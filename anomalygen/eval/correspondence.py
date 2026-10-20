# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defect correspondence scoring using frozen DINOv2 ViT-L/14 patch features.

For each generated image, we extract patch-level features from only the masked defect region,
then measure how well those features match real defect patches of the same anomaly type via
nearest-neighbour cosine similarity.

Two complementary scores are produced:
  - nn_score  : best-match similarity per generated patch, pooled       (higher = better)
  - mnn_score : same, but restricted to mutual nearest-neighbour pairs  (stricter, higher = better)

Feature extraction and pooling are configurable; the defaults are the validated best setting:
  - layer         : DINOv2 block to read (-1 = final post-LN tokens; e.g. 12 = block-12 output).
  - readout       : per-generated-patch pooling for nn ("mean", "p25" = the 25th-percentile value,
                    or "worst25" = mean of the lowest-matched 25%; p25 / worst25 weight the worst
                    patches more heavily).
  - region_policy : "full" (whole-image aspect-preserving resize) or "zoom" (per-instance square
                    crops so every defect fills a comparable input fraction).
  - inst_agg      : how the per-instance scores of a multi-part mask combine ("min" = a sample is
                    only as good as its worst part, or "mean").

These are added as extra rows in valid_kpi.csv alongside FID.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from cosmos_framework.utils import log
from PIL import Image
from scipy import ndimage
from torchvision.transforms import functional as TF

from anomalygen.models.vision_encoder.dinov2 import DEFAULT_BACKBONE, BackboneSpec, get_dinov2_model


def _image_mask_to_tensor(
    image_arr,
    mask_arr,
    min_size=224,
    max_size=518 * 2,
    patch_size=14,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """float32 numpy (H×W×3, 0-1) → normalised [3, H, W] tensor."""
    image = Image.fromarray((image_arr * 255).astype(np.uint8)).convert("RGB")
    mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).convert("L")
    height, width = image.height, image.width

    if height < min_size or width < min_size:
        # Upscale so the short side reaches min_size while preserving aspect ratio.
        scale = min_size / min(height, width)
        height, width = int(round(height * scale)), int(round(width * scale))
    if height > max_size or width > max_size:
        # Downscale so the long side fits in max_size while preserving aspect ratio.
        scale = max_size / max(height, width)
        height, width = int(round(height * scale)), int(round(width * scale))

    # Snap to a multiple of the patch size so the strided ViT conv doesn't drop pixels.
    # Ref: https://github.com/facebookresearch/dinov2/issues/86
    new_height = height - height % patch_size
    new_width = width - width % patch_size
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

    return (
        TF.normalize(TF.to_tensor(image), mean=mean, std=std),  # [3, H, W]
        (TF.to_tensor(mask)[0] > 0.5).float(),  # [H, W]
        new_height,
        new_width,
    )


def _patch_grid_mask(mask_tensor, height_patch, width_patch, patch_size=14):
    """Project a binary pixel mask [H, W] onto the ViT patch grid [H_p, W_p].

    Uses max-pooling: a patch is active if any pixel inside it is active.
    """
    # Crop to the exact pixel area the ViT conv covers (handles non-divisible sizes).
    patch_view = mask_tensor[: height_patch * patch_size, : width_patch * patch_size].reshape(
        height_patch, patch_size, width_patch, patch_size
    )
    return patch_view.amax(dim=(1, 3))  # [H_p, W_p]


def _extract_defect_features(
    model, image_arr, mask_arr, layer=-1, patch_size=14, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
) -> Optional[torch.Tensor]:
    """Return patch features for pixels inside the defect mask, shape [N_defect, D].

    ``layer`` selects the DINOv2 depth: -1 reads the final post-LN tokens; a block index reads
    ``hidden_states[layer]`` (index 0 = embeddings, i = output of block i, so 12 = block-12 output).
    ``output_hidden_states`` forces a one-time recompile of the compiled backbone but returns the
    same features. Returns None if the mask is empty after downsampling to patch resolution.
    """
    x, mask_tensor, new_height, new_width = _image_mask_to_tensor(
        image_arr, mask_arr, patch_size=patch_size, mean=mean, std=std
    )
    device = next(model.parameters()).device
    x = x.unsqueeze(0).to(device)  # [1, 3, H, W]
    mask_tensor = mask_tensor.to(device)  # [H, W]

    if layer == -1:
        patch_tokens = model(pixel_values=x).last_hidden_state[:, 1:, :].squeeze(0)  # [N, D]
    else:
        out = model(pixel_values=x, output_hidden_states=True)
        patch_tokens = out.hidden_states[layer][:, 1:, :].squeeze(0)  # [N, D]

    h_p, w_p = new_height // patch_size, new_width // patch_size
    grid = patch_tokens.reshape(h_p, w_p, -1)  # [H_p, W_p, D]
    defect_mask = _patch_grid_mask(mask_tensor, h_p, w_p, patch_size) > 0.5  # [H_p, W_p]

    if not defect_mask.any():
        return None

    # Move back to CPU to avoid massive memory usage for storing the features.
    return grid[defect_mask].to("cpu")  # [N_defect, D]


# --------------------------------------------------------------------------- zoom region policy
# Cap on per-mask zoom instances. `_region_feature_sets` batches one 518×518 DINOv2-L crop per
# instance, so peak GPU memory grows with the instance count. Measured on DINOv2-large (fp32, layer-12
# path), peak = weights + activations: N=1 → 1.4 GB, N=8 → 2.7 GB, N=32 → 7.1 GB. A speckled mask can
# label dozens of components ≥ min_area, which would OOM alongside the diffusion model on the same GPU.
# Cap at the largest 8 by area (the real defects; tiny specks drop) — 2.7 GB is safe on any target GPU,
# and 8 leaves headroom over generate.py's default of 5 real instances.
_MAX_ZOOM_INSTANCES = 8

# Re-validated nn/mnn scoring defaults. Hoisted to one place so `compute_correspondence_kpi`,
# `ValidationKPI`, and `add_nn_scoring_args` all reference the same source — the three cannot drift
# apart silently (the golden test only pins the first). See fine-tuning.md for the config change.
DEFAULT_NN_LAYER = 12
DEFAULT_NN_READOUT = "worst25"
DEFAULT_NN_REGION_POLICY = "zoom"
DEFAULT_NN_INST_AGG = "min"


def _split_instances(
    mask_bool: np.ndarray, min_area: int = 9, max_instances: int = _MAX_ZOOM_INSTANCES
) -> list[np.ndarray]:
    """Connected mask instances (8-connectivity, 3×3 closing first to merge fragments).

    Components below ``min_area`` pixels are dropped unless that would leave nothing. At most
    ``max_instances`` are returned — the largest by area — so the downstream DINOv2 batch is bounded.
    """
    closed = ndimage.binary_closing(mask_bool, structure=np.ones((3, 3), dtype=bool))
    labels, n = ndimage.label(closed, structure=np.ones((3, 3), dtype=int))
    comps = [c for i in range(1, n + 1) if (c := (labels == i) & mask_bool).any()]
    insts = [c for c in comps if int(c.sum()) >= min_area] or comps
    if len(insts) > max_instances:
        insts = sorted(insts, key=lambda c: int(c.sum()), reverse=True)[:max_instances]
    return insts


def _crop_instance(
    pil_img: Image.Image, inst_bool: np.ndarray, pad_ratio: float = 0.5, out: int = 518, min_side: int = 64
) -> tuple[Image.Image, np.ndarray]:
    """Square crop around one instance's bbox with ``pad_ratio``·side of context per side, clamped
    to the image then letterbox-padded back to square (so an elongated image doesn't stretch the
    crop), resized to ``(tgt, tgt)``. Returns (cropped_image, cropped_mask_uint8)."""
    ys, xs = np.nonzero(inst_bool)
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bbox = max(bottom - top, right - left)  # defect side before padding
    height, width = inst_bool.shape
    side = min(max(min_side, int(round(bbox * (1 + 2 * pad_ratio)))), max(height, width))
    cy, cx = (top + bottom) // 2, (left + right) // 2
    y0 = int(np.clip(cy - side // 2, 0, max(0, height - side)))
    x0 = int(np.clip(cx - side // 2, 0, max(0, width - side)))
    y1, x1 = min(height, y0 + side), min(width, x0 + side)

    tgt = max(56, out - out % 14)  # snap the target to the 14px patch grid (>= 4 patches)
    crop = pil_img.crop((x0, y0, x1, y1))
    inst_crop = inst_bool[y0:y1, x0:x1].astype(np.uint8)
    # The crop is non-square when `side` was clamped by an elongated image; letterbox-pad it to a
    # square before the resize so features keep their aspect ratio instead of being stretched. The
    # zero padding sits outside the mask, so it contributes no defect patches.
    ch, cw = inst_crop.shape
    if ch != cw:
        s = max(ch, cw)
        py, px = (s - ch) // 2, (s - cw) // 2
        square = Image.new("RGB", (s, s))
        square.paste(crop, (px, py))
        crop = square
        padded = np.zeros((s, s), dtype=np.uint8)
        padded[py : py + ch, px : px + cw] = inst_crop
        inst_crop = padded
    crop = crop.resize((tgt, tgt), Image.Resampling.BICUBIC)
    mask_region = Image.fromarray(inst_crop * 255)
    crop_mask = (np.asarray(mask_region.resize((tgt, tgt), Image.Resampling.NEAREST)) > 127).astype(np.uint8)
    return crop, crop_mask


def _zoom_regions(
    pil_img: Image.Image, mask_bool: np.ndarray, pad_ratio: float = 0.5, out: int = 518
) -> list[tuple[Image.Image, np.ndarray]]:
    """mask → [(cropped_image, cropped_mask_uint8), ...], one per connected instance."""
    return [_crop_instance(pil_img, inst, pad_ratio, out) for inst in _split_instances(mask_bool)]


def _region_feature_sets(model, image_arr, mask_arr, region_policy, layer, patch_size, mean, std) -> list[torch.Tensor]:
    """Per-instance defect-feature tensors (unnormalised).

    ``full`` → ``[whole-mask features]`` (length 0 or 1); ``zoom`` → one tensor per connected mask
    instance (empty tensors from instances with no active patch are skipped). All instance crops of
    one sample are the same square size, so they go through the backbone in a SINGLE batched forward
    (the ViT processes each image independently) rather than one forward per instance.
    """
    if region_policy == "full":
        feats = _extract_defect_features(
            model, image_arr, mask_arr, layer=layer, patch_size=patch_size, mean=mean, std=std
        )
        return [feats] if feats is not None else []
    if region_policy != "zoom":
        raise ValueError(f"Unknown region_policy '{region_policy}'. Choose 'full' or 'zoom'.")

    mask_bool = np.asarray(mask_arr) > 0.5
    if not mask_bool.any():
        return []
    pil_img = Image.fromarray((np.asarray(image_arr) * 255).astype(np.uint8)).convert("RGB")
    crops = _zoom_regions(pil_img, mask_bool)  # [(crop_image, crop_mask_uint8), ...], all same size
    if not crops:
        return []

    device = next(model.parameters()).device
    x = torch.stack([TF.normalize(TF.to_tensor(crop), mean=mean, std=std) for crop, _ in crops]).to(device)  # [N,3,H,W]
    if layer == -1:
        tokens = model(pixel_values=x).last_hidden_state[:, 1:, :]  # [N, P, D]
    else:
        tokens = model(pixel_values=x, output_hidden_states=True).hidden_states[layer][:, 1:, :]

    h_p, w_p = x.shape[-2] // patch_size, x.shape[-1] // patch_size
    feats_list: list[torch.Tensor] = []
    for i, (_, crop_mask) in enumerate(crops):
        grid = tokens[i].reshape(h_p, w_p, -1)  # [H_p, W_p, D]
        # Move the patch-grid mask to the token device so boolean-indexing a CUDA `grid` is always
        # valid (older torch raised on a CPU mask), then offload the gathered features to CPU so the
        # reference bank accumulated across all instances doesn't pin GPU memory (mirrors the
        # non-zoom path).
        defect_mask = (_patch_grid_mask(torch.from_numpy(crop_mask).float(), h_p, w_p, patch_size) > 0.5).to(
            grid.device
        )
        if defect_mask.any():
            feats_list.append(grid[defect_mask].to("cpu"))  # [N_defect, D]
    return feats_list


def _nn_mnn_scores(f_g: torch.Tensor, f_r: torch.Tensor, readout: str = "mean"):
    """NN and MNN cosine similarity scores between *L2-normalised* features.

    The caller is responsible for normalising the inputs (so per-gen and
    per-real normalisation happens once, not on every gen×real pair).

    ``readout`` pools the per-gen-patch best-match similarities for NN: "mean" (the baseline),
    "p25" (the 25th-percentile value), or "worst25" (mean of the lowest-matched 25%). p25 / worst25
    emphasise the worst-matched patches. MNN is always a plain mean over mutual pairs.

    Returns 0-D GPU tensors so the caller can batch device→host syncs:
      - NN  : pooled best-match similarity, gen → real direction.
      - MNN : mean similarity of mutual nearest-neighbour pairs only;
              0.0 (as a tensor) when no mutual pair exists.
    """
    sim = f_g @ f_r.T  # [N, M]

    # NN: best real match per gen patch (values + indices in one kernel), pooled per ``readout``.
    nn_values, nn_g2r = sim.max(dim=1)  # [N], [N]
    if readout == "mean":
        nn = nn_values.mean()
    elif readout == "p25":
        nn = nn_values.quantile(0.25)
    else:  # "worst25": mean of the lowest-matched 25% of gen patches (most pessimistic)
        k = max(1, int(0.25 * nn_values.numel()))
        nn = torch.sort(nn_values).values[:k].mean()

    # MNN: reuse sim; check mutual pairs without a host-side branch.
    nn_r2g = sim.argmax(dim=0)  # [M]
    idx = torch.arange(f_g.shape[0], device=sim.device)
    mutual = nn_r2g[nn_g2r] == idx  # [N], bool
    sim_diag = sim[idx, nn_g2r]  # [N]
    mut_f = mutual.float()
    # Safe mean → 0 when there are no mutual pairs (sum is 0 too).
    mnn = (sim_diag * mut_f).sum() / mut_f.sum().clamp(min=1)

    return nn, mnn


@torch.inference_mode()
def compute_correspondence_kpi(
    real_images_dict: dict,
    generated_images_dict: dict,
    backbone: str = DEFAULT_BACKBONE,
    top_k: int = 3,
    *,
    layer: int = DEFAULT_NN_LAYER,
    readout: str = DEFAULT_NN_READOUT,
    region_policy: str = DEFAULT_NN_REGION_POLICY,
    inst_agg: str = DEFAULT_NN_INST_AGG,
) -> dict:
    """Compute nn_score and mnn_score for each anomaly type and their macro average.

    Scoring strategy (per-instance + top-K):
        - Every real reference and every generated sample is turned into a list of per-instance
          feature sets by ``region_policy`` ("full" = one whole-mask set; "zoom" = one set per
          connected mask instance). All reference instances form a flat bank.
        - For each generated instance, score it against every reference instance individually,
          sort those pair scores by nn descending, take the top-K, and mean their nn / mnn.
        - Combine the generated sample's instances with ``inst_agg`` ("min" = worst part).
        - Per anomaly type, mean across all generated samples; "Average" is the macro-mean.

    Args:
        backbone: HuggingFace model ID (or local path) for the feature extractor. Defaults to the
                  bundled DINOv2 ViT-L/14.
        top_k:    Number of best-matching reference instances to average over per generated
                  instance. -1 (or a value ≥ the number of reference instances) means use all.
        layer / readout / region_policy / inst_agg: feature-extraction and pooling toggles; see the
                  module docstring. Defaults are the validated best setting (zoom + block 12 +
                  worst-25% readout + worst-part aggregation).

    Inputs use the same dict structure as compute_kpi():
        real_images_dict[anomaly_name]["original_image"]  — list of numpy (H×W×3, 0-1)
        real_images_dict[anomaly_name]["original_mask"]   — list of numpy (H×W, 0-1)
        generated_images_dict[anomaly_name]["reconstructed_image"] — same format
        generated_images_dict[anomaly_name]["original_mask"]       — inpainting-input masks

    Returns a dict mirroring compute_kpi output:
        {anomaly_name: {"nn_score": float, "mnn_score": float, "per_sample": [...]},
         "Average": {"nn_score": float, "mnn_score": float}}

    `per_sample` is one row per generated sample with `path`, `nn_score`, `mnn_score`;
    samples that fail feature extraction are recorded with NaN scores. Every anomaly type present
    in real_images_dict appears in the result (NaN where scoring failed), so the valid_kpi.csv
    writer can iterate safely.
    """
    if top_k == 0 or top_k < -1:
        raise ValueError(f"top_k must be -1 (use all) or positive, got {top_k}.")
    if backbone not in BackboneSpec:
        raise ValueError(f"Unknown backbone '{backbone}'. Available: {list(BackboneSpec.keys())}")
    if readout not in ("mean", "p25", "worst25"):
        raise ValueError(f"readout must be 'mean', 'p25', or 'worst25', got '{readout}'.")
    if region_policy not in ("full", "zoom"):
        raise ValueError(f"region_policy must be 'full' or 'zoom', got '{region_policy}'.")
    if inst_agg not in ("min", "mean"):
        raise ValueError(f"inst_agg must be 'min' or 'mean', got '{inst_agg}'.")

    model = get_dinov2_model(backbone)
    patch_size = BackboneSpec[backbone]["patch_size"]
    mean = BackboneSpec[backbone]["mean"]
    std = BackboneSpec[backbone]["std"]
    reduce_inst = min if inst_agg == "min" else (lambda vals: float(np.mean(vals)))

    result: dict = {}
    nn_vals: list[float] = []
    mnn_vals: list[float] = []

    for anomaly_name in sorted(real_images_dict.keys()):
        real_images = real_images_dict[anomaly_name].get("original_image", [])
        real_masks = real_images_dict[anomaly_name].get("original_mask", [])
        gen_images = generated_images_dict[anomaly_name].get("reconstructed_image", [])
        gen_masks = generated_images_dict[anomaly_name].get("original_mask")

        if gen_masks is None:
            # Each gen image is paired with the input mask used for its inpainting; falling back to
            # real_masks would mis-pair gen features with unrelated masks, so we skip instead.
            log.warning(f"[{anomaly_name}] generated_images_dict missing 'original_mask' — skipping.")
            result[anomaly_name] = {"nn_score": float("nan"), "mnn_score": float("nan"), "per_sample": []}
            continue

        # Gen images and their inpainting-input masks must line up 1:1 (same for real); a length
        # mismatch would silently mis-pair or drop tail entries under zip, corrupting the score.
        if len(real_images) != len(real_masks):
            raise ValueError(
                f"[{anomaly_name}] real image/mask count mismatch: {len(real_images)} images vs {len(real_masks)} masks"
            )
        if len(gen_images) != len(gen_masks):
            raise ValueError(
                f"[{anomaly_name}] generated image/mask count mismatch: {len(gen_images)} images vs {len(gen_masks)} "
                "masks"
            )

        # Flatten every real reference into a bank of per-instance, pre-normalised features; reused
        # across every generated sample of this anomaly type.
        ref_bank: list[torch.Tensor] = []
        for img, mask in zip(real_images, real_masks):
            for feats in _region_feature_sets(model, img, mask, region_policy, layer, patch_size, mean, std):
                ref_bank.append(F.normalize(feats, dim=-1, eps=1e-8))

        if not ref_bank:
            log.warning(f"[{anomaly_name}] No real defect patches — skipping.")
            result[anomaly_name] = {"nn_score": float("nan"), "mnn_score": float("nan"), "per_sample": []}
            continue

        # Resolve top-K once per anomaly type (bank size is fixed for this loop).
        n_refs = len(ref_bank)
        effective_k = n_refs if top_k == -1 or top_k >= n_refs else top_k

        gen_paths = generated_images_dict[anomaly_name].get("img_path", [None] * len(gen_images))

        type_nn: list[float] = []
        type_mnn: list[float] = []
        per_sample: list[dict] = []

        for path, img, mask in zip(gen_paths, gen_images, gen_masks):
            gen_sets = _region_feature_sets(model, img, mask, region_policy, layer, patch_size, mean, std)
            if not gen_sets:
                per_sample.append({"path": path, "nn_score": float("nan"), "mnn_score": float("nan")})
                continue

            # Per generated instance: score against every reference instance, average the top-K by
            # NN, then reduce across the mask's instances with ``inst_agg``.
            inst_nn: list[float] = []
            inst_mnn: list[float] = []
            for feats_gen in gen_sets:
                f_g = F.normalize(feats_gen, dim=-1, eps=1e-8)  # normalise once per gen instance

                # Collect per-pair score tensors and reduce them together (topk then mean) rather than
                # calling .item() per reference, so we sync once per metric per gen instance.
                scored = [_nn_mnn_scores(f_g, f_r, readout=readout) for f_r in ref_bank]
                nn_t = torch.stack([s[0] for s in scored])
                mnn_t = torch.stack([s[1] for s in scored])

                # Sort by NN desc, average both metrics over the same top-K reference instances.
                topk_nn, topk_idx = nn_t.topk(effective_k)
                inst_nn.append(topk_nn.mean().item())
                inst_mnn.append(mnn_t[topk_idx].mean().item())

            nn_s = reduce_inst(inst_nn)
            mnn_s = reduce_inst(inst_mnn)
            type_nn.append(nn_s)
            type_mnn.append(mnn_s)
            per_sample.append({"path": path, "nn_score": nn_s, "mnn_score": mnn_s})

        if not type_nn:
            log.warning(f"[{anomaly_name}] No generated defect patches — skipping.")
            result[anomaly_name] = {"nn_score": float("nan"), "mnn_score": float("nan"), "per_sample": per_sample}
            continue

        result[anomaly_name] = {
            "nn_score": float(np.mean(type_nn)),
            "mnn_score": float(np.mean(type_mnn)),
            "per_sample": per_sample,
        }
        nn_vals.append(result[anomaly_name]["nn_score"])
        mnn_vals.append(result[anomaly_name]["mnn_score"])

        log.info(
            f"[{anomaly_name}] nn_score={result[anomaly_name]['nn_score']:.4f}  "
            f"mnn_score={result[anomaly_name]['mnn_score']:.4f}  "
            f"(top_k={effective_k}/{n_refs}, layer={layer}, readout={readout}, "
            f"region={region_policy}, inst_agg={inst_agg})"
        )

    if nn_vals:
        result["Average"] = {"nn_score": float(np.mean(nn_vals)), "mnn_score": float(np.mean(mnn_vals))}

    return result


def add_nn_scoring_args(parser) -> None:
    """Add the nn/mnn scoring knobs to a CLI ``parser``. Defaults are the validated setting used by
    training-time ``ValidationKPI`` (``zoom`` / block-12 / ``worst25`` / ``min``); pass
    ``--nn_region_policy full --nn_layer -1 --nn_readout mean --nn_inst_agg mean`` to reproduce the
    pre-MR ``full`` / final-layer / ``mean`` defaults instead. These change the nn/mnn *values*, so a
    baseline recorded under one setting is not comparable to another."""
    parser.add_argument(
        "--nn_layer", type=int, default=DEFAULT_NN_LAYER, help="DINOv2 block to read (-1 = final post-LN tokens)"
    )
    parser.add_argument(
        "--nn_readout", choices=["mean", "p25", "worst25"], default=DEFAULT_NN_READOUT, help="per-gen-patch nn pooling"
    )
    parser.add_argument(
        "--nn_region_policy",
        choices=["full", "zoom"],
        default=DEFAULT_NN_REGION_POLICY,
        help="whole-image vs per-instance square crops",
    )
    parser.add_argument(
        "--nn_inst_agg",
        choices=["min", "mean"],
        default=DEFAULT_NN_INST_AGG,
        help="combine a multi-part mask's per-instance scores",
    )


def nn_scoring_kwargs(args) -> dict:
    """Map parsed :func:`add_nn_scoring_args` values to ``compute_correspondence_kpi`` keyword args."""
    return dict(
        layer=args.nn_layer, readout=args.nn_readout, region_policy=args.nn_region_policy, inst_agg=args.nn_inst_agg
    )
