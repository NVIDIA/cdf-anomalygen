# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-to-image anomaly-inpainting model ops.

The reusable inference building blocks shared by the ``anomalygen/scripts/texture/generate.py`` CLI and the
``ValidationKPI`` callback: load the trained model, build a single-crop inpaint batch,
generate, decode, and the ``build_inpaint_batch_fn`` closure the iterative orchestrator drives.
"""

from __future__ import annotations

import contextlib
import glob
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.common.config import ROOT_DIR
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint
from PIL import Image
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

import anomalygen  # noqa: F401  (registration side effects)
from anomalygen.checkpoint.trained_keys import TRAINED_KEY_PREFIXES, WARM_START_SKIP_PREFIXES, match_keys
from anomalygen.checkpoint.utils import (
    GEOMETRY_STAMP_KEY,
    pop_and_verify_geometry_stamp,
    resolve_shift,
    upgrade_legacy_state,
    verify_digest,
    verify_manifest,
)
from anomalygen.configs.loader import read_experiment_field, register_recipe
from anomalygen.configs.texture.constants import (
    BASE_CHECKPOINT_PATHS,
    DEFAULT_MODEL_SIZE,
    MODEL_SIZES,
)
from anomalygen.configs.texture.legacy import recipe_anomaly_types
from anomalygen.data.crop_geometry import (
    CropPlan,
    fit_to_input_size,
    resize_mask,
)
from anomalygen.data.utils import (
    MASK_FG_THRESHOLD,
    UNCONDITIONAL_CAPTION,
    build_source_item,
    caption_for_anomaly_type,
)

# Trained subset the fine-tuned .pt must contain, and the keys to skip when warm-starting the base
# network. Shared with the optimizer / checkpointer (see trained_keys).
_FT_TRAINED_KEYS = TRAINED_KEY_PREFIXES
_BASE_SKIP_KEYS = WARM_START_SKIP_PREFIXES


def load_model(
    checkpoint_path: str,
    experiment: str = "anomalygen_texture_ft",
    keys_to_skip_loading: Optional[List[str]] = None,
):
    """Load a model from a full DCP/safetensors checkpoint using its experiment config.

    Internal base-network loader; for an anomalygen fine-tuned run (a filtered ``.pt`` from
    ``SelectiveCheckpointer``) use :func:`load_finetuned_model` / :func:`load_for_inference`.
    The framework's own default config (``cosmos_framework/configs/base/config.py``) is used.
    """
    kwargs = {}
    if keys_to_skip_loading is not None:
        kwargs["keys_to_skip_loading"] = keys_to_skip_loading
    # Resolve the checkpoint to an absolute path against the caller's cwd before the chdir below,
    # so a relative --checkpoint / --base_checkpoint keeps pointing at the repo, not the framework tree.
    checkpoint_path = os.path.abspath(checkpoint_path)
    # Build with cwd at cosmos_framework's parent dir so the model reads its packaged resources
    # by "cosmos_framework/..."-prefixed paths relative to it (e.g. the Qwen3-VL config JSON,
    # the tokenizers/ tree), mirroring anomalygen/scripts/texture/train.py. ROOT_DIR is the
    # package dir; its parent is site-packages for the pip-installed framework.
    with contextlib.chdir(ROOT_DIR.parent):
        model, _config = load_model_from_checkpoint(
            experiment_name=experiment,
            checkpoint_path=checkpoint_path,
            experiment_opts=["model.config.rectified_flow_inference_config.scheduler_type=unipc"],
            **kwargs,
        )
    model.eval()

    return model


# Checkpoint pointers, in preference order. ``best_checkpoint.txt`` (written by the TrainingReport
# callback at train end) names the peak-scoring iteration; ``latest_checkpoint.txt`` (written by the
# checkpointer for resume) names the *last* one, which is frequently worse — small datasets peak early
# and then drift. Preferring "best" here means generating from a run dir does the right thing without
# the caller having to read the pointer themselves.
_CKPT_POINTERS = ("best_checkpoint.txt", "latest_checkpoint.txt")


def _resolve_ft_model_pt(path: str) -> str:
    """Resolve the fine-tuned model component ``.pt`` from ``path``.

    Accepts the ``.pt`` file directly, or a directory (run dir or its ``checkpoints/``) under which
    the SelectiveCheckpointer layout ``model/iter_<N>.pt`` lives. Honors ``best_checkpoint.txt``,
    then ``latest_checkpoint.txt``, else picks the highest iteration (filenames are zero-padded, so
    lexicographic). Logs which pointer decided it, since best vs. latest changes the output quality.
    """
    if path.endswith(".pt") and os.path.isfile(path):
        return path

    for root in (path, os.path.join(path, "checkpoints")):
        model_dir = os.path.join(root, "model")
        if not os.path.isdir(model_dir):
            continue
        for pointer_name in _CKPT_POINTERS:
            pointer = os.path.join(root, pointer_name)
            if not os.path.isfile(pointer):
                continue
            with open(pointer) as f:
                name = f.read().strip()
            if name:
                candidate = os.path.join(model_dir, name if name.endswith(".pt") else f"{name}.pt")
                if os.path.isfile(candidate):
                    log.info(f"Resolved fine-tuned checkpoint from {pointer_name}: {candidate}")
                    return candidate
        pts = sorted(glob.glob(os.path.join(model_dir, "iter_*.pt")))
        if pts:
            log.warning(
                f"No usable checkpoint pointer under {root!r}; falling back to the highest iteration "
                f"{pts[-1]} — this is the LAST checkpoint, not necessarily the best-scoring one."
            )
            return pts[-1]

    raise FileNotFoundError(
        f"No fine-tuned model checkpoint found under {path!r}. Expected a 'model/iter_<N>.pt' "
        "(SelectiveCheckpointer layout) or a direct .pt file."
    )


def _verify_overlay(ckpt_model: dict, full_state: dict) -> None:
    """Guard the strict=False overlay: reject unknown checkpoint keys (silently dropped otherwise)
    and a checkpoint missing trained keys (left at warm-started base values). Mirrors
    ``SelectiveCheckpointer._verify_load_filtered``."""
    unexpected = set(ckpt_model) - set(full_state)
    if unexpected:
        raise ValueError(
            f"Fine-tuned checkpoint has {len(unexpected)} key(s) absent from the model; refusing to load. "
            f"Examples: {sorted(unexpected)[:10]}. The checkpoint likely predates a model/key-layout change."
        )
    missing = match_keys(full_state, _FT_TRAINED_KEYS) - set(ckpt_model)
    if missing:
        raise ValueError(
            f"Fine-tuned checkpoint is missing {len(missing)} trained key(s) {_FT_TRAINED_KEYS} present in the "
            f"model — they would stay at base values. Examples: {sorted(missing)[:10]}."
        )


# Digests for the locally-converted base DCP shards, recorded by scripts/download_checkpoints.sh.
CONVERTED_MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "assets" / "checkpoint_manifest_converted.sha256"


def _verify_base_weights(base_checkpoint: str) -> None:
    """Check the base DCP's shards against the recorded manifest — opt-in, off by default.

    Hashing multi-GB shards costs tens of seconds, which is too much for every load but cheap next to
    a run that silently used substituted weights. Only the size being loaded is hashed. Skipped
    silently when the manifest is absent, so a checkout that never recorded one still runs.
    """
    if os.environ.get("ANOMALYGEN_VERIFY_BASE_WEIGHTS") != "1":
        return
    manifest = pathlib.Path(CONVERTED_MANIFEST)
    if not manifest.is_file():
        log.warning(f"Base-weight verification requested but {manifest} is absent; skipping.")
        return
    base = pathlib.Path(base_checkpoint).resolve()
    failures = verify_manifest(manifest, base.parent, only_under=base.name)
    if failures:
        detail = "\n".join(f"  {rel}: {why}" for rel, why in failures)
        raise ValueError(f"Base checkpoint integrity: {len(failures)} shard(s) under {base} failed.\n{detail}")
    log.info(f"Base checkpoint integrity: {base.name} shards match {manifest.name}.")


def _is_legacy(model_pt: str) -> bool:
    """True when the fine-tuned ``.pt`` carries no geometry stamp: a legacy checkpoint."""
    return GEOMETRY_STAMP_KEY not in torch.load(model_pt, map_location="cpu", weights_only=True, mmap=True)


def load_finetuned_model(
    ft_checkpoint: str,
    base_checkpoint: Optional[str] = None,
    experiment: str = "anomalygen_texture_ft",
    legacy_class_order: Optional[List[List[str]]] = None,
):
    """Load an anomalygen fine-tuned model saved by ``SelectiveCheckpointer``.

    Those checkpoints hold only the trained subset (LoRA + ``text_prompt_emb``), so this warm-starts
    the full base network from ``base_checkpoint`` (skipping the keys absent from it), then overlays
    the trained subset from ``ft_checkpoint``'s flat ``.pt`` — the model-only half of
    ``SelectiveCheckpointer._load_filtered``. ``ft_checkpoint`` may be the run dir, its ``checkpoints/``,
    or the ``iter_<N>.pt`` directly. The experiment node must already be registered (see
    :func:`load_for_inference`, which registers it from a recipe and calls this).
    ``base_checkpoint`` defaults to the DCP matching that node's ``model_size``.

    ``legacy_class_order`` is the recipe's ``anomaly_types`` listing for a legacy (unstamped)
    checkpoint: the state is stamped and its classes reordered in memory before the overlay.
    """
    if base_checkpoint is None:
        base_checkpoint = _base_checkpoint_for_experiment(experiment)
    model_pt = _resolve_ft_model_pt(ft_checkpoint)
    # Before the base network loads, so a checkpoint that fails costs seconds rather than a multi-GB
    # warm start. verify_digest says these are the bytes we trained; weights_only=True below says an
    # unpickle cannot execute code. This is the load that reads a .pt handed over from a training
    # host or a mounted directory, so both matter here more than on a resume.
    verify_digest(model_pt)
    _verify_base_weights(base_checkpoint)
    model = load_model(base_checkpoint, experiment=experiment, keys_to_skip_loading=list(_BASE_SKIP_KEYS))

    log.info(f"Overlaying fine-tuned weights from {model_pt} onto base {base_checkpoint}.")
    ckpt_model = dict(torch.load(model_pt, map_location="cpu", weights_only=True))
    if legacy_class_order is not None:
        upgrade_legacy_state(ckpt_model, model, legacy_class_order, os.path.basename(model_pt))
    # Generation has no dataset to scan: the per-class caps come from the checkpoint.
    model.class_ladders = pop_and_verify_geometry_stamp(ckpt_model, model, os.path.basename(model_pt))
    full_state = get_model_state_dict(model)
    _verify_overlay(ckpt_model, full_state)
    full_state.update(ckpt_model)
    set_model_state_dict(model, model_state_dict=full_state, options=StateDictOptions(strict=False))
    model.eval()

    return model


def _base_checkpoint_for_experiment(experiment: str) -> str:
    """Frozen base DCP matching the registered ``experiment`` node's ``model_size``.

    Read off the built node, not the recipe file, so it covers both recipe forms the loader accepts
    (YAML/JSON file or Python module). The other size's DCP would load silently as wrong weights.
    """
    model_size = str(read_experiment_field(experiment, "diffusion_expert_config.model_size") or DEFAULT_MODEL_SIZE)
    if model_size not in BASE_CHECKPOINT_PATHS:
        raise ValueError(f"Experiment {experiment!r} sets model_size={model_size!r}, not one of {MODEL_SIZES}.")
    return BASE_CHECKPOINT_PATHS[model_size]


def load_for_inference(
    recipe: str,
    ft_checkpoint: str,
    base_checkpoint: Optional[str] = None,
    experiment: Optional[str] = None,
):
    """Single entry point for inference scripts: register the experiment recipe, then load the model.

    Importing ``anomalygen`` only registers the reusable groups (model / ckpt_type / callbacks); the
    experiment node is registered here from ``recipe`` before the loader hydra-composes it. The
    experiment name comes from the recipe; ``experiment`` overrides it, and picks the node when a
    module recipe stores several. ``base_checkpoint`` defaults to the DCP matching the recipe's
    ``model_size``.

    A checkpoint with no geometry stamp is a legacy checkpoint. Its recipe (the run-dir copy) is then
    read as a legacy recipe, and the checkpoint is stamped from it and reordered to the sorted class
    ids in memory. Generation only: such a checkpoint cannot be resumed.
    """
    model_pt = _resolve_ft_model_pt(ft_checkpoint)
    legacy = _is_legacy(model_pt)
    if legacy:
        log.warning(
            f"{model_pt} carries no geometry stamp (legacy checkpoint): {recipe} is read as a legacy recipe and "
            "the checkpoint is stamped from it at load. Generation only; this checkpoint cannot be resumed."
        )
    # Never shard the model for inference. Multi-GPU generation is data-parallel — each rank runs
    # its own testcases with a full model copy — so sharding only adds parameter all-gathers. Their
    # count follows each testcase's instance depth, which differs per rank, so the ranks fall out of
    # step and the collective hangs. It also costs more memory per GPU than an unsharded copy.
    #
    # The builder reads this while register_recipe runs and bakes the choice into the model-group
    # node, so it only has to hold across that call. Restored after: leaving it set would pin any
    # experiment built later in the same interpreter to ddp.
    previous = os.environ.get("ANOMALYGEN_PARALLELISM")
    os.environ["ANOMALYGEN_PARALLELISM"] = "ddp"
    try:
        registered = register_recipe(recipe, experiment=experiment, legacy=legacy)
    finally:
        if previous is None:
            os.environ.pop("ANOMALYGEN_PARALLELISM", None)
        else:
            os.environ["ANOMALYGEN_PARALLELISM"] = previous
    name = experiment or registered
    if base_checkpoint is None:
        base_checkpoint = _base_checkpoint_for_experiment(name)
    return load_finetuned_model(
        model_pt,
        base_checkpoint=base_checkpoint,
        experiment=name,
        legacy_class_order=recipe_anomaly_types(recipe) if legacy else None,
    )


def _load_conditioning(image: Image.Image, h: int, w: int, device, dtype) -> torch.Tensor:
    """PIL RGB -> [3, 1, H, W] tensor in [-1, 1]."""
    img = image.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
    return t.unsqueeze(1).to(device=device, dtype=dtype)  # [3,1,H,W]


def build_inpaint_batch(
    model,
    image: Image.Image,
    mask: Image.Image,
    class_id: int,
    caption: str,
    seed: Optional[int] = None,
) -> Dict:
    """Build a single-sample two-vision-item batch for inpainting one crop.

    ``seed`` seeds the source item's conditioning noise; ``None`` uses the global RNG.
    """
    device = next(model.net.parameters()).device
    dtype = model.tensor_kwargs["dtype"]
    w, h = image.size
    cond = _load_conditioning(image, h, w, device, dtype)  # [3,1,H,W] clean (guide target)
    m = (torch.from_numpy(np.asarray(resize_mask(mask, (w, h)), dtype=np.float32)) >= MASK_FG_THRESHOLD).float()
    m = m.to(device=device)  # [H,W]

    m3 = m[None, None].expand(cond.shape)  # [3,1,H,W]
    # Defect region -> noise, background kept: the one source encoding, matching training with
    # background dropout off (the dropout is a training-only augmentation).
    source = build_source_item(cond, m3, seed=seed)

    image_size = torch.tensor([[h, w, h, w]], dtype=torch.float32, device=device)
    batch = {
        "dataset_name": "anomaly",
        "images": [source.unsqueeze(0), cond.unsqueeze(0)],  # each [1,3,1,H,W]
        "image_size": [image_size, image_size],
        "num_frames": [torch.tensor([2], dtype=torch.int64, device=device)],
        "num_vision_items_per_sample": [2],
        "is_preprocessed": True,
        "sequence_plan": [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])],
        "edit_mask": m[None, None],  # [1,1,H,W]
        "anomaly_class_id": torch.tensor([int(class_id)], dtype=torch.long, device=device),
        model.input_caption_key: [caption],
        # Key the framework reads when generate_samples_from_batch(has_negative_prompt=True).
        "neg_" + model.input_caption_key: [UNCONDITIONAL_CAPTION],
    }
    return batch


def _resolve_shift(model) -> float:
    """The recipe's sampler time-shift: one scalar, shared by every input size.

    The only source of a shift at inference, read off the ``--recipe`` the model was built from, so
    there is nothing to override and no default to fall back to. Pass the recipe the checkpoint
    trained under (the run-dir copy) or this silently samples on a different schedule.
    """
    shift = resolve_shift(model)
    if shift is None:
        raise AttributeError(
            "recipe config has no rectified_flow_training_config.shift; cannot match the sampler to what training used."
        )
    return float(shift)


def _build_inpaint_batch_multi(
    model,
    images: List[Image.Image],
    masks: List[Image.Image],
    class_ids: List[int],
    captions: List[str],
    seeds: Optional[List[int]] = None,
) -> Dict:
    """Merge B single-sample inpaint batches into one B-sample batch.

    Reuses :func:`build_inpaint_batch` per sample, then concatenates exactly as the framework's
    collate flattens per-sample list-valued keys (``images``/``image_size``/``num_frames``/
    ``sequence_plan`` flattened across samples; ``edit_mask``/``anomaly_class_id`` stacked on dim 0).
    ``seeds`` is one source-noise seed per sample, so noise never depends on batch position or size.
    """
    if seeds is None:
        seeds = [None] * len(images)
    singles = [
        build_inpaint_batch(model, img, msk, cid, cap, seed=sd)
        for img, msk, cid, cap, sd in zip(images, masks, class_ids, captions, seeds)
    ]
    merged = {
        "dataset_name": "anomaly",
        "is_preprocessed": True,
        "images": [vi for s in singles for vi in s["images"]],
        "image_size": [sz for s in singles for sz in s["image_size"]],
        "num_frames": [nf for s in singles for nf in s["num_frames"]],
        "num_vision_items_per_sample": [n for s in singles for n in s["num_vision_items_per_sample"]],
        "sequence_plan": [sp for s in singles for sp in s["sequence_plan"]],
        # Stack only when every crop shares an input size; under variable-resolution generation they do not,
        # and torch.cat would raise. A list is an accepted form here — model._per_item_pixel_masks
        # takes either a list of per-sample tensors or a stacked [B,1,H,W].
        "edit_mask": (
            torch.cat([s["edit_mask"] for s in singles], dim=0)
            if len({tuple(s["edit_mask"].shape[-2:]) for s in singles}) == 1
            else [s["edit_mask"] for s in singles]
        ),
        "anomaly_class_id": torch.cat([s["anomaly_class_id"] for s in singles], dim=0),
        model.input_caption_key: [c for s in singles for c in s[model.input_caption_key]],
    }
    neg_key = "neg_" + model.input_caption_key
    merged[neg_key] = [c for s in singles for c in s[neg_key]]
    return merged


def _decode_to_pil(model, vision_latent: torch.Tensor) -> Image.Image:
    """Decode a single sample's vision latent to a PIL RGB image.

    Normalizes the latent to 5D ``[B, C, T, H, W]`` as the VAE's conv3d requires.
    """
    latent = vision_latent
    while latent.dim() > 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    pix = model.decode(latent)  # [B,C,T,H,W] in [-1,1]

    pix = ((1.0 + pix) / 2.0).clamp(0, 1)[0, :, 0]  # [C,H,W]
    arr = (pix.permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)

    return Image.fromarray(arr)


@torch.inference_mode()
def build_inpaint_batch_fn(
    model,
    class_ids: Dict[str, int],
    num_steps: int,
    guidance: float,
):
    """Build a ``(crops, masks, names, seeds, plans) -> edited_crops`` batched closure for the
    iterative driver. Each crop is fitted to the plan the orchestrator passes in — the geometry is
    not recoverable from the cut alone — and handed back at its native size, so ``paste_back`` gets
    an exact paste. Returns one edited crop (PIL) per input crop.

    ``guidance``/``num_steps`` are batch-level scalars (the caller guarantees the batch is
    homogeneous in ``guidance``); ``seeds`` is one seed per crop, driving both its noise streams, so
    results are invariant to batch composition.
    """
    step_shift = _resolve_shift(model)

    def fn(
        crops: List[Image.Image],
        masks: List[Image.Image],
        names: List[str],
        seeds: List[int],
        plans: List[CropPlan],
    ) -> List[Image.Image]:
        cids: List[int] = []
        caps: List[str] = []
        rc: List[Image.Image] = []
        rm: List[Image.Image] = []
        natives: List[Tuple[int, int]] = []
        boxes: List[Optional[Tuple[int, int, int, int]]] = []
        for i, (img, msk, name) in enumerate(zip(crops, masks, names)):
            # anomaly_name is data-driven (testcase JSONL "anomaly_type"); a typo or a format mismatch
            # must fail loudly instead of silently generating with class 0.
            if name not in class_ids:
                raise KeyError(f"Unknown anomaly_name {name!r}; known: {sorted(class_ids)}")
            cids.append(class_ids[name])
            caps.append(caption_for_anomaly_type(name))
            natives.append(img.size)
            img, msk, box = fit_to_input_size(img, msk, plans[i])
            boxes.append(box)
            rc.append(img)
            rm.append(msk)

        batch = _build_inpaint_batch_multi(model, rc, rm, cids, caps, seeds=[int(s) for s in seeds])
        out = model.generate_samples_from_batch(
            batch,
            guidance=guidance,
            num_steps=num_steps,
            shift=step_shift,
            seed=list(seeds),
            has_negative_prompt=True,
        )

        outs: List[Image.Image] = []
        for i in range(len(crops)):
            o = _decode_to_pil(model, out["vision"][i])
            if boxes[i] is not None:
                o = o.crop(boxes[i])
            if o.size != natives[i]:
                o = o.resize(natives[i], Image.Resampling.BICUBIC)
            outs.append(o)
        return outs

    return fn
