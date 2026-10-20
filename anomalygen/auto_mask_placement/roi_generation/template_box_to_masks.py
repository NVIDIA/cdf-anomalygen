# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import math
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from skimage.feature import hog
from torchvision.ops import nms

from anomalygen.auto_mask_placement.roi_generation.utils import to_jsonable, to_rgb_uint8

# ── Module utilities (used only by this template pipeline) ────────────────────


def _mask_to_compressed_rle(mask):
    """Convert binary mask to COCO-compressed RLE dict."""
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    # rle["counts"] is a bytes object → convert to str for JSON
    rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": list(mask.shape), "counts": rle["counts"]}


def _crop_pad_resize_square(
    img,
    box,
    target_size,
    is_mask=False,
):
    """Crop an image or mask by box, pad to square, and resize to target_size."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = map(int, box)

    # Clip box to valid image bounds
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((target_size, target_size), dtype=img.dtype)

    # Crop region
    crop = img[y0:y1, x0:x1]

    # Normalize mask to 0/255 binary if needed
    if is_mask:
        crop = (crop > 0).astype(np.uint8) * 255

    # Pad to square (centered)
    h_c, w_c = crop.shape[:2]
    side = max(h_c, w_c)
    if img.ndim == 2:
        square = np.zeros((side, side), dtype=img.dtype)
    else:
        square = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
    y_off = (side - h_c) // 2
    x_off = (side - w_c) // 2
    square[y_off : y_off + h_c, x_off : x_off + w_c] = crop

    # Resize
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(square, (target_size, target_size), interpolation=interp)

    # Re-binarize masks after resize
    if is_mask:
        resized = (resized > 127).astype(np.uint8) * 255

    return resized


def _generate_augmented_variants(
    base_images,
    rotation_degrees,
    allow_flip_horizontal,
    allow_flip_vertical,
    is_mask=False,
):
    """Generate all rotation + flip variants with canonical."""

    def _canonicalize_transform(rotation, flip_lr, flip_ud):
        # --- Hard-coded dihedral group canonicalization ---
        r = round(rotation / 90.0) * 90 % 360
        if not flip_lr and not flip_ud:
            return (r, "none")
        if flip_lr and flip_ud:
            return ((r + 180) % 360, "none")

        lookup = {
            # flip_lr only
            (0, True, False): (0.0, "hflip"),
            (90, True, False): (0.0, "anti-diag"),
            (180, True, False): (0.0, "vflip"),
            (270, True, False): (0.0, "diag"),
            # flip_ud only
            (0, False, True): (0.0, "vflip"),
            (90, False, True): (0.0, "diag"),
            (180, False, True): (0.0, "hflip"),
            (270, False, True): (0.0, "anti-diag"),
        }
        return lookup.get((int(r), flip_lr, flip_ud), (r, "none"))

    flip_lr_opts = [False, True] if allow_flip_horizontal else [False]
    flip_ud_opts = [False, True] if allow_flip_vertical else [False]

    variants, metadata = [], []
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    for i, base in enumerate(base_images):
        # Dedup on the canonical transform key ONLY. The image-crop and
        # mask-crop invocations must stay index-aligned (HOGFilteringStage
        # zips their outputs); content-hash dedup collapses variants of a
        # symmetric mask that its image does not collapse, desynchronizing
        # the two lists.
        seen_xforms = set()
        for deg in rotation_degrees:
            M = cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), deg, 1.0)
            rotated = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), flags=interp, borderValue=0)

            for flr in flip_lr_opts:
                for fud in flip_ud_opts:
                    canon_rot, canon_flip = _canonicalize_transform(deg, flr, fud)
                    key = (canon_rot, canon_flip)
                    if key in seen_xforms:
                        continue

                    aug = rotated
                    if flr:
                        aug = cv2.flip(aug, 1)
                    if fud:
                        aug = cv2.flip(aug, 0)
                    aug = np.ascontiguousarray(aug)
                    if is_mask:
                        aug = (aug > 127).astype(np.uint8) * 255

                    seen_xforms.add(key)
                    variants.append(aug)
                    metadata.append(
                        {
                            "source": str(i),
                            "rotation": float(deg),
                            "flip_lr": bool(flr),
                            "flip_ud": bool(fud),
                            "canonical": {"rotation": canon_rot, "flip": canon_flip},
                        }
                    )
    return variants, metadata


def _to_serializable(obj):
    """Safely convert numpy / torch / OmegaConf / PIL to serializable JSON form."""
    if dataclasses.is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if hasattr(obj, "tolist"):  # torch tensor
        return obj.tolist()
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode(errors="ignore")
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "size") and hasattr(obj, "mode"):  # PIL.Image
        return {"size": obj.size, "mode": obj.mode}
    return obj


def _compute_hash_dict(d):
    """Return short stable SHA256 hash string for dict."""
    dumped = json.dumps(_to_serializable(d), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


def build_template_box_to_masks_stages(config, roi_generate_models):
    """Return the ordered list of stages for Template-Box-to-Masks pipeline."""
    return [
        TemplatePreparationStage(config, roi_generate_models),
        ProposalGenerationStage(config, roi_generate_models),
        BoxFilteringStage(config),
        SAMInferenceStage(config, roi_generate_models),
        MaskFilteringStage(config),
        HOGFilteringStage(config, device=roi_generate_models.device),
        ColorFilteringStage(config, device=roi_generate_models.device),
        PostProcessStage(config),
    ]


def _quantize_box_to_grid(box, img_w, img_h, grid_w, grid_h):
    """Map an image-space box onto the feature grid, guaranteeing at least
    one cell: a thin box can quantize to an empty slice whose mean() is NaN,
    silently zeroing that template's proposals."""
    x0, y0, x1, y1 = box
    # Clamp the origin into the grid first, then the far edge relative to
    # it — this order stays safe even for out-of-range inputs.
    tx0 = min(max(int(x0 / img_w * grid_w), 0), grid_w - 1)
    ty0 = min(max(int(y0 / img_h * grid_h), 0), grid_h - 1)
    tx1 = min(max(int(x1 / img_w * grid_w), tx0 + 1), grid_w)
    ty1 = min(max(int(y1 / img_h * grid_h), ty0 + 1), grid_h)
    return tx0, ty0, tx1, ty1


# The cache sits under the run's output directory, so anything able to write there could
# otherwise hand ``pickle.load`` code to run on the next resume. The hash in the sibling
# meta JSON is no defence: ``compute_dependency_hash`` covers the stage's inputs, not these
# bytes, so swapping only the .pkl leaves it valid. Restricting the unpickler rather than
# changing the format, matching ``weights_only=True`` in checkpoint/selective_dcp.py.
# Both numpy spellings appear because the package moved in NumPy 2.
_ALLOWED_PICKLE_GLOBALS = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        # What NumPy 2 emits for a contiguous array; _reconstruct/ndarray above still
        # appear for older pickles and other layouts, so both routes stay listed.
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    }
)


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that refuses any global the ROI stage cache does not legitimately store."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        # NumPy 2 dtype classes; naming them individually would turn a precision
        # change into an unloadable cache.
        if module == "numpy.dtypes" and name.endswith("DType"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} while loading the ROI stage cache")


def load_cached_context(stages, ctx, cache_dir):
    """Try to restore cached context. Returns (ctx, start_idx, prev_hash).

    prev_hash is the last verified stage hash: the caller must continue the
    hash chain from it — restarting the chain at None would mismatch every
    stage past the resume point on all subsequent runs."""

    prev_hash = None
    for i in range(len(stages) - 1):
        stage = stages[i]
        cached_hash = stage.load_cache_hash(cache_dir)
        if cached_hash is None:
            return ctx, i, prev_hash
        if cached_hash != stage.compute_dependency_hash(ctx, prev_hash):
            return ctx, i, prev_hash
        ctx[stage.name] = stage.load_cache_result(cache_dir)
        prev_hash = cached_hash
    return ctx, len(stages) - 1, prev_hash


class PipelineStage(ABC):
    """Base pipeline stage with optional visualisation hook."""

    def __init__(self, name: str, config=None):
        self.name = name
        self.config = config
        self.result = {}
        self.deps = {}

    @abstractmethod
    def run(self, ctx):
        raise NotImplementedError

    def save_cache(self, ctx, stage_hash):
        """Save cache by splitting lightweight hash metadata and heavy context."""
        cache_dir = os.path.join(ctx["input"]["output_dir"], "template_box_to_masks", "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # --- (1) Save lightweight hash meta (JSON, human-readable)
        hash_path = os.path.join(cache_dir, f"{self.name}_meta.json")
        with open(hash_path, "w") as f:
            json.dump({"hash": stage_hash}, f)

        # --- (2) Save heavy context data (Pickle)
        result_path = os.path.join(cache_dir, f"{self.name}_result.pkl")
        with open(result_path, "wb") as f:
            pickle.dump(self.result, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_cache_hash(self, cache_dir):
        """Load cached metadata (hash) first; context loaded only if needed."""
        hash_path = os.path.join(cache_dir, f"{self.name}_meta.json")
        result_path = os.path.join(cache_dir, f"{self.name}_result.pkl")

        if not (os.path.exists(hash_path) and os.path.exists(result_path)):
            return None

        # Quick read hash (for fast verification)
        with open(hash_path, "r") as f:
            meta = json.load(f)
        cached_hash = meta.get("hash")

        # Don't load ctx yet — return its path for optional later use
        return cached_hash

    def load_cache_result(self, cache_dir):
        """Load cached context through a restricted unpickler."""
        result_path = os.path.join(cache_dir, f"{self.name}_result.pkl")

        with open(result_path, "rb") as f:
            cached_result = _SafeUnpickler(f).load()

        return cached_result

    def compute_dependency_hash(self, ctx, prev_hash=None):
        """Compute hash of all dependent inputs and config fields."""
        dep_hash = _compute_hash_dict(self.deps)
        if prev_hash:
            dep_hash = hashlib.sha256((prev_hash + dep_hash).encode()).hexdigest()
        return dep_hash

    def save_visualization(self, ctx):
        # Optional hook: a stage with nothing to draw keeps this default.
        pass


class TemplatePreparationStage(PipelineStage):
    """
    Prepare template crops from a list of boxes on a given image:
      1) refine each template box by SAM2
      2) crop each box
      3) pad to square and resize to crop_resize
      4) augment by rotations and flips
    """

    def __init__(self, config, roi_generate_models):
        super().__init__("template_prepare", config)
        self.roi_generate_models = roi_generate_models
        self.refine_template = config.template_box_to_masks.refine_template
        self.crop_resize = config.template_box_to_masks.crop_resize

        allow_flip = config.template_box_to_masks.allow_flip
        self.allow_flip_horizontal = "horizontal" in allow_flip
        self.allow_flip_vertical = "vertical" in allow_flip
        rdeg = config.template_box_to_masks.rotation_degrees
        normalized = sorted({float(d) % 360.0 for d in rdeg})
        if 0.0 not in normalized:
            normalized = [0.0] + normalized
        self.rotation_degrees = normalized

        self.deps = {
            "image_resize": self.config.template_box_to_masks.image_resize,
            "refine_template": self.refine_template,
            "crop_resize": self.crop_resize,
            "allow_flip_horizontal": self.allow_flip_horizontal,
            "allow_flip_vertical": self.allow_flip_vertical,
            "rotation_degrees": self.rotation_degrees,
        }

    def compute_dependency_hash(self, ctx, prev_hash=None):
        """
        Dynamically update dependencies from context before computing hash.
        This ensures inputs like boxes are included in the hash.
        """
        input_for_hash = {
            "image_path": ctx["input"]["image_path"],
            "boxes": ctx["input"]["boxes"],
            "ori_image_size": ctx["input"]["ori_image_size"],
        }
        self.deps["input"] = input_for_hash
        return super().compute_dependency_hash(ctx, prev_hash)

    def run(self, ctx):
        image = ctx["input"]["image"]
        template_boxes = ctx["input"]["boxes"]

        image_np = to_rgb_uint8(image)
        if self.refine_template:
            refined_template_boxes = self._refine_template_boxes(template_boxes, image_np)
        else:
            refined_template_boxes = template_boxes

        template_crops = [_crop_pad_resize_square(image_np, b, self.crop_resize) for b in refined_template_boxes]
        aug_template_crops, transforms = _generate_augmented_variants(
            template_crops, self.rotation_degrees, self.allow_flip_horizontal, self.allow_flip_vertical
        )

        self.result.update(
            {
                "refined_template_boxes": refined_template_boxes,
                "template_crops": template_crops,
                "aug_template_crops": aug_template_crops,
                "transforms": transforms,
            }
        )
        ctx[self.name] = self.result
        return ctx

    def _refine_template_boxes(self, boxes, image_np):
        if not boxes:
            return boxes

        refined_boxes = []
        for box in boxes:
            # --- Run SAM2 segmentation ---
            masks, _ = self.roi_generate_models.forward_segmentation(image_np, boxes=[box])
            mask = masks[0].astype(np.uint8)

            # --- Compute new bbox from mask ---
            ys, xs = np.nonzero(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                refined_boxes.append(box)  # fallback to original
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            refined_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        return refined_boxes

    def save_visualization(self, ctx):
        out_dir = os.path.join(
            ctx["input"]["output_dir"],
            "template_box_to_masks",
            "visualization",
            "template",
        )
        os.makedirs(out_dir, exist_ok=True)

        image = ctx["input"]["image"]
        boxes = self.result["refined_template_boxes"]
        crops = self.result["aug_template_crops"]
        transforms = self.result["transforms"]

        # ---- Draw refined boxes ----
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        img.save(os.path.join(out_dir, "refined_template_boxes.png"))

        # ---- No aug crops, stop ----
        if not crops:
            return

        # Prepare grouping by template source
        per_box = defaultdict(list)
        for crop, tr in zip(crops, transforms):
            per_box[tr["source"]].append((crop, tr))

        font = ImageFont.load_default()
        MAX_COLS = 8

        # ---- Build crop grids ----
        for key, items in per_box.items():
            if not items:
                continue

            h, w = items[0][0].shape[:2]
            cols = min(MAX_COLS, len(items))
            rows = (len(items) + cols - 1) // cols
            canvas = Image.new("RGB", (cols * w, rows * h), "white")

            for idx, (crop_np, tr) in enumerate(items):
                r, c = divmod(idx, cols)
                crop_img = Image.fromarray(crop_np)
                x, y = c * w, r * h
                canvas.paste(crop_img, (x, y))

                # draw annotation
                deg = tr["rotation"]
                flip_lr = "L" if tr["flip_lr"] else "-"
                flip_ud = "U" if tr["flip_ud"] else "-"

                cdeg = tr["canonical"]["rotation"]
                cflip = tr["canonical"]["flip"]
                txt = f"rot{deg} {flip_lr}/{flip_ud} | canon rot{cdeg},{cflip}"
                ImageDraw.Draw(canvas).text((x + 4, y + 4), txt, fill=(255, 255, 255), font=font)

            canvas.save(os.path.join(out_dir, f"aug_template_box_{key}.png"))


class ProposalGenerationStage(PipelineStage):
    """
    Generates proposal boxes for a given image through a multi-step pipeline:

      1) Extract the image feature map.
      2) Crop the feature map using the template box to obtain the template feature map.
      3) Apply average pooling to the template feature map to derive the template feature vector.
      4) Identify similar locations on the full image feature map based on the template feature.
      5) Run SAM2 on the detected similar points to obtain instance masks and convert them into proposal boxes.
    """

    def __init__(self, config, roi_generate_models):
        super().__init__("proposal_generation", config)
        self.roi_generate_models = roi_generate_models
        self.max_proposal = config.template_box_to_masks.max_proposal
        self.proposal_similarity_tol = config.template_box_to_masks.proposal_similarity_tol
        self.nms_iou_threshold = config.template_box_to_masks.nms_iou_threshold
        self.proposal_seed = config.template_box_to_masks.proposal_seed

        self.deps = {
            "max_proposal": self.max_proposal,
            "proposal_similarity_tol": self.proposal_similarity_tol,
            "nms_iou_threshold": self.nms_iou_threshold,
            "proposal_seed": self.proposal_seed,
        }

    def run(self, ctx):
        image = ctx["input"]["image"]
        template_boxes = ctx["input"]["boxes"]

        feat = self.roi_generate_models.forward_feature_map(
            image,
        )

        feat_upsampled = torch.nn.functional.interpolate(
            feat.unsqueeze(0),
            size=(512, 512),
            mode="bilinear",
            align_corners=False,  # add batch dim
        )[0]
        feat = torch.nn.functional.normalize(feat_upsampled, dim=0)
        _, H, W = feat.shape

        img_w, img_h = image.size
        proposal_boxes, proposal_scores = [], []
        # Seeded generator: identical inputs must yield identical proposals,
        # or the dependency-hash cache can match while the real output would
        # differ. The seed participates in self.deps for the same reason.
        rng = np.random.default_rng(self.proposal_seed)

        def proposal_with_segmentation(prompts):
            for p in prompts:
                masks, scores = self.roi_generate_models.forward_segmentation(image, **p)
                if masks is None:
                    continue
                for m_idx, m in enumerate(masks):
                    if m is None or m.sum() == 0:
                        continue
                    ys, xs = np.nonzero(m > 0)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                    if x1 <= x0 or y1 <= y0:
                        continue
                    # scale back to original resolution
                    x0, x1 = x0 / m.shape[1] * img_w, x1 / m.shape[1] * img_w
                    y0, y1 = y0 / m.shape[0] * img_h, y1 / m.shape[0] * img_h
                    proposal_boxes.append([x0, y0, x1, y1])
                    proposal_scores.append(scores[m_idx])

        # Process each template box
        for x0, y0, x1, y1 in template_boxes:
            tx0, ty0, tx1, ty1 = _quantize_box_to_grid((x0, y0, x1, y1), img_w, img_h, W, H)

            # Extract template feature vector
            template_feat = feat[:, ty0:ty1, tx0:tx1].mean(dim=(1, 2))
            template_feat = torch.nn.functional.normalize(template_feat, dim=0)

            # Compute similarity map (dot product)
            sim_map = torch.einsum("chw,c->hw", feat, template_feat)
            sim_np = sim_map.cpu().numpy()

            # Generate candidate points from high-similarity regions
            top_percent = 100 * (1.0 - self.proposal_similarity_tol)
            thresh = np.percentile(sim_np, top_percent)
            ys, xs = np.nonzero(sim_np >= thresh)
            if len(xs) == 0:
                continue

            # Random subsample
            max_pts = self.max_proposal
            if len(xs) > max_pts:
                selected_ids = rng.choice(len(xs), max_pts, replace=False)
                xs, ys = xs[selected_ids], ys[selected_ids]

            # Convert to original image coordinates
            point_coords = np.stack([xs / W * img_w, ys / H * img_h], axis=1)
            # Removes duplicates
            _, unique_idx = np.unique(point_coords.round(decimals=0), axis=0, return_index=True)
            point_coords = point_coords[unique_idx]
            # Prepare for SAM
            point_coords = point_coords.tolist()

            # Generate proposals using SAM2 + point prompts
            proposal_with_segmentation(
                [{"point_coords": [p], "point_labels": [1], "boxes": None} for p in point_coords]
            )

            # Generate proposals using SAM2 + box prompts
            tpl_w, tpl_h = x1 - x0, y1 - y0
            w, h = tpl_w * 1.1, tpl_h * 1.1
            local_boxes = [
                [
                    max(0, cx - w / 2),
                    max(0, cy - h / 2),
                    min(img_w, cx + w / 2),
                    min(img_h, cy + h / 2),
                ]
                for (cx, cy) in point_coords
            ]
            proposal_with_segmentation(
                [{"boxes": [b], "point_coords": None, "point_labels": None} for b in local_boxes]
            )

        if not proposal_boxes:
            # Raise here with the descriptive message: an empty (0,) tensor
            # would make torchvision nms fail with a cryptic shape error
            # before SAMInferenceStage's friendly guard is ever reached.
            raise ValueError("No proposal boxes found, please check the proposal generation parameters.")

        proposal_boxes_np = np.array(proposal_boxes)
        proposal_scores_np = np.array(proposal_scores)
        proposal_boxes_tensor = torch.tensor(proposal_boxes_np, dtype=torch.float32)
        proposal_scores_tensor = torch.tensor(proposal_scores_np, dtype=torch.float32)

        keep_idx = nms(proposal_boxes_tensor, proposal_scores_tensor, iou_threshold=self.nms_iou_threshold)
        keep_idx_np = keep_idx.cpu().numpy()

        nms_boxes = proposal_boxes_np[keep_idx_np]
        nms_confidences = proposal_scores_np[keep_idx_np]

        if len(nms_boxes) > self.max_proposal:
            sorted_idx = np.argsort(nms_confidences)[::-1][: self.max_proposal]
            nms_boxes = nms_boxes[sorted_idx]
            nms_confidences = nms_confidences[sorted_idx]

        self.result.update({"proposal_boxes": nms_boxes, "proposal_confidences": nms_confidences})
        ctx[self.name] = self.result
        return ctx


class BoxFilteringStage(PipelineStage):
    """
    Compute the aspect-ratio difference and size (diagonal length) difference relative to the
    best-matched template box.
    """

    def __init__(self, config):
        super().__init__("box_filter", config)
        rdeg = config.template_box_to_masks.rotation_degrees
        normalized = sorted({float(d) % 360.0 for d in rdeg})
        if 0.0 not in normalized:
            normalized = [0.0] + normalized
        self.rotation_degrees = normalized

    def run(self, ctx):
        proposal_boxes = ctx["proposal_generation"]["proposal_boxes"]
        template_boxes = ctx["template_prepare"]["refined_template_boxes"]

        aug_templates = []
        for tb in template_boxes:
            x0, y0, x1, y1 = tb
            w = max(x1 - x0, 1e-6)
            h = max(y1 - y0, 1e-6)
            s = max(w, h) + 1e-6

            for deg in self.rotation_degrees:
                rw, rh = self._rotated_box_size(w, h, deg)
                aspect = rw / rh + 1e-6
                s_r = max(rw, rh) + 1e-6
                aug_templates.append((aspect, s_r))

        aspect_diff = []
        size_diff = []

        for proposal_box in proposal_boxes:
            x0, y0, x1, y1 = proposal_box
            w = max(x1 - x0, 1e-6)
            h = max(y1 - y0, 1e-6)
            aspect = w / h + 1e-6
            s = max(w, h) + 1e-6

            best_aspect_diff = float("inf")
            best_size_diff = float("inf")

            for tpl_aspect, tpl_s in aug_templates:
                diff_aspect = abs(aspect - tpl_aspect) / tpl_aspect
                diff_s = abs(s - tpl_s) / tpl_s

                if diff_aspect < best_aspect_diff:
                    best_aspect_diff = diff_aspect
                if diff_s < best_size_diff:
                    best_size_diff = diff_s

            aspect_diff.append(best_aspect_diff)
            size_diff.append(best_size_diff)

        self.result.update({"aspect_diff": np.array(aspect_diff), "size_diff": np.array(size_diff)})
        ctx[self.name] = self.result
        return ctx

    @staticmethod
    def _rotated_box_size(w, h, degree):
        """Compute the axis-aligned bounding box size after rotating a rectangle by an arbitrary degree."""
        rad = math.radians(degree % 360.0)
        cos_r = abs(math.cos(rad))
        sin_r = abs(math.sin(rad))
        rw = w * cos_r + h * sin_r
        rh = w * sin_r + h * cos_r
        return max(rw, 1e-6), max(rh, 1e-6)


class SAMInferenceStage(PipelineStage):
    """
    Runs SAM2 to generate instance masks from proposal boxes:
    """

    def __init__(self, config, roi_generate_models):
        super().__init__("sam_inference", config)
        self.roi_generate_models = roi_generate_models

    def run(self, ctx):
        proposal_boxes = ctx["proposal_generation"]["proposal_boxes"]
        template_boxes = ctx["template_prepare"]["refined_template_boxes"]
        image = ctx["input"]["image"]

        template_masks, _ = self.roi_generate_models.forward_segmentation(
            image=image,
            boxes=template_boxes,
        )
        if len(proposal_boxes) == 0:
            raise ValueError("No proposal boxes found, please check the proposal generation parameters.")
        else:
            candidate_masks, _ = self.roi_generate_models.forward_segmentation(
                image=image,
                boxes=proposal_boxes,
            )
        self.result.update({"template_masks": template_masks, "candidate_masks": candidate_masks})
        ctx[self.name] = self.result
        return ctx

    def save_visualization(self, ctx):
        out_dir = os.path.join(
            ctx["input"]["output_dir"],
            "template_box_to_masks",
            "visualization",
            "sam_inference",
        )
        os.makedirs(out_dir, exist_ok=True)

        image = ctx["input"]["image"]
        image_np = to_rgb_uint8(image)
        H, W = image_np.shape[:2]

        template_masks = self.result["template_masks"]
        candidate_masks = self.result["candidate_masks"]

        # -------- helper: overlay mask list on image -------- #
        def overlay(image_np, masks, fixed_color=None, alpha=0.5):
            img = image_np.astype(np.float32).copy()
            for i, m in enumerate(masks):
                color = (
                    np.array(fixed_color, dtype=np.float32) if fixed_color is not None else np.random.rand(3)
                ) * 255
                mask = (m > 0)[..., None]
                img[mask[..., 0]] = (1 - alpha) * img[mask[..., 0]] + alpha * color
            return Image.fromarray(img.astype(np.uint8))

        # -------- Template masks (blue) -------- #
        blue = np.array([0, 100, 255], dtype=np.float32) / 255.0
        out = overlay(image_np, template_masks, fixed_color=blue)
        out.save(os.path.join(out_dir, "template_masks.png"))

        # -------- Candidate masks (random colors) -------- #
        out = overlay(image_np, candidate_masks, fixed_color=None)
        out.save(os.path.join(out_dir, "candidate_masks.png"))


class MaskFilteringStage(PipelineStage):
    """
    Compute the connected-component count and the Contour Chamfer distance difference.
    """

    def __init__(self, config):
        super().__init__("mask_filter", config)
        self.crop_resize = config.template_box_to_masks.crop_resize
        allow_flip = config.template_box_to_masks.allow_flip
        self.allow_flip_horizontal = "horizontal" in allow_flip
        self.allow_flip_vertical = "vertical" in allow_flip

        rdeg = config.template_box_to_masks.rotation_degrees
        normalized = sorted({float(d) % 360.0 for d in rdeg})
        if 0.0 not in normalized:
            normalized = [0.0] + normalized
        self.rotation_degrees = normalized

    def run(self, ctx):
        template_masks = ctx["sam_inference"]["template_masks"]
        template_boxes = ctx["template_prepare"]["refined_template_boxes"]
        candidate_masks = ctx["sam_inference"]["candidate_masks"]
        candidate_boxes = ctx["proposal_generation"]["proposal_boxes"]

        component_diffs = []
        chamfer_score = []

        # Compute connected component
        template_components = [self._component_count(m) for m in template_masks]

        for candidate_mask in candidate_masks:
            component_component = self._component_count(candidate_mask)
            best_component_diff = float("inf")
            for template_component in template_components:
                component_diff = abs(template_component - component_component)
                if component_diff < best_component_diff:
                    best_component_diff = component_diff
            component_diffs.append(best_component_diff)

        component_diffs = np.array(component_diffs)

        # Template mask augmentation before compute chamfer distance
        template_mask_crops = [
            _crop_pad_resize_square(m, b, self.crop_resize, is_mask=True)
            for m, b in zip(template_masks, template_boxes)
        ]
        aug_template_mask_crops, transforms = _generate_augmented_variants(
            template_mask_crops,
            self.rotation_degrees,
            self.allow_flip_horizontal,
            self.allow_flip_vertical,
            is_mask=True,
        )

        # Compute chamfer distance
        template_dts = []
        template_contours = []
        for template_mask_crop in aug_template_mask_crops:
            template_contour = self._extract_contour(template_mask_crop)
            template_dt = self._distance_transform(template_contour)
            template_contours.append(template_contour)
            template_dts.append(template_dt)

        chamfer_distances = []
        candidate_mask_crops = []
        for candidate_mask, candidate_box in zip(candidate_masks, candidate_boxes):
            candidate_mask_crop = _crop_pad_resize_square(candidate_mask, candidate_box, self.crop_resize, is_mask=True)
            candidate_contour = self._extract_contour(candidate_mask_crop)
            candidate_dt = self._distance_transform(candidate_contour)
            candidate_mask_crops.append(candidate_mask_crop)

            best_chamfer = float("inf")
            for template_dt, template_contour in zip(template_dts, template_contours):
                c2t = template_dt[candidate_contour == 1]
                t2c = candidate_dt[template_contour == 1]
                if c2t.size == 0 or t2c.size == 0:
                    continue
                chamfer = 0.5 * (c2t.mean() + t2c.mean())
                if chamfer < best_chamfer:
                    best_chamfer = chamfer
            chamfer_distances.append(best_chamfer)
        diag = self.crop_resize * (2**0.5)
        chamfer_score = np.array(chamfer_distances) / diag

        self.result.update(
            {
                "template_mask_crops": template_mask_crops,
                "aug_template_mask_crops": aug_template_mask_crops,
                "transforms": transforms,
                "chamfer_score": np.array(chamfer_score),
                "component_diffs": np.array(component_diffs),
            }
        )
        ctx[self.name] = self.result
        return ctx

    @staticmethod
    def _extract_contour(binary_mask):
        kernel = np.ones((3, 3), dtype=np.uint8)
        contour = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        contour = (contour > 0).astype(np.uint8)
        if contour.sum() == 0:
            edges = cv2.Canny(binary_mask.astype(np.uint8) * 255, 50, 150)
            contour = (edges > 0).astype(np.uint8)
        return contour

    @staticmethod
    def _distance_transform(contour_mask):
        """Compute a distance map where each background pixel stores the distance to the nearest foreground pixel."""
        if contour_mask is None:
            return None
        inv = (contour_mask == 0).astype(np.uint8)
        return cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    @staticmethod
    def _component_count(binary_mask):
        mask_uint8 = (binary_mask > 0).astype(np.uint8)
        if mask_uint8.sum() == 0:
            return 0
        num_labels, _ = cv2.connectedComponents(mask_uint8, connectivity=8)
        return max(num_labels - 1, 0)


class HOGFilteringStage(PipelineStage):
    """
    Filters candidate instances using HOG feature similarity:

      1) Extract HOG descriptors from the template region.
      2) Extract HOG descriptors from each candidate region.
      3) Compute feature distance or similarity scores.
    """

    def __init__(self, config, device="cuda"):
        super().__init__("hog_filter", config)
        self.device = device

        self.orientations = config.template_box_to_masks.orientations
        self.pixels_per_cell = config.template_box_to_masks.pixels_per_cell
        self.cells_per_block = config.template_box_to_masks.cells_per_block
        self.deps = {
            "orientations": self.orientations,
            "pixels_per_cell": self.pixels_per_cell,
            "cells_per_block": self.cells_per_block,
        }

    def run(self, ctx):
        image = ctx["input"]["image"]

        aug_template_crops = ctx["template_prepare"]["aug_template_crops"]
        aug_template_mask_crops = ctx["mask_filter"]["aug_template_mask_crops"]

        if len(aug_template_crops) != len(aug_template_mask_crops):
            raise ValueError(
                f"Template image/mask augmentation lists desynchronized "
                f"({len(aug_template_crops)} images vs "
                f"{len(aug_template_mask_crops)} masks); both stages must "
                f"generate variants with identical transform configs."
            )

        aug_template_masked_crops = np.array(
            [
                img * (mask // 255)[..., None]  # expand mask to (H, W, 1)
                for img, mask in zip(aug_template_crops, aug_template_mask_crops)
            ]
        )

        candidate_masks = ctx["sam_inference"]["candidate_masks"]
        candidate_boxes = ctx["proposal_generation"]["proposal_boxes"]
        crop_resize = self.config.template_box_to_masks.crop_resize
        candidate_masked_crops = np.array(
            [
                _crop_pad_resize_square(image * mask[..., None], box, crop_resize)
                for mask, box in zip(candidate_masks, candidate_boxes)
            ]
        )

        sim_hog = self._hog_sim(aug_template_masked_crops, candidate_masked_crops)

        self.result.update({"sim_hog": np.array(sim_hog)})
        ctx[self.name] = self.result
        return ctx

    def _hog_sim(self, template_images, candidate_images):
        def to_gray_float32(image):
            """Convert any RGB/gray image to float32 grayscale [0, 1]."""
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.ndim == 3 and image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:  # already single-channel
                gray = image.squeeze()
            return gray.astype(np.float32) / 255.0

        # Convert all images to grayscale
        template_grays = [to_gray_float32(img) for img in template_images]
        candidate_grays = [to_gray_float32(img) for img in candidate_images]

        # Batch compute HOG features on GPU
        template_vecs = self._compute_hog(template_grays, self.orientations, self.pixels_per_cell, self.cells_per_block)
        candidate_vecs = self._compute_hog(
            candidate_grays, self.orientations, self.pixels_per_cell, self.cells_per_block
        )

        # Compute cosine similarity on GPU
        template_vecs_tensor = torch.tensor(template_vecs, dtype=torch.float32, device=self.device)
        candidate_vecs_tensor = torch.tensor(candidate_vecs, dtype=torch.float32, device=self.device)

        # Normalize vectors
        template_vecs_norm = torch.nn.functional.normalize(template_vecs_tensor, p=2, dim=1)
        candidate_vecs_norm = torch.nn.functional.normalize(candidate_vecs_tensor, p=2, dim=1)

        # Compute all similarities at once: (N_candidates, N_templates)
        similarities = torch.mm(candidate_vecs_norm, template_vecs_norm.t())

        # Find best template for each candidate
        best_sims, best_tpl_indices = torch.max(similarities, dim=1)

        # Move results back to CPU
        best_sims_cpu = best_sims.cpu().numpy()
        best_tpl_indices_cpu = best_tpl_indices.cpu().numpy()

        similarity_list = []
        for idx in range(len(candidate_images)):
            best_sim = float(best_sims_cpu[idx])
            best_tpl = int(best_tpl_indices_cpu[idx])
            similarity_list.append(best_sim)

        return similarity_list

    def _compute_hog(self, gray_images, orientations, pixels_per_cell, cells_per_block):
        """
        Compute HOG features for grayscale images.

        Args:
            gray_images: List of grayscale images as numpy arrays [0, 1]
            orientations: Number of orientation bins
            pixels_per_cell: Tuple (height, width) of cell size
            cells_per_block: Tuple (height, width) of block size

        Returns:
            vecs: numpy array of shape (N, feature_dim)
            vis_images: list of visualization images
        """
        if len(gray_images) == 0:
            return np.array([]), []

        vecs = []
        # Batch process to reduce overhead
        for gray in gray_images:
            vec = hog(
                gray,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True,
                visualize=False,
            )
            vecs.append(vec)

        return np.stack(vecs)


class ColorFilteringStage(PipelineStage):
    """
    Filters candidate regions based on color similarity:

      1) Compute the template's color histograms (color, lightness).
      2) Extract color histograms from each candidate region.
      3) Measure histogram distance between the candidate and the template.
    """

    def __init__(self, config, device="cuda"):
        super().__init__("color_filter", config)
        self.device = device
        self.color_bins = config.template_box_to_masks.color_bins
        self.lightness_bins = config.template_box_to_masks.lightness_bins
        self.crop_resize = config.template_box_to_masks.crop_resize
        self.deps = {
            "lightness_bins": self.lightness_bins,
            "color_bins": self.color_bins,
        }

    def run(self, ctx):
        image = ctx["input"]["image"]
        template_crops = ctx["template_prepare"]["template_crops"]
        template_mask_crops = ctx["mask_filter"]["template_mask_crops"]

        template_masked_crops = np.array(
            [
                img * (mask // 255)[..., None]  # expand mask to (H, W, 1)
                for img, mask in zip(template_crops, template_mask_crops)
            ]
        )

        candidate_masks = ctx["sam_inference"]["candidate_masks"]
        candidate_boxes = ctx["proposal_generation"]["proposal_boxes"]
        candidate_masked_crops = np.array(
            [
                _crop_pad_resize_square(image * mask[..., None], box, self.crop_resize)
                for mask, box in zip(candidate_masks, candidate_boxes)
            ]
        )

        sim_lightness, sim_color = self._color_hist_filter(template_masked_crops, candidate_masked_crops)

        self.result.update({"sim_lightness": np.array(sim_lightness), "sim_color": np.array(sim_color)})
        ctx[self.name] = self.result
        return ctx

    def _color_hist_filter(self, template_images, candidate_images):
        def compute_color_hist(image, ignore_black=True, black_thresh=0):
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            mask = None
            if ignore_black:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                mask = (gray > black_thresh).astype(np.uint8)

            hist_L = cv2.calcHist([lab], [0], mask, [self.lightness_bins], [0, 255])
            hist_L = gaussian_filter1d(hist_L, sigma=1.0)
            hist_L += 1e-6
            cv2.normalize(hist_L, hist_L, alpha=1, norm_type=cv2.NORM_L1)

            hist_AB = cv2.calcHist([lab], [1, 2], mask, [self.color_bins, self.color_bins], [-128, 127, -128, 127])
            hist_AB = gaussian_filter1d(hist_AB, sigma=1.5)
            hist_AB += 1e-6
            cv2.normalize(hist_AB, hist_AB, alpha=1, norm_type=cv2.NORM_L1)

            return hist_L, hist_AB

        # Precompute template histograms
        template_hists = [compute_color_hist(img) for img in template_images]

        # Compute candidate histograms
        candidate_hists = [compute_color_hist(img) for img in candidate_images]

        # Batch compute Bhattacharyya distances on GPU
        sim_lightness_list, sim_color_list, best_tpl_indices = self._compute_bhattacharyya_batch(
            template_hists, candidate_hists
        )

        return sim_lightness_list, sim_color_list

    def _compute_bhattacharyya_batch(self, template_hists, candidate_hists):
        """
        Compute Bhattacharyya distance between all candidate and template histograms on GPU.

        Bhattacharyya distance: d = sqrt(1 - sum(sqrt(p * q)))
        where p and q are normalized histograms.

        Args:
            template_hists: List of (hist_L, hist_AB) tuples for templates
            candidate_hists: List of (hist_L, hist_AB) tuples for candidates

        Returns:
            sim_lightness_list: Best lightness similarity for each candidate
            sim_color_list: Best color similarity for each candidate
            best_tpl_indices: Best template index for each candidate
        """
        if len(candidate_hists) == 0 or len(template_hists) == 0:
            return [], [], []

        # Extract and flatten histograms
        # Lightness: (N, lightness_bins)
        template_L = np.array([h[0].flatten() for h in template_hists], dtype=np.float32)
        candidate_L = np.array([h[0].flatten() for h in candidate_hists], dtype=np.float32)

        # Color: (N, color_bins * color_bins)
        template_AB = np.array([h[1].flatten() for h in template_hists], dtype=np.float32)
        candidate_AB = np.array([h[1].flatten() for h in candidate_hists], dtype=np.float32)

        # Move to GPU
        template_L_tensor = torch.tensor(template_L, dtype=torch.float32, device=self.device)
        candidate_L_tensor = torch.tensor(candidate_L, dtype=torch.float32, device=self.device)
        template_AB_tensor = torch.tensor(template_AB, dtype=torch.float32, device=self.device)
        candidate_AB_tensor = torch.tensor(candidate_AB, dtype=torch.float32, device=self.device)

        # Compute Bhattacharyya coefficient (BC) for all pairs
        # BC = sum(sqrt(p * q)) for normalized histograms p, q
        # Distance = sqrt(1 - BC)
        # Similarity = 1 - Distance = 1 - sqrt(1 - BC)

        # Lightness: (N_candidates, N_templates)
        # Using matrix multiplication: sqrt(p) @ sqrt(q).T
        template_L_sqrt = torch.sqrt(template_L_tensor + 1e-10)  # Add epsilon for numerical stability
        candidate_L_sqrt = torch.sqrt(candidate_L_tensor + 1e-10)
        bc_L = torch.mm(candidate_L_sqrt, template_L_sqrt.t())  # (N_cand, N_tpl)
        bc_L = torch.clamp(bc_L, 0.0, 1.0)  # Clamp to [0, 1] for numerical stability
        dist_L = torch.sqrt(1.0 - bc_L)
        sim_L = 1.0 - dist_L  # (N_cand, N_tpl)

        # Color: (N_candidates, N_templates)
        template_AB_sqrt = torch.sqrt(template_AB_tensor + 1e-10)
        candidate_AB_sqrt = torch.sqrt(candidate_AB_tensor + 1e-10)
        bc_AB = torch.mm(candidate_AB_sqrt, template_AB_sqrt.t())  # (N_cand, N_tpl)
        bc_AB = torch.clamp(bc_AB, 0.0, 1.0)
        dist_AB = torch.sqrt(1.0 - bc_AB)
        sim_AB = 1.0 - dist_AB  # (N_cand, N_tpl)

        # Combined similarity (average of lightness and color)
        sim_combined = (sim_L + sim_AB) / 2.0  # (N_cand, N_tpl)

        # Find best template for each candidate
        _, best_tpl_indices = torch.max(sim_combined, dim=1)

        # Get the corresponding lightness and color similarities
        # Use advanced indexing to select the best template's similarity for each candidate
        batch_indices = torch.arange(len(candidate_hists), device=self.device)
        best_sim_L = sim_L[batch_indices, best_tpl_indices]
        best_sim_AB = sim_AB[batch_indices, best_tpl_indices]

        # Move results back to CPU
        sim_lightness_list = best_sim_L.cpu().numpy().tolist()
        sim_color_list = best_sim_AB.cpu().numpy().tolist()
        best_tpl_indices_cpu = best_tpl_indices.cpu().numpy()

        return sim_lightness_list, sim_color_list, best_tpl_indices_cpu


class PostProcessStage(PipelineStage):
    """
    Drops candidates that exceed tolerance thresholds.
    Perform morphological refinement to generate the final binary mask.
    """

    def __init__(self, config):
        super().__init__("post_process", config)
        # Box Filter
        self.box_enabled = config.template_box_to_masks.box_enabled
        self.size_tol = config.template_box_to_masks.size_tol
        self.aspect_tol = config.template_box_to_masks.aspect_tol

        # Mask Filter
        self.mask_enabled = config.template_box_to_masks.mask_enabled
        self.component_count_tol = config.template_box_to_masks.component_count_tol
        self.chamfer_tol = config.template_box_to_masks.chamfer_tol

        # HOG Filter
        self.hog_enabled = config.template_box_to_masks.hog_enabled
        self.hog_similarity_tol = config.template_box_to_masks.hog_similarity_tol

        # Color Histogram Filter
        self.color_hist_enabled = config.template_box_to_masks.color_hist_enabled
        self.color_tol = config.template_box_to_masks.color_tol
        self.lightness_tol = config.template_box_to_masks.lightness_tol

        # Morphological Refinement
        self.kernel_size = config.morphological_kernel
        self.op_name = config.morphological_operation

    def run(self, ctx):
        candidate_masks = ctx["sam_inference"]["candidate_masks"]

        if candidate_masks is None or len(candidate_masks) == 0:
            raise ValueError("No mask after SAM2 inference.")
        else:
            # Merge all instance masks
            keep_mask_list = []

            if self.box_enabled:
                keep_box_size = ctx["box_filter"]["size_diff"] <= self.size_tol
                keep_box_aspect = ctx["box_filter"]["aspect_diff"] <= self.aspect_tol
                keep_mask_list.append(keep_box_size & keep_box_aspect)

            if self.mask_enabled:
                keep_component = ctx["mask_filter"]["component_diffs"] <= self.component_count_tol
                keep_chamfer = ctx["mask_filter"]["chamfer_score"] <= self.chamfer_tol
                keep_mask_list.append(keep_component & keep_chamfer)

            if self.hog_enabled:
                keep_hog = ctx["hog_filter"]["sim_hog"] >= 1 - self.hog_similarity_tol
                keep_mask_list.append(keep_hog)

            if self.color_hist_enabled:
                keep_lightness = ctx["color_filter"]["sim_lightness"] >= 1 - self.lightness_tol
                keep_color = ctx["color_filter"]["sim_color"] >= 1 - self.color_tol
                keep_mask_list.append(keep_lightness & keep_color)

            # Combine all enabled filters
            if len(keep_mask_list) > 0:
                keep_mask = np.logical_and.reduce(keep_mask_list)
            else:
                # If no filter enabled, keep all
                keep_mask = np.ones(len(candidate_masks), dtype=bool)

            filtered_masks = candidate_masks[keep_mask]
            filtered_masks = np.concatenate([filtered_masks, ctx["sam_inference"]["template_masks"]], axis=0)

        binary_mask = np.any(filtered_masks, axis=0).astype(np.uint8)  # binary mask with values {0, 1}

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        if self.op_name:
            if self.op_name == "close":
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            elif self.op_name == "open":
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            elif self.op_name == "dilate":
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            elif self.op_name == "erode":
                binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
            else:
                raise ValueError(
                    f"Unsupported morphological operation '{self.op_name}'. Supported ops: close, open, dilate, erode"
                )

        binary_mask = (binary_mask >= 0.5).astype(np.uint8) * 255  # binary mask with values {0, 255}

        self.result.update({"binary_mask": binary_mask})
        return ctx

    def save_result(self, ctx):
        """
        Save:
          - <output_dir>/template_box_to_masks/output/binary_mask.png
          - <output_dir>/template_box_to_masks/output/result.json
        """
        output_dir = os.path.join(ctx["input"]["output_dir"], "template_box_to_masks")
        binary_mask = self.result["binary_mask"]

        ori_w, ori_h = ctx["input"]["ori_image_size"]
        binary_mask = cv2.resize(binary_mask.astype(np.uint8), (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        output_dir = os.path.join(output_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "binary_mask.png")
        cv2.imwrite(output_path, binary_mask)

        image_path = ctx["input"]["image_path"]
        proc_w, proc_h = ctx["input"]["image"].size
        # All box/mask fields below are in the processed (resized) space of
        # processed_image_size, except "boxes", which is derived from the
        # mask after it is resized back to original_image_size. The two size
        # fields let consumers convert between the spaces.
        item = {
            "image_path": image_path,
            "original_image_size": [ori_w, ori_h],
            "processed_image_size": [proc_w, proc_h],
            "template_boxes": ctx["input"]["boxes"],
            "refined_template_boxes": ctx["template_prepare"]["refined_template_boxes"],
            "template_masks": [_mask_to_compressed_rle(m) for m in ctx["sam_inference"]["template_masks"]],
            "candidate_boxes": ctx["proposal_generation"]["proposal_boxes"],
            "candidate_masks": [_mask_to_compressed_rle(m) for m in ctx["sam_inference"]["candidate_masks"]],
            "size_diff": ctx["box_filter"]["size_diff"],
            "aspect_diff": ctx["box_filter"]["aspect_diff"],
            "component_diffs": ctx["mask_filter"]["component_diffs"],
            "chamfer_score": ctx["mask_filter"]["chamfer_score"],
            "sim_hog": ctx["hog_filter"]["sim_hog"],
            "sim_lightness": ctx["color_filter"]["sim_lightness"],
            "sim_color": ctx["color_filter"]["sim_color"],
            "boxes": [],
        }
        labeled_mask = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
        num_labels, _, stats, _ = labeled_mask
        for i in range(1, num_labels):  # skip background label 0
            x, y, w, h, _ = stats[i]
            item["boxes"].append([int(x), int(y), int(x + w), int(y + h)])

        # save JSON
        json_path = os.path.join(output_dir, "result.json")
        with open(json_path, "w") as f:
            json.dump(to_jsonable(item), f, indent=2)

    def save_visualization(self, ctx):
        out_dir = os.path.join(
            ctx["input"]["output_dir"],
            "template_box_to_masks",
            "visualization",
        )
        os.makedirs(out_dir, exist_ok=True)

        image_np = to_rgb_uint8(ctx["input"]["image"])
        mask = self.result["binary_mask"] > 0

        overlay = image_np.astype(np.float32)
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0], dtype=np.float32)

        out = Image.fromarray(overlay.astype(np.uint8))
        out.save(os.path.join(out_dir, "overlay_mask.png"))
