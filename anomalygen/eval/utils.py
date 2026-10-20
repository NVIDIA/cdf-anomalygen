# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared loaders over the SDG generation output tree, used by the texture evaluate/filter CLIs.

Both consume the layout produced by ``generate.py``:

  gen_root:  {gen}/reconstructed_image/{texture}+{defect}_{idx}.png
             {gen}/original_mask/{texture}+{defect}_{idx}.png
  real_root: {texture}/anomaly_image/{defect}/<stem>.png
             {texture}/mask/{defect}/<stem>_mask.png
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.data.utils import list_image_mask_pairs, require_same_size, validate_anomaly_type

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
RECON_SUBDIR = "reconstructed_image"
MASK_SUBDIR = "original_mask"
ORIG_IMAGE_SUBDIR = "original_image"


def _img_to_float(img: Image.Image, target_size: Optional[int] = None) -> np.ndarray:
    img = img.convert("RGB")
    if target_size is not None:
        img = img.resize((target_size, target_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def _mask_to_float(img: Image.Image, target_size: Optional[int] = None) -> np.ndarray:
    img = img.convert("L")
    if target_size is not None:
        img = img.resize((target_size, target_size))
    mask = np.asarray(img, dtype=np.float32) / 255.0
    if target_size is not None:
        # Resizing interpolates; re-binarize so masks stay {0, 1} (matches the old loader).
        mask = np.where(mask >= 0.5, 1.0, 0.0).astype(np.float32)
    return mask


def infer_anomaly_types(recon_dir: str) -> List[str]:
    """Distinct ``{texture}+{defect}`` keys from ``{key}_{idx:05d}.png`` filenames (idx is the
    trailing ``_<digits>`` segment, so rsplit once from the right)."""
    keys = set()
    for fn in os.listdir(recon_dir):
        if fn.lower().endswith(IMAGE_EXTS):
            keys.add(os.path.splitext(fn)[0].rsplit("_", 1)[0])
    return sorted(keys)


def resolve_anomaly_types(anomaly_types: Optional[List[str]], recipe: Optional[str], recon_dir: str) -> List[str]:
    """Explicit ``anomaly_types`` win; else derive from ``recipe``'s ``anomaly_types``; else infer
    from the generated filenames under ``recon_dir``. Every resolved key must be ``texture+defect``."""
    if anomaly_types:
        resolved = list(anomaly_types)
    elif recipe and recipe.lower().endswith((".yaml", ".yml", ".json")):
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(recipe), resolve=True)
        recipe_types = data.get("anomaly_types") if isinstance(data, dict) else None
        resolved = [f"{t}+{d}" for t, d in recipe_types] if recipe_types else infer_anomaly_types(recon_dir)
    else:
        resolved = infer_anomaly_types(recon_dir)

    # Validated, not merely checked for a '+': load_real splits each key and joins both halves onto
    # real_root, so a path-bearing key would score references read from outside the dataset.
    for key in resolved:
        validate_anomaly_type(key, field="anomaly_types")
    return resolved


def load_generated(
    gen_root: str,
    anomaly_types: List[str],
    target_size: Optional[int] = None,
    with_original_image: bool = False,
) -> Dict[str, Dict[str, List]]:
    """Generated images/masks keyed by anomaly type, read straight from the SDG output tree.

    ``target_size`` resizes each image/mask to a square of that side; ``None`` keeps native
    resolution, which is what compares variable-size outputs as they were generated.

    ``with_original_image`` additionally loads the pre-edit clean image from ``original_image/`` into
    an ``"original_image"`` list. Correspondence (NN/MNN) does not need it, but the anomaly-quality
    axes (completeness/precision/boundary_iou) diff against it, so filtering on those requires it. A
    sample whose clean image is missing is skipped so all four lists stay aligned.
    """
    recon_dir = os.path.join(gen_root, RECON_SUBDIR)
    mask_dir = os.path.join(gen_root, MASK_SUBDIR)
    orig_dir = os.path.join(gen_root, ORIG_IMAGE_SUBDIR)
    if not os.path.isdir(recon_dir):
        raise FileNotFoundError(f"{recon_dir} not found — expected generate.py output layout under {gen_root}.")

    all_files = sorted(os.listdir(recon_dir))
    generated: Dict[str, Dict[str, List]] = {}
    for key in anomaly_types:
        # Reconstructed and mask share the filename; prefix match avoids one type's name being a
        # substring of another's.
        files = [f for f in all_files if f.startswith(f"{key}_") and f.lower().endswith(IMAGE_EXTS)]
        recon, masks, paths, origs = [], [], [], []
        for f in files:
            mask_path = os.path.join(mask_dir, f)
            if not os.path.exists(mask_path):
                continue
            orig_path = os.path.join(orig_dir, f)
            if with_original_image and not os.path.exists(orig_path):
                log.warning(f"no original_image for {f!r} under {orig_dir}; skipping sample.")
                continue
            img = Image.open(os.path.join(recon_dir, f))
            mask = Image.open(mask_path)
            require_same_size(img, mask, f"{os.path.join(recon_dir, f)} + {mask_path}")
            recon.append(_img_to_float(img, target_size))
            masks.append(_mask_to_float(mask, target_size))
            paths.append(os.path.join(recon_dir, f))
            if with_original_image:
                origs.append(_img_to_float(Image.open(orig_path), target_size))
        if recon:
            entry = {"reconstructed_image": recon, "original_mask": masks, "img_path": paths}
            if with_original_image:
                entry["original_image"] = origs
            generated[key] = entry
        else:
            log.warning(f"no generated images for {key!r} under {recon_dir}; skipping.")
    return generated


def load_real(
    real_root: str, anomaly_types: List[str], target_size: Optional[int] = None
) -> Dict[str, Dict[str, List]]:
    """Real references from ``{texture}/anomaly_image/{defect}`` paired with ``{texture}/mask/{defect}/<stem>_mask``.

    ``target_size`` resizes each image/mask to a square of that side; ``None`` keeps native resolution.
    """
    real: Dict[str, Dict[str, List]] = {}
    for key in anomaly_types:
        texture, _, defect = key.partition("+")
        pairs = list_image_mask_pairs(
            os.path.join(real_root, texture, "anomaly_image", defect),
            os.path.join(real_root, texture, "mask", defect),
            mask_suffix="_mask",
        )
        imgs, masks = [], []
        for img_path, mask_path in pairs:
            if not os.path.exists(mask_path):
                continue
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            require_same_size(img, mask, f"{img_path} + {mask_path}")
            imgs.append(_img_to_float(img, target_size))
            masks.append(_mask_to_float(mask, target_size))
        if not imgs:
            raise RuntimeError(
                f"No real references for {key!r} under {os.path.join(real_root, texture, 'anomaly_image', defect)} "
                f"(+ matching {texture}/mask/{defect}/<stem>_mask). Check --real_root and that masks exist."
            )
        real[key] = {"original_image": imgs, "original_mask": masks}
    return real
