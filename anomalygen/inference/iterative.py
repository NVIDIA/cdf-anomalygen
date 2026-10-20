# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Iterative multi-instance inpainting orchestration: split a full-image mask into
# per-instance masks, then inpaint each in turn, feeding each result into the next.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from cosmos_framework.utils import log
from PIL import Image
from sklearn.cluster import KMeans

from anomalygen.data.crop_geometry import CropPlan, InputSizeSpec, ladder_for, mask_bbox, plan_for
from anomalygen.data.utils import MASK_FG_THRESHOLD, require_same_size
from anomalygen.inference.crop_paste import annotate, best_crop, paste_back


def _components(mask: Image.Image):
    """Connected components of ``mask`` at the foreground cut: ``(labels, stats, centroids)``, background dropped."""
    binary = np.where(np.asarray(mask.convert("L")) >= MASK_FG_THRESHOLD, 255, 0).astype(np.uint8)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return labels, stats[1:], centroids[1:]


def _instance_groups(centroids: np.ndarray, max_k: int) -> List[np.ndarray]:
    """Component indices per instance: one component each when there are at most ``max_k``, else
    KMeans clusters of the component centroids. The one grouping the splitter and the bbox scan share."""
    n = len(centroids)
    if n <= max_k:
        return [np.array([i]) for i in range(n)]
    if max_k == 1:
        return [np.arange(n)]
    labels = KMeans(n_clusters=max_k, n_init=10, random_state=0).fit(centroids).labels_
    return [np.flatnonzero(labels == c) for c in range(max_k)]


def split_mask_into_instances(mask: Image.Image, max_k: int = 5) -> List[Image.Image]:
    """Split a binary mask into up to ``max_k`` per-instance masks (CC + KMeans clustering)."""
    if max_k <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")
    labels, _, centroids = _components(mask)
    # Before the max_k == 1 shortcut: an empty mask must return [] at every max_k, or the caller
    # plans a window from a bbox that does not exist.
    if len(centroids) == 0:
        return []
    if max_k == 1:
        return [mask]
    # Component labels start at 1; label 0 is the background _components dropped.
    return [
        Image.fromarray(np.where(np.isin(labels, members + 1), 255, 0).astype(np.uint8))
        for members in _instance_groups(centroids, max_k)
    ]


def instance_bboxes(mask: Image.Image, max_k: int) -> List[Tuple[int, int, int, int]]:
    """Inclusive ``(left, upper, right, lower)`` of each instance :func:`split_mask_into_instances` would
    return, read off the component stats: no per-instance mask is built."""
    if max_k <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")
    _, stats, centroids = _components(mask)
    boxes = []
    for members in _instance_groups(centroids, max_k):
        left, top = stats[members, cv2.CC_STAT_LEFT], stats[members, cv2.CC_STAT_TOP]
        right = left + stats[members, cv2.CC_STAT_WIDTH] - 1
        lower = top + stats[members, cv2.CC_STAT_HEIGHT] - 1
        boxes.append((int(left.min()), int(top.min()), int(right.max()), int(lower.max())))
    return boxes


# Classes whose cap warning has been logged: one line per class per process is enough.
_CAP_WARNED: set[str] = set()


def _plan_window(
    inst_mask: Image.Image,
    img_w: int,
    img_h: int,
    input_spec: InputSizeSpec,
    *,
    crop_ratio: float,
    anomaly_name: str,
) -> Tuple[CropPlan, Tuple[int, int, int, int]]:
    """Return ``(plan, bbox)`` for this instance from the shared planner (:func:`plan_for`).

    ``bbox`` is the mask's ``(left, upper, right, lower)``, scanned once here and handed to every
    later step that needs it. The plan is passed onward rather than recomputed, which is what keeps
    the framing identical to training and guarantees the window contains the whole mask.

    Warns once per class when the ladder the class trained under cannot hold the defect at
    ``input_spec.scale``: the crop is then downsampled to fit and upsampled on paste-back, which
    reads downstream as a bad result rather than as a defect larger than any the class trained on.
    """
    bbox = mask_bbox(inst_mask)
    plan = plan_for(input_spec, bbox, img_w, img_h, crop_ratio=crop_ratio, anomaly_name=anomaly_name)
    squeezed = input_spec.crop_mode == "scale" and plan.scale < input_spec.scale - 1e-6
    if squeezed and anomaly_name not in _CAP_WARNED:
        _CAP_WARNED.add(anomaly_name)
        log.warning(
            f"{anomaly_name}: a {bbox[2] - bbox[0] + 1}x{bbox[3] - bbox[1] + 1} px defect plans at scale "
            f"{plan.scale:.2f}, under the inference scale {input_spec.scale}: the largest input size for this "
            f"class is {max(ladder_for(input_spec, anomaly_name))}, so the crop is downsampled to fit and "
            "upsampled on paste-back. Reported once per class."
        )
    return plan, bbox


def run_iterative_inpaint_batch(
    images: List[Image.Image],
    masks: List[Image.Image],
    anomaly_names: List[str],
    model_inpaint_batch,
    *,
    num_depth: Optional[int] = None,
    seeds: List[int],
    crop_ratios: List[float],
    poisson_blends: List[bool],
    max_instances_list: List[int],
    input_spec: InputSizeSpec,
):
    """Batched iterative inpaint: process the j-th instance of every image in one forward.

    For each depth ``j`` in ``range(num_depth)``, gather the j-th instance crop of every image
    (dummy-padding images with fewer instances so the batch width stays constant), run one
    batched ``model_inpaint_batch(crops, masks, names, seeds, plans)`` — each seed offset by ``j``, so a
    sample's instances don't share one noise draw — and paste each real result back into that
    image's running reconstruction. Crop windows derive from the (fixed) instance masks, so they
    are stable across depths; only the crop *content* updates from the running image. ``num_depth``
    defaults to this call's own max instance depth (generation: one image per call, unsharded). FSDP
    validation passes the max across ranks instead, and the depths past this rank's own still run a
    forward against all-dummy slots whose output is discarded, so the ranks stay in step.
    ``input_spec`` is the :class:`InputSizeSpec` published on the checkpoint; it re-plans each window
    through the shared planner.
    Returns per-image ``(composites, artifacts)`` where each artifact dict has per-instance lists
    keyed ``cropped_image``/``cropped_mask``/``annotated_image``/``mask_cropped_image``.
    """
    b = len(images)
    for i in range(b):
        require_same_size(images[i], masks[i], anomaly_names[i])
    inst_masks = [split_mask_into_instances(masks[i], max_k=max_instances_list[i]) for i in range(b)]
    if num_depth is None:
        num_depth = max((len(m) for m in inst_masks), default=0)
    for i in range(b):
        if not inst_masks[i]:
            log.warning(
                f"{anomaly_names[i]}: mask has no defect pixels at the >= {MASK_FG_THRESHOLD} cut; "
                f"image {i} is returned untouched."
            )
    running = [images[i].copy() for i in range(b)]
    artifacts: List[Dict[str, List[Image.Image]]] = [
        {"cropped_image": [], "cropped_mask": [], "annotated_image": [], "mask_cropped_image": []} for _ in range(b)
    ]
    # Most recent real (crop, plan), reused to pad depths where this rank has nothing real left.
    last_template: Optional[Tuple[Image.Image, CropPlan]] = None

    for j in range(num_depth):
        crops: List[Image.Image] = []
        cmasks: List[Image.Image] = []
        offsets: List[Tuple[int, int]] = []
        plans: List[Optional[CropPlan]] = []
        boxes: List[Optional[Tuple[int, int, int, int]]] = []
        is_real: List[bool] = []
        for i in range(b):
            real = j < len(inst_masks[i])
            is_real.append(real)
            if real:
                inst = inst_masks[i][j]
                plan, box = _plan_window(
                    inst, *running[i].size, input_spec, crop_ratio=crop_ratios[i], anomaly_name=anomaly_names[i]
                )
                ci, cm, off = best_crop(running[i], inst, plan.crop_w, plan.crop_h, box)
                artifacts[i]["annotated_image"].append(
                    annotate(running[i], inst, off, ci.size, anomaly_names[i], bbox=box)
                )
                artifacts[i]["cropped_image"].append(ci)
                artifacts[i]["cropped_mask"].append(cm)
                crops.append(ci)
                cmasks.append(cm)
                offsets.append(off)
                plans.append(plan)
                boxes.append(box)
            else:
                # Dummy slot keeps the batch width constant; its output is discarded. Mirror a real
                # crop rather than the whole image: a full frame would plan an enormous input (and
                # add a compile shape) purely to be thrown away.
                crops.append(None)
                cmasks.append(None)
                offsets.append((0, 0))
                plans.append(None)
                boxes.append(None)

        template = next((k for k, r in enumerate(is_real) if r), None)
        if template is not None:
            last_template = (crops[template], plans[template])
        elif last_template is None:
            # Nothing real on this rank at any depth so far, so there is no earlier crop to mirror.
            # Fall back to the smallest size: a shape the ladder always covers, so the throwaway
            # forward below adds no compile shape.
            size = input_spec.input_size_ladder[0]
            last_template = (Image.new("RGB", (size, size)), CropPlan(size, size, size, size, size, size, 1.0))

        # Every depth runs a forward even when this rank has nothing real here: num_depth is the
        # global max instance depth, so skipping one would leave this rank a parameter all-gather
        # behind its peers and hang the collective.
        tmpl_img, tmpl_plan = last_template
        for k, r in enumerate(is_real):
            if not r:
                crops[k] = tmpl_img.copy()
                cmasks[k] = Image.new("L", tmpl_img.size)
                plans[k] = tmpl_plan

        # One noise draw per instance: depth j of a sample is seeded ``seed + j``.
        edited = model_inpaint_batch(crops, cmasks, anomaly_names, [int(s) + j for s in seeds], plans)

        for i in range(b):
            if not is_real[i]:
                continue
            running[i] = paste_back(
                input_image=running[i],
                recon_image=edited[i],
                cropped_image=crops[i],
                cropped_mask=cmasks[i],
                offset=offsets[i],
                poisson_blend=poisson_blends[i],
            )
            # The bbox is inclusive; PIL crop right/lower are exclusive, so +1 (avoids a zero-area crop).
            lx, uy, rx, by = boxes[i]
            artifacts[i]["mask_cropped_image"].append(running[i].crop((lx, uy, rx + 1, by + 1)))

    return running, artifacts
