# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified ROI → AMP pipeline: auto-routes between CAD-, THT-, text- and free-based ROI generation.

Routing logic (``_get_sample_route``): a defect whose ``spatial_dependency`` is "cad_tht" takes the
tht2roi path; otherwise a sample carrying "cad_mask" takes the cad2roi path, and the rest split on
"free" vs "text".

Usage:
    python -m anomalygen.scripts.auto_mask_placement.roi_place \
        --input_pair_path amp_configs/amp_samples_template.json \
        --defect_desc amp_configs/defect_spec_template.jsonl \
        --output_dir results/output \
        --n_seeds 5 --seed 42

Input JSON format: list of samples, each with:
    Common (required):
        - clean_image:  path to image (PNG)
        - defect_type:  defect category string

    CAD route (when cad_mask is present):
        - cad_mask:     path to CAD semantic segmentation mask (PNG)
        - cad_mask_label:   path to color label JSON
        - submask:      path to defect submask (null for missing)

    CAD_THT route (spatial_dependency "cad_tht"): the same three fields, but the mask carries the
    disjoint pad / hole / pin classes that become regions B / C / D.

    Text route (when cad_mask is absent):
        - submask:      path to defect submask (null → ROI only)
"""

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.auto_mask_placement import AutoMaskPlacement
from anomalygen.auto_mask_placement.cad2roi import (
    BridgeMaskPlacer,
    CADToROIGenerator,
    ExcessSolderMaskPlacer,
    LessSolderMaskPlacer,
    MissingMaskPlacer,
    get_bridge_groups,
)
from anomalygen.auto_mask_placement.cad2roi.mask_utils import mask_area
from anomalygen.auto_mask_placement.roi_generation import ROIGenerationModels
from anomalygen.auto_mask_placement.text2roi import (
    SAM_IMAGE_SIZE,
    Text2BoxDetector,
    pick_best_mask,
    postprocess_sam_mask,
    resize_for_sam,
)
from anomalygen.auto_mask_placement.tht2roi import (
    BRIDGE,
    RingMaskPlacer,
    THTBridgeMaskPlacer,
    THTToROIGenerator,
    roi_rule,
)
from anomalygen.configs.texture.constants import DEFAULT_CROP_RATIO, DEFAULT_GUIDANCE, DEFAULT_NUM_STEPS
from anomalygen.data.utils import list_images, validate_anomaly_type

_ROI_MASK_PNG = "roi_mask.png"

# ── Visualization colors ──────────────────────────────────────────────────────
ROI_COLOR = np.array([0, 200, 0], dtype=np.float32)
CAD_ROI_COLOR = np.array([180, 180, 180], dtype=np.float32)
MASK_COLOR = np.array([0, 220, 255], dtype=np.float32)
CAD_MASK_COLOR = np.array([255, 140, 0], dtype=np.float32)
BBOX_COLOR = (0, 0, 255)
POINT_COLOR = (0, 255, 255)

# Ceiling on placements one pair record may request. The pair file is data, not code: a corrupt or
# hand-edited n_seeds must not turn a single record into an unbounded placement loop. Far above any
# allocation roi_pair produces (n_seeds = ceil(allocation / pair budget)), so it only bites on garbage.
MAX_SEEDS_PER_RECORD = 10_000


# ── Defect-description / routing helpers ──────────────────────────────────────


def _load_defect_descriptions(jsonl_path):
    """Load defect descriptions from JSONL file.

    Returns:
        (prompts, spatial_deps) dicts keyed by defect_type.
    """
    prompts = {}
    spatial_deps = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            dt = entry["defect_type"]
            spatial_deps[dt] = entry.get("spatial_dependency", "text")
            roi_prompt = entry.get("roi_prompt_defect_location", "")
            if roi_prompt:
                prompts[dt] = roi_prompt
    return prompts, spatial_deps


def _get_sample_route(sample, spatial_deps):
    """Determine route: 'free', 'cad', 'cad_tht', or 'text'.

    ``cad_tht`` is tested first because its records also carry ``cad_mask`` — the THT semantic mask
    lives at the same path — so the ``cad`` test below would otherwise claim them. The check is a
    guarded early return rather than a rewrite that makes ``spatial_dependency`` authoritative for
    every route: no ``cad`` / ``text`` / ``free`` defect can have this value, so all three reach
    exactly the code they reached before.
    """
    dep = spatial_deps.get(sample["defect_type"], "text")
    if dep == "cad_tht":
        return "cad_tht"
    if "cad_mask" in sample and sample["cad_mask"] is not None:
        return "cad"
    if dep == "free":
        return "free"
    return "text"


# ── Submask helpers ───────────────────────────────────────────────────────────


def _mask_area_from_path(path: str) -> int:
    """Read mask from file and return white pixel count."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask_area(img) if img is not None else 0


def _extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """Extract the largest connected component from a binary mask."""
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    # Label 0 is background; find largest among 1..N
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels_map == largest] = 255
    return result


def _preprocess_submask(submask_path: str, sample: dict) -> str:
    """Extract largest connected component from submask (default: on).
    Set submask_split_largest=false in sample to disable."""
    if not sample.get("submask_split_largest", True):
        return submask_path
    img = cv2.imread(submask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return submask_path
    _, mask_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    extracted = _extract_largest_component(mask_bin)
    p = Path(submask_path)
    out = Path(tempfile.mkdtemp()) / f"{p.stem}_largest.png"
    cv2.imwrite(str(out), extracted)
    return str(out)


# ── Visualization helpers ─────────────────────────────────────────────────────


def _overlay(bg, mask, color, alpha=0.6):
    """Blend color onto bg where mask > 0."""
    viz = bg.copy()
    m = mask > 0
    viz[m] = (viz[m].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
    return viz


def _make_amp_overlay(img_np, roi_mask, placed_mask, bbox, point):
    """Create AMP result overlay with ROI, placed mask, bbox, and point."""
    ov = img_np.astype(np.float32).copy()
    if roi_mask is not None:
        roi_bin = roi_mask > 0
        ov[roi_bin] = 0.65 * ov[roi_bin] + 0.35 * ROI_COLOR
    if placed_mask is not None:
        placed_bin = placed_mask > 0
        ov[placed_bin] = 0.45 * ov[placed_bin] + 0.55 * MASK_COLOR
    ov = ov.astype(np.uint8)
    if bbox:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(ov, (x0, y0), (x1, y1), BBOX_COLOR, 3)
    if point:
        px, py = int(point[0]), int(point[1])
        cv2.circle(ov, (px, py), 10, POINT_COLOR, -1)
        cv2.circle(ov, (px, py), 10, (0, 0, 0), 2)
    return ov


def _save_cad_seed(mask_bw, idx, prefix, out, assets, cad_img, clean_img):
    """Save mask + CAD/real overlays for one seed of a CAD defect."""
    Image.fromarray(mask_bw).save(out / f"{prefix}seed{idx}.png")
    Image.fromarray(_overlay(cad_img, mask_bw, CAD_MASK_COLOR)).save(assets / f"{prefix}seed{idx}_cad.png")
    Image.fromarray(_overlay(clean_img, mask_bw, CAD_MASK_COLOR)).save(assets / f"{prefix}seed{idx}_real.png")


def _sum_masks(cands):
    """Combine all candidate masks into one."""
    if not cands:
        return np.zeros((1, 1), dtype=np.uint8)
    result = cands[0].mask.copy()
    for c in cands[1:]:
        result = cv2.bitwise_or(result, c.mask)
    return result


# ── Free (whole-image ROI) processing ────────────────────────────────────────


def _process_free_sample(sample, output_dir, n_seeds, base_seed, min_area_ratio=0.5, max_area_retries=10):
    clean_image_path = sample["clean_image"]
    defect_type = sample["defect_type"]
    submask_path = sample.get("submask")

    name = Path(clean_image_path).stem
    out = Path(output_dir) / name / defect_type
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    pil = Image.open(clean_image_path).convert("RGB")
    ori_w, ori_h = pil.size
    img_np = np.array(pil)

    # ROI = entire image (white mask)
    roi_mask = np.full((ori_h, ori_w), 255, dtype=np.uint8)
    roi_mask_path = str(assets / _ROI_MASK_PNG)
    Image.fromarray(roi_mask).save(roi_mask_path)

    if not submask_path or not os.path.exists(submask_path):
        log.warning(f"  {name}/{defect_type}: free ROI, no submask for AMP")
        return {
            "name": name,
            "defect_type": defect_type,
            "route": "free",
            "status": "ROI_ONLY",
            "n_candidates": 1,
            "seeds": {},
        }

    submask_path = _preprocess_submask(submask_path, sample)
    submask_stem = Path(submask_path).stem
    base_area = _mask_area_from_path(submask_path)
    min_area = int(base_area * min_area_ratio)

    seed_results = {}

    for seed_i in range(n_seeds):
        final_mask_path = out / f"{submask_stem}__seed{seed_i}.png"
        overlay_path = assets / f"{submask_stem}__seed{seed_i}_overlay.png"

        if final_mask_path.exists() and overlay_path.exists():
            seed_results[seed_i] = {"status": "CACHED"}
            continue

        placed_ok = False
        amp_results = None

        with tempfile.TemporaryDirectory() as seed_work:
            for attempt in range(max_area_retries):
                sub_seed = base_seed + seed_i * max_area_retries + attempt
                amp = AutoMaskPlacement(
                    image_width=ori_w,
                    image_height=ori_h,
                    random_seed=sub_seed,
                    separate_rois=False,
                    max_retry_per_mask=15,
                )
                amp.load_combined_rois(roi_image_paths=[roi_mask_path])
                amp_results = amp.process_submask(
                    submask_path=submask_path,
                    n_instances=1,
                    output_dir=seed_work,
                )
                if not amp_results:
                    continue
                area = _mask_area_from_path(amp_results[0]["output_path"])
                if area > min_area:
                    placed_ok = True
                    break

            final_area = _mask_area_from_path(amp_results[0]["output_path"]) if amp_results else 0
            if amp_results and final_area > 0:
                shutil.copy(amp_results[0]["output_path"], str(final_mask_path))

                placed_np = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE)
                ov = _make_amp_overlay(img_np, None, placed_np, None, None)
                Image.fromarray(ov).save(overlay_path)

                seed_results[seed_i] = {"status": "OK" if placed_ok else "SMALL"}
            else:
                seed_results[seed_i] = {"status": "FAILED"}

    n_ok = sum(1 for s in seed_results.values() if s["status"] in ("OK", "CACHED"))
    log.info(f"  {name}/{defect_type}: free ROI, {n_ok}/{n_seeds} seeds OK")

    return {
        "name": name,
        "defect_type": defect_type,
        "route": "free",
        "status": "OK",
        "n_candidates": 1,
        "seeds": seed_results,
    }


# ── Text2ROI processing ─────────────────────────────────────────────────────


def _process_text2roi_sample(
    sample,
    defect_prompts,
    detector,
    sam,
    output_dir,
    n_seeds,
    base_seed,
    roi_only=False,
    min_area_ratio=0.5,
    max_area_retries=10,
    refresh_roi=False,
):
    clean_image_path = sample["clean_image"]
    defect_type = sample["defect_type"]
    submask_path = sample.get("submask") if not roi_only else None

    prompt = defect_prompts.get(defect_type)
    if prompt is None:
        raise ValueError(f"No roi_prompt_defect_location for '{defect_type}' in defect description JSONL.")

    name = Path(clean_image_path).stem
    cat, dc = defect_type.split("+", 1)
    out = Path(output_dir) / name / defect_type
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    pil = Image.open(clean_image_path).convert("RGB")
    ori_w, ori_h = pil.size
    img_np = np.array(pil)

    # ROI cache: reuse assets/roi_mask.png (skip the expensive detect + SAM2) unless refreshing.
    roi_mask_png = assets / _ROI_MASK_PNG
    meta_path = assets / "roi_meta.json"

    if roi_mask_png.exists() and not refresh_roi:
        log.info(f"  {name}/{defect_type}: reusing cached ROI")
        roi_mask = np.array(Image.open(roi_mask_png).convert("L"))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        bbox, point = meta.get("bbox"), meta.get("point")
    else:
        # Stage 1: Text2Box
        det_results = detector.detect(pil, prompt)
        if not det_results:
            log.warning(f"  {name}/{defect_type}: no detection, skipping")
            return {
                "name": name,
                "defect_type": defect_type,
                "route": "text2roi",
                "status": "NO_DETECTION",
                "n_candidates": 0,
                "seeds": {},
            }

        top = max(det_results, key=lambda r: r["confidence"])
        bbox = top["bbox"]
        point = top["point"]

        # Save bbox overlay
        bbox_viz = img_np.copy()
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(bbox_viz, (x0, y0), (x1, y1), BBOX_COLOR, 3)
        if point:
            px, py = int(point[0]), int(point[1])
            cv2.circle(bbox_viz, (px, py), 10, POINT_COLOR, -1)
            cv2.circle(bbox_viz, (px, py), 10, (0, 0, 0), 2)
        Image.fromarray(bbox_viz).save(assets / "bbox.png")

        # Stage 2: SAM2 → ROI mask
        res_pil, scale = resize_for_sam(pil, SAM_IMAGE_SIZE)
        sam.set_image(np.array(res_pil))

        bx = [int(v * scale) for v in bbox]
        pt_s = [point[0] * scale, point[1] * scale] if point else None

        if pt_s:
            sam_masks, scores, _ = sam.predict(
                point_coords=np.asarray([pt_s]),
                point_labels=np.asarray([1]),
                box=np.asarray(bx),
                multimask_output=True,
            )
        else:
            sam_masks, scores, _ = sam.predict(
                point_coords=None,
                point_labels=None,
                box=np.asarray(bx),
                multimask_output=True,
            )

        roi_mask = postprocess_sam_mask(pick_best_mask(sam_masks, scores), ori_w, ori_h)

        Image.fromarray(roi_mask).save(roi_mask_png)
        roi_viz = _overlay(img_np, roi_mask, ROI_COLOR, alpha=0.5)
        if bbox:
            cv2.rectangle(roi_viz, (x0, y0), (x1, y1), BBOX_COLOR, 3)
        if point:
            cv2.circle(roi_viz, (int(point[0]), int(point[1])), 10, POINT_COLOR, -1)
        Image.fromarray(roi_viz).save(assets / "roi_overlay.png")
        meta_path.write_text(json.dumps({"bbox": bbox, "point": point, "confidence": top["confidence"]}))

    # Stage 3: AMP
    if not submask_path or not os.path.exists(submask_path):
        log.warning(f"  {name}/{defect_type}: ROI OK, no submask for AMP")
        return {
            "name": name,
            "defect_type": defect_type,
            "route": "text2roi",
            "status": "ROI_ONLY",
            "bbox": bbox,
            "point": point,
            "n_candidates": 1,
            "seeds": {},
        }

    roi_mask_path = str(assets / _ROI_MASK_PNG)
    submask_path = _preprocess_submask(submask_path, sample)
    submask_stem = Path(submask_path).stem
    base_area = _mask_area_from_path(submask_path)
    min_area = int(base_area * min_area_ratio)

    seed_results = {}

    for seed_i in range(n_seeds):
        final_mask_path = out / f"{submask_stem}__seed{seed_i}.png"
        overlay_path = assets / f"{submask_stem}__seed{seed_i}_overlay.png"

        if final_mask_path.exists() and overlay_path.exists():
            seed_results[seed_i] = {"status": "CACHED"}
            continue

        placed_ok = False
        amp_results = None

        with tempfile.TemporaryDirectory() as seed_work:
            for attempt in range(max_area_retries):
                sub_seed = base_seed + seed_i * max_area_retries + attempt
                amp = AutoMaskPlacement(
                    image_width=ori_w,
                    image_height=ori_h,
                    random_seed=sub_seed,
                    separate_rois=False,
                    max_retry_per_mask=15,
                )
                amp.load_combined_rois(roi_image_paths=[roi_mask_path])
                amp_results = amp.process_submask(
                    submask_path=submask_path,
                    n_instances=1,
                    output_dir=seed_work,
                )
                if not amp_results:
                    continue
                area = _mask_area_from_path(amp_results[0]["output_path"])
                if area > min_area:
                    placed_ok = True
                    break

            final_area = _mask_area_from_path(amp_results[0]["output_path"]) if amp_results else 0
            if amp_results and final_area > 0:
                shutil.copy(amp_results[0]["output_path"], str(final_mask_path))

                placed_np = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE)
                ov = _make_amp_overlay(img_np, roi_mask, placed_np, bbox, point)
                Image.fromarray(ov).save(overlay_path)
                seed_results[seed_i] = {"status": "OK" if placed_ok else "SMALL"}
            else:
                seed_results[seed_i] = {"status": "FAILED"}

    n_ok = sum(1 for s in seed_results.values() if s["status"] in ("OK", "CACHED"))
    log.info(f"  {name}/{defect_type}: ROI OK, {n_ok}/{n_seeds} seeds OK")

    return {
        "name": name,
        "defect_type": defect_type,
        "route": "text2roi",
        "status": "OK",
        "bbox": bbox,
        "point": point,
        "n_candidates": 1,
        "seeds": seed_results,
    }


# ── CAD2ROI processing ──────────────────────────────────────────────────────


def _process_cad2roi_sample(sample, output_dir, n_seeds, base_seed, n_instances=-1, roi_only=False):
    cad_mask_path = sample["cad_mask"]
    clean_image_path = sample["clean_image"]
    defect_type = sample["defect_type"]
    submask_path = sample.get("submask")
    cad_mask_label = sample["cad_mask_label"]

    # defect_type is "PCB+missing" → cad_defect is "missing"
    cad_defect = defect_type.split("+", 1)[1] if "+" in defect_type else defect_type

    name = Path(cad_mask_path).stem
    out = Path(output_dir) / name / defect_type
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    cad_img = np.array(Image.open(cad_mask_path))[:, :, :3]
    clean_img = np.array(Image.open(clean_image_path).convert("RGB"))

    gen = CADToROIGenerator(cad_mask_label)
    candidates = gen.generate_all_candidates(cad_mask_path)
    cands = candidates.get(cad_defect, [])

    if not cands:
        log.warning(f"  {name}/{defect_type}: no ROI candidates, skipping")
        return {
            "name": name,
            "defect_type": defect_type,
            "route": "cad2roi",
            "status": "NO_CANDIDATES",
            "n_candidates": 0,
            "seeds": {},
        }

    # Save ROI overlays to assets
    roi_viz_cad = _overlay(cad_img, _sum_masks(cands), CAD_ROI_COLOR)
    roi_viz_real = _overlay(clean_img, _sum_masks(cands), CAD_ROI_COLOR)
    Image.fromarray(roi_viz_cad).save(assets / "roi_cad.png")
    Image.fromarray(roi_viz_real).save(assets / "roi_real.png")

    for i, cand in enumerate(cands):
        Image.fromarray(cand.mask).save(assets / f"roi_{i}_mask.png")

    if roi_only:
        log.info(f"  {name}/{defect_type}: {len(cands)} ROIs (roi-only)")
        return {
            "name": name,
            "defect_type": defect_type,
            "route": "cad2roi",
            "status": "ROI_ONLY",
            "n_candidates": len(cands),
            "seeds": {},
        }

    # AMP placement per seed
    seed_results = {}
    components, img_shape = gen.parser.parse(cad_mask_path)
    # Preprocess submask if needed
    if submask_path:
        submask_path = _preprocess_submask(submask_path, sample)
    sub_stem = Path(submask_path).stem if submask_path else ""
    prefix = f"{sub_stem}__" if sub_stem else ""

    if cad_defect == "missing":
        placer = MissingMaskPlacer()
        results = placer.place_all(cands, n_instances=n_instances, n_seeds=n_seeds, base_seed=base_seed)
        for seed_i, (n_sel, n_placed, mask_bw) in results.items():
            idx = seed_i - 1  # normalize to 0-based
            _save_cad_seed(mask_bw, idx, prefix, out, assets, cad_img, clean_img)
            seed_results[idx] = {"status": "OK", "n_selected": n_sel}

    elif cad_defect in ("less_solder", "excess_solder"):
        PlacerCls = ExcessSolderMaskPlacer if cad_defect == "excess_solder" else LessSolderMaskPlacer
        placer = PlacerCls()
        results = placer.place_all(
            submask_path, cands, img_shape, n_instances=n_instances, n_seeds=n_seeds, base_seed=base_seed
        )
        for seed_i, (n_req, n_placed, mask) in results.items():
            idx = seed_i - 1
            if mask is None:
                seed_results[idx] = {"status": "FAILED", "n_requested": n_req, "n_placed": 0}
                continue
            _save_cad_seed(mask, idx, prefix, out, assets, cad_img, clean_img)
            status = "OK" if n_placed >= n_req else "PARTIAL"
            seed_results[idx] = {"status": status, "n_requested": n_req, "n_placed": n_placed}

    elif cad_defect == "bridge":
        groups = get_bridge_groups(
            components, img_shape, classes=("pad", "solder"), min_area_abs=gen.parser.min_area_abs
        )
        placer = BridgeMaskPlacer()
        results = placer.place_all(
            submask_path, cands, groups, img_shape, n_instances=n_instances, n_seeds=n_seeds, base_seed=base_seed
        )
        for seed_i, (n_req, n_placed, mask) in results.items():
            idx = seed_i - 1
            if mask is None:
                seed_results[idx] = {"status": "FAILED", "n_requested": n_req, "n_placed": 0}
                continue
            _save_cad_seed(mask, idx, prefix, out, assets, cad_img, clean_img)
            status = "OK" if n_placed >= n_req else "PARTIAL"
            seed_results[idx] = {"status": status, "n_requested": n_req, "n_placed": n_placed}

    n_ok = sum(1 for s in seed_results.values() if s["status"] == "OK")
    log.info(f"  {name}/{defect_type}: {len(cands)} ROIs, {n_ok}/{n_seeds} seeds OK")

    return {
        "name": name,
        "defect_type": defect_type,
        "route": "cad2roi",
        "status": "OK",
        "n_candidates": len(cands),
        "seeds": seed_results,
    }


# ── CAD_THT (through-hole joint) processing ──────────────────────────────────


def _process_cad_tht_sample(sample, output_dir, n_seeds, base_seed, n_instances=1, roi_only=False):
    """Place one THT defect onto one joint crop.

    Reads the same artefacts as the cad route — ``cad_mask`` + ``cad_mask_label`` — but decodes them
    into the disjoint pad/hole/pin regions (B/C/D) that the THT ROI rules are written against.

    The output directory is keyed on the CLEAN image's stem, not the cad_mask's. They are required to
    be equal, but ``_build_testcase_jsonl`` looks masks up by clean stem, so keying on the clean image
    directly means a stem mismatch surfaces upstream in the preflight rather than as rows silently
    vanishing from the JSONL.
    """
    clean_image_path = sample["clean_image"]
    defect_type = sample["defect_type"]
    submask_path = sample.get("submask")

    tht_defect = defect_type.split("+", 1)[1] if "+" in defect_type else defect_type

    name = Path(clean_image_path).stem
    out = Path(output_dir) / name / defect_type
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    result = {"name": name, "defect_type": defect_type, "route": "tht2roi", "n_candidates": 0, "seeds": {}}

    # Each of these is one sample's problem, reported as that sample's status -- the shape the cad
    # and free routes already use. Left to raise, every one took the whole roi_place run down, and
    # this module runs standalone on a hand-written pair file where all three are easy to write.
    #
    # Skipping is only safe because run_pipeline now counts the testcase rows against num_sdg: a
    # skipped sample is a shortfall, and a shortfall fails the run with these statuses named from
    # summary.json. Without that gate this would trade a loud crash for a quietly short batch.
    missing = [k for k in ("cad_mask", "cad_mask_label") if not sample.get(k)]
    if missing:
        log.warning(f"  {name}/{defect_type}: pair record has no {' and no '.join(missing)}, skipping")
        return {**result, "status": "NO_CAD_MASK"}
    cad_mask_path = sample["cad_mask"]
    cad_mask_label = sample["cad_mask_label"]

    # Only when there is going to be a placement. --roi_only is the ROI cache warm-up
    # (roi_place_pipeline documents it as such) and never opens the submask, so demanding one here
    # skipped the sample before a single candidate or overlay was written -- on the one run whose
    # entire output is those overlays. The free and text routes null the submask out for roi_only at
    # :338 for the same reason, and this route did too before the guard was added.
    if not roi_only and (not submask_path or not os.path.exists(submask_path)):
        log.warning(f"  {name}/{defect_type}: no submask for AMP, skipping")
        return {**result, "status": "NO_SUBMASK"}

    # roi_rule raises on an unrecognised class -- deliberately, so it cannot degrade into an empty
    # candidate list that reads as "found nowhere to go". Asked here, where one bad row is one
    # skipped sample rather than the end of the run.
    try:
        roi_rule(tht_defect)
    except ValueError as exc:
        log.warning(f"  {name}/{defect_type}: {exc}, skipping")
        return {**result, "status": "UNKNOWN_DEFECT"}

    cad_img = np.array(Image.open(cad_mask_path).convert("RGB"))
    clean_img = np.array(Image.open(clean_image_path).convert("RGB"))

    # roi_allocate's preflight already rejects this for a dataset, but roi_place also runs standalone
    # on a hand-written pair file. Everything below is built at the cad_mask's resolution, and the
    # overlays index the clean image with a mask of that shape -- a mismatch would otherwise surface
    # as an opaque numpy IndexError instead of the dataset problem it is.
    if cad_img.shape[:2] != clean_img.shape[:2]:
        log.warning(
            f"  {name}/{defect_type}: cad_mask {cad_img.shape[1]}x{cad_img.shape[0]} != clean_image "
            f"{clean_img.shape[1]}x{clean_img.shape[0]}, skipping"
        )
        return {**result, "status": "SIZE_MISMATCH"}

    gen = THTToROIGenerator(cad_mask_label)
    joints, img_shape = gen.parse_joints(cad_mask_path)
    cands = gen.generate_candidates(joints, tht_defect)

    if not cands:
        log.warning(f"  {name}/{defect_type}: no ROI candidates from {len(joints)} joint(s), skipping")
        return {**result, "status": "NO_CANDIDATES"}

    roi_all = _sum_masks(cands)
    Image.fromarray(_overlay(cad_img, roi_all, CAD_ROI_COLOR)).save(assets / "roi_cad.png")
    Image.fromarray(_overlay(clean_img, roi_all, CAD_ROI_COLOR)).save(assets / "roi_real.png")
    for i, cand in enumerate(cands):
        Image.fromarray(cand.mask).save(assets / f"roi_{i}_mask.png")

    if roi_only:
        log.info(f"  {name}/{defect_type}: {len(cands)} ROIs (roi-only)")
        return {**result, "status": "ROI_ONLY", "n_candidates": len(cands)}

    submask_path = _preprocess_submask(submask_path, sample)
    prefix = f"{Path(submask_path).stem}__"

    seed_results = {}
    if tht_defect == BRIDGE:
        # bridge takes the JOINTS, not the candidates: it is not confined to a region, so the ROI
        # above is an overlay only and the fit is tested against the pad ring and the crop edge.
        results = THTBridgeMaskPlacer().place_all(
            submask_path, joints, img_shape, n_instances=n_instances, n_seeds=n_seeds, base_seed=base_seed
        )
    else:
        results = RingMaskPlacer().place_all(
            submask_path, cands, img_shape, n_instances=n_instances, n_seeds=n_seeds, base_seed=base_seed
        )
    for seed_i, (n_req, n_placed, mask, undersized) in results.items():
        idx = seed_i - 1
        if mask is None:
            seed_results[idx] = {"status": "FAILED", "n_requested": n_req, "n_placed": 0}
            continue
        _save_cad_seed(mask, idx, prefix, out, assets, cad_img, clean_img)
        # SMALL outranks PARTIAL: a seed that placed every ROI it was asked for but kept a sliver of
        # one of them is worse to receive than one that placed fewer well, because the sliver is
        # inpainted as a full defect. The free and text routes use the same label for the same thing.
        if undersized:
            status = "SMALL"
        else:
            status = "OK" if n_placed >= n_req else "PARTIAL"
        seed_results[idx] = {"status": status, "n_requested": n_req, "n_placed": n_placed}

    n_ok = sum(1 for s in seed_results.values() if s["status"] == "OK")
    log.info(f"  {name}/{defect_type}: {len(cands)} ROIs, {n_ok}/{n_seeds} seeds OK")
    return {**result, "status": "OK", "n_candidates": len(cands), "seeds": seed_results}


# ── SDG JSONL (folds in build_jsonl.py + verify_jsonl.py) ────────────────────


def _all_multi_instance_masks(mask_dir):
    """True only if EVERY training submask in mask_dir has ≥2 connected components."""
    imgs = list_images(mask_dir)
    if not imgs:
        return False
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        n_labels, *_ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
        if n_labels - 1 < 2:  # a single-instance mask exists
            return False
    return True


def _enumerate_amp_masks(amp_output_dir, full_type):
    """Yield (clean_stem, mask_path) for one defect_type from the AMP output layout."""
    validate_anomaly_type(full_type, field="defect_type")  # joined onto each output dir below
    amp_output_dir = Path(amp_output_dir)
    if not amp_output_dir.is_dir():
        return
    for name_dir in sorted(p for p in amp_output_dir.iterdir() if p.is_dir()):
        type_dir = name_dir / full_type
        if not type_dir.is_dir():
            continue
        for mask in sorted(type_dir.glob("*__seed*.png")):
            yield name_dir.name, mask


def _build_testcase_jsonl(samples, output_dir, jsonl_path, guidance=None, crop_ratio=None, num_steps=None):
    """Pair AMP-placed masks with their clean images into an SDG JSONL.

    guidance / crop_ratio / num_steps default to the shared
    ``anomalygen.configs.texture.constants`` values (DEFAULT_GUIDANCE /
    DEFAULT_CROP_RATIO / DEFAULT_NUM_STEPS) — the single source of truth. Pass None
    to omit a field (then the reader's own setdefault applies).

    iteration_generation_max_instance = 1 only when every training submask for a
    defect is multi-instance, else 5. The per-defect training-mask directory is
    taken from the ``submask`` paths in the input (``.../<texture>/mask/<anomaly>/``).
    """
    name_to_clean = {Path(s["clean_image"]).stem: s["clean_image"] for s in samples if s.get("clean_image")}
    defect_mask_dir = {}  # defect_type -> training-mask dir (parent of a submask)
    for s in samples:
        if s.get("submask") and s["defect_type"] not in defect_mask_dir:
            defect_mask_dir[s["defect_type"]] = Path(s["submask"]).parent
    defect_types = sorted({s["defect_type"] for s in samples})

    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    written = 0
    with open(jsonl_path, "w") as fp:
        for full_type in defect_types:
            pairs = list(_enumerate_amp_masks(output_dir, full_type))
            if not pairs:
                log.warning(f"  no AMP masks for {full_type}")
                continue
            mask_dir = defect_mask_dir.get(full_type)
            iter_max = 1 if (mask_dir and _all_multi_instance_masks(mask_dir)) else 5
            for stem, mask_path in pairs:
                clean_img = name_to_clean.get(stem)
                if clean_img is None:
                    log.warning(f"  no clean image for {stem}")
                    continue
                rec = {
                    "image_filename": str(clean_img),
                    "mask_filename": str(mask_path),
                    "anomaly_type": full_type,
                    "num_generated_images": 1,
                    "poisson_blend": False,
                    "iteration_generation_max_instance": iter_max,
                }
                # Only set the tunables when overridden; else InpaintInferenceDataset defaults apply.
                if guidance is not None:
                    rec["guidance"] = guidance
                if num_steps is not None:
                    rec["num_steps"] = num_steps
                if crop_ratio is not None:
                    rec["crop_ratio"] = crop_ratio
                fp.write(json.dumps(rec) + "\n")
                written += 1
    log.info(f"wrote {written} entries to {jsonl_path}")
    return written


def _verify_and_resize_jsonl(jsonl_path, cache_dir):
    """Check image/mask existence; resize mismatched masks (NEAREST) into cache_dir, rewriting the JSONL."""
    os.makedirs(cache_dir, exist_ok=True)
    entries = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line.strip()]
    resized = missing = 0
    for e in entries:
        if not Path(e["image_filename"]).exists():
            log.warning(f"missing image {e['image_filename']}")
            missing += 1
            continue
        if not Path(e["mask_filename"]).exists():
            log.warning(f"missing mask {e['mask_filename']}")
            missing += 1
            continue
        with Image.open(e["image_filename"]) as im:
            img_w, img_h = im.size
        with Image.open(e["mask_filename"]) as m:
            if m.size == (img_w, img_h):
                continue
            resized_im = m.resize((img_w, img_h), Image.NEAREST)
        key = os.path.relpath(e["mask_filename"]).replace(os.sep, "__")
        stem, ext = os.path.splitext(key)
        out_path = Path(cache_dir) / f"{stem}__{img_w}x{img_h}{ext}"
        if not out_path.exists():
            resized_im.save(out_path)
        e["mask_filename"] = str(out_path)
        resized += 1
    with open(jsonl_path, "w") as fp:
        fp.write("\n".join(json.dumps(e) for e in entries) + "\n")
    log.info(f"verified {len(entries)} entries (resized={resized}, missing={missing})")
    return missing


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified ROI → AMP pipeline (auto-routes cad2roi / text2roi)")
    parser.add_argument("--input_pair_path", required=True, help="Path to input samples JSON")
    parser.add_argument("--defect_desc", required=True, help="JSONL with roi_prompt_defect_location per defect_type")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds per sample")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_instances", type=int, default=-1, help="Number of instances per image for cad2roi (-1 = all ROIs)"
    )
    parser.add_argument("--model_id", default="nvidia/Cosmos3-Nano", help="VLM for text2box")
    parser.add_argument(
        "--max_area_retries", type=int, default=10, help="Max retry attempts per seed for mask placement"
    )
    parser.add_argument("--roi_only", action="store_true", help="Skip AMP + JSONL, only generate ROI")
    parser.add_argument(
        "--refresh_roi",
        action="store_true",
        help="Regenerate ROIs even if a cached assets/roi_mask.png exists (text2roi)",
    )
    # SDG JSONL (folds in build_jsonl.py + verify_jsonl.py) — always written to <output>/testcase.jsonl
    # Defaults are the shared InpaintInferenceDataset constants (no values hardcoded here).
    parser.add_argument(
        "--guidance", type=float, default=DEFAULT_GUIDANCE, help=f"JSONL guidance (default {DEFAULT_GUIDANCE})"
    )
    parser.add_argument(
        "--crop_ratio", type=float, default=DEFAULT_CROP_RATIO, help=f"JSONL crop_ratio (default {DEFAULT_CROP_RATIO})"
    )
    parser.add_argument(
        "--num_steps", type=int, default=DEFAULT_NUM_STEPS, help=f"JSONL num_steps (default {DEFAULT_NUM_STEPS})"
    )
    args = parser.parse_args(argv)

    # Load inputs
    defect_prompts, spatial_deps = _load_defect_descriptions(args.defect_desc)
    log.info(f"Loaded {len(spatial_deps)} defect descriptions from {args.defect_desc}")

    with open(args.input_pair_path) as f:
        samples = json.load(f)

    # Partition samples by route. Every route is named explicitly and an unrecognised one raises:
    # this used to end in a catch-all `else: free_indices.append(i)`, which would silently place a
    # new route's samples anywhere in the image and produce plausible-looking masks in the wrong spot.
    text_indices = []
    cad_indices = []
    tht_indices = []
    free_indices = []
    _buckets = {"text": text_indices, "cad": cad_indices, "cad_tht": tht_indices, "free": free_indices}
    for i, s in enumerate(samples):
        route = _get_sample_route(s, spatial_deps)
        if route not in _buckets:
            raise ValueError(f"sample {i} ({s['defect_type']}): unhandled route {route!r}")
        _buckets[route].append(i)

    log.info(
        f"Loaded {len(samples)} samples "
        f"({len(text_indices)} text2roi, {len(cad_indices)} cad2roi, "
        f"{len(tht_indices)} tht2roi, {len(free_indices)} free)"
    )

    # roi_pair stamps each record with the seed count its own defect type needs; --n_seeds is the
    # fallback for hand-written pair files. The stride uses the max so per-sample seed ranges stay
    # disjoint however the counts vary — overlapping ranges would repeat placements across samples.
    # The per-record value is clamped: it drives the placement loops, and comes from a file.
    def _n_seeds_of(sample):
        return min(int(sample.get("n_seeds") or args.n_seeds), MAX_SEEDS_PER_RECORD)

    # Said once per record here; _n_seeds_of itself is called several times per record.
    for s in samples:
        requested = int(s.get("n_seeds") or args.n_seeds)
        if requested > MAX_SEEDS_PER_RECORD:
            log.warning(f"n_seeds={requested} for {s.get('clean_image')} capped at {MAX_SEEDS_PER_RECORD}")

    seed_stride = max((_n_seeds_of(s) for s in samples), default=args.n_seeds)
    # Same-type records differ: the last one carries the remainder, so a plain dict comprehension
    # would report the remainder (allocation=9, n_seeds=2 -> records [2,2,2,2,1] would log 1). Take
    # the max per type — this is the number read back when auditing whether an allocation came out.
    per_type = {}
    for s in samples:
        dt = s["defect_type"]
        per_type[dt] = max(per_type.get(dt, 0), _n_seeds_of(s))
    log.info(f"Output: {args.output_dir}, n_seeds per type={per_type}, base_seed={args.seed}")

    all_results = [None] * len(samples)
    total = len(samples)

    # ── Text2ROI path (GPU-heavy, run first) ─────────────────────────────
    if text_indices:

        def _roi_cached(i):
            s = samples[i]
            roi = Path(args.output_dir) / Path(s["clean_image"]).stem / s["defect_type"] / "assets" / _ROI_MASK_PNG
            return roi.exists()

        need_model = args.refresh_roi or any(not _roi_cached(i) for i in text_indices)
        detector = sam = models = None
        if need_model:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"\nLoading Text2Box (Qwen) on {device}...")
            detector = Text2BoxDetector(model_id=args.model_id, device=str(device))
            detector.load()
            log.info("Loading SAM2...")
            models = ROIGenerationModels(device)
            sam = models._get_model("sam2")
            log.info("Models ready.\n")
        else:
            log.info("\nAll text2roi ROIs cached — skipping model load.\n")

        for idx in text_indices:
            sample = samples[idx]
            log.info(f"[{idx + 1}/{total}] {sample['clean_image']} -> {sample['defect_type']}")
            sample_seed = args.seed + idx * seed_stride * args.max_area_retries
            result = _process_text2roi_sample(
                sample,
                defect_prompts,
                detector,
                sam,
                args.output_dir,
                _n_seeds_of(sample),
                sample_seed,
                roi_only=args.roi_only,
                max_area_retries=args.max_area_retries,
                refresh_roi=args.refresh_roi,
            )
            all_results[idx] = result

        # Free VRAM
        if need_model:
            detector.unload()
            del sam, models, detector
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            log.info("\nText2ROI models unloaded.\n")

    # ── Free path (whole-image ROI, CPU-only) ────────────────────────────
    if free_indices:
        log.info("Processing free (whole-image ROI) samples...\n")
        for idx in free_indices:
            sample = samples[idx]
            log.info(f"[{idx + 1}/{total}] {sample['clean_image']} -> {sample['defect_type']}")
            sample_seed = args.seed + idx * seed_stride * args.max_area_retries
            result = _process_free_sample(
                sample,
                args.output_dir,
                _n_seeds_of(sample),
                sample_seed,
                max_area_retries=args.max_area_retries,
            )
            all_results[idx] = result

    # ── CAD2ROI path (CPU-bound) ─────────────────────────────────────────
    if cad_indices:
        log.info("Processing CAD2ROI samples...\n")
        for idx in cad_indices:
            sample = samples[idx]
            log.info(f"[{idx + 1}/{total}] {sample['cad_mask']} -> {sample['defect_type']}")
            sample_seed = args.seed + idx * seed_stride * args.max_area_retries
            result = _process_cad2roi_sample(
                sample,
                args.output_dir,
                _n_seeds_of(sample),
                sample_seed,
                n_instances=args.n_instances,
                roi_only=args.roi_only,
            )
            all_results[idx] = result

    # ── THT2ROI path (CPU-bound) ─────────────────────────────────────────
    if tht_indices:
        log.info("Processing THT2ROI samples...\n")
        # -1 means "a random count in [1, n_candidates]" everywhere else, which on a THT crop that
        # caught a neighbouring joint would paint two defects. One joint carries one defect, because
        # the generated crop is composited back against a single joint id. An explicit count is
        # still honoured, so asking for more remains possible.
        tht_n_instances = args.n_instances if args.n_instances > 0 else 1
        if args.n_instances != tht_n_instances:
            log.info(f"cad_tht: --n_instances {args.n_instances} -> {tht_n_instances} (one defect per joint)")
        for idx in tht_indices:
            sample = samples[idx]
            log.info(f"[{idx + 1}/{total}] {sample['clean_image']} -> {sample['defect_type']}")
            sample_seed = args.seed + idx * seed_stride * args.max_area_retries
            result = _process_cad_tht_sample(
                sample,
                args.output_dir,
                _n_seeds_of(sample),
                sample_seed,
                n_instances=tht_n_instances,
                roi_only=args.roi_only,
            )
            all_results[idx] = result

    # ── Summary ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSummary saved to {summary_path}")

    # Stats
    log.info(
        f"\nTotal: {total} samples ({len(text_indices)} text2roi, {len(cad_indices)} cad2roi, "
        f"{len(tht_indices)} tht2roi, {len(free_indices)} free)"
    )

    for route, indices in [("text2roi", text_indices), ("free", free_indices)]:
        if not indices:
            continue
        results = [all_results[i] for i in indices]
        has_roi = sum(1 for r in results if r["status"] not in ("NO_DETECTION",))
        n_ok = sum(sum(1 for s in r["seeds"].values() if s["status"] in ("OK", "CACHED")) for r in results)
        n_st = sum(len(r["seeds"]) for r in results)
        seeds_str = f", {n_ok}/{n_st} seeds OK" if n_st > 0 else ""
        log.info(f"{route}: {has_roi}/{len(results)} with ROI{seeds_str}")

    for route, indices in [("cad2roi", cad_indices), ("tht2roi", tht_indices)]:
        if not indices:
            continue
        route_results = [all_results[i] for i in indices]
        log.info(f"\n{route}:")
        for dt in sorted(set(r["defect_type"] for r in route_results)):
            dt_results = [r for r in route_results if r["defect_type"] == dt]
            n_with_roi = sum(1 for r in dt_results if r["n_candidates"] > 0)
            n_ok = sum(sum(1 for s in r["seeds"].values() if s["status"] == "OK") for r in dt_results)
            n_st = sum(len(r["seeds"]) for r in dt_results)
            log.info(f"  {dt}: {n_with_roi}/{len(dt_results)} with ROI, {n_ok}/{n_st} seeds OK")

    # ── SDG JSONL + allocation (always, unless roi-only) ─────────────────
    # Everything lands under --output_dir: the placed masks, summary.json, testcase.jsonl,
    # allocation.json, and resized_masks/.
    if not args.roi_only:
        out_dir = args.output_dir
        jsonl_path = os.path.join(out_dir, "testcase.jsonl")
        n = _build_testcase_jsonl(
            samples,
            args.output_dir,
            jsonl_path,
            guidance=args.guidance,
            crop_ratio=args.crop_ratio,
            num_steps=args.num_steps,
        )
        if n > 0:
            _verify_and_resize_jsonl(jsonl_path, os.path.join(out_dir, "resized_masks"))

        allocation = {}
        for s in samples:
            allocation[s["defect_type"]] = allocation.get(s["defect_type"], 0) + _n_seeds_of(s)
        allocation_path = os.path.join(out_dir, "allocation.json")
        with open(allocation_path, "w") as f:
            json.dump(allocation, f, indent=2)
        log.info(f"wrote allocation to {allocation_path}: {allocation}")


if __name__ == "__main__":
    main()
