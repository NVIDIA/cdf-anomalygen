# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defect-quality validation metrics: the raw geometry axes plus the ``aq_nn`` composite.

``aq_nn = completeness ① + nn_score ⑤`` — an equal-weight, bounded composite (each term in ~[0, 1],
no cross-sample standardisation) so it is directly comparable **across validation passes** and rises
as the model improves. (An earlier draft z-scored each term over the current pass; that recentres
every pass to ~0, so the cross-step ``Average`` the monitor / early-stop reads could not track
training — this formulation fixes that.) ``aq_nn`` is available and early-stop-selectable but is not
plotted by default and is secondary to ``nn_score``. Treat it as a dataset-native realism monitor,
not a human-alignment metric: on human 2AFC preferences it scored *below* its own ``completeness``
term (0.736 vs 0.763), because folding in ``nn_score`` (which anti-aligns with human preference,
AUROC ~ 0.48) pulls it down — so pick ``completeness`` if you want the better human-aligned early-stop.

Sub-metrics (each mirrors the offline implementation):

* ``completeness`` — diff-anchored A/B gated-SAM coverage of the WORST mask part. ``M_gen`` is
  ``A = SAM(whole-mask box)`` or ``B = SAM per diff-core cluster (union)``, both gated against the
  diff core; ``B`` wins when the change fragments into ≥2 clusters (else ``A``, else ``B``, else a
  low-threshold Otsu diff), which keeps a partially-generated *missing* defect low. The score is
  ``min_part |M_gen ∩ part| / |part|`` over the mask's connected parts. (①)
* ``precision`` — ``|M_gen ∩ mask| / |M_gen|``, the fraction of the segmented change kept inside the
  mask. (①)
* ``boundary_iou`` — Cheng CVPR21 boundary-band IoU of ``M_gen`` vs the mask. (①)
* ``nn_score`` — reused from :func:`compute_correspondence_kpi` (frozen DINOv2). (⑤)

``compute_anomaly_quality_kpi`` returns ``{anomaly_name: {"aq_nn", completeness, precision,
boundary_iou}, "Average": {...}}`` so :class:`ValidationKPI` can merge them into ``valid_kpi.csv``
alongside FID/NN; ``aq_nn`` is selectable as an early-stop / training-report metric by its
``METRIC_SPECS`` name.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from cosmos_framework.utils import log
from scipy import ndimage
from scipy.stats import rankdata
from skimage.filters import threshold_otsu

# Thresholds that decide whether completeness trusts the SAM mask (else it falls back to the diff).
_SAM_CAP = 1.1  # |M_SAM| / |mask| upper bound to trust the SAM segmentation
_SAM_OVL_MIN = 0.5  # min |M_SAM ∩ M_core| / |M_core| to trust it
_LOW_OTSU = 0.5  # fallback threshold = _LOW_OTSU * Otsu when SAM is unreliable
_OTSU_FLOOR = 8.0  # diff floor (0-255) rejecting sensor noise / faithful repaint
_MIN_COMP_PX = 9  # skip mask specks below this many pixels when splitting into parts
_MERGE_GAP_FRAC = 0.015  # dilate a diff core by this * image-diagonal before clustering
_MIN_CLUSTER_FRAC = 0.01  # a diff cluster must hold >= this * |mask| of core to count

# SAM2.1 hiera-large weights, resolved relative to the repo root (mirrors roi_generation.model).
_SAM2_CKPT = str(
    Path(__file__).resolve().parents[2] / "checkpoints" / "facebook" / "sam2.1-hiera-large" / "sam2.1_hiera_large.pt"
)
_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

_CACHE: dict = {}


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- lazy models
def _sam_predictor(device: str):
    key = ("sam", device)
    if key not in _CACHE:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(_SAM2_CONFIG, _SAM2_CKPT, device=device)
        _CACHE[key] = SAM2ImagePredictor(model)
    return _CACHE[key]


# --------------------------------------------------------------------------- geometry ①
def _extract_core(clean255: np.ndarray, gen255: np.ndarray, floor: float = _OTSU_FLOOR):
    """Binarise the change map: diff = channel-mean |clean - gen|, split by Otsu (floored)."""
    diff = np.abs(clean255 - gen255).mean(axis=2)  # H×W in [0, 255]
    if diff.max() < floor:
        return np.zeros(diff.shape, dtype=bool), diff, floor
    try:
        t = max(float(threshold_otsu(diff)), floor)
    except Exception:  # noqa: BLE001 — degenerate (constant) diff
        t = floor
    core = ndimage.binary_opening(diff > t, structure=np.ones((3, 3)))
    return core, diff, t


def _mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Cheng CVPR21 boundary band = mask - erode(mask, d), d = ratio * image diagonal."""
    h, w = mask.shape
    d = max(1, int(round(dilation_ratio * math.hypot(h, w))))
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), iterations=d, border_value=0)
    return mask & ~eroded


def precision(mgen: np.ndarray, mask_bool: np.ndarray) -> float:
    """|M_gen ∩ mask| / |M_gen| — fraction of the segmented change that stayed inside the mask.

    ``completeness`` returns an UNCLIPPED ``M_gen`` (core ∪ SAM), so a SAM segmentation that spills
    onto pre-existing structure outside the mask lowers this. NaN when ``M_gen`` is empty. (①)
    """
    a = int(mgen.sum())
    return float(np.logical_and(mgen, mask_bool).sum() / a) if a else float("nan")


def boundary_iou(mask: np.ndarray, mgen: np.ndarray, dilation_ratio: float = 0.02) -> float:
    gb = _mask_to_boundary(mask, dilation_ratio)
    db = _mask_to_boundary(mgen, dilation_ratio)
    union = np.logical_or(gb, db).sum()
    return float(np.logical_and(gb, db).sum() / union) if union > 0 else float("nan")


def _sam_segment_boxes(gen255: np.ndarray, boxes: np.ndarray, predictor) -> np.ndarray:
    """Segment N boxes in ONE batched SAM forward. ``boxes`` (N, 4) xyxy -> (N, H, W) bool."""
    predictor.set_image(gen255.astype(np.uint8))
    masks, _, _ = predictor.predict(box=np.asarray(boxes, np.float32), multimask_output=False)
    masks = np.asarray(masks)
    if masks.ndim == 4:  # (N, 1, H, W)
        masks = masks[:, 0]
    elif masks.ndim == 2:  # (H, W) — a single box collapsed
        masks = masks[None]
    return masks.astype(bool)


def _clusters(core: np.ndarray, mask_bool: np.ndarray):
    """Dilate-merge the diff ``core`` into clusters; return one xyxy box per surviving change region."""
    d = max(1, int(_MERGE_GAP_FRAC * np.hypot(*core.shape)))
    lab, n = ndimage.label(ndimage.binary_dilation(core, iterations=d))
    thr = _MIN_CLUSTER_FRAC * max(1, int(mask_bool.sum()))
    boxes = []
    for i in range(1, n + 1):
        g = lab == i
        if (g & core).sum() >= thr:
            ys, xs = np.nonzero(g)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes


def _mask_parts(mask_bool: np.ndarray):
    """Split a target mask into connected parts. Returns ``(labels, [part_indices])``."""
    lab, n = ndimage.label(mask_bool)
    parts = [i for i in range(1, n + 1) if int((lab == i).sum()) >= _MIN_COMP_PX]
    if not parts:  # all specks -> treat the whole mask as one part
        lab, parts = mask_bool.astype(int), [1]
    return lab, parts


def _min_over_parts(mgen: np.ndarray, mask_bool: np.ndarray) -> float:
    """Coverage of the WORST connected mask part — a defect is only as good as its least-filled part
    (denominator is always the input mask). ``_mask_parts`` always yields >=1 part, so ``covs`` is
    non-empty."""
    lab, parts = _mask_parts(mask_bool)
    covs = [float(np.logical_and(mgen, lab == i).sum() / max(1, int((lab == i).sum()))) for i in parts]
    return float(min(covs))


def _gate(sam: np.ndarray, core: np.ndarray, mask_area: int) -> bool:
    """SAM is trusted iff it agrees with the change core AND isn't much bigger than the mask."""
    overlap = np.logical_and(sam, core).sum() / max(1, int(core.sum()))
    return bool(overlap >= _SAM_OVL_MIN and sam.sum() / max(1, mask_area) <= _SAM_CAP)


def completeness(
    clean255: np.ndarray, gen255: np.ndarray, mask_bool: np.ndarray, predictor
) -> tuple[float, np.ndarray]:
    """Coverage of the WORST mask part (min) on a diff-anchored, gated-SAM segmentation.
    Returns ``(coverage_min, M_gen)``.

    Two SAM candidates for ``M_gen``, both gated against the diff core::

        whole    = SAM(whole-mask box)                 — one box over the whole mask
        clusters = SAM per diff-core cluster, unioned  — one box per localized change region

    Pick ``clusters`` when the change fragments into >=2 clusters (else ``whole``, else ``clusters``,
    else a low-threshold Otsu diff). ``clusters`` is the per-instance path — it never lets one box span
    disjoint regions — and preferring it on >=2 clusters keeps a partially-generated *missing* defect
    low, where a single whole-mask box would over-claim the still-present structure. SAM runs in ONE
    batched forward over ``[whole-mask box] + [cluster boxes]``. The score then splits the mask into
    connected parts and reports ``min_part |M_gen ∩ part| / |part|`` — only as good as the worst part.
    """
    at = int(mask_bool.sum())
    if at == 0:
        return float("nan"), np.zeros(mask_bool.shape, dtype=bool)
    core, diff, t = _extract_core(clean255, gen255)
    ys, xs = np.nonzero(mask_bool)
    whole_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    cluster_boxes = _clusters(core, mask_bool)
    sam = _sam_segment_boxes(gen255, np.asarray([whole_box] + cluster_boxes, np.float32), predictor)
    sam_whole = sam[0]
    sam_clusters = sam[1:].any(axis=0) if cluster_boxes else np.zeros(mask_bool.shape, dtype=bool)

    # Unclipped M_gen (no ∩ mask): precision must see change that spilled OUTSIDE the mask and
    # boundary_iou the true segmentation shape. Coverage re-intersects per mask part in
    # _min_over_parts, so clipping here would only hide spill (and pin precision at 1.0).
    mgen_whole = core | sam_whole
    mgen_clusters = core | sam_clusters
    # Never drop the fallback threshold below the noise floor — else sub-floor sensor noise / faithful
    # repaint (diff.max() < floor, so t == floor) would pass at 0.5·floor and score a full defect.
    mgen_diff = diff > max(_LOW_OTSU * t, _OTSU_FLOOR)  # neither SAM trusted: permissive diff
    whole_ok = _gate(sam_whole, core, at)
    clusters_ok = _gate(sam_clusters, core, at)

    # >=2 clusters: prefer the per-cluster union — a single whole-mask box would over-claim the intact
    # structure between fragments, which is what keeps a partially-generated *missing* defect low.
    if len(cluster_boxes) >= 2 and clusters_ok:
        mgen = mgen_clusters
    elif whole_ok:
        mgen = mgen_whole
    elif clusters_ok:
        mgen = mgen_clusters
    else:
        mgen = mgen_diff
    return _min_over_parts(mgen, mask_bool), mgen


# --------------------------------------------------------------------------- composite
_TERMS = ("completeness", "boundary_iou", "precision", "nn_score")

# Individual axes exposed as their own KPIs (raw, per-type macro) so each is visible on the
# training curve; the aq_nn composite is computed separately from _TERMS above.
_AXES = ("completeness", "precision", "boundary_iou")
# Public alias: the CLIs build their --score choices and CSV columns from the same tuple, so it is
# defined once here rather than restated next to each consumer.
GEOMETRY_AXES = _AXES


def _term(sample: dict, name: str, default: float = 0.0) -> float:
    v = sample.get(name)
    return default if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def _nanmean(values: list[float]) -> float:
    """Macro-mean that ignores NaN/None entries; NaN when nothing finite is present."""
    finite = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _aq_nn_score(sample: dict) -> float:
    """``aq_nn = completeness + nn_score`` — an equal-weight, bounded composite (both terms already in
    [0, 1], no cross-sample statistics) so it is directly comparable across validation passes and
    rises as the model improves (unlike a per-pass z-score, which recenters to ~0 every step and
    cannot track training). A missing (NaN) term contributes 0.
    """
    return _term(sample, "completeness") + _term(sample, "nn_score")


def compute_anomaly_quality_kpi(
    real_images_dict: dict,
    generated_images_dict: dict,
    correspondence_kpi: dict,
    device: Optional[str] = None,
) -> dict:
    """Compute ``aq_nn`` + the geometry axes per anomaly type and their macro average.

    Args:
        real_images_dict / generated_images_dict: the same dict structure
            :func:`compute_correspondence_kpi` consumes; ``generated`` must carry
            ``reconstructed_image`` (gen), ``original_image`` (clean), ``original_mask``,
            and ``img_path``.
        correspondence_kpi: the return value of :func:`compute_correspondence_kpi`; its
            per-sample ``nn_score`` supplies the ⑤ term of ``aq_nn`` (matched by ``path``).
        device: torch device for the SAM2 model (default: auto).

    Returns:
        ``{anomaly_name: {"aq_nn", "completeness", "precision", "boundary_iou"}, "Average": {...}}``.
        Each score is an absolute, cross-pass-comparable number (see :func:`_aq_nn_score`).
        Returns ``{}`` (with a logged warning) if the SAM2 model cannot be loaded, so the caller
        still writes NN / FID.
    """
    device = _resolve_device(device)
    try:
        predictor = _sam_predictor(device)
    except Exception as e:  # noqa: BLE001 — missing weights / deps must not crash training
        log.warning(f"anomaly_quality: SAM2 load failed ({e}); skipping aq_nn.")
        return {}

    result: dict = {}
    nn_means: list[float] = []
    axis_means: dict[str, list[float]] = {a: [] for a in _AXES}
    for anomaly_name in sorted(real_images_dict.keys()):
        gen = generated_images_dict.get(anomaly_name, {})
        cleans = gen.get("original_image", [])
        gens = gen.get("reconstructed_image", [])
        masks = gen.get("original_mask", [])
        paths = gen.get("img_path", [None] * len(gens))
        nn_by_path = {
            row.get("path"): row.get("nn_score")
            for row in correspondence_kpi.get(anomaly_name, {}).get("per_sample", [])
        }
        nn_vals: list[float] = []
        axis_vals: dict[str, list[float]] = {a: [] for a in _AXES}
        sample_rows: list[dict] = []
        for path, clean, gen_img, mask in zip(paths, cleans, gens, masks):
            clean255, gen255 = np.asarray(clean) * 255.0, np.asarray(gen_img) * 255.0
            mask_bool = np.asarray(mask) > 0.5
            try:
                comp, mgen = completeness(clean255, gen255, mask_bool, predictor)
                sample = {
                    "completeness": comp,
                    "boundary_iou": boundary_iou(mask_bool, mgen),
                    "precision": precision(mgen, mask_bool),
                    "nn_score": _to_float(nn_by_path.get(path)),
                }
            except Exception as e:  # noqa: BLE001 — one bad sample must not sink the batch
                log.warning(f"anomaly_quality[{anomaly_name}]: sample '{path}' failed ({e}); recording NaN.")
                sample = dict.fromkeys(_TERMS, float("nan"))
            # A wholly-failed sample (every term NaN — the except path above) is dropped from the
            # aq_nn mean via _nanmean below, matching how the axes drop it, rather than being counted
            # as a hard 0 that would drag the composite down while leaving the axes untouched.
            failed = math.isnan(_nanmean([sample.get(t) for t in _TERMS]))
            nn_vals.append(float("nan") if failed else _aq_nn_score(sample))
            for a in _AXES:
                axis_vals[a].append(_to_float(sample.get(a)))
            # Per-sample axes for downstream rankers (filtering and cumulative AQ-Rank refinement). Kept
            # under a distinct key from correspondence's ``per_sample`` so ValidationKPI's
            # ``kpi[name].update(vals)`` merge never clobbers the nn/mnn per-sample rows.
            sample_rows.append({"path": path, **{a: _to_float(sample.get(a)) for a in _AXES}})

        if nn_vals:
            result[anomaly_name] = {
                "aq_nn": _nanmean(nn_vals),
                **{a: _nanmean(axis_vals[a]) for a in _AXES},
                "per_sample_axes": sample_rows,
            }
            nn_means.append(result[anomaly_name]["aq_nn"])
            for a in _AXES:
                axis_means[a].append(result[anomaly_name][a])
        else:
            result[anomaly_name] = {k: float("nan") for k in ("aq_nn", *_AXES)}
            result[anomaly_name]["per_sample_axes"] = []

    if not nn_means:
        log.warning("anomaly_quality: no generated samples scored; skipping aq_nn.")
        return {}
    # Macro-average over anomaly types, matching the correspondence / FID KPI convention.
    result["Average"] = {
        "aq_nn": _nanmean(nn_means),
        **{a: _nanmean(axis_means[a]) for a in _AXES},
    }
    return result


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --- per-sample row augmentation -------------------------------------------------------------
# Shared by the `filter` and `evaluate` CLIs, so it lives here with compute_anomaly_quality_kpi
# rather than in either one — a CLI importing another CLI drags in that module's whole import
# graph and points the dependency sideways instead of down at the library.

# Compass hyper-parameters. The composite is aq_rank_score = rank(nn) + Σ ±rank(axis); these control
# how each axis's +/- sign is chosen from nn's own extremes (see compute_aq_rank_scores).
_COMPASS_PCT = 0.20  # top/bottom fraction of nn used as pseudo good/bad to orient each axis
# Permissive fixed-margin gate: |AUROC - 0.5| >= this admits the axis. It rejects constant/degenerate
# axes but, fit from only 2*k pseudo-labels, does NOT reliably reject a noisy one (a pure-noise axis
# passes well above chance). A significance-test gate (Mann-Whitney p-value) is planned follow-up.
_COMPASS_GATE = 0.05
COMPASS_MIN_N = 5  # fewer samples than this per type: skip the compass, rank on nn alone


def _rank01(values: Sequence) -> np.ndarray:
    """Percentile rank of each value in [0, 1] (ties averaged). NaN sorts lowest (rank 0)."""
    v = np.array([_to_float(x) for x in values], dtype=float)
    v = np.where(np.isnan(v), -np.inf, v)
    ranks = rankdata(v, method="average")  # 1..n
    return (ranks - 1.0) / max(len(v) - 1, 1)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """AUROC of ``scores`` vs binary ``labels`` via the rank-sum (Mann-Whitney U) identity.

    Returns None when one class is empty (no direction can be read).
    """
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_aq_rank_scores(rows: List[dict]) -> np.ndarray:
    """Compass-oriented composite over ``nn`` + the ``_AXES`` geometry axes, one score per row.

    Rank-normalise ``nn`` and each axis within these rows (a single anomaly type). Take ``nn``'s own
    top/bottom-``_COMPASS_PCT`` samples as pseudo good/bad, and for each axis add its rank with a
    ``+``/``-`` sign when its agreement with those pseudo labels clears ``_COMPASS_GATE``. That gate is
    a permissive fixed margin: it drops constant/degenerate axes but not merely-noisy ones, so the
    composite may carry weak terms (a Mann-Whitney significance gate is planned — see the module
    docstring). No ground-truth labels are used — the sign is self-supervised from ``nn``. Samples with
    a NaN ``nn`` (degenerate, no defect features) are forced last so they drop first, matching the
    plain-nn path. With fewer than ``COMPASS_MIN_N`` samples the direction is unreliable, so rank on
    ``nn`` alone.

    This is a **rank-relative candidate-pool** score, not a validation metric: filtering ranks one
    bucket, while quality refinement re-ranks Original plus the observed rounds before every draw and
    over the full pool for final selection. ``mean(rank01(x)) == 0.5`` for any pool, so do NOT register
    it in ``METRIC_SPECS`` / early-stop — a pass-relative score cannot track training across passes
    (the same trap that made the old z-scored composite unusable). The absolute, cross-pass ``aq_nn``
    is the composite for that. Living in this module keeps both consumers on the same implementation.
    """
    return compute_aq_rank_with_compass(rows)[0]


def compute_aq_rank_with_compass(rows: List[dict]) -> tuple[np.ndarray, dict]:
    """As :func:`compute_aq_rank_scores`, plus each axis's fitted ``auroc`` and ``sign`` for audit logs."""
    nn = np.array([_to_float(r.get("nn_score")) for r in rows], dtype=float)
    n = len(nn)
    nan_nn = np.isnan(nn)
    score = _rank01(nn)
    axes = {axis: {"auroc": None, "sign": 0} for axis in _AXES}
    if n >= COMPASS_MIN_N:
        order = np.argsort(np.where(nan_nn, -np.inf, nn))
        k = max(2, int(n * _COMPASS_PCT))
        pseudo_good = set(order[-k:].tolist())
        pseudo_bad = set(order[:k].tolist())
        idx = np.array([i for i in range(n) if i in pseudo_good or i in pseudo_bad])
        labels = np.array([1 if i in pseudo_good else 0 for i in idx])
        for axis in _AXES:
            ranked = _rank01([r.get(f"{axis}_score") for r in rows])
            auc = _auroc(ranked[idx], labels)
            sign = 0 if auc is None or abs(auc - 0.5) < _COMPASS_GATE else (1 if auc >= 0.5 else -1)
            axes[axis] = {"auroc": auc, "sign": sign}
            if sign:
                score = score + sign * ranked
    return np.where(nan_nn, -np.inf, score), {"pool_size": n, "axes": axes}


def augment_with_quality(kpi: Dict, aq_kpi: Dict) -> None:
    """Merge per-sample geometry axes from ``aq_kpi`` into ``kpi``'s per-sample rows (matched by
    ``path``) and add the ``aq_nn_score`` and compass ``aq_rank_score`` composites to each row, in place.

    Missing axes become NaN and simply drop out of the composite via the gate, so a partial
    anomaly_quality result degrades gracefully rather than crashing.
    """
    for anomaly_key, item in kpi.items():
        if anomaly_key == "Average":
            continue
        rows = item.get("per_sample", [])
        if not rows:
            continue
        axes_by_path = {r.get("path"): r for r in aq_kpi.get(anomaly_key, {}).get("per_sample_axes", [])}
        for row in rows:
            axis_row = axes_by_path.get(row["path"], {})
            for axis in _AXES:
                row[f"{axis}_score"] = _to_float(axis_row.get(axis))
            # aq_nn = completeness + nn_score (absolute, per-sample — the same composite validation
            # tracks). NaN in either term propagates, so route_by_scores drops an unscorable sample.
            row["aq_nn_score"] = row["completeness_score"] + _to_float(row.get("nn_score"))
        for row, q in zip(rows, compute_aq_rank_scores(rows)):
            row["aq_rank_score"] = float(q)
