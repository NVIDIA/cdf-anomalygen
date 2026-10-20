# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter generated anomaly images by a per-sample quality score, splitting into keep/drop trees.

Consumes the ``generate.py`` output tree (via :mod:`anomalygen.eval.utils`), scores every
generated sample, then keeps the top ``(1 - drop_ratio)`` fraction per anomaly type by the chosen
``--score``:

* ``nn`` / ``mnn`` — DINOv2 correspondence to the real refs (:func:`compute_correspondence_kpi`).
* ``completeness`` / ``precision`` / ``boundary_iou`` — one gated-SAM/geometry axis
  (:func:`compute_anomaly_quality_kpi`), each ranked on its own.
* ``aq_nn`` — the absolute, per-sample composite ``completeness + nn_score`` (the same one validation
  tracks). Unlike ``aq_rank`` it is *not* rank-relative — it ranks on absolute quality. Needs the axes.
* ``aq_rank`` — the composite ``aq_rank_score``, built on ``nn`` as the base: start from the ``nn``
  rank and layer the three geometry axes on top, ``aq_rank = rank(nn) + Σ ±rank(axis)``. Each axis is
  rank-normalised within the anomaly type and signed by whether it agrees with ``nn``'s own top/bottom
  samples (a label-free "compass"); a type with too few samples to fit a direction falls back to plain
  ``nn``. The sign gate is a permissive fixed-margin heuristic — it drops constant/degenerate axes but
  does *not* reliably reject a merely-noisy one, so the composite can carry weak terms (a significance
  gate is planned; see :func:`compute_aq_rank_scores`). ``nn`` stays the anchor, but the geometry axes
  can reorder samples
  (including the top one), so ``aq_rank`` is not guaranteed to match ``nn``'s ranking. Ranking per
  type removes any cross-type scale confound. See :func:`compute_aq_rank_scores`.

  ``aq_rank`` is opt-in, not the default. Across five internal datasets it had the best *mean*
  ranking of the schemes tried, but its sign is self-supervised from ``nn``'s extremes: a weak NN
  anchor can orient an axis incorrectly and score below plain ``nn`` (observed on one of the five).
  That filter-time counterexample is not replaced by refinement's later 2AFC pilot. AQ-Rank is not a
  human-alignment guarantee, so ``--score`` defaults to the simpler ``nn`` objective.

Note: this is the cosmos3 successor of the old GIQA-based ``filter.py``. cosmos3 ships NN/MNN
correspondence (and set-level FID) rather than the old per-sample GIQA metric, so filtering ranks
on NN/MNN (or the geometry axes above). FID is set-level and cannot rank individual samples, so it
is not a filter criterion.

  --gen_root     generate.py output dir:  {gen}/reconstructed_image/{key}_{idx}.png
                                          {gen}/original_mask/{key}_{idx}.png
                                          {gen}/original_image/{key}_{idx}.png
                                          {gen}/texture_ft_generation_result.csv  (optional)
  --real_root    training dataset root:   {texture}/anomaly_image/{defect}/<stem>.png
                                          {texture}/mask/{defect}/<stem>_mask.png
  --output_dir  keep/{recon,mask,orig} + drop/{recon,mask,orig}
                 + texture_ft_generation_result_filtered.csv
                 + keep,drop/texture_ft_generation_result.csv  (when the input CSV exists)
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Framework process setup (inference env, grad disabled, distributed init when WORLD_SIZE>1).
from cosmos_framework.inference.common.init import init_script
from cosmos_framework.utils import log

# Guarded because it disables autograd process-wide; importing this module must leave it alone.
if __name__ == "__main__":
    init_script(training=False)

from anomalygen.eval.anomaly_quality import GEOMETRY_AXES, augment_with_quality, compute_anomaly_quality_kpi
from anomalygen.eval.correspondence import (
    DEFAULT_BACKBONE,
    add_nn_scoring_args,
    compute_correspondence_kpi,
    nn_scoring_kwargs,
)
from anomalygen.eval.utils import (
    MASK_SUBDIR,
    ORIG_IMAGE_SUBDIR,
    RECON_SUBDIR,
    load_generated,
    load_real,
    resolve_anomaly_types,
)

_GEN_CSV = "texture_ft_generation_result.csv"
_FILTERED_CSV = "texture_ft_generation_result_filtered.csv"
_OUTPUT_FILENAME_COL = "output_filename"
_COPY_SUBDIRS = (RECON_SUBDIR, MASK_SUBDIR, ORIG_IMAGE_SUBDIR)

# Geometry axes (from compute_anomaly_quality_kpi) that the composite aq_rank_score rides on top of
# the DINOv2 nn anchor. Each is a bounded, absolute [0, 1] quantity (higher = better).
_GEOMETRY_AXES = GEOMETRY_AXES
# --score values that need the anomaly_quality axes (SAM + geometry), not just correspondence.
# ``aq_nn`` (= completeness + nn_score) needs the axes too, so it lives here alongside ``aq_rank``.
_AXIS_SCORES = frozenset({*_GEOMETRY_AXES, "aq_rank", "aq_nn"})
_SCORE_CHOICES = ["nn", "mnn", *_GEOMETRY_AXES, "aq_rank", "aq_nn"]


def filter_topk(scores: Sequence[float], drop_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Global top-K filtering by score, keeping the highest-scoring ``(1 - drop_ratio)`` fraction.

    Returns ``(keep_idx, drop_idx)`` as arrays of indices into ``scores``, keep first in descending
    score order.
    """
    sorted_idx = np.argsort(scores)[::-1]
    num_keep = int(round((1 - drop_ratio) * len(scores)))
    return sorted_idx[:num_keep], sorted_idx[num_keep:]


def _copy_sample(gen_root: str, output_path: str, split: str, recon_path: str) -> str:
    """Copy a sample's reconstructed/mask/original triple (same basename) into ``{split}/`` trees."""
    base = os.path.basename(recon_path)
    for sub in _COPY_SUBDIRS:
        shutil.copy(os.path.join(gen_root, sub, base), os.path.join(output_path, split, sub, base))
    return base


def route_by_scores(
    kpi: Dict, gen_root: str, output_path: str, drop_ratio: float, score: str = "nn"
) -> Tuple[List[str], List[str]]:
    """Split and copy generated samples into keep/drop trees given a correspondence ``kpi``.

    Ranks each anomaly type's ``per_sample`` rows by ``{score}_score`` and keeps the top
    ``(1 - drop_ratio)`` fraction. Returns ``(kept_basenames, dropped_basenames)``.
    """
    score_key = f"{score}_score"

    for split in ("keep", "drop"):
        for sub in _COPY_SUBDIRS:
            os.makedirs(os.path.join(output_path, split, sub), exist_ok=True)

    kept_basenames: List[str] = []
    dropped_basenames: List[str] = []
    for anomaly_key, item in kpi.items():
        if anomaly_key == "Average":
            continue
        rows = item.get("per_sample", [])
        if not rows:
            continue

        # NaN scores (degenerate samples with no defect features on the patch grid) must sort last so
        # they drop first — np.argsort otherwise places NaN at the front of the kept set.
        scores = np.nan_to_num(np.array([row[score_key] for row in rows], dtype=float), nan=-np.inf)
        paths = [row["path"] for row in rows]
        keep_idx, drop_idx = filter_topk(scores, drop_ratio)

        for idx in keep_idx:
            kept_basenames.append(_copy_sample(gen_root, output_path, "keep", paths[idx]))
        for idx in drop_idx:
            dropped_basenames.append(_copy_sample(gen_root, output_path, "drop", paths[idx]))

        log.info(f"[{anomaly_key}] kept: {len(keep_idx)} / dropped: {len(drop_idx)}")

    return kept_basenames, dropped_basenames


def filter_generated_images(
    gen_dict: Dict,
    real_dict: Dict,
    gen_root: str,
    output_path: str,
    drop_ratio: float,
    backbone: str = DEFAULT_BACKBONE,
    top_k: int = 3,
    score: str = "nn",
    nn_kwargs: Optional[dict] = None,
) -> Tuple[Dict, List[str], List[str]]:
    """Score generated samples, then split/copy them into keep/drop trees.

    Always runs correspondence (nn/mnn). ``nn_kwargs`` (from :func:`nn_scoring_kwargs`) overrides the
    nn/mnn scoring knobs (layer/readout/region_policy/inst_agg). For an axis/composite ``score`` it
    additionally runs :func:`compute_anomaly_quality_kpi` and folds the per-sample geometry axes +
    ``aq_rank_score`` into ``kpi`` via :func:`augment_with_quality`; if anomaly_quality produces
    nothing (e.g. the SAM2 weights are unavailable) it falls back to ranking on ``nn``.

    Returns ``(kpi, kept_basenames, dropped_basenames)`` where ``kpi`` is the correspondence result
    (augmented with the geometry axes when a non-correspondence ``score`` was requested).
    """
    kpi = compute_correspondence_kpi(real_dict, gen_dict, backbone=backbone, top_k=top_k, **(nn_kwargs or {}))
    if score in _AXIS_SCORES:
        aq_kpi = compute_anomaly_quality_kpi(real_dict, gen_dict, kpi)
        if aq_kpi:
            augment_with_quality(kpi, aq_kpi)
        else:
            log.warning(f"anomaly_quality produced no scores; falling back to '--score nn' from '{score}'.")
            score = "nn"
    kept_basenames, dropped_basenames = route_by_scores(kpi, gen_root, output_path, drop_ratio, score)
    return kpi, kept_basenames, dropped_basenames


def save_filter_result_csv(kpi: Dict, output_path: str) -> None:
    """Write per-sample scores for every anomaly type to ``{output}/<input>_filtered.csv``.

    Always writes nn/mnn; when :func:`augment_with_quality` has populated the geometry axes +
    composite (i.e. a non-correspondence ``--score`` was used) those columns are appended too.
    """
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, _FILTERED_CSV)
    axis_cols = [f"{a}_score" for a in _GEOMETRY_AXES] + ["aq_nn_score", "aq_rank_score"]
    has_composite = any(
        "aq_rank_score" in row for key, item in kpi.items() if key != "Average" for row in item.get("per_sample", [])
    )
    extra_cols = axis_cols if has_composite else []
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["anomaly_type", "filename", "nn_score", "mnn_score"] + extra_cols)
        for anomaly_key, item in kpi.items():
            if anomaly_key == "Average":
                continue
            for row in item.get("per_sample", []):
                writer.writerow(
                    [anomaly_key, os.path.basename(row["path"]), row["nn_score"], row["mnn_score"]]
                    + [row.get(c) for c in extra_cols]
                )
    log.info(f"Saved filtering results to {csv_path}")


def save_filtered_generation_csv(
    generated_path: str, output_path: str, kept_basenames: List[str], dropped_basenames: List[str]
) -> None:
    """Split ``generate.py``'s result CSV into keep/drop copies by output-filename basename."""
    gen_csv = os.path.join(generated_path, _GEN_CSV)
    if not os.path.exists(gen_csv):
        log.info(f"No {_GEN_CSV} found under {generated_path}, skipping CSV split")
        return

    with open(gen_csv, newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        rows = list(reader)

    def _write_split(split: str, target_names: List[str]) -> str:
        names = set(target_names)
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
        out_csv = os.path.join(output_path, split, _GEN_CSV)
        with open(out_csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                if os.path.basename(row[_OUTPUT_FILENAME_COL]) in names:
                    writer.writerow(row)
        return out_csv

    keep_csv = _write_split("keep", kept_basenames)
    drop_csv = _write_split("drop", dropped_basenames)
    log.info(f"Saved split generation CSVs to {keep_csv} and {drop_csv}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Filter generated anomaly images by a per-sample quality score.")
    parser.add_argument(
        "--gen_root", required=True, help="generate.py output dir (reconstructed_image/ + original_mask/)"
    )
    parser.add_argument(
        "--real_root", required=True, help="real dataset root ({texture}/anomaly_image/{defect} + mask)"
    )
    parser.add_argument("--output_dir", required=True, help="destination for keep/ and drop/ trees")
    parser.add_argument(
        "--anomaly_types",
        nargs="+",
        default=None,
        help="keys 'texture+defect' to score; default: derive from --recipe, else infer from --gen_root",
    )
    parser.add_argument(
        "--recipe", default=None, help="recipe YAML/JSON to derive anomaly_types from its 'anomaly_types'"
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="best-matching real refs to average per sample (-1 = use all)"
    )
    parser.add_argument(
        "--score",
        choices=_SCORE_CHOICES,
        default="nn",
        help="rank samples by this score (default: nn): nn/mnn (correspondence only, no SAM); a single "
        "axis completeness/precision/boundary_iou; aq_nn (absolute per-sample completeness + nn_score); "
        "or aq_rank (optional rank-relative compass composite of nn + the three axes; see the module docstring)",
    )
    parser.add_argument("--drop_ratio", type=float, required=True, help="fraction in [0, 1] to drop per anomaly type")
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=None,
        help="resize loaded images/masks to this square side; default None keeps native resolution",
    )
    add_nn_scoring_args(parser)  # nn/mnn knobs (default = validated zoom/12/worst25/min; override to restore old)
    args = parser.parse_args(argv)

    if not 0.0 <= args.drop_ratio <= 1.0:
        raise ValueError(f"--drop_ratio must be in [0, 1], got {args.drop_ratio}")

    recon_dir = os.path.join(args.gen_root, RECON_SUBDIR)
    anomaly_types = resolve_anomaly_types(args.anomaly_types, args.recipe, recon_dir)

    log.info("Loading generated images...")
    # The geometry axes / composite diff against the pre-edit clean image, so load it for those scores.
    generated = load_generated(
        args.gen_root,
        anomaly_types,
        target_size=args.model_input_size,
        with_original_image=args.score in _AXIS_SCORES,
    )
    if not generated:
        raise RuntimeError(f"No generated images found under {recon_dir} for {anomaly_types}.")

    log.info("Loading real images...")
    # Only score (and require real refs for) types that actually have generated images.
    real = load_real(args.real_root, list(generated.keys()), target_size=args.model_input_size)

    log.info(f"Filtering by {args.score.upper()} (top_k={args.top_k})...")
    kpi, kept_basenames, dropped_basenames = filter_generated_images(
        generated,
        real,
        args.gen_root,
        args.output_dir,
        args.drop_ratio,
        top_k=args.top_k,
        score=args.score,
        nn_kwargs=nn_scoring_kwargs(args),
    )

    save_filter_result_csv(kpi, args.output_dir)
    save_filtered_generation_csv(args.gen_root, args.output_dir, kept_basenames, dropped_basenames)


if __name__ == "__main__":
    main()
