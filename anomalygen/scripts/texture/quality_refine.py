# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-sample ``(guidance, crop_ratio)`` search — three subcommands.

``draw``   writes one search round's ``testcase.jsonl`` by re-drawing ``(guidance, crop_ratio)`` for
           **every** sample of the base testcase, keeping each mask placement fixed. Round 1 draws
           uniformly; given prior buckets (``--original`` / ``--rounds_dir``), rounds 2+ use per-sample
           Bayesian optimization — an independent GP over each sample's own observed
           ``(guidance, crop_ratio) -> score`` history, Thompson-sampled. Draws are seeded (``--seed``)
           so a round is reproducible for the same thread count and package versions.

``select`` picks the best-scoring render per sample across the original bucket and eligible rounds
           into a ``searched/`` bucket. Always runnable — with zero rounds every winner is the
           original, so ``searched/`` simply clones it and is guaranteed to exist for the downstream
           filter/regeneration + pseudo-label steps.

``run``    drives the whole loop: ``(draw -> generate -> evaluate) x --num_search_run``, then
           ``select`` and a closing ``evaluate`` on ``searched/``. It gates each round (one image per
           testcase row *and* its ``kpi.json``) and exits non-zero rather than assembling a stale
           ``searched/``. The scoring knobs it is given are forwarded to every round's ``evaluate``,
           so each round is scored the same way as the batch it is selected against.

``draw``'s Bayesian optimization is the only part that *computes* with numpy / scipy / scikit-learn /
threadpoolctl — ``select`` and the uniform draw path are pure stdlib logic — but all four are imported at
module load, so every subcommand needs them installed. scipy, scikit-learn and threadpoolctl are pinned in
``requirements.txt``; numpy is pinned by the vLLM install stage. ``anomalygen.eval.correspondence`` is imported at module load too, for the
``add_nn_scoring_args`` declaration ``run`` forwards to each round's ``evaluate.py``; it pulls in torch
and the DINOv2 backbone spec, so every subcommand needs those on the path as well. ``draw`` reads the
base ``testcase.jsonl`` (the mask-generation output, AMG or AMP, that produced the generation bucket); ``select`` reads
precomputed ``kpi.json`` files plus each bucket's generation CSV and copies the winning renders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from cosmos_framework.utils import log
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from threadpoolctl import threadpool_limits

from anomalygen import childproc
from anomalygen.configs.loader import read_experiment_field, register_recipe
from anomalygen.configs.texture.constants import DEFAULT_CROP_RATIO_RANGE, DEFAULT_GUIDANCE_RANGE, MIN_CROP_RATIO
from anomalygen.eval.anomaly_quality import COMPASS_MIN_N, GEOMETRY_AXES, compute_aq_rank_with_compass
from anomalygen.eval.correspondence import add_nn_scoring_args

_ORIGINAL_KPI_HELP = "kpi.json scored on --original"
_KPI_NAME = "kpi.json"
_KPI_NAME_HELP = "kpi filename inside each round dir"

_GEN_CSV = "texture_ft_generation_result.csv"

# Per-sample metrics the search can optimize and select on — every one of them is on the per_sample
# rows ``evaluate.py`` writes for each round. ``aq_rank`` is the exception to treating those stored
# values as fixed: whenever candidates compete, refinement recomputes it over Original + all observed
# rounds before every draw and over the full candidate pool for final selection. The zero-round clone
# path may copy the stored value into its audit CSV, but makes no ranking decision.
_SCORE_CHOICES = ["nn", "mnn", "completeness", "precision", "boundary_iou", "aq_nn", "aq_rank"]

# Scoring knobs forwarded verbatim to every round's evaluate.py. They all change the computed numbers,
# so a round scored without them is not comparable to the Step 5 KPI it is selected against. The nn/mnn
# knobs are declared by ``add_nn_scoring_args`` itself, so their names and defaults live in one place.
_EVAL_SCORING_FLAGS = ("top_k", "model_input_size", "nn_layer", "nn_readout", "nn_region_policy", "nn_inst_agg")


_SINGLE_KINDS = ("reconstructed_image", "original_image", "original_mask")
_MULTI_KINDS = ("annotated_image",)
_ROUND_DIGITS = 2  # decimal places for emitted guidance / crop_ratio (e.g. 1.23456 -> 1.23)


# --- draw: write one search round's testcase --------------------------------------------------


def _load_base_rows(base_testcase):
    """All rows from the base ``testcase.jsonl`` (blank lines skipped)."""
    rows = []
    with open(base_testcase) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _draw_rows(rows, rng, guidance_range, crop_ratio_range):
    """Copy every row and overwrite ``(guidance, crop_ratio)`` with a fresh uniform draw; every other
    field (the ``(clean, mask)`` pairing, seed, num_steps, ...) is preserved verbatim."""
    gl, gh = guidance_range
    cl, ch = crop_ratio_range
    drawn = []
    for row in rows:
        new = dict(row)
        new["guidance"] = round(rng.uniform(gl, gh), _ROUND_DIGITS)
        new["crop_ratio"] = round(rng.uniform(cl, ch), _ROUND_DIGITS)
        drawn.append(new)
    return drawn


# --- draw (BO): reconstruct prior observations ------------------------------------------------


def _numeric(v):
    """Finite float, or None for missing / "none" / NaN / infinity."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none")):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _round_number(name):
    """Positive integer from ``round_<N>``, or ``None`` for any other name."""
    name = str(name)
    suffix = name.removeprefix("round_")
    number = int(suffix) if name.startswith("round_") and suffix.isdecimal() else None
    return number if number is not None and number >= 1 and suffix == str(number) else None


def _round_dirs(rounds_dir, *, before_round=None, through_round=None):
    """Numeric ``round_<r>/`` buckets, ordered and optionally bounded by round number.

    Plain lexicographic sorting puts round_10 before round_2, which silently reorders the
    tie-break priority in ``_pick_best`` (and the select log) once a search exceeds 9 rounds.
    Non-numeric and non-positive names are not search rounds and are ignored consistently by draw
    and select.
    """
    if not rounds_dir or not Path(rounds_dir).is_dir():
        return []

    rounds = []
    for path in Path(rounds_dir).glob("round_*"):
        number = _round_number(path.name)
        if (
            path.is_dir()
            and number is not None
            and (before_round is None or number < before_round)
            and (through_round is None or number <= through_round)
        ):
            rounds.append((number, path))
    return [path for _, path in sorted(rounds)]


def _bucket_specs(
    original,
    original_kpi,
    rounds_dir,
    kpi_name,
    *,
    before_round=None,
    through_round=None,
    warn_incomplete=False,
):
    """Original plus complete, numeric round buckets inside the requested boundary."""
    specs = [(original, original_kpi, "original")]
    for round_dir in _round_dirs(rounds_dir, before_round=before_round, through_round=through_round):
        kpi = round_dir / kpi_name
        if (round_dir / _GEN_CSV).is_file() and kpi.is_file():
            specs.append((round_dir, kpi, round_dir.name))
        elif warn_incomplete:
            log.warning(f"skipping {round_dir}: missing {_GEN_CSV} or {kpi_name}")
    return specs


def _load_generation_rows(bucket_dir):
    """Read a generation CSV and report malformed/missing output names as a normal input error."""
    csv_path = Path(bucket_dir) / _GEN_CSV
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "output_filename" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path}: missing required output_filename column")
        rows = []
        for position, row in enumerate(reader, start=2):
            if not row.get("output_filename"):
                raise ValueError(f"{csv_path}: CSV row {position} has an empty output_filename")
            rows.append(row)
    return rows


def _generation_index(row, basename):
    """Strict sample index from a generation row; CSV position is not testcase position."""
    try:
        return int(row["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"generated output {basename} has no valid integer index") from exc


def _bucket_observations(bucket_dir, kpi_path, score_key):
    """One (index, guidance, crop_ratio, score) per sample index in a single bucket.

    Joins the bucket CSV to its kpi.json by output basename; keeps the best (max) score per index
    (a sample with num_generated_images>1 has several same-(g,c) outputs); drops rows whose score is
    missing/NaN or whose guidance/crop_ratio is non-numeric."""
    scores = _load_scores(kpi_path, score_key)  # {basename: score}
    best = {}  # index -> (g, c, score)
    for row in _load_generation_rows(bucket_dir):
        basename = os.path.basename(row["output_filename"])
        y = _numeric(scores.get(basename))
        g = _numeric(row.get("guidance"))
        c = _numeric(row.get("crop_ratio"))
        if y is None or g is None or c is None:
            continue
        idx = _generation_index(row, basename)
        if idx not in best or y > best[idx][2]:
            best[idx] = (g, c, y)
    return [(idx, g, c, y) for idx, (g, c, y) in best.items()]


def _load_observations(original, original_kpi, rounds_dir, score_key, kpi_name, before_round=None):
    """Per-sample observations over Original and complete rounds before ``before_round``."""
    obs = defaultdict(list)
    for bucket_dir, kpi_path, _ in _bucket_specs(
        original, original_kpi, rounds_dir, kpi_name, before_round=before_round
    ):
        for idx, g, c, y in _bucket_observations(bucket_dir, kpi_path, score_key):
            obs[idx].append((g, c, y))
    return obs


def _load_aq_rank_bucket(bucket_dir, kpi_path, source, generation_rows=None):
    """Load one candidate bucket and mark candidates eligible for pooled AQ-Rank.

    ``evaluate.py`` writes ``aq_rank_score`` inside each individual generation bucket, but that value
    is deliberately not used here: it was ranked against that bucket alone. Refinement must rank the
    observed cross-round pool anew before every BO draw (and once more for final selection).

    A candidate missing any finite raw axis is excluded from the rank pool and logged, while other
    candidates remain usable. Keeping the excluded candidate with a non-finite selection score lets
    final selection prefer a valid render without allowing a missing negatively-oriented axis to earn
    a favourable rank. Structural CSV/KPI join failures still stop the run.
    """
    data = json.loads(Path(kpi_path).read_text())
    ranked_inputs = {}
    for anomaly_type, block in data.items():
        if anomaly_type == "Average" or not isinstance(block, dict):
            continue
        for row in block.get("per_sample", []):
            basename = os.path.basename(row.get("path", ""))
            if not basename:
                continue
            metrics = {f"{axis}_score": _numeric(row.get(f"{axis}_score")) for axis in ("nn", *GEOMETRY_AXES)}
            ranked_inputs[basename] = {
                "anomaly_type": anomaly_type,
                "metrics": metrics,
                "mnn": row.get("mnn_score", float("nan")),
            }
    if not ranked_inputs:
        raise ValueError(f"AQ-Rank {kpi_path}: contains no per-sample rows")
    result = {}
    excluded = []
    rows = generation_rows if generation_rows is not None else _load_generation_rows(bucket_dir)
    for row in rows:
        basename = os.path.basename(row["output_filename"])
        item = ranked_inputs.get(basename)
        if item is None:
            raise ValueError(f"AQ-Rank {kpi_path}: no KPI row for generated output {basename}")
        missing = [key for key, value in item["metrics"].items() if value is None]
        if missing:
            excluded.append(f"{basename} ({', '.join(missing)})")
        result[basename] = {
            "row": row,
            "score": float("nan"),
            "nn": item["metrics"]["nn_score"],
            "mnn": item["mnn"],
            "metrics": item["metrics"],
            "rankable": not missing,
            "anomaly_type": item["anomaly_type"],
            "src_dir": str(bucket_dir),
            "source": source,
        }
    if not result:
        raise ValueError(f"AQ-Rank {bucket_dir}: {_GEN_CSV} has no candidate rows")
    if excluded:
        preview = "; ".join(excluded[:3]) + ("; ..." if len(excluded) > 3 else "")
        log.warning(
            f"{kpi_path}: excluding {len(excluded)}/{len(result)} generated candidate(s) from AQ-Rank "
            f"because a raw input is missing or non-finite: {preview}"
        )
    if not any(candidate["rankable"] for candidate in result.values()):
        raise ValueError(f"AQ-Rank {kpi_path}: no generated candidate has all finite raw inputs")
    return result


def _log_compass(anomaly_type, compass):
    """Log one anomaly type's compact, numeric compass diagnostics."""
    axis_details = []
    for axis, diagnostics in compass["axes"].items():
        auc = diagnostics["auroc"]
        auc_text = "n/a" if auc is None else f"{auc:.3f}"
        sign = diagnostics["sign"]
        sign_text = "+1" if sign > 0 else "-1" if sign < 0 else "disabled"
        axis_details.append(f"{axis}: AUROC={auc_text}, sign={sign_text}")
    log.info(
        f"AQ-Rank compass for anomaly type {anomaly_type!r}, pool={compass['pool_size']}; " + "; ".join(axis_details)
    )


def _rank_aq_pool(bucket_specs):
    """Load all buckets, then recompute AQ-Rank over their combined per-type candidate pools."""
    buckets = [_load_aq_rank_bucket(bucket_dir, kpi_path, source) for bucket_dir, kpi_path, source in bucket_specs]
    grouped = defaultdict(list)
    for bucket in buckets:
        for candidate in bucket.values():
            if candidate["rankable"]:
                grouped[candidate["anomaly_type"]].append(candidate)
    if grouped:
        log.info(
            "AQ-Rank compass directions are fitted across pooled samples and then applied when "
            "comparing variants within one sample"
        )
    for anomaly_type, candidates in grouped.items():
        scores, compass = compute_aq_rank_with_compass([candidate["metrics"] for candidate in candidates])
        if compass["pool_size"] < COMPASS_MIN_N:
            log.warning(
                f"AQ-Rank pool for anomaly type {anomaly_type!r} has {len(candidates)} candidate(s), "
                f"below {COMPASS_MIN_N}; ranking this type by NN only"
            )
        _log_compass(anomaly_type, compass)
        for candidate, score in zip(candidates, scores):
            candidate["score"] = float(score)
    return buckets


def _aq_rank_observations(original, original_kpi, rounds_dir, kpi_name, before_round=None):
    """Per-index BO observations after re-ranking Original plus every complete prior round.

    The same candidate's target is recomputed each invocation because adding a completed round changes
    its within-type percentile.  This is intentional cumulative acquisition, not a cross-round KPI.
    Raw files are intentionally re-read: refinement defaults to only three rounds, and operators may
    repair an incomplete CSV/KPI in place, for which a process-local cache would return stale history.
    """
    specs = _bucket_specs(original, original_kpi, rounds_dir, kpi_name, before_round=before_round)
    buckets = _rank_aq_pool(specs)
    observations = defaultdict(list)
    for bucket in buckets:
        best = {}
        for candidate in bucket.values():
            row = candidate["row"]
            guidance = _numeric(row.get("guidance"))
            crop_ratio = _numeric(row.get("crop_ratio"))
            if guidance is None or crop_ratio is None or not math.isfinite(candidate["score"]):
                continue
            basename = os.path.basename(row["output_filename"])
            index = _generation_index(row, basename)
            if index not in best or candidate["score"] > best[index][2]:
                best[index] = (guidance, crop_ratio, candidate["score"])
        for index, value in best.items():
            observations[index].append(value)
    return observations


def _draw_before_round(args):
    """Exclusive round-history boundary for one public ``draw`` invocation.

    ``run`` passes ``--current_round`` explicitly. For backward compatibility, direct ``draw`` calls
    using the canonical ``round_<N>/testcase.jsonl`` layout can still infer it from ``--output``. If
    history already exists and neither source identifies the round, fail rather than silently pooling
    current or future rounds left by an earlier run.
    """
    if args.current_round is not None:
        return args.current_round
    output_parent = Path(args.output).parent.name
    inferred = _round_number(output_parent)
    if inferred is not None:
        return inferred
    if _round_dirs(args.rounds_dir):
        raise ValueError(
            "--current_round is required when --rounds_dir already contains rounds and --output is "
            "not under canonical round_<N>/ with N >= 1; otherwise draw cannot exclude current/future stale rounds"
        )
    return None


# --- draw (BO): per-sample Gaussian-process proposal ------------------------------------------


def _distinct_xy(observations):
    """Number of distinct (guidance, crop_ratio) locations among observations, at emitted precision."""
    return len({(round(g, _ROUND_DIGITS), round(c, _ROUND_DIGITS)) for g, c, _ in observations})


def _bo_propose(observations, n, rng_seed, guidance_range, crop_ratio_range, n_candidates):
    """Thompson-sample n (guidance, crop_ratio) proposals from a GP fit on ``observations``.

    Fits a GP on the observations ([(g, c, y), ...], inputs scaled to [0,1]^2 by the ranges), then draws
    n independent posterior realizations over an LHS candidate pool and returns each realization's
    argmax. Deterministic for a fixed rng_seed."""
    gl, gh = guidance_range
    cl, ch = crop_ratio_range

    def _scale(g, c):
        return [(g - gl) / (gh - gl), (c - cl) / (ch - cl)]

    X = np.array([_scale(g, c) for g, c, _ in observations], dtype=float)
    y = np.array([yy for *_, yy in observations], dtype=float)

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=[0.2, 0.2], length_scale_bounds=(1e-2, 1e1), nu=2.5
    ) + WhiteKernel(1e-6, (1e-8, 1e-2))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=2, random_state=rng_seed
    )
    gp.fit(X, y)

    cand = LatinHypercube(d=2, seed=rng_seed).random(n_candidates)  # (n_candidates, 2) in [0,1)
    draws = gp.sample_y(cand, n_samples=n, random_state=rng_seed)  # (n_candidates, n)
    picks = cand[np.argmax(draws, axis=0)]  # (n, 2)
    return [
        (round(gl + px[0] * (gh - gl), _ROUND_DIGITS), round(cl + px[1] * (ch - cl), _ROUND_DIGITS)) for px in picks
    ]


def _sample_index(row, position):
    """The row's sample index — the key BO history is stored under.

    ``InpaintInferenceDataset`` does ``rec.setdefault("index", i)``, i.e. it honors an index already in
    the JSONL and only falls back to line position. Mirror that exactly: assuming position here would
    hand every sample another sample's history, silently, on any testcase that carries its own indices."""
    try:
        return int(row["index"])
    except (KeyError, TypeError, ValueError):
        return position


def _draw_rows_bo(rows, observations, seed, guidance_range, crop_ratio_range, bo_min_obs, n_candidates):
    """Overwrite (guidance, crop_ratio) on copies of rows via per-sample Bayesian optimization: fit an
    independent GP on each sample's own (guidance, crop_ratio) -> score history (``observations`` keyed
    by sample index) and Thompson-sample one point. A sample with < bo_min_obs observations over < 2
    distinct locations falls back to uniform. Row order and every other field are preserved. Fully
    seeded (per-sample seed derived from `seed`).

    Returns ``(drawn_rows, n_bo)`` — n_bo is how many samples actually used the GP, which is what the
    caller reports (an invocation with BO *enabled* can still be 100% uniform; see ``_run_draw``)."""
    gl, gh = guidance_range
    cl, ch = crop_ratio_range
    drawn, n_bo = [], 0
    # The cost of a draw is ``gp.sample_y``: numpy's ``multivariate_normal`` takes an SVD of the
    # ``n_candidates x n_candidates`` posterior covariance, so it grows cubically with --bo_candidates
    # (~0.5 s at 1024). BLAS spreads that SVD over one thread per core by default, and when the cores
    # are shared with other pipelines the threads mostly wait on each other: a batch that takes
    # ~0.5 s/sample capped took 5-50 s/sample uncapped. A few threads would be marginally faster than
    # one on an idle host, but one is the only setting that is safe on a shared one, and it also makes
    # the seeded picks reproducible across hosts (thread count changes floating-point order). Scoped to
    # the draw on purpose: ``_evaluate`` runs in this process afterwards and must not inherit the cap.
    with threadpool_limits(limits=1):
        for position, row in enumerate(rows):
            idx = _sample_index(row, position)
            obs = observations.get(idx, [])
            sample_seed = (seed * 100003 + idx) & 0x7FFFFFFF
            if len(obs) >= bo_min_obs and _distinct_xy(obs) >= 2:
                g, c = _bo_propose(obs, 1, sample_seed, guidance_range, crop_ratio_range, n_candidates)[0]
                n_bo += 1
            else:
                rng = random.Random(sample_seed)
                g = round(rng.uniform(gl, gh), _ROUND_DIGITS)
                c = round(rng.uniform(cl, ch), _ROUND_DIGITS)
            new = dict(row)
            new["guidance"] = g
            new["crop_ratio"] = c
            drawn.append(new)
    return drawn, n_bo


def _trained_crop_ratio_range(recipe):
    """The framing band ``recipe`` trained under, or ``None`` if it cannot be read.

    Generation clamps every requested ``crop_ratio`` into this band, so drawing outside it wastes
    the round on values that collapse onto its edges. Reads the registered experiment node rather
    than loading the model — the band is config, not weights.
    """
    if recipe is None:
        return None
    band = read_experiment_field(register_recipe(str(recipe)), "diffusion_expert_config.crop_ratio_range")
    return None if band is None else [float(band[0]), float(band[1])]


def _resolve_crop_ratio_range(args) -> list:
    """The crop_ratio draw range: the recipe's trained band, narrowed to an explicit
    ``--crop_ratio_range`` when one is given; the default when no band can be read.

    Generation clamps every crop_ratio into the band, so a draw outside it only repeats the band's
    edge — and the GP, scaled by the draw range, would keep proposing there. An explicit range is
    therefore intersected with the band rather than used as is. Re-checks LOW < HIGH because
    argparse only sees the explicit flag, not the band.
    """
    explicit = None if args.crop_ratio_range is None else [float(v) for v in args.crop_ratio_range]
    band = _trained_crop_ratio_range(getattr(args, "recipe", None))
    if band is None:
        if explicit is not None:
            return explicit
        # Silent here would sweep the default band against a recipe trained on another one.
        log.warning(
            f"could not read crop_ratio_range off the recipe; drawing over the default "
            f"{tuple(DEFAULT_CROP_RATIO_RANGE)}. Pass --crop_ratio_range to set it explicitly."
        )
        return list(DEFAULT_CROP_RATIO_RANGE)
    if not band[1] > band[0]:
        raise ValueError(f"recipe crop_ratio_range must be LOW < HIGH (got {band[0]} {band[1]})")
    if explicit is None:
        log.info(f"drawing crop_ratio over the recipe's trained band {tuple(band)}")
        return band
    low, high = max(explicit[0], band[0]), min(explicit[1], band[1])
    if not high > low:
        raise ValueError(
            f"--crop_ratio_range {tuple(explicit)} lies outside the recipe's trained band {tuple(band)}; "
            "generation clamps every draw into the band, so there is nothing to search there."
        )
    if [low, high] != explicit:
        log.warning(
            f"--crop_ratio_range {tuple(explicit)} narrowed to {(low, high)}, its overlap with the recipe's "
            f"trained band {tuple(band)}: generation clamps draws outside the band onto its edge."
        )
    return [low, high]


def _run_draw(args) -> None:
    args.crop_ratio_range = _resolve_crop_ratio_range(args)
    rows = _load_base_rows(args.base_testcase)
    if args.original and args.original_kpi:
        try:
            before_round = _draw_before_round(args)
            observations = _ScoreStrategy(args.score).observations(
                args.original,
                args.original_kpi,
                args.rounds_dir,
                args.kpi_name,
                before_round,
            )
        except (OSError, ValueError) as exc:
            log.error(str(exc))
            raise SystemExit(1) from exc
        drawn, n_bo = _draw_rows_bo(
            rows,
            observations,
            args.seed,
            args.guidance_range,
            args.crop_ratio_range,
            args.bo_min_obs,
            args.bo_candidates,
        )
        # Report the actual per-sample split, not just "BO was enabled". Round 1 supplies exactly one
        # observation per sample, so every sample falls back to uniform — logging "bayesopt" there
        # reads as if the search were already steering when it is still exploring blind.
        mode = (
            f"bayesopt {n_bo}/{len(drawn)} samples (rest uniform)" if n_bo else "uniform (no sample had enough history)"
        )
    else:
        drawn = _draw_rows(rows, random.Random(args.seed), args.guidance_range, args.crop_ratio_range)
        mode = "uniform"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in drawn:
            f.write(json.dumps(row) + "\n")
    log.info(
        f"wrote {len(drawn)} search-round samples -> {out} "
        f"(mode={mode}, guidance {tuple(args.guidance_range)}, crop_ratio {tuple(args.crop_ratio_range)}, seed {args.seed})"
    )


# --- select: pick the best render per sample across rounds ------------------------------------


def _load_scores(kpi_path, score_key, *, data=None):
    """``{output_basename: score}`` flattened across every anomaly type in a ``kpi.json`` (skips ``Average``)."""
    if data is None:
        data = json.loads(Path(kpi_path).read_text())
    scores = {}
    for name, block in data.items():
        if name == "Average" or not isinstance(block, dict):
            continue
        for row in block.get("per_sample", []):
            scores[os.path.basename(row["path"])] = row.get(score_key, float("nan"))
    return scores


def _has_usable_score(kpi_path, score_key) -> bool:
    """True when at least one sample in ``kpi_path`` carries a finite ``score_key``.

    A bucket whose kpi.json lacks the metric entirely does not merely rank low — it is removed from
    the competition. ``_load_scores`` defaults a missing key to NaN and ``_pick_best`` replaces an
    invalid incumbent with *any* finite candidate without comparing magnitudes, so an entirely
    unscored bucket loses every pick regardless of how good its renders are.
    """
    return any(_numeric(v) is not None for v in _load_scores(kpi_path, score_key).values())


def _unrankable_original(kpi_path, bucket_dir, score_key) -> str:
    """The error for an ``--original_kpi`` that cannot rank the original bucket on ``score_key``."""
    return (
        f"{kpi_path} carries no usable {score_key}: every value is missing or non-finite. select ranks the "
        "original bucket against the rounds, so it would be dropped from the competition and a round "
        "would win every sample even when strictly worse — while the run exited 0 and reported "
        f"'improved over original by search: N/N'. Re-run evaluate.py on {bucket_dir} so its KPI "
        f"carries {score_key} (an anomaly_quality metric needs the SAM2 checkpoint), then re-run."
    )


def _load_bucket(bucket_dir, kpi_path, score_key, source):
    """``{output_basename: {row, score, nn, mnn, src_dir, source}}`` for one generation bucket.

    ``score`` is the selection metric (``score_key``); ``nn``/``mnn`` are carried for the stitched
    per_sample.csv. All three are per-sample-independent (one render vs the real set), so copying them
    from the source bucket's kpi.json is exact — no re-eval needed."""
    data = json.loads(Path(kpi_path).read_text())
    nn = _load_scores(kpi_path, "nn_score", data=data)
    mnn = _load_scores(kpi_path, "mnn_score", data=data)
    # The anomaly-quality keys live on the same per_sample rows (evaluate.py folds them in), so they
    # load through the same path; nn/mnn are reused rather than re-read since they are always carried.
    if score_key == "nn_score":
        sel = nn
    elif score_key == "mnn_score":
        sel = mnn
    else:
        sel = _load_scores(kpi_path, score_key, data=data)
    out = {}
    for row in _load_generation_rows(bucket_dir):
        basename = os.path.basename(row["output_filename"])
        out[basename] = {
            "row": row,
            "score": sel.get(basename, float("nan")),
            "nn": nn.get(basename, float("nan")),
            "mnn": mnn.get(basename, float("nan")),
            "src_dir": str(bucket_dir),
            "source": source,
        }
    return out


def _pick_best(candidates):
    """Highest finite score; invalid scores rank worst and ties keep the earliest source."""
    best = None
    for cand in candidates:
        if best is None:
            best = cand
            continue
        cand_score = _numeric(cand["score"])
        best_score = _numeric(best["score"])
        if cand_score is not None and (best_score is None or cand_score > best_score):
            best = cand
    return best


class _ScoreStrategy:
    """Keep absolute-score and pooled AQ-Rank behavior behind one command-level interface."""

    def __init__(self, score):
        self.score_key = f"{score}_score"
        self.is_pooled_aq_rank = score == "aq_rank"

    def observations(self, original, original_kpi, rounds_dir, kpi_name, before_round):
        if self.is_pooled_aq_rank:
            return _aq_rank_observations(
                original,
                original_kpi,
                rounds_dir,
                kpi_name,
                before_round=before_round,
            )
        return _load_observations(
            original,
            original_kpi,
            rounds_dir,
            self.score_key,
            kpi_name,
            before_round=before_round,
        )

    def load_buckets(self, specs):
        if self.is_pooled_aq_rank and len(specs) > 1:
            return _rank_aq_pool(specs)
        return [_load_bucket(bucket_dir, kpi, self.score_key, source) for bucket_dir, kpi, source in specs]

    def check_original(self, original, original_kpi, original_dir, has_competition):
        """Raise when an absolute-score Original cannot fairly compete with a round."""
        if self.is_pooled_aq_rank or not has_competition or not original:
            return
        if any(_numeric(candidate["score"]) is not None for candidate in original.values()):
            return
        raise ValueError(_unrankable_original(original_kpi, original_dir, self.score_key))

    def pick(self, candidates, original, basename, has_competition):
        if (
            self.is_pooled_aq_rank
            and has_competition
            and not any(_numeric(candidate["score"]) is not None for candidate in candidates)
        ):
            log.warning(f"no eligible AQ-Rank candidate for {basename}; keeping Original without a ranking decision")
            return original
        return _pick_best(candidates)

    def validate(self, bucket_dir, kpi_path, source, generation_rows=None):
        """Raise when a bucket cannot be ranked by this strategy."""
        if self.is_pooled_aq_rank:
            _load_aq_rank_bucket(bucket_dir, kpi_path, source, generation_rows)
        elif not _has_usable_score(kpi_path, self.score_key):
            if source == "original":
                raise ValueError(_unrankable_original(kpi_path, bucket_dir, self.score_key))
            raise ValueError(
                f"{kpi_path}: no sample carries a usable {self.score_key} — every value is missing or "
                f"non-finite, so the search cannot rank the {source!r} bucket (an anomaly_quality metric "
                "needs the SAM2 checkpoint; check the evaluate step's log for an 'anomaly_quality "
                "computation failed' warning)"
            )


def _copy_sample_kinds(src_dir, basename, dst_dir) -> None:
    """Copy every KIND for one sample from ``src_dir`` into ``dst_dir`` under the same name. Single-file
    kinds copy 1:1; multi-instance kinds (annotated_image, ``<stem>_<NNNNN>.png``) are globbed per instance."""
    src_dir = Path(src_dir)
    stem = Path(basename).stem
    for k in _SINGLE_KINDS:
        src_f = src_dir / k / basename
        if src_f.exists():
            (dst_dir / k).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_dir / k / basename)
    for k in _MULTI_KINDS:
        for src_f in sorted((src_dir / k).glob(f"{stem}_*.png")):
            suffix = src_f.name[len(stem) :]  # e.g. "_00000.png"
            if not suffix[1:-4].isdigit():  # require _<digits>.png so a stem can't over-match a sibling
                continue
            (dst_dir / k).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_dir / k / src_f.name)


def _run_select(args) -> None:
    if Path(args.output).resolve() == Path(args.original).resolve():
        log.error("--output must differ from --original (assembling in place would delete the source bucket)")
        sys.exit(1)
    strategy = _ScoreStrategy(args.score)
    score_key = strategy.score_key
    specs = _bucket_specs(
        args.original,
        args.original_kpi,
        args.rounds_dir,
        args.kpi_name,
        through_round=args.max_round,
        warn_incomplete=True,
    )
    try:
        # With no competing round, use the ordinary loader even for aq_rank. This preserves the
        # promised clone-only path when Original predates the raw AQ axes; any stored bucket-local
        # score is audit data only and cannot affect the sole candidate's selection.
        buckets = strategy.load_buckets(specs)
        original = buckets[0]
        # The rounds are gated on carrying the metric; the original must clear the same bar or it
        # silently forfeits every pick. A clone-only path has nothing to compete with, so an unscored
        # original still wins by construction and searched/ can be produced for Step 7.
        strategy.check_original(
            original,
            args.original_kpi,
            args.original,
            has_competition=len(buckets) > 1,
        )
    except (OSError, ValueError) as exc:
        log.error(str(exc))
        raise SystemExit(1) from exc

    out_dir = Path(args.output)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for k in _SINGLE_KINDS:
        (out_dir / k).mkdir(parents=True, exist_ok=True)

    # Always emit BOTH metrics under their own names, plus which one selection ran on. Writing the
    # winning ``--score`` into a fixed "nn_score" column would silently put mnn values in a column
    # named nn_score under --score mnn; both are already loaded per bucket, so there is no reason to
    # collapse them.
    fieldnames = list(next(iter(original.values()))["row"].keys()) if original else []
    for extra in ("nn_score", "mnn_score", "selected_by", "source"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    def _fmt(x):
        value = _numeric(x)
        return f"{value:.6f}" if value is not None else ""

    out_rows, summary_rows, per_sample_rows = [], [], []
    by_source = defaultdict(int)
    improved = 0
    # Iterate the original bucket's basenames in CSV order so searched/ mirrors the original layout.
    for basename, orig in original.items():
        candidates = [b[basename] for b in buckets if basename in b]
        winner = strategy.pick(candidates, orig, basename, has_competition=len(buckets) > 1)
        _copy_sample_kinds(winner["src_dir"], basename, out_dir)

        by_source[winner["source"]] += 1
        if winner["source"] != "original":
            improved += 1
        new_row = dict(winner["row"])
        new_row["output_filename"] = basename
        new_row["nn_score"] = _fmt(winner["nn"])
        new_row["mnn_score"] = _fmt(winner["mnn"])
        new_row["selected_by"] = score_key
        new_row["source"] = winner["source"]
        out_rows.append(new_row)
        summary_rows.append(
            {
                "output_filename": basename,
                "source": winner["source"],
                "selected_by": score_key,
                "score": _fmt(winner["score"]),
                "original_score": _fmt(orig["score"]),
                "improved": "1" if winner["source"] != "original" else "0",
            }
        )
        per_sample_rows.append(
            {
                "anomaly_type": winner["row"].get("anomaly_type", ""),
                "path": str(out_dir / "reconstructed_image" / basename),
                "nn_score": _fmt(winner["nn"]),
                "mnn_score": _fmt(winner["mnn"]),
            }
        )

    with (out_dir / _GEN_CSV).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    with (out_dir / "search_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["output_filename", "source", "selected_by", "score", "original_score", "improved"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    # Per-sample eval table (stitched nn + mnn), mirroring the layout eval produces elsewhere.
    with (out_dir / "per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anomaly_type", "path", "nn_score", "mnn_score"])
        writer.writeheader()
        writer.writerows(per_sample_rows)

    n_rounds = len(buckets) - 1
    log.info(
        f"selected {len(out_rows)} best-per-sample renders into {out_dir} "
        f"({n_rounds} round(s), selected by {score_key})"
    )
    log.info("picked-from distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    log.info(f"improved over original by search: {improved}/{len(out_rows)}")


# --- run: the whole Step 6 loop (draw -> generate -> evaluate) x N, then select ------------------


def _round_dir(rounds_dir, r):
    return Path(rounds_dir) / f"round_{r}"


def _count_rows(testcase):
    return sum(1 for line in open(testcase) if line.strip())


def _blocked_count(round_dir):
    """Outputs the content-safety guardrail suppressed in this bucket.

    ``generate.py`` writes ``guardrail_blocked.csv`` whenever the guardrail ran — header-only when it
    blocked nothing — and not at all under ``--no-guardrail``, so an absent file means zero either way.
    """
    path = Path(round_dir) / "guardrail_blocked.csv"
    if not path.is_file():
        return 0
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _round_incomplete_reason(round_dir, expected_rows, kpi_name, score=None):
    """Why this round does not count, or None if it is complete.

    Both consumers key off ``kpi.json``: ``select`` scores buckets from it and the next round's BO
    pools it into history. A round that stopped early is otherwise dropped silently — the run still
    exits 0 — so the loop refuses to continue instead of wasting the remaining GPU time.

    A guardrail-blocked sample writes no image *by design*, so it counts toward the round rather than
    against it. Comparing images to rows alone would abort the whole search over one blocked caption.

    ``score`` is the metric the search optimises, and a present-but-unscored ``kpi.json`` is
    checked separately because it fails *silently* rather than loudly. ``evaluate.py`` degrades to
    NN/MNN(+FID) on any anomaly_quality failure — a missing SAM2 checkpoint is enough — so with an
    ``aq_*`` metric the rows can arrive complete and correctly counted but carry no ``score_key``.
    ``_load_scores`` then defaults every sample to NaN, ``_pick_best`` ranks NaN worst and keeps the
    earliest source on ties, and the original wins every sample: N rounds of generation burned for a
    byte-identical ``searched/``, exit 0. Requiring one real value turns that into a named failure.
    """
    round_dir = Path(round_dir)
    kpi_path = round_dir / kpi_name
    if not kpi_path.is_file():
        return f"missing {kpi_name}"
    image_names = {path.name for path in (round_dir / "reconstructed_image").glob("*.png")}
    images = len(image_names)
    blocked = _blocked_count(round_dir)
    if images + blocked != expected_rows:
        return f"{images} image(s) + {blocked} guardrail-blocked for {expected_rows} testcase row(s)"
    csv_path = round_dir / _GEN_CSV
    if not csv_path.is_file():
        return f"missing {_GEN_CSV}"
    try:
        generation_rows = _load_generation_rows(round_dir)
    except (OSError, ValueError) as exc:
        return str(exc)
    csv_names = [os.path.basename(row["output_filename"]) for row in generation_rows]
    if len(csv_names) != len(image_names) or set(csv_names) != image_names:
        return f"{_GEN_CSV} does not match the {images} reconstructed image(s)"
    if score:
        strategy = _ScoreStrategy(score)
        try:
            strategy.validate(round_dir, kpi_path, round_dir.name, generation_rows)
        except (OSError, ValueError) as exc:
            return str(exc)
    return None


def _generate_argv(args, testcase, output_dir):
    """The generate.py argv for one round.

    Every knob that changes what the model produces is forwarded, for the same reason
    ``_evaluate_argv`` forwards its scoring knobs: a round is only worth comparing against the
    original bucket if it was produced the same way. Leaving these to generate.py's own defaults
    made the rounds silently differ from Step 4 -- ``--guardrail`` defaults to ON, so a
    ``--no-guardrail`` batch was searched with the guardrail running, and a ``--base_checkpoint``
    chosen to override the recipe's model_size was ignored, putting different base weights behind
    the renders that ``select`` then picks between.
    """
    argv = [
        "--checkpoint",
        str(args.checkpoint),
        "--recipe",
        str(args.recipe),
        "--input_data_path",
        str(testcase),
        "--output_dir",
        str(output_dir),
    ]
    if getattr(args, "base_checkpoint", None):
        argv += ["--base_checkpoint", str(args.base_checkpoint)]
    if getattr(args, "base_seed", None) is not None:
        argv += ["--base_seed", str(args.base_seed)]
    # BooleanOptionalAction on the other side, so the negative form is a distinct flag, and only the
    # explicit `--no-guardrail` needs sending: absent, generate.py's own default (on) applies.
    if getattr(args, "guardrail", True) is False:
        argv.append("--no-guardrail")
    return argv


def _evaluate_argv(args, gen_root, output_file):
    argv = [
        "--gen_root",
        str(gen_root),
        "--real_root",
        str(args.real_root),
        "--recipe",
        str(args.recipe),
        "--output_file",
        str(output_file),
    ]
    # Forward only what the caller set, so evaluate.py's own defaults stay authoritative. Without this
    # a run that scored Step 5 with non-default knobs would search against differently-computed
    # numbers and select winners on a metric that is not the one being reported.
    for name in _EVAL_SCORING_FLAGS:
        value = getattr(args, name, None)
        if value is not None:
            argv += [f"--{name}", str(value)]
    return argv


def _generate_env(output_root, environ=None):
    """The environment for the torchrun child: ours, minus torchrun's own vars, plus an explicit
    output root.

    ``IMAGINAIRE_OUTPUT_ROOT`` is worth setting rather than inheriting because the framework falls
    back to ``/tmp/imaginaire4-output`` when it is unset (``utils/config.py``) — so an unset value does
    not fail, it quietly resolves the config's job paths somewhere else entirely.
    """
    env = childproc.without_torchrun_vars(environ)
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    return env


def _generate_cmd(args, testcase, output_dir):
    """The torchrun command for one round's generation.

    Every round launches a *fresh* torchrun, so each needs its own rendezvous: the static backend pins
    ``MASTER_PORT=29500``, and a round starting before the previous one released that port fails to
    bind — a flake that appears mid-search on a busy machine, not on the first round. ``--standalone``
    asks for a c10d rendezvous on an ephemeral port plus a unique run id (``run.py``: ``rdzv_backend =
    "c10d"``, ``rdzv_endpoint = "localhost:0"``, ``rdzv_id = uuid4()``).

    Requesting it explicitly rather than relying on the default: torch >=2.6 already applies the same
    three settings when ``--nnodes`` is 1 and no endpoint or master port is given, but that is an
    implicit fallback guarded by conditions a future caller could break simply by passing one more
    flag. Saying ``--standalone`` states the requirement instead of inheriting it.
    """
    return [
        *childproc.torchrun(args.num_gpus, Path(__file__).with_name("generate.py")),
        *_generate_argv(args, testcase, output_dir),
    ]


def _generate(args, testcase, output_dir):
    """generate.py needs torch.distributed, so it is launched through torchrun rather than called."""
    cmd = _generate_cmd(args, testcase, output_dir)
    log.info(f"$ IMAGINAIRE_OUTPUT_ROOT={args.output_root} " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=_generate_env(args.output_root)).returncode


def _evaluate(args, gen_root, output_file):
    argv = _evaluate_argv(args, gen_root, output_file)
    log.info("$ evaluate.py " + " ".join(argv))
    if args.dry_run:
        return
    from anomalygen.scripts.texture import evaluate

    evaluate.main(argv)


def _draw_round(args, out_testcase, round_number):
    argv = [
        "draw",
        "--base_testcase",
        str(args.base_testcase),
        "--output",
        str(out_testcase),
        "--seed",
        str(round_number),
        "--current_round",
        str(round_number),
        "--original",
        str(args.original),
        "--original_kpi",
        str(args.original_kpi),
        "--rounds_dir",
        str(args.rounds_dir),
        "--score",
        args.score,
        "--kpi_name",
        args.kpi_name,
        "--bo_min_obs",
        str(args.bo_min_obs),
        "--bo_candidates",
        str(args.bo_candidates),
        "--guidance_range",
        str(args.guidance_range[0]),
        str(args.guidance_range[1]),
        "--crop_ratio_range",
        str(args.crop_ratio_range[0]),
        str(args.crop_ratio_range[1]),
    ]
    log.info("$ quality_refine.py " + " ".join(argv))
    if args.dry_run:
        return
    main(argv)


def _run_run(args) -> None:
    """draw -> generate -> evaluate for each round, gate each round, then select and score."""
    # Resolve once here so every round draws over the same band and the forwarded argv records it.
    args.crop_ratio_range = _resolve_crop_ratio_range(args)
    expected_rows = 0 if args.dry_run else _count_rows(args.base_testcase)
    strategy = _ScoreStrategy(args.score)
    original_kpi = Path(args.original_kpi)
    # Fail before the first round rather than after the last: select ranks the original bucket against
    # the rounds, so an original that cannot be ranked forfeits every pick and the whole search is
    # decided before it starts. Skipped with no rounds (select then just clones the original), and
    # guarded on the file existing so a missing --original_kpi still surfaces at select, where it
    # always has.
    if not args.dry_run and args.num_search_run > 0 and original_kpi.is_file():
        try:
            strategy.validate(args.original, original_kpi, "original")
        except (OSError, ValueError) as exc:
            log.error(str(exc))
            sys.exit(1)

    for r in range(1, args.num_search_run + 1):
        rd = _round_dir(args.rounds_dir, r)
        log.info(f"--- refinement round {r}/{args.num_search_run} -> {rd} ---")
        testcase = rd / "testcase.jsonl"
        _draw_round(args, testcase, round_number=r)
        rc = _generate(args, testcase, rd)
        if rc != 0:
            log.error(f"round {r}: generate.py exited {rc} — stopping before select.")
            sys.exit(rc)
        _evaluate(args, rd, rd / args.kpi_name)
        if args.dry_run:
            continue
        reason = _round_incomplete_reason(rd, expected_rows, args.kpi_name, args.score)
        if reason:
            log.error(
                f"round {r} is incomplete ({reason}) — stopping. Re-run this round's generate.py "
                "(it overwrites) and evaluate.py; a searched/ assembled without it is stale."
            )
            sys.exit(1)

    log.info(f"--- select best-per-sample -> {args.output} ---")
    if not args.dry_run:
        args.max_round = args.num_search_run
        _run_select(args)
    _evaluate(args, args.output, args.final_kpi)
    log.info(f"refinement complete: {args.output} scored into {args.final_kpi}")


# --- CLI ---------------------------------------------------------------------------------------


def _get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sample (guidance, crop_ratio) search.")
    sub = parser.add_subparsers(dest="command", required=True)

    draw = sub.add_parser("draw", help="write one search round's testcase.jsonl (redraw every sample)")
    draw.add_argument(
        "--base_testcase", required=True, help="base testcase.jsonl (the mask-generation output, AMG or AMP)"
    )
    draw.add_argument("--output", required=True, help="round testcase.jsonl to write")
    draw.add_argument(
        "--guidance_range",
        type=float,
        nargs=2,
        default=DEFAULT_GUIDANCE_RANGE,
        metavar=("LOW", "HIGH"),
        help="uniform draw range for guidance",
    )
    draw.add_argument(
        "--recipe",
        default=None,
        help="run-dir recipe copy — read for the trained crop_ratio band when --crop_ratio_range is omitted",
    )
    draw.add_argument(
        "--crop_ratio_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "crop_ratio draw range; defaults to the band --recipe trained under "
            f"(or {tuple(DEFAULT_CROP_RATIO_RANGE)} without one) and is narrowed to that band when given, "
            "since generation clamps into it"
        ),
    )
    draw.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducible draws (vary per round, e.g. the round number)"
    )
    draw.add_argument(
        "--original", default=None, help="original generation bucket (enables BO together with --original_kpi)"
    )
    draw.add_argument("--original_kpi", default=None, help=_ORIGINAL_KPI_HELP)
    draw.add_argument(
        "--rounds_dir", default=None, help="dir of prior round_<r>/ buckets (each with a kpi.json) to pool into BO"
    )
    draw.add_argument(
        "--current_round",
        type=int,
        default=None,
        help="current 1-based round; excludes this and later stale rounds from BO history (run sets it automatically)",
    )
    draw.add_argument("--score", choices=_SCORE_CHOICES, default="nn", help="per-sample metric to optimize")
    draw.add_argument("--kpi_name", default=_KPI_NAME, help=_KPI_NAME_HELP)
    draw.add_argument(
        "--bo_min_obs",
        type=int,
        default=2,
        help="min observations per sample to use BO (else uniform); per-sample history is thin, keep low",
    )
    draw.add_argument("--bo_candidates", type=int, default=1024, help="Thompson-sampling candidate pool size")
    draw.set_defaults(func=_run_draw)

    select = sub.add_parser("select", help="select best-per-sample renders across rounds into searched/")
    select.add_argument("--original", required=True, help="original generation bucket (${OUT})")
    select.add_argument("--original_kpi", required=True, help=_ORIGINAL_KPI_HELP)
    select.add_argument(
        "--rounds_dir",
        default=None,
        help="dir of round_<r>/ buckets each with a kpi.json; omit/empty to clone --original",
    )
    select.add_argument(
        "--max_round",
        type=int,
        default=None,
        help="highest round_<N> allowed to compete; 0 clones Original, default includes every round",
    )
    select.add_argument("--output", required=True, help="searched/ bucket to write")
    select.add_argument("--score", choices=_SCORE_CHOICES, default="nn", help="per-sample metric to compare")
    select.add_argument("--kpi_name", default=_KPI_NAME, help=_KPI_NAME_HELP)
    select.set_defaults(func=_run_select)

    run = sub.add_parser("run", help="the whole Step 6 loop: (draw -> generate -> evaluate) xN, select, score")
    run.add_argument("--base_testcase", required=True, help="generation testcase.jsonl (AMG or AMP output)")
    run.add_argument("--original", required=True, help="original generation bucket (${OUT})")
    run.add_argument("--original_kpi", required=True, help=_ORIGINAL_KPI_HELP)
    run.add_argument("--rounds_dir", required=True, help="dir the round_<r>/ buckets are written under")
    run.add_argument("--output", required=True, help="searched/ bucket to assemble")
    run.add_argument("--final_kpi", required=True, help="where to score the assembled searched/ bucket")
    run.add_argument("--checkpoint", required=True, help="fine-tuned checkpoint for generate.py")
    run.add_argument("--recipe", required=True, help="run-dir recipe copy — must match --checkpoint")
    run.add_argument("--real_root", required=True, help="dataset root, the evaluation real-reference")
    run.add_argument("--num_search_run", type=int, default=3, help="search rounds; 0 = select-only (clone)")
    run.add_argument("--num_gpus", type=int, default=1, help="torchrun --nproc_per_node for generate.py")
    run.add_argument(
        "--output_root",
        default="results",
        help="IMAGINAIRE_OUTPUT_ROOT for generate.py; set explicitly because the framework otherwise "
        "falls back to /tmp/imaginaire4-output without complaining.",
    )
    # Forwarded to every round's generate.py so the rounds are produced the way the original bucket
    # was; see _generate_argv.
    run.add_argument("--base_checkpoint", default=None, help="base checkpoint for generate.py (default: from --recipe)")
    run.add_argument("--base_seed", type=int, default=None, help="generate.py --base_seed (default: its own)")
    run.add_argument(
        "--guardrail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the content-safety guardrail in each round, matching generate.py's default",
    )
    run.add_argument("--score", choices=_SCORE_CHOICES, default="nn", help="per-sample metric to optimize")
    run.add_argument("--kpi_name", default=_KPI_NAME, help=_KPI_NAME_HELP)
    run.add_argument("--guidance_range", type=float, nargs=2, default=DEFAULT_GUIDANCE_RANGE, metavar=("LOW", "HIGH"))
    run.add_argument(
        "--crop_ratio_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="defaults to the band --recipe trained under; narrowed to that band when given",
    )
    run.add_argument("--bo_min_obs", type=int, default=2, help="min observations per sample before BO")
    run.add_argument("--bo_candidates", type=int, default=1024, help="Thompson-sampling candidate pool size")
    # The scoring knobs each round's evaluate.py is run with. add_nn_scoring_args is evaluate's own
    # declaration, so the nn/mnn flag names, choices and defaults cannot drift apart from it.
    add_nn_scoring_args(run)
    run.add_argument("--top_k", type=int, default=None, help="evaluate.py --top_k for every round")
    run.add_argument("--model_input_size", type=int, default=None, help="evaluate.py --model_input_size")
    run.add_argument("--dry_run", action="store_true", help="print each stage's command without running it")
    run.set_defaults(func=_run_run)

    args = parser.parse_args(argv)
    if args.command in ("draw", "run"):
        # A degenerate range would divide by zero when scaling to the GP's unit box, producing
        # nan/inf proposals that argparse would otherwise let straight through into the testcase.
        for name in ("guidance_range", "crop_ratio_range"):
            pair = getattr(args, name)
            if pair is None:  # resolved from the recipe at run time
                continue
            low, high = pair
            if not high > low:
                parser.error(f"--{name} must be LOW < HIGH (got {low} {high})")
        # Same floor a recipe's band has: no draw under it survives generation's clamp.
        if args.crop_ratio_range is not None and args.crop_ratio_range[0] < MIN_CROP_RATIO:
            parser.error(f"--crop_ratio_range LOW must be at least {MIN_CROP_RATIO} (got {args.crop_ratio_range[0]})")
    if args.command == "draw" and args.current_round is not None and args.current_round < 1:
        parser.error(f"--current_round must be >= 1 (got {args.current_round})")
    if args.command == "select" and args.max_round is not None and args.max_round < 0:
        parser.error(f"--max_round must be >= 0 (got {args.max_round})")
    return args


def main(argv=None) -> None:
    args = _get_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
