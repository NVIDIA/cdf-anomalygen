# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the quality_refine subcommands.

``draw`` redraws every sample's params, ``select`` picks the best render per sample across rounds, and
``run`` is the Step 6 loop over both that the skill used to ask an agent to hand-execute.
"""

import csv
import importlib.util
import json
import pathlib
import random
import sys
import types
from collections import defaultdict

import pytest

from anomalygen.scripts.texture import quality_refine

_RANGES = {"guidance_range": (1.5, 10.0), "crop_ratio_range": (1.5, 8.0)}
_RICH = [(2.0, 2.0, 0.1), (8.0, 6.0, 0.4)]  # >=2 observations over 2 distinct locations -> BO engages
_POOR = [(6.0, 2.0, 0.2)]  # a single observation -> falls back to a uniform draw


def _main(cmd, **flags):
    """quality_refine.main with flags spelled as kwargs — the argv lists are long enough to bury the
    one flag a test is actually about."""
    argv = [cmd]
    for key, val in flags.items():
        vals = val if isinstance(val, (list, tuple)) else [val]
        argv += [f"--{key}", *(str(v) for v in vals)]
    return quality_refine.main(argv)


# --- builders ------------------------------------------------------------------------------------


def _write_gen_csv(bucket_dir, rows):
    """Writes the real texture_ft_generation_result.csv header so the loaders see production columns."""
    bucket_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "output_filename", "image_filename", "mask_filename", "anomaly_type", "guidance", "num_steps",
        "seed", "num_generated_images", "crop_ratio", "poisson_blend", "PSNR", "index",
    ]  # fmt: skip
    with (bucket_dir / quality_refine._GEN_CSV).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)


def _gen_rows(n, guidance, crop_ratio, atype="A+x"):
    return [
        {
            "output_filename": f"{atype}_{i:05d}.png",
            "anomaly_type": atype,
            "guidance": guidance,
            "crop_ratio": crop_ratio,
            "index": i,
        }
        for i in range(n)
    ]


def _write_base_testcase(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_bucket(bucket_dir, samples, tag):
    """samples: [(basename, index)]. Recon PNGs are tagged with ``tag`` so buckets are tellable apart."""
    (bucket_dir / "reconstructed_image").mkdir(parents=True, exist_ok=True)
    for base, _idx in samples:
        (bucket_dir / "reconstructed_image" / base).write_bytes(tag.encode())
    with (bucket_dir / quality_refine._GEN_CSV).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["output_filename", "anomaly_type", "index"])
        w.writeheader()
        for base, idx in samples:
            w.writerow({"output_filename": base, "anomaly_type": base.rsplit("_", 1)[0], "index": idx})


def _write_kpi_nn_mnn(path, scores):
    """kpi.json with independently-set metrics (``scores``: {basename: (nn, mnn)})."""
    per_type = defaultdict(list)
    for base, (nn, mnn) in scores.items():
        per_type[base.rsplit("_", 1)[0]].append(
            {"path": f"/x/reconstructed_image/{base}", "nn_score": nn, "mnn_score": mnn}
        )
    data = {atype: {"nn_score": 0.0, "mnn_score": 0.0, "per_sample": ps} for atype, ps in per_type.items()}
    data["Average"] = {"nn_score": 0.0, "mnn_score": 0.0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_aq_rank_kpi(path, scores):
    """CPU-only KPI fixture with every raw axis needed by refinement's ``aq_rank`` mode.

    ``scores`` maps a basename to ``(nn, completeness, precision, boundary_iou)``.
    """
    per_type = defaultdict(list)
    for base, (nn, completeness, precision, boundary_iou) in scores.items():
        per_type[base.rsplit("_", 1)[0]].append(
            {
                "path": f"/x/reconstructed_image/{base}",
                "nn_score": nn,
                "mnn_score": nn,
                "completeness_score": completeness,
                "precision_score": precision,
                "boundary_iou_score": boundary_iou,
            }
        )
    data = {atype: {"per_sample": rows} for atype, rows in per_type.items()}
    data["Average"] = {"nn_score": 0.0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _make_kpi(path, scores):
    """kpi.json where each sample's nn == mnn (``scores``: {basename: value})."""
    _write_kpi_nn_mnn(path, {base: (v, v) for base, v in scores.items()})


def _ramped_kpi(path, n, start, atype="A+x"):
    _make_kpi(path, {f"{atype}_{i:05d}.png": start + 0.1 * i for i in range(n)})


def _read_summary(searched):
    with (searched / "search_summary.csv").open() as f:
        return {r["output_filename"]: r for r in csv.DictReader(f)}


def _read_gen_csv(searched):
    with (searched / quality_refine._GEN_CSV).open() as f:
        return list(csv.DictReader(f))


# --- draw ----------------------------------------------------------------------------------------


def test_draw_rows_covers_every_sample_within_range_preserving_other_fields():
    rows = [
        {"index": i, "anomaly_type": "A+x", "mask_filename": f"m{i}.png", "guidance": 6.0, "crop_ratio": 2.0, "seed": 1}
        for i in range(2)
    ]
    drawn = quality_refine._draw_rows(rows, random.Random(0), (1.5, 10.0), (1.5, 8.0))
    assert len(drawn) == 2  # ALL samples redrawn, never gated on score
    for d in drawn:
        assert 1.5 <= d["guidance"] <= 10.0
        assert 1.5 <= d["crop_ratio"] <= 8.0
        assert d["mask_filename"] in ("m0.png", "m1.png") and d["seed"] == 1  # placement/other fields intact
    assert all(r["guidance"] == 6.0 for r in rows)  # base rows untouched


def test_draw_writes_a_full_round_for_every_sample(tmp_path):
    base = tmp_path / "testcase.jsonl"
    _write_base_testcase(
        base,
        [{"index": i, "anomaly_type": t, "guidance": 6.0, "crop_ratio": 2.0} for i, t in enumerate("xxy")],
    )
    base.write_text(base.read_text().replace("\n", "\n\n", 1))  # a blank line must not drop a sample
    out = tmp_path / "rounds" / "round_1" / "testcase.jsonl"
    _main("draw", base_testcase=base, output=out, seed=1)
    written = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert [r["index"] for r in written] == [0, 1, 2]  # every sample, always
    assert any(r["guidance"] != 6.0 for r in written)  # params were redrawn


def test_draw_primitives_are_seed_reproducible_and_two_decimal():
    """Rounds are replayed from disk, so a draw that drifts between runs makes a round
    irreproducible; the 2dp rounding keeps the emitted JSONL diffable. Covers both the uniform and
    the BO proposal path, plus Thompson sampling's per-sample diversity."""
    rows = [{"index": i, "guidance": 6.0, "crop_ratio": 2.0} for i in range(20)]
    obs = [(g, c, -((g - 7.0) ** 2 + (c - 4.0) ** 2)) for g in (2.0, 4.0, 6.0, 8.0) for c in (2.0, 4.0, 6.0)]

    def draw():
        return quality_refine._draw_rows(rows, random.Random(42), (1.5, 10.0), (1.5, 8.0))

    def propose():
        return quality_refine._bo_propose(obs, 8, 3, (1.5, 10.0), (1.5, 8.0), 256)

    assert draw() == draw() and propose() == propose()  # reproducible for a fixed seed
    assert len(set(propose())) > 1  # distinct per-sample proposals (Thompson sampling)
    for g, c in [(d["guidance"], d["crop_ratio"]) for d in draw()] + propose():
        assert round(g, 2) == g and round(c, 2) == c


def test_distinct_xy_counts_unique_locations():
    """Drives the _distinct_xy < 2 guard that decides BO vs uniform."""
    assert quality_refine._distinct_xy([(6.0, 2.0, 0.1), (6.0, 2.0, 0.9), (8.0, 5.0, 0.4)]) == 2


def test_numeric_parses_floats_and_rejects_sentinels():
    assert quality_refine._numeric("6.0") == 6.0
    assert quality_refine._numeric(2) == 2.0
    for bad in (None, "", "none", "None", float("nan"), float("inf"), float("-inf")):
        assert quality_refine._numeric(bad) is None


def test_bucket_observations_joins_maxaggregates_and_drops_invalid(tmp_path):
    b = tmp_path / "orig"
    _write_gen_csv(
        b,
        [
            {"output_filename": "A+x_00000.png", "anomaly_type": "A+x", "guidance": 6.0, "crop_ratio": 2.0, "index": 0},
            {"output_filename": "A+x_00001.png", "anomaly_type": "A+x", "guidance": 6.0, "crop_ratio": 2.0, "index": 0},
            {
                "output_filename": "A+x_00002.png",
                "anomaly_type": "A+x",
                "guidance": 7.0,
                "crop_ratio": "none",
                "index": 1,
            },  # noqa: E501
            {"output_filename": "A+x_00003.png", "anomaly_type": "A+x", "guidance": 8.0, "crop_ratio": 5.0, "index": 2},
        ],
    )
    _make_kpi(
        b / "kpi.json",
        {
            "A+x_00000.png": 0.3,
            "A+x_00001.png": 0.6,  # same index 0 -> max 0.6
            "A+x_00002.png": 0.9,  # non-numeric crop_ratio -> dropped
            "A+x_00003.png": float("nan"),  # NaN score -> dropped
        },
    )
    assert quality_refine._bucket_observations(b, b / "kpi.json", "nn_score") == [(0, 6.0, 2.0, 0.6)]


def test_load_observations_pools_original_and_rounds(tmp_path):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _write_gen_csv(orig, _gen_rows(1, 6.0, 2.0))
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.3})
    _write_gen_csv(rounds / "round_1", _gen_rows(1, 8.0, 5.0))
    _make_kpi(rounds / "round_1" / "kpi.json", {"A+x_00000.png": 0.7})
    (rounds / "round_2").mkdir(parents=True)  # no CSV/kpi -> skipped, must not raise
    obs = quality_refine._load_observations(orig, tmp_path / "orig_kpi.json", rounds, "nn_score", "kpi.json")
    assert obs[0] == [(6.0, 2.0, 0.3), (8.0, 5.0, 0.7)]  # keyed by sample index


def test_absolute_score_history_excludes_current_and_future_rounds(tmp_path):
    """All score modes use the same resume boundary, not only cumulative AQ-Rank."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _write_gen_csv(orig, _gen_rows(1, 3.0, 2.0))
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.3})
    for r in (1, 2, 3):
        _write_gen_csv(rounds / f"round_{r}", _gen_rows(1, r + 3.0, r + 2.0))
        _make_kpi(rounds / f"round_{r}" / "kpi.json", {"A+x_00000.png": 0.3 + r / 10})

    observations = quality_refine._load_observations(
        orig, tmp_path / "orig_kpi.json", rounds, "nn_score", "kpi.json", before_round=2
    )

    assert observations[0] == [(3.0, 2.0, 0.3), (4.0, 3.0, 0.4)]


def test_aq_rank_recomputes_observed_pool_before_each_draw(tmp_path, monkeypatch):
    """Adding a round changes earlier candidates' targets because the rank population grew."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _write_gen_csv(orig, _gen_rows(1, 3.0, 2.0))
    _write_aq_rank_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": (0.5, 0.5, 0.5, 0.5)})
    _write_gen_csv(rounds / "round_1", _gen_rows(1, 6.0, 4.0))
    _write_aq_rank_kpi(rounds / "round_1" / "kpi.json", {"A+x_00000.png": (0.6, 0.5, 0.5, 0.5)})
    messages = []
    monkeypatch.setattr(quality_refine.log, "info", messages.append)

    first = quality_refine._aq_rank_observations(orig, tmp_path / "orig_kpi.json", rounds, "kpi.json")
    assert len(first[0]) == 2 and first[0][1][2] == 1.0

    _write_gen_csv(rounds / "round_2", _gen_rows(1, 8.0, 6.0))
    _write_aq_rank_kpi(rounds / "round_2" / "kpi.json", {"A+x_00000.png": (0.7, 0.5, 0.5, 0.5)})
    second = quality_refine._aq_rank_observations(orig, tmp_path / "orig_kpi.json", rounds, "kpi.json")

    assert len(second[0]) == 3
    assert second[0][1][2] < first[0][1][2]  # Round 1 was re-ranked, not reused as a local 1.0.
    assert second[0][2][2] == 1.0
    assert sum("AQ-Rank compass for anomaly type" in message for message in messages) == 2
    assert sum("directions are fitted across pooled samples" in message for message in messages) == 2


def test_aq_rank_search_uses_the_four_axis_composite(tmp_path):
    """With enough candidates, geometry can beat a higher-NN render in the BO history."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _write_gen_csv(orig, _gen_rows(3, 3.0, 2.0))
    _write_aq_rank_kpi(
        tmp_path / "orig_kpi.json",
        {
            "A+x_00000.png": (0.9, 0.6, 0.6, 0.6),
            "A+x_00001.png": (0.7, 0.5, 0.5, 0.5),
            "A+x_00002.png": (0.5, 0.1, 0.1, 0.1),
        },
    )
    _write_gen_csv(rounds / "round_1", _gen_rows(3, 6.0, 4.0))
    _write_aq_rank_kpi(
        rounds / "round_1" / "kpi.json",
        {
            "A+x_00000.png": (0.8, 1.0, 1.0, 1.0),
            "A+x_00001.png": (0.6, 0.2, 0.2, 0.2),
            "A+x_00002.png": (0.4, 0.05, 0.05, 0.05),
        },
    )

    observations = quality_refine._aq_rank_observations(orig, tmp_path / "orig_kpi.json", rounds, "kpi.json")

    assert observations[0][1][2] > observations[0][0][2]


def test_aq_rank_history_excludes_the_current_and_future_rounds(tmp_path):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _write_gen_csv(orig, _gen_rows(1, 3.0, 2.0))
    _write_aq_rank_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": (0.5, 0.5, 0.5, 0.5)})
    for r in (1, 2, 3):
        _write_gen_csv(rounds / f"round_{r}", _gen_rows(1, r + 3.0, r + 2.0))
        _write_aq_rank_kpi(rounds / f"round_{r}" / "kpi.json", {"A+x_00000.png": (0.5 + r / 10, 0.5, 0.5, 0.5)})

    observations = quality_refine._aq_rank_observations(
        orig, tmp_path / "orig_kpi.json", rounds, "kpi.json", before_round=2
    )

    assert len(observations[0]) == 2  # Original + round_1 only.


def test_aq_rank_never_mixes_anomaly_types_in_one_rank_pool(tmp_path, monkeypatch):
    bucket = tmp_path / "out"
    rows = [
        {"output_filename": "A+x_00000.png", "index": 0, "guidance": 3.0, "crop_ratio": 2.0},
        {"output_filename": "A+x_00001.png", "index": 1, "guidance": 3.0, "crop_ratio": 2.0},
        {"output_filename": "B+y_00000.png", "index": 2, "guidance": 3.0, "crop_ratio": 2.0},
        {"output_filename": "B+y_00001.png", "index": 3, "guidance": 3.0, "crop_ratio": 2.0},
    ]
    _write_gen_csv(bucket, rows)
    kpi = tmp_path / "kpi.json"
    _write_aq_rank_kpi(
        kpi,
        {
            "A+x_00000.png": (0.1, 0.5, 0.5, 0.5),
            "A+x_00001.png": (0.2, 0.5, 0.5, 0.5),
            "B+y_00000.png": (0.8, 0.5, 0.5, 0.5),
            "B+y_00001.png": (0.9, 0.5, 0.5, 0.5),
        },
    )
    seen_nn_groups = []

    def fake_rank(group):
        seen_nn_groups.append([row["nn_score"] for row in group])
        compass = {
            "pool_size": len(group),
            "axes": {axis: {"auroc": None, "sign": 0} for axis in ("completeness", "precision", "boundary_iou")},
        }
        return [0.0] * len(group), compass

    monkeypatch.setattr(quality_refine, "compute_aq_rank_with_compass", fake_rank)

    quality_refine._rank_aq_pool([(bucket, kpi, "original")])

    assert seen_nn_groups == [[0.1, 0.2], [0.8, 0.9]]


def test_aq_rank_logs_when_a_type_pool_is_too_small_for_geometry(tmp_path, monkeypatch):
    bucket = tmp_path / "out"
    _write_gen_csv(bucket, _gen_rows(2, 3.0, 2.0))
    kpi = tmp_path / "kpi.json"
    _write_aq_rank_kpi(
        kpi,
        {
            "A+x_00000.png": (0.5, 0.5, 0.5, 0.5),
            "A+x_00001.png": (0.6, 0.6, 0.6, 0.6),
        },
    )
    warnings = []
    monkeypatch.setattr(quality_refine.log, "warning", warnings.append)

    quality_refine._rank_aq_pool([(bucket, kpi, "original")])

    assert any("'A+x' has 2 candidate(s), below 5" in message and "NN only" in message for message in warnings)


def test_aq_rank_logs_each_cross_sample_compass_direction(tmp_path, monkeypatch):
    bucket = tmp_path / "out"
    _write_gen_csv(bucket, _gen_rows(6, 3.0, 2.0))
    kpi = tmp_path / "kpi.json"
    _write_aq_rank_kpi(
        kpi,
        {f"A+x_{i:05d}.png": (i / 5, 1 - i / 5, i / 5, 0.5) for i in range(6)},
    )
    messages = []
    monkeypatch.setattr(quality_refine.log, "info", messages.append)

    quality_refine._rank_aq_pool([(bucket, kpi, "original")])

    assert any("across pooled samples" in text and "within one sample" in text for text in messages)
    message = next(text for text in messages if "AQ-Rank compass for anomaly type" in text)
    assert "completeness: AUROC=0.000, sign=-1" in message
    assert "precision: AUROC=1.000, sign=+1" in message
    assert "boundary_iou: AUROC=0.500, sign=disabled" in message


def test_aq_rank_history_rejects_a_missing_generation_index(tmp_path):
    orig = tmp_path / "out"
    rows = _gen_rows(2, 3.0, 2.0)
    rows[1].pop("index")
    _write_gen_csv(orig, rows)
    kpi = tmp_path / "kpi.json"
    _write_aq_rank_kpi(
        kpi,
        {
            "A+x_00000.png": (0.5, 0.5, 0.5, 0.5),
            "A+x_00001.png": (0.6, 0.6, 0.6, 0.6),
        },
    )

    with pytest.raises(ValueError, match="A\\+x_00001.png has no valid integer index"):
        quality_refine._aq_rank_observations(orig, kpi, tmp_path / "no-rounds", "kpi.json")


def test_absolute_score_history_rejects_a_garbled_generation_index(tmp_path):
    bucket = tmp_path / "out"
    rows = _gen_rows(1, 3.0, 2.0)
    rows[0]["index"] = "not-an-index"
    _write_gen_csv(bucket, rows)
    kpi = tmp_path / "kpi.json"
    _make_kpi(kpi, {"A+x_00000.png": 0.5})

    with pytest.raises(ValueError, match="A\\+x_00000.png has no valid integer index"):
        quality_refine._bucket_observations(bucket, kpi, "nn_score")


def test_bo_propose_steers_toward_the_optimum():
    import numpy as np

    gr, cr = (1.5, 10.0), (1.5, 8.0)
    g_star, c_star = 7.0, 4.0
    rng = np.random.default_rng(0)
    obs = []
    for _ in range(30):
        g, c = float(rng.uniform(*gr)), float(rng.uniform(*cr))
        obs.append((g, c, -((g - g_star) ** 2 + (c - c_star) ** 2)))  # smooth concave peak at (7,4)
    picks = quality_refine._bo_propose(obs, 15, 1, gr, cr, 512)
    assert len(picks) == 15
    gs, cs = sorted(p[0] for p in picks), sorted(p[1] for p in picks)
    assert abs(gs[len(gs) // 2] - g_star) <= 2.0  # median proposal near the optimum
    assert abs(cs[len(cs) // 2] - c_star) <= 1.5
    for g, c in picks:
        assert gr[0] <= g <= gr[1] and cr[0] <= c <= cr[1]

    # BO must beat uniform random: its proposals sit closer to the optimum than uniform draws over the
    # same box. The median-band asserts above alone would also admit a uniform sampler.
    ur = np.random.default_rng(123)
    uniform_pts = [(float(ur.uniform(*gr)), float(ur.uniform(*cr))) for _ in picks]

    def _median_dist(pts):
        d = sorted(((g - g_star) ** 2 + (c - c_star) ** 2) ** 0.5 for g, c in pts)
        return d[len(d) // 2]

    assert _median_dist(picks) < _median_dist(uniform_pts)


def _draw_bo(rows, observations, **over):
    kw = dict(seed=1, bo_min_obs=2, n_candidates=64, **_RANGES)
    kw.update(over)
    return quality_refine._draw_rows_bo(rows, observations, **kw)


def test_draw_rows_bo_routes_per_sample_by_data_richness():
    """``n_bo`` is the routing decision made observable — the caller logs the split from it — so the
    real GP can run and the count still says which branch each sample took."""
    rows = [
        {"index": i, "anomaly_type": "A+x", "guidance": 6.0, "crop_ratio": 2.0, "mask_filename": "m.png"}
        for i in range(2)
    ]
    drawn, n_bo = _draw_bo(rows, {0: _RICH, 1: _POOR})  # keyed by sample index
    assert n_bo == 1  # only sample 0 had enough history
    assert [d["index"] for d in drawn] == [0, 1]  # order preserved
    assert all(d["mask_filename"] == "m.png" for d in drawn)  # other fields preserved
    for d in drawn:
        assert 1.5 <= d["guidance"] <= 10.0 and 1.5 <= d["crop_ratio"] <= 8.0


def test_draw_rows_bo_fits_its_gaussian_processes_single_threaded(monkeypatch):
    """Every GP fit must run INSIDE ``threadpool_limits(limits=1)``, entered once for the batch and
    exited before the draw returns. The double tracks enter/exit rather than construction: the real
    class applies the limit in ``__init__`` and lifts it only in ``__exit__``, so a bare statement or
    a context that closes before the loop would both pass a construction-only check while either
    leaving the process capped for the in-process ``evaluate`` that follows, or not capping the fits."""
    state = {"inside": False, "enters": 0, "exits": 0, "limits": []}

    class _Recorder:
        def __init__(self, limits=None, user_api=None):  # threadpoolctl's real signature
            state["limits"].append(limits)

        def __enter__(self):
            state["inside"] = True
            state["enters"] += 1
            return self

        def __exit__(self, *exc):
            state["inside"] = False
            state["exits"] += 1
            return False

    real_propose = quality_refine._bo_propose
    fits_inside = []

    def _spy(*a, **kw):
        fits_inside.append(state["inside"])
        return real_propose(*a, **kw)

    monkeypatch.setattr(quality_refine, "threadpool_limits", _Recorder)
    monkeypatch.setattr(quality_refine, "_bo_propose", _spy)
    rows = [{"index": i, "guidance": 6.0, "crop_ratio": 2.0} for i in range(2)]

    drawn, n_bo = _draw_bo(rows, {0: _RICH, 1: _RICH})

    assert n_bo == 2 and len(drawn) == 2
    assert fits_inside == [True, True], "every GP fit must happen inside the thread cap"
    assert state["limits"] == [1] and state["enters"] == 1, "one cap of 1 thread, entered once per batch"
    assert state["exits"] == 1 and not state["inside"], "the cap must be lifted before the draw returns"


def test_draw_rows_bo_keys_history_by_preset_index_not_row_position():
    """InpaintInferenceDataset honors an ``index`` already in the JSONL (setdefault), so a testcase
    whose rows carry non-positional indices must still get ITS OWN history — not the neighbour's.

    The decoys at 0/1 are deliberately too thin to route: keying by position would find them and
    report n_bo=0, keying by index finds the rich histories at 7/3 and reports 2.
    """
    rows = [{"index": 7, "guidance": 6.0, "crop_ratio": 2.0}, {"index": 3, "guidance": 6.0, "crop_ratio": 2.0}]
    drawn, n_bo = _draw_bo(rows, {7: _RICH, 3: _RICH, 0: _POOR, 1: _POOR})
    assert n_bo == 2, "history was looked up by row position, not by the row's own index"
    assert [d["index"] for d in drawn] == [7, 3]


def test_draw_rows_bo_falls_back_to_position_without_an_index_field():
    """The companion to the test above: with no ``index`` in the JSONL there is nothing to key on,
    so history is looked up by row position instead."""
    rows = [{"guidance": 6.0, "crop_ratio": 2.0} for _ in range(2)]  # no "index"
    _, n_bo = _draw_bo(rows, {1: _RICH})  # only position 1 has history
    assert n_bo == 1


def test_draw_rows_bo_is_reproducible():
    """Both branches must replay identically for a seed, so a round can be reconstructed from disk."""
    rows = [{"index": i, "anomaly_type": "B+y", "guidance": 6.0, "crop_ratio": 2.0} for i in range(3)]
    assert _draw_bo(rows, {}, seed=5) == _draw_bo(rows, {}, seed=5)  # no history -> all uniform
    rich = {i: _RICH for i in range(3)}
    assert _draw_bo(rows, rich, seed=5) == _draw_bo(rows, rich, seed=5)  # all BO


def test_draw_rejects_a_degenerate_range():
    """LOW == HIGH would divide by zero when scaling into the GP's unit box -> nan proposals."""
    argv = ["draw", "--base_testcase", "b.jsonl", "--output", "o.jsonl"]
    for flag in ("--guidance_range", "--crop_ratio_range"):
        with pytest.raises(SystemExit):
            quality_refine._get_args([*argv, flag, "5", "5"])
    assert quality_refine._get_args([*argv, "--guidance_range", "2", "9"]).guidance_range == [2.0, 9.0]


def test_draw_rejects_a_crop_ratio_low_under_the_recipe_floor():
    """A recipe's band floors at MIN_CROP_RATIO, so every draw below it would be clamped away."""
    argv = ["draw", "--base_testcase", "b.jsonl", "--output", "o.jsonl", "--crop_ratio_range"]
    with pytest.raises(SystemExit):
        quality_refine._get_args([*argv, "1.0", "2.5"])
    assert quality_refine._get_args([*argv, "1.5", "2.5"]).crop_ratio_range == [1.5, 2.5]


def test_draw_rejects_a_non_positive_current_round():
    argv = ["draw", "--base_testcase", "b.jsonl", "--output", "o.jsonl", "--current_round", "0"]
    with pytest.raises(SystemExit):
        quality_refine._get_args(argv)


def test_select_rejects_a_negative_max_round():
    argv = [
        "select",
        "--original",
        "original",
        "--original_kpi",
        "kpi.json",
        "--output",
        "searched",
        "--max_round",
        "-1",
    ]
    with pytest.raises(SystemExit):
        quality_refine._get_args(argv)


def test_draw_allows_round_zero_output_when_there_is_no_history(tmp_path):
    base = tmp_path / "testcase.jsonl"
    _write_base_testcase(base, [{"index": 0, "guidance": 3.0, "crop_ratio": 2.0}])
    orig = tmp_path / "out"
    _write_gen_csv(orig, _gen_rows(1, 3.0, 2.0))
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.5})

    rounds = tmp_path / "rounds"
    output = rounds / "round_0" / "testcase.jsonl"
    _main(
        "draw",
        base_testcase=base,
        output=output,
        original=orig,
        original_kpi=tmp_path / "orig_kpi.json",
        rounds_dir=rounds,
    )

    assert output.is_file()


def _bo_history_buckets(tmp_path, n, round1_guidance, round1_crop):
    """original + round_1 covering n samples, so each sample has two observations. Whether those two
    land on DISTINCT (g,c) locations is what decides BO vs uniform, so the caller picks round 1's."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _write_gen_csv(orig, _gen_rows(n, 3.0, 2.0))
    _ramped_kpi(tmp_path / "orig_kpi.json", n, 0.2)
    _write_gen_csv(rounds / "round_1", _gen_rows(n, round1_guidance, round1_crop))
    _ramped_kpi(rounds / "round_1" / "kpi.json", n, 0.5)
    base = tmp_path / "testcase.jsonl"
    _write_base_testcase(
        base,
        [
            {"index": i, "anomaly_type": "A+x", "guidance": 3.0, "crop_ratio": 2.0, "mask_filename": f"m{i}.png"}
            for i in range(n)
        ],
    )
    return base, orig, rounds


def _run_draw_cli(tmp_path, base, orig, rounds, out):
    _main(
        "draw",
        base_testcase=base,
        output=out,
        seed=2,
        original=orig,
        original_kpi=tmp_path / "orig_kpi.json",
        rounds_dir=rounds,
        current_round=2,
        bo_min_obs=2,
        bo_candidates=128,
    )


def test_draw_cli_uses_bo_when_prior_buckets_given(tmp_path):
    base, orig, rounds = _bo_history_buckets(tmp_path, 3, round1_guidance=8.0, round1_crop=6.0)
    out = tmp_path / "rounds" / "round_2" / "testcase.jsonl"
    _run_draw_cli(tmp_path, base, orig, rounds, out)

    written = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert [r["index"] for r in written] == [0, 1, 2]  # every sample, order kept
    assert all(r["mask_filename"] == f"m{r['index']}.png" for r in written)  # other fields intact
    for r in written:
        assert 1.5 <= r["guidance"] <= 10.0 and 1.5 <= r["crop_ratio"] <= 8.0

    out2 = tmp_path / "rounds" / "round_2b" / "testcase.jsonl"
    _run_draw_cli(tmp_path, base, orig, rounds, out2)
    assert out2.read_text() == out.read_text()  # reproducible


@pytest.mark.parametrize(
    ("round1", "expected"),
    [
        # Distinct (g,c) from the original -> the real GP runs for every sample.
        ((8.0, 6.0), "mode=bayesopt 3/3 samples"),
        # round_1 repeats the original's location: 2 observations, 1 distinct point. A GP has nothing
        # to fit, so the _distinct_xy < 2 guard forces uniform even though bo_min_obs is satisfied.
        ((3.0, 2.0), "mode=uniform (no sample had enough history)"),
    ],
    ids=["engages-bo", "falls-back-to-uniform"],
)
def test_draw_cli_reports_the_real_bo_uniform_split(tmp_path, loguru_lines, round1, expected):
    """Guards the highest-value path with the REAL GP (no mock). Index preservation, in-range bounds
    and reproducibility are all satisfied by a 100% uniform fallback too, so assert the reported
    split: if routing silently stopped reaching _bo_propose, the first case fails."""
    logged = loguru_lines
    base, orig, rounds = _bo_history_buckets(tmp_path, 3, *round1)
    _run_draw_cli(tmp_path, base, orig, rounds, tmp_path / "r2" / "testcase.jsonl")

    summary = [ln for ln in logged if "mode=" in ln]
    assert summary, logged
    assert expected in summary[0], summary[0]


def test_draw_cli_without_bo_args_is_uniform(tmp_path):
    """No --original/--rounds_dir means no history to fit, so draw must still emit a full round."""
    base = tmp_path / "testcase.jsonl"
    _write_base_testcase(base, [{"index": 0, "anomaly_type": "A+x", "guidance": 6.0, "crop_ratio": 2.0}])
    out = tmp_path / "round_1" / "testcase.jsonl"
    _main("draw", base_testcase=base, output=out, seed=1)
    written = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert [r["index"] for r in written] == [0]
    assert 1.5 <= written[0]["guidance"] <= 10.0 and 1.5 <= written[0]["crop_ratio"] <= 8.0


@pytest.mark.parametrize("output_rel", ["manual.jsonl", "round_0/testcase.jsonl"])
def test_draw_with_history_requires_an_unambiguous_current_round(tmp_path, output_rel):
    base, orig, rounds = _bo_history_buckets(tmp_path, 1, round1_guidance=8.0, round1_crop=6.0)

    with pytest.raises(SystemExit) as exc:
        _main(
            "draw",
            base_testcase=base,
            output=tmp_path / output_rel,
            original=orig,
            original_kpi=tmp_path / "orig_kpi.json",
            rounds_dir=rounds,
        )

    assert exc.value.code == 1


def test_draw_infers_current_round_from_the_canonical_output_path(tmp_path):
    base, orig, rounds = _bo_history_buckets(tmp_path, 1, round1_guidance=8.0, round1_crop=6.0)
    out = rounds / "round_2" / "testcase.jsonl"

    _main(
        "draw",
        base_testcase=base,
        output=out,
        original=orig,
        original_kpi=tmp_path / "orig_kpi.json",
        rounds_dir=rounds,
        bo_min_obs=3,
    )

    assert out.is_file()


# --- select --------------------------------------------------------------------------------------


def _select(orig, orig_kpi, rounds_dir, out, **over):
    _main("select", original=orig, original_kpi=orig_kpi, rounds_dir=rounds_dir, output=out, **over)


def test_pick_best_prefers_highest_and_keeps_earliest_on_tie():
    orig = {"score": 0.5, "source": "original"}
    r1 = {"score": 0.5, "source": "round_1"}
    r2 = {"score": 0.7, "source": "round_2"}
    assert quality_refine._pick_best([orig, r1, r2])["source"] == "round_2"
    assert quality_refine._pick_best([orig, r1])["source"] == "original"  # tie -> earliest
    assert quality_refine._pick_best([{"score": float("nan"), "source": "original"}, r1])["source"] == "round_1"
    assert quality_refine._pick_best([{"score": float("inf"), "source": "original"}, r1])["source"] == "round_1"


def test_select_picks_best_render_per_sample_across_rounds(tmp_path):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    samples = [("A+x_00000.png", 0), ("A+x_00001.png", 1)]
    _make_bucket(orig, samples, tag="ORIG")
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.5, "A+x_00001.png": 0.3})
    _make_bucket(rounds / "round_1", samples, tag="R1")
    _make_kpi(rounds / "round_1" / "kpi.json", {"A+x_00000.png": 0.4, "A+x_00001.png": 0.7})

    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", rounds, searched)

    summary = _read_summary(searched)
    assert summary["A+x_00000.png"]["source"] == "original"  # 0.5 > 0.4
    assert summary["A+x_00001.png"]["source"] == "round_1"  # 0.7 > 0.3
    assert (searched / "reconstructed_image" / "A+x_00000.png").read_bytes() == b"ORIG"
    assert (searched / "reconstructed_image" / "A+x_00001.png").read_bytes() == b"R1"
    rows = {r["output_filename"]: r for r in _read_gen_csv(searched)}
    assert rows["A+x_00001.png"]["source"] == "round_1" and rows["A+x_00001.png"]["nn_score"] == "0.700000"
    assert rows["A+x_00001.png"]["selected_by"] == "nn_score"


def test_select_aq_rank_uses_one_pool_for_original_and_rounds(tmp_path, monkeypatch):
    """Final selection uses the four-axis composite, not the bucket-local value or plain NN."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    samples = [("A+x_00000.png", 0), ("A+x_00001.png", 1), ("A+x_00002.png", 2)]
    _make_bucket(orig, samples, tag="ORIG")
    _write_aq_rank_kpi(
        tmp_path / "orig_kpi.json",
        {
            # The original has the higher NN for sample 0 but weaker geometry than round 1.
            "A+x_00000.png": (0.9, 0.6, 0.6, 0.6),
            "A+x_00001.png": (0.7, 0.5, 0.5, 0.5),
            "A+x_00002.png": (0.5, 0.1, 0.1, 0.1),
        },
    )
    _make_bucket(rounds / "round_1", samples, tag="R1")
    _write_aq_rank_kpi(
        rounds / "round_1" / "kpi.json",
        {
            # Across all six candidates, the axes orient positively from NN's extremes. Their
            # combined ranks make this render beat the original despite its lower NN (0.8 < 0.9).
            "A+x_00000.png": (0.8, 1.0, 1.0, 1.0),
            "A+x_00001.png": (0.6, 0.2, 0.2, 0.2),
            "A+x_00002.png": (0.4, 0.05, 0.05, 0.05),
        },
    )

    messages = []
    monkeypatch.setattr(quality_refine.log, "info", messages.append)
    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", rounds, searched, score="aq_rank")

    summary = _read_summary(searched)
    assert summary["A+x_00000.png"]["source"] == "round_1"
    assert summary["A+x_00001.png"]["source"] == "original"
    assert summary["A+x_00000.png"]["selected_by"] == "aq_rank_score"
    assert float(summary["A+x_00000.png"]["score"]) > float(summary["A+x_00000.png"]["original_score"])
    assert any("AQ-Rank compass" in message for message in messages)


def test_aq_rank_excludes_one_invalid_candidate_without_rejecting_the_bucket(tmp_path, monkeypatch):
    bucket = tmp_path / "out"
    _write_gen_csv(bucket, _gen_rows(2, 3.0, 2.0))
    kpi = tmp_path / "kpi.json"
    _write_aq_rank_kpi(
        kpi,
        {
            "A+x_00000.png": (0.5, float("inf"), 0.5, 0.5),
            "A+x_00001.png": (0.6, 0.6, 0.6, 0.6),
        },
    )
    warnings = []
    monkeypatch.setattr(quality_refine.log, "warning", warnings.append)

    ranked = quality_refine._rank_aq_pool([(bucket, kpi, "original")])[0]

    assert ranked["A+x_00000.png"]["rankable"] is False
    assert ranked["A+x_00000.png"]["score"] != ranked["A+x_00000.png"]["score"]
    assert ranked["A+x_00001.png"]["score"] == 0.0
    assert any("excluding 1/2" in message for message in warnings)


def test_aq_rank_rejects_a_bucket_with_no_valid_candidate(tmp_path):
    bucket = tmp_path / "out"
    _write_gen_csv(bucket, _gen_rows(1, 3.0, 2.0))
    kpi = tmp_path / "kpi.json"
    _write_aq_rank_kpi(kpi, {"A+x_00000.png": (0.5, float("inf"), 0.5, 0.5)})

    with pytest.raises(ValueError, match="no generated candidate has all finite raw inputs"):
        quality_refine._load_aq_rank_bucket(bucket, kpi, "original")


def test_aq_rank_loader_reads_kpi_once_and_preserves_mnn(tmp_path, monkeypatch):
    bucket = tmp_path / "out"
    _write_gen_csv(bucket, _gen_rows(1, 3.0, 2.0))
    kpi = tmp_path / "kpi.json"
    _write_kpi_keys(
        kpi,
        {
            "A+x_00000.png": {
                "nn_score": 0.8,
                "mnn_score": 0.1,
                "completeness_score": 0.7,
                "precision_score": 0.6,
                "boundary_iou_score": 0.5,
            }
        },
    )
    real_read_text = pathlib.Path.read_text
    reads = 0

    def count_kpi_reads(path, *args, **kwargs):
        nonlocal reads
        if path == kpi:
            reads += 1
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", count_kpi_reads)

    candidate = quality_refine._load_aq_rank_bucket(bucket, kpi, "original")["A+x_00000.png"]

    assert reads == 1
    assert candidate["nn"] == 0.8 and candidate["mnn"] == 0.1


def test_aq_rank_loader_derives_raw_inputs_from_shared_geometry_axes(tmp_path, monkeypatch):
    bucket = tmp_path / "out"
    _write_gen_csv(bucket, _gen_rows(1, 3.0, 2.0))
    kpi = tmp_path / "kpi.json"
    _write_kpi_keys(
        kpi,
        {
            "A+x_00000.png": {
                "nn_score": 0.8,
                "completeness_score": 0.7,
                "precision_score": 0.6,
                "boundary_iou_score": 0.5,
                "shape_score": 0.4,
            }
        },
    )
    monkeypatch.setattr(quality_refine, "GEOMETRY_AXES", (*quality_refine.GEOMETRY_AXES, "shape"))

    candidate = quality_refine._load_aq_rank_bucket(bucket, kpi, "original")["A+x_00000.png"]

    assert candidate["metrics"]["shape_score"] == 0.4


def test_absolute_score_loader_reads_kpi_once_for_all_reported_metrics(tmp_path, monkeypatch):
    bucket = tmp_path / "out"
    _write_gen_csv(bucket, _gen_rows(1, 3.0, 2.0))
    kpi = tmp_path / "kpi.json"
    _write_kpi_keys(
        kpi,
        {"A+x_00000.png": {"nn_score": 0.8, "mnn_score": 0.1, "completeness_score": 0.7}},
    )
    real_read_text = pathlib.Path.read_text
    reads = 0

    def count_kpi_reads(path, *args, **kwargs):
        nonlocal reads
        if path == kpi:
            reads += 1
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", count_kpi_reads)

    candidate = quality_refine._load_bucket(bucket, kpi, "completeness_score", "original")["A+x_00000.png"]

    assert reads == 1
    assert candidate["score"] == 0.7 and candidate["nn"] == 0.8 and candidate["mnn"] == 0.1


def test_select_by_mnn_never_labels_an_mnn_value_as_nn_score(tmp_path):
    """--score mnn must not write mnn numbers into a column literally named nn_score. Both metrics
    are already loaded per bucket, so each goes out under its own name plus a selected_by marker."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _make_bucket(orig, [("A+x_00000.png", 0)], tag="ORIG")
    _make_bucket(rounds / "round_1", [("A+x_00000.png", 0)], tag="R1")
    # nn prefers the original (0.9 > 0.1); mnn prefers round_1 (0.8 > 0.2). Selecting on mnn must
    # pick round_1 AND report 0.1/0.8 under the right column names.
    _write_kpi_nn_mnn(tmp_path / "orig_kpi.json", {"A+x_00000.png": (0.9, 0.2)})
    _write_kpi_nn_mnn(rounds / "round_1" / "kpi.json", {"A+x_00000.png": (0.1, 0.8)})

    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", rounds, searched, score="mnn")

    row = _read_gen_csv(searched)[0]
    assert row["source"] == "round_1"  # selection followed mnn, not nn
    assert row["selected_by"] == "mnn_score"
    assert row["nn_score"] == "0.100000"  # round_1's REAL nn — not its mnn
    assert row["mnn_score"] == "0.800000"

    summary = _read_summary(searched)["A+x_00000.png"]
    assert summary["selected_by"] == "mnn_score"
    assert summary["score"] == "0.800000" and summary["original_score"] == "0.200000"
    assert "nn_score" not in summary  # no metric-agnostic column masquerading as nn


def test_select_orders_rounds_numerically_not_lexicographically(tmp_path):
    """round_10 must not sort before round_2 — that silently reorders _pick_best's tie-break."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    _make_bucket(orig, [("A+x_00000.png", 0)], tag="ORIG")
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.1})
    for r in (2, 10):
        _make_bucket(rounds / f"round_{r}", [("A+x_00000.png", 0)], tag=f"R{r}")
        _make_kpi(rounds / f"round_{r}" / "kpi.json", {"A+x_00000.png": 0.5})  # TIE between the two

    assert [p.name for p in quality_refine._round_dirs(rounds)] == ["round_2", "round_10"]

    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", rounds, searched)
    # Tie -> earliest source wins, and "earliest" means round 2, not the lexicographically-first 10.
    assert _read_summary(searched)["A+x_00000.png"]["source"] == "round_2"
    assert (searched / "reconstructed_image" / "A+x_00000.png").read_bytes() == b"R2"


def test_round_discovery_ignores_noncanonical_numeric_aliases(tmp_path):
    rounds = tmp_path / "rounds"
    for name in ("round_1", "round_01", "round_0", "round_²", "round_backup"):
        (rounds / name).mkdir(parents=True)

    assert [path.name for path in quality_refine._round_dirs(rounds)] == ["round_1"]


def test_run_selection_ignores_stale_rounds_after_the_requested_boundary(tmp_path):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    sample = [("A+x_00000.png", 0)]
    _make_bucket(orig, sample, tag="ORIG")
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.1})
    for number, score in ((1, 0.2), (2, 0.3), (3, 0.99)):
        _make_bucket(rounds / f"round_{number}", sample, tag=f"R{number}")
        _make_kpi(rounds / f"round_{number}" / "kpi.json", {"A+x_00000.png": score})
    _make_bucket(rounds / "round_backup", sample, tag="STALE")
    _make_kpi(rounds / "round_backup" / "kpi.json", {"A+x_00000.png": 1.0})

    args = quality_refine._get_args(
        [
            "select",
            "--original",
            str(orig),
            "--original_kpi",
            str(tmp_path / "orig_kpi.json"),
            "--rounds_dir",
            str(rounds),
            "--output",
            str(tmp_path / "searched"),
            "--max_round",
            "2",
        ]
    )
    quality_refine._run_select(args)

    assert _read_summary(tmp_path / "searched")["A+x_00000.png"]["source"] == "round_2"


def test_aq_rank_pool_excludes_stale_rounds_after_the_requested_boundary(tmp_path, monkeypatch):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    sample = [("A+x_00000.png", 0)]
    _make_bucket(orig, sample, tag="ORIG")
    _write_aq_rank_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": (0.5, 0.5, 0.5, 0.5)})
    for number in (1, 2, 3):
        _make_bucket(rounds / f"round_{number}", sample, tag=f"R{number}")
        _write_aq_rank_kpi(
            rounds / f"round_{number}" / "kpi.json",
            {"A+x_00000.png": (0.5 + number / 10, 0.5, 0.5, 0.5)},
        )

    loaded_sources = []
    real_rank = quality_refine._rank_aq_pool

    def rank_and_record(specs):
        loaded_sources.extend(source for _, _, source in specs)
        return real_rank(specs)

    monkeypatch.setattr(quality_refine, "_rank_aq_pool", rank_and_record)
    args = quality_refine._get_args(
        [
            "select",
            "--original",
            str(orig),
            "--original_kpi",
            str(tmp_path / "orig_kpi.json"),
            "--rounds_dir",
            str(rounds),
            "--output",
            str(tmp_path / "searched"),
            "--score",
            "aq_rank",
            "--max_round",
            "2",
        ]
    )

    quality_refine._run_select(args)

    assert loaded_sources == ["original", "round_1", "round_2"]


def test_aq_rank_selection_explicitly_keeps_original_when_one_sample_has_no_eligible_render(tmp_path, monkeypatch):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    samples = [("A+x_00000.png", 0), ("A+x_00001.png", 1)]
    _make_bucket(orig, samples, tag="ORIG")
    _make_bucket(rounds / "round_1", samples, tag="R1")
    for kpi in (tmp_path / "orig_kpi.json", rounds / "round_1" / "kpi.json"):
        _write_aq_rank_kpi(
            kpi,
            {
                "A+x_00000.png": (0.5, 0.5, 0.5, 0.5),
                "A+x_00001.png": (0.5, float("nan"), 0.5, 0.5),
            },
        )
    warnings = []
    monkeypatch.setattr(quality_refine.log, "warning", warnings.append)

    _select(orig, tmp_path / "orig_kpi.json", rounds, tmp_path / "searched", score="aq_rank")

    summary = _read_summary(tmp_path / "searched")["A+x_00001.png"]
    assert summary["source"] == "original" and summary["score"] == ""
    assert any("no eligible AQ-Rank candidate for A+x_00001.png" in message for message in warnings)


@pytest.mark.parametrize(
    ("invalid_source", "expected_source"),
    [("original", "round_1"), ("round_1", "original")],
)
def test_aq_rank_selection_never_picks_an_ineligible_candidate(tmp_path, invalid_source, expected_source):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    samples = [("A+x_00000.png", 0), ("A+x_00001.png", 1)]
    _make_bucket(orig, samples, tag="ORIG")
    _make_bucket(rounds / "round_1", samples, tag="R1")
    valid = (0.6, 0.6, 0.6, 0.6)
    invalid = (0.5, float("nan"), 0.5, 0.5)
    original_sample = invalid if invalid_source == "original" else valid
    round_sample = invalid if invalid_source == "round_1" else valid
    _write_aq_rank_kpi(
        tmp_path / "orig_kpi.json",
        {"A+x_00000.png": (0.5, 0.5, 0.5, 0.5), "A+x_00001.png": original_sample},
    )
    _write_aq_rank_kpi(
        rounds / "round_1" / "kpi.json",
        {"A+x_00000.png": (0.5, 0.5, 0.5, 0.5), "A+x_00001.png": round_sample},
    )

    _select(orig, tmp_path / "orig_kpi.json", rounds, tmp_path / "searched", score="aq_rank")

    assert _read_summary(tmp_path / "searched")["A+x_00001.png"]["source"] == expected_source


_MALFORMED_GENERATION_CSVS = [
    "wrong_header\nA+x_00000.png\n",
    "index,output_filename\n0\n",
]


def _malformed_generation_bucket(tmp_path, csv_text):
    orig = tmp_path / "out"
    (orig / "reconstructed_image").mkdir(parents=True)
    (orig / "reconstructed_image" / "A+x_00000.png").write_bytes(b"ORIG")
    (orig / quality_refine._GEN_CSV).write_text(csv_text)
    kpi = orig / "kpi.json"
    _make_kpi(kpi, {"A+x_00000.png": 0.5})
    return orig, kpi


@pytest.mark.parametrize("csv_text", _MALFORMED_GENERATION_CSVS)
def test_select_reports_a_malformed_generation_csv_as_an_input_error(tmp_path, monkeypatch, csv_text):
    orig, kpi = _malformed_generation_bucket(tmp_path, csv_text)
    errors = []
    monkeypatch.setattr(quality_refine.log, "error", errors.append)

    with pytest.raises(SystemExit) as exc:
        _select(orig, kpi, tmp_path / "nope", tmp_path / "searched")

    assert exc.value.code == 1
    assert errors and "output_filename" in errors[0]


@pytest.mark.parametrize("csv_text", _MALFORMED_GENERATION_CSVS)
def test_draw_reports_a_malformed_generation_csv_as_an_input_error(tmp_path, monkeypatch, csv_text):
    orig, kpi = _malformed_generation_bucket(tmp_path, csv_text)
    base = tmp_path / "testcase.jsonl"
    _write_base_testcase(base, [{"index": 0, "guidance": 3.0, "crop_ratio": 2.0}])
    errors = []
    monkeypatch.setattr(quality_refine.log, "error", errors.append)

    with pytest.raises(SystemExit) as exc:
        _main("draw", base_testcase=base, output=tmp_path / "next.jsonl", original=orig, original_kpi=kpi)

    assert exc.value.code == 1
    assert errors and "output_filename" in errors[0]


@pytest.mark.parametrize("csv_text", _MALFORMED_GENERATION_CSVS)
def test_round_gate_reports_a_malformed_generation_csv_as_an_input_error(tmp_path, csv_text):
    round_dir, _ = _malformed_generation_bucket(tmp_path, csv_text)

    reason = quality_refine._round_incomplete_reason(round_dir, 1, "kpi.json", "nn")

    assert reason is not None and "output_filename" in reason


# ---------------------------------------------------------------------------
# The original bucket has to clear the same rankability bar as the rounds
# ---------------------------------------------------------------------------
def _write_kpi_keys(path, rows):
    """kpi.json carrying arbitrary per-sample keys (``rows``: {basename: {key: value}})."""
    per_type = defaultdict(list)
    for base, vals in rows.items():
        per_type[base.rsplit("_", 1)[0]].append({"path": f"/x/reconstructed_image/{base}", **vals})
    data = {atype: {"per_sample": ps} for atype, ps in per_type.items()}
    data["Average"] = {"nn_score": 0.0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _aq_buckets(tmp_path, original_scores, round_score):
    """Original + one round, both scored on aq_nn. ``original_scores`` sets the original's per-sample
    keys verbatim so a test can omit aq_nn_score entirely."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    samples = [("A+x_00000.png", 0)]
    _make_bucket(orig, samples, tag="ORIG")
    _write_kpi_keys(tmp_path / "orig_kpi.json", original_scores)
    _make_bucket(rounds / "round_1", samples, tag="R1")
    _write_kpi_keys(rounds / "round_1" / "kpi.json", {"A+x_00000.png": {"aq_nn_score": round_score}})
    return orig, rounds


def test_select_aborts_when_the_original_carries_no_usable_score(tmp_path):
    """Mike's Medium. _load_scores defaults a missing key to NaN and _pick_best replaces a NaN
    incumbent with ANY scored candidate without comparing magnitudes, so an original whose kpi.json
    predates the aq axes forfeits every pick — a strictly worse round wins and the run exits 0
    reporting 'improved over original by search: N/N'. Reachable because --score now takes aq_*."""
    orig, rounds = _aq_buckets(tmp_path, {"A+x_00000.png": {"nn_score": 0.9}}, round_score=0.01)

    with pytest.raises(SystemExit) as e:
        _select(orig, tmp_path / "orig_kpi.json", rounds, tmp_path / "searched", score="aq_nn")
    assert e.value.code == 1
    assert not (tmp_path / "searched").exists(), "must not assemble a bucket the original never contested"


def test_select_ranks_the_original_when_it_does_carry_the_score(tmp_path):
    """The guard must not fire on a well-formed original — and with it scored, the better render wins
    on magnitude rather than on being the only non-NaN candidate."""
    orig, rounds = _aq_buckets(tmp_path, {"A+x_00000.png": {"aq_nn_score": 0.9}}, round_score=0.01)

    _select(orig, tmp_path / "orig_kpi.json", rounds, tmp_path / "searched", score="aq_nn")
    assert _read_summary(tmp_path / "searched")["A+x_00000.png"]["source"] == "original"  # 0.9 > 0.01


def test_select_allows_a_partially_scored_original(tmp_path):
    """One usable value is enough to rank, matching the round gate — only an entirely unscored
    original is a hard failure."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    samples = [("A+x_00000.png", 0), ("A+x_00001.png", 1)]
    _make_bucket(orig, samples, tag="ORIG")
    _write_kpi_keys(
        tmp_path / "orig_kpi.json",
        {"A+x_00000.png": {"nn_score": 0.5}, "A+x_00001.png": {"aq_nn_score": 0.9}},
    )
    _make_bucket(rounds / "round_1", samples, tag="R1")
    _write_kpi_keys(
        rounds / "round_1" / "kpi.json",
        {"A+x_00000.png": {"aq_nn_score": 0.4}, "A+x_00001.png": {"aq_nn_score": 0.4}},
    )

    _select(orig, tmp_path / "orig_kpi.json", rounds, tmp_path / "searched", score="aq_nn")
    summary = _read_summary(tmp_path / "searched")
    assert summary["A+x_00001.png"]["source"] == "original"  # 0.9 > 0.4
    assert summary["A+x_00000.png"]["source"] == "round_1"  # original is NaN here, round is the only value


def test_select_clones_original_with_zero_rounds(tmp_path):
    orig = tmp_path / "out"
    _make_bucket(orig, [("A+x_00000.png", 0)], tag="ORIG")
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.5})
    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", tmp_path / "nope", searched)  # rounds_dir absent
    assert (searched / "reconstructed_image" / "A+x_00000.png").read_bytes() == b"ORIG"
    assert _read_summary(searched)["A+x_00000.png"]["source"] == "original"


def test_select_max_round_zero_clones_original_even_when_rounds_exist(tmp_path):
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    sample = [("A+x_00000.png", 0)]
    _make_bucket(orig, sample, tag="ORIG")
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.1})
    _make_bucket(rounds / "round_1", sample, tag="R1")
    _make_kpi(rounds / "round_1" / "kpi.json", {"A+x_00000.png": 0.9})

    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", rounds, searched, max_round=0)

    assert _read_summary(searched)["A+x_00000.png"]["source"] == "original"
    assert (searched / "reconstructed_image" / "A+x_00000.png").read_bytes() == b"ORIG"


def test_select_aq_rank_with_zero_rounds_clones_an_unrankable_original(tmp_path, monkeypatch):
    """Clone-only mode must not require pooled ranking because there is no competing candidate."""
    orig = tmp_path / "out"
    _make_bucket(orig, [("A+x_00000.png", 0)], tag="ORIG")
    _write_kpi_keys(tmp_path / "orig_kpi.json", {"A+x_00000.png": {"nn_score": 0.5}})
    searched = tmp_path / "searched"

    warnings = []
    monkeypatch.setattr(quality_refine.log, "warning", warnings.append)

    _select(orig, tmp_path / "orig_kpi.json", tmp_path / "nope", searched, score="aq_rank")

    assert (searched / "reconstructed_image" / "A+x_00000.png").read_bytes() == b"ORIG"
    summary = _read_summary(searched)["A+x_00000.png"]
    assert summary["source"] == "original"
    assert summary["score"] == ""
    assert not any("no eligible AQ-Rank candidate" in message for message in warnings)


def test_select_round_wins_when_a_sample_is_unscored(tmp_path):
    """A NaN on *one* sample is a per-sample outcome: that pick goes to the round. It is only an
    entire bucket with no usable value that is a hard failure, so the second sample keeps this
    bucket rankable (see test_select_aborts_when_the_original_carries_no_usable_score)."""
    orig, rounds = tmp_path / "out", tmp_path / "out" / "rounds"
    samples = [("A+x_00000.png", 0), ("A+x_00001.png", 1)]
    _make_bucket(orig, samples, tag="ORIG")
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": float("nan"), "A+x_00001.png": 0.9})
    _make_bucket(rounds / "round_1", samples, tag="R1")
    _make_kpi(rounds / "round_1" / "kpi.json", {"A+x_00000.png": 0.2, "A+x_00001.png": 0.2})
    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", rounds, searched)
    assert (searched / "reconstructed_image" / "A+x_00000.png").read_bytes() == b"R1"
    assert (searched / "reconstructed_image" / "A+x_00001.png").read_bytes() == b"ORIG"  # 0.9 > 0.2


def test_select_refuses_output_equal_to_original(tmp_path):
    orig = tmp_path / "out"
    _make_bucket(orig, [("A+x_00000.png", 0)], tag="ORIG")
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.5})
    with pytest.raises(SystemExit):  # rmtree(--output) must never target the source bucket
        _main("select", original=orig, original_kpi=tmp_path / "orig_kpi.json", output=orig)


def test_select_reports_a_missing_original_generation_csv_cleanly(tmp_path):
    orig = tmp_path / "out"
    orig.mkdir()
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.5})

    with pytest.raises(SystemExit) as exc:
        _select(orig, tmp_path / "orig_kpi.json", tmp_path / "nope", tmp_path / "searched")

    assert exc.value.code == 1
    assert not (tmp_path / "searched").exists()


def test_load_scores_skips_average_and_uses_basename(tmp_path):
    """The Average block is not a sample; keying it in would create a phantom render to select."""
    kpi = tmp_path / "kpi.json"
    _make_kpi(kpi, {"A+x_00000.png": 0.5})
    assert quality_refine._load_scores(kpi, "nn_score") == {"A+x_00000.png": 0.5}


def test_select_writes_per_sample_csv_with_nn_and_mnn(tmp_path):
    orig = tmp_path / "out"
    _make_bucket(orig, [("A+x_00000.png", 0), ("A+x_00001.png", 1)], tag="ORIG")
    # DISTINCT nn vs mnn so we prove both are stitched (not nn copied into both columns).
    _write_kpi_nn_mnn(tmp_path / "orig_kpi.json", {"A+x_00000.png": (0.5, 0.8), "A+x_00001.png": (0.3, 0.7)})
    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", tmp_path / "nope", searched)  # 0 rounds -> clone original

    with (searched / "per_sample.csv").open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["anomaly_type", "path", "nn_score", "mnn_score"]
        rows = {r["path"].rsplit("/", 1)[-1]: r for r in reader}
    assert set(rows) == {"A+x_00000.png", "A+x_00001.png"}
    r0 = rows["A+x_00000.png"]
    assert r0["anomaly_type"] == "A+x"
    assert r0["path"].endswith("searched/reconstructed_image/A+x_00000.png")
    assert r0["nn_score"] == "0.500000" and r0["mnn_score"] == "0.800000"
    assert rows["A+x_00001.png"]["nn_score"] == "0.300000" and rows["A+x_00001.png"]["mnn_score"] == "0.700000"


def test_select_copies_multi_instance_annotated_images_and_rejects_stem_overmatch(tmp_path):
    """annotated_image is the one multi-instance kind: `<stem>_<NNNNN>.png`, globbed per instance. The
    digit guard matters because a plain `<stem>_*` glob also matches a longer sibling's files."""
    orig = tmp_path / "out"
    _make_bucket(orig, [("A+x_00000.png", 0), ("A+x_000001.png", 1)], tag="ORIG")  # note the longer sibling stem
    ann = orig / "annotated_image"
    ann.mkdir(parents=True)
    (ann / "A+x_00000_00000.png").write_bytes(b"I0")  # instance 0 of sample A+x_00000
    (ann / "A+x_00000_00001.png").write_bytes(b"I1")  # instance 1
    (ann / "A+x_00000_extra.png").write_bytes(b"NOPE")  # not _<digits>.png -> must be skipped
    (ann / "A+x_000001_00000.png").write_bytes(b"SIB")  # belongs to the OTHER sample
    _make_kpi(tmp_path / "orig_kpi.json", {"A+x_00000.png": 0.5, "A+x_000001.png": 0.4})

    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", tmp_path / "nope", searched)

    got = sorted(p.name for p in (searched / "annotated_image").glob("*.png"))
    assert got == ["A+x_000001_00000.png", "A+x_00000_00000.png", "A+x_00000_00001.png"]
    assert (searched / "annotated_image" / "A+x_00000_00000.png").read_bytes() == b"I0"
    assert not (searched / "annotated_image" / "A+x_00000_extra.png").exists()  # digit guard held


def test_select_writes_empty_cells_for_nan_scores(tmp_path):
    """_fmt renders NaN as an empty cell rather than the string 'nan', so downstream CSV readers get a
    blank instead of a value that parses as a float."""
    orig = tmp_path / "out"
    _make_bucket(orig, [("A+x_00000.png", 0)], tag="ORIG")
    _write_kpi_nn_mnn(tmp_path / "orig_kpi.json", {"A+x_00000.png": (float("nan"), float("nan"))})

    searched = tmp_path / "searched"
    _select(orig, tmp_path / "orig_kpi.json", tmp_path / "nope", searched)

    row = _read_gen_csv(searched)[0]
    assert row["nn_score"] == "" and row["mnn_score"] == ""
    summary = _read_summary(searched)["A+x_00000.png"]
    assert summary["score"] == "" and summary["original_score"] == ""
    with (searched / "per_sample.csv").open() as f:
        ps = next(csv.DictReader(f))
    assert ps["nn_score"] == "" and ps["mnn_score"] == ""


# --- run -----------------------------------------------------------------------------------------


def _args(tmp_path, **over):
    base = tmp_path / "gen" / "testcase.jsonl"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text('{"index": 0}\n{"index": 1}\n')
    a = dict(
        base_testcase=base,
        original=tmp_path / "out",
        original_kpi=tmp_path / "out" / "kpi.json",
        rounds_dir=tmp_path / "out" / "rounds",
        output=tmp_path / "out" / "searched",
        final_kpi=tmp_path / "out" / "searched" / "final_kpi.json",
        checkpoint=tmp_path / "ckpt.pt",
        recipe=tmp_path / "recipe.yaml",
        real_root=tmp_path / "ds",
        num_search_run=2,
        num_gpus=1,
        score="nn",
        kpi_name="kpi.json",
        guidance_range=(1.5, 10.0),
        crop_ratio_range=(1.5, 8.0),
        bo_min_obs=2,
        bo_candidates=64,
        dry_run=False,
    )
    a.update(over)
    return types.SimpleNamespace(**a)


def _round_dir(root, images=2, kpi="kpi.json", blocked=None, name=None, generation_csv=True):
    """A round dir: ``images`` renders, optionally a kpi.json and a guardrail_blocked.csv."""
    rd = root if name is None else root / name
    (rd / "reconstructed_image").mkdir(parents=True, exist_ok=True)
    for i in range(images):
        (rd / "reconstructed_image" / f"A+x_{i:05d}.png").write_bytes(b"x")
    if generation_csv:
        _write_gen_csv(rd, _gen_rows(images, 3.0, 2.0))
    if kpi:
        # Per-type per_sample rows, not just an Average block: the round gate checks that the metric
        # being optimised is actually on the rows, and a real evaluate.py always writes them.
        rows = [{"path": f"/x/reconstructed_image/A+x_{i:05d}.png", "nn_score": 0.5} for i in range(images)]
        (rd / kpi).write_text(json.dumps({"A+x": {"nn_score": 0.5, "per_sample": rows}, "Average": {"nn_score": 0.5}}))
    if blocked is not None:
        rows = "\n".join(f"{i},0,A+x,c.png,m.png,text,blocked" for i in range(blocked))
        (rd / "guardrail_blocked.csv").write_text(
            "index,output_idx,anomaly_type,image_filename,mask_filename,guardrail,message\n" + rows + "\n"
        )
    return rd


def _stub(monkeypatch, calls, gen_rc=0, **round_kw):
    """Replace draw/select/generate/evaluate with recorders. ``round_kw`` shapes what each generate
    leaves behind — omit it for a well-formed round."""
    # The stub recipe is not a file; with no readable band the explicit range is drawn from as is.
    monkeypatch.setattr(quality_refine, "_trained_crop_ratio_range", lambda _r: None)
    monkeypatch.setattr(quality_refine, "main", lambda argv: calls.append(("draw", argv)))
    monkeypatch.setattr(
        quality_refine,
        "_run_select",
        lambda a: calls.append(("select", a, a.max_round)),
    )

    def gen(args, testcase, out):
        calls.append(("generate", str(out)))
        if round_kw.get("rounds_ok", True):
            _round_dir(out, **{k: v for k, v in round_kw.items() if k != "rounds_ok"})
        return gen_rc

    monkeypatch.setattr(quality_refine, "_generate", gen)
    monkeypatch.setattr(
        quality_refine, "_evaluate", lambda args, gen_root, out: calls.append(("evaluate", str(gen_root))) or 0
    )  # noqa: E501


def test_runs_every_round_then_selects_and_scores(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    quality_refine._run_run(_args(tmp_path))
    assert [c[0] for c in calls] == ["draw", "generate", "evaluate"] * 2 + ["select", "evaluate"]
    assert next(call for call in calls if call[0] == "select")[2] == 2


def test_each_round_gets_its_own_dir_and_seed(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    quality_refine._run_run(_args(tmp_path))
    draws = [call[1] for call in calls if call[0] == "draw"]
    seeds = [d[d.index("--seed") + 1] for d in draws]
    current_rounds = [d[d.index("--current_round") + 1] for d in draws]
    outs = [d[d.index("--output") + 1] for d in draws]
    assert seeds == ["1", "2"], "seed varies per round so rounds are not identical draws"
    assert current_rounds == ["1", "2"], "history must exclude the active and any stale future rounds"
    assert outs[0].endswith("round_1/testcase.jsonl") and outs[1].endswith("round_2/testcase.jsonl")


def test_stops_when_a_round_is_incomplete(tmp_path, monkeypatch):
    """The whole reason this is code: an unfinished round is silently dropped by select and by the
    next round's BO history, so the loop must refuse to continue rather than waste the GPU time."""
    calls = []
    _stub(monkeypatch, calls, rounds_ok=False)
    with pytest.raises(SystemExit) as e:
        quality_refine._run_run(_args(tmp_path))
    assert e.value.code == 1
    assert "select" not in [c[0] for c in calls], "must not select from an incomplete search"
    assert [c[0] for c in calls].count("draw") == 1, "must not start round 2"


@pytest.mark.parametrize(
    "round_kw",
    [{"kpi": None}, {"images": 1}, {"generation_csv": False}],
    ids=["missing-kpi", "short-image-count", "missing-generation-csv"],
)
def test_a_malformed_round_fails_the_round(tmp_path, monkeypatch, round_kw):
    """Either shortfall aborts: images present but no kpi.json, or 1 image for 2 testcase rows."""
    _stub(monkeypatch, [], **round_kw)
    with pytest.raises(SystemExit):
        quality_refine._run_run(_args(tmp_path))


def test_generate_failure_stops_before_select(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls, gen_rc=7)
    with pytest.raises(SystemExit) as e:
        quality_refine._run_run(_args(tmp_path))
    assert e.value.code == 7
    assert "select" not in [c[0] for c in calls]


def test_run_aborts_before_the_first_round_when_the_original_cannot_be_ranked(tmp_path, monkeypatch):
    """Checking this only in select would let the whole search run first: N generation passes, then a
    failure that was decidable before round 1. Same reason the round gate stops the loop."""
    calls = []
    _stub(monkeypatch, calls)
    errors = []
    monkeypatch.setattr(quality_refine.log, "error", errors.append)
    args = _args(tmp_path, score="aq_nn")
    args.original_kpi.parent.mkdir(parents=True, exist_ok=True)
    _write_kpi_keys(args.original_kpi, {"A+x_00000.png": {"nn_score": 0.9}})  # no aq_nn_score

    with pytest.raises(SystemExit) as e:
        quality_refine._run_run(args)
    assert e.value.code == 1
    assert calls == [], "must not draw or generate a single round"
    assert any(f"Re-run evaluate.py on {args.original}" in message for message in errors)


def test_aq_rank_run_aborts_before_generation_when_original_lacks_an_axis(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    args = _args(tmp_path, score="aq_rank")
    _write_gen_csv(args.original, _gen_rows(2, 3.0, 2.0))
    _write_kpi_keys(
        args.original_kpi,
        {
            "A+x_00000.png": {
                "nn_score": 0.9,
                "completeness_score": 0.8,
                "precision_score": 0.7,
                # boundary_iou_score is intentionally absent.
            },
            "A+x_00001.png": {
                "nn_score": 0.8,
                "completeness_score": 0.7,
                "precision_score": 0.6,
                # Every generated candidate is intentionally ineligible.
            },
        },
    )

    with pytest.raises(SystemExit) as exc:
        quality_refine._run_run(args)

    assert exc.value.code == 1
    assert calls == []


def test_run_proceeds_when_the_original_carries_the_score(tmp_path, monkeypatch):
    """The fail-fast guard must not block a well-formed run."""
    calls = []
    _stub(monkeypatch, calls)
    args = _args(tmp_path, num_search_run=1)
    args.original_kpi.parent.mkdir(parents=True, exist_ok=True)
    _write_kpi_keys(args.original_kpi, {"A+x_00000.png": {"nn_score": 0.9}})

    quality_refine._run_run(args)
    assert [c[0] for c in calls] == ["draw", "generate", "evaluate", "select", "evaluate"]


def test_aq_rank_run_reaches_round_two_with_re_ranked_round_one_history(tmp_path, monkeypatch):
    args = _args(tmp_path, score="aq_rank", num_search_run=2, bo_min_obs=99)
    monkeypatch.setattr(quality_refine, "_trained_crop_ratio_range", lambda _r: None)  # stub recipe, no band
    _write_gen_csv(args.original, _gen_rows(2, 3.0, 2.0))
    _write_aq_rank_kpi(
        args.original_kpi,
        {
            "A+x_00000.png": (0.4, 0.4, 0.4, 0.4),
            "A+x_00001.png": (0.5, 0.5, 0.5, 0.5),
        },
    )
    history_snapshots = []
    real_observations = quality_refine._aq_rank_observations

    def record_observations(*call_args, **call_kwargs):
        observations = real_observations(*call_args, **call_kwargs)
        history_snapshots.append(
            (
                call_kwargs.get("before_round"),
                {index: [score for _, _, score in values] for index, values in observations.items()},
            )
        )
        return observations

    def generate(_args, testcase, output_dir):
        rows = [json.loads(line) for line in pathlib.Path(testcase).read_text().splitlines() if line]
        (pathlib.Path(output_dir) / "reconstructed_image").mkdir(parents=True, exist_ok=True)
        generated = []
        for row in rows:
            basename = f"A+x_{int(row['index']):05d}.png"
            (pathlib.Path(output_dir) / "reconstructed_image" / basename).write_bytes(b"round")
            generated.append(
                {
                    "output_filename": basename,
                    "index": row["index"],
                    "guidance": row["guidance"],
                    "crop_ratio": row["crop_ratio"],
                }
            )
        _write_gen_csv(output_dir, generated)
        return 0

    def evaluate(_args, gen_root, output_file):
        if pathlib.Path(gen_root) != pathlib.Path(args.output):
            round_number = int(pathlib.Path(gen_root).name.removeprefix("round_"))
            _write_aq_rank_kpi(
                pathlib.Path(output_file),
                {
                    "A+x_00000.png": (0.4 + round_number / 10, 0.5, 0.5, 0.5),
                    "A+x_00001.png": (0.5 + round_number / 10, 0.6, 0.6, 0.6),
                },
            )
        return 0

    selected_bounds = []
    monkeypatch.setattr(quality_refine, "_aq_rank_observations", record_observations)
    monkeypatch.setattr(quality_refine, "_generate", generate)
    monkeypatch.setattr(quality_refine, "_evaluate", evaluate)
    monkeypatch.setattr(
        quality_refine,
        "_run_select",
        lambda select_args: selected_bounds.append(select_args.max_round),
    )

    quality_refine._run_run(args)

    history_sizes = [
        (round_number, {index: len(scores) for index, scores in samples.items()})
        for round_number, samples in history_snapshots
    ]
    assert history_sizes == [(1, {0: 1, 1: 1}), (2, {0: 2, 1: 2})]
    assert history_snapshots[0][1][1][0] == 1.0
    assert history_snapshots[1][1][1][0] == 0.5
    assert selected_bounds == [2]


def test_zero_rounds_still_selects_and_scores(tmp_path, monkeypatch):
    """num_search_run=0 reduces to cloning the original into searched/ — searched/ must still exist."""
    calls = []
    _stub(monkeypatch, calls)
    quality_refine._run_run(_args(tmp_path, num_search_run=0))
    assert [c[0] for c in calls] == ["select", "evaluate"]


def test_draw_round_forwards_aq_rank_and_the_explicit_round_boundary(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(quality_refine, "main", lambda argv: calls.append(argv))
    args = _args(tmp_path, score="aq_rank")

    quality_refine._draw_round(args, tmp_path / "round_2" / "testcase.jsonl", round_number=2)

    argv = calls[0]
    assert argv[argv.index("--score") + 1] == "aq_rank"
    assert argv[argv.index("--current_round") + 1] == "2"


@pytest.mark.parametrize(
    ("images", "blocked", "rows", "expected"),
    [
        # The guardrail is on by default and writes no image for a blocked sample. Counting images
        # against rows alone would abort the whole search over one blocked caption.
        (4, 1, 5, None),
        # The guardrail allowance must not swallow a genuinely truncated round.
        (3, 1, 5, "3 image(s) + 1 guardrail-blocked for 5"),
        # generate.py writes a header-only file whenever the guardrail ran with nothing blocked.
        (5, 0, 5, None),
        # --no-guardrail writes no sidecar at all; absent must read as zero, not as an error.
        (5, None, 5, None),
    ],
    ids=["blocked-counts-as-done", "short-for-another-reason", "header-only-csv", "no-csv-at-all"],
)
def test_guardrail_blocked_samples_count_toward_a_finished_round(tmp_path, images, blocked, rows, expected):
    rd = _round_dir(tmp_path, images=images, blocked=blocked, name="round_1")
    reason = quality_refine._round_incomplete_reason(rd, rows, "kpi.json")
    if expected is None:
        assert reason is None
    else:
        assert reason and expected in reason
    if blocked is not None or images == 5:
        assert quality_refine._blocked_count(rd) == (blocked or 0)


def test_missing_kpi_still_fails_regardless_of_the_guardrail(tmp_path):
    rd = _round_dir(tmp_path, images=5, kpi=None, blocked=0, name="round_1")
    assert quality_refine._round_incomplete_reason(rd, 5, "kpi.json") == "missing kpi.json"


def test_round_incomplete_reason_reports_what_is_wrong(tmp_path):
    """The message is what the operator acts on, so it must name the shortfall, not just fail."""
    rd = _round_dir(tmp_path, name="round_1")
    assert quality_refine._round_incomplete_reason(rd, 2, "kpi.json") is None
    assert quality_refine._round_incomplete_reason(rd, 1, "kpi.json") == (
        "2 image(s) + 0 guardrail-blocked for 1 testcase row(s)"
    )
    (rd / "kpi.json").unlink()
    assert "missing kpi.json" in quality_refine._round_incomplete_reason(rd, 2, "kpi.json")


def test_generate_argv_passes_the_checkpoint_and_io_paths(tmp_path):
    a = _args(tmp_path, num_gpus=4, dry_run=True)
    argv = quality_refine._generate_argv(a, "tc.jsonl", "out")
    assert argv[:2] == ["--checkpoint", str(a.checkpoint)]
    assert "--input_data_path" in argv and "--output_dir" in argv


def test_generate_env_sets_the_output_root_explicitly():
    """Unset IMAGINAIRE_OUTPUT_ROOT does not fail — the framework falls back to /tmp/imaginaire4-output
    — so the child must be told, not left to inherit."""
    env = quality_refine._generate_env("/w/results", environ={"PATH": "/usr/bin"})
    assert env["IMAGINAIRE_OUTPUT_ROOT"] == "/w/results"
    assert env["PATH"] == "/usr/bin", "the rest of the environment is passed through"


@pytest.mark.parametrize(
    "dirty",
    [
        # A stale RANK/WORLD_SIZE from an outer launch would fight the child torchrun's own rendezvous,
        # which generate.py reports as a world-size mismatch rather than anything diagnostic.
        {
            "RANK": "0", "WORLD_SIZE": "8", "LOCAL_RANK": "0", "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "29500", "TORCHELASTIC_RUN_ID": "abc",
        },
        # PET_<ARG> is torchrun's env fallback for its own CLI args. The three we pass explicitly are
        # already immune (a flag beats the env default), but PET_NNODES would leave the child waiting
        # on a multi-node rendezvous that never arrives — so strip the namespace rather than enumerate.
        {
            "PET_NNODES": "4", "PET_RDZV_ID": "shared", "PET_MAX_RESTARTS": "9",
            "TORCHELASTIC_RUN_ID": "abc", "TORCHELASTIC_ERROR_FILE": "/tmp/e.json",
        },
    ],
    ids=["torchrun-rendezvous", "PET-namespace"],
)  # fmt: skip
def test_generate_env_strips_inherited_launcher_variables(dirty):
    env = quality_refine._generate_env("results", environ={"PATH": "/usr/bin", **dirty})
    assert set(env) == {"PATH", "IMAGINAIRE_OUTPUT_ROOT"}


def test_generate_env_keeps_lookalikes_and_leaves_the_caller_environment_alone():
    """Prefix stripping must not eat a user's own variables, and must copy rather than mutate — the
    caller's environ is reused by the next round."""
    keep = {"PETSC_DIR": "/opt/petsc", "MY_PET": "cat", "HF_HOME": "/hf"}
    dirty = {**keep, "RANK": "3"}
    env = quality_refine._generate_env("results", environ=dirty)
    assert {k: env[k] for k in keep} == keep
    assert "RANK" not in env
    assert dirty == {**keep, "RANK": "3"}, "the caller's mapping must be untouched"


def test_generate_uses_a_free_rendezvous_port_per_round():
    """Each round is a fresh torchrun; the static backend pins MASTER_PORT=29500, so a round starting
    before the previous one released it fails to bind. --standalone asks for an ephemeral port."""
    a = _args(pathlib.Path("/tmp"), num_gpus=4)
    cmd = quality_refine._generate_cmd(a, "tc.jsonl", "out")
    # This interpreter's torch.distributed.run, not the `torchrun` console script: that name is only
    # on PATH with the venv activated, and the entrypoint is invoked as .venv/bin/python -m ...
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in cmd
    assert f"--nproc_per_node={a.num_gpus}" in cmd
    assert cmd.index("--standalone") < [i for i, c in enumerate(cmd) if c.endswith("generate.py")][0], (
        "launcher flags must precede the script path"
    )


def test_standalone_is_what_torchrun_itself_means_by_it():
    """--standalone is only worth trusting if torchrun still expands it to a c10d rendezvous on an
    ephemeral port. Pinned against the installed torch so a change in that meaning fails here rather
    than as a mid-search bind error."""
    run_py = pathlib.Path(importlib.util.find_spec("torch.distributed.run").origin).read_text()
    body = run_py.split("if args.standalone:", 1)[1].split("logger.info", 1)[0]
    assert 'args.rdzv_backend = "c10d"' in body
    assert 'args.rdzv_endpoint = "localhost:0"' in body
    assert "args.rdzv_id = str(uuid.uuid4())" in body, "unique run id per round"


# ---------------------------------------------------------------------------
# Hand-off to evaluate.py
#
# The search shells out to evaluate.py once per round, so whatever it fails to forward is silently
# reset to evaluate's defaults — the rounds are then scored with a *different* metric than the Step 5
# KPI they are selected against, and select picks winners on it. These lock the argv it builds, the
# per-sample score lookup that reads the result back, and the gate that refuses an unrankable round.
# ---------------------------------------------------------------------------
_REQUIRED_RUN = [
    "run",
    "--base_testcase", "base.jsonl",
    "--original", "out",
    "--original_kpi", "out/kpi.json",
    "--rounds_dir", "out/rounds",
    "--output", "out/searched",
    "--final_kpi", "out/searched/kpi.json",
    "--checkpoint", "ckpt.pt",
    "--recipe", "run/exp.yaml",
    "--real_root", "datasets/ds",
]  # fmt: skip


def _eval_args(**over):
    base = dict(real_root="datasets/ds", recipe="run/exp.yaml", score="nn")
    base.update(over)
    return types.SimpleNamespace(**base)


def _tail(argv):
    """The part after the four always-present core flags."""
    return argv[8:]


# ---------------------------------------------------------------------------
# _evaluate_argv — what reaches each round's evaluate.py
# ---------------------------------------------------------------------------
def test_core_flags_are_always_present():
    argv = quality_refine._evaluate_argv(_eval_args(), "rounds/round_1", "rounds/round_1/kpi.json")
    assert argv[:8] == [
        "--gen_root",
        "rounds/round_1",
        "--real_root",
        "datasets/ds",
        "--recipe",
        "run/exp.yaml",
        "--output_file",
        "rounds/round_1/kpi.json",
    ]


def test_scoring_knobs_are_forwarded():
    """The knobs that change the computed numbers must reach every round."""
    argv = _tail(
        quality_refine._evaluate_argv(
            _eval_args(top_k=5, model_input_size=256, nn_region_policy="full", nn_layer=-1, nn_readout="mean"),
            "g",
            "o.json",
        )
    )
    for flag, value in [
        ("--top_k", "5"),
        ("--model_input_size", "256"),
        ("--nn_region_policy", "full"),
        ("--nn_layer", "-1"),
        ("--nn_readout", "mean"),
    ]:
        assert flag in argv and argv[argv.index(flag) + 1] == value


def test_unset_optional_knobs_are_not_forwarded():
    """``None`` means "use evaluate's default" — forwarding it would stringify to 'None'."""
    assert _tail(quality_refine._evaluate_argv(_eval_args(top_k=None, model_input_size=None), "g", "o.json")) == []


def test_parsed_run_args_carry_the_nn_defaults_through():
    """End-to-end: a real `run` invocation forwards the nn scoring config it was parsed with, so the
    rounds are scored the same way as the Step 5 KPI they are compared against."""
    args = quality_refine._get_args([*_REQUIRED_RUN, "--nn_region_policy", "full", "--nn_readout", "mean"])
    tail = _tail(quality_refine._evaluate_argv(args, "g", "o.json"))
    assert tail[tail.index("--nn_region_policy") + 1] == "full"
    assert tail[tail.index("--nn_readout") + 1] == "mean"
    # the knobs left at their defaults still travel, so a round is never scored under a silent fallback
    assert "--nn_layer" in tail and "--nn_inst_agg" in tail


# ---------------------------------------------------------------------------
# score vocabulary
# ---------------------------------------------------------------------------
def test_anomaly_quality_scores_are_selectable():
    """evaluate.py always scores the axes now, so the search can optimize on them."""
    assert set(quality_refine._SCORE_CHOICES) == {
        "nn",
        "mnn",
        "completeness",
        "precision",
        "boundary_iou",
        "aq_nn",
        "aq_rank",
    }


def test_filter_aq_rank_is_also_selectable_for_refinement():
    """The same public score name uses cumulative search and full-pool final selection internally."""
    assert "aq_rank" in quality_refine._SCORE_CHOICES


# ---------------------------------------------------------------------------
# _load_scores — reading the axes back out of a round KPI
# ---------------------------------------------------------------------------
def test_load_scores_reads_axis_keys(tmp_path):
    """evaluate.py folds the axes into the same ``per_sample`` rows, so one lookup path serves all."""
    kpi = {
        "Phone+oil": {
            "per_sample": [
                {"path": "/gen/a.png", "nn_score": 0.5, "completeness_score": 0.9, "aq_nn_score": 1.4},
                {"path": "/gen/b.png", "nn_score": 0.7, "completeness_score": 0.2, "aq_nn_score": 0.9},
            ]
        },
        "Average": {"nn_score": 0.6},  # must be skipped
    }
    path = tmp_path / "kpi.json"
    path.write_text(json.dumps(kpi))

    assert quality_refine._load_scores(path, "nn_score") == {"a.png": 0.5, "b.png": 0.7}
    assert quality_refine._load_scores(path, "completeness_score") == {"a.png": 0.9, "b.png": 0.2}
    assert quality_refine._load_scores(path, "aq_nn_score") == {"a.png": 1.4, "b.png": 0.9}


def test_load_scores_missing_key_is_nan(tmp_path):
    """A key absent from the rows must not silently rank every sample equal."""
    path = tmp_path / "kpi.json"
    path.write_text(json.dumps({"Phone+oil": {"per_sample": [{"path": "/gen/a.png", "nn_score": 0.5}]}}))
    got = quality_refine._load_scores(path, "completeness_score")["a.png"]
    assert got != got  # NaN


# ---------------------------------------------------------------------------
# _round_incomplete_reason — a round only counts if it can actually be ranked
# ---------------------------------------------------------------------------
def _round_with_rows(tmp_path, rows, images=2):
    """A round dir that passes the file/count checks, carrying ``rows`` as its per-sample block."""
    rd = tmp_path / "round_1"
    (rd / "reconstructed_image").mkdir(parents=True)
    names = [pathlib.Path(row.get("path", f"{i}.png")).name for i, row in enumerate(rows[:images])]
    for name in names:
        (rd / "reconstructed_image" / name).write_bytes(b"")
    _write_gen_csv(
        rd,
        [{"output_filename": name, "index": i, "guidance": 3.0, "crop_ratio": 2.0} for i, name in enumerate(names)],
    )
    (rd / "kpi.json").write_text(json.dumps({"Phone+oil": {"per_sample": rows}}))
    return rd


def test_round_with_the_scored_metric_is_complete(tmp_path):
    rd = _round_with_rows(
        tmp_path, [{"path": "/gen/a.png", "aq_nn_score": 1.4}, {"path": "/gen/b.png", "aq_nn_score": 0.9}]
    )
    assert quality_refine._round_incomplete_reason(rd, 2, "kpi.json", "aq_nn") is None


def test_round_missing_the_scored_metric_is_incomplete(tmp_path):
    """The silent failure John flagged: evaluate.py degrades to NN/MNN when anomaly_quality raises
    (a missing SAM2 checkpoint is enough), so the rows arrive complete and correctly counted but
    carry no aq_nn_score. Unchecked, every sample loads as NaN, _pick_best keeps the earliest source,
    and the original wins every sample — N rounds burned for a byte-identical searched/, exit 0."""
    rd = _round_with_rows(tmp_path, [{"path": "/gen/a.png", "nn_score": 0.5}, {"path": "/gen/b.png", "nn_score": 0.7}])
    reason = quality_refine._round_incomplete_reason(rd, 2, "kpi.json", "aq_nn")
    assert reason is not None and "aq_nn_score" in reason


def test_round_missing_aq_rank_axes_is_incomplete(tmp_path):
    rd = _round_with_rows(
        tmp_path,
        [{"path": "/gen/a.png", "nn_score": 0.5}, {"path": "/gen/b.png", "nn_score": 0.7}],
    )
    reason = quality_refine._round_incomplete_reason(rd, 2, "kpi.json", "aq_rank")
    assert reason is not None and "no generated candidate has all finite raw inputs" in reason


def test_round_with_one_invalid_aq_rank_candidate_is_still_complete(tmp_path):
    rd = _round_with_rows(
        tmp_path,
        [
            {
                "path": "/gen/a.png",
                "nn_score": 0.5,
                "completeness_score": 0.5,
                "precision_score": 0.5,
                "boundary_iou_score": 0.5,
            },
            {
                "path": "/gen/b.png",
                "nn_score": 0.7,
                "completeness_score": float("nan"),
                "precision_score": 0.7,
                "boundary_iou_score": 0.7,
            },
        ],
    )

    assert quality_refine._round_incomplete_reason(rd, 2, "kpi.json", "aq_rank") is None


def test_round_aq_rank_kpi_must_join_every_generated_candidate(tmp_path):
    rd = _round_with_rows(
        tmp_path,
        [
            {
                "path": "/gen/a.png",
                "nn_score": 0.5,
                "completeness_score": 0.5,
                "precision_score": 0.5,
                "boundary_iou_score": 0.5,
            }
        ],
        images=1,
    )
    # Add a second generated candidate without adding its KPI row.
    (rd / "reconstructed_image" / "b.png").write_bytes(b"")
    _write_gen_csv(
        rd,
        [
            {"output_filename": "a.png", "index": 0, "guidance": 3.0, "crop_ratio": 2.0},
            {"output_filename": "b.png", "index": 1, "guidance": 3.0, "crop_ratio": 2.0},
        ],
    )

    reason = quality_refine._round_incomplete_reason(rd, 2, "kpi.json", "aq_rank")

    assert reason is not None and "AQ-Rank" in reason and "no KPI row" in reason


def test_round_with_one_usable_value_is_complete(tmp_path):
    """Partial anomaly_quality still ranks — only an entirely unscored round is unrankable."""
    rd = _round_with_rows(tmp_path, [{"path": "/gen/a.png"}, {"path": "/gen/b.png", "aq_nn_score": 0.9}])
    assert quality_refine._round_incomplete_reason(rd, 2, "kpi.json", "aq_nn") is None


def test_round_metric_check_is_opt_in(tmp_path):
    """Omitting score keeps the original file/count-only behaviour for callers that don't rank."""
    rd = _round_with_rows(tmp_path, [{"path": "/gen/a.png"}, {"path": "/gen/b.png"}])
    assert quality_refine._round_incomplete_reason(rd, 2, "kpi.json") is None


def test_round_count_mismatch_still_wins_over_the_metric_check(tmp_path):
    """An incomplete generation is the more actionable diagnosis, so it must be reported first."""
    rd = _round_with_rows(tmp_path, [{"path": "/gen/a.png"}], images=1)
    reason = quality_refine._round_incomplete_reason(rd, 2, "kpi.json", "aq_nn")
    assert reason is not None and "image(s)" in reason


# --- the recipe's trained crop_ratio_range --------------------------------------------------------


def test_the_band_is_read_off_the_node_the_recipe_registered(monkeypatch, register_experiment):
    """Falling back to the default here would sweep a band the model never trained on."""
    # A register_recipe stand-in: stores the node the way a recipe does and returns its name.
    monkeypatch.setattr(
        quality_refine, "register_recipe", lambda _r: register_experiment("from_a_module", crop_ratio_range=[1.5, 8.0])
    )
    assert quality_refine._trained_crop_ratio_range("ag_config.some_recipe") == [1.5, 8.0]


def test_a_recipe_that_overwrites_an_existing_node_is_read_fresh(monkeypatch, register_experiment):
    """A module names its node itself, so it can replace one already registered; the replacement is
    what must be read."""
    register_experiment("shared_name", crop_ratio_range=[1.5, 2.5])
    monkeypatch.setattr(
        quality_refine, "register_recipe", lambda _r: register_experiment("shared_name", crop_ratio_range=[2.0, 6.0])
    )
    assert quality_refine._trained_crop_ratio_range("ag_config.some_recipe") == [2.0, 6.0]


def test_an_unreadable_band_warns_instead_of_silently_defaulting(monkeypatch, loguru_lines):
    """The failure this guards is silent: sweeping the default band against another trained band."""
    monkeypatch.setattr(quality_refine, "_trained_crop_ratio_range", lambda _r: None)
    args = types.SimpleNamespace(crop_ratio_range=None, recipe="ag_config.some_recipe")
    assert quality_refine._resolve_crop_ratio_range(args) == list(quality_refine.DEFAULT_CROP_RATIO_RANGE)
    assert any("crop_ratio_range" in line for line in loguru_lines)


def _resolve(monkeypatch, band, explicit):
    monkeypatch.setattr(quality_refine, "_trained_crop_ratio_range", lambda _r: band)
    return quality_refine._resolve_crop_ratio_range(
        types.SimpleNamespace(crop_ratio_range=explicit, recipe="ag_config.some_recipe")
    )


def test_an_explicit_range_is_narrowed_to_the_trained_band(monkeypatch, loguru_lines):
    """Generation clamps every draw into the band, so the part outside it only repeats the band's
    edge — while the GP, scaled by the draw range, keeps proposing there. The crop axis would be dead."""
    assert _resolve(monkeypatch, [1.5, 2.5], (1.5, 8.0)) == [1.5, 2.5]
    assert any("narrowed" in line for line in loguru_lines)


def test_an_explicit_range_inside_the_band_is_kept_as_is(monkeypatch, loguru_lines):
    """Narrowing the search below the band is a legitimate choice, not a mistake to warn about."""
    assert _resolve(monkeypatch, [1.5, 2.5], (2.0, 2.4)) == [2.0, 2.4]
    assert not any("narrowed" in line for line in loguru_lines)


def test_an_explicit_range_outside_the_band_is_refused(monkeypatch):
    """Nothing to search: every draw would collapse onto one band edge."""
    with pytest.raises(ValueError, match="outside the recipe's trained band"):
        _resolve(monkeypatch, [1.5, 2.5], (3.0, 5.0))
    with pytest.raises(ValueError, match="outside"):
        _resolve(monkeypatch, [1.5, 2.5], (2.5, 8.0))  # touching the edge is a degenerate range


def test_an_explicit_range_is_used_as_is_when_no_band_is_readable(monkeypatch, loguru_lines):
    """This is how ``run`` hands each round's draw its already-resolved range without a recipe."""
    assert _resolve(monkeypatch, None, (1.5, 8.0)) == [1.5, 8.0]
    assert not any("crop_ratio_range" in line for line in loguru_lines)


def test_generate_rounds_inherit_the_batch_sampling_setup(tmp_path):
    """A round is only comparable to the original bucket if it was produced the same way -- the same
    reason _evaluate_argv forwards its scoring knobs.

    generate.py's --guardrail defaults to ON, so before this a --no-guardrail batch had its search
    rounds generated WITH the guardrail, and the winners select() picked came from those rounds.
    """
    args = types.SimpleNamespace(
        checkpoint="ckpt.pt", recipe="r.yaml", base_checkpoint="base.pt", base_seed=7, guardrail=False
    )
    argv = quality_refine._generate_argv(args, "t.jsonl", "out")
    flags = {a: argv[i + 1] for i, a in enumerate(argv) if a.startswith("--") and i + 1 < len(argv)}
    assert flags["--base_checkpoint"] == "base.pt"
    assert flags["--base_seed"] == "7"
    assert "--no-guardrail" in argv


def test_generate_rounds_send_only_what_the_caller_set(tmp_path):
    """Unset knobs must leave generate.py's own defaults authoritative, not arrive as None."""
    args = types.SimpleNamespace(
        checkpoint="ckpt.pt", recipe="r.yaml", base_checkpoint=None, base_seed=None, guardrail=True
    )
    argv = quality_refine._generate_argv(args, "t.jsonl", "out")
    assert "--base_checkpoint" not in argv and "--base_seed" not in argv
    # The positive form is generate.py's default; only the negative needs sending.
    assert "--guardrail" not in argv and "--no-guardrail" not in argv
