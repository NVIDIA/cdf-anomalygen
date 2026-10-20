# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NN/MNN-based generated-image filter.

The selection/routing/CSV logic is tested directly with real inputs — ``route_by_scores`` takes a
plain correspondence-KPI dict, so no model (and no patching) is needed. The full model-backed path
(``filter_generated_images``) is covered by a golden test gated on the DINOv2 checkpoint.
"""

import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anomalygen.eval import anomaly_quality as aq_module
from anomalygen.models.vision_encoder.dinov2 import DEFAULT_BACKBONE
from anomalygen.scripts.texture import filter as filter_module

_GEN_CSV = "texture_ft_generation_result.csv"
_FILTERED_CSV = "texture_ft_generation_result_filtered.csv"
_SUBDIRS = ("reconstructed_image", "original_mask", "original_image")
_DINOV2_WEIGHTS = os.path.join(DEFAULT_BACKBONE, "model.safetensors")


def _write_gen_tree(gen_root, key, count):
    """Create ``count`` recon/mask/orig triples sharing a filename; return the filenames."""
    names = []
    for idx in range(count):
        name = f"{key}_{idx:05d}.png"
        for sub in _SUBDIRS:
            d = gen_root / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_text(sub)  # content marks which subdir it came from
        names.append(name)
    return names


def _kpi(recon_dir, names, nn_scores, mnn_scores=None):
    mnn_scores = nn_scores if mnn_scores is None else mnn_scores
    per_sample = [
        {"path": str(recon_dir / n), "nn_score": nn, "mnn_score": mnn}
        for n, nn, mnn in zip(names, nn_scores, mnn_scores)
    ]
    return {
        "K+d": {
            "nn_score": float(np.mean(nn_scores)),
            "mnn_score": float(np.mean(mnn_scores)),
            "per_sample": per_sample,
        }
    }


def test_filter_topk_basic():
    keep_idx, drop_idx = filter_module.filter_topk(np.array([0.1, 0.3, 0.2, 0.4]), drop_ratio=0.25)
    assert list(keep_idx) == [3, 1, 2]
    assert list(drop_idx) == [0]


def test_filter_topk_drop_all():
    keep_idx, drop_idx = filter_module.filter_topk(np.array([1.0, 0.5]), drop_ratio=1.0)
    assert list(keep_idx) == []
    assert set(drop_idx) == {0, 1}


def test_route_by_scores_routes_highest_scores_to_keep(tmp_path):
    gen_root = tmp_path / "gen"
    names = _write_gen_tree(gen_root, "K+d", count=4)
    recon_dir = gen_root / "reconstructed_image"
    # idx1 (0.4) and idx3 (0.3) are the top-2 -> kept at drop_ratio 0.5.
    kpi = _kpi(recon_dir, names, [0.1, 0.4, 0.2, 0.3])

    output_dir = tmp_path / "out"
    kept, dropped = filter_module.route_by_scores(kpi, str(gen_root), str(output_dir), drop_ratio=0.5)

    assert set(kept) == {"K+d_00001.png", "K+d_00003.png"}
    assert set(dropped) == {"K+d_00000.png", "K+d_00002.png"}
    # every kept/dropped file is copied into all three subdirs of its split
    for split, group in (("keep", kept), ("drop", dropped)):
        for sub in _SUBDIRS:
            assert {p.name for p in (output_dir / split / sub).iterdir()} == set(group)


@pytest.mark.parametrize("drop_ratio", [0.0, 0.1, 0.3, 0.5, 1.0])
def test_route_by_scores_drop_ratio_math(tmp_path, drop_ratio):
    gen_root = tmp_path / "gen"
    count = 20
    names = _write_gen_tree(gen_root, "K+d", count=count)
    recon_dir = gen_root / "reconstructed_image"
    kpi = _kpi(recon_dir, names, list(np.linspace(0.0, 1.0, count)))

    output_dir = tmp_path / "out"
    kept, dropped = filter_module.route_by_scores(kpi, str(gen_root), str(output_dir), drop_ratio=drop_ratio)

    expected_keep = int(round((1 - drop_ratio) * count))
    assert len(kept) == expected_keep
    assert len(dropped) == count - expected_keep


def test_route_by_scores_can_rank_on_mnn(tmp_path):
    gen_root = tmp_path / "gen"
    names = _write_gen_tree(gen_root, "K+d", count=2)
    recon_dir = gen_root / "reconstructed_image"
    # nn favours idx0, mnn favours idx1; selecting on mnn keeps idx1.
    kpi = _kpi(recon_dir, names, nn_scores=[0.9, 0.1], mnn_scores=[0.1, 0.9])

    kept, _ = filter_module.route_by_scores(kpi, str(gen_root), str(tmp_path / "out"), drop_ratio=0.5, score="mnn")
    assert set(kept) == {names[1]}


def test_route_by_scores_can_rank_on_aq_nn(tmp_path):
    """--score aq_nn ranks by the absolute per-sample completeness + nn_score, distinct from either term."""
    gen_root = tmp_path / "gen"
    names = _write_gen_tree(gen_root, "K+d", count=3)
    recon_dir = gen_root / "reconstructed_image"
    kpi = _kpi(recon_dir, names, nn_scores=[0.0, 0.5, 0.7])  # nn favours idx2
    # completeness favours idx0, but aq_nn = completeness + nn peaks at idx1.
    aq_kpi = {
        "K+d": {
            "per_sample_axes": [
                {"path": str(recon_dir / names[i]), "completeness": c, "precision": 0.5, "boundary_iou": 0.5}
                for i, c in enumerate([0.9, 0.6, 0.0])
            ]
        }
    }
    aq_module.augment_with_quality(kpi, aq_kpi)
    rows = kpi["K+d"]["per_sample"]
    assert [round(r["aq_nn_score"], 3) for r in rows] == [0.9, 1.1, 0.7]  # completeness + nn_score

    kept, _ = filter_module.route_by_scores(kpi, str(gen_root), str(tmp_path / "out"), drop_ratio=2 / 3, score="aq_nn")
    assert set(kept) == {names[1]}  # aq_nn's top, not nn's idx2 nor completeness's idx0


@pytest.mark.parametrize("bad", ["1.5", "-0.3"])
def test_main_rejects_out_of_range_drop_ratio(bad):
    with pytest.raises(ValueError):
        filter_module.main(["--gen_root", "g", "--real_root", "r", "--output_dir", "o", "--drop_ratio", bad])


def test_route_by_scores_routes_nan_scores_to_drop(tmp_path):
    # compute_correspondence_kpi emits NaN for degenerate (no-foreground) samples; they must be
    # DROPPED, not kept — np.argsort would otherwise sort NaN to the front of keep.
    gen_root = tmp_path / "gen"
    names = _write_gen_tree(gen_root, "K+d", count=4)
    recon_dir = gen_root / "reconstructed_image"
    kpi = _kpi(recon_dir, names, nn_scores=[float("nan"), 0.1, 0.5, 0.9])

    kept, dropped = filter_module.route_by_scores(kpi, str(gen_root), str(tmp_path / "out"), drop_ratio=0.25)

    assert names[0] in dropped  # the NaN sample is dropped
    assert names[0] not in kept


def test_route_by_scores_skips_type_with_no_samples(tmp_path):
    kept, dropped = filter_module.route_by_scores(
        {"K+d": {"per_sample": []}}, str(tmp_path / "gen"), str(tmp_path / "out"), drop_ratio=0.5
    )
    assert kept == [] and dropped == []


def test_save_filter_result_csv_writes_per_sample_scores(tmp_path):
    kpi = {
        "K+d": {
            "per_sample": [
                {"path": "/x/a.png", "nn_score": 0.8, "mnn_score": 0.5},
                {"path": "/x/b.png", "nn_score": float("nan"), "mnn_score": float("nan")},
            ]
        },
        "Average": {"nn_score": 0.8, "mnn_score": 0.5},
    }
    filter_module.save_filter_result_csv(kpi, str(tmp_path))

    df = pd.read_csv(tmp_path / _FILTERED_CSV)
    assert list(df.columns) == ["anomaly_type", "filename", "nn_score", "mnn_score"]
    assert set(df["filename"]) == {"a.png", "b.png"}
    assert "Average" not in set(df["anomaly_type"])
    assert df[df["filename"] == "a.png"].iloc[0]["nn_score"] == pytest.approx(0.8)


def test_save_filtered_generation_csv_splits_by_basename(tmp_path):
    gen_dir = tmp_path / "gen"
    recon_dir = gen_dir / "reconstructed_image"
    recon_dir.mkdir(parents=True)
    keep_file, drop_file = recon_dir / "keep.png", recon_dir / "drop.png"
    pd.DataFrame(
        [{"output_filename": str(keep_file), "metric": 1.0}, {"output_filename": str(drop_file), "metric": 0.1}]
    ).to_csv(gen_dir / _GEN_CSV, index=False)

    output_dir = tmp_path / "out"
    for split in ("keep", "drop"):
        (output_dir / split).mkdir(parents=True)

    filter_module.save_filtered_generation_csv(str(gen_dir), str(output_dir), ["keep.png"], ["drop.png"])

    keep_df = pd.read_csv(output_dir / "keep" / _GEN_CSV)
    drop_df = pd.read_csv(output_dir / "drop" / _GEN_CSV)
    assert list(keep_df["output_filename"]) == [str(keep_file)]
    assert list(drop_df["output_filename"]) == [str(drop_file)]


def test_save_filtered_generation_csv_missing_file_is_noop(tmp_path):
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    output_dir = tmp_path / "out"
    for split in ("keep", "drop"):
        (output_dir / split).mkdir(parents=True)

    filter_module.save_filtered_generation_csv(str(gen_dir), str(output_dir), ["foo.png"], ["bar.png"])

    assert not (output_dir / "keep" / _GEN_CSV).exists()
    assert not (output_dir / "drop" / _GEN_CSV).exists()


# --------------------------------------------------------------------------- compass composite (pure)
def _rows(nn, **axes):
    """Build per-sample rows: nn plus any {precision,completeness,boundary_iou}_score sequences."""
    return [{"nn_score": nn[i], **{f"{a}_score": v[i] for a, v in axes.items()}} for i in range(len(nn))]


def test_rank01_and_auroc_primitives():
    r = aq_module._rank01([0.1, 0.9, 0.5, float("nan")])
    assert r[3] == 0.0 and r[1] == 1.0 and 0.0 < r[2] < 1.0  # NaN sorts lowest
    assert aq_module._auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert aq_module._auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert aq_module._auroc(np.array([0.5, 0.5]), np.array([1, 1])) is None  # one class only


def test_aq_rank_agreeing_axis_is_added_and_keeps_nn_order():
    xs = np.linspace(0, 1, 12)
    q = aq_module.compute_aq_rank_scores(_rows(xs, precision=xs, completeness=[0.5] * 12, boundary_iou=[0.5] * 12))
    # precision agrees with nn's extremes -> '+', flat axes gate out -> order still tracks nn
    assert np.argmax(q) == 11 and np.argmin(q) == 0
    assert list(q) == sorted(q)


def test_aq_rank_flips_disagreeing_axis():
    xs = np.linspace(0, 1, 10)
    q = aq_module.compute_aq_rank_scores(
        _rows(xs, precision=1.0 - xs, completeness=[0.5] * 10, boundary_iou=[0.5] * 10)
    )
    # anti-correlated axis gets '-', which realigns it with nn — the ranking is not inverted
    assert list(np.argsort(q)) == list(range(10))


def test_aq_rank_gates_signalless_axes_to_plain_nn():
    xs = np.linspace(0, 1, 8)
    q = aq_module.compute_aq_rank_scores(_rows(xs, precision=[0.5] * 8, completeness=[0.5] * 8, boundary_iou=[0.5] * 8))
    assert np.allclose(q, aq_module._rank01(xs))  # every axis constant -> gated -> aq_rank == rank(nn)


def test_aq_rank_nan_nn_sorts_last():
    rows = _rows([0.1, 0.3, 0.5, 0.7, 0.9], precision=[0.1, 0.3, 0.5, 0.7, 0.9])
    rows.append({"nn_score": float("nan"), "precision_score": 1.0})  # degenerate: no nn, high axis
    q = aq_module.compute_aq_rank_scores(rows)
    assert q[-1] == -np.inf and np.isfinite(q[:-1]).all()


def test_aq_rank_small_set_falls_back_to_nn_rank():
    nn = [0.2, 0.9, 0.5, 0.1]  # n=4 < COMPASS_MIN_N -> no compass
    q = aq_module.compute_aq_rank_scores(_rows(nn, precision=[0.9, 0.1, 0.5, 0.2]))
    assert list(np.argsort(q)) == list(np.argsort(nn))  # order == nn order, axis ignored


def test_aq_rank_compass_api_preserves_scores_and_returns_only_consumed_diagnostics():
    xs = np.linspace(0, 1, 6)
    rows = _rows(xs, precision=xs, completeness=[0.5] * 6, boundary_iou=[0.5] * 6)

    scores, compass = aq_module.compute_aq_rank_with_compass(rows)

    assert np.array_equal(scores, aq_module.compute_aq_rank_scores(rows))
    assert set(compass) == {"pool_size", "axes"}
    assert compass["pool_size"] == 6
    assert all(set(axis) == {"auroc", "sign"} for axis in compass["axes"].values())


def _write_rgb(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def _synthetic_images(n, size=224):
    """Return ``n`` distinct deterministic RGB uint8 images — hermetic (no network) for CI."""
    rng = np.random.default_rng(0)
    return [(rng.random((size, size, 3)) * 255).astype(np.uint8) for _ in range(n)]


@pytest.mark.gpu  # DINOv2 forward is ~30s/img on the CPU CI runner; run this integration check on GPU
@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_filter_generated_images_end_to_end(tmp_path):
    """Full model-backed path: score two synthetic-image samples with real DINOv2, then split.
    drop_ratio=0.5 over 2 samples keeps exactly 1 and copies its whole triple into keep/."""
    size = 224
    imgs = _synthetic_images(2, size)
    real_root, gen_root = tmp_path / "real", tmp_path / "gen"
    key, texture, defect = "SEM_IC+crack", "SEM_IC", "crack"
    mask = np.zeros((size, size), np.uint8)
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    for idx, arr in enumerate(imgs):
        _write_rgb(real_root / texture / "anomaly_image" / defect / f"r{idx}.png", arr)
        _write_mask(real_root / texture / "mask" / defect / f"r{idx}_mask.png", mask)
        name = f"{key}_{idx:05d}.png"
        _write_rgb(gen_root / "reconstructed_image" / name, arr)
        _write_mask(gen_root / "original_mask" / name, mask)
        _write_rgb(gen_root / "original_image" / name, arr)

    from anomalygen.eval.utils import load_generated, load_real

    generated = load_generated(str(gen_root), [key])
    real = load_real(str(real_root), [key])
    output_dir = tmp_path / "out"
    kpi, kept, dropped = filter_module.filter_generated_images(
        generated,
        real,
        str(gen_root),
        str(output_dir),
        drop_ratio=0.5,
        top_k=1,
    )

    assert len(kept) == 1 and len(dropped) == 1
    assert len(kpi[key]["per_sample"]) == 2
    for sub in _SUBDIRS:
        assert {p.name for p in (output_dir / "keep" / sub).iterdir()} == set(kept)


@pytest.mark.gpu
@pytest.mark.skipif(not os.path.exists(_DINOV2_WEIGHTS), reason="DINOv2 checkpoint not present")
def test_filter_main_end_to_end(tmp_path):
    """CLI end-to-end: split two synthetic samples and write both the score CSV and split manifest CSVs."""
    size = 224
    imgs = _synthetic_images(2, size)
    real_root, gen_root = tmp_path / "real", tmp_path / "gen"
    key, texture, defect = "SEM_IC+crack", "SEM_IC", "crack"
    mask = np.zeros((size, size), np.uint8)
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    recon_names = []
    for idx, arr in enumerate(imgs):
        _write_rgb(real_root / texture / "anomaly_image" / defect / f"r{idx}.png", arr)
        _write_mask(real_root / texture / "mask" / defect / f"r{idx}_mask.png", mask)
        name = f"{key}_{idx:05d}.png"
        _write_rgb(gen_root / "reconstructed_image" / name, arr)
        _write_mask(gen_root / "original_mask" / name, mask)
        _write_rgb(gen_root / "original_image" / name, arr)
        recon_names.append(name)
    pd.DataFrame([{"output_filename": str(gen_root / "reconstructed_image" / n)} for n in recon_names]).to_csv(
        gen_root / _GEN_CSV, index=False
    )

    output_dir = tmp_path / "out"
    filter_module.main(
        [
            "--gen_root",
            str(gen_root),
            "--real_root",
            str(real_root),
            "--output_dir",
            str(output_dir),
            "--anomaly_types",
            key,
            "--top_k",
            "1",
            "--drop_ratio",
            "0.5",
        ]
    )

    filtered = pd.read_csv(output_dir / _FILTERED_CSV)
    assert set(filtered["filename"]) == set(recon_names)  # per-sample scores for every sample
    assert len(list((output_dir / "keep" / "reconstructed_image").iterdir())) == 1
    assert len(list((output_dir / "drop" / "reconstructed_image").iterdir())) == 1
    assert (output_dir / "keep" / _GEN_CSV).exists() and (output_dir / "drop" / _GEN_CSV).exists()
