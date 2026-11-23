# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared SDG-output loaders in eval.utils (CPU-only, no model)."""

import numpy as np
import pytest
from PIL import Image

from anomalygen.eval.utils import (
    infer_anomaly_types,
    load_generated,
    load_real,
    resolve_anomaly_types,
)


def _write_rgb(path, size=(8, 8), color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_mask(path, size=(8, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[2:5, 2:5] = 255
    Image.fromarray(arr, mode="L").save(path)


def test_infer_anomaly_types_strips_trailing_index(tmp_path):
    recon = tmp_path / "reconstructed_image"
    for name in ("SEM_IC+crack_00001.png", "SEM_IC+crack_00002.png", "OPT+scratch_00007.png"):
        _write_rgb(recon / name)
    assert infer_anomaly_types(str(recon)) == ["OPT+scratch", "SEM_IC+crack"]


def test_resolve_prefers_explicit_over_recipe_and_infer(tmp_path):
    recon = tmp_path / "reconstructed_image"
    _write_rgb(recon / "A+b_00001.png")
    assert resolve_anomaly_types(["X+y"], None, str(recon)) == ["X+y"]


def test_resolve_reads_anomaly_types_from_recipe(tmp_path):
    recon = tmp_path / "reconstructed_image"
    _write_rgb(recon / "A+b_00001.png")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("anomaly_types:\n  - [SEM_IC, crack]\n  - [OPT, scratch]\n")
    assert resolve_anomaly_types(None, str(recipe), str(recon)) == ["SEM_IC+crack", "OPT+scratch"]


def test_resolve_falls_back_to_inference(tmp_path):
    recon = tmp_path / "reconstructed_image"
    _write_rgb(recon / "A+b_00001.png")
    assert resolve_anomaly_types(None, None, str(recon)) == ["A+b"]


def test_resolve_rejects_anomaly_type_without_plus(tmp_path):
    with pytest.raises(ValueError):
        resolve_anomaly_types(["missing_plus"], None, str(tmp_path))


@pytest.mark.parametrize("bad", ["../escape+x", "wood+../../escape", "/tmp/escape+x", "a/b+c"])
def test_resolve_rejects_a_path_bearing_anomaly_type(tmp_path, bad):
    """load_real joins texture and defect straight onto real_root, so a path-bearing key would read
    references from outside the dataset and score them as real."""
    with pytest.raises(ValueError, match="anomaly type"):
        resolve_anomaly_types([bad], None, str(tmp_path))


def test_load_generated_pairs_recon_with_mask_and_skips_unmatched(tmp_path):
    gen_root = tmp_path / "gen"
    _write_rgb(gen_root / "reconstructed_image" / "K+d_00001.png")
    _write_rgb(gen_root / "reconstructed_image" / "K+d_00002.png")  # no matching mask -> skipped
    _write_mask(gen_root / "original_mask" / "K+d_00001.png")

    out = load_generated(str(gen_root), ["K+d"])

    assert list(out.keys()) == ["K+d"]
    assert len(out["K+d"]["reconstructed_image"]) == 1
    assert len(out["K+d"]["original_mask"]) == 1
    assert [p.split("/")[-1] for p in out["K+d"]["img_path"]] == ["K+d_00001.png"]


def test_load_generated_does_not_mix_types_sharing_a_prefix(tmp_path):
    gen_root = tmp_path / "gen"
    for name in ("K+d_00001.png", "K+d_extra_00002.png", "K+d_extra_00003.png"):
        _write_rgb(gen_root / "reconstructed_image" / name)
        _write_mask(gen_root / "original_mask" / name)

    out = load_generated(str(gen_root), ["K+d", "K+d_extra"])

    assert [p.split("/")[-1] for p in out["K+d"]["img_path"]] == ["K+d_00001.png"]
    assert len(out["K+d_extra"]["img_path"]) == 2


def test_load_generated_missing_recon_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_generated(str(tmp_path / "nonexistent"), ["K+d"])


def test_load_generated_resizes_to_target_size(tmp_path):
    gen_root = tmp_path / "gen"
    _write_rgb(gen_root / "reconstructed_image" / "K+d_00001.png", size=(20, 30))  # non-square
    _write_mask(gen_root / "original_mask" / "K+d_00001.png", size=(20, 30))

    out = load_generated(str(gen_root), ["K+d"], target_size=16)

    assert out["K+d"]["reconstructed_image"][0].shape == (16, 16, 3)
    assert out["K+d"]["original_mask"][0].shape == (16, 16)
    # a resized mask is re-binarized to {0, 1}
    assert set(np.unique(out["K+d"]["original_mask"][0]).tolist()).issubset({0.0, 1.0})


def test_load_real_pairs_image_and_mask(tmp_path):
    real_root = tmp_path / "real"
    _write_rgb(real_root / "TEX" / "anomaly_image" / "def" / "s0.png")
    _write_mask(real_root / "TEX" / "mask" / "def" / "s0_mask.png")

    out = load_real(str(real_root), ["TEX+def"])

    assert len(out["TEX+def"]["original_image"]) == 1
    assert len(out["TEX+def"]["original_mask"]) == 1


def test_load_generated_raises_on_mask_image_size_mismatch(tmp_path):
    gen_root = tmp_path / "gen"
    _write_rgb(gen_root / "reconstructed_image" / "K+d_00001.png", size=(16, 16))
    _write_mask(gen_root / "original_mask" / "K+d_00001.png", size=(8, 8))  # mismatched
    with pytest.raises(ValueError):
        load_generated(str(gen_root), ["K+d"])


def test_load_real_raises_on_mask_image_size_mismatch(tmp_path):
    real_root = tmp_path / "real"
    _write_rgb(real_root / "TEX" / "anomaly_image" / "def" / "s0.png", size=(16, 16))
    _write_mask(real_root / "TEX" / "mask" / "def" / "s0_mask.png", size=(8, 8))  # mismatched
    with pytest.raises(ValueError):
        load_real(str(real_root), ["TEX+def"])


def test_load_generated_omits_type_with_no_masked_pairs(tmp_path):
    gen_root = tmp_path / "gen"
    _write_rgb(gen_root / "reconstructed_image" / "K+d_00001.png")  # recon exists, no mask -> no pairs
    assert load_generated(str(gen_root), ["K+d"]) == {}


def test_load_real_skips_image_missing_its_mask(tmp_path):
    real_root = tmp_path / "real"
    _write_rgb(real_root / "TEX" / "anomaly_image" / "def" / "s0.png")
    _write_mask(real_root / "TEX" / "mask" / "def" / "s0_mask.png")
    _write_rgb(real_root / "TEX" / "anomaly_image" / "def" / "s1.png")  # no mask for s1 -> skipped

    out = load_real(str(real_root), ["TEX+def"])
    assert len(out["TEX+def"]["original_image"]) == 1


def test_load_real_resizes_to_target_size(tmp_path):
    real_root = tmp_path / "real"
    _write_rgb(real_root / "TEX" / "anomaly_image" / "def" / "s0.png", size=(20, 30))
    _write_mask(real_root / "TEX" / "mask" / "def" / "s0_mask.png", size=(20, 30))

    out = load_real(str(real_root), ["TEX+def"], target_size=16)

    assert out["TEX+def"]["original_image"][0].shape == (16, 16, 3)
    assert out["TEX+def"]["original_mask"][0].shape == (16, 16)


def test_load_real_raises_when_no_references(tmp_path):
    with pytest.raises(RuntimeError):
        load_real(str(tmp_path / "real"), ["TEX+def"])
