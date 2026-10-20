# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``roi_allocate`` — the SDG sample-planning stage.

Locks the pure, deterministic planning logic that has no other guard:

  * uniform / proportional (largest-remainder) allocation,
  * the validation-mode "≥1 per defect" KPI floor,
  * the ``--per_defect_counts`` override + guard rails,
  * the migrated preflight (``_validate_amp_inputs``) that cross-checks the
    (dataset_dir, clean_dir, defect_spec) triple before allocating.

All logic is exercised through the module's internal functions (white-box) — the
point is to lock the current, correct behavior, not to change it.
"""

import json

import numpy as np
import pytest
from PIL import Image

from anomalygen.scripts.auto_mask_placement import roi_allocate as A


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_png(path, arr=None):
    """Write a small valid PNG (content is irrelevant — only existence matters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None:
        arr = np.zeros((8, 8), np.uint8)
        arr[2:6, 2:6] = 255
    Image.fromarray(arr).save(path)


def _dataset(
    tmp_path,
    *,
    texture="Phone",
    anomaly="scratch",
    n_masks=3,
    n_cleans=2,
    spatial_dependency="free",
    prompt=None,
    with_clean=True,
):
    """A single-defect dataset + its one-line defect_spec JSONL."""
    ds = tmp_path / "ds"
    for i in range(n_masks):
        _write_png(ds / texture / "mask" / anomaly / f"m{i}.png")
    if with_clean:
        for i in range(n_cleans):
            _write_png(ds / texture / "clean_image" / f"c{i}.png")
    entry = {"defect_type": f"{texture}+{anomaly}", "spatial_dependency": spatial_dependency}
    if prompt is not None:
        entry["roi_prompt_defect_location"] = prompt
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text(json.dumps(entry) + "\n")
    return ds, spec


# ===========================================================================
# uniform allocation
# ===========================================================================
def test_uniform_sum_preserved_and_balanced():
    defects = ["a", "b", "c"]
    assert A._uniform(10, defects) == {"a": 4, "b": 3, "c": 3}
    assert A._uniform(9, defects) == {"a": 3, "b": 3, "c": 3}
    # invariants across a range of totals: sum preserved, spread <= 1, the
    # first (num_sdg % N) defects get the +1.
    for num_sdg in range(0, 40):
        for defs in (["a"], ["a", "b"], ["a", "b", "c", "d"]):
            alloc = A._uniform(num_sdg, defs)
            counts = list(alloc.values())
            assert sum(counts) == num_sdg
            assert max(counts) - min(counts) <= 1
            rem = num_sdg % len(defs)
            for d in defs[:rem]:
                assert alloc[d] == num_sdg // len(defs) + 1


def test_allocate_inference_delegates_to_uniform():
    defects = ["x+a", "y+b", "z+c"]
    assert A._allocate(7, defects, {}, mode="inference") == A._uniform(7, defects)


# ===========================================================================
# proportional (largest-remainder) allocation
# ===========================================================================
def test_proportional_largest_remainder_known_case():
    # raw: a=2.5 (floor 2, rem .5), b=7.5 (floor 7, rem .5); remainder 1 goes to
    # the first tied type in stable order -> a. Total is preserved.
    assert A._proportional(10, ["a", "b"], {"a": 1, "b": 3}) == {"a": 3, "b": 7}


def test_proportional_sum_preserved_and_monotone():
    defects = ["a", "b", "c"]
    counts = {"a": 1, "b": 2, "c": 7}
    for num_sdg in (1, 5, 10, 13, 50, 101):
        alloc = A._proportional(num_sdg, defects, counts)
        assert sum(alloc.values()) == num_sdg
        # a larger mask count never gets a smaller allocation
        assert alloc["a"] <= alloc["b"] <= alloc["c"]


def test_proportional_raises_when_no_masks():
    with pytest.raises(ValueError, match="mask counts must be > 0"):
        A._proportional(10, ["a", "b"], {"a": 0, "b": 0})


# ===========================================================================
# validation-mode KPI floor
# ===========================================================================
def test_validation_kpi_floor_raises_on_zero_coverage():
    # b rounds to 0 with num_sdg=2 (raw b = 2*1/101 ≈ 0.02) -> KPI floor trips.
    with pytest.raises(ValueError) as ei:
        A._allocate(2, ["a", "b"], {"a": 100, "b": 1}, mode="validation")
    msg = str(ei.value)
    assert "coverage broken" in msg
    assert "'b'" in msg  # names the starved defect
    assert "Increase num_sdg" in msg


def test_validation_ok_when_all_covered():
    alloc = A._allocate(100, ["a", "b"], {"a": 100, "b": 1}, mode="validation")
    assert sum(alloc.values()) == 100
    assert all(v >= 1 for v in alloc.values())


# ===========================================================================
# --per_defect_counts override + guard rails
# ===========================================================================
def test_override_projects_onto_defect_types():
    # unlisted defect -> 0; unknown key ignored (warning logged).
    alloc = A._allocate(
        15,
        ["a", "b", "c"],
        {},
        mode="inference",
        override={"a": 5, "b": 10, "zzz": 99},
    )
    assert alloc == {"a": 5, "b": 10, "c": 0}


def test_override_rejected_in_validation_mode():
    with pytest.raises(ValueError, match="only valid with --mode inference"):
        A._allocate(10, ["a"], {"a": 5}, mode="validation", override={"a": 10})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(num_sdg=-1, defect_types=["a"], counts={}), "must be >= 0"),
        (dict(num_sdg=5, defect_types=[], counts={}), "non-empty"),
        (dict(num_sdg=5, defect_types=["a"], counts={}, mode="bogus"), "unknown --mode"),
    ],
)
def test_allocate_guard_rails(kwargs, match):
    with pytest.raises(ValueError, match=match):
        A._allocate(**kwargs)


# ===========================================================================
# _defect_types_from_spec + _count_masks
# ===========================================================================
def test_defect_types_from_spec_ordered_and_deduped(tmp_path):
    spec = tmp_path / "spec.jsonl"
    spec.write_text(
        json.dumps({"defect_type": "Phone+oil"})
        + "\n"
        + json.dumps({"defect_type": "Phone+scratch"})
        + "\n"
        + json.dumps({"defect_type": "Phone+oil"})
        + "\n"  # duplicate
        + "\n"  # blank line ignored
    )
    assert A._defect_types_from_spec(spec) == ["Phone+oil", "Phone+scratch"]


def test_count_masks_counts_images_and_zero_for_missing(tmp_path):
    ds, _ = _dataset(tmp_path, texture="Phone", anomaly="scratch", n_masks=3)
    counts = A._count_masks(ds, ["Phone+scratch", "Phone+missing"])
    assert counts == {"Phone+scratch": 3, "Phone+missing": 0}


def test_count_masks_counts_every_extension_case_insensitively(tmp_path):
    """The submask count that weights the proportional split: png/jpg/jpeg/bmp by lower-cased suffix,
    a non-image left out. Pinned so the shared lister cannot change a single count."""
    d = tmp_path / "ds" / "Phone" / "mask" / "scratch"
    for name in ("m0.png", "m1.JPG", "m2.jpeg", "m3.BMP"):
        _write_png(d / name)
    (d / "notes.txt").write_text("not an image")
    assert A._count_masks(tmp_path / "ds", ["Phone+scratch"]) == {"Phone+scratch": 4}


# ===========================================================================
# _validate_amp_inputs preflight
# ===========================================================================
def test_validate_ok_for_free(tmp_path):
    ds, spec = _dataset(tmp_path, spatial_dependency="free")
    assert A._validate_amp_inputs(ds, spec) == []


def test_validate_ok_for_text_with_prompt(tmp_path):
    ds, spec = _dataset(tmp_path, spatial_dependency="text", prompt="the screen surface")
    assert A._validate_amp_inputs(ds, spec) == []


def test_validate_text_requires_prompt(tmp_path):
    ds, spec = _dataset(tmp_path, spatial_dependency="text", prompt="   ")  # blank
    errors = A._validate_amp_inputs(ds, spec)
    assert any("roi_prompt_defect_location" in e for e in errors)


@pytest.mark.parametrize("bad_type", ["../escape+x", "/tmp/escape+x", "Phone+../../escape"])
def test_validate_rejects_a_path_bearing_defect_type(tmp_path, bad_type):
    """defect_type reaches path joins here and a directory name downstream, so it is rejected at the
    dataset spec rather than several stages later."""
    ds, spec = _dataset(tmp_path)
    entry = json.loads(spec.read_text().strip())
    entry["defect_type"] = bad_type
    spec.write_text(json.dumps(entry) + "\n")
    errors = A._validate_amp_inputs(ds, spec)
    assert any("anomaly type" in e for e in errors), errors


def test_validate_missing_submask_dir(tmp_path):
    ds, spec = _dataset(tmp_path, n_masks=0)  # no mask/<anomaly>/ dir written
    errors = A._validate_amp_inputs(ds, spec)
    assert any("missing submask source" in e for e in errors)


def test_validate_no_clean_images(tmp_path):
    ds, spec = _dataset(tmp_path, with_clean=False)
    errors = A._validate_amp_inputs(ds, spec)
    assert any("no clean images" in e for e in errors)


def test_validate_bad_defect_type(tmp_path):
    spec = tmp_path / "spec.jsonl"
    spec.write_text(json.dumps({"defect_type": "NoPlusSign", "spatial_dependency": "free"}) + "\n")
    errors = A._validate_amp_inputs(tmp_path, spec)
    # The error names the offending value and the shape expected of it.
    assert any("NoPlusSign" in e and "{texture}+{defect}" in e for e in errors), errors


def test_validate_unknown_spatial_dependency(tmp_path):
    ds, spec = _dataset(tmp_path, spatial_dependency="bogus")
    errors = A._validate_amp_inputs(ds, spec)
    assert any("unknown spatial_dependency" in e for e in errors)


def test_validate_cad_checks(tmp_path):
    # cad defect with masks + cleans but NO labels file, NO cad_mask/ dir.
    texture, anomaly = "PCB", "bridge"
    ds = tmp_path / "ds"
    for i in range(2):
        _write_png(ds / texture / "mask" / anomaly / f"m{i}.png")
    for i in range(2):
        _write_png(ds / texture / "clean_image" / f"c{i}.png")
    spec = tmp_path / "spec.jsonl"
    spec.write_text(json.dumps({"defect_type": f"{texture}+{anomaly}", "spatial_dependency": "cad"}) + "\n")

    errors = A._validate_amp_inputs(ds, spec)
    joined = "\n".join(errors)
    assert "semantic_segmentation_labels.json" in joined  # missing labels
    assert "cad_mask" in joined  # missing cad_mask dir

    # add labels + a cad_mask dir covering only one of the two cleans -> per-clean gap
    (ds / "semantic_segmentation_labels.json").write_text("{}")
    _write_png(ds / texture / "cad_mask" / "c0.png")
    errors2 = A._validate_amp_inputs(ds, spec)
    assert any("without matching cad_mask" in e for e in errors2)


# ===========================================================================
# cad_tht preflight
# ===========================================================================
# Each check here exists because the failure it catches is otherwise silent at placement time:
# a `cad` label file parses cleanly and yields zero joints; a misspelled defect finds no ROI rule;
# and a clean image sized differently from its cad_mask gets resampled against a mask built at the
# other resolution.
THT_PALETTE = {
    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
    "(0, 0, 0, 255)": {"class": "UNLABELLED"},
    "(240, 4, 111, 255)": {"class": "hole"},
    "(27, 186, 239, 255)": {"class": "pad"},
    "(220, 179, 62, 255)": {"class": "pin"},
}
# The newer annotation: the lead is split into pin_tip (its face) and pin (the exposed side wall).
TIP_PALETTE = {
    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
    "(0, 0, 0, 255)": {"class": "UNLABELLED"},
    "(0, 0, 255, 255)": {"class": "hole"},
    "(0, 255, 0, 255)": {"class": "pad"},
    "(0, 255, 255, 255)": {"class": "pin_tip"},
    "(255, 0, 0, 255)": {"class": "pin"},
}
# Nothing requires a board to expose a side wall, so pin_tip alone has to be enough.
TIP_ONLY_PALETTE = {k: v for k, v in TIP_PALETTE.items() if v["class"] != "pin"}

CAD_PALETTE = {
    "(0, 0, 0, 255)": {"class": "UNLABELLED"},
    "(35, 198, 233, 255)": {"class": "pad"},
    "(93, 220, 11, 255)": {"class": "capacitor"},
    "(243, 69, 141, 255)": {"class": "solder"},
}


def _tht_dataset(tmp_path, *, anomaly="interrupted", palette=None, cad_size=None, clean_size=(8, 8)):
    """A minimal cad_tht dataset; ``cad_size`` defaults to matching the clean images."""
    texture = "SolderPin"
    ds = tmp_path / "ds"
    _write_png(ds / texture / "mask" / anomaly / "m0.png")
    for i in range(2):
        _write_png(ds / texture / "clean_image" / f"c{i}.png", np.zeros(clean_size[::-1], np.uint8))
        _write_png(ds / texture / "cad_mask" / f"c{i}.png", np.zeros((cad_size or clean_size)[::-1], np.uint8))
    (ds / "semantic_segmentation_labels.json").write_text(json.dumps(palette or THT_PALETTE))
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text(json.dumps({"defect_type": f"{texture}+{anomaly}", "spatial_dependency": "cad_tht"}) + "\n")
    return ds, spec


def test_cad_tht_dataset_passes_preflight(tmp_path):
    ds, spec = _tht_dataset(tmp_path)
    assert A._validate_amp_inputs(ds, spec) == []


def test_cad_tht_needs_the_pad_hole_pin_classes(tmp_path):
    """A cad_tht run against a `cad` label file parses fine and finds no joints at all."""
    ds, spec = _tht_dataset(tmp_path, palette=CAD_PALETTE)
    errors = A._validate_amp_inputs(ds, spec)
    assert len(errors) == 1
    assert "hole" in errors[0] and "pin" in errors[0] and "needs the class" in errors[0]


@pytest.mark.parametrize("palette", [TIP_PALETTE, TIP_ONLY_PALETTE], ids=["tip-and-pin", "tip-only"])
def test_cad_tht_accepts_either_spelling_of_the_lead(tmp_path, palette):
    """D is the lead face, which newer annotations name pin_tip and older ones name pin. Requiring
    both would reject the very boards this route was extended for."""
    ds, spec = _tht_dataset(tmp_path, palette=palette)
    assert A._validate_amp_inputs(ds, spec) == []


def test_cad_tht_rejects_a_palette_with_no_lead_class_at_all(tmp_path):
    """pad + hole alone leaves D empty, so `hidden` would place into nothing."""
    ds, spec = _tht_dataset(tmp_path, palette={k: v for k, v in THT_PALETTE.items() if v["class"] != "pin"})
    errors = A._validate_amp_inputs(ds, spec)
    assert len(errors) == 1
    assert "pin or pin_tip" in errors[0]


def test_cad_tht_rejects_a_defect_outside_the_published_enum(tmp_path):
    ds, spec = _tht_dataset(tmp_path, anomaly="not_a_tht_defect")
    errors = A._validate_amp_inputs(ds, spec)
    assert any("unknown cad_tht defect" in e for e in errors)


def test_cad_tht_rejects_a_cad_mask_of_a_different_size(tmp_path):
    """Everything downstream is built at the cad_mask's resolution."""
    ds, spec = _tht_dataset(tmp_path, clean_size=(8, 8), cad_size=(12, 12))
    errors = A._validate_amp_inputs(ds, spec)
    assert any("different size" in e for e in errors)


def test_cad_tht_still_requires_the_shared_cad_inputs(tmp_path):
    """cad_tht reuses the cad contract, so the labels file and a per-clean cad_mask are required."""
    ds, spec = _tht_dataset(tmp_path)
    (ds / "semantic_segmentation_labels.json").unlink()
    assert any("semantic_segmentation_labels.json" in e for e in A._validate_amp_inputs(ds, spec))

    ds2, spec2 = _tht_dataset(tmp_path / "b")
    (ds2 / "SolderPin" / "cad_mask" / "c1.png").unlink()
    assert any("without matching cad_mask" in e for e in A._validate_amp_inputs(ds2, spec2))


def test_the_cad_route_gains_no_new_errors(tmp_path):
    """The size and enum checks are cad_tht-only; `cad`'s accepted inputs must not change."""
    texture, anomaly = "PCB", "bridge"
    ds = tmp_path / "ds"
    _write_png(ds / texture / "mask" / anomaly / "m0.png")
    for i in range(2):
        _write_png(ds / texture / "clean_image" / f"c{i}.png", np.zeros((8, 8), np.uint8))
        _write_png(ds / texture / "cad_mask" / f"c{i}.png", np.zeros((12, 12), np.uint8))  # different size
    (ds / "semantic_segmentation_labels.json").write_text(json.dumps(CAD_PALETTE))  # no hole/pin
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text(json.dumps({"defect_type": f"{texture}+{anomaly}", "spatial_dependency": "cad"}) + "\n")
    assert A._validate_amp_inputs(ds, spec) == []


def test_cad_tht_is_an_accepted_spatial_dependency(tmp_path):
    """It used to be rejected as unknown — that rejection was the whole baseline for this work."""
    ds, spec = _tht_dataset(tmp_path)
    assert not any("unknown spatial_dependency" in e for e in A._validate_amp_inputs(ds, spec))
