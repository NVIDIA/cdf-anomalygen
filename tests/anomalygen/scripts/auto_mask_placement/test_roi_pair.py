# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``roi_pair`` — (submask, clean) pairing + n_seeds.

The pairing / n_seeds derivation lives inside ``main()``, so it is driven
end-to-end through ``main(argv)`` over an on-disk fixture (plus the importable
``_clean_pool`` helper). The point is to lock the current,
correct behavior:

  * within pair budget (submasks × cleans) every (submask, clean) pair is unique,
  * at budget every combination is used exactly once,
  * above budget n_seeds scales as ⌈allocation / budget⌉ and record count as
    ⌈allocation / n_seeds⌉,
  * a fixed seed is deterministic, and the ``name`` field is ``<clean>__<submask>``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomalygen.scripts.auto_mask_placement import roi_pair as B


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_png(path, arr=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None:
        arr = np.zeros((8, 8), np.uint8)
        arr[2:6, 2:6] = 255
    Image.fromarray(arr).save(path)


def _amp_fixture(tmp_path, *, n_submasks, n_cleans, allocation, texture="metal", anomaly="scratch"):
    full = f"{texture}+{anomaly}"
    dataset_dir = tmp_path / "dataset"
    for i in range(n_submasks):
        _write_png(dataset_dir / texture / "mask" / anomaly / f"sub{i}.png")
    for i in range(n_cleans):
        _write_png(dataset_dir / texture / "clean_image" / f"clean{i}.png")
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text(json.dumps({"defect_type": full, "spatial_dependency": "free"}) + "\n")
    alloc = tmp_path / "allocation.json"
    alloc.write_text(json.dumps({full: allocation}))
    out = tmp_path / "amp_samples.json"
    return dataset_dir, spec, alloc, out, full


def _run_build(tmp_path, *, n_submasks, n_cleans, allocation, seed=0):
    dataset_dir, spec, alloc, out, full = _amp_fixture(
        tmp_path, n_submasks=n_submasks, n_cleans=n_cleans, allocation=allocation
    )
    B.main(
        [
            "--dataset_dir",
            str(dataset_dir),
            "--defect_desc",
            str(spec),
            "--allocation",
            str(alloc),
            "--output_pair_path",
            str(out),
            "--seed",
            str(seed),
        ]
    )
    records = json.loads(out.read_text())
    n_seeds = int(out.with_suffix(out.suffix + ".n_seeds").read_text())
    return records, n_seeds, full


# ===========================================================================
# pairing + n_seeds
# ===========================================================================
def test_pairs_unique_when_within_budget(tmp_path):
    # budget = 3 * 4 = 12; allocation 6 <= budget -> n_seeds 1, all pairs unique.
    records, n_seeds, _ = _run_build(tmp_path, n_submasks=3, n_cleans=4, allocation=6)
    assert n_seeds == 1
    assert len(records) == 6
    pairs = {(r["submask"], r["clean_image"]) for r in records}
    assert len(pairs) == len(records)  # no repeated (submask, clean) pair


def test_full_coverage_at_budget(tmp_path):
    # allocation == budget (12) -> every (submask, clean) combination used once.
    records, n_seeds, _ = _run_build(tmp_path, n_submasks=3, n_cleans=4, allocation=12)
    assert n_seeds == 1
    assert len(records) == 12
    submasks = {r["submask"] for r in records}
    cleans = {r["clean_image"] for r in records}
    pairs = {(r["submask"], r["clean_image"]) for r in records}
    assert len(submasks) == 3 and len(cleans) == 4
    assert pairs == {(s, c) for s in submasks for c in cleans}


def test_n_seeds_scales_above_budget(tmp_path):
    # budget = 2 * 2 = 4; allocation 10 -> n_seeds = ceil(10/4) = 3, records = ceil(10/3) = 4.
    records, n_seeds, _ = _run_build(tmp_path, n_submasks=2, n_cleans=2, allocation=10)
    assert n_seeds == 3
    assert len(records) == 4
    # 3 full records + a 1-seed remainder = 10 images exactly, not 4 x 3 = 12.
    assert sorted(r["n_seeds"] for r in records) == [1, 3, 3, 3]
    assert sum(r["n_seeds"] for r in records) == 10


def _run_multi(tmp_path, specs, seed=0):
    """specs: {anomaly: (n_submasks, n_cleans, allocation)} — one texture per anomaly, so each
    defect type gets its own independent (submask, clean) budget."""
    dataset_dir = tmp_path / "dataset"
    spec_lines, alloc = [], {}
    for anomaly, (n_sub, n_clean, count) in specs.items():
        texture = f"tex_{anomaly}"
        for i in range(n_sub):
            _write_png(dataset_dir / texture / "mask" / anomaly / f"sub{i}.png")
        for i in range(n_clean):
            _write_png(dataset_dir / texture / "clean_image" / f"clean{i}.png")
        full = f"{texture}+{anomaly}"
        spec_lines.append(json.dumps({"defect_type": full, "spatial_dependency": "free"}))
        alloc[full] = count
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text("\n".join(spec_lines) + "\n")
    alloc_path = tmp_path / "allocation.json"
    alloc_path.write_text(json.dumps(alloc))
    out = tmp_path / "amp_samples.json"
    # fmt: off
    B.main([
        "--dataset_dir", str(dataset_dir), "--defect_desc", str(spec),
        "--allocation", str(alloc_path), "--output_pair_path", str(out), "--seed", str(seed),
    ])
    # fmt: on
    records = json.loads(out.read_text())
    sidecar = int(out.with_suffix(out.suffix + ".n_seeds").read_text())
    return records, sidecar


def test_a_starved_defect_does_not_raise_n_seeds_for_the_others(tmp_path):
    """The pcb case: one type with a single clean image needs 2 placements per record, but a global
    n_seeds would apply that 2 to every type — and since each type emits ceil(alloc/n_seeds) records
    worth n_seeds images, the rounding pushes the run's total past num_sdg."""
    records, sidecar = _run_multi(
        tmp_path,
        {
            "starved": (8, 1, 9),  # budget 8 < 9 -> needs 2
            "roomy_a": (16, 4, 8),  # budget 64 -> needs 1
            "roomy_b": (62, 4, 8),  # budget 248 -> needs 1
        },
    )
    by_type = {}
    for r in records:
        by_type.setdefault(r["defect_type"], []).append(r["n_seeds"])

    # The starved type needs 2 placements per record; the roomy ones stay at 1 instead of being
    # dragged up to 2, so they spend their budget on 8 distinct pairs rather than 4 pairs placed twice.
    assert sorted(by_type["tex_starved+starved"]) == [1, 2, 2, 2, 2]  # last record takes the remainder
    assert set(by_type["tex_roomy_a+roomy_a"]) == {1}
    assert set(by_type["tex_roomy_b+roomy_b"]) == {1}
    assert len(by_type["tex_roomy_a+roomy_a"]) == 8, "one record per image, not 4 records x 2 seeds"

    # Every type lands on its request exactly — num_sdg is a target, not a floor.
    images = sum(r["n_seeds"] for r in records)
    assert images == 9 + 8 + 8
    assert sidecar == 2, "the sidecar still reports the max, as roi_place's scalar fallback"


def test_an_exactly_divisible_request_needs_no_rounding(tmp_path):
    """With every type inside its budget nothing is inflated and the total is exactly what was asked."""
    records, sidecar = _run_multi(tmp_path, {"a": (8, 2, 8), "b": (16, 4, 8), "c": (62, 4, 9)})
    assert sidecar == 1
    assert sum(r["n_seeds"] for r in records) == 25
    assert len(records) == 25


def test_deterministic_for_fixed_seed(tmp_path):
    r1, _, _ = _run_build(tmp_path / "a", n_submasks=3, n_cleans=4, allocation=8)
    r2, _, _ = _run_build(tmp_path / "b", n_submasks=3, n_cleans=4, allocation=8)
    assert [(Path(r["submask"]).name, Path(r["clean_image"]).name) for r in r1] == [
        (Path(r["submask"]).name, Path(r["clean_image"]).name) for r in r2
    ]


def test_name_field_format_and_free_has_no_cad(tmp_path):
    records, _, _ = _run_build(tmp_path, n_submasks=2, n_cleans=2, allocation=3)
    for r in records:
        assert r["name"] == f"{Path(r['clean_image']).stem}__{Path(r['submask']).stem}"
        assert r["cad_mask"] is None and r["cad_mask_label"] is None  # spatial_dependency == "free"


# ===========================================================================
# clean images are sourced from <dataset_dir>/<TEXTURE>/clean_image/
# ===========================================================================
def test_cleans_sourced_from_dataset_clean_image(tmp_path):
    # the fixture only creates cleans under <dataset_dir>/<texture>/clean_image/;
    # every paired clean must come from there (no separate clean root).
    records, _, _ = _run_build(tmp_path, n_submasks=2, n_cleans=2, allocation=4)
    assert len(records) == 4
    assert all("/clean_image/" in r["clean_image"] for r in records)


# ===========================================================================
# _clean_pool fallback resolution
# ===========================================================================
def test_clean_pool_prefers_nested_clean_image(tmp_path):
    tex = "metal"
    _write_png(tmp_path / tex / "clean_image" / "a.png")
    _write_png(tmp_path / tex / "stray.png")  # sibling, must be ignored when nested exists
    pool = B._clean_pool(tmp_path, tex)
    assert [p.name for p in pool] == ["a.png"]


def test_clean_pool_falls_back_to_texture_dir(tmp_path):
    tex = "metal"
    _write_png(tmp_path / tex / "a.png")
    _write_png(tmp_path / tex / "b.png")
    pool = B._clean_pool(tmp_path, tex)
    assert {p.name for p in pool} == {"a.png", "b.png"}


def test_clean_pool_is_every_extension_case_insensitively_in_name_order(tmp_path):
    """The listing AMP pairs against: png/jpg/jpeg/bmp by lower-cased suffix, sorted by name, a
    non-image left out. Pinned so the shared lister cannot move a single clean or reorder the pool."""
    d = tmp_path / "metal" / "clean_image"
    for name in ("b.png", "a.jpeg", "C.JPG", "d.BMP"):
        _write_png(d / name)
    (d / "notes.txt").write_text("not an image")
    assert [p.name for p in B._clean_pool(tmp_path, "metal")] == ["C.JPG", "a.jpeg", "b.png", "d.BMP"]


# ===========================================================================
# --per_clean_pairing
# ===========================================================================
# The round-robin above maximises (submask, clean) diversity and does NOT cover every clean image:
# clean_idx = (i % S + i // S) % C depends only on the SUM of the two indices, so B rounds over S
# submasks reach B + S - 1 distinct cleans, not B * S. When each clean image is a work item rather
# than a backdrop — a joint crop that must be composited back onto a board — the required invariant is
# the other one, and the shortfall is invisible because the record count is still right.
def _run_pairing(tmp_path, *, n_submasks, n_cleans, allocation, mode, sd="cad_tht", seed=0, texture="SolderPin"):
    anomaly = "interrupted"
    full = f"{texture}+{anomaly}"
    ds = tmp_path / "dataset"
    for i in range(n_submasks):
        _write_png(ds / texture / "mask" / anomaly / f"sub{i}.png")
    for i in range(n_cleans):
        _write_png(ds / texture / "clean_image" / f"clean{i}.png")
        _write_png(ds / texture / "cad_mask" / f"clean{i}.png")
    (ds / "semantic_segmentation_labels.json").write_text("{}")
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text(json.dumps({"defect_type": full, "spatial_dependency": sd}) + "\n")
    alloc = tmp_path / "allocation.json"
    alloc.write_text(json.dumps({full: allocation}))
    out = tmp_path / "amp_samples.json"
    # fmt: off
    B.main([
        "--dataset_dir", str(ds), "--defect_desc", str(spec), "--allocation", str(alloc),
        "--output_pair_path", str(out), "--seed", str(seed), "--per_clean_pairing", mode,
    ])
    # fmt: on
    return json.loads(out.read_text()), full


def test_on_covers_every_clean(tmp_path):
    """The invariant a composite-back-onto-a-board flow needs: every clean gets exactly one record."""
    records, _ = _run_pairing(tmp_path, n_submasks=5, n_cleans=40, allocation=40, mode="on")
    assert len(records) == 40
    assert len({r["clean_image"] for r in records}) == 40
    assert all(r["n_seeds"] == 1 for r in records)


def test_the_default_is_the_diversity_pairing_for_every_route(tmp_path):
    """The flag decides; the route does not.

    An earlier design inferred per-clean pairing from ``spatial_dependency == "cad_tht"``, which made
    one route behave differently by default. Pairing answers what the clean images ARE -- backdrops or
    work items -- which no defect's placement route can know, so both of these take the round-robin.
    """
    for sd in ("cad_tht", "free"):
        records, _ = _run_pairing(tmp_path / sd, n_submasks=5, n_cleans=40, allocation=40, mode="off", sd=sd)
        assert len(records) == 40
        # ceil(alloc/S) + S - 1 = 8 + 4 = 12 distinct cleans out of 40.
        assert len({r["clean_image"] for r in records}) == 12, f"{sd} did not take the default pairing"


def test_per_clean_uses_each_clean_exactly_once(tmp_path):
    records, _ = _run_pairing(tmp_path, n_submasks=3, n_cleans=10, allocation=10, mode="on")
    cleans = [r["clean_image"] for r in records]
    assert len(cleans) == len(set(cleans)) == 10


def test_per_clean_still_rotates_the_submask_bank(tmp_path):
    """Submasks may repeat across joints — they are shape templates, not work items."""
    records, _ = _run_pairing(tmp_path, n_submasks=3, n_cleans=9, allocation=9, mode="on")
    assert len({r["submask"] for r in records}) == 3


def test_per_clean_rejects_more_than_one_defect_type(tmp_path):
    """One record per clean means one defect per clean; two types would both claim the same images."""
    ds = tmp_path / "dataset"
    spec_lines, alloc = [], {}
    for anomaly in ("interrupted", "trapped"):
        for i in range(2):
            _write_png(ds / "SolderPin" / "mask" / anomaly / f"sub{i}.png")
        alloc[f"SolderPin+{anomaly}"] = 2
        spec_lines.append(json.dumps({"defect_type": f"SolderPin+{anomaly}", "spatial_dependency": "cad_tht"}))
    for i in range(4):
        _write_png(ds / "SolderPin" / "clean_image" / f"clean{i}.png")
    spec = tmp_path / "defect_spec.jsonl"
    spec.write_text("\n".join(spec_lines) + "\n")
    alloc_path = tmp_path / "allocation.json"
    alloc_path.write_text(json.dumps(alloc))

    with pytest.raises(SystemExit, match="exactly one defect type"):
        # fmt: off
        B.main([
            "--dataset_dir", str(ds), "--defect_desc", str(spec), "--allocation", str(alloc_path),
            "--output_pair_path", str(tmp_path / "out.json"), "--per_clean_pairing", "on",
        ])
        # fmt: on


def test_per_clean_refuses_an_allocation_larger_than_the_clean_pool(tmp_path):
    """Unsatisfiable: 12 records cannot each take a distinct clean out of a pool of 8."""
    with pytest.raises(SystemExit, match="only 8 clean image"):
        _run_pairing(tmp_path, n_submasks=3, n_cleans=8, allocation=12, mode="on")


def test_per_clean_warns_when_the_allocation_leaves_cleans_uncovered(tmp_path, loguru_lines):
    """A deliberate subset is legitimate, but under-coverage must never be silent again."""
    records, _ = _run_pairing(tmp_path, n_submasks=2, n_cleans=10, allocation=4, mode="on")
    assert len(records) == 4
    assert len({r["clean_image"] for r in records}) == 4
    assert any("6 will get" in line for line in loguru_lines)


def test_off_is_the_round_robin_on_a_cad_tht_dataset(tmp_path):
    """A THT validation set wants diversity, not per-joint coverage."""
    records, _ = _run_pairing(tmp_path, n_submasks=5, n_cleans=40, allocation=40, mode="off")
    assert len(records) == 40
    assert len({r["clean_image"] for r in records}) < 40


def test_cad_tht_records_carry_the_cad_fields(tmp_path):
    records, _ = _run_pairing(tmp_path, n_submasks=2, n_cleans=4, allocation=4, mode="on")
    for r in records:
        assert r["cad_mask"] is not None and r["cad_mask_label"] is not None
        assert Path(r["cad_mask"]).stem == Path(r["clean_image"]).stem
