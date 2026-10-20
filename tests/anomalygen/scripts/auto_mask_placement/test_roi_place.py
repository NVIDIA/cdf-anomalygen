# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``roi_place``'s route selection.

``_get_sample_route`` is the one place where adding ``cad_tht`` could have changed an existing
route's behaviour, because a THT pair record carries ``cad_mask`` too — the semantic mask lives at
the same path — so the ``cad`` test would have claimed it. The fix is a guarded early return, and
what these tests lock is that it is *additive*: every cad / text / free input reaches exactly the
branch it reached before.

Also locked: an unrecognised route raises. The partition in ``main()`` used to end in a catch-all
``else: free_indices.append(i)``, so a new route name silently became whole-image placement —
plausible-looking masks in entirely the wrong place.
"""

import pytest

from anomalygen.scripts.auto_mask_placement import roi_place as P

CAD_MASK = {"cad_mask": "/ds/tex/cad_mask/x.png", "cad_mask_label": "/ds/labels.json"}


def _sample(defect_type="tex+anom", **over):
    s = {"defect_type": defect_type, "clean_image": "/ds/tex/clean_image/x.png", "submask": "/ds/m.png"}
    s.update(over)
    return s


# ===========================================================================
# the pre-existing routes are untouched
# ===========================================================================
@pytest.mark.parametrize(
    "spatial_dependency, extra, expected",
    [
        ("cad", CAD_MASK, "cad"),
        ("text", {}, "text"),
        ("free", {}, "free"),
        # A cad record is routed on the FIELD, not the value: that was true before cad_tht and stays
        # true, so a hand-written pair file carrying cad_mask still reaches cad2roi.
        ("text", CAD_MASK, "cad"),
        ("free", CAD_MASK, "cad"),
        (None, {}, "text"),  # absent spatial_dependency defaults to text
    ],
    ids=["cad", "text", "free", "text-with-cad-mask", "free-with-cad-mask", "missing-dep"],
)
def test_existing_routes_are_unchanged(spatial_dependency, extra, expected):
    deps = {"tex+anom": spatial_dependency} if spatial_dependency else {}
    assert P._get_sample_route(_sample(**extra), deps) == expected


def test_a_null_cad_mask_is_not_a_cad_record():
    """roi_pair writes ``cad_mask: null`` for every non-cad defect, so None must not route to cad."""
    assert P._get_sample_route(_sample(cad_mask=None, cad_mask_label=None), {"tex+anom": "free"}) == "free"


# ===========================================================================
# cad_tht
# ===========================================================================
def test_cad_tht_wins_over_the_cad_mask_field():
    """The whole reason the check is first: a THT record carries cad_mask and would become cad."""
    assert P._get_sample_route(_sample(**CAD_MASK), {"tex+anom": "cad_tht"}) == "cad_tht"


def test_cad_tht_is_selected_by_spatial_dependency_alone():
    assert P._get_sample_route(_sample(), {"tex+anom": "cad_tht"}) == "cad_tht"


def test_routing_is_per_defect_type_not_per_run():
    """A mixed spec must split, so one dataset can carry cad_tht and free defects at once."""
    deps = {"tex+tht": "cad_tht", "tex+other": "free"}
    assert P._get_sample_route(_sample("tex+tht", **CAD_MASK), deps) == "cad_tht"
    assert P._get_sample_route(_sample("tex+other"), deps) == "free"


# ===========================================================================
# an unhandled route must not fall through to free
# ===========================================================================
def test_an_unrecognised_route_raises(tmp_path, monkeypatch):
    """Guards the bucket table in main(): a route with no bucket used to become 'free' silently."""
    monkeypatch.setattr(P, "_get_sample_route", lambda sample, deps: "cad_tht_v2")

    pair = tmp_path / "pairs.json"
    pair.write_text('[{"defect_type": "tex+anom", "clean_image": "c.png", "submask": "m.png"}]')
    spec = tmp_path / "spec.jsonl"
    spec.write_text('{"defect_type": "tex+anom", "spatial_dependency": "free"}\n')

    with pytest.raises(ValueError, match="unhandled route 'cad_tht_v2'"):
        # fmt: off
        P.main([
            "--input_pair_path", str(pair), "--defect_desc", str(spec),
            "--output_dir", str(tmp_path / "out"),
        ])
        # fmt: on


# ===========================================================================
# the two bridge placers must stay distinguishable
# ===========================================================================
def test_tht_does_not_export_a_bare_bridge_mask_placer():
    """The property that makes the collision impossible rather than merely avoided.

    cad2roi exports ``BridgeMaskPlacer``. If tht2roi exported that name too, any module importing
    both would bind whichever import ran last, and the cad route would end up calling the THT placer
    -- a TypeError about argument counts, raised mid-batch, naming nothing useful. Aliasing at the
    import site fixes one call site; keeping the names distinct at the source fixes all of them,
    including ones not written yet.
    """
    import anomalygen.auto_mask_placement.tht2roi as tht

    assert not hasattr(tht, "BridgeMaskPlacer")
    assert hasattr(tht, "THTBridgeMaskPlacer")


def test_each_route_binds_its_own_bridge_placer():
    from anomalygen.auto_mask_placement.cad2roi import BridgeMaskPlacer as CadBridge
    from anomalygen.auto_mask_placement.tht2roi import THTBridgeMaskPlacer as ThtBridge

    assert CadBridge is not ThtBridge
    assert P.BridgeMaskPlacer is CadBridge
    assert P.THTBridgeMaskPlacer is ThtBridge


def test_the_two_bridge_placers_have_incompatible_signatures():
    """Why the names have to differ: swapping them is not a no-op, it is a TypeError mid-run."""
    import inspect

    from anomalygen.auto_mask_placement.cad2roi import BridgeMaskPlacer as CadBridge
    from anomalygen.auto_mask_placement.tht2roi import THTBridgeMaskPlacer as ThtBridge

    cad = list(inspect.signature(CadBridge.place_all).parameters)
    tht = list(inspect.signature(ThtBridge.place_all).parameters)
    assert cad != tht
    assert "groups" in cad and "groups" not in tht


# ===========================================================================
# the placement path itself
# ===========================================================================
# Everything above stops at routing. Nothing exercised what _process_cad_tht_sample DOES, and the
# guards added for the crash findings all return before any of it -- so the seed-file naming that
# _enumerate_amp_masks globs for, and therefore every row in testcase.jsonl, was resting on nothing.
# A rename of the suffix would have emptied the testcase for this route silently, with only the
# row-count gate noticing and its message blaming placement.
#
# The fixtures are test_tht2roi's, loaded by path rather than copied: the geometry they build is
# already agreed there, and a second copy of it would be a second thing to keep true.
import importlib.util
import json as _json
import pathlib

import numpy as np
from PIL import Image

_T2R = importlib.util.spec_from_file_location(
    "_t2r",
    pathlib.Path(__file__).resolve().parents[2] / "auto_mask_placement" / "test_tht2roi.py",
)
_t2r = importlib.util.module_from_spec(_T2R)
_T2R.loader.exec_module(_t2r)


def _tht_sample(tmp_path, defect="interrupted", joints=((40, 40, 30, 18, 8),), size=(80, 80)):
    """A pair record for one crop, with the clean image and the cad_mask the same size."""
    cad = _t2r._mask_png(tmp_path, list(joints), size=size)
    clean = tmp_path / "clean.png"
    Image.fromarray(np.full((size[1], size[0], 3), 120, np.uint8)).save(clean)
    return {
        "clean_image": str(clean),
        "defect_type": f"SolderPin+{defect}",
        "submask": _t2r._submask(tmp_path),
        "cad_mask": cad,
        "cad_mask_label": _t2r._labels(tmp_path),
    }


@pytest.mark.parametrize("defect", ["interrupted", "bridge"], ids=["ring", "bridge"])
@pytest.mark.parametrize("split_largest", [True, False], ids=["largest-component", "as-annotated"])
def test_a_placed_sample_names_its_seed_files_the_way_the_testcase_looks_them_up(tmp_path, defect, split_largest):
    """``<submask stem>__seed<i>.png`` is a contract between this function and _enumerate_amp_masks,
    and the two were never tested together.

    The stem is the PREPROCESSED submask's, so it carries ``_largest`` unless the record turns the
    largest-component split off -- the name the glob has to match is not the one in the pair record.
    Both dispatch arms, because bridge takes the JOINTS and the ring classes take the candidates: a
    different placer, the same naming.
    """
    sample = _tht_sample(tmp_path, defect)
    sample["submask_split_largest"] = split_largest
    out = tmp_path / "amp"

    result = P._process_cad_tht_sample(sample, out, n_seeds=2, base_seed=42)

    assert result["status"] == "OK"
    assert result["route"] == "tht2roi"
    assert sorted(result["seeds"]) == [0, 1]
    assert all(s["status"] in ("OK", "PARTIAL") for s in result["seeds"].values())

    stem = pathlib.Path(sample["submask"]).stem + ("_largest" if split_largest else "")
    written = sorted(p.name for p in (out / "clean" / sample["defect_type"]).glob("*.png"))
    assert written == [f"{stem}__seed0.png", f"{stem}__seed1.png"]

    # and the glob the testcase is built from finds exactly those, whichever stem they carry
    found = list(P._enumerate_amp_masks(out, sample["defect_type"]))
    assert [name for name, _ in found] == ["clean", "clean"]
    assert sorted(pathlib.Path(m).name for _, m in found) == written


def test_the_testcase_row_points_at_the_mask_that_was_placed(tmp_path):
    """The other end of the same contract: a row whose mask path does not exist generates nothing."""
    sample = _tht_sample(tmp_path)
    out = tmp_path / "amp"
    P._process_cad_tht_sample(sample, out, n_seeds=1, base_seed=42)

    jsonl = tmp_path / "testcase.jsonl"
    P._build_testcase_jsonl([sample], out, jsonl)
    rows = [_json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]

    assert len(rows) == 1
    assert rows[0]["image_filename"] == sample["clean_image"]
    assert pathlib.Path(rows[0]["mask_filename"]).is_file()


def test_roi_only_writes_the_overlays_and_places_nothing(tmp_path):
    """roi_place_pipeline documents --roi_only as the ROI cache warm-up, so the assets are the whole
    output and a seed file would mean it had generated work nobody asked for."""
    sample = _tht_sample(tmp_path)
    out = tmp_path / "amp"

    result = P._process_cad_tht_sample(sample, out, n_seeds=2, base_seed=42, roi_only=True)

    assert result["status"] == "ROI_ONLY"
    assert result["n_candidates"] >= 1
    assets = out / "clean" / sample["defect_type"] / "assets"
    assert (assets / "roi_cad.png").is_file() and (assets / "roi_real.png").is_file()
    assert not list((out / "clean" / sample["defect_type"]).glob("*__seed*.png"))


@pytest.mark.parametrize("submask", [None, "/does/not/exist.png"], ids=["absent", "missing-file"])
def test_roi_only_does_not_need_a_submask(tmp_path, submask):
    """--roi_only never opens one: it is the ROI cache warm-up and the overlays are its whole output.

    The guard added for the crash finding ran before the roi_only branch, so a pair file written for
    a warm-up -- which has no reason to name submasks at all -- produced nothing but skips."""
    sample = _tht_sample(tmp_path)
    sample["submask"] = submask
    out = tmp_path / "amp"

    result = P._process_cad_tht_sample(sample, out, n_seeds=1, base_seed=42, roi_only=True)

    assert result["status"] == "ROI_ONLY"
    assets = out / "clean" / sample["defect_type"] / "assets"
    assert (assets / "roi_cad.png").is_file() and (assets / "roi_real.png").is_file()


@pytest.mark.parametrize("submask", [None, "/does/not/exist.png"], ids=["absent", "missing-file"])
def test_a_placing_run_still_needs_one(tmp_path, submask):
    """The guard is not removed, only scoped: without --roi_only the submask is what gets placed."""
    sample = _tht_sample(tmp_path)
    sample["submask"] = submask
    result = P._process_cad_tht_sample(sample, tmp_path / "amp", n_seeds=1, base_seed=42)
    assert result["status"] == "NO_SUBMASK"


def test_a_cad_mask_that_is_not_the_clean_image_size_is_one_skipped_sample(tmp_path):
    """Everything below is built at the cad_mask's resolution and the overlays index the clean image
    with it, so the alternative is an opaque numpy IndexError mid-run."""
    sample = _tht_sample(tmp_path)
    Image.fromarray(np.full((40, 40, 3), 120, np.uint8)).save(sample["clean_image"])

    result = P._process_cad_tht_sample(sample, tmp_path / "amp", n_seeds=1, base_seed=42)
    assert result["status"] == "SIZE_MISMATCH"


def test_a_joint_with_no_region_for_the_class_is_one_skipped_sample(tmp_path):
    """dry_contact is region C only; a joint annotated with no hole ring has nowhere to put it."""
    sample = _tht_sample(tmp_path, "dry_contact", joints=((40, 40, 30, 8, 8),))

    result = P._process_cad_tht_sample(sample, tmp_path / "amp", n_seeds=1, base_seed=42)
    assert result["status"] in ("NO_CANDIDATES", "OK")  # geometry-dependent; the run must not raise


def test_n_instances_is_coerced_to_one(tmp_path):
    """-1 means "every ROI" elsewhere in AMP. Here one joint carries one defect: the crop is
    composited back against a single joint id, so a second defect would have nowhere to be recorded."""
    sample = _tht_sample(tmp_path)
    result = P._process_cad_tht_sample(sample, tmp_path / "amp", n_seeds=1, base_seed=42, n_instances=-1)

    assert result["status"] == "OK"
    assert result["seeds"][0]["n_requested"] == 1
