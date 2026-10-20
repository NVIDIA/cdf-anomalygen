# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tht2roi`` — the geometry layer behind ``spatial_dependency: cad_tht``.

Fixtures are synthetic: concentric circles painted in the delivered palette. Nothing here reads a
real dataset, because the THT corpora are customer data and never reach CI.

What is locked, and why each one is otherwise silent:

  * the three annotated classes ARE regions B / C / D, and the pad disc is their union — the whole
    premise of the module,
  * ``roi_inner`` is measured against that disc, not against the ``pad`` class, which is already a
    ring and would give a radius roughly a third too small (so the cut would barely bite),
  * a joint the crop edge clipped yields NO candidate — it keeps a large slice of its pad, so it
    would otherwise pass any area floor while its "B+C" was really just "B",
  * a component belonging to no pad is reported rather than dropped in silence,
  * the two ROI tables are checked at import, where a typo'd key reads back as "no inner cut".
"""

import json
import re
import types

import cv2
import numpy as np
import pytest
from PIL import Image

from anomalygen.auto_mask_placement import tht2roi
from anomalygen.auto_mask_placement.tht2roi import regions as R
from anomalygen.auto_mask_placement.tht2roi import spec as S
from anomalygen.auto_mask_placement.tht2roi.defects import ring as RING

# The delivered palette (full_board_roi/.../semantic_segmentation_labels_*.json).
PALETTE = {
    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
    "(0, 0, 0, 255)": {"class": "UNLABELLED"},
    "(240, 4, 111, 255)": {"class": "hole"},
    "(27, 186, 239, 255)": {"class": "pad"},
    "(220, 179, 62, 255)": {"class": "pin"},
    "(0, 255, 255, 255)": {"class": "pin_tip"},
}
RGBA = {
    "pad": (27, 186, 239, 255),
    "hole": (240, 4, 111, 255),
    "pin": (220, 179, 62, 255),
    "pin_tip": (0, 255, 255, 255),
}
UNLABELLED = (0, 0, 0, 255)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _labels(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "semantic_segmentation_labels.json"
    p.write_text(json.dumps(PALETTE))
    return str(p)


def _disc(shape, cx, cy, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return np.hypot(yy - cy, xx - cx) <= r


def _paint_joint(canvas, cx, cy, r_pad, r_hole, r_pin, r_tip=None):
    """Paint one joint as DISJOINT rings, the way the delivered annotation carries them.

    ``r_tip`` splits the lead the way the newer annotation does: ``pin_tip`` is the face in the
    middle and ``pin`` the sliver of side wall around it. Omitted, the lead is one ``pin`` disc —
    every dataset built before the split.
    """
    shape = canvas.shape[:2]
    pad_d, hole_d, pin_d = (_disc(shape, cx, cy, r) for r in (r_pad, r_hole, r_pin))
    canvas[pad_d & ~hole_d] = RGBA["pad"]
    canvas[hole_d & ~pin_d] = RGBA["hole"]
    canvas[pin_d] = RGBA["pin"]
    if r_tip is not None:
        canvas[_disc(shape, cx, cy, r_tip)] = RGBA["pin_tip"]
    return canvas


def _mask_png(tmp_path, joints, size=(80, 80), name="cad.png"):
    """``joints``: list of (cx, cy, r_pad, r_hole, r_pin[, r_tip]). Returns the written path."""
    canvas = np.zeros((size[1], size[0], 4), np.uint8)
    canvas[:] = UNLABELLED
    for j in joints:
        _paint_joint(canvas, *j)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    Image.fromarray(canvas, "RGBA").save(p)
    return str(p)


def _one_joint(tmp_path, **over):
    """A single centred joint: r_pad 30, r_hole 18, r_pin 8 on an 80x80 canvas."""
    geom = {"cx": 40, "cy": 40, "r_pad": 30, "r_hole": 18, "r_pin": 8}
    geom.update(over)
    path = _mask_png(tmp_path, [(geom["cx"], geom["cy"], geom["r_pad"], geom["r_hole"], geom["r_pin"])])
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, shape = gen.parse_joints(path)
    return gen, joints, shape, geom


# ===========================================================================
# which class supplies D
# ===========================================================================
# Newer annotations split the lead: pin_tip is its face seen from above, pin the sliver of side wall
# a tilted lead exposes. On the reference board every joint carries a pin_tip and only 12 of 184 also
# carry a pin, so D is the pin_tip and the sliver belongs to the hole ring it sits inside.
def _lead_joint(tmp_path, r_tip=None, name="cad.png"):
    path = _mask_png(tmp_path, [(40, 40, 30, 18, 8, r_tip)], name=name)
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, _ = gen.parse_joints(path)
    return joints


def test_d_is_the_pin_tip_when_the_annotation_splits_the_lead(tmp_path):
    (joint,) = _lead_joint(tmp_path, r_tip=4)
    tip = _disc((80, 80), 40, 40, 4)
    assert joint.D.any()
    assert np.array_equal(joint.D, joint.D & tip), "D must be the tip, not the whole lead"


def test_the_side_wall_sliver_folds_into_the_hole_ring(tmp_path):
    """It is annotated lead, but it is not the face D is meant to be -- and leaving it in no region
    would punch a hole in the coverage of a joint that is otherwise perfectly annotated."""
    (split,) = _lead_joint(tmp_path, r_tip=4)
    (whole,) = _lead_joint(tmp_path, r_tip=None, name="whole.png")
    sliver = _disc((80, 80), 40, 40, 8) & ~_disc((80, 80), 40, 40, 4)
    assert (split.C & sliver).sum() == sliver.sum(), "every sliver pixel belongs to C"
    assert split.C.sum() == whole.C.sum() + sliver.sum()
    assert not (split.D & sliver).any()


def test_d_falls_back_to_pin_when_the_annotation_does_not_split_the_lead(tmp_path):
    """Every dataset built before the split parses exactly as it did."""
    (joint,) = _lead_joint(tmp_path, r_tip=None)
    assert np.array_equal(joint.D, _disc((80, 80), 40, 40, 8))
    assert joint.missing("BCD") == []


def test_the_choice_is_made_per_joint_not_per_mask(tmp_path):
    """A board where only some leads are tilted enough to expose a side wall must not force every
    joint onto one reading of the annotation."""
    path = _mask_png(tmp_path, [(40, 40, 30, 18, 8, 4), (120, 40, 30, 18, 8, None)], size=(160, 80))
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, _ = gen.parse_joints(path)
    by_x = sorted(joints, key=lambda j: j.centre[0])
    assert len(by_x) == 2
    assert by_x[0].D.sum() < by_x[1].D.sum(), "the split joint's D is the tip; the other's is the pin"
    assert np.array_equal(by_x[0].D, by_x[0].D & _disc((80, 160), 40, 40, 4))
    assert np.array_equal(by_x[1].D, _disc((80, 160), 120, 40, 8))


def test_the_regions_stay_disjoint_and_the_disc_is_unchanged_by_the_split(tmp_path):
    """The disc is the filled pad contour, so it does not depend on how the lead is annotated --
    which is what keeps radius, centre and the roi_inner cuts identical across the two forms."""
    (split,) = _lead_joint(tmp_path, r_tip=4)
    (whole,) = _lead_joint(tmp_path, r_tip=None, name="whole.png")
    assert np.array_equal(split.disc, whole.disc)
    assert split.radius == whole.radius and split.centre == whole.centre
    assert not (split.B & split.C).any() and not (split.C & split.D).any() and not (split.B & split.D).any()


# ===========================================================================
# the classes are the regions
# ===========================================================================
def test_the_three_classes_decode_to_disjoint_regions(tmp_path):
    _, joints, _, g = _one_joint(tmp_path)
    assert len(joints) == 1
    j = joints[0]
    assert not (j.B & j.C).any() and not (j.C & j.D).any() and not (j.B & j.D).any()
    # Each region is the annulus between two radii, to within rasterisation.
    for region, (r_out, r_in) in [(j.B, (g["r_pad"], g["r_hole"])), (j.C, (g["r_hole"], g["r_pin"]))]:
        expected = np.pi * (r_out**2 - r_in**2)
        assert region.sum() == pytest.approx(expected, rel=0.05)
    assert j.D.sum() == pytest.approx(np.pi * g["r_pin"] ** 2, rel=0.05)


def test_disc_is_the_union_of_the_three_regions(tmp_path):
    _, joints, _, _ = _one_joint(tmp_path)
    j = joints[0]
    assert np.array_equal(j.disc, j.B | j.C | j.D)


def test_disc_survives_a_gap_in_the_hole_ring(tmp_path):
    """``disc`` is filled from the pad's OUTER contour, so a hole in the annotation cannot leak it.

    The union would; that is the whole reason ``_fill`` exists rather than ``B | C | D``.
    """
    _, joints, _, _ = _one_joint(tmp_path)
    j = joints[0]
    holed = j.B.copy()
    holed[38:42, 38:42] = False  # punch the middle out of the pad ring's interior
    assert np.array_equal(R._fill(holed), R._fill(j.B))


def test_centre_and_radius_come_from_the_disc_not_the_pad_ring(tmp_path):
    """``roi_inner`` is stated in PAD radii; the ``pad`` class is a ring and would be far too small.

    Uses a hole/pad ratio near the delivered geometry (~0.73). The ``pad`` ring's own equivalent
    radius is then ``r_pad * sqrt(1 - 0.73**2)`` ~= 0.68 r_pad, so a cut computed against it would
    reach only two-thirds as far as intended.
    """
    _, joints, _, g = _one_joint(tmp_path, r_pad=30, r_hole=22, r_pin=8)
    j = joints[0]
    assert j.centre == pytest.approx((g["cx"], g["cy"]), abs=1.0)
    assert j.radius == pytest.approx(g["r_pad"], rel=0.05)
    ring_only_radius = np.sqrt(j.B.sum() / np.pi)
    assert ring_only_radius < 0.75 * j.radius


def test_centre_is_x_then_y(tmp_path):
    """Ordering matches CADParser and ROICandidate; a swap would shift the cut diagonally."""
    _, joints, _, _ = _one_joint(tmp_path, cx=50, cy=25, r_pad=20, r_hole=12, r_pin=5)
    assert joints[0].centre == pytest.approx((50, 25), abs=1.0)


# ===========================================================================
# roi_for
# ===========================================================================
def test_roi_for_unions_the_letters(tmp_path):
    _, joints, _, _ = _one_joint(tmp_path)
    j = joints[0]
    assert np.array_equal(R.roi_for(j, "B") > 127, j.B)
    assert np.array_equal(R.roi_for(j, "BC") > 127, j.B | j.C)
    assert np.array_equal(R.roi_for(j, "BCD") > 127, j.disc)


def test_roi_for_returns_a_binary_uint8_mask(tmp_path):
    _, joints, _, _ = _one_joint(tmp_path)
    roi = R.roi_for(joints[0], "BC")
    assert roi.dtype == np.uint8 and set(np.unique(roi)) <= {0, 255}


def test_inner_cut_removes_everything_within_k_pad_radii(tmp_path):
    _, joints, _, _ = _one_joint(tmp_path)
    j = joints[0]
    cut = R.roi_for(j, "BC", 0.5) > 127
    ys, xs = np.nonzero(cut)
    cx, cy = j.centre
    assert np.hypot(ys - cy, xs - cx).min() >= 0.5 * j.radius - 1.0
    assert cut.sum() < (R.roi_for(j, "BC") > 127).sum()


def test_inner_cut_pushes_the_inner_boundary_outward(tmp_path):
    """What the cut buys: legal positions start further from the pad centre.

    Note it is NOT that the ROI's own bbox centre becomes illegal — ``B+C`` already excludes the pin,
    so that centre pixel was never in the region. The problem the cut solves is that the region
    reaches almost to the centre, so a submask anchored there still lands mostly on it.
    """
    _, joints, _, _ = _one_joint(tmp_path)
    j = joints[0]
    cx, cy = j.centre

    def inner_radius(mask):
        ys, xs = np.nonzero(mask)
        return np.hypot(ys - cy, xs - cx).min()

    uncut = inner_radius(R.roi_for(j, "BC") > 127)
    cut = inner_radius(R.roi_for(j, "BC", 0.5) > 127)
    assert uncut == pytest.approx(8, abs=1.5)  # the pin's edge
    assert cut == pytest.approx(0.5 * j.radius, abs=1.5)
    assert cut > uncut


def test_inner_cut_of_zero_is_a_no_op(tmp_path):
    _, joints, _, _ = _one_joint(tmp_path)
    j = joints[0]
    assert np.array_equal(R.roi_for(j, "BC", 0.0), R.roi_for(j, "BC"))


# ===========================================================================
# joint grouping
# ===========================================================================
def test_two_intact_joints_yield_two_candidates(tmp_path):
    """Why joints and not crops: a neighbour in frame must not merge into one region."""
    path = _mask_png(tmp_path, [(25, 40, 18, 11, 5), (75, 40, 18, 11, 5)], size=(100, 80))
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, _ = gen.parse_joints(path)
    assert len(joints) == 2
    assert len(gen.generate_candidates(joints, "interrupted")) == 2
    assert not (joints[0].disc & joints[1].disc).any()


def test_hole_and_pin_are_assigned_to_their_own_pad(tmp_path):
    path = _mask_png(tmp_path, [(25, 40, 18, 11, 5), (75, 40, 18, 11, 5)], size=(100, 80))
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, _ = gen.parse_joints(path)
    for j in joints:
        assert j.C.any() and j.D.any()
        assert (j.C | j.D).sum() == (j.disc & ~j.B).sum()  # nothing borrowed from the other joint


def test_a_component_with_no_pad_is_reported_not_dropped_silently(tmp_path, loguru_lines):
    """A stray hole (its pad clipped away entirely) must leave a trace in the log."""
    canvas = np.zeros((80, 80, 4), np.uint8)
    canvas[:] = UNLABELLED
    _paint_joint(canvas, 40, 40, 25, 15, 6)
    # Radius 10, not 4: CADParser drops any component under fragment_ratio (0.1) of the largest of
    # its class, and the joint's own hole ring is ~594 px, so a smaller orphan never reaches the
    # grouping code at all.
    canvas[_disc((80, 80), 12, 12, 10)] = RGBA["hole"]  # orphan: no pad anywhere near it
    p = tmp_path / "orphan.png"
    Image.fromarray(canvas, "RGBA").save(p)

    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, _ = gen.parse_joints(str(p))
    assert len(joints) == 1
    assert any("outside every pad disc" in line for line in loguru_lines)


# ===========================================================================
# the clipped-joint failure this module exists to prevent
# ===========================================================================
def _clipped_neighbour(tmp_path):
    """One intact joint plus a neighbour whose pad is cut by the frame and whose hole is outside."""
    canvas = np.zeros((80, 80, 4), np.uint8)
    canvas[:] = UNLABELLED
    _paint_joint(canvas, 30, 40, 22, 13, 6)
    shape = (80, 80)
    pad_d = _disc(shape, 95, 40, 22)  # centre well outside the frame -> only an arc is visible
    canvas[pad_d] = RGBA["pad"]
    p = tmp_path / "clipped.png"
    Image.fromarray(canvas, "RGBA").save(p)
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    return gen, *gen.parse_joints(str(p))


def test_a_clipped_joint_is_parsed_but_carries_no_hole(tmp_path):
    _, joints, _ = _clipped_neighbour(tmp_path)
    assert len(joints) == 2
    clipped = [j for j in joints if not j.C.any()]
    assert len(clipped) == 1
    assert clipped[0].B.sum() > RING.MIN_ROI_PX  # large enough to pass any area floor


def _pad_clipped_hole_intact(tmp_path, cx=8, cy=40, r_pad=30):
    """A joint the crop edge cuts through the PAD of, leaving the hole and lead whole.

    The existing clipped-joint tests cover the cut that takes C with it, which joints_carrying then
    drops. This one keeps every region, so nothing drops it -- and it is the geometry the crop edge
    produces most often, since the pad is the outermost ring.
    """
    canvas = np.zeros((80, 80, 4), np.uint8)
    canvas[:] = UNLABELLED
    _paint_joint(canvas, cx, cy, r_pad, 6, 3)
    path = tmp_path / "clipped_pad.png"
    tmp_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGBA").save(path)
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, shape = gen.parse_joints(str(path))
    return gen, joints, shape


def test_a_clipped_pad_does_not_move_the_joint_centre_or_shrink_its_radius(tmp_path):
    """Both came from the truncated arc: centroid and sqrt(area/pi).

    Measured before the fix, on a pad of true centre (8, 40) and radius 30 clipped at the left edge:
    centre (16.3, 40.0) and radius 24.7 -- so roi_for's inner cut was displaced 8 px and 18% too
    small, and for a rule with inner_k the ring meant to be removed was neither centred nor
    half-radius. The joint is NOT dropped, because every region survives the cut.
    """
    _, joints, _ = _pad_clipped_hole_intact(tmp_path)
    assert len(joints) == 1
    joint = joints[0]
    assert joint.B.any() and joint.C.any() and joint.D.any(), "this fixture must keep every region"
    assert joint.centre == pytest.approx((8.0, 40.0), abs=1.0)
    assert joint.radius == pytest.approx(30.0, abs=1.5)


def test_an_intact_joint_is_measured_exactly_as_before(tmp_path):
    """The clipped path must not move a placement that was never wrong -- so the area formula stays
    for every joint the crop edge did not touch, which is nearly all of them."""
    _, joints, shape, geom = _one_joint(tmp_path)
    joint = joints[0]
    assert joint.centre == pytest.approx((float(geom["cx"]), float(geom["cy"])), abs=0.5)
    assert joint.radius == pytest.approx(np.sqrt(joint.disc.sum() / np.pi), abs=1e-6)


def test_the_annulus_test_asks_the_point_amp_actually_centres_on(tmp_path):
    """AMP aligns on the ROI's BOUNDING-BOX centre; this asked the pixel centroid.

    They agree for a symmetric ring and part company for anything else. The fixture is a crescent:
    its centroid sits on ROI pixels, so the ROI read as solid, the mask was centred onto the hole,
    and the entire draw sweep was spent before the retention fallback rescued it.
    """
    # An eccentric annulus: outer circle centred, hole pushed right. The ring is then thick on the
    # left, which drags the centroid onto ring pixels, while the bounding box -- fixed by the outer
    # circle -- keeps its centre inside the hole. A partly clipped joint gives the same shape.
    yy, xx = np.ogrid[:80, :80]
    roi = (np.hypot(yy - 40, xx - 40) <= 30) & (np.hypot(yy - 40, xx - 55) > 18)

    ys, xs = np.nonzero(roi)
    centroid = (int(round(ys.mean())), int(round(xs.mean())))
    bbox = (int((ys.min() + ys.max() + 1) // 2), int((xs.min() + xs.max() + 1) // 2))
    assert roi[centroid] != roi[bbox], "fixture no longer separates the two points"
    assert RING._is_annulus(roi) is (not roi[bbox])


@pytest.mark.parametrize("defect", ["bright", "interrupted", "less_solder", "trapped", "hidden", "dry_contact"])
def test_a_clipped_joint_yields_no_candidate(tmp_path, defect):
    """Every ring rule names C, so a joint without one is not a placement target for any of them."""
    gen, joints, _ = _clipped_neighbour(tmp_path)
    assert len(gen.generate_candidates(joints, defect)) == 1


def test_the_skipped_clipped_joint_is_logged(tmp_path, loguru_lines):
    gen, joints, _ = _clipped_neighbour(tmp_path)
    gen.generate_candidates(joints, "interrupted")
    assert any("has no C region" in line for line in loguru_lines)


def test_missing_names_only_the_absent_letters(tmp_path):
    _, joints, _ = _clipped_neighbour(tmp_path)
    clipped = next(j for j in joints if not j.C.any())
    intact = next(j for j in joints if j.C.any())
    assert clipped.missing("BC") == ["C"]
    assert clipped.missing("BCD") == ["C", "D"]
    assert clipped.missing("B") == []
    assert intact.missing("BCD") == []


def test_a_roi_below_the_area_floor_is_skipped(tmp_path):
    """A ROI too small to hold a defect is dropped rather than forced.

    The floor is driven from the ROI's measured area rather than from a hand-picked geometry, so the
    boundary is exercised exactly and the test does not depend on how a small circle rasterises.
    """
    _, joints, _, _ = _one_joint(tmp_path)
    area = int((R.roi_for(joints[0], "C") > 127).sum())
    assert tht2roi.get_ring_candidates(joints, "dry_contact", min_roi_px=area + 1) == []
    assert len(tht2roi.get_ring_candidates(joints, "dry_contact", min_roi_px=area)) == 1


# ===========================================================================
# candidates
# ===========================================================================
def test_candidate_carries_the_roi_and_its_letters(tmp_path):
    _, joints, _, _ = _one_joint(tmp_path)
    cand = tht2roi.get_ring_candidates(joints, "bright")[0]
    letters, inner = S.roi_rule("bright")
    assert cand.source_classes == list(letters)
    assert cand.area == int((R.roi_for(joints[0], letters, inner) > 127).sum())
    assert np.array_equal(cand.mask, R.roi_for(joints[0], letters, inner))


def test_candidate_centroid_is_x_then_y(tmp_path):
    """Matches CADParser/ROICandidate; ring.py measures its outward tie-break against this."""
    _, joints, _, _ = _one_joint(tmp_path, cx=55, cy=30, r_pad=20, r_hole=12, r_pin=5)
    cand = tht2roi.get_ring_candidates(joints, "hidden")[0]
    assert cand.centroid == pytest.approx((55, 30), abs=1.5)


def test_bridge_is_not_a_ring_class(tmp_path):
    """bridge spans pad -> crop edge, so it is fitted rather than confined; it has its own module.

    It still yields a candidate, but only as the ROI overlay every THT class writes — see
    ``test_bridge_candidates_are_an_overlay_not_a_confinement``.
    """
    assert S.BRIDGE not in S.RING_CLASSES and S.BRIDGE in S.THT_CLASSES
    assert set(S.RING_CLASSES) | {S.BRIDGE} == set(S.THT_CLASSES)


# ===========================================================================
# the ROI tables
# ===========================================================================
def test_roi_rule_returns_letters_and_inner_cut():
    assert S.roi_rule("dry_contact") == ("C", 0.0)
    letters, inner = S.roi_rule("interrupted")
    assert set(letters) <= S.VALID_LETTERS and inner > 0


def test_roi_rule_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="unknown THT defect"):
        S.roi_rule("not_a_defect")


def test_every_shipped_class_has_a_usable_rule():
    for cls in S.THT_CLASSES:
        letters, inner = S.roi_rule(cls)
        assert letters and set(letters) <= S.VALID_LETTERS
        assert 0.0 <= inner < 1.0


@pytest.mark.parametrize(
    "roi_spec, roi_inner, match",
    [
        ({"bright": "BX"}, {}, "unknown region letter"),
        ({"bright": ""}, {}, "is empty"),
        ({"bright": "BC"}, {"brigth": 0.5}, "unknown class"),
        ({"bright": "BC"}, {"bright": 1.0}, r"\[0.0, 1.0\)"),
    ],
    ids=["unknown-letter", "empty-letters", "typo-in-inner-key", "inner-cut-too-large"],
)
def test_the_tables_are_checked_at_import(monkeypatch, roi_spec, roi_inner, match):
    """Each of these is silent otherwise — the typo'd key in particular reads back as 'no cut'."""
    monkeypatch.setattr(S, "THT_ROI_SPEC", roi_spec)
    monkeypatch.setattr(S, "THT_ROI_INNER", roi_inner)
    with pytest.raises(ValueError, match=match):
        S._validate_tables()


# ===========================================================================
# the placer (drives AMP — ~0.03s per placement)
# ===========================================================================
def _submask(tmp_path, size=80, blob=(30, 46, 12, 26)):
    """A defect-shaped blob on its own canvas, the way a base mask arrives from the bank."""
    m = np.zeros((size, size), np.uint8)
    y0, y1, x0, x1 = blob
    m[y0:y1, x0:x1] = 255
    p = tmp_path / "submask.png"
    Image.fromarray(m).save(p)
    return str(p)


def test_a_placement_never_leaves_its_roi(tmp_path):
    """AMP clips to the ROI, so this is the region rules actually being enforced end to end."""
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    results = RING.RingMaskPlacer().place_all(_submask(tmp_path), cands, shape, n_instances=1, n_seeds=2)

    assert len(results) == 2
    roi = cands[0].mask > 127
    for _, n_placed, mask, _small in results.values():
        assert n_placed == 1 and mask is not None
        assert (mask > 127).any()
        assert not ((mask > 127) & ~roi).any()


def test_different_seeds_give_different_placements(tmp_path):
    """Every (seed, ROI) pair owns a disjoint block of draw seeds, so seeds do not collapse."""
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    results = RING.RingMaskPlacer().place_all(_submask(tmp_path), cands, shape, n_instances=1, n_seeds=4)
    rendered = {m.tobytes() for _, _, m, _small in results.values() if m is not None}
    assert len(rendered) == 4


def test_n_instances_caps_at_the_number_of_joints(tmp_path):
    """-1 never reaches the placer; an over-large explicit count clamps rather than raising."""
    path = _mask_png(tmp_path, [(25, 40, 18, 11, 5), (75, 40, 18, 11, 5)], size=(100, 80))
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, shape = gen.parse_joints(path)
    cands = gen.generate_candidates(joints, "interrupted")
    results = RING.RingMaskPlacer().place_all(_submask(tmp_path), cands, shape, n_instances=9, n_seeds=1)
    n_requested, _, _, _ = results[1]
    assert n_requested == 2


def test_an_annulus_roi_keeps_most_of_the_annotation(tmp_path):
    """The regression this placer exists for: a centred mask on an annulus lands on the hole.

    ``interrupted`` is a "BC" rule with a 0.50 inner cut, so its ROI is a ring whose centroid is not
    on the ROI at all. AMP centre-aligns, so centring parks the submask over the hole and the clip
    takes most of it -- measured against the reference batch, 46% of the annotation survived where
    the reference kept 70%. Retention matters twice over: it is the conditioning signal, and the mask
    bbox is what sizes the generation crop, so a shrunken mask also moves the model off its trained
    scale.

    The bound is deliberately well under what is achieved (measured ~0.70-0.76 on real joints) so
    this fails on a regression to centring rather than on ordinary draw-to-draw variation.
    """
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    roi = cands[0].mask > 127
    assert not roi[int(round(np.nonzero(roi)[0].mean())), int(round(np.nonzero(roi)[1].mean()))], (
        "fixture is not an annulus, so it cannot exercise the case this guards"
    )

    submask = _submask(tmp_path)
    target = RING._annotation_area(submask, shape[1], shape[0])
    mask, note = RING.RingMaskPlacer().place_on_roi(submask, cands[0], shape, seed=1)

    assert mask is not None, note
    kept = int((mask > 127).sum())
    assert kept >= 0.55 * target, f"kept {kept}/{target}px ({kept / target:.0%}) -- {note}"


def test_the_kept_figure_is_a_share_of_the_draw_and_cannot_exceed_it(tmp_path):
    """A retention that can read 106% is not measuring retention.

    AMP was given a scale RANGE, so a draw carried 1.0-1.44x the annotation's pixels before the clip
    while the denominator stayed the raw annotation area. Measured on this fixture, by scale:

        scale   pre-clip   kept   printed   true
        1.0        224      186     83%      83%
        1.1        271      230    103%      85%
        1.2        323      238    106%      74%

    The largest-survivor rule therefore picked the largest SCALE -- the worst-retained draw of the
    three -- and its inflated figure tripped the early stop first, so the remaining draws were never
    made. The note reports a share of the draw now, which is bounded by construction.
    """
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    submask = _submask(tmp_path)

    for seed in range(1, 6):
        mask, note = RING.RingMaskPlacer().place_on_roi(submask, cands[0], shape, seed=seed)
        assert mask is not None, note
        kept, drawn = (int(v) for v in re.search(r"kept (\d+)/(\d+)px", note).groups())
        assert kept <= drawn, f"kept more of the draw than the draw held: {note}"
        assert f"{kept / drawn:.0%}" in note


def test_the_draw_that_is_selected_is_the_best_kept_one(monkeypatch):
    """The selection rule itself, on synthetic draws, because the real ones cannot separate it.

    Written after the fixture tests above failed to notice the rule being reverted: they assert the
    reported figure and the shape of the conflict, and both survive a `max(area)` selection intact.
    What had to be pinned is which draw comes back, so this hands _best_draw three known areas at
    three known scales and names the winner.

        scale   drawn   kept   retention
        1.0      100      80      80%     <- best kept, smallest
        1.1      121      90      74%
        1.2      144     110      76%     <- biggest, and what max(area) would have taken
    """
    placer = RING.RingMaskPlacer()
    target = 100
    areas = {1.0: 80, 1.1: 90, 1.2: 110}

    def fake_draw(roi_path, submask_path, shift, w, h, seed, work, scale=1.0):
        m = np.zeros((10, 20), np.uint8)
        m.flat[: areas[scale]] = 255
        return m

    monkeypatch.setattr(placer, "_draw", fake_draw)
    mask, kept, drawn = placer._best_draw(None, None, 0, 20, 10, 0, None, target)

    assert (kept, drawn) == (80, 100), "selected the biggest draw rather than the best-kept one"
    assert int((mask > 127).sum()) == 80


def test_the_early_stop_is_measured_against_the_draw_too(monkeypatch):
    """Otherwise a scaled draw trips it first and the remaining draws are never made -- so the loop
    ends on whichever draw happened to be inflated enough, not on a good one."""
    placer = RING.RingMaskPlacer()
    target = 100
    seen = []

    def fake_draw(roi_path, submask_path, shift, w, h, seed, work, scale=1.0):
        seen.append(scale)
        m = np.zeros((10, 20), np.uint8)
        # 96 px: 96% of an unscaled draw, but only 67% of a 1.2 one. Against `target` the very first
        # scaled draw would clear TARGET_FRAC and stop; against the draw, only the 1.0 one does.
        m.flat[:96] = 255
        return m

    monkeypatch.setattr(placer, "_draw", fake_draw)
    placer._best_draw(None, None, 0, 20, 10, 0, None, target)
    assert seen[-1] == 1.0, f"stopped on a scaled draw that kept {96 / (100 * 1.2 * 1.2):.0%} of itself"


def test_the_biggest_draw_is_not_the_best_kept_one(tmp_path, monkeypatch):
    """Why the denominator matters, as a property rather than as arithmetic.

    Absolute survivors rise with scale; the share that survives does not. On this fixture, with the
    selection fixed:

        scale   drawn   kept   retention
        1.0      224    175      78%
        1.1      271    230      85%
        1.2      323    262      81%

    So ranking draws by pixels picks 1.2 and ranking them by retention picks 1.1 -- the two
    orderings disagree, which is the whole of the finding. The old code ranked by pixels against a
    denominator that never moved, so it picked the largest scale every time and called it "kept
    238/224px".
    """
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    submask = _submask(tmp_path)
    target = RING._annotation_area(submask, shape[1], shape[0])

    by_scale = {}
    for scale in RING.SCALES:
        # The sweep is what _draw reads, so narrowing it to one value is how a single scale is
        # measured. Overriding the placer's own aug does nothing now, which is the point of the fix:
        # the scale of a draw is this module's to know rather than AMP's to sample.
        monkeypatch.setattr(RING, "SCALES", (scale,))
        kept = []
        for seed in range(1, 9):
            mask, _note = RING.RingMaskPlacer().place_on_roi(submask, cands[0], shape, seed=seed)
            if mask is not None:
                kept.append(int((mask > 127).sum()))
        by_scale[scale] = sum(kept) / len(kept)

    retention = {s: k / (target * s * s) for s, k in by_scale.items()}
    assert max(by_scale, key=by_scale.get) == max(RING.SCALES), "pixels no longer rise with scale"
    assert max(retention, key=retention.get) != max(by_scale, key=by_scale.get), (
        "the fixture no longer shows the conflict the denominator exists to resolve"
    )


def test_a_placement_that_keeps_almost_nothing_is_labelled_small(tmp_path):
    """free and text flag SMALL below min_area_ratio and cad2roi's LessSolderMaskPlacer rejects
    below half the ROI; cad_tht had no floor, so a placement keeping 2% of its draw was recorded OK
    and inpainted as a full defect on the strength of one log line."""
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "dry_contact")  # region C only: a thin ring
    # A submask far larger than the ring can hold, so most of every draw is clipped away.
    big = tmp_path / "big.png"
    m = np.zeros((80, 80), np.uint8)
    m[5:75, 5:75] = 255
    Image.fromarray(m).save(big)

    mask, note = RING.RingMaskPlacer().place_on_roi(str(big), cands[0], shape, seed=1)
    if mask is None:
        pytest.skip(f"nothing landed at all on this fixture ({note}), which is a different outcome")
    kept, drawn = (int(v) for v in re.search(r"kept (\d+)/(\d+)px", note).groups())
    assert kept / drawn < RING.MIN_RETENTION, "fixture no longer produces an undersized placement"
    assert note.startswith("SMALL"), note


def test_an_undersized_placement_says_how_undersized(tmp_path, loguru_lines, monkeypatch):
    """summary.json records SMALL; the arithmetic behind it was being dropped.

    place_all logged a note only when the placement returned None, and an undersized one returns a
    mask -- so a real run reported four SMALL seeds and nothing anywhere said whether they kept 49%
    or 2%. Those want different answers: one is a threshold to argue about, the other is a joint the
    geometry cannot serve.
    """
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    placer = RING.RingMaskPlacer()

    # A placement that lands but keeps far too little. Built FROM the ROI: a few pixels anywhere
    # else would trip the escape check, which is a different failure and would hide this one.
    def undersized(submask_path, roi_candidate, img_shape, seed=42):
        mask = np.zeros(img_shape[:2], np.uint8)
        ys, xs = np.nonzero(roi_candidate.mask > 127)
        mask[ys[:4], xs[:4]] = 255
        # The label place_on_roi puts in front when the retention floor is missed; place_all keys
        # off it, so a fixture without it tests nothing.
        return mask, "SMALL, kept 4/1673px (0% of the draw)"

    monkeypatch.setattr(placer, "place_on_roi", undersized)
    results = placer.place_all(_submask(tmp_path), cands, shape, n_seeds=1)

    assert results[1][3] is True, "the seed was not reported as undersized"
    said = [line for line in loguru_lines if "kept 4/1673px" in line]
    assert said, f"the retention figure never reached the log: {loguru_lines}"


def test_an_undersized_seed_reaches_the_summary_as_small(tmp_path):
    """The label has to travel: place_all carries it and _process_cad_tht_sample records it, so the
    row-count gate and summary.json name it like the other routes' failures."""
    from anomalygen.scripts.auto_mask_placement import roi_place as P

    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    results = RING.RingMaskPlacer().place_all(_submask(tmp_path), cands, shape, n_seeds=1)
    assert len(results[1]) == 4, "place_all must report whether the placement was undersized"

    n_req, n_placed, mask, undersized = results[1]
    assert mask is not None and undersized is False
    assert P is not None  # the consumer of that fourth element; exercised in test_roi_place.py


def test_a_roi_covering_its_own_centroid_is_placed_centred(tmp_path):
    """The translate/centre choice is read off the ROI's SHAPE, never off the class name.

    ``hidden`` is "BCD" -- the whole filled disc -- so its centroid IS on the ROI and there is no
    hole to fall into. Translating such a placement would only push it toward the rim for no reason,
    so it stays strictly centred; the note is what records which branch ran.
    """
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "hidden")
    roi = cands[0].mask > 127
    ys, xs = np.nonzero(roi)
    assert roi[int(round(ys.mean())), int(round(xs.mean()))], "fixture disc does not cover its centroid"

    mask, note = RING.RingMaskPlacer().place_on_roi(_submask(tmp_path), cands[0], shape, seed=1)
    assert mask is not None and note.startswith("centred"), note


def test_the_placer_reports_no_result_rather_than_falling_back_to_the_roi(tmp_path):
    """There is deliberately no 'use the ROI itself' fallback — that would paint the whole ring."""
    _, joints, shape, _ = _one_joint(tmp_path)
    cands = tht2roi.get_ring_candidates(joints, "interrupted")
    empty = tmp_path / "empty.png"
    Image.fromarray(np.zeros((80, 80), np.uint8)).save(empty)
    mask, note = RING.RingMaskPlacer().place_on_roi(str(empty), cands[0], shape, seed=1)
    assert mask is None and note in {"EMPTY_ROI", "NO_PLACEMENT"}


# ===========================================================================
# bridge — fitted, not confined
# ===========================================================================
from anomalygen.auto_mask_placement.tht2roi.defects import bridge as BR  # noqa: E402


def _ring(shape=(60, 60), cx=30, cy=30, r_out=20, r_in=12):
    return _disc(shape, cx, cy, r_out) & ~_disc(shape, cx, cy, r_in)


def _bar(shape, rows, cols):
    m = np.zeros(shape, np.uint8)
    m[rows[0] : rows[1], cols[0] : cols[1]] = 255
    return m


def test_bridge_ok_accepts_one_component_spanning_the_ring_and_the_edge():
    B = _ring()
    m = _bar((60, 60), (26, 34), (0, 30))  # from the left edge into the ring
    assert BR.bridge_ok(m, B, min_b=10)


def test_bridge_ok_rejects_two_separate_blobs():
    """The substance of the check: a blob on the pad plus a blob at the edge is not a bridge.

    Both halves of the naive test pass — something touches B, something touches the edge — so
    checking them separately would accept this and the generator would render two defects.
    """
    B = _ring()
    m = _bar((60, 60), (26, 34), (0, 6))  # at the edge, nowhere near the ring
    m |= _bar((60, 60), (26, 34), (12, 24))  # on the ring, not connected to the first
    assert m[26:34, 6:12].sum() == 0, "the fixture must leave a real gap"
    assert not BR.bridge_ok(m, B, min_b=10)


def test_bridge_ok_rejects_a_mask_that_never_reaches_the_edge():
    B = _ring()
    assert not BR.bridge_ok(_bar((60, 60), (26, 34), (12, 24)), B, min_b=10)


def test_bridge_ok_rejects_a_graze_of_the_pad_ring():
    """A bridge has to grow FROM the joint, so a one-pixel contact does not qualify."""
    B = _ring()
    m = _bar((60, 60), (29, 31), (0, 12))  # reaches the edge, barely clips the ring
    overlap = int(((m > 127) & B).sum())
    assert BR.bridge_ok(m, B, min_b=overlap)
    assert not BR.bridge_ok(m, B, min_b=overlap + 1)


def test_bridge_ok_rejects_an_empty_mask():
    assert not BR.bridge_ok(np.zeros((60, 60), np.uint8), _ring(), min_b=1)


@pytest.mark.parametrize(
    "direction, probe",
    [("left", (30, 0)), ("right", (30, 59)), ("up", (0, 30)), ("down", (59, 30))],
)
def test_extend_to_edge_reaches_the_named_edge(direction, probe):
    m = _bar((60, 60), (28, 32), (28, 32))
    out = BR.extend_to_edge(m, direction)
    assert out[probe] > 0


def test_extend_to_edge_keeps_the_frontier_width():
    """Extrudes the profile of the last few rows, so the tail keeps the width the mask has there."""
    m = _bar((60, 60), (25, 35), (20, 30))
    out = BR.extend_to_edge(m, "left")
    assert (out[:, 0] > 0).sum() == 10


def test_fit_bridge_takes_the_smallest_shift():
    """Searching outward by Chebyshev radius means the annotation moves as little as it can."""
    B = _ring()
    m = _bar((60, 60), (26, 34), (0, 11))  # one pixel short of the ring
    assert not BR.bridge_ok(m, B, min_b=20)
    fitted, shift = BR.fit_bridge(m, B, min_b=20)
    assert fitted is not None
    assert BR.bridge_ok(fitted, B, min_b=20)
    assert max(abs(shift[0]), abs(shift[1])) <= 3


def test_fit_bridge_is_deterministic():
    B = _ring()
    m = _bar((60, 60), (26, 34), (0, 11))
    a, sa = BR.fit_bridge(m, B, min_b=20)
    b, sb = BR.fit_bridge(m, B, min_b=20)
    assert sa == sb and np.array_equal(a, b)


def test_fit_bridge_gives_up_when_the_ring_cannot_be_spanned():
    """An unsatisfiable request returns None rather than an invalid mask."""
    B = _ring()
    m = _bar((60, 60), (29, 31), (28, 32))
    assert BR.fit_bridge(m, B, min_b=10**6) == (None, None)


def test_ring_offsets_walk_the_chebyshev_ring_in_row_major_order():
    """The order decides which of several equally-small shifts wins, so it must be stable."""
    assert list(BR._ring_offsets(0)) == [(0, 0)]
    got = list(BR._ring_offsets(1))
    assert got == [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for radius in (2, 3, 5):
        offs = list(BR._ring_offsets(radius))
        assert len(offs) == len(set(offs)) == (2 * radius + 1) ** 2 - (2 * radius - 1) ** 2
        assert all(max(abs(dx), abs(dy)) == radius for dx, dy in offs)


def test_min_b_combines_a_fraction_with_an_absolute_floor(tmp_path):
    """The fraction alone collapses on a small pad; the floor alone lets a large one be grazed."""
    _, joints, _, _ = _one_joint(tmp_path)
    big = joints[0]
    placer = BR.THTBridgeMaskPlacer(min_b_frac=0.30, min_b_floor=30)
    assert placer.min_b(big) == int(0.30 * big.B.sum())  # fraction wins on a large ring

    _, small, _, _ = _one_joint(tmp_path / "s", r_pad=8, r_hole=6, r_pin=3)
    assert placer.min_b(small[0]) == 30  # floor wins on a small one


# ---------------------------------------------------------------------------
# bridge through the placer
# ---------------------------------------------------------------------------
def _bridge_submask(tmp_path, size=80):
    """A bar running from the frame edge towards the joint — the shape a real annotation has."""
    p = tmp_path / "bridge_sub.png"
    Image.fromarray(_bar((size, size), (34, 46), (0, 34))).save(p)
    return str(p)


def _bridge_placer(deform=False):
    """A placer tuned for the fixtures. Deformation is OFF unless a test is about it.

    The threshold is lower than the shipped default so the fixture's bar is a valid bridge on an
    80 px crop; the default's arithmetic is covered by test_min_b_combines_... above.
    """
    return BR.THTBridgeMaskPlacer(min_b_frac=0.05, min_b_floor=10, deform=deform)


def test_a_placed_bridge_spans_the_pad_ring_and_the_crop_edge(tmp_path):
    _, joints, shape, _ = _one_joint(tmp_path)
    placer = _bridge_placer()
    results = placer.place_all(_bridge_submask(tmp_path), joints, shape, n_instances=1, n_seeds=1)
    _, n_placed, mask, _small = results[1]
    assert n_placed == 1 and mask is not None
    assert BR.bridge_ok(mask, joints[0].B, placer.min_b(joints[0]))


def test_a_bridge_deliberately_leaves_the_roi(tmp_path):
    """The opposite of every ring class: clipping a bridge to B+C would stop it reaching the edge."""
    gen, joints, shape, _ = _one_joint(tmp_path)
    roi = gen.generate_candidates(joints, S.BRIDGE)[0].mask > 127
    results = _bridge_placer().place_all(_bridge_submask(tmp_path), joints, shape, n_seeds=1)
    mask = results[1][2] > 127
    assert (mask & ~roi).any(), "a bridge that stayed inside its ROI could not touch the frame"


def test_bridge_candidates_are_an_overlay_not_a_confinement(tmp_path):
    """They exist so every THT class writes the same assets; the placer never clips to them."""
    gen, joints, _, _ = _one_joint(tmp_path)
    cand = gen.generate_candidates(joints, S.BRIDGE)[0]
    letters, _ = S.roi_rule(S.BRIDGE)
    assert np.array_equal(cand.mask, R.roi_for(joints[0], letters))


def test_bridge_skips_a_clipped_joint(tmp_path):
    gen, joints, shape = _clipped_neighbour(tmp_path)
    assert len(gen.generate_candidates(joints, S.BRIDGE)) == 1
    results = _bridge_placer().place_all(_bridge_submask(tmp_path, size=80), joints, shape, n_seeds=1)
    assert results[1][0] == 1, "only the intact joint is a placement target"


def test_bridge_warns_that_extra_seeds_repeat_one_mask(tmp_path, loguru_lines):
    """The fit is the smallest shift that works, so it is unique — seeds cannot vary it."""
    _, joints, shape, _ = _one_joint(tmp_path)
    results = _bridge_placer().place_all(_bridge_submask(tmp_path), joints, shape, n_seeds=3)
    rendered = {m.tobytes() for _, _, m, _small in results.values() if m is not None}
    assert len(rendered) == 1
    assert any("repeats" in line and "submask bank" in line for line in loguru_lines)


def test_bridge_warns_when_every_joint_is_used_regardless_of_seed(tmp_path, loguru_lines):
    """Seeds only choose WHICH joints are used, so taking all of them removes the last variation.

    ``rng.choice(n, size=n, replace=False)`` returns a permutation, and the masks are OR-ed together,
    so every seed produces a byte-identical result — with two joints exactly as with one.
    """
    path = _mask_png(tmp_path, [(25, 40, 18, 11, 5), (75, 40, 18, 11, 5)], size=(100, 80))
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    joints, shape = gen.parse_joints(path)
    assert len(joints) == 2

    sub = tmp_path / "sub2.png"
    Image.fromarray(_bar((80, 100), (36, 46), (0, 30))).save(sub)
    results = _bridge_placer().place_all(str(sub), joints, shape, n_instances=2, n_seeds=3)

    rendered = {m.tobytes() for _, _, m, _small in results.values() if m is not None}
    assert len(rendered) == 1
    assert any("repeats" in line for line in loguru_lines)


def test_bridge_reports_a_submask_it_cannot_fit(tmp_path):
    _, joints, shape, _ = _one_joint(tmp_path)
    empty = tmp_path / "empty.png"
    Image.fromarray(np.zeros((80, 80), np.uint8)).save(empty)
    mask, note = BR.THTBridgeMaskPlacer().place_on_joint(str(empty), joints[0], shape)
    assert mask is None and note == "EMPTY_SUBMASK"


# ---------------------------------------------------------------------------
# bridge: both ends need real width, and a corner bridge keeps both arms
# ---------------------------------------------------------------------------
def test_bridge_ok_rejects_a_graze_of_the_crop_border():
    """Symmetry with min_b. Without min_edge, "reaches the edge" means a single pixel."""
    B = _ring()
    m = np.zeros((60, 60), np.uint8)
    m[12:24, 26:34] = 255  # a body sitting on the ring
    m[0:13, 29] = 255  # a 1 px wide thread up to the border
    assert int(((m > 127) & B).sum()) >= 20
    assert BR.bridge_ok(m, B, min_b=20, min_edge=1)  # bare contact: accepted
    assert not BR.bridge_ok(m, B, min_b=20, min_edge=3)  # real width: rejected


def test_min_edge_scales_with_the_crop_and_has_a_floor():
    placer = BR.THTBridgeMaskPlacer(min_edge_frac=0.05, min_edge_floor=3)
    assert placer.min_edge((200, 200)) == 10  # fraction wins on a large crop
    assert placer.min_edge((40, 40)) == 3  # floor wins on a small one
    assert placer.min_edge((200, 40)) == 3, "scaled off the SHORTER side"


def test_target_edges_returns_every_edge_already_touched():
    """A corner annotation touches two; extending only the nearest would drop the other."""
    corner = np.zeros((60, 60), bool)
    corner[50:60, 45:60] = True  # touches both bottom and right
    assert set(BR._target_edges(corner)) == {"down", "right"}

    side = np.zeros((60, 60), bool)
    side[20:40, 45:60] = True
    assert BR._target_edges(side) == ("right",)


def test_target_edges_falls_back_to_the_nearest_when_none_is_touched():
    floating = np.zeros((60, 60), bool)
    floating[25:35, 40:50] = True  # nearest border is the right one
    assert BR._target_edges(floating) == ("right",)


def test_a_corner_bridge_keeps_both_contacts_after_being_shifted(tmp_path):
    """The bug this fixes: the shift broke the second contact and nothing repaired it.

    Measured on the sample corpus, every fitted row lost its bottom contact and kept a 1-4 px gap.
    """
    _, joints, shape, _ = _one_joint(tmp_path)
    h, w = shape
    sub = np.zeros((h, w), np.uint8)
    sub[h - 18 :, w - 14 :] = 255  # a corner bridge: touches bottom AND right
    sub[h - 30 : h - 14, w - 30 : w - 12] = 255  # a body reaching back towards the pad
    p = tmp_path / "corner.png"
    Image.fromarray(sub).save(p)

    assert set(BR._target_edges(sub > 127)) == {"down", "right"}
    mask, note = _bridge_placer().place_on_joint(str(p), joints[0], shape)
    assert mask is not None, note
    m = mask > 127
    assert m[-1, :].any(), "bottom contact lost"
    assert m[:, -1].any(), "right contact lost"


def test_extending_two_edges_does_not_fan_the_mask_out(tmp_path):
    """Each extension reads the SAME base mask; chaining would let one widen the other's profile."""
    base = np.zeros((60, 60), np.uint8)
    base[40:50, 40:50] = 255
    chained = BR.extend_to_edge(BR.extend_to_edge(base, "right"), "down")
    unioned = BR._extend_all(base, ("right", "down"))
    assert int((unioned > 0).sum()) < int((chained > 0).sum())
    # both still reach both borders
    for m in (unioned,):
        assert (m[-1, :] > 0).any() and (m[:, -1] > 0).any()


# ---------------------------------------------------------------------------
# bridge: elastic deformation
# ---------------------------------------------------------------------------
def _fitted_bridge(tmp_path):
    """A placed, un-deformed bridge plus the joint and directions it was fitted to."""
    _, joints, shape, _ = _one_joint(tmp_path)
    placer = BR.THTBridgeMaskPlacer(min_b_frac=0.05, min_b_floor=10, deform=False)
    mask, _ = placer.place_on_joint(_bridge_submask(tmp_path), joints[0], shape)
    assert mask is not None
    return mask, joints[0], shape, BR._target_edges(mask > 127)


def test_deform_preserves_contact_area_and_unity(tmp_path):
    """The three properties a warp must not break, checked on deform_bridge's own output."""
    mask, _, _, directions = _fitted_bridge(tmp_path)
    target = int((mask > 0).sum())
    warped = BR.deform_bridge(mask, np.random.default_rng(3), directions)
    assert warped is not None

    gaps = BR._edge_gaps(warped > 0)
    assert all(gaps[d] == 0 for d in directions), "lost a border it was anchored to"
    assert cv2.connectedComponents((warped > 0).astype(np.uint8), 8)[0] - 1 == 1, "fragmented"
    assert abs(int((warped > 0).sum()) - target) <= BR.DEFORM_TOL * target, "area drifted"
    assert not np.array_equal(warped > 0, mask > 0), "did not actually change the mask"


def test_deform_is_what_makes_seeds_matter_for_bridge(tmp_path):
    """Without it the smallest valid shift is unique, so every seed returns the same mask."""
    _, joints, shape, _ = _one_joint(tmp_path)
    sub = _bridge_submask(tmp_path)
    kw = dict(min_b_frac=0.05, min_b_floor=10)

    plain = BR.THTBridgeMaskPlacer(deform=False, **kw)
    assert len({plain.place_on_joint(sub, joints[0], shape, seed=s)[0].tobytes() for s in (1, 2, 3)}) == 1

    warping = BR.THTBridgeMaskPlacer(deform=True, **kw)
    assert len({warping.place_on_joint(sub, joints[0], shape, seed=s)[0].tobytes() for s in (1, 2, 3)}) == 3


def test_a_deformed_bridge_is_still_a_valid_bridge(tmp_path):
    """The warp knows about borders and area but nothing about the pad ring, so this is not free."""
    _, joints, shape, _ = _one_joint(tmp_path)
    placer = BR.THTBridgeMaskPlacer(min_b_frac=0.05, min_b_floor=10, deform=True)
    for seed in (1, 2, 3, 4, 5):
        mask, note = placer.place_on_joint(_bridge_submask(tmp_path), joints[0], shape, seed=seed)
        assert mask is not None, note
        assert BR.bridge_ok(mask, joints[0].B, placer.min_b(joints[0]), placer.min_edge(shape)), note


def test_deform_leaves_the_row_alone_when_it_cannot_converge(tmp_path):
    """A shape no field can warp costs variety, never the row."""
    mask, joint, shape, _ = _fitted_bridge(tmp_path)
    placer = BR.THTBridgeMaskPlacer(min_b_frac=0.05, min_b_floor=10, deform=True)
    # tol=0 is unsatisfiable, so deform_bridge always gives up.
    placer.deform_alpha, placer.deform_sigma = 3.0, 6.0
    import unittest.mock as mock

    with mock.patch.object(BR, "deform_bridge", return_value=None):
        out, note = placer.place_on_joint(_bridge_submask(tmp_path), joint, shape, seed=1)
    assert out is not None and "did not converge" in note
    assert np.array_equal(out, mask), "should fall back to the un-warped placement"


def test_elastic_does_not_move_pixels_at_an_anchored_border():
    """Damping to zero near the border is what keeps the contact through the warp."""
    m = np.zeros((60, 60), np.uint8)
    m[40:60, 30:50] = 255  # touches the bottom border
    warped = BR._elastic(m, np.random.default_rng(0), alpha=6.0, sigma=6.0, directions=("down",))
    assert np.array_equal(warped[-1, :] > 0, m[-1, :] > 0), "the anchored row moved"
    assert not np.array_equal(warped > 0, m > 0), "nothing moved at all"


@pytest.mark.parametrize("frac", [0.5, 0.8, 1.2, 2.0])
def test_set_area_hits_the_target_both_ways(frac):
    """Shrink and grow, on a boundary that is not axis-aligned.

    The shape matters. A square's boundary crosses each level everywhere at once, so its reachable
    areas jump in steps of ~100 px and a target between two of them is simply unreachable (measured:
    7.6% off for a 30x30 square halved). A disc's boundary crosses gradually, and the same code lands
    within a couple of percent. That is a property of level sets, not a defect here -- real defect
    annotations have irregular boundaries and behave like the disc.
    """
    m = np.zeros((80, 80), np.uint8)
    cv2.circle(m, (40, 40), 17, 255, -1)
    start = int((m > 0).sum())
    target = int(start * frac)
    assert abs(int((BR.set_area(m, target) > 0).sum()) - target) <= 0.03 * target


def test_keep_main_prefers_the_component_holding_the_most_anchored_borders():
    """A corner bridge must keep the piece holding BOTH contacts, not whichever is largest."""
    m = np.zeros((60, 60), np.uint8)
    m[52:60, 52:60] = 255  # small, touches down AND right
    m[10:40, 10:40] = 255  # much larger, touches neither
    kept = BR.keep_main(m, ("down", "right"))
    assert (kept[-1, :] > 0).any() and (kept[:, -1] > 0).any()
    assert int((kept > 0).sum()) < 900, "kept the large component instead"


def test_no_repeat_warning_when_deformation_is_on(tmp_path, loguru_lines):
    """The warning is about seeds being inert; with a warp they are not."""
    _, joints, shape, _ = _one_joint(tmp_path)
    BR.THTBridgeMaskPlacer(min_b_frac=0.05, min_b_floor=10, deform=True).place_all(
        _bridge_submask(tmp_path), joints, shape, n_seeds=3
    )
    assert not any("repeats" in line for line in loguru_lines)


def test_the_shipped_deformation_is_strong_enough_to_read_as_variety(tmp_path):
    """Guards the DEFAULT, not the mechanism.

    The mechanism works at any alpha; the question the default answers is whether a batch's bridges
    look like different defects or like one defect jittered. The reference implementation's 3.0 left
    IoU ~0.95 against the un-warped mask, which is not distinguishable by eye. Ten seeds on one joint
    must produce ten genuinely different masks, all still valid bridges.
    """
    _, joints, shape, _ = _one_joint(tmp_path)
    sub = _bridge_submask(tmp_path)
    kw = dict(min_b_frac=0.05, min_b_floor=10)
    plain = BR.THTBridgeMaskPlacer(deform=False, **kw).place_on_joint(sub, joints[0], shape)[0] > 127
    placer = BR.THTBridgeMaskPlacer(**kw)  # shipped deform defaults

    masks, ious = [], []
    for seed in range(1, 11):
        m, note = placer.place_on_joint(sub, joints[0], shape, seed=seed)
        assert m is not None and BR.bridge_ok(m, joints[0].B, placer.min_b(joints[0]), placer.min_edge(shape)), note
        masks.append(m.tobytes())
        mm = m > 127
        ious.append((mm & plain).sum() / max(1, (mm | plain).sum()))

    assert len(set(masks)) == 10, "seeds collapsed onto the same mask"
    assert np.mean(ious) < 0.92, "the default warp is too weak to read as a different defect"


def test_a_stroke_too_thin_to_survive_the_crop_is_reported_not_raised(tmp_path):
    """A bridge annotation is cut at the training mask's resolution and fitted at the joint crop's.
    NEAREST samples rather than averages, so a 2 px stroke in a 1024 px mask can lose every pixel on
    the way down -- and the emptiness test ran BEFORE the resize, so the placer carried an all-false
    mask into _target_edges and reached xs.min() on an empty array. That ValueError left place_all
    and killed the whole roi_place run, for an annotation that is perfectly good where it was drawn.
    """
    from anomalygen.auto_mask_placement.tht2roi.defects import bridge as BR

    big = np.zeros((1024, 1024), np.uint8)
    big[500:502, 200:800] = 255  # 2 px tall, the width of the crop
    path = tmp_path / "thin_bridge.png"
    Image.fromarray(big).save(path)
    assert (np.array(Image.open(path).convert("L")) > 127).any(), "not empty where it was drawn"

    joint = types.SimpleNamespace(B=np.zeros((64, 64), np.uint8))
    mask, note = BR.THTBridgeMaskPlacer().place_on_joint(str(path), joint, (64, 64), seed=0)

    assert mask is None
    assert note == "SUBMASK_LOST_IN_RESIZE", "an empty file and one lost in the resize need different remedies"


def test_an_annotation_that_survives_the_crop_still_places(tmp_path):
    """The added check must not refuse a mask that is merely resized."""
    from anomalygen.auto_mask_placement.tht2roi.defects import bridge as BR

    big = np.zeros((1024, 1024), np.uint8)
    big[400:700, 100:900] = 255  # thick enough to survive 16x down
    path = tmp_path / "thick_bridge.png"
    Image.fromarray(big).save(path)

    resized = cv2.resize(
        (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8) * 255,
        (64, 64),
        interpolation=cv2.INTER_NEAREST,
    )
    assert (resized > 127).any(), "the fixture must survive the resize or it tests nothing"

    joint = types.SimpleNamespace(B=np.zeros((64, 64), np.uint8))
    _, note = BR.THTBridgeMaskPlacer().place_on_joint(str(path), joint, (64, 64), seed=0)
    assert note != "SUBMASK_LOST_IN_RESIZE"


# ===========================================================================
# one bad pair record is one skipped sample, not the end of the run
# ===========================================================================
def _tht_pair(tmp_path):
    """``(cad_mask path, label path, shape)`` -- the two artefacts a cad_tht pair record names."""
    cad = _mask_png(tmp_path, [(40, 40, 30, 18, 8)])
    gen = tht2roi.THTToROIGenerator(_labels(tmp_path))
    _, shape = gen.parse_joints(cad)
    return cad, _labels(tmp_path), shape


@pytest.mark.parametrize(
    ("drop", "status"),
    [("cad_mask", "NO_CAD_MASK"), ("cad_mask_label", "NO_CAD_MASK"), ("submask", "NO_SUBMASK")],
)
def test_a_pair_record_missing_a_field_skips_that_sample(tmp_path, drop, status):
    """roi_place runs standalone on a hand-written pair file -- the module docstring says so -- and
    a missing cad_mask used to raise KeyError straight out of the run, taking every other sample
    with it. The cad and free routes have reported this per sample all along."""
    from anomalygen.scripts.auto_mask_placement import roi_place as P

    cad, labels, shape = _tht_pair(tmp_path)
    clean = tmp_path / "clean.png"
    Image.fromarray(np.zeros((*shape, 3), np.uint8)).save(clean)
    sample = {
        "clean_image": str(clean),
        "defect_type": "T+interrupted",
        "cad_mask": cad,
        "cad_mask_label": labels,
        "submask": _submask(tmp_path, size=shape[0]),
    }
    sample.pop(drop)

    result = P._process_cad_tht_sample(sample, str(tmp_path / "out"), n_seeds=1, base_seed=0)
    assert result["status"] == status


def test_an_unknown_defect_class_skips_that_sample(tmp_path):
    """roi_rule raises on an unrecognised class on purpose -- so it cannot degrade into an empty
    candidate list that reads as 'found nowhere to go'. Asked per sample, one bad row costs one row.

    Safe only because run_pipeline counts the testcase rows against num_sdg: the shortfall this
    creates fails the run, with the status named out of summary.json."""
    from anomalygen.scripts.auto_mask_placement import roi_place as P

    cad, labels, shape = _tht_pair(tmp_path)
    clean = tmp_path / "clean.png"
    Image.fromarray(np.zeros((*shape, 3), np.uint8)).save(clean)
    sample = {
        "clean_image": str(clean),
        "defect_type": "T+not_a_real_defect",
        "cad_mask": cad,
        "cad_mask_label": labels,
        "submask": _submask(tmp_path, size=shape[0]),
    }

    result = P._process_cad_tht_sample(sample, str(tmp_path / "out"), n_seeds=1, base_seed=0)
    assert result["status"] == "UNKNOWN_DEFECT"


def test_a_complete_pair_record_still_places(tmp_path):
    """The guards must not cost the ordinary sample."""
    from anomalygen.scripts.auto_mask_placement import roi_place as P

    cad, labels, shape = _tht_pair(tmp_path)
    clean = tmp_path / "clean.png"
    Image.fromarray(np.zeros((*shape, 3), np.uint8)).save(clean)
    sample = {
        "clean_image": str(clean),
        "defect_type": "T+interrupted",
        "cad_mask": cad,
        "cad_mask_label": labels,
        "submask": _submask(tmp_path, size=shape[0]),
    }

    result = P._process_cad_tht_sample(sample, str(tmp_path / "out"), n_seeds=1, base_seed=0)
    assert result["status"] == "OK", result
