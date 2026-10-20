# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for anomalygen.data.crop_geometry (the shared train/inference crop planner).

The planner is the single source of truth for both sides, so these pin the invariants a caller
relies on: the window always contains the whole mask, the input size is always a ladder size, and a
mask survives resizing.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from anomalygen.data.crop_geometry import (
    InputSizeSpec,
    apply_crop_plan,
    ceil_input_size,
    fit_to_input_size,
    input_size_spec_from_config,
    ladder_for,
    normalize_input_size_ladder,
    place_window,
    plan_crop,
    plan_crop_ratio,
    plan_window,
    resize_mask,
    snap_input_size,
    validate_geometry,
)
from anomalygen.data.utils import MASK_FG_THRESHOLD

# --- resize_mask ---------------------------------------------------------------------------------


def _line_mask(size=400, thickness=1, length=300):
    m = np.zeros((size, size), np.uint8)
    y = size // 2
    m[y : y + thickness, (size - length) // 2 : (size + length) // 2] = 255
    return Image.fromarray(m, "L")


@pytest.mark.parametrize("factor", [0.5, 0.25, 0.1, 0.05])
def test_resize_mask_never_loses_a_thin_defect(factor):
    # NEAREST samples one source pixel per output pixel, so a 1px scratch survives only by luck.
    # Any-touch must keep it at every downsample factor, including ones where a uint8 coverage
    # intermediate would round the fraction back to zero.
    mask = _line_mask()
    side = max(1, int(400 * factor))
    out = resize_mask(mask, (side, side))

    assert out.size == (side, side)
    assert int((np.asarray(out) >= MASK_FG_THRESHOLD).sum()) > 0
    # A resized mask must stay binary: a soft edge would re-enter the pipeline as partial foreground.
    assert set(np.unique(np.asarray(out)).tolist()) <= {0, 255}


def test_resize_mask_keeps_an_empty_mask_empty():
    # Any-touch must not invent foreground where there is none.
    empty = Image.fromarray(np.zeros((400, 400), np.uint8), "L")
    assert int(np.asarray(resize_mask(empty, (50, 50))).sum()) == 0


def test_resize_mask_keeps_a_thin_row_when_one_axis_grows_and_the_other_shrinks():
    # INTER_AREA covers area only when both axes shrink; on a mixed resize OpenCV falls back to a
    # two-tap kernel on the shrinking axis and drops the rows between its taps.
    arr = np.zeros((512, 512), np.uint8)
    arr[259, :] = 255
    out = np.asarray(resize_mask(Image.fromarray(arr, "L"), (1024, 64)))
    assert out.shape == (64, 1024) and (out >= MASK_FG_THRESHOLD).any()


def test_resize_mask_normalises_the_identity_case_too():
    # Callers threshold the result at MASK_FG_THRESHOLD as a 0/255 "L" array. A mode-"1" mask handed
    # back unconverted reads as all-background there; converted, its True pixels are 255.
    bits = Image.fromarray(np.pad(np.full((4, 4), 255, np.uint8), 6), "L").convert("1")
    assert int((np.asarray(bits) >= MASK_FG_THRESHOLD).sum()) == 0  # the raw object reads as empty
    out = resize_mask(bits, bits.size)
    assert out is not bits and out.mode == "L"
    assert int((np.asarray(out) >= MASK_FG_THRESHOLD).sum()) == 16
    assert set(np.unique(np.asarray(out)).tolist()) <= {0, 255}


# --- snap_input_size ---------------------------------------------------------------------------------

_LADDER = (256, 512, 1024, 2048)


@pytest.mark.parametrize(
    "ideal, expected",
    [
        (300.0, 256),  # just above a size -> that size
        (384.0, 256),  # exactly halfway -> ties go down
        (400.0, 512),
        (1500.0, 1024),
        (1600.0, 2048),
    ],
)
def test_snap_input_size_is_the_nearest_size_without_an_rng(ideal, expected):
    # Inference must be deterministic: the same testcase always plans the same geometry.
    assert snap_input_size(ideal, _LADDER, None) == expected


def test_normalize_input_size_ladder_sorts_and_drops_duplicates():
    # The geometry stamp compares ladders as lists, so [512, 512] must stamp the same as [512].
    assert normalize_input_size_ladder([512, 256, 512]) == (256, 512)
    assert normalize_input_size_ladder([512]) == normalize_input_size_ladder([512, 512])


@pytest.mark.parametrize("ideal, expected", [(-5.0, 256), (0.0, 256), (255.0, 256), (2049.0, 2048), (99999.0, 2048)])
def test_snap_input_size_clamps_to_the_ends_of_the_ladder(ideal, expected):
    assert snap_input_size(ideal, _LADDER, None) == expected
    assert snap_input_size(ideal, _LADDER, np.random.default_rng(0)) == expected


def test_snap_input_size_stochastic_path_only_returns_neighbouring_sizes():
    # Training rounds stochastically so the model sees both roundings of a framing — but never a
    # size that is not adjacent to the ideal, which would silently change the requested geometry.
    rng = np.random.default_rng(0)
    draws = [snap_input_size(384.0, _LADDER, rng) for _ in range(400)]
    assert set(draws) == {256, 512}  # both roundings appear at the midpoint
    # Proportional to proximity: an ideal 3/4 of the way up lands on the upper size most of the time.
    high = [snap_input_size(256 + 0.75 * 256, _LADDER, rng) for _ in range(400)]
    assert set(high) <= {256, 512}
    assert 0.6 < high.count(512) / len(high) < 0.9


# --- plan_crop / place_window: the containment invariant ------------------------------------------


def _contains(origin: int, crop: int, b_lo: int, b_hi: int) -> bool:
    return origin <= b_lo and origin + crop - 1 >= b_hi


def test_place_window_keeps_an_even_bbox_whole_when_the_crop_equals_it():
    # Regression: with crop == bbox there is no slack at all, and an even extent makes the
    # centroid-derived origin (mid - crop//2) land one pixel too low, cutting the last mask row.
    b_lo, b_hi, crop = 10, 13, 4  # extent 4 (even), crop == bbox
    assert place_window(b_lo, b_hi, crop, 64, None) == b_lo
    assert _contains(place_window(b_lo, b_hi, crop, 64, None), crop, b_lo, b_hi)
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert place_window(b_lo, b_hi, crop, 64, rng) == b_lo  # no slack -> no jitter is possible


def test_place_window_jitter_uses_all_the_slack_and_no_more():
    # The jitter range is the SLACK, not a fraction of the window: bounding it by the window
    # truncates the mask whenever the slack is smaller than that fraction.
    b_lo, b_hi, crop, extent = 40, 49, 12, 100  # bbox 10, crop 12 -> 2px of slack
    rng = np.random.default_rng(0)
    seen = {place_window(b_lo, b_hi, crop, extent, rng) for _ in range(200)}
    assert seen == {38, 39, 40}  # every origin that still contains the mask, and only those


def test_place_window_returns_zero_when_the_window_exceeds_the_extent():
    # A window wider than the image cannot be placed; the plan pads the remainder instead.
    assert place_window(2, 5, 50, 20, None) == 0
    assert place_window(2, 5, 50, 20, np.random.default_rng(0)) == 0


@pytest.mark.parametrize("jitter", [True, False], ids=["training-rng", "inference-deterministic"])
def test_planned_window_always_contains_the_whole_mask(jitter):
    """Randomised property test of plan_crop's hard invariants, on both call paths.

    ``crop >= bbox`` on each axis, and the window ``place_window`` chooses holds the whole mask.
    A window that truncates the defect makes the model inpaint half a defect and the compositor
    paste it back. And ``crop >= bbox * ratio_floor`` unless the image is smaller: the snap to the
    ladder may widen the framing but never tighten it under the floor.
    """
    rng = np.random.default_rng(20260816)
    ladders = [(256, 512, 1024, 2048), (256,), (64, 128, 256), (32, 64, 128, 256, 512)]
    for _ in range(600):
        ladder = ladders[int(rng.integers(len(ladders)))]
        img_w, img_h = int(rng.integers(8, 3000)), int(rng.integers(8, 3000))
        bbox_w = int(rng.integers(1, img_w + 1))
        bbox_h = int(rng.integers(1, img_h + 1))
        x1 = int(rng.integers(0, img_w - bbox_w + 1))
        y1 = int(rng.integers(0, img_h - bbox_h + 1))
        ctx = float(rng.uniform(1.0, 8.0))
        s = float(np.exp(rng.uniform(np.log(0.25), np.log(8.0))))
        plan_rng = rng if jitter else None

        plan = plan_crop(bbox_w, bbox_h, img_w, img_h, ctx, s, ladder, ratio_floor=1.5, rng=plan_rng)

        ctxt = f"bbox={bbox_w}x{bbox_h} img={img_w}x{img_h} ctx={ctx:.2f} s={s:.3f} ladder={ladder}"
        assert plan.crop_w >= bbox_w and plan.crop_h >= bbox_h, ctxt
        assert plan.crop_w <= img_w and plan.crop_h <= img_h, ctxt
        # The framing floor, to within the half-pixel that rounding the window can take.
        assert plan.crop_w >= min(1.5 * bbox_w, img_w) - 0.5 - 1e-6, ctxt
        assert plan.crop_h >= min(1.5 * bbox_h, img_h) - 0.5 - 1e-6, ctxt
        assert plan.input_w in ladder and plan.input_h in ladder, ctxt
        assert 0 < plan.content_w <= plan.input_w and 0 < plan.content_h <= plan.input_h, ctxt
        assert 0 < plan.scale <= s, ctxt
        # The axes snap sizes independently but must share ONE scale, or the cut is stretched and
        # every defect's aspect ratio is silently distorted. Both sides are integers, so allow the
        # half-pixel each rounding can contribute.
        tol = 0.5 * plan.scale + 0.5 + 1e-9
        assert abs(plan.content_w - plan.crop_w * plan.scale) <= tol, ctxt
        assert abs(plan.content_h - plan.crop_h * plan.scale) <= tol, ctxt

        ox = place_window(x1, x1 + bbox_w - 1, plan.crop_w, img_w, plan_rng)
        oy = place_window(y1, y1 + bbox_h - 1, plan.crop_h, img_h, plan_rng)
        assert _contains(ox, plan.crop_w, x1, x1 + bbox_w - 1), ctxt
        assert _contains(oy, plan.crop_h, y1, y1 + bbox_h - 1), ctxt
        assert 0 <= ox <= img_w - plan.crop_w and 0 <= oy <= img_h - plan.crop_h, ctxt


def test_plan_crop_lifts_a_snap_that_would_undercut_the_framing_floor():
    # A 256 px bbox at ctx 1.5 asks for 384 px. Nearest-snapping lands on 256, a window with zero
    # context; the size is lifted to 512 instead so the window keeps at least the floor.
    plan = plan_crop(256, 256, 4000, 4000, 1.5, 1.0, _LADDER, ratio_floor=1.5)
    assert plan.input_w == 512 and plan.crop_w == 512


def test_plan_crop_keeps_the_frame_when_the_image_cannot_give_the_floor():
    # The defect spans most of a 300 px frame: 1.5x would need 384 px that do not exist, so the window
    # is the whole frame and no size is lifted beyond what holds it.
    plan = plan_crop(256, 256, 300, 300, 1.5, 1.0, _LADDER, ratio_floor=1.5)
    assert plan.crop_w == 300 and plan.input_w == 512 and plan.needs_pad


def test_plan_crop_gives_up_scale_before_framing_when_the_ceiling_binds():
    # Priority order: honour s, then shrink ctx toward ratio_floor, then and only then reduce s.
    # A 2000px defect at s=1 cannot fit a 2048 input with 1.5x context, so s must yield.
    plan = plan_crop(2000, 2000, 6000, 6000, 4.0, 1.0, _LADDER, ratio_floor=1.5)
    assert plan.scale < 1.0
    assert plan.crop_w >= 2000 and plan.input_w == 2048
    # A small defect leaves headroom, so s is honoured exactly and the framing is the one requested.
    small = plan_crop(50, 50, 6000, 6000, 4.0, 1.0, _LADDER, ratio_floor=1.5)
    assert small.scale == 1.0


def test_plan_crop_survives_a_defect_spanning_a_whole_axis():
    # Regression: s_eff is a float quotient, so a defect filling an axis can make bbox * s_eff
    # overshoot the largest size by an ulp (1536.0000000000002). The size search must tolerate that
    # instead of raising StopIteration into the dataloader worker.
    plan = plan_crop(929, 2723, 3609, 2723, 2.70, 0.892, (256, 512, 1024, 1536), ratio_floor=1.5)
    assert plan.input_h == 1536
    assert plan.crop_h >= 2723  # the invariant still holds


@pytest.mark.parametrize("seed", [0, 1])
def test_plan_crop_does_not_pad_when_the_image_never_ran_out(seed):
    # Regression: round(round(input/s) * s) is not an identity, so deriving the content size from
    # the rounded crop reported a 1-2px shortfall on a window the image filled completely — and
    # apply_crop_plan then pasted a black stripe along the canvas edge.
    rng = np.random.default_rng(seed)
    for _ in range(200):
        ctx = 1.5 + float(rng.random()) * 2.5
        s = float(np.exp(np.log(0.5) + float(rng.random()) * (np.log(4.0) - np.log(0.5))))
        # A small defect centred in a large frame: the window can always be cut in full.
        plan = plan_crop(60, 60, 800, 600, ctx, s, (256, 512, 1024, 1536), ratio_floor=1.5, rng=rng)
        assert not plan.needs_pad, f"spurious padding for ctx={ctx}, s={s}: {plan}"


def test_plan_crop_still_pads_when_the_image_is_genuinely_too_small():
    # The complement of the test above: padding is legitimate when the frame really does run out.
    plan = plan_crop(40, 40, 100, 100, 4.0, 4.0, (256, 512, 1024, 1536), ratio_floor=1.5)
    assert plan.needs_pad and plan.content_w < plan.input_w


def test_apply_crop_plan_emits_the_input_size_and_keeps_the_whole_mask(mask_with_square):
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    mask = mask_with_square(size=100, lo=40, hi=60)  # 20x20 of foreground, x and y both in [40, 59]
    plan = plan_crop(20, 20, 100, 100, 2.0, 1.0, (64, 128, 256), ratio_floor=1.5)
    # bbox is (x1, y1, x2, y2), inclusive.
    out_img, out_mask = apply_crop_plan(img, mask, plan, (40, 40, 59, 59), np.random.default_rng(0))

    assert out_img.size == out_mask.size == (plan.input_w, plan.input_h)
    # EVERY defect pixel survives: at scale 1.0 nothing is resampled away, so a placement that
    # clipped the mask (or transposed the axes) would land below 400.
    assert int((np.array(out_mask) >= MASK_FG_THRESHOLD).sum()) == 400


# --- fit_to_input_size: the padded path -----------------------------------------------------------


def test_fit_to_input_size_centres_the_content_and_reports_where_it_is(mask_with_square):
    # The no-pad path is covered above; this is the branch where the image ran out. A wrong offset or
    # an off-by-one in content_box silently pastes a black border into every generated crop, or
    # crops back a region shifted by half the padding.
    plan = plan_crop(40, 40, 100, 100, 4.0, 4.0, (256, 512, 1024, 1536), ratio_floor=1.5)
    assert plan.needs_pad and (plan.content_w, plan.content_h) != (plan.input_w, plan.input_h)

    cut = Image.fromarray(np.full((plan.crop_h, plan.crop_w, 3), 200, np.uint8), mode="RGB")
    cut_mask = mask_with_square(size=plan.crop_w, lo=0, hi=plan.crop_w)  # all foreground
    out_i, out_m, box = fit_to_input_size(cut, cut_mask, plan)

    assert out_i.size == out_m.size == (plan.input_w, plan.input_h)
    assert box == (
        (plan.input_w - plan.content_w) // 2,
        (plan.input_h - plan.content_h) // 2,
        (plan.input_w - plan.content_w) // 2 + plan.content_w,
        (plan.input_h - plan.content_h) // 2 + plan.content_h,
    )
    # content_box round trip: cropping it back recovers the real pixels and NONE of the padding.
    inner = np.asarray(out_i.crop(box))
    assert inner.shape[:2] == (plan.content_h, plan.content_w)
    assert inner.min() > 0, "padding leaked into the content box"
    # ...and everything outside it is padding.
    full = np.asarray(out_i)
    assert full[0, 0].sum() == 0 and full[-1, -1].sum() == 0
    # The mask survives the resize into the content area rather than being smeared or dropped.
    assert int((np.asarray(out_m) >= MASK_FG_THRESHOLD).sum()) == plan.content_w * plan.content_h


def test_fit_to_input_size_reports_no_box_when_the_content_fills_the_input(mask_with_square):
    # content_box is None exactly when there is nothing to strip; a spurious box would make the
    # caller crop a full-size output down for no reason.
    plan = plan_crop(20, 20, 100, 100, 2.0, 1.0, (64, 128, 256), ratio_floor=1.5)
    assert not plan.needs_pad
    img = Image.fromarray(np.full((plan.crop_h, plan.crop_w, 3), 50, np.uint8), mode="RGB")
    _, _, box = fit_to_input_size(img, mask_with_square(size=plan.crop_w), plan)
    assert box is None


# --- plan_crop_ratio ("ratio" mode) ---------------------------------------------------------------


@pytest.mark.parametrize("bb", [4, 20, 200], ids=["tiny", "small", "large"])
def test_plan_crop_ratio_frames_the_window_off_the_bbox(bb):
    # The whole point of the mode: the window is bbox * ratio at EVERY defect size, so the defect
    # fills 1/ratio of the input whatever its pixel size. Nothing floors the window either — a 4px
    # defect cuts 12px and is magnified onto the smallest input, rather than being framed wider the
    # way plan_crop's ladder floor would frame it.
    plan = plan_crop_ratio(bb, bb, 4000, 4000, 3.0, _LADDER)
    assert plan.crop_w == plan.crop_h == 3 * bb


def test_plan_crop_ratio_on_a_single_entry_ladder_pins_one_input_size():
    # This is how a recipe asks for one fixed model input size. Every defect, at any framing, must
    # come back on exactly that size, and stretched onto it rather than padded.
    for bb, ratio in ((10, 1.5), (100, 4.0), (900, 8.0)):
        plan = plan_crop_ratio(bb, bb, 1000, 1000, ratio, (512,))
        assert (plan.input_w, plan.input_h) == (512, 512)
        assert not plan.needs_pad


def test_plan_crop_ratio_snaps_each_axis_on_a_multi_entry_ladder():
    # With several sizes the window stays near its native pixel size instead of being resampled onto
    # a fixed one, and a window the image clamps rectangular keeps its aspect per axis.
    plan = plan_crop_ratio(50, 50, 4000, 4000, 4.0, _LADDER)
    assert plan.input_w == plan.input_h == snap_input_size(200, _LADDER)

    clamped = plan_crop_ratio(300, 300, 4000, 400, 4.0, _LADDER)  # short frame clamps the h axis
    assert clamped.crop_w == 1200 and clamped.crop_h == 400
    assert clamped.input_w > clamped.input_h


def test_plan_crop_ratio_rejects_a_nonpositive_ratio():
    with pytest.raises(ValueError, match="crop ratio"):
        plan_crop_ratio(10, 10, 100, 100, 0.0, _LADDER)


@pytest.mark.parametrize("jitter", [True, False], ids=["training-rng", "inference-deterministic"])
def test_plan_crop_ratio_window_always_contains_the_whole_mask(jitter):
    """The same hard invariant as plan_crop: a window that truncates the defect makes the model
    inpaint half a defect and the compositor paste it back."""
    rng = np.random.default_rng(20260830)
    ladders = [(512,), (256, 512, 1024, 1536), (64, 128, 256)]
    for _ in range(600):
        ladder = ladders[int(rng.integers(len(ladders)))]
        img_w, img_h = int(rng.integers(8, 3000)), int(rng.integers(8, 3000))
        bbox_w = int(rng.integers(1, img_w + 1))
        bbox_h = int(rng.integers(1, img_h + 1))
        x1 = int(rng.integers(0, img_w - bbox_w + 1))
        y1 = int(rng.integers(0, img_h - bbox_h + 1))
        ratio = float(rng.uniform(1.0, 8.0))
        plan_rng = rng if jitter else None

        plan = plan_crop_ratio(bbox_w, bbox_h, img_w, img_h, ratio, ladder, rng=plan_rng)

        ctxt = f"bbox={bbox_w}x{bbox_h} img={img_w}x{img_h} ratio={ratio:.2f} ladder={ladder}"
        assert plan.crop_w >= bbox_w and plan.crop_h >= bbox_h, ctxt
        assert plan.crop_w <= img_w and plan.crop_h <= img_h, ctxt
        assert plan.input_w in ladder and plan.input_h in ladder, ctxt
        # The cut is always resized onto the input, never padded, so there are no black borders.
        assert not plan.needs_pad, ctxt

        ox = place_window(x1, x1 + bbox_w - 1, plan.crop_w, img_w, plan_rng)
        oy = place_window(y1, y1 + bbox_h - 1, plan.crop_h, img_h, plan_rng)
        assert _contains(ox, plan.crop_w, x1, x1 + bbox_w - 1), ctxt
        assert _contains(oy, plan.crop_h, y1, y1 + bbox_h - 1), ctxt


# --- one scale for both axes ----------------------------------------------------------------------


def test_a_shared_scale_still_survives_the_input_size_ceiling_together():
    # A scalar request must keep ONE effective scale even when the ceiling binds on one axis only,
    # or a plain isotropic recipe would start distorting aspect.
    plan = plan_crop(100, 2000, 4000, 4000, 2.0, 1.0, (256, 512), ratio_floor=1.5)
    assert plan.content_w / plan.crop_w == pytest.approx(plan.content_h / plan.crop_h, rel=0.02)


# --- plan_window dispatch -------------------------------------------------------------------------


def test_plan_window_dispatches_on_crop_mode():
    args = dict(ctx=2.0, scale=1.0, ladder=_LADDER, ratio_floor=1.5)
    assert plan_window(50, 50, 4000, 4000, crop_mode="ratio", **args) == plan_crop_ratio(
        50, 50, 4000, 4000, 2.0, _LADDER
    )
    assert plan_window(50, 50, 4000, 4000, crop_mode="scale", **args) == plan_crop(
        50, 50, 4000, 4000, 2.0, 1.0, _LADDER, ratio_floor=1.5
    )


# --- validate_geometry: the one place a recipe's geometry is checked ------------------------------


def test_validate_geometry_normalises_what_it_accepts():
    geometry = validate_geometry("scale", [512, 256, 512], [1.5, 2.5], [0.5, 2.0], inference_scale=1.0)
    assert geometry == ("scale", (256, 512), (1.5, 2.5), (0.5, 2.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(crop_mode="fixed"), "crop_mode"),
        (dict(input_size_ladder=(100, 200)), "input_size_ladder"),
        (dict(crop_ratio_range=(1.0, 2.5)), "crop_ratio_range"),
        (dict(crop_ratio_range=(2.0, 2.0)), "low < high"),
        (dict(scale_range=(0.0, 2.0)), "scale_range"),
        (dict(inference_scale=4.0), "inference_scale"),
    ],
    ids=["mode", "ladder", "floor", "pinned-band", "scale", "inference-scale"],
)
def test_validate_geometry_names_the_bad_knob(kwargs, message):
    # A typo in a recipe must fail at parse, not fall back to a geometry the checkpoint was not
    # trained under.
    args = dict(crop_mode="scale", input_size_ladder=_LADDER, crop_ratio_range=(1.5, 2.5), scale_range=(0.5, 2.0))
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        validate_geometry(**args)


def test_validate_geometry_ignores_inference_scale_under_ratio_mode():
    # There is no scale axis to be off-distribution on.
    validate_geometry("ratio", _LADDER, (1.5, 8.0), (0.5, 2.0), inference_scale=4.0)


@pytest.mark.parametrize("ideal, expected", [(0, 256), (256, 256), (257, 512), (1025, 2048), (9999, 2048)])
def test_ceil_input_size_takes_the_smallest_size_that_holds_the_ideal(ideal, expected):
    assert ceil_input_size(ideal, _LADDER) == expected


# --- training and inference plan the same crop ----------------------------------------------------


@pytest.mark.parametrize("crop_mode", ["scale", "ratio"])
def test_training_and_inference_build_the_same_planner_call(monkeypatch, crop_mode, mask_with_square):
    """The two entry points must assemble identical ``plan_window`` arguments, in either mode.

    The training crop stage and ``plan_for`` (generation, and the validation token bound) each build
    the call themselves — the context ratio, the scale and ``ratio_floor`` all come from different
    places on the two sides. Nothing else pins that they agree, and a drift here is invisible: both
    sides keep producing valid crops, just of different framings.
    """
    from anomalygen.data import augmentations as aug_mod
    from anomalygen.data import crop_geometry as geo_mod
    from anomalygen.inference import iterative as inf_mod

    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, {k: v for k, v in kwargs.items() if k != "rng"}))
        return plan_window(*args, **kwargs)

    monkeypatch.setattr(aug_mod, "plan_window", _spy)
    monkeypatch.setattr(geo_mod, "plan_window", _spy)  # plan_for's namespace

    ratio, ladder = 2.0, (64, 128, 256)
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    mask = mask_with_square(size=100, lo=40, hi=60)

    # Pinning both ranges to a point makes training's draws deterministic, so any difference that
    # shows up is a real disagreement rather than the sampled context ratio or scale.
    if crop_mode == "ratio":
        train_crop = aug_mod.RandomRatioCrop(
            crop_ratio_range=(ratio, ratio), input_size_ladder=ladder, rng=np.random.default_rng(0)
        )
    else:
        train_crop = aug_mod.RandomScaleCrop(
            crop_ratio_range=(ratio, ratio),
            scale_range=(1.0, 1.0),
            input_size_ladder=ladder,
            rng=np.random.default_rng(0),
        )
    train_crop(img, mask)
    spec = InputSizeSpec(ladder, 1.0, (ratio, ratio), crop_mode)
    inf_mod._plan_window(mask, 100, 100, spec, crop_ratio=ratio, anomaly_name="t+d")

    assert len(calls) == 2
    assert calls[0] == calls[1], f"training {calls[0]} != inference {calls[1]}"


# --- input_size_spec_from_config -----------------------------------------------------------------


def _dec(**overrides):
    """Stand-in for a checkpoint's ``diffusion_expert_config``."""
    fields = {
        "input_size_ladder": [64, 128],
        "crop_mode": "scale",
        "crop_ratio_range": (1.5, 2.0),
        "inference_scale": 0.5,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_the_spec_carries_the_per_class_caps_normalised():
    # The caps are handed in beside the config: dataset state the checkpoint carries, not a config field.
    spec = input_size_spec_from_config(_dec(), class_ladders={"t+small": [64, 64], "t+big": [128, 64]})
    assert spec.class_ladders == {"t+small": (64,), "t+big": (64, 128)}
    assert input_size_spec_from_config(_dec()).class_ladders is None


def test_ladder_for_uses_the_class_cap_when_there_is_one():
    # A class capped in training must be planned against that cap at generation; an unknown class or
    # a checkpoint without caps falls back to the full ladder.
    spec = InputSizeSpec((64, 128, 256), 1.0, (1.5, 2.5), "scale", class_ladders={"t+small": (64,)})
    assert ladder_for(spec, "t+small") == (64,)
    assert ladder_for(spec, "t+other") == (64, 128, 256)
    assert ladder_for(InputSizeSpec((64, 128, 256)), "t+small") == (64, 128, 256)


def test_input_size_spec_is_inherited_from_the_config():
    # Values distinct from the module defaults, so a defaulting body would fail here.
    spec = input_size_spec_from_config(_dec())
    assert spec.input_size_ladder == (64, 128)
    assert spec.crop_ratio_range == (1.5, 2.0)
    assert spec.scale == 0.5


def test_absent_diffusion_expert_config_raises():
    # generate.py passes getattr(model.config, "diffusion_expert_config", None), so None is reachable.
    with pytest.raises(ValueError, match="no diffusion_expert_config"):
        input_size_spec_from_config(None)


@pytest.mark.parametrize("crop_mode", ["ratoi", None], ids=["typo", "none"])
def test_the_config_reader_rejects_an_unknown_crop_mode(crop_mode):
    # train.py composes any key=value override after the recipe was validated, so the model config
    # can carry a mode the parse never saw. This reader is the one place the stamp, validation and
    # generation share; letting it fall through would plan under "scale" semantics with no log.
    with pytest.raises(ValueError, match="crop_mode must be one of"):
        input_size_spec_from_config(_dec(crop_mode=crop_mode))


# --- the image/mask pair invariant ----------------------------------------------------------------


def test_fit_to_input_size_rejects_a_mismatched_cut(mask_with_square):
    # The choke point every path funnels through, so it backstops the entry-point checks.
    plan = plan_crop(20, 20, 100, 100, 2.0, 1.0, (64, 128, 256), ratio_floor=1.5)
    img = Image.fromarray(np.full((plan.crop_h, plan.crop_w, 3), 50, np.uint8), mode="RGB")
    with pytest.raises(ValueError, match="image/mask size mismatch"):
        fit_to_input_size(img, mask_with_square(size=plan.crop_w // 2), plan)


def test_fit_to_input_size_emits_both_at_the_input_size(mask_with_square):
    plan = plan_crop(40, 40, 100, 100, 4.0, 4.0, (256, 512, 1024, 1536), ratio_floor=1.5)
    cut = Image.fromarray(np.full((plan.crop_h, plan.crop_w, 3), 200, np.uint8), mode="RGB")
    out_i, out_m, _ = fit_to_input_size(cut, mask_with_square(size=plan.crop_w), plan)
    assert out_i.size == out_m.size == (plan.input_w, plan.input_h)


def test_apply_crop_plan_rejects_a_mismatched_pair(mask_with_square):
    plan = plan_crop(20, 20, 100, 100, 2.0, 1.0, (64, 128, 256), ratio_floor=1.5)
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    with pytest.raises(ValueError, match="image/mask size mismatch"):
        apply_crop_plan(img, mask_with_square(size=50), plan, (20, 20, 40, 40))
