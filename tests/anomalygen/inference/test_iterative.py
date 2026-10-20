# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the iterative inpainting orchestrator (used by generate.py and validation).

``run_iterative_inpaint_batch`` is driven with a fake batched paint closure, so the whole
crop -> generate -> paste-back loop (real crop_paste geometry) is exercised without a model.
"""

import numpy as np
import pytest
from PIL import Image

from anomalygen.configs.texture.constants import DEFAULT_CROP_RATIO, DEFAULT_MAX_INSTANCES
from anomalygen.data.crop_geometry import InputSizeSpec, clamp_crop_ratio, mask_bbox
from anomalygen.data.utils import MASK_FG_THRESHOLD

# Every window is planned against an input_spec; a small ladder keeps the 100x100 fixtures realistic.
_INPUT_SPEC = InputSizeSpec(input_size_ladder=(64, 128, 256), scale=1.0)
from anomalygen.inference import iterative
from anomalygen.inference.iterative import (
    instance_bboxes,
    run_iterative_inpaint_batch,
    split_mask_into_instances,
)


def _mask(size=100, blobs=((40, 40),), side=15):
    arr = np.zeros((size, size), dtype=np.uint8)
    for r, c in blobs:
        arr[r : r + side, c : c + side] = 255
    return Image.fromarray(arr, mode="L")


def _gray(size=100, value=128):
    return Image.fromarray(np.full((size, size, 3), value, np.uint8), mode="RGB")


def _fg(pil_mask):
    return int((np.array(pil_mask) > 127).sum())


# --- split_mask_into_instances -------------------------------------------------------------------


def test_split_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        split_mask_into_instances(_mask(), max_k=0)


def test_split_k1_returns_mask_unchanged():
    m = _mask()
    assert split_mask_into_instances(m, max_k=1) == [m]


def test_split_empty_mask_returns_empty():
    empty = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    assert split_mask_into_instances(empty, max_k=5) == []


def test_split_components_below_k_one_mask_each():
    m = _mask(blobs=((10, 10), (70, 70)), side=10)
    instances = split_mask_into_instances(m, max_k=5)
    assert len(instances) == 2
    # Each returned mask carries exactly one blob; together they cover the original foreground.
    assert sum(_fg(i) for i in instances) == _fg(m)


def test_split_more_components_than_k_clusters_deterministically():
    m = _mask(blobs=((5, 5), (5, 80), (80, 5), (80, 80)), side=8)
    instances = split_mask_into_instances(m, max_k=2)  # 4 blobs -> KMeans into 2 clusters
    assert len(instances) == 2
    # Clustering partitions the blobs, so total foreground is preserved.
    assert sum(_fg(i) for i in instances) == _fg(m)


@pytest.mark.parametrize(
    ("blobs", "max_k"),
    [(((10, 10), (70, 70)), 5), (((5, 5), (5, 80), (80, 5), (80, 80)), 2), (((10, 10), (70, 70)), 1)],
    ids=["one_per_component", "clustered", "whole_mask"],
)
def test_instance_bboxes_match_the_split_masks(blobs, max_k):
    """The bbox scan must describe exactly the instances the splitter returns, clustered or not:
    the validation token bound is planned from these boxes and generation from those masks."""
    m = _mask(blobs=blobs, side=8)
    assert instance_bboxes(m, max_k) == [mask_bbox(inst) for inst in split_mask_into_instances(m, max_k)]


def test_instance_bboxes_use_the_shared_foreground_threshold():
    # A blob under the cut is background for the splitter, so it must be for the scan too, or a
    # testcase sorts as having an instance and then generates nothing.
    arr = np.zeros((40, 40), np.uint8)
    arr[2:7, 2:7] = 255
    arr[20:25, 20:25] = MASK_FG_THRESHOLD  # exactly at the cut: foreground
    arr[32:37, 32:37] = MASK_FG_THRESHOLD - 1  # just below: background
    assert len(instance_bboxes(Image.fromarray(arr, mode="L"), max_k=5)) == 2
    assert instance_bboxes(Image.fromarray(np.zeros((40, 40), np.uint8), mode="L"), max_k=5) == []


# --- one image per call (how generate.py drives it) ---------------------------------------------


def _recording_paint(color=(255, 0, 0)):
    """Fake batched paint: records ``(crop size, anomaly name, plan)`` per slot and returns a
    solid-colour crop. ``plan`` is the shared crop plan the orchestrator passes through so inference
    cannot re-derive (and drift from) the geometry."""
    calls = []

    def paint(crops, masks, names, seeds, plans):
        for crop, name, plan in zip(crops, names, plans):
            calls.append((crop.size, name, plan))
        return [Image.new("RGB", c.size, color) for c in crops]

    return paint, calls


def _run_one(image, mask, name, paint, spec, *, crop_ratio=DEFAULT_CROP_RATIO, max_instances=DEFAULT_MAX_INSTANCES):
    """One image through the orchestrator, as generate.py calls it: ``(composite, artifacts)``."""
    composites, artifacts = run_iterative_inpaint_batch(
        [image],
        [mask],
        [name],
        paint,
        seeds=[1],
        crop_ratios=[crop_ratio],
        poisson_blends=[False],
        max_instances_list=[max_instances],
        input_spec=spec,
    )
    return composites[0], artifacts[0]


@pytest.mark.parametrize("requested, used", [(8.0, 2.5), (1.0, 1.5), (2.0, 2.0)], ids=["above", "below", "inside"])
def test_clamp_crop_ratio_pins_a_request_into_the_trained_band(requested, used):
    # generate.py records the clamped value, so this is the one place the clamp is defined.
    spec = InputSizeSpec(crop_ratio_range=(1.5, 2.5))
    assert clamp_crop_ratio(requested, spec) == used


def test_an_empty_mask_returns_a_copy_and_warns(loguru_lines):
    # The JSONL dataset refuses such a mask up front; a direct caller still gets the input back, but
    # not silently — the warning names the threshold that made the mask empty.
    image = _gray()
    empty = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    paint, calls = _recording_paint()
    out, artifacts = _run_one(image, empty, "t+d", paint, _INPUT_SPEC)
    assert calls == []  # nothing to inpaint
    assert np.array_equal(np.array(out), np.array(image))
    assert artifacts["cropped_image"] == []
    assert any("t+d" in ln and "no defect pixels" in ln for ln in loguru_lines)


def test_each_instance_is_planned_against_its_class_cap():
    # The checkpoint carries the per-class input-size caps training applied; a class capped at 64
    # must not be planned at a larger size just because the full ladder allows one.
    image = _gray(size=200)
    mask = _mask(size=200, blobs=((60, 60),), side=50)  # 50 px at ctx 2.0 -> a 100 px crop
    paint, calls = _recording_paint()
    spec = InputSizeSpec(input_size_ladder=(64, 128, 256), scale=1.0, class_ladders={"t+small": (64,)})
    _run_one(image, mask, "t+small", paint, spec, max_instances=1)
    _run_one(image, mask, "t+other", paint, spec, max_instances=1)
    assert (calls[0][2].input_w, calls[0][2].input_h) == (64, 64)  # capped: the crop is downscaled to fit
    assert (calls[1][2].input_w, calls[1][2].input_h) == (128, 128)  # full ladder: 100 snaps to 128


def test_a_defect_larger_than_its_class_trained_on_warns_once(loguru_lines, monkeypatch):
    # The cap keeps generation at sizes the class trained on (an untrained size may generate nothing),
    # so a bigger test defect is downsampled to fit. That must be said, once per class, not silent.
    monkeypatch.setattr(iterative, "_CAP_WARNED", set())
    image = _gray(size=200)
    mask = _mask(size=200, blobs=((60, 60),), side=50)  # 50 px at ctx 2.0 -> a 100 px crop, capped onto 64
    paint, _ = _recording_paint()
    spec = InputSizeSpec(input_size_ladder=(64, 128, 256), scale=1.0, class_ladders={"t+small": (64,)})
    for _ in range(2):
        _run_one(image, mask, "t+small", paint, spec, max_instances=1)
    _run_one(image, mask, "t+other", paint, spec, max_instances=1)
    warned = [ln for ln in loguru_lines if "downsampled to fit" in ln]
    assert len(warned) == 1 and "t+small" in warned[0] and "64" in warned[0]


def test_the_reconstruction_is_pasted_into_the_mask_region():
    image = _gray(value=128)
    mask = _mask(blobs=((40, 40),), side=16)
    paint, calls = _recording_paint(color=(255, 0, 0))
    out, _ = _run_one(image, mask, "scratch+dent", paint, _INPUT_SPEC, max_instances=1)

    assert len(calls) == 1
    assert calls[0][1] == "scratch+dent"  # anomaly_name threaded through
    assert calls[0][2] is not None  # the shared plan is threaded through to the painter
    # A pixel well inside the mask is replaced by the reconstruction colour.
    px = np.array(out)[47, 47]
    assert px[0] == 255 and px[1] == 0 and px[2] == 0


def test_a_wide_ratio_is_clamped_into_the_trained_band():
    image = _gray()
    mask = _mask(blobs=((40, 40),), side=16)
    paint, calls = _recording_paint()
    # A ratio far above crop_ratio_range is clamped back into the trained framing band rather than
    # handing the model a framing it never saw.
    _run_one(image, mask, "t+d", paint, _INPUT_SPEC, crop_ratio=50.0, max_instances=1)
    assert calls[0][0][0] < image.size[0] and calls[0][0][1] < image.size[1]


def test_the_window_comes_from_the_shared_plan():
    # The window is sized by the SHARED planner and the plan is handed to the painter, so inference
    # cannot re-derive a framing that drifts from training's.
    image = _gray()
    mask = _mask(blobs=((40, 40),), side=16)
    paint, calls = _recording_paint()
    _run_one(image, mask, "t+d", paint, _INPUT_SPEC, max_instances=1)

    (size, _, plan) = calls[0]
    assert plan is not None
    assert size == (plan.crop_w, plan.crop_h)
    assert (plan.input_w, plan.input_h) == (64, 64)  # ladder size, not the fixed 512 square
    assert plan.crop_w >= 16 and plan.crop_h >= 16  # the window still holds the whole mask
    assert plan.crop_w == plan.crop_h  # a square bbox under a square-capable ladder stays square


def test_a_ratio_mode_spec_is_honoured():
    # Generation must frame the way the checkpoint trained: under "ratio" the window is bbox * ratio
    # and a single-entry ladder pins the input size. Reading the mode off the spec is the only thing
    # that keeps the two sides together.
    image = _gray()
    mask = _mask(blobs=((40, 40),), side=16)
    paint, calls = _recording_paint()
    spec = InputSizeSpec(input_size_ladder=(128,), scale=1.0, crop_ratio_range=(1.5, 8.0), crop_mode="ratio")
    _run_one(image, mask, "t+d", paint, spec, crop_ratio=4.0, max_instances=1)

    (_, _, plan) = calls[0]
    assert (plan.crop_w, plan.crop_h) == (64, 64)  # 16px bbox * 4.0, framed off the bbox alone
    assert (plan.input_w, plan.input_h) == (128, 128)  # the one ladder size


def test_artifacts_hold_one_entry_per_instance():
    image = _gray()
    mask = _mask(blobs=((20, 20), (70, 70)), side=10)
    paint, _ = _recording_paint()
    out, artifacts = _run_one(image, mask, "t+d", paint, _INPUT_SPEC)
    assert isinstance(out, Image.Image)
    assert set(artifacts) == {"cropped_image", "cropped_mask", "annotated_image", "mask_cropped_image"}
    # One entry per instance (two blobs -> two instances).
    for key in artifacts:
        assert len(artifacts[key]) == 2


# --- run_iterative_inpaint_batch -----------------------------------------------------------------


def _batch_paint(color=(255, 0, 0)):
    """Fake batched paint that records the seeds of each depth's call; one solid crop per slot."""
    calls = []

    def paint(crops, masks, names, seeds, plans):
        assert len(crops) == len(masks) == len(names) == len(seeds) == len(plans)
        calls.append(list(seeds))
        return [Image.new("RGB", c.size, color) for c in crops]

    return paint, calls


def test_a_batch_matches_one_image_per_call_for_different_instance_counts():
    imgs = [_gray(value=128), _gray(value=64)]
    masks = [_mask(blobs=[(40, 40)], side=16), _mask(blobs=[(20, 20), (70, 70)], side=10)]  # 1 vs 2 instances
    names = ["t+d", "t+d"]

    # One-image-per-call reference (generate.py's shape) with the same solid-colour paint.
    single_paint = _recording_paint()[0]
    reference = [_run_one(imgs[i], masks[i], names[i], single_paint, _INPUT_SPEC)[0] for i in range(2)]
    composites, artifacts = run_iterative_inpaint_batch(
        imgs,
        masks,
        names,
        _batch_paint()[0],
        input_spec=_INPUT_SPEC,
        num_depth=2,
        seeds=[1, 1],
        crop_ratios=[DEFAULT_CROP_RATIO, DEFAULT_CROP_RATIO],
        poisson_blends=[False, False],
        max_instances_list=[DEFAULT_MAX_INSTANCES, DEFAULT_MAX_INSTANCES],
    )
    assert len(composites) == 2
    for i in range(2):
        # Batched instance-depth generation (with dummy padding) matches one image per call.
        assert np.array_equal(np.array(composites[i]), np.array(reference[i]))
    assert len(artifacts[0]["cropped_image"]) == 1  # image 0: 1 instance
    assert len(artifacts[1]["cropped_image"]) == 2  # image 1: 2 instances


def test_num_depth_defaults_to_the_calls_own_max_instance_depth():
    # Generation runs unsharded, one image per call: no lockstep to keep, so the depth is this call's.
    imgs = [_gray(), _gray()]
    masks = [_mask(blobs=[(40, 40)], side=16), _mask(blobs=[(20, 20), (70, 70)], side=10)]
    paint, calls = _batch_paint()
    run_iterative_inpaint_batch(
        imgs,
        masks,
        ["t+d", "t+d"],
        paint,
        input_spec=_INPUT_SPEC,
        seeds=[1, 1],
        crop_ratios=[DEFAULT_CROP_RATIO] * 2,
        poisson_blends=[False] * 2,
        max_instances_list=[DEFAULT_MAX_INSTANCES] * 2,
    )
    assert len(calls) == 2  # one forward per depth; the deeper image sets the count


def test_batch_seeds_are_offset_by_instance_depth():
    # Every instance of a sample must get its OWN noise draw: depth j is generated with
    # ``seed + j``; generate.py strides its per-output seeds so the offsets never reach the next
    # output. Reusing one seed across depths would give a multi-instance sample identical noise.
    imgs = [_gray(value=128), _gray(value=64)]
    masks = [_mask(blobs=[(20, 20), (70, 70)], side=10), _mask(blobs=[(20, 20), (70, 70)], side=10)]
    paint, calls = _batch_paint()
    run_iterative_inpaint_batch(
        imgs,
        masks,
        ["t+d", "t+d"],
        paint,
        input_spec=_INPUT_SPEC,
        num_depth=2,
        seeds=[500, 9000],
        crop_ratios=[DEFAULT_CROP_RATIO, DEFAULT_CROP_RATIO],
        poisson_blends=[False, False],
        max_instances_list=[DEFAULT_MAX_INSTANCES, DEFAULT_MAX_INSTANCES],
    )
    # One call per depth; each sample keeps its own base seed and is offset by the depth only.
    assert calls == [[500, 9000], [501, 9001]]


def test_batch_empty_mask_returns_original():
    imgs = [_gray()]
    empty = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    composites, artifacts = run_iterative_inpaint_batch(
        imgs,
        [empty],
        ["t+d"],
        _batch_paint()[0],
        input_spec=_INPUT_SPEC,
        num_depth=1,
        seeds=[1],
        crop_ratios=[DEFAULT_CROP_RATIO],
        poisson_blends=[False],
        max_instances_list=[DEFAULT_MAX_INSTANCES],
    )
    assert np.array_equal(np.array(composites[0]), np.array(imgs[0]))
    assert artifacts[0]["cropped_image"] == []


@pytest.mark.parametrize(
    "blobs_per_image, num_depth",
    [
        ([[(20, 20)], [(70, 70)]], 3),  # both images run out after depth 0; depths 1-2 are all dummy
        ([[], []], 2),  # this rank has no real instance at ANY depth
    ],
)
def test_batch_runs_one_forward_per_depth_even_when_every_slot_is_a_dummy(blobs_per_image, num_depth):
    # num_depth is the max instance depth ACROSS ranks, so a rank that has run out of real instances
    # must still issue the remaining forwards: skipping one leaves it a parameter all-gather behind
    # its peers and hangs the collective. The dummy output is discarded rather than pasted.
    imgs = [_gray() for _ in blobs_per_image]
    masks = [_mask(blobs=tuple(b), side=10) for b in blobs_per_image]
    widths = []

    def paint(crops, cmasks, names, seeds, plans):
        # A dummy slot must still carry a real crop and plan; None would reach the model instead.
        assert all(c is not None for c in crops) and all(p is not None for p in plans)
        widths.append(len(crops))
        return [Image.new("RGB", c.size, (255, 0, 0)) for c in crops]

    composites, _ = run_iterative_inpaint_batch(
        imgs,
        masks,
        ["t+d"] * len(imgs),
        paint,
        input_spec=_INPUT_SPEC,
        num_depth=num_depth,
        seeds=[1] * len(imgs),
        crop_ratios=[DEFAULT_CROP_RATIO] * len(imgs),
        poisson_blends=[False] * len(imgs),
        max_instances_list=[DEFAULT_MAX_INSTANCES] * len(imgs),
    )
    assert widths == [len(imgs)] * num_depth  # one full-width forward per depth, real or not
    for img, blobs, comp in zip(imgs, blobs_per_image, composites):
        if not blobs:
            assert np.array_equal(np.array(comp), np.array(img))  # nothing real -> nothing pasted
