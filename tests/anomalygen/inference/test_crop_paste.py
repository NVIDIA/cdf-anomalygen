# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for anomalygen.inference.crop_paste (pure PIL/cv2/numpy geometry).

Reachable from generate.py via inference.iterative; tested directly here for the branches the
orchestrator does not exercise (poisson blend, feathering, annotation, empty-mask guards).
"""

import numpy as np
import pytest
from PIL import Image

from anomalygen.configs.texture.constants import MIN_CROP_RATIO
from anomalygen.data.utils import MASK_FG_THRESHOLD
from anomalygen.inference.crop_paste import (
    annotate,
    best_crop,
    enlarge_mask,
    feathered_alpha,
    mask_bbox,
    paste_back,
)


def _mask(size=50, box=(5, 10, 15, 20)):
    """L-mode mask with a filled foreground rectangle (left, upper, right, lower), inclusive."""
    lx, uy, rx, by = box
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[uy : by + 1, lx : rx + 1] = 255
    return Image.fromarray(arr, mode="L")


def test_mask_bbox_tight_and_inclusive():
    assert mask_bbox(_mask(box=(5, 10, 15, 20))) == (5, 10, 15, 20)


def test_mask_bbox_empty_raises():
    with pytest.raises(ValueError):
        mask_bbox(Image.fromarray(np.zeros((10, 10), np.uint8), mode="L"))


def test_best_crop_centers_and_clamps():
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    mask = _mask(size=100, box=(40, 40, 60, 60))
    cropped_image, cropped_mask, offset = best_crop(img, mask, grid_x=32, grid_y=32)
    assert offset == (34, 34)  # center 50; clamp(50-16) = 34
    assert cropped_image.size == (32, 32)
    assert cropped_mask.size == (32, 32)


def test_best_crop_centers_on_a_defect_larger_than_the_window():
    # Regression: the shared placement rule confines the origin to the range that CONTAINS the mask,
    # which is empty when the window is smaller than the defect. Returning 0 there put the window at
    # the far edge of the image with no defect pixels in it at all, so generation silently no-opped
    # (empty mask -> all-background source -> zero alpha -> composite identical to the input).
    img = Image.fromarray(np.full((200, 200, 3), 50, np.uint8), mode="RGB")
    mask = _mask(size=200, box=(120, 120, 180, 180))  # 61px defect, wider than the 32px window
    cropped_image, cropped_mask, offset = best_crop(img, mask, grid_x=32, grid_y=32)

    assert cropped_image.size == cropped_mask.size == (32, 32)
    assert int((np.array(cropped_mask) >= MASK_FG_THRESHOLD).sum()) > 0, "window must show the defect"
    assert offset == (134, 134)  # centred on the bbox, not pinned to the origin


def test_best_crop_empty_mask_raises():
    img = Image.fromarray(np.zeros((40, 40, 3), np.uint8), mode="RGB")
    empty = Image.fromarray(np.zeros((40, 40), np.uint8), mode="L")
    with pytest.raises(ValueError):
        best_crop(img, empty, grid_x=32, grid_y=32)


def test_enlarge_mask_grows_foreground():
    mask = _mask(size=50, box=(20, 20, 25, 25))
    enlarged = enlarge_mask(mask)
    assert enlarged.mode == "L"
    assert int((np.array(enlarged) > 127).sum()) > int((np.array(mask) > 127).sum())


def test_feathered_alpha_hard_when_feather_nonpositive():
    alpha = feathered_alpha(_mask(size=40, box=(10, 10, 20, 20)), feather=0.0)
    assert set(np.unique(alpha)).issubset({0.0, 1.0})


def test_feathered_alpha_interior_one_and_ramps_outward():
    mask = _mask(size=60, box=(20, 20, 40, 40))
    alpha = feathered_alpha(mask, feather=8.0)
    binary = np.array(mask.convert("L")) >= 128
    # Alpha is exactly 1 everywhere inside the original mask (interior never eroded).
    assert np.allclose(alpha[binary], 1.0)
    # Values stay in [0, 1] and a soft ramp (strictly between 0 and 1) exists outside.
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert np.any((alpha > 0.0) & (alpha < 1.0))


def test_paste_back_feather_zero_keeps_recon_inside_and_source_outside():
    input_image = Image.fromarray(np.zeros((100, 100, 3), np.uint8), mode="RGB")  # black
    cropped_image = Image.fromarray(np.full((50, 50, 3), 10, np.uint8), mode="RGB")  # grey source
    recon_image = Image.fromarray(np.full((50, 50, 3), 200, np.uint8), mode="RGB")  # bright recon
    cropped_mask = _mask(size=50, box=(20, 20, 30, 30))

    out = paste_back(input_image, recon_image, cropped_image, cropped_mask, offset=(25, 25), feather=0.0)
    out_arr = np.array(out)
    assert out_arr[50, 50, 0] == 200  # mask centre -> reconstruction
    assert out_arr[30, 30, 0] == 10  # inside crop, outside mask -> original cropped value
    assert out_arr[5, 5, 0] == 0  # outside the paste window -> untouched input


def test_paste_back_resizes_mismatched_recon():
    input_image = Image.fromarray(np.zeros((80, 80, 3), np.uint8), mode="RGB")
    cropped_image = Image.fromarray(np.full((40, 40, 3), 10, np.uint8), mode="RGB")
    recon_image = Image.fromarray(np.full((20, 20, 3), 200, np.uint8), mode="RGB")  # wrong size
    cropped_mask = _mask(size=40, box=(10, 10, 30, 30))
    out = paste_back(input_image, recon_image, cropped_image, cropped_mask, (10, 10), feather=0.0)
    assert out.size == (80, 80)  # recon resized to crop, no error


def test_paste_back_poisson_empty_mask_falls_back_to_paste():
    input_image = Image.fromarray(np.zeros((60, 60, 3), np.uint8), mode="RGB")
    cropped_image = Image.fromarray(np.full((30, 30, 3), 10, np.uint8), mode="RGB")
    recon_image = Image.fromarray(np.full((30, 30, 3), 200, np.uint8), mode="RGB")
    empty_mask = Image.fromarray(np.zeros((30, 30), np.uint8), mode="L")
    out = paste_back(
        input_image, recon_image, cropped_image, cropped_mask=empty_mask, offset=(15, 15), poisson_blend=True
    )
    # Empty enlarged mask -> the untouched cropped image is pasted (no reconstruction blended in).
    assert np.array(out)[30, 30, 0] == 10


def test_paste_back_poisson_blends_into_mask_region():
    input_image = Image.fromarray(np.zeros((80, 80, 3), np.uint8), mode="RGB")
    cropped_image = Image.fromarray(np.full((40, 40, 3), 10, np.uint8), mode="RGB")
    recon_image = Image.fromarray(np.full((40, 40, 3), 200, np.uint8), mode="RGB")
    cropped_mask = _mask(size=40, box=(12, 12, 28, 28))
    out = paste_back(input_image, recon_image, cropped_image, cropped_mask, offset=(20, 20), poisson_blend=True)
    # Seamless-clone runs and composites the crop back; output stays full-size and valid.
    assert out.size == (80, 80)
    assert out.mode == "RGB"


def test_annotate_returns_rgb_same_size():
    img = Image.fromarray(np.full((60, 60, 3), 100, np.uint8), mode="RGB")
    mask = _mask(size=60, box=(10, 10, 30, 30))
    out = annotate(img, mask, crop_offset=(5, 5), crop_size=(40, 40), label="crop")
    assert out.mode == "RGB"
    assert out.size == img.size


# --- the image/mask pair invariant ----------------------------------------------------------------


def test_best_crop_takes_its_extent_from_the_image_not_the_mask():
    # Regression: best_crop clamped against ``mask.size``, but the window was planned against the
    # IMAGE's extent, so a mask-derived clamp cut a window smaller than the plan that
    # fit_to_input_size then stretched onto the input size — wrong scale and aspect, no error.
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    mask = _mask(size=100, box=(40, 40, 60, 60))
    cropped_image, cropped_mask, _ = best_crop(img, mask, grid_x=64, grid_y=64)
    assert cropped_image.size == cropped_mask.size == (64, 64)


def test_best_crop_rejects_a_mismatched_pair():
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    with pytest.raises(ValueError, match="image/mask size mismatch"):
        best_crop(img, _mask(size=50, box=(10, 10, 20, 20)), grid_x=32, grid_y=32)


def test_paste_back_rejects_a_mismatched_crop():
    crop = Image.fromarray(np.full((40, 40, 3), 10, np.uint8), mode="RGB")
    recon = Image.fromarray(np.full((40, 40, 3), 200, np.uint8), mode="RGB")
    with pytest.raises(ValueError, match="image/mask size mismatch"):
        paste_back(
            input_image=Image.fromarray(np.zeros((100, 100, 3), np.uint8), mode="RGB"),
            recon_image=recon,
            cropped_image=crop,
            cropped_mask=_mask(size=20, box=(2, 2, 8, 8)),
            offset=(0, 0),
        )


@pytest.mark.parametrize("crop", [33, 34, 40, 64, 128, 211])
def test_paste_back_ramp_finishes_inside_a_small_crop(crop):
    """A crop small relative to the ramp would otherwise still be blending at its border, leaving
    the rectangular seam feathering exists to hide. Reachable under crop_mode="ratio"."""
    side = max(1, round(crop / MIN_CROP_RATIO))
    image = Image.fromarray(np.full((crop * 3, crop * 3, 3), 40, np.uint8), "RGB")
    arr = np.zeros((crop, crop), np.uint8)
    lo = (crop - side) // 2
    arr[lo : lo + side, lo : lo + side] = 255
    mask = Image.fromarray(arr, "L")
    off = (crop, crop)
    cropped = image.crop((off[0], off[1], off[0] + crop, off[1] + crop))
    recon = Image.fromarray(np.full((crop, crop, 3), 220, np.uint8), "RGB")

    out = np.asarray(paste_back(image, recon, cropped, mask, off).convert("RGB"), np.int16)
    inside = out[off[1], off[0] : off[0] + crop, 0]
    outside = out[off[1] - 1, off[0] : off[0] + crop, 0]
    assert np.abs(inside - outside).max() == 0, "composite steps at the crop border"
