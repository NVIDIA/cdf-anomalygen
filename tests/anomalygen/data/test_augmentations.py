# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for anomalygen.data.augmentations (paired train-time image/mask transforms).

RNG is injected so the probabilistic transforms are deterministic under test.
"""

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tF

from anomalygen.configs.texture.constants import DEFAULT_INPUT_SIZE_LADDER
from anomalygen.data.augmentations import (
    RandomHorizontalFlip,
    RandomInstanceDrop,
    RandomOrthogonalRotation,
    RandomRatioCrop,
    RandomRingJitter,
    RandomRotation,
    RandomScaleCrop,
    RandomVerticalFlip,
)


@pytest.mark.parametrize(
    "augmentation",
    [
        RandomRingJitter,
        RandomInstanceDrop,
        RandomVerticalFlip,
        RandomHorizontalFlip,
        RandomOrthogonalRotation,
        RandomRotation,
        RandomScaleCrop,
        RandomRatioCrop,
    ],
)
def test_every_augmentation_requires_an_explicit_rng(augmentation):
    """No augmentation may fall back to an unseeded generator: the caller's seed is the contract.

    An implicit default would still draw, so a run would stay reproducible-looking while quietly
    ignoring base_seed. Refusing to construct says so at the call site instead.
    """
    with pytest.raises(TypeError, match="rng"):
        augmentation()


def _asymmetric_image():
    # Distinct top-left marker so a flip is observable.
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[0, 0] = 255
    return Image.fromarray(arr, mode="RGB")


def test_horizontal_flip_applies_when_probability_certain(mask_with_square):
    img = _asymmetric_image()
    out_img, _ = RandomHorizontalFlip(p=1.0, rng=np.random.default_rng(0))(img, mask_with_square())
    # Top-left marker moves to top-right after an h-flip.
    assert np.array(out_img)[0, -1, 0] == 255
    assert np.array(out_img)[0, 0, 0] == 0


def test_horizontal_flip_skipped_when_probability_zero(mask_with_square):
    img = _asymmetric_image()
    out_img, _ = RandomHorizontalFlip(p=0.0, rng=np.random.default_rng(0))(img, mask_with_square())
    assert np.array_equal(np.array(out_img), np.array(img))


def test_vertical_flip_applies_when_probability_certain(mask_with_square):
    img = _asymmetric_image()
    out_img, _ = RandomVerticalFlip(p=1.0, rng=np.random.default_rng(0))(img, mask_with_square())
    # Top-left marker moves to bottom-left after a v-flip.
    assert np.array(out_img)[-1, 0, 0] == 255


def _thin_diagonal_mask(size=64):
    """1px-wide diagonal defect — the shape (crack / scratch / bridge) rotation most easily erodes."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for i in range(10, size - 10):
        arr[i, i] = 255
    return Image.fromarray(arr, mode="L")


class _FixedAngleRng:
    """np.random.Generator stand-in pinning RandomRotation to one known positive angle."""

    def __init__(self, angle: int) -> None:
        self._angle = angle

    def integers(self, low, high=None):
        return self._angle

    def random(self):
        return 0.0  # < 0.5 -> keep the positive angle


def test_random_rotation_preserves_thin_mask_area():
    angle = 13
    img = Image.fromarray(np.full((64, 64, 3), 120, np.uint8), mode="RGB")
    mask = _thin_diagonal_mask()
    area = int((np.array(mask) > 0).sum())

    _, out_mask = RandomRotation(max_angle=angle, p=1.0, rng=_FixedAngleRng(angle))(img, mask)
    out = np.array(out_mask)

    # NEAREST rotation: the mask stays binary and every defect pixel survives the turn.
    assert set(np.unique(out)).issubset({0, 255})
    assert int((out > 0).sum()) == area
    # Reference: the BILINEAR + 0.5-threshold path this replaced erodes the same thin defect.
    bilinear = np.array(tF.rotate(mask, angle, interpolation=transforms.InterpolationMode.BILINEAR))
    assert int((bilinear >= 128).sum()) < area


def test_flip_stack_preserves_size_and_mask_binary(mask_with_square):
    """The flips are standalone transforms; composing them is the caller's job."""
    rng = np.random.default_rng(1)
    img = Image.fromarray(np.full((64, 64, 3), 120, np.uint8), mode="RGB")
    mask = mask_with_square()
    out_img, out_mask = img, mask
    for aug in (
        RandomVerticalFlip(0.5, rng=rng),
        RandomHorizontalFlip(0.5, rng=rng),
        RandomOrthogonalRotation(rng=rng),
    ):
        out_img, out_mask = aug(out_img, out_mask)
    assert out_img.size == img.size
    assert out_mask.size == mask.size
    assert set(np.unique(np.array(out_mask))).issubset({0, 255})


# --- RandomScaleCrop -----------------------------------------------------------------------------


def test_random_scale_crop_requires_l_mode_mask():
    img = _asymmetric_image()
    with pytest.raises(ValueError):
        RandomScaleCrop(rng=np.random.default_rng(0))(img, img.convert("RGB"))


def test_random_scale_crop_rejects_a_nonpositive_scale_low():
    # The recipe validator checks the band, but a direct construction bypasses it; log(0) is -inf and
    # every draw would then be 0 with no error.
    with pytest.raises(ValueError, match="scale_range low must be > 0"):
        RandomScaleCrop(scale_range=(0.0, 2.0), rng=np.random.default_rng(0))


def test_random_scale_crop_rejects_an_empty_mask():
    # The dataset drops empty masks at scan time, so one reaching the crop was emptied by an earlier
    # stage; passing it through untouched would train a native-size sample with a zero defect loss.
    img = Image.fromarray(np.zeros((100, 100, 3), np.uint8), mode="RGB")
    empty_mask = Image.fromarray(np.zeros((100, 100), np.uint8), mode="L")
    with pytest.raises(ValueError, match="no defect pixels"):
        RandomScaleCrop(rng=np.random.default_rng(2))(img, empty_mask)


def test_random_scale_crop_emits_the_pair_already_at_input_size(mask_with_square):
    rng = np.random.default_rng(3)
    img = Image.fromarray(np.full((100, 100, 3), 50, np.uint8), mode="RGB")
    mask = mask_with_square(size=100, lo=40, hi=60)
    crop = RandomScaleCrop(
        crop_ratio_range=(1.5, 4.0), scale_range=(0.5, 4.0), input_size_ladder=(64, 128, 256), rng=rng
    )
    for _ in range(20):
        out_img, out_mask = crop(img, mask)
        # The dataset must not resize afterwards, so the transform owns the final size: image and
        # mask come back identically sized, on ladder sizes, per axis.
        assert out_img.size == out_mask.size
        assert out_img.size[0] in (64, 128, 256) and out_img.size[1] in (64, 128, 256)
        assert int((np.array(out_mask) > 0).sum()) > 0  # the defect is never cropped away


def test_random_scale_crop_input_may_be_rectangular_for_an_elongated_defect():
    """Axes are planned independently, so an elongated defect gets an elongated input.

    Forcing a square here would squeeze the crop and distort the defect's aspect ratio.
    """
    rng = np.random.default_rng(0)
    img = Image.fromarray(np.zeros((400, 1600, 3), np.uint8), mode="RGB")  # wide frame
    m = np.zeros((400, 1600), np.uint8)
    m[190:210, 400:1200] = 255  # 800x20 defect
    mask = Image.fromarray(m, mode="L")

    crop = RandomScaleCrop(
        crop_ratio_range=(1.5, 1.5), scale_range=(1.0, 1.0), input_size_ladder=DEFAULT_INPUT_SIZE_LADDER, rng=rng
    )
    w, h = crop(img, mask)[0].size
    assert w > h, "the wide defect must get a wide input, not a square"


def test_random_ratio_crop_pins_one_input_size_on_a_single_entry_ladder(mask_with_square):
    # How a recipe asks for a fixed model input size: every sample, at any drawn framing, comes back
    # on exactly that size — which is what makes the batch shape constant.
    rng = np.random.default_rng(5)
    img = Image.fromarray(np.full((300, 300, 3), 50, np.uint8), mode="RGB")
    mask = mask_with_square(size=300, lo=100, hi=140)
    crop = RandomRatioCrop(crop_ratio_range=(1.5, 8.0), input_size_ladder=(128,), rng=rng)

    for _ in range(20):
        out_img, out_mask = crop(img, mask)
        assert out_img.size == out_mask.size == (128, 128)
        assert int((np.array(out_mask) > 0).sum()) > 0  # the defect is never cropped away


def test_random_ratio_crop_frames_off_the_bbox_not_the_pixel_size(mask_with_square):
    # The distinguishing property of "ratio" mode: at a pinned ratio, defects of very different
    # pixel sizes fill the same fraction of the input. Under "scale" mode they would not.
    img = Image.fromarray(np.full((1000, 1000, 3), 50, np.uint8), mode="RGB")
    fractions = []
    for side in (20, 200):
        mask = mask_with_square(size=1000, lo=400, hi=400 + side)
        crop = RandomRatioCrop(crop_ratio_range=(4.0, 4.0), input_size_ladder=(256,), rng=np.random.default_rng(0))
        out_mask = crop(img, mask)[1]
        fractions.append(int((np.array(out_mask) > 0).sum()) / (256 * 256))
    assert fractions[0] == pytest.approx(fractions[1], rel=0.1)


def test_the_shared_scale_keeps_a_square_defect_square(mask_with_square):
    # One scale for both axes, so a square defect renders square whatever the draw. The single-size
    # ladder keeps the stochastic size snap out of it.
    img = Image.fromarray(np.zeros((2000, 2000, 3), np.uint8), mode="RGB")
    mask = mask_with_square(size=2000, lo=800, hi=860)
    crop = RandomScaleCrop(
        crop_ratio_range=(1.5, 4.0), scale_range=(0.333, 3.0), input_size_ladder=(256,), rng=np.random.default_rng(0)
    )
    for _ in range(20):
        arr = np.array(crop(img, mask)[1])
        ys, xs = np.nonzero(arr >= 128)
        assert (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) == pytest.approx(1.0, abs=0.1)


def test_random_scale_crop_honours_a_per_class_ladder_override(mask_with_square):
    # The per-class input-size cap is applied by passing a truncated ladder for this sample only.
    rng = np.random.default_rng(4)
    img = Image.fromarray(np.full((512, 512, 3), 50, np.uint8), mode="RGB")
    mask = mask_with_square(size=512, lo=200, hi=260)
    crop = RandomScaleCrop(scale_range=(1.0, 4.0), input_size_ladder=(64, 128, 256, 512), rng=rng)
    for _ in range(20):
        out_img, _ = crop(img, mask, ladder=(64, 128))
        assert max(out_img.size) <= 128
