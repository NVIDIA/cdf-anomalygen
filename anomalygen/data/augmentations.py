# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Paired (image, mask) augmentations for I2I anomaly inpainting. Operates on PIL Image
# pairs so the mask stays aligned with the image through every transform.

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tF
from torchvision.transforms import v2

from anomalygen.configs.texture.constants import (
    DEFAULT_CROP_RATIO_RANGE,
    DEFAULT_INPUT_SIZE_LADDER,
    DEFAULT_SCALE_RANGE,
)
from anomalygen.data.crop_geometry import apply_crop_plan, mask_bbox_or_none, plan_window
from anomalygen.data.utils import MASK_FG_THRESHOLD, require_same_size


def _fires(rng, p: float) -> bool:
    """Sample a probability gate, drawing ONLY when the outcome is uncertain.

    ``p <= 0`` and ``p >= 1`` short-circuit without touching ``rng``, so turning an augmentation
    fully off or fully on leaves the random stream — and therefore every other augmentation's
    draws — byte-identical to a run without it.
    """
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(rng.random() < p)


class RandomRingJitter:
    """Colour-jitter a band AROUND every defect; the defect pixels themselves stay untouched.

    Motivation: training crops can carry a colour fringe a few pixels from the defect edge that the
    held-out images do not, a leak the model would otherwise absorb as part of "what a defect looks
    like". Recolouring the band breaks the correlation. Because the band lies outside the mask it
    never enters the defect-only training loss — it only changes what the source item SHOWS.

    ``band_px`` is a FIXED width in ORIGINAL-image pixels, not a fraction of the instance: the leak
    sits a fixed distance from the edge, so it does not scale with defect size. One dilation of the
    whole mask suffices — dilation is a union, so this equals dilating each instance separately.
    The band is recoloured by a SINGLE jitter draw, so the whole fringe shifts together. Run BEFORE
    :class:`RandomInstanceDrop`, so bands are cut against the full mask.

    Returns ``mask`` unchanged; only the image is touched.
    """

    def __init__(self, p: float = 0.5, band_px: int = 10, *, rng: np.random.Generator) -> None:
        self.p = float(p)
        self.band_px = int(band_px)
        self.rng = rng
        # ``hue`` spans the full circle on purpose: the goal is to destroy the hue cue, not nudge it.
        self.photometric = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5)

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.band_px <= 0 or not _fires(self.rng, self.p):
            return image, mask
        m = (np.asarray(mask) >= MASK_FG_THRESHOLD).astype(np.uint8)
        r = self.band_px
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ring = (cv2.dilate(m, kern) > 0) & (m == 0)  # subtract the mask: defect pixels never recoloured
        ys, xs = np.nonzero(ring)
        if xs.size == 0:
            return image, mask
        arr = np.asarray(image).copy()
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        patch = torch.from_numpy(arr[y0:y1, x0:x1].astype(np.float32) / 255.0).permute(2, 0, 1)
        jittered = self.photometric(patch).clamp(0.0, 1.0)
        jittered = (jittered.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        sel = ring[y0:y1, x0:x1]
        arr[y0:y1, x0:x1][sel] = jittered[sel]
        return Image.fromarray(arr), mask


class RandomInstanceDrop:
    """Keep one random connected instance of the mask; black the others out in the IMAGE too.

    A no-op when the mask has 0 or 1 instances, so its effective rate is the multi-instance fraction
    of the dataset rather than ``p``. Motivation: inference inpaints ONE instance at a time, while
    training otherwise asks for every defect at once. Instances use the same threshold and
    8-connectivity as the loss and the inference-side splitter, so "one defect" means the same thing
    everywhere.

    The blackout covers each dropped instance PLUS a ``band_px`` margin, unconditionally: it removes
    a jittered ring that would else remain as a coloured halo, and covers the instance edge so no
    trace of a dropped defect survives unmarked.
    """

    def __init__(self, p: float = 0.5, band_px: int = 10, *, rng: np.random.Generator) -> None:
        self.p = float(p)
        self.band_px = int(band_px)
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image):
        if not _fires(self.rng, self.p):
            return image, mask
        m = (np.asarray(mask) >= MASK_FG_THRESHOLD).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(m, connectivity=8)
        if n_labels <= 2:  # background + at most one instance
            return image, mask
        keep = int(self.rng.integers(1, n_labels))
        dropped = ((labels > 0) & (labels != keep)).astype(np.uint8)
        r = self.band_px
        if r > 0:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            dropped = cv2.dilate(dropped, kern)
        dropped = (dropped > 0) & (labels != keep)
        arr = np.asarray(image).copy()
        arr[dropped] = 0  # black in the IMAGE too, so the pair keeps no unmarked defect
        kept_mask = np.where(labels == keep, 255, 0).astype(np.uint8)
        return Image.fromarray(arr), Image.fromarray(kept_mask, mode="L")


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5, *, rng: np.random.Generator) -> None:
        self.p = p
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if _fires(self.rng, self.p):
            image, mask = tF.vflip(image), tF.vflip(mask)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5, *, rng: np.random.Generator) -> None:
        self.p = p
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if _fires(self.rng, self.p):
            image, mask = tF.hflip(image), tF.hflip(mask)
        return image, mask


class RandomOrthogonalRotation:
    """Rotate the pair by a random quarter turn (0 / 90 / 180 / 270 degrees).

    Lossless and border-free: PIL fast-paths exact quarter turns to a transpose, so there is no
    interpolation (the mask stays binary without re-thresholding) and no black corners to crop away.
    That is why this is applied to EVERY sample, unlike the opt-in small-angle
    :class:`RandomRotation`. Note 90/270 swap width and height, so a non-square input comes back
    transposed.
    """

    def __init__(self, *, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        angle = int(self.rng.integers(0, 4)) * 90
        if angle == 0:
            return image, mask
        # expand=True keeps the full frame on 90/270 (and triggers PIL's exact transpose path).
        image = tF.rotate(image, angle, expand=True)
        mask = tF.rotate(mask, angle, expand=True)
        return image, mask


class RandomRotation:
    """Small-angle rotation; NEAREST keeps the mask binary without re-thresholding.

    ``p`` defaults to 0: a free rotation leaves black corners and can swing a small edge defect off
    canvas, so it is opt-in and must be followed by a crop that trims the corners away. The lossless
    quarter turn (:class:`RandomOrthogonalRotation`) is the one applied by default.
    """

    def __init__(self, max_angle: int = 20, p: float = 0.0, *, rng: np.random.Generator) -> None:
        self.max_angle = int(max_angle)
        self.p = float(p)
        self.rng = rng

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not _fires(self.rng, self.p):
            return image, mask
        angle = int(self.rng.integers(0, self.max_angle + 1))
        angle = angle if self.rng.random() < 0.5 else -angle
        image = tF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        # NEAREST keeps the mask binary. Rotation is area-preserving, so a thin defect survives it
        # (unlike a downsample), though it can pick up staircase gaps.
        mask = tF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask


class _MaskCentredCrop:
    """Shared plumbing for the two crop stages: the framing band, the ladder, and the mask bbox.

    Each stage owns the input-size contract, so it runs on every sample: skipping it would emit the
    pair at its native size, off the ladder and mis-tokenised rather than rejected. The band and
    the ladder arrive checked (see :func:`validate_geometry`); this stage trusts them.
    """

    CROP_MODE: str  # set by each subclass: the ``crop_mode`` this stage is the training half of

    def __init__(
        self,
        crop_ratio_range: Tuple[float, float] = DEFAULT_CROP_RATIO_RANGE,
        input_size_ladder: Sequence[int] = DEFAULT_INPUT_SIZE_LADDER,
        *,
        rng: np.random.Generator,
    ) -> None:
        self.crop_ratio_range = (float(crop_ratio_range[0]), float(crop_ratio_range[1]))
        self.input_size_ladder = tuple(int(v) for v in input_size_ladder)
        self.rng = rng

    def _bbox(self, image: Image.Image, mask: Image.Image):
        """Validate the pair and return the mask's bbox.

        Raises on an empty mask: the dataset drops those at scan time, so one arriving here was
        emptied by an earlier stage, and training on it would teach nothing while the loss reads 0.
        """
        if mask.mode != "L":
            raise ValueError(f"{type(self).__name__} expects an 'L'-mode mask, got {mask.mode!r}")
        require_same_size(image, mask, f"the pair handed to {type(self).__name__}")
        box = mask_bbox_or_none(mask)
        if box is None:
            raise ValueError(
                f"{type(self).__name__}: the mask has no defect pixels; an earlier augmentation stage emptied it."
            )
        return box

    def _draw_ratio(self) -> float:
        """Linear-uniform draw from the framing band."""
        lo, hi = self.crop_ratio_range
        return lo + float(self.rng.random()) * (hi - lo)

    def _plan(self, image, bbox_wh, ctx, scale, ladder: Optional[Sequence[int]]):
        """Plan this sample's crop. Generation plans through the same :func:`plan_window`, which is
        what keeps the two sides' framing identical. ``ladder`` overrides the configured one for this
        sample (the per-class input-size cap)."""
        return plan_window(
            *bbox_wh,
            *image.size,
            ctx=ctx,
            scale=scale,
            ladder=self.input_size_ladder if ladder is None else ladder,
            ratio_floor=self.crop_ratio_range[0],  # never frame tighter than the band the model trains on
            crop_mode=self.CROP_MODE,
            rng=self.rng,
        )


class RandomScaleCrop(_MaskCentredCrop):
    """Mask-centered cut-out on two independent axes: context ratio and resample scale.

    ``crop_ratio_range`` is the FRAMING axis (how much context surrounds the defect); ``scale_range`` is
    the SCALE axis (resampling as augmentation, which is what teaches defect-size invariance).

    The framing axis is drawn linear-uniformly; the SCALE axis is drawn LOG-uniformly, because scale
    is multiplicative and a linear draw over-weights the upscale half of the band — which would sit
    training above ``inference_scale`` (1.0) and send generation to input sizes training rarely saw.
    A log draw is symmetric about the geometric mean, so 1x sits mid-band and both halves are covered.

    Emits the pair already at input size; the dataset must not resize afterwards.
    """

    CROP_MODE = "scale"

    def __init__(
        self,
        crop_ratio_range: Tuple[float, float] = DEFAULT_CROP_RATIO_RANGE,
        scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
        input_size_ladder: Sequence[int] = DEFAULT_INPUT_SIZE_LADDER,
        *,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(crop_ratio_range, input_size_ladder, rng=rng)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        if self.scale_range[0] <= 0:
            raise ValueError(f"scale_range low must be > 0, got {self.scale_range[0]}: its log is the draw's floor.")
        # Precomputed log bounds for the scale draw (see __call__).
        self._log_s_lo = float(np.log(self.scale_range[0]))
        self._log_s_span = float(np.log(self.scale_range[1])) - self._log_s_lo

    def __call__(
        self, image: Image.Image, mask: Image.Image, ladder: Optional[Sequence[int]] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """``ladder`` overrides the configured one for this sample — used to apply the per-class
        input-size cap, so a class of small defects never draws an input sized for a large one."""
        x1, y1, x2, y2 = self._bbox(image, mask)

        ctx = self._draw_ratio()
        # Log-uniform from one draw, the same as a linear form would take, so the choice does not
        # perturb what any other stage consumes from the shared generator.
        s = float(np.exp(self._log_s_lo + float(self.rng.random()) * self._log_s_span))

        plan = self._plan(image, (x2 - x1 + 1, y2 - y1 + 1), ctx, s, ladder)
        return apply_crop_plan(image, mask, plan, (x1, y1, x2, y2), self.rng)


class RandomRatioCrop(_MaskCentredCrop):
    """Mask-centered zoom-crop on the framing axis alone: the window is ``max(bbox) * ratio``.

    The crop stage of ``crop_mode="ratio"``. There is no scale draw — the defect fills 1/ratio of the
    input whatever its pixel size, so with a single-entry ladder every sample lands on the same fixed
    input size.

    Emits the pair already at input size; the dataset must not resize afterwards.
    """

    CROP_MODE = "ratio"

    def __call__(
        self, image: Image.Image, mask: Image.Image, ladder: Optional[Sequence[int]] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """``ladder`` overrides the configured one for this sample (the per-class input-size cap)."""
        x1, y1, x2, y2 = self._bbox(image, mask)

        # scale=1.0 is a placeholder: "ratio" mode has no scale axis and plan_window ignores it.
        plan = self._plan(image, (x2 - x1 + 1, y2 - y1 + 1), self._draw_ratio(), 1.0, ladder)
        return apply_crop_plan(image, mask, plan, (x1, y1, x2, y2), self.rng)
