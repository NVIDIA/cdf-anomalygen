# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crop geometry shared by training and generation.

One planner decides, for a defect, what window to cut from the image and what input size to fit it
onto; training and generation both call it, so their framing cannot drift.

Two framing policies, selected by ``crop_mode`` and dispatched by :func:`plan_window`: "scale"
(:func:`plan_crop`, context + resample axes) and "ratio" (:func:`plan_crop_ratio`, one bbox-relative
axis). Both snap to ``input_size_ladder``, so the mode and the input sizes stay independent.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from typing import NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from anomalygen.configs.texture.constants import (
    CROP_MODES,
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_RATIO_RANGE,
    DEFAULT_INPUT_SIZE_LADDER,
    DEFAULT_SCALE_RANGE,
    MAX_INPUT_SIZE,
    MIN_CROP_RATIO,
    VISION_TOKEN_PX,
)
from anomalygen.data.utils import MASK_FG_THRESHOLD, require_same_size


def mask_bbox_or_none(mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Tight ``(left, upper, right, lower)`` box of the mask's foreground, ``None`` if empty."""
    arr = np.asarray(mask if mask.mode == "L" else mask.convert("L"))
    ys, xs = np.nonzero(arr >= MASK_FG_THRESHOLD)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_bbox(mask: Image.Image) -> Tuple[int, int, int, int]:
    """:func:`mask_bbox_or_none`, raising on an empty mask."""
    box = mask_bbox_or_none(mask)
    if box is None:
        raise ValueError("No mask region found in input mask.")
    return box


def normalize_input_size_ladder(ladder: Sequence[int]) -> Tuple[int, ...]:
    """Ascending, deduplicated, validated sizes. Raises so a bad recipe fails at parse, not in a worker.

    Sizes must be multiples of :data:`VISION_TOKEN_PX` (one vision token) or the input cannot be
    tokenized, and must stay within :data:`MAX_INPUT_SIZE`. A repeated size is dropped: the geometry
    stamp compares ladders as lists, so ``[512, 512]`` would otherwise refuse to resume a ``[512]``
    checkpoint.
    """
    sizes = sorted({int(v) for v in ladder})
    if not sizes or any(v < VISION_TOKEN_PX or v % VISION_TOKEN_PX for v in sizes):
        raise ValueError(
            f"input_size_ladder must be non-empty positive multiples of {VISION_TOKEN_PX}, got {list(ladder)}."
        )
    if sizes[-1] > MAX_INPUT_SIZE:
        raise ValueError(f"input_size_ladder tops out at {sizes[-1]}, above the {MAX_INPUT_SIZE} ceiling.")
    return tuple(sizes)


def ceil_input_size(ideal: float, ladder: Sequence[int]) -> int:
    """Smallest entry of the ascending ``ladder`` that is >= ``ideal``, or its largest when none is."""
    return ladder[min(bisect.bisect_left(ladder, ideal), len(ladder) - 1)]


class Geometry(NamedTuple):
    """A recipe's crop geometry, normalised and checked once by :func:`validate_geometry`."""

    crop_mode: str
    input_size_ladder: Tuple[int, ...]
    crop_ratio_range: Tuple[float, float]
    scale_range: Tuple[float, float]


def validate_geometry(
    crop_mode: str,
    input_size_ladder: Sequence[int],
    crop_ratio_range: Sequence[float],
    scale_range: Sequence[float] = DEFAULT_SCALE_RANGE,
    inference_scale: Optional[float] = None,
) -> Geometry:
    """Normalise and check a recipe's crop geometry in one place.

    Called by the recipe builder and the training dataset, so a bad recipe fails at parse rather
    than in a DataLoader worker; the crop stages and the planner trust what comes out.

    ``crop_ratio_range`` floors at :data:`MIN_CROP_RATIO` — a tighter window leaves too little
    context to place a coherent defect — and is open at the top so training sees more than one
    framing. ``scale_range`` is a resample factor and may sit below 1. Under "scale" mode
    ``inference_scale``, when given, must sit inside ``scale_range``, or generation would resample
    every crop to a scale the model never saw.
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}.")
    ladder = normalize_input_size_ladder(input_size_ladder)
    ratio = tuple(float(v) for v in crop_ratio_range)
    if len(ratio) != 2 or not MIN_CROP_RATIO <= ratio[0] < ratio[1]:
        raise ValueError(
            f"crop_ratio_range must be [low, high] with {MIN_CROP_RATIO} <= low < high, got {list(crop_ratio_range)}. "
            "A pinned band (low == high) gives training a single framing; widen high instead."
        )
    scale = tuple(float(v) for v in scale_range)
    if len(scale) != 2 or not 0 < scale[0] <= scale[1]:
        raise ValueError(f"scale_range must be [low, high] with 0 < low <= high, got {list(scale_range)}.")
    if inference_scale is not None and crop_mode == "scale" and not scale[0] <= float(inference_scale) <= scale[1]:
        raise ValueError(
            f"inference_scale {inference_scale} is outside the trained scale_range {list(scale)}; "
            "generation would resample every crop to a scale the model never saw."
        )
    return Geometry(crop_mode, ladder, ratio, scale)


def snap_input_size(ideal: float, ladder: Sequence[int], rng: Optional[np.random.Generator] = None) -> int:
    """Choose an input size for ``ideal``.

    ``rng=None`` -> nearest size, deterministic. Used at INFERENCE so a testcase always produces the
    same geometry.

    ``rng`` given -> stochastic between the two neighbouring sizes, with probability proportional to
    proximity (an ideal just above a size lands on it almost always; one halfway is a coin flip).
    Used at TRAINING so the model sees both roundings of any given framing instead of a hard,
    deterministic quantisation of the ratio axis.
    """
    if ideal <= ladder[0]:
        return ladder[0]
    if ideal >= ladder[-1]:
        return ladder[-1]
    hi_i = bisect.bisect_left(ladder, ideal)
    lo, hi = ladder[hi_i - 1], ladder[hi_i]
    p_hi = (ideal - lo) / (hi - lo)  # 0 at lo, 1 at hi
    if rng is None:
        return lo if p_hi <= 0.5 else hi
    return hi if float(rng.random()) < p_hi else lo


class CropPlan(NamedTuple):
    """Geometry for one sample, per axis. The cut, the input it is fitted onto, and the resampling."""

    crop_w: int  # native window to cut (real pixels, already clamped to the image)
    crop_h: int
    input_w: int  # final tensor size, a ladder entry
    input_h: int
    content_w: int  # size the cut occupies inside the input after scaling
    content_h: int
    # Effective resample factor, one for both axes. Reported for inspection — the resampling itself
    # is driven by content_w/content_h.
    scale: float

    @property
    def needs_pad(self) -> bool:
        return (self.content_w, self.content_h) != (self.input_w, self.input_h)


def plan_crop(
    bbox_w: int,
    bbox_h: int,
    img_w: int,
    img_h: int,
    ctx: float,
    s: float,
    ladder: Sequence[int] = DEFAULT_INPUT_SIZE_LADDER,
    ratio_floor: float = DEFAULT_CROP_RATIO_RANGE[0],
    rng: Optional[np.random.Generator] = None,
) -> CropPlan:
    """Plan one crop from the two axes, per axis. Single source of truth for training and inference.

    INVARIANT: ``crop_w >= bbox_w and crop_h >= bbox_h``, always — a window that cannot hold the mask
    makes the model inpaint half a defect. Every other quantity yields to it.

    When the input-size ceiling binds, in order: honour ``s`` -> shrink ``ctx`` toward ``ratio_floor``
    -> only then reduce ``s``. An unseen scale costs sharpness; an unseen framing costs structure.

    The ladder FLOOR binds the other way, and by design: a defect small enough that ``need * s`` lands
    under the smallest size snaps up to it, and ``ideal_crop`` re-derives the window from that size — so
    the realized context ratio ends up ABOVE ``ctx``, well outside ``crop_ratio_range``. Small defects
    therefore train at the smallest size with extra surrounding context rather than at a smaller input
    size. Inference plans through this same function, so the framing still matches training.

    ``ratio_floor`` holds on the realized window too. ``snap_input_size`` rounds to the nearest size,
    so a size that would leave the mask less than ``ratio_floor`` of context is lifted to the next
    one — unless the image itself is smaller, in which case the window keeps what the frame has.

    Axes are planned independently (elongated defects need elongated inputs) but share one ``ctx``
    and one ``s``, so the defect keeps its aspect.
    """
    ladder = sorted(int(v) for v in ladder)
    input_max = ladder[-1]
    ctx, s = float(ctx), float(s)
    if s <= 0:
        raise ValueError(f"scale must be > 0, got {s}")
    bbox_w, bbox_h = max(1, int(bbox_w)), max(1, int(bbox_h))
    img_w, img_h = max(1, int(img_w)), max(1, int(img_h))
    ratio_floor = max(1.0, float(ratio_floor))

    # Computed first because it bounds everything else: the largest s at which the defect still
    # fits the input size WITH the minimum context. Only bites when bbox * ratio_floor * s > input_max.
    # The window is capped by the image, so reserve input size for context that actually exists — a defect
    # spanning most of the frame cannot be given ratio_floor of it, and budgeting for it would snap the
    # input size one step lower and resample the crop for nothing.
    fit_w = min(bbox_w * ratio_floor, float(img_w))
    fit_h = min(bbox_h * ratio_floor, float(img_h))
    # One scale for both axes: a ceiling that binds on one must not break aspect.
    s_eff = min(s, input_max / fit_w, input_max / fit_h)

    axes = []
    for bb, img, fit in ((bbox_w, img_w, fit_w), (bbox_h, img_h, fit_h)):
        # Widest context this axis can afford at s_eff, floored at ratio_floor.
        ctx_axis = max(ratio_floor, min(ctx, input_max / (bb * s_eff)))
        need = min(max(bb * ctx_axis, float(bb)), float(img))  # real pixels; never below the bbox

        input_size = snap_input_size(need * s_eff, ladder, rng)
        # Snapping DOWN must not push the window under the floored one: ceil to the smallest size that
        # holds ``fit`` (bbox * ratio_floor, or the image when that is smaller). In exact arithmetic
        # fit * s_eff <= input_max, but s_eff is a float quotient and can overshoot the largest size by
        # an ulp, so compare with a tolerance; ceil_input_size falls back to that size.
        if input_size / s_eff < fit:
            input_size = ceil_input_size(fit * s_eff - 1e-6, ladder)

        ideal_crop = int(round(input_size / s_eff))
        crop = max(min(ideal_crop, img), bb)  # real pixels only (bb <= img, so the floor survives)
        # Pad only when the IMAGE ran out. round(round(input_size/s)*s) is not an identity, so deriving
        # content from the rounded crop would report a 1-2px shortfall on a window the image filled
        # completely and paste a black stripe along the input edge.
        content = input_size if crop >= ideal_crop else max(1, min(input_size, int(round(crop * s_eff))))
        axes.append((crop, input_size, content))

    (cw, kw, tw), (ch, kh, th) = axes
    return CropPlan(cw, ch, kw, kh, tw, th, s_eff)


def plan_crop_ratio(
    bbox_w: int,
    bbox_h: int,
    img_w: int,
    img_h: int,
    ctx: float,
    ladder: Sequence[int] = DEFAULT_INPUT_SIZE_LADDER,
    rng: Optional[np.random.Generator] = None,
) -> CropPlan:
    """Plan one crop from the FRAMING axis alone: a square ``max(bbox) * ctx`` window.

    The defect fills 1/``ctx`` of the input whatever its pixel size — there is no scale axis, so
    ``ctx`` alone sets the framing. The window is clamped per axis to the image (a window wider than
    the frame comes back rectangular), then each axis snaps to ``ladder``.

    A single-entry ladder therefore pins one fixed input size and stretches the cut onto it; a longer
    ladder keeps each axis near its native pixel size instead. ``content == input`` either way, so the
    cut is always resized, never padded.

    Nothing floors the window, so a small enough defect cuts a handful of pixels and is magnified
    hard onto the input. Deliberate: framing stays bbox-relative at EVERY defect size, which is what
    separates this mode from :func:`plan_crop` (where the ladder floor widens a small defect's window
    instead). Generation plans the same window, so the model is asked for the magnification it
    trained on — the resolution is lost, but the two sides do not disagree.

    Same invariant as :func:`plan_crop`: ``crop_w >= bbox_w and crop_h >= bbox_h``, always.
    """
    ladder = sorted(int(v) for v in ladder)
    ctx = float(ctx)
    if ctx <= 0:
        raise ValueError(f"crop ratio must be > 0, got {ctx}")
    bbox_w, bbox_h = max(1, int(bbox_w)), max(1, int(bbox_h))
    img_w, img_h = max(1, int(img_w)), max(1, int(img_h))

    # Both axes are framed off the SAME bbox extent, so the window is square until the image clamps it.
    base = max(bbox_w, bbox_h)

    axes = []
    for bb, img in ((bbox_w, img_w), (bbox_h, img_h)):
        side = int(round(base * ctx))
        # Clamped to the image, never below the mask — the latter is also what keeps a sub-1px
        # request (a ratio below 1 on a 1px bbox) from producing an empty window.
        crop = max(min(side, img), bb)
        axes.append((crop, snap_input_size(crop, ladder, rng)))

    (cw, kw), (ch, kh) = axes
    return CropPlan(cw, ch, kw, kh, kw, kh, kw / cw)


def plan_window(
    bbox_w: int,
    bbox_h: int,
    img_w: int,
    img_h: int,
    *,
    ctx: float,
    scale: float,
    ladder: Sequence[int],
    ratio_floor: float,
    crop_mode: str = DEFAULT_CROP_MODE,
    rng: Optional[np.random.Generator] = None,
) -> CropPlan:
    """Plan one crop under ``crop_mode``. The one entry point training and generation share.

    ``scale`` and ``ratio_floor`` apply to "scale" mode only; "ratio" mode frames off ``ctx`` alone.
    ``crop_mode`` is trusted: :func:`validate_geometry` checks it when the recipe is read and
    :func:`input_size_spec_from_config` when a model's config is.
    """
    if crop_mode == "ratio":
        return plan_crop_ratio(bbox_w, bbox_h, img_w, img_h, ctx, ladder, rng)
    return plan_crop(bbox_w, bbox_h, img_w, img_h, ctx, scale, ladder, ratio_floor=ratio_floor, rng=rng)


def place_window(b_lo: int, b_hi: int, crop: int, extent: int, rng: Optional[np.random.Generator] = None) -> int:
    """Window origin on one axis. Shared by training and inference so placement cannot drift.

    Containment is enforced by construction: the origin is confined to the range that keeps the whole
    mask extent ``[b_lo, b_hi]`` inside the window AND the window inside ``[0, extent]``. Deriving it
    from the centroid instead loses a pixel whenever the extent is even and ``crop == bbox``.

    ``rng`` given -> uniform over that range (training jitter). ``rng=None`` -> the centroid
    placement, clamped into it (deterministic at inference).

    A window too small to hold the mask cannot satisfy containment; it centres on the mask instead,
    so the crop still shows the defect. Defensive: every production window comes from ``plan_crop``,
    which never produces one.
    """
    if crop < b_hi - b_lo + 1:
        return max(0, min((b_lo + b_hi) // 2 - crop // 2, max(0, extent - crop)))
    lo = max(0, b_hi - crop + 1)
    hi = min(b_lo, extent - crop)
    if hi < lo:  # window wider than the image (a direct caller only; plan_crop clamps to it)
        return 0
    if rng is not None:
        return int(rng.integers(lo, hi + 1))
    return max(lo, min((b_lo + b_hi) // 2 - crop // 2, hi))


def resize_mask(mask: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize a binary mask, keeping every output pixel that touches foreground.

    NEAREST samples one source pixel per output pixel, so shrinking drops a defect thinner than the
    downsample factor — entirely, or into fragments the image still shows. Area coverage then
    "any non-zero" keeps it, matching how the mask is later pooled down to latent resolution.
    Resizes in float: a uint8 intermediate rounds a small coverage fraction back to 0.

    INTER_AREA covers area only when both axes shrink, so a mixed resize shrinks first and grows
    second. Always returns a 0/255 "L" mask, the identity case included.
    """
    src = (np.asarray(mask.convert("L")) >= MASK_FG_THRESHOLD).astype(np.float32)
    w, h = mask.size
    mid = (min(size[0], w), min(size[1], h))
    out = cv2.resize(src, mid, interpolation=cv2.INTER_AREA) if mid != (w, h) else src
    if mid != size:
        out = cv2.resize(out, size, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(np.where(out > 0, 255, 0).astype(np.uint8), "L")


def fit_to_input_size(
    image: Image.Image, mask: Image.Image, plan: CropPlan
) -> Tuple[Image.Image, Image.Image, Optional[Tuple[int, int, int, int]]]:
    """Scale an already-cut pair to the plan's content size and pad it to the input size.

    Shared by training (:func:`apply_crop_plan`) and generation, so the two cannot drift.

    Pads the image with 0 (black), which the dataset's [-1,1] normalisation maps to -1, and the mask
    with 0, so the defect-region-only loss ignores padded pixels entirely.

    Returns ``(image, mask, content_box)``; ``content_box`` locates the real pixels inside the input
    so a caller can strip the padding back off, and is ``None`` when the input is all real content.
    """
    require_same_size(image, mask, "the cut handed to fit_to_input_size")
    if (image.width, image.height) != (plan.content_w, plan.content_h):
        image = image.resize((plan.content_w, plan.content_h), Image.Resampling.BICUBIC)
        mask = resize_mask(mask, (plan.content_w, plan.content_h))

    if not plan.needs_pad:
        return image, mask, None

    # The image ran out before the input was filled: centre the real content and pad the rest.
    out_i = Image.new("RGB", (plan.input_w, plan.input_h), (0, 0, 0))
    out_m = Image.new("L", (plan.input_w, plan.input_h), 0)
    off = ((plan.input_w - plan.content_w) // 2, (plan.input_h - plan.content_h) // 2)
    out_i.paste(image, off)
    out_m.paste(mask, off)
    return out_i, out_m, (off[0], off[1], off[0] + plan.content_w, off[1] + plan.content_h)


def apply_crop_plan(
    image: Image.Image,
    mask: Image.Image,
    plan: CropPlan,
    bbox: Tuple[int, int, int, int],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Image.Image, Image.Image]:
    """Cut the planned window, then fit it to the input size.

    ``bbox`` is the inclusive mask extent ``(x1, y1, x2, y2)``; the window is placed to contain it.
    """
    require_same_size(image, mask, "the pair handed to apply_crop_plan")
    W, H = image.size
    crop_w, crop_h = plan.crop_w, plan.crop_h

    x1 = place_window(bbox[0], bbox[2], crop_w, W, rng)
    y1 = place_window(bbox[1], bbox[3], crop_h, H, rng)
    cut_img = image.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    cut_msk = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))

    out_i, out_m, _ = fit_to_input_size(cut_img, cut_msk, plan)
    return out_i, out_m


class InputSizeSpec(NamedTuple):
    """How a native cut-out becomes the input fed to the model at generation time.

    ``crop_ratio_range`` is the FRAMING band training sampled; inference clamps its requested context
    ratio into it. ``scale`` is the inference value of the SCALE axis; 1.0 means generate at native
    resolution, and training samples it over ``scale_range``, so any value inside that range is
    in-distribution. Under ``crop_mode="ratio"`` there is no scale axis and ``scale`` is unused.
    ``class_ladders`` are the per-class caps training applied to ``input_size_ladder`` (see
    :func:`ladder_for`); ``None`` plans every class against the full ladder.
    """

    input_size_ladder: Tuple[int, ...] = DEFAULT_INPUT_SIZE_LADDER
    scale: float = 1.0
    crop_ratio_range: Tuple[float, float] = DEFAULT_CROP_RATIO_RANGE
    crop_mode: str = DEFAULT_CROP_MODE
    class_ladders: Optional[Mapping[str, Tuple[int, ...]]] = None


def ladder_for(input_spec: InputSizeSpec, anomaly_name: str) -> Tuple[int, ...]:
    """The ladder to plan ``anomaly_name`` against: the cap its class trained under when the checkpoint
    carries one, else the full ladder."""
    if input_spec.class_ladders and anomaly_name in input_spec.class_ladders:
        return input_spec.class_ladders[anomaly_name]
    return input_spec.input_size_ladder


def input_size_spec_from_config(dec, class_ladders: Optional[Mapping[str, Sequence[int]]] = None) -> InputSizeSpec:
    """Read the input-size configuration off a model's ``diffusion_expert_config``.

    Training publishes the crop mode, ladder, inference scale and framing band it used, so generation
    inherits them rather than re-specifying them — training and inference disagreeing on geometry is
    a silent, expensive class of bug. ``dec`` is the attrs config of a built model or the mapping a
    recipe node carries. The crop mode is checked here as well as at recipe parse: a ``key=value``
    override composed after the parse reaches the model config unchecked.

    ``class_ladders`` are the per-class caps the checkpoint carries beside its stamp
    (``model.class_ladders``): dataset state, not config, so they are passed in rather than read off.
    """
    if dec is None:
        raise ValueError("model config has no diffusion_expert_config; cannot recover training geometry.")
    get = dec.get if isinstance(dec, Mapping) else lambda key, default=None: getattr(dec, key, default)
    crop_ratio_range = get("crop_ratio_range")
    crop_mode = get("crop_mode")  # no default: a config without one was not built by a current recipe
    if crop_mode not in CROP_MODES:
        raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}.")
    return InputSizeSpec(
        input_size_ladder=normalize_input_size_ladder(get("input_size_ladder")),
        scale=float(get("inference_scale")),
        crop_ratio_range=(float(crop_ratio_range[0]), float(crop_ratio_range[1])),
        crop_mode=str(crop_mode),
        class_ladders=(
            None if not class_ladders else {str(k): normalize_input_size_ladder(v) for k, v in class_ladders.items()}
        ),
    )


def clamp_crop_ratio(crop_ratio: float, input_spec: InputSizeSpec) -> float:
    """``crop_ratio`` clamped into the band training sampled — the only framing the model has seen."""
    lo, hi = input_spec.crop_ratio_range
    return min(max(float(crop_ratio), lo), hi)


def plan_for(
    input_spec: InputSizeSpec,
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    *,
    crop_ratio: float,
    anomaly_name: str,
) -> CropPlan:
    """Plan one instance's window at generation. ``bbox`` is the inclusive ``(left, upper, right, lower)``.

    ``crop_ratio`` is clamped into the trained band, the scale axis is pinned at ``input_spec.scale``
    and the ladder is the one ``anomaly_name``'s class trained under. Generation and the validation
    token bound both plan through this call, so they cannot disagree on a window.
    """
    left, upper, right, lower = bbox
    return plan_window(
        right - left + 1,
        lower - upper + 1,
        img_w,
        img_h,
        ctx=clamp_crop_ratio(crop_ratio, input_spec),
        scale=input_spec.scale,
        ladder=ladder_for(input_spec, anomaly_name),
        ratio_floor=input_spec.crop_ratio_range[0],
        crop_mode=input_spec.crop_mode,
        rng=None,
    )
