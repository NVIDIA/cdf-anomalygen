# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mask VOCABULARY: the mask *spec* the VLM emits, its rasterizer into a binary {0,255} mask, and
the OpenCV mask-edit ops.

Coordinates are normalized to [0, 1]; ``width`` / ``radius`` are fractions of ``min(H, W)``. There
are no per-defect-type branches — all defect-specific intelligence lives in *which* primitives the
VLM chooses and *where* it puts them. To add an op, subclass `Op` with a `kind`, `group`, `prompt`,
`params` and an `apply` — or `_Shape` with a `draw` — and register it in `OPS`. What the model LOOKS
at is `imaging.py`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from cosmos_framework.utils import log
from PIL import Image, ImageDraw
from scipy.ndimage import label as _lbl

from anomalygen.agentic_mask_generation.utils import resize_labels

_SHORT_SIDE = "short side"
_FRAC_OF_SHORT_SIDE = "fraction of the short side"
_GROUP_REGULARIZE = "REGULARIZE (tidy the outline; adds no new information)"
_GROUP_REGION_CLEANUP = "REGION CLEANUP"
_POINT_NORMALIZED = "[x,y] normalized"
_FRAC_OF_WIDTH = "fraction of the image WIDTH"
_FRAC_OF_HEIGHT = "fraction of the image HEIGHT"

# --------------------------------------------------------------------------------------------------
# Shape helpers — the pixel maths every shape op shares, plus stamp-shape extraction / stats.
# --------------------------------------------------------------------------------------------------


def _px(xy: list[float], w: int, h: int) -> tuple[int, int]:
    # Clamped, so a coordinate slightly out of [0,1] does not truncate the shape at the border.
    x = min(max(float(xy[0]), 0.0), 1.0)
    y = min(max(float(xy[1]), 0.0), 1.0)
    return min(int(round(x * w)), w - 1), min(int(round(y * h)), h - 1)


def _floor_note(name: str, frac: float, ref: int, axis: str, outcome: str = "floored to 1px") -> str:
    """The one sentence every op says when it substitutes for a sub-pixel size.

    `outcome` varies: most ops floor to one pixel, while `corner` and a sub-pixel morphology `frac`
    floor to ZERO (`corner: 0` is the legal way to ask for an unrounded outline).

    `floored to`, deliberately not the clamp channel's `raised to`, so a substitution and a clamp read
    apart.
    """
    return f"{name} {outcome} ({name} {frac:g} of a {ref}px {axis} is {frac * ref:.3g}px, under one pixel)"


def _scale(frac: float, w: int, h: int, *, adj: list | None = None, name: str = "") -> int:
    # width/radius are fractions of the short side, so they look the same on any aspect.
    px = int(round(frac * min(w, h)))
    if px < 1 and adj is not None and name:
        adj.append(_floor_note(name, frac, min(w, h), _SHORT_SIDE))
    return max(1, px)


def _scale_xy(
    fx: float, fy: float, w: int, h: int, *, adj: list | None = None, nx: str = "", ny: str = ""
) -> tuple[int, int]:
    """Per-axis extents: rx is a fraction of WIDTH and ry of HEIGHT, matching how the VLM reads them
    off the gridded image's axes. Scaling both by min(w,h) squeezes the long axis on a non-square
    board."""
    px, py = int(round(fx * w)), int(round(fy * h))
    if adj is not None:
        if px < 1 and nx:
            adj.append(_floor_note(nx, fx, w, "width"))
        if py < 1 and ny:
            adj.append(_floor_note(ny, fy, h, "height"))
    return max(1, px), max(1, py)


def _bounded_extent(px: int, w: int, h: int) -> int:
    """The pixel extent of a primitive that rasterizes into its OWN temp buffer — `(2*extent+1)^2` for
    `ellipse`, `arc` and `ring`, `tw x th` for `stamp` — so the model's number sizes an ALLOCATION.

    ONLY `stamp` can reach the cap: every other extent is a fraction with a declared `hi` of 1.0 that
    `check_input` clamps first, while `stamp.scale` is a multiplier of the pasted shape's native size
    with NO upper bound, so uncapped `stamp{scale:100000}` on a 512x512 board would ask for a patch
    tens of millions of pixels a side. This line is what hands back 2048 instead.

    Four times the LONG side is past the diagonal from any centre, so a shape at the cap already
    reaches beyond every pixel of the board, and the buffer stays ~17 MB on a 512 board.
    """
    return max(1, min(int(px), 4 * max(w, h)))


def _draw_capsule(d: ImageDraw.ImageDraw, a, b, r: int) -> None:
    """A capsule = thick line with rounded caps (two end disks)."""
    d.line([a, b], fill=255, width=2 * r)
    for c in (a, b):
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=255)


def extract_shapes(exemplars, max_total: int = 12) -> list[dict]:
    """Build a 'shape library' from the EXAMPLE masks so the agent can STAMP a REAL defect shape:
    each exemplar mask's FULL shape plus its largest connected components, as normalized patches.

    ``wf``/``hf`` are both measured against the exemplar's SHORT side, and `Stamp.draw` scales both
    by the target's short side, so a stamp keeps the example shape's aspect ratio on any board."""
    lib: list[dict] = []
    for ei, ex in enumerate(exemplars):
        a = (np.asarray(ex.mask.convert("L")) > 127).astype(np.uint8)
        if not a.any():
            continue
        eh, ew = a.shape
        eref = min(ew, eh)  # one reference for BOTH extents -> the stored fractions carry true aspect
        ys, xs = np.nonzero(a > 0)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        n, lab, stats, _ = cv2.connectedComponentsWithStats(a)
        lib.append(
            {
                "patch": a[y0 : y1 + 1, x0 : x1 + 1].astype(bool),
                "wf": (x1 - x0 + 1) / eref,
                "hf": (y1 - y0 + 1) / eref,
                "ncomp": n - 1,
                "src": ei + 1,
                "kind": "full",
            }
        )
        for c in sorted(range(1, n), key=lambda c: -stats[c, cv2.CC_STAT_AREA])[:3]:
            if stats[c, cv2.CC_STAT_AREA] < 0.0002 * a.size:
                continue
            x, y, ww, hh, _ = stats[c]
            lib.append(
                {
                    "patch": (lab[y : y + hh, x : x + ww] == c),
                    "wf": ww / eref,
                    "hf": hh / eref,
                    "ncomp": 1,
                    "src": ei + 1,
                    "kind": "component",
                }
            )
    lib = lib[:max_total]
    for i, s in enumerate(lib):
        s["id"] = i
    return lib


# HOW THE FIELDS LISTED HERE SCALE, declared once by the module that scales them.
#
# `regions._remap_primitives` maps a shape authored in a crop frame back onto the board and imports
# this table rather than keeping a second field list. A wrong entry distorts a shape SILENTLY, and a
# field MISSING from the table is left in the CROP frame while every other coordinate moves.
#
# NOT here: anything meaning the same number on either canvas — angles, enum axes, multipliers
# (`factor`, `sx`, `sy`), pixel lengths (`px`, `corner_px`) and the fields measured against the MASK
# rather than the canvas (`width_rel`, `eps`).
FIELD_SCALE = {
    "center": "point",
    "a": "point",
    "b": "point",
    "points": "points",
    "rx": "width",  # a fraction of the image WIDTH
    "ry": "height",  # a fraction of its HEIGHT
    "dx": "width",  # translate's offsets, declared "fraction of W" / "fraction of H"
    "dy": "height",
    "radius": "short",
    "inner": "short",
    "width": "short",
    "scale": "short",
    "corner": "short",  # the polygon ops' rounding radius, `_scale`d like every other length
    "frac": "short",  # the morphology ops' kernel, as a fraction of the short side
    "at": "point",
    "path": "points",
    # THE ONE FIELD NO OP DECLARES: `target` is cross-cutting (`CROSS_CUTTING`), so it is reachable
    # from neither `OPS[k].params` nor a per-op `apply`, but the remapper reads the REQUEST and there
    # it is a normalized point like any other.
    "target": "point",
    # A THIRD kind: a pixel in the crop IS a pixel on the board (same resolution, different origin),
    # so `points_px` maps by pure translation while fractional fields map by scale-and-offset.
    # Folding it into "points" multiplies it by the crop fraction — a silent 5x error on a
    # fifth-of-the-board crop.
    "points_px": "points_px",
}


# --------------------------------------------------------------------------------------------------
# CV ops — deterministic geometric edits of a MASK (pixels), via OpenCV. Applying one of these is the
# cheap alternative to re-running the generator when the fix is a move/resize/rotate/thicken.
# --------------------------------------------------------------------------------------------------


def _to_np(mask: Image.Image) -> np.ndarray:
    return ((np.asarray(mask.convert("L")) > 127).astype(np.uint8)) * 255


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr > 127).astype(np.uint8) * 255).convert("L")


def _kernel(px: int) -> np.ndarray:
    px = max(1, px)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def _centroid(m: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(m > 0)
    if len(xs) == 0:
        h, w = m.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def _conform(m: np.ndarray, clean: Image.Image, band_rel: float = 0.12) -> np.ndarray:
    """Snap the mask to the real structure boundary in the clean image: GrabCut, seeded by the mask.

    The boundary may travel at most ``band_rel`` of the mask's own short side. Unbounded, the mask
    becomes the whole structure it sits on — six deliberately different shapes on one joint all
    converge to within IoU 0.6-0.8 of each other. Set ``band_rel`` to 0 for that behaviour, right
    only when the defect IS the whole structure (a missing component, say).
    """
    if clean is None or not m.any():
        return m
    rgb = np.asarray(clean.convert("RGB"))
    H, W = m.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))
    ys, xs = np.nonzero(m > 0)
    pad = int(0.06 * min(H, W))
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    sub, subm = rgb[y0:y1, x0:x1].copy(), (m[y0:y1, x0:x1] > 0)
    if sub.size == 0 or min(sub.shape[:2]) < 5:
        return m
    gc = np.full(subm.shape, cv2.GC_PR_BGD, np.uint8)
    gc[subm] = cv2.GC_PR_FGD
    if band_rel > 0:
        # Band scaled to the MASK, not the board: a fixed pixel budget is a different fraction of a
        # 20 px defect than of a 200 px one.
        _ys, _xs = np.nonzero(subm)
        _short = max(1, min(int(_xs.max() - _xs.min() + 1), int(_ys.max() - _ys.min() + 1)))
        band = max(2, int(round(band_rel * _short)))
        # Only the ring within `band` px of the outline is left undecided, so the outline can move by
        # at most that much either way.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        u8 = subm.astype(np.uint8)
        gc[cv2.erode(u8, k) > 0] = cv2.GC_FGD
        gc[cv2.dilate(u8, k) == 0] = cv2.GC_BGD
    else:
        # Only a SMALL central core is definite-FG, so GrabCut is free to carve an oversized blob
        # down to the real feature; the core is what stops it collapsing entirely.
        dt = cv2.distanceTransform(subm.astype(np.uint8), cv2.DIST_L2, 5)
        if dt.max() > 0:
            gc[dt >= 0.6 * dt.max()] = cv2.GC_FGD
        else:
            gc[cv2.erode(subm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(
            sub, gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK
        )
        ref = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    except Exception:
        return m
    if ref.sum() == 0:
        return m
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = ref * 255
    return out


def _snap(m: np.ndarray, clean: Image.Image) -> np.ndarray:
    """Intensity-based snap-to-feature: Otsu-split the WHOLE clean image (only the keep step is local)
    and drop the mask pixels off the cluster its core sits on. Cheaper than GrabCut `conform`, which is
    the one that crops to a neighbourhood, and only ever SHRINKS."""
    if clean is None or not m.any():
        return m
    g = np.asarray(clean.convert("L"))
    H, W = m.shape
    if g.shape != (H, W):
        g = cv2.resize(g, (W, H))
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core = cv2.erode(m, np.ones((3, 3), np.uint8))
    bright = otsu > 0
    on_bright = int((bright & (core > 0)).sum())
    on_dark = int((~bright & (core > 0)).sum())
    feature = bright if on_bright >= on_dark else ~bright
    out = ((m > 0) & feature).astype(np.uint8) * 255
    return out if out.any() else m


def _layer_union(m: np.ndarray, labels: np.ndarray | None) -> np.ndarray | None:
    """Union of the guidance-layer structures the mask overlaps (None when no layer / no overlap)."""
    if labels is None or not m.any():
        return None
    H, W = m.shape
    if labels.shape != (H, W):
        labels = resize_labels(labels, W, H)
    ids = np.unique(labels[(m > 0) & (labels > 0)])
    if ids.size == 0:
        return None
    return np.isin(labels, ids)


def layer_fit(m: np.ndarray, labels: np.ndarray | None, mode: str) -> np.ndarray:
    """Grow each mask component to its overlapped guidance-layer structure(s).
      mode 'fill'  — union with the structure(s); the component itself is kept, so this never shrinks
      mode 'scale' — scale the component about its centroid (factors floored at 1.0) to cover the
                     structures' bbox. Only the SCALED copy is kept, so a concave component can lose
                     coverage: an L scaled about its centroid does not contain the original L.
    Components with no overlap pass through unchanged."""
    if labels is None or not m.any():
        return m > 0
    H, W = m.shape
    if labels.shape != (H, W):
        labels = resize_labels(labels, W, H)
    out = np.zeros((H, W), bool)
    cc, n = _lbl(m > 0)
    for i in range(1, n + 1):
        c = cc == i
        ids = np.unique(labels[c & (labels > 0)])
        if ids.size == 0:
            out |= c
            continue
        S = np.isin(labels, ids)
        if mode == "fill":
            out |= c | S
        else:
            ys, xs = np.nonzero(c)
            seg_ys, seg_xs = np.nonzero(S)
            tw, th = seg_xs.max() - seg_xs.min() + 1, seg_ys.max() - seg_ys.min() + 1
            w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            fx, fy = max(1.0, tw / w), max(1.0, th / h)  # grow only
            cx, cy = xs.mean(), ys.mean()
            A = np.float32([[fx, 0, cx * (1 - fx)], [0, fy, cy * (1 - fy)]])
            g = cv2.warpAffine(c.astype(np.uint8) * 255, A, (W, H))
            out |= g > 127
    return out


# --------------------------------------------------------------------------------------------------
# THE REGISTRY — every op declares itself next to the code that applies it. One declaration feeds the
# catalogue the model chooses from, the validation the engine applies, and the correction the model
# gets back when it got one wrong.
#
# `prompt` is MODEL-FACING TEXT and the catalogue is a tuned experimental artifact: editing one of
# these strings is a PROMPT change, and baselines being compared against each other stop being
# comparable.
# --------------------------------------------------------------------------------------------------
def _enum_choices(unit: str) -> frozenset:
    """The values an `enum` parameter accepts, read off the SAME string the model is shown.

    An `enum` whose unit quotes no values yields an empty set and then accepts nothing, so every
    declared one has to quote its values.
    """
    return frozenset(re.findall(r'"([^"]*)"', unit))


@dataclass(frozen=True)
class Param:
    """One declared parameter: what it is called, what it is, what UNIT it is in, and whether the op
    can proceed without it. The unit is not decoration — `px` and `frac` are two different fields on
    the same op precisely because the name states the unit."""

    name: str
    type: str  # number | int | points | enum | layer
    unit: str
    required: bool
    default: "Any" = None
    lo: "Any" = None  # declared bounds: every numeric parameter needs a `lo`, and a `hi` unless
    hi: "Any" = None  # only the image size can supply one (10 of 45 have none — see _Morphology.px)
    alias: bool = False  # accepted by the engine, NOT advertised — `points` for `path`, `points_px`
    #                       and `corner_px` for `points` and `corner`; the catalogue stays at one name
    #                       per thing.
    edit_only: bool = False  # meaningless without an existing mask, so turn 1 is never taught it:
    #                       `width_rel` is a fraction of THE MASK'S OWN thickness, which on a blank
    #                       canvas is a fraction of nothing.


@dataclass(frozen=True)
class Context:
    """Everything an op may need BESIDES the mask and its own parameters — one object, so adding a
    new one does not change every op's signature."""

    clean: "Any" = None
    layers: "Any" = None
    shapes: "Any" = None  # the stamp library, for `stamp`


class Op:
    """One op: what the model is TOLD, what it will ACCEPT, and what it DOES — in one place.

    A subclass gives `kind`, `group`, `prompt`, `params` and an `apply`. `check_input` is inherited
    and is the ONE place a request becomes values: declared names, types and bounds only.
    """

    kind: str = ""
    group: str = ""
    prompt: str = ""
    # The same tool said in the voice of turn 1, which draws onto a BLANK canvas — `add_polygon`'s
    # "ADD everything inside that closed outline" has nothing to add to. Empty means `prompt` already
    # reads correctly in both voices.
    init_prompt: str = ""
    params: tuple = ()
    rigid: bool = False
    requires: tuple = ()
    one_of: tuple = ()
    # `one_of` is "at least one of these"; `overrides` is "at most one of these MATTERS" — two
    # spellings of one quantity where supplying both means one is ignored. First member wins, and the
    # loser is REPORTED rather than dropped in silence.
    overrides: tuple = ()

    @property
    def desc(self) -> str:  # an alias of `prompt` with no readers: the catalogue renders
        return self.prompt  # `sp.prompt` / `sp.init_prompt` itself

    def required_params(self) -> tuple:
        return tuple(p for p in self.params if p.required)

    def check_input(self, request: dict) -> "dict | Refused":
        """The request -> declared values, or a Refused naming what is missing and in what unit.

        Every DECLARED parameter crosses HERE and nowhere else. The cross-cutting `target` is the
        exception, because no op declares it: `apply_op` reads it straight off the request, and
        `_apply_to_one_region` parses and refuses it.
        """
        missing = [p.name for p in self.required_params() if _absent(request.get(p.name))]
        for group in self.one_of:
            if all(_absent(request.get(n)) for n in group):
                missing.append(" or ".join(group))
        if missing:
            return Refused(f"{self.kind} was asked for without {', '.join(missing)}", self.signature())
        out, adjusted = {}, []
        for prm in self.params:
            raw = request.get(prm.name)
            if _absent(raw):
                out[prm.name] = prm.default
                continue
            try:
                if prm.type in ("enum", "layer"):
                    v = str(raw)
                elif prm.type == "int":
                    v = int(float(raw))
                elif prm.type == "points":
                    v = [[float(x), float(y)] for x, y in raw]
                elif prm.type == "point":
                    x, y = raw  # ONE [x, y]; `float()` reads neither this nor `points`
                    v = [float(x), float(y)]
                else:
                    v = float(raw)
            except (TypeError, ValueError, OverflowError):
                # OverflowError because `1e999` parses to `inf` and `int(inf)` raises it. Named here
                # rather than left to `apply_op`'s handler so the agent is told which PARAMETER it got
                # wrong instead of "dilate failed: OverflowError".
                return Refused(f"{self.kind}: {prm.name} is not a {prm.type} ({raw!r})", self.signature())
            # Unchecked, `str(raw)` accepts anything and `flip` takes its ELSE branch, so `axis:"X"`
            # would mirror TOP/BOTTOM — the opposite of what was asked — with `adjustments=()` saying
            # nothing had changed.
            if prm.type == "enum" and v not in _enum_choices(prm.unit):
                return Refused(f"{self.kind}: {prm.name} must be one of {prm.unit}, got {raw!r}", self.signature())
            if prm.type in ("point", "points"):
                # A coordinate is a list element, so the scalar finiteness check below never sees it.
                # Unchecked, `_px` clamps an infinite coordinate to the border and the shape is drawn
                # there as `Applied` with no adjustment; a NaN one reaches PIL as a generic failure.
                coords = v if prm.type == "point" else [c for pt in v for c in pt]
                if not all(math.isfinite(c) for c in coords):
                    what = "a finite point" if prm.type == "point" else "a list of finite points"
                    return Refused(f"{self.kind}: {prm.name} is not {what} ({raw!r})", self.signature())
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # NON-FINITE FIRST, because every comparison against NaN is False and so every
                # declared bound below is VACUOUS for it: `translate{dx:"NaN"}` builds a NaN affine
                # matrix and `warpAffine` returns an empty mask, reported as `Applied`.
                # Infinity is refused rather than clamped, because clamping turns a nonsense request
                # into a MAXIMAL one — `dx: 1.0` translates the mask clean off the canvas.
                if isinstance(v, float) and not math.isfinite(v):
                    return Refused(f"{self.kind}: {prm.name} is not a finite number ({raw!r})", self.signature())
                if prm.lo is not None and v < prm.lo:
                    v = prm.lo
                    adjusted.append(f"{prm.name} raised to {prm.lo}")
                if prm.hi is not None and v > prm.hi:
                    v = prm.hi
                    adjusted.append(f"{prm.name} capped at {prm.hi}")
            out[prm.name] = v
        _sent_now = {k for k, v in request.items() if isinstance(k, str) and not _absent(v)}
        # Generated from the declaration, so no op has to remember to write the note itself.
        for _grp in self.overrides:
            _got = [n for n in _grp if n in _sent_now]
            for _lost in _got[1:]:
                adjusted.append(
                    f"{_lost} {request[_lost]!r} ignored — {_got[0]} was also given "
                    f"and takes precedence; send one or the other"
                )
        out["_adjusted"] = tuple(adjusted)
        # WHAT THE MODEL ACTUALLY SENT, as distinct from what the defaults filled in: `p["width"]` is
        # 0.02 on every brush request, so `p` alone cannot answer "did they give me both spellings?".
        # `_absent`, not key-presence — `{"width": null}` HAS the key and supplies nothing.
        out["_sent"] = frozenset(k for k, v in request.items() if isinstance(k, str) and not _absent(v))
        return out

    def apply(self, m, p: dict, ctx: Context) -> "Applied | Refused":
        raise NotImplementedError(f"{self.kind} declares no behaviour")

    def example(self) -> str:
        """A JSON request for this op, DERIVED from its own parameters — so an op added tomorrow has
        one, and no example can quietly describe a spelling the declaration no longer has."""

        def value(prm):
            # by UNIT, not by type: "0.05" is a sensible fraction, a meaningless angle (0.05 deg) and
            # a destructive multiplier (shrink to 5%).
            u = prm.unit
            if prm.type == "enum":
                return prm.default
            if prm.type == "points":
                return [[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]]
            if prm.type == "point":
                return [0.5, 0.5]
            if prm.type == "layer":
                return "sam2"
            if "multiplier" in u:
                return 1.2
            if "sector end" in u:
                # BEFORE `degrees`, which this unit also contains and would otherwise swallow it. The
                # second angle of a pair must DIFFER from the first or the sector draws nothing.
                return 120
            if "degrees" in u:
                return 12
            if "arc length" in u:
                return 0.01
            if "hole" in u:
                # `ring.radius` and `ring.inner` share a unit, so one number for both is an annulus
                # of zero thickness, which the op refuses.
                return 0.02
            return 0.05  # any fraction-of-the-image length

        req = {"kind": self.kind}
        # ONE member of each `overrides` group, and it is the one with a declared DEFAULT rather than
        # the one that wins at runtime: `brush` would otherwise teach `width_rel`, whose reference is
        # the mask's own thickness, in a turn-1 catalogue where there is no mask yet.
        # `overrides` is in runtime-precedence order and its first member may be an alias this loop
        # will not advertise (`_Polygon` declares `("points_px", "points")`), so the choice is made
        # among the survivors of the alias filter — shadow nothing when fewer than two survive, or
        # the group emits NEITHER spelling.
        _by_name = {prm.name: prm for prm in self.params}

        def _showable(n):
            prm = _by_name.get(n)
            return prm is not None and not prm.alias and "PIXEL" not in prm.unit and prm.unit != "pixels"

        shadowed = set()
        for g in self.overrides:
            cands = [n for n in g if _showable(n)]
            if len(cands) < 2:
                continue  # nothing to disambiguate once aliases are dropped
            keep = next((n for n in cands if _by_name[n].default is not None), cands[0])
            shadowed |= set(cands) - {keep}
        for prm in self.params:
            if prm.alias or "PIXEL" in prm.unit or prm.unit == "pixels":
                continue  # accepted, never advertised: aliases + pixel spellings
            if prm.name in shadowed:
                continue
            req[prm.name] = value(prm)
        return json.dumps(req, separators=(",", ":"))

    def signature(self) -> str:
        """What to tell a model that got the parameters wrong — names and UNITS, never the prose.

        `one_of` members are NOT "[optional]": calling them that gives `{"kind":"dilate"}` back a
        refusal whose explanation says both may be omitted, in the same sentence as the refusal for
        omitting both.
        """
        if not self.params:
            return f"{self.kind} takes no parameters"
        alt = {n for g in self.one_of for n in g}

        def _need(p):
            if p.required:
                return " [required]"
            return " [give one of these]" if p.name in alt else " [optional]"

        return f"{self.kind} takes " + ", ".join(f"{p.name} ({p.unit}){_need(p)}" for p in self.params)


# Every registered value is an `Op` subclass, so no caller has to narrow a union.
OPS: dict[str, Op] = {}


# --- GEOMETRY -------------------------------------------------------------------------------------
class _Geometry(Op):
    requires = ("mask",)  # moves or resizes the mask it is given
    group = "GEOMETRY (move or resize the WHOLE mask; its shape is preserved)"


class Translate(_Geometry):
    kind, rigid = "translate", True
    prompt = "translate{dx,dy} — slide it; +dx right, +dy DOWN"
    params = (
        Param("dx", "number", "fraction of W", False, 0.0, -1.0, 1.0),
        Param("dy", "number", "fraction of H", False, 0.0, -1.0, 1.0),
    )

    def apply(self, m, p, ctx):
        H, W = m.shape
        M = np.float32([[1, 0, p["dx"] * W], [0, 1, p["dy"] * H]])
        return Applied(cv2.warpAffine(m, M, (W, H)), p["_adjusted"])


class Rotate(_Geometry):
    kind, rigid = "rotate", True
    prompt = "rotate{angle} — spin it about its centre, degrees, + is counter-clockwise"
    params = (Param("angle", "number", "degrees, + counter-clockwise", False, 0.0, -360.0, 360.0),)

    def apply(self, m, p, ctx):
        H, W = m.shape
        return Applied(
            cv2.warpAffine(m, cv2.getRotationMatrix2D(_centroid(m), p["angle"], 1.0), (W, H)), p["_adjusted"]
        )


class Scale(_Geometry):
    kind = "scale"
    prompt = "scale{factor} — bigger (>1) or smaller (<1)"
    params = (Param("factor", "number", "multiplier", False, 1.0, 0.01, 20.0),)

    def apply(self, m, p, ctx):
        H, W = m.shape
        return Applied(cv2.warpAffine(m, cv2.getRotationMatrix2D(_centroid(m), 0, p["factor"]), (W, H)), p["_adjusted"])


class Stretch(_Geometry):
    kind = "stretch"
    prompt = "stretch{sx,sy} — lengthen or thin along one axis about the centre"
    params = (
        Param("sx", "number", "multiplier on x", False, 1.0, 0.01, 20.0),
        Param("sy", "number", "multiplier on y", False, 1.0, 0.01, 20.0),
    )

    def apply(self, m, p, ctx):
        H, W = m.shape
        cx, cy = _centroid(m)
        sx, sy = p["sx"], p["sy"]
        M = np.float32([[sx, 0, cx * (1 - sx)], [0, sy, cy * (1 - sy)]])
        return Applied(cv2.warpAffine(m, M, (W, H)), p["_adjusted"])


class Flip(_Geometry):
    kind, rigid = "flip", True
    prompt = (
        'flip{axis:"x"|"y"} — mirror it IN PLACE inside its own box; "x" swaps left/right, "y" '
        "top/bottom. For a shape that is right but handed the wrong way round"
    )
    params = (Param("axis", "enum", '"x" | "y"', False, "x"),)

    def apply(self, m, p, ctx):
        ys, xs = np.nonzero(m)
        if not xs.size:
            return Applied(m, p["_adjusted"])
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        sub = m[y0:y1, x0:x1]
        sub = sub[:, ::-1] if p["axis"] == "x" else sub[::-1, :]
        out = np.zeros_like(m)
        out[y0:y1, x0:x1] = sub
        return Applied(out, p["_adjusted"])


# --- MORPHOLOGY -------------------------------------------------------------------------------------
# Five ops differing only in which cv2 morphology call they make, plus `smooth`, which borrows their
# two size spellings and blurs instead. Unbounded, `close frac=1.0` is 49 seconds and `px=100000` a
# 40 GB allocation, both reported as success — hence `frac`'s declared `hi` and, since `px` has none,
# `_kernel_px`'s cap against the mask.
# --------------------------------------------------------------------------------------------------
class _Morphology(Op):
    requires = ("mask",)  # thickens or thins an existing outline
    group = "MORPHOLOGY (thicken or thin the outline everywhere at once)"
    # Neither spelling has a default, because an amount the engine invents is an edit the model did
    # not ask for. `hi` caps the fraction at half the short side: past that a kernel does not thicken
    # the outline, it swallows the mask.
    params = (
        Param("px", "int", "pixels", False, None, 0, None),
        Param("frac", "number", _FRAC_OF_SHORT_SIDE, False, None, 0.0, 0.5),
    )
    one_of = (("px", "frac"),)
    # ONE kernel in two units, so a request carrying both has a loser: on an 80px square
    # `dilate{px:2}` and `dilate{frac:0.2}` light 7048 and 24136 pixels, and a model told nothing
    # reads the `px` result as what its `frac` asked for.
    overrides = (("px", "frac"),)
    cv_op = None

    def _kernel_px(self, m, p) -> int:
        """The kernel number in pixels: a RADIUS for the five cv2 ops, whose `_kernel` builds a
        (2n+1)^2 element from it, and the kernel SIZE itself for `smooth`, which passes it to
        `GaussianBlur` — which is why `smooth` redeclares the unit `px` is written in.

        `px` is still capped against THIS mask, because a declared bound cannot know the image size."""
        s = min(m.shape)
        n = int(p["px"]) if p["px"] is not None else int(p["frac"] * s)
        return max(0, min(n, s // 2))

    def apply(self, m, p, ctx):
        n = self._kernel_px(m, p)
        adj = list(p["_adjusted"])
        if p["px"] is not None and n != int(p["px"]):
            adj.append(f"px capped at {n} (half this mask's short side)")
        if n <= 0:
            # cv2 has no zero-sized kernel and `_kernel(0)` is a 3x3, which moves the outline a
            # pixel, so zero has to be handled here as no change. A sub-pixel request that rounded to
            # zero is REPORTED (an explicit `frac: 0` is a legitimate no-op and stays silent).
            if p["frac"] and p["px"] is None:
                adj.append(
                    _floor_note("frac", p["frac"], min(m.shape), _SHORT_SIDE, outcome="dropped — the mask is unchanged")
                )
            return Applied(m, tuple(adj))
        k = _kernel(n)
        out = (
            cv2.dilate(m, k)
            if self.cv_op == "dilate"
            else cv2.erode(m, k)
            if self.cv_op == "erode"
            else cv2.morphologyEx(m, self.cv_op, k)
        )
        return Applied(out, tuple(adj))


class Dilate(_Morphology):
    kind, cv_op = "dilate", "dilate"
    prompt = "dilate{px} — grow the outline outward by px PIXELS"


class Erode(_Morphology):
    kind, cv_op = "erode", "erode"
    prompt = "erode{px} — shrink it inward by px PIXELS"


class Close(_Morphology):
    kind, cv_op = "close", cv2.MORPH_CLOSE
    prompt = "close{px} — fill gaps and holes smaller than px"


class Open(_Morphology):
    kind, cv_op = "open", cv2.MORPH_OPEN
    prompt = "open{px} — remove specks smaller than px"


class Gradient(_Morphology):
    kind, cv_op = "gradient", cv2.MORPH_GRADIENT
    prompt = "gradient{px} — keep only a rim of that thickness"


class Smooth(_Morphology):
    """NOT a morphology call at all — a Gaussian blur re-thresholded, which rounds a jagged edge
    without eating into it. It shares this group's two spellings and their bounds, and nothing else:
    the number they resolve to is a kernel RADIUS there and the kernel SIZE here."""

    kind = "smooth"
    group = _GROUP_REGULARIZE
    # `px` REDECLARED in the unit this op reads it in: `apply` hands the number to `GaussianBlur` as
    # the kernel SIZE, where the five cv2 ops make a (2*px+1) element out of it, so `smooth{px:5}`
    # blurs with a 5x5 kernel where `dilate{px:5}` grows by an 11x11 one.
    # NO COMMA in the unit string: `prompts._units_line` keys the size sentence on
    # `unit.split(",")[-1]`, so a comma would file this parameter under the tail of its own sentence.
    params = tuple(
        Param("px", "int", "PIXELS of Gaussian kernel width (not a radius)", False, None, 0, None)
        if prm.name == "px"
        else prm
        for prm in _Morphology.params
    )
    prompt = (
        "smooth{px} — round off a jagged edge; px is the WIDTH of the Gaussian kernel in "
        "pixels, and frac is that width as a fraction of the short side"
    )

    def apply(self, m, p, ctx):
        n = self._kernel_px(m, p)
        adj = list(p["_adjusted"])
        if p["px"] is not None and n != int(p["px"]):
            adj.append(f"px capped at {n} (half this mask's short side)")
        # `cv2.GaussianBlur(m, (1,1), 0)` is the IDENTITY: the smallest odd kernel that BLURS is 3,
        # so everything below n=2 leaves the mask byte-identical — including `smooth{px:1}`, the
        # spelling the catalogue advertises, and `frac` in [0.005, 0.0099] on a 200px mask.
        if n < 2:
            # NOT `_floor_note`: its sentence ends "under one pixel", and the cause here is the
            # KERNEL, not the pixel grid — `frac: 0.006` on a 200px mask is 1.2px and still cannot
            # blur.
            _asked = (
                f"px {int(p['px'])}"
                if p["px"] is not None
                else f"frac {p['frac']:g} of a {min(m.shape)}px short side ({p['frac'] * min(m.shape):.3g}px)"
            )
            adj.append(f"{_asked} dropped — a Gaussian kernel under 3px is the identity, so the mask is unchanged")
        k = max(1, n) | 1  # Gaussian needs an ODD kernel, and 0 is not one
        out = (cv2.GaussianBlur(m, (k, k), 0) > 127).astype(np.uint8) * 255
        return Applied(out, tuple(adj))


# --- REGULARIZE / REGION CLEANUP --------------------------------------------------------------------
class Clear(Op):
    kind = "clear"
    group = _GROUP_REGION_CLEANUP
    # `clear{}` and not a bare `clear`: the leak guard needs a syntactic form for any op name that
    # could be ordinary English ("a clear edge").
    prompt = "clear{} — empty the mask, to start this candidate over"
    requires = ("mask",)  # on a blank canvas it is a no-op, so turn 1 is never offered it

    def apply(self, m, p, ctx):
        return Applied(np.zeros_like(m), p["_adjusted"])


class Convex(Op):
    requires = ("mask",)  # the hull OF something
    kind = "convex"
    group = _GROUP_REGULARIZE
    prompt = "convex — fill every dent, leaving a convex blob"

    def apply(self, m, p, ctx):
        pts = cv2.findNonZero(m)
        if pts is None or len(pts) < 3:
            return Applied(m, p["_adjusted"])  # a hull needs three points; fewer is already convex
        out = np.zeros_like(m)
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
        return Applied(out, p["_adjusted"])


class Simplify(Op):
    requires = ("mask",)  # simplifies an existing outline
    kind = "simplify"
    group = _GROUP_REGULARIZE
    prompt = "simplify{eps} — drop the small wiggles from a noisy polygon"
    # `hi` is 0.2 because 0.5 collapses any outline to a triangle, which is not simplification.
    params = (Param("eps", "number", "fraction of the contour's arc length", False, 0.01, 0.0, 0.2),)

    def apply(self, m, p, ctx):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return Applied(m, p["_adjusted"])
        out = np.zeros_like(m)
        for c in cnts:
            cv2.fillPoly(out, [cv2.approxPolyDP(c, p["eps"] * cv2.arcLength(c, True), True)], 255)
        return Applied(out, p["_adjusted"])


class KeepLargest(Op):
    requires = ("mask",)  # selects among existing components
    kind = "keep_largest"
    group = _GROUP_REGION_CLEANUP
    prompt = "keep_largest — delete stray pieces, keep the main one"

    def apply(self, m, p, ctx):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return Applied(m, p["_adjusted"])
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        return Applied((lab == big).astype(np.uint8) * 255, p["_adjusted"])


class FillHoles(Op):
    requires = ("mask",)  # fills holes IN something
    kind = "fill_holes"
    group = _GROUP_REGION_CLEANUP
    prompt = "fill_holes — fill enclosed gaps inside the mask"

    def apply(self, m, p, ctx):
        # Flood the OUTSIDE; what the flood cannot reach is an enclosed hole. Padding with a 1 px
        # zero ring makes (0, 0) always background AND connects every background component that
        # touches a border — flooding from one unpadded border pixel reaches only its own component,
        # and a mask that splits the background (an edge-to-edge stroke) had the far side filled in
        # as a "hole".
        h, w = m.shape
        ff = np.pad(m, 1)
        cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
        return Applied(m | cv2.bitwise_not(ff[1:-1, 1:-1]), p["_adjusted"])


# --- FREEFORM ---------------------------------------------------------------------------------------
class _Stroke(Op):
    group = "FREEFORM (a brush and an eraser — ONE op does a dot, a stroke or a simple curve)"
    # ONE FRAME: normalized image coords. There is deliberately no pixel spelling of `path`, so a
    # model sizing off a zoom caption cannot have a pixel path silently honoured against a turn that
    # says "the FULL image, normalised [0,1]".
    #
    # `width_rel` is a fraction of THE MASK being edited — the practical one for a defect covering a
    # fraction of a percent of a board. It is taught by `_BRUSH_NORM` in every default proposer
    # prompt.
    params = (
        Param("path", "points", "[[x,y],...] normalized", False, None),
        Param("points", "points", "[[x,y],...] normalized", False, None, alias=True),
        Param("width", "number", _FRAC_OF_SHORT_SIDE, False, 0.02, 0.0, 1.0),
        Param("width_rel", "number", "fraction of THE MASK'S OWN thickness", False, None, 0.0, 10.0, edit_only=True),
    )
    # `path`/`points` are one list under two names; `width_rel`/`width` one thickness in two units.
    overrides = (("width_rel", "width"), ("path", "points"))
    paint = True

    def apply(self, m, p, ctx):
        H, W = m.shape
        s = min(H, W)
        pts = p["path"] or p["points"] or []
        pix = [(int(x * W), int(y * H)) for x, y in pts]
        if not pix:
            return Refused(f"{self.kind} was given no path", self.signature())
        # Its own pixel width rather than `_scale`, because `width_rel`'s reference is THE MASK'S OWN
        # extent, not the image, so there is no (frac, w, h) for the shared reader to work from. The
        # sub-pixel SENTENCE is still shared — `_floor_note`.
        adj = list(p["_adjusted"])
        if p["width_rel"] is not None:
            ys, xs = np.nonzero(m > 0)
            ref = max(8, int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))) if len(xs) else s
            name, frac, base, axis = "width_rel", p["width_rel"], ref, "mask"
        else:
            name, frac = "width", (p["width"] if p["width"] is not None else 0.02)
            base, axis = s, _SHORT_SIDE
        want = frac * base
        wpx = max(1, int(want))
        if want < 1:
            adj.append(_floor_note(name, frac, base, axis))
        # PIL, NOT cv2: `cv2.polylines(thickness=t)` renders t+1 px for even t (it centres on the
        # pixel grid), so `width: 0.01` on a 200 px board asks for 2 px and draws 3 — +50% at the
        # widths a defect actually uses. The shape ops rasterize through PIL too, so both agree.
        canvas = Image.fromarray(np.zeros_like(m))
        d = ImageDraw.Draw(canvas)
        if len(pix) >= 2:
            d.line(pix, fill=255, width=wpx, joint="curve")
        r = wpx // 2
        for c in pix:  # round caps, and a single point is a dot
            d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=255)
        layer = np.asarray(canvas)
        out = cv2.bitwise_or(m, layer) if self.paint else cv2.bitwise_and(m, cv2.bitwise_not(layer))
        return Applied(out, tuple(adj))


class Brush(_Stroke):
    init_prompt = "brush{path:[[x,y],...],width} — a stroke of that width along the path (a dot, if one point)"
    kind, paint = "brush", True
    prompt = "brush{...} — PAINT coverage along the path"


class Erase(_Stroke):
    requires = ("mask",)  # removes area FROM something; on a blank canvas it is a no-op
    kind, paint = "erase", False
    prompt = "erase{...} — REMOVE coverage along it. 1 point = a dot, 2+ = a stroke"


# --- FILLED OUTLINE ---------------------------------------------------------------------------------
class _Polygon(Op):
    group = "FILLED OUTLINE (straight edges and real corners, which a round brush cannot make)"
    params = (
        Param("points", "points", "[[x,y],...] normalized, 3+", False, None),
        Param("points_px", "points", "[[x,y],...] in PIXELS, 3+", False, None, alias=True),
        Param("corner", "number", _FRAC_OF_SHORT_SIDE, False, 0.0, 0.0, 0.5),
        Param("corner_px", "int", "PIXELS", False, None, 0, None, alias=True),
    )
    one_of = (("points", "points_px"),)
    # BOTH pixel/fraction pairs are declared, so each announces its winner instead of picking one in
    # silence — the two spellings differ by thousands of lit pixels on the same request.
    overrides = (("points_px", "points"), ("corner_px", "corner"))
    paint = True

    # How far outside the canvas a vertex may sit, in multiples of the long side. Not a `lo`/`hi` on
    # `points`: the bound is in PIXELS and a declared bound cannot know the image size.
    _MAX_OUTSIDE = 10

    def apply(self, m, p, ctx):
        H, W = m.shape
        pix = (
            [(int(x), int(y)) for x, y in p["points_px"]]
            if p["points_px"]
            else [(int(x * W), int(y * H)) for x, y in p["points"]]
        )
        if len(pix) < 3:
            return Refused(f"{self.kind} needs 3+ points, got {len(pix)}", self.signature())
        # REFUSED, not clamped: a far vertex changes the SLOPE of the edges entering the canvas, so
        # pulling it back to the frame draws a different polygon — a vertex at x=-1.0 clamped to
        # x=0.0 loses 35% of the lit area. It bounds a real cost too: cv2 refuses past int32 on its
        # own, but inside that window far-out points still cost whole seconds per op on a large mask.
        #
        # 10x the long side, not the frame: a defect running off the edge of the board is a real
        # request and lands a vertex just outside. Ten canvases out is a pixels-for-fraction slip.
        lim = self._MAX_OUTSIDE * max(W, H)
        if any(abs(x) > lim or abs(y) > lim for x, y in pix):
            far = max(max(abs(x), abs(y)) for x, y in pix)
            return Refused(
                f"{self.kind}: a vertex sits {far} px from the origin, past the {lim} px this "
                f"{W}x{H} mask allows (10x its long side) — normalized `points` are 0-1, and "
                f"`points_px` are pixels of THIS mask",
                self.signature(),
            )
        layer = np.zeros_like(m)
        cv2.fillPoly(layer, [np.array(pix, np.int32)], 255)
        # Opening rounds the convex corners, closing the concave ones; the same disk for both leaves
        # the edges between corners where they were drawn.
        # CAPPED AGAINST THIS MASK and reported, as in `_Morphology._kernel_px`: the PIXEL alias
        # `corner_px` is a raw int no declared bound can size against the image. Uncapped it builds a
        # (2*cr+1)^2 element — uncapped values turn a tiny mask into a minutes-long
        # op and then into a tens-of-GB allocation.
        s_short = min(H, W)
        cr = int(p["corner_px"]) if p["corner_px"] is not None else int(p["corner"] * s_short)
        cr = max(0, min(cr, s_short // 2))
        adj = list(p["_adjusted"])
        if p["corner_px"] is not None and cr != int(p["corner_px"]):
            adj.append(f"corner_px capped at {cr} (half this mask's short side)")
        elif p["corner_px"] is None and p["corner"] and cr == 0:
            # a sub-pixel corner resolves to NO rounding, and that is the honest outcome to report
            adj.append(
                _floor_note(
                    "corner", p["corner"], s_short, _SHORT_SIDE, outcome="dropped — the outline is left unrounded"
                )
            )
        if cr > 0:
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cr + 1, 2 * cr + 1))
            layer = cv2.morphologyEx(layer, cv2.MORPH_OPEN, disk)
            layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, disk)
        out = cv2.bitwise_or(m, layer) if self.paint else cv2.bitwise_and(m, cv2.bitwise_not(layer))
        return Applied(out, tuple(adj))


class AddPolygon(_Polygon):
    init_prompt = "add_polygon{points:[[x,y],...],corner} — the filled area inside a closed outline"
    kind, paint = "add_polygon", True
    prompt = "add_polygon{points:[[x,y],...],corner} — ADD everything inside that closed outline"


class ErasePolygon(_Polygon):
    requires = ("mask",)  # same: it draws, but what it draws is a HOLE
    kind, paint = "erase_polygon", False
    prompt = (
        "erase_polygon{...same...} — REMOVE it. 3+ points. Reach for these when the defect "
        'follows a STRAIGHT edge of the hardware — along a lead, across a pad. "corner" rounds '
        "the outline off by that radius (omit or 0 = sharp), so one op spans a crisp rectangle "
        "through to a rounded blob"
    )


# --- IMAGE-GUIDED -----------------------------------------------------------------------------------
class Conform(Op):
    kind = "conform"
    group = "IMAGE-GUIDED (let the CLEAN IMAGE decide the outline — best for Shape and Fill)"
    requires = ("mask", "clean")
    prompt = (
        "conform — pull the outline onto whatever real edge lies under it. The mask KEEPS its "
        "place and roughly its size; only the boundary moves, and by at most a fraction of the "
        "mask. Use it once the mask is on the right feature and you want its edge to follow that "
        "feature instead of a hand-drawn approximation — it does in one op what many small brush "
        "corrections would"
    )

    def apply(self, m, p, ctx):
        return Applied(_conform(m, ctx.clean), p["_adjusted"])


class Snap(Op):
    kind = "snap"
    group = "IMAGE-GUIDED (let the CLEAN IMAGE decide the outline — best for Shape and Fill)"
    requires = ("mask", "clean")
    prompt = (
        "snap — keep only the mask pixels sitting on the bright/real feature and drop the rest; "
        "only ever shrinks, so it is the way to carve a blob off the dark gaps between structures"
    )

    def apply(self, m, p, ctx):
        return Applied(_snap(m, ctx.clean), p["_adjusted"])


# --- STRUCTURE LAYERS -------------------------------------------------------------------------------
# A layer is ANY int label map (0 = background): fillet highlights, SAM2 segments, clusters. These
# ops never know what the structures MEAN, which is why one class serves every visual prompt.
class _LayerOp(Op):
    group = "STRUCTURE LAYERS (a precomputed map of the real structures; needs a visual prompt that provides one)"
    requires = ("mask", "layer")
    params = (Param("layer", "layer", "the injected visual prompt's own map", True, None),)
    fit = None  # "fill" | "scale", or None for the trim

    def apply(self, m, p, ctx):
        lab = (ctx.layers or {}).get(p["layer"])
        if self.fit is not None:
            out = layer_fit(m, lab, self.fit)
            return Applied(out.astype(np.uint8) * 255, p["_adjusted"])
        u = _layer_union(m, lab)
        if u is None:
            # no overlap: trimming to nothing would delete the candidate this op was asked to refine
            return Applied(m, p["_adjusted"])
        return Applied(((m > 0) & u).astype(np.uint8) * 255, p["_adjusted"])


class SnapLayer(_LayerOp):
    kind = "snap_layer"
    prompt = (
        "snap_layer{layer:__LAYERS__} — TRIM the mask down to the mapped structure it "
        "overlaps. Use it when the mask spills past the real feature into gaps or background"
    )


class FillLayer(_LayerOp):
    kind, fit = "fill_layer", "fill"
    prompt = (
        "fill_layer{layer:__LAYERS__} — GROW each region OUT until it fully covers the mapped "
        "structure it overlaps (it never shrinks). Use it when the mask is smaller than the "
        "feature it should occupy — a defect that takes the WHOLE joint rather than a dot on it"
    )


class AddRegion(_LayerOp):
    """The one layer op that does not need a mask to start from: it addresses a structure directly,
    by a point inside it, where the other three can only make an EXISTING overlap agree.

    ADDITIVE — nothing here erases, so no sequence of `add_region`s can lose the candidate.

    NO FALLBACK: a point on no mapped structure is REFUSED with the point named. Nearest-region, or a
    disk at the point, would place the defect somewhere the model did not ask for, and the model
    reads the resulting mask as confirmation that its coordinate was right.
    """

    kind = "add_region"
    requires = ("layer",)  # NOT "mask": this op is how a mask can begin
    params = (
        Param("layer", "layer", "the injected visual prompt's own map", True, None),
        Param("at", "point", "a point INSIDE the structure to add, in 0..1 of the image", True, None),
    )
    prompt = (
        "add_region{layer:__LAYERS__,at:[x,y]} — PAINT the whole mapped structure under that "
        "point into the mask. Use it to take a WHOLE joint/pad/lead; repeat it to take another"
    )
    init_prompt = (
        "add_region{layer:__LAYERS__,at:[x,y]} — the whole mapped structure under that "
        "point (a whole joint, pad or lead), taken as the mask"
    )

    def apply(self, m, p, ctx):
        # No `lab is None` branch: `apply_op` validates the `layer` name against this run's maps
        # before `apply` is reached.
        lab = np.asarray(ctx.layers[p["layer"]])
        H, W = m.shape
        x, y = p["at"]
        # An address outside the image is refused as exactly that. Clamped in, it would land on the
        # border and then report "on no mapped structure", sending the model looking for a better
        # structure instead of a better coordinate. `1.0` is the right edge and is kept.
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return Refused(
                f"{self.kind}: [{x:g}, {y:g}] is outside the image — `at` is in 0..1 of the image, not pixels",
                self.signature(),
            )
        cx, cy = min(int(x * W), W - 1), min(int(y * H), H - 1)
        lid = int(lab[cy, cx])
        if lid <= 0:
            return Refused(
                f"{self.kind}: [{x:g}, {y:g}] is on no mapped structure in {p['layer']!r} — "
                f"put the point inside a structure the map shows",
                self.signature(),
            )
        return Applied(cv2.bitwise_or(m, ((lab == lid).astype(np.uint8) * 255)), p["_adjusted"])


class ScaleLayer(_LayerOp):
    kind, fit = "scale_layer", "scale"
    # No cross-reference to the sibling layer ops: an op's prompt is filtered in on its own, so
    # naming another one offers a run that allows this op and not that one something it cannot use.
    prompt = (
        "scale_layer{layer:__LAYERS__} — FIT: scale each region up to the structure's bounding "
        "box, keeping its own shape. Use it when you want the extent but not the outline"
    )


OPS["translate"] = Translate()

OPS["rotate"] = Rotate()

OPS["scale"] = Scale()

OPS["stretch"] = Stretch()

OPS["flip"] = Flip()

OPS["dilate"] = Dilate()

OPS["erode"] = Erode()

OPS["close"] = Close()

OPS["open"] = Open()

OPS["gradient"] = Gradient()

OPS["smooth"] = Smooth()

OPS["clear"] = Clear()
OPS["convex"] = Convex()

OPS["simplify"] = Simplify()

OPS["keep_largest"] = KeepLargest()

OPS["fill_holes"] = FillHoles()

OPS["brush"] = Brush()

OPS["erase"] = Erase()

OPS["add_polygon"] = AddPolygon()

OPS["erase_polygon"] = ErasePolygon()

OPS["conform"] = Conform()

OPS["snap"] = Snap()

OPS["snap_layer"] = SnapLayer()

OPS["fill_layer"] = FillLayer()

OPS["scale_layer"] = ScaleLayer()

OPS["add_region"] = AddRegion()


# --- SHAPES — turn 1's vocabulary, as ops: they produce area from nothing ---------------------------
# `draw` rasterizes through PIL onto the canvas it is handed. Moving the pixels to numpy would change
# them: PIL and cv2 rasterize edges differently.
class _Shape(Op):
    group = "SHAPE (draws area from nothing; turn 1 starts from a blank canvas)"

    def draw(self, d, p, w, h, canvas, ctx, adj: list) -> str:
        raise NotImplementedError

    def apply(self, m, p, ctx):
        canvas = Image.fromarray(m)
        # `adj` is threaded rather than returned, because `draw`'s return value is already the other
        # outcome channel (the refusal reason).
        adj = list(p["_adjusted"])
        why = self.draw(ImageDraw.Draw(canvas), p, m.shape[1], m.shape[0], canvas, ctx, adj)
        if why:
            return Refused(why, self.signature())
        return Applied(np.asarray(canvas), tuple(adj))


class Disk(_Shape):
    kind = "disk"
    prompt = "disk{center,radius} — a filled circle"
    params = (
        Param("center", "point", _POINT_NORMALIZED, True, None),
        Param("radius", "number", _FRAC_OF_SHORT_SIDE, True, None, 0.0, 1.0),
    )

    def draw(self, d, p, w, h, canvas, ctx, adj) -> str:
        cx, cy = _px(p["center"], w, h)
        r = _scale(p["radius"], w, h, adj=adj, name="radius")
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        return ""


class Capsule(_Shape):
    kind = "capsule"
    prompt = "capsule{a,b,width} — an elongated blob connecting two points"
    params = (
        Param("a", "point", _POINT_NORMALIZED, True, None),
        Param("b", "point", _POINT_NORMALIZED, True, None),
        Param("width", "number", _FRAC_OF_SHORT_SIDE, False, 0.012, 0.0, 1.0),
    )

    def draw(self, d, p, w, h, canvas, ctx, adj) -> str:
        a, b = _px(p["a"], w, h), _px(p["b"], w, h)
        r = _scale(p["width"], w, h, adj=adj, name="width") // 2 or 1
        _draw_capsule(d, a, b, r)
        return ""


class Ellipse(_Shape):
    kind = "ellipse"
    prompt = "ellipse{center,rx,ry,angle} — a filled ellipse, optionally rotated"
    params = (
        Param("center", "point", _POINT_NORMALIZED, True, None),
        Param("rx", "number", _FRAC_OF_WIDTH, False, 0.04, 0.0, 1.0),
        Param("ry", "number", _FRAC_OF_HEIGHT, False, 0.04, 0.0, 1.0),
        Param("angle", "number", "degrees", False, 0.0, -360.0, 360.0),
    )

    def draw(self, d, p, w, h, canvas, ctx, adj) -> str:
        cx, cy = _px(p["center"], w, h)
        rx, ry = _scale_xy(p["rx"], p["ry"], w, h, adj=adj, nx="rx", ny="ry")
        rx, ry = _bounded_extent(rx, w, h), _bounded_extent(ry, w, h)
        tmp = Image.new("L", (2 * rx + 1, 2 * ry + 1), 0)
        ImageDraw.Draw(tmp).ellipse([0, 0, 2 * rx, 2 * ry], fill=255)
        tmp = tmp.rotate(p["angle"], expand=True)
        canvas.paste(255, (cx - tmp.width // 2, cy - tmp.height // 2), tmp)
        return ""


class Rectangle(_Shape):
    kind = "rectangle"
    prompt = "rectangle{center,rx,ry,angle} — a filled rectangle (rx,ry are HALF-extents), rotatable"
    params = (
        Param("center", "point", _POINT_NORMALIZED, True, None),
        Param("rx", "number", _FRAC_OF_WIDTH, False, 0.05, 0.0, 1.0),
        Param("ry", "number", _FRAC_OF_HEIGHT, False, 0.05, 0.0, 1.0),
        Param("angle", "number", "degrees", False, 0.0, -360.0, 360.0),
    )

    def draw(self, d, p, w, h, canvas, ctx, adj) -> str:
        cx, cy = _px(p["center"], w, h)
        rx, ry = _scale_xy(p["rx"], p["ry"], w, h, adj=adj, nx="rx", ny="ry")
        # Degrees COUNTER-clockwise on screen, matching Ellipse (PIL's rotate() convention) — not the
        # y-up textbook formula, which turns clockwise once applied to these y-DOWN pixel coordinates.
        ang = math.radians(p["angle"])
        corners = [(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)]
        pts = [
            (int(cx + x * math.cos(ang) + y * math.sin(ang)), int(cy - x * math.sin(ang) + y * math.cos(ang)))
            for x, y in corners
        ]
        d.polygon(pts, fill=255)
        return ""


class Ring(_Shape):
    kind = "ring"
    prompt = "ring{center,radius,inner} — an annulus: the outer disk minus the inner one"
    params = (
        Param("center", "point", _POINT_NORMALIZED, True, None),
        Param("radius", "number", "outer, fraction of the short side", True, None, 0.0, 1.0),
        Param("inner", "number", "hole, fraction of the short side", False, None, 0.0, 1.0),
    )

    def draw(self, d, p, w, h, canvas, ctx, adj) -> str:
        cx, cy = _px(p["center"], w, h)
        R = _bounded_extent(_scale(p["radius"], w, h, adj=adj, name="radius"), w, h)
        # `inner` capped against the already-bounded OUTER radius: uncapped, an absurd `inner`
        # effectively never finishes.
        inner = p["inner"] if p["inner"] is not None else p["radius"] * 0.6
        ri = min(R, _bounded_extent(_scale(inner, w, h, adj=adj, name="inner"), w, h))
        if ri >= R:
            # ZERO THICKNESS is a refusal, not an empty `Applied`: two independently legal values can
            # floor onto each other (`radius: 0.001, inner: 0.0005` on a 200px board are both 1px),
            # and `proposer_short` counts `Refused` reasons, so an empty `Applied` reaches it as a
            # candidate that rendered nothing for no stated reason.
            return (
                f"ring has no thickness: outer radius {p['radius']:g} and inner {inner:g} both "
                f"land on {R}px at this size — widen the gap between them"
            )
        tmp = Image.new("L", (2 * R + 1, 2 * R + 1), 0)
        td = ImageDraw.Draw(tmp)
        td.ellipse([0, 0, 2 * R, 2 * R], fill=255)
        td.ellipse([R - ri, R - ri, R + ri, R + ri], fill=0)  # punch the hole
        canvas.paste(255, (cx - R, cy - R), tmp)
        return ""


class Arc(_Shape):
    kind = "arc"
    prompt = "arc{center,rx,ry,start,end,angle} — a filled elliptic sector (a pie slice)"
    params = (
        Param("center", "point", _POINT_NORMALIZED, True, None),
        Param("rx", "number", _FRAC_OF_WIDTH, False, 0.05, 0.0, 1.0),
        Param("ry", "number", _FRAC_OF_HEIGHT, False, 0.05, 0.0, 1.0),
        Param("start", "number", "sector start, degrees", False, 0.0, -360.0, 360.0),
        Param("end", "number", "sector end, degrees", False, 90.0, -360.0, 360.0),
        Param("angle", "number", "rotate the whole sector, degrees", False, 0.0, -360.0, 360.0),
    )

    def draw(self, d, p, w, h, canvas, ctx, adj) -> str:
        cx, cy = _px(p["center"], w, h)
        rx, ry = _scale_xy(p["rx"], p["ry"], w, h, adj=adj, nx="rx", ny="ry")
        rx, ry = _bounded_extent(rx, w, h), _bounded_extent(ry, w, h)
        tmp = Image.new("L", (2 * rx + 1, 2 * ry + 1), 0)
        ImageDraw.Draw(tmp).pieslice([0, 0, 2 * rx, 2 * ry], p["start"], p["end"], fill=255)
        if p["angle"]:
            tmp = tmp.rotate(p["angle"], expand=True)
        canvas.paste(255, (cx - tmp.width // 2, cy - tmp.height // 2), tmp)
        return ""


class Stamp(_Shape):
    kind = "stamp"
    prompt = "stamp{shape_id,center,scale,angle} — paste a REAL shape from the example masks"
    requires = ("shapes",)
    params = (
        Param("shape_id", "int", "index into the stamp library", True, None, 0, None),
        Param("center", "point", _POINT_NORMALIZED, True, None),
        Param("scale", "number", "multiplier of the shape's native size", False, 1.0, 0.0, None),
        Param("angle", "number", "degrees", False, 0.0, -360.0, 360.0),
    )

    def draw(self, d, p, w, h, canvas, ctx, adj) -> str:
        shapes = getattr(ctx, "shapes", None)
        if shapes is None:
            return "stamp needs a shape library on this run"
        sid = int(p["shape_id"])
        if not (0 <= sid < len(shapes)):
            return f"stamp shape_id {sid} is outside 0..{len(shapes) - 1}"
        s = shapes[sid]
        sc = float(p["scale"])
        # ISOTROPIC: both extents against the SAME reference, so a stamped shape keeps the aspect
        # ratio of the example mask it came from. Sizing width against W and height against H would
        # restretch the shape by the board's own aspect, which a verbatim "stamp" must not do.
        ref = min(w, h)
        # Reported, because unreported `scale` 0.0005 through 0.05 all collapse to the SAME single
        # lit pixel — a hundredfold span the model cannot tell apart.
        if min(s["wf"], s["hf"]) * sc * ref < 1:
            adj.append(
                _floor_note(
                    "scale",
                    sc,
                    int(min(s["wf"], s["hf"]) * ref),
                    "native size of shape %d" % sid,
                    outcome="floored to a 1px stamp",
                )
            )
        tw, th = (_bounded_extent(s["wf"] * sc * ref, w, h), _bounded_extent(s["hf"] * sc * ref, w, h))
        pim = Image.fromarray((s["patch"].astype("uint8")) * 255).resize((tw, th))
        pim = pim.point(lambda v: 255 if v > 127 else 0)  # re-binarize after resample
        if p["angle"]:
            pim = pim.rotate(p["angle"], expand=True)
        cx, cy = _px(p["center"], w, h)
        canvas.paste(255, (cx - pim.width // 2, cy - pim.height // 2), pim)
        return ""


# DIRECTLY into `OPS`: turn 1 and refine apply their ops through the same `apply_cv_sequence`, so
# there is one registry.
for _c in (Disk, Capsule, Ellipse, Rectangle, Ring, Arc, Stamp):
    OPS[_c.kind] = _c()

# `polygon` and `stroke` are `add_polygon` and `brush` under another name — same parameters, same
# pixels. Only the registry names go in the catalogue, so the model is taught one name per tool.
_MERGED_AWAY = {"polygon": "add_polygon", "stroke": "brush"}

# Parameters that belong to ALL ops rather than one, honoured by `apply_op` around whatever op it
# dispatches to — so no `Op` declares them and none can be found by walking `OPS[...].params`.
# `prompts` renders the cheat-sheet line from this table.
CROSS_CUTTING = {
    "target": "[x,y] normalized — apply this op to ONLY the connected region nearest that point",
}


def canonical_kind(kind) -> str:
    """The registry name for a kind as the model may have spelled it.

    PUBLIC, and used by every filter: a filter that does not resolve the alias drops the request
    before `apply_op` can resolve it, then reports it as an op the run does not offer.
    """
    return _MERGED_AWAY.get(str(kind), str(kind))


def filter_to_vocabulary(ops, allowed) -> tuple[list, list[str], list[dict]]:
    """Keep only the ops this run offers, and say what was dropped and why.

    Returns `(kept, withdrawn, no_kind)`: the ops that survive, the DEDUPED kinds `allowed` does not
    carry, and the entries that named no `kind` at all. The caller does the logging, because turn 1
    and a refine round word the same event differently.

    ALIAS-AWARE, because `apply_op` is. `canonical_kind` stringifies, so a `"kind"` arriving as a
    list or a number is dropped as unknown rather than raising — but see `record_label` for what such
    a kind must NOT then be used as.
    """
    kept, withdrawn, no_kind = [], {}, []
    for o in ops:
        if not isinstance(o, dict):
            continue
        # `str(None)` is `"None"` — a valid identifier, so it survives every sanitizer and becomes
        # `degraded[op_withdrawn:None]`, a record naming an op that does not exist. `{"kind": null}`
        # goes to the no-kind bucket instead.
        _k = o.get("kind")
        kind = "" if _k is None else str(_k)
        if not kind:
            no_kind.append(o)
        elif canonical_kind(kind) in allowed:
            kept.append(o)
        else:
            # CANONICAL, so the histogram counts REGISTRY OPS rather than giving `polygon` and
            # `add_polygon` a bucket each. The cost is that a run log says "asked for add_polygon"
            # where the reply said `polygon`.
            withdrawn[canonical_kind(kind)] = None
    return kept, list(withdrawn), no_kind


# A degradation label is parsed back out of the log with `degraded\[([^]]*)\]`, and the bucket it
# names interpolates a KIND — a value the MODEL chose, which the JSON repair path can hand through
# as anything (`{"kind": ["disk","brush"]}`). Only the BUCKET is constrained; the kind itself goes
# in the sentence, where it is not parsed.
#
# `\A`/`\Z`, not `^`/`$`: Python's `$` also matches BEFORE a trailing newline, so `"dilate\n"` is a
# name by that test. It buckets as `op_withdrawn:dilate\n`, which PRINTS as `op_withdrawn:dilate`
# and compares unequal to it — one op's withdrawals arrive as two rows no reader can tell apart.
_LABEL_SAFE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


def record_label(kind: str) -> str:
    """`kind` as a degradation bucket label, or `unparsable` when it is not a name."""
    return kind if _LABEL_SAFE.match(kind) else "unparsable"


def unavailable(available) -> frozenset:
    """Ops this beam cannot run, because something they DECLARE they need is not here.

    `"mask"` is excluded on purpose: it is a property of the TURN, not the beam — turn 1 has no mask
    by definition, and that is what makes an op an init op rather than an unavailable one.
    """
    have = set(available)
    return frozenset(k for k, sp in OPS.items() if set(sp.requires) - have - {"mask"})


# DERIVED, not restated: a hand-written copy is a second list that has to agree with the registry.
CV_KINDS = frozenset(OPS)

# Shape-PRESERVING subset. Fixed-shape mode restricts refine to these so a stamped REAL
# training-mask shape is only repositioned/oriented, never deformed.
RIGID_CV_KINDS = frozenset(k for k, sp in OPS.items() if sp.rigid)


# --------------------------------------------------------------------------------------------------
# Every adjustment to a MODEL-SUPPLIED value made by the SHARED HELPERS, keyed by the FUNCTION that
# makes it. A bound added to a helper already listed is covered; a bound in a NEW helper belongs here,
# classified. Per-op caps live with their op instead — `Op.check_input`'s declared lo/hi,
# `_Polygon`'s `corner_px`, `_Stroke`'s `width`.
#
#   SILENT — each stands in for a request the rasterizer cannot carry out as written, so the
#   substitute is the only reading there was:
#     _px              normalized coords clamped to [0,1], then to the pixel grid
#     _kernel          max(1, px)                                        a zero-size kernel
#     _bounded_extent  max(1, min(px, 4 * long side))                    a BUFFER past the diagonal
#
#   REPORTED in `Applied.adjustments` — each hands back a mask of a DIFFERENT size than was asked
#   for, which the model reads as the size it chose unless it is told:
#     _scale/_scale_xy max(1, frac * ref)      sub-pixel length -> 1px, via `_floor_note`
#     _kernel_px       max(0, min(n, short // 2))                        a kernel past half the mask
#
# `_apply_to_one_region` is in neither class: `target:[x,y]` CHOOSES rather than being clamped, and a
# target on an empty mask is refused outright.


def _absent(v) -> bool:
    """Was this parameter supplied? Not `in (None, "", [], ())` — a caller that pre-parses can hand
    in a numpy array, and a membership test on one is ambiguous."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, tuple)):
        return len(v) == 0
    return False


@dataclass(frozen=True)
class Applied:
    """The op ran. `adjustments` names anything the engine changed about what was asked for."""

    mask: Any
    adjustments: tuple = ()


@dataclass(frozen=True)
class Refused:
    """The op did not run, the mask is untouched, and here is what to send instead.

    Nothing is invented on the model's behalf: a request with no parameters is refused rather than
    applied as an identity, which the agent would read as "N to N px (1.0x)" — "too timid" instead
    of "you did not say how far".
    """

    reason: str
    correction: str


def _apply_to_one_region(sp, arr, p, ctx, target, kind: str) -> "Applied | Refused":
    """`target:[x,y]` — run the op against ONE connected region and put the rest back untouched.

    The op sees a mask containing only its region, so no op knows about this at all.

    It REFUSES rather than falling back to the whole mask when there is no region to pick: a fallback
    edits everything reasonable-looking, and the model reads the result as confirmation that its
    point selected what it meant.
    """
    n, lab = cv2.connectedComponents((arr > 127).astype(np.uint8))
    if n <= 1:
        return Refused(
            f"{kind}: target was given but this mask has no regions to select", "omit `target` to edit the whole mask"
        )
    H, W = arr.shape
    try:
        tx, ty = float(target[0]) * W, float(target[1]) * H
        if not (math.isfinite(tx) and math.isfinite(ty)):
            raise ValueError("a non-finite coordinate is nearest to nothing")
    except (TypeError, ValueError, IndexError):
        return Refused(
            f"{kind}: target must be [x,y] normalized 0..1, got {target!r}",
            "target:[x,y] — a point inside the region you mean",
        )
    # NEAREST BY DISTANCE TO THE REGION, not to its centroid: on an elongated defect the centroid can
    # sit further from a point inside the region than another region's centroid does.
    best, best_d = 0, None
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        d = float(np.min((xs - tx) ** 2 + (ys - ty) ** 2))
        if best_d is None or d < best_d:
            best, best_d = i, d
    only = np.where(lab == best, np.uint8(255), np.uint8(0))
    rest = np.where((arr > 127) & (lab != best), np.uint8(255), np.uint8(0))
    r = sp.apply(only, p, ctx)
    if isinstance(r, Refused):
        return r
    # `sp.apply` returns a NUMPY mask (it was handed one); `_to_np` is the PIL->numpy reader at the
    # `apply_op` boundary and does not accept an array.
    got = r.mask if isinstance(r.mask, np.ndarray) else _to_np(r.mask)
    out = cv2.bitwise_or(got, rest)
    return Applied(
        _to_pil(out),
        tuple(r.adjustments)
        + (f"applied to the region nearest {list(target)} only, {n - 2} other region(s) left untouched",),
    )


def apply_op(
    op: dict, mask: Image.Image, clean: Image.Image | None = None, *, layers=None, shapes=None
) -> "Applied | Refused":
    """Apply ONE op. Returns an outcome; never raises.

    An op that raises takes its whole beam with it. The outcome is the contract instead, so a run
    that is failing every op still completes and still records them.
    """
    if not isinstance(op, dict):
        # "never raises" is unconditional, so the FIRST line has to hold for anything a reply can
        # contain — `apply_op([])` and `apply_cv_sequence([None])` reach here.
        return Refused(
            f"an op must be an object, got {type(op).__name__}",
            f'send {{"kind": ...}}; one of: {", ".join(sorted(OPS))}',
        )
    # `canonical_kind` stringifies BEFORE any lookup: an unhashable kind (`{"kind": ["dilate"]}`)
    # would otherwise raise on set/dict membership rather than being refused.
    kind = canonical_kind(op.get("kind", ""))
    sp = OPS.get(kind)
    if sp is None:
        return Refused(f"{kind!r} is not an op", f"choose one of: {', '.join(sorted(OPS))}")
    missing = [p.name for p in sp.required_params() if _absent(op.get(p.name))]
    for group in sp.one_of:
        if all(_absent(op.get(n)) for n in group):
            missing.append(" or ".join(group))
    if missing:
        return Refused(f"{kind} was asked for without {', '.join(missing)}", sp.signature())
    if "layer" in sp.requires:
        # `isinstance` and `is None`, never truthiness: a layer map IS a numpy array, so both
        # `layers or {}` and `not have.get(...)` raise rather than answering. `sorted()` is over
        # stringified keys because it must not raise while CONSTRUCTING the Refused.
        have = layers if isinstance(layers, dict) else {}
        if have.get(str(op.get("layer", ""))) is None:
            names = ", ".join(sorted(str(k) for k in have)) or "none on this board"
            return Refused(
                f"{kind} refused: this clean has no layer named {op.get('layer')!r}", f"available layers: {names}"
            )
    if "clean" in sp.requires and clean is None:
        return Refused(f"{kind} refused: it needs the clean image and none was given", sp.signature())
    ctx = Context(clean=clean, layers=layers, shapes=shapes)
    # `check_input` sits inside this try as belt-and-braces: it names `OverflowError` itself
    # (`{"px":1e999}` is `inf`, and `int(inf)` raises neither TypeError nor ValueError), but a type
    # branch added tomorrow must not be able to make "never raises" false again.
    try:
        p = sp.check_input(op)
        if isinstance(p, Refused):
            return p
        arr = _to_np(mask)
        if not _absent(op.get("target")):
            return _apply_to_one_region(sp, arr, p, ctx, op["target"], kind)
        r = sp.apply(arr, p, ctx)
        return Applied(_to_pil(r.mask), r.adjustments) if isinstance(r, Applied) else r
    except Exception as e:  # noqa: BLE001 — an op reports, it does not raise
        log.warning(f"degraded[op_failed:{kind}]: {type(e).__name__}: {e}")
        return Refused(f"{kind} failed: {type(e).__name__}: {e}", sp.signature())


def apply_cv_sequence(
    ops: list[dict],
    prev_mask: Image.Image,
    clean: Image.Image | None = None,
    layers: dict | None = None,
    outcomes: list | None = None,
    shapes: list | None = None,
) -> Image.Image:
    """Apply a sequence of basic CV ops in order — lets the reasoner compose primitives
    (e.g. rotate, then stretch, then translate, then conform-to-image) in one fix."""
    m = prev_mask
    for op in ops:
        # NO PRE-FILTER: `apply_op` is the only thing that decides what a kind is. A second copy of
        # that judgement here would `continue` past a request it did not recognise — an alias like
        # `polygon` — leaving no outcome recorded and a candidate that drew nothing for no stated
        # reason. `apply_op` refuses a non-dict and an unhashable kind, so nothing needs guarding.
        r = apply_op(op, m, clean, layers=layers, shapes=shapes)
        if outcomes is not None:
            outcomes.append((op.get("kind") if isinstance(op, dict) else op, r))
        if isinstance(r, Applied):
            m = r.mask
        # a Refused leaves the mask exactly as it was — that IS the refusal
    return m
