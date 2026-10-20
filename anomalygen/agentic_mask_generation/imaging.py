# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the model perceives: the board and the mask, as pixels and as words — renderers that draw the
board with a coordinate grid, the mask overlaid, a magnified crop or a comparison montage, and
describers that say the same state in a sentence the prompt can carry.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


def describe_shapes(lib: list[dict]) -> str:
    """One-line catalogue of the stamp shape library for the prompt."""
    if not lib:
        return "(none)"
    return "; ".join(
        f"id {s['id']}: {s['kind']} of example {s['src']} "
        f"(~{s['wf'] * 100:.0f}%×{s['hf'] * 100:.0f}% of its image, {s['ncomp']} region(s))"
        for s in lib
    )


# View-building CV: none of the free helpers below sets a DELIVERY size — they magnify, or (in
# `beam_montage`) shrink to a fixed layout panel; `Views` alone brings a view down to what is sent,
# before any grid is drawn.
ZOOM_MAGNIFY = 1024  # a small crop is UPSCALED to this so fine detail is legible
MASK_PAD = 1.5  # pad around a TIGHT mask bbox (magnify window keeps the host feature in frame)
REGION_PAD = 0.3  # pad around an ALREADY-generous localize work-region box
ZOOM_FLOOR = 0.1  # floors the HALF-window, so the smallest magnify window is 20% of the image
MONTAGE_COLS = 3  # candidates per row in the comparison grid
MONTAGE_PANEL = 512  # per-candidate tile px in the grid (a layout size, not a detail size)


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """The one rule for resampling a binary mask: ``mask`` as an ``L`` image at ``size``, NEAREST.

    Every reader in this package applies ``> 127``. An interpolating filter averages a mask's edges
    and thin parts with the background into grey values that fall below that bar — a 1-2 px
    annotation downscaled 3x is dropped from the zoom, the outline ring, the caption and the geometry
    note. NEAREST keeps every value binary and the area unbiased (no dilation, no erosion of what the
    judge's structure numbers are compared against). Its limitation, stated plainly: it keeps one
    source pixel per destination pixel, so a line thinner than the downscale factor is lost where no
    sample falls on it. Same size in: the same pixels out.
    """
    return mask.convert("L").resize(tuple(size), Image.NEAREST)


def overlay_mask(clean: Image.Image, mask: Image.Image, *, judge_vis: bool = False) -> Image.Image:
    """Paint ``mask`` red over ``clean`` — a 150/255 fill, and every caller in this package gets that
    one unless it asks otherwise.

    ``judge_vis`` renders it the way a mask that is about to be COMPARED is drawn instead: a lighter
    90/255 (35%) tint, so the structure UNDER the mask stays readable, plus a 2px solid contour, so
    the exact boundary — what every geometric judgment is actually about — is DRAWN rather than
    implied. Its one caller is the pairwise selector
    (:func:`~anomalygen.agentic_mask_generation.loop._beam_pairwise_winner`), which draws the
    boundary it asks about.

    The contour is ``outline_mask``'s ring, not a polyline stroke: a stroke CENTRED on the contour
    at thickness 2 covers 3px of an axis-aligned edge, while the ring is exactly the 2px asked for;
    and the ring follows EVERY boundary, so a mask with a hole gets the hole outlined too — a hole IS
    a boundary the comparison is being asked about.
    """
    base = clean.convert("RGB")
    if mask.size != base.size:
        mask = resize_mask(mask, base.size)
    red = Image.new("RGB", base.size, (255, 0, 0))
    if not judge_vis:
        return Image.composite(red, base, mask.point(lambda v: 150 if v > 127 else 0))
    return outline_mask(
        Image.composite(red, base, mask.point(lambda v: 90 if v > 127 else 0)), mask, width=2, color=(255, 0, 0)
    )


def outline_mask(
    img: Image.Image, mask: Image.Image, *, width: int | None = None, color: tuple[int, int, int] = (255, 40, 40)
) -> Image.Image:
    """Return ``img`` with the ``mask`` BOUNDARY drawn as a red contour (not filled), at NATIVE size —
    image_max_dim (in _img_block) is the sole shrink cap.

    ``width`` (in px) and ``color`` are for a caller that has to match a FIXED stroke rather than one
    that scales with the board — `overlay_mask(judge_vis=True)` is the only one. Both default to what
    every other caller already got: a band that grows with the image height, in the softer red.
    ``width is not None``, not ``if width``: only None means "did not ask" and gets the board-scaled
    band. A width that was asked for is used as asked, rounded up to the next odd kernel (so 2px is
    the narrowest band this can draw).
    """
    img = img.convert("RGB")
    mask = mask.convert("L")
    if mask.size != img.size:
        mask = resize_mask(mask, img.size)
    binm = mask.point(lambda p: 255 if p > 127 else 0)
    w = int(width) if width is not None else max(2, img.height // 300)
    k = max(3, w if w % 2 else w + 1)  # MaxFilter/MinFilter kernels must be odd
    ring = ImageChops.subtract(binm.filter(ImageFilter.MaxFilter(k)), binm.filter(ImageFilter.MinFilter(k)))
    stroke = Image.new("RGB", img.size, color)
    return Image.composite(stroke, img, ring)


def _crop_upscale(img: Image.Image, box_px, magnify_to: int):
    """Shared zoom core: crop ``img`` to pixel box (L,T,R,B) and UPSCALE the long side up to
    ``magnify_to`` if the crop is smaller. A crop already that big is left NATIVE — image_max_dim in
    _img_block is the sole shrink cap. Returns (crop, box_norm) or (None, None) if degenerate."""
    L, T, R, B = box_px
    if R - L < 4 or B - T < 4:
        return None, None
    W, H = img.size
    crop = img.crop((L, T, R, B))
    scale = magnify_to / max(1, max(crop.size))
    if scale > 1:
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    return crop, (L / W, T / H, R / W, B / H)


def zoom_around_mask(
    img: Image.Image, mask: Image.Image, pad: float = MASK_PAD, magnify_to: int = ZOOM_MAGNIFY, region=None
):
    """Crop ``img`` to a padded box AROUND ``mask``, a small crop upscaled so fine detail is visible.
    Each axis is padded from its OWN bbox extent (no square window from the long side), and the window
    is widened to CONTAIN ``region`` — the beam's normalized work-region box — when one is given, so
    the whole host feature stays in frame.
    Returns (crop, box_norm) in FULL-IMAGE normalized coords, or (None, None) if the mask is empty."""
    if mask.size != img.size:
        mask = resize_mask(mask, img.size)  # window math + box_norm are in img space — keep one frame
    a = np.asarray(mask.convert("L")) > 127
    if not a.any():
        return None, None
    ys, xs = np.nonzero(a)
    W, H = img.size
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    floor = min(W, H) * ZOOM_FLOOR  # floor: a SMALL defect still gets a genuine magnification
    hx = max((x1 - x0) * (1 + pad) / 2, floor)
    hy = max((y1 - y0) * (1 + pad) / 2, floor)
    L, T, R, B = cx - hx, cy - hy, cx + hx, cy + hy
    if region is not None:
        try:
            rx0, ry0, rx1, ry1 = (float(v) for v in region)
            if rx1 > rx0 and ry1 > ry0:
                L, T = min(L, rx0 * W), min(T, ry0 * H)
                R, B = max(R, rx1 * W), max(B, ry1 * H)
        except Exception:  # noqa: BLE001 — a malformed region loses only the containment widening, never the zoom
            pass
    box = (max(0, int(L)), max(0, int(T)), min(W, int(R)), min(H, int(B)))
    return _crop_upscale(img, box, magnify_to)


def zoom_to_box(img: Image.Image, box, magnify_to: int = ZOOM_MAGNIFY, pad: float = REGION_PAD):
    """Crop ``img`` to a GIVEN normalized bbox (x0,y0,x1,y1) — e.g. the localize work_region — padded,
    small crop upscaled. Returns (crop, box_norm) or (None, None) on a degenerate box."""
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except Exception:  # noqa: BLE001 — a malformed box just means "no zoom", not a crash
        return None, None
    if not (x1 > x0 and y1 > y0):
        return None, None
    W, H = img.size
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = max(0.0, x0 - pad * bw), min(1.0, x1 + pad * bw)
    y0, y1 = max(0.0, y0 - pad * bh), min(1.0, y1 + pad * bh)
    return _crop_upscale(img, (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)), magnify_to)


def beam_montage(
    overlays: list[Image.Image], labels: list[str], cols: int = MONTAGE_COLS, panel: int = MONTAGE_PANEL
) -> Image.Image:
    """Lay overlays in a labelled grid so the agent compares them in ONE image. Aspect-preserving
    (letterboxed) — a forced-square resize squashes non-square boards below comparable detail."""
    n = len(overlays)
    rows = max(1, math.ceil(n / cols))
    M = Image.new("RGB", (cols * panel, rows * (panel + 20)), "white")
    d = ImageDraw.Draw(M)
    for i, (ov, lab) in enumerate(zip(overlays, labels)):
        r, c = divmod(i, cols)
        x, y = c * panel, r * (panel + 20)
        d.text((x + 5, y + 4), lab, fill="black")
        t = ov.convert("RGB").copy()
        t.thumbnail((panel - 8, panel - 8))
        M.paste(t, (x + 4 + (panel - 8 - t.width) // 2, y + 18 + (panel - 8 - t.height) // 2))
    return M


# --- visual aids: a coordinate grid (poor-man's Set-of-Mark) + mask geometry ---
# No per-defect-type logic here: these describe what is being shown, never encode a rule about a type.
def draw_grid(img: Image.Image, n: int = 10) -> Image.Image:
    """Overlay a light grid, ticked every 1/n, labelled in normalized coords — the only coordinate
    unit the agent reads or writes."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for i in range(n + 1):
        x, y = round(i / n * (w - 1)), round(i / n * (h - 1))
        col = (90, 200, 255) if i % 5 == 0 else (60, 90, 110)
        d.line([(x, 0), (x, h)], fill=col, width=1)
        d.line([(0, y), (w, y)], fill=col, width=1)
        if i % 5 == 0:
            lx = ly = f"{i / n:.1f}"
            d.text((min(x + 2, w - 22), 2), lx, fill=(120, 220, 255))
            d.text((2, min(y + 1, h - 12)), ly, fill=(120, 220, 255))
    return out


def draw_grid_fullcoords(img: Image.Image, box: tuple[float, float, float, float], step: float = 0.05) -> Image.Image:
    """Grid on a ZOOM crop, labelled in FULL-IMAGE normalized coords (the crop covers ``box``), so a
    position read at the zoomed scale can be written straight back as a full-image dx/dy."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    x0, y0, x1, y1 = box
    bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    X = math.ceil(x0 / step) * step
    while X <= x1:
        px = int((X - x0) / bw * (w - 1))
        d.line([(px, 0), (px, h)], fill=(90, 200, 255), width=1)
        d.text((min(px + 2, w - 30), 2), f"{X:.2f}", fill=(120, 220, 255))
        X += step
    Y = math.ceil(y0 / step) * step
    while Y <= y1:
        py = int((Y - y0) / bh * (h - 1))
        d.line([(0, py), (w, py)], fill=(90, 200, 255), width=1)
        d.text((2, min(py + 1, h - 12)), f"{Y:.2f}", fill=(120, 220, 255))
        Y += step
    return out


class Views:
    """Every image the model LOOKS at, built at the size it will be DELIVERED at.

    THE INVARIANT: every image returned here is already at its delivered size, and any grid on it was
    drawn after that resize. `draw_grid` writes with PIL's default font and ``width=1`` lines,
    both ABSOLUTE pixels, so a grid drawn before the transport's ``image_max_dim`` shrink degrades by
    exactly that factor — and reading the zoom's grid is the judge prompt's mandatory first step.

    Knowing the delivered size here relies on ``sent_png_bytes``'s SECOND shrink (a base64 byte-cap
    loop) never firing: its cap is 4.5 MB and a 2048px gridded PNG is 1.7 MB of base64. If that stops
    being true, the grid has to move into the transport. ``deliver_at=0`` means native.
    """

    def __init__(self, deliver_at: int = 0):
        self.deliver_at = max(0, int(deliver_at or 0))

    @classmethod
    def from_vlm(cls, vlm) -> "Views":
        """Read the delivery cap off the transport (`Vlm(img_maxdim=cfg.image_max_dim)`), as a plain
        attribute read with NO default: a transport that cannot say its cap has to fail here rather
        than silently deliver native — a missing delivery cap is an error, not a zero.
        """
        return cls(vlm.image_max_dim)

    # --- the invariant's one enforcement point ---------------------------------------------------
    def fit(self, img: Image.Image) -> Image.Image:
        """Bring an image to its delivered size; never upscales (that is `zoom`'s magnification)."""
        cap = self.deliver_at
        if not cap or img is None or max(img.size) <= cap:
            return img
        out = img.copy()
        out.thumbnail((cap, cap), Image.LANCZOS)
        return out

    @property
    def magnify_to(self) -> int:
        """Never magnify past the delivered size: upscaling to 1024 and then delivering 384 spends the
        interpolation twice and throws it away."""
        return min(ZOOM_MAGNIFY, self.deliver_at) if self.deliver_at else ZOOM_MAGNIFY

    def grid(self, img: Image.Image, n: int = 10) -> Image.Image:
        return draw_grid(self.fit(img), n)

    def grid_fullcoords(self, img: Image.Image, box, step: float = 0.05) -> Image.Image:
        return draw_grid_fullcoords(self.fit(img), box, step)

    def zoom_around_mask(self, img, mask, **kw):
        kw.setdefault("magnify_to", self.magnify_to)
        crop, box = zoom_around_mask(img, mask, **kw)
        return self.fit(crop), box

    def zoom_to_box(self, img, box, **kw):
        kw.setdefault("magnify_to", self.magnify_to)
        crop, b = zoom_to_box(img, box, **kw)
        return self.fit(crop), b

    def montage(self, overlays: list[Image.Image], labels: list[str], cols: int = MONTAGE_COLS) -> Image.Image:
        """The candidate comparison grid, with its tiles built at the size they are DELIVERED at.

        Most exposed to stacked shrinks: a full-size tile thumbnailed into a panel, inside a grid the
        transport shrinks again. Tiles are sized first and gridded at tile size, so the labels the
        agent is asked to read are the same height in every arm."""
        n = max(1, len(overlays))
        rows = max(1, math.ceil(n / cols))
        panel = MONTAGE_PANEL
        if self.deliver_at:
            # bound BOTH axes: a tall montage hits the cap on its height, the axis nobody checks
            panel = min(panel, self.deliver_at // cols, self.deliver_at // rows - 20)
        panel = max(64, int(panel))
        tiles = [self._tile(ov, panel - 8) for ov in overlays]
        return beam_montage(tiles, labels, cols=cols, panel=panel)

    def _tile(self, ov: Image.Image, side: int) -> Image.Image:
        t = ov.convert("RGB").copy()
        t.thumbnail((side, side), Image.LANCZOS)
        return draw_grid(t)


def mask_geometry(mask: Image.Image) -> dict:
    """Normalized descriptors of a binary mask: area, centroid, bbox, fill-within-bbox."""
    a = np.asarray(mask.convert("L")) > 127
    h, w = a.shape
    on = int(a.sum())
    if on == 0:
        return {"area_ratio": 0.0, "centroid": None, "bbox": None, "fill_in_bbox": 0.0}
    ys, xs = np.nonzero(a)
    x0, x1, y0, y1 = xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h
    bbox_area = max(1, (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return {
        "area_ratio": round(on / a.size, 4),
        "centroid": [round(float(xs.mean()) / w, 3), round(float(ys.mean()) / h, 3)],
        "bbox": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
        "fill_in_bbox": round(on / bbox_area, 3),  # low => thin/scattered, high => solid region
    }


def describe_geometry(g: dict, include_position: bool = False) -> str:
    if not g["centroid"]:
        return "empty"
    tag = "scattered/thin" if g["fill_in_bbox"] < 0.4 else "solid region"
    if not include_position:
        # SHAPE ONLY: omitting absolute centroid/bbox is what stops an EXEMPLAR's geometry being
        # copied as a position — scale and framing differ, placement must be RELATIVE to structure.
        bb = g["bbox"]
        aspect = round((bb[2] - bb[0]) / max(1e-6, bb[3] - bb[1]), 2) if bb else None
        return (
            f"area={g['area_ratio'] * 100:.2f}% of image, fill_in_bbox={g['fill_in_bbox']}, "
            f"aspect(w/h)={aspect} ({tag}) — SHAPE only; position is NOT given (read it relative "
            "to the structure, do not copy absolute coords)"
        )
    return (
        f"area={g['area_ratio'] * 100:.2f}% of image, centroid=({g['centroid'][0]},{g['centroid'][1]}), "
        f"bbox={g['bbox']}, fill_in_bbox={g['fill_in_bbox']} "
        f"({tag})"
    )


def binarize(raw) -> tuple[np.ndarray, list[str]]:
    """Any caller's image, as the lit/not-lit array this system means by "a mask".

    ONE rule, because a per-call-site threshold splits by dtype or by maximum and the branches then
    disagree with each other: **a mask's level is the 99th percentile of its non-zero values, and the
    lit half is what is above half of it.** The percentile stops one stray bright pixel from BEING the
    level; half of it reads an antialiased edge the same way in bool, 0/1, 0..255 and float 0..1.
    Inputs that are not amplitudes at all are normalized first, each named in the returned `notes`.
    Every encoding of one mask enters the beam as the same mask.
    """
    notes: list[str] = []
    arr = np.asarray(raw)
    if arr.dtype == bool or not arr.size:
        return arr.astype(bool), notes
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        n = int((~np.isfinite(arr)).sum())
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        notes.append(f"{n} non-finite value(s) read as 0 — NaN and inf are not amplitudes")
    if np.issubdtype(arr.dtype, np.signedinteger) and arr.min() < 0:
        notes.append(
            f"{arr.dtype} with negative values reinterpreted as unsigned — that is 255 "
            f"stored in a signed byte, not an amplitude"
        )
        arr = arr.view(np.dtype(f"uint{arr.dtype.itemsize * 8}"))
    elif arr.min() < 0:
        n = int((arr < 0).sum())
        notes.append(
            f"{n} negative value(s) clipped to 0 — a resampling undershoot, or the "
            f"outside of a distance field, is not amplitude"
        )
        arr = np.clip(arr, 0, None)
    nz = arr[arr > 0]
    if not nz.size:
        return np.zeros(arr.shape, bool), notes
    return arr > (float(np.percentile(nz, 99)) / 2), notes
