# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable visual prompts injected into the proposer/judge.

A visual prompt is an extra bundle of image/text blocks appended to the model's context to help it
place a defect mask. The maps are computed OFFLINE and cached under a ``vp_dir`` (keyed by the clean
image's hash); the loader here only READS those PNGs, and every visual prompt is a silent no-op
(returns ``[]``) when its artifact is absent. Select one by name with :func:`by_name`.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from cosmos_framework.utils import log
from matplotlib import colormaps
from PIL import Image

from anomalygen.agentic_mask_generation import sam2
from anomalygen.agentic_mask_generation.imaging import Views, resize_mask
from anomalygen.agentic_mask_generation.utils import atomic_write, clean_hash, resize_labels


def _labels_path(png: Path) -> Path:
    return png.with_suffix(".labels.npz")


def save_labels(png: Path, lab: np.ndarray, *, shape: tuple | None = None) -> None:
    """Persist the structure labels a precompute already computed, beside the PNG it drew from them.

    The labels cannot be recovered from that PNG: it is a lossy blend of colour onto the board for
    the MODEL, which a board pixel of the right hue imitates without ever having been in a segment.

    ``shape`` resamples to the CLEAN's (H, W) first — 2048x2048 int32 is 16 MB a board and one
    dataset here has 531 of them — with NEAREST, because any interpolation invents ids that belong
    to no structure.
    """
    if shape is not None:
        lab = resize_labels(lab, int(shape[1]), int(shape[0]))

    # Atomically: `_usable` re-decodes the PNG but nothing re-checks the labels beside it, so a
    # half-written .npz survives every run after — the guard that would rebuild it sees a fine PNG.
    def _save(t: Path) -> None:
        with open(t, "wb") as fh:  # a handle, not the path: savez appends `.npz` to a name without it
            np.savez_compressed(fh, labels=lab.astype(np.int32))

    atomic_write(_labels_path(png), _save)


def load_labels(png: Path) -> np.ndarray | None:
    """The labels for a map, or None when there are none (caller withdraws the layer ops)."""
    p = _labels_path(png)
    if not p.is_file():
        return None
    try:
        with np.load(p) as z:
            return z["labels"]
    except Exception as e:  # noqa: BLE001
        # BROAD on purpose: a truncated .npz raises zipfile.BadZipFile and a zero-byte one raises
        # EOFError, neither of which derives from OSError or ValueError. `degraded[KIND]: detail` is
        # the package's ledger form (test_a_degradation_line_is_the_shape_its_readers_parse).
        log.warning(f"degraded[vp_labels_unreadable]: {p}: {type(e).__name__}: {e} — treating the layer as absent")
        return None


def _components(mask: np.ndarray, *, min_area_frac: float = 2e-5) -> np.ndarray:
    """A boolean region map -> one label per connected STRUCTURE.

    LABELLED, never binary: ops.layer_fit unions a mask with every pixel carrying the ids it overlaps,
    so a binary layer makes every highlighted patch on the board ONE structure and fill_layer grows
    the mask over all of them. The area floor is load-bearing too — a physics gate answers per PIXEL,
    and without it a real board yields 60-210 "structures", median THREE pixels, and `fill_layer`
    becomes a silent no-op on most real solder joints.
    """
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    floor = max(6, int(min_area_frac * m.shape[0] * m.shape[1]))  # sensor speckle, not a structure
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= floor]
    return np.where(np.isin(lab, keep), lab, 0).astype(np.int32)


def _load_rgb(path: Path) -> Image.Image | None:
    """Read a cached visual-prompt PNG as RGB; ``None`` if it is missing or unreadable.

    MISSING is silent — the documented no-op, the state every clean is in before its first precompute.
    PRESENT-BUT-UNREADABLE is logged: a precompute killed mid-write leaves a truncated PNG, ``build``
    then returns no blocks at all, and nothing else here would say so.
    """
    if not path.is_file():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001 — see load_labels: a truncated PNG raises from several trees
        log.warning(f"degraded[vp_map_unreadable]: {path}: {type(e).__name__}: {e} — treating the cue as absent")
        return None


def _usable(*paths: Path) -> bool:
    """True iff every artifact is not merely PRESENT but actually loads.

    A truncated map makes ``build()`` return ``[]`` forever while the file is still there, so the
    precompute that would replace it never runs and the cue is gone until someone deletes the file by
    hand. Costs a decode of each map — small against the precompute it guards (a CV pass for the
    physics prompts, a GPU pass for sam2) — and short-circuits, so a cold cache pays nothing.
    """
    return all(_load_rgb(p) is not None for p in paths)


SOLDERLIGHT_CAPTION = (
    "SOLDER-LIGHT SLOPE MAP (physics, no ML). This board is imaged under tri-color ring light: surface "
    "SLOPE is encoded as color (flat→red, mid slope→green, steep→blue). Image 1 = per-pixel slope "
    "heatmap (warm = flat, cold = steep): a HEALTHY solder joint shows a continuous steep→mid→flat "
    "gradient (its wetted fillet); the space BETWEEN pads is flat/dark — a raised slope there means a "
    "bridge. Image 2 = the VALID FILLET regions highlighted GREEN: this is where wetted solder actually "
    "is. Use them to pick the exact solder joint (for solder-deficit defects place ON one joint's "
    "green/fillet zone) or the exact pad gap (for bridge defects span BETWEEN two adjacent fillets). "
    "Relative signal, uncalibrated — weigh with the real image."
)
SOLDERLIGHT_JUDGE_CAPTION = (
    "SOLDER-LIGHT visual prompt (physics): slope heatmap + valid-fillet regions (GREEN = where wetted solder is). "
    "Verify the candidate mask against these — a solder-deficit mask should overlap ONE joint's fillet "
    "zone; a bridge mask should span the gap BETWEEN two adjacent fillets. Weigh with the real image."
)


def build_solderlight_blocks(vp_dir, clean_path, defect_type, judge=False, style="overlay", views=None) -> list:
    """Solder-light physics visual prompt: slope heatmap + fillet highlight as ready-to-inject blocks.
    Precomputed by :class:`SolderlightVP` below; ``[]`` when the cache is absent. ``style`` — 'overlay'
    (blended on the photo) or 'grid' (+ the 0-1 grid) — selects FORMAT only, never which file is read."""
    views = views or Views()
    maps = Path(vp_dir)
    h = clean_hash(clean_path)
    slope = _load_rgb(maps / f"solderlight_slope_{h}.png")
    fillet = _load_rgb(maps / f"solderlight_fillet_{h}.png")
    if slope is None or fillet is None:
        return []
    cap = SOLDERLIGHT_JUDGE_CAPTION if judge else SOLDERLIGHT_CAPTION
    if style == "grid":
        slope, fillet = views.grid(slope), views.grid(fillet)
        cap = cap + " Both maps carry the SAME 0-1 coordinate grid as the clean image."
    return [cap, "Slope heatmap (warm=flat, cold=steep):", slope, "Valid solder-fillet regions (green):", fillet]


NORMALLIGHT_CAPTION = (
    "SOLDER MAP under NORMAL (white) light — physics, no ML. This board is NOT imaged under tri-color "
    "ring light, so slope is not encoded in colour here. What is usable instead: solder is metal with a "
    "curved surface, so it returns a SMALL BRIGHT specular highlight, while solder mask, silkscreen and "
    "component bodies are matte. The maps isolate exactly that. Image 1 = the specular response (warm = "
    "strong): every bright compact blob is a candidate joint, and the space between two of them is a pad "
    "gap. Image 2 = the detected SOLDER highlighted GREEN — this is where metal actually is. Place a "
    "solder-deficit defect ON one green joint; span a bridge defect BETWEEN two adjacent green joints. "
    "It marks the specular CORE of a joint, not its full wetted extent, so a real joint is usually a "
    "little larger than its green patch. Relative and uncalibrated — weigh it with the real image."
)
NORMALLIGHT_JUDGE_CAPTION = (
    "SOLDER MAP under normal white light (physics): specular response + detected solder (GREEN = where "
    "metal is, marking the joint's bright core rather than its full extent). Verify the candidate mask "
    "against these — a solder-deficit mask should sit on ONE green joint; a bridge mask should span the "
    "gap BETWEEN two adjacent green joints. Weigh with the real image."
)


def build_normallight_blocks(vp_dir, clean_path, defect_type, judge=False, style="grid", views=None) -> list:
    """Normal-light solder maps as ready-to-inject blocks; ``[]`` when not precomputed."""
    views = views or Views()
    maps = Path(vp_dir)
    h = clean_hash(clean_path)
    spec = _load_rgb(maps / f"normallight_specular_{h}.png")
    solder = _load_rgb(maps / f"normallight_solder_{h}.png")
    if spec is None or solder is None:
        return []
    cap = NORMALLIGHT_JUDGE_CAPTION if judge else NORMALLIGHT_CAPTION
    if style == "grid":
        spec, solder = views.grid(spec), views.grid(solder)
        cap = cap + " Both maps carry the SAME 0-1 coordinate grid as the clean image."
    return [cap, "Specular response (warm = strong):", spec, "Detected solder (green):", solder]


SAM2SEG_CAPTION = (
    "AUTOMATIC SEGMENTATION of THIS clean board (SAM2 'segment everything'; class-agnostic — colors are "
    "arbitrary IDs, NOT classes). Each colored region is one visually distinct structure/part (component "
    "body, individual leads, individual solder joints, board zones) with its EXACT boundary. Use it in "
    "two steps: (1) from the EXAMPLES decide which KIND of structure hosts the defect; (2) pick the "
    "specific segment(s) of that kind here and align your mask to that segment's actual outline. Colors "
    "carry no meaning; boundaries do. It may over- or under-segment — trust the real image."
)
SAM2SEG_JUDGE_CAPTION = (
    "AUTOMATIC SEGMENTATION (SAM2, class-agnostic) of the clean board — arbitrary colors, exact "
    "boundaries. Use it to verify the candidate mask sits ON a specific structure segment (hugging its "
    "outline) rather than straddling background or spilling across unrelated segments; weigh it with "
    "the real image."
)


def build_sam2seg_blocks(vp_dir, clean_path, defect_type, judge=False, style="grid", views=None) -> list:
    """SAM2 segment-everything blocks from the ``sam2seg_<hash>.png`` :class:`Sam2VP` precomputed;
    ``[]`` when absent. ``defect_type`` only selects the caption, ``style`` only the grid."""
    views = views or Views()
    im = _load_rgb(Path(vp_dir) / f"sam2seg_{clean_hash(clean_path)}.png")
    if im is None:
        return []
    cap = SAM2SEG_JUDGE_CAPTION if judge else SAM2SEG_CAPTION
    if style == "grid":
        im = views.grid(im)
        cap = cap + " The map carries the SAME 0-1 coordinate grid as the clean image."
    return [cap, im]


class VisualPrompt(Protocol):
    """The pluggable visual-prompt contract: turn a clean image + defect type into injectable blocks."""

    name: str

    def build(self, clean_path: str, defect_type: str) -> list:
        """Image/text blocks injected into the proposer (``[]`` = no visual prompt)."""
        ...

    def build_for_judge(self, clean_path: str, defect_type: str) -> list:
        """Image/text blocks injected into the judge (may return ``[]``)."""
        ...

    def exists(self, clean_path: str) -> bool:
        """True iff the precomputed artifacts this visual prompt needs are present for ``clean_path``.

        ``build`` silently returns ``[]`` when its cache is missing, so a missing precompute degrades
        quality invisibly; callers confirm availability here. :class:`NoneVP` needs none, returns True.
        """
        ...

    def precompute(self, clean_path: str) -> None:
        """Generate + cache this visual prompt's map(s) for clean_path into vp_dir (idempotent: no-op if exists())."""
        ...

    def layers(self, clean_path: str) -> dict:
        """``{name: int label map}`` for ops' snap_layer/fill_layer/scale_layer, or ``{}``.

        ONLY this visual prompt's own layer: the model is offered exactly the layer belonging to the
        cue it is looking at, never a structure map it was never shown. ``{}`` — no layer, or a cache
        carrying none — makes the caller withdraw the layer ops rather than offer one that does nothing.
        """
        ...


class SolderlightVP:
    """Solder-light physics visual prompt — slope heatmap + valid-fillet highlight, drawn with the coordinate grid."""

    name = "solderlight"

    def __init__(self, vp_dir: str, views=None):
        self.views = views or Views()
        self.vp_dir = vp_dir

    def build(self, clean_path: str, defect_type: str) -> list:
        return build_solderlight_blocks(self.vp_dir, clean_path, defect_type, style="grid", views=self.views)

    def build_for_judge(self, clean_path: str, defect_type: str) -> list:
        return build_solderlight_blocks(
            self.vp_dir, clean_path, defect_type, judge=True, style="grid", views=self.views
        )

    def _fillet(self, clean_path: str) -> Path:
        return Path(self.vp_dir) / f"solderlight_fillet_{clean_hash(clean_path)}.png"

    def painted_map(self, path: str):
        """The precomputed fillet-highlight map for ``path``, or None."""
        return _load_rgb(self._fillet(path))

    def exists(self, clean_path: str) -> bool:
        """True iff both precomputed maps (slope + fillet) AND the fillet labels LOAD for this hash.

        The labels are part of the artifact: leaving them out of the check would leave a picture-only
        cache permanently without layer ops, where counting them re-runs a precompute that is pure
        numpy over one image. LOAD, not `is_file` — see :func:`_usable`.
        """
        h = clean_hash(clean_path)
        maps = Path(self.vp_dir)
        return (
            _usable(maps / f"solderlight_slope_{h}.png", self._fillet(clean_path))
            and load_labels(self._fillet(clean_path)) is not None
        )

    def layers(self, clean_path: str) -> dict:
        lab = load_labels(self._fillet(clean_path))
        return {self.name: lab} if lab is not None else {}

    def precompute(self, clean_path: str) -> None:
        """Compute + cache the slope + fillet maps for ``clean_path`` (no-op if :meth:`exists`).

        The cached PNGs are the RAW maps — ``build`` applies the coordinate grid at load time — so no
        grid is drawn here.
        """
        if self.exists(clean_path):
            return
        # Constants of the tri-colour ring-light capture geometry (slope weights, then the
        # slope/colour/brightness/chroma gates) — properties of the illumination rig, not of any
        # dataset. Recalibrate them if the rig's ring angles or channel colours differ.
        alpha, beta, gamma = 0.05, 0.60, 1.15  # slope weights: red (flat) / green (mid) / blue (steep)
        m_min, gb_min = 0.40, 0.55  # Th_slope_min / Th_color (G+B ratio)
        bright, chroma_min = 170.0, 55.0  # channel-sum gate (substrate isn't dark) + chroma gate

        h = clean_hash(clean_path)
        maps = Path(self.vp_dir)
        maps.mkdir(parents=True, exist_ok=True)
        img = np.array(Image.open(clean_path).convert("RGB"), dtype=np.float32)
        S = img.sum(axis=2)
        Sq = np.where(S == 0, 1, S)
        Rr, Gr, Br = img[..., 0] / Sq, img[..., 1] / Sq, img[..., 2] / Sq
        M = alpha * Rr + beta * Gr + gamma * Br
        M[S < 45] = 0.0  # spec Th_dark: no reflection -> slope 0

        heat = (colormaps["jet"](np.clip(M / gamma, 0, 1))[..., :3] * 255).astype(np.float32)
        blend = img * 0.45 + heat * 0.55
        blend[S < 45] = img[S < 45] * 0.6
        Image.fromarray(blend.astype(np.uint8)).save(maps / f"solderlight_slope_{h}.png")

        chroma = img.max(axis=2) - img.min(axis=2)
        fillet = (M >= m_min) & ((Gr + Br) > gb_min) & (S > bright) & (chroma > chroma_min)
        fl = img.copy()
        fl[fillet] = fl[fillet] * 0.35 + np.array([40, 255, 60], dtype=np.float32) * 0.65
        png = maps / f"solderlight_fillet_{h}.png"
        Image.fromarray(fl.astype(np.uint8)).save(png)
        save_labels(png, _components(fillet))  # from `fillet`, never re-derived from the blend


class Sam2VP:
    """SAM2 segment-everything visual prompt — precomputed class-agnostic segmentation map, drawn with the grid."""

    name = "sam2"

    def __init__(self, vp_dir: str, views=None):
        self.views = views or Views()
        self.vp_dir = vp_dir

    def build(self, clean_path: str, defect_type: str) -> list:
        return build_sam2seg_blocks(self.vp_dir, clean_path, defect_type, style="grid", views=self.views)

    def build_for_judge(self, clean_path: str, defect_type: str) -> list:
        return build_sam2seg_blocks(self.vp_dir, clean_path, defect_type, judge=True, style="grid", views=self.views)

    def _seg(self, clean_path: str) -> Path:
        return Path(self.vp_dir) / f"sam2seg_{clean_hash(clean_path)}.png"

    def painted_map(self, path: str):
        """The precomputed painted map for ``path`` (any image, not only cleans), or None."""
        return _load_rgb(self._seg(path))

    def exists(self, clean_path: str) -> bool:
        """True iff the precomputed segmentation map AND its labels LOAD for this clean image's hash.

        Unlike the two physics prompts, a miss here costs a GPU pass — but a picture with no labels
        regenerates anyway, they cannot be recovered from it (see :func:`save_labels`). Unreadable
        counts as a miss, see :func:`_usable`.
        """
        seg = self._seg(clean_path)
        return _usable(seg) and load_labels(seg) is not None

    def layers(self, clean_path: str) -> dict:
        lab = load_labels(self._seg(clean_path))
        return {self.name: lab} if lab is not None else {}

    def precompute(self, clean_path: str) -> None:
        """Run SAM2 'segment everything' over the 2x-upscaled board + cache the painted map as
        ``sam2seg_<hash>.png`` (no-op if :meth:`exists`)."""
        if self.exists(clean_path):
            return
        gen = sam2.segmenter()  # ONE model for the process; sam2.py owns its lifetime
        # The 2x upscale is CAPPED so the point grid + GPU memory stay bounded on a huge board.
        upscale, min_area, upscale_cap = 2.0, 0.0002, 2048

        h = clean_hash(clean_path)
        maps = Path(self.vp_dir)
        maps.mkdir(parents=True, exist_ok=True)
        pil = Image.open(clean_path).convert("RGB")
        W0, H0 = pil.size
        scale = min(upscale, upscale_cap / max(W0, H0))  # 2x, but never past upscale_cap on the long side
        img = np.array(
            pil
            if abs(scale - 1.0) < 1e-3
            else pil.resize((max(1, int(W0 * scale)), max(1, int(H0 * scale))), Image.LANCZOS)
        )
        H, W = img.shape[:2]

        try:
            masks = gen.generate(img)
            masks = [m for m in masks if m["area"] >= min_area * H * W]
            masks.sort(key=lambda m: -m["area"])  # big first so small segments paint on top

            over = img.astype(np.float32)
            lab = np.zeros((H, W), np.int32)
            for i, m in enumerate(masks):
                # golden-ratio hue rotation -> maximally distinct arbitrary colors
                r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
                col = np.array([r * 255, g * 255, b * 255], dtype=np.float32)
                seg = m["segmentation"]
                over[seg] = over[seg] * 0.55 + col * 0.45
                lab[seg] = i + 1  # later (smaller) segments win, as they do in paint
                cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(over, cnts, -1, col.tolist(), 1)

            png = maps / f"sam2seg_{h}.png"
            Image.fromarray(over.astype(np.uint8)).save(png)
            save_labels(png, lab, shape=(H0, W0))  # stored in the CLEAN's frame, not the 2x one
            # Hand back the per-image working set: it accumulates to an OOM over the hundreds of
            # cleans one process precomputes.
            del masks, over, img, lab
        finally:
            # `finally`, not the last statement of the try: a generator that RAISES is the OOM case,
            # and skipping the reset + `empty_cache()` there pins that image's working set into the next.
            sam2.release()


class NoneVP:
    """The empty baseline — injects no visual prompt on either role."""

    name = "none"

    def build(self, clean_path: str, defect_type: str) -> list:
        return []

    def build_for_judge(self, clean_path: str, defect_type: str) -> list:
        return []

    def exists(self, clean_path: str) -> bool:
        return True  # nothing to precompute

    def precompute(self, clean_path: str) -> None:
        pass  # no visual prompt -> nothing to generate

    def layers(self, clean_path: str) -> dict:
        return {}  # no cue -> no structures -> the caller withdraws the layer ops


class NormallightVP:
    """Solder under NORMAL (white) light — specular response + detected solder, drawn with the grid.

    Under white light the tri-colour ring-light ratios :class:`SolderlightVP` reads carry no slope
    information at all, so that prompt is not merely weaker here, it is meaningless. Which
    illumination a dataset was shot under is a fact about the capture rig, NOT something to infer
    from a thumbnail — solder-light imagery can look like flat per-material colour while carrying
    full slope information, so appearance-based routing misclassifies exactly the boards that
    benefit most from the slope prompt.

    What survives the change of illumination is that solder is metal with a curved surface, returning
    a small bright specular highlight where mask, silkscreen and bodies are matte. Hence a white
    top-hat — brightness minus its own LOCAL background, so a big pale region cancels where a small
    joint does not; thresholding globally instead halves the median lift over the real defect masks
    and takes the marked area from ~10% of the board past 40% — on min(R,G,B), the dichromatic
    channel (max and a saturation-weighted value measure the same within noise).

    Measured on the four genuinely white-light benchmark classes (wfm), against the REAL defect masks:
    median lift 3.6x over painting the board at random, hitting 28-66% of real defect pixels while
    marking 10-16% of the board.

    KNOWN FALSE POSITIVE, left in deliberately: laser-etched part numbers and white silkscreen pass
    every stage. The obvious thickness filter is rejected on measurement -- dropping components
    thinner than 0.15 of the element leaves the median lift where it was and takes fine-pitch IC
    bridges to ZERO, because those joints are as thin as a character stroke.
    """

    name = "normallight"

    def __init__(self, vp_dir: str, views=None):
        self.views = views or Views()
        self.vp_dir = vp_dir

    def build(self, clean_path: str, defect_type: str) -> list:
        return build_normallight_blocks(self.vp_dir, clean_path, defect_type, style="grid", views=self.views)

    def build_for_judge(self, clean_path: str, defect_type: str) -> list:
        return build_normallight_blocks(
            self.vp_dir, clean_path, defect_type, judge=True, style="grid", views=self.views
        )

    def _solder(self, clean_path: str) -> Path:
        return Path(self.vp_dir) / f"normallight_solder_{clean_hash(clean_path)}.png"

    def exists(self, clean_path: str) -> bool:
        # LOAD, not `is_file` — see :func:`_usable`.
        h = clean_hash(clean_path)
        maps = Path(self.vp_dir)
        return (
            _usable(maps / f"normallight_specular_{h}.png", self._solder(clean_path))
            and load_labels(self._solder(clean_path)) is not None
        )

    def layers(self, clean_path: str) -> dict:
        lab = load_labels(self._solder(clean_path))
        return {self.name: lab} if lab is not None else {}

    def precompute(self, clean_path: str, joint_frac: float = 0.045) -> None:
        """Compute + cache the specular + solder maps (no-op if :meth:`exists`).

        ``joint_frac`` is the largest a joint may be, as a fraction of the short side; it sizes the
        top-hat element, and anything wider is background. A sweep from 0.015 to 0.045 moves lift by
        less than 1x, so this is a scale declaration, not a tuning knob.
        """
        if self.exists(clean_path):
            return
        maps = Path(self.vp_dir)
        maps.mkdir(parents=True, exist_ok=True)
        h = clean_hash(clean_path)
        img = np.array(Image.open(clean_path).convert("RGB"))
        base = img.astype(np.float32).min(axis=2)  # dichromatic: the specular floor across channels
        r = max(3, int(round(joint_frac * min(img.shape[:2]))) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        top = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_TOPHAT, k).astype(np.float32)
        norm = np.clip(top / max(top.max(), 1e-6), 0, 1)

        heat = (colormaps["jet"](norm)[..., :3] * 255).astype(np.float32)
        blend = img.astype(np.float32) * 0.45 + heat * 0.55
        Image.fromarray(blend.astype(np.uint8)).save(maps / f"normallight_specular_{h}.png")

        # Otsu, not a fixed cut: what counts as bright depends on this board's exposure.
        _, m = cv2.threshold((norm * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lab = _components(m > 0)  # close + area floor, the same helper the fillet labels use
        keep = lab > 0
        sol = img.astype(np.float32).copy()
        sol[keep] = sol[keep] * 0.35 + np.array([40, 255, 60], np.float32) * 0.65
        png = maps / f"normallight_solder_{h}.png"
        Image.fromarray(sol.astype(np.uint8)).save(png)
        save_labels(png, lab)  # `lab` as it is, never re-derived from the green blend


# One caption for BOTH renderers (context turns and the judge prompt), so the two never drift.
EXEMPLAR_VP_CAPTION = (
    "STRUCTURE VIEW — the same kind of painted map the candidate side is shown, computed on this "
    "example's own image, with the example's REAL mask outlined in white/black: read WHERE the real "
    "annotation sits on the painted features and HOW FULLY it covers them, and hold the candidate to "
    "the same convention on ITS map."
)


def annotate_exemplar(vp, ex, *, max_side: int = 640):
    """The exemplar re-rendered in the CANDIDATE's modality: ``vp``'s painted map of the exemplar's
    defect image with its REAL mask outlined — the words of a card or caption can be read against the
    wrong feature (an 'end termination' has both a metal cap and a solder fillet; only the picture
    says which one the annotations live on). None whenever it cannot be built — absence changes
    nothing. The map precompute is content-hash cached, so repeat exemplars are free."""
    try:
        if not getattr(ex, "defect_path", None) or not hasattr(vp, "painted_map"):
            return None
        vp.precompute(ex.defect_path)  # no-op on a cache hit; one GPU/CV pass per UNIQUE image
        pm = vp.painted_map(ex.defect_path)
        if pm is None:
            return None
        mask_src = Image.open(ex.mask_path).convert("L") if getattr(ex, "mask_path", None) else ex.mask
        m = np.asarray(resize_mask(mask_src, pm.size)) > 127  # the same resample the exemplar's own mask gets
        over = np.array(pm)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t = max(2, round(max(pm.size) / 400))
        cv2.drawContours(over, cnts, -1, (0, 0, 0), t + 2)
        cv2.drawContours(over, cnts, -1, (255, 255, 255), t)
        out = Image.fromarray(over)
        out.thumbnail((max_side, max_side))
        return out
    except Exception as e:  # noqa: BLE001 — the parity view is an extra; a failure must not cost the run
        log.warning(f"degraded[exemplar_vp_failed]: {getattr(ex, 'defect_path', '?')}: {type(e).__name__}: {e}")
        return None


def by_name(name: str, *, vp_dir: str | None = None, views=None) -> VisualPrompt:
    """Construct a visual prompt by name; an unknown name is :class:`NoneVP`, the empty baseline."""
    if name == "solderlight":
        if vp_dir is None:
            raise ValueError("SolderlightVP requires vp_dir")
        return SolderlightVP(vp_dir, views)
    if name == "normallight":
        if vp_dir is None:
            raise ValueError("NormallightVP requires vp_dir")
        return NormallightVP(vp_dir, views)
    if name == "sam2":
        if vp_dir is None:
            raise ValueError("Sam2VP requires vp_dir")
        return Sam2VP(vp_dir, views)
    return NoneVP()
