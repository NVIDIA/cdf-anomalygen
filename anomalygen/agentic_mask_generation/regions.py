# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WHERE on this board the defect may go — and how those sites are described to the model.

Enumerate the valid sites on a clean (a VLM call, cached per request), widen an assigned site when the
defect's own examples say it needs more room, write the site list into the words the proposer reads,
and map coordinates back out of a region crop into the full board.

Enumeration is inference-legal by construction: it reads the clean board and this type's EXAMPLE
defect images — never a ground-truth mask for the image being placed on. Answers are cached under
`clean content . defect type . model . exemplar set . ask` — each model enumerates its own sites.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from anomalygen.agentic_mask_generation.card_distill import ask_stamp
from anomalygen.agentic_mask_generation.imaging import (
    Views,
    overlay_mask,
)
from anomalygen.agentic_mask_generation.judging import component_count, nn_spacing
from anomalygen.agentic_mask_generation.ops import FIELD_SCALE
from anomalygen.agentic_mask_generation.utils import atomic_write, clean_hash, log, slug
from anomalygen.agentic_mask_generation.vlm import (
    MAX_TOKENS_REGIONS,
    SchemaInvalid,
    validate_reply,
    watch_reply_repair,
)


def ex_dist_summary(exemplars) -> str:
    """Measured distribution of the EXAMPLE masks (count + clustering spacing) so PROPOSE targets the
    right grouping/tightness."""
    ns = [component_count(e.mask) for e in exemplars]
    sp = [nn_spacing(e.mask) for e in exemplars]
    return (
        f"MEASURED example distribution: instance counts {ns}; inter-instance spacing (median "
        f"nearest-neighbour, smaller=tighter clusters) {sp}. If multi-instance, REPRODUCE this "
        "grouping CHARACTER — same count order, and instances clustered/spaced like the examples "
        "(a few evenly-spread dots is NOT the same as a tight group). Read it; don't assume."
    )


def _widen_to_examples(regs, exemplars, headroom: float = 1.25):
    """Grow each assigned region until it could hold a whole example of this defect.

    A region is one SITE, and a sites-spanning defect covers several, so the assigned region is too small
    for the mask by construction — on WFM bridge the regions have a median long edge of 0.125 of the
    image against real annotations spanning 0.420. Widened rather than dropped: the crop is what tells
    the model how big the defect is relative to its host.

    The target comes from the EXAMPLE masks this run was given, so it is this type's own measured
    extent rather than a constant; `headroom` leaves the defect somewhere to sit inside the crop.
    Extents are paired as (long, short) — sorted, not x/y — so the target is rotation-safe: an
    exemplar photographed landscape still says how wide a portrait region's short axis must be.
    """
    exts = []
    for ex in exemplars or ():
        try:
            a = np.asarray(ex.mask.convert("L")) > 127
        except Exception as e:
            log.warning(f"degraded[exemplar_unreadable]: {type(e).__name__}: {e}")
            continue
        if not a.any():
            continue
        eh, ew = a.shape
        ys, xs = np.nonzero(a)
        exts.append(sorted(((xs.max() - xs.min() + 1) / ew, (ys.max() - ys.min() + 1) / eh), reverse=True))
    if not exts:
        return regs  # nothing measured -> leave the assignment alone
    want_long = min(1.0, float(np.median([lg for lg, _ in exts])) * headroom)
    want_short = min(1.0, float(np.median([sh for _, sh in exts])) * headroom)
    out = []
    for r in regs:
        x0, y0, x1, y1 = r.get("x0", 0.0), r.get("y0", 0.0), r.get("x1", 1.0), r.get("y1", 1.0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        # each axis grows toward ITS OWN target — the region's long axis toward the examples' long
        # extent, its short toward their short — and never shrinks; a wide-thin pad box gets its
        # short axis widened to component scale instead of inheriting the long axis's factor of 1
        wx, wy = (want_long, want_short) if w >= h else (want_short, want_long)
        fx, fy = max(1.0, wx / max(w, 1e-6)), max(1.0, wy / max(h, 1e-6))
        hw, hh = w * fx / 2, h * fy / 2
        # clip by SHIFTING rather than truncating: a region near an edge keeps its size instead of
        # silently coming back smaller than every other one on the board
        hw, hh = min(hw, 0.5), min(hh, 0.5)
        cx, cy = min(max(cx, hw), 1 - hw), min(max(cy, hh), 1 - hh)
        out.append({**r, "x0": cx - hw, "y0": cy - hh, "x1": cx + hw, "y1": cy + hh})
    return out


def site_menu(regions, covered, W, H):
    """The enumerated valid sites, with the ones already used marked, so the AGENT picks where to place.

    The system decides HOW MANY defects a site may hold; the agent decides WHICH site this one goes
    on. Reached only by `coverage_feedback="choose"`, which no test drives — so nothing pins this.
    """
    if not regions:
        return []
    taken = set()
    for m in covered or []:
        a = np.asarray(m.convert("L")) if hasattr(m, "convert") else np.asarray(m)
        ys, xs = np.nonzero(a > 127)
        if not len(xs):
            continue
        # By AREA OVERLAP, not by centroid: a defect that spans two sites — which is what a bridge IS —
        # has its centroid in the gap between their boxes and would mark neither as used.
        hit = []
        for i, r in enumerate(regions):
            inside = (
                ((xs / W) >= r["x0"]) & ((xs / W) <= r["x1"]) & ((ys / H) >= r["y0"]) & ((ys / H) <= r["y1"])
            ).sum()
            if inside:
                hit.append((inside, i))
        if hit:
            taken.update(i for n, i in hit if n >= 0.2 * max(h for h, _ in hit))
        else:  # nowhere near an enumerated site — attribute it to the nearest one
            cx, cy = xs.mean() / W, ys.mean() / H
            taken.add(
                min(
                    range(len(regions)),
                    key=lambda i: (
                        (max(regions[i]["x0"] - cx, 0, cx - regions[i]["x1"])) ** 2
                        + (max(regions[i]["y0"] - cy, 0, cy - regions[i]["y1"])) ** 2
                    ),
                )
            )
    free = [i for i in range(len(regions)) if i not in taken]
    lst = "; ".join(
        f"{i}:[{r['x0']:.2f},{r['y0']:.2f},{r['x1']:.2f},{r['y1']:.2f}]{' TAKEN' if i in taken else ''}"
        for i, r in enumerate(regions)
    )
    return [
        f"VALID SITES on this board ({len(regions)} enumerated, normalized bboxes): {lst}\n"
        f"CHOOSE ONE site for this defect and place EVERY candidate on it. "
        + (
            f"Sites {free} have no defect yet — prefer one of those; "
            if free
            else "Every site already holds a defect; pick the least crowded one; "
        )
        + "a site that is wrong for this defect is worse than a repeat, so never pick an invalid one."
    ]


def coverage_blocks(clean, covered, mode, views: Views):
    """Blocks telling the agent which sites on THIS clean already carry a defect, so it targets an
    UNCOVERED valid site (spreads the run-to-run distribution). Empty when nothing is covered yet.
    ``covered`` = list of prior-accepted binary masks on this clean; ``mode`` = 'text' | 'visual'.

    ``views`` carries the delivery cap and is REQUIRED, never defaulted — a native run says so with
    ``Views(0)``. omitting the delivery cap raises instead of returning a native view."""
    if not covered:
        return []
    W, H = clean.size
    bbs = []
    for m in covered:
        arr = np.asarray(m.convert("L")) if hasattr(m, "convert") else np.asarray(m)
        a = arr > 127
        ys, xs = np.nonzero(a)
        if len(xs):
            bbs.append((xs.min() / W, ys.min() / H, xs.max() / W, ys.max() / H))
    if not bbs:
        return []
    instr = (
        f"COVERAGE — {len(bbs)} defect(s) are ALREADY placed on THIS board. To build a diverse "
        "dataset, target a DIFFERENT valid site that has no defect yet; do NOT reuse a covered spot. "
        "Only ever choose a genuinely valid location for this defect (a wrong site is worse than a repeat)."
    )
    if mode == "visual":
        ov = clean.copy()
        for m in covered:
            ov = overlay_mask(ov, m)
        return [
            "Defects already placed on this board (red = TAKEN; put the new one on an untaken valid site):",
            views.grid(ov),
            instr,
        ]
    lst = "; ".join(f"[{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}]" for x0, y0, x1, y1 in bbs)
    return [
        f"Defects already placed on this board — normalized bboxes to AVOID (put the new one elsewhere): {lst}",
        instr,
    ]


# --------------------------------------------------------------------------------------------------
# Enumeration: the ask, and what an answer to it is keyed by.
#
# The ask is named rather than written inline because the cache key is a FINGERPRINT of it
# (`ask_stamp`): what the model was asked is what decides what it answered, so a reworded question has
# to re-enumerate instead of being served the answer to the old one.
# --------------------------------------------------------------------------------------------------
REGION_SYS = (
    "You are a precise vision annotator. Report only regions that are visible AND "
    "valid sites for this specific defect type."
)
REGION_EXAMPLE = "EXAMPLE %d — a defect of this type occupies ONE such local region:"
REGION_CLEAN = "CLEAN image (coordinate grid, normalized 0-1):"
REGION_ASK = (
    "List the DISTINCT local regions where a '%s' defect could VALIDLY sit on this image, "
    "one tight normalized bbox each. A 'region' is a repeated valid site of THIS defect (per the "
    "examples: e.g. each solder joint, each component, each trace segment). If the defect is whole-"
    "component or global — only ONE valid area, not a repeated site — return just that ONE region. If "
    "there is no discrete valid local site, return an empty list. Do NOT invent sites the defect can't "
    'occur on. Output JSON {"regions":[{"x0":..,"y0":..,"x1":..,"y1":..}, ...]}.'
)


def _exemplar_fp(exemplars) -> str:
    """Fingerprint of the exemplar images this call SENDS — all of them.

    The key must be a function of the whole request, and the prompt carries the exemplar defect
    images, which vary with `--num-examples` and with the clean they were DINO-ranked against.

    Hashed from PIXELS, not from a path: `Exemplar` carries PIL images with no source, so what was
    SENT is the only identity available. NOT covered: `image_max_dim` also changes the bytes that
    leave the process (`_img_block` downscales after this), but it is a recorded `Config` field, so a
    wrong answer stays attributable.
    """
    h = hashlib.md5(usedforsecurity=False)
    for ex in exemplars:
        im = ex.defect
        h.update(f"{im.mode}{im.size}".encode())
        h.update(im.tobytes())
    return h.hexdigest()[:8]


def _boxes_with_area(regs, source: str) -> list:
    """The boxes that enclose something, naming the ones that do not.

    The four keys are PRESENT, NUMERIC and in [0,1] by the "regions" schema — `p["x1"] > p["x0"]` on
    STRING coordinates is a legal comparison that returns True. What is left is the ORDER, which a
    static JSON schema cannot express: an inverted box crops to nothing, so it is dropped and named.

    ONE definition, run on the fresh reply AND on a cache entry read back, since both reach
    `_widen_to_examples` and `Views.zoom_to_box`, neither of which re-checks. ``source`` is the board
    for a fresh answer, the file for a stored one.
    a stored region faces the check a fresh one faces.
    """
    out = []
    for p in regs:
        if p["x1"] > p["x0"] and p["y1"] > p["y0"]:
            out.append(p)
        else:
            log.warning(f"degraded[region_inverted]: {source}: {p} has no area")
    return out


def _repair_frame(regs: list, size, source: str) -> tuple[list, bool]:
    """Map a reply back to normalized [0,1] and say whether it needed repair.

    One frame per REPLY — a reply is authored in one coordinate system, so the max coordinate
    decides. <=1: already normalized. <=2: normalized with overshoot (a box running past the edge);
    clamping is the repair, rescaling would collapse the reply. <=1000: the 0-1000 integer
    convention many VLMs ground in; divide by 1000. Beyond: pixels of the image that was sent.
    Repair branches clamp into [0,1]; inverted/degenerate boxes still die in _boxes_with_area.

    Returns ``(regs, repaired)`` — the caller must not cache a repaired reply, so a frame read
    wrong once is wrong for one run, not forever.
    """
    vals = [
        float(p[k])
        for p in regs
        for k in ("x0", "y0", "x1", "y1")
        if isinstance(p, dict) and isinstance(p.get(k), (int, float))
    ]
    m = max(vals, default=0.0)
    if m <= 1.0:
        return regs, False
    W, H = size
    # The band the max coordinate falls in decides the frame: (upper bound, (sx, sy), label). The last
    # row has no upper bound, so a max beyond 1000 — or a NaN, which compares false against every
    # bound — is read as pixels.
    bands = (
        (2.0, (1.0, 1.0), "overshoot clamped"),
        (1000.0, (1000.0, 1000.0), "0-1000 convention"),
        (math.inf, (float(W), float(H)), "pixel frame"),
    )
    for bound, (sx, sy), frame in bands:
        if m <= bound:
            break
    out = []
    for p in regs:
        if not (isinstance(p, dict) and all(isinstance(p.get(k), (int, float)) for k in ("x0", "y0", "x1", "y1"))):
            out.append(p)
            continue
        out.append(
            {
                **p,
                "x0": min(max(p["x0"] / sx, 0.0), 1.0),
                "x1": min(max(p["x1"] / sx, 0.0), 1.0),
                "y0": min(max(p["y0"] / sy, 0.0), 1.0),
                "y1": min(max(p["y1"] / sy, 0.0), 1.0),
            }
        )
    log.warning(f"degraded[regions_frame_repaired]: {source}: reply max coord {m:g} -> {frame}")
    return out, True


def enumerate_regions(vlm, clean_path, defect_type, exemplars, model, cache_dir=None):
    """Inference-legal (NO GT) one-shot enumeration of the DISTINCT valid local regions where a defect of
    this type could sit, as normalized bboxes. A region is whatever repeated valid site THIS defect type
    has — the notion is defect-agnostic, never tied to one structure class. A multi-site defect
    (insufficient-solder -> each solder joint, scratch -> each surface) returns MANY; a whole-component /
    single-site defect returns ONE; a global / non-localizable defect returns NONE. The count is what lets
    --coverage roundrobin decide whether it can split across sites at all.

    With ``cache_dir`` the answer is kept on disk under

        ``<clean content>.<defect type>.<model>.<exemplar set>.<ask>.json``

    — every input that can change the answer, in a name a human can read: the clean's CONTENT hash,
    the type, WHO WAS ASKED, a fingerprint of the exemplars shown (`_exemplar_fp`) and one of the
    question itself (`ask_stamp`). The fingerprints are in the FILENAME, not inside the file, so two
    arms' answers COEXIST rather than evicting each other.
    each model enumerates its own sites.
    """
    views = Views.from_vlm(vlm)
    ck = None
    if cache_dir:
        key = ".".join(
            (
                clean_hash(clean_path),
                slug(defect_type),
                slug(str(model)),
                _exemplar_fp(exemplars),
                ask_stamp(REGION_SYS, REGION_EXAMPLE, REGION_CLEAN, REGION_ASK),
            )
        )
        ck = Path(cache_dir) / f"{key}.json"
        if ck.is_file():
            try:
                got = json.loads(ck.read_text())
                # Validated on READ, not only on write: an entry written by another build is still on
                # disk and still keyed by content, so a bad one would replay a string coordinate into
                # `_widen_to_examples` — a call site with no `try`.
                # a stored region faces the check a fresh one faces.
                if isinstance(got, list):
                    validate_reply({"regions": got}, "regions")
                    kept = _boxes_with_area(got, ck.name)
                    # Raised as a schema failure because that is what it is, and because the handler
                    # below already knows what to do with one.
                    if len(kept) != len(got):
                        raise SchemaInvalid(
                            "regions", f"{len(got) - len(kept)} of {len(got)} cached boxes have no area"
                        )
                    return kept
            except (json.JSONDecodeError, OSError) as e:
                # SAME kind as the schema rejection below: both end with the entry discarded and this
                # board re-enumerated at the cost of a VLM call, which is the thing the record is for.
                # What differs is the reason, and the reason is the detail's job.
                log.warning(f"degraded[region_cache_invalid]: {ck.name}: unreadable ({type(e).__name__}: {e})")
            except SchemaInvalid as e:
                log.warning(f"degraded[region_cache_invalid]: {ck.name}: {e.detail}")
                # DELETE it. The write below is guarded by `if out`, so a re-enumeration that answers
                # nothing would leave the bad file in place and repeat this read-reject-re-enumerate
                # cycle — one VLM call per run, forever, with nothing but a log line to show it.
                try:
                    ck.unlink()
                except OSError:
                    pass
    try:
        clean = Image.open(clean_path).convert("RGB")
    except Exception as e:
        # [] here is indistinguishable from "this clean has no valid site": the beam places freely,
        # with no crop and no round-robin, and the run looks ordinary.
        log.warning(f"degraded[regions_unavailable]: {Path(clean_path).name}: unreadable clean ({e})")
        return []
    blocks = []
    # EVERY exemplar handed in: the caller already applied `--num-examples`, and slicing again here
    # would override the control plane.
    for i, ex in enumerate(exemplars, 1):
        blocks += [REGION_EXAMPLE % i, ex.defect]
    blocks += [REGION_CLEAN, views.grid(clean), REGION_ASK % defect_type]
    try:
        with watch_reply_repair() as parse:
            r = vlm.call_json(REGION_SYS, blocks, model=model, max_tokens=MAX_TOKENS_REGIONS, schema="regions")
        regs = r.get("regions") if isinstance(r.get("regions"), list) else []
        regs, repaired = _repair_frame(regs, clean.size, Path(clean_path).name)
        out = _boxes_with_area(regs, Path(clean_path).name)
        # Never cache an empty answer (usually a failed call), a frame-repaired one (a frame guess),
        # nor a salvaged one (a list cut short by max_tokens, whole-looking).
        if ck is not None and out and not (repaired or parse.repaired):
            # Its OWN except: SAVING the answer is not part of producing it, and under the outer
            # handler an unwritable cache dir would turn a call that ANSWERED into [] on every clean.
            try:
                atomic_write(ck, lambda t: t.write_text(json.dumps(out, indent=1)))
            except OSError as e:
                log.warning(
                    f"degraded[region_cache_unwritable]: {ck}: {type(e).__name__}: {e} — "
                    "this board's sites are re-enumerated on every run"
                )
        if not out:
            # ITS OWN degraded KIND, not the failure's: the prompt above SAYS an empty list is a
            # valid answer, so pooling it with real failures would read as a failure per clean on a
            # dataset where "no discrete site" is right.
            log.warning(f"degraded[regions_empty]: {Path(clean_path).name}: the model returned no regions")
        return out
    except Exception as e:
        log.warning(f"degraded[regions_unavailable]: {Path(clean_path).name}: {type(e).__name__}: {e}")
        return []


# --------------------------------------------------------------------------------------------------
# The engine.
# --------------------------------------------------------------------------------------------------
def _is_xy(p) -> bool:
    """Whether ``p`` is a coordinate pair this module can map: two REAL numbers.

    A coordinate the model got wrong must not raise here: a raise costs the whole propose reply, a
    field left as authored costs one op. `bool` is an `int` and no coordinate, and a non-finite one
    maps to a point that draws nothing while looking mapped — the op layer refuses both BY NAME.
    every coordinate field survives a value that is not a number.
    """
    return (
        isinstance(p, (list, tuple))
        and len(p) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p)
    )


def _remap_primitives(specs: list, box, W: int = 0, H: int = 0) -> list:
    """Map primitives authored in the assigned region's CROP frame into FULL-image coordinates.

    ``box`` = (x0,y0,x1,y1), the region in normalized full-image coords. Points map affinely into the
    box. Each size field must follow the dimension the RENDERER scales it by, or the shape comes out
    distorted: ``rx``/``ry`` scale by image WIDTH/HEIGHT and so take the box's own x/y spans, while
    ``radius``/``inner``/``width``/``scale`` scale by ``min(W, H)`` and so take the ratio of the CROP's
    short side to the image's — hence ``W``/``H``.
    a size authored in a crop is a fraction of the CROP on every field."""
    x0, y0, x1, y1 = [float(v) for v in box]
    sx, sy = (x1 - x0), (y1 - y0)
    # crop short side / image short side — the factor the shape ops effectively apply to
    # radius-like fields, since `_scale` measures them against `min(w, h)`
    ss = (min(sx * W, sy * H) / min(W, H)) if (W and H) else min(sx, sy)

    def _pt(p):
        return [x0 + float(p[0]) * sx, y0 + float(p[1]) * sy]

    def _pt_px(p):
        # translation only — see FIELD_SCALE's note: the crop and the board share a resolution, so a
        # crop pixel differs from a board pixel by the origin alone
        return [float(p[0]) + x0 * W, float(p[1]) + y0 * H]

    out = []
    for spec in specs:
        mapped = []
        for p in spec:
            if not isinstance(p, dict):
                continue
            q = dict(p)
            # Driven by `ops.FIELD_SCALE`, declared by the module that SCALES these fields —
            # an edit after a draw scales with the crop like the draw did.
            factor = {"width": sx, "height": sy, "short": ss}
            for k, how in FIELD_SCALE.items():
                v = q.get(k)
                # A coordinate this cannot map is LEFT AS AUTHORED: the op layer parses the same
                # field one step later and refuses it BY NAME, which is the correction worth reaching.
                if how == "point":
                    if _is_xy(v):
                        q[k] = _pt(v)
                    elif k in q:
                        log.warning(
                            f"degraded[remap_coord_unusable]: {p.get('kind')}.{k}={v!r} is not "
                            "an [x, y] pair of numbers — left in the crop frame, for the op "
                            "layer to refuse"
                        )
                elif how in ("points", "points_px"):
                    into_frame = _pt if how == "points" else _pt_px
                    if isinstance(v, list):
                        keep = [pp for pp in v if _is_xy(pp)]
                        if len(keep) != len(v):
                            # DROPPED, not left: the rest of this shape's vertices ARE mapped, so a
                            # survivor in the crop frame would draw a shape that exists in neither.
                            log.warning(
                                f"degraded[remap_coord_unusable]: {p.get('kind')}.{k}: "
                                f"{len(v) - len(keep)} of {len(v)} points are not [x, y] pairs "
                                "of numbers — dropped"
                            )
                        q[k] = [into_frame(pp) for pp in keep]
                    elif k in q:
                        log.warning(
                            f"degraded[remap_coord_unusable]: {p.get('kind')}.{k}={v!r} is not "
                            "a list of points — left in the crop frame, for the op layer to "
                            "refuse"
                        )
                elif isinstance(v, (int, float, str)) and not isinstance(v, bool):
                    # Read as the op layer reads it — `float()` — so "0.2" scales with the crop
                    # exactly as 0.2 does instead of reaching the board unscaled.
                    try:
                        q[k] = float(v) * factor[how]
                    except ValueError:
                        log.warning(
                            f"degraded[remap_coord_unusable]: {p.get('kind')}.{k}={v!r} is not a "
                            "number — left in the crop frame, for the op layer to refuse"
                        )
            mapped.append(q)
        out.append(mapped)
    return out


@dataclass(frozen=True)
class Siting:
    """The sites this beam was assigned, their crops, and WHOSE coordinate frame the model authors in.

    The invariant: **the canvas is only really a crop once the crop exists** — otherwise the proposer
    authors crop coordinates that nothing maps back. `build` is the only way to make one, so `hard`
    cannot disagree with `crops`.
    a hard region run that could not crop books the constraint it dropped.
    """

    regions: tuple = ()
    crops: tuple = ()  # (slot index, gridded crop, its box in full-image normalized coords)
    hard: bool = False

    @classmethod
    def build(cls, clean, target_region, exemplars, *, want_hard: bool, views: Views) -> "Siting":
        """Widen each region to the examples' scale, crop each site, and decide whether the crop IS the canvas.

        ``views`` carries the delivery cap so a crop is gridded at the size it is SENT at, and is
        REQUIRED — a native run says ``Views(0)``.

        `target_region` is one region for the beam or one per candidate slot; a slot whose crop cannot
        be built is dropped, falling back to the full frame rather than taking the beam down.
        """
        regs = (
            list(target_region)
            if isinstance(target_region, (list, tuple))
            else ([target_region] if target_region is not None else [])
        )
        # Every canvas is restored to the exemplars' measured scale; never-shrink makes a
        # correctly-sized enumeration a no-op, so no card flag reaches this decision.
        if regs:
            regs = _widen_to_examples(regs, exemplars)
        hard = bool(want_hard and regs)
        crops = []
        for i, wr in enumerate(regs):
            try:
                z, b = views.zoom_to_box(clean, (wr.get("x0"), wr.get("y0"), wr.get("x1"), wr.get("y1")))
            except Exception as e:  # noqa: BLE001 — this slot loses its crop, the beam does not stop
                log.warning(f"degraded[region_crop_failed]: {type(e).__name__}: {e}")
                continue
            if z is None:
                continue
            # Hard-region grids the crop in ITS OWN [0,1] frame (that is the frame the proposer authors
            # in, and to_board maps back); otherwise the grid carries full-image coords.
            crops.append((i, views.grid(z) if hard else views.grid_fullcoords(z, b), b))
        if hard and not crops:
            # `hard`, not `want_hard`: `want_hard` is True for every beam, so keying the line on it
            # would log on every unassigned beam — every beam that takes the localize path.
            # a hard region run that could not crop books the constraint it dropped.
            log.warning(
                "degraded[hard_region_dropped]: no crop could be built for any assigned region "
                "— this beam authors full-image coordinates, so nothing confines its mask to "
                "the site it was given"
            )
        return cls(regions=tuple(regs), crops=tuple(crops), hard=bool(hard and crops))

    def primary(self):
        """The first crop, which is what a single-canvas prompt shows. (None, None) when there is none."""
        return (self.crops[0][1], self.crops[0][2]) if self.crops else (None, None)

    def frame_for(self, candidate: int):
        """The [0,1] frame candidate `candidate` authored in — crops taken ROUND-ROBIN.

        More candidates are asked for than there are crops, so the surplus WRAPS rather than piling
        onto the last one; `canvas_label` and `assignment_text` state the same rule to the model.
        Indexing is POSITIONAL over the crops that EXIST, so a dropped slot cannot make the label
        shown and the frame mapped through disagree. each candidate slot gets its own region.
        """
        boxes = [b for _i, _z, b in self.crops]
        return boxes[candidate % len(boxes)]

    def canvas_label(self, position: int) -> str:
        """How a crop is named to the model. Positional, matching `frame_for`."""
        return f"CROP {position}"

    def assignment_text(self, n_candidates: int) -> str:
        """The candidate -> crop assignment, written out rather than left to be inferred."""
        by_crop: dict[int, list[int]] = {}
        for k in range(max(0, n_candidates)):
            by_crop.setdefault(k % len(self.crops), []).append(k)
        return "; ".join(
            f"candidate(s) {', '.join(str(k) for k in ks)} -> {self.canvas_label(c)}"
            for c, ks in sorted(by_crop.items())
        )

    def to_board(self, spec: dict, candidate: int, W: int, H: int) -> dict:
        """Map one candidate's crop-frame primitives back onto the full board."""
        return _remap_primitives([spec], self.frame_for(candidate), W, H)[0]
