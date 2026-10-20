# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge placement: the one THT class that is fitted rather than confined.

WHY IT IS NOT A RING CLASS
A solder bridge is solder that has run from the joint to something outside the crop, so it must span
the pad ring and reach the crop edge. Confining it to a region would guarantee the opposite: a mask
clipped to B+C can never touch the edge. So there is no ROI clip here and AMP is not involved at all
-- the base mask is translated until it satisfies both conditions, and nothing else about it changes.

WHAT "SATISFIES" MEANS
``bridge_ok`` demands that ONE connected component reaches BOTH ends with real width. A blob resting
on the pad plus a separate blob at the border is two defects that happen to share a mask, not a
bridge. Both ends carry a threshold, each a fraction with an absolute floor, so neither collapses on
a small joint nor is satisfied by a graze on a large one.

WHICH BORDER IT GROWS TOWARDS
Every edge the annotation ALREADY reached, not merely the closest. A bridge cut from a corner touches
two, and repairing only one turns it into a side bridge with an unexplained gap along the other.

WHERE THE VARIETY COMES FROM
``fit_bridge`` takes the SMALLEST shift that works, so for a given (base mask, joint) there is exactly
ONE answer. On its own that makes the class deterministic and ``n_seeds`` inert. The elastic warp
below is what makes a seed mean something here: it perturbs the fitted mask while holding the three
properties that would otherwise turn a variation into a different defect. Turn it off and the
placement is reproducible but every seed returns the same mask.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.auto_mask_placement.cad2roi.parser import ROICandidate
from anomalygen.auto_mask_placement.tht2roi.regions import joints_carrying, roi_for
from anomalygen.auto_mask_placement.tht2roi.spec import BRIDGE, roi_rule

# Required overlap with the pad ring, as a fraction of the ring's area, with an absolute floor in
# pixels. Both are needed: the fraction alone collapses to nothing on a small pad, and the floor
# alone would let a large pad be grazed. A bridge has to GROW from the joint, and an earlier build
# without this produced masks whose pad contact was a single pixel.
MIN_B_FRAC = 0.30
MIN_B_FLOOR = 30

# The same idea on the other end of the bridge. Without it "reaches the edge" means a single pixel,
# which is not a bridge running out of frame but an annotation grazing the border. Kept deliberately
# low -- extend_to_edge produces contacts an order of magnitude wider, so this only rejects grazes.
MIN_EDGE_FRAC = 0.05
MIN_EDGE_FLOOR = 3

EDGE_BAND = 3  # rows/columns of the frontier profile extruded to the edge


def _edge_mask(shape) -> np.ndarray:
    edge = np.zeros(shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return edge


def _edge_gaps(mask: np.ndarray) -> dict:
    """Distance from the mask to each crop edge, in pixels; 0 means it already touches."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return {"left": int(xs.min()), "right": int(w - 1 - xs.max()), "up": int(ys.min()), "down": int(h - 1 - ys.max())}


def _target_edges(mask: np.ndarray) -> tuple:
    """Every edge the annotation ALREADY reaches; the nearest one if it reaches none.

    A bridge annotation cut from a corner touches two edges, and extending only the nearest one
    silently turns it into a side bridge: the shift breaks the second contact and nothing repairs it.
    Measured on the sample corpus, every fitted row lost its bottom contact that way and kept a 1-4 px
    gap. Restoring only edges the annotation already had keeps its character without inventing a
    contact it never showed.
    """
    gaps = _edge_gaps(mask)
    touching = tuple(d for d, gap in gaps.items() if gap == 0)
    return touching or (min(gaps, key=gaps.get),)


def bridge_ok(mask: np.ndarray, B: np.ndarray, min_b: int, min_edge: int = 1) -> bool:
    """True when a SINGLE component covers ``min_b`` of the pad ring and ``min_edge`` of the border.

    The single-component requirement is the substance. Testing "touches B" and "touches the edge"
    separately passes a mask made of two unconnected blobs, which is not a bridge and which the
    generator would render as two independent defects.

    ``min_edge`` defaults to 1 -- bare contact -- so the predicate on its own means exactly what it
    says. Callers that are shipping a mask pass a real width; see ``THTBridgeMaskPlacer.min_edge``.
    """
    m = mask > 127
    if not m.any():
        return False
    edge = _edge_mask(m.shape)
    n_labels, labels = cv2.connectedComponents(m.astype(np.uint8), 8)
    for k in range(1, n_labels):
        component = labels == k
        if int((component & B).sum()) >= min_b and int((component & edge).sum()) >= min_edge:
            return True
    return False


def extend_to_edge(mask: np.ndarray, direction: str, band: int = EDGE_BAND) -> np.ndarray:
    """Extrude the mask's leading profile to the crop edge.

    Extrudes the profile of the last ``band`` rows/columns rather than the single frontier line, so
    the extension keeps the width the mask actually has there instead of narrowing to whatever the
    outermost pixel happened to be.
    """
    out = mask.copy()
    ys, xs = np.nonzero(out)
    if len(ys) == 0:
        return out
    if direction in ("right", "left"):
        frontier = xs.max() if direction == "right" else xs.min()
        strip = (
            out[:, max(frontier - band + 1, 0) : frontier + 1]
            if direction == "right"
            else out[:, frontier : frontier + band]
        )
        rows = strip.any(axis=1)
        if direction == "right":
            out[rows, frontier:] = 255
        else:
            out[rows, : frontier + 1] = 255
    else:
        frontier = ys.max() if direction == "down" else ys.min()
        strip = (
            out[max(frontier - band + 1, 0) : frontier + 1, :]
            if direction == "down"
            else out[frontier : frontier + band, :]
        )
        cols = strip.any(axis=0)
        if direction == "down":
            out[frontier:, cols] = 255
        else:
            out[: frontier + 1, cols] = 255
    return out


def _ring_offsets(radius: int):
    """Offsets on the Chebyshev ring of the given radius, in row-major order.

    Row-major matters: ``fit_bridge`` returns the FIRST offset that works, so the order decides which
    of several equally-small shifts is chosen, and with it whether a rebuild reproduces byte for byte.
    """
    if radius == 0:
        yield 0, 0
        return
    for dy in range(-radius, radius + 1):
        if abs(dy) == radius:
            for dx in range(-radius, radius + 1):
                yield dx, dy
        else:
            yield -radius, dy
            yield radius, dy


def _extend_all(mask: np.ndarray, directions) -> np.ndarray:
    """Extrude towards each direction from the SAME base mask, then union the results.

    Chaining the calls instead would let the first extension widen the frontier profile the second
    one reads, so a corner bridge would fan out into a band rather than gaining a second arm.
    """
    out = mask
    for d in directions:
        out = cv2.bitwise_or(out, extend_to_edge(mask, d))
    return out


def fit_bridge(mask: np.ndarray, B: np.ndarray, min_b: int, min_edge: int = 1, directions=None):
    """Translate ``mask`` by the smallest offset that makes it a valid bridge.

    Searches outward by Chebyshev radius and stops at the first offset that passes, so the annotation
    keeps as much of its own geometry as the constraints allow.

    Returns ``(mask, (dx, dy))``, or ``(None, None)`` if no offset works — which happens when the pad
    ring is too far from every edge for this mask to span both.

    A success is cheap because the search stops at the first hit, typically within a few pixels. A
    FAILURE walks every offset in the crop and costs ~0.2 s at 60x60, ~1.4 s at 120x120, since each
    candidate runs a connected-components pass. That is a property of the joint geometry rather than
    of one annotation, so a corpus that cannot be spanned pays it on every row rather than on a few.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 127)
    if len(ys) == 0:
        return None, None
    # Taken from the mask's ORIGINAL position: they are a property of where the annotation sits, and
    # recomputing them per candidate offset would let the targets flip mid-search.
    directions = directions or _target_edges(mask > 127)

    for radius in range(max(h, w)):
        for dx, dy in _ring_offsets(radius):
            y2, x2 = ys + dy, xs + dx
            inside = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
            if not inside.any():
                continue
            shifted = np.zeros_like(mask)
            shifted[y2[inside], x2[inside]] = 255
            shifted = _extend_all(shifted, directions)
            if bridge_ok(shifted, B, min_b, min_edge):
                return shifted, (dx, dy)
    return None, None


# ── Elastic deformation ───────────────────────────────────────────────────────
# Without this a bridge is a translated copy of its annotation, so the same (base mask, joint) pair
# always yields the same mask and n_seeds buys nothing. The warp is what makes seeds mean something
# for this class. It has to preserve three properties or it changes the defect rather than varying it:
#
#   contact  the mask must still reach every border it was fitted to,
#   area     the defect must stay the size it was observed at,
#   unity    it must remain ONE component -- the generator treats each component as its own instance
#            and would render a fragmented mask as several defects in separate crops.
#
# Each is handled explicitly below rather than hoped for: the displacement is damped to zero near the
# anchored borders, the area is restored by a level set, and fragments are dropped.

ANCHOR = 10.0  # px over which the displacement is damped to zero at an anchored border

# Peak displacement and the smoothness of the field that carries it. At 12 / 12 the placements are
# visibly distinct while the outline stays a plausible solder flow.
#
# Beyond roughly alpha 18 the shapes start to pinch and neck: still one component, still a valid
# bridge, but no longer a shape solder would make.
DEFORM_ALPHA = 12.0  # peak displacement, px
DEFORM_SIGMA = 12.0  # smoothness of the displacement field, px
DEFORM_TOL = 0.045  # area tolerance against the source annotation
DEFORM_ATTEMPTS = 12  # fresh fields to try before giving up
DEFORM_REPAIRS = 8  # repair passes per field


def _elastic(mask: np.ndarray, rng, alpha: float, sigma: float, directions) -> np.ndarray:
    """A smooth random displacement, damped to zero at every anchored border.

    Damping by the distance to the NEAREST anchored border, rather than to one of them, is what lets
    a corner bridge keep both of its contacts through the warp: near either border the displacement
    is zero, so the pixels that make contact cannot move away from it.
    """
    h, w = mask.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    fields = []
    for _ in range(2):
        f = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        fields.append(f / (np.abs(f).max() + 1e-8) * alpha)
    dx, dy = fields

    dist = np.full((h, w), np.inf, np.float32)
    for d in directions:
        dist = np.minimum(dist, {"right": (w - 1) - xs, "left": xs, "down": (h - 1) - ys, "up": ys}[d])
    damp = np.clip(dist / ANCHOR, 0.0, 1.0)

    return cv2.remap(
        mask,
        (xs + dx * damp).astype(np.float32),
        (ys + dy * damp).astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def set_area(mask: np.ndarray, target: int) -> np.ndarray:
    """Move the boundary until the mask covers ``target`` px, via a level set of its signed distance.

    Dilate/erode would move the boundary a whole pixel at a time, which on masks this small overshoots
    a few-percent window in one step. A level set is continuous in the threshold, and area is monotone
    in it, so bisection converges on the exact target.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return mask
    sdf = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3) - cv2.distanceTransform(m, cv2.DIST_L2, 3)
    lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (sdf <= mid).sum() < target:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda t: abs(int((sdf <= t).sum()) - target))
    return (sdf <= best).astype(np.uint8) * 255


def keep_main(mask: np.ndarray, directions) -> np.ndarray:
    """Drop warp fragments, keeping the component that best holds the anchored borders.

    Ranked by how many of the anchored borders a component touches, then by area, so a corner bridge
    keeps the piece holding both contacts rather than whichever happens to be largest.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n_labels <= 2:
        return mask
    h, w = mask.shape
    touches = {
        "left": lambda s: s[cv2.CC_STAT_LEFT] == 0,
        "right": lambda s: s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] >= w,
        "up": lambda s: s[cv2.CC_STAT_TOP] == 0,
        "down": lambda s: s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] >= h,
    }
    keep = max(
        range(1, n_labels),
        key=lambda i: (sum(touches[d](stats[i]) for d in directions), stats[i, cv2.CC_STAT_AREA]),
    )
    return (labels == keep).astype(np.uint8) * 255


def deform_bridge(
    mask: np.ndarray,
    rng,
    directions,
    alpha: float = DEFORM_ALPHA,
    sigma: float = DEFORM_SIGMA,
    tol: float = DEFORM_TOL,
    attempts: int = DEFORM_ATTEMPTS,
) -> Optional[np.ndarray]:
    """Warp a fitted bridge so it is not a copy of the annotation it came from.

    Returns None if no field converges within ``attempts``; the caller then keeps the un-warped mask,
    so a difficult shape costs variety rather than the row.

    ``tol`` bounds the area of what this returns. It does NOT bound the area the caller ships: a warp
    that slides off the pad ring is re-fitted, and that re-fit extrudes to the border again and can
    add a few percent more. Measured on the sample corpus, re-fitted rows landed at +6% against the
    un-warped mask where warped-only rows stayed inside the tolerance.
    """
    target = int((mask > 0).sum())
    if target == 0:
        return None
    gaps_ok = lambda m: all(_edge_gaps(m > 0)[d] == 0 for d in directions)  # noqa: E731

    for _ in range(attempts):
        warped = _elastic(mask, rng, alpha, sigma, directions)
        if not (warped > 0).any():
            continue
        for _ in range(DEFORM_REPAIRS):
            # Order matters: dropping fragments first means set_area sizes the SURVIVING blob, and
            # repeating it afterwards catches pieces the resize itself pinched off.
            warped = keep_main(warped, directions)
            warped = set_area(warped, target)
            warped = keep_main(warped, directions)
            if gaps_ok(warped) and abs(int((warped > 0).sum()) - target) <= tol * target:
                break
            if not gaps_ok(warped):
                warped = _extend_all(warped, directions)
        n_components = cv2.connectedComponents((warped > 0).astype(np.uint8), 8)[0] - 1
        if gaps_ok(warped) and abs(int((warped > 0).sum()) - target) <= tol * target and n_components == 1:
            return warped
    return None


def get_bridge_candidates(joints, min_roi_px: int = 1) -> List[ROICandidate]:
    """One candidate per joint, carrying B+C — for the ROI overlay only.

    Unlike a ring class this mask is NOT a confinement; the placer tests against the joint's pad ring
    and the crop edge, and never clips to it. It exists so the assets a run writes look the same for
    every THT class.
    """
    letters, _ = roi_rule(BRIDGE)
    candidates = []
    for joint in joints_carrying(joints, letters, BRIDGE):
        roi = roi_for(joint, letters)
        area = int((roi > 127).sum())
        if area < min_roi_px:
            continue
        ys, xs = np.nonzero(roi > 127)
        candidates.append(
            ROICandidate(
                mask=roi,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                centroid=(float(xs.mean()), float(ys.mean())),
                area=area,
                source_classes=list(letters),
                component_ids=[],
            )
        )
    return candidates


class THTBridgeMaskPlacer:
    """Fit a bridge annotation onto a joint so it spans the pad ring and the crop edge.

    Named with the THT prefix, unlike its sibling ``RingMaskPlacer``, because ``cad2roi`` exports a
    ``BridgeMaskPlacer`` of its own that takes different arguments. Two identically named classes
    would collide at any import site that wanted both, and the failure is a TypeError raised mid-run
    rather than anything naming the clash — so the names are kept distinct at the source instead of
    relying on every caller to alias correctly.
    """

    def __init__(
        self,
        min_b_frac: float = MIN_B_FRAC,
        min_b_floor: int = MIN_B_FLOOR,
        min_edge_frac: float = MIN_EDGE_FRAC,
        min_edge_floor: int = MIN_EDGE_FLOOR,
        deform: bool = True,
        deform_alpha: float = DEFORM_ALPHA,
        deform_sigma: float = DEFORM_SIGMA,
    ):
        self.min_b_frac = min_b_frac
        self.min_b_floor = min_b_floor
        self.min_edge_frac = min_edge_frac
        self.min_edge_floor = min_edge_floor
        # On by default: without it every bridge is a translated copy of its annotation, so a batch
        # repeats a handful of shapes however many rows it has.
        self.deform = deform
        self.deform_alpha = deform_alpha
        self.deform_sigma = deform_sigma

    def min_b(self, joint) -> int:
        """Required overlap with the pad ring, in pixels."""
        return max(self.min_b_floor, int(self.min_b_frac * joint.B.sum()))

    def min_edge(self, img_shape) -> int:
        """Required contact with the crop border, in pixels.

        Scaled off the crop's shorter side so it means the same thing on joints of different sizes.
        """
        return max(self.min_edge_floor, int(self.min_edge_frac * min(img_shape)))

    def place_on_joint(
        self, submask_path: str, joint, img_shape: tuple, seed: int = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """Fit one bridge onto one joint. Returns ``(mask, note)``; ``mask`` is None if it cannot fit.

        ``seed`` drives the deformation only. Without deformation the fit is the smallest shift that
        works and so is unique, and the seed changes nothing.
        """
        h, w = img_shape
        raw = np.array(Image.open(submask_path).convert("L")) > 127
        if not raw.any():
            return None, "EMPTY_SUBMASK"
        # Resized here rather than by AMP, which this class never reaches. NEAREST keeps the mask
        # binary; anything smoother would leave grey edges for bridge_ok to threshold away.
        mask = cv2.resize(raw.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        # Emptiness is re-tested HERE and not only above. NEAREST samples rather than averages, so a
        # 2 px stroke in a 1024 px annotation can lose every pixel on the way down to a joint crop --
        # raw.any() is true, this is entirely false, and _target_edges below then reaches xs.min()
        # on an empty array. That ValueError leaves place_all and kills the whole roi_place run, for
        # an annotation that is perfectly good at its own resolution. A distinct note because the
        # remedy is distinct: the mask is not empty, it is too thin for this crop size.
        if not (mask > 127).any():
            return None, "SUBMASK_LOST_IN_RESIZE"

        min_b, min_edge = self.min_b(joint), self.min_edge(img_shape)
        directions = _target_edges(mask > 127)

        if bridge_ok(mask, joint.B, min_b, min_edge):
            placed, note = mask, "as annotated"
        else:
            placed, shift = fit_bridge(mask, joint.B, min_b, min_edge, directions)
            if placed is None:
                return None, "NO_FIT"
            note = f"shifted by {max(abs(shift[0]), abs(shift[1]))} px"

        if not self.deform or seed is None:
            return placed, note

        warped = deform_bridge(
            placed, np.random.default_rng(seed), directions, alpha=self.deform_alpha, sigma=self.deform_sigma
        )
        if warped is None:
            # A shape no field could warp within tolerance. Keeping the un-deformed mask costs
            # variety on this row rather than losing the row.
            return placed, f"{note}, warp did not converge"
        if bridge_ok(warped, joint.B, min_b, min_edge):
            return warped, f"{note}, deformed"
        # The warp preserves border contact and area, but nothing in it knows about the pad ring, so
        # it can slide the mask off B. Re-fitting recovers that instead of discarding the warp.
        refit, _ = fit_bridge(warped, joint.B, min_b, min_edge, directions)
        if refit is not None:
            return refit, f"{note}, deformed + refitted"
        return placed, f"{note}, warp left the pad ring"

    def place_all(
        self,
        submask_path: str,
        joints: List,
        img_shape: tuple,
        n_instances: int = 1,
        n_seeds: int = 1,
        base_seed: int = 42,
    ) -> dict:
        """Fit the bridge onto ``n_instances`` joints per seed.

        Returns ``{seed_index: (n_requested, n_placed, mask or None)}``, the same contract the ring
        placer and the cad2roi placers use.

        ``base_seed`` selects which joints a seed uses; it does not vary the bridge itself, which is
        the smallest shift that works and therefore unique. Asking for several seeds on a single
        joint repeats one mask, so the caller is told rather than left to find duplicates later.
        """
        eligible = joints_carrying(joints, roi_rule(BRIDGE)[0], BRIDGE)
        if not eligible:
            return {}
        h, w = img_shape
        n_total = len(eligible)
        n_sel = min(max(1, n_instances), n_total)
        # Without deformation, seeds only vary WHICH joints are used. When every joint is taken
        # anyway, rng.choice returns a permutation of the same set and the OR below is
        # order-independent, so each seed yields a byte-identical mask.
        if n_seeds > 1 and n_sel == n_total and not self.deform:
            log.warning(
                f"  bridge: {n_seeds} seeds over {n_total} joint(s) with n_instances={n_sel} repeats "
                "one mask — the fit is the smallest shift that works and so is unique; variety has "
                "to come from the submask bank"
            )

        results = {}
        for seed_i in range(n_seeds):
            rng = np.random.RandomState(base_seed + seed_i)
            chosen = rng.choice(n_total, size=n_sel, replace=False)

            combined = np.zeros((h, w), dtype=np.uint8)
            n_ok = 0
            # Always False, and the tuple carries it so both THT placers answer the same shape. This
            # placer has had a minimum all along and it is a stricter one: MIN_B_FRAC/MIN_B_FLOOR on
            # the pad ring and the same again at the crop edge, checked as a FIT rather than as a
            # share of the annotation, so a bridge that only grazes never becomes a mask at all --
            # there is nothing here for an undersized-but-written label to describe.
            small = False
            for ji in chosen:
                joint = eligible[ji]
                # A distinct warp per (seed, joint), so two seeds cannot deform the same joint alike.
                mask, note = self.place_on_joint(
                    submask_path, joint, img_shape, seed=base_seed + seed_i * n_total + int(ji)
                )
                if mask is None:
                    # The note carries the reason; asserting one here named the fit for every cause,
                    # including an annotation that never reached the fit at all.
                    log.warning(f"  bridge: no mask for this joint ({note}), skipping joint")
                    continue
                combined = cv2.bitwise_or(combined, mask)
                n_ok += 1
                log.info(f"  bridge: placed ({note})")

            results[seed_i + 1] = (n_sel, n_ok, combined if combined.sum() > 0 else None, small)
        return results
