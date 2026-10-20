# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group a parsed THT semantic mask into joints, and build a joint's ROI from region letters.

WHY THE CLASSES ARE THE REGIONS
The delivered THT annotation is pixel-disjoint, so each class is already the ring the placement rules
want and none of them has to be computed:

    pad  == B == pad_disc - hole      hole == C == hole_disc - lead      lead == D

WHICH CLASS IS THE PIN
``pin_tip`` is its face seen from above, and ``pin`` is the sliver of side wall a tilted lead exposes.
D is therefore the ``pin_tip``, and the sliver folds into the hole ring it sits inside rather than
becoming a second, disconnected D.

An annotation without ``pin_tip`` is unchanged: D falls back to ``pin``.
The choice is made per joint, not per mask, so a mask that carries both degrades one joint at a time.

WHY JOINTS, NOT CROPS
A joint crop is cut from a full board, and ``crop_ratio`` can pull a neighbouring pad into frame. One
ROI per crop would then silently merge two joints into one region. Components are grouped into joints
instead, so a crop with a neighbour yields two candidates and ``n_instances`` stays meaningful.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from cosmos_framework.utils import log

PAD_CLASS, HOLE_CLASS, PIN_CLASS, PIN_TIP_CLASS = "pad", "hole", "pin", "pin_tip"

# The lead classes, in the order D prefers them: a joint's D is the first one it carries.
_LEAD_CLASSES = (PIN_TIP_CLASS, PIN_CLASS)


@dataclass
class THTJoint:
    """One solder joint: its three disjoint regions plus the geometry ``roi_inner`` is stated in."""

    B: np.ndarray  # bool (H, W) -- solder ring
    C: np.ndarray  # bool (H, W) -- hole ring
    D: np.ndarray  # bool (H, W) -- the lead face (pin_tip, or pin when the split is absent)
    disc: np.ndarray  # bool (H, W) -- the filled pad disc, i.e. B|C|D closed over annotation gaps
    centre: tuple  # (cx, cy) of disc -- x first, as CADParser and ROICandidate both use
    radius: float  # equivalent radius of disc, in pixels

    def regions(self) -> dict:
        """``{letter: mask}`` for the three region letters."""
        return {"B": self.B, "C": self.C, "D": self.D}

    def missing(self, letters: str) -> list:
        """Which of ``letters`` name a region this joint does not actually carry.

        A joint clipped by the crop edge keeps a slice of its pad but loses the hole and pin, because
        filling a partial arc's contour no longer encloses them. The result is a joint that still
        looks placeable -- ``B`` alone is easily above any area floor -- while the region the rule
        asked for is not what it would be placed into. Callers use this to drop such a joint instead
        of silently degrading "B+C" to "B".
        """
        region = self.regions()
        return [letter for letter in letters if not region[letter].any()]


def _fill(mask: np.ndarray) -> np.ndarray:
    """The filled interior of an annulus.

    ``B|C|D`` is the same region whenever the annotation is perfectly nested, but a one-pixel gap in
    the hole ring would leak the union and drag the centroid with it. Filling the pad's outer contour
    depends only on its outline, so the centre and radius stay stable under that kind of noise.
    """
    filled = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def joints_from_components(components: dict, img_shape: tuple) -> list:
    """Group ``CADParser`` components into ``THTJoint``s, one per ``pad`` component.

    ``hole``, ``pin`` and ``pin_tip`` components are assigned to the pad whose filled disc contains
    their centroid; a component that lands in no pad is dropped, since it has no joint to belong to.
    """
    h, w = img_shape
    pads = components.get(PAD_CLASS, [])
    discs = [_fill(p.mask > 0) for p in pads]

    def owner(comp):
        # First match wins. Pad discs on a board do not overlap, so at most one can contain a given
        # centroid and the iteration order does not matter.
        cx, cy = comp.centroid  # CADParser stores (x, y)
        y, x = int(round(cy)), int(round(cx))
        if not (0 <= y < h and 0 <= x < w):
            return None
        for i, d in enumerate(discs):
            if d[y, x]:
                return i
        return None

    tracked = (HOLE_CLASS, *_LEAD_CLASSES)
    per_pad = {i: {cls: [] for cls in tracked} for i in range(len(pads))}
    orphans = {}
    for cls in tracked:
        for comp in components.get(cls, []):
            i = owner(comp)
            if i is None:
                # Almost always a joint the crop edge cut through: its pad survives as an arc whose
                # filled contour no longer encloses this component. Reported because the joint it
                # belonged to then carries an empty region, which changes what a rule naming that
                # letter would place into.
                orphans[cls] = orphans.get(cls, 0) + 1
                continue
            per_pad[i][cls].append(comp.mask > 0)
    if orphans:
        detail = ", ".join(f"{n} {cls}" for cls, n in sorted(orphans.items()))
        log.warning(f"  {detail} component(s) outside every pad disc, dropped (joint clipped by the crop edge?)")

    def union(masks):
        out = np.zeros((h, w), bool)
        for m in masks:
            out |= m
        return out

    def centre_and_radius(disc, C, D):
        """The joint's centre and outer radius, read so that a clipped pad cannot move them.

        A joint the crop edge cuts through keeps a TRUNCATED arc of its pad, and it is not dropped:
        ``joints_carrying`` only rejects a joint whose region is wholly empty, and a cut that takes a
        slice of the pad while leaving the hole intact takes nothing. The centroid of that arc is not
        the joint's centre and ``sqrt(area/pi)`` is not its radius -- measured on a pad of true
        centre (8, 40) and radius 30 clipped at the left edge, they come out (16.3, 40.0) and 24.7.
        ``roi_for``'s inner cut is then displaced 8 px and 18% too small, so for a rule with
        ``inner_k`` the ring that is meant to be removed is neither centred nor half-radius and the
        defect is allowed where the physics says it cannot sit.

        The regions are concentric, so a clipped joint is measured from the innermost one that is
        actually there -- the lead, then the hole -- and the radius from the furthest pad pixel,
        which is the true outer radius as long as ANY un-clipped pad survives, whereas an area-based
        radius shrinks with whatever was cut away.

        Only for a clipped joint. An intact one keeps the area formula it has always used, so this
        cannot move a placement that was not already wrong.
        """
        ys, xs = np.nonzero(disc)
        clipped = disc[0].any() or disc[-1].any() or disc[:, 0].any() or disc[:, -1].any()
        if not clipped:
            return (float(xs.mean()), float(ys.mean())), float(np.sqrt(disc.sum() / np.pi))
        for inner in (D, C):
            iys, ixs = np.nonzero(inner)
            if len(iys):
                centre = (float(ixs.mean()), float(iys.mean()))
                break
        else:
            centre = (float(xs.mean()), float(ys.mean()))
        return centre, float(np.hypot(xs - centre[0], ys - centre[1]).max())

    joints = []
    for i, pad in enumerate(pads):
        disc = discs[i]
        ys, xs = np.nonzero(disc)
        if len(ys) == 0:
            continue
        # D is the lead face; anything else annotated on the lead is part of the hole ring it sits
        # inside. Per joint rather than per mask, so a board where only some leads are tilted enough
        # to expose a side wall is handled one joint at a time.
        lead = next((cls for cls in _LEAD_CLASSES if per_pad[i][cls]), PIN_CLASS)
        rest = [m for cls in _LEAD_CLASSES if cls != lead for m in per_pad[i][cls]]
        C = union(per_pad[i][HOLE_CLASS] + rest)
        D = union(per_pad[i][lead])
        centre, radius = centre_and_radius(disc, C, D)
        joints.append(THTJoint(B=pad.mask > 0, C=C, D=D, disc=disc, centre=centre, radius=radius))
    return joints


def joints_carrying(joints, letters: str, label: str = "") -> list:
    """The joints that actually have every region in ``letters``; the rest are reported and dropped.

    A joint the crop edge cut through keeps a slice of its pad but loses the hole and pin, because
    filling a partial arc's contour no longer encloses them. It then still looks placeable -- ``B``
    alone clears any area floor -- while the region the rule asked for is not what would be placed
    into. Shared by every defect module so the rule cannot drift between them.
    """
    kept = []
    for joint in joints:
        absent = joint.missing(letters)
        if absent:
            prefix = f"  {label}: " if label else "  "
            log.warning(f"{prefix}joint has no {'/'.join(absent)} region, skipping (clipped or unannotated)")
            continue
        kept.append(joint)
    return kept


def roi_for(joint: THTJoint, letters: str, inner_k: float = 0.0) -> np.ndarray:
    """The union of the joint's region letters, optionally with its inside cut away.

    ``inner_k`` is measured in PAD radii against the filled disc -- never against the ``pad`` class
    alone, which is already the ring and would give a radius roughly a third too small.

    Returns a uint8 0/255 mask so it can be handed straight to AMP through a temp PNG.
    """
    region = joint.regions()
    m = np.zeros(joint.B.shape, bool)
    for letter in letters:
        m |= region[letter]
    if inner_k > 0:
        cx, cy = joint.centre
        yy, xx = np.ogrid[: m.shape[0], : m.shape[1]]
        m &= np.hypot(yy - cy, xx - cx) >= inner_k * joint.radius
    return m.astype(np.uint8) * 255
