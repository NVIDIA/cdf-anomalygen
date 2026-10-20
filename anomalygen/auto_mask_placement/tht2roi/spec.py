# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The THT defect enum and its per-class ROI rules.

This module is the single source of truth for which THT defect classes exist and where each one is
allowed to be placed. It deliberately lives in code rather than in ``defect_spec.jsonl``: the spec
row for a ``cad_tht`` defect carries exactly the fields a ``cad`` row carries, so a dataset cannot
put a defect in the wrong region, and the supported enum is a constant that documentation and
configuration validation can both point at.

THE REGION LETTERS

Three disjoint regions, read straight off the semantic mask's classes (see ``regions.py``, which
also covers which class supplies D when the annotation splits the lead):

    B  the solder ring   the pad, excluding the hole
    C  the hole ring     the hole, excluding the lead face
    D  the lead face     the pin tip

``A`` (the whole crop) is not offered. Every class in the shipped spec is confined to B/C/D, and a
whole-crop ROI is what ``spatial_dependency: free`` already means.

THE INNER CUT

A region built from these letters is not necessarily an annulus. Whenever it spans the pin — ``B+C``
is the pad with only the pin removed — its inner boundary lies close to the pad centre, so that
centre is itself a legal position. AMP anchors a submask on the ROI's bounding-box centre, which for
a concentric region is exactly that point, so placements concentrate there.

For a defect whose physics puts it at a radius rather than at the centre, that concentration is
wrong, and no amount of retrying fixes it: the centre is legal, so it keeps being chosen.
``THT_ROI_INNER`` removes it from the legal set instead, cutting the region's inside back to a given
fraction of the pad radius and leaving an annulus the anchor point cannot land in.

The cut is stated in pad radii rather than pixels so one value holds across joint sizes, and it is
applied after the letters, which makes "the outer part of B+C" expressible without inventing a new
region letter.

Both tables carry defaults tuned for through-hole solder joints. They are the shipped policy, not a
property of any particular dataset — a corpus whose defects sit differently needs the values
revisited, and the values a run actually used are worth recording alongside its output.
"""

# class -> the region letters it may occupy
THT_ROI_SPEC = {
    "bridge": "BC",  # plus crop-edge contact; NOT used as a confinement — see defects/bridge.py
    "bright": "BC",
    "dry_contact": "C",
    "hidden": "BCD",
    "interrupted": "BC",
    "less_solder": "BC",
    "trapped": "BC",
}

# class -> how much of the region's inside to cut away, in pad radii. Absent means no cut.
# bright / interrupted real ROI: B + C - 0.5 * disc of pad radii
THT_ROI_INNER = {
    "bright": 0.50,
    "interrupted": 0.50,
}

# bridge is placed by its own mechanism: it must span pad -> crop edge, so it is shifted to fit
# rather than confined to a region. Everything else shares one ring-confined placer.
BRIDGE = "bridge"
RING_CLASSES = tuple(c for c in THT_ROI_SPEC if c != BRIDGE)
THT_CLASSES = tuple(THT_ROI_SPEC)

VALID_LETTERS = frozenset("BCD")


def _validate_tables():
    """Check the two tables above at import time.

    Every mistake this catches is otherwise silent. An unknown letter raises a ``KeyError`` deep in
    ``roi_for``; a misspelled ``THT_ROI_INNER`` key reads back as "no inner cut", so the class quietly
    keeps the centre it was supposed to lose; and an ``inner_k`` at or above 1.0 removes the whole pad
    disc, leaving an empty ROI and a run in which every joint is skipped for looking too small.

    Done here rather than in ``roi_rule`` because these are properties of the tables, not of one
    lookup — a bad entry for a class nobody places this run is still a bad entry.
    """
    for cls, letters in THT_ROI_SPEC.items():
        if not letters:
            raise ValueError(f"THT_ROI_SPEC[{cls!r}] is empty; a class must name at least one region")
        bad = sorted(set(letters) - VALID_LETTERS)
        if bad:
            raise ValueError(
                f"THT_ROI_SPEC[{cls!r}] = {letters!r} uses unknown region letter(s) {bad}; "
                f"valid letters are {sorted(VALID_LETTERS)}"
            )
    unknown = sorted(set(THT_ROI_INNER) - set(THT_ROI_SPEC))
    if unknown:
        raise ValueError(f"THT_ROI_INNER has entries for unknown class(es) {unknown}")
    for cls, k in THT_ROI_INNER.items():
        if not 0.0 <= k < 1.0:
            raise ValueError(f"THT_ROI_INNER[{cls!r}] = {k}; the inner cut must be in [0.0, 1.0) pad radii")


_validate_tables()


def roi_rule(tht_defect: str):
    """``(letters, inner_k)`` for a THT class. Raises on an unknown class.

    Unknown names fail loudly here rather than yielding an empty candidate list further down, where
    the run would look like a defect that simply found nowhere to go.
    """
    if tht_defect not in THT_ROI_SPEC:
        raise ValueError(f"unknown THT defect {tht_defect!r}; supported: {', '.join(sorted(THT_ROI_SPEC))}")
    return THT_ROI_SPEC[tht_defect], THT_ROI_INNER.get(tht_defect, 0.0)
