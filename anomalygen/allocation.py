# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How ``num_sdg`` is split across anomaly types — the one rule, shared by both mask backends.

AMP (``roi_allocate``) and AMG (``scripts/agentic_mask_generation/generate.py``) place masks by
completely different means, but they answer to the same request: *produce ``num_sdg`` samples*. That
they answer it identically should be a fact rather than a coincidence, so the arithmetic lives here
and neither backend owns it.

Deliberately dependency-free — no torch, no cv2, no framework logging. Both importers sit in
packages the other has no business pulling in, and this is arithmetic.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, Optional, Sequence


def _refuse_repeats(types: Sequence[str]) -> None:
    """Every split is a dict keyed by type, which holds one entry per type: ``[a, a, b]`` collapses to
    two keys and the result stops summing to the total. Refused at every entry point, naming the repeat."""
    repeated = sorted(t for t, k in Counter(types).items() if k > 1)
    if repeated:
        raise ValueError(f"types must be distinct (repeated: {repeated})")


def uniform(total: int, types: Sequence[str]) -> Dict[str, int]:
    """Split ``total`` as evenly as whole samples allow: ``total // n`` each, remainder to the front.

    The result sums to EXACTLY ``total``. That is the whole point of allocating rather than rounding
    a per-type count up: ``ceil(25 / 3) = 9`` over three types delivers 27, and the two extra masks
    cost real API calls and GPU time before anything discards them. ``{9, 8, 8}`` never produces them.

    The remainder goes to the FIRST types in the given order, so the split is a pure function of
    ``(total, types)``. Order is therefore load-bearing: a run resumed with its types in a different
    order would target a different split, which is why callers pass the user's declared order and
    compare it when deciding whether two runs are the same run.

    ``total`` below ``len(types)`` gives the leading types 1 and the rest 0. A zero-target type is a
    real answer to "spread 2 samples over 5 types", not an error — the caller decides whether to run
    it at all.
    """
    n = len(types)
    if n == 0:
        raise ValueError("types must be non-empty")
    if total < 0:
        raise ValueError(f"total must be non-negative (got {total})")
    _refuse_repeats(types)
    base, remainder = divmod(total, n)
    alloc = {t: base for t in types}
    for t in list(types)[:remainder]:
        alloc[t] += 1
    return alloc


def proportional(total: int, types: Sequence[str], counts: Dict[str, int]) -> Dict[str, int]:
    """Split ``total`` in proportion to ``counts``, by largest-remainder rounding.

    Sums to EXACTLY ``total``, like :func:`uniform`: every share is floored and the leftover goes to
    the types with the largest discarded fractions. ``counts`` is how many real training masks each
    type has, so a validation set built this way mirrors the distribution the model actually trains
    on — which is the point of using it there rather than an even split.
    """
    _refuse_repeats(types)
    denom = float(sum(counts.get(t, 0) for t in types))
    if denom <= 0:
        raise ValueError(f"sum of mask counts must be > 0 (got {counts})")
    raw = {t: total * counts.get(t, 0) / denom for t in types}
    floors = {t: int(math.floor(r)) for t, r in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(types, key=lambda t: raw[t] - floors[t], reverse=True)
    for t in order[:remainder]:
        floors[t] += 1
    return floors


def allocate(
    total: int,
    types: Sequence[str],
    counts: Dict[str, int],
    *,
    mode: str = "inference",
    override: Optional[Dict[str, int]] = None,
    warn: Optional[Callable[[str], None]] = None,
    total_label: str = "total",
    mode_label: str = "mode",
    override_label: str = "explicit per-type counts",
) -> Dict[str, int]:
    """The whole policy: which split a request gets, and when a request is refused.

    ``mode``:
      * ``"inference"`` — the generation set. Uniform, no per-type floor: a type may legitimately get
        0 when the total is smaller than the type count.
      * ``"validation"`` — the set Step 3 validates against. Proportional to ``counts``, and a zero
        share is REFUSED rather than delivered: validation scores per type, so a type with no samples
        silently drops out of the KPI that picks the best checkpoint.

    ``override`` sets exact per-type counts and is inference-only; types it omits get 0. Its sum wins
    when it disagrees with ``total`` — that is what asking for exact counts means — and ``warn`` is
    told so. Callers pass their own logger; this module keeps no logging dependency of its own.

    The ``*_label`` arguments are what the messages call these things. The two backends spell the same
    request differently (``--num_sdg`` for AMP, ``--num-sdg`` for AMG), and an error is worth more when
    it names the flag the user actually typed than when it names an internal parameter.
    """
    if mode not in ("inference", "validation"):
        raise ValueError(f"unknown {mode_label} {mode!r} (expected 'inference' or 'validation')")
    if total < 0:
        raise ValueError(f"{total_label} must be >= 0 (got {total})")
    if not types:
        raise ValueError("types must be non-empty")
    _refuse_repeats(types)

    if override is not None:
        if mode != "inference":
            raise ValueError(f"{override_label} is only valid with {mode_label} inference (got {mode_label} {mode})")
        unknown = [k for k in override if k not in types]
        if unknown and warn:
            warn(f"{override_label} names unknown type(s) {unknown} not in {list(types)}; ignored")
        # A count is a whole number of masks >= 0 — REFUSED rather than coerced, naming each type.
        # int() would hand the fill loop a negative target it can never meet (nothing would launch
        # and nothing would settle) and would silently floor 2.5 to 2; neither is what was asked
        # for. An integral float (12.0) IS accepted as 12: JSON round-trips through floats, and
        # refusing a value that names an exact whole count would be pedantry. Only the types that
        # participate are validated: an unknown type is ignored with its value, as warned above.
        bad = {
            k: v
            for k, v in override.items()
            if k in types
            and (isinstance(v, bool) or not (isinstance(v, (int, float)) and v >= 0 and float(v).is_integer()))
        }
        if bad:
            raise ValueError(
                f"{override_label} must be whole numbers >= 0; got "
                + ", ".join(f"{k}: {v!r}" for k, v in sorted(bad.items()))
            )
        alloc = {t: int(override.get(t, 0)) for t in types}
        if sum(alloc.values()) != total and warn:
            warn(
                f"{override_label} sum ({sum(alloc.values())}) differs from {total_label} "
                f"({total}); using the override sum as the effective {total_label}"
            )
        return alloc

    if mode == "inference":
        return uniform(total, types)

    alloc = proportional(total, types, counts)
    # KPI floor: training validation needs per-type samples for stable nn_score / mnn_score. Refuse to
    # silently allocate 0 — the type would vanish from the score that chooses the best checkpoint, and
    # nothing downstream reports which type went missing.
    zero_types = [t for t in types if alloc[t] == 0]
    if zero_types:
        # Two different starvations, two different remedies. A type with NO real masks gets a zero
        # share from every total — the proportion itself is 0 — so a larger total is not advice for it.
        # A type that HAS masks but rounded to 0 is starved by this total, and a larger one is the fix.
        no_data = [t for t in zero_types if counts.get(t, 0) <= 0]
        rounded_out = [t for t in zero_types if counts.get(t, 0) > 0]
        problems = []
        if no_data:
            problems.append(
                f"types {no_data} have no real masks in the dataset, so {mode_label} validation cannot "
                f"give them a proportional share. Drop them, or use {mode_label} inference (uniform "
                f"split, or exact counts via {override_label})"
            )
        if rounded_out:
            n_total = int(sum(counts.get(t, 0) for t in types))
            n_min = min(counts.get(t, 0) for t in types if counts.get(t, 0) > 0)
            suggested = max(total, math.ceil(3 * n_total / n_min))
            problems.append(
                f"types {rounded_out} got 0 entries with {total_label}={total}. "
                f"Increase {total_label} to >= {suggested} or reduce the types"
            )
        raise ValueError("validation coverage broken: " + "; ".join(problems) + ".")
    return alloc
