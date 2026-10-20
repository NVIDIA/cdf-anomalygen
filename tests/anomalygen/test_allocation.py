# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared num_sdg split: it sums to the total, and AMP really does use it.

Only what is NEW here is tested. `proportional`, the validation floor and the override all moved out
of `roi_allocate` unchanged and keep their coverage in
`tests/anomalygen/scripts/auto_mask_placement/test_roi_allocate.py`, which reaches them through the
delegating `_proportional` / `_allocate` — so re-testing them here would only duplicate it.
"""

import pytest

from anomalygen import allocation


@pytest.mark.parametrize(
    ("total", "n", "expected"),
    [
        (25, 3, [9, 8, 8]),  # the case that motivated this: a per-type ceil would deliver 27
        (25, 5, [5, 5, 5, 5, 5]),  # exact division: no remainder to place
        (2, 5, [1, 1, 0, 0, 0]),  # fewer samples than types — a 0 share is an answer, not an error
        (0, 3, [0, 0, 0]),
        (7, 1, [7]),
    ],
)
def test_the_split_sums_to_the_total_with_the_remainder_at_the_front(total, n, expected):
    types = [f"t{i}" for i in range(n)]
    alloc = allocation.uniform(total, types)
    assert [alloc[t] for t in types] == expected
    assert sum(alloc.values()) == total, "an allocation that does not sum to the total is the bug"


def test_order_decides_who_gets_the_remainder():
    """A pure function of (total, types) INCLUDING their order. That is what makes an allocated run
    resumable — re-deriving the split from the same inputs reproduces every type's target."""
    assert allocation.uniform(5, ["a", "b", "c"]) == {"a": 2, "b": 2, "c": 1}
    assert allocation.uniform(5, ["c", "b", "a"]) == {"c": 2, "b": 2, "a": 1}


@pytest.mark.parametrize(("total", "types"), [(5, []), (-1, ["a"])])
def test_refuses_what_it_cannot_split(total, types):
    with pytest.raises(ValueError):
        allocation.uniform(total, types)


@pytest.mark.parametrize(
    "split",
    [
        lambda types: allocation.uniform(5, types),
        lambda types: allocation.proportional(5, types, {"a": 1, "b": 1}),
        lambda types: allocation.allocate(5, types, {"a": 1, "b": 1}),
        lambda types: allocation.allocate(5, types, {"a": 1, "b": 1}, mode="validation"),
        lambda types: allocation.allocate(5, types, {}, override={"a": 2, "b": 3}),
    ],
    ids=["uniform", "proportional", "allocate-inference", "allocate-validation", "allocate-override"],
)
def test_a_repeated_type_is_refused_by_every_entry_point_naming_it(split):
    """A dict keyed by type holds one entry per type, so ``[a, a, b]`` collapses to two keys and the
    split stops summing to the total (uniform would deliver 4 of 5). Refused, naming the repeat."""
    with pytest.raises(ValueError, match=r"'a'"):
        split(["a", "a", "b"])


def test_validation_names_a_type_with_no_real_masks_instead_of_suggesting_the_same_total():
    """No total gives a count-0 type a proportional share, so "increase the total to >= <the total you
    passed>" is not advice. The message states the fact and the real options, in the caller's own
    flag names — AMP and AMG spell them differently."""
    with pytest.raises(ValueError) as ei:
        allocation.allocate(
            5,
            ["a", "b"],
            {"a": 10, "b": 0},
            mode="validation",
            total_label="--num-sdg",
            mode_label="--mode",
            override_label="--per-defect-counts",
        )
    msg = str(ei.value)
    assert "'b'" in msg and "no real masks" in msg, msg
    assert ">= 5" not in msg, msg
    assert "--mode inference" in msg and "--per-defect-counts" in msg, msg


def test_validation_suggests_a_larger_total_for_a_type_that_rounded_to_zero():
    """b has real masks; 5 samples over 101 is what starves it, so a larger total IS the fix."""
    with pytest.raises(ValueError, match=r"'b'.*>= 303"):
        allocation.allocate(5, ["a", "b"], {"a": 100, "b": 1}, mode="validation")


@pytest.mark.parametrize("value", [-3, 2.5], ids=["negative", "fractional"])
def test_an_override_that_is_not_a_whole_count_is_refused_naming_the_type(value):
    """A coercing int() would hand the fill loop a negative target nothing can meet — the quota
    would launch no beams and never settle — or silently floor 2.5. The refusal names the type.
    An integral float stays accepted: JSON round-trips whole counts through floats."""
    with pytest.raises(ValueError, match=r"B\+y"):
        allocation.allocate(5, ["A+x", "B+y"], {}, override={"A+x": 5, "B+y": value})
    assert allocation.allocate(2, ["A+x"], {}, override={"A+x": 2.0}) == {"A+x": 2}


def test_amp_allocates_by_this_rule_rather_than_its_own_copy():
    """A second implementation would drift, and the two backends answering num_sdg differently is
    exactly the surprise this module exists to remove."""
    from anomalygen.scripts.auto_mask_placement.roi_allocate import _uniform

    types = ["A+x", "B+y", "C+z"]
    assert _uniform(25, types) == allocation.uniform(25, types) == {"A+x": 9, "B+y": 8, "C+z": 8}


def test_override_warnings_name_the_flag_the_user_typed():
    """Both override warnings interpolate override_label/total_label — the messages hardcoded
    'per-type counts' / 'the requested total', so neither could name --per-defect-counts or
    --num-sdg, the flags the user actually passed."""
    warnings: list[str] = []
    allocation.allocate(
        5,
        ["A+x"],
        {},
        override={"A+x": 3, "zzz": 1},
        warn=warnings.append,
        total_label="--num-sdg",
        override_label="--per-defect-counts",
    )
    unknown, mismatch = warnings
    assert "--per-defect-counts" in unknown, unknown
    assert "--per-defect-counts" in mismatch and "--num-sdg" in mismatch, mismatch
