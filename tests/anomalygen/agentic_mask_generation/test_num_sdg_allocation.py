# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`--num-sdg` sizes a run the way AMP does: one total, split across the declared types, and a resume
owes the recorded split. These pin what the flag does to the DELIVERABLE through the real
`generate.main()`; the split arithmetic itself is pinned in `tests/anomalygen/test_allocation.py`."""

from __future__ import annotations

from collections import Counter

import pytest

from .test_cross_type_scheduler import _drive

_THREE = ("T0+D", "T1+D", "T2+D")


def _types(rows):
    return Counter(r["anomaly_type"] for r in rows)


@pytest.mark.parametrize(
    ("total", "expected"),
    [(25, {"T0+D": 9, "T1+D": 8, "T2+D": 8}), (9, {"T0+D": 3, "T1+D": 3, "T2+D": 3})],
    ids=["remainder", "exact"],
)
def test_num_sdg_delivers_the_total_asked_for(monkeypatch, tmp_path, total, expected):
    """MUTANTS this kills: allocating with ceil (27 rows), flooring (24), handing every type the full
    total (75), or dropping the remainder so the split sums to 24."""
    r = _drive(monkeypatch, tmp_path, out=tmp_path / "o", types=_THREE, argv=("--num-sdg", str(total)))
    assert r.exit is None, f"the run should finish: {r.exit}"
    assert len(r.rows) == total and _types(r.rows) == expected


def test_an_allocated_run_resumes_to_its_own_per_type_shortfall(monkeypatch, tmp_path):
    """Resume re-derives each type's target from the same total, so only the shortfall is produced.
    MUTANT this kills: resuming against the run's TOTAL per type — a complete run re-run would grow."""
    out = tmp_path / "o"
    assert len(_drive(monkeypatch, tmp_path, out=out, types=_THREE, argv=("--num-sdg", "25")).rows) == 25
    again = _drive(monkeypatch, tmp_path, out=out, types=_THREE, argv=("--num-sdg", "25"))
    assert again.exit is None and len(again.rows) == 25, "a complete run re-run must not grow"
    assert not again.beams, "nothing was short, so no beam should have been launched"


def test_a_resume_is_held_to_the_earlier_runs_split_not_just_its_flags(monkeypatch, tmp_path):
    """The resolved per-type split is recorded and compared on resume — the flags cannot see a dataset
    that grew between runs. MUTANT this kills: trusting the flags alone, which lets a second process
    target a different split and blend two allocations into one testcase.jsonl."""
    out = tmp_path / "o"
    first = _drive(monkeypatch, tmp_path, out=out, types=_THREE, argv=("--num-sdg", "9", "--mode", "validation"))
    assert first.exit is None and len(first.rows) == 9

    # Same command, same everything the control plane can see — but the dataset now has more masks for
    # one type, so the proportional split moves.
    again = _drive(
        monkeypatch,
        tmp_path,
        out=out,
        types=_THREE,
        argv=("--num-sdg", "9", "--mode", "validation"),
        per_type={"T0+D": 9, "T1+D": 1, "T2+D": 1},
    )
    assert again.exit is not None, "a moved split must be refused, not blended in"
    assert "per-type split DIFFERS" in str(again.exit)
    assert len(again.rows) == 9, "the refused run must leave the first run's deliverable alone"
