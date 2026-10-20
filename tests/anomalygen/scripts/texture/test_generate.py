# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generate.py's distributed work-splitting / result-merging helpers.

Importing generate.py runs the framework's ``init_script`` at module load, so these tests are kept
in their own module. They exercise the pure single-rank paths with real inputs (no GPU, no
distributed init, no mocks) — in particular the guardrail block-record -> merge -> ``guardrail_blocked.csv``
path, which is otherwise only exercised end-to-end.
"""

import csv

from anomalygen.inference.guardrail import BLOCKED_CSV_HEADER, blocked_row
from anomalygen.scripts.texture.generate import (
    _build_sample_output_plans,
    _get_rank_work_items,
    _merge_rank_rows,
    gather_from_ranks,
)


def test_get_rank_work_items_round_robin():
    assert _get_rank_work_items(7, rank=0, world_size=3) == [0, 3, 6]
    assert _get_rank_work_items(7, rank=1, world_size=3) == [1, 4]
    assert _get_rank_work_items(5, rank=0, world_size=1) == [0, 1, 2, 3, 4]


def test_build_sample_output_plans_offsets_per_anomaly_type():
    data = [
        {"index": 10, "anomaly_type": "Phone+oil", "num_generated_images": 2},
        {"index": 11, "anomaly_type": "Phone+oil", "num_generated_images": 1},
        {"index": 12, "anomaly_type": "Phone+scratch", "num_generated_images": 1},
    ]
    plans = _build_sample_output_plans(data)
    assert (plans[10].global_order, plans[10].anomaly_offset) == (0, 0)
    assert plans[11].anomaly_offset == 2  # after the two Phone+oil outputs
    assert plans[12].anomaly_offset == 0  # a different anomaly_type starts its own offset


def test_blocked_rows_merge_single_rank_and_write_guardrail_csv(tmp_path):
    # Blocked records arrive out of order (as ranks would): two per-output image blocks of the SAME
    # sample (index 1, output_idx 0 and 1) plus a whole-sample text block (index 2, output_idx -1).
    # The single-rank merge must order by (index, output_idx) — keeping the two same-sample rows
    # distinct — and round-trip through the same DictWriter generate.py uses for guardrail_blocked.csv.
    rows = [
        blocked_row(1, 1, "Phone+oil", "a.png", "a_mask.png", "image", "blocked a out1"),
        blocked_row(1, 0, "Phone+oil", "a.png", "a_mask.png", "image", "blocked a out0"),
        blocked_row(2, -1, "Phone+scratch", "b.png", "b_mask.png", "text", "blocked b"),
    ]
    merged = _merge_rank_rows(gather_from_ranks(rows, world_size=1))

    assert [(r["index"], r["output_idx"]) for r in merged] == [(1, 0), (1, 1), (2, -1)]
    assert set(merged[0]) == set(BLOCKED_CSV_HEADER)

    out = tmp_path / "guardrail_blocked.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BLOCKED_CSV_HEADER)
        writer.writeheader()
        writer.writerows(merged)

    with open(out, newline="") as f:
        read_back = list(csv.DictReader(f))
    assert [(r["index"], r["output_idx"]) for r in read_back] == [("1", "0"), ("1", "1"), ("2", "-1")]
    assert read_back[0]["message"] == "blocked a out0"


def test_importing_the_module_does_not_disable_autograd():
    """``init_script`` turns autograd off process-wide, so it is guarded behind ``__main__``.
    Unguarded, importing any pipeline script would silently kill gradients for the whole process."""
    import importlib

    import torch

    torch.set_grad_enabled(True)
    for name in ("generate", "evaluate", "filter"):
        importlib.reload(importlib.import_module(f"anomalygen.scripts.texture.{name}"))
        assert torch.is_grad_enabled(), f"importing {name}.py disabled autograd"
