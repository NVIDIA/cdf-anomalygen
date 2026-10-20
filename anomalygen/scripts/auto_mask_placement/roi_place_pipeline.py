# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the three auto-mask-placement CLIs as one step: allocate -> pair -> place.

The three are never useful apart — ``roi_allocate`` writes ``allocation.json``, ``roi_pair`` consumes it
and writes ``amp_samples.json`` plus an ``amp_samples.json.n_seeds`` sidecar, and ``roi_place`` needs
both that pair file *and* the sidecar's value passed back in as ``--n_seeds``. Driving them by hand
means retyping four derived paths and remembering to feed the sidecar through, which is the step people
get wrong: stopping after ``roi_pair`` leaves no ``testcase.jsonl`` and Step 4 has nothing to read.

Each stage is invoked **in process** via its own ``main(argv)``, so there is no second interpreter to
mismatch and a stage's traceback surfaces directly.

This wrapper derives every intermediate path from ``--output_dir`` and fails loudly if a stage does not
produce what the next one needs. Pass-through flags for ``roi_place`` (``--n_instances``,
``--refresh_roi``, ``--roi_only``, generation defaults, ...) are forwarded verbatim after ``--``::

    python -m anomalygen.scripts.auto_mask_placement.roi_place_pipeline \\
        --num_sdg 25 --mode validation --defect_desc <spec> \\
        --dataset_dir <dataset> --output_dir <amp_dir> --seed 42 \\
        -- --refresh_roi
"""

from __future__ import annotations

import argparse
import pathlib

from cosmos_framework.utils import log

from anomalygen.scripts.auto_mask_placement import roi_allocate, roi_pair, roi_place

_ALLOCATION = "allocation.json"
_PAIRS = "amp_samples.json"
_TESTCASE = "testcase.jsonl"
_N_SEEDS = "{n_seeds}"  # placeholder resolved once roi_pair has written its sidecar

# Each stage's entry point, in run order. Every one takes an explicit argv.
_STAGE_MAIN = {"roi_allocate": roi_allocate.main, "roi_pair": roi_pair.main, "roi_place": roi_place.main}


def _stage_argvs(args, passthrough):
    """``[(stage, argv, produces)]`` with every intermediate path derived from --output_dir.

    Returned as data (not run) so the composition is unit-testable without importing the stage modules.
    """
    out = pathlib.Path(args.output_dir)
    allocation, pairs = out / _ALLOCATION, out / _PAIRS
    allocate = [
        "--num_sdg",
        str(args.num_sdg),
        "--mode",
        args.mode,
        "--defect_desc",
        str(args.defect_desc),
        "--dataset_dir",
        str(args.dataset_dir),
        "--output_allocation",
        str(allocation),
    ]
    if args.per_defect_counts:
        allocate += ["--per_defect_counts", args.per_defect_counts]
    pair = [
        "--dataset_dir",
        str(args.dataset_dir),
        "--defect_desc",
        str(args.defect_desc),
        "--allocation",
        str(allocation),
        "--output_pair_path",
        str(pairs),
        "--seed",
        str(args.seed),
        "--per_clean_pairing",
        args.per_clean_pairing,
    ]
    place = [
        "--input_pair_path",
        str(pairs),
        "--defect_desc",
        str(args.defect_desc),
        "--output_dir",
        str(out),
        "--seed",
        str(args.seed),
        "--n_seeds",
        _N_SEEDS,
        *passthrough,
    ]
    return [
        ("roi_allocate", allocate, allocation),
        ("roi_pair", pair, pairs),
        ("roi_place", place, out / _TESTCASE),
    ]


def _read_n_seeds(pairs_path):
    """The ``n_seeds`` roi_pair computed, from its sidecar. Absent/blank -> roi_place's own default."""
    sidecar = pathlib.Path(f"{pairs_path}.n_seeds")
    if not sidecar.is_file():
        return "1"
    return sidecar.read_text().strip() or "1"


def _stage_rc(main, argv) -> int:
    """Run one stage's ``main(argv)`` and return its exit code."""
    try:
        return main(argv) or 0
    except SystemExit as exc:  # NOSONAR — a stage's exit (argparse error, sys.exit) becomes the return code
        # A stage that exits with a MESSAGE puts it in `code` -- argparse does, and so does a
        # stage naming what the caller has to change. Dropping it left only "stage exited 1",
        # which is precisely the information the message existed to supply.
        if exc.code is None:  # bare sys.exit() means success
            return 0
        if isinstance(exc.code, int):
            return exc.code
        log.error(str(exc.code))
        return 1


def _run(args, passthrough, stage_main=None) -> int:
    """Run the three stages. ``stage_main`` maps stage name -> ``main(argv)``; it defaults to the real
    CLIs and exists so callers (and tests) can substitute them without patching module state."""
    stage_main = _STAGE_MAIN if stage_main is None else stage_main
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stages = _stage_argvs(args, passthrough)
    pairs_path = stages[1][2]
    roi_only = "--roi_only" in passthrough

    for stage, argv, produces in stages:
        argv = [a.replace(_N_SEEDS, _read_n_seeds(pairs_path)) for a in argv]
        log.info(f"{stage} {' '.join(argv)}")
        if args.dry_run:
            continue
        rc = _stage_rc(stage_main[stage], argv)
        if rc != 0:
            log.error(f"{stage} exited {rc} — stopping.")
            return rc
        # --roi_only warms the ROI cache on purpose and writes no testcase.jsonl.
        if roi_only and stage == "roi_place":
            continue
        # Fail here rather than let the next stage read a missing file and report something vaguer.
        if not produces.exists():
            log.error(f"{stage} did not produce {produces} — stopping.")
            return 1

    if not args.dry_run and not roi_only:
        log.success(f"done -> {pathlib.Path(args.output_dir) / _TESTCASE}")
    return 0


def _get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="allocate -> pair -> place, as one step.",
        epilog="Flags after -- are forwarded verbatim to roi_place.",
    )
    parser.add_argument("--num_sdg", type=int, required=True)
    parser.add_argument("--defect_desc", type=pathlib.Path, required=True, help="defect_spec JSONL")
    parser.add_argument("--dataset_dir", type=pathlib.Path, required=True, help="Dataset root in the Step 1 layout.")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Where to write allocation.json, amp_samples.json and testcase.jsonl.",
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "validation"],
        default="inference",
        help="Forwarded to roi_allocate: inference (uniform) or validation (proportional, >=1 per defect).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed shared by roi_pair and roi_place.")
    parser.add_argument(
        "--per_defect_counts",
        default=None,
        help="JSON dict of explicit per-defect counts, forwarded to roi_allocate. Only valid with --mode inference.",
    )
    # roi_pair's flag, exposed here because the `--` passthrough reaches roi_place only.
    parser.add_argument(
        "--per_clean_pairing",
        choices=["on", "off"],
        default="off",
        help="on: every clean image used exactly once (one defect type only). "
        "off (default): round-robin, which does not cover every clean image.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print each stage's argv without running it.")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def main(argv=None) -> int:
    args, passthrough = _get_args(argv)
    return _run(args, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
