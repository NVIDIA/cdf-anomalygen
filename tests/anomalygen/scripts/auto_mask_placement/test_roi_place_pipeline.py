# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the allocate -> pair -> place wrapper.

The wrapper's job is the plumbing the three CLIs otherwise push onto the caller: deriving every
intermediate path from one --output_dir, feeding roi_pair's n_seeds sidecar into roi_place, and
refusing to continue when a stage produced nothing. The stage CLIs have their own tests, so these
drive the composition with stand-in stages passed through ``_run``'s ``stage_main`` argument.
"""

import inspect
import pathlib
import types

import pytest

from anomalygen.scripts.auto_mask_placement import roi_place_pipeline as rpp

_FULL_RUN = {
    "roi_allocate": [("allocation.json", "{}")],
    "roi_pair": [("amp_samples.json", "[]"), ("amp_samples.json.n_seeds", "7")],
    "roi_place": [("testcase.jsonl", "{}\n")],
}


def _args(tmp_path, **over):
    a = dict(
        num_sdg=25,
        defect_desc=tmp_path / "spec.jsonl",
        dataset_dir=tmp_path / "ds",
        output_dir=tmp_path / "amp",
        mode="inference",
        seed=42,
        per_defect_counts=None,
        per_clean_pairing="off",
        dry_run=False,
    )
    a.update(over)
    return types.SimpleNamespace(**a)


def _flags(argv):
    """{flag: value} for the --flag value pairs in an argv list."""
    return {a: argv[i + 1] for i, a in enumerate(argv) if a.startswith("--") and i + 1 < len(argv)}


def _stages(tmp_path, writes=_FULL_RUN, rc=0, seen=None, raises=None):
    """A ``stage_main`` mapping of stand-in stages that record argv and write the given artifacts."""

    def make(stage):
        def _main(argv):
            if seen is not None:
                seen.append((stage, argv))
            if raises is not None:
                raise raises
            out = tmp_path / "amp"
            out.mkdir(parents=True, exist_ok=True)
            for fname, body in writes.get(stage, []):
                (out / fname).write_text(body)
            return rc

        return _main

    return {stage: make(stage) for stage in rpp._STAGE_MAIN}


def test_every_intermediate_path_is_derived_from_output_dir(tmp_path):
    """The caller supplies one directory; allocation/pairs/testcase are the wrapper's business."""
    stages = rpp._stage_argvs(_args(tmp_path), [])
    assert [n for n, _, _ in stages] == ["roi_allocate", "roi_pair", "roi_place"]

    amp = tmp_path / "amp"
    alloc_out = _flags(stages[0][1])["--output_allocation"]
    pair_in, pair_out = _flags(stages[1][1])["--allocation"], _flags(stages[1][1])["--output_pair_path"]

    assert alloc_out == str(amp / "allocation.json")
    assert pair_in == alloc_out, "roi_pair must read exactly what roi_allocate wrote"
    assert _flags(stages[2][1])["--input_pair_path"] == pair_out, "roi_place must read roi_pair's output"
    assert [p for _, _, p in stages] == [amp / "allocation.json", amp / "amp_samples.json", amp / "testcase.jsonl"]


def test_seed_is_shared_by_pair_and_place_but_not_allocate(tmp_path):
    stages = rpp._stage_argvs(_args(tmp_path, seed=43), [])
    assert _flags(stages[1][1])["--seed"] == "43"
    assert _flags(stages[2][1])["--seed"] == "43"
    assert "--seed" not in _flags(stages[0][1])  # roi_allocate has no seed


def test_n_seeds_sidecar_is_read_with_a_safe_fallback(tmp_path):
    """roi_pair computes n_seeds into a sidecar and roi_place needs it back — the step people miss."""
    pairs = tmp_path / "amp_samples.json"
    assert rpp._read_n_seeds(pairs) == "1"  # absent -> roi_place's own default
    (tmp_path / "amp_samples.json.n_seeds").write_text("4\n")
    assert rpp._read_n_seeds(pairs) == "4"
    (tmp_path / "amp_samples.json.n_seeds").write_text("   ")  # blank -> default, never ""
    assert rpp._read_n_seeds(pairs) == "1"


def test_run_feeds_the_sidecar_value_into_roi_place(tmp_path):
    seen = []
    assert rpp._run(_args(tmp_path), [], stage_main=_stages(tmp_path, seen=seen)) == 0
    place_argv = dict(seen)["roi_place"]
    assert _flags(place_argv)["--n_seeds"] == "7", "roi_place must receive the value roi_pair computed"
    assert rpp._N_SEEDS not in " ".join(place_argv), "the placeholder must be resolved"


@pytest.mark.parametrize(
    ("stage_kw", "rc", "stages_run"),
    [
        # A non-zero exit propagates, and nothing downstream runs.
        ({"writes": {}, "rc": 3}, 3, 1),
        # argparse inside a stage calls sys.exit; that must surface as our exit code, not a traceback.
        ({"raises": SystemExit(2)}, 2, 1),
        # roi_pair returning 0 without writing amp_samples.json would otherwise surface later as a
        # vaguer 'file not found' from roi_place.
        ({"writes": {"roi_allocate": [("allocation.json", "{}")]}}, 1, 2),
    ],
    ids=["nonzero-exit", "systemexit", "exited-zero-wrote-nothing"],
)
def test_a_failing_stage_stops_the_chain(tmp_path, stage_kw, rc, stages_run):
    seen = []
    assert rpp._run(_args(tmp_path), [], stage_main=_stages(tmp_path, seen=seen, **stage_kw)) == rc
    assert len(seen) == stages_run, "a failed stage must not let the next one run"


def test_roi_only_does_not_require_a_testcase(tmp_path):
    """--roi_only warms the ROI cache on purpose and writes no testcase.jsonl — not a failure."""
    no_testcase = {k: v for k, v in _FULL_RUN.items() if k != "roi_place"}
    assert rpp._run(_args(tmp_path), ["--roi_only"], stage_main=_stages(tmp_path, writes=no_testcase)) == 0


def test_passthrough_flags_reach_roi_place_only():
    argv = "--num_sdg 5 --defect_desc s.jsonl --dataset_dir ds --output_dir amp".split()
    args, extra = rpp._get_args([*argv, "--", "--refresh_roi", "--n_instances", "3"])
    assert extra == ["--refresh_roi", "--n_instances", "3"]
    assert isinstance(args.output_dir, pathlib.Path), "path args follow the sibling CLIs' convention"
    stages = rpp._stage_argvs(args, extra)
    assert "--refresh_roi" in stages[2][1] and "--n_instances" in stages[2][1]
    assert "--refresh_roi" not in stages[0][1] and "--refresh_roi" not in stages[1][1]


def test_per_defect_counts_only_reaches_allocate(tmp_path):
    stages = rpp._stage_argvs(_args(tmp_path, per_defect_counts='{"A+x": 5}'), [])
    assert _flags(stages[0][1])["--per_defect_counts"] == '{"A+x": 5}'
    assert "--per_defect_counts" not in " ".join(stages[1][1] + stages[2][1])


def test_dry_run_executes_no_stage(tmp_path):
    def _never(argv):
        raise AssertionError("--dry_run must not execute a stage")

    assert rpp._run(_args(tmp_path, dry_run=True), [], stage_main=dict.fromkeys(rpp._STAGE_MAIN, _never)) == 0


def test_stage_main_defaults_to_the_real_clis():
    """The injection point must not drift from the modules it stands in for."""
    from anomalygen.scripts.auto_mask_placement import roi_allocate, roi_pair, roi_place

    assert rpp._STAGE_MAIN == {
        "roi_allocate": roi_allocate.main,
        "roi_pair": roi_pair.main,
        "roi_place": roi_place.main,
    }


@pytest.mark.parametrize("stage", ["roi_allocate", "roi_pair", "roi_place"])
def test_each_stage_main_accepts_an_explicit_argv(stage):
    """_run passes argv positionally; a stage whose main() ignored it would silently read sys.argv."""
    assert "argv" in inspect.signature(rpp._STAGE_MAIN[stage]).parameters


def test_enumerate_amp_masks_rejects_a_path_bearing_defect_type(tmp_path):
    """_enumerate_amp_masks joins the defect type onto each AMP output dir, so a path-bearing value
    would enumerate masks from outside the run tree."""
    from anomalygen.scripts.auto_mask_placement import roi_place

    (tmp_path / "sample").mkdir()
    with pytest.raises(ValueError, match="anomaly type"):
        list(roi_place._enumerate_amp_masks(tmp_path, "../../escape+x"))


def test_a_stage_message_reaches_the_caller(tmp_path, loguru_lines):
    """A stage that exits with a string is telling the caller what to change; the wrapper must not eat it.

    Dropping it leaves only "roi_pair exited 1 — stopping", which names the stage but not the cause,
    and the pipeline is the documented way to run AMP — so the message would be reachable only by
    re-running the stage by hand.
    """
    stages = _stages(tmp_path, raises=SystemExit("pass --per_clean_pairing off for a validation set"))
    assert rpp._run(_args(tmp_path), [], stage_main=stages) == 1
    assert any("--per_clean_pairing off" in line for line in loguru_lines)


def test_a_bare_sys_exit_is_success(tmp_path):
    """``sys.exit()`` carries code None, which is 0 by Python's convention, not a failure."""
    stages = _stages(tmp_path, raises=SystemExit())
    assert rpp._run(_args(tmp_path), [], stage_main=stages) == 1, "stage wrote nothing, so the artifact check fails"

    complete = {k: [(f, "{}") for f, _ in v] for k, v in _FULL_RUN.items()}

    def _exit_clean(argv):
        out = tmp_path / "amp"
        out.mkdir(parents=True, exist_ok=True)
        for stage_files in complete.values():
            for fname, body in stage_files:
                (out / fname).write_text(body)
        raise SystemExit()

    assert rpp._run(_args(tmp_path), [], stage_main=dict.fromkeys(rpp._STAGE_MAIN, _exit_clean)) == 0
