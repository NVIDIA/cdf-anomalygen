# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for scripts/skill_utility/set_pipeline_vars.sh.

The script is the pipeline's one input-validation choke point: every later step splices ${NAME} into a
path or a command line, so a name that gets through here reaches a shell. These drive it as the skill
does — sourced into a bash process — and read back what it set.
"""

import pathlib
import re
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "skill_utility" / "set_pipeline_vars.sh"

# Every variable the skill's steps refer to. Missing one leaves a step expanding to an empty path.
_EXPECTED_VARS = {
    "NAME", "TASK", "SCRIPTS", "NUM_SDG", "NUM_GPUS", "NUM_SEARCH_RUN", "DATASET_DIR",
    "DEFECT_SPEC", "RECIPE", "VAL_DIR", "GEN_DIR", "OUT", "JOB_NAME", "RUN_DIR", "LOG_DIR",
}  # fmt: skip

# 'NUM_GPUS       = 8    (set)' -> ('NUM_GPUS', '8', 'set')
_PRINTED = re.compile(r"^(\w+)\s+=\s+(.*?)\s+\((set|default|derived)\)$")


def _source(args, epilogue="", execute=False):
    """Source the script with ``args``, then run ``epilogue`` in the same shell."""
    verb = "source" if not execute else "bash"
    script = f'{verb} "{_SCRIPT}" {args}\nrc=$?\n{epilogue}\nexit $rc\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=_REPO)


def _vars_after(args):
    """{name: value} for every variable the script set, read back from the sourcing shell."""
    dump = "\n".join(f'echo "{v}=${{{v}-<unset>}}"' for v in sorted(_EXPECTED_VARS))
    out = _source(args, epilogue=dump)
    assert out.returncode == 0, out.stderr
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line and " " not in line)


def _printed(args):
    """{name: (value, provenance)} parsed from what the script echoes back."""
    out = _source(args)
    assert out.returncode == 0, out.stderr
    got = {}
    for line in out.stdout.splitlines():
        m = _PRINTED.match(line)
        if m:
            got[m.group(1)] = (m.group(2), m.group(3))
    return got


def test_sets_every_variable_the_steps_reference():
    got = _vars_after("--name phone_screen --dataset_dir datasets/phone_screen --num_sdg 25")
    assert _EXPECTED_VARS <= set(got)
    assert "<unset>" not in got.values()


def test_derived_paths_agree_with_each_other():
    """The steps assume these line up — e.g. Step 3 writes RUN_DIR and Step 4 reads the recipe copy
    from it, and the eval real-reference is the same DATASET_DIR training used."""
    v = _vars_after("--name phone_screen --dataset_dir datasets/phone_screen --num_sdg 25")
    assert v["DATASET_DIR"] == "datasets/phone_screen", "DATASET_DIR is whatever the caller passed"
    assert v["DEFECT_SPEC"] == f"{v['DATASET_DIR']}/defect_spec.jsonl"
    assert v["RUN_DIR"] == f"results/anomalygen/{v['NAME']}/{v['JOB_NAME']}"
    assert v["JOB_NAME"] == f"anomalygen_{v['TASK']}"
    assert v["RECIPE"] == f"ag_config/exp_{v['TASK']}_{v['NAME']}.yaml"
    assert v["VAL_DIR"] != v["GEN_DIR"], "validation and generation AMP sets must not share a directory"


def test_optional_inputs_default_to_the_documented_values_and_are_overridable():
    """The defaults are the skill's contract (SKILL.md's Inputs table). num_search_run=0 is a
    documented value — clone-only refinement — not a missing input, so it must survive the override."""
    v = _vars_after("--name x --dataset_dir datasets/x --num_sdg 1")
    assert (v["NUM_GPUS"], v["NUM_SEARCH_RUN"], v["TASK"]) == ("1", "3", "texture_ft")
    v = _vars_after("--name x --dataset_dir datasets/x --num_sdg 1 --num_gpus 8 --num_search_run 0")
    assert (v["NUM_GPUS"], v["NUM_SEARCH_RUN"]) == ("8", "0")


def test_printed_provenance_separates_set_from_default_and_derived():
    """The printed block is what a reviewer reads back off a run log. '(set)' beside a non-default is
    the audit signal — the caller overrode the skill — so it must survive even when the value happens
    to equal the default, or '--num_gpus 1' and 'defaulted to 1' become indistinguishable."""
    plain = _printed("--name x --dataset_dir datasets/x --num_sdg 1")
    assert _EXPECTED_VARS <= set(plain), "a value with no provenance cannot be audited"
    assert plain["NUM_SEARCH_RUN"] == ("3", "default")
    assert plain["TASK"] == ("texture_ft", "default")
    for v in ("RUN_DIR", "RECIPE", "SCRIPTS", "OUT", "JOB_NAME"):
        assert plain[v][1] == "derived", f"{v} should be derived, got {plain[v][1]}"
    # DATASET_DIR is required, so it is always the caller's — never 'derived' from ${NAME} as it once was.
    assert plain["DATASET_DIR"] == ("datasets/x", "set")

    # The case that matters: a run launched with --num_gpus 2 when the skill's default is 1.
    assert _printed("--name x --dataset_dir datasets/x --num_sdg 1 --num_gpus 2")["NUM_GPUS"] == ("2", "set")
    assert _printed("--name x --dataset_dir datasets/x --num_sdg 1 --num_gpus 1")["NUM_GPUS"] == ("1", "set")
    assert _printed("--name x --dataset_dir datasets/x --num_sdg 1 --job_name v2")["JOB_NAME"] == ("v2", "set")


def test_log_dir_points_into_results_not_the_repo_root():
    """Steps redirect their logs to ${LOG_DIR}. A bare './foo.log' is invisible next to the run that
    produced it, so LOG_DIR must never resolve to '.' or an absolute path outside the run tree."""
    log_dir = _vars_after("--name x --dataset_dir datasets/x --num_sdg 1")["LOG_DIR"]
    assert log_dir == "results"
    assert not log_dir.startswith(("/", "."))


@pytest.mark.parametrize(
    "name",
    [
        "x; rm -rf /",  # command separator
        "x$(id)",  # command substitution
        "x`id`",  # legacy substitution
        "x y",  # word split -> later args shift
        "../escape",  # path traversal out of datasets/
        "-flag",  # leads with '-', reads as an option downstream
        "",  # empty -> paths collapse to 'datasets/'
    ],
)
def test_rejects_names_that_would_reach_a_shell(name):
    out = _source(f"--name {name!r} --dataset_dir datasets/x --num_sdg 1")
    assert out.returncode == 64, f"accepted {name!r}"
    assert "invalid --name" in out.stderr


@pytest.mark.parametrize("value", ["3.5", "-1", "1e3", "abc", "", "1 2"])
def test_rejects_non_integer_counts(value):
    out = _source(f"--name x --dataset_dir datasets/x --num_sdg {value!r}")
    assert out.returncode == 64, f"accepted {value!r}"
    assert "must be a non-negative integer" in out.stderr


def test_rejects_zero_gpus():
    """0 passes the integer check but makes torchrun launch no workers at all."""
    out = _source("--name x --dataset_dir datasets/x --num_sdg 1 --num_gpus 0")
    assert out.returncode == 64
    assert "need at least 1" in out.stderr


def test_job_name_defaults_to_the_recipes_and_drives_the_run_dir():
    v = _vars_after("--name x --dataset_dir datasets/x --num_sdg 1")
    assert v["JOB_NAME"] == "anomalygen_texture_ft", "must match the recipe's job_name default"
    assert v["RUN_DIR"].endswith(f"/{v['JOB_NAME']}")


def test_job_name_is_overridable_because_the_recipe_can_rename_it():
    """The documented escape from a resumed checkpoint is a new job_name in the recipe. RUN_DIR's last
    segment IS that job_name, so without this override Steps 4-7 would read the wrong run dir."""
    v = _vars_after("--name x --dataset_dir datasets/x --num_sdg 1 --job_name rerun_v2")
    assert v["JOB_NAME"] == "rerun_v2"
    assert v["RUN_DIR"] == "results/anomalygen/x/rerun_v2"


def test_rejects_a_job_name_that_would_reach_a_shell():
    out = _source("--name x --dataset_dir datasets/x --num_sdg 1 --job_name 'a;b'")
    assert out.returncode == 64 and "invalid --job_name" in out.stderr


def test_a_bad_task_is_reported_as_a_task_not_as_a_job_name():
    """job_name defaults to anomalygen_${task}, so validation order decides which error the user sees."""
    out = _source("--name x --dataset_dir datasets/x --num_sdg 1 --task 'a;b'")
    assert out.returncode == 64
    assert "invalid --task" in out.stderr and "invalid --job_name" not in out.stderr


def test_rejects_a_task_with_no_scripts_directory():
    """${SCRIPTS} must name a real directory; guessing one sends every step to a missing path."""
    out = _source("--name x --dataset_dir datasets/x --num_sdg 1 --task made_up")
    assert out.returncode == 64
    assert "unknown --task" in out.stderr


def test_scripts_dir_for_the_default_task_exists():
    assert (_REPO / _vars_after("--name x --dataset_dir datasets/x --num_sdg 1")["SCRIPTS"] / "train.py").is_file()


def test_rejects_an_unknown_option_instead_of_ignoring_it():
    out = _source("--name x --dataset_dir datasets/x --num_sdg 1 --numgpus 8")
    assert out.returncode == 64 and "unknown option" in out.stderr


def test_required_inputs_are_required():
    assert _source("--dataset_dir datasets/x --num_sdg 1").returncode == 64
    assert _source("--name x --dataset_dir datasets/x").returncode == 64
    assert _source("--name x --num_sdg 1").returncode == 64


def test_dataset_dir_is_required_and_never_derived_from_the_name():
    """`name` is an experiment label; the dataset it trains on is a separate input. Deriving one from
    the other was right only when the two happened to coincide — the exceptional case — and wrong
    silently the rest of the time, which is what a default buys here."""
    out = _source("--name uc1pega-opus5-e2e --num_sdg 25")
    assert out.returncode == 64, "a missing --dataset_dir must stop the caller, not fall back"
    assert "--dataset_dir is required" in out.stderr
    assert "datasets/uc1pega-opus5-e2e" not in out.stdout, "no name-derived path may be offered"

    # The shape the fallback used to get wrong: label and dataset name differ, as they normally do.
    v = _vars_after("--name uc1pega-opus5-e2e --dataset_dir datasets/uc1pega --num_sdg 25")
    assert v["DATASET_DIR"] == "datasets/uc1pega"
    assert v["NAME"] == "uc1pega-opus5-e2e"
    assert v["RUN_DIR"].startswith("results/anomalygen/uc1pega-opus5-e2e/"), "label still names the run"
    assert v["VAL_DIR"] == "datasets/validation_uc1pega-opus5-e2e", "mask sets are named by the label"


def test_dataset_dir_existence_is_left_to_the_step_1_gate():
    """Sourcing stays free of I/O (the header promises it), and Step 1's dataset gate checks the layout
    one step later — including the `seen=0` case that catches a path pointing nowhere."""
    v = _vars_after("--name x --dataset_dir /no/such/dataset --num_sdg 1")
    assert v["DATASET_DIR"] == "/no/such/dataset"
    assert v["DEFECT_SPEC"] == "/no/such/dataset/defect_spec.jsonl"


def test_executing_instead_of_sourcing_is_refused():
    """An executed copy sets the variables in a child that exits — the caller would then run every
    step with empty paths. Fail instead."""
    out = _source("--name x --dataset_dir datasets/x --num_sdg 1", execute=True)
    assert out.returncode == 64
    assert "source this script" in out.stderr


def test_failure_leaves_no_half_set_variables():
    """A rejected input must not leave NAME set from the attempt — the next step would use it."""
    out = _source("--name 'bad name' --dataset_dir datasets/x --num_sdg 1", epilogue='echo "NAME=${NAME-<unset>}"')
    assert out.returncode == 64
    assert "NAME=<unset>" in out.stdout


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_the_header_and_nothing_below_it(flag):
    out = _source(flag)
    assert "validate the AnomalyGen pipeline's inputs" in out.stdout
    assert "SPDX" not in out.stdout, "help leaked the licence preamble"
    assert "_ag_vars()" not in out.stdout, "help ran past the header into the code"


def test_does_not_leave_its_helper_function_behind():
    out = _source("--name x --dataset_dir datasets/x --num_sdg 1", epilogue='declare -F _ag_vars || echo "cleaned"')
    assert "cleaned" in out.stdout
