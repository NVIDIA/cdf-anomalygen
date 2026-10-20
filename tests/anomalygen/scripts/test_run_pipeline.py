# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the config-driven entrypoint.

The entrypoint's own logic is the config contract, the stage argv it builds, and the export mapping;
the five sub-CLIs have their own tests. So these drive ``load_config``/``plan_export`` directly and
substitute the stage runners rather than launching subprocesses.
"""

import csv
import dataclasses
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import typing

import pytest
import yaml

from anomalygen import catalogue
from anomalygen.scripts import run_pipeline as rp

_DEFECT_TYPES = ("A+x", "A+y", "A+z")


def _write_spec(ds, routes):
    """defect_spec.jsonl with a spatial_dependency per type; None omits the key entirely."""
    lines = []
    for defect_type, route in routes.items():
        row = {"defect_type": defect_type}
        if route is not None:
            row["spatial_dependency"] = route
        lines.append(json.dumps(row))
    (ds / "defect_spec.jsonl").write_text("\n".join(lines) + "\n")


def _dataset(tmp_path):
    """The paths load_config now stats: a dataset with a defect_spec, a recipe and a checkpoint."""
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    _write_spec(ds, {t: "cad_tht" for t in _DEFECT_TYPES})
    (tmp_path / "r.yaml").touch()
    (tmp_path / "ckpt.pt").touch()
    return {"input": str(ds), "recipe": str(tmp_path / "r.yaml"), "checkpoint": str(tmp_path / "ckpt.pt")}


def _touch(path):
    """A file at ``path``. base_checkpoint is a checked path now, so a test naming one must have it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def _cfg(tmp_path, **over):
    body = {"output": "out", "num_sdg": 4, **_dataset(tmp_path)}
    body.update(over)
    for k in [k for k, v in body.items() if v is None]:
        del body[k]
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def _stub_stage_run(monkeypatch, seen=None, rows=None):
    """Stand in for `run`, doing the part of a mask stage the gates after it depend on.

    stage_amp now refuses a testcase with fewer rows than the run asked for -- roi_place drops a
    sample it cannot place and still exits 0 -- so a stub that only records argv trips that gate.
    Writing the file a real roi_place would keeps these tests about what they are about, and keeps
    the gate honest rather than stubbed out.
    """

    def _run(cmd, log_path, env=None):
        argv = [str(c) for c in cmd]
        if seen is not None:
            seen.append(cmd)
        if "--output_dir" not in argv or "--num_sdg" not in argv:
            return
        out = pathlib.Path(argv[argv.index("--output_dir") + 1])
        n = int(argv[argv.index("--num_sdg") + 1])
        out.mkdir(parents=True, exist_ok=True)
        written = rows if rows is not None else [{"image_filename": f"clean_{i}.png"} for i in range(n)]
        out.joinpath("testcase.jsonl").write_text("".join(json.dumps(r) + "\n" for r in written))

    monkeypatch.setattr(rp, "run", _run)
    return seen


def _flags(argv):
    return {a: argv[i + 1] for i, a in enumerate(argv) if str(a).startswith("--") and i + 1 < len(argv)}


# ── config ────────────────────────────────────────────────────────────────────


def test_the_derived_count_is_not_recorded_as_a_setting(tmp_path):
    """defect_counts is computed from per_defect_counts, so it must not appear in the resolved
    config as though it were something the caller could set."""
    cfg = rp.load_config(_cfg(tmp_path, per_defect_counts={"A+x": 4}))
    assert cfg.defect_counts == {"A+x": 4}
    assert "defect_counts" not in dataclasses.asdict(cfg)


def test_required_keys_are_the_fields_without_a_default():
    """Requiredness is the dataclass's own notion, so the two cannot drift apart."""
    assert rp._required_names() == ["input", "output", "recipe"]


def test_a_misspelled_key_is_an_error_not_a_silent_default(tmp_path):
    """`guidence: 3.0` would otherwise run at the default guidance and look entirely successful."""
    with pytest.raises(SystemExit, match="guidence"):
        rp.load_config(_cfg(tmp_path, guidence=3.0))


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"input": None}, "input"),
        ({"stages": ["amp", "polish"]}, "polish"),
        ({"stages": ["generate"], "checkpoint": None}, "checkpoint"),  # generate cannot run without one
        ({"stages": ["amp"], "num_sdg": None}, "num_sdg"),
        ({"stages": ["amg"], "num_sdg": None}, "num_sdg"),
        ({"output_naming": "sourced"}, "output_naming"),
        ({"mode": "train"}, "mode"),
        ({"amp_per_clean_pairing": "yes"}, "amp_per_clean_pairing"),
    ],
    ids=[
        "missing-required",
        "unknown-stage",
        "generate-needs-ckpt",
        "amp-needs-num_sdg",
        "amg-needs-num_sdg",
        "naming",
        "mode",
        "pairing",
    ],
)
def test_config_rejects_what_would_fail_late(tmp_path, over, match):
    if over.get("stages") == ["amg"]:
        over = {**over, **_amg_keys(tmp_path)}
    with pytest.raises(SystemExit, match=match):
        rp.load_config(_cfg(tmp_path, **over))


def test_amp_and_amg_together_is_an_error_naming_both(tmp_path):
    """Two mask stages would both write a testcase.jsonl and generate would read one of them by
    accident of path resolution."""
    with pytest.raises(SystemExit, match=r"amp.*amg|amg.*amp"):
        rp.load_config(_cfg(tmp_path, stages=["amp", "amg", "generate"]))


def test_the_default_stage_list_picks_amp(tmp_path):
    cfg = rp.load_config(_cfg(tmp_path))
    assert cfg.stages == list(rp.DEFAULT_STAGES)
    assert "amg" not in cfg.stages and "amp" in cfg.stages


def _amg_keys(tmp_path):
    """The three keys the amg stage requires, pointing at a registry file that exists."""
    reg = tmp_path / "endpoints.yaml"
    reg.write_text(yaml.safe_dump([{"model": "m", "base_url": "http://x/v1", "key": "EMPTY"}]))
    return {"amg_gen_model": "m", "amg_judge_model": "m", "amg_model_registry": str(reg)}


@pytest.mark.parametrize("missing", ["amg_gen_model", "amg_judge_model", "amg_model_registry"])
def test_amg_requires_its_three_keys_only_when_selected(tmp_path, missing):
    keys = _amg_keys(tmp_path)
    keys[missing] = None
    # Selected: the absent key is named.
    with pytest.raises(SystemExit, match=missing):
        rp.load_config(_cfg(tmp_path, stages=["amg"], **keys))
    # Not selected: the same config loads.
    rp.load_config(_cfg(tmp_path, stages=["amp"], **keys))


def test_amg_serve_must_name_a_known_recipe(tmp_path):
    with pytest.raises(SystemExit, match=r"gemma-4-31b-it"):  # the available list is in the message
        rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="llama-99b"))


def test_amg_serve_defaults_the_model_names_and_makes_the_registry_optional(tmp_path):
    _mask_dirs(tmp_path, "A+x")
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it"))
    assert cfg.amg_gen_model == cfg.amg_judge_model == "nvidia/google/gemma-4-31b-it"
    assert cfg.amg_model_registry is None


def test_an_explicit_judge_survives_amg_serve(tmp_path):
    _mask_dirs(tmp_path, "A+x")
    keys = _amg_keys(tmp_path)  # registry with model "m"
    cfg = rp.load_config(
        _cfg(
            tmp_path,
            stages=["amg"],
            amg_serve="gemma-4-31b-it",
            amg_judge_model="m",
            amg_model_registry=keys["amg_model_registry"],
        )
    )
    assert cfg.amg_gen_model == "nvidia/google/gemma-4-31b-it" and cfg.amg_judge_model == "m"


def test_amg_serve_without_the_amg_stage_is_logged_not_refused(tmp_path, loguru_lines):
    cfg = rp.load_config(_cfg(tmp_path, stages=["amp"], amg_serve="gemma-4-31b-it"))
    assert rp._serve_recipe(cfg) is None
    assert any("amg_serve" in line for line in loguru_lines)


def test_without_amg_serve_the_registry_is_still_required(tmp_path):
    with pytest.raises(SystemExit, match="amg_model_registry"):
        rp.load_config(_cfg(tmp_path, stages=["amg"], amg_gen_model="m", amg_judge_model="m"))


def test_amp_per_clean_pairing_on_is_refused_under_amg(tmp_path):
    """AMG has no pairing step; accepting the key would promise coverage it cannot deliver."""
    with pytest.raises(SystemExit, match="amp_per_clean_pairing"):
        rp.load_config(_cfg(tmp_path, stages=["amg"], amp_per_clean_pairing="on", **_amg_keys(tmp_path)))


def test_an_explicit_seed_under_amg_is_logged_not_refused(tmp_path, loguru_lines):
    """A config switches backend by editing `stages` alone; a seed left over from the amp run must
    not fail it, but the reader must learn it has no effect."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amp_seed=7, **_amg_keys(tmp_path)))
    assert cfg.amp_seed == 7
    assert any("amp_seed" in line and "amg" in line for line in loguru_lines)
    assert not any("WARNING" in line and "amp_seed" in line for line in loguru_lines)


def test_on_and_off_survive_yaml_parsing_them_as_booleans(tmp_path):
    """YAML 1.1 turns bare `on` into True; requiring quotes would be a trap with no visible cause."""
    assert rp.load_config(_cfg(tmp_path, amp_per_clean_pairing=True)).amp_per_clean_pairing == "on"
    assert rp.load_config(_cfg(tmp_path, amp_per_clean_pairing=False)).amp_per_clean_pairing == "off"
    assert rp.load_config(_cfg(tmp_path, amp_per_clean_pairing="on")).amp_per_clean_pairing == "on"


def test_explicit_counts_are_refused_for_a_validation_set(tmp_path):
    """roi_allocate rejects --per_defect_counts outside --mode inference, four stage-frames deep."""
    with pytest.raises(SystemExit, match="mode: inference"):
        rp.load_config(_cfg(tmp_path, mode="validation", per_defect_counts={"A+x": 4}))


def test_num_sdg_disagreeing_with_explicit_counts_warns_and_adopts_the_counts(tmp_path, loguru_lines):
    """Same rule as roi_allocate -- the counts win -- but said before any stage runs rather than
    from inside the AMP log, which is where it would otherwise be buried."""
    cfg = rp.load_config(_cfg(tmp_path, num_sdg=40, per_defect_counts={"A+x": 3}))
    assert any("per_defect_counts sums to 3" in line for line in loguru_lines)
    assert cfg.num_sdg == 3, "the resolved config must record what is generated, not what was asked"


def test_matching_counts_and_num_sdg_do_not_warn(tmp_path, loguru_lines):
    rp.load_config(_cfg(tmp_path, num_sdg=4, per_defect_counts={"A+x": 3, "A+y": 1}))
    assert not [line for line in loguru_lines if "per_defect_counts sums" in line]


def test_explicit_counts_may_state_the_total_by_themselves(tmp_path):
    cfg = rp.load_config(_cfg(tmp_path, num_sdg=None, per_defect_counts={"A+x": 3, "A+y": 1}))
    assert cfg.num_sdg == 4


def test_num_sdg_is_coerced_from_a_yaml_quoted_string(tmp_path):
    """A YAML-quoted "40" must not fail the row-count gate's int comparison forever."""
    cfg = rp.load_config(_cfg(tmp_path, num_sdg="4"))
    assert cfg.num_sdg == 4 and isinstance(cfg.num_sdg, int)

    with pytest.raises(SystemExit, match="num_sdg"):
        rp.load_config(_cfg(tmp_path, num_sdg="four"))


def test_num_gpus_is_coerced_from_a_yaml_quoted_string(tmp_path):
    """Same trap as num_sdg: a YAML-quoted "2" must not fail check_gpus's int comparison later."""
    cfg = rp.load_config(_cfg(tmp_path, num_gpus="2"))
    assert cfg.num_gpus == 2 and isinstance(cfg.num_gpus, int)

    with pytest.raises(SystemExit, match="num_gpus"):
        rp.load_config(_cfg(tmp_path, num_gpus="two"))


def test_num_gpus_null_is_a_systemexit_not_a_typeerror(tmp_path):
    """int(None) raises TypeError, not ValueError; the coercion must catch both."""
    body = {"output": "out", "num_sdg": 4, "num_gpus": None, **_dataset(tmp_path)}
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(SystemExit, match="num_gpus"):
        rp.load_config(path)


def test_a_defect_type_absent_from_defect_spec_is_an_error(tmp_path):
    """roi_allocate warns about an unrecognised type and allocates it zero images, so a typo runs
    the whole of AMP and produces nothing while reporting success."""
    with pytest.raises(SystemExit, match=r"A\+bridg"):
        rp.load_config(_cfg(tmp_path, per_defect_counts={"A+bridg": 4}))


@pytest.mark.parametrize("key", ["input", "recipe", "checkpoint"])
def test_a_path_that_does_not_exist_is_caught_before_a_stage_pays_to_find_out(tmp_path, key):
    with pytest.raises(SystemExit, match=key):
        rp.load_config(_cfg(tmp_path, **{key: str(tmp_path / "nope")}))


def test_a_missing_checkpoint_is_ignored_when_no_stage_needs_one(tmp_path):
    """An amp-only or export-only run must not demand a checkpoint it will never load."""
    rp.load_config(_cfg(tmp_path, stages=["amp"], checkpoint=str(tmp_path / "nope")))


def test_stages_are_run_in_pipeline_order_however_they_are_written(tmp_path):
    cfg = rp.load_config(_cfg(tmp_path, stages=["export", "amp", "amp"]))
    assert cfg.stages == ["amp", "export"]


def test_a_single_stage_may_be_written_as_a_string(tmp_path):
    assert rp.load_config(_cfg(tmp_path, stages="amp")).stages == ["amp"]


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        (None, None),  # every type in defect_spec, allocated by roi_allocate
        ({"A+x": 3, "A+y": 1}, {"A+x": 3, "A+y": 1}),
        ({"A+x": "3"}, {"A+x": 3}),  # YAML quoting a number must not reach roi_allocate as a string
    ],
    ids=["omitted", "counts", "stringified-count"],
)
def test_per_defect_counts_is_the_only_type_selection(counts, expected):
    assert rp.normalise_per_defect_counts(counts) == expected


@pytest.mark.parametrize(("bad", "match"), [(["A+x"], "mapping"), ({}, "omit it")], ids=["list", "empty"])
def test_per_defect_counts_rejects_what_roi_allocate_could_not_use(bad, match):
    with pytest.raises(SystemExit, match=match):
        rp.normalise_per_defect_counts(bad)


def _mask_dirs(tmp_path, *types):
    """`<ds>/<texture>/mask/<defect>/` for each 'texture+defect', with one mask file in it."""
    for t in types:
        texture, defect = t.split("+", 1)
        d = tmp_path / "ds" / texture / "mask" / defect
        d.mkdir(parents=True, exist_ok=True)
        (d / "a_mask.png").touch()


def test_an_amg_run_needs_no_defect_spec(tmp_path):
    """AMG never reads defect_spec.jsonl; demanding it would fail the datasets the backend is for."""
    keys = _amg_keys(tmp_path)
    config = _cfg(tmp_path, stages=["amg"], **keys)  # lays down the default dataset, spec included
    (tmp_path / "ds" / "defect_spec.jsonl").unlink()
    _mask_dirs(tmp_path, "A+x")
    cfg = rp.load_config(config)
    assert cfg.stages == ["amg"]


def test_amg_checks_explicit_counts_against_the_mask_directories(tmp_path):
    keys = _amg_keys(tmp_path)
    _mask_dirs(tmp_path, "A+x", "A+y")
    rp.load_config(_cfg(tmp_path, stages=["amg"], per_defect_counts={"A+x": 4}, **keys))
    with pytest.raises(SystemExit, match=r"A\+zz.*Available.*A\+x"):
        rp.load_config(_cfg(tmp_path, stages=["amg"], per_defect_counts={"A+zz": 4}, **keys))


def test_mask_types_on_disk_lists_every_texture_defect_pair(tmp_path):
    _mask_dirs(tmp_path, "B+q", "A+y", "A+x")
    (tmp_path / "ds" / "A" / "clean_image").mkdir(parents=True)  # not a mask dir, not a type
    assert rp._mask_types_on_disk(str(tmp_path / "ds")) == ["A+x", "A+y", "B+q"]


# ── layout ────────────────────────────────────────────────────────────────────


def test_testcase_follows_the_selected_mask_stage(tmp_path):
    amp = rp.load_config(_cfg(tmp_path, stages=["amp"], output=str(tmp_path / "o")))
    amg = rp.load_config(_cfg(tmp_path, stages=["amg"], output=str(tmp_path / "o"), **_amg_keys(tmp_path)))
    assert rp.layout(amp)["testcase"] == tmp_path / "o" / "amp" / "testcase.jsonl"
    assert rp.layout(amg)["testcase"] == tmp_path / "o" / "amg" / "testcase.jsonl"
    assert rp.layout(amg)["amg_cache"] == tmp_path / "o" / "amg_cache"


def test_a_resumed_tail_reads_whichever_testcase_exists(tmp_path):
    out = tmp_path / "o"
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(out)))
    assert rp.layout(cfg)["testcase"] == out / "amp" / "testcase.jsonl", "nothing on disk: the classic path"
    (out / "amg").mkdir(parents=True)
    (out / "amg" / "testcase.jsonl").write_text("{}\n")
    assert rp.layout(cfg)["testcase"] == out / "amg" / "testcase.jsonl"


def test_two_testcases_on_disk_is_an_error_asking_for_the_stage(tmp_path):
    out = tmp_path / "o"
    for backend in ("amp", "amg"):
        (out / backend).mkdir(parents=True)
        (out / backend / "testcase.jsonl").write_text("{}\n")
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], output=str(out)))
    with pytest.raises(SystemExit, match="stages"):
        rp.layout(cfg)


def test_generate_depends_on_either_mask_stage():
    assert rp._skipped_dependencies("generate", ["amg", "generate"]) == []
    assert rp._skipped_dependencies("generate", ["amp", "generate"]) == []
    assert rp._skipped_dependencies("generate", ["generate"]) == ["amp|amg"]
    assert rp._skipped_dependencies("refine", ["refine"]) == ["amp|amg", "generate", "evaluate"]


# ── stage argv ────────────────────────────────────────────────────────────────


def test_amp_argv_carries_the_pairing_and_defect_selection(tmp_path, monkeypatch):
    cfg = rp.load_config(
        _cfg(tmp_path, per_defect_counts={"A+x": 4}, amp_per_clean_pairing="on", guidance=3.0, num_steps=35)
    )
    seen = _stub_stage_run(monkeypatch, [])
    rp.stage_amp(cfg, rp.layout(cfg))

    argv = [str(c) for c in seen[0]]
    flags = _flags(argv)
    assert flags["--per_clean_pairing"] == "on"
    assert json.loads(flags["--per_defect_counts"]) == {"A+x": 4}
    # The sampling knobs belong to roi_place, which only sees what follows `--`.
    assert argv.index("--") < argv.index("--guidance")
    assert flags["--num_steps"] == "35"


def test_generation_knobs_absent_from_the_config_are_not_forced(tmp_path, monkeypatch):
    """Omitting `guidance` must leave the shared texture default in force, not pass None."""
    cfg = rp.load_config(_cfg(tmp_path))
    seen = []
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.append(cmd))
    rp.stage_amp(cfg, rp.layout(cfg))
    argv = [str(c) for c in seen[0]]
    assert "--guidance" not in argv and "--" not in argv


def test_generate_pins_the_framework_output_root_inside_the_run(tmp_path, monkeypatch):
    """Generation and refinement must resolve the framework's job paths to the same place.

    quality_refine sets this for the generate.py children it launches, so a direct generate that
    inherited the ambient value instead would put the two stages' framework-side artifacts in
    different roots -- and unset, that root is a shared /tmp path outside the run tree.
    """
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")

    seen = {}
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.update(cmd=cmd, env=env))
    rp.stage_generate(cfg, paths)
    assert seen["env"]["IMAGINAIRE_OUTPUT_ROOT"] == str(paths["imaginaire"])
    assert "--guardrail" in [str(c) for c in seen["cmd"]]


def test_guardrail_false_reaches_generate_as_the_negative_flag(tmp_path, monkeypatch):
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], guardrail=False, output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")
    seen = {}
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.update(cmd=[str(c) for c in cmd]))
    rp.stage_generate(cfg, paths)
    assert "--no-guardrail" in seen["cmd"] and "--guardrail" not in seen["cmd"]


def test_a_stage_run_alone_says_which_artifact_it_is_missing(tmp_path):
    """Resuming into `generate` without an AMP run must name testcase.jsonl, not fail inside torchrun."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], output=str(tmp_path / "out")))
    with pytest.raises(SystemExit, match="testcase.jsonl"):
        rp.stage_generate(cfg, rp.layout(cfg))


def test_refine_is_told_the_same_sampling_setup_as_generate(tmp_path, monkeypatch):
    """quality_refine forwarded neither, so a --no-guardrail batch was searched with the guardrail on
    and an explicit base_checkpoint was ignored -- the rounds differed from the bucket they are
    selected against."""
    cfg = rp.load_config(
        _cfg(
            tmp_path,
            output=str(tmp_path / "out"),
            guardrail=False,
            base_seed=7,
            base_checkpoint=str(_touch(tmp_path / "base.pt")),
        )
    )
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    paths["kpi"].write_text("{}")

    seen = []
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.append([str(c) for c in cmd]))
    rp.stage_refine(cfg, paths)
    argv = seen[0]
    assert "--no-guardrail" in argv and "--guardrail" not in argv
    assert _flags(argv)["--base_seed"] == "7"
    # The path the preflight checked, not the raw config string: prepare, the preflight and the CLI
    # all have to name one file, and generate.py would otherwise abspath its own default against the
    # working directory.
    assert _flags(argv)["--base_checkpoint"] == str(tmp_path / "base.pt")


def test_both_stages_are_given_the_base_checkpoint_the_preflight_checked(tmp_path, monkeypatch):
    """generate.py's own default is relative and is abspath'd against the WORKING DIRECTORY, so an
    orchestrator started outside the repository preflighted one path and loaded another -- or loaded
    whatever tree happened to sit at <cwd>/checkpoints, silently."""
    cfg = rp.load_config(_cfg(tmp_path, output=str(tmp_path / "out")))  # no base_checkpoint set
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    paths["kpi"].write_text("{}")

    checked = {t for label, t, _ in rp.required_weights(cfg) if "base checkpoint" in label}
    seen = []
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.append([str(c) for c in cmd]))
    rp.stage_generate(cfg, paths)
    # what the stub generate did not write, and what take_bucket cleared on its way in
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    paths["kpi"].write_text("{}")
    rp.stage_refine(cfg, paths)

    for argv in seen:
        passed = _flags(argv)["--base_checkpoint"]
        assert pathlib.Path(passed).is_absolute(), passed
        assert pathlib.Path(passed) in checked, f"{passed} is not what the preflight checked ({checked})"


def test_a_model_size_the_loader_will_reject_is_refused_before_the_weights_are_fetched(tmp_path):
    """`model_size` is free text. `Nano` used to preflight as base-nano, provision 30 GB and then die
    inside generate.py hours later, which checks the same value against MODEL_SIZES."""
    (tmp_path / "r.yaml").write_text(yaml.safe_dump({"model_size": "Huge"}))
    with pytest.raises(SystemExit, match="not one of"):
        rp._model_size(tmp_path / "r.yaml")


@pytest.mark.parametrize("written", ["nano", "Nano", "NANO", " edge "])
def test_one_spelling_reaches_every_reader_of_the_size(tmp_path, monkeypatch, written):
    """The readers disagreed: base_group lower-cased, the caption-tokenizer branch compared raw. So
    `model_size: Edge` asked for the Edge base checkpoint and the NANO caption tokenizer."""
    _weights(tmp_path, monkeypatch)
    (tmp_path / "r.yaml").write_text(yaml.safe_dump({"model_size": written}))
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"]))
    groups = {g for _, _, g in rp.required_weights(cfg)}

    expected_base = f"base-{written.strip().lower()}"
    tokenizer = "edge-processor" if written.strip().lower() == "edge" else "qwen-tokenizer"
    assert expected_base in groups
    assert tokenizer in groups, f"{written!r} took the wrong caption-tokenizer branch"
    assert groups <= set(catalogue.GROUPS)


def test_an_empty_directory_is_not_a_provisioned_checkpoint(tmp_path, monkeypatch):
    """A k8s subPath volume is created empty by the kubelet. It passed the preflight, kept base-nano
    out of the provisioning request, and generate.py then died in load_for_inference."""
    root = _weights(tmp_path, monkeypatch, hf=_TOKENIZER + _GUARDRAIL, files=_GEN_FILES)
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"]))
    assert rp.missing_weights(cfg) == []

    base = root / "checkpoints" / "Cosmos3-Nano"
    for child in list(base.iterdir()):
        child.unlink() if child.is_file() else shutil.rmtree(child)
    assert base.is_dir() and not any(base.iterdir())

    assert any("base checkpoint" in m for m in rp.missing_weights(cfg))
    assert "base-nano" in rp.missing_groups(cfg)


# The VALUES each stage puts on the command line, not only the flag names.
# test_every_flag_the_entrypoint_emits_exists_on_the_cli_it_targets checks that the names exist on
# the CLI they are aimed at, which is a different property: with names alone, swapping paths["sdg"]
# for cfg.input, or paths["kpi"] for anything else, passes. evaluate had no test at all -- the three
# flags it emits appeared nowhere in this file -- and refine asserted three of its fourteen values.


def _argv_of(stage, cfg, paths, monkeypatch):
    """The argv a stage would run, with the stage's own preconditions satisfied."""
    seen = []
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.append([str(c) for c in cmd]))
    rp._RUNNERS[stage](cfg, paths)
    assert len(seen) == 1, f"{stage} ran {len(seen)} commands"
    return seen[0]


def test_evaluate_scores_this_run_against_the_dataset_it_came_from(tmp_path, monkeypatch):
    """Every value here is a path that could plausibly be another one, and none was asserted."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["evaluate"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)

    flags = _flags(_argv_of("evaluate", cfg, paths, monkeypatch))
    assert flags["--gen_root"] == str(paths["sdg"]), "scored a bucket other than the one generated"
    assert flags["--real_root"] == cfg.input, "compared against something other than the real images"
    assert flags["--recipe"] == cfg.recipe
    assert flags["--output_file"] == str(paths["kpi"]), "wrote the KPI where refine will not find it"


def test_generate_is_told_the_checkpoint_recipe_and_testcase_of_this_run(tmp_path, monkeypatch):
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], output=str(tmp_path / "out"), base_seed=11))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")

    argv = _argv_of("generate", cfg, paths, monkeypatch)
    # childproc.torchrun joins this one with '=', so it is not a (flag, value) pair like the rest.
    assert f"--nproc_per_node={cfg.num_gpus}" in argv
    flags = _flags(argv)
    assert flags["--checkpoint"] == cfg.checkpoint
    assert flags["--recipe"] == cfg.recipe
    assert flags["--input_data_path"] == str(paths["testcase"]), "generated from a testcase this run did not write"
    assert flags["--output_dir"] == str(paths["sdg"])
    assert flags["--base_seed"] == "11"


def test_refine_is_told_every_path_it_reads_and_writes(tmp_path, monkeypatch):
    """Fourteen values, of which three were asserted. The rest are the buckets the search reads,
    writes and is scored against -- a swap among them is silent and ships the wrong images."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["refine"], output=str(tmp_path / "out"), num_search_run=4))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    paths["kpi"].write_text("{}")

    flags = _flags(_argv_of("refine", cfg, paths, monkeypatch))
    assert flags["--base_testcase"] == str(paths["testcase"])
    assert flags["--original"] == str(paths["sdg"])
    assert flags["--original_kpi"] == str(paths["kpi"])
    assert flags["--rounds_dir"] == str(paths["rounds"])
    assert flags["--output"] == str(paths["searched"])
    assert flags["--final_kpi"] == str(paths["final_kpi"])
    assert flags["--real_root"] == cfg.input
    assert flags["--output_root"] == str(paths["imaginaire"])
    assert flags["--num_search_run"] == "4"
    assert flags["--num_gpus"] == str(cfg.num_gpus)


def test_no_stage_confuses_the_raw_bucket_with_the_refined_one(tmp_path, monkeypatch):
    """The pair most worth pinning: evaluate scores sdg/ and refine SELECTS into searched/ from it.
    Either one pointed at the other reads as a working run and ships the wrong images."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["evaluate", "refine"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    paths["kpi"].write_text("{}")

    ev = _flags(_argv_of("evaluate", cfg, paths, monkeypatch))
    rf = _flags(_argv_of("refine", cfg, paths, monkeypatch))
    assert ev["--gen_root"] != str(paths["searched"])
    assert rf["--original"] == ev["--gen_root"], "refine searched a bucket evaluate did not score"
    assert rf["--output"] != rf["--original"], "refine would select into the bucket it reads from"


# ── export ────────────────────────────────────────────────────────────────────


def _rows(pairs):
    return [{"output_filename": o, "image_filename": i} for o, i in pairs]


def test_export_restores_the_source_name():
    rows = _rows([("SolderPin+bridge_00000.png", "/ds/clean_image/A1__T101__3.png")])
    assert rp.plan_export(rows, "source") == {"A1__T101__3.png": "SolderPin+bridge_00000.png"}


def test_export_keeps_the_generated_name_when_asked():
    rows = _rows([("SolderPin+bridge_00000.png", "/ds/clean_image/A1__T101__3.png")])
    assert rp.plan_export(rows, "indexed") == {"SolderPin+bridge_00000.png": "SolderPin+bridge_00000.png"}


def test_two_outputs_for_one_source_is_an_error_naming_both():
    """Silently renaming past a collision would drop a generated image and leave the caller
    compositing whichever was written last."""
    rows = _rows([("A+x_00000.png", "/ds/c/j1.png"), ("A+y_00000.png", "/ds/c/j1.png")])
    with pytest.raises(SystemExit) as exc:
        rp.plan_export(rows, "source")
    assert "A+x_00000.png" in str(exc.value) and "A+y_00000.png" in str(exc.value)
    assert "indexed" in str(exc.value), "the error must name the way out"


def test_a_collision_report_stays_readable_with_many_clashes():
    rows = _rows([(f"A+{t}_{i:05d}.png", f"/ds/c/j{i}.png") for i in range(20) for t in ("x", "y")])
    with pytest.raises(SystemExit, match=r"20 source image"):
        rp.plan_export(rows, "source")


@pytest.mark.parametrize(
    ("naming", "second_input", "targets"),
    [
        ("source", "/ds/A/clean_image/crop_000004.jpg", ("crop_000123.png", "crop_000004.png")),
        ("indexed", "/ds/A/clean_image/crop_000004.jpg", ("A+y_00000.png", "A+x_00000.png")),
        ("indexed", "/ds/A/clean_image/crop_000123.jpg", ("A+y_00000.png", "A+x_00000.png")),
    ],
    ids=["source", "indexed", "indexed-reused-input"],
)
def test_export_copies_the_files_and_records_the_mapping(tmp_path, naming, second_input, targets):
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out"), output_naming=naming))
    paths = rp.layout(cfg)
    recon = paths["sdg"] / "reconstructed_image"
    recon.mkdir(parents=True)
    pairs = [("A+y_00000.png", "/ds/A/clean_image/crop_000123.jpg"), ("A+x_00000.png", second_input)]
    for output, _ in pairs:
        (recon / output).write_bytes(output.encode())
    with open(paths["sdg"] / rp._GEN_CSV, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["output_filename", "image_filename"])
        w.writeheader()
        w.writerows(_rows(pairs))

    rp.stage_export(cfg, paths)
    for target, (output, _) in zip(targets, pairs):
        assert (paths["export"] / target).read_bytes() == output.encode()
    manifest = json.loads((paths["export"] / "export_manifest.json").read_text())
    assert manifest["images"] == {targets[0]: "A+y_00000.png", targets[1]: "A+x_00000.png"}
    assert manifest["input_images"] == {targets[0]: "/ds/A/clean_image/crop_000123.jpg", targets[1]: second_input}


def test_a_missing_csv_names_the_stage_that_writes_the_bucket_being_read(tmp_path):
    """A half-finished refine leaves searched/ without a CSV; telling the caller to run generate
    sends them to a stage that cannot produce it."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    (paths["searched"] / "reconstructed_image").mkdir(parents=True)
    with pytest.raises(SystemExit, match="'refine'"):
        rp.stage_export(cfg, paths)

    # The raw bucket still points at generate.
    shutil.rmtree(paths["searched"])
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    with pytest.raises(SystemExit, match="'generate'"):
        rp.stage_export(cfg, paths)


def test_export_reads_the_refined_bucket_when_refine_was_selected(tmp_path):
    """After refinement the searched bucket is the run's result; exporting the raw one would quietly
    ship the images the search rejected."""
    cfg = rp.load_config(_cfg(tmp_path, output=str(tmp_path / "out")))  # default stages include refine
    paths = rp.layout(cfg)
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    (paths["searched"] / "reconstructed_image").mkdir(parents=True)
    assert rp.gen_root(cfg, paths) == paths["searched"]


def _refined_run(paths, run="r1"):
    """The state a completed [.., generate, refine, ..] invocation leaves: both buckets, both marked.

    Built through the same helpers the stages use rather than by hand, so a test cannot describe a
    tree no run can produce -- which is exactly how the version of this that aged directories with
    os.utime passed against a check that could never fire.
    """
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True, exist_ok=True)
    rp._stamp(paths["sdg"], run)
    (paths["searched"] / "reconstructed_image").mkdir(parents=True, exist_ok=True)
    rp._stamp(paths["searched"], run, source=run)


def test_a_resumed_tail_still_finds_a_refined_bucket_on_disk(tmp_path):
    """stages [export] over an output a previous run refined: the selection says nothing, so the
    markers the previous run left have to."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    rp._stamp(paths["sdg"], "r1")
    assert rp.gen_root(cfg, paths) == paths["sdg"]

    (paths["searched"] / "reconstructed_image").mkdir(parents=True)
    rp._stamp(paths["searched"], "r1", source="r1")
    assert rp.gen_root(cfg, paths) == paths["searched"]


def test_a_refined_bucket_from_another_generate_run_is_refused(tmp_path):
    """Run 1 refines. Run 2 in the same output, with a different amp_seed and no refine stage,
    rewrites sdg/ -- and export used to still read searched/, shipping run 1's images under run 2's
    recorded config, with nothing to say so.

    Driven through take_bucket, which is what run 2's generate stage actually calls. mtime cannot
    express this state: generate overwrites its images in place and an in-place overwrite leaves the
    directory's mtime alone, so the sdg/ of run 2 is not 'newer' by any timestamp on disk.
    """
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    _refined_run(paths, run="r1")

    rp.take_bucket(paths["sdg"], "r2")  # run 2's generate stage
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)

    with pytest.raises(SystemExit, match="different generate run"):
        rp.gen_root(cfg, paths)


def test_a_refined_bucket_nobody_marked_is_refused_rather_than_read(tmp_path):
    """A tree from before this check, or assembled by hand. It cannot be shown to belong to the sdg/
    beside it, and reading it would ship images the run did not produce."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    (paths["searched"] / "reconstructed_image").mkdir(parents=True)

    with pytest.raises(SystemExit, match="no run marker"):
        rp.gen_root(cfg, paths)


def test_a_refine_tail_over_an_earlier_generate_is_not_a_mismatch(tmp_path):
    """stages [refine, export] against a previous run's sdg/ is a legitimate pairing.

    Stamping searched/ with the REFINE run's id would make this read as stale, which is why the
    marker records the sdg/ it was assembled from instead of who assembled it.
    """
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    rp._stamp(paths["sdg"], "r1")
    (paths["searched"] / "reconstructed_image").mkdir(parents=True)
    rp._stamp(paths["searched"], "r2", source="r1")  # refined later, from r1's generate

    assert rp.gen_root(cfg, paths) == paths["searched"]


def test_generate_takes_the_sdg_bucket_so_an_earlier_runs_images_cannot_be_scored(tmp_path, monkeypatch):
    """evaluate scores the DIRECTORY, not the CSV, so a leftover image lands in the KPI unremarked.

    A first run of 40 images then a second of 2 into the same output used to leave 42 in sdg/, and
    sdg/kpi.json averaged over all of them. Nothing in generate.py removes a file it did not write.
    """
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
    paths["testcase"].write_text("{}\n")
    stale = paths["sdg"] / "reconstructed_image" / "SolderPin+bridge_00039.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"from the run before")

    monkeypatch.setattr(rp, "run", lambda *a, **k: None)
    rp.stage_generate(cfg, paths)

    assert not stale.exists(), "an earlier run's image survived into this run's bucket"
    assert rp._stamp_of(paths["sdg"])["run"] == paths["run"]


def test_a_second_run_does_not_clear_the_bucket_it_just_filled(tmp_path, monkeypatch):
    """take_bucket keys off the marker, not off existence: the stage must not delete its own output
    if it is reached twice with the same paths."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    rp.take_bucket(paths["sdg"], paths["run"])
    mine = paths["sdg"] / "reconstructed_image" / "a.png"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"this run's")

    rp.take_bucket(paths["sdg"], paths["run"])
    assert mine.exists()


def test_refine_takes_the_rounds_directory(tmp_path, monkeypatch):
    """quality_refine counts a round's pngs against the testcase rows and stops the search when they
    disagree, advising a re-run that overwrites -- which cannot remove a previous run's extras."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["refine"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
    paths["testcase"].write_text("{}\n")
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    rp._stamp(paths["sdg"], "r1")  # what run 1's generate left
    paths["kpi"].write_text("{}")
    stale = paths["rounds"] / "round_1" / "reconstructed_image" / "old.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"round from the run before")

    monkeypatch.setattr(rp, "run", lambda *a, **k: None)
    rp.stage_refine(cfg, paths)

    assert not stale.exists()
    # and searched/ now records which sdg/ it was assembled from, which is what export reads
    assert rp._stamp_of(paths["searched"])["source"] == "r1"


def test_refining_an_unmarked_sdg_leaves_searched_unprovable(tmp_path, monkeypatch):
    """An output tree written before this check has no marker to inherit, so searched/ cannot claim
    a provenance it does not have -- and a later export-only run refuses it rather than guessing.

    The alternative would be for refine to stamp sdg/ on its way past, which would have it vouch for
    a bucket it did not write.
    """
    cfg = rp.load_config(_cfg(tmp_path, stages=["refine"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
    paths["testcase"].write_text("{}\n")
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)  # no marker: an older tree
    paths["kpi"].write_text("{}")

    monkeypatch.setattr(rp, "run", lambda *a, **k: None)
    rp.stage_refine(cfg, paths)
    assert "source" not in rp._stamp_of(paths["searched"])

    (paths["searched"] / "reconstructed_image").mkdir(parents=True, exist_ok=True)
    tail = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out")))
    with pytest.raises(SystemExit, match="no run marker"):
        rp.gen_root(tail, paths)


def test_a_testcase_longer_than_the_run_asked_for_is_not_reported_as_a_shortfall(tmp_path):
    """The gate fired on != and borrowed the shortfall's words, so a SURPLUS was reported as
    'a dropped sample is a placement that could not be made' -- with nothing dropped at all.

    Measured: a second amp into one output, amp_seed 42 -> 7, wrote 12 rows for a 7-row request
    while roi_place logged 7/7 seeds OK.
    """
    cfg = rp.load_config(_cfg(tmp_path, num_sdg=7, output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
    paths["testcase"].write_text("".join('{"image_filename": "c.png"}\n' for _ in range(12)))

    with pytest.raises(SystemExit) as exc:
        rp.require_full_testcase(cfg, paths, "amp", "SHORTFALL REMEDY")
    assert "MORE, not fewer" in str(exc.value)
    assert "amp_seed" in str(exc.value)
    assert "SHORTFALL REMEDY" not in str(exc.value)
    assert "could not be made" not in str(exc.value)


def test_the_shortfall_remedy_reads_the_seeds_not_the_sample(tmp_path):
    """Every route returns the sample as OK once placement was attempted, however its seeds went, so
    counting sample statuses told the operator 'no unplaced sample' in the very run that had one."""
    cfg = rp.load_config(_cfg(tmp_path, num_sdg=2, output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["amp"].mkdir(parents=True)
    (paths["amp"] / "summary.json").write_text(
        json.dumps(
            [
                {"name": "a", "status": "OK", "seeds": {"0": {"status": "OK"}}},
                {"name": "b", "status": "OK", "seeds": {"0": {"status": "FAILED"}}},
                {"name": "c", "status": "NO_CANDIDATES"},  # never reached seeds
            ]
        )
    )
    remedy = rp._amp_shortfall_remedy(paths)
    assert "1x FAILED" in remedy and "1x NO_CANDIDATES" in remedy
    assert "no unplaced sample" not in remedy
    assert "OK" not in remedy.split("Every sample")[0], "a placed seed is not a reason for a shortfall"


def test_guardrail_blocked_sources_are_named(tmp_path, loguru_lines):
    """A blocked joint yields no image at all; the caller must not have to count files to find out."""
    (tmp_path / rp._BLOCKED_CSV).write_text("image_filename\n/ds/clean_image/A1__T101__3.png\n")
    rp.report_blocked(tmp_path)
    assert any("A1__T101__3.png" in line for line in loguru_lines)


def test_export_refuses_to_report_success_over_a_missing_file(tmp_path):
    cfg = rp.load_config(_cfg(tmp_path, stages=["export"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["sdg"].mkdir(parents=True)
    with open(paths["sdg"] / rp._GEN_CSV, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["output_filename", "image_filename"])
        w.writeheader()
        w.writerows(_rows([("A+x_00000.png", "/ds/c/j1.png")]))
    with pytest.raises(SystemExit, match="missing though the CSV lists it"):
        rp.stage_export(cfg, paths)


# ── composition ───────────────────────────────────────────────────────────────


def test_every_stage_has_a_runner():
    assert set(rp._RUNNERS) == set(rp.STAGES)


def test_dry_run_executes_no_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("--dry_run must not run a stage"))
    rp.main(["--config", str(_cfg(tmp_path)), "--dry_run"])


def test_dry_run_still_checks_the_weights(tmp_path, monkeypatch):
    """--dry_run is what an orchestrator runs as a cheap preflight task: it has to answer "is this
    container wired up to run", not just "does the YAML parse"."""
    _weights(tmp_path, monkeypatch)  # empty tree
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("--dry_run must not run a stage"))
    with pytest.raises(SystemExit, match="weight"):
        rp.main(["--config", str(_cfg(tmp_path, stages=["generate"])), "--dry_run"])


def test_a_failing_stage_stops_the_run(tmp_path, monkeypatch):
    cfg_path = _cfg(tmp_path, stages=["amp", "export"], output=str(tmp_path / "out"))
    seen = []

    def _fail(cmd, log_path, env=None):
        seen.append(cmd)
        raise SystemExit("stage failed (exit 2)")

    monkeypatch.setattr(rp, "run", _fail)
    with pytest.raises(SystemExit, match="exit 2"):
        rp.main(["--config", str(cfg_path)])
    assert len(seen) == 1, "export must not run after amp failed"


def test_the_resolved_config_is_written_beside_the_results(tmp_path, monkeypatch):
    """Every knob that shaped a run, defaults included, recorded next to what it produced."""
    cfg_path = _cfg(tmp_path, stages=["amp"], output=str(tmp_path / "out"))
    _stub_stage_run(monkeypatch)
    rp.main(["--config", str(cfg_path)])
    resolved = yaml.safe_load((tmp_path / "out" / "pipeline_config.resolved.yaml").read_text())
    assert resolved["base_seed"] == 1 and resolved["output_naming"] == "source"


def test_terminating_signals_become_a_systemexit_installed_before_any_stage_runs(tmp_path, monkeypatch):
    """A bare process kill would skip childproc.stream's finally and Server.__exit__ and orphan a
    torchrun or a running server; only an exception unwinds those, so main must install a handler
    for every signal a run may be ended with -- and install it BEFORE the first stage, or a
    cancellation during stage one is the bare kill again. The order is asserted, not implied."""
    cfg_path = _cfg(tmp_path, stages=["amp"], output=str(tmp_path / "out"))
    _stub_stage_run(monkeypatch)
    order, installed = [], {}

    def _fake_signal(signum, handler):
        installed[signum] = handler
        order.append(("signal", signum))

    real_amp = rp._RUNNERS["amp"]
    monkeypatch.setattr(rp.signal, "signal", _fake_signal)
    monkeypatch.setitem(rp._RUNNERS, "amp", lambda cfg, paths: (order.append(("stage", "amp")), real_amp(cfg, paths)))
    rp.main(["--config", str(cfg_path)])
    assert set(installed) == {rp.signal.SIGTERM, rp.signal.SIGHUP, rp.signal.SIGINT}
    assert order.index(("stage", "amp")) > max(i for i, e in enumerate(order) if e[0] == "signal")
    for sig in (rp.signal.SIGTERM, rp.signal.SIGHUP):
        with pytest.raises(SystemExit) as exc:
            installed[sig](sig, None)
        assert exc.value.code == 128 + sig


def test_a_sigterm_during_a_running_stage_stops_the_child_and_the_server(tmp_path, monkeypatch):
    """The whole chain, with a real child and the real signal: SIGTERM arrives while stage_amg is
    reading its CLI's output inside `with serving.Server(...)`. The handler raises, stream's finally
    ends the child's process group, Server.__exit__ runs, and main reports 143. Every earlier test
    covers one link with the others faked; this one is the reason the links exist."""
    FakeServer.events = []
    monkeypatch.setattr(rp.serving, "Server", FakeServer)
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0", "1"])
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp, "check_amg", lambda cfg: None)
    _mask_dirs(tmp_path, "A+x", "A+y")
    cfg_path = _cfg(tmp_path, stages=["amg"], output=str(tmp_path / "out"), amg_serve="gemma-4-31b-it")
    child_pid = []

    def _run(cmd, log_path, env=None):
        # In place of the amg CLI: a child that prints its pid and then sleeps, like a CLI mid-call.
        def _on_line(line):
            if line.startswith("PID "):
                child_pid.append(int(line.split()[1]))
                os.kill(os.getpid(), rp.signal.SIGTERM)  # the orchestrator cancels the job here

        code = "import os, time; print('PID', os.getpid(), flush=True); time.sleep(120)"
        return rp.childproc.stream([sys.executable, "-c", code], _on_line)

    monkeypatch.setattr(rp, "run", _run)
    with pytest.raises(SystemExit) as exc:
        rp.main(["--config", str(cfg_path)])
    assert exc.value.code == 143
    assert FakeServer.events == ["enter", "exit"], "the server was stopped on the way out"
    (pid,) = child_pid
    for _ in range(100):  # the group terminate waits for the child; give the kernel a moment to reap
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, rp.signal.SIGKILL)
        pytest.fail(f"the stage child {pid} outlived the cancelled run")


def _accepted_flags(module_path):
    """The long flags a CLI's argparse accepts, read from its source.

    Statically rather than by running ``--help``: importing any of these modules pulls in torch and
    the framework, and the point is only to compare two lists of strings.
    """
    src = pathlib.Path(module_path).read_text()
    return set(re.findall(r"add_argument\(\s*\n?\s*[\"'](--[\w-]+)", src))


_CLI = {
    "amp": "anomalygen/scripts/auto_mask_placement/roi_place_pipeline.py",
    "amg": "anomalygen/scripts/agentic_mask_generation/generate.py",
    "generate": "anomalygen/scripts/texture/generate.py",
    "evaluate": "anomalygen/scripts/texture/evaluate.py",
    "refine": "anomalygen/scripts/texture/quality_refine.py",
}


@pytest.mark.parametrize("stage", sorted(_CLI))
def test_every_flag_the_entrypoint_emits_exists_on_the_cli_it_targets(tmp_path, monkeypatch, stage):
    """The entrypoint's whole risk is drift: a renamed flag in a sub-CLI turns into a failure four
    stages deep, or -- worse for a flag with a default -- a run that silently ignores the config."""
    extra = {}
    if stage == "amg":
        _mask_dirs(tmp_path, "A+x")
        extra = {
            **_amg_keys(tmp_path),
            "stages": ["amg"],
            "amp_per_clean_pairing": "off",
            "amg_beam_multi": 4,
            "amg_beam_pairwise": True,
            "amg_exemplar_captions": True,
            "amg_exemplar_visual_prompt": True,
            "amg_coverage_note": True,
        }
    body = dict(
        output=str(tmp_path / "out"),
        base_checkpoint=str(_touch(tmp_path / "b.pt")),
        per_defect_counts={"A+x": 4},
        guidance=3.0,
        num_steps=35,
        crop_ratio=2.0,
        amp_per_clean_pairing="on",
    )
    body.update(extra)
    cfg = rp.load_config(_cfg(tmp_path, **body))
    paths = rp.layout(cfg)
    if stage != "amg":
        # As many rows as the run asks for, each naming a distinct clean image: the stage gates on
        # the count, and on source naming two rows over one clean image would collide.
        paths["testcase"].parent.mkdir(parents=True)
        paths["testcase"].write_text(
            "".join(json.dumps({"image_filename": f"clean_{i}.png"}) + "\n" for i in range(cfg.num_sdg))
        )
    (paths["sdg"] / "reconstructed_image").mkdir(parents=True)
    paths["kpi"].write_text("{}")

    seen = []

    def _fake_run(cmd, log_path, env=None):
        seen.append([str(c) for c in cmd])
        if stage == "amg":
            paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
            paths["testcase"].write_text("".join(json.dumps({"image_filename": f"c{i}.png"}) + "\n" for i in range(4)))

    monkeypatch.setattr(rp, "run", _fake_run)
    rp._RUNNERS[stage](cfg, paths)
    argv = seen[0]

    # roi_place's flags ride after `--`, so they are checked against roi_place, not the wrapper.
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1 :]
        place = _accepted_flags("anomalygen/scripts/auto_mask_placement/roi_place.py")
        assert {a for a in passthrough if a.startswith("--")} <= place

    # Everything before the TARGET belongs to the launcher: torchrun's flags are not generate.py's
    # to accept. Anchored on _CLI[stage] rather than on "the first .py or -m", which lands on
    # torch.distributed.run instead -- and rather than naming the launcher flags, which needed a
    # second exception the moment --standalone appeared beside --nproc_per_node.
    target = pathlib.PurePath(_CLI[stage]).stem
    for i, a in enumerate(argv):
        if pathlib.PurePath(a).stem == target:
            argv = argv[i + 1 :]
            break
    emitted = {a for a in argv if a.startswith("--")}
    accepted = _accepted_flags(_CLI[stage])
    assert emitted and accepted, "a regex that matched nothing would make this assertion vacuous"
    # generate.py's --guardrail/--no-guardrail is one BooleanOptionalAction argument.
    accepted |= {"--no-" + f[2:] for f in accepted}
    assert emitted <= accepted, f"{stage}: {sorted(emitted - accepted)} not accepted by {_CLI[stage]}"


def test_the_example_config_documents_every_key_and_invents_none():
    """The example is the only place a reader learns a key exists; a default added without a line
    here is one nobody will find."""
    example = yaml.safe_load(pathlib.Path("ag_config/pipeline/example.yaml").read_text())
    assert set(example) == rp._field_names()


def test_the_amg_example_is_a_runnable_config(tmp_path, monkeypatch):
    """The amg example is what a reader copies; it must load, resolve every key, and dry-run against
    a stub tree without touching the network."""
    example = yaml.safe_load(pathlib.Path("ag_config/pipeline/example_amg.yaml").read_text())
    assert set(example) == rp._field_names(), "every key documented, none invented"
    assert example["stages"][0] == "amg" and "amp" not in example["stages"]
    assert example["amg_beam_multi"] == 4 and example["amg_beam_pairwise"] is True

    # Point every path at a stub so load_config's existence checks pass.
    _mask_dirs(tmp_path, "passive_component+missing")
    keys = _amg_keys(tmp_path)
    example.update(
        input=str(tmp_path / "ds"),
        output=str(tmp_path / "out"),
        recipe=str(tmp_path / "r.yaml"),
        checkpoint=str(tmp_path / "ckpt.pt"),
        amg_model_registry=keys["amg_model_registry"],
        amg_gen_model="m",
        amg_judge_model="m",
    )
    (tmp_path / "r.yaml").touch()
    (tmp_path / "ckpt.pt").touch()
    cfg_path = tmp_path / "example_amg.yaml"
    cfg_path.write_text(yaml.safe_dump(example))
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    # The example serves Gemma (tp 2), so its GPU preflight wants two devices. Pin what is "visible"
    # rather than inherit the host's: a one-GPU CI runner is not a defect in the example.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("--dry_run must not run a stage"))
    rp.main(["--config", str(cfg_path), "--dry_run"])


def test_the_use_case_example_is_a_runnable_config(tmp_path, monkeypatch):
    """The use-case example is the one a reader can run without owning any data, so what it claims
    has to hold: every key documented, and a config that loads and resolves."""
    example = yaml.safe_load(pathlib.Path("ag_config/pipeline/example_usecase.yaml").read_text())
    assert set(example) == rp._field_names(), "every key documented, none invented"

    fetch = example["staging"]["fetch"]
    assert any(str(u).startswith("usecase://") for u in fetch["input"]), "the file is about use-cases"
    assert "prepare" in example["stages"], "nothing else runs the use-case"
    # Six outputs over five clean images claims one of them twice, which `source` naming refuses --
    # an example that fails the moment it is copied teaches the wrong thing.
    assert example["output_naming"] == "indexed"

    # NOTHING is substituted but the output directory. The header claims this config runs as
    # written, and a test that swapped the recipe and the checkpoint for stubs is what let that
    # claim be false: `--dry_run` on the shipped file exited with "checkpoint does not exist".
    assert example["checkpoint"] is None, "a path to a checkpoint nobody has is what broke this before"
    assert example["stages"] == ["prepare", "amp"], "the stages that need no model, so it can run"
    example.update(output=str(tmp_path / "out"))
    cfg_path = tmp_path / "example_usecase.yaml"
    cfg_path.write_text(yaml.safe_dump(example))
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("--dry_run must not run a stage"))
    # --dry_run returns before any stage, so the use-case script never runs and nothing is
    # downloaded -- which is what makes it safe to pin a network-fed example in a unit test.
    rp.main(["--config", str(cfg_path), "--dry_run"])


_TEMPLATES = sorted(pathlib.Path("ag_config/pipeline/templates").glob("*_day[01]*.yaml"))
# The one cloud model the registry templates name (docs/pipeline/config-from-request.md, the fixed
# table). Six hand-edited files carry it twice each; a typo in one would pass a gen == judge check
# and die at check_amg in the pod.
_CLOUD_MODEL = "azure/openai/gpt-6-astra"
assert len(_TEMPLATES) >= 12, "the template directory moved, or the glob no longer matches it"


def _use_case_and_day(template) -> tuple:
    """``(usecase name, dayN)`` off a template's file name -- the name is the loader's own
    (``usecase://<name>``), so it may itself contain underscores."""
    from anomalygen.staging import usecase

    for name in usecase.names():
        if template.stem.startswith(f"{name}_day"):
            return name, template.stem[len(name) + 1 :].split("_")[0]
    pytest.fail(f"{template.name} is not named after a use case this image ships: {usecase.names()}")


def _trained_pairs(recipe_path) -> set:
    """``texture+defect`` for every pair the recipe names -- what the checkpoint can generate."""
    recipe = yaml.safe_load(pathlib.Path(recipe_path).read_text())
    return {f"{t}+{d}" for t, d in recipe["anomaly_types"]}


@pytest.mark.parametrize("template", _TEMPLATES, ids=[g.stem for g in _TEMPLATES])
def test_every_template_has_the_shape_the_request_doc_promises(template):
    """docs/pipeline/config-from-request.md tells a config generator to match these files outside the
    marked values; what the doc promises about their shape is pinned here, and the loader's own
    verdict is the dry-run test below."""
    cfg = yaml.safe_load(template.read_text())
    uc, day = _use_case_and_day(template)
    assert set(cfg) <= rp._field_names(), f"{template.name} invents a key"
    assert cfg["stages"] == ["prepare", "amg", "generate", "export", "publish"]
    assert cfg["recipe"].startswith("ag_config/") and pathlib.Path(cfg["recipe"]).is_file()
    # The guardrail is on everywhere but magnetic_tile, where the text check blocks every MT_Crack
    # caption on the word "crack" and the run still passes -- the doc says why it is off there.
    assert cfg["guardrail"] is (uc != "magnetic_tile")
    assert "\\n" not in template.read_text(), "a literal escape in a header: the file is matched byte for byte"
    # Decision 2: every type named is one the recipe trained, five of each, num_sdg their sum.
    counts = cfg["per_defect_counts"]
    assert set(counts) <= _trained_pairs(cfg["recipe"]), f"{template.name} names an untrained type"
    assert all(n == 5 for n in counts.values()) and sum(counts.values()) == cfg["num_sdg"]
    # The model is fixed per use case and day (the fixed table): pcb Day 1 and phone_screen use Astra through
    # a registry; pcb Day 0 and magnetic_tile serve Gemma from the run.
    if (uc, day) in {("pcb", "day1"), ("phone_screen", "day1")}:
        assert cfg["amg_serve"] is None
        assert cfg["amg_gen_model"] == cfg["amg_judge_model"] == _CLOUD_MODEL
        assert cfg["amg_model_registry"]
    else:
        assert cfg["amg_serve"] == "gemma-4-31b-it" and cfg["amg_model_registry"] is None
        assert cfg["amg_beam_multi"] == 4 and cfg["amg_beam_pairwise"] is True
    fetch = cfg["staging"]["fetch"]
    (source,) = fetch["input"]
    assert source.startswith("usecase://"), "the dataset comes from the use case's own prepare script"
    if uc == "pcb":
        # pcb generates on an image the request supplies: one directory key, one source, the texture
        # per_defect_counts names.
        ((pair, _),) = counts.items()
        texture = pair.split("+")[0]
        ((key, sources),) = fetch["input_override"].items()
        assert key == f"{texture}/clean_image", "Decision 1 and the override key must agree; a directory, not a file"
        (src,) = sources
        if day == "day1":
            assert src.startswith(("hf://", "s3://", "<"))
            if src.startswith("hf://"):
                assert re.search(r"@[0-9a-f]{40}/", src), "pin the revision"
        else:
            assert src.startswith("s3://") and src.endswith(f"/{texture}/")
    else:
        # magnetic_tile and phone_screen generate on the dataset's own clean images: every trained type, no override.
        assert "input_override" not in fetch
        assert set(counts) == _trained_pairs(cfg["recipe"])
    # One publish rule (the guide, "One publish rule"): <root>/<day>/<usecase>/, then for pcb alone
    # <component>/<defect>/ -- the two things a pcb request names. Pinned exactly, not by prefix.
    root = "s3://sdg-workflow-controller-output/generated/defect_image_generation/"
    tail = f"{day}/{uc}/"
    if uc == "pcb":
        rest = template.stem[len(f"pcb_{day}_") :]
        defect = next(dft for dft in ("excess_solder", "missing", "bridge") if rest.endswith(dft))
        component = rest[: -len(defect)].rstrip("_") or "<component>"
        tail += f"{component}/{defect}/"
    assert cfg["staging"]["publish"]["output"] == root + tail, f"{template.name}: publish tail is not {tail}"


@pytest.mark.parametrize("template", _TEMPLATES, ids=[g.stem for g in _TEMPLATES])
def test_every_template_dry_runs(template, tmp_path, monkeypatch):
    """A template the loader would reject is worse than none: every generated config would then be
    wrong in the same way and pass the diff.

    Two kinds of value cannot be real in a checked-in file and are stood in for here, exactly as a
    deployment would: the DEPLOYMENT mounts (checkpoint, registry) become stub files, and a
    template's placeholder image URL becomes a real pinned one. Everything else is dry-run as
    written -- the usecase:// source, the override key, the publish prefix shape."""
    cfg = yaml.safe_load(template.read_text())
    (tmp_path / "ckpt.pt").touch()
    cfg["checkpoint"] = str(tmp_path / "ckpt.pt")
    if cfg["amg_model_registry"]:
        registry = tmp_path / "endpoints.yaml"
        registry.write_text(
            yaml.safe_dump([{"model": cfg["amg_gen_model"], "base_url": "https://gateway/v1", "key": "k"}])
        )
        cfg["amg_model_registry"] = str(registry)
    for key, sources in cfg["staging"]["fetch"].get("input_override", {}).items():
        cfg["staging"]["fetch"]["input_override"][key] = [
            "hf://datasets/nvidia/Spark-AnomalyGen-USD@0bbac7b7322afcfa0d58c837c17fddaee84d9b76/input_real_image/0402_H060.jpg"
            if s.startswith("<")
            else s
            for s in sources
        ]
    cfg_path = tmp_path / template.name
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp, "check_publish_destination", lambda cfg: None)
    # The Gemma-serving templates (tp 2) want two visible devices; the others want num_gpus.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1" if cfg["amg_serve"] else "0")
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("--dry_run must not run a stage"))
    rp.main(["--config", str(cfg_path), "--dry_run"])


def test_the_example_config_lists_every_stage_in_run_order():
    """Every stage but amg: amp and amg exclude each other, and the example is the amp one."""
    example = yaml.safe_load(pathlib.Path("ag_config/pipeline/example.yaml").read_text())
    assert example["stages"] == [s for s in rp.STAGES if s != "amg"]


# ── weights ───────────────────────────────────────────────────────────────────


# The catalogue rows whose weight is a DIRECTORY, so the fixture can build one as a directory. It
# touched every path as a file, which meant no test here had ever seen the shape the base checkpoint
# and the DINOv2 tree really have -- and "an empty directory reads as present" was therefore not
# something these tests could express, let alone catch.
_DIR_WEIGHTS = frozenset(f"checkpoints/{w.location}" for w in catalogue.WEIGHTS if w.kind == "dir")


def _weights(tmp_path, monkeypatch, *, hf=(), files=()):
    """A fake checkpoints tree: an HF cache plus whatever repo-relative weights are asked for."""
    root = tmp_path / "ckpt"
    monkeypatch.setattr(rp, "_REPO_ROOT", root)
    monkeypatch.setenv("HF_HUB_CACHE", str(root / "checkpoints" / "hf"))
    for repo, rel in hf:
        target = root / "checkpoints" / "hf" / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    for rel in files:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel in _DIR_WEIGHTS:
            target.mkdir(exist_ok=True)
            (target / ".metadata").touch()  # a DCP is a directory, and an empty one is not one
        else:
            target.touch()
    return root


_GEN_FILES = ("checkpoints/Cosmos3-Nano", "checkpoints/wan2pt2/Wan2.2_VAE.pth")
_EVAL_FILES = (
    "checkpoints/facebook/dinov2-large",
    "checkpoints/nvidia/C-RADIO-V3/model.safetensors",
    "checkpoints/facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt",
)
_TOKENIZER = [("Qwen/Qwen3-VL-8B-Instruct", "tokenizer.json")]
_GUARDRAIL = [
    ("nvidia/Cosmos-Guardrail1", "blocklist"),
    ("nvidia/Cosmos-Guardrail1", "face_blur_filter/Resnet50_Final.pth"),
    ("Qwen/Qwen3Guard-Gen-0.6B", "model.safetensors"),
]


@pytest.mark.parametrize(
    ("stages", "expect"),
    [
        (["amp"], []),  # cad_tht placement loads no model; text2roi's are the roi_place CLI's own
        (["export"], []),  # file copying
        (["evaluate"], ["DINOv2 (NN/MNN)", "C-RADIO v3 (FID)", "SAM2.1 (anomaly-quality axes)"]),
        (["generate"], ["base checkpoint (nano)", "Wan2.2 VAE", "caption tokenizer"]),
        (["amg"], ["AMG segmentation (SAM2.1)", "AMG exemplar embedding (DINOv2)"]),
    ],
    ids=["amp", "export", "evaluate", "generate", "amg"],
)
def test_only_the_weights_the_selected_stages_load_are_required(tmp_path, stages, expect):
    """An evaluate-only run has no use for the 30 GB base checkpoint; demanding the whole tree would
    fail a legitimate partial run on weights it never opens."""
    extra = _amg_keys(tmp_path) if "amg" in stages else {}
    cfg = rp.load_config(_cfg(tmp_path, stages=stages, guardrail=False, **extra))
    assert [label for label, _, _ in rp.required_weights(cfg)] == expect


def test_the_guardrail_weights_are_required_only_when_it_is_enabled(tmp_path):
    on = rp.load_config(_cfg(tmp_path, stages=["generate"], guardrail=True))
    off = rp.load_config(_cfg(tmp_path, stages=["generate"], guardrail=False))
    assert [t for _, t, _ in rp.required_weights(on) if "Guard" in str(t)] == [
        "nvidia/Cosmos-Guardrail1",
        "Qwen/Qwen3Guard-Gen-0.6B",
    ]
    assert not [t for _, t, _ in rp.required_weights(off) if "Guard" in str(t)]


def test_the_face_blur_weight_is_checked_not_just_the_blocklist(tmp_path, monkeypatch):
    """The repo's own preflight probes `blocklist` alone, and a cache holding only that passes it --
    then RetinaFaceFilter reaches for Resnet50_Final.pth and falls back to a Hub download."""
    _weights(
        tmp_path,
        monkeypatch,
        hf=[("nvidia/Cosmos-Guardrail1", "blocklist"), ("Qwen/Qwen3Guard-Gen-0.6B", "model.safetensors"), *_TOKENIZER],
        files=_GEN_FILES,
    )
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], guardrail=True))
    with pytest.raises(SystemExit, match="Resnet50_Final.pth"):
        rp.check_weights(cfg)


def test_a_complete_tree_passes(tmp_path, monkeypatch):
    _weights(tmp_path, monkeypatch, hf=[*_GUARDRAIL, *_TOKENIZER], files=_GEN_FILES + _EVAL_FILES)
    # Every stage but amg (amp and amg exclude each other), so publish is included and it needs
    # somewhere to publish to.
    staging = {"publish": {"output": "s3://bucket/prefix/"}}
    stages = [s for s in rp.STAGES if s != "amg"]
    cfg = rp.load_config(_cfg(tmp_path, stages=stages, guardrail=True, staging=staging))
    rp.check_weights(cfg)


# ── amg preflight ─────────────────────────────────────────────────────────────


def _registry(tmp_path, entries):
    reg = tmp_path / "endpoints.yaml"
    reg.write_text(yaml.safe_dump(entries) if not isinstance(entries, str) else entries)
    return str(reg)


def test_a_registry_that_resolves_both_models_passes(tmp_path):
    reg = _registry(
        tmp_path, [{"model": "g", "base_url": "http://x/v1"}, {"model": "j", "base_url": "http://y/v1", "key": "k"}]
    )
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_gen_model="g", amg_judge_model="j", amg_model_registry=reg))
    rp.check_amg(cfg)
    assert set(rp._read_registry(reg)) == {"g", "j"}


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        ("- just a string\n", "model"),
        ({"model": "g"}, "list"),
        ([{"model": "g"}], "base_url"),
        ([{"base_url": "http://x/v1"}], "model"),
        ([{"model": "other", "base_url": "http://x/v1"}], r"g.*other"),
    ],
    ids=["entry-not-a-mapping", "not-a-list", "no-base_url", "no-model", "name-absent"],
)
def test_check_amg_names_what_is_wrong_with_the_registry(tmp_path, entries, match):
    reg = _registry(tmp_path, entries)
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_gen_model="g", amg_judge_model="g", amg_model_registry=reg))
    with pytest.raises(SystemExit, match=match):
        rp.check_amg(cfg)


def test_a_missing_registry_file_is_caught_by_the_path_check(tmp_path):
    with pytest.raises(SystemExit, match="amg_model_registry"):
        rp.load_config(
            _cfg(
                tmp_path,
                stages=["amg"],
                amg_gen_model="g",
                amg_judge_model="g",
                amg_model_registry=str(tmp_path / "nope.yaml"),
            )
        )


def test_check_amg_is_skipped_when_amg_is_not_selected(tmp_path):
    """A registry pointed at a NON-existent path proves the function returns without reading it."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["amp"], amg_model_registry=str(tmp_path / "nope.yaml")))
    assert rp.check_amg(cfg) is None


def test_dry_run_runs_the_amg_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("--dry_run must not run a stage"))
    reg = _registry(tmp_path, [{"model": "other", "base_url": "http://x/v1"}])
    cfg_path = _cfg(tmp_path, stages=["amg"], amg_gen_model="g", amg_judge_model="g", amg_model_registry=reg)
    with pytest.raises(SystemExit, match="other"):
        rp.main(["--config", str(cfg_path), "--dry_run"])


def test_amg_serve_adds_the_recipes_weights_to_the_preflight(tmp_path, monkeypatch):
    _weights(tmp_path, monkeypatch)  # empty tree
    _mask_dirs(tmp_path, "A+x")
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it"))
    labels = [label for label, _, _ in rp.required_weights(cfg)]
    assert "gemma-4-31b-it.sh weights" in labels
    assert any("google/gemma-4-31b-it" in m for m in rp.missing_weights(cfg))

    _weights(
        tmp_path,
        monkeypatch,
        hf=[
            ("google/gemma-4-31b-it", "model-00001-of-00002.safetensors"),
            ("google/gemma-4-31b-it", "model-00002-of-00002.safetensors"),
        ],
        files=["checkpoints/facebook/dinov2-large", f"checkpoints/{catalogue.entry('sam2').location}"],
    )
    assert not [m for m in rp.missing_weights(cfg) if "gemma" in m]


def test_no_amg_serve_means_no_server_weights(tmp_path):
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], **_amg_keys(tmp_path)))
    assert "gemma-4-31b-it.sh weights" not in [label for label, _, _ in rp.required_weights(cfg)]


def test_check_amg_accepts_the_served_name_without_a_registry(tmp_path):
    _mask_dirs(tmp_path, "A+x")
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it"))
    rp.check_amg(cfg)


def test_check_amg_still_needs_a_registry_for_a_model_the_recipe_does_not_serve(tmp_path):
    _mask_dirs(tmp_path, "A+x")
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it", amg_judge_model="azure/opus"))
    with pytest.raises(SystemExit, match="azure/opus"):
        rp.check_amg(cfg)


@pytest.mark.parametrize(
    ("serve", "num_gpus", "expect"),
    [(None, 1, 1), ("gemma-4-31b-it", 1, 2), ("gemma-4-31b-it", 4, 4), (None, 3, 3)],
    ids=["cloud-1", "gemma-1", "gemma-4", "cloud-3"],
)
def test_gpus_required_is_the_larger_of_the_two_consumers(tmp_path, serve, num_gpus, expect):
    _mask_dirs(tmp_path, "A+x")
    over = {"amg_serve": serve} if serve else _amg_keys(tmp_path)
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg", "generate"], num_gpus=num_gpus, **over))
    assert rp.gpus_required(cfg) == expect


def test_too_few_visible_gpus_is_an_error_naming_both_numbers(tmp_path, monkeypatch):
    _mask_dirs(tmp_path, "A+x")
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0"])
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it"))
    with pytest.raises(SystemExit, match=r"1.*2|2.*1"):
        rp.check_gpus(cfg)


def test_too_few_visible_gpus_with_a_yaml_quoted_num_gpus_is_still_a_systemexit(tmp_path, monkeypatch):
    """num_gpus: "4" must not slip past load_config's coercion and blow up check_gpus's int compare."""
    _mask_dirs(tmp_path, "A+x")
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0"])
    cfg = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it", num_gpus="4"))
    with pytest.raises(SystemExit, match="4"):
        rp.check_gpus(cfg)


def test_no_visible_gpu_is_a_warning_not_an_error(tmp_path, monkeypatch, loguru_lines):
    """A CPU-only preflight host cannot answer the question; the real pod will."""
    _mask_dirs(tmp_path, "A+x")
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: [])
    rp.check_gpus(rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it")))
    assert any("WARNING" in line and "GPU" in line for line in loguru_lines)


def test_dry_run_reports_the_service_and_the_gpu_budget(tmp_path, monkeypatch, loguru_lines):
    _mask_dirs(tmp_path, "A+x")
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0", "1"])
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("dry run"))
    rp.main(["--config", str(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it")), "--dry_run"])
    text = "\n".join(loguru_lines)
    assert "gpus_required" in text and "gemma-4-31b-it" in text and "service" in text


def test_every_missing_weight_is_named_at_once(tmp_path, monkeypatch):
    """An operator wiring up a volume mount wants one list, not a failure per round trip through a
    20-minute stage."""
    _weights(tmp_path, monkeypatch)
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate", "evaluate"], guardrail=True))
    with pytest.raises(SystemExit) as exc:
        rp.check_weights(cfg)
    assert "9 weight(s)" in str(exc.value)
    for expected in ("Cosmos3-Nano", "Wan2.2_VAE.pth", "dinov2-large", "C-RADIO-V3", "Resnet50_Final.pth", "sam2.1"):
        assert expected in str(exc.value)


def test_hf_missing_names_the_repo_when_no_probe_is_known(tmp_path, monkeypatch):
    """A repo the catalogue does not carry must not pass silently: with no probes, every snapshot
    trivially satisfies `not absent`, so the check would green-light a cache that holds nothing."""
    _weights(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="nobody/nothing"):
        rp._hf_missing("nobody/nothing")


def test_an_inherited_hf_cache_is_honoured(tmp_path, monkeypatch):
    """generate.py only setdefaults HF_HUB_CACHE, so a caller that exported one is what the run uses
    -- checking the repo-local path instead would report weights the run will never look for."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    assert rp._hf_cache() == tmp_path / "elsewhere"
    monkeypatch.delenv("HF_HUB_CACHE")
    assert rp._hf_cache() == rp._REPO_ROOT / "checkpoints" / "hf"


@pytest.mark.parametrize(
    ("recipe_body", "expected"),
    [("model_size: edge\n", "edge"), ("model_size: nano\n", "nano"), ("anomaly_types: [a]\n", "nano")],
    ids=["edge", "nano", "absent-defaults"],
)
def test_the_base_checkpoint_follows_the_recipe(tmp_path, recipe_body, expected):
    recipe = tmp_path / "r.yaml"
    recipe.write_text(recipe_body)
    assert rp._model_size(recipe) == expected


def test_a_python_recipe_falls_back_rather_than_failing(tmp_path):
    """A .py recipe is a module; guessing wrong would only misname the base dir in an error."""
    assert rp._model_size(tmp_path / "exp.py") == "nano"


def test_an_explicit_base_checkpoint_is_what_gets_checked(tmp_path):
    cfg = rp.load_config(
        _cfg(tmp_path, stages=["generate"], base_checkpoint=str(_touch(tmp_path / "base.pt")), guardrail=False)
    )
    base = [t for label, t, _ in rp.required_weights(cfg) if "base checkpoint" in label]
    # The absolute path, and the same one prepare would fetch to and generate is handed. It used to
    # be anchored on the repository here while the fetch destination stayed cwd-relative.
    assert base == [tmp_path / "base.pt"]


@pytest.mark.parametrize("key", ["recipe", "checkpoint", "base_checkpoint", "amg_model_registry"])
def test_a_use_case_is_refused_on_a_single_file_destination(tmp_path, key):
    """A use-case script writes a dataset -- a tree of textures plus defect_spec.jsonl -- so it can
    only fill a directory. Every other fetchable key names one file, and the mismatch would surface
    from inside the copy as an IsADirectoryError naming neither the key nor the use-case."""
    # The key needs a destination of its own, or the earlier "set the key as well" check answers
    # first -- correctly, since a fetch with nowhere to write is a different mistake.
    cfg = _cfg(
        tmp_path,
        stages=["prepare"],
        base_checkpoint=str(_touch(tmp_path / "base.pt")),
        amg_model_registry=str(_touch(tmp_path / "reg.yaml")),
        **{"staging": {"fetch": {key: ["usecase://pcb"]}}},
    )
    with pytest.raises(SystemExit, match="prepares a whole dataset"):
        rp.load_config(cfg)


def test_a_use_case_is_accepted_on_the_dataset_destination(tmp_path):
    cfg = _cfg(tmp_path, stages=["prepare"], **{"staging": {"fetch": {"input": ["usecase://pcb"]}}})
    assert rp.load_config(cfg).staging["fetch"]["input"] == ["usecase://pcb"]


def test_a_use_case_typo_is_caught_before_any_stage_runs(tmp_path):
    """It costs a config read; the alternative is finding out after prepare has downloaded."""
    cfg = _cfg(tmp_path, stages=["prepare"], **{"staging": {"fetch": {"input": ["usecase://pcbb"]}}})
    with pytest.raises(SystemExit, match="not a use-case this image ships"):
        rp.load_config(cfg)


def test_the_fetch_order_is_the_same_on_every_run(tmp_path, monkeypatch):
    """`_field_names()` is a SET, and str hashing is randomised per process.

    Measured across four interpreters: four different orders. fetch replaces its destination, so
    nested destinations -- `input: /work/data` with `recipe: /work/data/recipe.yaml` -- depend on
    it, and the same config succeeded or failed depending on the hash seed. Verified both ways: with
    input last the foreign-destination guard refuses the run; with recipe last both survive.
    """
    cfg = rp.load_config(_cfg(tmp_path, stages=["prepare"], output=str(tmp_path / "out")))
    order = [f.name for f in dataclasses.fields(rp.Config)]

    seen = []
    monkeypatch.setattr(rp, "_fetch_spec", lambda c: {k: ["s3://b/x"] for k in rp._FETCHABLE})
    monkeypatch.setattr("anomalygen.staging.fetch", lambda src, dest, override=None: seen.append(dest))
    monkeypatch.setattr("anomalygen.staging.ensure_base_weights", lambda *a, **k: False)
    rp.stage_prepare(cfg, paths := rp.layout(cfg))
    assert paths

    fetched_keys = [k for k in order if k in rp._FETCHABLE]
    assert [getattr(cfg, k) for k in fetched_keys] == seen, "not in the order Config declares"


def test_prepare_fetches_the_base_checkpoint_to_the_path_generate_loads(tmp_path, monkeypatch):
    """One root, from a working directory that is not the repository -- the case the entrypoint is for.

    prepare hands fetch the raw config value, which fetch resolves against the cwd; this file used to
    resolve the same value against the repository. So an orchestrator running
    /repo/.venv/bin/python -m ...run_pipeline from /work downloaded the DCP to /work/ft and then
    looked for /repo/ft. Where a tree happens to sit at the repo path, it is loaded silently.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rp, "_REPO_ROOT", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    _touch(tmp_path / "ft.pt")

    cfg = rp.load_config(
        _cfg(tmp_path, stages=["prepare", "generate"], base_checkpoint="ft.pt", output=str(tmp_path / "out"))
    )
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
    paths["testcase"].write_text("{}\n")

    fetched = []
    monkeypatch.setattr(rp, "_fetch_spec", lambda c: {"base_checkpoint": ["s3://b/ft.pt"]})
    monkeypatch.setattr(
        "anomalygen.staging.fetch", lambda src, dest, override=None: fetched.append(pathlib.Path(dest).resolve())
    )
    monkeypatch.setattr("anomalygen.staging.ensure_base_weights", lambda *a, **k: False)
    rp.stage_prepare(cfg, paths)

    seen = []
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.append([str(c) for c in cmd]))
    rp.stage_generate(cfg, paths)

    loaded = pathlib.Path(_flags(seen[0])["--base_checkpoint"])
    checked = [t for label, t, _ in rp.required_weights(cfg) if "base checkpoint" in label]
    assert fetched == [loaded], f"prepare wrote {fetched} and generate loads {loaded}"
    assert checked == [loaded], "the preflight checked a path neither of them uses"
    assert loaded != rp._REPO_ROOT / "ft.pt", "anchored on the repository, not on the config's own root"


def test_the_sibling_clis_are_addressed_by_package_not_cwd(tmp_path, monkeypatch):
    """Under an orchestrator the working directory is whatever the image declares, so a relative
    'anomalygen/scripts/texture/generate.py' resolves to nothing."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    paths["testcase"].parent.mkdir(parents=True)
    paths["testcase"].write_text("{}\n")
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    seen = []
    monkeypatch.setattr(rp, "run", lambda cmd, log_path, env=None: seen.append([str(c) for c in cmd]))
    rp.stage_generate(cfg, paths)
    script = next(c for c in seen[0] if c.endswith("generate.py"))
    assert pathlib.Path(script).is_absolute() and pathlib.Path(script).is_file()


@pytest.mark.parametrize(
    ("routes", "counts", "wanted"),
    [
        ({"A+x": "cad_tht"}, None, False),
        ({"A+x": "free"}, None, False),
        ({"A+x": "cad"}, None, False),
        ({"A+x": "text"}, None, True),
        ({"A+x": None}, None, True),  # roi_place defaults an absent key to text, so this one routes there
        ({"A+x": "cad_tht", "A+y": "text"}, None, True),  # any selected type is enough
        ({"A+x": "cad_tht", "A+y": "text"}, {"A+x": 4}, False),  # the text row is allocated zero
    ],
    ids=["cad_tht", "free", "cad", "text", "absent-key", "mixed", "mixed-but-not-selected"],
)
def test_text2roi_weights_follow_the_routes_the_run_will_actually_place(tmp_path, routes, counts, wanted):
    """roi_place loads the VLM and SAM2 unguarded, so a text-routed run without them dies seconds in;
    a cad_tht run must not be asked for 30 GB of VLM it never opens."""
    config = _cfg(tmp_path, stages=["amp"], per_defect_counts=counts, num_sdg=4)
    _write_spec(tmp_path / "ds", routes)  # after _cfg, which lays down the default spec
    cfg = rp.load_config(config)
    assert rp._text_routed(cfg) is wanted
    labels = [label for label, _, _ in rp.required_weights(cfg)]
    assert ("text2roi grounding VLM" in labels) is wanted


def test_text_routes_in_the_spec_do_not_pull_the_vlm_under_amg(tmp_path):
    """The spec may still say `text` from an earlier amp run; amg does not read it."""
    keys = _amg_keys(tmp_path)
    _mask_dirs(tmp_path, "A+x")
    config = _cfg(tmp_path, stages=["amg"], **keys)
    _write_spec(tmp_path / "ds", {"A+x": "text"})
    cfg = rp.load_config(config)
    assert "text2roi grounding VLM" not in [label for label, _, _ in rp.required_weights(cfg)]


def test_scoring_requires_sam2_for_the_anomaly_quality_axes(tmp_path):
    """anomaly_quality tolerates a missing checkpoint and warns, so this makes a soft dependency hard
    on purpose: a KPI quietly one column short is worse to receive than a preflight naming the file."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["evaluate"]))
    assert "SAM2.1 (anomaly-quality axes)" in [label for label, _, _ in rp.required_weights(cfg)]


def test_the_two_sam2_users_resolve_to_one_file(tmp_path):
    """roi_generation and anomaly_quality anchor the same checkpoint separately; a run needing both
    must not report it twice or check two paths."""
    config = _cfg(tmp_path, stages=["amp", "evaluate"])
    _write_spec(tmp_path / "ds", {"A+x": "text"})
    cfg = rp.load_config(config)
    sam2 = [t for label, t, _ in rp.required_weights(cfg) if "SAM2" in label]
    assert len(set(sam2)) == 1, sam2


# ── amg stage ─────────────────────────────────────────────────────────────────


def _amg_cfg(tmp_path, **over):
    _mask_dirs(tmp_path, "A+x", "A+y")
    body = {"stages": ["amg"], "output": str(tmp_path / "out"), **_amg_keys(tmp_path)}
    body.update(over)
    return rp.load_config(_cfg(tmp_path, **body))


def _amg_rows(n):
    return "".join(json.dumps({"image_filename": f"/d/c{i}.png", "anomaly_type": "A+x"}) + "\n" for i in range(n))


def _run_amg(cfg, monkeypatch, rows):
    """Run stage_amg with a fake CLI that writes `rows` testcase rows; return the argv it saw."""
    paths = rp.layout(cfg)
    seen = []

    def _fake(cmd, log_path, env=None):
        seen.append([str(c) for c in cmd])
        paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
        paths["testcase"].write_text(rows)

    monkeypatch.setattr(rp, "run", _fake)
    rp.stage_amg(cfg, paths)
    return seen[0], paths


def test_amg_argv_addresses_the_cli_by_package_and_points_everything_into_the_run(tmp_path, monkeypatch):
    cfg = _amg_cfg(tmp_path)
    argv, paths = _run_amg(cfg, monkeypatch, _amg_rows(4))
    assert argv[:3] == [sys.executable, "-m", rp._AMG_CLI]
    flags = _flags(argv)
    assert flags["--dataset-path"] == cfg.input
    assert flags["--out-dir"] == str(paths["amg"])
    assert flags["--cache-dir"] == str(paths["amg_cache"]), "the CLI's default is CWD-relative"
    assert flags["--model-registry"] == cfg.amg_model_registry
    assert flags["--gen-model"] == "m" and flags["--judge-model"] == "m"
    assert flags["--num-sdg"] == "4" and flags["--mode"] == "inference"
    assert flags["--visual-prompt"] == "auto" and flags["--defect-library"] == "auto"


def test_anomaly_types_come_from_the_counts_or_the_disk(tmp_path, monkeypatch):
    cfg = _amg_cfg(tmp_path, per_defect_counts={"A+y": 4})
    argv, _ = _run_amg(cfg, monkeypatch, _amg_rows(4))
    i = argv.index("--anomaly-types")
    assert argv[i + 1 : i + 2] == ["A+y"] and argv[i + 2].startswith("--")
    assert json.loads(_flags(argv)["--per-defect-counts"]) == {"A+y": 4}

    cfg = _amg_cfg(tmp_path)
    argv, _ = _run_amg(cfg, monkeypatch, _amg_rows(4))
    i = argv.index("--anomaly-types")
    assert argv[i + 1 : i + 3] == ["A+x", "A+y"]
    assert "--per-defect-counts" not in argv


def test_types_and_counts_argv_order_is_independent_of_config_key_order(tmp_path, monkeypatch):
    """Reordering per_defect_counts in the config must not change the argv, or a resume into the
    same amg out-dir would be refused as a settings mismatch."""
    cfg = _amg_cfg(tmp_path, per_defect_counts={"A+y": 2, "A+x": 2})
    argv, _ = _run_amg(cfg, monkeypatch, _amg_rows(4))
    i = argv.index("--anomaly-types")
    assert argv[i + 1 : i + 3] == ["A+x", "A+y"]
    assert list(json.loads(_flags(argv)["--per-defect-counts"]).items()) == [("A+x", 2), ("A+y", 2)]


def test_amg_cache_dir_overrides_the_default_location(tmp_path, monkeypatch):
    shared = tmp_path / "shared_cache"
    cfg = _amg_cfg(tmp_path, amg_cache_dir=str(shared))
    assert rp.layout(cfg)["amg_cache"] == shared

    argv, _ = _run_amg(cfg, monkeypatch, _amg_rows(4))
    assert _flags(argv)["--cache-dir"] == str(shared)


def test_amg_cache_dir_unset_defaults_under_output(tmp_path, monkeypatch):
    cfg = _amg_cfg(tmp_path)
    assert rp.layout(cfg)["amg_cache"] == tmp_path / "out" / "amg_cache"

    argv, _ = _run_amg(cfg, monkeypatch, _amg_rows(4))
    assert _flags(argv)["--cache-dir"] == str(tmp_path / "out" / "amg_cache")


def test_gemma_flags_reach_argv_exactly_when_set(tmp_path, monkeypatch):
    off, _ = _run_amg(_amg_cfg(tmp_path), monkeypatch, _amg_rows(4))
    for flag in (
        "--beam-multi",
        "--beam-pairwise",
        "--exemplar-captions",
        "--exemplar-visual-prompt",
        "--coverage-note",
    ):
        assert flag not in off
    on, _ = _run_amg(
        _amg_cfg(
            tmp_path,
            amg_beam_multi=4,
            amg_beam_pairwise=True,
            amg_exemplar_captions=True,
            amg_exemplar_visual_prompt=True,
            amg_coverage_note=True,
        ),
        monkeypatch,
        _amg_rows(4),
    )
    assert _flags(on)["--beam-multi"] == "4"
    for flag in ("--beam-pairwise", "--exemplar-captions", "--exemplar-visual-prompt", "--coverage-note"):
        assert flag in on

    negative, _ = _run_amg(_amg_cfg(tmp_path, amg_beam_multi=-1), monkeypatch, _amg_rows(4))
    assert "--beam-multi" not in negative


def test_fewer_rows_than_asked_is_an_error_naming_both_numbers(tmp_path, monkeypatch):
    """AMG writes testcase.jsonl incrementally; an interrupted run leaves a short one that exists."""
    cfg = _amg_cfg(tmp_path)
    with pytest.raises(SystemExit, match=r"3.*4|4.*3") as exc:
        _run_amg(cfg, monkeypatch, _amg_rows(3))
    assert "run.log" in str(exc.value), "the remedy must point at where a legit exit-0-short shows up"


def test_sampling_knobs_are_backfilled_into_every_row(tmp_path, monkeypatch):
    cfg = _amg_cfg(tmp_path, guidance=3.5, crop_ratio=2.0)  # num_steps left unset
    _, paths = _run_amg(cfg, monkeypatch, _amg_rows(4))
    rows = [json.loads(line) for line in paths["testcase"].read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["guidance"] == 3.5 and r["crop_ratio"] == 2.0 for r in rows)
    assert all("num_steps" not in r for r in rows), "an unset knob keeps generate.py's default"
    assert all(r["image_filename"] == f"/d/c{i}.png" for i, r in enumerate(rows)), "row order and content kept"


def test_backfill_leaves_the_overflow_file_alone(tmp_path, monkeypatch):
    cfg = _amg_cfg(tmp_path, guidance=3.5)
    paths = rp.layout(cfg)
    paths["amg"].mkdir(parents=True)
    overflow = paths["amg"] / "testcase_overflow.jsonl"
    overflow.write_text(_amg_rows(2))
    _run_amg(cfg, monkeypatch, _amg_rows(4))
    assert overflow.read_text() == _amg_rows(2)


def test_no_knobs_set_means_no_rewrite(tmp_path, monkeypatch):
    cfg = _amg_cfg(tmp_path)
    _, paths = _run_amg(cfg, monkeypatch, _amg_rows(4))
    assert paths["testcase"].read_text() == _amg_rows(4)


def _serve_cfg(tmp_path, **over):
    _mask_dirs(tmp_path, "A+x", "A+y")
    body = {"stages": ["amg"], "output": str(tmp_path / "out"), "amg_serve": "gemma-4-31b-it"}
    body.update(over)
    return rp.load_config(_cfg(tmp_path, **body))


class FakeServer:
    events = []

    def __init__(self, recipe, gpus, log_path, **kw):
        self.recipe, self.gpus, self.log_path, self.started = recipe, gpus, log_path, False

    def __enter__(self):
        FakeServer.events.append("enter")
        return self

    def __exit__(self, *exc):
        FakeServer.events.append("exit")
        return False


def test_the_merged_registry_holds_the_served_entry_and_the_users(tmp_path):
    reg = tmp_path / "endpoints.yaml"
    reg.write_text(
        yaml.safe_dump(
            [
                {"model": "m", "base_url": "http://x/v1", "key": "EMPTY"},
                {"model": "unrelated/model", "base_url": "http://z/v1", "key": "EMPTY"},
            ]
        )
    )
    cfg = _serve_cfg(tmp_path, amg_judge_model="m", amg_model_registry=str(reg))
    path = rp._effective_registry(cfg, rp.layout(cfg))
    entries = {e["model"]: e for e in yaml.safe_load(path.read_text())}
    assert entries["nvidia/google/gemma-4-31b-it"]["base_url"] == "http://localhost:8902/v1"
    assert entries["m"]["base_url"] == "http://x/v1"
    assert "unrelated/model" not in entries, "only the models this run names may be copied"
    assert path == pathlib.Path(cfg.output) / "amg_endpoints.yaml"
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_the_served_entry_carries_the_api_key_the_server_was_locked_with(tmp_path, monkeypatch):
    """The recipe script locks the server with API_KEY when set; a registry saying EMPTY would 401."""
    cfg = _serve_cfg(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    entries = {e["model"]: e for e in yaml.safe_load(rp._effective_registry(cfg, rp.layout(cfg)).read_text())}
    assert entries["nvidia/google/gemma-4-31b-it"]["key"] == "EMPTY"
    monkeypatch.setenv("API_KEY", "s3cret")
    entries = {e["model"]: e for e in yaml.safe_load(rp._effective_registry(cfg, rp.layout(cfg)).read_text())}
    assert entries["nvidia/google/gemma-4-31b-it"]["key"] == "s3cret"


def test_the_served_entry_wins_a_name_clash(tmp_path):
    reg = tmp_path / "r.yaml"
    reg.write_text(yaml.safe_dump([{"model": "nvidia/google/gemma-4-31b-it", "base_url": "http://elsewhere/v1"}]))
    cfg = _serve_cfg(tmp_path, amg_model_registry=str(reg))
    entries = {e["model"]: e for e in yaml.safe_load(rp._effective_registry(cfg, rp.layout(cfg)).read_text())}
    assert entries["nvidia/google/gemma-4-31b-it"]["base_url"] == "http://localhost:8902/v1"


def test_the_merged_registry_lands_outside_the_cli_out_dir(tmp_path):
    """The CLI's --out-dir is paths['amg']; a non-empty one with no run.json is refused as a
    corrupt resume. The merged registry must not be the thing that makes it non-empty."""
    cfg = _serve_cfg(tmp_path)
    paths = rp.layout(cfg)
    path = rp._effective_registry(cfg, paths)
    assert not paths["amg"].exists() or not any(paths["amg"].iterdir())
    assert paths["amg"] not in path.parents


def test_a_malformed_user_entry_is_refused_under_amg_serve_too(tmp_path):
    reg = tmp_path / "r.yaml"
    reg.write_text(yaml.safe_dump([{"base_url": "http://x/v1"}]))  # no 'model'
    cfg = _serve_cfg(tmp_path, amg_model_registry=str(reg))
    with pytest.raises(SystemExit, match="model"):
        rp._effective_registry(cfg, rp.layout(cfg))


def test_stage_amg_serves_around_the_cli_and_passes_the_merged_registry(tmp_path, monkeypatch):
    FakeServer.events = []
    monkeypatch.setattr(rp.serving, "Server", FakeServer)
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0", "1", "2"])
    cfg = _serve_cfg(tmp_path)
    paths = rp.layout(cfg)
    seen = []

    def _fake(cmd, log_path, env=None):
        FakeServer.events.append("run")
        seen.append([str(c) for c in cmd])
        paths["testcase"].parent.mkdir(parents=True, exist_ok=True)
        paths["testcase"].write_text(_amg_rows(4))

    monkeypatch.setattr(rp, "run", _fake)
    rp.stage_amg(cfg, paths)
    assert FakeServer.events == ["enter", "run", "exit"]
    assert _flags(seen[0])["--model-registry"] == str(pathlib.Path(cfg.output) / "amg_endpoints.yaml")


def test_the_server_gets_the_first_tp_visible_gpus(tmp_path, monkeypatch):
    captured = {}

    class Spy(FakeServer):
        def __init__(self, recipe, gpus, log_path, **kw):
            captured.update(gpus=gpus, log_path=log_path)
            super().__init__(recipe, gpus, log_path, **kw)

    monkeypatch.setattr(rp.serving, "Server", Spy)
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["3", "5", "7"])
    cfg = _serve_cfg(tmp_path)
    paths = rp.layout(cfg)
    monkeypatch.setattr(
        rp,
        "run",
        lambda cmd, log_path, env=None: (
            paths["testcase"].parent.mkdir(parents=True, exist_ok=True),
            paths["testcase"].write_text(_amg_rows(4)),
        ),
    )
    rp.stage_amg(cfg, paths)
    assert captured["gpus"] == ["3", "5"] and captured["log_path"] == paths["logs"] / "serve.log"


def test_a_gate_failure_still_stops_the_server(tmp_path, monkeypatch):
    FakeServer.events = []
    monkeypatch.setattr(rp.serving, "Server", FakeServer)
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0", "1"])
    cfg = _serve_cfg(tmp_path)
    paths = rp.layout(cfg)
    monkeypatch.setattr(
        rp,
        "run",
        lambda cmd, log_path, env=None: (
            paths["testcase"].parent.mkdir(parents=True, exist_ok=True),
            paths["testcase"].write_text(_amg_rows(3)),
        ),
    )
    with pytest.raises(SystemExit):
        rp.stage_amg(cfg, paths)
    assert FakeServer.events == ["enter", "exit"]


def test_fewer_visible_gpus_than_tp_is_a_warning_naming_both(tmp_path, monkeypatch, loguru_lines):
    FakeServer.events = []
    monkeypatch.setattr(rp.serving, "Server", FakeServer)
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0"])  # recipe wants tp=2
    cfg = _serve_cfg(tmp_path)
    paths = rp.layout(cfg)
    monkeypatch.setattr(
        rp,
        "run",
        lambda cmd, log_path, env=None: (
            paths["testcase"].parent.mkdir(parents=True, exist_ok=True),
            paths["testcase"].write_text(_amg_rows(4)),
        ),
    )
    rp.stage_amg(cfg, paths)
    assert any("WARNING" in line and "1" in line and "tp=2" in line for line in loguru_lines)


def test_a_reused_server_warns_that_it_keeps_its_gpus_for_the_run(tmp_path, monkeypatch, loguru_lines):
    """FakeServer.started is False, standing in for a reused (not started-by-us) server; a run that
    also selects generate/refine must be told those stages will see fewer GPUs."""
    FakeServer.events = []
    monkeypatch.setattr(rp.serving, "Server", FakeServer)
    monkeypatch.setattr(rp.serving, "visible_gpus", lambda: ["0", "1", "2"])
    cfg = _serve_cfg(tmp_path, stages=["amg", "generate"])
    paths = rp.layout(cfg)
    monkeypatch.setattr(
        rp,
        "run",
        lambda cmd, log_path, env=None: (
            paths["testcase"].parent.mkdir(parents=True, exist_ok=True),
            paths["testcase"].write_text(_amg_rows(4)),
        ),
    )
    rp.stage_amg(cfg, paths)
    assert any("WARNING" in line and "8902" in line for line in loguru_lines)


def test_without_amg_serve_no_server_is_touched(tmp_path, monkeypatch):
    FakeServer.events = []
    monkeypatch.setattr(rp.serving, "Server", FakeServer)
    cfg = _amg_cfg(tmp_path)
    _run_amg(cfg, monkeypatch, _amg_rows(4))
    assert FakeServer.events == []


# ── publish ───────────────────────────────────────────────────────────────────


def _export_bucket(tmp_path, images=("a.png", "b.png"), manifest_images=None):
    """An export/ bucket the way stage_export leaves one: images plus the manifest listing them."""
    out = tmp_path / "out" / "export"
    out.mkdir(parents=True)
    for name in images:
        (out / name).write_bytes(b"\x89PNG")
    listed = {n: f"src_{n}" for n in (manifest_images if manifest_images is not None else images)}
    (out / "export_manifest.json").write_text(json.dumps({"source_bucket": "sdg", "images": listed}))
    return out


def test_publish_is_last_whatever_order_the_config_uses(tmp_path):
    """Stage order is the canonical one, so an orchestrator cannot upload before exporting."""
    cfg = rp.load_config(
        _cfg(tmp_path, stages=["publish", "export", "amp"], staging={"publish": {"output": "s3://b/p"}})
    )
    assert cfg.stages == ["amp", "export", "publish"]


def test_publish_is_not_in_the_default_stages(tmp_path):
    """A default run has no destination, so it must not try to reach one."""
    assert "publish" not in rp.load_config(_cfg(tmp_path)).stages
    assert "publish" in rp.STAGES  # ... but it is still a stage the canonical order knows about


def test_a_publish_without_a_destination_fails_at_load(tmp_path):
    """publish runs last, so a missing destination found at stage time would cost the whole run."""
    with pytest.raises(SystemExit, match="staging.publish.output"):
        rp.load_config(_cfg(tmp_path, stages=["export", "publish"]))


def test_a_local_publish_destination_is_rejected(tmp_path):
    """A plain path is not an object store; staging exists only to cross that boundary."""
    with pytest.raises(SystemExit, match="not a supported object-store URL"):
        rp.load_config(_cfg(tmp_path, stages=["export", "publish"], staging={"publish": {"output": "/tmp/x"}}))


def test_credentials_are_not_a_config_field():
    """main() writes the resolved config into the output tree, which publish then uploads."""
    fields = rp._field_names()
    assert not [f for f in fields if any(w in f for w in ("token", "secret", "key", "credential", "password"))]


def test_publish_uploads_what_the_manifest_lists_not_what_is_on_disk(tmp_path, monkeypatch):
    """export/ is never cleared, so a glob would ship a previous run's images alongside this run's."""
    from anomalygen import staging

    bucket = _export_bucket(tmp_path, images=("a.png", "b.png", "stale.png"), manifest_images=("a.png", "b.png"))
    sent = []
    monkeypatch.setattr(staging, "upload", lambda local, uri: sent.append((local.name, uri)))
    published = staging.publish(bucket, "s3://bucket/run/")

    assert [n for n, _ in sent] == ["a.png", "b.png", "export_manifest.json"]
    assert "stale.png" not in published
    assert published["a.png"] == "s3://bucket/run/a.png"


def test_publish_checks_every_file_before_sending_any(tmp_path, monkeypatch):
    """A half-published prefix cannot be told apart from a complete one by the consumer."""
    from anomalygen import staging

    bucket = _export_bucket(tmp_path, images=("a.png",), manifest_images=("a.png", "gone.png"))
    sent = []
    monkeypatch.setattr(staging, "upload", lambda local, uri: sent.append(local.name))
    with pytest.raises(RuntimeError, match="not on disk"):
        staging.publish(bucket, "s3://bucket/run/")
    assert sent == []


def test_publish_refuses_an_empty_manifest(tmp_path):
    """Zero images is the guardrail-blocked case, and silently uploading nothing hides it."""
    from anomalygen import staging

    with pytest.raises(RuntimeError, match="lists no images"):
        staging.publish(_export_bucket(tmp_path, images=(), manifest_images=()), "s3://bucket/run/")


def test_the_manifest_name_is_the_same_one_export_writes():
    """Two literals for one filename would let the writer and the reader drift apart."""
    from anomalygen.staging import MANIFEST_NAME

    source = pathlib.Path(rp.__file__).read_text()
    assert f'"{MANIFEST_NAME}"' in source


def test_inline_msc_configuration_is_written_private(tmp_path, monkeypatch):
    """A fixed name in the shared temp dir would leave credentials world-readable on a shared node."""
    from anomalygen.staging import store

    monkeypatch.delenv(store.MSC_CONFIG_ENV, raising=False)
    monkeypatch.setenv(store.MSC_INLINE_ENV, json.dumps({"profiles": {"p": {}}}))
    store.configure()

    written = pathlib.Path(os.environ[store.MSC_CONFIG_ENV])
    assert json.loads(written.read_text()) == {"profiles": {"p": {}}}
    assert written.stat().st_mode & 0o077 == 0, "config is readable beyond its owner"


def test_an_explicit_msc_config_path_wins_over_the_inline_form(tmp_path, monkeypatch):
    """Overwriting the more specific instruction would make the two disagree with no way to tell."""
    from anomalygen.staging import store

    monkeypatch.setenv(store.MSC_CONFIG_ENV, "/explicit/path.json")
    monkeypatch.setenv(store.MSC_INLINE_ENV, json.dumps({"profiles": {}}))
    store.configure()
    assert os.environ[store.MSC_CONFIG_ENV] == "/explicit/path.json"


# ── prepare / fetch ───────────────────────────────────────────────────────────


def test_prepare_runs_before_every_stage_that_reads_what_it_fetched(tmp_path):
    """Canonical order, so a config cannot ask to generate before its inputs have arrived."""
    cfg = rp.load_config(
        _cfg(tmp_path, stages=["export", "prepare", "amp"], staging={"fetch": {"recipe": "s3://b/r.yaml"}})
    )
    assert cfg.stages[0] == "prepare"


def test_the_staging_stages_are_not_in_the_default_stages(tmp_path):
    """A run that names no locations must not reach for any."""
    assert {"prepare", "publish"}.isdisjoint(rp.load_config(_cfg(tmp_path)).stages)


def test_a_fetch_key_that_is_not_a_config_key_is_rejected(tmp_path):
    """The destination IS the config key, so a key with no field has nowhere to put its download."""
    with pytest.raises(SystemExit, match="not a config key"):
        rp.load_config(_cfg(tmp_path, stages=["prepare"], staging={"fetch": {"inpt": "s3://b/x"}}))


def test_a_fetch_whose_destination_field_is_unset_is_rejected(tmp_path):
    """`checkpoint` is optional, so fetching one without setting it would download into nowhere."""
    with pytest.raises(SystemExit, match="no destination"):
        rp.load_config(
            _cfg(tmp_path, stages=["prepare"], checkpoint=None, staging={"fetch": {"checkpoint": "s3://b/m.pt"}})
        )


def test_a_local_fetch_source_is_allowed_when_it_exists(tmp_path):
    """A caller that mounts the work item rather than uploading it lists the directory here."""
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    cfg = rp.load_config(_cfg(tmp_path, stages=["prepare"], staging={"fetch": {"input": str(mounted)}}))
    assert cfg.staging["fetch"]["input"] == str(mounted)


def test_a_local_fetch_source_that_is_absent_is_rejected(tmp_path):
    """Nothing fetches a local source, so one that is missing at load will still be missing later."""
    with pytest.raises(SystemExit, match="nor a path that exists"):
        rp.load_config(_cfg(tmp_path, stages=["prepare"], staging={"fetch": {"recipe": "/no/such/r.yaml"}}))


def _fetched(dest):
    """What a fetch put at ``dest``, minus the marker it leaves to claim the directory as its own."""
    from anomalygen import staging

    return sorted(
        p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file() and p.name != staging.FETCH_MARKER
    )


def test_fetching_from_the_destination_itself_is_refused(tmp_path):
    """The merge replaces the destination, so this would delete the very files being fetched."""
    from anomalygen import staging

    dest = tmp_path / "ds"
    (dest / "T").mkdir(parents=True)
    (dest / "T" / "provided.png").write_text("x")
    with pytest.raises(ValueError, match="is the destination itself"):
        staging.fetch([str(dest), "s3://b/assets/"], dest)
    assert (dest / "T" / "provided.png").exists(), "the guard must not have destroyed anything"


def test_a_mounted_work_item_merges_with_a_downloaded_prefix(tmp_path, monkeypatch):
    """The scenario where the orchestrator supplies the clean image and cad_mask on disk."""
    from anomalygen import staging

    mounted = tmp_path / "mounted" / "T" / "clean_image"
    mounted.mkdir(parents=True)
    (mounted / "j1.png").write_text("provided")

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(
        staging,
        "iter_objects",
        lambda p: iter([(p + "spec.jsonl", "spec.jsonl"), (p + "T/mask/m.png", "T/mask/m.png")]),
    )
    monkeypatch.setattr(
        staging, "download", lambda uri, local: (local.parent.mkdir(parents=True, exist_ok=True), local.write_text(uri))
    )

    dest = tmp_path / "ds"
    staging.fetch(["s3://b/assets/", str(tmp_path / "mounted")], dest)

    assert _fetched(dest) == [
        "T/clean_image/j1.png",
        "T/mask/m.png",
        "spec.jsonl",
    ]
    assert (dest / "T" / "clean_image" / "j1.png").read_text() == "provided"


def test_the_weights_preflight_waits_for_prepare(tmp_path, monkeypatch):
    """It reads the recipe and defect_spec that prepare fetches, so running it first would call a
    correctly wired container broken."""
    _weights(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(rp, "check_weights", lambda cfg: calls.append(len(calls)))
    monkeypatch.setattr(rp, "_RUNNERS", {"prepare": lambda cfg, paths: calls.append("prepare")})
    cfg_path = _cfg(
        tmp_path, stages=["prepare"], output=str(tmp_path / "out"), staging={"fetch": {"recipe": "s3://b/r.yaml"}}
    )
    rp.main(["--config", str(cfg_path)])
    assert calls == ["prepare", 1], "check_weights must run after prepare, not before"


def test_the_amg_preflight_waits_for_prepare_too(tmp_path, monkeypatch):
    """The registry file is a config key, so staging.fetch can bring it in -- which is how an
    orchestrator hands the run its model endpoint. check_amg must then read it after prepare."""
    _weights(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp, "check_amg", lambda cfg: calls.append("check_amg"))
    monkeypatch.setattr(rp, "_RUNNERS", {"prepare": lambda cfg, paths: calls.append("prepare")})
    keys = _amg_keys(tmp_path)
    keys["amg_model_registry"] = str(tmp_path / "fetched-later.yaml")  # does not exist yet
    _mask_dirs(tmp_path, "A+x")
    cfg_path = _cfg(
        tmp_path,
        stages=["prepare"],
        output=str(tmp_path / "out"),
        staging={"fetch": {"amg_model_registry": "s3://b/endpoints.yaml"}},
        **keys,
    )
    rp.main(["--config", str(cfg_path)])
    assert calls == ["prepare", "check_amg"], "check_amg must run after prepare, not before"


def test_amg_counts_are_not_checked_against_a_dataset_prepare_has_yet_to_fetch(tmp_path):
    """The mask directories the check reads do not exist until prepare has run."""
    keys = _amg_keys(tmp_path)
    cfg = rp.load_config(
        _cfg(
            tmp_path,
            stages=["prepare", "amg"],
            per_defect_counts={"A+x": 4},  # no <input>/A/mask/x/ on disk yet
            staging={"fetch": {"input": "s3://b/ds/"}},
            **keys,
        )
    )
    assert cfg.defect_counts == {"A+x": 4}


def test_a_path_prepare_will_fetch_is_not_required_to_exist_yet(tmp_path):
    """Otherwise every remote run fails at load, before the stage that would create the path."""
    body = {"output": "out", "num_sdg": 4, **_dataset(tmp_path)}
    body["recipe"] = str(tmp_path / "not-here-yet.yaml")
    body["stages"] = ["prepare", "generate"]
    body["staging"] = {"fetch": {"recipe": "s3://b/r.yaml"}}
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(body))
    assert rp.load_config(path).recipe.endswith("not-here-yet.yaml")


def test_a_path_nobody_will_fetch_is_still_required_at_load(tmp_path):
    """Deferring only the fetched keys is what keeps the typo check alive for the rest."""
    body = {"output": "out", "num_sdg": 4, **_dataset(tmp_path)}
    body["recipe"] = str(tmp_path / "typo.yaml")
    body["stages"] = ["prepare", "generate"]
    body["staging"] = {"fetch": {"input": "s3://b/ds/"}}  # input is fetched, recipe is not
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(SystemExit, match="recipe does not exist"):
        rp.load_config(path)


def test_a_deferred_path_is_checked_once_prepare_has_run(tmp_path, monkeypatch):
    """A fetch that landed the wrong layout must not reach a stage that pays to discover it."""
    _weights(tmp_path, monkeypatch)
    monkeypatch.setattr(rp, "check_weights", lambda cfg: None)
    monkeypatch.setattr(rp, "_RUNNERS", {"prepare": lambda cfg, paths: None})  # fetches nothing
    body = {"output": str(tmp_path / "out"), "num_sdg": 4, **_dataset(tmp_path)}
    body["recipe"] = str(tmp_path / "never-arrives.yaml")
    body["stages"] = ["prepare"]
    body["staging"] = {"fetch": {"recipe": "s3://b/r.yaml"}}
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(SystemExit, match="recipe does not exist"):
        rp.main(["--config", str(path)])


def test_fetch_merges_sources_in_order(tmp_path, monkeypatch):
    """Shared assets and the per-run item live in different places and become one dataset here."""
    from anomalygen import staging

    remote = {
        "s3://b/shared/": [("s3://b/shared/spec.jsonl", "spec.jsonl"), ("s3://b/shared/T/mask/m.png", "T/mask/m.png")],
        "s3://b/item/": [("s3://b/item/T/cad/c.png", "T/cad/c.png"), ("s3://b/item/spec.jsonl", "spec.jsonl")],
    }
    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter(remote[p]))
    monkeypatch.setattr(
        staging, "download", lambda uri, local: (local.parent.mkdir(parents=True, exist_ok=True), local.write_text(uri))
    )

    dest = tmp_path / "ds"
    staging.fetch(["s3://b/shared/", "s3://b/item/"], dest)

    assert _fetched(dest) == ["T/cad/c.png", "T/mask/m.png", "spec.jsonl"]
    assert (dest / "spec.jsonl").read_text() == "s3://b/item/spec.jsonl", "later source must win"


def test_a_failed_merge_leaves_the_previous_dataset_alone(tmp_path, monkeypatch):
    """A half-merged dataset does not fail, it quietly generates a smaller batch."""
    from anomalygen import staging

    dest = tmp_path / "ds"
    (dest / "T").mkdir(parents=True)
    (dest / "T" / "keep.png").write_text("previous")
    (dest / staging.FETCH_MARKER).touch()  # a re-run: the previous contents are a previous fetch's

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([(f"{p}a.png", "a.png")]))

    def boom(uri, local):
        raise RuntimeError("network went away")

    monkeypatch.setattr(staging, "download", boom)
    with pytest.raises(RuntimeError, match="network went away"):
        staging.fetch(["s3://b/x/"], dest)
    assert (dest / "T" / "keep.png").read_text() == "previous"


def test_an_empty_prefix_is_an_error_not_an_empty_dataset(tmp_path, monkeypatch):
    """A prefix that lists nothing is a typo or the wrong profile; silently producing an empty
    dataset would surface much later as a batch that generated nothing."""
    from anomalygen import staging

    monkeypatch.setattr(staging, "is_object", lambda uri: False)
    monkeypatch.setattr(staging, "iter_objects", lambda p: iter([]))
    with pytest.raises(RuntimeError, match="nothing to fetch"):
        staging.fetch(["s3://b/typo/"], tmp_path / "ds")


# ── base weights ──────────────────────────────────────────────────────────────


def test_present_weights_are_not_re_provisioned(tmp_path):
    """The mounted-volume case, which is every task after the first: nothing should happen."""
    from anomalygen.staging import ensure_base_weights

    script = tmp_path / "never.sh"  # absent on purpose: running it would fail loudly
    assert ensure_base_weights(script, tmp_path / "ckpt", lambda: []) is False


def test_provisioning_reports_what_is_missing_when_the_directory_cannot_be_made(tmp_path):
    """The motivating case is a read-only mount, which is fixed by changing the mount -- so the
    message names the absent weights and the directory rather than an errno.

    The unwritable directory is one whose parent is a FILE, not one with its permission bits cleared:
    root ignores those bits, and CI runs as root, so a chmod-based version of this passed here and
    failed there. ENOTDIR is returned whatever the uid, and it reaches the same `except OSError`."""
    from anomalygen.staging import ensure_base_weights

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    with pytest.raises(RuntimeError, match="cannot be written to"):
        ensure_base_weights(tmp_path / "s.sh", blocker / "ckpt", lambda: ["base checkpoint -> x"])


def test_the_lock_is_rechecked_before_the_download_runs(tmp_path):
    """Whoever held the lock was most likely doing this work, so waiting for it must not repeat it."""
    from anomalygen.staging import ensure_base_weights

    script = tmp_path / "s.sh"
    script.write_text("#!/bin/sh\nexit 1\n")  # would fail if it ever ran
    answers = iter([["base checkpoint -> x"], []])  # missing before the lock, satisfied after
    assert ensure_base_weights(script, tmp_path / "ckpt", lambda: next(answers)) is False


def test_a_provisioning_script_that_leaves_a_gap_is_an_error(tmp_path):
    """Exit 0 is not evidence: a licence-gated repo is skipped with a warning, not a failure."""
    from anomalygen.staging import ensure_base_weights

    script = tmp_path / "s.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    with pytest.raises(RuntimeError, match="still missing"):
        ensure_base_weights(script, tmp_path / "ckpt", lambda: ["base checkpoint -> x"])


def test_prepare_asks_for_only_what_the_selected_stages_open(tmp_path, monkeypatch):
    """A cad_tht amp-only run loads no model, so prepare must not reach for one."""
    _weights(tmp_path, monkeypatch)
    cfg = rp.load_config(_cfg(tmp_path, stages=["prepare", "amp"], staging={"fetch": {"input": "s3://b/ds/"}}))
    assert rp.missing_weights(cfg) == []


def test_missing_weights_and_the_preflight_agree(tmp_path, monkeypatch):
    """Provisioning and the check that gates the run must not disagree about what 'present' means."""
    _weights(tmp_path, monkeypatch)
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"], guardrail=True))
    listed = rp.missing_weights(cfg)
    with pytest.raises(SystemExit) as exc:
        rp.check_weights(cfg)
    assert f"{len(listed)} weight(s)" in str(exc.value)


def test_a_disagreeing_hf_cache_is_refused_before_downloading(tmp_path):
    """The script exports HF_HUB_CACHE=$CKPT_DIR/hf whatever the caller inherited, so the two can
    point at different directories -- and then nothing the download produces is ever found."""
    from anomalygen.staging import ensure_base_weights

    with pytest.raises(RuntimeError, match="always writes hub entries to"):
        ensure_base_weights(
            tmp_path / "s.sh",
            tmp_path / "ckpt",
            lambda: ["guardrail -> x"],
            hf_cache=tmp_path / "elsewhere" / "hf",
        )


def test_a_matching_hf_cache_is_accepted(tmp_path):
    ck = tmp_path / "ckpt"
    from anomalygen.staging import ensure_base_weights

    assert ensure_base_weights(tmp_path / "s.sh", ck, lambda: [], hf_cache=ck / "hf") is False


# ── object keys ───────────────────────────────────────────────────────────────


def test_a_rewritten_scheme_still_places_objects_in_their_tree(monkeypatch):
    """msc.list does not echo the URL it was given: asked for s3://bucket/x/ it answers with
    msc://_s3-bucket/x/..., resolving the bucket to an implicit profile. Comparing those against the
    caller's prefix as strings fails on exactly the backend that matters, and the version that fell
    back to the basename turned that into a dataset with every class collapsed into one directory."""
    from anomalygen.staging import store

    keys = [
        "msc://_s3-lliou/pre/fix/T/mask/b.png",
        "msc://_s3-lliou/pre/fix/T/clean/c.png",
        "msc://_s3-lliou/pre/fix/a.txt",
    ]
    monkeypatch.setattr(store, "configure", lambda: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "multistorageclient",
        type("M", (), {"list": staticmethod(lambda *a, **k: [type("I", (), {"key": k})() for k in keys])})(),
    )
    assert sorted(rel for _, rel in store.iter_objects("s3://lliou/pre/fix/")) == [
        "T/clean/c.png",
        "T/mask/b.png",
        "a.txt",
    ]


def test_a_key_outside_the_prefix_is_an_error_not_a_guess(monkeypatch):
    """Guessing wrong flattens the tree, and nothing errors until a stage reads the result."""
    from anomalygen.staging import store

    monkeypatch.setattr(store, "configure", lambda: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "multistorageclient",
        type(
            "M", (), {"list": staticmethod(lambda *a, **k: [type("I", (), {"key": "msc://p/somewhere/else.png"})()])}
        )(),
    )
    with pytest.raises(RuntimeError, match="not under it"):
        list(store.iter_objects("s3://lliou/pre/fix/"))


def test_key_path_reduces_both_url_forms_to_the_same_thing():
    from anomalygen.staging.store import _key_path

    assert _key_path("s3://bucket/a/b.png") == "a/b.png"
    assert _key_path("msc://_s3-bucket/a/b.png") == "a/b.png"
    assert _key_path("msc://profile/a/") == "a/"


def test_a_provisioning_failure_reports_what_the_script_said(tmp_path, capsys):
    """The message must name the actual failure. An earlier version asserted a licence problem, and
    the first real one it met was a digest mismatch -- which sent the reader to Hugging Face."""
    from anomalygen.staging import ensure_base_weights

    script = tmp_path / "s.sh"
    script.write_text(
        "#!/bin/sh\n"
        "echo 'downloading something large'\n"
        "echo 'ERROR: Locally-converted weights do not match assets/checkpoint_manifest_converted.sha256.' >&2\n"
        "exit 1\n"
    )
    with pytest.raises(RuntimeError) as exc:
        ensure_base_weights(script, tmp_path / "ckpt", lambda: ["base checkpoint -> x"])

    message = str(exc.value)
    assert "weights do not match" in message, "the script's own ERROR line must reach the caller"
    assert "stale record" in message, "and what that failure means"
    assert "usual cause" not in message, "no single cause may be asserted"
    assert "downloading something large" in capsys.readouterr().out, "progress must still stream"


def test_a_provisioning_failure_with_no_error_line_says_so(tmp_path):
    """Claiming a cause the script did not report is what this whole message exists to stop."""
    from anomalygen.staging import ensure_base_weights

    script = tmp_path / "s.sh"
    script.write_text("#!/bin/sh\nexit 3\n")
    with pytest.raises(RuntimeError, match="reported no ERROR line"):
        ensure_base_weights(script, tmp_path / "ckpt", lambda: ["base checkpoint -> x"])


# ── selective provisioning ────────────────────────────────────────────────────


def test_the_groups_asked_for_are_the_ones_the_script_offers(tmp_path, monkeypatch):
    """The mapping is a set of strings on one side of a subprocess boundary and a bash array on the
    other, with nothing between them to catch a rename. So ask the script itself what it accepts.

    Every group, not only the ones some fixture happens to need: a renamed block would otherwise make
    prepare exit 1 on 'unknown group' at the point a 30 GB download was meant to start."""
    known = subprocess.run(
        ["bash", str(rp._REPO_ROOT / rp._DOWNLOAD_CHECKPOINTS), "--list"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    _weights(tmp_path, monkeypatch)
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate", "evaluate"]))
    mapped = {group for _, _, group in rp.required_weights(cfg)}
    assert mapped, "the fixture must exercise some weights or this asserts nothing"
    assert mapped <= set(known), f"not offered by the script: {sorted(mapped - set(known))}"


@pytest.mark.parametrize("written, group", [("nano", "base-nano"), ("Nano", "base-nano"), ("EDGE", "base-edge")])
def test_the_base_group_survives_however_the_recipe_spelled_the_size(tmp_path, monkeypatch, written, group):
    """`model_size` is free text in the recipe, and the group name used to be interpolated from it.

    A recipe saying `Nano` therefore had prepare ask the provisioning script for `base-Nano`, which
    it rejects as an unknown group -- so nothing was downloaded, seconds after the preflight had
    named the 30 GB as missing. The size is looked up in the catalogue now.
    """
    _weights(tmp_path, monkeypatch)
    (tmp_path / "r.yaml").write_text(yaml.safe_dump({"model_size": written}))
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"]))
    groups = {g for _, _, g in rp.required_weights(cfg)}
    assert group in groups
    assert groups <= set(catalogue.GROUPS)


def test_gemma_is_asked_for_only_by_a_run_that_serves_it(tmp_path, monkeypatch):
    """62.6 GB. The amp path never opens it, and neither does an amg run against a registry endpoint
    (the server is someone else's); only `amg_serve` starts the vLLM here and so needs the weights.
    Asking for every group used to pull it on any cold run."""
    _weights(tmp_path, monkeypatch)
    amp = rp.load_config(_cfg(tmp_path, stages=[s for s in rp.STAGES if s not in ("publish", "amg")]))
    assert "gemma" not in {group for _, _, group in rp.required_weights(amp)}
    registry = rp.load_config(_cfg(tmp_path, stages=["amg"], **_amg_keys(tmp_path)))
    assert "gemma" not in {group for _, _, group in rp.required_weights(registry)}
    served = rp.load_config(_cfg(tmp_path, stages=["amg"], amg_serve="gemma-4-31b-it", **_amg_keys(tmp_path)))
    assert "gemma" in {group for _, _, group in rp.required_weights(served)}


def test_only_the_missing_groups_are_requested(tmp_path, monkeypatch):
    """A present weight must not put its group on the command line: the script would re-hash it."""
    root = _weights(tmp_path, monkeypatch, hf=_TOKENIZER + _GUARDRAIL, files=_GEN_FILES)
    cfg = rp.load_config(_cfg(tmp_path, stages=["generate"]))
    assert rp.missing_groups(cfg) == []

    (root / "checkpoints" / "wan2pt2" / "Wan2.2_VAE.pth").unlink()
    assert rp.missing_groups(cfg) == ["wan-vae"]


def test_the_groups_reach_the_script_as_arguments(tmp_path):
    """The whole point of the mapping is the argv it produces, so assert on that and not on a log."""
    from anomalygen.staging import ensure_base_weights

    script = tmp_path / "s.sh"
    script.write_text('#!/bin/sh\necho "$@" > "$CKPT_DIR/argv"\n')
    (tmp_path / "ckpt").mkdir()
    answers = iter([["a"], ["a"], []])  # missing, still missing under the lock, provisioned after
    ensure_base_weights(script, tmp_path / "ckpt", lambda: next(answers), groups=lambda: ["base-edge", "wan-vae"])
    assert (tmp_path / "ckpt" / "argv").read_text().split() == ["base-edge", "wan-vae"]


def test_no_group_mapping_still_provisions_everything(tmp_path):
    """Bare is what the script reads as 'every group'. Callers without a mapping must keep getting
    that: asking for too little is the direction that breaks a run."""
    from anomalygen.staging import ensure_base_weights

    script = tmp_path / "s.sh"
    script.write_text('#!/bin/sh\necho "[$@]" > "$CKPT_DIR/argv"\n')
    (tmp_path / "ckpt").mkdir()
    answers = iter([["a"], ["a"], []])
    ensure_base_weights(script, tmp_path / "ckpt", lambda: next(answers))
    assert (tmp_path / "ckpt" / "argv").read_text().strip() == "[]"


def test_the_publish_destination_is_checked_before_any_stage_runs(tmp_path, monkeypatch):
    """publish is last, so a typo in the bucket used to cost amp, generate and export first -- 100 s
    for one image, a whole batch for a real run, and the images stranded in a dying container."""
    _weights(tmp_path, monkeypatch, hf=_TOKENIZER + _GUARDRAIL, files=_GEN_FILES)
    monkeypatch.setattr(rp, "run", lambda *a, **k: pytest.fail("no stage may run"))
    monkeypatch.setattr(
        "anomalygen.staging.check_reachable",
        lambda uri: (_ for _ in ()).throw(RuntimeError("NoSuchBucket")),
    )
    cfg = _cfg(
        tmp_path,
        stages=["amp", "generate", "export", "publish"],
        staging={"publish": {"output": "s3://nope/out/"}},
        output=str(tmp_path / "out"),
    )
    with pytest.raises(SystemExit, match="NoSuchBucket"):
        rp.main(["--config", str(cfg)])


def test_a_run_that_does_not_publish_makes_no_network_call(tmp_path, monkeypatch):
    """The check belongs to publish, so a local run must not pay for it -- or fail on it offline."""
    _weights(tmp_path, monkeypatch, hf=_TOKENIZER + _GUARDRAIL, files=_GEN_FILES)
    monkeypatch.setattr(
        "anomalygen.staging.check_reachable",
        lambda uri: pytest.fail("no publish stage, so nothing should be probed"),
    )
    cfg = rp.load_config(_cfg(tmp_path, stages=["amp", "generate"]))
    rp.check_publish_destination(cfg)


# ── the config reference doc ──────────────────────────────────────────────────

_CONFIG_DOC = pathlib.Path("docs/pipeline/config.md")
_TABLE_BEGIN = "<!-- BEGIN generated key table -->"
_TABLE_END = "<!-- END generated key table -->"


def _doc_key_table():
    """``{key: (type cell, default cell)}`` from the doc's key table.

    Cells are stripped rather than compared as text: rumdl-fmt pads the columns to align them, so
    the formatter owns the whitespace and a test that conflated the two would fail on every reformat
    while catching nothing.

    Split on unescaped pipes only. A type cell reads `string \\| null`, and splitting on every pipe
    turns that one cell into two -- every column after it shifts, and the test then reports the doc
    as wrong about defaults it states correctly.
    """
    body = _CONFIG_DOC.read_text().split(_TABLE_BEGIN, 1)[1].split(_TABLE_END, 1)[0]
    rows = {}
    for line in body.strip().splitlines()[2:]:  # past the header and the separator
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        rows[cells[0].strip("`")] = (cells[1], cells[2])
    return rows


def test_the_doc_lists_every_config_key_and_invents_none():
    """The doc is the specification the workflow layer generates configs from, so a key missing from
    it is a key nobody will emit, and one it invents is a config the loader will reject."""
    assert set(_doc_key_table()) == rp._field_names()


def test_the_doc_states_the_defaults_the_dataclass_actually_has():
    """A default restated by hand goes stale, and a reader who trusts it writes a config built
    around a value the loader no longer uses -- silently, because the key is still valid."""
    table = _doc_key_table()
    wrong = {}
    for field in dataclasses.fields(rp.Config):
        if field.default is not dataclasses.MISSING:
            expected = f"`{json.dumps(field.default)}`"
        elif field.default_factory is not dataclasses.MISSING:
            expected = f"`{json.dumps(field.default_factory())}`"
        else:
            expected = "**required**"
        actual = table[field.name][1]
        if actual != expected:
            wrong[field.name] = f"doc says {actual}, dataclass has {expected}"
    assert not wrong, wrong


def test_the_doc_states_the_types_the_dataclass_actually_has():
    """Checked as a prefix, not an equality: the type cell may carry a hand-written value domain
    after it (`string — one of ...`), which is prose the dataclass cannot express."""
    names = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}
    hints = typing.get_type_hints(rp.Config)
    table = _doc_key_table()
    wrong = {}
    for field in dataclasses.fields(rp.Config):
        parts = typing.get_args(hints[field.name]) or (hints[field.name],)
        expected = r" \| ".join("null" if p is type(None) else names[p] for p in parts)
        actual = table[field.name][0]
        if not actual.startswith(expected):
            wrong[field.name] = f"doc says {actual!r}, dataclass has {expected!r}"
    assert not wrong, wrong


@pytest.mark.parametrize(
    ("key", "match"),
    [
        ("output", "names an output of this run"),
        ("num_sdg", "names an output of this run"),
        ("guardrail", "names an output of this run"),
        ("nonesuch", "not a config key at all"),
    ],
)
def test_only_an_input_path_can_be_a_fetch_destination(tmp_path, key, match):
    """fetch REPLACES its destination, and the destination is whatever the config key already says.
    `staging.fetch.output` therefore deleted the output tree after the resolved config and the logs
    had been written into it, and `staging.fetch.num_sdg` reached `Path(40)` and died there."""
    cfg = _cfg(tmp_path, stages=["prepare", "amp"], staging={"fetch": {key: "s3://b/x/"}})
    with pytest.raises(SystemExit, match=match):
        rp.load_config(cfg)


@pytest.mark.parametrize("key", ["input", "recipe", "checkpoint", "base_checkpoint"])
def test_the_input_paths_remain_fetchable(tmp_path, key):
    """The guard above must not cost the keys an orchestrator actually stages."""
    over = {key: str(tmp_path / "somewhere")} if key in ("base_checkpoint",) else {}
    cfg = _cfg(tmp_path, stages=["prepare", "amp"], staging={"fetch": {key: "s3://b/x/"}}, **over)
    rp.load_config(cfg)


# ── a mask stage that placed fewer masks than asked ───────────────────────────


def test_amp_refuses_a_short_testcase(tmp_path, monkeypatch):
    """roi_place drops a sample it cannot place and still exits 0, so 40 of 200 crops at
    NO_CANDIDATES wrote 160 rows and generate, evaluate, refine, export and publish all succeeded.
    The 40 missing joints then surfaced when the caller composited back onto the board."""
    cfg = rp.load_config(_cfg(tmp_path, per_defect_counts={"A+x": 4}, output=str(tmp_path / "out")))
    paths = rp.layout(cfg)
    _stub_stage_run(monkeypatch, rows=[{"image_filename": "clean_0.png"}])  # one row for four asked
    (paths["amp"]).mkdir(parents=True, exist_ok=True)
    # The statuses roi_place actually writes, read off a real summary.json rather than invented. An
    # earlier version used "SUCCESS", which matched an equally invented filter in the code -- both
    # wrong together, and the test passed anyway.
    (paths["amp"] / "summary.json").write_text(json.dumps([{"status": "OK"}] + [{"status": "NO_CANDIDATES"}] * 3))

    with pytest.raises(SystemExit) as exc:
        rp.stage_amp(cfg, paths)
    message = str(exc.value)
    assert "1 row(s)" in message and "asked for 4" in message
    assert "3x NO_CANDIDATES" in message, "the reason is already in summary.json; name it"
    reasons = message.split("ended on:")[1].split(".")[0]
    assert "OK" not in reasons, f"a placed sample is not a reason to report: {reasons!r}"


def test_amp_accepts_a_full_testcase(tmp_path, monkeypatch):
    """The gate must not cost the ordinary run."""
    cfg = rp.load_config(_cfg(tmp_path, per_defect_counts={"A+x": 4}, output=str(tmp_path / "out")))
    _stub_stage_run(monkeypatch)
    rp.stage_amp(cfg, rp.layout(cfg))


# ── a source-naming collision, decided when the masks are ─────────────────────


def test_a_source_name_collision_is_caught_at_the_mask_stage(tmp_path, monkeypatch):
    """Every collision is decidable from the testcase rows, which already name the clean image each
    output will be called after. export raises on the same condition -- after generate, evaluate and
    refine have run on a batch that was never going to export."""
    cfg = rp.load_config(
        _cfg(tmp_path, per_defect_counts={"A+x": 2}, output_naming="source", output=str(tmp_path / "out"))
    )
    _stub_stage_run(monkeypatch, rows=[{"image_filename": "c.png"}, {"image_filename": "c.png"}])
    with pytest.raises(SystemExit, match="claimed more than once"):
        rp.stage_amp(cfg, rp.layout(cfg))


def test_one_row_generating_several_images_collides_with_itself(tmp_path, monkeypatch):
    """Two rows over one clean image and one row asking for two images are the same mistake."""
    cfg = rp.load_config(
        _cfg(tmp_path, per_defect_counts={"A+x": 1}, output_naming="source", output=str(tmp_path / "out"))
    )
    _stub_stage_run(monkeypatch, rows=[{"image_filename": "c.png", "num_generated_images": 2}])
    with pytest.raises(SystemExit, match="claimed more than once"):
        rp.stage_amp(cfg, rp.layout(cfg))


def test_indexed_naming_has_no_collision_to_check(tmp_path, monkeypatch):
    """indexed names by defect type and index, so two outputs over one clean image are expected."""
    cfg = rp.load_config(
        _cfg(tmp_path, per_defect_counts={"A+x": 2}, output_naming="indexed", output=str(tmp_path / "out"))
    )
    _stub_stage_run(monkeypatch, rows=[{"image_filename": "c.png"}, {"image_filename": "c.png"}])
    rp.stage_amp(cfg, rp.layout(cfg))


# ── overriding paths inside the dataset a run fetches ─────────────────────────


def test_an_override_is_not_read_as_a_config_key(tmp_path):
    """It is the one key under staging.fetch that is not a config field: it names paths INSIDE the
    dataset `input` points at, so the "not a config key at all" guard must let it through."""
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    cfg = rp.load_config(
        _cfg(
            tmp_path,
            stages=["prepare"],
            staging={"fetch": {"input": str(mounted), "input_override": {"T/clean_image": str(mounted)}}},
        )
    )
    assert cfg.staging["fetch"]["input_override"] == {"T/clean_image": str(mounted)}


def test_an_override_without_a_fetched_dataset_is_refused(tmp_path):
    """The paths it names are paths in a tree this run fetches. With no fetch there is no tree, and
    an override applied to a mounted dataset would be deleting the operator's own files."""
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    with pytest.raises(SystemExit, match="does not fetch one"):
        rp.load_config(
            _cfg(tmp_path, stages=["prepare"], staging={"fetch": {"input_override": {"T/clean_image": str(mounted)}}})
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"T/clean_image": "/no/such/dir"}, "nor a path that exists"),
        ({"T/clean_image": []}, "lists no sources"),
        ({"../outside": "s3://b/x/"}, "INSIDE the destination"),
        ({"T/clean_image": "usecase://pcb"}, "prepares a whole dataset"),
        ("not-a-mapping", "must be a mapping"),
    ],
)
def test_an_override_is_checked_before_the_run_starts(tmp_path, override, match):
    """After the input has been fetched is too late to find out that the one line naming the images
    to generate from was misspelt -- by then the dataset it replaces has been downloaded."""
    with pytest.raises(SystemExit, match=match):
        rp.load_config(
            _cfg(tmp_path, stages=["prepare"], staging={"fetch": {"input": "s3://b/ds/", "input_override": override}})
        )


def test_the_override_goes_to_the_fetch_of_the_tree_it_addresses(tmp_path, monkeypatch):
    """Its paths are relative to `input`, so handing it to any other fetch would address a file."""
    cfg = rp.load_config(_cfg(tmp_path, stages=["prepare"], output=str(tmp_path / "out")))
    override = {"T/clean_image": ["s3://b/demo/"]}

    seen = {}
    monkeypatch.setattr(
        rp, "_fetch_spec", lambda c: {"input": ["s3://b/ds/"], "recipe": ["s3://b/r.yaml"], "input_override": override}
    )
    monkeypatch.setattr("anomalygen.staging.fetch", lambda src, dest, override=None: seen.update({str(dest): override}))
    monkeypatch.setattr("anomalygen.staging.ensure_base_weights", lambda *a, **k: False)
    rp.stage_prepare(cfg, rp.layout(cfg))

    assert seen[str(cfg.input)] == override
    assert seen[str(cfg.recipe)] is None, "the recipe is one file; an override has nowhere to go in it"


def test_a_fetch_block_that_will_not_run_says_so(tmp_path, monkeypatch):
    """`prepare` is the stage that fetches. Without it a config carrying staging.fetch downloads
    nothing and runs on whatever is already at those paths -- and for input_override that is exactly
    the outcome the key exists to prevent: the named images are absent, the dataset's own are used,
    and the run succeeds."""
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    cfg = _cfg(
        tmp_path,
        stages=["amp", "generate"],
        staging={"fetch": {"input": str(mounted), "input_override": {"T/clean_image": str(mounted)}}},
    )
    said = []
    monkeypatch.setattr(rp.log, "warning", said.append)
    rp.load_config(cfg)

    warned = "\n".join(said)
    assert "NOTHING WILL BE FETCHED" in warned
    assert "input_override" in warned, "the override is the case where silence is worst"


def test_a_resume_is_not_refused_for_carrying_its_fetch_block(tmp_path):
    """Re-running later stages over an existing output, with the same config and a shorter `stages`,
    is how a run resumes -- and that config still carries the block that filled those paths."""
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    cfg = _cfg(tmp_path, stages=["export"], staging={"fetch": {"input": str(mounted)}})
    assert rp.load_config(cfg).stages == ["export"]


def test_a_fetch_block_that_runs_warns_about_nothing(tmp_path, monkeypatch):
    """The warning must not fire for the ordinary shape, or it teaches readers to ignore it."""
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    cfg = _cfg(tmp_path, stages=["prepare", "amp"], staging={"fetch": {"input": str(mounted)}})
    said = []
    monkeypatch.setattr(rp.log, "warning", said.append)
    rp.load_config(cfg)
    assert "NOTHING WILL BE FETCHED" not in "\n".join(said)
