# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The serving module: recipe table read from the scripts, GPU discovery, and the Server lifecycle
built on serverctl.sh. Nothing here starts a process or opens a socket; Popen, serverctl.sh and the
group terminate are faked."""

import subprocess

import pytest

from anomalygen import catalogue, serving

# ── recipe table ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", serving.recipe_names())
def test_every_recipe_on_disk_parses(name):
    r = serving.recipe(name)
    assert (serving.SERVING_DIR / r.script).is_file() and r.tp >= 1 and r.port > 0


def test_recipes_are_read_lazily_so_a_broken_script_breaks_only_the_run_that_names_it(tmp_path, monkeypatch):
    """run_pipeline imports this module for every run, amp-only and --help included. Parsing every
    recipe at import turned a bash-legal edit to one script into a failure of all of them."""
    monkeypatch.setattr(serving, "SERVING_DIR", tmp_path)
    serving.recipe.cache_clear()
    (tmp_path / "serverctl.sh").write_text("#!/bin/bash\n")
    (tmp_path / "good.sh").write_text("PORT=${PORT:-9000}\nTP=${TP:-1}\nGROUP=${GROUP:-gemma}\nSERVED=${SERVED:-x}\n")
    (tmp_path / "broken.sh").write_text("PORT=${PORT:-9000}\n")  # no TP / GROUP / SERVED
    assert serving.recipe_names() == ["broken", "good"], "listing names reads no script"
    assert serving.recipe("good").port == 9000
    with pytest.raises(RuntimeError, match=r"broken\.sh"):
        serving.recipe("broken")
    with pytest.raises(RuntimeError, match=r"nope\.sh: no such serving recipe.*broken.*good"):
        serving.recipe("nope")
    serving.recipe.cache_clear()


def test_the_table_is_read_from_the_script_not_copied(tmp_path):
    """The script is the only place a serving knob is declared; the table is what it says."""
    script = tmp_path / "r.sh"
    script.write_text("PORT=${PORT:-9000}\nTP=${TP:-4}\nGROUP=${GROUP:-gemma}\nSERVED=${SERVED:-nvidia/org/model}\n")
    assert serving.script_defaults(script) == {
        "PORT": "9000",
        "TP": "4",
        "GROUP": "gemma",
        "SERVED": "nvidia/org/model",
    }


def test_every_spelling_bash_accepts_for_a_default_is_read(tmp_path):
    """Quotes around the expansion and a leading `export` are what shellcheck asks for; a recipe
    edited that way must still be a recipe."""
    script = tmp_path / "r.sh"
    script.write_text(
        'PORT="${PORT:-9000}"\nexport TP=${TP:-4}\n'
        "export GROUP='${GROUP:-gemma}'\nSERVED=${SERVED:-nvidia/org/model}\n"
        'OTHER=plain\nMODEL="$(w_location "$GROUP")"\n'
    )
    assert serving.script_defaults(script) == {
        "PORT": "9000",
        "TP": "4",
        "GROUP": "gemma",
        "SERVED": "nvidia/org/model",
    }


def _recipe_from(tmp_path, monkeypatch, body):
    monkeypatch.setattr(serving, "SERVING_DIR", tmp_path)
    serving.recipe.cache_clear()
    (tmp_path / "r.sh").write_text(body)
    try:
        return serving.recipe("r")
    finally:
        serving.recipe.cache_clear()


def test_a_script_missing_a_knob_is_an_error_naming_it(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="SERVED"):
        _recipe_from(tmp_path, monkeypatch, "PORT=${PORT:-9000}\nTP=${TP:-1}\nGROUP=${GROUP:-gemma}\n")


def test_a_non_integer_knob_is_an_error_not_a_bare_valueerror(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match=r"r\.sh: TP and PORT defaults must be integers"):
        _recipe_from(
            tmp_path, monkeypatch, "PORT=${PORT:-9000}\nTP=${TP:-two}\nGROUP=${GROUP:-gemma}\nSERVED=${SERVED:-x}\n"
        )


def test_a_script_naming_a_group_the_catalogue_lacks_is_an_error_naming_both(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match=r"r\.sh.*GROUP='llama'.*not a weights group"):
        _recipe_from(
            tmp_path, monkeypatch, "PORT=${PORT:-9000}\nTP=${TP:-1}\nGROUP=${GROUP:-llama}\nSERVED=${SERVED:-x}\n"
        )


def test_a_script_naming_a_non_hub_group_is_refused(tmp_path, monkeypatch):
    """vLLM resolves the model from the hub cache by repo id; a dir or file row cannot be served."""
    with pytest.raises(RuntimeError, match="dir row"):
        _recipe_from(
            tmp_path, monkeypatch, "PORT=${PORT:-9000}\nTP=${TP:-1}\nGROUP=${GROUP:-dinov2}\nSERVED=${SERVED:-x}\n"
        )


def test_the_gemma_recipe_carries_the_scripts_defaults_and_the_catalogues_repo():
    r = serving.recipe("gemma-4-31b-it")
    text = (serving.SERVING_DIR / r.script).read_text()
    for var, value in (("SERVED", r.served), ("GROUP", r.group), ("TP", str(r.tp)), ("PORT", str(r.port))):
        assert f"{var}=${{{var}:-{value}}}" in text
    assert "MODEL=${MODEL:-" not in text, "the repo id is the catalogue's to declare, not the script's"
    assert r.hf_repo == catalogue.entry(r.group).location


# ── visible_gpus ──────────────────────────────────────────────────────────────


def test_visible_gpus_reads_the_env_first(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,6")
    assert serving.visible_gpus() == ["1", "6"]


def test_visible_gpus_asks_nvidia_smi_when_the_env_is_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        serving.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="0\n1\n2\n", stderr="")
    )
    assert serving.visible_gpus() == ["0", "1", "2"]


def test_visible_gpus_is_empty_without_a_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(serving.subprocess, "run", boom)
    assert serving.visible_gpus() == []


def test_an_empty_env_value_means_no_gpu(monkeypatch):
    """CUDA_VISIBLE_DEVICES= (empty) is how a CPU-only test run hides every device."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert serving.visible_gpus() == []


# ── Server ────────────────────────────────────────────────────────────────────

GEMMA = serving.recipe("gemma-4-31b-it")
UP, DEAD, LOADING, OTHER, DEAD_ENGINE, LOCKED = (
    serving.UP,
    serving.DEAD,
    serving.LOADING,
    serving.OTHER_MODEL,
    serving.DEAD_ENGINE,
    serving.LOCKED,
)


class FakeProc:
    """Stands in for Popen: records the launch and reports alive/dead."""

    launches = []

    def __init__(self, argv, **kw):
        self.argv, self.kw, self.pid, self.rc = argv, kw, 4242, None
        FakeProc.launches.append(self)

    def poll(self):
        return self.rc


class Ctl:
    """Stands in for serverctl.sh: answers `health` and `wait` from two scripted sequences and records
    every call, so a test can assert what the Server asked and in which order."""

    def __init__(self, health=(), wait=()):
        self.health, self.wait, self.calls = list(health), list(wait), []

    def __call__(self, argv, **kw):
        argv = [str(a) for a in argv]
        assert argv[0].endswith("serverctl.sh"), argv
        cmd = argv[1]
        self.calls.append(argv[1:])
        rc = (self.health if cmd == "health" else self.wait).pop(0)
        if isinstance(rc, BaseException):
            raise rc
        return subprocess.CompletedProcess(argv, rc, stdout=f"{cmd} said {rc}\n", stderr="")


def _harness(monkeypatch, health, wait=()):
    FakeProc.launches = []
    ctl = Ctl(health, wait)
    monkeypatch.setattr(serving.subprocess, "run", ctl)
    monkeypatch.setattr(serving.subprocess, "Popen", FakeProc)
    stopped = []
    monkeypatch.setattr(serving.childproc, "terminate_group", lambda proc, grace_s=30: stopped.append(proc))
    return ctl, stopped


def _server(tmp_path, **kw):
    return serving.Server(GEMMA, ["0", "1"], tmp_path / "serve.log", **kw)


def test_a_server_already_up_is_reused_not_started_and_not_stopped(monkeypatch, tmp_path):
    ctl, stopped = _harness(monkeypatch, health=[UP])
    with _server(tmp_path) as s:
        assert s.started is False
        assert s.base_url == "http://localhost:8902/v1"
    assert FakeProc.launches == [] and stopped == []
    assert ctl.calls == [["health", "8902", GEMMA.served]]


def test_a_server_serving_another_model_is_an_error_not_a_restart(monkeypatch, tmp_path):
    _, stopped = _harness(monkeypatch, health=[OTHER])
    with pytest.raises(SystemExit, match="another model"):
        with _server(tmp_path):
            pass
    assert FakeProc.launches == [] and stopped == []


def test_a_server_with_a_dead_engine_is_an_error_naming_the_fix(monkeypatch, tmp_path):
    _harness(monkeypatch, health=[DEAD_ENGINE])
    with pytest.raises(SystemExit, match="engine is dead.*serverctl.sh stop"):
        with _server(tmp_path):
            pass
    assert FakeProc.launches == []


def test_a_locked_server_is_an_error_naming_the_key_not_a_wait(monkeypatch, tmp_path):
    """A server started with API_KEY=X while this run has none answers 401 to every probe. That used
    to read as "loading" and cost the full start-up timeout; it is its own exit code now."""
    monkeypatch.delenv("API_KEY", raising=False)
    _, stopped = _harness(monkeypatch, health=[LOCKED])
    with pytest.raises(SystemExit, match=r"401.*API_KEY unset"):
        with _server(tmp_path):
            pass
    assert FakeProc.launches == [] and stopped == []


def test_a_server_that_locks_itself_during_start_up_is_reported_the_same_way(monkeypatch, tmp_path):
    """Ours, started with the wrong key in the environment: `wait` sees the 401 and says so."""
    monkeypatch.setenv("API_KEY", "wrong")
    _, stopped = _harness(monkeypatch, health=[DEAD], wait=[LOCKED])
    with pytest.raises(SystemExit, match=r"401.*API_KEY set"):
        with _server(tmp_path):
            pass
    assert stopped == FakeProc.launches


def test_a_controller_that_hangs_is_a_timeout_not_a_hang(monkeypatch, tmp_path):
    def _hang(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

    monkeypatch.setattr(serving.subprocess, "run", _hang)
    with pytest.raises(SystemExit, match="serverctl.sh health did not return within"):
        with _server(tmp_path):
            pass


def test_a_server_still_loading_is_waited_for_not_duplicated(monkeypatch, tmp_path, loguru_lines):
    """A hand-started recipe spends minutes before /v1/models answers. Seeing that window as "no
    server" and starting a second copy puts two 31B servers on the same cards; one OOMs."""
    ctl, stopped = _harness(monkeypatch, health=[LOADING, UP], wait=[LOADING, UP])
    with _server(tmp_path) as s:
        assert s.started is False
    assert FakeProc.launches == [], "nothing was started"
    assert stopped == [], "and nothing is stopped: it was never ours"
    assert [c[0] for c in ctl.calls] == ["health", "wait", "wait", "health"]
    assert any("coming up" in line and "waiting" in line for line in loguru_lines)


def test_nothing_on_the_port_starts_the_recipe_with_its_env_in_a_new_session(monkeypatch, tmp_path):
    ctl, stopped = _harness(monkeypatch, health=[DEAD, UP], wait=[UP])
    with _server(tmp_path) as s:
        assert s.started is True
    (proc,) = FakeProc.launches
    assert proc.argv == [str(serving.SERVING_DIR / GEMMA.script)]
    env = proc.kw["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and env["TP"] == "2" and env["PORT"] == "8902"
    assert proc.kw["start_new_session"] is True
    assert proc.kw["cwd"] == str(serving.SERVING_DIR.parents[1])
    assert (tmp_path / "serve.log").exists()
    assert stopped == [proc], "what was started is stopped on exit, as one process group"


def test_up_on_the_socket_is_verified_with_health_before_use(monkeypatch, tmp_path):
    """`wait` only sees /v1/models answer; `health` checks the served name and a real completion."""
    _harness(monkeypatch, health=[DEAD, OTHER], wait=[UP])
    with pytest.raises(SystemExit, match="not usable"):
        with _server(tmp_path):
            pass


def test_the_wait_is_sliced_within_the_startup_budget(monkeypatch, tmp_path):
    ctl, _ = _harness(monkeypatch, health=[DEAD, UP], wait=[LOADING, LOADING, UP])
    with _server(tmp_path, startup_timeout_s=1000, wait_slice_s=400):
        pass
    slices = [int(c[c.index("--timeout") + 1]) for c in ctl.calls if c[0] == "wait"]
    assert len(slices) == 3 and all(0 < s <= 400 for s in slices)


def test_a_server_that_dies_while_loading_is_an_error_naming_the_log(monkeypatch, tmp_path):
    _, stopped = _harness(monkeypatch, health=[DEAD], wait=[DEAD])
    with pytest.raises(SystemExit, match=r"exited during start-up.*serve\.log"):
        with _server(tmp_path):
            pass
    assert stopped == FakeProc.launches, "stopped anyway: workers may outlive a dead API server"


def test_a_loading_server_that_vanishes_is_reported_as_not_ours(monkeypatch, tmp_path):
    _, stopped = _harness(monkeypatch, health=[LOADING], wait=[DEAD])
    with pytest.raises(SystemExit, match="loading is gone"):
        with _server(tmp_path):
            pass
    assert stopped == []


def test_a_startup_timeout_stops_the_server_then_errors(monkeypatch, tmp_path):
    _, stopped = _harness(monkeypatch, health=[DEAD], wait=[LOADING] * 50)
    clock = iter(range(0, 100_000, 600))  # each monotonic() call advances 10 minutes
    monkeypatch.setattr(serving.time, "monotonic", lambda: next(clock))
    with pytest.raises(SystemExit, match="not up after 1800s"):
        with _server(tmp_path, startup_timeout_s=1800):
            pass
    assert stopped == FakeProc.launches


def test_an_interrupt_during_the_wait_still_stops_what_was_started(monkeypatch, tmp_path):
    """`with` never calls __exit__ when __enter__ raises. A SIGTERM-driven SystemExit (or a Ctrl-C)
    landing in the start-up wait must therefore stop the server from inside __enter__, or the vLLM
    that was just started outlives the run holding its GPUs -- which is what a killed run_pipeline
    left behind once on a real host."""
    _, stopped = _harness(monkeypatch, health=[DEAD], wait=[SystemExit(143)])
    with pytest.raises(SystemExit) as exc:
        with _server(tmp_path):
            pytest.fail("the body must never run when start-up was interrupted")
    assert exc.value.code == 143, "the interrupt is what propagates, not a stop-side error"
    assert stopped == FakeProc.launches


def test_exit_does_not_mask_the_stage_exception(monkeypatch, tmp_path):
    _, stopped = _harness(monkeypatch, health=[DEAD, UP], wait=[UP])
    with pytest.raises(SystemExit, match="stage failed"):
        with _server(tmp_path):
            raise SystemExit("stage failed (exit 1)")
    assert stopped == FakeProc.launches


def test_stop_is_the_shared_group_terminate_not_a_pgid_lookup(monkeypatch, tmp_path):
    """The leader may already be reaped by the time we stop; a getpgid on it raises and leaves the
    workers on the GPUs. childproc.terminate_group signals by pid (== pgid in a new session)."""
    calls = []
    _harness(monkeypatch, health=[DEAD, UP], wait=[UP])
    monkeypatch.setattr(serving.childproc, "terminate_group", lambda proc, grace_s=30: calls.append(proc.pid))
    with _server(tmp_path):
        pass
    assert calls == [4242]


def test_api_key_comes_from_the_environment_the_script_and_serverctl_read(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert _server(tmp_path).api_key == "s3cret"
    monkeypatch.delenv("API_KEY")
    assert _server(tmp_path).api_key is None
