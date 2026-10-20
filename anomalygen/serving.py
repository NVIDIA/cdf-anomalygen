# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starting, waiting for and stopping a local VLM server for the amg stage.

WHY THIS IS NOT IN run_pipeline.py
The entrypoint's stages are thin wrappers; this module is the work behind one of them, the way
``anomalygen.staging`` is behind prepare and publish and ``anomalygen.childproc`` is behind every
stage that runs a child. Testable without a GPU: every process interaction is behind a call a test
can replace.

ONE LIFECYCLE, NOT TWO
``scripts/agentic_mask_generation_serving/serverctl.sh`` already tells a dead server from one that
is still loading (a process check beside the socket probe) and already knows how to wait for one.
Nothing here re-implements that. This module shells out to it for ``health`` and ``wait``, does the
one thing a shell cannot do for a Python caller -- start the recipe script, hold the handle, and stop
it on ANY exit from the stage -- and stops it with ``childproc.terminate_group``, the same call every
other child here is stopped with.

The recipe table is read from the scripts' own ``VAR=${VAR:-default}`` lines rather than copied, so
the script is the only place a serving knob is declared. What the entrypoint has to know BEFORE the
server runs -- GPUs it takes, port it answers on, the name it serves under -- comes from there. Which
weights it serves is the one thing the script does not spell out: it names a catalogue GROUP, and
the repository id and the files that prove the weights are present come from
``scripts/weights_catalogue.sh``, the same row the provisioner downloads against.

WHY ``Server`` ONLY STOPS WHAT IT STARTED
An operator on a dev host may have started the same server by hand, and it may still be loading
when this run begins. Either is a reason to use it, not to start a second copy on the same cards --
which would OOM the first -- and not to kill it on the way out.
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import subprocess
import time
from dataclasses import dataclass

from cosmos_framework.utils import log

from anomalygen import catalogue, childproc

SERVING_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "agentic_mask_generation_serving"

# serverctl.sh's exit-code contract (see its header). `health` uses all six; `wait` UP, DEAD, LOADING
# and LOCKED.
UP, DEAD, LOADING, OTHER_MODEL, DEAD_ENGINE, LOCKED = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class Recipe:
    script: str  # file under SERVING_DIR
    served: str  # the script's SERVED default: the name the server calls its model
    group: str  # the script's GROUP default: the catalogue row holding its weights
    hf_repo: str  # that row's repository id -- what the script loads and the provisioner fetches
    tp: int  # the script's TP default: GPUs the server takes
    port: int  # the script's PORT default


# `VAR=${VAR:-default}`, with or without `export`, with or without quotes around the expansion --
# every spelling bash reads the same way and shellcheck may ask for.
_DEFAULT = re.compile(r"^(?:export\s+)?(?P<var>[A-Z_]+)=(?P<q>[\"']?)\$\{(?P=var):-(?P<value>[^}]+?)\}(?P=q)\s*$", re.M)


def script_defaults(script: pathlib.Path) -> dict:
    """The ``VAR=${VAR:-default}`` defaults a recipe script declares, ``{VAR: default}``."""
    return {m["var"]: m["value"] for m in _DEFAULT.finditer(script.read_text())}


def recipe_names() -> list:
    """The recipes on disk: every script under SERVING_DIR except the controller. Names only -- nothing
    is parsed until :func:`recipe` asks for one, so run_pipeline imports this module without reading
    a shell script, and an edit that breaks the parse breaks only the run that names that recipe."""
    return sorted(p.stem for p in SERVING_DIR.glob("*.sh") if p.name != "serverctl.sh")


@functools.lru_cache(maxsize=None)
def recipe(name: str) -> Recipe:
    """The recipe ``name`` (a script stem under SERVING_DIR), read from its script on first use.

    Neither the repository id nor the files that prove the weights are present are declared here:
    the script names a catalogue GROUP, and both come from that row. They used to be copied -- here
    as ``hf_probe``, in the script as MODEL, in run_pipeline and in the provisioner -- and the copies
    were what drifted. See anomalygen/catalogue.py.

    Every failure is a RuntimeError naming the script and the line it wanted: a recipe is read only
    when a config names it, so this is that run's error and nobody else's.
    """
    script = f"{name}.sh"
    path = SERVING_DIR / script
    if not path.is_file():
        raise RuntimeError(f"{script}: no such serving recipe; available: {recipe_names()}")
    defaults = script_defaults(path)
    try:
        served, group, tp, port = defaults["SERVED"], defaults["GROUP"], defaults["TP"], defaults["PORT"]
    except KeyError as e:
        var = e.args[0]
        raise RuntimeError(
            f"{script}: no `{var}=${{{var}:-...}}` line; the entrypoint reads a recipe's knobs from its script"
        ) from None
    try:
        tp, port = int(tp), int(port)
    except ValueError as e:
        raise RuntimeError(f"{script}: TP and PORT defaults must be integers -- {e}") from None
    try:
        weights = catalogue.entry(group)
    except KeyError as e:
        raise RuntimeError(f"{script}: GROUP={group!r} -- {e.args[0]}") from None
    if weights.kind != "hf":
        raise RuntimeError(
            f"{script}: GROUP={group!r} is a {weights.kind} row; a served model must be a hub-cache (hf) entry"
        )
    return Recipe(script=script, served=served, group=group, hf_repo=weights.location, tp=tp, port=port)


def visible_gpus() -> list:
    """The devices this process may use, as CUDA index strings.

    ``CUDA_VISIBLE_DEVICES`` when set (an empty value means none -- that is how a CPU-only run hides
    every device); otherwise every device ``nvidia-smi`` lists; otherwise none.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [d.strip() for d in env.split(",") if d.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


class Server:
    """Start a recipe's server on ``gpus`` unless one is up or coming up, wait until it serves the
    recipe's model, stop it on exit if this instance started it.

    ``started`` says whether this instance launched the process; only then does ``__exit__`` stop it.
    The process is launched in its own session so the whole vLLM tree (engine, workers) can be
    signalled as one group -- and so its pid is its pgid, which is what ``childproc`` relies on.

    ``API_KEY`` in the environment reaches the recipe script (which locks the server with it) and
    ``serverctl.sh`` (which then probes with it) through the environment alone; ``api_key`` exposes
    the same value so the caller can write it into the registry the CLI reads.
    """

    def __init__(self, recipe, gpus, log_path, startup_timeout_s=1800, wait_slice_s=300, script_dir=SERVING_DIR):
        self.recipe, self.gpus, self.log_path = recipe, list(gpus), pathlib.Path(log_path)
        self.startup_timeout_s, self.wait_slice_s = startup_timeout_s, wait_slice_s
        self.script_dir = pathlib.Path(script_dir)
        self.started = False
        self._proc = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.recipe.port}/v1"

    @property
    def api_key(self):
        """The bearer the served model expects, or None when the server is open."""
        return os.environ.get("API_KEY") or None

    # serverctl.sh bounds its own probes (PROBE_TIMEOUT, COMPLETION_TIMEOUT); these bound the whole
    # subcommand on top, so a controller that hangs for a reason of its own cannot hang the run.
    _HEALTH_TIMEOUT_S = 180

    def _ctl(self, cmd, *extra, timeout_s=_HEALTH_TIMEOUT_S) -> tuple:
        """Run one ``serverctl.sh`` subcommand against this recipe; ``(exit code, what it said)``."""
        argv = [str(self.script_dir / "serverctl.sh"), cmd, str(self.recipe.port), self.recipe.served, *extra]
        try:
            done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"serve: serverctl.sh {cmd} did not return within {timeout_s}s") from None
        return done.returncode, done.stdout.strip()

    def _locked(self) -> SystemExit:
        have = "set" if self.api_key else "unset"
        return SystemExit(
            f"serve: the server on :{self.recipe.port} rejects our credentials (401); it was started with an "
            f"API_KEY and this run has API_KEY {have}. Export the key it was started with, or stop it."
        )

    def __enter__(self):
        state, said = self._ctl("health")
        port, served = self.recipe.port, self.recipe.served
        if state == UP:
            log.info(f"serve: reusing the {served} already serving on :{port}")
            return self
        if state == OTHER_MODEL:
            raise SystemExit(
                f"serve: :{port} is serving another model, not {served} -- {said}. Stop it or free the port."
            )
        if state == DEAD_ENGINE:
            raise SystemExit(
                f"serve: the server on :{port} answers HTTP but its engine is dead -- {said}. "
                f"Stop it (serverctl.sh stop {port}) before this run."
            )
        if state == LOCKED:
            raise self._locked()
        if state == LOADING:
            # Someone else's, still coming up. A second copy on the same cards would OOM it.
            log.info(f"serve: a server is still coming up on :{port} (not this run's); waiting for it")
        elif state == DEAD:
            self._start()
        else:
            raise SystemExit(f"serve: serverctl.sh health exited {state}: {said}")
        # `with` only calls __exit__ once __enter__ has returned. Anything that unwinds the wait --
        # the SIGTERM handler's SystemExit, a Ctrl-C, the timeout's own SystemExit -- would otherwise
        # leave a server that was started but never handed to anyone holding its GPUs. Seen once for
        # real: a killed run_pipeline left the vLLM re-parented to init.
        try:
            self._wait()
        except BaseException:
            try:
                self._stop()
            except OSError as e:
                log.warning(f"serve: stop after a failed start-up failed: {e}")
            raise
        return self

    def _start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpus)
        env["TP"] = str(len(self.gpus))
        env["PORT"] = str(self.recipe.port)
        log.info(
            f"serve: starting {self.recipe.script} on GPUs {env['CUDA_VISIBLE_DEVICES']} (TP {env['TP']}) -> {self.log_path}"
        )
        with open(self.log_path, "ab") as fp:
            self._proc = subprocess.Popen(
                [str(self.script_dir / self.recipe.script)],
                env=env,
                stdout=fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                # The script defaults HF_HUB_CACHE=$PWD/checkpoints/hf, while run_pipeline._hf_cache()
                # falls back to <repo>/checkpoints/hf. Anchor the child at the repo root (SERVING_DIR
                # is <repo>/scripts/agentic_mask_generation_serving) so both land on the same cache
                # regardless of run_pipeline's own cwd.
                cwd=str(self.script_dir.parents[1]),
            )
        self.started = True

    def _wait(self):
        """Block until the server answers with the recipe's model, in ``serverctl.sh wait`` slices.

        Slices rather than one long call so a start-up that dies is noticed within a slice, and so the
        overall budget is this module's ``startup_timeout_s``, not the script's default.
        """
        t0 = time.monotonic()
        deadline = t0 + self.startup_timeout_s
        port, served = self.recipe.port, self.recipe.served
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit(f"serve: {served} not up after {self.startup_timeout_s}s. See {self.log_path}")
            slice_s = max(1, int(min(self.wait_slice_s, remaining)))
            rc, said = self._ctl("wait", "--timeout", str(slice_s), timeout_s=slice_s + 60)
            if rc == UP:
                break
            if rc == LOCKED:
                raise self._locked()
            if rc == DEAD:
                whose = "it exited during start-up" if self.started else "the one that was loading is gone"
                raise SystemExit(f"serve: no server process on :{port} -- {whose}. See {self.log_path}")
            if rc != LOADING:
                raise SystemExit(f"serve: serverctl.sh wait exited {rc}: {said}")
        # Answering /v1/models is not the same as serving OUR model with a live engine.
        state, said = self._ctl("health")
        if state != UP:
            raise SystemExit(f"serve: the server on :{port} came up but is not usable -- {said}")
        log.info(f"serve: {served} up after {int(time.monotonic() - t0)}s")

    def _stop(self):
        if not self.started or self._proc is None:
            return
        log.info(f"serve: stopping {self.recipe.script} (pgid {self._proc.pid})")
        childproc.terminate_group(self._proc)
        self.started = False

    def __exit__(self, exc_type, exc, tb):
        try:
            self._stop()
        except OSError as e:  # already gone; the stage's own outcome is what matters
            log.warning(f"serve: stop failed: {e}")
        return False  # never swallow the stage's exception
