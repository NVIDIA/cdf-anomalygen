# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running a child process: starting it cleanly, and not orphaning it.

WHY THIS EXISTS
Every long-running step here is a child process read line by line:

    proc = Popen(...)
    for line in proc.stdout:
        ...

That loop can be left without finishing. k8s and Airflow cancel a job with SIGTERM, and the
entrypoint turns SIGTERM into ``SystemExit`` so a served model gets stopped -- which means the
exception unwinds through the loop above, and nothing stops the child. What it leaves behind
depends on what it was:

  a generate stage      torchrun keeps both GPUs and writes into a closed pipe
  a provisioning run    download_checkpoints.sh keeps converting into checkpoints/Cosmos3-<Size>
                        while the lock that was protecting it is released. The next task sees a
                        directory that exists, provisions nothing, and loads a half-written DCP.
                        Kill the orphan afterwards and every later run says "already present" while
                        load_for_inference fails on a missing .metadata -- with nothing to fix it.

WHY A NEW SESSION AND NOT proc.terminate()
``terminate()`` signals the child alone. These children are launchers: ``torchrun`` forks a rank per
GPU, ``download_checkpoints.sh`` runs ``uvx`` and a converter. Killing the launcher leaves the work
holding the GPUs. ``start_new_session=True`` puts the child and everything it spawns in one process
group whose id IS the child's pid, so one ``killpg`` reaches all of them -- and no ``getpgid`` lookup
is needed, which is the call that raises ``ProcessLookupError`` once ``wait()`` has reaped the leader.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

# Long enough for a launcher to pass the signal on and its ranks to exit, short enough that a
# cancelled task does not sit past the grace period an orchestrator gives it before SIGKILL.
STOP_GRACE_S = 30

# How often the group is checked while it drains. Short enough that an ordinary stop is not padded
# by it, long enough not to spin.
_POLL_S = 0.05

# Whole namespaces torchrun owns. ``PET_<ARG>`` is its env fallback for its own CLI args
# (torch.distributed.argparse_util.env): an explicit flag beats it, so the ones we pass are already
# immune. The exposure is the ones we do NOT pass, above all PET_NNODES, which leaves the child
# waiting on a multi-node rendezvous that never arrives. Stripping by prefix also covers whatever a
# future torch adds.
TORCHRUN_OWNED_PREFIXES = ("PET_", "TORCHELASTIC_")

# Set by torchrun for its children. Inheriting them from OUR environment -- when this command is
# itself run inside an outer torchrun -- makes the child see two conflicting rendezvous
# configurations, which surfaces as a world-size mismatch rather than as anything obvious.
TORCHRUN_OWNED = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def torchrun(nproc_per_node: int, script) -> list:
    """The argv prefix for launching ``script`` under torchrun, from THIS interpreter.

    ``sys.executable -m torch.distributed.run`` rather than the ``torchrun`` console script, which
    is only on PATH when the venv is activated. An orchestrator invoking
    ``/repo/.venv/bin/python -m anomalygen.scripts.run_pipeline`` -- the portability case this
    entrypoint exists for -- has no activated venv, so the bare name either is not found at all
    (FileNotFoundError, after the mask stage has already run) or resolves to a system torchrun built
    against a different torch. The console script is a shim for exactly this module:
    ``torchrun = torch.distributed.run:main``.

    ``--standalone`` asks for a c10d rendezvous on an ephemeral port with a unique run id. Without
    it the static backend pins MASTER_PORT=29500, and a second launch before the first released that
    port fails to bind -- a flake that shows up on a busy machine, not on the first run. torch >=2.6
    applies the same settings implicitly for a single node, but that is a fallback guarded by
    conditions one more flag could break; saying it states the requirement.
    """
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        "--standalone",
        str(script),
    ]


def without_torchrun_vars(environ=None) -> dict:
    """A copy of ``environ`` with everything torchrun owns removed."""
    source = os.environ if environ is None else environ
    return {k: v for k, v in source.items() if k not in TORCHRUN_OWNED and not k.startswith(TORCHRUN_OWNED_PREFIXES)}


def _group_alive(pgid: int) -> bool:
    """Whether any process is still in the group. Signal 0 checks without delivering anything."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # something is there; it is simply not ours to signal
    return True


def terminate_group(proc: subprocess.Popen, grace_s: float = STOP_GRACE_S) -> None:
    """Stop ``proc`` and everything it spawned. A no-op once the whole group has gone.

    ``proc`` must have been started with ``start_new_session=True``; its pid is then its process
    group id, so this never needs ``os.getpgid`` -- a lookup that raises once the leader is reaped
    and leaves the group running if the exception is swallowed.

    CALL THIS BEFORE REAPING THE LEADER. The group id IS the leader's pid, and a pid is only free to
    be recycled once it has been waited for -- so while the caller still holds an unreaped child, a
    zombie included, ``killpg`` cannot reach anything but this group. Reap first and the number may
    by then belong to some other process that called ``setsid``, which is one of our own later
    children. Nothing here can detect that, so the ordering is the guarantee: ``stream`` no longer
    waits before its ``finally``, and this function does its own waiting.

    THE LEADER IS NOT THE GROUP, which is what the first version of this assumed three times over.
    It returned early when ``proc.poll()`` showed the leader had exited, so a launcher that died
    while its ranks held the GPUs was never signalled at all. It then waited on the LEADER for the
    grace period, so a member that ignores SIGTERM -- ``uvx``, a converter, a worker whose handler
    only logs -- was never escalated to SIGKILL. Both are the ordinary shape of the processes this
    module exists for: they fork, and the thing worth killing is not the thing that was started.

    So the group is signalled whatever the leader is doing, and the grace period is spent watching
    the GROUP empty rather than one member of it. The leader is reaped first because a zombie
    still holds the group id, which would make the poll below never finish.
    """
    signalled = True
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        signalled = False  # the whole group is already gone
    except PermissionError:
        return  # not ours to signal, and escalating would not be either
    if not signalled and proc.poll() is not None:
        return

    deadline = time.monotonic() + grace_s
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))

    while time.monotonic() < deadline:
        if not _group_alive(proc.pid):
            break
        time.sleep(_POLL_S)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)

    if proc.poll() is None:
        proc.wait()


def stream(cmd: list, on_line, *, env: dict | None = None, grace_s: float = STOP_GRACE_S) -> int:
    """Run ``cmd``, hand each output line to ``on_line``, and return its exit code.

    stderr is folded into stdout so there is ONE stream to read and one log to keep. Their relative
    order is still the child's to decide -- a block-buffered stdout can land after an unbuffered
    stderr -- but reading them separately would mean interleaving two logs after the fact, which
    cannot be done at all.

    Whatever leaves this function -- a normal return, a failure in ``on_line``, or the SystemExit a
    SIGTERM raises -- the child and its process group are stopped first.
    """
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    try:
        for line in proc.stdout:
            on_line(line)
    finally:
        # Closed BEFORE the stop, and this is not tidiness. Nothing drains the pipe once this
        # function is unwinding, so a child that is still talking blocks in write() on a full one:
        # it never reaches its own SIGTERM handling, sits out the whole grace period, and is
        # SIGKILLed for being slow when it was only unheard. Closing the read end turns its next
        # write into EPIPE.
        with contextlib.suppress(OSError):
            proc.stdout.close()
        # Before any wait() of ours, so the pid is still held by the child -- see terminate_group.
        terminate_group(proc, grace_s)
    # Reaped here rather than trusted to have been: terminate_group returns early when the group is
    # not ours to signal, and `proc.returncode` is then None, which the caller reports as
    # "stage failed (exit None)". A no-op on every path that already waited.
    return proc.wait()
