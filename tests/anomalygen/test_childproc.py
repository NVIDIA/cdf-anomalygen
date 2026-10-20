# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running a child without orphaning it, and its grandchildren with it."""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from anomalygen import childproc

# A launcher: prints a line so the reader is inside the loop, spawns a grandchild that would outlive
# it, prints the grandchild's pid, then sleeps. torchrun and download_checkpoints.sh are both this
# shape -- kill the launcher alone and the work carries on holding GPUs or writing a checkpoint.
_LAUNCHER = r"""
import os, subprocess, sys, time
kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(f"GRANDCHILD {kid.pid}", flush=True)
print("READY", flush=True)
time.sleep(120)
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def test_an_exception_in_the_reader_takes_the_whole_process_group(tmp_path):
    """SIGTERM becomes SystemExit inside `for line in proc.stdout`. Before this, that unwound past
    the child: torchrun kept both GPUs, and download_checkpoints.sh kept converting into the
    checkpoint tree after the lock protecting it was released."""
    import sys

    grandchild = {}

    def on_line(line):
        if line.startswith("GRANDCHILD"):
            grandchild["pid"] = int(line.split()[1])
        if line.startswith("READY"):
            raise SystemExit("cancelled, as SIGTERM does")

    with pytest.raises(SystemExit):
        childproc.stream([sys.executable, "-c", _LAUNCHER], on_line)

    assert "pid" in grandchild, "the launcher must have reported its grandchild"
    assert _wait_gone(grandchild["pid"]), "the grandchild outlived the run"


def test_a_normal_run_is_not_killed_and_reports_its_code(tmp_path):
    """The guard must not disturb the ordinary path: every line delivered, real exit code returned."""
    import sys

    seen = []
    code = childproc.stream([sys.executable, "-c", "import sys; print('a'); print('b'); sys.exit(3)"], seen.append)
    assert [line.strip() for line in seen] == ["a", "b"]
    assert code == 3


def test_stderr_arrives_on_the_same_stream_as_stdout():
    """One reader and one log. Their ORDER is the child's to decide -- a block-buffered stdout can
    land after an unbuffered stderr -- so only the merge is asserted here, not the sequence."""
    import sys

    seen = []
    childproc.stream([sys.executable, "-c", "import sys; print('out'); sys.stderr.write('err\\n')"], seen.append)
    assert sorted(line.strip() for line in seen) == ["err", "out"]


def test_terminate_group_is_a_no_op_once_the_child_has_exited():
    """Called from a finally on every path, so the ordinary one must not raise. os.getpgid would:
    it raises ProcessLookupError once wait() has reaped the session leader, which is why the pid is
    used directly -- start_new_session makes them the same number."""
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait()
    childproc.terminate_group(proc)  # must not raise


def test_a_child_that_ignores_sigterm_is_killed(tmp_path):
    """A grace period, not an indefinite wait: an orchestrator SIGKILLs the task after its own."""
    import sys

    ignorer = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('READY', flush=True); time.sleep(120)"

    def on_line(line):
        if line.startswith("READY"):
            raise SystemExit("cancelled")

    started = time.monotonic()
    with pytest.raises(SystemExit):
        childproc.stream([sys.executable, "-c", ignorer], on_line, grace_s=1)
    assert time.monotonic() - started < 10, "should have SIGKILLed rather than waited the full grace"


def test_a_launcher_that_exits_first_does_not_leave_its_ranks_running(tmp_path):
    """The leader is not the group. torchrun forks a rank per GPU and download_checkpoints.sh runs
    uvx and a converter, so the process worth killing is routinely not the one that was started --
    and `if proc.poll() is not None: return` skipped the killpg for exactly that shape.
    """
    import sys

    # The launcher spawns a child that outlives it, reports both pids, and exits at once.
    launcher = (
        "import os, subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print('RANK', c.pid, flush=True)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", launcher], stdout=subprocess.PIPE, text=True, start_new_session=True)
    rank = int(proc.stdout.readline().split()[1])
    proc.wait()  # the leader is gone and reaped; the rank is not
    assert _alive(rank)

    childproc.terminate_group(proc)
    assert _wait_gone(rank), "the rank outlived a launcher that had already exited"


def test_a_group_member_that_ignores_sigterm_is_killed_not_waited_out(tmp_path):
    """The grace period used to be spent in proc.wait() on the LEADER, so once IT exited the stop
    returned and a member holding the GPUs was never escalated to SIGKILL."""
    import sys

    launcher = (
        "import signal, subprocess, sys\n"
        "worker = 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(1, flush=True); time.sleep(120)'\n"
        "c = subprocess.Popen([sys.executable, '-c', worker])\n"
        "print('WORKER', c.pid, flush=True)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", launcher], stdout=subprocess.PIPE, text=True, start_new_session=True)
    worker = int(proc.stdout.readline().split()[1])
    proc.wait()
    assert _alive(worker)

    started = time.monotonic()
    childproc.terminate_group(proc, grace_s=1)
    assert _wait_gone(worker), "a SIGTERM-ignoring member survived the stop"
    assert time.monotonic() - started < 10


def test_a_child_still_talking_is_not_left_blocked_on_a_pipe_nobody_reads(tmp_path):
    """Nothing drains stdout once stream() is unwinding, so a chatty child blocks in write() on a
    full pipe: it never reaches its SIGTERM handler, sits out the whole grace, and is SIGKILLed for
    being slow when it was only unheard. Closing the read end makes its next write EPIPE.
    """
    import sys

    chatty = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *a: None)\n"  # handled, but only after a write returns
        "print('READY', flush=True)\n"
        "while True: sys.stdout.write('x' * 4096 + chr(10))\n"
    )

    def on_line(line):
        if line.startswith("READY"):
            raise SystemExit("cancelled")

    started = time.monotonic()
    with pytest.raises(SystemExit):
        childproc.stream([sys.executable, "-c", chatty], on_line, grace_s=20)
    assert time.monotonic() - started < 15, "waited out the grace on a child that was only unheard"


def test_the_child_is_not_reaped_before_its_group_is_signalled(monkeypatch):
    """A pid is free to be recycled the moment it is waited for, and the group id IS that pid.

    stream used to `return proc.wait()` inside the try, so by the time the finally ran the number had
    been released -- and the next child started with start_new_session becomes a group leader, so it
    can be given that exact number. The killpg would then reach it. Nothing can detect that after the
    fact, so the ordering is the guarantee, and this asserts the ordering.
    """
    import sys

    order = []
    real_wait = subprocess.Popen.wait
    real_killpg = os.killpg

    def note_wait(self, timeout=None):
        order.append("wait")
        return real_wait(self, timeout=timeout) if timeout is not None else real_wait(self)

    def note_killpg(pgid, sig):
        order.append(f"killpg:{sig}")
        return real_killpg(pgid, sig)

    monkeypatch.setattr(subprocess.Popen, "wait", note_wait)
    monkeypatch.setattr(os, "killpg", note_killpg)
    assert childproc.stream([sys.executable, "-c", "print('x')"], lambda line: None) == 0

    assert order, "neither wait nor killpg was called"
    assert order[0].startswith("killpg"), f"reaped before signalling the group: {order}"


def test_a_group_we_may_not_signal_still_yields_a_real_exit_code(monkeypatch):
    """terminate_group returns early when the group is not ours to signal, and `proc.returncode` is
    None at that point -- which run_pipeline reports as "stage failed (exit None)" for a stage that
    exited 0."""
    import sys

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(PermissionError()))
    code = childproc.stream([sys.executable, "-c", "import sys; sys.exit(7)"], lambda line: None)
    assert code == 7, f"expected the child's own code, got {code!r}"


# ── launching under torchrun ──────────────────────────────────────────────────


def test_the_launcher_is_this_interpreter_not_a_name_on_path():
    """`torchrun` is on PATH only with the venv activated. An orchestrator running
    `/repo/.venv/bin/python -m anomalygen.scripts.run_pipeline` -- the portability case the
    entrypoint exists for -- has no activated venv, so the bare name is either absent (raising
    FileNotFoundError after the mask stage has already run) or a system torchrun built against a
    different torch. The console script is a shim for this very module."""
    import sys

    cmd = childproc.torchrun(2, "/x/generate.py")
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "torchrun" not in cmd


def test_the_rendezvous_is_asked_for_explicitly():
    """Without --standalone the static backend pins MASTER_PORT=29500, and a second launch before
    the first released it fails to bind. torch >=2.6 applies the same settings implicitly for one
    node, but that is a fallback guarded by conditions one more flag could break."""
    cmd = childproc.torchrun(4, "/x/generate.py")
    assert "--standalone" in cmd
    assert "--nproc_per_node=4" in cmd
    assert cmd.index("--standalone") < cmd.index("/x/generate.py"), "launcher flags precede the script"


def test_the_parent_torchrun_variables_do_not_reach_the_child():
    """Launched from inside an outer torchrun -- which an orchestrator may well do -- the child would
    otherwise see two rendezvous configurations and report a world-size mismatch rather than
    anything that names the cause."""
    env = childproc.without_torchrun_vars(
        {
            "PATH": "/usr/bin",
            "HOME": "/home/x",
            "RANK": "0",
            "WORLD_SIZE": "8",
            "MASTER_PORT": "29500",
            "PET_NNODES": "3",
            "TORCHELASTIC_RUN_ID": "abc",
        }
    )
    assert sorted(env) == ["HOME", "PATH"]
