# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provisioning the base weights, once per machine rather than once per run.

WHERE THEY HAVE TO LIVE
The paths are hardcoded relative to the repository root -- ``<repo>/checkpoints/Cosmos3-Nano``,
``<repo>/checkpoints/facebook/dinov2-large`` and so on -- so this cannot put them anywhere else and
have the code find them. That is not a limitation this module invents: run_pipeline's own preflight
already tells an operator to "mount an existing checkpoints tree at <repo>/checkpoints".

WHY IT IS A CACHE AND NOT A DOWNLOAD
A container's writable layer disappears with the container, so weights fetched into one are fetched
again by the next task -- tens of gigabytes per image generated. The directory therefore has to be a
volume that outlives the container; then the presence check below succeeds for every task after the
first, and this module does nothing at all. "Cache" means only that.

WHY THE LOCK
Fan-out puts several tasks on one node against one volume. Without a lock they race: two copies of
the provisioning script writing the same snapshot, and a presence check that can see a half-written
tree as a complete one. The lock is held across the check AND the download, and the check is repeated
once it is held, because the process that was waiting is usually waiting for exactly this work.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable
from pathlib import Path

from cosmos_framework.utils import log

from anomalygen import childproc

LOCK_NAME = ".provisioning.lock"

# What the script's own failures mean. Listed rather than ranked.
_WHAT_ITS_ERRORS_MEAN = """What the failures it reports mean:
  "weights do not match"   a digest differs from assets/checkpoint_manifest*.sha256. Against the Hub
                           manifest that is an integrity problem, because those digests are what
                           HuggingFace publishes. Against the locally-converted one it is usually a
                           stale record -- those bytes depend on the source revision and the
                           framework pin -- and the remedy is to delete THAT manifest and re-run, so
                           it records afresh.
  "does not cover"         the script's weight list grew and the manifest did not; same remedy.
  "not downloaded"         a licence not accepted on the model's Hugging Face page, or no HF_TOKEN."""


def _run_streaming(cmd: list, env: dict) -> tuple:
    """Run ``cmd``, echoing its output as it arrives, and return ``(returncode, ERROR lines)``.

    Streamed rather than captured because a cold provisioning run spends tens of minutes downloading,
    and an operator watching a silent terminal cannot tell that from a hang. The ERROR lines are kept
    as they go past so the exception can name the failure rather than point at the scrollback -- the
    one line that matters is otherwise buried under gigabytes of progress output.
    """
    notable = []

    def echo(line):
        print(line, end="")
        if line.startswith("ERROR:") or line.startswith("[FAIL]"):
            notable.append(line.rstrip())

    # Through childproc so a cancelled task cannot leave the downloader converting into the
    # checkpoint tree after the lock protecting it is gone -- the next task would then find a
    # directory that exists, provision nothing, and load a DCP still being written.
    return childproc.stream(cmd, echo, env=env), notable


def ensure_base_weights(
    script: Path,
    checkpoint_dir: Path,
    missing: Callable[[], list],
    *,
    groups: Callable[[], list] | None = None,
    hf_cache: Path | None = None,
) -> bool:
    """Provision whatever ``missing()`` still reports, under a lock. True when a download ran.

    ``missing`` is a callable rather than a list so it can be re-evaluated after the lock is taken;
    the caller owns the question of which weights this run needs, which is why it is not asked here.

    ``groups`` names the script's download blocks to ask for, and is evaluated under the lock beside
    ``missing`` so the two cannot describe different moments. Omitted -- or returning nothing -- the
    script is called bare, which provisions every group; that is the safe direction, since asking for
    too little only moves the failure to the re-check below.

    ``hf_cache`` is where the caller's check looks for hub-cache entries. It is compared against the
    one the script writes to, because they can disagree: the script exports
    ``HF_HUB_CACHE=$CKPT_DIR/hf`` unconditionally, while a caller that inherited HF_HUB_CACHE reads
    somewhere else entirely. Left to itself that downloads tens of gigabytes into a directory nothing
    will look in, and reports the same weights missing afterwards.
    """
    outstanding = missing()
    if not outstanding:
        return False

    if hf_cache is not None and Path(hf_cache).resolve() != (Path(checkpoint_dir) / "hf").resolve():
        raise RuntimeError(
            f"HF_HUB_CACHE is {hf_cache}, but {script.name} always writes hub entries to "
            f"{Path(checkpoint_dir) / 'hf'}. Provisioning would download into one directory while "
            "this run reads the other. Unset HF_HUB_CACHE so both use the checkpoints tree, or "
            "provision that cache separately and mount it."
        )

    checkpoint_dir = Path(checkpoint_dir)
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lock_path = checkpoint_dir / LOCK_NAME
        lock_path.touch()
    except OSError as exc:
        # The common cause is a read-only mount, and the useful answer is not the errno but which
        # weights are absent from where -- an operator fixes this by changing the mount, not the run.
        raise RuntimeError(
            f"{len(outstanding)} base weight(s) are missing and {checkpoint_dir} cannot be written "
            f"to ({exc}). Either mount a writable volume there, or provision it before the run:\n"
            + "\n".join(f"    {m}" for m in outstanding)
        ) from exc

    with lock_path.open("a+") as lock_file:
        log.info(f"[weights] waiting for the provisioning lock at {lock_path}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Whoever held it was most likely doing this exact work, so ask again before repeating it.
        outstanding = missing()
        if not outstanding:
            log.info("[weights] another process provisioned them while this one waited")
            return False

        log.info(f"[weights] {len(outstanding)} missing, provisioning into {checkpoint_dir}:")
        for item in outstanding:
            log.info(f"    {item}")
        # Calling the script rather than downloading here: the revision pins and the integrity
        # manifests live in it, and a second implementation would be a second place for them to
        # drift. It skips what is already present, so a warm volume costs a re-hash, not a download.
        wanted = list(groups()) if groups is not None else []
        if wanted:
            log.info(f"[weights] asking {script.name} for: {' '.join(wanted)}")
        else:
            # No arguments means every group: ≈ 140 GB, including a 62.6 GB Gemma that no stage here
            # opens. Reached only when the caller gave no group mapping, so say which run this is.
            log.warning("[weights] no group selection given — this provisions EVERY checkpoint")

        env = {**os.environ, "CKPT_DIR": str(checkpoint_dir)}
        returncode, errors = _run_streaming(["bash", str(script), *wanted], env)
        if returncode != 0:
            reported = "\n".join(f"    {e}" for e in errors) or "    (it reported no ERROR line; its output is above)"
            raise RuntimeError(f"{script} exited {returncode}. It reported:\n{reported}\n\n{_WHAT_ITS_ERRORS_MEAN}")

    still = missing()
    if still:
        raise RuntimeError(
            f"{script} finished but {len(still)} weight(s) are still missing:\n" + "\n".join(f"    {m}" for m in still)
        )
    log.success(f"[weights] provisioned -> {checkpoint_dir}")
    return True
