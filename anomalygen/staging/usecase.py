# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A prepared use-case dataset as a source for ``fetch``.

WHAT IT IS
``usecase://pcb`` names one of the dataset scripts this repository ships under
``anomalygen/scripts/datasets/``. The script downloads whatever its data actually comes from -- a
Hugging Face dataset repo, a pinned GitHub archive -- and writes the layout the pipeline reads:
``<TEXTURE>/{anomaly_image,mask,clean_image}/`` plus ``defect_spec.jsonl``. This module runs it and
hands the directory it produced to ``fetch`` as an ordinary local source.

WHY IT MATERIALISES RATHER THAN IMPLEMENTING A BACKEND
``hub`` and ``registry`` answer "list what is there" and then "download one of them". A prepare
script cannot be listed before it has run: it IS the thing that decides what exists. So rather than
inventing a listing that has to run the script as a side effect, the script is run once into a
directory and that directory is a local source -- which ``fetch`` already merges, places by relative
path, and installs with the same atomic swap and the same guards. Nothing downstream is new.

WHY A NAME AND NOT A PATH
The obvious spelling is the script's path, and it is the wrong one. A path is ambiguous under an
orchestrator, whose working directory is whatever the image declares -- the same trap the base
checkpoint fell into. More to the point, ``staging.fetch`` sources name DATA today, and the configs
that reach this entrypoint are increasingly generated rather than written: letting one name an
executable is a capability, not a spelling. A name resolves against what the IMAGE ships, so the
set of runnable things is decided by the build and not by the config that arrives.

A caller who wants to run a script of their own already can -- run it, then list its output
directory as a local source. That path needs nothing from this module and is not restricted by it.

WHAT A PARAMETER IS FOR, AND WHY IT IS NOT A LIST HERE
Not every dataset can be downloaded. The phone-screen images come from a Roboflow export that a
human downloads through a browser, so the script needs that file handed to it -- which is a
DEPLOYMENT value: somebody downloads it once, puts it in a bucket, and every config afterwards
points at the same object. ``usecase://phone_screen?zip=s3://bucket/mobile-screen.zip`` says so, and
``fetch`` brings the file down like any other source before the script runs.

What this module deliberately does NOT hold is a table of which script needs which parameter. That
table would be a second copy of something only the script knows, and the copy is the part that goes
stale: a fourth script added by someone who never reads this file would be offered as complete while
failing on every run. So a parameter is passed through and the script validates it -- which is why
the phone-screen script refuses a run with no zip in its own first phase, naming the download page.
Checked here: the syntax, the parameter names, and that each value is a source ``fetch`` can reach.
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cosmos_framework.utils import log

from anomalygen import childproc

SCHEME = "usecase://"

# What a prepare script leaves behind that is its own working state rather than the dataset. Both
# HF-backed scripts download with `local_dir=`, which keeps per-file metadata in `.cache/huggingface/`,
# and the magnetic-tile script keeps the raw GitHub archive under `.cache/` -- 58 MB of the 61 MB it
# produces, which then rides into `input` and is copied twice on the way. Named, and only at the top
# level: a dataset is free to carry a directory called `.cache` deeper in its tree, and dropping data
# silently is worse than carrying a few files nobody wanted.
_NOT_DATASET = frozenset({".cache"})

# The package the names resolve against, and the shape they resolve to.
_PACKAGE = "anomalygen.scripts.datasets"
_MODULE = "prepare_{name}_defect"


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts" / "datasets"


def names() -> list:
    """Every use-case this image can prepare, from what is on disk rather than from a list here."""
    return sorted(
        p.stem[len("prepare_") : -len("_defect")] for p in _scripts_dir().glob("prepare_*_defect.py") if p.is_file()
    )


# A parameter name becomes a command-line flag, so it is checked against this rather than passed on.
# `?output_dir=/etc` would otherwise give the flag this module owns a second value, and anything
# spelled with a leading dash would be a way for a config -- a GENERATED artifact here -- to put an
# arbitrary flag in an argv.
_PARAM = re.compile(r"^[a-z][a-z0-9_]*$")
_OURS = "output_dir"


@dataclass(frozen=True)
class Spec:
    """A parsed ``usecase://`` source: the script to run and what to run it with."""

    name: str
    files: dict = field(default_factory=dict)  # {parameter: source to fetch, passed as --<p> <path>}
    flags: tuple = ()  # parameters given without a value, passed as --<p>


def parse(uri: str) -> Spec:
    """The use-case behind ``uri``, checked against what this image ships.

    ``usecase://<name>[?<p>=<source>][&<p>]``. A parameter WITH a value names a file to fetch and
    hand the script; a bare one is a flag. The split is deliberate: the alternative -- guessing from
    the value whether it is a path to download or a literal to pass through -- is a guess that is
    wrong silently, and every parameter these scripts take today is a file.
    """
    rest = str(uri)[len(SCHEME) :]
    raw_name, _, query = rest.partition("?")
    name = raw_name.strip("/")
    if not name:
        raise ValueError(f"{uri!r} names no use-case. Write {SCHEME}<name>; available: {names()}.")
    if name not in names():
        raise ValueError(
            f"{uri!r} is not a use-case this image ships. Available: {names()}. The name resolves to "
            f"{_PACKAGE}.{_MODULE.format(name=name)}, so a use-case is added by adding that module, "
            "not by pointing this at a path."
        )

    files, flags = {}, []
    for item in (part for part in query.split("&") if part):
        key, has_value, value = item.partition("=")
        # A PREFIX of our own flag is refused too, not just the name. These scripts build a default
        # ArgumentParser, and argparse accepts any unambiguous abbreviation of a long option, so
        # `?output=/x` -- or `?o=/x` -- reached the child as `--output /x` and argparse read it as
        # --output_dir. The dataset then went wherever the config said, and the run failed here with
        # "wrote no files", pointing away from the cause. That is exactly the capability the comment
        # above says a generated config must not have.
        if not _PARAM.match(key) or _OURS.startswith(key):
            raise ValueError(
                f"{uri!r}: {key!r} is not a usable parameter name. A parameter becomes the flag "
                f"--{key}, so it must match {_PARAM.pattern} and must not be {_OURS!r} or an "
                "abbreviation of it -- argparse would read it as that flag."
            )
        if key in files or key in flags:
            raise ValueError(f"{uri!r}: parameter {key!r} is given twice, and only one value can be passed.")
        if not has_value:
            flags.append(key)
            continue
        if not value:
            raise ValueError(
                f"{uri!r}: parameter {key!r} has no value. Give it a source to fetch, or write it "
                f"as a bare '{key}' to pass --{key} as a flag."
            )
        if is_usecase(value):
            # It would have to be prepared into a directory to have a path at all, and what the
            # script is being handed is one file.
            raise ValueError(f"{uri!r}: parameter {key!r} cannot be another use-case ({value}).")
        files[key] = value
    return Spec(name=name, files=files, flags=tuple(flags))


def is_usecase(uri) -> bool:
    return str(uri).startswith(SCHEME)


def prepare(uri: str, into: Path, files: dict | None = None) -> Path:
    """Run the use-case's script so that ``into`` holds its dataset. Returns ``into``.

    A subprocess, not an import. These scripts report failure with ``sys.exit`` -- an import would
    turn that into a SystemExit unwinding through the middle of a fetch -- and they download tens of
    megabytes while printing progress, so a cancelled run has to be able to stop them. childproc
    gives both: the output joins this run's log, and the whole process group is stopped on the way
    out rather than left downloading into a directory nobody will read.
    """
    spec = parse(uri)
    module = f"{_PACKAGE}.{_MODULE.format(name=spec.name)}"
    into.mkdir(parents=True, exist_ok=True)
    log.info(f"[fetch] preparing use-case {spec.name!r} via {module}")

    # The caller resolves the file parameters, because fetching is what `fetch` does and this module
    # cannot import it without a cycle. Checked rather than trusted: a missing key here would reach
    # the script as a flag with no argument, which argparse reports as a usage error about a flag
    # nobody wrote.
    files = dict(files or {})
    if set(files) != set(spec.files):
        raise RuntimeError(
            f"{uri}: the file parameter(s) {sorted(spec.files)} were not all resolved; got {sorted(files)}."
        )
    # -u, because the child's stdout is a pipe and therefore block-buffered: without it the phase
    # lines these scripts print sit in that buffer until exit, so a download that takes minutes logs
    # nothing while it runs, a SIGTERM loses them entirely, and -- measured -- a failure arrives as
    # the stderr message BEFORE the phases that led to it, under a RuntimeError saying "its own
    # output above says why".
    argv = [sys.executable, "-u", "-m", module]
    for key, local in sorted(files.items()):
        argv += [f"--{key}", str(local)]
    for key in sorted(spec.flags):
        argv.append(f"--{key}")
    # LAST, so that ours wins. The check above refuses an abbreviation of --output_dir, and this is
    # the half that does not depend on knowing how argparse abbreviates: whatever a parameter is
    # matched to, a later value for the same destination replaces an earlier one.
    argv += ["--output_dir", str(into)]

    code = childproc.stream(argv, lambda line: log.info(f"  {line.rstrip()}"))
    if code != 0:
        raise RuntimeError(f"{uri}: {module} exited {code}. Its own output above says why.")

    produced = [p for p in into.rglob("*") if p.is_file()]
    if not produced:
        # Exit 0 and an empty tree is the one outcome that would otherwise pass: fetch would install
        # nothing over the destination and the run would fail later, at a stage that cannot say why.
        raise RuntimeError(f"{uri}: {module} exited 0 but wrote no files to {into}.")

    dropped = 0
    for name in sorted(_NOT_DATASET):
        work = into / name
        if work.is_dir():
            dropped += sum(1 for p in work.rglob("*") if p.is_file())
            shutil.rmtree(work)
    if dropped:
        log.info(f"[fetch] dropped {dropped} file(s) of {sorted(_NOT_DATASET)} working state")
    # Checked again, because the count above was taken before the drop: a script whose entire output
    # landed under one of those names would otherwise reach the destination as an empty dataset, past
    # the guard that exists to catch exactly that.
    kept = len(produced) - dropped
    if kept <= 0:
        raise RuntimeError(f"{uri}: {module} wrote only {sorted(_NOT_DATASET)} working state to {into}.")
    log.info(f"[fetch] use-case {spec.name!r} prepared {kept} file(s)")
    return into
