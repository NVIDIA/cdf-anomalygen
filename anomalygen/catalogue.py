# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The weights catalogue, read from the shell file that also provisions against it.

WHY THE SHELL FILE IS THE ORIGINAL
``scripts/weights_catalogue.sh`` is what ``download_checkpoints.sh`` and ``preflight_env_ckpt.sh``
source; both are shell, and neither can read a Python table. Declaring it there and parsing it here
costs one regex. Declaring it here and generating the shell would put a build step between a
one-line edit and the script that has to run inside an air-gapped image.

It is also the honest direction. The provisioner is the thing that decides where a weight lands;
everything on this side is a check that it landed. A check that disagrees with the provisioner is
wrong by definition, so the provisioner's copy is the one to read.

See that file's header for what a row means. This module adds nothing to it -- no defaults, no
fallbacks, no second opinion about what "present" means.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

CATALOGUE = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "weights_catalogue.sh"

# Bounded to the array, not run over the file: a regex loose enough to find rows anywhere would also
# find them in the header's worked example, and the catalogue would silently gain an entry from its
# own documentation.
_ARRAY = re.compile(r"^WEIGHTS=\(\n(?P<rows>.*?)^\)$", re.M | re.S)
_ROW = re.compile(r'^\s*"(?P<group>[^|"]+)\|(?P<kind>[^|"]+)\|(?P<location>[^|"]+)\|(?P<probes>[^|"]*)"\s*$')

KINDS = ("dir", "file", "hf")


@dataclass(frozen=True)
class Weight:
    """One row. ``location`` is under ``checkpoints/`` for dir and file, and a repo id for hf."""

    group: str
    kind: str
    location: str
    probes: tuple


def _parse(text: str) -> tuple:
    block = _ARRAY.search(text)
    if not block:
        raise RuntimeError(f"{CATALOGUE} has no WEIGHTS=( ... ) array; the catalogue cannot be read")
    rows = []
    for line in block["rows"].splitlines():
        if not line.strip():
            continue
        m = _ROW.match(line)
        if not m:
            raise RuntimeError(f"{CATALOGUE}: cannot read catalogue row {line.strip()!r}")
        if m["kind"] not in KINDS:
            raise RuntimeError(f"{CATALOGUE}: row {m['group']!r} has kind {m['kind']!r}, not one of {list(KINDS)}")
        rows.append(Weight(m["group"], m["kind"], m["location"], tuple(m["probes"].split())))
    if not rows:
        raise RuntimeError(f"{CATALOGUE}: the WEIGHTS array is empty")
    return tuple(rows)


WEIGHTS = _parse(CATALOGUE.read_text())

# Order preserved, duplicates dropped -- the same list ``download_checkpoints.sh --list`` prints,
# because it derives it from the same array the same way.
GROUPS = tuple(dict.fromkeys(w.group for w in WEIGHTS))


def entries(group: str) -> tuple:
    """Every row in ``group``. Empty is an error, because a caller naming one expects it to exist."""
    found = tuple(w for w in WEIGHTS if w.group == group)
    if not found:
        raise KeyError(f"{group!r} is not a weights group; known groups: {list(GROUPS)}")
    return found


def entry(group: str, location: str | None = None) -> Weight:
    """The single row in ``group``, or the one at ``location`` when the group holds several.

    A group with more than one row and no location given is refused rather than resolved to the
    first: the guardrail is two repositories, and answering with either would leave the caller
    checking half of what it asked about.
    """
    found = [w for w in entries(group) if location is None or w.location == location]
    if len(found) != 1:
        where = f" at {location!r}" if location else ""
        raise KeyError(f"{group!r}{where} names {len(found)} catalogue rows, not one: {[w.location for w in found]}")
    return found[0]


def hf_probes(repo: str) -> tuple:
    """The files a complete hub-cache entry for ``repo`` holds.

    These are what the loader opens, never ``config.json``: ``hf`` links each file into the snapshot
    as it completes, so the small metadata is present within seconds of a pull that may never finish.
    """
    for w in WEIGHTS:
        if w.kind == "hf" and w.location == repo:
            return w.probes
    return ()


def base_group(model_size: str) -> str:
    """The group holding the base checkpoint for ``model_size``.

    Lower-cased and then CHECKED, rather than interpolated into ``f"base-{size}"``. That is what this
    used to be, and a recipe saying ``model_size: Nano`` -- the field is free text -- made prepare
    ask the provisioning script for ``base-Nano``, which it rejects as an unknown group. The run then
    failed having downloaded nothing, seconds after the preflight had named exactly what was missing.
    """
    group = f"base-{str(model_size).strip().lower()}"
    if group not in GROUPS:
        sizes = sorted(g[len("base-") :] for g in GROUPS if g.startswith("base-"))
        raise KeyError(f"model_size {model_size!r} has no base checkpoint; known sizes: {sizes}")
    return group
