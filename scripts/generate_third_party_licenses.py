# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generate LICENSE-3rd-party.txt — the full third-party dependency closure.

Covers *everything* installed in the environment, transitive packages included, because that is
what a user actually ends up with on disk.

Two exclusions, both by definition rather than convenience:

  * NVIDIA first-party distributions (``nvidia-*``, ``cuda-*``, ``transformer-engine*``, apex,
    multi-storage-client). This is a *third-party* notice file; our own packages are not a third
    party. This mirrors what the predecessor repo did by hand in cosmos-anomalygen commit c3ed9ef
    ("Exclude NVIDIA packages from 3rd-party licenses").

    First-party authorship is not the same as first-party *content*: ``megatron-core`` and
    ``cosmos-framework`` are NVIDIA projects that incorporate third-party code, so the notice
    obligations for that code apply to the container regardless of who packaged it. Both are
    attributed here — see ``FIRST_PARTY_WITH_THIRD_PARTY_CONTENT``, which also records which
    first-party packages were audited and found to carry no such material.
  * This repo's own packages (cdf-anomalygen and the lerobot, pycountry and nltk stubs), and
    anything installed from this checkout under another name — see ``installed_from_this_repo``.

Pass ``--include-nvidia`` to emit the unfiltered closure instead.

Every license fact is read from the *installed* distribution — dist-info METADATA and the license
files the wheel ships. Nothing is asserted from memory. Where a wheel ships no license text, it is
fetched from the pinned upstream URL in ``UPSTREAM_LICENSE_URLS`` on every run — pinned to a tag or
commit, so the text is the one that shipped with the installed version rather than whatever the
default branch says today. A distribution with neither a bundled text nor a mapped URL is a hard
error rather than a silent omission. Network access is therefore required.

Run inside the environment whose contents you are attributing. For the committed
``LICENSE-3rd-party.txt`` that means the `develop` Docker image, not a local venv — a dev machine
accumulates packages the image never has.

`develop`, not `product`, is what this project releases, so its ``requirements-dev.txt`` tooling is
in scope deliberately. That file is pinned to keep this artefact reproducible:

    docker run --rm -v "$PWD:/workspace/cdf-anomalygen" -w /workspace/cdf-anomalygen \
        "$IMAGE" /opt/venv/bin/python scripts/generate_third_party_licenses.py

Write the output into the bind-mounted repo; a container-local path is lost on exit. Re-run after
changing dependencies, then diff the result before committing it.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

_APACHE_2_0 = "Apache-2.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
SEPARATOR = "-" * 80
LICENSE_FILE_RE = re.compile(r"(LICEN[CS]E|COPYING|NOTICE)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Policy data. Kept together at the top so a reviewer can audit every judgement
# call in one place rather than finding them scattered through the logic.
# ---------------------------------------------------------------------------

# Distributions whose wheel ships no license text, mapped to the upstream file(s) of record.
# Adding an entry is deliberate: cite the canonical repo, never a mirror.
# Pinned to a tag, or a commit where the repo publishes none (fvcore, pycocotools, qwen-vl-utils):
# a branch ref would rewrite the terms recorded for a version already shipped. Bump with the pin.
UPSTREAM_LICENSE_URLS: dict[str, tuple[str, ...]] = {
    "antlr4-python3-runtime": ("https://raw.githubusercontent.com/antlr/antlr4/4.9.3/LICENSE.txt",),
    "flask-cors": ("https://raw.githubusercontent.com/corydolphin/flask-cors/6.0.5/LICENSE",),
    "fvcore": (
        "https://raw.githubusercontent.com/facebookresearch/fvcore/c74b80272065b4629f7c6b8fb67d09608859b8f1/LICENSE",
    ),
    "loguru": ("https://raw.githubusercontent.com/Delgan/loguru/0.7.3/LICENSE",),
    # 0.19.1 has no tag; this is the core_r0.19.0 release-branch commit whose package_info says 0.19.1.
    "megatron-core": (
        "https://raw.githubusercontent.com/NVIDIA/Megatron-LM/8e2a184627bbbb09107acefe2df25b0c98dd8757/LICENSE",
    ),
    "mistral-common": ("https://raw.githubusercontent.com/mistralai/mistral-common/v1.11.7/LICENCE",),
    "model-hosting-container-standards": (
        "https://raw.githubusercontent.com/aws/model-hosting-container-standards/v0.1.16/LICENSE",
    ),
    "multi-storage-client": ("https://raw.githubusercontent.com/NVIDIA/multi-storage-client/1.1.0/LICENSE",),
    "obstore": ("https://raw.githubusercontent.com/developmentseed/obstore/py-v0.11.1/LICENSE",),
    "opentelemetry-semantic-conventions-ai": (
        "https://raw.githubusercontent.com/traceloop/openllmetry/ddcff1c205bf041f262a823a08ac8915cc8d156b/LICENSE",
    ),
    "pycocotools": (
        "https://raw.githubusercontent.com/ppwwyyxx/cocoapi/2971b0430e0990f119c60c8708ca66d4a5fed5d0/license.txt",
    ),
    "qwen-vl-utils": (
        "https://raw.githubusercontent.com/QwenLM/Qwen2-VL/96588727e44c78b25ba03ea03b8e12f7e64fd0da/LICENSE",
    ),
    "sentencepiece": ("https://raw.githubusercontent.com/google/sentencepiece/v0.2.2/LICENSE",),
    "tokenizers": ("https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/LICENSE",),
    "transformer-engine-torch": ("https://raw.githubusercontent.com/NVIDIA/TransformerEngine/v2.15/LICENSE",),
    "z3-solver": ("https://raw.githubusercontent.com/Z3Prover/z3/z3-4.15.4/LICENSE.txt",),
}

# Distributions that publish no license text anywhere — not in the wheel, not in a public repo.
# We record the declared license and say so plainly, rather than substituting a generic copy of
# that license and implying it came from upstream.
# Only reached under --include-nvidia (every key is NVIDIA first-party, filtered out in collect()
# before build_package consults this). A non-NVIDIA entry here would affect the shipped document.
NO_UPSTREAM_TEXT: dict[str, str] = {
    "nvidia-cusparselt-cu13": (
        "Distributed under the NVIDIA Software License Agreement. The wheel ships no license file "
        "and there is no public source repository; see "
        "https://docs.nvidia.com/cuda/cusparselt/license.html."
    ),
    "nvidia-ml-py": (
        "This distribution declares `License: BSD` (and the OSI BSD classifier) in its metadata but "
        "ships no license file, and NVIDIA publishes it to PyPI only — there is no public source "
        "repository from which the text could be quoted. Recorded as declared-but-unquotable."
    ),
    "cuda-toolkit": (
        "Distributed under the NVIDIA CUDA Toolkit End User License Agreement. The wheel ships no "
        "license file; see https://docs.nvidia.com/cuda/eula/index.html."
    ),
}

# Distributions whose METADATA declares no usable license label ("UNKNOWN", or nothing at all).
# Each SPDX ID below was read off the license text the distribution actually ships or that
# UPSTREAM_LICENSE_URLS resolves to — identified by its operative clauses, not from memory. Verify
# the same way before adding a row: an unlabelled package is not the same as an unlicensed one.
LICENSE_LABEL_OVERRIDES: dict[str, str] = {
    # 3-clause BSD: retain-notice, reproduce-in-binary, and no-endorsement-by-name.
    "apex": "BSD-3-Clause",
    # Megatron-LM is a mixed-license distribution and its own metadata is self-contradictory
    # (`License: Apache 2.0` alongside `Classifier: License :: OSI Approved :: BSD License`).
    # Per open-source review, all three apply: BSD-3-Clause for NVIDIA's own code, plus
    # Apache-2.0 and MIT for third-party components incorporated into it. Conjunctive, not a choice
    # — different files carry different terms, so a consumer must satisfy all three. The pinned
    # UPSTREAM_LICENSE_URLS entry resolves to a single LICENSE file that concatenates all three.
    "megatron-core": "BSD-3-Clause AND Apache-2.0 AND MIT",
    # Verbatim MIT permission grant and warranty disclaimer.
    "better-profanity": "MIT",
    "natten": "MIT",
    # Full Apache 2.0 text, including the TERMS AND CONDITIONS preamble.
    "sentencepiece": _APACHE_2_0,
    "transformer-engine": _APACHE_2_0,
    "transformer-engine-cu13": _APACHE_2_0,
    "transformer-engine-torch": _APACHE_2_0,
    "yacs": _APACHE_2_0,
    # No text to read; NO_UPSTREAM_TEXT explains the provenance for these.
    "cuda-toolkit": "NVIDIA CUDA Toolkit EULA",
}

# License texts a wheel ships in its *payload* rather than its .dist-info. bundled_texts() only
# reads .dist-info on purpose (a LICENSE module inside a package is code, not a notice), so any
# genuine attribution material outside it has to be named here — explicitly, one judgement per row,
# rather than by globbing the package tree.
#
# Paths are relative to the installed distribution root (i.e. site-packages).
PAYLOAD_LICENSE_FILES: dict[str, tuple[str, ...]] = {
    # matplotlib ships the DejaVu and STIX font families, whose terms are NOT in its .dist-info
    # LICENSE. Raised in open-source review.
    #
    # DejaVu is the gap: its text appears nowhere in the .dist-info metadata, so neither the font
    # terms nor the Bitstream Vera name-restriction clause were being recorded at all.
    #
    # STIX is listed too even though the .dist-info LICENSE already embeds it (under
    # "Name: Stix fonts / License: OFL-1.1"). Carrying it as its own labelled entry is what the
    # review asked for and makes the attribution findable; the duplication is harmless, and it
    # stops the entry vanishing silently if matplotlib restructures that concatenated LICENSE.
    "matplotlib": (
        "matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU",
        "matplotlib/mpl-data/fonts/ttf/LICENSE_STIX",
    ),
}

# Stanzas removed from a reproduced third-party NOTICE because the files they attribute are NOT
# shipped in this image — the notice file must describe what the container actually contains, and
# reproducing an attribution for deleted code over-claims.
#
# Each entry is (stanza title, path of the file it attributes, declaration).
#
# EVERY ENTRY MUST BE PAIRED WITH THE BUILD STEP THAT DELETES THE FILE, and the pairing is CHECKED
# IN BOTH DIRECTIONS, because only one of the two ways they can drift is self-announcing:
#
#   * upstream restructures or drops the stanza  -> redact_notice() finds nothing and raises.
#   * the build step is dropped or refactored    -> the stanza is still in the NOTICE, so stripping
#                                                   it would succeed silently while the file it
#                                                   describes is sitting in site-packages.
#
# The second is the dangerous direction: the artefact would positively assert that the file is
# absent from a container that ships it, and because the redaction runs before the copyleft gates
# read package.texts, it would also erase the declaration those gates key on. So the path is part
# of the rule and redact_notice() refuses to strip a stanza whose file is still present.
#
# Empty means no build step deletes attributed code today.
NOTICE_REDACTIONS: dict[str, tuple[tuple[str, str, str], ...]] = {}

# NVIDIA first-party distributions. Excluded from the *third-party* notice file — they are ours,
# not a third party's. Matched as exact names or `prefix-*` / `prefix_*` families.
NVIDIA_FIRST_PARTY_PREFIXES = ("nvidia-", "cuda-", "transformer-engine")
# This set answers "who wrote it", not "is it excluded". megatron-core and cosmos-framework are
# listed here AND in FIRST_PARTY_WITH_THIRD_PARTY_CONTENT below, which overrides the exclusion —
# they are still first-party, they just carry third-party content that has to be attributed.
# collect() applies both; keeping authorship and exclusion separate is what lets the second table
# state its reason per package instead of silently deleting names from this one.
NVIDIA_FIRST_PARTY_NAMES = frozenset({"apex", "megatron-core", "multi-storage-client", "cosmos-framework"})

# First-party by authorship, but they incorporate or vendor third-party code, so the container's
# attribution obligations follow the third-party content inside them regardless of who packaged it.
# These are attributed despite being first-party; the mapped reason is emitted into the notice file
# so the apparent inconsistency is self-explaining.
#
# Audited 2026-08-12 against the installed distributions:
#   megatron-core       INCLUDED — mixed BSD-3-Clause / Apache-2.0 / MIT (see LICENSE_LABEL_OVERRIDES).
#   cosmos-framework    INCLUDED — its wheel ships a NOTICE enumerating vendored third-party source
#                       (HuggingFace Transformers/Qwen, ByteDance-Seed Bagel, HuggingFace Diffusers,
#                       Detectron2). That NOTICE was being discarded entirely.
#   apex                not included — .dist-info ships only its own BSD-3-Clause LICENSE (1449 B);
#                       no NOTICE, no third-party attribution material in the wheel.
#   transformer-engine  not included — .dist-info ships only the Apache-2.0 text (10142 B); no NOTICE.
#   multi-storage-client not included — no third-party notice material in the wheel.
# Re-run the audit when any of these versions change: absence of a NOTICE is evidence about the
# wheel, not proof that the upstream project vendors nothing.
FIRST_PARTY_WITH_THIRD_PARTY_CONTENT: dict[str, str] = {
    "megatron-core": (
        "Listed although Megatron-LM is an NVIDIA project: it incorporates third-party components "
        "under Apache-2.0 and MIT alongside NVIDIA's own BSD-3-Clause code, so the notice "
        "obligations for those components apply to this distribution."
    ),
    "cosmos-framework": (
        "Listed although Cosmos is an NVIDIA project: its NOTICE (reproduced below) attributes "
        "third-party source vendored into the package, so those notices travel with the container. "
        "It covers the vendored source this image ships; see NOTICE_REDACTIONS in "
        "scripts/generate_third_party_licenses.py for any stanza omitted because the file it "
        "attributes is removed during the build."
    ),
}

# Copyleft policy. Two gates, because copyleft reaches the image by two routes.
#
# 1. The package's own metadata label — what a distribution DECLARES about itself.
# 2. The `License:` declaration lines inside the notice texts it ships — what it declares about
#    third-party code it vendors. A wheel that bundles someone else's binaries labels itself in
#    METADATA and describes the bundled ones in its notice, so gate 1 is blind to them by
#    construction. That blind spot is not hypothetical: vendored GPL-3.0 source reached this image
#    once already behind a non-copyleft package label, and only a human review caught it.
STRONG_COPYLEFT = re.compile(r"\bAGPL|\bGPL-?[23]|GNU General Public|\bGPLv[23]|\bSSPL", re.I)
WEAK_COPYLEFT = re.compile(r"\bLGPL|Lesser General Public|\bMPL|Mozilla Public|\bEPL\b|Eclipse Public|\bCDDL", re.I)

# Gate 2 reads ONLY these declaration lines, never the full text. License bodies cross-reference
# each other — MPL-2.0's Exhibit wording and every LGPL text name the GNU GPL — so a full-text scan
# would fire on packages we already ship, and a gate that cries wolf is a gate someone switches off.
#
# SCOPE, stated plainly because the narrow form fails silently and that is the worse direction for
# a gate: this catches STRUCTURED, Debian-style `License:` declarations on a single line. It does
# NOT catch, and these still need human eyes on a dependency bump:
#   * dual-licensing described in PROSE. pillow ships the same FreeType as matplotlib under the
#     same terms, in prose, with no `License:` line anywhere in its section — matplotlib trips the
#     gate and pillow does not. Neither is a compliance problem; the asymmetry is the point.
#   * a value on the FOLLOWING line (`License:\n  <text>`), which `(.+)` cannot see. Two such
#     stanzas already exist in the artefact; both are non-copyleft today.
# The blind spot is narrowed, not closed. Read this gate as "no structured copyleft declaration
# went unrecorded", never as "the vendored-copyleft question has been swept".
NOTICE_DECLARED_LICENSE_RE = re.compile(r"^[ \t]*License:[ \t]*(.+)$", re.M)

# Vendored copyleft we knowingly ship: package -> (exact declarations accepted, why).
#
# Acceptance is per DECLARATION, not per package. Keying on the package alone would mean that a
# dependency accepted once for one bundled library ships anything it vendors later unnoticed — a
# numpy bump that added a genuinely non-exempt GPL binary would pass on the strength of the
# libgfortran decision. The strings must match the notice's declaration exactly.
#
# Same bar as ACCEPTED_WEAK_COPYLEFT: a licensing decision, recorded, not a formality. The reason
# is emitted into the notice file, so an auditor reading only the container sees the basis.
ACCEPTED_VENDORED_COPYLEFT: dict[str, tuple[frozenset[str], str]] = {
    "numpy": (
        frozenset({"GPL-3.0-or-later WITH GCC-exception-3.1", "LGPL-2.1-or-later"}),
        "Bundles libgfortran (GPL-3.0-or-later WITH GCC-exception-3.1) and libquadmath "
        "(LGPL-2.1-or-later) in numpy.libs, both unmodified from the upstream wheel. The GCC "
        "Runtime Library Exception exists precisely so that linking libgfortran imposes no GPL "
        "obligation on the linked work; libquadmath is dynamically linked and unmodified.",
    ),
    "scipy": (
        frozenset({"GPL-3.0-or-later WITH GCC-exception-3.1", "LGPL-2.1-or-later"}),
        "Bundles libgfortran and libquadmath in scipy.libs — the same upstream binaries on the "
        "same terms as numpy above, and accepted on the same basis.",
    ),
    "matplotlib": (
        frozenset({"FTL OR GPL-2.0-or-later"}),
        "Bundles FreeType, offered as `FTL OR GPL-2.0-or-later`. Unlike a conjunctive `AND` "
        "expression this is a genuine choice of terms, and the FreeType License is the one taken. "
        "Recorded rather than relabelled, so the choice upstream offers stays visible.",
    ),
}

# Weak-copyleft dependencies we knowingly ship. Both are MPL-2.0 (file-level, and
# unmodified here) and neither is removable: requests hard-imports certifi, and
# transformers/datasets/peft hard-import tqdm. Adding to this list is a licensing
# decision, not a formality — record why.
#
# tqdm specifically: its metadata expression is `MPL-2.0 AND MIT`, which is CONJUNCTIVE. In SPDX,
# AND means both apply to different parts of the work, not that a licensee may choose. Its LICENCE
# confirms it — `files: *` is MPL-2.0, and MIT survives only for `tqdm/_tqdm.py`, `README.rst` and
# `.gitignore`. So there is nothing to elect: the MPL-2.0 obligations attach to the container and
# tqdm belongs here, recorded, rather than relabelled.
ACCEPTED_WEAK_COPYLEFT = frozenset({"certifi", "tqdm"})

# This repo's own packages — never a dependency of anything, never attributed.
LOCAL_PACKAGES = frozenset({"cdf-anomalygen", "lerobot-stub", "pycountry-stub", "nltk-stub"})


# ---------------------------------------------------------------------------
# License discovery
# ---------------------------------------------------------------------------


def normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_nvidia_first_party(name: str) -> bool:
    name = normalize(name)
    return name in NVIDIA_FIRST_PARTY_NAMES or name.startswith(NVIDIA_FIRST_PARTY_PREFIXES)


@dataclass
class Package:
    """A distribution and everything needed to attribute it."""

    name: str
    version: str
    author: str
    license_label: str
    url: str
    texts: list[tuple[str, str]] = field(default_factory=list)  # (source label, verbatim text)
    note: str = ""
    # Copyleft this package DECLARES about code it vendors, read from the notices it ships itself.
    # Populated only from locally-shipped texts, never from an upstream text fetched over the
    # network — see build_package. Gate 2 in main() checks it.
    vendored_copyleft: set[str] = field(default_factory=set)

    @property
    def sort_key(self) -> str:
        return self.name.lower()


def fetch_license_text(url: str, dist_name: str) -> str:
    """Fetch an upstream license text from the pinned ref in UPSTREAM_LICENSE_URLS.

    Fetched live rather than vendored, but every URL points at an immutable tag or commit, so the
    result is the text that shipped with the installed version and does not drift when upstream
    edits its default branch.
    """
    if not url.startswith("https://"):
        raise ValueError(f"refusing to fetch a non-https license URL: {url}")
    # urlopen raises on non-2xx, so there is no status to check — but it never says which package.
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310  (https enforced above)
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"{dist_name}: could not fetch the pinned license URL {url} ({exc}). "
            "Check that the tag or commit still exists upstream."
        ) from exc


def installed_distributions() -> dict[str, metadata.Distribution]:
    """Normalized name -> Distribution for everything importable in the active environment.

    An editable install can leave both a .dist-info and a stale .egg-info for the same project;
    keeping the first occurrence is enough since the metadata we read is identical.
    """
    found: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            found.setdefault(normalize(name), dist)
    return found


def installed_from_this_repo(dist: metadata.Distribution) -> bool:
    """True for an install of this checkout, whatever project name it was installed under.

    ``LOCAL_PACKAGES`` matches by name, which misses a checkout installed before a rename: its old
    metadata is still importable and would be attributed as an unknown third party. Two traces
    identify such an install without naming it:
      * its metadata directory sits directly in the repo root (the ``*.egg-info`` an editable
        build leaves there) — not anywhere under it, since a local ``.venv`` lives there too;
      * its ``direct_url.json`` records an editable install of the repo root itself — not of a
        directory under it, which could hold a vendored third-party project.
    """
    if Path(str(dist.locate_file(""))).resolve() == REPO_ROOT:
        return True
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return False
    origin = json.loads(direct_url)
    url = urllib.parse.urlparse(origin.get("url", ""))
    if url.scheme != "file" or not origin.get("dir_info", {}).get("editable"):
        return False
    return Path(urllib.request.url2pathname(url.path)).resolve() == REPO_ROOT


def bundled_texts(dist: metadata.Distribution) -> list[tuple[str, str]]:
    """License/notice files the wheel itself ships, as (filename, verbatim text)."""
    found = []
    for file in dist.files or []:
        path = Path(str(file))
        # Only files inside the .dist-info directory; a LICENSE module inside the package is code.
        if not any(part.endswith(".dist-info") for part in path.parts):
            continue
        if not LICENSE_FILE_RE.search(path.name):
            continue
        try:
            text = dist.locate_file(file).read_text(errors="replace")  # type: ignore[union-attr]
        except OSError:
            continue
        if text.strip():
            found.append((path.name, text))
    return sorted(set(found))


def payload_texts(dist: metadata.Distribution, normalized: str) -> list[tuple[str, str]]:
    """License texts the wheel ships outside .dist-info, named in PAYLOAD_LICENSE_FILES.

    A missing path is a hard error rather than a silent skip: the entry exists because someone
    established the file carries terms we must reproduce, so it disappearing (a restructured wheel,
    a version bump) has to surface here and not as a quietly shorter notice file.
    """
    found = []
    for relative in PAYLOAD_LICENSE_FILES.get(normalized, ()):
        try:
            text = dist.locate_file(relative).read_text(errors="replace")  # type: ignore[union-attr]
        except OSError as exc:
            raise RuntimeError(
                f"{normalized}: PAYLOAD_LICENSE_FILES names {relative}, which this installed "
                f"distribution does not ship ({exc}). Confirm where the terms moved and update the "
                "entry — do not just delete it."
            ) from exc
        if not text.strip():
            raise RuntimeError(f"{normalized}: PAYLOAD_LICENSE_FILES entry {relative} is empty.")
        found.append((relative, text))
    return found


def redact_notice(
    dist: metadata.Distribution, texts: list[tuple[str, str]], normalized: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """Strip NOTICE_REDACTIONS stanzas, returning the edited texts and the declarations to record.

    Stanzas are the blank-line-separated blocks these NOTICE files are written in; a stanza is
    matched by its first line being exactly the configured title.

    Two hard errors, one for each way the rule and the build step that deletes the file can drift
    apart. Both need a human; neither may degrade to a silent no-op:

    * The file is STILL PRESENT in the environment being attributed. The deletion did not run, so
      removing its attribution would make the notice file under-claim what the container ships.
      This is the check the stanza-title match cannot make on its own — upstream's NOTICE lists the
      file whether or not we deleted it, so stripping always "works".
    * The stanza is NOT FOUND. Either upstream restructured the NOTICE (the rule needs rewriting)
      or the attribution is already gone (the rule should be retired).
    """
    rules = NOTICE_REDACTIONS.get(normalized, ())
    if not rules:
        return texts, []

    declarations: list[str] = []
    edited = list(texts)
    for title, path, declaration in rules:
        # Order matters: verify absence BEFORE stripping, so a failed build step is reported as
        # itself rather than as a downstream mismatch.
        try:
            still_present = dist.locate_file(path).exists()  # type: ignore[union-attr]
        except OSError:
            still_present = False
        if still_present:
            raise RuntimeError(
                f"{normalized}: NOTICE_REDACTIONS drops the '{title}' stanza, but {path} IS present "
                "in the environment being attributed. The build step that deletes it did not run — "
                "refusing to write a notice file that under-claims what the container ships. "
                "Generate from the develop image (see this module's docstring), not a local venv."
            )
        hit = False
        for index, (label, text) in enumerate(edited):
            blocks = text.split("\n\n")
            kept = [b for b in blocks if b.strip().splitlines()[:1] != [title]]
            if len(kept) != len(blocks):
                edited[index] = (label, "\n\n".join(kept))
                hit = True
        if not hit:
            raise RuntimeError(
                f"{normalized}: NOTICE_REDACTIONS expects a '{title}' stanza, which none of this "
                f"distribution's notice texts contains. Upstream may have restructured or already "
                "removed it — reconcile this entry with the build step that deletes the file."
            )
        declarations.append(declaration)
    return edited, declarations


def license_label(meta) -> str:
    """Best available license label: SPDX expression > License field > OSI classifiers."""
    for key in ("License-Expression", "License"):
        value = meta.get(key)
        if value and len(value.splitlines()) == 1:
            return value.strip()
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        # " OR ", not " AND ". Multiple classifiers express a CHOICE of licence, so
        # joining them conjunctively would assert the consumer must satisfy all of
        # them at once — the opposite of what dual licensing grants. These are
        # classifier display names, not SPDX ids, so the result is not a parseable
        # SPDX expression either; it is a human-readable label.
        return " OR ".join(dict.fromkeys(classifiers))
    # A multi-line License field means the full text was inlined; take its first meaningful line.
    for line in (meta.get("License") or "").splitlines():
        if line.strip():
            return line.strip()
    return "See license text below"


def author_label(meta) -> str:
    for key in ("Author", "Author-email", "Maintainer", "Maintainer-email"):
        value = meta.get(key)
        if value and value.strip().upper() != "UNKNOWN":
            return " ".join(value.split())
    return f"The {meta.get('Name')} team"


def project_url(meta) -> str:
    """Canonical project URL, preferring a source repository, falling back to the PyPI page."""
    candidates = []
    for entry in meta.get_all("Project-URL") or []:
        label, _, value = entry.partition(",")
        candidates.append((label.strip().lower(), value.strip()))
    if meta.get("Home-page"):
        candidates.append(("homepage", meta["Home-page"].strip()))

    for preferred in ("source", "repository", "source code", "homepage", "home-page", "documentation"):
        for label, value in candidates:
            if label == preferred and value.startswith("http"):
                return value
    for _, value in candidates:
        if value.startswith("http"):
            return value
    return f"https://pypi.org/project/{meta.get('Name')}/"


def build_package(name: str, dist: metadata.Distribution) -> tuple[Package | None, str]:
    """Assemble a Package, or return (None, reason) if no license text can be sourced."""
    meta = dist.metadata
    normalized = normalize(name)
    package = Package(
        name=normalized,
        version=dist.version or "unknown",
        author=author_label(meta),
        license_label=LICENSE_LABEL_OVERRIDES.get(normalized) or license_label(meta),
        url=project_url(meta),
    )
    package.texts, redaction_notes = redact_notice(
        dist, bundled_texts(dist) + payload_texts(dist, normalized), normalized
    )

    # Gate 2's input, collected HERE rather than in main() so it sees only texts this distribution
    # ships itself. An upstream LICENSE fetched below would otherwise be scanned too, and an Apache
    # project using structured Name:/License:/Files: headers in its own LICENSE would trip the gate
    # for code it does not vendor — unblockable except by recording a vendored-copyleft acceptance
    # that would be false. Redaction has already run, so a stanza dropped above is not scanned; that
    # is safe only because redact_notice refuses to drop a stanza whose file is still present.
    for _, text in package.texts:
        for match in NOTICE_DECLARED_LICENSE_RE.finditer(text):
            declared = match.group(1).strip()
            if STRONG_COPYLEFT.search(declared) or WEAK_COPYLEFT.search(declared):
                package.vendored_copyleft.add(declared)

    # Notes stack: a distribution can be attributed despite being first-party, carry a redaction
    # declaration, and record a vendored-copyleft acceptance — so the rendering must not drop any.
    accepted_vendored = ACCEPTED_VENDORED_COPYLEFT.get(normalized)
    vendored_note = accepted_vendored[1] if accepted_vendored and package.vendored_copyleft else ""
    notes = [
        n for n in (FIRST_PARTY_WITH_THIRD_PARTY_CONTENT.get(normalized, ""), *redaction_notes, vendored_note) if n
    ]
    package.note = "\n\n".join(notes)

    if package.texts:
        return package, ""

    if package.name in UPSTREAM_LICENSE_URLS:
        for url in UPSTREAM_LICENSE_URLS[package.name]:
            package.texts.append((url, fetch_license_text(url, package.name)))
        return package, ""
    if package.name in NO_UPSTREAM_TEXT:
        # Append, never assign: a first-party rationale or redaction note set above is also part of
        # the record, and overwriting it here would drop it from the artefact.
        package.note = "\n\n".join(filter(None, (package.note, NO_UPSTREAM_TEXT[package.name])))
        return package, ""
    return None, (
        f"{package.name}: the wheel ships no license text and there is no UPSTREAM_LICENSE_URLS "
        f"or NO_UPSTREAM_TEXT entry for it (project URL: {package.url})"
    )


# ---------------------------------------------------------------------------
# Collection and rendering
# ---------------------------------------------------------------------------


def collect(include_nvidia: bool) -> tuple[list[Package], list[str]]:
    packages: list[Package] = []
    problems: list[str] = []
    # Prefix matching is on name shape, not provenance — report what it drops so it stays auditable.
    by_prefix: list[str] = []
    for name, dist in installed_distributions().items():
        if name in LOCAL_PACKAGES or installed_from_this_repo(dist):
            continue
        if not include_nvidia and is_nvidia_first_party(name) and name not in FIRST_PARTY_WITH_THIRD_PARTY_CONTENT:
            if normalize(name) not in NVIDIA_FIRST_PARTY_NAMES:
                by_prefix.append(f"{name} {dist.version}")
            continue
        package, reason = build_package(name, dist)
        if package is None:
            problems.append(reason)
            continue
        packages.append(package)
    if by_prefix:
        print(f"  excluded as NVIDIA first-party by name prefix ({len(by_prefix)}):")
        for entry in sorted(by_prefix):
            print(f"    {entry}")
    return sorted(packages, key=lambda p: p.sort_key), problems


def render(packages: list[Package], include_nvidia: bool) -> str:
    scope = (
        "every distribution installed in the environment"
        if include_nvidia
        else "every third-party distribution installed in the environment; this project's own "
        "packages are excluded, as are NVIDIA first-party packages except those that incorporate "
        "or vendor third-party code (each such entry says so, and why, in its own section)"
    )
    parts = [
        "# Dependencies Licenses",
        "",
        "This file contains the license texts for all dependencies used in this project.",
        "",
        f"Scope: {scope}, transitive dependencies included.",
        "CDF AnomalyGen does not redistribute these packages — they are downloaded when you",
        "build the environment.",
        "",
        f"Total packages: {len(packages)}",
        "",
        "This file is generated. Run `python scripts/generate_third_party_licenses.py` after",
        "changing dependencies rather than editing it by hand.",
        "",
        "---",
        "",
    ]
    for package in packages:
        parts += [
            SEPARATOR,
            "",
            f"## {package.name} ({package.version})",
            "",
            f"**License:** {package.license_label}",
            "",
            f"**License URL:** {package.url}",
            "",
        ]
        if package.note:
            parts += [package.note, ""]
        for label, text in package.texts:
            parts += [f"**Source:** {label}", "", "```", text.strip("\n"), "```", ""]
    return "\n".join(parts).rstrip() + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=str(REPO_ROOT / "LICENSE-3rd-party.txt"), help="Output path.")
    parser.add_argument("--include-nvidia", action="store_true", help="Do not filter NVIDIA first-party packages.")
    args = parser.parse_args(argv)

    packages, problems = collect(args.include_nvidia)
    if problems:
        raise SystemExit("Cannot generate third-party licenses:\n  " + "\n  ".join(problems))

    strong = [p for p in packages if STRONG_COPYLEFT.search(p.license_label)]
    weak = [
        p for p in packages if not STRONG_COPYLEFT.search(p.license_label) and WEAK_COPYLEFT.search(p.license_label)
    ]
    if strong:
        raise SystemExit(
            "Strong copyleft (GPL/AGPL/SSPL) in the dependency closure:\n  "
            + "\n  ".join(f"{p.name} {p.version}: {p.license_label}" for p in strong)
            + "\nRemove the dependency, or make shipping it an explicit, recorded decision."
        )
    unexpected = [p for p in weak if normalize(p.name) not in ACCEPTED_WEAK_COPYLEFT]
    if unexpected:
        raise SystemExit(
            "Unrecorded weak copyleft (LGPL/MPL/EPL/CDDL):\n  "
            + "\n  ".join(f"{p.name} {p.version}: {p.license_label}" for p in unexpected)
            + "\nAdd it to ACCEPTED_WEAK_COPYLEFT with a reason, or drop the dependency."
        )
    for p in weak:
        print(f"  weak copyleft (accepted): {p.name} {p.version} — {p.license_label}")

    # Gate 2: copyleft a package declares about code it vendors, collected in build_package.
    # Compared per DECLARATION, so a package already accepted for one bundled library does not get
    # a free pass on the next one it starts shipping.
    # Keyed by name, not by Package: Package is a mutable dataclass and so unhashable.
    vendored = sorted(((p.name, p.vendored_copyleft) for p in packages if p.vendored_copyleft))
    unrecorded = {
        name: decls - ACCEPTED_VENDORED_COPYLEFT.get(normalize(name), (frozenset(), ""))[0] for name, decls in vendored
    }
    unrecorded = {name: decls for name, decls in unrecorded.items() if decls}
    if unrecorded:
        raise SystemExit(
            "Copyleft declared in a vendored-code notice, with no recorded decision:\n  "
            + "\n  ".join(f"{name}: {', '.join(sorted(decls))}" for name, decls in sorted(unrecorded.items()))
            + "\nThis is third-party code bundled INSIDE a dependency, so the package's own license "
            "label does not cover it.\nAdd the exact declaration to ACCEPTED_VENDORED_COPYLEFT with "
            "a reason, drop the dependency, or\nremove the bundled file at image build (see "
            "NOTICE_REDACTIONS)."
        )
    for name, decls in vendored:
        print(f"  vendored copyleft (accepted): {name} — {', '.join(sorted(decls))}")

    output = Path(args.output)
    output.write_text(render(packages, args.include_nvidia))
    print(f"Wrote {output} — {len(packages)} packages, {sum(len(p.texts) for p in packages)} license texts")


if __name__ == "__main__":
    main()
