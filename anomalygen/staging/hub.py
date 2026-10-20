# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face as a source for ``fetch``.

WHY THE URL FORM IS NOT OURS
``hf://datasets/<org>/<repo>@<rev>/<path>`` is what ``huggingface_hub.HfFileSystem`` already parses.
Inventing a scheme would mean writing and maintaining a parser for something that exists, and would
read differently from every other tool the workflow side uses.

WHY THE REPO TYPE IS REQUIRED
HfFileSystem defaults a bare ``hf://<org>/<repo>`` to a MODEL repo, so a dataset given that way
resolves to a model repo that does not exist and the error names the wrong thing entirely. Both
dataset and model repos are in play here -- the sample datasets are dataset repos, a fine-tuned
checkpoint is a model repo -- so the type is asked for rather than guessed.

WHY THE REVISION IS RESOLVED ONCE, UP FRONT
Listing and downloading are separate calls. Against a branch they can see different commits, and the
result is a merge of two revisions that nothing reports. So the branch is resolved to a commit sha
before the listing, and every per-file URL carries that sha: what gets listed is what gets fetched.

It is also the only reproducibility record there is. A branch moves, and a dataset fetched from one
is not the dataset the same config fetched last week -- for training or generation input that is an
integrity property, not a convenience. This cannot refuse an unpinned source (an operator trying a
public dataset should not have to look a sha up first), so it logs the one it resolved instead, and
says plainly that the source did not name it.

CREDENTIALS COME FROM THE ENVIRONMENT
``HF_TOKEN``, or the token ``hf auth login`` persists. Never from the pipeline config -- see the
module docstring of ``store.py`` for the two paths that would leak it.
"""

from __future__ import annotations

from pathlib import Path

from cosmos_framework.utils import log

SCHEME = "hf://"
_TYPES = ("datasets/", "models/")

# Hugging Face plumbing that is present in every repo and is never content. The reference workflow
# on the orchestrator side deletes this file after its download for the same reason; dropping it
# here means no consumer has to know to. Named, not a dotfile wildcard: a dataset may legitimately
# carry one, and silently dropping data is worse than carrying one file nobody wanted.
_NOT_CONTENT = frozenset({".gitattributes"})

_AUTH_HINT = (
    "A gated or private repo needs HF_TOKEN in the environment, or a token persisted by "
    "`hf auth login`, and the licence accepted once on the repo's Hugging Face page. Credentials "
    "are read from the environment only -- never put one in the pipeline config, which is written "
    "to the output directory verbatim."
)


def _fs():
    """An HfFileSystem using the ambient token.

    No explicit ``token=``: the default resolution covers both HF_TOKEN and the file ``hf auth
    login`` writes, and passing the environment variable alone would break the login case.
    """
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def _plural(repo_type: str) -> str:
    """``dataset`` -> ``datasets``, the segment the URL carries."""
    return f"{repo_type}s"


def _parse(url: str):
    """The resolved repo, revision and in-repo path behind ``url``."""
    rest = str(url)[len(SCHEME) :]
    if not rest.startswith(_TYPES):
        raise ValueError(
            f"{url!r} must name the repo type: {SCHEME}datasets/<org>/<repo> or {SCHEME}models/"
            "<org>/<repo>. Without it a dataset repo is looked up as a model repo, and the failure "
            "reports that the model does not exist."
        )
    try:
        return _fs().resolve_path(str(url))
    except Exception as exc:  # noqa: BLE001 -- re-raised with what actually resolves it
        raise RuntimeError(f"{url} could not be resolved on Hugging Face: {exc}\n{_AUTH_HINT}") from exc


def _fs_path(resolved, revision: str, path_in_repo: str) -> str:
    """The path HfFileSystem itself uses for this location.

    The type prefix comes from the library's own table, not from ``f"{repo_type}s/"``: datasets,
    spaces and kernels carry one and MODELS DO NOT. Building it by hand made every
    ``hf://models/...`` prefix fetch fail its own "not under" guard on the first entry -- the
    fine-tuned-checkpoint case this module exists to serve -- while the dataset case worked, so
    nothing noticed.

    Assembled here rather than by ``dataclasses.replace`` on the resolved path: that object keeps
    the revision string the URL was written with, and ``__post_init__`` prefers it over the field,
    so replacing the revision on a source spelled ``@main`` silently listed main instead of the sha
    just resolved -- defeating the pin this module resolves the sha for.
    """
    from huggingface_hub import constants

    prefix = constants.REPO_TYPES_URL_PREFIXES.get(resolved.repo_type, "")
    return f"{prefix}{resolved.repo_id}@{revision}/{path_in_repo}".rstrip("/")


def _url(resolved, revision: str, path_in_repo: str) -> str:
    """One ``hf://`` URL, pinned to ``revision``."""
    tail = f"/{path_in_repo}" if path_in_repo else ""
    return f"{SCHEME}{_plural(resolved.repo_type)}/{resolved.repo_id}@{revision}{tail}"


def _commit(resolved) -> str:
    """``resolved.revision`` as a commit sha, warning when the source did not name one."""
    from huggingface_hub import HfApi

    try:
        sha = HfApi().repo_info(resolved.repo_id, repo_type=resolved.repo_type, revision=resolved.revision).sha
    except Exception as exc:  # noqa: BLE001 -- re-raised with what actually resolves it
        raise RuntimeError(
            f"could not read {resolved.repo_id} ({resolved.repo_type}) at revision "
            f"{resolved.revision!r} on Hugging Face: {exc}\n{_AUTH_HINT}"
        ) from exc
    if sha != resolved.revision:
        log.warning(
            f"[fetch] {resolved.repo_id}@{resolved.revision} is a branch, not a commit -- it resolved "
            f"to {sha} now and may resolve elsewhere later. Pin it with "
            f"{SCHEME}{_plural(resolved.repo_type)}/{resolved.repo_id}@{sha} to make this repeatable."
        )
    else:
        log.info(f"[fetch] {resolved.repo_id}@{sha}")
    return sha


def is_object(uri: str) -> bool:
    """True when ``uri`` names a file in the repo rather than a directory or the repo root."""
    resolved = _parse(uri)
    if not resolved.path_in_repo:
        return False
    return _fs().isfile(_fs_path(resolved, resolved.revision, resolved.path_in_repo))


def iter_objects(prefix: str):
    """``(pinned url, path relative to prefix)`` for every file under ``prefix``, recursively."""
    fs = _fs()
    resolved = _parse(prefix)
    sha = _commit(resolved)
    base = resolved.path_in_repo.rstrip("/")

    dropped = 0
    for raw in fs.find(_fs_path(resolved, sha, base)):
        # Resolved back rather than string-stripped. find() may or may not echo the revision it was
        # asked for, and prefixes the repo type for some kinds and not others; resolve_path knows
        # both, so the in-repo path comes from the library instead of from assumptions about its
        # output format. Getting that wrong does not raise -- it flattens the tree into one
        # directory, or drops every file behind a guard, depending on which way it is wrong.
        in_repo = fs.resolve_path(raw).path_in_repo
        # Still checked: a listing that answers outside the prefix would need a guess to place.
        if base and not in_repo.startswith(f"{base}/"):
            raise RuntimeError(
                f"listing {prefix} returned {in_repo!r}, which is not under {base!r}. Placing it "
                "would need a guess, and guessing wrong flattens the tree silently."
            )
        rel = in_repo[len(base) + 1 :] if base else in_repo
        if Path(rel).name in _NOT_CONTENT:
            dropped += 1
            continue
        yield _url(resolved, sha, in_repo), rel

    if dropped:
        log.info(f"[fetch] skipped {dropped} Hugging Face metadata file(s): {sorted(_NOT_CONTENT)}")


def download(uri: str, local: Path) -> None:
    """Download one file to ``local``, via a temporary name in the same directory.

    Streamed straight to the destination rather than through ``hf_hub_download``: that one populates
    the shared HF cache and then links or copies out of it, which on a per-run staging directory
    means the bytes land twice and the cache grows by every dataset any run ever fetched.
    """
    resolved = _parse(uri)
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(f".{local.name}.partial")
    try:
        _fs().get_file(_fs_path(resolved, resolved.revision, resolved.path_in_repo), str(tmp))
        tmp.replace(local)
    except Exception as exc:  # noqa: BLE001 -- re-raised with what actually resolves it
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {uri} -> {local}: {exc}\n{_AUTH_HINT}") from exc
