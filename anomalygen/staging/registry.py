# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NGC as a source for ``fetch``.

THE URL FORM
    ngc://models/<org>[/<team>]/<name>:<version>[/<path>]
    ngc://resources/<org>[/<team>]/<name>:<version>[/<path>]

Shaped after the ``hf://`` one -- artifact type first -- because the two sit side by side in the same
config list and a reader should not have to learn two conventions. ``<name>:<version>`` is NGC's own
way of writing it, and the team segment is optional because NGC itself makes it so: a model may be
published directly under an org or under a team within it, and the REST path differs accordingly.

THE VERSION IS THE PIN, AND IT IS MANDATORY
Unlike a Hugging Face branch there is nothing to resolve: NGC has no moving pointer, the version IS
the artifact, and the API has no endpoint that does not name one. So a URL without a version is
rejected rather than defaulted -- there is no default that would be right.

REST AND NOT THE ngc CLI
The CLI is a separate binary that is not in this image and would have to be installed and configured
to be used at all. The two REST calls this needs -- list a version's files, download one of them --
are reachable without credentials for public artifacts, which is what lets the tests exercise this
path with no key at all.

That convenience hid a bug for the whole of the first release: the endpoints were the public
catalogue's, which serves nothing private, and no test or e2e run ever asked for a private artifact.
See ``_base`` for what that cost and how it was found.

CREDENTIALS COME FROM THE ENVIRONMENT
``NGC_API_KEY``, sent as a bearer token. Never from the pipeline config -- see the module docstring
of ``store.py`` for the two paths that would leak it.
"""

from __future__ import annotations

import os
from pathlib import Path

from cosmos_framework.utils import log

SCHEME = "ngc://"
API = "https://api.ngc.nvidia.com/v2"

# The key holding the file list differs per artifact type, and NGC's naming for a resource is its
# older one ("recipe"). Mapped explicitly: reading whichever key happens to end in "Files" would
# turn a future response change into a wrong answer rather than a failure.
_KINDS = {"models": ("modelFiles", "modelVersion"), "resources": ("recipeFiles", "recipeVersion")}

_TIMEOUT = 60
_AUTH_HINT = (
    "A private artifact needs NGC_API_KEY in the environment. Credentials are read from the "
    "environment only -- never put one in the pipeline config, which is written to the output "
    "directory verbatim."
)


def _headers() -> dict:
    key = os.environ.get("NGC_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _parse(url: str) -> tuple:
    """``(kind, scope, version, path)`` behind ``url``; ``scope`` is ``org[/team]/name``."""
    rest = str(url)[len(SCHEME) :]
    kind, _, rest = rest.partition("/")
    if kind not in _KINDS:
        raise ValueError(
            f"{url!r} must name the artifact type: "
            + " or ".join(f"{SCHEME}{k}/<org>[/<team>]/<name>:<version>" for k in _KINDS)
            + f". Got {kind!r}."
        )
    head, sep, tail = rest.partition(":")
    if not sep:
        raise ValueError(
            f"{url!r} names no version. NGC has no moving pointer to fall back on -- a version IS "
            f"the artifact -- so write {SCHEME}{kind}/<org>[/<team>]/<name>:<version>."
        )
    scope = [s for s in head.split("/") if s]
    if len(scope) not in (2, 3):
        raise ValueError(
            f"{url!r} has {len(scope)} segment(s) before the version; expected <org>/<name> or <org>/<team>/<name>."
        )
    version, _, path = tail.partition("/")
    if not version:
        raise ValueError(f"{url!r} has an empty version.")
    return kind, "/".join(scope), version, path


def _base(url: str) -> tuple:
    """``(version endpoint, kind, path)`` for ``url``.

    TWO ENDPOINTS, CHOSEN BY WHETHER THERE IS A KEY, because neither one serves both cases:

        /v2/<kind>/<org>/<team>/<name>/...        the public catalogue. Anonymous reads work;
                                                  a PRIVATE artifact answers 404 here whatever
                                                  credentials are sent.
        /v2/org/<org>[/team/<team>]/<kind>/...    the org-scoped path. Private artifacts work;
                                                  it answers 401 with no credentials, even for a
                                                  PUBLIC artifact.

    Only the public form shipped, which is why no private fetch has ever worked: every test and
    every e2e run used a public artifact, where that form is right and the Authorization header is
    ignored, so nothing here was ever asked the question that separates the two.

    404 is what makes it expensive to diagnose. NGC will not answer 403 for an artifact the caller
    may not see -- that would confirm the name exists -- so the one code means both "no such thing"
    and "not yours", and the reader is sent to check a spelling that was correct all along.

    MEASURED, not reasoned about. The cell that decides whether choosing by key is safe is a PUBLIC
    artifact fetched WITH a key: a pod that holds NGC_API_KEY for its private sources sends its
    public ones through the org-scoped form too, including artifacts of an org the key is not a
    member of. Against nvidia/tao/pretrained_classification:resnet18, a public model of another org:

        org-scoped, with a key      200     <- the case this function's choice creates
        org-scoped, no key          401
        catalogue,  with a key      200     <- so a fallback has a route, if one is ever needed
        catalogue,  private artifact 404    (what made every private fetch fail before this)

    So a key costs nothing here: it is honoured where it is needed and ignored where it is not.
    """
    kind, scope, version, path = _parse(url)
    if not _headers():
        # No key: the org-scoped path would 401 before it could 404, which is a worse error for the
        # public case than the one the public catalogue gives.
        return f"{API}/{kind}/{scope}/versions/{version}", kind, path
    *owner, name = scope.split("/")
    team = f"/team/{owner[1]}" if len(owner) > 1 else ""
    return f"{API}/org/{owner[0]}{team}/{kind}/{name}/versions/{version}", kind, path


def _get(url: str, **kwargs):
    import requests

    try:
        response = requests.get(url, headers=_headers(), timeout=_TIMEOUT, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- network errors reach the caller as one kind of failure
        raise RuntimeError(f"NGC request failed: {url}: {exc}") from exc
    if response.status_code in (401, 403):
        raise RuntimeError(f"NGC returned {response.status_code} for {url}.\n{_AUTH_HINT}")
    if not response.ok:
        raise RuntimeError(f"NGC returned {response.status_code} for {url}: {response.text[:200]}")
    return response


def _files(uri: str) -> list:
    """Every file path in the version named by ``uri``.

    Paged, and then counted against the version's own ``totalFileCount``. NGC answers with 100 files
    and a ``paginationInfo`` block whether or not more exist, so a single-request implementation
    returns a short list and reports success -- and a dataset that is 100 files instead of 205 does
    not fail, it quietly generates from less than it was given. The count is the check that a page
    was not lost; disagreeing with it is a failure, not a warning.
    """
    base, kind, _ = _base(uri)
    files_key, version_key = _KINDS[kind]

    paths, expected, page = [], None, 0
    while True:
        body = _get(f"{base}/files", params={"page-number": page}).json()
        if expected is None:
            expected = (body.get(version_key) or {}).get("totalFileCount")
        paths += [entry["path"] for entry in body.get(files_key, [])]
        info = body.get("paginationInfo") or {}
        nxt = info.get("nextPage")
        if nxt is None or str(nxt) == str(page):
            break
        page = int(nxt)

    if expected is not None and len(paths) != expected:
        raise RuntimeError(
            f"{uri} lists {len(paths)} file(s) but the version reports {expected}. A page was lost, "
            "and fetching this would produce a partial tree that looks complete."
        )
    return paths


def is_object(uri: str) -> bool:
    """True when ``uri`` names one file in the version rather than a directory or the whole version."""
    _, _, path = _base(uri)
    return bool(path) and path in _files(uri)


def iter_objects(prefix: str):
    """``(url, path relative to prefix)`` for every file under ``prefix``, recursively.

    NGC has no directories -- a version is a flat list of paths that happen to contain slashes -- so
    "under" is a prefix match on the path, with the separator required: without it ``mask`` would
    also match ``mask_v2/...`` and pull in a sibling nobody asked for.
    """
    kind, scope, version, path = _parse(prefix)
    inner = f"{path.rstrip('/')}/" if path else ""
    found = 0
    for entry in _files(prefix):
        if not entry.startswith(inner):
            continue
        found += 1
        yield f"{SCHEME}{kind}/{scope}:{version}/{entry}", entry[len(inner) :]
    log.info(f"[fetch] {found} file(s) under {prefix}")


def download(uri: str, local: Path) -> None:
    """Download one file to ``local``, via a temporary name in the same directory.

    Streamed in chunks: a model version file is routinely several GB, and reading the body into
    memory first would make the container's memory limit, not the disk, decide what can be fetched.
    """
    base, _, path = _base(uri)
    if not path:
        raise ValueError(f"{uri!r} names a version, not a file in it.")
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(f".{local.name}.partial")
    try:
        with _get(f"{base}/files/{path}", stream=True, allow_redirects=True) as response, tmp.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)
        tmp.replace(local)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"download failed: {uri} -> {local}: {exc}\n{_AUTH_HINT}") from exc
