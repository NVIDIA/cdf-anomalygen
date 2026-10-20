# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object-store access for the staging stages, over multi-storage-client.

FOUR SCHEMES IN, TWO OUT
``fetch`` reads from object storage (``s3://``, ``msc://``), Hugging Face (``hf://``) and NGC
(``ngc://``); ``publish`` writes only to the first two. That asymmetry is deliberate: Hugging Face
and NGC are where a dataset or a checkpoint is PUBLISHED, on a human decision, with a licence and a
model card attached. A pipeline task uploading a run's output there is not a thing anyone wants to
happen by a typo in a config field, so there is no code here that could.

The three listing-and-download functions dispatch on the scheme; ``hub`` and ``registry`` hold the
two non-MSC backends. Everything above them -- the overlay merge, the atomic swap, the partial-file
rename -- is shared, so a source is a source regardless of where it came from.

WHY MSC AND NOT boto3
Both are already dependencies. MSC is the one the orchestrator side already speaks, and one call
covers S3, GCS, Azure, OCI and a plain filesystem -- so a backend change is a URL change rather than
a code change, and the filesystem backend makes this module testable without any cloud credentials.

TWO URL FORMS, BOTH SUPPORTED
    s3://bucket/key             implicit: credentials come from the AWS chain (env vars, IAM role)
    msc://profile/prefix/key    explicit: `profile` is defined in the MSC configuration
Neither is preferred; the caller picks. Verified against msc 1.0.1: the implicit form reaches the S3
provider and fails only on credentials, so it is a supported scheme rather than a parse error.

CREDENTIALS COME FROM THE ENVIRONMENT, NEVER FROM THE CONFIG
Two paths in this repo would otherwise leak them:

  * ``run_pipeline.run()`` logs the whole argv of every stage it launches, so a credential passed as
    a command-line argument lands in ``logs/*.log``.
  * ``run_pipeline.main()`` writes ``asdict(cfg)`` to ``pipeline_config.resolved.yaml`` inside the
    OUTPUT directory -- the very tree a publish step uploads.

So nothing here reads a secret from the pipeline config, and nothing puts one in an argv.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from cosmos_framework.utils import log

from anomalygen.staging import hub, registry, usecase

# Inline JSON the orchestrator may set. MSC itself reads only MSC_CONFIG (a path) -- verified against
# msc 1.0.1, whose config module references MSC_CONFIG / XDG_CONFIG_HOME / XDG_CONFIG_DIRS and
# nothing else -- so the inline form has to be materialised before any MSC call.
MSC_INLINE_ENV = "MULTISTORAGECLIENT_CONFIGURATION"
MSC_CONFIG_ENV = "MSC_CONFIG"

_SUPPORTED_SCHEMES = ("msc://", "s3://")
_FETCH_SCHEMES = (*_SUPPORTED_SCHEMES, hub.SCHEME, registry.SCHEME)
# What an upload or download failure can mean. LISTED, NOT RANKED, and deliberately not led by
# credentials: a run that published seven batches and then met a typo'd bucket got told to check its
# AWS keys, which were fine. MSC does not help here -- it translates a 404 on a PUT into
# "Object ... does not exist", which reads like a missing file rather than a missing bucket.
_STORE_HINT = """What a store failure here can mean:
  the bucket or prefix does not exist   a typo, or a name that exists in another account. On a PUT
                                        this arrives as "does not exist" about the object, not the
                                        bucket, which is the same 404 either way.
  no write permission on it             the credentials are valid and simply not allowed here.
  the endpoint is wrong                 AWS_ENDPOINT_URL must be set for a non-AWS store (PBSS);
                                        without it an s3:// URL resolves to AWS, where the bucket
                                        is someone else's or nobody's.
  no credentials at all                 set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or an
                                        instance role) for s3://, or MULTISTORAGECLIENT_CONFIGURATION
                                        / MSC_CONFIG for an msc:// profile.
Credentials are read from the environment only -- never put one in the pipeline config, which is
written to the output directory verbatim."""


def is_remote(uri: str) -> bool:
    """True when ``uri`` names something to download rather than a local path."""
    return str(uri).startswith(_FETCH_SCHEMES)


def _backend(uri: str):
    """The module that speaks ``uri``'s scheme, or None for the MSC-backed object stores."""
    if str(uri).startswith(hub.SCHEME):
        return hub
    if str(uri).startswith(registry.SCHEME):
        return registry
    return None


def check_uri(uri: str) -> None:
    """Reject a destination this module cannot write before a run spends GPU hours reaching it."""
    if not str(uri).startswith(_SUPPORTED_SCHEMES):
        published = f"{hub.SCHEME} and {registry.SCHEME} are fetch sources, not upload destinations."
        raise ValueError(
            f"{uri!r} is not a supported object-store URL. Expected one of "
            f"{', '.join(_SUPPORTED_SCHEMES)}. A local destination needs no staging step. "
            f"{published}"
        )


def check_source(uri: str) -> None:
    """Accept an object-store URL or a local directory as something to fetch FROM.

    A local source is how a caller that already has part of the input on disk -- a mounted work item,
    say -- gets it merged with the parts that are downloaded. It goes through the same staging
    directory as a remote one, so it cannot be destroyed by the swap that installs the merge; the
    alternative, letting the caller pre-place files in the destination, cannot work at all, because
    the destination is replaced wholesale and whatever was there is gone.
    """
    if is_remote(uri):
        return
    if usecase.is_usecase(uri):
        spec = usecase.parse(uri)  # raises with the available names when it is not one of them
        # Its file parameters are fetched before the script runs, so they are sources like any
        # other and are checked here -- at config load, rather than after the script has been
        # started and is waiting for a file that was never going to arrive.
        for param, source in spec.files.items():
            try:
                check_source(source)
            except ValueError as exc:
                raise ValueError(f"{uri}: parameter {param!r} -- {exc}") from exc
        return
    path = Path(uri)
    if not path.exists():
        raise ValueError(
            f"{uri!r} is neither a remote URL ({', '.join(_FETCH_SCHEMES)}), a {usecase.SCHEME} name, "
            "nor a path that exists. "
            "A local source must be present when the run starts -- nothing fetches it."
        )


def configure() -> None:
    """Materialise inline MSC configuration, if that is how it arrived. Idempotent.

    ``MSC_CONFIG`` wins when both are set: an explicit path is the more specific instruction, and
    silently overwriting it with the inline form would make the two disagree with no way to tell.

    The file is created 0600 in a private directory. The orchestrator-side reference implementation
    writes a fixed name straight into the shared temp dir, which on a multi-tenant node leaves the
    credentials world-readable; mkdtemp gives a fresh 0700 directory instead.
    """
    if os.environ.get(MSC_CONFIG_ENV):
        return
    inline = os.environ.get(MSC_INLINE_ENV)
    if not inline:
        return
    try:
        parsed = json.loads(inline)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{MSC_INLINE_ENV} must contain JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{MSC_INLINE_ENV} must contain a JSON object, got {type(parsed).__name__}.")

    path = Path(tempfile.mkdtemp(prefix="anomalygen-msc-")) / "config.json"
    path.write_text(json.dumps(parsed))
    path.chmod(0o600)
    os.environ[MSC_CONFIG_ENV] = str(path)
    # The path, never the contents.
    log.info(f"[staging] wrote {MSC_INLINE_ENV} to {path} and set {MSC_CONFIG_ENV}")


def upload(local: Path, uri: str) -> None:
    """Upload one file, then confirm the object is really there.

    The read-back is not paranoia about MSC: an upload to a prefix the caller cannot write can come
    back without raising on some providers, and a publish step that reports success while writing
    nothing is the worst failure this module could have -- the batch is deleted downstream.
    """
    import multistorageclient as msc

    configure()
    try:
        msc.upload_file(url=uri, local_path=str(local))
    except Exception as exc:  # noqa: BLE001 -- re-raised with the one hint that resolves most of these
        raise RuntimeError(f"upload failed: {local} -> {uri}: {exc}\n{_STORE_HINT}") from exc
    if not msc.is_file(uri):
        raise RuntimeError(f"upload reported success but {uri} does not exist afterwards.\n{_STORE_HINT}")


def check_reachable(uri: str) -> None:
    """Fail now if the store behind ``uri`` cannot be reached, rather than after the GPU work.

    Publishing is the LAST stage, so until this existed a typo in the destination cost a whole run to
    discover.

    It lists the prefix rather than writing a probe object. Writing one would prove more, but only by
    needing delete permission to clean up after itself -- and a prefix that allows PUT and not DELETE
    would then fail a check it should pass, with a stray object left behind. So this is honest about
    its limit: it catches a wrong bucket, a wrong endpoint and absent credentials, which is what
    typos and misconfigured containers actually produce. It does NOT prove the prefix is writable.
    """
    import multistorageclient as msc

    check_uri(uri)
    configure()
    try:
        # One object is enough to prove the store answers; an empty prefix is not an error, since
        # publishing into a fresh one is the normal case.
        next(iter(msc.list(str(uri).rstrip("/") + "/", include_directories=False)), None)
    except Exception as exc:  # noqa: BLE001 -- re-raised with what actually resolves it
        raise RuntimeError(f"publish destination {uri} is not reachable: {exc}\n{_STORE_HINT}") from exc


def join(base: str, name: str) -> str:
    """``base`` and ``name`` as one object URL, with exactly one separator."""
    return f"{str(base).rstrip('/')}/{str(name).lstrip('/')}"


def is_object(uri: str) -> bool:
    """True when ``uri`` names a single object rather than a prefix.

    Asked of the store rather than inferred from a trailing slash: a caller who omits the slash on a
    prefix would otherwise get one file named after the whole directory, and the mistake would only
    surface as a dataset with no images in it.
    """
    backend = _backend(uri)
    if backend is not None:
        return backend.is_object(uri)

    import multistorageclient as msc

    configure()
    return msc.is_file(uri)


def _key_path(url: str) -> str:
    """The path below the bucket or profile: ``s3://bucket/a/b`` and ``msc://p/a/b`` both give ``a/b``.

    Needed because msc.list does not echo the URL it was given. Asked for ``s3://abc/x/``, msc 1.0.1
    answers with ``msc://_s3-abc/x/...`` -- it resolves the bucket to an implicit profile and
    rewrites the scheme. Comparing the answers against the caller's prefix as strings therefore fails
    on exactly the backend that matters, so both sides are reduced to this common form instead.
    """
    rest = str(url).split("://", 1)[-1]
    return rest.split("/", 1)[1] if "/" in rest else ""


def iter_objects(prefix: str):
    """``(url, path relative to prefix)`` for every object under ``prefix``, recursively.

    The relative path is what preserves ``<TEXTURE>/mask/<class>/`` on the way down.
    """
    backend = _backend(prefix)
    if backend is not None:
        yield from backend.iter_objects(prefix)
        return

    import multistorageclient as msc

    configure()
    base = str(prefix).rstrip("/") + "/"
    base_key = _key_path(base)
    for item in msc.list(base, include_directories=False):
        url = str(item.key)
        key = _key_path(url)
        if not key.startswith(base_key):
            raise RuntimeError(
                f"listing {prefix} returned {url}, which is not under it ({key!r} vs {base_key!r}). "
                "Placing it would need a guess, and guessing wrong flattens the tree silently."
            )
        yield url, key[len(base_key) :]


def download(uri: str, local: Path) -> None:
    """Download one object to ``local``, via a temporary name in the same directory.

    Written aside and renamed because a partial file left at the final path is indistinguishable from
    a complete one on the next run -- and the next run is exactly when nobody is watching.
    """
    backend = _backend(uri)
    if backend is not None:
        backend.download(uri, local)
        return

    import multistorageclient as msc

    configure()
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(f".{local.name}.partial")
    try:
        msc.download_file(url=uri, local_path=str(tmp))
        tmp.replace(local)
    except Exception as exc:  # noqa: BLE001 -- re-raised with the hint that resolves most of these
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {uri} -> {local}: {exc}\n{_STORE_HINT}") from exc
