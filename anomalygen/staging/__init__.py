# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Moving a run's inputs and outputs between object storage and local disk.

WHY THIS IS NOT IN run_pipeline.py
The entrypoint's stages are thin wrappers -- ``amp`` shells out to ``roi_place_pipeline``, ``refine``
to ``quality_refine.py`` -- and staging is the same shape: the stage assembles a call, this package
does the work. Keeping it here is what lets a second use case (SMT) import ``publish`` directly
without inheriting the entrypoint's config schema, stage ordering or CLI.

WHAT PUBLISH UPLOADS, AND WHY IT IS THE MANIFEST AND NOT A GLOB
``export/`` is not cleared between runs: ``stage_export`` writes the images this run produced and
leaves anything already there. Globbing the directory would therefore ship a previous run's images
alongside this one's, silently, and the consumer has no way to tell them apart. ``export_manifest.json``
lists exactly what this run wrote, so it is the only correct source of truth for what to upload.

The manifest is uploaded too. It carries ``{generated name -> source crop}``, which is what a step
compositing results back onto a board needs; without it the consumer has a bag of files and no
mapping.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from cosmos_framework.utils import log

from anomalygen.staging import usecase
from anomalygen.staging.store import (
    check_reachable,
    check_source,
    check_uri,
    download,
    is_object,
    is_remote,
    iter_objects,
    join,
    upload,
)
from anomalygen.staging.weights import ensure_base_weights

MANIFEST_NAME = "export_manifest.json"

# Dropped into a destination this function built, and looked for before it replaces one. Without it
# `fetch` cannot tell a directory it owns from one the operator populated, and the only thing
# standing between a mistyped `input:` and an `rmtree` of their dataset is that the two paths are
# not literally equal.
FETCH_MARKER = ".anomalygen_fetch"

__all__ = [
    "MANIFEST_NAME",
    "check_override",
    "check_reachable",
    "check_source",
    "check_uri",
    "ensure_base_weights",
    "fetch",
    "is_remote",
    "publish",
]


def publish(export_dir: Path, destination: str) -> dict:
    """Upload everything ``export_manifest.json`` lists, plus the manifest. Returns ``{name: url}``.

    ``export_dir`` is the local ``export/`` bucket; ``destination`` is an object-store prefix.
    """
    check_uri(destination)
    export_dir = Path(export_dir)
    manifest_path = export_dir / MANIFEST_NAME

    # The caller (stage_publish) has already required this to exist, so reaching here without it
    # means a manifest that is present but unreadable -- a different fault, and worth its own message.
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{manifest_path} could not be read as JSON: {exc}") from exc

    names = sorted(manifest.get("images", {}))
    if not names:
        raise RuntimeError(
            f"{manifest_path} lists no images, so there is nothing to publish. An export that "
            "produced no output usually means every sample was blocked by the guardrail; "
            "logs/generate.log records that per sample."
        )

    # Every file is checked before any is sent: a half-published prefix is worse than one that was
    # never written, because the consumer cannot distinguish it from a complete one.
    missing = [n for n in names if not (export_dir / n).is_file()]
    if missing:
        raise RuntimeError(
            f"{manifest_path} lists {len(missing)} file(s) that are not on disk: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}. The export bucket and its manifest disagree; "
            "re-run the 'export' stage."
        )

    log.info(f"[publish] {len(names)} image(s) + manifest -> {destination}")
    published = {}
    for name in names:
        url = join(destination, name)
        upload(export_dir / name, url)
        published[name] = url
    manifest_url = join(destination, MANIFEST_NAME)
    upload(manifest_path, manifest_url)
    published[MANIFEST_NAME] = manifest_url

    log.success(f"[publish] {len(published)} object(s) at {destination}")
    return published


def fetch(sources, destination, override=None) -> int:
    """Download ``sources`` to ``destination``. Returns the number of files written.

    ``sources`` is one source or a list of them. A source is an object-store URL or a local path.
    A single source naming one file copies that file to ``destination``; a source naming a prefix or
    a directory copies everything under it, preserving the tree below it so ``<TEXTURE>/mask/<class>/``
    survives the trip.

    A LOCAL SOURCE IS FOR INPUT THE CALLER ALREADY HAS
    An orchestrator that mounts the work item rather than uploading it lists that directory beside
    the remote prefixes and gets the same merge. It must be a SEPARATE directory, not ``destination``
    itself: this function replaces ``destination`` wholesale, so anything pre-placed there is
    deleted, silently, before the first stage reads it.

    A LIST IS AN OVERLAY, IN ORDER, LATER WINNING
    That is what the split between shared and per-item inputs needs: a dataset is a submask bank and
    a defect_spec that many runs share, plus the one clean image and cad_mask this run was given, and
    the two live in different places because they have different lifetimes. Merging them here keeps
    that a property of the config rather than something a caller has to stage by hand.

    The merge is assembled in a temporary directory and moved into place, so ``destination`` is
    either the previous contents or the complete new ones. Downloading in place would leave a
    half-merged dataset that looks finished -- and a dataset missing a few masks does not fail, it
    quietly generates a smaller batch.

    AN OVERRIDE REPLACES ONE PATH INSIDE THE RESULT
    ``override`` is ``{path relative to destination: sources}``, applied to the merged tree before it
    is moved into place. It is not another overlay: the path it names is REPLACED, because the thing
    it exists for -- "generate from these three boards, not the dataset's" -- is not satisfied by
    adding three images to a pool of two hundred that are all still sampled.
    """
    urls = [sources] if isinstance(sources, str) else list(sources)
    if not urls:
        raise ValueError("fetch was given no sources")
    for url in urls:
        check_source(url)
    override = check_override(override)

    destination = Path(destination)
    # A use-case is RUN into a directory and then treated as the local source it produced, so
    # everything below -- the merge, the relative placement, the swap, the guards -- is the path a
    # mounted directory already takes. See staging/usecase.py for why it is not a backend.
    # Listed BEFORE anything is prepared or copied. check_override only reads the shape and the
    # scheme, so an override naming a prefix its upstream task has not written yet used to be found
    # after the whole dataset had been fetched -- including a `usecase://` download of several
    # minutes -- and reported as "nothing to fetch under". This is the pass that can say it first.
    listed_override = _list_override(override)

    prepared = []
    if any(usecase.is_usecase(u) for u in urls):
        # Before the script runs, not after. A use-case fills a DIRECTORY, so the one guard that can
        # be answered without listing anything can be answered now -- and it is the expensive one to
        # answer late: `input: /data/mine` with a use-case fetched the whole gated dataset and then
        # refused to replace the directory, having moved several minutes of bytes for an outcome
        # decided before the first one.
        _refuse_foreign_destination(destination, one_file=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resolved = []
        for u in urls:
            if not usecase.is_usecase(u):
                resolved.append(u)
                continue
            work = Path(tempfile.mkdtemp(prefix=f".{destination.name}-usecase-", dir=destination.parent))
            prepared.append(work)
            # The dataset and the script's own inputs go in separate directories: the dataset
            # directory becomes a fetch SOURCE, so a zip left in it would be copied into the input
            # tree as if the script had produced it.
            resolved.append(str(usecase.prepare(u, work / "dataset", _usecase_files(u, work / "args"))))
        urls = resolved

    try:
        return _fetch_resolved(urls, destination, listed_override)
    finally:
        for work in prepared:
            shutil.rmtree(work, ignore_errors=True)


def _usecase_files(uri: str, into: Path) -> dict:
    """Fetch each file parameter of ``uri``; return ``{parameter: local path}``.

    A parameter names ONE file -- the script is given a path, not a directory to search -- so a
    prefix is refused here rather than handed over as a directory the script will not understand.
    """
    files = {}
    for param, source in usecase.parse(uri).files.items():
        if not _names_one_file(source):
            raise ValueError(
                f"{uri}: parameter {param!r} is given {source}, which is a prefix or a directory. A "
                "use-case parameter is one file handed to the script; name the file itself."
            )
        local = into / f"{param}-{_basename(source)}"
        _copy_one(source, local)
        log.info(f"[fetch] use-case parameter --{param} <- {source}")
        files[param] = local
    return files


def _fetch_resolved(urls: list, destination: Path, override: dict) -> int:
    """The body of :func:`fetch`, once every source names something on disk or in a store."""
    one_file = len(urls) == 1 and _names_one_file(urls[0])
    if one_file:
        if override:
            # `override` addresses paths inside a tree, and this destination is one file. Said here
            # rather than let the first override raise "not in the fetched input" about a path that
            # could never have been in it.
            raise ValueError(
                f"fetch has an override, but {urls[0]} names a single file and {destination} is "
                f"therefore a file too. An override replaces a path inside a fetched tree; "
                f"{sorted(override)} has nowhere to go here."
            )
        # Guarded like the prefix path below. It was not: `recipe`, `checkpoint` and
        # `amg_model_registry` are file destinations, so a config that names an operator's file AND
        # a fetch for it replaced that file with one log line and no way back.
        _refuse_foreign_destination(destination, one_file=True)
        _copy_one(urls[0], destination)
        _marker_for(destination, one_file=True).touch()
        log.info(f"[fetch] {urls[0]} -> {destination}")
        return 1

    # Guarding here rather than trusting the caller: a source that IS the destination would be
    # deleted by the swap below, and the run would then fail somewhere else entirely.
    resolved = destination.resolve()
    for url in urls:
        if not is_remote(url) and Path(url).resolve() == resolved:
            raise ValueError(
                f"fetch source {url!r} is the destination itself. This function replaces the "
                "destination, so that would delete the very files being fetched. Put the local input "
                "in its own directory and list it here."
            )

    # After the guard above, which names a more specific mistake when both apply.
    _refuse_foreign_destination(destination, one_file=False)

    objects = [(url, _list_source(url)) for url in urls]
    empty = [url for url, found in objects if not found]
    if empty:
        raise RuntimeError(
            f"nothing to fetch under {empty}. A prefix that lists no objects is usually a typo or a "
            "profile pointing at the wrong bucket; an object meant as a single file should be given "
            "on its own, not in a list."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    # Renamed aside rather than deleted, and only removed once the new tree is in place. Deleting
    # first left a window with NEITHER copy on disk: the entrypoint turns SIGTERM into SystemExit, so
    # a cancellation between the rmtree and the rename unwound through the finally below, which then
    # removed the fully-downloaded replacement as well. And the window was not the two lines it looks
    # like -- an interrupt during the rmtree of a 5k-image dataset leaves it half-deleted, which is
    # the same loss arrived at more slowly. Both renames are atomic and both paths are siblings, so
    # this is either the old contents or the new ones, which is what the docstring promises.
    replaced = destination.with_name(f".{destination.name}.replaced")
    kept_aside = False
    try:
        written = 0
        for url, found in objects:  # in order, so a later source overwrites an earlier one
            for obj, rel in found:
                _copy_one(obj, _placeable(staged, rel))
                written += 1
            log.info(f"[fetch] {len(found)} object(s) from {url}")
        # Inside the staging directory, before the swap. After it would mean the destination spends
        # a moment holding the dataset the config asked NOT to generate from, and a run cancelled in
        # that moment leaves exactly that on disk, looking complete.
        written += _apply_override(staged, override)
        (staged / FETCH_MARKER).touch()
        # A leftover aside copy is reclaimed FIRST, and its meaning depends on what is at the
        # destination. The recovery below only runs when this process reaches its own finally, so a
        # hard kill -- a pod eviction, the SIGKILL after an orchestrator's grace period -- leaves the
        # old tree here with no destination beside it, and nothing would ever have come back for it:
        # the next fetch would rename its own tree into place and orphan a 5k-image dataset on the
        # volume for good.
        if replaced.exists() and not destination.exists():
            replaced.replace(destination)  # that run died mid-swap; this IS the destination
        if destination.exists():
            if replaced.exists():
                shutil.rmtree(replaced)  # a stale aside copy, with the destination already back
            destination.replace(replaced)
            kept_aside = True
        staged.replace(destination)
        log.info(f"[fetch] {written} file(s) from {len(urls)} source(s) -> {destination}")
        return written
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        if kept_aside and replaced.exists():
            if destination.exists():
                shutil.rmtree(replaced, ignore_errors=True)
            else:
                replaced.replace(destination)  # the second rename never happened; put the old back


def check_override(override) -> dict:
    """``override`` as ``{relative path: [source, ...]}``, or ``{}``.

    Shape only: the keys, and that every value is a source ``fetch`` can reach. WHAT those sources
    hold is read by :func:`_list_override`, which also runs before anything is downloaded -- this one
    is separate because it is also called from config load, where no network is wanted.
    """
    if not override:
        return {}
    if not isinstance(override, dict):
        raise ValueError(f"fetch override must be a mapping of path -> sources, got {type(override).__name__}.")
    checked = {}
    for rel, sources in override.items():
        rel = str(rel)
        # `.` and `./` are neither empty, absolute nor climbing, and `Path(".").parts` is (), so a
        # key of `.` resolved to the staging directory itself: "replace one path inside the dataset"
        # became "rmtree the dataset and refill it from this key", after the use-case script had run
        # for nothing. Hence the parts check rather than a truth test on the string.
        if not rel or not Path(rel).parts or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(
                f"fetch override key {rel!r} must be a path INSIDE the destination, relative to it "
                "-- an absolute path or one climbing out of the tree names something this fetch "
                "does not own."
            )
        urls = [sources] if isinstance(sources, str) else list(sources or [])
        if not urls:
            raise ValueError(f"fetch override {rel!r} lists no sources.")
        for url in urls:
            # A use-case script writes a whole dataset -- textures and a defect_spec -- so it is
            # something to fetch INTO the destination, never into one path inside it.
            if usecase.is_usecase(url):
                raise ValueError(
                    f"fetch override {rel!r} names {url}, which prepares a whole dataset. A use-case "
                    "belongs on the destination itself; an override replaces one path inside it."
                )
            check_source(url)
        checked[rel] = urls
    return checked


def _list_override(override: dict) -> dict:
    """``{path: [(url, names one file, listing)]}`` -- what each override source holds, nothing written.

    Separated from placing it so that it can run before the first byte moves. A source that lists
    nothing is refused here: replacing the path with an empty directory would leave the run with no
    images and no error until AMP.
    """
    listed = {}
    for rel, urls in override.items():
        entries = []
        for url in urls:
            if _names_one_file(url):
                entries.append((url, True, None))
                continue
            found = _list_source(url)
            if not found:
                raise RuntimeError(
                    f"override {rel!r}: nothing to fetch under {url}. Replacing the path with an "
                    "empty directory would leave the run with no images and no error until AMP."
                )
            entries.append((url, False, found))
        listed[rel] = entries
    return listed


def _apply_override(staged: Path, listed: dict) -> int:
    """Replace each overridden path in the merged tree. Returns the number of files written.

    THE PATH HAS TO BE THERE ALREADY
    An override that names nothing is almost always a typo -- ``clean_image`` written ``clean_images``
    -- and the damage is silent: the fetch succeeds, an unread directory appears beside the real one,
    and the run generates from the dataset's own images as if no override had been written. Adding a
    texture or a mask class is what listing another source on the destination is for.

    A DIRECTORY MAY BE FILLED FROM NAMED FILES
    "Generate from THIS board" is the common request, and the board is one file in a repo of three.
    A source that names a single file therefore lands in the directory under its own name, beside
    whatever the prefixes brought. That guess is available here and nowhere else: the key already
    names the directory the file belongs in, whereas a bare file listed on ``input`` would have to be
    placed somewhere in a dataset tree by guessing which texture and which role it is.

    What each source holds was read by :func:`_list_override` before any of this ran.
    """
    written = 0
    for rel, entries in listed.items():
        target = _placeable(staged, rel)
        if not target.exists():
            parent = target.parent
            near = sorted(p.name for p in parent.iterdir()) if parent.is_dir() else []
            raise RuntimeError(
                f"override {rel!r} is not in the fetched input, so there is nothing to override. "
                f"{parent.relative_to(staged) or '.'} holds {near if near else 'nothing'}."
            )

        # A FILE is replaced by exactly one file. The reverse of the paragraph above: a prefix onto
        # `defect_spec.jsonl` would replace that file with a directory of the same name, and the
        # failure surfaces stages later as a JSON parse error on something that is not a file.
        if target.is_file():
            if len(entries) != 1 or not entries[0][1]:
                raise RuntimeError(
                    f"override {rel!r} is a file in the fetched input, but the source(s) given for it "
                    f"name a prefix: {[url for url, _, _ in entries]}. Give the one file that replaces it."
                )
            url = entries[0][0]
            target.unlink()
            _copy_one(url, target)
            written += 1
            log.info(f"[fetch] override {rel} <- {url}")
            continue

        shutil.rmtree(target)
        target.mkdir(parents=True)
        for url, one_file, found in entries:  # in order, later winning, as the destination's own do
            if one_file:
                # Its own name, not the key's: the key is the directory. A stem still has to match
                # its cad_mask for a cad texture, and renaming under the hood would break that in a
                # way nothing here could report.
                _copy_one(url, _placeable(target, _basename(url)))
                written += 1
                log.info(f"[fetch] override {rel} <- {url}")
                continue
            for obj, sub in found:
                _copy_one(obj, _placeable(target, sub))
                written += 1
            log.info(f"[fetch] override {rel} <- {len(found)} object(s) from {url}")
    return written


def _basename(source: str) -> str:
    """The file name ``source`` ends in, for a source that names one file.

    The last segment, split on ``/``, for a local path and a URL alike -- ``Path`` would normalise a
    URL as if it were a filesystem path, and nothing here wants it to.
    """
    name = str(source).rstrip("/").rsplit("/", 1)[-1]
    if not name or name in (".", ".."):
        raise RuntimeError(f"{source!r} names a single file but does not end in a file name.")
    return name


def _placeable(staged: Path, rel: str) -> Path:
    """``staged / rel``, refusing anything that would land outside ``staged``.

    A listing is remote input, so its relative paths are too. An S3 key with a double slash -- legal
    in S3 -- yields a ``rel`` beginning with ``/``, and ``Path(staged) / "/etc/cron.d/job"`` IS
    ``/etc/cron.d/job``; an NGC entry containing ``..`` climbs out the same way. The file would then
    survive the swap, outside the tree, while the run reported N files fetched.
    """
    target = staged / rel
    try:
        inside = target.resolve().is_relative_to(staged.resolve())
    except OSError:  # a symlinked component that cannot be resolved is not one to trust either
        inside = False
    if not inside:
        raise RuntimeError(
            f"listing returned {rel!r}, which does not land inside the staging directory. A relative "
            "path that is absolute or climbs out of the tree is not one this can place."
        )
    return target


def _marker_for(destination: Path, one_file: bool) -> Path:
    """Where the "a fetch wrote this" marker for ``destination`` lives.

    Inside a directory, beside a file. A file cannot carry a marker within itself, and the
    alternative -- trusting that a file destination is always ours -- is what let a fetch onto
    ``checkpoint:`` overwrite whatever was there.
    """
    return destination.parent / f".{destination.name}{FETCH_MARKER}" if one_file else destination / FETCH_MARKER


def _refuse_foreign_destination(destination: Path, one_file: bool) -> None:
    """Refuse to replace a destination that a previous fetch did not create.

    ``fetch`` replaces its destination wholesale, and the destination is whatever the config field
    already says. ``input: /data/tht`` plus a ``staging.fetch.input`` therefore deleted a 5k-image
    dataset and put the download in its place -- the same-path guard does not catch it, because the
    source and the destination are different paths.

    A destination a fetch wrote carries a marker; one it did not is somebody's data. That is the
    only claim being made here: the marker is not proof of anything except that this code put the
    contents there, which is exactly the question "may I delete it" turns on. An operator who wants
    a directory adopted -- a PVC whose filesystem left a ``lost+found`` in it, a tree fetched before
    the marker existed -- creates the marker themselves, which the message says; that is an explicit
    assertion of ownership rather than a config flag that would have to mean the same thing.

    The KIND is checked too. A prefix fetch onto an existing file used to pass this guard (it is not
    a directory, so there was nothing to refuse) and die in ``shutil.rmtree`` with a bare
    ``NotADirectoryError`` from three frames down, which names neither the path nor the config key.
    """
    marked = _marker_for(destination, one_file).exists()
    if not destination.exists():
        return

    if one_file and destination.is_dir():
        raise RuntimeError(
            f"{destination} is a directory, but this fetch names a single file. The config key "
            "points at something a previous prefix fetch wrote; give the file its own path."
        )
    if not one_file and not destination.is_dir():
        raise RuntimeError(
            f"{destination} is a file, but this fetch lists a prefix and would replace it with a "
            "directory. The config key points at the wrong thing, or the source should have been "
            "given on its own as a single object."
        )
    if marked:
        return
    if not one_file and not any(destination.iterdir()):
        return

    what = "file" if one_file else "directory"
    raise RuntimeError(
        f"{destination} is an existing {what} and no previous fetch wrote it, so replacing it would "
        f"destroy data this run did not put there. fetch REPLACES its destination -- point the "
        f"config key at a path this run owns, remove {destination} if it really is disposable, or "
        f"adopt it deliberately with `touch {_marker_for(destination, one_file)}`."
    )


def _names_one_file(source: str) -> bool:
    """True when ``source`` is a single file rather than a prefix or a directory."""
    return is_object(source) if is_remote(source) else Path(source).is_file()


def _list_source(source: str) -> list:
    """``(source, path relative to it)`` for everything under ``source``, remote or local."""
    if is_remote(source):
        return list(iter_objects(source))
    root = Path(source)
    return [(str(f), f.relative_to(root).as_posix()) for f in sorted(root.rglob("*")) if f.is_file()]


def _copy_one(source: str, local: Path) -> None:
    """Put one file at ``local``, downloading or copying as the source requires.

    A local copy is written aside and renamed like a download, for the same reason: a half-written
    file at the final path is indistinguishable from a complete one on the next run.
    """
    if is_remote(source):
        download(source, local)
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(f".{local.name}.partial")
    try:
        shutil.copy2(source, tmp)
        tmp.replace(local)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"copy failed: {source} -> {local}: {exc}") from exc
