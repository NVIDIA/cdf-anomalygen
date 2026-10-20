# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PIL glue for Cosmos 3's content-safety guardrail, for the standalone texture SDG script.

The framework's ``GuardrailRunner`` (``cosmos_framework.auxiliary.guardrail.common.core``) already
is the guardrail abstraction: ``generate.py`` builds the text and image runners directly from
``presets.create_{text,video}_guardrail_runner`` and calls ``run_safety_check`` on them. This module
only adds what the framework does not provide for this repo: the ``[C,T,H,W]``-float API is
video-tensor oriented, whereas our backend produces PIL images — so :func:`guard_image` converts a
PIL image to the ``[1,H,W,C]`` uint8 frames the runner expects, runs it, and converts back. It also
holds the blocked-sample manifest helpers, and :func:`create_guardrail_runners`. ``cosmos_framework``
is imported only inside that function, so the module stays importable on CPU without downloading any
models; everything else takes the runner as a duck-typed argument.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile
from typing import Any

import nltk
import numpy as np
import PIL.Image
from huggingface_hub import constants as hf_constants

# Columns of the separate blocked-sample manifest (guardrail_blocked.csv). Kept out of
# texture_ft_generation_result.csv because downstream (pseudo_label/filter) reads that manifest
# by output_filename and would choke on rows pointing at files that were never saved.
BLOCKED_CSV_HEADER = [
    "index",
    "output_idx",
    "anomaly_type",
    "image_filename",
    "mask_filename",
    "guardrail",
    "message",
]


GUARDRAIL_REPO_DIR = "models--nvidia--Cosmos-Guardrail1"


def _read_guardrail_corpora_for_nltk() -> str | None:
    """Let nltk read the guardrail's corpora out of the Hugging Face cache.

    nltk >= 3.10 fixed CVE-2026-54293 (path traversal in ``nltk.data.load()``) by restricting
    loads to an allowlist built from ``NLTK_DATA`` and ``nltk.data.path``. Cosmos-Guardrail1 ships
    its corpora inside the HF cache, which is not on that list, so the guardrail dies with
    ``PermissionError: Security Violation [pathsec.open]``. Add just that one model directory —
    not the whole cache — so the traversal fix keeps applying everywhere else.

    The cache location is read from the environment on every call, not from
    ``huggingface_hub.constants``: those are frozen when that module is first imported, and
    generate.py does ``os.environ.setdefault("HF_HUB_CACHE", <repo>/checkpoints/hf)`` at import,
    which can land afterwards. Reading the constant would then point at the wrong cache.

    Idempotent: returns the model directory whenever it exists, whether this call appended it or a
    previous one already did. Returns None only when the directory is absent.
    """
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        home = os.environ.get("HF_HOME")
        cache = os.path.join(home, "hub") if home else hf_constants.HF_HUB_CACHE
    model_dir = pathlib.Path(cache) / GUARDRAIL_REPO_DIR
    if not model_dir.is_dir():
        return None
    entry = str(model_dir)
    if entry not in nltk.data.path:
        nltk.data.path.append(entry)
    return entry


# Marker written into a materialised nltk_data copy before it is renamed into place, so a directory
# without it is a copy that was interrupted mid-way and is rebuilt rather than trusted.
_COMPLETE_MARKER = ".complete"


def _nltk_data_cache_root() -> pathlib.Path:
    """``$XDG_CACHE_HOME/anomalygen/nltk_data`` (``~/.cache`` when unset): per user, writable under
    every deployment, including containers that run with ``HOME`` pointed at a scratch directory."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(base) / "anomalygen" / "nltk_data"


def _has_symlinks(tree: pathlib.Path) -> bool:
    return any(p.is_symlink() for p in tree.rglob("*"))


def _copy_without_symlinks(src: pathlib.Path, dst: pathlib.Path) -> None:
    shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
    (dst / _COMPLETE_MARKER).touch()


def _materialized_nltk_data(src: pathlib.Path) -> pathlib.Path:
    """A symlink-free copy of ``src``, built once per source path.

    Keyed by the resolved source path: an HF snapshot path carries the revision, so a new revision
    gets a new copy and an unchanged one is reused. The copy is assembled in a sibling temp directory
    and renamed into place, so concurrent ranks either win the rename or find a finished copy; the
    marker distinguishes both from a copy that was interrupted. Falls back to a private temp
    directory when the user cache cannot be written.
    """
    src = src.resolve()
    root = _nltk_data_cache_root()
    dst = root / hashlib.sha256(str(src).encode()).hexdigest()[:16]
    if (dst / _COMPLETE_MARKER).is_file():
        return dst
    tmp = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"{dst.name}.tmp-", dir=root))
        _copy_without_symlinks(src, tmp)
    except OSError:
        # The user cache cannot be written, before or during the copy: a private temp dir per process.
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        fallback = pathlib.Path(tempfile.mkdtemp(prefix="anomalygen-nltk_data-")) / "nltk_data"
        _copy_without_symlinks(src, fallback)
        return fallback
    try:
        os.rename(tmp, dst)
    except OSError as e:
        # Another rank finished first; its copy is complete by construction, since the marker is written
        # before the rename. A marker-less `dst` can only be hand-damaged: say so rather than reuse it.
        shutil.rmtree(tmp, ignore_errors=True)
        if not (dst / _COMPLETE_MARKER).is_file():
            raise RuntimeError(f"{dst} exists without its {_COMPLETE_MARKER} marker; remove it and retry") from e
    return dst


def _dereference_guardrail_nltk_data(model_dir: str) -> list[str]:
    """Shadow every guardrail ``nltk_data`` entry on ``nltk.data.path`` with a symlink-free copy.

    nltk >= 3.10.3 opens data files with ``O_NOFOLLOW`` and refuses a final-component symlink (its
    pathsec TOCTOU guard). The Hugging Face cache stores every file of a snapshot as a symlink into
    ``blobs/``, so the punkt and wordnet data the framework's ``Blocklist`` registers from there are
    unreadable in place. A copy with the links resolved, inserted just ahead of the original, is what
    ``nltk.data.find`` reaches first; the original stays listed so the allowlist it grants is
    unchanged. Only ``nltk_data`` directories below ``model_dir`` are touched — never ``model_dir``
    itself, which :func:`_read_guardrail_corpora_for_nltk` lists as an allowlist root and which holds
    the guardrail's model weights. Returns the copies inserted.
    """
    root = pathlib.Path(model_dir).resolve()
    inserted: list[str] = []
    for entry in list(nltk.data.path):
        if not isinstance(entry, str):
            continue
        original = pathlib.Path(entry)
        resolved = original.resolve()
        if (
            original.name != "nltk_data"
            or resolved == root
            or not resolved.is_relative_to(root)
            or not original.is_dir()
            or not _has_symlinks(original)
        ):
            continue
        copy = str(_materialized_nltk_data(original))
        if copy in nltk.data.path:
            continue
        nltk.data.path.insert(nltk.data.path.index(entry), copy)
        inserted.append(copy)
    return inserted


def _is_real_nltk() -> bool:
    """Whether ``nltk`` is the real package rather than ``assets/nltk_stub``. The stub reads the
    checkpoint's data with plain file I/O, so only the real package needs the workarounds above."""
    return not getattr(nltk, "IS_STUB", False)


def create_guardrail_runners(*, offload_model_to_cpu: bool = False) -> tuple[Any, Any]:
    """Build the framework's ``(text, video)`` guardrail runners. With the real nltk, widen its
    allowlist first and shadow the symlinked corpora after the text runner registers them."""
    # Imported here, not at module scope, for the reason in the module docstring: this module stays
    # importable on CPU without pulling the framework in. ``log`` comes from the framework too.
    from cosmos_framework.auxiliary.guardrail.common import presets
    from cosmos_framework.utils import log

    real_nltk = _is_real_nltk()
    if real_nltk:
        _read_guardrail_corpora_for_nltk()
    text_guardrail = presets.create_text_guardrail_runner(offload_model_to_cpu=offload_model_to_cpu)
    # After the text runner: building it downloads the guardrail and registers its nltk_data.
    model_dir = _read_guardrail_corpora_for_nltk() if real_nltk else None
    if model_dir is not None:
        for copy in _dereference_guardrail_nltk_data(model_dir):
            if pathlib.Path(copy).is_relative_to(_nltk_data_cache_root()):
                log.info(f"[guardrail] nltk reads the blocklist corpora from a symlink-free copy: {copy}")
            else:
                log.warning(
                    f"[guardrail] the user cache is not writable; nltk reads the blocklist corpora from a "
                    f"per-process temp copy that is not removed at exit: {copy}"
                )
    video_guardrail = presets.create_video_guardrail_runner(offload_model_to_cpu=offload_model_to_cpu)
    if not is_enforcing(video_guardrail):
        log.warning(
            "[guardrail] image content-safety screening is NOT enforcing: the framework preset "
            "supplies no safety model, so no generated image can be blocked. Face blurring still "
            "runs. Text screening is unaffected. Recorded as image_guardrail_enforcing=false in the "
            "run summary."
        )
    return text_guardrail, video_guardrail


def is_enforcing(runner: Any) -> bool:
    """Whether ``runner`` can actually reach a *deny* decision.

    A ``GuardrailRunner`` with an empty ``safety_models`` list always returns ``is_safe=True`` — its
    postprocessors (face blur) still run, but nothing can block. That is the current state of the
    framework's video preset, where the content-safety filter is commented out upstream.

    This matters beyond a log line: ``timing_summary.json`` records ``guardrail_enabled``, and a
    reader would reasonably take that as evidence screening occurred. Reporting a flag the operator
    set, rather than a capability the run actually had, is the part worth fixing here — whether the
    upstream model comes back is not ours to decide.
    """
    return bool(getattr(runner, "safety_models", None))


def _pil_to_frames(img: PIL.Image.Image) -> np.ndarray:
    """PIL image -> a single-frame ``[1, H, W, C]`` uint8 RGB array for the guardrail runner."""
    return np.array(img.convert("RGB"), dtype=np.uint8)[None, ...]


def _frames_to_pil(frames: np.ndarray) -> PIL.Image.Image:
    """``[1, H, W, C]`` uint8 array -> PIL RGB image (inverse of :func:`_pil_to_frames`)."""
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"Expected [1, H, W, C] frames, got shape {arr.shape}")
    return PIL.Image.fromarray(arr[0].astype(np.uint8), mode="RGB")


def guard_image(video_guardrail: Any, img: PIL.Image.Image) -> tuple[PIL.Image.Image | None, str]:
    """Run the image guardrail (content-safety check + face-blur) on a PIL image.

    ``video_guardrail`` is a framework ``GuardrailRunner`` (from
    ``presets.create_video_guardrail_runner``). Returns ``(image, message)``: the (possibly
    face-blurred) image, or ``None`` if the content-safety check blocks it. Blocking is unreachable
    with the current preset (its ``safety_models`` list is empty — only the face-blur postprocessor
    runs), but handled for parity with the framework's video guardrail.
    """
    frames = _pil_to_frames(img)
    is_safe, message = video_guardrail.run_safety_check(frames)
    if not is_safe:
        return None, message
    return _frames_to_pil(video_guardrail.postprocess(frames)), message


def blocked_row(
    index: Any,
    output_idx: Any,
    anomaly_type: str,
    image_filename: str,
    mask_filename: str,
    guardrail: str,
    message: str,
) -> dict:
    """Build a ``{"sort_key", "row"}`` payload for the blocked-sample manifest, matching the merge
    shape used for the main CSV so ``_merge_rank_rows`` can gather and order it across ranks.

    ``output_idx`` is the 0-based generated-output index for a per-output image block, or ``-1`` for a
    whole-sample text block (which skips every output of the sample). It is part of the sort key so a
    sample with several blocked outputs yields distinct, individually-identifiable rows.
    """
    return {
        "sort_key": [int(index), int(output_idx)],
        "row": {
            "index": index,
            "output_idx": output_idx,
            "anomaly_type": anomaly_type,
            "image_filename": image_filename,
            "mask_filename": mask_filename,
            "guardrail": guardrail,
            "message": message,
        },
    }
