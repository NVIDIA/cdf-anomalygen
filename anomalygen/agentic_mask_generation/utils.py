# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The few helpers that belong to no module in particular.

A function qualifies only if it knows nothing of masks, defects, boards or models, and has more than
one consumer."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from cosmos_framework.utils import log

_clean_hash_cache: dict = {}  # (path, mtime_ns, size) -> hash


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write ``path`` via a scratch file beside it, then ``os.replace`` it into place, so a failed
    write leaves the previous file untouched.

    The scratch name carries pid AND thread id because both writers are real: runs share one cache
    directory across processes and a run's beams are a thread pool, so one shared scratch name would
    let two writers interleave into it. It APPENDS rather than replaces the extension: a glob for
    ``*.npy`` or ``*.png`` must not match a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # nothing on success; the partial write on failure


def clean_hash(clean_path) -> str:
    """Stable cache key = md5 of an image file's bytes (first 8 hex). Mirrors the precompute scripts.
    Memoized on (path, mtime_ns, size); mtime+size keep an edited file from serving a stale hash."""
    p = Path(clean_path)
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _clean_hash_cache:
        return _clean_hash_cache[key]
    h = hashlib.md5(p.read_bytes(), usedforsecurity=False).hexdigest()[:8]
    if key is not None:
        _clean_hash_cache[key] = h
    return h


def slug(t: str) -> str:
    """A string as a filename component. One definition: a second copy whose character class or
    length cap drifts splits a defect type's cached data across two files, each half looking complete."""
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", t)[:80]


def resize_labels(lab: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resample an int label map to (h, w). NEAREST always: these are ids, and interpolation invents
    labels that belong to no structure — a mask fitted to one still looks like a mask downstream."""
    if lab.shape == (h, w):
        return lab.astype(np.int32)
    return cv2.resize(lab.astype(np.int32), (int(w), int(h)), interpolation=cv2.INTER_NEAREST)


__all__ = ["atomic_write", "clean_hash", "log", "resize_labels", "slug"]
