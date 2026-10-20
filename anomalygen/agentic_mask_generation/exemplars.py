# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load ``k`` (defect image, mask) exemplars for a (texture, defect) from an AnomalyGen dataset.

Pairs come from the AnomalyGen on-disk layout via :func:`anomalygen.data.utils.list_image_mask_pairs`:

    ``{dataset_path}/{texture}/anomaly_image/{defect}/<name>.<ext>``  (defect image)
    ``{dataset_path}/{texture}/mask/{defect}/<name>_mask.<ext>``      (its binary mask)

Sampling is ``firstk`` (first ``k`` by filename) or ``similar`` (the ``k`` closest to the clean image
by DINOv2, fail-soft to ``firstk``), the latter on the eval's own ``facebook/dinov2-large`` backbone.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.agentic_mask_generation.imaging import resize_mask
from anomalygen.agentic_mask_generation.utils import atomic_write
from anomalygen.data.utils import list_image_mask_pairs
from anomalygen.models.vision_encoder.dinov2 import get_dinov2_model


@dataclass
class Exemplar:
    """One in-context demonstration: a real defect image and its binary mask (both PIL images)."""

    defect: Any  # PIL.Image (RGB)
    mask: Any  # PIL.Image (L)
    # Source paths, so the driver can line this exemplar up with its own distill-time caption and
    # repaint its visual-prompt map at the file's own scale.
    defect_path: Optional[str] = None
    mask_path: Optional[str] = None
    # The exemplar in the candidate's modality — the vp's painted map with the REAL mask outlined.
    # None = absent, changes nothing.
    annotated: Any = None
    # The distill pass's caption of THIS exemplar (host / extent / shape...), rendered next to its
    # images in every prompt. "" = absent, changes nothing — same contract as every optional block.
    observation: str = ""


def _dinov2_embed(model, img: Image.Image, *, patch_size: int = 14, min_size: int = 224, max_size: int = 448) -> Any:
    """L2-normalised CLS-token global embedding of a PIL RGB image, using the eval's DINOv2 backbone.
    Preprocessing mirrors ``anomalygen.eval.correspondence``, so the feature space is the evaluator's."""
    w, h = img.size
    if min(h, w) < min_size:
        s = min_size / min(h, w)
        h, w = round(h * s), round(w * s)
    if max(h, w) > max_size:
        s = max_size / max(h, w)
        h, w = round(h * s), round(w * s)
    h, w = max(patch_size, h - h % patch_size), max(patch_size, w - w % patch_size)
    img = img.resize((w, h), Image.Resampling.BICUBIC)
    device = next(model.parameters()).device
    x = TF.normalize(TF.to_tensor(img), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    with torch.no_grad():
        out = model(pixel_values=x.unsqueeze(0).to(device))
    cls = out.last_hidden_state[:, 0, :].squeeze(0).float().cpu()  # [D]
    return cls / (cls.norm() + 1e-8)


# Bump _EMB_VERSION whenever the model or _dinov2_embed preprocessing changes: cache keys are file
# identity + version, so without a bump every stale vector on disk is reused as if it were current.
_EMB_VERSION = "dinov2-large.cls448.v1"
_EMB_MEM: dict[str, np.ndarray] = {}  # process-global in-memory layer over the disk cache
_DINO_MODEL = None  # once per process: a missing local ckpt raises UNCACHED, so without this every
#                     clean would re-attempt — and re-log — the failing local load


def _dino_model():
    """The eval's DINOv2 backbone, loaded once per process."""
    global _DINO_MODEL
    if _DINO_MODEL is None:
        try:
            _DINO_MODEL = get_dinov2_model()
        except Exception:  # noqa: BLE001 — local ckpt absent → fall back to the hub id (cached after first fetch)
            _DINO_MODEL = get_dinov2_model("facebook/dinov2-large")
        _DINO_MODEL.eval()
    return _DINO_MODEL


def _emb_key(img_path: str) -> str:
    """Identity that fully determines an image's embedding: file (abspath+size+mtime) + embed version."""
    st = os.stat(img_path)
    return f"{os.path.abspath(img_path)}|{st.st_size}|{st.st_mtime_ns}|{_EMB_VERSION}"


def _cached_embed(img_path: str, cache_dir: Optional[str]) -> np.ndarray:
    """L2-normalised DINOv2 CLS embedding of ``img_path`` — memoised in-process and (when ``cache_dir``
    is set) on disk. The backbone loads LAZILY, on a real miss only, so a fully-cached run never loads it."""
    key = _emb_key(img_path)
    hit = _EMB_MEM.get(key)
    if hit is not None:
        return hit
    fp = (
        Path(cache_dir) / f"{hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:20]}.npy"
        if cache_dir
        else None
    )
    if fp is not None and fp.exists():
        e = np.load(fp)
        _EMB_MEM[key] = e
        return e
    e = _dinov2_embed(_dino_model(), Image.open(img_path).convert("RGB")).numpy()  # load backbone on miss
    if fp is not None:
        # Atomically, because a truncated .npy is PERMANENT: `_similar_order` catches the load error
        # and every run from here on ranks by firstk. Through a file HANDLE, not the path: `np.save`
        # appends `.npy` to a name that lacks it, so the scratch path would be written beside the
        # scratch file instead of into it.
        def _save(t: Path) -> None:
            with open(t, "wb") as fh:
                np.save(fh, e)

        atomic_write(fp, _save)
    _EMB_MEM[key] = e
    return e


def _similar_order(pairs: list, clean_path: str, *, cache_dir: Optional[str] = None) -> Optional[list]:
    """Rank ``(defect_img, mask)`` pairs by DINOv2 cosine similarity of the defect image to the clean
    board (most-similar first). Returns ``None`` on ANY failure so the caller falls back to firstk."""
    try:
        ec = _cached_embed(clean_path, cache_dir)
        scored = [(float((ec * _cached_embed(im, cache_dir)).sum()), (im, mk)) for im, mk in pairs]
        scored.sort(key=lambda t: -t[0])
        return [p for _, p in scored]
    except Exception as e:  # noqa: BLE001 — firstk is always a valid answer; ranking is the extra
        # ...but falling back to it in SILENCE is not: nothing else records which exemplars a run
        # actually used, so an unlogged fallback is indistinguishable from a run that really ranked
        # them. `os.path.basename(str(...))`, not `Path(clean_path).name`: `clean_path`
        # may be None here, and a TypeError raised INSIDE the handler replaces a fail-soft with a crash.
        log.warning(
            f"degraded[exemplar_ranking_unavailable]: {os.path.basename(str(clean_path))}: "
            f"{type(e).__name__}: {e} — ranking skipped, taking the first k instead"
        )
        return None


def load_exemplars(
    dataset_path: str,
    texture: str,
    defect: str,
    *,
    k: int = 3,
    sampling: str = "firstk",
    max_side: int = 640,
    clean_path: Optional[str] = None,
    emb_cache_dir: Optional[str] = None,
) -> list[Exemplar]:
    """Load up to ``k`` ``(defect image, mask)`` exemplars for ``texture``/``defect``: defect RGB,
    mask ``L`` (0/255), both at the defect image's ``max_side`` thumbnail size. ``sampling="firstk"``
    takes the first ``k`` pairs by filename; ``sampling="similar"`` (needs ``clean_path``) takes the
    ``k`` closest to the clean board by DINOv2, fail-softing to firstk."""
    img_dir = f"{dataset_path}/{texture}/anomaly_image/{defect}"
    mask_dir = f"{dataset_path}/{texture}/mask/{defect}"
    pairs = [(im, mk) for im, mk in list_image_mask_pairs(img_dir, mask_dir, "_mask") if os.path.exists(mk)]
    if not pairs:
        raise FileNotFoundError(f"no (image, mask) pairs found for {texture}/{defect} under {dataset_path}")

    if sampling == "similar" and clean_path:  # rank by DINOv2 closeness to the clean (None -> firstk)
        ordered = _similar_order(pairs, clean_path, cache_dir=emb_cache_dir)
        if ordered is not None:
            pairs = ordered
    chosen = pairs[:k]
    out: list[Exemplar] = []
    for img_path, mask_path in chosen:
        defect_img = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        defect_img.thumbnail((max_side, max_side))
        # The mask follows the image's thumbnail size through `resize_mask` (NEAREST, so it stays
        # 0/255), never `thumbnail`: an interpolating thumbnail greys a thin annotation below the
        # `> 127` every reader applies, and preflight checks the full-resolution file, so only this
        # step could lose it.
        mask_img = resize_mask(mask_img, defect_img.size)
        out.append(Exemplar(defect=defect_img, mask=mask_img, defect_path=img_path, mask_path=mask_path))
    return out


class DatasetError(RuntimeError):
    """A dataset defect that stops the run before it spends anything."""


@dataclass(frozen=True)
class PreflightReport:
    """What preflight looked at, and what it cost."""

    types: int
    pairs: int
    cleans: int
    seconds: float
    # Per type, the same numbers `pairs` totals. The proportional validation split weights each type
    # by how many real training masks it has, and this walk already visited every one of them —
    # recomputing them elsewhere would mean a second pass over the tree to learn what was just counted.
    per_type: "dict[str, int]" = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"dataset preflight OK: {self.pairs} (image, mask) pairs across {self.types} type(s) "
            f"+ {self.cleans} clean image(s), in {self.seconds:.1f}s"
        )


def _check_pair(img_path: str, mask_path: str) -> str:
    """One (defect image, mask) pair, or a sentence saying what is wrong with it."""
    try:
        mask = Image.open(mask_path)
        mask.load()  # Image.open is LAZY — this is what reads the pixels
    except Exception as e:  # noqa: BLE001 — every failure mode is reportable, none is fatal here
        return f"{mask_path}: unreadable mask ({type(e).__name__}: {e})"
    try:
        img = Image.open(img_path)
        img.load()
    except Exception as e:  # noqa: BLE001
        return f"{img_path}: unreadable defect image ({type(e).__name__}: {e})"
    try:
        hi = mask.convert("L").getextrema()[1]
    except Exception as e:  # noqa: BLE001 — `convert` raises on some source modes (La, LAB), and this
        # is a gate: escaping as a non-DatasetError would turn a refusal into a traceback.
        return f"{mask_path}: mask cannot be read as greyscale (mode {mask.mode}; {type(e).__name__}: {e})"
    if hi <= 127:
        # Readable is not usable: an all-black mask loads fine and is then silently skipped by
        # patch_histogram and _widen_to_examples, dropping a demonstration from a pool of three.
        return (
            f"{mask_path}: mask has pixels but none above the {{0,255}} threshold (max={hi}) — it "
            "looks 0/1-encoded, not 0/255"
            if hi > 0
            else f"{mask_path}: mask is entirely zero (no annotated region)"
        )
    if mask.size != img.size:
        return f"{mask_path}: mask is {mask.size}, but its defect image {img_path} is {img.size}"
    return ""


def _check_image(path: str) -> str:
    try:
        Image.open(path).load()
    except Exception as e:  # noqa: BLE001
        return f"{path}: unreadable ({type(e).__name__}: {e})"
    return ""


def preflight(
    dataset_path: str,
    types,
    cleans_by_texture: Optional[dict] = None,
    *,
    workers: int = 16,
    max_reported: int = 20,
) -> PreflightReport:
    """Read every exemplar pair and clean image the run may touch, ONCE, before the first model call
    (preflight runs before every side effect and every model call).

    Global and all-at-once, because a broken file in a LATE type otherwise surfaces only after every
    earlier type has been paid for, one file per round. Discovery goes through the SAME `list_image_mask_pairs`
    the loader uses, so a stray file the loader never pairs (mask directories exported from Windows
    carry a ``Thumbs.db``) cannot refuse a run that would have succeeded. Every declared type is checked, so
    one bad file fails the whole run.

    Threaded because it is pure I/O: on a large exemplar tree discovery drops from tens of seconds
    serially to a few seconds threaded; a small tree is instant either way.
    """
    t0 = time.perf_counter()
    types = tuple(types)  # counted AND iterated; a generator would report 0 types
    jobs, structural, problems = [], [], []
    n_pairs = 0
    per_type: dict = {}
    for texture, defect in types:
        img_dir = f"{dataset_path}/{texture}/anomaly_image/{defect}"
        mask_dir = f"{dataset_path}/{texture}/mask/{defect}"
        pairs = [(im, mk) for im, mk in list_image_mask_pairs(img_dir, mask_dir, "_mask") if os.path.exists(mk)]
        if not pairs:
            structural.append(f"{texture}/{defect}: no (image, mask) pairs under {img_dir}")
            continue
        n_pairs += len(pairs)
        per_type[f"{texture}+{defect}"] = len(pairs)
        jobs += [(_check_pair, (im, mk)) for im, mk in pairs]
    cleans = [str(p) for t, ps in (cleans_by_texture or {}).items() for p in ps]
    jobs += [(_check_image, (c,)) for c in cleans]

    with ThreadPoolExecutor(max(1, workers)) as ex:
        problems += [p for p in ex.map(lambda j: j[0](*j[1]), jobs) if p]

    took = time.perf_counter() - t0
    if structural or problems:
        # STRUCTURAL problems first and never truncated away: a whole missing type is the most
        # actionable line, and sorted in with the file messages it lands LAST — its relative
        # "<texture>/<defect>" sorts after their absolute paths, exactly where truncation hides it.
        shown = structural + sorted(problems)[:max_reported]
        hidden = max(0, len(problems) - max_reported)
        more = f"\n  ... and {hidden} more file(s)" if hidden else ""
        counts = ", ".join(
            x
            for x in (
                f"{len(structural)} type(s) with no data" if structural else "",
                f"{len(problems)} unusable file(s)" if problems else "",
            )
            if x
        )
        raise DatasetError(f"{counts} in {dataset_path} — refusing to start:\n  " + "\n  ".join(shown) + more)
    return PreflightReport(types=len(types), pairs=n_pairs, cleans=len(cleans), seconds=took, per_type=per_type)


def patch_histogram(dataset_path: str, texture: str, defect: str, cap: int = 200) -> dict[int, int]:
    """{separate patches in one real mask: how many masks have that many}, over the type's own GT.
    The evidence for `distribution_kind`, counted from the ANNOTATIONS rather than asked of a model —
    a model answers how many defect EVENTS it thinks a photo shows, not how many patches were drawn."""

    img_dir = f"{dataset_path}/{texture}/anomaly_image/{defect}"
    mask_dir = f"{dataset_path}/{texture}/mask/{defect}"
    out: dict[int, int] = {}
    for _im, mk in list_image_mask_pairs(img_dir, mask_dir, "_mask")[:cap]:
        if not os.path.exists(mk):
            continue
        try:
            a = (np.asarray(Image.open(mk).convert("L")) > 127).astype(np.uint8)
        except Exception as e:  # noqa: BLE001 — one unreadable mask must not sink the histogram
            log.warning(f"degraded[real_mask_unreadable]: {mk}: {type(e).__name__}: {e}")
            continue
        if not a.any():
            continue
        n = int(cv2.connectedComponents(a)[0] - 1)
        out[n] = out.get(n, 0) + 1
    return dict(sorted(out.items()))


def count_pairs(dataset_path: str, texture: str, defect: str) -> int:
    """Number of available ``(image, mask)`` exemplar pairs for ``texture``/``defect`` — the same
    filename discovery :func:`load_exemplars` uses, but WITHOUT loading any images."""
    img_dir = f"{dataset_path}/{texture}/anomaly_image/{defect}"
    mask_dir = f"{dataset_path}/{texture}/mask/{defect}"
    return sum(1 for _, mk in list_image_mask_pairs(img_dir, mask_dir, "_mask") if os.path.exists(mk))
