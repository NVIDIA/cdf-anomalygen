# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one owner of SAM2 in this package: the model, its checkpoint, and its lifetime.

Nothing else builds one. The router makes a fresh ``Sam2VP`` per clean, so a SAM2 apiece would be
hundreds of models on the GPU and an OOM; this is ONE model, built once, called from the driver's
single-threaded precompute."""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from cosmos_framework.utils import log
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from anomalygen.models.device import best_cuda_device

# Repo-relative so the path is CWD-independent; provisioned by scripts/download_checkpoints.sh.
CKPT = str(
    Path(__file__).resolve().parents[2] / "checkpoints" / "facebook" / "sam2.1-hiera-large" / "sam2.1_hiera_large.pt"
)
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # a hydra config NAME resolved by the sam2 package, not a path

_GENERATOR = None


def segmenter() -> SAM2AutomaticMaskGenerator:
    """The shared 'segment everything' generator, built on first use.

    The grid and the crop count are the library's own defaults, spelled out. What this changes is the
    two keep-filters, RELAXED from 0.8/0.95 to 0.6/0.85: a visual-prompt structure hint wants more of
    the board outlined, not only the regions SAM2 is confident about."""
    global _GENERATOR
    if _GENERATOR is None:
        dev = best_cuda_device()
        log.info(f"SAM2 segmenter loading on {dev}")
        # Build UNDER the chosen device, not merely move to it: PositionEmbeddingSine warms its
        # cache on the current device at construction and never moves it, so a build on cuda:0
        # would re-copy the position encodings cross-GPU on every set_image.
        ctx = torch.cuda.device(dev) if dev.startswith("cuda") else contextlib.nullcontext()
        with ctx:
            _GENERATOR = SAM2AutomaticMaskGenerator(
                build_sam2(CFG, CKPT, device=dev),
                points_per_side=32,
                pred_iou_thresh=0.6,
                stability_score_thresh=0.85,
                crop_n_layers=0,
            )
    return _GENERATOR


def release() -> None:
    """At the end of a precompute pass, hand back the per-image features SAM2 caches on every call —
    not the model — because they accumulate across hundreds of cleans in one process."""
    if _GENERATOR is not None:
        try:
            _GENERATOR.predictor.reset_predictor()
        except Exception:  # noqa: BLE001 — best-effort; not every SAM2 version exposes it
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
