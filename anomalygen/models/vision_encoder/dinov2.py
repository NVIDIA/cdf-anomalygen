# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 ViT-L/14 vision-encoder backbone.

Loaded from the local checkpoint (or the HF hub id ``facebook/dinov2-large``) via 🤗
transformers and cached per model id. Used as the frozen patch-feature extractor for
correspondence (NN/MNN) KPI scoring in ``anomalygen.eval.correspondence``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from cosmos_framework.utils import log
from transformers import AutoModel

from anomalygen.models.device import best_cuda_device

# Local checkpoint dir: <repo_root>/checkpoints/facebook/dinov2-large
_DINOV2_DIR = str(Path(__file__).resolve().parents[3] / "checkpoints" / "facebook" / "dinov2-large")
_DINOV2_SPEC = {"patch_size": 14, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

# Backbone id / path -> patch grid + normalisation spec.
BackboneSpec: dict[str, dict] = {
    _DINOV2_DIR: _DINOV2_SPEC,
    "facebook/dinov2-large": _DINOV2_SPEC,
}
DEFAULT_BACKBONE = _DINOV2_DIR
_model_cache: dict[str, torch.nn.Module] = {}


def get_dinov2_model(model_id: str = DEFAULT_BACKBONE) -> torch.nn.Module:
    """Load the DINOv2 backbone once per model_id (CUDA when available, else CPU).

    Raises RuntimeError if the checkpoint can't be loaded (e.g. not pre-downloaded).
    """
    if model_id not in _model_cache:
        device = best_cuda_device()
        log.info(f"Loading {model_id} for correspondence scoring on {device}...")
        try:
            model = AutoModel.from_pretrained(model_id, device_map=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load DINOv2 backbone ({model_id}). Pre-download the checkpoint.") from exc

        # torch.compile only pays off on CUDA; skip it on CPU to avoid a slow, pointless warmup.
        if device.startswith("cuda") and "facebook/dinov2-large" in model_id:
            model = torch.compile(model, mode="default", fullgraph=False)

        model.eval()
        _model_cache[model_id] = model

    return _model_cache[model_id]
