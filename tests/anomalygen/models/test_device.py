# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Placement of the single-GPU towers (SAM2, DINOv2): the roomiest GPU, and inputs follow the model.

Everything here stubs the CUDA probes, so the suite runs identically on CPU-only runners and GPU
hosts. The recording backbone keeps its parameter on the ``meta`` device — a device no independent
derivation would ever produce — so an input that fails to follow the model is caught on any runner.
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anomalygen.models import device as device_mod

_GB = 1 << 30


def _stub_cuda(monkeypatch, free_bytes, distributed=False, current=0):
    """NVML-path stubs: one free-bytes entry per visible GPU; empty means CUDA is unavailable."""
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: bool(free_bytes))
    monkeypatch.setattr(device_mod.torch.cuda, "device_count", lambda: len(free_bytes))
    monkeypatch.setattr(
        device_mod.torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=96 * _GB)
    )
    monkeypatch.setattr(device_mod.torch.cuda, "device_memory_used", lambda i: 96 * _GB - free_bytes[i])
    monkeypatch.setattr(device_mod.torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(device_mod.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(device_mod.torch.distributed, "is_initialized", lambda: distributed)
    # The probe must stay NVML-only: mem_get_info creates a CUDA context on every GPU it touches.
    monkeypatch.setattr(device_mod.torch.cuda, "mem_get_info", lambda i: pytest.fail("mem_get_info creates a context"))


@pytest.mark.parametrize(
    "free_bytes, expected",
    [
        pytest.param([9 * _GB, 96 * _GB, 96 * _GB, 40 * _GB], "cuda:1", id="server hogs gpu0: roomiest of the rest"),
        pytest.param([96 * _GB], "cuda:0", id="single gpu"),
        pytest.param([50 * _GB, 50 * _GB, 10 * _GB], "cuda:0", id="tie goes to the lowest index"),
        pytest.param([], "cpu", id="no gpu: cpu path"),
    ],
)
def test_best_cuda_device_picks_the_roomiest_gpu(monkeypatch, free_bytes, expected):
    _stub_cuda(monkeypatch, free_bytes)
    assert device_mod.best_cuda_device() == expected


def test_a_pinned_rank_stays_on_its_own_device(monkeypatch):
    """Under torchrun every other GPU belongs to another rank; the probe must not pick one."""
    _stub_cuda(monkeypatch, [9 * _GB, 96 * _GB, 96 * _GB, 96 * _GB], distributed=True, current=0)
    assert device_mod.best_cuda_device() == "cuda:0"


def test_sam2_builds_on_the_picked_device(monkeypatch):
    """The segmenter hands best_cuda_device()'s answer to build_sam2 and builds under that device."""
    from anomalygen.agentic_mask_generation import sam2 as sam2_mod

    seen = {}
    monkeypatch.setattr(sam2_mod, "_GENERATOR", None)
    monkeypatch.setattr(sam2_mod, "best_cuda_device", lambda: "cuda:1")
    monkeypatch.setattr(
        sam2_mod.torch.cuda, "device", lambda dev: (seen.setdefault("ctx", dev), contextlib.nullcontext())[1]
    )
    monkeypatch.setattr(sam2_mod, "build_sam2", lambda cfg, ckpt, device: seen.setdefault("device", device))
    monkeypatch.setattr(sam2_mod, "SAM2AutomaticMaskGenerator", lambda model, **kw: "generator")
    assert sam2_mod.segmenter() == "generator"
    assert seen["device"] == "cuda:1"
    assert seen["ctx"] == "cuda:1"  # built UNDER the device: SAM2 warms caches on the current device


def test_dinov2_loads_and_compiles_on_the_picked_device(monkeypatch):
    from anomalygen.models.vision_encoder import dinov2 as dinov2_mod

    seen = {}

    class _Model:
        def eval(self):
            return self

    monkeypatch.setattr(dinov2_mod, "best_cuda_device", lambda: "cuda:1")
    monkeypatch.setattr(
        dinov2_mod.AutoModel,
        "from_pretrained",
        staticmethod(lambda model_id, device_map: (seen.setdefault("device_map", device_map), _Model())[1]),
    )
    monkeypatch.setattr(dinov2_mod.torch, "compile", lambda m, **kw: (seen.setdefault("compiled", True), m)[1])
    monkeypatch.setattr(dinov2_mod, "_model_cache", {})
    dinov2_mod.get_dinov2_model("facebook/dinov2-large")
    assert seen["device_map"] == "cuda:1"
    assert seen.get("compiled")  # torch.compile must fire for cuda:N, not only a literal "cuda"


# ---------------------------------------------------------------- inputs follow the model's device


class _RecordingBackbone:
    """A backbone whose parameter lives on ``meta`` and which records its inputs' devices.

    Returns correctly shaped CPU tokens so the callers' post-processing runs unchanged.
    """

    def __init__(self, dim=8):
        self.dim = dim
        self.seen: list[torch.device] = []
        self._p = torch.nn.Parameter(torch.empty(1, device="meta"))

    def parameters(self):
        yield self._p

    def __call__(self, pixel_values=None, output_hidden_states=False):
        self.seen.append(pixel_values.device)
        n, _, h, w = pixel_values.shape
        tokens = torch.zeros(n, 1 + (h // 14) * (w // 14), self.dim)
        return SimpleNamespace(last_hidden_state=tokens, hidden_states=[tokens] * 13)


def test_dinov2_embed_input_follows_the_model():
    from PIL import Image

    from anomalygen.agentic_mask_generation import exemplars as ex_mod

    model = _RecordingBackbone()
    emb = ex_mod._dinov2_embed(model, Image.new("RGB", (64, 64)))
    assert model.seen == [torch.device("meta")]
    assert emb.shape == (model.dim,)


def test_extract_defect_features_input_follows_the_model(monkeypatch):
    from anomalygen.eval import correspondence as corr_mod

    model = _RecordingBackbone()
    # The pixel mask rides to the model's device with the image; a meta mask cannot be evaluated,
    # so project it to a concrete all-True patch grid and let the rest of the function run.
    monkeypatch.setattr(corr_mod, "_patch_grid_mask", lambda m, h_p, w_p, patch_size=14: torch.ones(h_p, w_p))
    feats = corr_mod._extract_defect_features(model, np.ones((64, 64, 3), np.float32), np.ones((64, 64), np.float32))
    assert model.seen == [torch.device("meta")]
    assert feats is not None and feats.device.type == "cpu"


def test_region_feature_sets_input_follows_the_model():
    from anomalygen.eval import correspondence as corr_mod

    model = _RecordingBackbone()
    image = np.ones((64, 64, 3), np.float32)
    mask = np.zeros((64, 64), np.float32)
    mask[8:40, 8:40] = 1.0  # one instance, so _zoom_regions yields one crop
    feats = corr_mod._region_feature_sets(
        model, image, mask, region_policy="zoom", layer=-1, patch_size=14, mean=(0.5,) * 3, std=(0.5,) * 3
    )
    assert model.seen and all(d == torch.device("meta") for d in model.seen)
    assert feats and all(f.device.type == "cpu" for f in feats)
