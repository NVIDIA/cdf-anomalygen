# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the anomalygen test tree."""

from types import SimpleNamespace

import numpy as np
import pytest
from hydra.core.config_store import ConfigStore
from PIL import Image


@pytest.fixture
def mask_with_square():
    """Factory for an "L" mask with one filled square: ``mask_with_square(size, lo, hi)``."""

    def _make(size=64, lo=20, hi=40):
        arr = np.zeros((size, size), dtype=np.uint8)
        arr[lo:hi, lo:hi] = 255
        return Image.fromarray(arr, mode="L")

    return _make


@pytest.fixture
def stub_model():
    """Factory for the bare model stand-in the config readers and the stamp accept: a ``config`` carrying
    the sampler shift and the recipe settings the stamp records, plus the ``class_ladders`` training attaches."""

    def _make(
        shift=1,
        ladder=(256, 512, 1024, 1536),
        crop_mode="scale",
        model_size="nano",
        anomaly_types=(("t", "big"), ("t", "small")),
        class_ladders=None,
    ):
        return SimpleNamespace(
            config=SimpleNamespace(
                diffusion_expert_config=SimpleNamespace(
                    input_size_ladder=list(ladder),
                    crop_mode=crop_mode,
                    crop_ratio_range=(1.5, 2.5),
                    inference_scale=1.0,
                    model_size=model_size,
                    anomaly_types=[list(p) for p in anomaly_types],
                ),
                rectified_flow_training_config=SimpleNamespace(shift=shift),
            ),
            class_ladders=class_ladders,
        )

    return _make


@pytest.fixture
def register_experiment():
    """Register throwaway ``experiment`` nodes in the global ConfigStore, removed again on teardown.

    The node carries only ``model.config.diffusion_expert_config`` with the given fields, which is
    what the readers under test look at: no model build, no checkpoint, no GPU.
    """
    names = []

    def _register(name, **expert_config):
        ConfigStore.instance().store(
            group="experiment",
            package="_global_",
            name=name,
            node={"model": {"config": {"diffusion_expert_config": expert_config}}},
        )
        names.append(f"{name}.yaml")
        return name

    yield _register

    repo = ConfigStore.instance().repo.get("experiment", {})
    for key in names:
        repo.pop(key, None)
