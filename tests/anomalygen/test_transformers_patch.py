# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``anomalygen/transformers_patch.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from anomalygen import transformers_patch


def test_the_shim_is_what_transformers_resolves_for_default_rope():
    import anomalygen  # noqa: F401  (the package import is what registers it)

    assert ROPE_INIT_FUNCTIONS["default"] is transformers_patch.default_rope_parameters


_FRAMEWORK_QWEN3_VL = "cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl"
_needs_framework = pytest.mark.skipif(
    importlib.util.find_spec("cosmos_framework") is None, reason="cosmos_framework is not installed"
)


def _default_rope_module(*imports: str) -> str:
    """The module of the ``"default"`` rope init after ``imports``, in a fresh process: this test module
    has already imported the patch, so only a new interpreter shows what an import order leaves."""
    code = "; ".join(f"import {m}" for m in imports) + (
        "; from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as R; print(R['default'].__module__)"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    return out.strip().splitlines()[-1]


@_needs_framework
def test_the_framework_alone_registers_its_own_default():
    """Premise of the next test: without the patch, the framework's ``rope_theta`` reader is the default."""
    assert _default_rope_module(_FRAMEWORK_QWEN3_VL) == _FRAMEWORK_QWEN3_VL


@_needs_framework
@pytest.mark.parametrize(
    "imports",
    [(_FRAMEWORK_QWEN3_VL, "anomalygen"), ("anomalygen", _FRAMEWORK_QWEN3_VL)],
    ids=["framework-first", "anomalygen-first"],
)
def test_the_shim_outlives_the_framework_module_that_registers_its_own(imports):
    """cosmos_framework registers a ``"default"`` that reads ``config.rope_theta``; ours must be the one
    left in place no matter which module was imported first."""
    assert _default_rope_module(*imports) == "anomalygen.transformers_patch"


def test_matches_transformers_for_a_transformers_5_config():
    """A stock config carries the base frequency in ``rope_parameters``; the result must equal what the
    model's own rotary module would compute, or weight init would silently drift for every stock model."""
    config = Qwen3Config(hidden_size=64, num_attention_heads=4, rope_theta=12345.0)
    assert not hasattr(config, "rope_theta"), "premise: transformers 5 configs expose no rope_theta"

    inv_freq, scale = transformers_patch.default_rope_parameters(config)
    expected, expected_scale = Qwen3RotaryEmbedding.compute_default_rope_parameters(config)

    torch.testing.assert_close(inv_freq, expected)
    assert scale == expected_scale == 1.0


def test_reads_the_framework_style_rope_theta_attribute():
    # cosmos_framework's own configs predate rope_parameters and keep the attribute.
    config = SimpleNamespace(rope_theta=10000.0, hidden_size=32, num_attention_heads=2, partial_rotary_factor=0.5)

    inv_freq, _ = transformers_patch.default_rope_parameters(config)

    dim = int((32 // 2) * 0.5)
    torch.testing.assert_close(inv_freq, 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)))


def test_a_config_with_neither_field_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="rope_theta"):
        transformers_patch.default_rope_parameters(SimpleNamespace(hidden_size=32, num_attention_heads=2))
