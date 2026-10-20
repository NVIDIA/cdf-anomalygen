# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shims that reconcile ``cosmos_framework`` with the installed ``transformers``.

Imported by :mod:`anomalygen` before anything else, so they are in place for every entry point.
"""

from __future__ import annotations

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def default_rope_parameters(config, device=None, **kwargs) -> tuple[torch.Tensor, float]:
    """Plain RoPE inverse frequencies for either generation of model config.

    transformers 5 dropped ``"default"`` from ``ROPE_INIT_FUNCTIONS`` and moved the base frequency
    into ``config.rope_parameters``; cosmos_framework puts the key back with a function that reads
    the ``config.rope_theta`` attribute its own configs still carry. Since transformers 5.17 weight
    initialisation resolves that key for every ``RotaryEmbedding`` module, so the framework's version
    is applied to stock models too and fails on their configs (the Qwen3Guard guardrail). This one
    accepts both, and replaces the framework's registration whichever module is imported first.
    """
    params = getattr(config, "rope_parameters", None) or {}
    base = params.get("rope_theta", getattr(config, "rope_theta", None))
    if base is None:
        raise ValueError(f"{type(config).__name__} declares neither rope_parameters['rope_theta'] nor rope_theta")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq.to(device), 1.0


ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters
