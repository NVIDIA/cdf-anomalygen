# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a single-GPU model loads: the GPU with the most free memory.

A literal ``"cuda"`` is cuda:0, whatever is already running there. On a multi-GPU host whose first
GPU serves a vLLM model — the serving recipes reserve 90% of their GPU's memory by design — a tower
placed by that literal competes with the server for the remainder while the other GPUs idle, and its
allocations fail or degrade. Placement therefore reads each GPU's free memory and takes the
roomiest, lowest index on a tie. The towers this places are ~1 GB, so any GPU fits them: there is no
capacity gate and no sharding fallback to arbitrate (the captioner, whose model may NOT fit one GPU,
layers its own gate on top of the same probe and pick).

Two deliberate constraints:

- The probe reads NVML (`torch.cuda.device_memory_used`), not ``torch.cuda.mem_get_info``: the
  latter creates a ~0.5 GB primary context on EVERY probed GPU — including GPUs other processes
  own — and raises outright on a device in exclusive-process mode.
- Under ``torch.distributed`` (the fine-tune's validation callback runs this under torchrun), the
  answer is the rank's own pinned device: every other GPU belongs to another rank whose memory
  headroom never accounted for a tower parked on it.

Inputs must follow the model (``next(model.parameters()).device``), never re-derive a device of
their own: two independent derivations agree only by accident.
"""

from __future__ import annotations

import torch


def free_bytes() -> list[int]:
    """Free memory per visible GPU, via NVML — no CUDA context is created on any device."""
    return [
        torch.cuda.get_device_properties(i).total_memory - torch.cuda.device_memory_used(i)
        for i in range(torch.cuda.device_count())
    ]


def roomiest(free: "list[int]") -> int:
    """Index of the largest entry; the lowest index wins a tie."""
    return max(range(len(free)), key=lambda i: (free[i], -i))


def best_cuda_device() -> str:
    """``cuda:<i>`` of the GPU with the most free memory, or ``cpu`` without CUDA.

    A pinned rank (``torch.distributed`` initialized) stays on its own device.
    """
    if not torch.cuda.is_available():
        return "cpu"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return f"cuda:{torch.cuda.current_device()}"
    return f"cuda:{roomiest(free_bytes())}"
