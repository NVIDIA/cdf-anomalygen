# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-defect-type LoRA on the gen-attention projections, routed per gen token.

A defect type is a *class* (``num_classes`` / ``gen_class_ids`` / ``per_class_lora_*`` keys). Each
type gets its own adapter and every gen token routes to its type's, so all types train in one mixed
batch. The adapters are stacked (``A: [C, r, in]``, ``B: [C, out, r]``) and selected per row::

    y = base(x) + scale * B[c] @ (A[c] @ x),   c = class of each row of x

Routing arrives as the ``gen_class_ids`` buffer, refilled each forward by the eager encode head
(``AnomalyVFMNetwork._encode_vision``) before the compiled blocks run — a buffer, not a Python
attribute (``fullgraph=True`` would bake an attribute as a constant). Needs ``cp_world_size=1`` so every gen
row aligns 1:1 with the routing vector, in ``full_only_seq`` order (== ``sorted(sequence_indexes)``);
one vector serves every layer. The packer pads that stream with a trailing segment (for CUDA-graph
bucketing, and one row unconditionally so attention sees the padding as a segment of its own), which
the routing vector does not cover — those rows are left on the frozen base.

DDP: every class's slice of ``lora_A``/``lora_B`` enters the graph each step, so no parameter is
unused and an absent class gets a zero grad.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class PerClassLoraLinear(nn.Linear):
    """``nn.Linear`` + a stack of ``C`` LoRA adapters, routed per input row by ``gen_class_ids``.

    The adapter params are named ``lora_A``/``lora_B``, so the optimizer's ``keys_to_select=["lora_",
    ...]`` and the checkpoint save-filter pick them up with no extra wiring (see
    ``anomalygen.checkpoint.trained_keys``).
    """

    def __init__(self, base: nn.Linear, num_classes: int, rank: int, alpha: float) -> None:
        # Inheriting nn.Linear makes F.linear(x, self.weight, ...) the base path; reuse base geometry.
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None, device="meta")
        self.weight = base.weight  # keep the pretrained weight identity (and its state-dict key)
        if base.bias is not None:
            self.bias = base.bias
        # Stacked adapters, left on meta; materialized by init_per_class_lora_weights after to_empty.
        self.lora_A = nn.Parameter(torch.empty(num_classes, rank, base.in_features, device="meta"))
        self.lora_B = nn.Parameter(torch.empty(num_classes, base.out_features, rank, device="meta"))
        self._num_classes = int(num_classes)
        self._lora_rank = int(rank)
        self._lora_alpha = float(alpha)
        # Per-gen-token class ids in full_only_seq order, rewritten each forward by the encode head.
        # Non-persistent: transient per-batch routing, never a checkpointed weight.
        self.register_buffer("gen_class_ids", torch.zeros(0, dtype=torch.long, device="meta"), persistent=False)

    @property
    def _lora_scale(self) -> float:
        return self._lora_alpha / self._lora_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        c = self.gen_class_ids
        n = x.shape[0]
        m = c.numel()
        # The routing vector covers the real gen rows. The packer appends a trailing pad segment
        # to the gen stream, so `n` is the routed rows plus that padding; those rows belong to no
        # sample and get no adapter. Empty -> routing unset (non-anomaly path) => frozen base.
        # More ids than rows means the rows were sharded away (context-parallel, unsupported) =>
        # fall back to base rather than mis-route.
        if m == 0 or m > n:
            return base_out
        x_routed = x[:m]
        # Contract against all C adapters and mask with a one-hot: the non-matching classes
        # contribute exactly zero, so this equals the definition above. Both adapters enter the
        # graph on every call (an absent class gets a zero grad), so DDP sees no unused param.
        # Do NOT rewrite as the per-row gather lora_A[c] / lora_B[c]. It has 1/C the FLOPs but
        # expands the adapter to a per-token [m, r, in] copy (134 MiB at m=2048), so it is
        # memory-bound and measures slower in both directions despite the arithmetic saving.
        onehot = F.one_hot(c, self._num_classes).to(x.dtype)  # [m, C]
        z = torch.einsum("cri,ni->ncr", self.lora_A, x_routed)  # [m, C, r]
        d = torch.einsum("cor,ncr->no", self.lora_B, z * onehot.unsqueeze(-1))  # [m, out]
        if m < n:
            d = F.pad(d, (0, 0, 0, n - m))  # pad rows keep the frozen base output
        return base_out + self._lora_scale * d


def inject_per_class_lora(
    net: nn.Module,
    targets: Sequence[str],
    num_classes: int,
    rank: int,
    alpha: float,
) -> int:
    """Replace each named ``nn.Linear`` child in ``targets`` with a :class:`PerClassLoraLinear`.

    Matches by exact child name (like the framework's ``_inject_lora_inplace``), then freezes every
    non-LoRA parameter so the ``keys_to_select=["lora_"]`` optimizer filter trains adapters only.
    Must run on the meta-device network BEFORE FSDP wrap (unsharded shapes). Returns the count.
    """
    target_set = set(targets)
    replaced = 0
    for _parent_name, parent in list(net.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear):
                setattr(parent, child_name, PerClassLoraLinear(child, num_classes, rank, alpha))
                replaced += 1
    for name, param in net.named_parameters():
        param.requires_grad_("lora_" in name)
    return replaced


def init_per_class_lora_weights(net: nn.Module) -> None:
    """Initialise the stacked adapters after ``to_empty`` materialises their storage.

    ``lora_A ~ kaiming_uniform_(a=sqrt(5))``, ``lora_B = 0`` so the initial delta is exactly zero (the
    net starts identical to the frozen base, gradients flow from step 1). Each pair is cast to its
    base weight's dtype so the einsums see matching dtypes in fp32/bf16/fp16.
    """
    for module in net.modules():
        if not isinstance(module, PerClassLoraLinear):
            continue
        with torch.no_grad():
            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B)
            dt = module.weight.dtype
            module.lora_A.data = module.lora_A.data.to(dt)
            module.lora_B.data = module.lora_B.data.to(dt)


def build_gen_class_ids(
    packed_sequence: torch.Tensor,
    token_shapes: Sequence[Sequence[int]],
    sequence_indexes: torch.Tensor,
    class_ids: Sequence[int],
    items_per_sample: int,
) -> Optional[torch.Tensor]:
    """Per-gen-token class-id vector in ``full_only_seq`` order (or ``None`` if unavailable).

    Scatters each vision item's class id onto its packed rows (item -> owning sample -> class),
    then reads them back in ascending packed-row order — which equals ``full_only_seq`` order,
    since ``_full_indices`` is ascending.
    """
    if not isinstance(sequence_indexes, torch.Tensor) or sequence_indexes.numel() == 0 or not token_shapes:
        return None
    ips = int(items_per_sample)
    cids = list(class_ids)
    device = packed_sequence.device
    cls_full = torch.zeros(int(packed_sequence.shape[0]), dtype=torch.long, device=device)
    row_offset = 0
    for item_idx, shape in enumerate(token_shapes):
        n = int(shape[0]) * int(shape[1]) * int(shape[2])
        sample_i = item_idx // ips if ips > 0 else item_idx
        ci = int(cids[sample_i]) if 0 <= sample_i < len(cids) else 0
        cls_full[sequence_indexes[row_offset : row_offset + n]] = ci
        row_offset += n
    return cls_full[torch.sort(sequence_indexes).values]


def set_gen_class_ids(net: nn.Module, gen_class_ids: Optional[torch.Tensor]) -> None:
    """Write ``gen_class_ids`` into every :class:`PerClassLoraLinear`'s routing buffer.

    One vector serves every layer (all gen linears share ``full_only_seq`` order). ``None`` clears
    the buffer to length 0, so a routing-less forward cleanly falls back to the frozen base instead
    of reusing the previous batch's ids.
    """
    for module in net.modules():
        if isinstance(module, PerClassLoraLinear):
            if gen_class_ids is None:
                module.gen_class_ids = torch.zeros(0, dtype=torch.long, device=module.weight.device)
            else:
                module.gen_class_ids = gen_class_ids


def has_per_class_lora(net: nn.Module) -> bool:
    """True when ``net`` holds at least one :class:`PerClassLoraLinear`."""
    return any(isinstance(m, PerClassLoraLinear) for m in net.modules())


def per_class_lora_buffer_names(root: nn.Module) -> List[str]:
    """Fully-qualified ``gen_class_ids`` buffer names under ``root``, for DDP's ignore list.

    Routing is per-batch/per-rank and differs in length across ranks, so ``broadcast_buffers`` would
    mismatch (hang/error). Feed these to the DDP module's ``_ddp_params_and_buffers_to_ignore`` to
    skip them in buffer sync. Names are relative to ``root`` — pass the exact module DDP wraps.
    """
    return [f"{name}.gen_class_ids" for name, module in root.named_modules() if isinstance(module, PerClassLoraLinear)]


def per_class_lora_target_modules() -> List[str]:
    """Default injection targets: the gen-expert attention projections (every layer)."""
    return ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"]
