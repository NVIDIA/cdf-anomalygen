# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract of :class:`PerClassLoraLinear`.

For each row ``i`` of the input, with ``c`` the class ``gen_class_ids[i]`` routes it to::

    y[i] = base(x[i]) + (alpha / rank) * B[c] @ A[c] @ x[i]

and every class's adapter takes part in the backward, whether or not the batch contains it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anomalygen.models.texture.per_class_lora import PerClassLoraLinear

NUM_CLASSES = 3
RANK = 4
ALPHA = 8.0
IN_FEATURES = 16
OUT_FEATURES = 8


def _build() -> PerClassLoraLinear:
    """A materialized module. lora_B is non-zero (its trained state), else every class looks alike."""
    torch.manual_seed(0)
    base = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    module = PerClassLoraLinear(base, num_classes=NUM_CLASSES, rank=RANK, alpha=ALPHA)
    module.weight = base.weight
    module.bias = base.bias
    module.lora_A = nn.Parameter(torch.randn(NUM_CLASSES, RANK, IN_FEATURES))
    module.lora_B = nn.Parameter(torch.randn(NUM_CLASSES, OUT_FEATURES, RANK))
    return module


def _expected(module: PerClassLoraLinear, x: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """The contract above, evaluated one row at a time."""
    scale = ALPHA / RANK
    rows = []
    for i in range(x.shape[0]):
        c = int(class_ids[i])
        rows.append(x[i] @ module.weight.T + module.bias + scale * (module.lora_B[c] @ (module.lora_A[c] @ x[i])))
    return torch.stack(rows)


@pytest.mark.parametrize(
    "class_ids",
    [
        [0, 0, 0, 1, 1, 1, 2, 2, 2],  # class-contiguous, as packing lays rows out
        [2, 0, 1, 2, 1, 0, 0, 2, 1],  # interleaved: catches an implementation assuming contiguity
    ],
    ids=["contiguous", "interleaved"],
)
def test_forward_applies_each_rows_own_class_adapter(class_ids):
    # float64 so the tolerance leaves no room for a mis-routed row to pass as rounding.
    module = _build().double()
    ids = torch.tensor(class_ids, dtype=torch.long)
    module.gen_class_ids = ids
    x = torch.randn(len(class_ids), IN_FEATURES, dtype=torch.float64)

    torch.testing.assert_close(module(x), _expected(module, x, ids), rtol=0, atol=1e-12)


def test_adapter_delta_scales_with_alpha_over_rank():
    """The alpha/rank factor multiplies the adapter delta, leaving the frozen base untouched."""
    module = _build().double()
    ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    module.gen_class_ids = ids
    x = torch.randn(4, IN_FEATURES, dtype=torch.float64)

    base_only = torch.nn.functional.linear(x, module.weight, module.bias)
    delta = module(x) - base_only

    torch.testing.assert_close(delta, _expected(module, x, ids) - base_only, rtol=0, atol=1e-12)
    assert not torch.allclose(delta, torch.zeros_like(delta)), "a trained adapter must move the output"


def test_every_class_adapter_receives_a_gradient():
    """Absent classes get a zero grad of full shape, so DDP's all-reduce agrees across ranks."""
    module = _build()
    present, absent = 0, 2
    module.gen_class_ids = torch.full((6,), present, dtype=torch.long)

    # Explicit: importing an inference entry point anywhere in the suite disables autograd globally.
    with torch.enable_grad():
        module(torch.randn(6, IN_FEATURES)).sum().backward()

    for param in (module.lora_A, module.lora_B):
        assert param.grad is not None and param.grad.shape == param.shape
        assert torch.count_nonzero(param.grad[present]) > 0
        assert torch.count_nonzero(param.grad[absent]) == 0


def test_unroutable_batches_fall_back_to_the_frozen_base():
    """Routing that cannot be trusted must not be guessed: emit the frozen base instead.

    Unset (empty) is the non-anomaly path. MORE ids than rows means the rows were sharded away
    (context-parallel, unsupported). Fewer ids than rows is deliberately not here — that is the
    packer's trailing pad segment, which is routable and covered by the next test.
    """
    module = _build()
    x = torch.randn(6, IN_FEATURES)
    base_only = torch.nn.functional.linear(x, module.weight, module.bias)

    for ids in (torch.zeros(0, dtype=torch.long), torch.zeros(7, dtype=torch.long)):
        module.gen_class_ids = ids
        torch.testing.assert_close(module(x), base_only)


def test_pad_rows_beyond_the_routing_vector_keep_the_frozen_base():
    """The packer appends a pad segment to the gen stream, so rows outnumber routing ids.

    Those trailing rows belong to no sample and must come back as the plain base output, while
    every routed row still gets its adapter. Falling back wholesale here — the pre-packing
    behaviour — would silently drop the adapters for an ordinary packed batch.
    """
    module = _build()
    x = torch.randn(6, IN_FEATURES)
    ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)  # 4 routed rows, 2 trailing pad rows
    module.gen_class_ids = ids

    out = module(x)
    base_only = torch.nn.functional.linear(x, module.weight, module.bias)
    torch.testing.assert_close(out[4:], base_only[4:])  # pad rows: untouched base
    torch.testing.assert_close(out[:4], _expected(module, x[:4], ids))  # routed rows: adapted
    assert not torch.allclose(out[:4], base_only[:4]), "routed rows must differ from the base"
