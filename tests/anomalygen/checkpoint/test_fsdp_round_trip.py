# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-rank FSDP round trip through the checkpointer's real ``save()`` and the load half of ``_load_filtered``.

Under FSDP the trained subset and the Adam moments are DTensors. ``save()`` must write them gathered
(plain tensors, so single-GPU generation can read the file) and the load half must re-shard them.
Runs on two GPUs with ``torch.multiprocessing.spawn``; skipped below that, and by the ``gpu`` marker
on a CPU-only torch. The CPU-side counterpart (a one-rank mesh with DTensor moments) lives in
``test_selective_dcp.py``.
"""

import os
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu


def _geometry():
    """The recipe settings the stamp records, as a stand-in ``model.config``: every save needs one."""
    return SimpleNamespace(
        diffusion_expert_config=SimpleNamespace(
            input_size_ladder=[256, 512],
            crop_mode="scale",
            crop_ratio_range=(1.5, 2.5),
            inference_scale=1.0,
            model_size="nano",
            anomaly_types=[["t", "d"]],
        ),
        rectified_flow_training_config=SimpleNamespace(shift=1),
    )


def _toy(seed: int, mesh):
    """A frozen base layer plus a trainable ``lora_`` layer, the shape the filter and the optimizer see."""
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(seed)
    model = torch.nn.Sequential(OrderedDict(base=torch.nn.Linear(8, 8), lora_head=torch.nn.Linear(8, 4))).cuda()
    model.config = _geometry()
    model.base.requires_grad_(False)
    fully_shard(model, mesh=mesh)
    return model, torch.optim.AdamW(model.lora_head.parameters(), lr=1e-3)


def _checkpointer(save_dir: str):
    """A ``SelectiveCheckpointer`` with only what ``save()`` reads: the filter, no callbacks, a local save dir."""
    import anomalygen.checkpoint.selective_dcp as selective_dcp

    ckpt = selective_dcp.SelectiveCheckpointer.__new__(selective_dcp.SelectiveCheckpointer)
    ckpt.save_keys_filter, ckpt.callbacks = ["lora_"], None
    ckpt.save_to_object_store, ckpt._local_dirname, ckpt.save_s3_backend_key = False, save_dir, None
    return ckpt


def _round_trip(rank: int, world_size: int, save_dir: str) -> None:
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    import anomalygen.checkpoint.selective_dcp as selective_dcp

    def no_dtensor(state) -> bool:
        return not any(
            isinstance(t, DTensor) for s in state.values() for t in (s.values() if isinstance(s, dict) else [s])
        )

    torch.cuda.set_device(rank)
    # The ranks meet through a file in the test's own temp dir: no port to pick, nothing to clean up.
    dist.init_process_group("nccl", init_method=f"file://{save_dir}/rendezvous", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        model, optimizer = _toy(0, mesh)
        model(torch.ones(2, 8, device="cuda")).sum().backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        assert all(isinstance(v, DTensor) for v in get_model_state_dict(model).values())
        assert all(isinstance(s["exp_avg"], DTensor) for s in optimizer.state.values())
        expected = {k: v.full_tensor().cpu() for k, v in get_model_state_dict(model).items() if k.startswith("lora_")}
        expected_moments = [s["exp_avg"].full_tensor().cpu() for s in optimizer.state.values()]

        # The gather itself: values, not key names. Without full_state_dict the state is still
        # name-keyed but every moment is a DTensor, which is the bug this pins.
        gathered = selective_dcp._gather_optimizer_state(model, optimizer)
        assert gathered["state"] and no_dtensor(gathered["state"])

        # The real save: rank 0 writes the four components, gathered through the same calls.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
        _checkpointer(save_dir).save(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False), 1)
        next_draw = torch.rand(3, device="cuda")  # the first draw after the state save() recorded
        dist.barrier()
        ckpt_model = torch.load(
            os.path.join(save_dir, "model", "iter_000000001.pt"), map_location="cpu", weights_only=True
        )
        ckpt_optim = torch.load(
            os.path.join(save_dir, "optim", "iter_000000001.pt"), map_location="cpu", weights_only=True
        )
        # The stamp rides in the model file; the load path pops and checks it before the overlay.
        from anomalygen.checkpoint.utils import pop_and_verify_geometry_stamp

        assert pop_and_verify_geometry_stamp(ckpt_model, model, "iter_000000001.pt") is None  # a toy has no caps
        assert set(ckpt_model) == set(expected), sorted(ckpt_model)  # the filtered subset, nothing else
        assert no_dtensor(ckpt_model) and no_dtensor(ckpt_optim["state"])
        trainer_state = torch.load(os.path.join(save_dir, "trainer", "iter_000000001.pt"), weights_only=True)
        assert trainer_state["iteration"] == 1
        # Every rank's generators are in the file; restoring this rank's replays the draw made after save().
        assert {f"rng_state_{r}" for r in range(world_size)} <= set(trainer_state)
        selective_dcp._restore_rng_state(trainer_state, "iter_000000001.pt")
        assert torch.equal(torch.rand(3, device="cuda"), next_draw)

        # Load half: a differently initialised sharded model and a fresh optimizer take the files back
        # through the same calls _load_filtered makes after popping the stamp.
        fresh, fresh_opt = _toy(1, mesh)
        # The trainer builds its scheduler before the load; it adds initial_lr to the param groups,
        # which the saved groups carry, and torch's re-shard requires the two group layouts to agree.
        torch.optim.lr_scheduler.LambdaLR(fresh_opt, lr_lambda=lambda _step: 1.0)
        set_model_state_dict(
            fresh, model_state_dict=ckpt_model, options=StateDictOptions(full_state_dict=True, strict=False)
        )
        selective_dcp._load_optimizer_state(fresh, fresh_opt, ckpt_optim)

        after = get_model_state_dict(fresh)
        assert all(isinstance(v, DTensor) for v in after.values())  # still sharded
        for key, value in expected.items():
            assert torch.equal(after[key].full_tensor().cpu(), value), key
        # A key outside the subset keeps the fresh model's own value: strict=False, not a wipe.
        assert not torch.equal(
            after["base.weight"].full_tensor(), get_model_state_dict(model)["base.weight"].full_tensor()
        )
        moments = [s["exp_avg"].full_tensor().cpu() for s in fresh_opt.state.values()]
        assert len(moments) == len(expected_moments)
        for got, want in zip(moments, expected_moments):
            assert torch.equal(got, want)
    finally:
        dist.destroy_process_group()


def test_fsdp_checkpoint_round_trips_across_two_ranks():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs to shard across")
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as save_dir:
        mp.spawn(_round_trip, args=(2, save_dir), nprocs=2, join=True)
