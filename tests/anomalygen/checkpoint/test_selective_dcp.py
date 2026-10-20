# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SelectiveCheckpointer's restricted checkpoint loading.

``load()`` reads all four components — ``model``, ``optim``, ``scheduler``, ``trainer`` — with
``weights_only=True``. Model state is plain tensors and loads under that restriction unmodified;
the other three are why the module registers a numpy allowlist at import. A missing entry in that
allowlist does not fail at save time. It fails on the way back, on a resume, hours into a run.

So the check that matters is a real training checkpoint, which is gated on one existing. The CPU
round trip below covers the same shape in CI, and deliberately carries numpy scalars: a bare
``AdamW`` state dict contains none, so a round trip built only from stock torch objects would pass
whether or not the allowlist were there, and prove nothing.
"""

import glob
import os
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.tensor import DTensor

import anomalygen.checkpoint.selective_dcp as selective_dcp
from anomalygen.checkpoint.selective_dcp import _COMPONENTS
from anomalygen.checkpoint.utils import GEOMETRY_STAMP_KEY

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(selective_dcp.__file__)), "..", "results")


def _real_component(name):
    """Newest saved <component>/iter_*.pt from any completed training run, or None."""
    found = sorted(glob.glob(os.path.join(_RESULTS, "**", "checkpoints", name, "iter_*.pt"), recursive=True))
    return found[-1] if found else None


def test_module_registers_the_numpy_allowlist_on_import():
    """Importing the module is what makes the restricted load work — the loads rely on that."""
    allowed = {getattr(g, "__name__", str(g)) for g in torch.serialization.get_safe_globals()}
    assert "scalar" in allowed, "numpy scalar reconstructor must be allowlisted"
    assert any(name.endswith("DType") for name in allowed), "numpy dtype classes must be allowlisted"


@pytest.mark.parametrize("component", _COMPONENTS)
def test_real_training_checkpoint_component_loads_under_the_restriction(component):
    """The check that answers the question: a component this pipeline actually wrote.

    Skips where no training run is present (CI); runs anywhere `results/` has one.
    """
    path = _real_component(component)
    if path is None:
        pytest.skip(f"no saved {component} checkpoint under results/")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded, f"{component} checkpoint loaded empty"


def test_saved_component_shape_round_trips_under_the_restriction(tmp_path):
    """The four-key shape ``save()`` writes, carrying the numpy scalars real optimizer state has."""
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    # enable_grad explicitly: importing the pipeline scripts runs init_script(training=False),
    # which disables grad process-wide, so this backward fails when the full suite runs it after
    # them and passes when this module runs alone.
    with torch.enable_grad():
        model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    scheduler.step()

    optim_state = optimizer.state_dict()
    # Stock torch keeps these as python floats; the framework's optimizer carries numpy scalars, and
    # those are what needed allowlisting. Inject them so this exercises the same path.
    optim_state["_numpy_scalars"] = {"lr": np.float64(1e-3), "beta": np.float32(0.9), "step": np.int64(1)}
    saved = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optim": optim_state,
        "scheduler": scheduler.state_dict(),
        "trainer": {"grad_scaler": grad_scaler.state_dict(), "iteration": 7},
    }
    assert set(saved) == set(_COMPONENTS), "test shape must track _COMPONENTS"

    for component, state in saved.items():
        path = tmp_path / f"{component}.pt"
        torch.save(state, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert set(loaded) == set(state), f"{component} lost keys through a restricted load"

    reloaded = torch.load(tmp_path / "optim.pt", map_location="cpu", weights_only=True)
    assert float(reloaded["_numpy_scalars"]["lr"]) == pytest.approx(1e-3)
    assert int(reloaded["_numpy_scalars"]["step"]) == 1


# --- FSDP portability -----------------------------------------------------------------------------


def test_to_full_tensors_passes_plain_tensors_through():
    """The DDP path: nothing is a DTensor, so the state dict must come back untouched."""
    state = {"a": torch.zeros(2, 3), "b": torch.ones(4)}
    out = selective_dcp._to_full_tensors(state)
    assert out.keys() == state.keys()
    assert all(out[k] is state[k] for k in state)


def _holds_dtensor(state: dict) -> bool:
    return any(isinstance(t, DTensor) for s in state.values() for t in s.values())


def _bare_checkpointer(save_dir, **attrs):
    """A ``SelectiveCheckpointer`` carrying only what ``save()`` and ``_load_filtered()`` read: the filter,
    no callbacks, a local checkpoint dir, no base network to warm-start, nothing left out of a resume."""
    ckpt = selective_dcp.SelectiveCheckpointer.__new__(selective_dcp.SelectiveCheckpointer)
    fields = dict(
        save_keys_filter=["lora_"],
        callbacks=None,
        save_to_object_store=False,
        load_from_object_store=False,
        _local_dirname=str(save_dir),
        save_s3_backend_key=None,
        config_checkpoint=SimpleNamespace(save_to_object_store=None),
        load_path=None,
        keys_not_to_resume=[],
        _suppress_latest_checkpoint=False,
    )
    fields.update(attrs)
    for name, value in fields.items():
        setattr(ckpt, name, value)
    return ckpt


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


def _trained_toy(seed: int = 0):
    """A frozen base layer plus a trainable ``lora_`` layer, one step in, with a scheduler attached the
    way the trainer attaches one (it adds ``initial_lr`` to the param groups)."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(OrderedDict(base=torch.nn.Linear(4, 4), lora_head=torch.nn.Linear(4, 2)))
    model.config = _geometry()
    model.base.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.lora_head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    with torch.enable_grad():
        model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    model.zero_grad(set_to_none=True)
    return model, optimizer, scheduler


@pytest.fixture
def sharded_moments():
    """A trained toy model whose Adam moments are DTensors, the shape FSDP leaves them in.

    A gloo group of one rank (an in-memory store, no network) and a CPU mesh over it are enough to
    build DTensors, so the gather's value contract is checked on a CPU runner; the two-rank GPU test
    covers the real sharding.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = torch.nn.Sequential(OrderedDict(base=torch.nn.Linear(4, 4), lora_head=torch.nn.Linear(4, 2)))
        model.config = _geometry()
        model.base.requires_grad_(False)  # the frozen network; only the lora_ subset trains and is saved
        optimizer = torch.optim.AdamW(model.lora_head.parameters(), lr=1e-3)
        with torch.enable_grad():
            model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = distribute_tensor(state[key], mesh, [Shard(0)])
        yield model, optimizer
    finally:
        dist.destroy_process_group()


def test_the_optimizer_gather_returns_plain_tensors_for_sharded_moments(sharded_moments):
    """Without ``full_state_dict`` torch returns the state name-keyed but with the moments still
    DTensors, which only an identically sharded model can load. The gather must hand back plain
    tensors, so the check is on the values, not the key type."""
    model, optimizer = sharded_moments
    assert _holds_dtensor(get_optimizer_state_dict(model, optimizer)["state"])  # the shape the bug produced
    gathered = selective_dcp._gather_optimizer_state(model, optimizer)
    assert gathered["state"] and not _holds_dtensor(gathered["state"])


def test_save_writes_the_optimizer_gathered(tmp_path, sharded_moments):
    """The gather has to be wired into ``save()``, not only exist: the optim file must hold plain tensors."""
    model, optimizer = sharded_moments
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    _bare_checkpointer(tmp_path).save(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False), 3)

    saved = torch.load(tmp_path / "optim" / "iter_000000003.pt", map_location="cpu", weights_only=True)
    assert saved["state"] and not _holds_dtensor(saved["state"])
    model_file = torch.load(tmp_path / "model" / "iter_000000003.pt", map_location="cpu", weights_only=True)
    assert set(model_file) == {"lora_head.weight", "lora_head.bias", GEOMETRY_STAMP_KEY}  # the subset plus its stamp
    assert (tmp_path / "latest_checkpoint.txt").read_text().strip() == "iter_000000003.pt"


def test_optimizer_state_round_trips():
    """Save gathers the optimizer state and load restores it; the moments must survive the pair."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    saved = selective_dcp._gather_optimizer_state(model, optimizer)

    # torch initialises the fresh optimizer's state by faking zero gradients and refuses to do so
    # over real ones, so clear them — as on a resume, where no backward has run yet.
    model.zero_grad(set_to_none=True)
    fresh_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert not fresh_opt.state_dict()["state"]
    selective_dcp._load_optimizer_state(model, fresh_opt, saved)
    loaded = fresh_opt.state_dict()["state"]
    assert all(
        torch.equal(loaded[i]["exp_avg"], optimizer.state[p]["exp_avg"]) for i, p in enumerate(model.parameters())
    )


def test_an_unstepped_optimizer_is_saved_empty_and_left_alone():
    # torch's gather steps an un-stepped optimizer once at lr=0 to create state, which would leave
    # Adam one step ahead of the weights on resume; there is nothing to gather, so nothing is.
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert selective_dcp._gather_optimizer_state(model, optimizer) == {}
    assert not optimizer.state


def test_an_empty_optimizer_state_loads_as_a_fresh_optimizer():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    selective_dcp._load_optimizer_state(model, optimizer, {})
    assert not optimizer.state


class _Container:
    """The shape of the framework's OptimizersContainer: iterable over its optimizers, no ``.state``."""

    def __init__(self, *optimizers):
        self.optimizers = list(optimizers)

    def __iter__(self):
        return iter(self.optimizers)


def test_the_trainer_optimizer_container_is_gathered_too():
    # The trainer hands save() an OptimizersContainer, not an Optimizer; the un-stepped check must
    # look through it rather than at an attribute it does not have.
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert selective_dcp._gather_optimizer_state(model, _Container(optimizer)) == {}
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    assert selective_dcp._gather_optimizer_state(model, _Container(optimizer))["state"]


# --- resume path ----------------------------------------------------------------------------------


def test_save_refuses_a_model_it_cannot_stamp(tmp_path):
    # An unstamped checkpoint could never be loaded again, so it is not written in the first place.
    model, optimizer, scheduler = _trained_toy()
    del model.config
    with pytest.raises(ValueError, match="cannot stamp"):
        _bare_checkpointer(tmp_path).save(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False), 1)
    assert not (tmp_path / "latest_checkpoint.txt").exists()


def test_a_filtered_resume_round_trips_on_cpu(tmp_path):
    """``save()`` then ``_load_filtered()``: the trained subset, the optimizer moments, the scheduler and
    the iteration come back; the frozen base keeps the fresh model's own values."""
    model, optimizer, scheduler = _trained_toy()
    _bare_checkpointer(tmp_path).save(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False), 7)
    assert (tmp_path / "latest_checkpoint.txt").read_text().strip() == "iter_000000007.pt"

    fresh, fresh_opt, fresh_sched = _trained_toy(seed=1)
    fresh_opt.state.clear()  # an un-stepped optimizer, as on a resume
    ckpt = _bare_checkpointer(tmp_path)
    assert ckpt.resume_iteration() == 7
    iteration = ckpt._load_filtered(
        "iter_000000007.pt", fresh, fresh_opt, fresh_sched, torch.amp.GradScaler("cuda", enabled=False)
    )

    assert iteration == 7 and fresh_sched.last_epoch == 7
    assert torch.equal(fresh.lora_head.weight, model.lora_head.weight)
    assert not torch.equal(fresh.base.weight, model.base.weight)  # outside the filter: strict=False, not a wipe
    for got, want in zip(fresh_opt.state.values(), optimizer.state.values()):
        assert torch.equal(got["exp_avg"], want["exp_avg"])


def test_keys_not_to_resume_leaves_a_component_fresh(tmp_path, loguru_lines):
    """The base checkpointer honours ``checkpoint.keys_not_to_resume``; the filtered resume must too, or a
    deliberate optimizer reset after a divergence is silently undone."""
    model, optimizer, scheduler = _trained_toy()
    _bare_checkpointer(tmp_path).save(model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False), 7)

    fresh, fresh_opt, fresh_sched = _trained_toy(seed=1)
    fresh_opt.state.clear()
    ckpt = _bare_checkpointer(tmp_path, keys_not_to_resume=["optim"])
    ckpt._load_filtered("iter_000000007.pt", fresh, fresh_opt, fresh_sched, torch.amp.GradScaler("cuda", enabled=False))
    assert not fresh_opt.state  # left fresh
    assert torch.equal(fresh.lora_head.weight, model.lora_head.weight)  # everything else still resumes
    assert any("not resuming ['optim']" in ln for ln in loguru_lines)

    with pytest.raises(ValueError, match="not the model"):
        _bare_checkpointer(tmp_path, keys_not_to_resume=["model"])._load_filtered(
            "iter_000000007.pt", fresh, fresh_opt, fresh_sched, torch.amp.GradScaler("cuda", enabled=False)
        )


def test_a_resume_without_a_saved_rng_state_warns(loguru_lines):
    # Every checkpoint written before the RNG state was recorded has none; the resume goes on from the
    # initial seed and must say so, since that replays the run's first iterations.
    selective_dcp._restore_rng_state({"iteration": 5}, "iter_000000005.pt")
    assert any("no RNG state" in ln for ln in loguru_lines)


def test_load_filtered_passes_a_load_source_to_the_existence_check(tmp_path, monkeypatch):
    """``_check_checkpoint_exists`` takes a ``CheckpointLoadSource`` and reads ``.path`` off it.

    Only ``_load_filtered`` reaches it, so a bare string raises ``AttributeError`` on resume, hours
    into a run — and no other test walks that path.
    """
    seen = []
    ckpt = object.__new__(selective_dcp.SelectiveCheckpointer)
    object.__setattr__(ckpt, "callbacks", None)
    object.__setattr__(ckpt, "save_to_object_store", False)
    object.__setattr__(ckpt, "save_s3_backend_key", None)
    object.__setattr__(ckpt, "config_checkpoint", SimpleNamespace(save_to_object_store=None))

    monkeypatch.setattr(type(ckpt), "load_dirname", property(lambda self: str(tmp_path)))
    monkeypatch.setattr(type(ckpt), "_warm_start_base_network", lambda self, model: None)
    monkeypatch.setattr(type(ckpt), "_check_checkpoint_exists", lambda self, source: seen.append(source))
    # Stop right after the existence checks; the load itself is covered elsewhere.
    monkeypatch.setattr(selective_dcp, "verify_digest", lambda path: (_ for _ in ()).throw(RuntimeError("stop")))

    with pytest.raises(RuntimeError, match="stop"):
        ckpt._load_filtered("iter_000001000.pt", model=None, optimizer=None, scheduler=None, grad_scaler=None)

    assert seen, "the existence check never ran"
    assert seen[0].path.endswith(os.path.join("model", "iter_000001000.pt"))
    assert seen[0].warm_start is False  # this job wrote it, so it is not a warm start
