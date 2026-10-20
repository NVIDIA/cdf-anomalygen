# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filtered ``torch.save`` checkpointer that saves only the filtered trainable subset.

Writes one ``.pt`` file per component (model/optim/scheduler/trainer). Because the saved
model holds only the trainable subset (LoRA + ``text_prompt_emb``), resume first
warm-starts the full base network from ``load_path`` (as a fresh run does), then overlays
that subset and restores the optimizer, scheduler, iteration and every rank's random generators.
A first run with no filtered checkpoint falls back to the framework's warm-start loader.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy
import torch
from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
from cosmos_framework.utils import distributed, log, misc
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor

from anomalygen.checkpoint.trained_keys import match_keys
from anomalygen.checkpoint.utils import (
    CLASS_LADDERS_KEY,
    GEOMETRY_STAMP_KEY,
    build_geometry_stamp,
    pop_and_verify_geometry_stamp,
    verify_digest,
    write_digest,
)

_COMPONENTS = ("model", "optim", "scheduler", "trainer")

# Checkpoints are loaded with ``weights_only=True`` so a ``.pt`` file cannot execute code while
# being unpickled. Optimizer and scheduler state are the components that do not load under that
# restriction unmodified: they store numpy scalars, and neither the scalar reconstructor nor the
# concrete dtype classes are on torch's default allowlist.
#
# Allowing these back is narrow and safe — they rebuild numpy scalars and dtype objects and nothing
# else, unlike the arbitrary ``__reduce__`` execution ``weights_only=False`` permits.
#
# Every concrete dtype class is allowed rather than the one or two a checkpoint happens to contain
# today: which dtypes appear depends on the optimizer, the schedule, and the training precision, so
# naming them individually turns a precision change into an unloadable checkpoint.
#
# Both numpy lookups are probed rather than assumed, because the package moved in NumPy 2:
# ``numpy._core`` does not exist on 1.x and ``numpy.dtypes`` was added in 2.0. On 2.x ``numpy.core``
# is a shim that forwards ``multiarray``, so either path reaches the same ``scalar`` function and
# allowlisting one covers both. The default is written as ``numpy.core`` alone — evaluating it is
# harmless, whereas reaching through it to ``numpy.core.multiarray`` emits a DeprecationWarning.
_SAFE_CHECKPOINT_GLOBALS = [getattr(numpy, "_core", numpy.core).multiarray.scalar, numpy.dtype]
_SAFE_CHECKPOINT_GLOBALS += [
    getattr(numpy.dtypes, name) for name in dir(getattr(numpy, "dtypes", object())) if name.endswith("DType")
]
torch.serialization.add_safe_globals(_SAFE_CHECKPOINT_GLOBALS)

# Segments that torch.compile / activation checkpointing / FSDP add to parameter paths but
# ``get_model_state_dict`` omits. Stripped so ``named_parameters()`` names match state-dict keys.
_WRAPPER_SEGMENTS = frozenset({"_orig_mod", "_checkpoint_wrapped_module", "_fsdp_wrapped_module"})


def _to_full_tensors(state: dict) -> dict:
    """Replace any DTensor with a plain, gathered tensor.

    Under FSDP the params are DTensors. Saving those gives a checkpoint that only loads back into
    an identically sharded model, so single-GPU generation cannot read it. Collective: every rank
    must call this, and every rank gets the full tensors. No-op under DDP.

    The trained-subset form of torch's ``get_model_state_dict(full_state_dict=True)``
    (``_gather_state_dict``), which would gather the whole network per rank and not fit. The load
    side needs no counterpart: ``set_model_state_dict(full_state_dict=True)`` re-shards.
    """
    return {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state.items()}


def _gather_optimizer_state(model, optimizer) -> dict:
    """Optimizer state gathered onto every rank; under FSDP the Adam moments are DTensors and only
    rank 0 writes, so saving them sharded would resume every rank from rank 0's shard. Collective.

    Empty for an un-stepped optimizer: it has nothing to gather, and torch's gather would step it
    once at lr=0 to create state, leaving Adam one step ahead of the weights.

    ``optimizer`` is one optimizer or an iterable of them (the trainer passes the framework's
    OptimizersContainer), the two shapes torch's gather accepts.
    """
    optimizers = [optimizer] if isinstance(optimizer, torch.optim.Optimizer) else list(optimizer)
    if not any(o.state for o in optimizers):
        return {}
    return get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=True))


def _rng_states() -> dict:
    """Every rank's random-generator states, keyed ``rng_state_<rank>``: rank 0 writes the file, so it
    needs all of them. Collective. Empty on a CUDA-less build, whose state the framework cannot take."""
    if not torch.cuda.is_available():
        return {}
    return {f"rng_state_{rank}": s for rank, s in enumerate(distributed.all_gather_object(get_rand_state_dict()))}


def _restore_rng_state(trainer_state: dict, source: str) -> None:
    """Put this rank's generators back where the checkpoint left them, so a resume continues the run's
    random stream (timesteps, noise) instead of replaying it from the initial seed."""
    rank = distributed.get_rank()
    state = trainer_state.get(f"rng_state_{rank}")
    if state is None:
        log.warning(
            f"{source} carries no RNG state for rank {rank} (written before it was recorded, or by fewer "
            "ranks): the random stream restarts from the run's initial seed."
        )
        return
    set_rand_state_dict(state)


def _load_optimizer_state(model, optimizer, state: dict) -> None:
    """Restore the gathered optimizer state, re-sharding it to this rank under FSDP. An empty state
    (saved before the first step) leaves the optimizer fresh."""
    if not state:
        return
    set_optimizer_state_dict(model, optimizer, optim_state_dict=state, options=StateDictOptions(full_state_dict=True))


class SelectiveCheckpointer(DistributedCheckpointer):
    def __init__(
        self,
        config_checkpoint,
        config_job,
        callbacks=None,
        disable_async: bool = False,
        save_keys_filter: Optional[List[str]] = None,
    ):
        super().__init__(config_checkpoint, config_job, callbacks=callbacks, disable_async=disable_async)
        self.save_keys_filter = save_keys_filter
        # When True, hide the filtered resume pointer so the base loader takes its warm-start branch.
        self._suppress_latest_checkpoint = False

    def _read_latest_checkpoint_file(self):
        if self._suppress_latest_checkpoint:
            return None
        return super()._read_latest_checkpoint_file()

    def resume_iteration(self) -> int:
        """The iteration a resume will continue from, read off the latest-checkpoint pointer; 0 for a fresh run."""
        latest = self._read_latest_checkpoint_file()
        return int(re.search(r"\d+", latest).group()) if latest else 0

    def _verify_save(self, model, full_state: dict, kept_keys: set) -> None:
        if not kept_keys:
            raise ValueError(
                f"save_keys_filter {self.save_keys_filter} matched none of {len(full_state)} params; "
                "refusing to checkpoint zero trained weights."
            )
        if not self.save_keys_filter:
            return
        # Catch a trained weight the filter forgot to list: every trainable param must be saved.
        # (Checking requires_grad, not the filter itself, is what makes a missing filter entry fail
        # here instead of silently dropping the weight from the checkpoint.)
        trainable = {
            ".".join(seg for seg in n.split(".") if seg not in _WRAPPER_SEGMENTS)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        missing = trainable - set(kept_keys)
        if missing:
            raise ValueError(
                f"SelectiveCheckpointer: {len(missing)} trainable param(s) not in the saved subset "
                f"(save_keys_filter {self.save_keys_filter}) — the filter is likely missing an entry. "
                f"Missing: {sorted(missing)[:10]}."
            )

    def _verify_load_filtered(self, ckpt_model: dict, full_state: dict) -> None:
        unexpected = set(ckpt_model) - set(full_state)
        if unexpected:
            raise ValueError(
                f"SelectiveCheckpointer: {len(unexpected)} checkpoint key(s) absent from the model would be "
                f"silently dropped under strict=False — refusing to load. Examples: {sorted(unexpected)[:10]}. "
                f"The checkpoint likely predates a model/key-layout change."
            )
        if self.save_keys_filter:
            missing = match_keys(full_state, self.save_keys_filter) - set(ckpt_model)
            if missing:
                raise ValueError(
                    f"SelectiveCheckpointer: checkpoint is missing {len(missing)} trained key(s) that "
                    f"save_keys_filter {self.save_keys_filter} captures in the live model — resuming would "
                    f"leave them at warm-started base values. Examples: {sorted(missing)[:10]}. The checkpoint "
                    f"was likely saved with a different filter or predates a key-layout change."
                )

    def _warm_start_base_network(self, model) -> None:
        """Load the full base network from ``load_path`` via the base loader's warm-start branch,
        so the filtered resume only has to overlay the trained subset on top of it."""
        if not self.load_path:
            log.warning("SelectiveCheckpointer: no load_path set; resuming without a base network.")
            return

        # Suppress the filtered resume pointer (forces the warm-start branch) and silence callbacks so
        # the on_load_* hooks fire only once, around the overlay below.
        saved_callbacks = self.callbacks
        self.callbacks = None
        self._suppress_latest_checkpoint = True
        try:
            super().load(model)
        finally:
            self.callbacks = saved_callbacks
            self._suppress_latest_checkpoint = False

    def _load_filtered(self, checkpoint_file, model, optimizer, scheduler, grad_scaler) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        # The filtered checkpoint holds only the trainable subset, so populate the full base network first.
        self._warm_start_base_network(model)

        paths = {k: os.path.join(self.load_dirname, k, checkpoint_file) for k in _COMPONENTS}
        loaded = {}
        for key, path in paths.items():
            # These are checkpoints this job wrote, so the same-job load source applies.
            self._check_checkpoint_exists(self._same_job_load_source(path))
            log.info(f"SelectiveCheckpointer: loading checkpoint from {path}")
            # Verify before loading: the point is to refuse bytes we did not write, not to notice
            # afterwards that we already loaded them.
            verify_digest(path)
            loaded[key] = torch.load(path, map_location="cpu", weights_only=True)

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint(model, state_dict=loaded)

        # Overlay the trained subset onto the warm-started full state dict, then load it.
        ckpt_model = dict(loaded["model"])
        # A resume keeps its dataset's per-class caps (train.py set them); the checkpoint's are the
        # previous run's reading of the same data, so the returned ones are not attached.
        pop_and_verify_geometry_stamp(ckpt_model, model, checkpoint_file)
        self._verify_load_filtered(ckpt_model, get_model_state_dict(model))
        # Full tensors for a subset of the keys: torch re-shards each onto the model's own DTensor
        # placements (the path the optimizer already takes) and keeps the model's values for the rest.
        set_model_state_dict(
            model, model_state_dict=ckpt_model, options=StateDictOptions(full_state_dict=True, strict=False)
        )

        iteration = loaded["trainer"]["iteration"]
        # checkpoint.keys_not_to_resume, as the base checkpointer honours it: a listed component starts
        # fresh. Not the model — a resume without its weights is a new run, not a resume.
        skip = set(self.keys_not_to_resume)
        if "model" in skip or not skip <= set(self.CHECKPOINT_KEYS):
            raise ValueError(
                f"checkpoint.keys_not_to_resume={sorted(skip)}: a filtered resume can leave out "
                f"{[k for k in self.CHECKPOINT_KEYS if k != 'model']}, not the model."
            )
        if skip:
            log.info(f"SelectiveCheckpointer: not resuming {sorted(skip)} (checkpoint.keys_not_to_resume).")
        if scheduler is not None and "scheduler" not in skip:
            scheduler.load_state_dict(loaded["scheduler"])
            scheduler.last_epoch = iteration
        if optimizer is not None and "optim" not in skip:
            _load_optimizer_state(model, optimizer, loaded["optim"])
        if "trainer" not in skip:
            if grad_scaler is not None:
                grad_scaler.load_state_dict(loaded["trainer"]["grad_scaler"])
            _restore_rng_state(loaded["trainer"], checkpoint_file)
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=paths["model"])
        log.success(f"SelectiveCheckpointer: loaded checkpoint, iteration {iteration}.")
        return iteration

    def save(self, model, optimizer, scheduler, grad_scaler, iteration: int) -> None:
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}.pt"
        full_state = get_model_state_dict(model)
        if self.save_keys_filter:
            kept_keys = match_keys(full_state, self.save_keys_filter)
            self._verify_save(model, full_state, kept_keys)
            log.info(f"SelectiveCheckpointer: filtered model state_dict {len(full_state)} -> {len(kept_keys)} keys.")
            # sorted(): _to_full_tensors runs a collective per key, so every rank must walk them
            # in the same order. match_keys returns a set, whose order differs between processes.
            ckpt_state = {k: full_state[k] for k in sorted(kept_keys)}
        else:
            # No filter means a whole-network save. Under FSDP that would gather every parameter
            # onto every rank, which does not fit — refuse instead of OOM-ing. The texture_ft path
            # always sets a filter (see register.py), so this only guards a bare construction.
            if any(isinstance(v, DTensor) for v in full_state.values()):
                raise ValueError(
                    "SelectiveCheckpointer: an unfiltered save is not supported under FSDP; the "
                    "whole network cannot be gathered per rank. Set save_keys_filter."
                )
            ckpt_state = full_state
        ckpt_state = _to_full_tensors(ckpt_state)
        ckpt_state[GEOMETRY_STAMP_KEY] = build_geometry_stamp(model)  # raises rather than write an unstamped .pt
        if getattr(model, "class_ladders", None):
            ckpt_state[CLASS_LADDERS_KEY] = model.class_ladders

        state_dicts_to_save = {
            "model": ckpt_state,
            "optim": _gather_optimizer_state(model, optimizer),
            "scheduler": scheduler.state_dict(),
            "trainer": {"grad_scaler": grad_scaler.state_dict(), "iteration": iteration, **_rng_states()},
        }

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=state_dicts_to_save)

        if distributed.get_rank() == 0:
            for folder in _COMPONENTS:
                checkpoint_path = os.path.join(self.save_dirname, folder, checkpoint_file)
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(misc.to(state_dicts_to_save[folder], device="cpu"), checkpoint_path)
                # Record the digest immediately after the write, while the bytes are still the ones
                # this process produced. Written before the latest-checkpoint pointer below, so a
                # checkpoint is never advertised as resumable before it is verifiable.
                write_digest(checkpoint_path)
            self._write_latest_checkpoint_file(checkpoint_file)
            log.success(
                f"SelectiveCheckpointer: saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}"
            )

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def load(self, model, optimizer=None, scheduler=None, grad_scaler=None) -> int:
        latest = self._read_latest_checkpoint_file()
        if latest is not None and str(latest).endswith(".pt"):
            log.info(f"SelectiveCheckpointer: found latest checkpoint '{latest}'.")
            return self._load_filtered(latest, model, optimizer, scheduler, grad_scaler)

        log.info(f"SelectiveCheckpointer: no checkpoint found — warm-start from base ({self.load_path}).")
        return super().load(model, optimizer, scheduler, grad_scaler)
