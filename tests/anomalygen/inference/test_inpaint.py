# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CPU-testable parts of the inference-loading public API.

Covers checkpoint resolution, the fine-tuned-overlay guard, the inpaint-closure's
unknown-anomaly guard, and the recipe-name error path — none of which build a model.
The full model-loading / generation path needs a fine-tuned checkpoint and a GPU and is
not exercised here.
"""

import glob
import hashlib
import os
import pathlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import anomalygen
import anomalygen.checkpoint.utils as ckpt_utils
import anomalygen.inference.inpaint as inpaint_module
from anomalygen.configs.texture.constants import (
    BASE_CHECKPOINT_PATHS,
    DEFAULT_MODEL_SIZE,
    DEFAULT_SHIFT,
    MODEL_SIZES,
)
from anomalygen.data.crop_geometry import plan_crop
from anomalygen.data.utils import UNCONDITIONAL_CAPTION
from anomalygen.inference.inpaint import (
    _base_checkpoint_for_experiment,
    _build_inpaint_batch_multi,
    _is_legacy,
    _resolve_ft_model_pt,
    _resolve_shift,
    _verify_overlay,
    build_inpaint_batch_fn,
    load_finetuned_model,
    load_for_inference,
)


def _plan_64():
    """Plan for a 64x64 crop on a single-size 64 ladder: no resample, no padding."""
    return plan_crop(32, 32, 64, 64, 2.0, 1.0, (64,), ratio_floor=1.5)


# --- _resolve_ft_model_pt ------------------------------------------------------------------------


def test_resolve_direct_pt_file(tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    assert _resolve_ft_model_pt(str(pt)) == str(pt)


def test_resolve_picks_highest_iter(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "iter_000001.pt").write_bytes(b"x")
    (model_dir / "iter_000002.pt").write_bytes(b"x")
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000002.pt")


def test_resolve_honours_latest_pointer(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "iter_000001.pt").write_bytes(b"x")
    (model_dir / "iter_000002.pt").write_bytes(b"x")
    (tmp_path / "latest_checkpoint.txt").write_text("iter_000001")  # no .pt suffix -> appended
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000001.pt")


def test_resolve_prefers_best_over_latest(tmp_path):
    """best_checkpoint.txt names the peak-scoring iteration; latest_checkpoint.txt names the last one,
    which is frequently worse. Generating from a run dir must pick the best without the caller
    having to read the pointer itself."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for it in ("iter_000001.pt", "iter_000002.pt", "iter_000003.pt"):
        (model_dir / it).write_bytes(b"x")
    (tmp_path / "latest_checkpoint.txt").write_text("iter_000003.pt")
    (tmp_path / "best_checkpoint.txt").write_text("iter_000002.pt\n")  # trailing newline is stripped
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000002.pt")


def test_resolve_falls_back_to_latest_when_best_is_absent_or_dangling(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "iter_000001.pt").write_bytes(b"x")
    (model_dir / "iter_000009.pt").write_bytes(b"x")
    (tmp_path / "latest_checkpoint.txt").write_text("iter_000001.pt")
    # A pointer naming a checkpoint that was rotated away must not win, and must not raise.
    (tmp_path / "best_checkpoint.txt").write_text("iter_000005.pt")
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000001.pt")


def test_resolve_checkpoints_subdir_layout(tmp_path):
    model_dir = tmp_path / "checkpoints" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "iter_000005.pt").write_bytes(b"x")
    assert _resolve_ft_model_pt(str(tmp_path)) == str(model_dir / "iter_000005.pt")


def test_resolve_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_ft_model_pt(str(tmp_path))


# --- _verify_overlay -----------------------------------------------------------------------------

# A model state with one base key plus the two trained keys the overlay must supply.
_FULL_STATE = {
    "net.blocks.0.weight": 0,
    "net.blocks.0.lora_A.weight": 0,
    "net.text_prompt_emb": 0,
}


def test_verify_overlay_accepts_valid_subset():
    ckpt = {"net.blocks.0.lora_A.weight": 1, "net.text_prompt_emb": 1}
    assert _verify_overlay(ckpt, _FULL_STATE) is None  # no raise


def test_verify_overlay_rejects_unexpected_key():
    ckpt = {"net.blocks.0.lora_A.weight": 1, "net.does_not_exist": 1}
    with pytest.raises(ValueError):
        _verify_overlay(ckpt, _FULL_STATE)


def test_verify_overlay_rejects_missing_trained_key():
    # Subset of the model (no unexpected keys) but missing a trained key -> would stay at base value.
    ckpt = {"net.text_prompt_emb": 1}  # lora_A missing
    with pytest.raises(ValueError):
        _verify_overlay(ckpt, _FULL_STATE)


# --- _base_checkpoint_for_experiment --------------------------------------------------------------


def test_base_checkpoint_follows_the_edge_size(register_experiment):
    # The sizes use different frozen backbones, so loading the other size's DCP would silently be
    # the wrong weights; the base checkpoint must track the experiment's model_size.
    name = register_experiment("unittest_size_edge", model_size="edge")
    assert _base_checkpoint_for_experiment(name) == "checkpoints/Cosmos3-Edge"


def test_base_checkpoint_follows_the_nano_size(register_experiment):
    name = register_experiment("unittest_size_nano", model_size="nano")
    assert _base_checkpoint_for_experiment(name) == "checkpoints/Cosmos3-Nano"


def test_base_checkpoint_rejects_unknown_size(register_experiment):
    name = register_experiment("unittest_size_bogus", model_size="giant")
    with pytest.raises(ValueError):
        _base_checkpoint_for_experiment(name)


@pytest.mark.parametrize(
    "label,expert_config", [("absent", {}), ("null", {"model_size": None}), ("empty", {"model_size": ""})]
)
def test_base_checkpoint_falls_back_to_the_default_size(register_experiment, label, expert_config):
    # DELIBERATE, not an accident: a recipe that never mentions model_size (every recipe predating
    # the Edge size) resolves to the default size instead of failing. Pinned so the silent fallback
    # stays a decision — an unset size must never become "whatever size ran last".
    name = register_experiment(f"unittest_size_unset_{label}", **expert_config)
    assert _base_checkpoint_for_experiment(name) == BASE_CHECKPOINT_PATHS[DEFAULT_MODEL_SIZE]
    assert BASE_CHECKPOINT_PATHS[DEFAULT_MODEL_SIZE] == "checkpoints/Cosmos3-Nano"


def test_size_table_and_default_agree():
    # Drift guard: a size added to one table and not the other resolves to a KeyError/ValueError at
    # load time, long after the recipe was written.
    assert set(BASE_CHECKPOINT_PATHS) == set(MODEL_SIZES)
    assert DEFAULT_MODEL_SIZE in MODEL_SIZES


# --- build_inpaint_batch_fn ---------------------------------------------------------------------


def test_build_inpaint_batch_fn_rejects_unknown_anomaly(stub_model):
    # Same guard, batched: unknown name fails before any model call.
    fn = build_inpaint_batch_fn(model=stub_model(shift=7), class_ids={"metal+scratch": 0}, num_steps=1, guidance=1.0)
    img = Image.new("RGB", (16, 16))
    mask = Image.new("L", (16, 16))
    with pytest.raises(KeyError):
        fn([img], [mask], ["unknown+type"], [1], [_plan_64()])


# --- _build_inpaint_batch_multi ------------------------------------------------------------------


class _StubNet:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))  # CPU param -> device = cpu


class _StubModel:
    """Minimal surface build_inpaint_batch(_multi) reads; no real forward.

    ``config`` carries the scalar shift table ``_resolve_shift`` reads when the caller pins no shift.
    """

    net = _StubNet()
    tensor_kwargs = {"dtype": torch.float32}
    input_caption_key = "ai_caption"
    config = SimpleNamespace(rectified_flow_training_config=SimpleNamespace(shift=DEFAULT_SHIFT))


def test_build_inpaint_batch_multi_merges_samples():
    model = _StubModel()
    imgs = [Image.fromarray(np.full((64, 64, 3), v, np.uint8), "RGB") for v in (100, 200)]
    m = np.zeros((64, 64), np.uint8)
    m[20:40, 20:40] = 255
    masks = [Image.fromarray(m, "L"), Image.fromarray(m, "L")]

    batch = _build_inpaint_batch_multi(model, imgs, masks, [0, 1], ["a b", "c d"])

    assert batch["dataset_name"] == "anomaly" and batch["is_preprocessed"] is True
    assert len(batch["images"]) == 4  # 2 vision items * 2 samples, flattened
    assert all(t.shape == (1, 3, 1, 64, 64) for t in batch["images"])
    assert batch["num_vision_items_per_sample"] == [2, 2]
    assert len(batch["image_size"]) == 4
    assert batch["edit_mask"].shape == (2, 1, 64, 64)
    assert batch["anomaly_class_id"].tolist() == [0, 1]
    assert batch[model.input_caption_key] == ["a b", "c d"]
    # One unconditional prompt per sample, merged alongside the conditional captions.
    assert batch["neg_" + model.input_caption_key] == [UNCONDITIONAL_CAPTION] * 2


# --- CFG unconditional branch --------------------------------------------------------------------


def test_the_batched_closure_feeds_the_unconditional_prompt():
    # The batch key and has_negative_prompt must move together, or the framework rejects the batch.
    model = _RecordingModel()
    img, mask = _crop_pair()
    build_inpaint_batch_fn(model, {"metal+scratch": 0}, num_steps=1, guidance=7.0)(
        [img], [mask], ["metal+scratch"], [1], [_plan_64()]
    )

    assert model.has_negative_prompt == [True]
    assert model.batches[0]["neg_" + model.input_caption_key] == [UNCONDITIONAL_CAPTION]


# --- load_for_inference error path ---------------------------------------------------------------


def test_load_for_inference_refuses_a_module_outside_the_allowed_namespaces(tmp_path):
    # Registration runs before any model load, so a recipe module the loader will not import fails
    # here, loudly. "json" is importable but not under an allowed namespace. The checkpoint is
    # stamped so the legacy-recipe refusal, which comes first, stays out of the way.
    with pytest.raises(ValueError, match="outside the allowed namespaces"):
        load_for_inference("json", _pt(tmp_path, stamped=True))


def _capture_parallelism_during_registration(monkeypatch, tmp_path):
    """Run load_for_inference far enough to register, returning what the builder would have read."""
    seen = {}

    def _register(recipe, experiment=None, **_kwargs):
        seen["value"] = os.environ.get("ANOMALYGEN_PARALLELISM")
        raise ValueError("stop after registering")  # enough to reach the finally, no model load

    monkeypatch.setattr(inpaint_module, "register_recipe", _register)
    with pytest.raises(ValueError, match="stop after registering"):
        load_for_inference("json", _pt(tmp_path, stamped=True))
    return seen["value"]


def test_load_for_inference_pins_unsharded_parallelism_while_registering(monkeypatch, tmp_path):
    # Inference must never shard: ranks run different testcases, so the per-rank all-gather counts
    # diverge and the collective hangs. The builder reads the variable during register_recipe, so
    # that call is the only window where the pin has to hold.
    monkeypatch.setenv("ANOMALYGEN_PARALLELISM", "fsdp")
    assert _capture_parallelism_during_registration(monkeypatch, tmp_path) == "ddp"


def test_load_for_inference_restores_parallelism_afterwards(monkeypatch, tmp_path):
    # The pin is scoped to registration. Leaving it set would silently force any experiment built
    # later in the same interpreter -- a training run in a notebook, say -- onto ddp.
    monkeypatch.setenv("ANOMALYGEN_PARALLELISM", "fsdp")
    _capture_parallelism_during_registration(monkeypatch, tmp_path)
    assert os.environ["ANOMALYGEN_PARALLELISM"] == "fsdp"


def test_load_for_inference_leaves_parallelism_unset_if_it_was(monkeypatch, tmp_path):
    # Nothing set it before, so nothing should be set after either.
    monkeypatch.delenv("ANOMALYGEN_PARALLELISM", raising=False)
    _capture_parallelism_during_registration(monkeypatch, tmp_path)
    assert "ANOMALYGEN_PARALLELISM" not in os.environ


# --- legacy checkpoints ----------------------------------------------------------------------------
# A fine-tuned .pt with no geometry stamp is a legacy checkpoint. load_for_inference reads its recipe
# as legacy and hands the recipe's class listing down; load_finetuned_model stamps and reorders the
# state in memory. Both halves are checked here without a real model.


def _pt(tmp_path, stamped):
    path = tmp_path / "iter_000000100.pt"
    state = {"net.blk.lora_A": torch.zeros(2, 1, 1)}
    if stamped:
        state[ckpt_utils.GEOMETRY_STAMP_KEY] = {"crop_mode": "ratio"}
    torch.save(state, path)
    return str(path)


def test_a_checkpoint_without_a_stamp_is_legacy(tmp_path):
    assert _is_legacy(_pt(tmp_path, stamped=False))
    assert not _is_legacy(_pt(tmp_path, stamped=True))


@pytest.mark.parametrize("stamped", [False, True], ids=["legacy", "stamped"])
def test_load_for_inference_reads_the_recipe_as_legacy_only_for_a_legacy_checkpoint(
    tmp_path, monkeypatch, register_experiment, stamped
):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["t", "small"], ["t", "big"]]}))
    name = register_experiment("unit_legacy", model_size="nano")
    seen = {}

    def _register(recipe, experiment=None, legacy=False):
        seen["legacy"] = legacy
        return name

    def _load(model_pt, base_checkpoint=None, experiment=None, legacy_class_order=None):
        seen["order"] = legacy_class_order
        return "model"

    monkeypatch.setattr(inpaint_module, "register_recipe", _register)
    monkeypatch.setattr(inpaint_module, "load_finetuned_model", _load)

    assert load_for_inference(str(recipe), _pt(tmp_path, stamped)) == "model"
    assert seen == {"legacy": not stamped, "order": None if stamped else [["t", "small"], ["t", "big"]]}


class _TinyModel(torch.nn.Module):
    """Stands in for the base network: one per-class LoRA tensor, a frozen layer, the config the stamp reads."""

    def __init__(self, config, num_classes=2):
        super().__init__()
        self.net = torch.nn.Module()
        self.net.blk = torch.nn.Module()
        self.net.blk.lora_A = torch.nn.Parameter(torch.zeros(num_classes, 2, 3))
        self.net.base = torch.nn.Linear(3, 3)
        self.config = config
        self.class_ladders = None


def test_load_finetuned_model_stamps_and_reorders_a_legacy_checkpoint_in_memory(tmp_path, monkeypatch, stub_model):
    # Listed class 0 = small (filled with 0), listed class 1 = big (filled with 1); the sorted order puts big first.
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"net.blk.lora_A": torch.stack([torch.zeros(2, 3), torch.ones(2, 3)])}, model_pt)
    model = _TinyModel(stub_model(ladder=(512,), crop_mode="ratio").config)
    monkeypatch.setattr(inpaint_module, "load_model", lambda *a, **k: model)

    loaded = load_finetuned_model(
        str(model_pt), base_checkpoint="unused", legacy_class_order=[["t", "small"], ["t", "big"]]
    )

    assert loaded is model
    assert model.net.blk.lora_A[0].eq(1.0).all() and model.net.blk.lora_A[1].eq(0.0).all()
    assert model.class_ladders is None, "a legacy checkpoint carries no per-class caps"


def test_load_finetuned_model_still_refuses_an_unstamped_checkpoint_without_a_class_order(
    tmp_path, monkeypatch, stub_model
):
    # Only load_for_inference opts a checkpoint into the upgrade; a direct call keeps the strict rule.
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"net.blk.lora_A": torch.zeros(2, 2, 3)}, model_pt)
    monkeypatch.setattr(inpaint_module, "load_model", lambda *a, **k: _TinyModel(stub_model().config))
    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_finetuned_model(str(model_pt), base_checkpoint="unused")


# --- per-instance seed wiring (serial path) ------------------------------------------------------
# The batched path is covered in test_iterative.py; this is the half generate.py actually runs, plus
# the build_inpaint_batch(seed=) -> build_source_item(seed=) hand-off that IS the seeding fix. A
# regression that dropped `seed=` on the way in would leave test_utils.py's build_source_item suite
# green while silently restoring the global-CUDA-RNG behaviour.


class _RecordingModel(_StubModel):
    """``_StubModel`` plus the two hooks the inpaint closures drive, recording what it saw."""

    def __init__(self):
        self.seeds = []
        self.sources = []
        self.batches = []
        self.has_negative_prompt = []

    def generate_samples_from_batch(self, batch, has_negative_prompt=False, **kwargs):
        self.seeds.append(kwargs.get("seed"))
        self.sources.append(batch["images"][0].clone())  # [source, cond] -> source item
        self.batches.append(batch)
        self.has_negative_prompt.append(has_negative_prompt)
        return {"vision": [torch.zeros(1, 3, 1, 8, 8)]}

    def decode(self, latent):
        return torch.zeros(1, 3, 1, 8, 8)


def _crop_pair(size=64, fill=120):
    img = Image.fromarray(np.full((size, size, 3), fill, np.uint8), "RGB")
    m = np.zeros((size, size), np.uint8)
    m[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return img, Image.fromarray(m, "L")


def test_batched_closures_are_reproducible_for_the_same_seed():
    """Two closures given the same seed produce identical source items — the --base_seed contract.
    The per-instance offset (depth j -> seed + j) belongs to the orchestrator and is tested there."""
    img, mask = _crop_pair()
    runs = []
    for _ in range(2):
        model = _RecordingModel()
        build_inpaint_batch_fn(model, {"t+d": 0}, num_steps=1, guidance=1.0)([img], [mask], ["t+d"], [77], [_plan_64()])
        runs.append(model.sources[0])
    assert torch.equal(runs[0], runs[1])


def test_batch_source_items_are_invariant_to_batch_position():
    """A sample's conditioning noise depends only on its own seed, not on where it sits in the batch."""
    model = _StubModel()
    img_a, mask_a = _crop_pair(fill=100)
    img_b, mask_b = _crop_pair(fill=200)

    both = _build_inpaint_batch_multi(model, [img_a, img_b], [mask_a, mask_b], [0, 1], ["a", "b"], seeds=[11, 22])
    swapped = _build_inpaint_batch_multi(model, [img_b, img_a], [mask_b, mask_a], [1, 0], ["b", "a"], seeds=[22, 11])

    # images is flattened [src0, cond0, src1, cond1]; b is at index 2 in one and index 0 in the other.
    assert torch.equal(both["images"][2], swapped["images"][0]), "sample b's source moved with its position"
    assert not torch.equal(both["images"][0], both["images"][2]), "different seeds must give different noise"


# --- safe checkpoint loading ----------------------------------------------------------------------
# ``load_for_inference`` reads a fine-tuned ``.pt`` with ``torch.load``. That file is the one most
# likely to arrive from elsewhere — copied between machines, shared, or read from a mounted directory
# — so the unpickler stays restricted, and the numpy state real checkpoints carry has to keep loading
# under that restriction.

_PACKAGE_ROOT = pathlib.Path(anomalygen.__file__).resolve().parent
_FT_CKPTS = sorted(glob.glob(str(_PACKAGE_ROOT.parent / "results" / "**" / "iter_*.pt"), recursive=True))


class _ExecOnUnpickle:
    """Unpickling this calls a global; the payload is inert on purpose (see test_cradio.py)."""

    def __reduce__(self):
        return (print, ("unpickling executed a global",))


def test_torch_load_rejects_a_crafted_checkpoint(tmp_path):
    """Without the restriction this pickle would run a shell command instead of returning weights."""
    path = tmp_path / "evil.pt"
    with open(path, "wb") as handle:
        pickle.dump({"state_dict": _ExecOnUnpickle()}, handle)
    with pytest.raises(Exception):
        torch.load(path, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("scalar", [np.float64(0.5), np.float32(0.9), np.int64(7)])
def test_numpy_scalar_state_loads_under_the_restriction(tmp_path, scalar):
    """Optimizer and scheduler state carry numpy scalars at whatever precision training used.

    Importing ``anomalygen`` allowlists the numpy scalar and dtype constructors; without it this
    raises ``UnpicklingError``, which is what made real optim and scheduler checkpoints unloadable
    when the restriction was first switched on.
    """
    path = tmp_path / "optim.pt"
    torch.save({"step": scalar, "dtype": np.dtype("float64"), "tensor": torch.zeros(2)}, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert float(loaded["step"]) == pytest.approx(float(scalar))
    assert loaded["dtype"] == np.dtype("float64")


@pytest.mark.skipif(not _FT_CKPTS, reason="no fine-tuned checkpoint under results/")
def test_real_finetuned_checkpoint_loads_under_the_restriction():
    """A checkpoint this pipeline actually produced, not just a synthetic state dict."""
    state = torch.load(_FT_CKPTS[0], map_location="cpu", weights_only=True)
    assert state, "expected a non-empty state dict"
    assert all(isinstance(v, torch.Tensor) for v in state.values())


def test_no_call_site_disables_the_unpickler_restriction():
    """``weights_only=False`` must not reappear — including in a module no test imports."""
    offenders = [
        f"{source.relative_to(_PACKAGE_ROOT.parent)}:{lineno}"
        for source in _PACKAGE_ROOT.rglob("*.py")
        for lineno, line in enumerate(source.read_text().splitlines(), 1)
        if "weights_only=False" in line.split("#", 1)[0].replace(" ", "")
    ]
    assert offenders == [], "checkpoint loads must keep weights_only=True; found: " + ", ".join(offenders)


# --- checkpoint integrity at the inference load site ----------------------------------------------


def test_load_finetuned_model_refuses_a_tampered_checkpoint(tmp_path):
    """Inference reads the .pt that travelled, so the digest is checked here and not only on resume.

    The refusal lands before the base network loads, so no model fixture is needed — reaching
    load_model at all would mean the check moved back after it.
    """
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"w": torch.zeros(2)}, model_pt)
    ckpt_utils.write_digest(str(model_pt))
    torch.save({"w": torch.ones(2)}, model_pt)  # same path, different bytes

    with pytest.raises(ValueError, match="does not match its recorded digest"):
        load_finetuned_model(str(model_pt), base_checkpoint="unused")


def test_load_finetuned_model_accepts_an_untampered_checkpoint(tmp_path, monkeypatch):
    """The digest must not be what stops a legitimate checkpoint: it gets past the check."""
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"w": torch.zeros(2)}, model_pt)
    ckpt_utils.write_digest(str(model_pt))

    def _stop_after_the_check(*_args, **_kwargs):
        raise RuntimeError("reached load_model")

    monkeypatch.setattr(inpaint_module, "load_model", _stop_after_the_check)
    with pytest.raises(RuntimeError, match="reached load_model"):
        load_finetuned_model(str(model_pt), base_checkpoint="unused")


# --- opt-in base-weight verification --------------------------------------------------------------
# verify_digest covers the fine-tuned .pt. The frozen base DCP had no load-time check at all: its
# digests are recorded by download_checkpoints.sh and were only compared on a later download run,
# while checkpoints/ is a bind mount anything on the host can rewrite in between.


def _base_tree(tmp_path, payload=b"shard"):
    """A checkpoints/ root holding one base DCP shard, plus a manifest recording it."""
    ckpt_dir = tmp_path / "checkpoints"
    shard = ckpt_dir / "Cosmos3-Nano" / "model" / "__0_0.distcp"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(payload)
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(f"{hashlib.sha256(payload).hexdigest()}  Cosmos3-Nano/model/__0_0.distcp\n")
    return ckpt_dir, shard, manifest


def _ft_checkpoint(tmp_path):
    model_pt = tmp_path / "iter_000000100.pt"
    torch.save({"w": torch.zeros(2)}, model_pt)
    ckpt_utils.write_digest(str(model_pt))
    return str(model_pt)


def test_base_weights_are_not_verified_unless_opted_in(tmp_path, monkeypatch):
    """Hashing multi-GB shards on every load is not the default; the check is opt-in."""
    ckpt_dir, shard, manifest = _base_tree(tmp_path)
    shard.write_bytes(b"tampered")
    monkeypatch.setattr(inpaint_module, "CONVERTED_MANIFEST", manifest)
    monkeypatch.delenv("ANOMALYGEN_VERIFY_BASE_WEIGHTS", raising=False)
    monkeypatch.setattr(
        inpaint_module, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached load_model"))
    )

    with pytest.raises(RuntimeError, match="reached load_model"):
        load_finetuned_model(_ft_checkpoint(tmp_path), base_checkpoint=str(ckpt_dir / "Cosmos3-Nano"))


def test_opted_in_base_weight_verification_refuses_a_tampered_shard(tmp_path, monkeypatch):
    ckpt_dir, shard, manifest = _base_tree(tmp_path)
    shard.write_bytes(b"tampered")
    monkeypatch.setattr(inpaint_module, "CONVERTED_MANIFEST", manifest)
    monkeypatch.setenv("ANOMALYGEN_VERIFY_BASE_WEIGHTS", "1")

    with pytest.raises(ValueError, match="does not match the recorded digest"):
        load_finetuned_model(_ft_checkpoint(tmp_path), base_checkpoint=str(ckpt_dir / "Cosmos3-Nano"))


def test_opted_in_base_weight_verification_passes_an_untouched_shard(tmp_path, monkeypatch):
    """The check must not be what stops a legitimate load: it gets past to load_model."""
    ckpt_dir, _, manifest = _base_tree(tmp_path)
    monkeypatch.setattr(inpaint_module, "CONVERTED_MANIFEST", manifest)
    monkeypatch.setenv("ANOMALYGEN_VERIFY_BASE_WEIGHTS", "1")
    monkeypatch.setattr(
        inpaint_module, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached load_model"))
    )

    with pytest.raises(RuntimeError, match="reached load_model"):
        load_finetuned_model(_ft_checkpoint(tmp_path), base_checkpoint=str(ckpt_dir / "Cosmos3-Nano"))


# --- batched closure: mixed input sizes ----------------------------------------------------------


class _ScatterModel(_StubModel):
    """Returns a crop tagged with its own sample's seed, so a mis-scatter is visible in the output."""

    def generate_samples_from_batch(self, batch, seed=None, **kwargs):
        return {"vision": [torch.full((1, 3, 1, 8, 8), s / 100.0) for s in seed]}

    def decode(self, latent):
        return latent  # _decode_to_pil maps [-1,1] -> [0,255]


def _solid(img):
    return int(np.asarray(img)[0, 0, 0])


def test_batched_closure_returns_crops_in_input_order():
    """One sampler call for the whole batch; results must stay in the caller's order. Getting that
    wrong pastes one sample's crop into another's image."""
    sizes = [64, 512, 64]
    crops = [Image.fromarray(np.full((n, n, 3), 120, np.uint8), "RGB") for n in sizes]
    masks = []
    for n in sizes:
        m = np.zeros((n, n), np.uint8)
        m[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 255
        masks.append(Image.fromarray(m, "L"))
    plans = [plan_crop(n // 2, n // 2, n, n, 2.0, 1.0, (n,), ratio_floor=1.5) for n in sizes]
    seeds = [10, 20, 30]

    fn = build_inpaint_batch_fn(
        _ScatterModel(),
        {"t+d": 0},
        num_steps=1,
        guidance=1.0,
    )
    got = fn(crops, masks, ["t+d"] * 3, seeds, plans)

    assert [_solid(o) for o in got] == [int(round((s / 100.0 + 1) * 127.5)) for s in seeds]
    assert len({_solid(o) for o in got}) == 3  # the three seeds really are distinguishable
    assert [o.size for o in got] == [(n, n) for n in sizes]


# --- sampler shift ------------------------------------------------------------------------------


def test_shift_comes_from_the_recipe_not_a_default(stub_model):
    # The shift is read off the recipe the model was built from. A value distinct from DEFAULT_SHIFT,
    # so a body that returned the default would fail here.
    assert _resolve_shift(stub_model(shift=7)) == 7.0


def test_shift_raises_when_the_recipe_does_not_carry_one():
    # Silently substituting a default would desync the sampler from the schedule training used.
    with pytest.raises(AttributeError, match="rectified_flow_training_config.shift"):
        _resolve_shift(SimpleNamespace(config=SimpleNamespace()))
