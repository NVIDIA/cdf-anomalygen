# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable config builders for the texture I2I anomaly-inpainting experiment.

A recipe (e.g. ``ag_config/exp_texture_ft_phone_screen.yaml``) supplies the dataset-specific values
(paths, ``anomaly_types``, ``dataset_name``) and calls ``build_anomalygen_texture_ft_experiment`` to
get the full experiment ``LazyDict`` to ``cs.store``. Two layers:

* ``build_anomalygen_texture_ft_model_config`` — what the model *is*: architecture, adapters, loss,
  and the crop geometry the trained weights are only valid under. Also used at inference.
* ``build_anomalygen_texture_ft_experiment`` — that config plus everything needed to *train* it:
  data paths, dataloaders, optimizer, scheduler, trainer schedule, callbacks, checkpointing.

Fixed architecture choices (any-touch latent mask, replacement trick, LoRA target modules) are
hard-coded here rather than exposed; everything commonly tuned is a keyword argument with a default.

``crop_mode`` sets how each window is framed; every sample is then cut onto an input size drawn from
``input_size_ladder`` (a single-size ladder pins one fixed input size). A step is bounded by
``max_sequence_length`` (a token budget), so samples per step vary with the sizes drawn; set it to
None to batch by ``batch_size`` instead.

Env overrides, so the documented launch command is unchanged: ``ANOMALYGEN_PARALLELISM`` (ddp/fsdp),
``ANOMALYGEN_COMPILE`` (0/1).
"""

from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from anomalygen.checkpoint.trained_keys import TRAINED_KEY_PREFIXES, WARM_START_SKIP_PREFIXES
from anomalygen.configs.texture.constants import (
    AC_MODES,
    BASE_CHECKPOINT_PATHS,
    COMPILE_RECOMPILE_LIMIT,
    DEFAULT_AC_MODE,
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_RATIO_RANGE,
    DEFAULT_INPUT_SIZE_LADDER,
    DEFAULT_MODEL_SIZE,
    DEFAULT_SCALE_RANGE,
    DEFAULT_SHIFT,
    MODEL_SIZES,
    PATCH_SPATIAL,
    VAE_SPATIAL_COMPRESSION,
)
from anomalygen.data.crop_geometry import validate_geometry
from anomalygen.data.inpaint_inference_dataset import get_inpaint_val_dataloader
from anomalygen.data.inpainting_dataset import get_inpainting_dataset
from anomalygen.data.utils import CAPTION_TOKEN_BUDGET, anomalygen_collate_fn, sample_token_count

# Frozen base ``model.config`` per size. Framework constants — deep-copy before mutating, which
# ``_base_model_config`` is the only place that does.
_SIZE_MODEL_CONFIGS = {
    "nano": NANO_MODEL_CONFIG,
    "edge": EDGE_MODEL_CONFIG,
}


def _base_model_config(model_size: str) -> dict:
    """Deep-copy the frozen base ``model.config`` for ``model_size``."""
    if model_size not in _SIZE_MODEL_CONFIGS:
        raise ValueError(f"model_size={model_size!r} is not one of {MODEL_SIZES}.")
    return copy.deepcopy(_SIZE_MODEL_CONFIGS[model_size])


# Repo root, derived from this module's location. Recipe-supplied relative paths are absolutized
# against it because training chdir's to the framework checkout for the model build.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _abs(path: str) -> str:
    """Absolutize ``path`` against the repo root (no-op if already absolute)."""
    return path if os.path.isabs(path) else str(_REPO_ROOT / path)


def build_anomalygen_texture_ft_model_config(
    anomaly_num_classes: int,
    vae_path: str,
    *,
    # --- backbone ---
    model_size: str = DEFAULT_MODEL_SIZE,
    anomaly_types: Sequence[Sequence[str]] = (),
    # --- global LoRA (all classes share one adapter) ---
    lora_enabled: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    # --- per-defect-type LoRA (routed per gen token) ---
    per_class_lora_enabled: bool = True,
    per_class_lora_rank: int = 8,
    per_class_lora_alpha: int = 8,
    # --- per-defect-type learnable text prompt ---
    learnable_text_prompt_enabled: bool = False,
    text_prompt_num_tokens: int = CAPTION_TOKEN_BUDGET,
    text_prompt_init_word: str = "anomaly",
    # --- loss ---
    inpaint_loss_background_weight: float = 0.0,
    # --- sampler + crop geometry (published so generation mirrors training) ---
    shift: float = DEFAULT_SHIFT,
    crop_mode: str = DEFAULT_CROP_MODE,
    input_size_ladder: Sequence[int] = DEFAULT_INPUT_SIZE_LADDER,
    crop_ratio_range: Sequence[float] = DEFAULT_CROP_RATIO_RANGE,
    inference_scale: float = 1.0,
    # --- runtime ---
    activation_checkpointing_mode: str = DEFAULT_AC_MODE,
    parallelism: str = "ddp",
) -> dict:
    """Base model config for ``model_size`` plus the texture anomaly-inpainting overrides.

    Deep-copies the size baseline (``nano`` -> Qwen3-VL-8B, ``edge`` -> Nemotron-3 Dense VL 2B) so
    the vlm_config / tokenizer / net wiring is inherited; every override below is size-independent.

    The two per-defect-type conditioning mechanisms — per-class LoRA and the learnable text prompt —
    are independent and stack; either or both may be on.
    """
    cfg = _base_model_config(model_size)
    cfg["lora_enabled"] = lora_enabled
    cfg["lora_rank"] = lora_rank
    cfg["lora_alpha"] = lora_alpha
    cfg["lora_target_modules"] = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"

    _dec = cfg.setdefault("diffusion_expert_config", {})
    # Recorded so inference can read the size off a registered node (see inference/inpaint.py).
    _dec["model_size"] = model_size
    _dec["anomaly_num_classes"] = anomaly_num_classes
    _dec["anomaly_types"] = [list(p) for p in anomaly_types]
    _dec["inpaint_latent_mask_mode"] = "any"
    _dec["inpaint_loss_background_weight"] = inpaint_loss_background_weight
    _dec["inpaint_replacement_trick"] = True
    # Per-class LoRA on the gen-expert attention projections.
    _dec["per_class_lora_enabled"] = per_class_lora_enabled
    _dec["per_class_lora_rank"] = per_class_lora_rank
    _dec["per_class_lora_alpha"] = per_class_lora_alpha
    # Learnable text prompt: whole text sequence tunable, template-initialized.
    _dec["learnable_text_prompt_enabled"] = learnable_text_prompt_enabled
    _dec["text_prompt_num_tokens"] = text_prompt_num_tokens
    _dec["text_prompt_init_word"] = text_prompt_init_word
    # Read by set_up_parallelism: "ddp" skips fully_shard so each rank keeps the full model for the
    # DDP wrapper to broadcast.
    _dec["parallelism"] = parallelism
    # Trained crop mode / input-size / framing bands, published so generation mirrors training.
    _dec["crop_mode"] = crop_mode
    _dec["input_size_ladder"] = [int(v) for v in input_size_ladder]
    _dec["crop_ratio_range"] = (float(crop_ratio_range[0]), float(crop_ratio_range[1]))
    _dec["inference_scale"] = inference_scale

    # One scalar shift for every input size, replacing the framework's per-resolution dict. Must be an
    # int: the framework takes the scalar path on ``isinstance(shift, int)`` and reads a float as a dict.
    if int(shift) != shift:
        raise ValueError(f"shift must be a whole number, got {shift}.")
    cfg["rectified_flow_training_config"]["shift"] = int(shift)

    # Local VAE checkpoint, overriding the framework's relative default.
    cfg["tokenizer"]["vae_path"] = vae_path
    # Disable the model-level EMA (on in the nano base). It builds a second, float32 copy of the net
    # — ~2x the bf16 net and the bulk of single-GPU memory — that we never use (trainer `/ema` is
    # None and net_ema. is skipped at load). Off, the model fits one GPU under DDP.
    cfg["ema"]["enabled"] = False
    # torch.compile: ON by default, ANOMALYGEN_COMPILE=0 disables. (USE_TORCH_COMPILE is not read.)
    cfg["compile"]["enabled"] = os.environ.get("ANOMALYGEN_COMPILE", "1").strip() == "1"
    # "language" compiles only the MoT transformer blocks — the heavy compute. They take the packed
    # sequence as input and never read the live inpaint context, so network.set_anomaly_context
    # forces no recompiles. "all" would also compile the encode/decode heads, where our
    # _encode_text / _encode_vision injections live: stale context + recompiles. Pinned, not inherited.
    cfg["compile"]["compiled_region"] = "language"
    if cfg["compile"].get("compiled_region") not in (None, "language"):
        raise ValueError("AnomalyGen Texture FT requires compiled_region in {None, 'language'}")
    # Activation checkpointing for the MoT blocks. The size baselines ship "full" for long video
    # sequences; a step here packs at most ``max_sequence_length`` tokens and does not need it.
    if activation_checkpointing_mode not in AC_MODES:
        raise ValueError(
            f"activation_checkpointing_mode must be one of {AC_MODES}, got {activation_checkpointing_mode!r}."
        )
    cfg["activation_checkpointing"]["mode"] = activation_checkpointing_mode
    # Derive the FSDP shard degree from the torchrun world size (nano pins 8) so one recipe scales to
    # any GPU count. Ignored under DDP.
    cfg["parallelism"]["data_parallel_shard_degree"] = -1
    return cfg


def build_anomalygen_texture_ft_experiment(
    *,
    # --- identity ---
    dataset_name: str,
    anomaly_types: Sequence[Sequence[str]],
    job_name: str = "anomalygen_texture_ft",
    # Caller's own file, snapshotted into the run dir by TrainingReport. Defaults to the caller.
    recipe_path: Optional[str] = None,
    # --- data + checkpoint paths (repo-root-relative or absolute) ---
    dataset_path: str,
    testcase_jsonl: str,
    # Frozen backbone size ("nano" | "edge"); also picks base_checkpoint_path when that is omitted.
    model_size: str = DEFAULT_MODEL_SIZE,
    base_checkpoint_path: Optional[str] = None,
    vae_path: Optional[str] = None,
    # --- global LoRA (all classes share one adapter) ---
    lora_enabled: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    # --- per-defect-type LoRA (routed per gen token; all classes in one mixed-batch run) ---
    per_class_lora_enabled: bool = True,
    per_class_lora_rank: int = 8,
    per_class_lora_alpha: int = 8,
    # --- per-defect-type learnable text prompt (stacks with the per-class LoRA) ---
    learnable_text_prompt_enabled: bool = False,
    text_prompt_num_tokens: int = CAPTION_TOKEN_BUDGET,
    text_prompt_init_word: str = "anomaly",
    # --- loss ---
    # Weight on the background term; the defect term is always unweighted. 0.0 = defect-only.
    inpaint_loss_background_weight: float = 0.0,
    # --- sampler + crop geometry ---
    shift: float = DEFAULT_SHIFT,
    # Where the framing comes from: "scale" (crop_ratio_range + scale_range) or "ratio"
    # (crop_ratio_range alone, so scale_range and inference_scale are unused).
    crop_mode: str = DEFAULT_CROP_MODE,
    # FRAMING axis: context multiplier — the window is bbox * ratio, so the defect fills 1/ratio of it.
    # Under crop_mode="ratio" this is the only framing axis.
    crop_ratio_range: Sequence[float] = DEFAULT_CROP_RATIO_RANGE,
    # SCALE axis ("scale" crop mode only): the defect's apparent pixel size (bbox * s), drawn LOG-uniformly
    # as an augmentation (a linear draw would over-weight upscaling and sit training above
    # inference_scale). Orthogonal to crop_ratio_range. See RandomScaleCrop.
    scale_range: Sequence[float] = DEFAULT_SCALE_RANGE,
    # Legal per-axis input sizes in pixels; each axis snaps to one. Multiples of 32 (one vision
    # token) — validated and sorted below. A single entry pins one fixed input size.
    input_size_ladder: Sequence[int] = DEFAULT_INPUT_SIZE_LADDER,
    # Scale used at GENERATION time ("scale" crop mode only); 1.0 = native pixels. Must sit inside scale_range.
    inference_scale: float = 1.0,
    # --- data loading ---
    # Token budget per packed step; this, not batch_size, is the training batch (the framework
    # accepts exactly one of the two). Bounds activation memory, which under variable input sizes
    # tracks the sizes drawn rather than a sample count. None => batch by sample count.
    max_sequence_length: Optional[int] = 5120,
    # Inner DataLoader per-pull size, and the training batch only when max_sequence_length is None.
    batch_size: int = 4,
    validation_batch_size: int = 16,
    num_workers: int = 4,
    # --- runtime ---
    # Activation checkpointing for the MoT blocks: "none" | "selective" | "full". Trades recompute
    # for memory and changes nothing the model trains on, so it is the first knob to raise on an OOM.
    activation_checkpointing_mode: str = DEFAULT_AC_MODE,
    # --- augmentation (per-sample probability; 0 = off) ---
    # Drop the training source item's background to -1 (black); the defect region is noise either
    # way. Training-only — inference always keeps the background.
    background_dropout_prob: float = 0.0,
    # Keep one defect instance, blacking the dropped ones out in the image.
    inst_aug_prob: float = 0.2,
    # Recolour a band around the defect (band width stays on $ANOMALYGEN_RING_JITTER_PX).
    ring_jitter_prob: float = 0.0,
    # --- optimizer (AdamW) ---
    lr: float = 1.0e-03,
    betas: Sequence[float] = (0.9, 0.95),
    eps: float = 1.0e-06,
    weight_decay: float = 0.0,
    # Name-substring -> factor on the base ``lr``, applied by the framework optimizer to any param
    # whose name contains it; e.g. give the learnable text prompt a larger step. Empty => uniform LR.
    lr_multipliers: Optional[Dict[str, float]] = None,
    # --- LambdaCosine scheduler (single warmup -> cosine-decay cycle) ---
    cycle_lengths: int = 15000,
    warm_up_steps: int = 500,
    # --- trainer schedule ---
    max_iter: int = 15000,
    validation_iter: int = 1000,
    run_validation_on_start: bool = True,
    save_iter: int = 1000,
    logging_iter: int = 10,
    seed: int = 42,
    # --- validation metrics ---
    # Anomaly-quality axes (completeness/precision/boundary_iou) + the aq_nn composite
    # (= completeness + nn_score). False skips their per-validation SAM2 load.
    compute_anomaly_quality: bool = True,
    # --- early stopping (off by default) ---
    early_stop_enabled: bool = False,
    early_stop_metric: str = "nn",
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.0,
    early_stop_min_delta_mode: str = "rel",
) -> LazyDict:
    """Assemble the full ``anomalygen_texture_ft`` experiment ``LazyDict`` for a texture dataset.

    The caller (a recipe) supplies the dataset-specific values and then ``cs.store``s the result.
    ``anomaly_types`` is a list of ``[texture, defect]`` pairs. It is sorted and deduplicated here, so
    the class ids (its positions) do not depend on the order a recipe lists them; the checkpoint
    stamps the sorted list and generation reads the ids back off it.
    """
    # Absolutize every path: recipes may pass repo-root-relative ones, but the model build runs under
    # a chdir to the framework root. The YAML/JSON loader passes recipe_path explicitly; a .py recipe
    # relies on the caller-frame default.
    dataset_path = _abs(dataset_path)
    testcase_jsonl = _abs(testcase_jsonl)
    recipe_path = _abs(recipe_path or inspect.stack()[1].filename)
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size={model_size!r} is not one of {MODEL_SIZES}.")
    base_checkpoint_path = _abs(base_checkpoint_path or BASE_CHECKPOINT_PATHS[model_size])
    vae_path = _abs(vae_path or "checkpoints/wan2pt2/Wan2.2_VAE.pth")

    anomaly_types = [list(p) for p in sorted({(str(t), str(d)) for t, d in anomaly_types})]
    anomaly_num_classes = len(anomaly_types)

    # The crop geometry is checked once, here, so a bad recipe fails at parse rather than inside a
    # DataLoader worker; the dataset and the crop stages trust the normalised values.
    geometry = validate_geometry(crop_mode, input_size_ladder, crop_ratio_range, scale_range, inference_scale)
    input_size_ladder = list(geometry.input_size_ladder)
    crop_ratio_range, scale_range = geometry.crop_ratio_range, geometry.scale_range
    if int(text_prompt_num_tokens) > CAPTION_TOKEN_BUDGET:
        raise ValueError(f"text_prompt_num_tokens must be <= {CAPTION_TOKEN_BUDGET}, got {text_prompt_num_tokens}.")
    if crop_mode == "ratio":
        log.info(
            "crop_mode='ratio': crop_ratio_range is the only framing axis; scale_range and inference_scale are unused."
        )
    # Headroom check: PackingDataLoader DISCARDS a sample at ``>=`` the budget, so the largest size
    # must fit strictly. Token count mirrors _compute_num_tokens_per_sample: per vision item
    # (h/32)*(w/32) + 2 delimiters, x2 items, plus the caption + 1 EOS.
    if max_sequence_length is not None:
        # Mirrors the dataset: padded to an exact length with the learnable prompt on, else
        # free-length capped at CAPTION_TOKEN_BUDGET.
        caption_tokens = text_prompt_num_tokens if learnable_text_prompt_enabled else CAPTION_TOKEN_BUDGET
        top_size_tokens = sample_token_count(input_size_ladder[-1], input_size_ladder[-1], caption_tokens)
        if top_size_tokens >= max_sequence_length:
            raise ValueError(
                f"input_size_ladder largest size {input_size_ladder[-1]} needs {top_size_tokens} tokens/sample "
                f"(2 vision items + {caption_tokens} caption) but max_sequence_length is "
                f"{max_sequence_length}; the packer discards samples at >= the budget."
            )

    # ANOMALYGEN_PARALLELISM=ddp (default; each rank holds the full, DDP-replicated model) or =fsdp
    # (shard params/optimizer across all launched GPUs). One value drives three things that must
    # agree: the model group node, trainer.distributed_parallelism, and ``_dec.parallelism``.
    parallelism = os.environ.get("ANOMALYGEN_PARALLELISM", "ddp").strip().lower()
    if parallelism not in ("ddp", "fsdp"):
        raise ValueError(f"ANOMALYGEN_PARALLELISM must be 'ddp' or 'fsdp', got {parallelism!r}")
    model_group = "anomalygen_texture_ft_mot_fsdp" if parallelism == "fsdp" else "anomalygen_texture_ft_mot_ddp"

    model_config = build_anomalygen_texture_ft_model_config(
        anomaly_num_classes,
        vae_path,
        model_size=model_size,
        anomaly_types=anomaly_types,
        lora_enabled=lora_enabled,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        per_class_lora_enabled=per_class_lora_enabled,
        per_class_lora_rank=per_class_lora_rank,
        per_class_lora_alpha=per_class_lora_alpha,
        learnable_text_prompt_enabled=learnable_text_prompt_enabled,
        text_prompt_num_tokens=text_prompt_num_tokens,
        text_prompt_init_word=text_prompt_init_word,
        inpaint_loss_background_weight=inpaint_loss_background_weight,
        shift=shift,
        crop_mode=crop_mode,
        input_size_ladder=input_size_ladder,
        crop_ratio_range=crop_ratio_range,
        inference_scale=inference_scale,
        activation_checkpointing_mode=activation_checkpointing_mode,
        parallelism=parallelism,
    )

    # Token-budget packing gives each rank a data-dependent sample count per step, so the framework's
    # default rank-level gradient average would weight a lone large-input sample as heavily as a peer
    # rank's whole packed batch. Average per sample instead.
    if max_sequence_length is not None:
        model_config["rectified_flow_training_config"]["sample_level_loss_averaging"] = True

    return LazyDict(
        dict(
            defaults=[
                {"override /model": model_group},
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /optimizer": "adamw"},
                {"override /scheduler": "lambdacosine"},
                {"override /checkpoint": "local"},
                {"override /callbacks": ["basic", "optimization", "anomalygen_texture_ft"]},
                {"override /ema": None},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": None},
                {"override /vlm_config": None},
                {"override /ckpt_type": "selective_dcp_anomalygen_texture_ft"},  # tiny filtered checkpoints
                "_self_",
            ],
            job=dict(project="anomalygen", group=dataset_name, name=job_name, wandb_mode="disabled"),
            model=dict(config=model_config),
            optimizer=dict(
                betas=list(betas),
                eps=eps,
                fused=True,
                # Train only the LoRA adapters and the learnable text prompt.
                keys_to_select=list(TRAINED_KEY_PREFIXES),
                lr_multipliers=dict(lr_multipliers or {}),
                lr=lr,
                optimizer_type="AdamW",
                weight_decay=weight_decay,
            ),
            scheduler=dict(
                # The scheduler takes parallel per-cycle lists; one cycle => one-element lists.
                lr_scheduler_type="LambdaCosine",
                cycle_lengths=[cycle_lengths],
                f_max=[1.0],
                f_min=[0.1],
                f_start=[0.0],
                warm_up_steps=[warm_up_steps],
            ),
            trainer=dict(
                distributed_parallelism=parallelism,
                grad_accum_iter=1,
                logging_iter=logging_iter,
                max_iter=max_iter,
                run_validation=True,
                validation_iter=validation_iter,
                run_validation_on_start=run_validation_on_start,
                seed=seed,
                grad_scaler_args=dict(enabled=False),
                # torch.compile recompiles per distinct PACKED SEQUENCE LENGTH, so the ceiling counts
                # combinations of input sizes per step, not input sizes. Must be set HERE: the trainer
                # calls set_torch_compile_options at init and overwrites any torch._dynamo.config set
                # outside. Training-only; generation keeps torch's defaults.
                compile_config=dict(recompile_limit=COMPILE_RECOMPILE_LIMIT, use_duck_shape=False),
                # Hand this recipe's values to the callbacks registered (arg-free) by
                # anomalygen.register; merges onto the `anomalygen_texture_ft` callbacks group.
                callbacks=dict(
                    # Tighter than the optimization group's default clip_norm of 1.0.
                    grad_clip=dict(clip_norm=0.1),
                    # Writes valid_kpi.csv each validation_iter: nn/mnn/fid always, axes + aq_nn
                    # (SAM2) when compute_anomaly_quality is on.
                    validation_kpi=dict(compute_anomaly_quality=compute_anomaly_quality),
                    # Reads that valid_kpi.csv and stops when `metric` fails to improve for `patience`
                    # consecutive validations. metric ∈ {nn, mnn, fid, aq_nn, completeness,
                    # precision, boundary_iou}.
                    early_stop=dict(
                        enabled=early_stop_enabled,
                        metric=early_stop_metric,
                        patience=early_stop_patience,
                        scope="Average",
                        min_delta=early_stop_min_delta,
                        min_delta_mode=early_stop_min_delta_mode,
                        cumulative_delta=False,
                    ),
                    # best_checkpoint.txt is selected on the same metric the run is monitored on,
                    # whether or not early stopping is enabled.
                    training_report=dict(recipe_path=recipe_path, best_metric=early_stop_metric),
                ),
            ),
            checkpoint=dict(
                dcp_async_mode_enabled=False,  # required for the save-keys filter (sync save path)
                # The trained subset (+ EMA) is absent from the base DCP checkpoint — it is trained
                # from scratch / template-initialized, so skip loading it.
                keys_to_skip_loading=list(WARM_START_SKIP_PREFIXES),
                load_path=base_checkpoint_path,
                save_iter=save_iter,
                strict_resume=False,
                verbose=True,
            ),
            dataloader_train=L(PackingDataLoader)(
                dataset_name="anomaly",
                patch_spatial=PATCH_SPATIAL,
                tokenizer_spatial_compression_factor=VAE_SPATIAL_COMPRESSION,
                tokenizer_temporal_compression_factor=4,
                # Exactly one of the two must be non-None (framework assert). Under a token budget
                # there is no sample ceiling, so samples/step varies with the input sizes drawn.
                max_sequence_length=max_sequence_length,
                max_samples_per_batch=None if max_sequence_length is not None else batch_size,
                dataloader=L(RankPartitionedDataLoader)(
                    # Inner per-pull size only; the packer above caps the actual training batch.
                    batch_size=batch_size,
                    num_workers=num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    prefetch_factor=4,
                    # Overrides the framework's custom_collate_fn (only set via setdefault). Required
                    # for variable resolution: edit_mask is not one of the framework's list-collated
                    # keys, so it would be stacked and raise on any mixed-size batch.
                    collate_fn=anomalygen_collate_fn,
                    datasets=dict(
                        image=dict(
                            ratio=1,
                            dataset=L(get_inpainting_dataset)(
                                dataset_dir=dataset_path,
                                anomaly_types=anomaly_types,
                                crop_mode=crop_mode,
                                crop_ratio_range=list(crop_ratio_range),
                                scale_range=list(scale_range),
                                input_size_ladder=list(input_size_ladder),
                                # Drives the per-worker augmentation rng, so the augmentation stream
                                # is reproducible from the experiment seed.
                                seed=seed,
                                # Cycle the (few-shot) sample set to a large virtual length so the
                                # PackingDataLoader's one-shot iterator never exhausts mid-run (it
                                # busy-spins on StopIteration otherwise). 100000x covers any max_iter.
                                repeat=100000,
                                background_dropout_prob=background_dropout_prob,
                                inst_aug_prob=inst_aug_prob,
                                ring_jitter_prob=ring_jitter_prob,
                                # Pad every caption to the learnable-prompt length so each sample
                                # emits a fixed text-token count (None when the feature is off).
                                text_prompt_pad_to=(text_prompt_num_tokens if learnable_text_prompt_enabled else None),
                                text_prompt_pad_word=text_prompt_init_word,
                                tokenizer_config="${model.config.vlm_config.tokenizer}",
                            ),
                        ),
                    ),
                ),
            ),
            # Real sharded validation loader: each rank generates a disjoint, (guidance, num_steps)-homogeneous
            # shard in model.validation_step; ValidationKPI.on_validation_end gathers + scores.
            dataloader_val=L(get_inpaint_val_dataloader)(
                input_data_path=testcase_jsonl,
                val_batch_size=validation_batch_size,
                num_workers=num_workers,
                # Dataloaders are instantiated under a chdir to the framework checkout, so
                # repo-root-relative testcase image/mask paths need an explicit root.
                base_dir=str(_REPO_ROOT),
                # Bound one validation forward by the training token budget: batches are cut by count,
                # and a batch of large defects planned at the top of the ladder would otherwise pack
                # many times the tokens of a training step.
                token_budget=max_sequence_length,
                diffusion_expert_config="${model.config.diffusion_expert_config}",
            ),
        ),
        flags={"allow_objects": True},
    )
