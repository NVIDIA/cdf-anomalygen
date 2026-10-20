# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``anomalygen_texture_ft_legacy`` — I2I anomaly-inpainting training recipe (phone_screen dataset), legacy mechanism.

Python form of ``ag_config/exp_texture_ft_phone_screen_legacy.yaml``. Copied from
``exp_texture_ft_phone_screen.py``: ONLY ``job_name`` and the "legacy mechanism" block below differ. That
block spells the legacy framing and augmentations in current keys, following the "Fixed-input-size
training" section of docs/tutorial/texture/4-fine-tuning.md: every sample is cropped around its defect
onto one fixed 512 px input, with the legacy augmentation rates. A legacy recipe's own keys
(``image_size``, ``model_input_size``, ``ratio_range``) are accepted by the file loader only; a module
recipe must use the current keys, as here.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from anomalygen.configs.texture.exp_config import build_anomalygen_texture_ft_experiment

cs = ConfigStore.instance()

# ---- dataset (required); paths are relative to the repo root, the builder absolutizes them -------
DATASET_NAME = "phone_screen"
ANOMALY_TYPES = [["Phone", "oil"], ["Phone", "scratch"], ["Phone", "stain"]]
DATASET_PATH = "datasets/phone_screen"
TESTCASE_JSONL = "datasets/validation_phone_screen/testcase.jsonl"

anomalygen_texture_ft = build_anomalygen_texture_ft_experiment(
    dataset_name=DATASET_NAME,
    anomaly_types=ANOMALY_TYPES,
    job_name="anomalygen_texture_ft_legacy",
    dataset_path=DATASET_PATH,
    testcase_jsonl=TESTCASE_JSONL,
    # Cosmos3 backbone selection
    model_size="nano",  # "nano" (Qwen3-VL-8B, recommended default) | "edge" (Nemotron-3 Dense VL 2B, experimental)
    # per-defect-type LoRA rank and alpha; default to 8
    per_class_lora_rank=8,
    per_class_lora_alpha=8,
    # legacy mechanism: fixed 512 px input, bbox-relative framing, legacy augmentation rates
    input_size_ladder=[512],
    crop_mode="ratio",
    crop_ratio_range=[1.5, 8.0],
    background_dropout_prob=0.5,
    ring_jitter_prob=0.5,
    inst_aug_prob=0.5,
    # schedule / trainer
    max_iter=15000,
    validation_iter=1000,
    run_validation_on_start=True,  # run one validation pass at iter 0, before training
    save_iter=1000,
    cycle_lengths=15000,  # LambdaCosine single-cycle length (steps)
    warm_up_steps=500,
    logging_iter=10,
    seed=42,
    # optimizer (AdamW)
    lr=1.0e-03,
    betas=(0.9, 0.95),
    eps=1.0e-06,
    weight_decay=0.0,
    # validation generation
    validation_batch_size=16,
    # early stopping (disabled by default)
    early_stop_enabled=False,
    # nn | mnn | fid | aq_nn | completeness | precision | boundary_iou
    # fid is lower-better; every other metric is higher-better
    early_stop_metric="nn",
    early_stop_patience=5,
    early_stop_min_delta=0.0,
    early_stop_min_delta_mode="rel",  # rel | abs
)

cs.store(group="experiment", package="_global_", name="anomalygen_texture_ft", node=anomalygen_texture_ft)
