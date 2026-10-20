# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``DiffusionExpertConfig`` extended with the I2I anomaly-inpainting knobs.

With every flag at its default (all off), it behaves identically to the base config. Field groups
mirror ``build_anomalygen_texture_ft_model_config``, which is what populates them; inference reads
them back off the registered node instead of re-parsing the recipe.
"""

from __future__ import annotations

from typing import List, Tuple

import attrs
from cosmos_framework.configs.base.defaults.model_config import DiffusionExpertConfig

from anomalygen.configs.texture.constants import (
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_RATIO_RANGE,
    DEFAULT_INPUT_SIZE_LADDER,
    DEFAULT_MODEL_SIZE,
)
from anomalygen.data.utils import CAPTION_TOKEN_BUDGET


@attrs.define(slots=False)
class AnomalyGenTextureDiffusionExpertConfig(DiffusionExpertConfig):
    """``DiffusionExpertConfig`` + the I2I anomaly-inpainting knobs."""

    # --- backbone ---------------------------------------------------------------------------------

    # Size this config was built for, so inference can pick the matching base DCP.
    model_size: str = DEFAULT_MODEL_SIZE

    # `anomaly_num_classes` sizes `text_prompt_emb`; `anomaly_types` are the ``[texture, defect]``
    # pairs in class-id order, used to rebuild each class's caption for the text-prompt template init.
    anomaly_num_classes: int = 0
    anomaly_types: List[List[str]] = attrs.field(factory=list)

    # --- per-defect-type conditioning: two independent mechanisms; either or both stack -----------

    # 1) Per-class LoRA on the gen-attention projections: each class gets its own stacked adapter and
    # gen tokens are routed per forward, so all classes train in one mixed-batch run. Mutually
    # exclusive with the shared ``lora_enabled`` (same targets). Needs ``cp_world_size=1`` so every
    # gen row aligns with the routing vector.
    per_class_lora_enabled: bool = False
    per_class_lora_rank: int = 8
    per_class_lora_alpha: int = 8

    # 2) Per-class learnable text prompt (`text_prompt_emb`): the whole text-token sequence becomes
    # tunable, initialized from the caption template.
    learnable_text_prompt_enabled: bool = False
    # Fixed prompt length = caption tokens emitted per sample; the builder rejects anything above it.
    text_prompt_num_tokens: int = CAPTION_TOKEN_BUDGET
    # Token embedding that initializes the padded tail of each class's prompt.
    text_prompt_init_word: str = "anomaly"

    # --- inpainting -------------------------------------------------------------------------------

    # Latent edit-mask downsampling: "any" (max-pool any-touch, keeps thin defects) | "nearest".
    inpaint_latent_mask_mode: str = "any"

    # Weight on the background term of the region-split loss (see ``masked_flow_matching_loss``); the
    # defect term is always unweighted. 0.0 = defect-only. Both terms are per-region means, so 1.0
    # makes one cell of background error count as much as one of defect error — a far stronger
    # background pull than the base whole-image loss applied.
    inpaint_loss_background_weight: float = 0.0

    # Latent hard-composite outside the mask at every denoising step (inference).
    inpaint_replacement_trick: bool = False

    # --- crop geometry: recorded by the experiment builder so generation mirrors training ---------

    # Where the framing comes from: "scale" (context + resample axes) or "ratio" (context axis only).
    crop_mode: str = DEFAULT_CROP_MODE
    # A single-size ladder is how you pin one fixed model input size.
    input_size_ladder: List[int] = attrs.field(factory=lambda: list(DEFAULT_INPUT_SIZE_LADDER))
    # Trained framing band; inference clamps a testcase's ``crop_ratio`` into it and reads the low
    # bound as the framing floor.
    crop_ratio_range: Tuple[float, float] = DEFAULT_CROP_RATIO_RANGE
    # "scale" mode only; unused under "ratio", where the framing has no scale axis.
    inference_scale: float = 1.0

    # --- runtime ----------------------------------------------------------------------------------

    # Mirrors ``trainer.distributed_parallelism``: "ddp" (each rank keeps the full model) or "fsdp"
    # (shard params/optimizer). Read by ``AnomalyGenTextureMoTModel.set_up_parallelism``, which under
    # "ddp" skips fully_shard so the DDP wrapper can broadcast plain-tensor params.
    parallelism: str = "ddp"
