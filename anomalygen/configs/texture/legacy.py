# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy recipes under the current loader.

Legacy training put every sample on one fixed input size, cut around the defect at a random ratio.
The current recipe spells that setup as ``input_size_ladder: [S]``, ``crop_mode: ratio`` and
``crop_ratio_range``. A legacy recipe (the copy in a legacy run dir) is rewritten into that spelling
here, so it registers as it is. Its ``anomaly_types`` listing order is also read back: a legacy
checkpoint numbered its classes in that order, the current code numbers them sorted (see
``checkpoint.utils.upgrade_legacy_state``).
"""

from __future__ import annotations

from typing import Any, Mapping

from omegaconf import OmegaConf

# Legacy builder defaults the current builder changed. A legacy recipe that omits one meant the legacy value.
LEGACY_DEFAULTS: dict[str, Any] = {
    "image_size": [512, 512],
    "model_input_size": 512,
    "ratio_range": [1.5, 8.0],
    "shift": 5.0,
    "background_dropout_prob": 0.5,
    "inst_aug_prob": 0.5,
    "ring_jitter_prob": 0.5,
}
# Legacy features the current model removed. Their weights have no slot, so a recipe that turns one
# on is refused.
_REMOVED_FEATURES = ("vision_prompt_item_enabled", "inpaint_class_hidden_embed_enabled")
# Legacy knobs with no current counterpart and no effect on the weights: dropped.
_DROPPED_KEYS = ("vision_prompt_num_tokens", "vision_prompt_freeze", "text_prompt_freeze")
# Keys only a legacy recipe carries; any one of them marks the recipe as legacy.
LEGACY_KEYS = frozenset({"image_size", "model_input_size", "ratio_range", *_REMOVED_FEATURES, *_DROPPED_KEYS})
# Current geometry keys. A recipe carrying one is not a legacy recipe and is never rewritten as one.
_GEOMETRY_KEYS = ("input_size_ladder", "crop_mode", "crop_ratio_range", "scale_range", "inference_scale")


def is_legacy_recipe(data: Mapping[str, Any]) -> bool:
    """True when ``data`` (builder kwargs from a recipe file) carries a key only a legacy recipe had."""
    return not LEGACY_KEYS.isdisjoint(data)


def translate_legacy_recipe(data: Mapping[str, Any]) -> dict[str, Any]:
    """The current spelling of a legacy recipe: legacy defaults filled in, legacy keys rewritten, the rest kept.

    Works on a recipe with no legacy key too: a legacy checkpoint's recipe may rely on the legacy
    defaults throughout and still mean them.
    """
    mixed = [k for k in _GEOMETRY_KEYS if k in data]
    if mixed:
        raise ValueError(
            f"Recipe carries current geometry keys {mixed}, so it cannot be read as a legacy recipe. "
            "A legacy checkpoint needs the recipe it trained under (the run-dir copy)."
        )
    out = {**LEGACY_DEFAULTS, **data}
    enabled = [k for k in _REMOVED_FEATURES if out.pop(k, False)]
    if enabled:
        raise ValueError(
            f"Legacy recipe enables {enabled}, which the current model removed; those weights cannot load."
        )
    for key in _DROPPED_KEYS:
        out.pop(key, None)

    size, side = out.pop("image_size"), int(out.pop("model_input_size"))
    if [int(v) for v in size] != [side, side]:
        raise ValueError(
            f"Legacy recipe: image_size {list(size)} must equal [model_input_size, model_input_size] = [{side}, {side}]."
        )
    low, high = out.pop("ratio_range")
    shift = out.pop("shift")
    if int(shift) != shift:
        raise ValueError(f"Legacy recipe: shift must be a whole number, got {shift}.")
    out.update(
        input_size_ladder=[side], crop_mode="ratio", crop_ratio_range=[float(low), float(high)], shift=int(shift)
    )
    if out.get("learnable_text_prompt_enabled"):
        # The legacy default length; the builder refuses it over its token budget.
        out.setdefault("text_prompt_num_tokens", 512)
    return out


def recipe_anomaly_types(path: str) -> list[list[str]]:
    """``anomaly_types`` in the order a YAML/JSON recipe lists them: the class order a legacy checkpoint used."""
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return [[str(t), str(d)] for t, d in data["anomaly_types"]]
