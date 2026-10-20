# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading a legacy recipe under the current loader: the rewrite is pure, so no builder or model is needed."""

import pathlib

import pytest
import yaml

import anomalygen
from anomalygen.configs.texture.legacy import (
    LEGACY_KEYS,
    is_legacy_recipe,
    recipe_anomaly_types,
    translate_legacy_recipe,
)

# The shipped recipe that spells the legacy mechanism by hand. The rewrite must produce the same block.
_SHIPPED_LEGACY_RECIPE = (
    pathlib.Path(anomalygen.__file__).resolve().parents[1] / "ag_config" / "exp_texture_ft_phone_screen_legacy.yaml"
)
_MECHANISM_KEYS = (
    "input_size_ladder",
    "crop_mode",
    "crop_ratio_range",
    "background_dropout_prob",
    "ring_jitter_prob",
    "inst_aug_prob",
)


def _legacy_recipe(**extra):
    """A legacy run-dir recipe copy: the dataset block plus the keys legacy recipes wrote out."""
    return {
        "dataset_name": "phone_screen",
        "anomaly_types": [["Phone", "stain"], ["Phone", "oil"]],
        "dataset_path": "datasets/phone_screen",
        "testcase_jsonl": "datasets/validation_phone_screen/testcase.jsonl",
        "image_size": [512, 512],
        "ratio_range": [1.5, 8.0],
        "model_input_size": 512,
        "shift": 5.0,
        **extra,
    }


def test_any_legacy_only_key_marks_a_recipe_as_legacy():
    for key in LEGACY_KEYS:
        assert is_legacy_recipe({key: 1}), key
    assert not is_legacy_recipe({"input_size_ladder": [512], "crop_mode": "ratio"})


def test_the_rewrite_matches_the_shipped_legacy_mechanism_block():
    shipped = yaml.safe_load(_SHIPPED_LEGACY_RECIPE.read_text())
    out = translate_legacy_recipe(_legacy_recipe())
    assert {k: out[k] for k in _MECHANISM_KEYS} == {k: shipped[k] for k in _MECHANISM_KEYS}
    # The one difference: legacy training used shift 5, and a checkpoint must keep it; the shipped recipe uses 1.
    assert out["shift"] == 5 and isinstance(out["shift"], int)
    assert LEGACY_KEYS.isdisjoint(out)


def test_a_recipe_with_no_legacy_key_is_read_with_the_legacy_defaults():
    # A legacy checkpoint's recipe may have relied on the defaults for every key that marks it.
    bare = _legacy_recipe()
    for key in ("image_size", "ratio_range", "model_input_size", "shift"):
        bare.pop(key)
    assert translate_legacy_recipe(bare) == translate_legacy_recipe(_legacy_recipe())


def test_explicit_values_win_over_the_legacy_defaults():
    out = translate_legacy_recipe(
        _legacy_recipe(image_size=[256, 256], model_input_size=256, ratio_range=[2, 4], shift=3, inst_aug_prob=0.1)
    )
    assert out["input_size_ladder"] == [256]
    assert out["crop_ratio_range"] == [2.0, 4.0]
    assert out["shift"] == 3
    assert out["inst_aug_prob"] == 0.1


def test_the_class_listing_order_and_every_other_key_are_kept():
    out = translate_legacy_recipe(_legacy_recipe(lr=0.5, job_name="j"))
    assert out["anomaly_types"] == [["Phone", "stain"], ["Phone", "oil"]], "the order is the checkpoint's class order"
    assert out["lr"] == 0.5 and out["job_name"] == "j"


@pytest.mark.parametrize("geometry_key", ["input_size_ladder", "crop_mode", "scale_range"])
def test_a_recipe_carrying_current_geometry_is_not_rewritten(geometry_key):
    # Filling legacy defaults over a current recipe would silently replace its geometry.
    with pytest.raises(ValueError, match="current geometry keys"):
        translate_legacy_recipe({"anomaly_types": [["t", "d"]], geometry_key: [512]})


@pytest.mark.parametrize(
    "bad", [dict(image_size=[512, 256]), dict(model_input_size=256)], ids=["non-square", "train-vs-model-size"]
)
def test_an_input_size_the_ladder_cannot_express_is_refused(bad):
    with pytest.raises(ValueError, match="image_size"):
        translate_legacy_recipe(_legacy_recipe(**bad))


def test_a_fractional_shift_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        translate_legacy_recipe(_legacy_recipe(shift=2.5))


@pytest.mark.parametrize("feature", ["vision_prompt_item_enabled", "inpaint_class_hidden_embed_enabled"])
def test_an_enabled_removed_feature_is_refused(feature):
    with pytest.raises(ValueError, match=feature):
        translate_legacy_recipe(_legacy_recipe(**{feature: True}))


def test_disabled_removed_features_and_knobs_without_a_counterpart_vanish():
    out = translate_legacy_recipe(
        _legacy_recipe(vision_prompt_item_enabled=False, vision_prompt_num_tokens=256, text_prompt_freeze=True)
    )
    assert not {"vision_prompt_item_enabled", "vision_prompt_num_tokens", "text_prompt_freeze"} & set(out)


def test_the_learnable_prompt_keeps_its_legacy_length_unless_the_recipe_set_one():
    assert translate_legacy_recipe(_legacy_recipe(learnable_text_prompt_enabled=True))["text_prompt_num_tokens"] == 512
    explicit = _legacy_recipe(learnable_text_prompt_enabled=True, text_prompt_num_tokens=128)
    assert translate_legacy_recipe(explicit)["text_prompt_num_tokens"] == 128
    assert "text_prompt_num_tokens" not in translate_legacy_recipe(_legacy_recipe())


def test_recipe_anomaly_types_reads_the_listing_order(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(_legacy_recipe()))
    assert recipe_anomaly_types(str(path)) == [["Phone", "stain"], ["Phone", "oil"]]
