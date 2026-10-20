# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe validation in ``build_anomalygen_texture_ft_experiment``.

These guards are the only thing between a bad recipe and a training run that is silently wrong
rather than failing, so each one is pinned with the bad input it exists to reject.
"""

import pytest

# The builder imports the cosmos_framework model stack, which pulls in triton (a CUDA dep absent
# from the CPU-only test env). Skip the module there rather than failing collection, which would
# take every test below with it. Same guard as tests/anomalygen/configs/test_loader.py.
pytest.importorskip("triton")

from anomalygen.configs.texture.constants import MIN_CROP_RATIO  # noqa: E402
from anomalygen.configs.texture.exp_config import build_anomalygen_texture_ft_experiment  # noqa: E402
from anomalygen.data.utils import CAPTION_TOKEN_BUDGET  # noqa: E402

_MINIMAL = dict(
    dataset_name="d",
    anomaly_types=[["metal", "scratch"]],
    dataset_path="datasets/d",
    testcase_jsonl="datasets/d/testcase.jsonl",
)


def _build(**overrides):
    return build_anomalygen_texture_ft_experiment(**{**_MINIMAL, **overrides})


def _dec(exp):
    return exp["model"]["config"]["diffusion_expert_config"]


def test_shift_is_written_as_an_int():
    # The framework takes the scalar path on isinstance(shift, int) and treats a float as a
    # per-resolution dict, which raises TypeError mid-training.
    shift = _build(shift=3)["model"]["config"]["rectified_flow_training_config"]["shift"]
    assert isinstance(shift, int) and shift == 3


def test_fractional_shift_is_rejected():
    with pytest.raises(ValueError, match="whole number"):
        _build(shift=1.5)


@pytest.mark.parametrize("ladder", [(), (100, 200)], ids=["empty", "not-multiples-of-32"])
def test_unusable_input_size_ladder_is_rejected(ladder):
    # A size that is not a whole number of vision tokens cannot be patchified.
    with pytest.raises(ValueError):
        _build(input_size_ladder=ladder)


def test_input_size_ladder_is_sorted_and_deduplicated_onto_the_config():
    # Generation reads the ladder back off the recipe and snaps to neighbouring sizes, so it must be
    # ascending whatever order a recipe happened to list; a repeated size would also fail the
    # geometry stamp against a checkpoint that listed it once.
    assert tuple(_dec(_build(input_size_ladder=(1024, 256, 512, 512)))["input_size_ladder"]) == (256, 512, 1024)


def test_anomaly_types_are_sorted_and_deduplicated_onto_the_config():
    # Class ids are positions in this list, so they must not depend on the order a recipe wrote it in.
    dec = _dec(_build(anomaly_types=[["wood", "hole"], ["metal", "scratch"], ["wood", "hole"]]))
    assert dec["anomaly_types"] == [["metal", "scratch"], ["wood", "hole"]]
    assert dec["anomaly_num_classes"] == 2


def test_a_size_above_the_input_size_ceiling_is_rejected():
    # A size above MAX_INPUT_SIZE has to fail at parse, not inside a dataloader worker.
    with pytest.raises(ValueError, match="ceiling"):
        _build(input_size_ladder=(256, 4096), max_sequence_length=None)


def test_inference_scale_outside_scale_range_is_rejected():
    # Generating at a scale training never sampled is off-distribution with no error at run time.
    with pytest.raises(ValueError):
        _build(scale_range=(0.333, 3.0), inference_scale=4.0)


def test_a_largest_size_over_the_token_budget_is_rejected():
    with pytest.raises(ValueError, match="max_sequence_length"):
        _build(input_size_ladder=(256, 1536), max_sequence_length=1024)


def test_text_prompt_over_the_ceiling_is_rejected():
    with pytest.raises(ValueError, match="text_prompt_num_tokens must be <="):
        _build(learnable_text_prompt_enabled=True, text_prompt_num_tokens=CAPTION_TOKEN_BUDGET + 1)


# --- crop_mode ------------------------------------------------------------------------------------


def test_an_unknown_crop_mode_is_rejected():
    # A typo must fail at parse rather than silently training under the default framing.
    with pytest.raises(ValueError, match="crop_mode"):
        _build(crop_mode="fixed")


@pytest.mark.parametrize("crop_mode", ["scale", "ratio"])
def test_crop_mode_is_published_for_generation(crop_mode):
    # Generation reads the mode back off the config; unpublished, it would frame every crop the
    # default way whatever the model trained under.
    assert _dec(_build(crop_mode=crop_mode))["crop_mode"] == crop_mode


@pytest.mark.parametrize("crop_mode", ["scale", "ratio"])
def test_the_framing_band_is_published_verbatim(crop_mode):
    # Generation clamps a testcase's crop_ratio into this band, so it must be the band training drew
    # from, in either mode — one default, no per-mode rewriting behind the recipe's back.
    assert tuple(_dec(_build(crop_mode=crop_mode, crop_ratio_range=(1.5, 8.0)))["crop_ratio_range"]) == (1.5, 8.0)


def test_ratio_mode_ignores_the_unused_scale_axis():
    # scale_range and inference_scale have no effect under "ratio", so the check that keeps
    # generation inside the trained scale band must not fire on a value that is simply unused.
    exp = _build(crop_mode="ratio", inference_scale=4.0, scale_range=(0.333, 3.0))
    assert _dec(exp)["crop_mode"] == "ratio"


@pytest.mark.parametrize("low", [0.8, 1.0, 1.2], ids=["below-one", "one", "under-the-floor"])
def test_a_crop_ratio_under_the_floor_is_rejected(low):
    # Too little context around the defect to generate a coherent one, so it is refused at parse.
    with pytest.raises(ValueError, match="crop_ratio_range"):
        _build(crop_ratio_range=(low, 3.0))


def test_the_floor_itself_is_allowed():
    assert tuple(_dec(_build(crop_ratio_range=(MIN_CROP_RATIO, 3.0)))["crop_ratio_range"]) == (MIN_CROP_RATIO, 3.0)


def test_scale_range_may_sit_below_one():
    # scale is a resample factor, not a context multiplier — the crop_ratio floor must not reach it.
    assert _dec(_build(scale_range=(0.25, 4.0)))["crop_ratio_range"] is not None


# --- activation checkpointing ---------------------------------------------------------------------


def test_activation_checkpointing_mode_is_a_recipe_key():
    # The memory-only OOM remedy has to live in the recipe, so it is snapshotted with the run.
    mode = _build(activation_checkpointing_mode="full")["model"]["config"]["activation_checkpointing"]["mode"]
    assert mode == "full"


def test_an_unknown_activation_checkpointing_mode_is_rejected():
    with pytest.raises(ValueError, match="activation_checkpointing_mode"):
        _build(activation_checkpointing_mode="always")


# --- framing band ---------------------------------------------------------------------------------


def test_a_pinned_crop_ratio_band_is_rejected():
    # One framing only trains no context diversity, and quality_refine cannot draw over it.
    with pytest.raises(ValueError, match="low < high"):
        _build(crop_ratio_range=(2.0, 2.0))


# --- validation loader ----------------------------------------------------------------------------


def test_the_validation_loader_is_bounded_by_the_training_token_budget():
    # Validation batches are cut by count, so the loader needs the budget to shrink them under.
    assert _build()["dataloader_val"]["token_budget"] == 5120
    assert _build(max_sequence_length=None)["dataloader_val"]["token_budget"] is None
