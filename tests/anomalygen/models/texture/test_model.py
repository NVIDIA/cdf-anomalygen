# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``anomalygen/models/texture/model.py``.

One file per source module, per the layout in CONTRIBUTING.md. Sections, in order: the region-split
flow-matching loss, the inference replacement trick, flat per-sample item addressing, the bounded
CFG blend, and batched vision encoding.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

from anomalygen.configs.texture.constants import DEFAULT_GUIDANCE_INTERVAL
from anomalygen.models.texture.model import (
    AnomalyGenTextureMoTModel,
    _group_is_encodable,
    _per_item_pixel_masks,
    _Region,
    downsample_edit_mask,
    masked_flow_matching_loss,
    replacement_velocity,
)

# --------------------------------------------------------------------------------------------------
# masked_flow_matching_loss — region-split loss
# --------------------------------------------------------------------------------------------------

C, T, H, W = 3, 1, 8, 8


class _UniformRectifiedFlow:
    """Stand-in for ``RectifiedFlow``; ``TrainTimeWeight`` only supports "uniform" (all-ones)."""

    @staticmethod
    def train_time_weight(t, tensor_kwargs):
        return torch.ones_like(t)


def _mask(n_cells: int) -> torch.Tensor:
    """``[1,T,H,W]`` latent mask with the first ``n_cells`` spatial cells set."""
    m = torch.zeros(1, T, H * W)
    m[0, :, :n_cells] = 1.0
    return m.reshape(1, T, H, W)


def _call(*, edit_masks, pred, target, condition_mask=None, bg_loss_weight=0.0):
    n = len(pred)
    if condition_mask is None:
        condition_mask = [torch.zeros(T, 1, 1) for _ in range(n)]  # 0 = noisy = supervised
    return masked_flow_matching_loss(
        pred=pred,
        target=target,
        condition_mask=condition_mask,
        edit_masks=edit_masks,
        timesteps=torch.full((n, 1), 0.5),
        rectified_flow=_UniformRectifiedFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32, "device": torch.device("cpu")},
        bg_loss_weight=bg_loss_weight,
    )


def _constant_error_item(err: float):
    """One item whose per-element squared error is exactly ``err`` everywhere."""
    target = torch.zeros(C, T, H, W)
    pred = torch.full((C, T, H, W), err**0.5)
    return pred, target


@pytest.mark.parametrize("n_cells", [1, 4, 40, H * W])
def test_defect_term_is_area_invariant(n_cells):
    """Same per-element error, wildly different mask areas -> identical defect term. Both terms are
    per-region means, so a small defect is weighed like a large one."""
    pred, target = _constant_error_item(2.0)
    loss, _, defect, _ = _call(edit_masks=[_mask(n_cells)], pred=[pred], target=[target])
    # per-cell sum over C channels: C * err
    assert defect.item() == pytest.approx(C * 2.0, rel=1e-5)
    assert loss.item() == pytest.approx(defect.item(), rel=1e-5)


def test_defect_term_is_the_masked_mean_of_the_squared_error():
    """``sum(sqerr * em) / sum(em)`` exactly — the scale recipe learning rates are tuned for."""
    torch.manual_seed(0)
    pred, target = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    em = _mask(9)

    _, _, defect, _ = _call(edit_masks=[em], pred=[pred], target=[target])

    sqerr = (pred - target) ** 2
    expected = (sqerr * em).sum() / em.sum()
    assert defect.item() == pytest.approx(expected.item(), rel=1e-5)


def test_defect_term_over_a_full_frame_mask_is_c_times_the_plain_mean():
    """Pins the scale vs the framework's whole-image loss: numerators sum over C channels while
    denominators count cells, so a full-frame mask lands at ``C * mean(sqerr)``."""
    torch.manual_seed(1)
    pred, target = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    _, _, defect, _ = _call(edit_masks=[_mask(H * W)], pred=[pred], target=[target])
    assert defect.item() == pytest.approx(C * ((pred - target) ** 2).mean().item(), rel=1e-5)


def test_background_weight_zero_gives_the_background_no_gradient():
    """The default. Background error is measured and reported, but must not reach the parameters."""
    torch.manual_seed(2)
    pred = torch.randn(C, T, H, W, requires_grad=True)
    target = torch.randn(C, T, H, W)
    em = _mask(5)

    loss, _, _, bg = _call(edit_masks=[em], pred=[pred], target=[target], bg_loss_weight=0.0)
    loss.backward()

    assert bg.item() > 0.0  # measured...
    assert torch.all(pred.grad * (1.0 - em) == 0.0)  # ...but not in the objective
    assert torch.any(pred.grad * em != 0.0)


def test_background_weight_scales_only_the_background_term():
    torch.manual_seed(3)
    pred, target = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    em = _mask(6)

    _, _, defect_0, bg_0 = _call(edit_masks=[em], pred=[pred], target=[target], bg_loss_weight=0.0)
    loss_w, _, defect_w, bg_w = _call(edit_masks=[em], pred=[pred], target=[target], bg_loss_weight=0.25)

    assert defect_w.item() == pytest.approx(defect_0.item(), rel=1e-6)
    assert bg_w.item() == pytest.approx(bg_0.item(), rel=1e-6)
    assert loss_w.item() == pytest.approx(defect_0.item() + 0.25 * bg_0.item(), rel=1e-5)


def test_background_term_is_area_invariant_too():
    """Both sides are per-region means, so ``bg_loss_weight=1.0`` weighs one cell of background
    error like one cell of defect error, independent of either area."""
    pred, target = _constant_error_item(3.0)
    _, _, _, bg_small = _call(edit_masks=[_mask(1)], pred=[pred], target=[target])
    _, _, _, bg_large = _call(edit_masks=[_mask(H * W - 1)], pred=[pred], target=[target])
    assert bg_small.item() == pytest.approx(bg_large.item(), rel=1e-5)
    assert bg_small.item() == pytest.approx(C * 3.0, rel=1e-5)


def test_empty_mask_is_finite_and_contributes_no_defect_term():
    """Denominator clamps to 1 while the numerator is structurally 0 — no NaN, no phantom gradient."""
    torch.manual_seed(4)
    pred = torch.randn(C, T, H, W, requires_grad=True)
    target = torch.randn(C, T, H, W)

    loss, _, defect, bg = _call(edit_masks=[_mask(0)], pred=[pred], target=[target], bg_loss_weight=1.0)
    loss.backward()

    assert defect.item() == 0.0
    assert torch.isfinite(loss) and torch.isfinite(bg)
    assert torch.all(torch.isfinite(pred.grad))


def test_full_frame_mask_is_finite_and_contributes_no_background_term():
    torch.manual_seed(5)
    pred, target = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    loss, _, defect, bg = _call(edit_masks=[_mask(H * W)], pred=[pred], target=[target], bg_loss_weight=1.0)

    assert bg.item() == 0.0
    assert torch.isfinite(loss) and defect.item() > 0.0


def test_conditioning_items_are_excluded_but_keep_their_per_item_slot():
    """Only the last item of [source, target] is noised; averaging in the clean one would halve the
    loss. ``per_item`` must stay aligned with ``sigma``, so skipped slots are zero-filled."""
    torch.manual_seed(6)
    pred_src, target_src = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    pred_tgt, target_tgt = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    em = _mask(7)

    both = _call(
        edit_masks=[em, em],
        pred=[pred_src, pred_tgt],
        target=[target_src, target_tgt],
        condition_mask=[torch.ones(T, 1, 1), torch.zeros(T, 1, 1)],  # source clean, target noisy
    )
    only = _call(edit_masks=[em], pred=[pred_tgt], target=[target_tgt])

    loss, per_item, _, _ = both
    assert loss.item() == pytest.approx(only[0].item(), rel=1e-5)
    assert per_item.shape == (2,)
    assert per_item[0].item() == 0.0
    assert per_item[1].item() == pytest.approx(loss.item(), rel=1e-5)


def test_item_with_no_mask_is_skipped():
    """``_anomaly_edit_masks`` carries None at any item without an edit mask."""
    torch.manual_seed(7)
    pred, target = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    em = _mask(4)

    loss, per_item, _, _ = _call(edit_masks=[None, em], pred=[pred, pred], target=[target, target])
    assert per_item.shape == (2,)
    assert per_item[0].item() == 0.0
    assert torch.isfinite(loss)


def test_returns_none_when_nothing_qualifies():
    """The caller's signal to fall back to the base whole-image loss."""
    pred, target = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    assert _call(edit_masks=[], pred=[pred], target=[target]) is None
    assert _call(edit_masks=[None], pred=[pred], target=[target]) is None
    # every item is pure conditioning
    assert (
        _call(
            edit_masks=[_mask(4)],
            pred=[pred],
            target=[target],
            condition_mask=[torch.ones(T, 1, 1)],
        )
        is None
    )


def test_shape_mismatch_between_mask_and_prediction_is_skipped():
    pred, target = torch.randn(C, T, H, W), torch.randn(C, T, H, W)
    wrong = torch.ones(1, T, H // 2, W // 2)
    assert _call(edit_masks=[wrong], pred=[pred], target=[target]) is None


def test_reduction_runs_in_fp32_for_bf16_predictions():
    """Sums run over ~200k elements; bf16 accumulation would visibly drift."""
    torch.manual_seed(8)
    pred = torch.randn(C, T, H, W).to(torch.bfloat16)
    target = torch.randn(C, T, H, W).to(torch.bfloat16)
    loss, per_item, defect, bg = _call(edit_masks=[_mask(5)], pred=[pred], target=[target], bg_loss_weight=0.5)

    for t in (loss, per_item, defect, bg):
        assert t.dtype == torch.float32

    sqerr = (pred.float() - target.float()) ** 2
    em = _mask(5)
    assert defect.item() == pytest.approx(((sqerr * em).sum() / em.sum()).item(), rel=1e-5)


# --- _per_item_pixel_masks: how the batch's masks reach the loss ----------------------------------


def _pix(h, w, dims):
    """A distinct pixel mask at the requested rank: [1,H,W] from training, [1,1,H,W] from inference."""
    m = torch.full((h, w), float(h))
    return m.unsqueeze(0).unsqueeze(0) if dims == 4 else m.unsqueeze(0)


@pytest.mark.parametrize("dims", [3, 4], ids=["training-1HW", "inference-11HW"])
def test_ragged_mask_list_is_repeated_per_vision_item(dims):
    # The two producers hand this different ranks, and under variable resolution the batch is a LIST
    # of differently-sized masks. Every item of a sample must get THAT sample's mask, normalised to
    # [1,H,W] — a mis-expansion silently supervises the wrong region.
    masks = [_pix(8, 8, dims), _pix(16, 4, dims), _pix(32, 32, dims)]
    flat = _per_item_pixel_masks(masks, [2, 2, 2], 6)

    assert [tuple(m.shape) for m in flat] == [(1, 8, 8), (1, 8, 8), (1, 16, 4), (1, 16, 4), (1, 32, 32), (1, 32, 32)]
    assert [float(m.flatten()[0]) for m in flat] == [8, 8, 16, 16, 32, 32]  # in sample order


def test_stacked_uniform_mask_tensor_is_split_per_sample():
    # The fixed-resolution path: collation stacks to [B,1,H,W] instead of a list.
    stacked = torch.stack([torch.full((1, 8, 8), float(i)) for i in range(3)])
    flat = _per_item_pixel_masks(stacked, [2, 2, 2], 6)
    assert [float(m.flatten()[0]) for m in flat] == [0, 0, 1, 1, 2, 2]


def test_a_single_mask_is_broadcast_to_every_item():
    flat = _per_item_pixel_masks(torch.ones(1, 8, 8), None, 4)
    assert len(flat) == 4 and all(tuple(m.shape) == (1, 8, 8) for m in flat)


@pytest.mark.parametrize("edit", [None, [], [object()]], ids=["none", "empty", "no-tensors"])
def test_nothing_usable_returns_none(edit):
    # None disables the masked loss entirely and falls back to the base one, so it must happen only
    # when there really is no mask to use.
    assert _per_item_pixel_masks(edit, [2], 2) is None


def test_a_mask_count_that_cannot_be_reconciled_returns_none():
    # Two masks for three items is ambiguous; broadcasting the first would supervise sample 1's
    # region against sample 0's mask.
    assert _per_item_pixel_masks([_pix(8, 8, 3), _pix(8, 8, 3)], None, 3) is None


# --------------------------------------------------------------------------------------------------
# replacement trick — downsample_edit_mask / replacement_velocity
# --------------------------------------------------------------------------------------------------

# --- downsample_edit_mask -------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3, 4], ids=["HW", "1HW", "11HW"])
def test_any_mode_keeps_a_defect_that_touches_only_one_pixel(dim):
    # "any" exists so a defect thinner than the latent stride survives: nearest sampling would drop
    # it, leaving the region frozen to the background and the defect never generated.
    mask = torch.zeros(16, 16)
    mask[3, 5] = 1.0
    for _ in range(dim - 2):
        mask = mask.unsqueeze(0)

    out = downsample_edit_mask(mask, 1, 4, 4, "any", torch.float32, torch.device("cpu"))

    assert out.shape == (1, 1, 4, 4)
    assert out.sum() == 1.0
    assert out[0, 0, 3 // 4, 5 // 4] == 1.0  # and in the right cell


def test_any_mode_does_not_invent_a_defect():
    out = downsample_edit_mask(torch.zeros(16, 16), 1, 4, 4, "any", torch.float32, torch.device("cpu"))
    assert out.sum() == 0.0


def test_output_is_binary_and_broadcast_over_the_time_axis():
    # A soft value would make the replacement trick blend rather than anchor.
    mask = torch.rand(16, 16)
    out = downsample_edit_mask(mask, 3, 8, 8, "any", torch.float32, torch.device("cpu"))
    assert out.shape == (1, 3, 8, 8)
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(out[0, 0], out[0, 2])  # every frame carries the same mask


def test_an_unexpected_mask_rank_raises():
    with pytest.raises(ValueError, match="Unexpected edit_mask shape"):
        downsample_edit_mask(torch.zeros(2, 1, 1, 16, 16), 1, 4, 4, "any", torch.float32, torch.device("cpu"))


# --- replacement_velocity -------------------------------------------------------------------------


def _flat(*items):
    return torch.cat([it.reshape(-1) for it in items])


def test_background_lands_exactly_on_the_guide_and_the_defect_is_untouched():
    """The whole point: after one step the background IS the guide, whatever the model predicted.

    Checked by re-deriving x0 the way the sampler does (``x0 = x_t - sigma * v``), so a sign error in
    the velocity conversion shows up here rather than as a slightly-off background at generation.
    """
    sigma = 0.5
    shape = (1, 1, 2, 4)
    guide = torch.full(shape, 7.0)
    latent = torch.full(shape, 3.0)
    velocity = torch.full(shape, -2.0)  # whatever the network said
    mask = torch.zeros(shape)
    mask[..., :2] = 1.0  # left half is the defect, right half is background

    lead = torch.zeros(6)  # a preceding item, so the offset is exercised
    out = replacement_velocity(
        flat_velocity=_flat(lead, velocity),
        flat_latent=_flat(lead, latent),
        guide_latent=guide,
        keep_latent=1.0 - mask,
        offset=lead.numel(),
        sigma=sigma,
    )

    vel_out = out[lead.numel() :].reshape(shape)
    x0 = latent - sigma * vel_out

    assert torch.allclose(x0[..., 2:], guide[..., 2:]), "background must be pinned to the guide"
    assert torch.allclose(vel_out[..., :2], velocity[..., :2]), "the defect region must be untouched"
    assert torch.equal(out[: lead.numel()], lead), "items before the target must not be rewritten"


def test_an_all_defect_mask_leaves_the_velocity_alone():
    # keep = 1 - mask = 0 everywhere, so there is nothing to anchor.
    shape = (1, 1, 2, 2)
    velocity = torch.randn(shape)
    out = replacement_velocity(
        flat_velocity=velocity.reshape(-1),
        flat_latent=torch.randn(shape).reshape(-1),
        guide_latent=torch.randn(shape),
        keep_latent=torch.zeros(shape),
        offset=0,
        sigma=0.5,
    )
    assert torch.allclose(out, velocity.reshape(-1))


def test_an_empty_mask_anchors_the_whole_item():
    # The complement: with no defect region every position is held at the guide.
    shape = (1, 1, 2, 2)
    guide, latent, sigma = torch.full(shape, 4.0), torch.full(shape, 1.0), 0.25
    out = replacement_velocity(
        flat_velocity=torch.randn(shape).reshape(-1),
        flat_latent=latent.reshape(-1),
        guide_latent=guide,
        keep_latent=torch.ones(shape),
        offset=0,
        sigma=sigma,
    )
    assert torch.allclose(latent - sigma * out.reshape(shape), guide)


def test_the_input_velocity_is_not_mutated():
    # The caller keeps the conditional branch's velocity; writing through would corrupt the CFG blend.
    shape = (1, 1, 2, 2)
    velocity = torch.randn(shape).reshape(-1)
    before = velocity.clone()
    replacement_velocity(
        flat_velocity=velocity,
        flat_latent=torch.randn(shape).reshape(-1),
        guide_latent=torch.randn(shape),
        keep_latent=torch.ones(shape),
        offset=0,
        sigma=0.5,
    )
    assert torch.equal(velocity, before)


# --------------------------------------------------------------------------------------------------
# flat item addressing — _clean_source / _target_regions
# --------------------------------------------------------------------------------------------------

CLEAN_SOURCE = AnomalyGenTextureMoTModel._clean_source


def _case(items_per_sample: int, n_samples: int = 2, shape=(2, 1, 3, 3)):
    """Per-sample flat tensors plus the matching x0 list, all items distinct by fill value."""
    x0, noise_x = [], []
    for s in range(n_samples):
        items = [torch.full(shape, float(10 * s + j)) for j in range(items_per_sample)]
        x0 += items
        noise_x.append(torch.cat([it.reshape(-1) for it in items]))
    return noise_x, SimpleNamespace(x0_tokens_vision=x0)


def test_source_slice_becomes_the_target_latent():
    noise_x, gen = _case(items_per_sample=2)
    out = CLEAN_SOURCE(SimpleNamespace(_anomaly_items_per_sample=2), noise_x, gen)

    n = gen.x0_tokens_vision[0].numel()
    for s in range(2):
        source, target = out[s][:n], out[s][n:]
        assert torch.equal(source, gen.x0_tokens_vision[2 * s + 1].reshape(-1))  # source <- target
        assert torch.equal(target, gen.x0_tokens_vision[2 * s + 1].reshape(-1))  # target untouched


def test_input_is_not_mutated():
    """The sampler owns noise_x; writing through would corrupt the conditional branch's trajectory."""
    noise_x, gen = _case(items_per_sample=2)
    before = [t.clone() for t in noise_x]
    CLEAN_SOURCE(SimpleNamespace(_anomaly_items_per_sample=2), noise_x, gen)
    assert all(torch.equal(a, b) for a, b in zip(noise_x, before))


# --- _target_regions -----------------------------------------------------------------------------

TARGET_REGIONS = AnomalyGenTextureMoTModel._target_regions


def _regions_self(items_per_sample, masks):
    """A stub ``self``. Real instance so the warn-once helper on the no-regions path resolves."""
    model = object.__new__(AnomalyGenTextureMoTModel)
    for key, value in dict(
        _anomaly_items_per_sample=items_per_sample,
        _anomaly_edit_masks=masks,
        _warned_target_regions=False,
    ).items():
        object.__setattr__(model, key, value)
    return model


def test_target_region_offset_skips_every_item_before_the_target():
    """The offset addresses the LAST item, so it must sum the numel of all the ones before it."""
    ips = 2
    _, gen = _case(items_per_sample=ips, n_samples=2)
    masks = [torch.full((1, 1, 1, 1), float(i)) for i in range(ips * 2)]
    n = gen.x0_tokens_vision[0].numel()

    regions = TARGET_REGIONS(_regions_self(ips, masks), gen, n_samples=2)

    assert [r[0] for r in regions] == [(ips - 1) * n] * 2  # every item before the target is skipped
    for s, (_, guide, mask) in enumerate(regions):
        assert torch.equal(guide, gen.x0_tokens_vision[ips * s + ips - 1])  # the target, not item 0
        assert torch.equal(mask, masks[ips * s + ips - 1])  # its own mask, not the sample's first


def test_inconsistent_item_layout_raises():
    # A wrong items-per-sample would otherwise hand back offsets pointing into another sample.
    _, gen = _case(items_per_sample=2, n_samples=2)
    masks = [torch.zeros(1, 1, 1, 1) for _ in range(4)]
    with pytest.raises(ValueError, match="vision items per sample"):
        TARGET_REGIONS(_regions_self(3, masks), gen, n_samples=2)


def test_returns_none_when_there_are_no_edit_masks():
    _, gen = _case(items_per_sample=2, n_samples=2)
    assert TARGET_REGIONS(_regions_self(2, None), gen, n_samples=2) is None


def test_generate_samples_pins_the_cfg_policy_outright(monkeypatch):
    """The x0 bound gates on DEFAULT_GUIDANCE_INTERVAL and assumes no norm rescale, so a caller's
    own values could not be honoured there; they are overwritten, not taken as defaults."""
    seen = {}

    def _base(self, *args, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(OmniMoTModel, "generate_samples_from_batch", _base, raising=True)
    model = object.__new__(AnomalyGenTextureMoTModel)

    model.generate_samples_from_batch(guidance=3.0, guidance_interval=[1.0, 2.0], normalize_cfg=True)
    assert seen["guidance_interval"] == list(DEFAULT_GUIDANCE_INTERVAL)
    assert seen["normalize_cfg"] is False


# --------------------------------------------------------------------------------------------------
# CFG blend override — _run_classifier_free_guidance
# --------------------------------------------------------------------------------------------------

# One sample: a flat [source, target] latent of 8 elements, target starting at 4.
_OFFSET, _LENGTH = 4, 4
# num_train_timesteps 1000 at timestep 500 -> sigma 0.5, and x_t is 0, so x0 = -0.5*v.
_TIMESTEP, _SIGMA = 500.0, 0.5
_GUIDE = torch.tensor([-1.0, 0.0, 1.0, 2.0])  # clean target latent; its min/max are the bound


def _model(
    monkeypatch,
    cond_v,
    uncond_v,
    *,
    guidance=2.0,
    timestep=_TIMESTEP,
    regions=...,
    noise_x=...,
):
    monkeypatch.setattr(
        OmniMoTModel,
        "_run_classifier_free_guidance",
        lambda self, **kwargs: (cond_v, uncond_v),
        raising=True,
    )
    model = object.__new__(AnomalyGenTextureMoTModel)
    noise = [torch.zeros(8)] if noise_x is ... else noise_x
    if regions is ...:
        regions = [_Region(_OFFSET, _GUIDE, None, _GUIDE.min(), _GUIDE.max())]
    for key, value in dict(
        _cfg_guidance=guidance,
        _cfg_step=None if noise is None else (timestep, noise),
        _cfg_regions=regions or [],
        _cfg_clamp_bound=None,
        config=SimpleNamespace(rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000)),
    ).items():
        object.__setattr__(model, key, value)
    return model


def _branches(target_cond):
    """``(cond_v, uncond_v)`` for one sample; uncond is 0, so the blend is ``g * cond``."""
    cond = torch.zeros(8)
    cond[_OFFSET : _OFFSET + _LENGTH] = torch.tensor(target_cond)
    cond[0] = 7.0  # source-region value, which must pass through untouched
    return [cond], [torch.zeros(8)]


def test_an_in_range_blend_is_the_plain_cfg_formula(monkeypatch):
    # The x0 round trip is exact where the clamp does not bind, so bounding must not perturb a
    # sample that was already in range.
    cond_v, uncond_v = _branches([0.1, 0.2, -0.1, -0.2])
    out_c, _ = _model(monkeypatch, cond_v, uncond_v)._run_classifier_free_guidance()

    assert torch.allclose(out_c[0], uncond_v[0] + 2.0 * (cond_v[0] - uncond_v[0]))


def test_the_result_is_returned_as_both_branches(monkeypatch):
    cond_v, uncond_v = _branches([0.1, 0.2, -0.1, -0.2])
    out_c, out_u = _model(monkeypatch, cond_v, uncond_v)._run_classifier_free_guidance()

    assert out_c is out_u
    # The base re-applies u + g*(c - u) to this; same object both sides makes it an exact identity.
    assert torch.equal(out_c[0] + 2.0 * (out_c[0] - out_u[0]), out_c[0])


def test_the_region_outside_the_target_item_is_left_alone(monkeypatch):
    # Only the target item is bounded; the shared source slice must survive, or the conditioning the
    # two branches share stops cancelling.
    cond_v, uncond_v = _branches([-3.0] * _LENGTH)
    out_c, _ = _model(monkeypatch, cond_v, uncond_v)._run_classifier_free_guidance()

    assert out_c[0][0].item() == pytest.approx(14.0)


@pytest.mark.parametrize(
    ("cond_value", "expected_v"),
    [
        (-3.0, -4.0),  # v=-6 -> x0=3, above guide max 2 -> clamped -> v=(0-2)/0.5
        (2.0, 2.0),  # v=4 -> x0=-2, below guide min -1 -> clamped -> v=(0+1)/0.5
    ],
    ids=["above-max", "below-min"],
)
def test_an_out_of_range_x0_is_pulled_back_into_the_clean_latent_range(monkeypatch, cond_value, expected_v):
    # Rules out the saturated patch that unbounded extrapolation produces.
    cond_v, uncond_v = _branches([cond_value] * _LENGTH)
    out_c, _ = _model(monkeypatch, cond_v, uncond_v)._run_classifier_free_guidance()

    target = out_c[0][_OFFSET : _OFFSET + _LENGTH]
    assert torch.allclose(target, torch.full((_LENGTH,), expected_v))
    x0 = -_SIGMA * target
    assert float(_GUIDE.min()) - 1e-6 <= float(x0.min()) and float(x0.max()) <= float(_GUIDE.max()) + 1e-6


def test_the_clamp_counts_the_values_it_moved_and_reports_once(monkeypatch, loguru_lines):
    # Summed on device across the steps of one generate call and read back once at the end, so a
    # clamp that binds often is visible without a host sync per step.
    cond_v, uncond_v = _branches([-3.0] * _LENGTH)  # every target value lands above the guide max
    model = _model(monkeypatch, cond_v, uncond_v)
    model._run_classifier_free_guidance()
    model._run_classifier_free_guidance()
    bound, total = model._cfg_clamp_bound
    assert int(bound) == 2 * _LENGTH and total == 2 * _LENGTH

    model._log_clamp_rate()
    assert model._cfg_clamp_bound is None
    assert any("clamp bound 100.00%" in ln for ln in loguru_lines)


def test_an_in_range_blend_moves_nothing(monkeypatch):
    cond_v, uncond_v = _branches([0.1, 0.2, -0.1, -0.2])
    model = _model(monkeypatch, cond_v, uncond_v)
    model._run_classifier_free_guidance()
    bound, total = model._cfg_clamp_bound
    assert int(bound) == 0 and total == _LENGTH


@pytest.mark.parametrize(
    "overrides",
    [{"guidance": 1.0}, {"timestep": 0.0}, {"regions": None}, {"noise_x": None}],
    ids=["guidance-1", "sigma-0", "no-regions", "no-noise-x"],
)
def test_the_branches_pass_straight_through(monkeypatch, overrides):
    # Each guard returns the base's own branches as the SAME objects, so no blend and no clamp.
    cond_v, uncond_v = _branches([-3.0] * _LENGTH)  # would visibly clamp if the blend ran
    out_c, out_u = _model(monkeypatch, cond_v, uncond_v, **overrides)._run_classifier_free_guidance()

    assert out_c is cond_v and out_u is uncond_v


@pytest.mark.parametrize("timestep", [150.0, 200.0, 1000.0, 1200.0], ids=["below", "at-lo", "at-hi", "above"])
def test_a_timestep_outside_the_guidance_interval_does_not_blend(monkeypatch, timestep):
    """Outside the interval the base does not blend, so neither may this override.

    Such a rank gets ``cond_v`` alone from the base; the blend would hand it a guided velocity
    instead -- silently, and only on multi-rank runs. Both ends exclusive, as the base compares.
    """
    cond_v, uncond_v = _branches([-3.0] * _LENGTH)  # would visibly clamp if the blend ran
    out_c, out_u = _model(monkeypatch, cond_v, uncond_v, timestep=timestep)._run_classifier_free_guidance()

    assert out_c is cond_v and out_u is uncond_v


# --------------------------------------------------------------------------------------------------
# batched vision encode — _encode_vision_x0_tokens
# --------------------------------------------------------------------------------------------------


class _FakeEncoder:
    """Deterministic per-sample encode; cross-sample leakage would show up as a changed result."""

    _encode_vision_x0_tokens = AnomalyGenTextureMoTModel._encode_vision_x0_tokens

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(state.shape))
        return state.mean(dim=(-2, -1), keepdim=True) * 3.0 + 1.0


def _items(n: int = 4) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(1, 3, 1, 8, 8) for _ in range(n)]


def test_returns_one_latent_per_item_equal_to_that_items_encoding():
    model = _FakeEncoder()
    items = _items()

    out = model._encode_vision_x0_tokens(items, None, None, None)

    assert len(out) == len(items)
    for latent, item in zip(out, items):
        assert latent.shape[0] == 1, "each item's latent keeps its own leading batch dim"
        torch.testing.assert_close(latent, _FakeEncoder().encode(item).contiguous().float())


def test_uniform_training_items_are_encoded_in_a_single_call():
    model = _FakeEncoder()

    model._encode_vision_x0_tokens(_items(n=6), None, None, None)

    assert model.calls == [(6, 3, 1, 8, 8)], "all items stacked into one encode"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vision_condition_indexes": [[0], [0]]},  # inference: only a pixel prefix is encoded
        {"num_views_per_vision_item": [1, 1]},  # multiview: one camera at a time
    ],
)
def test_conditioned_and_multiview_inputs_are_not_batched(kwargs):
    assert not _group_is_encodable(
        _items(n=2), kwargs.get("vision_condition_indexes"), kwargs.get("num_views_per_vision_item")
    )


def test_only_uniform_shape_groups_are_encodable():
    assert _group_is_encodable(_items(n=8), None, None)
    # Mixed resolutions cannot share one call.
    assert not _group_is_encodable([torch.randn(1, 3, 1, 8, 8), torch.randn(1, 3, 1, 16, 16)], None, None)
    # A lone item is passed to the VAE unchanged, so it qualifies with nothing to agree with.
    assert _group_is_encodable(_items(n=1), None, None)
    assert not _group_is_encodable([], None, None)


def _bare_model(monkeypatch, *, balancing_available: bool):
    """``AnomalyGenTextureMoTModel`` without ``__init__``; the VAE and the lb-group probe are stubs."""
    model = object.__new__(AnomalyGenTextureMoTModel)
    calls: list[tuple[int, ...]] = []
    object.__setattr__(model, "encode", lambda state: calls.append(tuple(state.shape)) or state.float())
    monkeypatch.setattr(
        model, "_vision_encoder", lambda: SimpleNamespace(balancing_available=lambda: balancing_available)
    )
    return model, calls


def test_balanced_encode_defers_to_the_base_when_the_lb_group_can_take_it(monkeypatch):
    """The base's balancing runs collectives over the ``lb`` group, so when it would balance, every
    rank must go through it rather than through the local batched encode."""
    model, calls = _bare_model(monkeypatch, balancing_available=True)
    seen = {}

    def _base(self, raw, num_items, cond_idx, num_views=None, balance_vae_encode=False):
        seen["balance_vae_encode"] = balance_vae_encode
        return list(raw)

    monkeypatch.setattr(OmniMoTModel, "_encode_vision_x0_tokens", _base, raising=True)

    model._encode_vision_x0_tokens(_items(n=2), None, None, None, balance_vae_encode=True)

    assert seen == {"balance_vae_encode": True}
    assert calls == [], "nothing encoded locally"


def test_balanced_encode_still_batches_when_balancing_is_unavailable(monkeypatch):
    # Training always opts in; on a single rank (or before the tokenizer benchmark) the base would
    # loop item by item, so the batched path is still the one taken.
    model, calls = _bare_model(monkeypatch, balancing_available=False)

    out = model._encode_vision_x0_tokens(_items(n=3), None, None, None, balance_vae_encode=True)

    assert calls == [(3, 3, 1, 8, 8)]
    assert len(out) == 3


def test_ragged_items_are_grouped_by_shape_and_returned_in_input_order():
    """Variable-resolution batches are ragged. Grouping reorders the encodes, so the latents have to
    be scattered back to their own slots — otherwise a sample is trained against another's latent."""
    model = _FakeEncoder()
    torch.manual_seed(0)
    small, big = torch.randn(1, 3, 1, 8, 8), torch.randn(1, 3, 1, 16, 16)
    items = [small, big, small.clone()]  # interleaved, so the groups are not contiguous

    out = model._encode_vision_x0_tokens(items, None, None, None)

    assert sorted(model.calls) == [(1, 3, 1, 16, 16), (2, 3, 1, 8, 8)], "one encode per distinct shape"
    for latent, item in zip(out, items):
        torch.testing.assert_close(latent, _FakeEncoder().encode(item).contiguous().float())


def test_the_background_term_carries_no_gradient_at_weight_zero():
    """At the default weight the background term contributes exactly zero gradient, so it is
    detached — the value still logs, but the backward skips a second full-size masked sum."""
    pred, target = _constant_error_item(4.0)
    pred.requires_grad_(True)
    _, _, defect_mean, bg_mean = _call(edit_masks=[_mask(8)], pred=[pred], target=[target])

    assert not bg_mean.requires_grad, "background term must not be in the graph at weight 0"
    assert defect_mean.requires_grad, "defect term still trains"
    assert float(bg_mean) > 0.0, "the value is still reported for logging"


def test_the_background_term_keeps_its_gradient_at_a_nonzero_weight():
    pred, target = _constant_error_item(4.0)
    pred.requires_grad_(True)
    _, _, _, bg_mean = _call(edit_masks=[_mask(8)], pred=[pred], target=[target], bg_loss_weight=0.2)
    assert bg_mean.requires_grad


def test_detaching_the_background_does_not_change_the_reported_values():
    """The detach is a graph change only; every logged number must be identical."""
    pred, target = _constant_error_item(4.0)
    total, per_item, defect_mean, bg_mean = _call(edit_masks=[_mask(8)], pred=[pred], target=[target])
    # weight 0 means the background never enters the total, detached or not.
    assert float(total) == pytest.approx(float(defect_mean))
    assert float(per_item[0]) == pytest.approx(float(defect_mean))
    # Numerator sums over C channels, denominator counts cells (see the loss docstring).
    assert float(bg_mean) == pytest.approx(C * 4.0)


# --------------------------------------------------------------------------------------------------
# _compute_losses — the base's module-level loss call is routed to the masked objective
# --------------------------------------------------------------------------------------------------


def _loss_model():
    """``AnomalyGenTextureMoTModel`` without ``__init__``, carrying one supervised item with a mask."""
    model = object.__new__(AnomalyGenTextureMoTModel)
    for key, value in dict(
        _anomaly_edit_masks=[_mask(8)],
        _loss_parts=None,
        _warned_loss_fallback=False,
        tensor_kwargs_fp32={"dtype": torch.float32, "device": torch.device("cpu")},
        config=SimpleNamespace(diffusion_expert_config=SimpleNamespace(inpaint_loss_background_weight=0.0)),
    ).items():
        object.__setattr__(model, key, value)
    return model


def _fake_base_losses(pred):
    """Stand-in for the base ``_compute_losses``: one vision term through the module-level name, the
    way the framework calls it, with no method in between."""
    from cosmos_framework.model.generator import omni_mot_model

    def _base(self, *args, **kwargs):
        loss, per_instance = omni_mot_model.compute_flow_matching_loss(
            pred=pred,
            target=[torch.zeros_like(p) for p in pred],
            condition_mask=[torch.zeros(T, 1, 1) for _ in pred],
            timesteps=torch.full((len(pred), 1), 0.5),
            has_valid_tokens=True,
            rectified_flow=_UniformRectifiedFlow(),
            tensor_kwargs_fp32=self.tensor_kwargs_fp32,
            normalize_by_active=False,
        )
        return loss, {"flow_matching_loss_vision": loss, "flow_matching_loss_vision_per_instance": per_instance}

    return _base


def test_the_vision_term_reaches_the_masked_loss_and_reports_the_split(monkeypatch):
    """The regression this guards: a base that stops calling the method makes the masked objective
    dead code silently — the run trains on the whole image and logs nothing different."""
    pred, _ = _constant_error_item(4.0)
    monkeypatch.setattr(OmniMoTModel, "_compute_losses", _fake_base_losses([pred]), raising=True)
    model = _loss_model()

    total, losses = model._compute_losses()

    assert "flow_matching_loss_vision_defect" in losses and "flow_matching_loss_vision_background" in losses
    # Masked 8 of 64 cells at error 4 -> defect mean is C*4 (see the loss docstring); whole-image
    # would be the same here, so pin the split keys rather than the scalar.
    assert float(losses["flow_matching_loss_vision_defect"]) == pytest.approx(C * 4.0)
    assert model._loss_parts is None, "consumed once, so a later modality cannot re-log it"


def test_non_vision_terms_fall_through_to_the_original_loss(monkeypatch):
    # Action/sound preds are 2-D; the router must hand them straight back to the framework's loss.
    pred = [torch.ones(4, 6)]  # [T, D]
    monkeypatch.setattr(OmniMoTModel, "_compute_losses", _fake_base_losses(pred), raising=True)
    model = _loss_model()
    object.__setattr__(model, "_anomaly_edit_masks", None)  # a batch with no vision items to supervise

    total, losses = model._compute_losses()

    assert "flow_matching_loss_vision_defect" not in losses
    assert torch.isfinite(total)


def test_the_module_binding_is_restored_after_the_call(monkeypatch):
    from cosmos_framework.model.generator import omni_mot_model

    before = omni_mot_model.compute_flow_matching_loss
    pred, _ = _constant_error_item(1.0)
    monkeypatch.setattr(OmniMoTModel, "_compute_losses", _fake_base_losses([pred]), raising=True)

    _loss_model()._compute_losses()

    assert omni_mot_model.compute_flow_matching_loss is before


def test_a_base_that_bypasses_the_module_name_is_caught_rather_than_trained_through(monkeypatch):
    """The failure the router cannot see on its own: a framework that imports the loss under another
    name (or goes back to a method) never enters ``routed``, and the run would silently train on the
    whole-image loss. With edit masks present, no vision term through the router must raise."""
    from cosmos_framework.model.generator.algorithm.loss import flow_matching

    pred, _ = _constant_error_item(1.0)

    def _base(self, *args, **kwargs):
        loss, per_instance = flow_matching.compute_flow_matching_loss(  # not the omni_mot_model binding
            pred=[pred],
            target=[torch.zeros_like(pred)],
            condition_mask=[torch.zeros(T, 1, 1)],
            timesteps=torch.full((1, 1), 0.5),
            has_valid_tokens=True,
            rectified_flow=_UniformRectifiedFlow(),
            tensor_kwargs_fp32=self.tensor_kwargs_fp32,
        )
        return loss, {"flow_matching_loss_vision": loss}

    monkeypatch.setattr(OmniMoTModel, "_compute_losses", _base, raising=True)

    with pytest.raises(RuntimeError, match="module-level name"):
        _loss_model()._compute_losses()
