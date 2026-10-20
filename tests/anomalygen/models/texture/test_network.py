# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CFG unconditional pass must disable every learned per-defect-type pathway.

If any one of them leaks into the unconditional branch, ``c - u`` collapses toward zero and guidance
goes silently inert — the outputs still look plausible, so nothing downstream flags it. Both gates
are covered here; each is driven with a stub ``self`` reading only the attributes set up here.

The last section covers the other half: *when* the pass is entered at all.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

from anomalygen.models.texture import network
from anomalygen.models.texture.model import AnomalyGenTextureMoTModel
from anomalygen.models.texture.network import AnomalyVFMNetwork

NULL_CONDITIONING = AnomalyVFMNetwork.null_conditioning


def _net(**overrides):
    """Stub network with the flag off and every learned pathway present."""
    base = dict(_uncond_pass=False, _anomaly_class_ids=[0], text_prompt_emb=object())
    base.update(overrides)
    return SimpleNamespace(**base)


def test_null_conditioning_restores_the_flag_even_on_an_exception():
    # The conditional branch runs after the unconditional one; a leaked True would disable the
    # adapters for the REST of sampling, not just the failed step.
    net = _net()
    with NULL_CONDITIONING(net):
        assert net._uncond_pass is True
    assert net._uncond_pass is False

    with contextlib.suppress(RuntimeError), NULL_CONDITIONING(net):
        raise RuntimeError("boom")
    assert net._uncond_pass is False


class _SpyPackedSeq:
    """Records whether anything past the ``_uncond_pass`` gate looked at the sequence."""

    def __init__(self):
        self.touched = False

    def __getattr__(self, name):
        # __getattr__ only fires for attributes not set in __init__, i.e. the ones the scatter reads.
        object.__setattr__(self, "touched", True)
        return None


def test_prompt_scatter_is_skipped_on_the_unconditional_pass():
    """The learnable text prompt routes off ``anomaly_class_id``, not the caption, so dropping the
    caption alone would leave it active."""
    fn = AnomalyVFMNetwork._apply_text_prompt

    uncond = _SpyPackedSeq()
    fn(_net(_uncond_pass=True), uncond, None)
    assert not uncond.touched, "the unconditional pass must return before reading the sequence"

    # Control: with the flag off the same stub IS read, so the assertion above pins the gate rather
    # than some other shared early return.
    cond = _SpyPackedSeq()
    fn(_net(_uncond_pass=False), cond, None)
    assert cond.touched


def test_per_class_lora_routing_is_cleared_on_the_unconditional_pass(monkeypatch):
    """The LoRA gate must CLEAR the routing, not merely skip it.

    Skipping would leave the previous (conditional) forward's ``gen_class_ids`` buffer in place, so
    the adapters would still fire on the unconditional branch and ``c - u`` would collapse.
    """
    seen = []
    monkeypatch.setattr(network, "has_per_class_lora", lambda self: True)
    monkeypatch.setattr(network, "set_gen_class_ids", lambda self, ids: seen.append(ids))
    monkeypatch.setattr(network, "build_gen_class_ids", lambda **kw: "routed")

    vision = SimpleNamespace(token_shapes=[(1, 1, 1)], sequence_indexes=[0])
    packed_seq = SimpleNamespace(vision=vision)
    routing = AnomalyVFMNetwork._set_per_class_routing

    routing(_net(_uncond_pass=True, _anomaly_items_per_sample=2), packed_seq, None)
    assert seen == [None], "the unconditional pass must clear the buffer, not leave it stale"

    # Control: the conditional pass still routes, so the assertion above is the gate and not a
    # blanket failure to build ids.
    routing(_net(_uncond_pass=False, _anomaly_items_per_sample=2), packed_seq, None)
    assert seen == [None, "routed"]


# --- entering the pass: the producer/consumer contract -------------------------------------------
# ``_get_velocity`` recognises the unconditional branch by ``is`` on the token list that
# ``_get_inference_text_tokens`` stashed. Every test above drives ``null_conditioning`` directly with
# the flag already set, so a break in that linkage leaves all of them green while the pass is never
# entered, the adapters stay on, and ``c - u`` collapses. Nothing else pins it.

_ITEM_SHAPE = (1, 1, 2, 2)


def _velocity_model(monkeypatch, seen):
    """Bare model wired for ``_get_velocity``: no nn.Module init, no real forward.

    ``object.__new__`` skips ``nn.Module.__init__``, so attributes go on with ``object.__setattr__``.
    The base ``_get_velocity`` is replaced by a recorder, since the branch decision is taken around
    the forward rather than inside it.
    """
    net = object.__new__(AnomalyVFMNetwork)
    object.__setattr__(net, "_uncond_pass", False)

    def _record(self, *, noise_x, **kwargs):
        # Read the flag DURING the call: null_conditioning restores it on exit, so checking
        # afterwards would always see False whether or not the pass was entered.
        seen.append((net._uncond_pass, [t.clone() for t in noise_x]))
        return [torch.zeros_like(t) for t in noise_x]

    monkeypatch.setattr(OmniMoTModel, "_get_velocity", _record, raising=True)

    model = object.__new__(AnomalyGenTextureMoTModel)
    for key, value in dict(
        net=net,
        _uncond_text_tokens=None,
        _anomaly_items_per_sample=2,  # [source, target]
        _anomaly_edit_masks=None,  # -> _target_regions returns None -> replacement trick is skipped
        _cfg_regions=None,  # not cached yet, as at the first forward of a generate call
        _warned_target_regions=False,  # real __init__ attribute; this stub bypasses __init__
        # ``_dec`` is a read-only property over config.diffusion_expert_config, so set it there.
        config=SimpleNamespace(diffusion_expert_config=SimpleNamespace(learnable_text_prompt_enabled=False)),
    ).items():
        object.__setattr__(model, key, value)
    return model


def _gen_data_clean():
    """One sample's [source, target] clean latents, distinct so a substitution is visible."""
    return SimpleNamespace(x0_tokens_vision=[torch.full(_ITEM_SHAPE, 1.0), torch.full(_ITEM_SHAPE, 9.0)])


def _drive(model, text_tokens):
    gen = _gen_data_clean()
    noise_x = [torch.zeros(2 * gen.x0_tokens_vision[0].numel())]
    model._get_velocity(
        net=None,
        noise_x=noise_x,
        timestep=torch.tensor([500.0]),
        text_tokens=text_tokens,
        sequence_plans=None,
        gen_data_clean=gen,
    )


def test_the_stashed_token_list_is_what_opens_the_unconditional_pass(monkeypatch):
    """End to end: the list ``_get_inference_text_tokens`` returns must be the one ``_get_velocity``
    recognises. Taking the list from the producer rather than assigning it by hand is the point —
    that is the linkage a refactor breaks."""
    seen = []
    model = _velocity_model(monkeypatch, seen)
    monkeypatch.setattr(
        OmniMoTModel,
        "_get_inference_text_tokens",
        lambda self, batch, has_neg, caption_groups=None: ([[1, 2]], [[3, 4]]),
        raising=True,
    )

    _cond, uncond = model._get_inference_text_tokens({}, True)
    _drive(model, uncond)

    uncond_pass, noise_x = seen[-1]
    assert uncond_pass, "the stashed unconditional token list must open the pass"
    # ... and the source slice was swapped to the clean target, so the branch is a plain
    # reconstruction rather than a noise-hole the disabled adapters cannot read.
    assert torch.equal(noise_x[0][:4], torch.full((4,), 9.0))


def test_an_equal_but_copied_token_list_does_not_open_the_pass(monkeypatch):
    """The regression this guards: any refactor that copies, slices or re-tokenizes the list on the
    way to ``_get_velocity`` leaves it EQUAL but not IDENTICAL, and the pass silently stops firing."""
    seen = []
    model = _velocity_model(monkeypatch, seen)
    monkeypatch.setattr(
        OmniMoTModel,
        "_get_inference_text_tokens",
        lambda self, batch, has_neg, caption_groups=None: ([[1, 2]], [[3, 4]]),
        raising=True,
    )

    _cond, uncond = model._get_inference_text_tokens({}, True)
    copied = list(uncond)
    assert copied == uncond and copied is not uncond  # the fixture really is the near-miss case

    _drive(model, copied)

    uncond_pass, noise_x = seen[-1]
    assert not uncond_pass
    assert torch.equal(noise_x[0], torch.zeros(8)), "no substitution on the conditional branch"


def test_the_conditional_token_list_does_not_open_the_pass(monkeypatch):
    # The ordinary case, so the two tests above pin the identity check rather than a body that
    # happens to enter the pass for everything or nothing.
    seen = []
    model = _velocity_model(monkeypatch, seen)
    monkeypatch.setattr(
        OmniMoTModel,
        "_get_inference_text_tokens",
        lambda self, batch, has_neg, caption_groups=None: ([[1, 2]], [[3, 4]]),
        raising=True,
    )

    cond, _uncond = model._get_inference_text_tokens({}, True)
    _drive(model, cond)

    assert not seen[-1][0]


def test_no_pass_is_entered_before_any_unconditional_list_is_stashed(monkeypatch):
    # ``_uncond_text_tokens`` starts None; a body that skipped the None check would treat the first
    # forward of a non-CFG run as unconditional and disable the adapters for it.
    seen = []
    model = _velocity_model(monkeypatch, seen)

    _drive(model, None)

    assert not seen[-1][0]
