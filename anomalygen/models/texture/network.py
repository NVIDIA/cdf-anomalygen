# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Cosmos3VFMNetwork`` extended with the per-defect-type conditioning pathways.

The learnable text prompt overwrites each sample's leading text positions, and the
per-defect-type LoRA routes gen tokens to their class's adapter. Class ids are supplied
per-forward via :meth:`set_anomaly_context` (read live, so requires ``compile.enabled=False``).
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    Cosmos3VFMNetworkConfig,
)
from torch import nn

from anomalygen.data.utils import CAPTION_TOKEN_BUDGET
from anomalygen.models.texture.per_class_lora import build_gen_class_ids, has_per_class_lora, set_gen_class_ids


class AnomalyVFMNetwork(Cosmos3VFMNetwork):
    def __init__(self, language_model, config: Cosmos3VFMNetworkConfig):
        super().__init__(language_model, config)

        num_classes = int(getattr(config, "anomaly_num_classes", 0) or 0)

        # Per-defect-type learnable text prompt: [num_classes, P, hidden]. The whole text-token
        # sequence (P positions) is learnable per defect type, initialized from the caption
        # template (see _maybe_init_text_prompt). Materialized empty here (meta device);
        # _init_text_prompt_post_materialization zeros it so all-zero == "not yet template-inited".
        self.text_prompt_emb: Optional[nn.Parameter] = None
        self.text_prompt_num_tokens: int = int(
            getattr(config, "text_prompt_num_tokens", CAPTION_TOKEN_BUDGET) or CAPTION_TOKEN_BUDGET
        )
        if getattr(config, "learnable_text_prompt_enabled", False) and num_classes > 0:
            self.text_prompt_emb = nn.Parameter(torch.zeros(num_classes, self.text_prompt_num_tokens, self.hidden_size))
        # Per-defect-type [num_classes, P] init token ids, stashed by the model (which owns the
        # tokenizer + anomaly types); consumed once for the lazy template init.
        self._text_prompt_init_ids: Optional[torch.Tensor] = None
        self._text_prompt_init_done: bool = False

        # Transient per-forward context.
        self._anomaly_class_ids: Optional[Sequence[int]] = None
        self._anomaly_items_per_sample: int = 1
        # True only inside null_conditioning(), i.e. for the CFG unconditional forward.
        self._uncond_pass: bool = False

    # ------------------------ CFG unconditional (null-class) pass ------------------------
    @contextlib.contextmanager
    def null_conditioning(self) -> Iterator[None]:
        """Disable every *learned* per-defect-type pathway for the duration of one forward.

        The CFG unconditional branch, driven by :meth:`AnomalyGenTextureMoTModel._get_velocity`.
        CFG blends ``v = u + g*(c - u)``, so ``u`` sets what guidance steers away from. The disabled
        pathways (the learnable text prompt, the per-defect-type LoRA) route off ``anomaly_class_id``
        rather than the caption, so dropping the caption alone would leave them active and collapse
        ``c - u`` toward zero, leaving guidance nothing to scale.

        The caller pairs this with a clean source latent and ``UNCONDITIONAL_CAPTION`` so the branch
        asks for an unchanged reconstruction. Mask and background stay shared by both branches, i.e.
        off the guided axis, so a high guidance weight cannot destroy the background.
        """
        prev = self._uncond_pass
        self._uncond_pass = True
        try:
            yield
        finally:
            self._uncond_pass = prev

    def set_anomaly_context(
        self,
        class_ids: Optional[Sequence[int]],
        items_per_sample: int,
    ) -> None:
        """Stash per-forward anomaly context."""
        self._anomaly_class_ids = class_ids
        self._anomaly_items_per_sample = int(items_per_sample)

    # ------------------------ Learnable text prompt ------------------------
    def set_text_prompt_init_ids(self, init_ids: Optional[torch.Tensor]) -> None:
        """Stash the per-defect-type [num_classes, P] template token ids for the lazy prompt init.

        Called by the model (which owns the tokenizer and anomaly types). The actual embedding
        lookup is deferred to the first forward, where ``embed_tokens`` is guaranteed loaded.
        """
        self._text_prompt_init_ids = init_ids

    def _init_text_prompt_post_materialization(self) -> None:
        """Zero ``text_prompt_emb`` after ``to_empty`` (which leaves uninitialised storage).

        All-zero is the sentinel for "not yet template-initialised": on a fresh run the base
        checkpoint skips this param so it stays zero and the lazy template init fires; on resume
        the trained (non-zero) values are loaded and the lazy init is skipped.
        """
        if self.text_prompt_emb is not None:
            with torch.no_grad():
                self.text_prompt_emb.zero_()

    def _maybe_init_text_prompt(self) -> None:
        """One-shot template init of ``text_prompt_emb`` from the stashed token ids.

        Runs on the first forward (so ``embed_tokens`` weights are loaded). Skips when the param
        was already populated from a resume checkpoint (non-zero) or no init ids were provided.
        """
        if self._text_prompt_init_done or self.text_prompt_emb is None:
            return
        self._text_prompt_init_done = True
        init_ids = self._text_prompt_init_ids
        self._text_prompt_init_ids = None
        if init_ids is None:
            return
        # Resume checkpoint already filled the prompt -> keep it, don't overwrite with the template.
        if torch.count_nonzero(self.text_prompt_emb) != 0:
            return
        with torch.no_grad():
            ids = init_ids.to(device=self.text_prompt_emb.device)
            emb = self.language_model.model.embed_tokens(ids)  # [num_classes, P, hidden]
            self.text_prompt_emb.copy_(emb.to(self.text_prompt_emb.dtype))

    def _apply_text_prompt(self, packed_seq, packed_sequence: torch.Tensor) -> None:
        """Overwrite each sample's first-P text positions with its class's learnable prompt.

        The packed text block for a sample is ``[caption(P), EOS, start_of_generation]`` (no BOS
        for this tokenizer), contiguous and in sample order, so reshaping ``text_indexes`` to
        ``[n_samples, block_len]`` and scattering ``text_prompt_emb[class_id]`` into the leading P
        rows is exact. Skipped outright on the CFG unconditional pass (``null_conditioning``). The
        block-length gate below only catches a caption shorter than P, so it does not stand in for
        that check whenever the negative prompt is longer.
        """
        if self.text_prompt_emb is None or self._anomaly_class_ids is None or self._uncond_pass:
            return
        idx = getattr(packed_seq, "text_indexes", None)
        if not isinstance(idx, torch.Tensor) or idx.numel() == 0:
            return

        class_ids = list(self._anomaly_class_ids)
        n_samples = len(class_ids)
        p = self.text_prompt_num_tokens
        total = int(idx.numel())
        if n_samples <= 0 or total % n_samples != 0:
            return
        block_len = total // n_samples
        if block_len < p:  # unconditional / non-padded captions: nothing to inject
            return

        idx2 = idx.view(n_samples, block_len)
        num_classes = int(self.text_prompt_emb.shape[0])
        for i, ci in enumerate(class_ids):
            if 0 <= int(ci) < num_classes:
                rows = idx2[i, :p]
                packed_sequence[rows] = self.text_prompt_emb[int(ci)].to(
                    dtype=packed_sequence.dtype, device=packed_sequence.device
                )

    def _encode_text(self, packed_seq):
        packed_sequence, target_dtype = super()._encode_text(packed_seq)
        self._maybe_init_text_prompt()
        self._apply_text_prompt(packed_seq, packed_sequence)
        return packed_sequence, target_dtype

    def _encode_vision(
        self,
        packed_seq,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> List[Tuple[int, int, int]] | None:
        original_latent_shapes = super()._encode_vision(packed_seq, packed_sequence, target_dtype)

        # Route each gen token to its class's LoRA adapter. Runs here (eager encode head), before the
        # compiled transformer blocks, so the class ids reach them as a tensor buffer — never a Python
        # attribute, which fullgraph=True would bake as a stale constant inside the compiled region.
        self._set_per_class_routing(packed_seq, packed_sequence)

        return original_latent_shapes

    def _set_per_class_routing(self, packed_seq, packed_sequence: torch.Tensor) -> None:
        """Refresh every :class:`PerClassLoraLinear`'s ``gen_class_ids`` buffer for this forward.

        Builds a per-gen-token class-id vector in ``full_only_seq`` order (identical at every gen
        layer) when the class ids and vision tokens are available; otherwise clears the buffer so the
        adapters fall back to the frozen base rather than reuse the previous batch's routing.
        """
        if not has_per_class_lora(self):
            return
        # CFG unconditional branch: leave the routing cleared, so the adapters fall back to the base.
        if self._uncond_pass:
            set_gen_class_ids(self, None)
            return
        vision = getattr(packed_seq, "vision", None)
        gen_class_ids = None
        if self._anomaly_class_ids is not None and vision is not None and vision.token_shapes is not None:
            gen_class_ids = build_gen_class_ids(
                packed_sequence=packed_sequence,
                token_shapes=vision.token_shapes,
                sequence_indexes=vision.sequence_indexes,
                class_ids=list(self._anomaly_class_ids),
                items_per_sample=int(self._anomaly_items_per_sample),
            )
        set_gen_class_ids(self, gen_class_ids)  # None -> clears the buffer (frozen base this forward)
