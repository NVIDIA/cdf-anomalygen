# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OmniMoTModel`` wired for I2I anomaly-inpainting generation."""

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from cosmos_framework.model.generator import omni_mot_model
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import distributed, log, misc
from cosmos_framework.utils.flags import DEVICE, Device
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from torch.distributed.fsdp import MixedPrecisionPolicy

from anomalygen.configs.texture.constants import (
    DEFAULT_GUIDANCE,
    DEFAULT_GUIDANCE_INTERVAL,
)
from anomalygen.data.crop_geometry import input_size_spec_from_config
from anomalygen.data.utils import (
    CAPTION_TOKEN_BUDGET,
    anomaly_class_ids,
    build_caption,
    pad_or_truncate,
    resolve_word_token_id,
)
from anomalygen.inference.inpaint import build_inpaint_batch_fn
from anomalygen.inference.iterative import run_iterative_inpaint_batch, split_mask_into_instances
from anomalygen.models.texture.network import AnomalyVFMNetwork
from anomalygen.models.texture.per_class_lora import (
    PerClassLoraLinear,
    init_per_class_lora_weights,
    inject_per_class_lora,
    per_class_lora_buffer_names,
    per_class_lora_target_modules,
)


def downsample_edit_mask(
    mask: torch.Tensor,
    t_lat: int,
    h_lat: int,
    w_lat: int,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Downsample a pixel-space binary mask to a ``[1, T_lat, H_lat, W_lat]`` latent mask.

    ``mode`` is ``"any"`` (max-pool any-touch) or ``"nearest"``.
    """
    m = mask
    if m.dim() == 2:  # [H, W]
        m_in = m.unsqueeze(0).unsqueeze(0).float()
    elif m.dim() == 3:  # [1, H, W]
        m_in = m.unsqueeze(0).float()
    elif m.dim() == 4:  # [1, 1, H, W]
        m_in = m.float()
    else:
        raise ValueError(f"Unexpected edit_mask shape: {tuple(m.shape)}")

    m_in = m_in.to(device=device)

    if mode == "any":
        h_pix, w_pix = m_in.shape[-2:]
        kh, kw = max(1, h_pix // h_lat), max(1, w_pix // w_lat)
        m_down = F.max_pool2d(m_in, kernel_size=(kh, kw), stride=(kh, kw))  # [1,1,H_lat,W_lat]
    else:
        m_down = F.interpolate(m_in, size=(h_lat, w_lat), mode="nearest")

    m_down = (m_down > 0.5).to(dtype=dtype)
    return m_down.squeeze(0).expand(1, t_lat, h_lat, w_lat).contiguous()


def to_4d_chtw(t: torch.Tensor) -> torch.Tensor:
    """Drop leading singleton dims so a ``[1,C,T,H,W]`` prediction becomes ``[C,T,H,W]``."""
    while t.dim() > 4 and t.shape[0] == 1:
        t = t[0]
    return t


def _detach_if_unweighted(term: torch.Tensor, weight: float) -> torch.Tensor:
    """A term with weight 0 contributes no gradient, so keep its value for logging but drop the
    backward over a second full-size masked sum per item per step."""
    return term.detach() if math.isclose(weight, 0.0) else term


def masked_flow_matching_loss(
    pred: List[torch.Tensor],
    target: List[torch.Tensor],
    condition_mask: List[torch.Tensor],
    edit_masks: Sequence[Optional[torch.Tensor]],
    timesteps: torch.Tensor,
    rectified_flow,
    tensor_kwargs_fp32: Dict,
    bg_loss_weight: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Region-split vision flow-matching loss, averaged over supervised (noised) items::

        defect_i = sum(sqerr * em * w)     / sum(em * nm)        w  = nm * tw
        bg_i     = sum(sqerr * (1-em) * w) / sum((1-em) * nm)    em = defect mask [1,T,H,W]
        loss_i   = defect_i + bg_loss_weight * bg_i              nm = noisy frames, tw = time weight

    Each term is a mean over its own region, so the loss is defect-area invariant. Denominators count
    latent *cells* while numerators sum over C channels, which puts ``defect_i`` at ``C * mean(sqerr)``
    over the mask — the scale recipe LRs are tuned for. Dividing by C too would be a pure ~48x
    rescale; it cancels from the defect/bg ratio.

    Empty regions clamp the denominator to 1 against a zero numerator: no NaN, no gradient.

    ``pred`` / ``target`` / ``edit_masks`` / ``timesteps`` are all per vision ITEM (the framework
    expands the per-sample schedule before the loss), so item ``i`` reads row ``i``.

    Returns ``(loss, per_item_loss, defect_mean, bg_mean)``, or ``None`` when no item qualifies (the
    caller's signal to fall back to the base loss). ``per_item_loss`` has one entry per ``pred``
    element, 0 where skipped, to stay aligned with ``sigma`` for per-instance telemetry.
    """
    if not edit_masks:
        return None

    n_items = len(pred)
    defect_terms: List[torch.Tensor] = []
    bg_terms: List[torch.Tensor] = []
    per_item: List[Optional[torch.Tensor]] = [None] * n_items

    for i in range(n_items):
        if i >= len(edit_masks) or edit_masks[i] is None:
            continue
        pred_i, target_i = to_4d_chtw(pred[i]), to_4d_chtw(target[i])
        if pred_i.dim() != 4:
            continue

        em_i = edit_masks[i].to(dtype=torch.float32, device=pred_i.device)  # [1,T,H,W]
        if em_i.shape[-3:] != pred_i.shape[-3:]:
            continue

        # Upcast before subtracting: differencing nearby bf16 values loses precision a later cast
        # cannot recover, and these are sums over ~200k elements.
        sqerr_i = (pred_i.to(torch.float32) - target_i.to(torch.float32)) ** 2  # [C,T,H,W]

        # noisy_mask: 1 on generated frames, 0 on clean/source frames; broadcast over [C,T,H,W].
        t_dim = sqerr_i.shape[-3]
        cm = condition_mask[i].to(dtype=torch.float32, device=pred_i.device).reshape(-1)
        if cm.numel() == t_dim:
            nm_i = (1.0 - cm).view(1, t_dim, 1, 1)
        else:
            nm_i = (1.0 - cm).mean().reshape(1, 1, 1, 1)

        # Skip pure conditioning items (source / prompt): both terms are structurally 0 there, and
        # averaging them in would halve the loss for the standard [source, target] pair.
        if float(nm_i.max()) <= 0.0:
            continue

        # Time weight w(sigma_t), as in the base loss. Identically 1.0 today (TrainTimeWeight only
        # supports "uniform") — wired through for when it isn't.
        ts_i = timesteps[i, :t_dim] if timesteps.dim() > 1 else timesteps[i]
        tw_i = rectified_flow.train_time_weight(ts_i, tensor_kwargs_fp32).to(torch.float32).reshape(-1)
        tw_i = tw_i.view(1, -1, 1, 1) if tw_i.numel() == t_dim else tw_i.mean().reshape(1, 1, 1, 1)
        w_i = nm_i * tw_i

        bgm_i = 1.0 - em_i
        defect_i = (sqerr_i * em_i * w_i).sum() / (em_i * nm_i).sum().clamp(min=1.0)
        defect_terms.append(defect_i)

        bg_i = (sqerr_i * bgm_i * w_i).sum() / (bgm_i * nm_i).sum().clamp(min=1.0)
        bg_i = _detach_if_unweighted(bg_i, bg_loss_weight)
        bg_terms.append(bg_i)
        per_item[i] = defect_i + bg_loss_weight * bg_i

    if not defect_terms:
        return None

    defect_mean = torch.stack(defect_terms).mean()
    bg_mean = torch.stack(bg_terms).mean()
    zero = torch.zeros((), dtype=defect_mean.dtype, device=defect_mean.device)
    per_item_loss = torch.stack([x if x is not None else zero for x in per_item])
    return defect_mean + bg_loss_weight * bg_mean, per_item_loss, defect_mean, bg_mean


def replacement_velocity(
    flat_velocity: torch.Tensor,
    flat_latent: torch.Tensor,
    guide_latent: torch.Tensor,
    keep_latent: torch.Tensor,
    offset: int,
    sigma: float,
) -> torch.Tensor:
    """Apply the inference replacement trick to one sample's flat velocity.

    Overwrites the background (non-mask) latent positions of the target vision item with the
    clean guide at each step, then converts back to velocity:
    ``x0 = x_t - sigma*v`` → ``x0_new = keep*guide + (1-keep)*x0`` → ``v_new = (x_t - x0_new)/sigma``,
    where ``keep = 1 - edit_mask`` (1 = anchor to guide). ``guide_latent`` and ``keep_latent`` are
    already in the latent's dtype and device (cast once per generate call, see :class:`_Region`);
    ``offset`` is the target item's start index within the flat ``[D]`` vectors; ``sigma`` is the
    noise level in ``[0, 1]``.
    """
    shape = tuple(guide_latent.shape)
    length = int(guide_latent.numel())
    lat_target = flat_latent[offset : offset + length].reshape(shape)
    vel_target = flat_velocity[offset : offset + length].reshape(shape)

    x0_pred = lat_target - sigma * vel_target
    x0_new = keep_latent * guide_latent + (1.0 - keep_latent) * x0_pred
    vel_new = (lat_target - x0_new) / sigma

    out = flat_velocity.clone()
    out[offset : offset + length] = vel_new.reshape(-1)
    return out


class _Region(NamedTuple):
    """One sample's target vision item as the x0 bound and the replacement trick read it every step,
    cast once per generate call to the working latent's dtype and device."""

    offset: int  # where the item starts in the sample's flat latent
    guide: torch.Tensor  # clean target latent, [C, T, H, W]
    keep: Optional[torch.Tensor]  # 1 - edit mask (1 = anchor to the guide); None when the sample has no mask
    lo: torch.Tensor  # the guide's range, the bounds the x0 clamp uses (0-d)
    hi: torch.Tensor


class AnomalyGenTextureMoTModel(OmniMoTModel):
    def __init__(self, config):
        super().__init__(config)
        # Transient per-batch latent edit masks, read by the loss and the replacement trick.
        # (Class ids also live on the network via set_anomaly_context.)
        self._anomaly_edit_masks: Optional[List[Optional[torch.Tensor]]] = None
        # Vision items per sample for the current batch: always 2 ([source, target]).
        self._anomaly_items_per_sample: int = 2
        # (defect_mean, bg_mean) from the last flow-matching loss; detached telemetry for logging.
        self._loss_parts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # Per-class input-size caps from the training data (INPUT_SIZE_CAP_HEADROOM): train.py sets them
        # from the dataset's mask scan, the checkpoint carries them beside its geometry stamp, and
        # generation attaches them at load. None = every class plans against the full ladder.
        self.class_ladders: Optional[Dict[str, List[int]]] = None
        # The current generate call's unconditional token list, stashed by
        # _get_inference_text_tokens so _get_velocity can identify the CFG unconditional branch.
        self._uncond_text_tokens: Optional[List[List[int]]] = None
        # CFG state of the current generate call, read back by the x0 bound in
        # _run_classifier_free_guidance and by the replacement trick. ``_cfg_regions`` caches each
        # sample's target-item constants once per call (empty when the batch carries no masks or
        # guides); ``_cfg_step`` is the current forward's timestep and noisy latents.
        self._cfg_guidance: float = 1.0
        self._cfg_regions: Optional[List[_Region]] = None
        self._cfg_step: Optional[Tuple[float, List[torch.Tensor]]] = None
        # (values the x0 clamp moved, values it saw) over the current generate call, summed on device.
        self._cfg_clamp_bound: Optional[Tuple[torch.Tensor, int]] = None
        # One-shot latches so a silent fallback is reported once instead of every step.
        self._warned_loss_fallback: bool = False
        self._warned_clean_source: bool = False
        self._warned_target_regions: bool = False
        try:
            self._device = next(self.net.parameters()).device
        except (StopIteration, AttributeError):
            self._device = getattr(self, "device", torch.device("cpu"))

    def set_up_parallelism(self) -> None:
        """Skip FSDP under DDP so each rank keeps the full, unsharded model.

        For any multi-GPU VFM run, ``ParallelDims.dp_enabled`` is always True, so the base
        ``set_up_parallelism`` builds a shard mesh and ``build_net`` applies ``fully_shard`` — the
        params become DTensors. The trainer then *also* wraps the model with DDP
        (``distributed_parallelism="ddp"``), and DDP's init-time param broadcast cannot operate on
        DTensors (``c10d.broadcast_`` / uneven-flatten errors). To run true DDP data-parallel
        (each rank holds the full model; grads all-reduced by the DDP wrapper), we must not shard:
        leaving ``parallel_dims=None`` makes ``build_net`` skip FSDP and behave like the single-GPU
        path (compile/activation-checkpointing still apply). The ``parallelism`` mode is set by the
        experiment recipe from the same ``ANOMALYGEN_PARALLELISM`` switch that picks the ddp/fsdp
        model group + trainer ``distributed_parallelism``, so the three stay consistent. Under
        ``parallelism="fsdp"``, defer to the base sharded setup.
        """
        if getattr(self._dec, "parallelism", "ddp") == "ddp":
            self.parallel_dims = None
            return
        super().set_up_parallelism()

    @torch.inference_mode()
    def validation_step(self, data_batch: Dict, iteration: int = 0):
        """Batched, multi-GPU validation generation. Accumulates onto the model; scoring happens in
        the ValidationKPI callback's ``on_validation_end``. No-op (harmless tuple) when no KPI pass
        is active — ``validation_generated_images_by_anomaly`` is initialized by the callback's
        on_validation_start, so its absence means the empty val loader / no ValidationKPI callback."""
        zero = torch.zeros((), device=self._device)
        if getattr(self, "validation_generated_images_by_anomaly", None) is None:
            return {}, zero
        self._generate_validation_batch(data_batch)
        return {}, zero

    def _generate_validation_batch(self, data_batch: Dict) -> None:
        """Generate one sharded batch of testcases and accumulate results onto ``self``.

        Reads per-record generation params from ``data_batch`` and the fixed generation config from
        ``self._dec`` (the input-size band; ``class_ids`` derived from
        ``anomaly_types``), builds the batched ``(crops, masks, names, seeds, plans) -> edited_crops``
        inpaint callable via :func:`build_inpaint_batch_fn`, batches generation across images at each
        iterative instance-depth, and appends to ``self.validation_generated_images_by_anomaly`` /
        ``self.validation_sample_indices_by_anomaly`` (the callback gathers + scores these in
        on_validation_end).
        """

        def _to_float(img):
            return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

        def _mask_to_float(img):
            return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

        dec = self._dec  # fixed generation config lives on the model config, not the callback
        class_ids = anomaly_class_ids(dec.anomaly_types)
        names: List[str] = list(data_batch["anomaly_type"])
        images = list(data_batch["image"])
        masks = list(data_batch["mask"])
        indices = [int(x) for x in data_batch["index"]]
        seeds = [int(s) for s in data_batch["seed"]]
        b = len(names)

        # One spec drives both the window sizing (iterative) and the resize (inpaint).
        input_spec = input_size_spec_from_config(dec, class_ladders=getattr(self, "class_ladders", None))

        # (guidance, num_steps) are homogeneous within the batch: _build_val_batch_indices groups on
        # both. The sampler shift is not a per-testcase knob: the inpaint path always takes the
        # recipe's own.
        guidance = float(data_batch["guidance"][0])
        num_steps = int(data_batch["num_steps"][0])
        max_inst = [int(x) for x in data_batch["iteration_generation_max_instance"]]
        # Normalized to a float by the inference dataset.
        crop_ratios = [float(x) for x in data_batch["crop_ratio"]]
        poisson = [bool(x) for x in data_batch["poisson_blend"]]

        # FSDP-consistent instance depth: the max instance count across ranks for THIS batch, so every
        # rank runs the same number of forwards (guarded on world_size to stay single-GPU-safe).
        local_depth = max((len(split_mask_into_instances(masks[i], max_k=max_inst[i])) for i in range(b)), default=0)
        num_depth = local_depth
        if distributed.get_world_size() > 1:
            t = torch.tensor([local_depth], device="cuda")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            num_depth = int(t.item())

        inpaint_fn = build_inpaint_batch_fn(
            self,
            class_ids=class_ids,
            num_steps=num_steps,
            guidance=guidance,
        )
        composites, artifacts = run_iterative_inpaint_batch(
            images,
            masks,
            names,
            inpaint_fn,
            num_depth=num_depth,
            seeds=seeds,
            crop_ratios=crop_ratios,
            poisson_blends=poisson,
            max_instances_list=max_inst,
            input_spec=input_spec,
        )

        gd = self.validation_generated_images_by_anomaly
        si = self.validation_sample_indices_by_anomaly
        for i in range(b):
            name = names[i]
            bucket = gd.setdefault(
                name,
                {
                    "reconstructed_image": [],
                    "original_image": [],
                    "original_mask": [],
                    "cropped_image": [],
                    "cropped_mask": [],
                    "annotated_image": [],
                    "mask_cropped_image": [],
                },
            )
            bucket["reconstructed_image"].append(_to_float(composites[i]))
            bucket["original_image"].append(_to_float(images[i]))
            bucket["original_mask"].append(_mask_to_float(masks[i]))
            bucket["cropped_image"].append([_to_float(a) for a in artifacts[i]["cropped_image"]])
            bucket["cropped_mask"].append([_mask_to_float(a) for a in artifacts[i]["cropped_mask"]])
            bucket["annotated_image"].append([_to_float(a) for a in artifacts[i]["annotated_image"]])
            bucket["mask_cropped_image"].append([_to_float(a) for a in artifacts[i]["mask_cropped_image"]])
            si.setdefault(name, []).append(indices[i])

    @property
    def _dec(self):
        return self.config.diffusion_expert_config

    def build_net(
        self,
        dtype: torch.dtype,
        *,
        mp_policy: MixedPrecisionPolicy | None = None,
        lora_enabled: bool | None = None,
    ) -> torch.nn.Module:
        """Build the anomaly VFM network, parallelize it, and materialize it on device.

        Mirrors ``OmniMoTModel.build_net``'s signature because ``set_up_model`` calls it:
        ``dtype`` is the storage ("master") dtype of the parameters, and ``mp_policy`` is the
        FSDP2 policy that casts them down for the forward/backward. ``lora_enabled`` overrides
        ``config.lora_enabled`` for the shared framework LoRA; per-defect-type LoRA is chosen
        from ``diffusion_expert_config`` and is unaffected by it.
        """
        if lora_enabled is None:
            lora_enabled = getattr(self.config, "lora_enabled", False)
        if mp_policy is not None and self.parallel_dims is None:
            # The policy is only ever installed by an FSDP wrap, and ``set_up_parallelism`` leaves
            # ``parallel_dims`` unset on the DDP path — so keeping the master dtype here would store
            # *and* compute in it. Collapse to the compute dtype instead, which is what this path
            # holds when nothing casts around it.
            dtype = mp_policy.param_dtype or dtype
            mp_policy = None
            log.info(f"AnomalyGenTextureMoTModel: no FSDP wrap on this path; parameters stored in {dtype}.")
        with torch.device("meta"):
            if self.vlm_config.model_instance is None:
                raise ValueError("Model instance should be specified")
            language_model = lazy_instantiate(self.vlm_config.model_instance)
            num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps

            network_config = Cosmos3VFMNetworkConfig(
                vlm_config=language_model.config,
                latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
                latent_downsample_factor=self.config.latent_downsample_factor,
                latent_channel_size=self.config.state_ch,
                max_latent_h=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_w=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_t=self.config.state_t,
                enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
                base_fps=self.config.diffusion_expert_config.base_fps,
                vision_gen=self.config.vision_gen,
                action_gen=self.config.action_gen,
                sound_gen=self.config.sound_gen,
                joint_attn_implementation=self.config.joint_attn_implementation,
                timestep_scale=1.0 / float(num_train_timesteps) * self.config.diffusion_expert_config.timestep_range,
                action_dim=self.config.max_action_dim,
                num_embodiment_domains=self.config.num_embodiment_domains,
                temporal_compression_factor_vision=self.tokenizer_vision_gen.temporal_compression_factor,
                natten_parameter_list=self.config.natten_parameter_list,
                video_temporal_causal=self.config.video_temporal_causal,
                sound_dim=self.config.sound_dim,
                sound_latent_fps=self.config.sound_latent_fps,
            )
            network_config._attn_implementation_internal = "eager"
            # Read by AnomalyVFMNetwork.__init__.
            network_config.anomaly_num_classes = int(getattr(self._dec, "anomaly_num_classes", 0) or 0)
            network_config.learnable_text_prompt_enabled = bool(
                getattr(self._dec, "learnable_text_prompt_enabled", False)
            )
            network_config.text_prompt_num_tokens = int(
                getattr(self._dec, "text_prompt_num_tokens", CAPTION_TOKEN_BUDGET) or CAPTION_TOKEN_BUDGET
            )

            net = AnomalyVFMNetwork(language_model=language_model, config=network_config)
            net.pad_for_cuda_graphs = self.config.compile.use_cuda_graphs

            # Per-defect-type LoRA and the shared framework LoRA adapt the same gen-attention
            # projections, so they are mutually exclusive; per-defect-type takes priority when enabled.
            if bool(getattr(self._dec, "per_class_lora_enabled", False)):
                if lora_enabled:
                    raise ValueError(
                        "per_class_lora_enabled and lora_enabled are both True: they target the same "
                        "gen-attention projections and cannot both be injected. Set lora_enabled=False "
                        "when using per-defect-type LoRA."
                    )
                targets = per_class_lora_target_modules()
                replaced = inject_per_class_lora(
                    net,
                    targets=targets,
                    num_classes=int(getattr(self._dec, "anomaly_num_classes", 0) or 0),
                    rank=int(getattr(self._dec, "per_class_lora_rank", 8)),
                    alpha=int(getattr(self._dec, "per_class_lora_alpha", 8)),
                )
                log.info(
                    f"AnomalyGenTextureMoTModel: injected per-defect-type LoRA into {replaced} modules ({targets})."
                )
            elif lora_enabled:
                net = self.add_lora(
                    net,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_target_modules=self.config.lora_target_modules,
                )

        self.install_attention_dispatch(net)

        net = parallelize_vfm_network(
            net,
            parallel_dims=self.parallel_dims,
            compile_config=self.config.compile,
            ac_config=self.config.activation_checkpointing,
            mp_policy=mp_policy,
        )

        with misc.timer("meta to cuda and broadcast model states"):
            net = net.to(dtype=dtype)
            net.to_empty(device=DEVICE)
            if DEVICE == Device.CUDA:
                net.init_weights(buffer_device=DEVICE)
                if bool(getattr(self._dec, "per_class_lora_enabled", False)):
                    init_per_class_lora_weights(net)  # lora_A ~ kaiming, lora_B = 0 (delta starts at 0)
                elif lora_enabled:
                    self._init_lora_weights_post_materialization(net)
                # Zero the learnable text prompt after to_empty (uninitialised storage); the
                # template values are filled lazily on the first forward (embed_tokens loaded).
                if getattr(net, "text_prompt_emb", None) is not None:
                    net._init_text_prompt_post_materialization()

        return net

    def set_up_model(self):
        super().set_up_model()
        # Learnable text prompt: re-enable grad and hand the per-defect-type template token ids to
        # the network for the one-shot lazy init on the first forward.
        prompt = getattr(self.net, "text_prompt_emb", None)
        if prompt is not None:
            prompt.requires_grad_(True)
            self.net.set_text_prompt_init_ids(self._build_text_prompt_init_ids())
            log.info(f"AnomalyGenTextureMoTModel: text_prompt_emb {tuple(prompt.shape)}; template init ids staged.")

        # Per-defect-type LoRA: our stacked adapters are named lora_A/lora_B, so the base LoRA-freeze
        # in super().set_up_model() leaves them trainable — re-assert grad defensively (as above).
        if bool(getattr(self._dec, "per_class_lora_enabled", False)):
            n_mod = 0
            for m in self.net.modules():
                if isinstance(m, PerClassLoraLinear):
                    m.lora_A.requires_grad_(True)
                    m.lora_B.requires_grad_(True)
                    n_mod += 1
            log.info(f"AnomalyGenTextureMoTModel: re-enabled grad on per-defect-type LoRA in {n_mod} modules.")

            # Exclude the per-batch routing buffers from DDP buffer sync. broadcast_buffers
            # (framework default) would broadcast gen_class_ids from rank 0 each forward, but its
            # length differs per rank, so the collective mismatches. DDP reads this attribute on the
            # module it wraps (this model) at construction time in trainer.train(), after set_up_model.
            ignore = list(getattr(self, "_ddp_params_and_buffers_to_ignore", []))
            ignore += per_class_lora_buffer_names(self)  # names relative to the model (the DDP module)
            self._ddp_params_and_buffers_to_ignore = ignore
            log.info(
                f"AnomalyGenTextureMoTModel: excluded {len(ignore)} per-defect-type routing buffers "
                "from DDP buffer sync (_ddp_params_and_buffers_to_ignore)."
            )

    def _build_text_prompt_init_ids(self) -> Optional[torch.Tensor]:
        """Build the per-defect-type ``[num_classes, P]`` template token ids for the prompt init.

        Mirrors the dataset exactly: each class's caption is tokenized and padded/truncated to
        ``text_prompt_num_tokens`` with the ``text_prompt_init_word`` token, so the template init
        equals the base model's embedding of the captions the dataset actually feeds.
        """
        anomaly_types = list(getattr(self._dec, "anomaly_types", []) or [])
        if not anomaly_types:
            log.warning("AnomalyGenTextureMoTModel: learnable text prompt enabled but anomaly_types is empty.")
            return None
        p = int(getattr(self._dec, "text_prompt_num_tokens", CAPTION_TOKEN_BUDGET) or CAPTION_TOKEN_BUDGET)
        pad_id = resolve_word_token_id(self.vlm_tokenizer, str(getattr(self._dec, "text_prompt_init_word", "anomaly")))
        rows: List[torch.Tensor] = []
        for texture, defect in anomaly_types:
            caption = build_caption(defect=defect, texture=texture)
            ids = tokenize_caption(caption, self.vlm_tokenizer, is_video=False, use_system_prompt=False)
            ids = pad_or_truncate(ids, p, pad_id)
            rows.append(torch.tensor(ids, dtype=torch.long))
        return torch.stack(rows)  # [num_classes, P]

    def _get_inference_text_tokens(self, data_batch: dict, has_negative_prompt: bool, caption_groups=None):
        """Pad the conditional caption tokens to ``P`` so inference matches training.

        Training emits exactly ``P`` caption tokens per sample; the framework's inference
        tokenizer does not pad, so without this the conditional text block would be shorter than
        the learnable prompt and the per-defect-type scatter would not fire. Only the conditional
        branch is padded; the unconditional one is left as tokenized. What keeps the trained prompt
        out of that branch is the ``null_conditioning`` gate, not its block length — the negative
        prompt is free to be longer than ``P``.

        Also stashes the unconditional token list so :meth:`_get_velocity` can tell the two CFG
        branches apart by object identity — the same test the framework's own
        ``_single_velocity_fn`` uses to pick its packed-sequence template.
        """
        cond_tokens, uncond_tokens = super()._get_inference_text_tokens(data_batch, has_negative_prompt, caption_groups)
        if getattr(self._dec, "learnable_text_prompt_enabled", False):
            p = int(getattr(self._dec, "text_prompt_num_tokens", CAPTION_TOKEN_BUDGET) or CAPTION_TOKEN_BUDGET)
            pad_id = resolve_word_token_id(
                self.vlm_tokenizer, str(getattr(self._dec, "text_prompt_init_word", "anomaly"))
            )
            cond_tokens = [pad_or_truncate(ids, p, pad_id) for ids in cond_tokens]
        self._uncond_text_tokens = uncond_tokens
        return cond_tokens, uncond_tokens

    def get_data_and_condition(
        self,
        data_batch: Dict[str, torch.Tensor],
        vision_condition_indexes: Optional[List[List[int]]] = None,
        retain_raw_state_vision: bool = True,
        balance_vae_encode: bool = False,
    ):
        # ``retain_raw_state_vision`` is accepted only so the base's keyword call sites bind.
        # It gates the per-camera multiview VAE path, which anomalygen never takes.
        gen_data_clean = super().get_data_and_condition(
            data_batch,
            vision_condition_indexes=vision_condition_indexes,
            retain_raw_state_vision=retain_raw_state_vision,
            balance_vae_encode=balance_vae_encode,
        )

        x0 = gen_data_clean.x0_tokens_vision  # flat list of [1, C, T_lat, H_lat, W_lat]
        num_vis = gen_data_clean.num_vision_items_per_sample
        class_ids = _harvest_class_ids(data_batch, gen_data_clean.batch_size)

        # Dataloader always emits [source, target]: 2 vision items per sample.
        if not (num_vis and set(num_vis) == {2}):
            raise ValueError(f"expected 2 vision items per sample, got num_vis={num_vis}")

        # Build a per-vision-item list of latent masks aligned 1:1 with x0_tokens_vision, computed
        # over the [source, target] items.
        latent_masks: Optional[List[Optional[torch.Tensor]]] = None
        pixel_masks = _per_item_pixel_masks(data_batch.get("edit_mask"), num_vis, len(x0))
        if pixel_masks is not None:
            mode = getattr(self._dec, "inpaint_latent_mask_mode", "any")
            latent_masks = []
            for mask, latent in zip(pixel_masks, x0):
                if mask is None:
                    latent_masks.append(None)
                    continue

                _, _, t_lat, h_lat, w_lat = latent.shape
                latent_masks.append(downsample_edit_mask(mask, t_lat, h_lat, w_lat, mode, latent.dtype, latent.device))

        items_per_sample = num_vis[0]
        self._anomaly_edit_masks = latent_masks
        self._anomaly_items_per_sample = items_per_sample

        # Pass to the network for vision encoding (read live each forward; needs compile off).
        net = getattr(self, "net", None)
        if isinstance(net, AnomalyVFMNetwork):
            net.set_anomaly_context(class_ids, items_per_sample)

        return gen_data_clean

    def _encode_vision_x0_tokens(
        self,
        raw_state_vision,
        num_vision_items_per_sample,
        vision_condition_indexes=None,
        num_views_per_vision_item=None,
        balance_vae_encode: bool = False,
    ):
        """Encode the vision items in one VAE call per distinct shape instead of one call per item.

        The base loops the tokenizer over each item, i.e. ``batch_size * 2`` batch-of-1 encodes.
        Batching is equivalent (the VAE is causal over T, independent over B) and saves a slice of the
        step's GPU time. Variable-resolution batches are ragged, so items are grouped by shape and
        each group gets its own encode call. Falls back to the base loop for the WHOLE batch as soon
        as any one group is not batchable; see :func:`_group_is_encodable`.

        ``balance_vae_encode`` is honoured by deferring to the base whenever it would actually
        balance: that path runs collectives over the ``lb`` group, so every rank must take it together.
        """
        if balance_vae_encode and self._vision_encoder().balancing_available():
            return super()._encode_vision_x0_tokens(
                raw_state_vision,
                num_vision_items_per_sample,
                vision_condition_indexes,
                num_views_per_vision_item,
                balance_vae_encode=True,
            )

        groups: Dict[Tuple, List[int]] = {}
        for i, item in enumerate(raw_state_vision):
            key = (tuple(item.shape), item.dtype, item.device) if isinstance(item, torch.Tensor) else (i,)
            groups.setdefault(key, []).append(i)
        grouped = [[raw_state_vision[i] for i in idx] for idx in groups.values()]

        # Every group must qualify on its own: the ruled-out cases are mostly batch-global, but the
        # 5-D/leading-1 shape test is per item, and concatenating 4-D items on dim 0 would join them
        # along channels instead.
        if not all(
            _group_is_encodable(items, vision_condition_indexes, num_views_per_vision_item) for items in grouped
        ):
            return super()._encode_vision_x0_tokens(
                raw_state_vision,
                num_vision_items_per_sample,
                vision_condition_indexes,
                num_views_per_vision_item,
            )

        out: List[Optional[torch.Tensor]] = [None] * len(raw_state_vision)
        for idx, items in zip(groups.values(), grouped):
            stacked = torch.cat(items, dim=0) if len(items) > 1 else items[0]  # [N,C,T,H,W]
            latents = self.encode(stacked).contiguous().float()  # [N,C_lat,T_lat,H_lat,W_lat]
            # Back to the base's one-[1,C,...]-per-item contract; these slices are views, not copies.
            for slot, i in enumerate(idx):
                out[i] = latents[slot : slot + 1]
        return out

    def _compute_flow_matching_loss(
        self, pred, target, condition_mask, timesteps, has_valid_tokens, rectified_flow, **kwargs
    ):
        """Replace the base whole-image loss with :func:`masked_flow_matching_loss`.

        At the default ``inpaint_loss_background_weight = 0.0`` the background is not in the
        objective, so a falling loss says nothing about background reconstruction — read
        ``flow_matching_loss_vision_background`` for that.

        Falls back to the base loss when the masked path does not apply: no edit masks (action/sound
        preds are 2-D), no supervised item qualified, or ``has_valid_tokens`` is False. The last one
        is load-bearing — the base emits a dummy loss wired to every pred that keeps the backward
        graph consistent across ranks.

        ``per_instance`` becomes the per-item objective value (0 at conditioning items).
        ``normalize_by_active`` is moot: the region means already normalize by active elements.
        """
        masked = None
        applies = (
            self._anomaly_edit_masks is not None
            and has_valid_tokens
            and rectified_flow is not None
            and len(pred) > 0
            and pred[0].dim() in (4, 5)  # vision modality only (action/sound are 2-D); 5-D = [1,C,T,H,W]
        )
        if applies:
            # `pred`/`target`/`condition_mask` are one-per-vision-item, and `_anomaly_edit_masks` is
            # the aligned per-item mask list (None where an item has no edit mask). Clean items are
            # skipped inside, so passing the full list is correct.
            masked = masked_flow_matching_loss(
                pred=pred,
                target=target,
                condition_mask=condition_mask,
                edit_masks=self._anomaly_edit_masks,
                timesteps=timesteps,
                rectified_flow=rectified_flow,
                tensor_kwargs_fp32=self.tensor_kwargs_fp32,
                bg_loss_weight=float(getattr(self._dec, "inpaint_loss_background_weight", 0.0)),
            )

        if masked is None:
            # Silently swaps the defect-region objective for the base whole-image one.
            if applies and not self._warned_loss_fallback:
                self._warned_loss_fallback = True
                log.warning(
                    "AnomalyGenTextureMoTModel: edit masks did not line up with the batch's vision "
                    "items; falling back to the base whole-image flow-matching loss."
                )
            # The module-level function, not the base method: the base's is what _compute_losses
            # re-routes here, so going through it would loop.
            return compute_flow_matching_loss(
                pred=pred,
                target=target,
                condition_mask=condition_mask,
                timesteps=timesteps,
                has_valid_tokens=has_valid_tokens,
                rectified_flow=rectified_flow,
                tensor_kwargs_fp32=self.tensor_kwargs_fp32,
                **kwargs,
            )

        loss, per_instance, defect_mean, bg_mean = masked
        self._loss_parts = (defect_mean.detach(), bg_mean.detach())  # read back by _compute_losses
        return loss, per_instance

    def _compute_losses(self, *args, **kwargs):
        """Base losses with every modality's loss routed through :meth:`_compute_flow_matching_loss`,
        plus the defect / background split. At ``bg_loss_weight = 0.0`` the background term carries
        no gradient, so this is the only way to see background drift during a run.

        The base calls the loss function through its module namespace rather than a method, so the
        binding is swapped for the duration of the call; the router hands the vision term to the
        masked loss and everything else back to the original. The swap only works while the base
        looks the name up at call time, so a batch that carries edit masks must be seen to pass a
        vision-shaped prediction through the router — otherwise the masked objective has silently
        become dead code, and that is raised rather than logged.
        """
        seen_vision = False

        def routed(pred, target, condition_mask, timesteps, has_valid_tokens, rectified_flow, tensor_kwargs_fp32, **kw):
            nonlocal seen_vision
            if len(pred) > 0 and pred[0].dim() in (4, 5):
                seen_vision = True
            return self._compute_flow_matching_loss(
                pred, target, condition_mask, timesteps, has_valid_tokens, rectified_flow, **kw
            )

        original = omni_mot_model.compute_flow_matching_loss
        omni_mot_model.compute_flow_matching_loss = routed
        try:
            total_loss, losses_dict = super()._compute_losses(*args, **kwargs)
        finally:
            omni_mot_model.compute_flow_matching_loss = original
        if self._anomaly_edit_masks is not None and not seen_vision:
            raise RuntimeError(
                "OmniMoTModel._compute_losses no longer reaches compute_flow_matching_loss through its "
                "module-level name; the masked flow-matching loss would be dead code."
            )
        # Consumed here, not reset per modality: the base calls the loss once per modality, and a
        # later action/sound call would otherwise clear the vision split before it is read.
        parts, self._loss_parts = self._loss_parts, None
        if parts is not None:
            losses_dict["flow_matching_loss_vision_defect"] = parts[0]
            losses_dict["flow_matching_loss_vision_background"] = parts[1]
        return total_loss, losses_dict

    @torch.no_grad()
    def generate_samples_from_batch(self, *args, guidance: float = DEFAULT_GUIDANCE, **kwargs):
        """Pin the CFG policy: guidance over :data:`DEFAULT_GUIDANCE_INTERVAL`, no norm rescale, and
        the x0 bound in :meth:`_run_classifier_free_guidance`.

        Set here rather than per call site so standalone generation and in-training validation
        cannot drift apart. The interval and the rescale are assigned outright, not defaulted: the
        bound below gates on the same constants, so a caller's own values could not be honoured.
        """
        self._cfg_guidance = float(guidance)
        self._cfg_regions = None
        self._cfg_step = None
        self._cfg_clamp_bound = None
        kwargs["guidance_interval"] = list(DEFAULT_GUIDANCE_INTERVAL)
        kwargs["normalize_cfg"] = False
        try:
            out = super().generate_samples_from_batch(*args, guidance=guidance, **kwargs)
        finally:
            # These hold the batch's latents. Clearing on the way out rather than on the next call
            # keeps a validation batch off the GPU for the training steps that follow it.
            self._cfg_regions = None
            self._cfg_step = None
        self._log_clamp_rate()
        return out

    def _log_clamp_rate(self) -> None:
        """Report how much of the guided x0 the clamp moved over the call just finished: one host sync
        per generate call, so a clamp that binds often is visible without slowing the steps."""
        if self._cfg_clamp_bound is None:
            return
        bound, total = self._cfg_clamp_bound
        self._cfg_clamp_bound = None
        log.info(
            f"AnomalyGenTextureMoTModel: guided-x0 clamp bound {int(bound) / max(total, 1):.2%} of target-latent values."
        )

    def _run_classifier_free_guidance(self, **kwargs):
        """Blend here so the guided x0 can be clamped to the clean latent's range.

        Unbounded extrapolation occasionally drives the target latent outside the range the VAE
        decodes sanely, which surfaces as a saturated or blacked-out patch filling the mask. Clamping
        the blended x0 to the range of that sample's clean target latent leaves everything in range
        untouched and rules the failure out. Handing the finished velocity back as BOTH branches
        makes the base's own blend an identity, so the result reaches the sampler unchanged.

        Only blend on a rank the base would blend for. When a peer needs CFG and this one does not,
        the base returns ``cond_v`` alone -- the uncond forward ran only to keep the FSDP allgather
        aligned -- so handing it the blend would give it a guided velocity instead.

        Not reached when a ``velocity_postprocess`` hook is passed: that branch calls the single
        velocity fn directly, so the clamp does not apply there. Nothing in-tree passes one.
        """
        cond_v, uncond_v = super()._run_classifier_free_guidance(**kwargs)
        if math.isclose(self._cfg_guidance, 1.0) or self._cfg_step is None or not self._cfg_regions:
            return cond_v, uncond_v
        timestep, noise_x = self._cfg_step

        # The base's own gate, strict on both ends as it is there. Ranks disagree whenever their
        # timestep grids differ, which is exactly when the base returns cond_v unblended.
        t_lo, t_hi = DEFAULT_GUIDANCE_INTERVAL
        if not t_lo < timestep < t_hi:
            return cond_v, uncond_v

        sigma = timestep / float(self.config.rectified_flow_inference_config.num_train_timesteps)
        if sigma <= 1e-6:
            return cond_v, uncond_v

        blended = []
        bound, total = self._cfg_clamp_bound or (0, 0)
        for i, (c_i, u_i) in enumerate(zip(cond_v, uncond_v)):
            v = u_i + self._cfg_guidance * (c_i - u_i)
            region = self._cfg_regions[i]
            start, stop = region.offset, region.offset + int(region.guide.numel())
            x_t = noise_x[i][start:stop]
            x0_free = x_t - sigma * v[start:stop]
            x0 = x0_free.clamp(min=region.lo, max=region.hi)
            bound = bound + (x0 != x0_free).sum()
            total += stop - start
            # In-place is safe: the blend above allocated v, and nothing else holds it.
            v[start:stop] = (x_t - x0) / sigma
            blended.append(v)
        self._cfg_clamp_bound = (bound, total)
        return blended, blended

    def _target_regions(self, gen_data_clean, n_samples: int):
        """Per sample: ``(offset, guide, latent_mask)`` for the target vision item, or ``None``.

        ``offset`` is where that item starts in the sample's flat latent; the mask is ``None`` when
        the sample has none.
        """
        masks = self._anomaly_edit_masks
        guides = getattr(gen_data_clean, "x0_tokens_vision", None)
        if masks is None or guides is None:
            # Disables the x0 clamp AND the replacement trick, so say so rather than degrade quietly.
            self._warn_target_regions("edit masks" if masks is None else "clean guide latents")
            return None
        ips = int(self._anomaly_items_per_sample)
        if not (len(guides) == len(masks) == ips * n_samples):
            raise ValueError(f"expected {ips} vision items per sample, got {len(guides)} for {n_samples} samples")
        regions = []
        for i in range(n_samples):
            base = ips * i
            items = guides[base : base + ips]
            offset = int(sum(int(it.numel()) for it in items[:-1]))
            regions.append((offset, items[-1], masks[base + ips - 1]))
        return regions

    def _warn_clean_source(self, reason: str) -> None:
        """Report a skipped source substitution, once per model."""
        if self._warned_clean_source:
            return
        self._warned_clean_source = True
        log.warning(
            f"AnomalyGenTextureMoTModel: left the CFG unconditional source unpatched ({reason}); "
            "guidance is steering away from a noise-filled source rather than a reconstruction."
        )

    def _warn_target_regions(self, missing: str) -> None:
        """Report a skipped target-region lookup, once per model."""
        if self._warned_target_regions:
            return
        self._warned_target_regions = True
        log.warning(
            f"AnomalyGenTextureMoTModel: no target regions ({missing} unavailable); the guided-x0 "
            "clamp and the replacement trick are both inactive for this run."
        )

    def _clean_source(self, noise_x, gen_data_clean):
        """Return ``noise_x`` with each sample's source item replaced by the clean target latent.

        The source item normally carries the defect region as uniform noise, an encoding only the
        trained adapters can read; with them disabled the network cannot interpret it. Swapping in
        the clean latent turns the unconditional branch into a plain reconstruction, which the base
        network does natively.

        ``noise_x[i]`` is one flat tensor per sample holding its ``[source, target]`` items in order,
        so the source is the leading slice. It is a conditioning item (constant over the sampling
        loop), so writing to a copy affects only this forward and never the sampler state.

        Bailing out leaves the anchor reading a noise-filled source with the adapters off, so it warns.
        """
        x0 = gen_data_clean.x0_tokens_vision
        ips = int(self._anomaly_items_per_sample)
        if x0 is None or ips != 2 or len(x0) != ips * len(noise_x):
            self._warn_clean_source("the batch is not one [source, target] pair per sample")
            return noise_x

        out = []
        for i, flat in enumerate(noise_x):
            source, target = x0[2 * i], x0[2 * i + 1]
            if source.shape != target.shape:  # nothing sensible to substitute
                self._warn_clean_source(f"source {tuple(source.shape)} != target {tuple(target.shape)}")
                return noise_x
            patched = flat.clone()
            patched[: source.numel()] = target.reshape(-1).to(dtype=flat.dtype, device=flat.device)
            out.append(patched)
        return out

    def _get_velocity(
        self,
        *,
        net=None,
        noise_x,
        timestep,
        text_tokens,
        sequence_plans,
        gen_data_clean,
        skip_text_tokens=False,
        **kwargs,
    ):
        # `**kwargs` passes through sampler arguments this override doesn't use (has_noisy_actions /
        # packed_sequence_template / memory); the base always calls by keyword.
        #
        # Identify the CFG unconditional branch by ``is`` on the token list (see
        # ``_get_inference_text_tokens``).
        target_net = net or self.net
        null_uncond = (
            self._uncond_text_tokens is not None
            and text_tokens is self._uncond_text_tokens
            and isinstance(target_net, AnomalyVFMNetwork)
        )
        forward_ctx = target_net.null_conditioning() if null_uncond else contextlib.nullcontext()
        with forward_ctx:
            vel = super()._get_velocity(
                net=net,
                noise_x=self._clean_source(noise_x, gen_data_clean) if null_uncond else noise_x,
                timestep=timestep,
                text_tokens=text_tokens,
                sequence_plans=sequence_plans,
                gen_data_clean=gen_data_clean,
                skip_text_tokens=skip_text_tokens,
                **kwargs,
            )
        # Flat vision items are [source, target] per sample; the target is the last item and carries
        # the defect mask. Its constants are cached on the first forward of a generate call: the
        # guides and masks do not change across steps, so neither do the casts and clamp bounds.
        if self._cfg_regions is None:
            self._cfg_regions = self._cache_regions(gen_data_clean, noise_x)
        timestep_value = float(timestep.flatten()[0].item())
        self._cfg_step = (timestep_value, noise_x)

        # The replacement trick below is deliberately OUTSIDE that context: it carries no learned
        # defect signal, and must apply identically to both branches or the CFG blend would mix two
        # different backgrounds.
        if not getattr(self._dec, "inpaint_replacement_trick", False) or not self._cfg_regions:
            return vel

        sigma = timestep_value / float(self.config.rectified_flow_inference_config.num_train_timesteps)
        if sigma <= 1e-6:
            return vel

        out: List[torch.Tensor] = []
        for i, region in enumerate(self._cfg_regions):
            if region.keep is None:
                out.append(vel[i])
                continue
            out.append(replacement_velocity(vel[i], noise_x[i], region.guide, region.keep, region.offset, sigma))
        return out

    def _cache_regions(self, gen_data_clean, noise_x: List[torch.Tensor]) -> List[_Region]:
        """Per-sample target-item constants for one generate call, cast once to the working latent's
        dtype and device. Empty when the batch carries no edit masks or clean guides (warned about
        once), which leaves the x0 clamp and the replacement trick off for the call."""
        regions = self._target_regions(gen_data_clean, len(noise_x))
        if regions is None:
            return []
        cached = []
        for (offset, guide, mask), flat in zip(regions, noise_x):
            guide = guide.to(dtype=flat.dtype, device=flat.device)
            keep = None if mask is None else (1.0 - mask).to(dtype=flat.dtype, device=flat.device)
            cached.append(_Region(offset, guide, keep, guide.min(), guide.max()))
        return cached


def _group_is_encodable(items, vision_condition_indexes, num_views_per_vision_item) -> bool:
    """Whether a shape group can go through ``self.encode`` — concatenated, or alone if it is one item.

    Each clause rules out a case the base loop handles differently. Asked once per shape group; the
    caller falls back to the base loop unless every group qualifies.
    """
    # Inference: the base may encode only a pixel prefix per item.
    if vision_condition_indexes is not None:
        return False
    # Multiview: the base encodes one camera at a time.
    if num_views_per_vision_item is not None:
        return False
    if not items:
        return False
    first = items[0]
    if not isinstance(first, torch.Tensor) or first.dim() != 5 or first.shape[0] != 1:
        return False
    # Ragged shapes or mixed dtype/device cannot share one call; vacuous for a lone item.
    return all(
        isinstance(item, torch.Tensor)
        and item.shape == first.shape
        and item.dtype == first.dtype
        and item.device == first.device
        for item in items
    )


def _harvest_class_ids(data_batch: Dict, batch_size: int) -> List[int]:
    """Pull per-sample anomaly class ids from the batch; all-zeros when absent."""
    raw = data_batch.get("anomaly_class_id")
    if raw is None:
        return [0] * int(batch_size)
    if isinstance(raw, torch.Tensor):
        return [int(x) for x in raw.flatten().tolist()]
    if isinstance(raw, (list, tuple)):
        out: List[int] = []
        for x in raw:
            out.append(int(x.flatten()[0].item()) if isinstance(x, torch.Tensor) else int(x))
        return out
    return [int(raw)] * int(batch_size)


def _squeeze_mask(m: torch.Tensor) -> torch.Tensor:
    """Reduce a mask to ``[1, H, W]`` from ``[1,1,H,W]`` / ``[1,H,W]`` / ``[H,W]``."""
    if m.dim() == 4:
        return m[0]
    if m.dim() == 2:
        return m.unsqueeze(0)
    return m


def _per_item_pixel_masks(edit, num_vis: Optional[List[int]], x0_len: int) -> Optional[List[Optional[torch.Tensor]]]:
    """Expand edit masks into a flat per-vision-item list of length ``x0_len`` (or ``None``)."""
    if edit is None:
        return None

    per_sample: List[torch.Tensor] = []
    if isinstance(edit, (list, tuple)):
        per_sample = [_squeeze_mask(m) for m in edit if isinstance(m, torch.Tensor)]
    elif isinstance(edit, torch.Tensor):
        if edit.dim() == 4:
            per_sample = [edit[b] for b in range(edit.shape[0])]
        else:
            per_sample = [_squeeze_mask(edit)]
    if not per_sample:
        return None

    if num_vis:
        flat: List[Optional[torch.Tensor]] = []
        for b, n in enumerate(num_vis):
            mb = per_sample[b] if b < len(per_sample) else None
            flat.extend([mb] * int(n))
    else:
        flat = list(per_sample)

    if len(flat) != x0_len:
        if len(per_sample) == 1:
            flat = [per_sample[0]] * x0_len  # broadcast single mask to all items
        else:
            return None
    return flat
