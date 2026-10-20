# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anomaly captioner backed by the Cosmos3-Nano reasoner VLM (via HuggingFace ``transformers``).

Given the (mask, clean, generated) image triple plus per-image metadata, the captioner produces a
structured anomaly caption. It runs ``nvidia/Cosmos3-Nano``
(``Cosmos3OmniForConditionalGeneration``) and shares two conventions with
:class:`anomalygen.auto_mask_placement.text2roi.Text2BoxDetector`: generation is seeded per call, and
a local ``checkpoints/<org>/<name>`` copy is preferred over the HF hub. It deliberately diverges on
decoding — captions decode greedily, whereas ``Text2BoxDetector`` keeps its model's sampling
distribution because greedy degenerates to a full-image box on that task. That precedent is a
different model on a different task (Qwen3-VL detection); captioning was measured against it before
greedy became the default here and does not reproduce it, so re-check if the captioning model
changes. Unlike the original SDG captioner this runs on ``transformers`` (no vLLM) and processes one
image triple at a time.
"""

from __future__ import annotations

import json
import typing
from datetime import datetime
from pathlib import Path

import torch
import transformers
from cosmos_framework.utils import log
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration, LogitsProcessor, LogitsProcessorList

from anomalygen.models import device as device_placement

# The captioner runs the Cosmos3-Nano reasoner exclusively.
MODEL_ID = "nvidia/Cosmos3-Nano"

# Default captioning prompt shipped with the package (system + user prompt with {placeholders}).
DEFAULT_CAPTION_PROMPT_PATH = Path(__file__).parent / "default_caption_prompt.yaml"

# Room a single GPU must have beyond the weights: the KV cache of a 4096-token caption, the vision
# tower's activations and allocator slack. A quarter of the weights covers all three with margin.
_PLACEMENT_HEADROOM = 0.25


class _UsableLogitsGuard(LogitsProcessor):
    """Fail loudly when a decoding step leaves no token that can meaningfully be picked.

    Greedy decoding does not need a well-formed probability distribution: ``argmax`` over a tensor
    containing ``nan`` returns *some* index and generation continues, so a numerically broken step
    becomes a quietly wrong caption rather than an error. Sampling has the opposite failure — the
    same condition trips a device-side assert inside ``torch.multinomial``, which aborts with a
    stack trace pointing at the sampler rather than at whatever produced the values, and poisons the
    CUDA context on the way out.

    Neither is useful for pseudo-label data, where a wrong caption is worse than a missing one. This
    raises at the first bad step instead, naming the sample and the step so the failure is
    actionable, and it is installed on both decoding paths for that reason.

    What counts as bad is deliberately narrower than "not finite". ``-inf`` is the standard sentinel
    for a *masked* token, written on purpose by stock processors — top-k/top-p, ``suppress_tokens``,
    ``min_length``, ``forced_eos_token_id`` (which masks everything but one token). Rejecting it
    outright would only be safe while this guard runs ahead of those, and ``transformers`` does not
    commit to that ordering. So only ``nan`` and ``+inf``, which are never legitimate, are treated
    as defects on their own; a ``-inf`` is one only when a whole row is masked, leaving no candidate
    at all. That is the case which actually breaks ``argmax`` and ``torch.multinomial``, and it stays
    fatal wherever the guard ends up in the chain.
    """

    def __init__(self, context: str = ""):
        self.context = context
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        self.step += 1
        broken = torch.isnan(scores) | (scores == float("inf"))
        exhausted = torch.isneginf(scores).all(dim=-1)
        if bool(broken.any()) or bool(exhausted.any()):
            n_nan = int(torch.isnan(scores).sum())
            n_posinf = int((scores == float("inf")).sum())
            n_exhausted = int(exhausted.sum())
            raise RuntimeError(
                f"model produced unusable logits at decode step {self.step} "
                f"({n_nan} nan and {n_posinf} +inf of {scores.numel()} scores; "
                f"{n_exhausted} of {exhausted.numel()} rows fully masked)"
                + (f" while captioning {self.context}" if self.context else "")
                + ". The caption is not recoverable from this state, so no output was written. "
                "Re-run this sample in isolation to confirm it is reproducible. If the load log shows "
                "the model sharded across GPUs, re-run with a single GPU visible: a failure that "
                "disappears then, or that repeats with identical counts for every GPU pair, is the "
                "host corrupting device-to-device copies, not the model or the input."
            )
        return scores


def _resolve_model_path(model_id: str) -> str:
    """Prefer a local ``checkpoints/<org>/<name>`` copy (from scripts/download_checkpoints.sh) over the
    HF hub. Falls back to the given ``model_id`` (repo id or explicit path) when no local copy exists."""
    if "/" in model_id and not Path(model_id).exists():
        local = Path(__file__).resolve().parents[2] / "checkpoints" / model_id
        if (local / "config.json").exists():
            return str(local)
    return model_id


def _checkpoint_bytes(model_path: str) -> typing.Optional[int]:
    """Size of the weights on disk, or ``None`` when it cannot be known before the load (a hub id)."""
    root = Path(model_path)
    if not root.is_dir():
        return None
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            total = json.loads(index.read_text()).get("metadata", {}).get("total_size")
        except (OSError, ValueError):
            total = None
        if total:
            return int(total)
    return sum(f.stat().st_size for f in root.rglob("*.safetensors")) or None


def _device_map(weight_bytes: typing.Optional[int], free_bytes: typing.Sequence[int]) -> typing.Union[str, dict]:
    """One GPU when any has room for the weights plus headroom, otherwise accelerate's ``auto`` split.

    Sharding a model that fits is all cost: every shard boundary becomes a device-to-device copy of
    the hidden state, and a host that corrupts or drops those copies feeds the next shard a wrong
    hidden state that is still finite, so the damage surfaces only as a few ``nan``/``+inf`` logits.
    Placement therefore takes the GPU with the most free memory (lowest index on a tie) and splits
    only when no single one is big enough. ``auto`` is also the answer when the size is unknown or
    there is no GPU at all, which is the CPU path.
    """
    if weight_bytes is None or not free_bytes:
        return "auto"
    best = device_placement.roomiest(free_bytes)
    if free_bytes[best] < weight_bytes * (1 + _PLACEMENT_HEADROOM):
        return "auto"
    return {"": f"cuda:{best}"}


class Captioner:
    """Generate anomaly captions with the Cosmos3-Nano reasoner.

    The model is loaded lazily on the first :meth:`generate_caption` call (or explicit :meth:`load`),
    so constructing a ``Captioner`` is cheap and does not require the checkpoint until inference.
    """

    def __init__(
        self,
        prompt_data: typing.Dict[str, str],
        temperature: typing.Optional[float] = 0.0,
        max_new_tokens: int = 4096,
        seed: int = 42,
    ):
        """Initialize the captioner with the prompt config and generation settings.

        Args:
            prompt_data: Dict with ``system_prompt`` and ``user_prompt`` (may contain ``{placeholders}``).
            temperature: Sampling temperature. ``0`` (default) decodes greedily; ``> 0`` samples at
                that temperature; ``None`` defers to the model's own ``generation_config``. Greedy is the
                default for robustness.
            max_new_tokens: Maximum number of tokens to generate. Defaults to ``4096``.
            seed: RNG seed pinned before each generation for reproducibility. Defaults to ``42``.
        """
        self.system_prompt = prompt_data["system_prompt"]
        self.user_prompt = prompt_data["user_prompt"]
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load the Cosmos3-Nano processor and model (idempotent).

        Placement is one GPU when one has room (see :func:`_device_map`), else ``device_map="auto"``:
        a split across the CUDA devices when present, otherwise CPU. The decision, the resulting
        device map and the library versions are logged, since they are what a numerically broken
        decode has to be triaged against.
        """
        if self.model is not None:
            return

        # Prefer a local checkpoints/<org>/<name> copy if present; else load from the HF hub.
        model_path = _resolve_model_path(MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(model_path)
        free_bytes = device_placement.free_bytes() if torch.cuda.is_available() else []
        device_map = _device_map(_checkpoint_bytes(model_path), free_bytes)
        self.model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device_map,
        )
        # A single-device load leaves ``hf_device_map`` unset; the model's device is the placement then.
        placement = getattr(self.model, "hf_device_map", None) or str(self.model.device)
        log.info(
            f"Captioner loaded {model_path} with transformers {transformers.__version__} / torch "
            f"{torch.__version__}; requested device_map={device_map}, "
            f"placement={json.dumps(placement, sort_keys=True, default=str)}"
        )
        devices = set(placement.values()) if isinstance(placement, dict) else {placement}
        if len(devices) > 1:
            log.warning(
                f"Captioner is sharded across {len(devices)} devices, so every decode step moves the "
                "hidden state between GPUs. A host that corrupts those copies shows up as nan/+inf "
                "logits; re-run with a single GPU visible to tell that apart from a model problem."
            )

    def replace_placeholders(self, prompt: str, meta: dict) -> str:
        """Replace ``{key}`` placeholders in the prompt with values from ``meta``.

        Predefined placeholders are ``{image_type}``, ``{anomaly_type}``, ``{bboxes}``.
        """
        for key, value in meta.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt and value is not None:
                prompt = prompt.replace(placeholder, str(value))
        return prompt

    def postprocess_response(self, response: str, num_bboxes: int, context: str = "") -> str:
        """Trim the ``<answer>`` block to at most ``num_bboxes`` anomalies.

        Falling back to the raw response is deliberate -- a caption the model shaped differently is
        still better label text than a corrupted slice of one -- but it is worth saying out loud,
        because from here on nothing distinguishes it from a well-formed caption.
        """
        start_tag = "<answer>"
        end_tag = "</answer>"
        # str.find / rfind return -1 when the tag is absent (they never raise),
        # so guard explicitly instead of relying on a dead try/except. Without
        # this, a response lacking the tags produced a corrupted slice
        # (response[-1 + len(start_tag) : -1]) rather than falling back to raw.
        start_pos = response.find(start_tag)
        end_pos = response.rfind(end_tag)
        where = f" for {context}" if context else ""
        if start_pos == -1 or end_pos == -1 or end_pos < start_pos + len(start_tag):
            log.warning(f"Caption{where} has no usable <answer> block; keeping the raw response.")
            return response
        content = response[start_pos + len(start_tag) : end_pos].strip()
        # The delimiter is a blank line followed by the bolded "Anomaly " text.
        delimiter = "\n\n**Anomaly "
        parts = content.split(delimiter)
        # The first part is always the caption.
        caption = parts[0]
        # The rest of the parts are the anomalies.
        anomalies = parts[1:]
        if not anomalies:
            log.warning(f"Caption{where} has an <answer> block but no anomaly sections; keeping the raw response.")
            return response
        # Keep only the desired number of anomalies from the list.
        kept_anomalies = anomalies[:num_bboxes]
        rebuilt_parts = [caption]
        for anomaly in kept_anomalies:
            rebuilt_parts.append("**Anomaly " + anomaly)
        final_content = "\n\n".join(rebuilt_parts)
        return f"{start_tag}\n{final_content}\n{end_tag}"

    def _generate_raw(self, messages: typing.List[dict], context: str = "") -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Pin the RNG so an explicit temperature > 0 still decodes reproducibly per call. The greedy
        # default takes argmax and never draws from it.
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Temperature drives the decoding strategy: the default 0 disables sampling outright rather
        # than passing temperature=0 to generate(), which would divide the logits by zero. Only an
        # explicit None leaves gen_kwargs empty and defers to the model's own generation_config.
        gen_kwargs = {}
        if self.temperature is not None:
            gen_kwargs = (
                {"do_sample": True, "temperature": self.temperature} if self.temperature > 0.0 else {"do_sample": False}
            )
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            logits_processor=LogitsProcessorList([_UsableLogitsGuard(context)]),
            **gen_kwargs,
        )
        trimmed = output_ids[:, inputs.input_ids.shape[1] :]

        # ``generate`` returns for one of two reasons: the model emitted a stop token, or it ran into
        # ``max_new_tokens``. Only the first means the caption is whole, and nothing downstream
        # checks -- ``postprocess_response`` trims the ``<answer>`` block but hands an untagged
        # string straight back, so a decode that never terminated would reach the label file looking
        # like a success. Length alone is not the signal: a caption may legitimately end on the last
        # allowed token, so the stop token is what separates the two.
        stop_ids = getattr(self.model.generation_config, "eos_token_id", None)
        stop_ids = {stop_ids} if isinstance(stop_ids, int) else set(stop_ids or ())
        terminated = trimmed.shape[1] > 0 and int(trimmed[0, -1]) in stop_ids
        if not terminated and trimmed.shape[1] >= self.max_new_tokens:
            raise RuntimeError(
                f"decoding hit the {self.max_new_tokens}-token cap without emitting a stop token"
                + (f" while captioning {context}" if context else "")
                + ". A caption that does not terminate on its own is the signature of a repetition "
                "loop, so no output was written. Raise --captioner_max_tokens if captions for this "
                "dataset are legitimately this long."
            )
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def generate_caption(
        self,
        ori_image_path: typing.Union[str, Path],
        ori_mask_path: typing.Union[str, Path],
        gen_image_path: typing.Union[str, Path],
        meta: dict,
    ) -> str:
        """Caption one (clean, mask, generated) triple. The three images share the same field of view.

        The user message provides the images in the order the prompt expects: mask, clean, generated.
        """
        self.load()
        messages = [
            {"role": "system", "content": self.replace_placeholders(self.system_prompt, meta)},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(ori_mask_path)},
                    {"type": "image", "image": str(ori_image_path)},
                    {"type": "image", "image": str(gen_image_path)},
                    {"type": "text", "text": self.replace_placeholders(self.user_prompt, meta)},
                ],
            },
        ]
        name = Path(gen_image_path).name
        raw = self._generate_raw(messages, context=name)
        # Post-process the response to ensure it matches the number of bounding boxes.
        return self.postprocess_response(raw, int(meta["num_bboxes"]), context=name)


def format_response(response: str, meta: dict) -> typing.Tuple[str, str]:
    """Strip the ``<answer>`` tags and return ``(caption, caption_with_meta)``.

    ``caption_with_meta`` prefixes the caption with a creation timestamp and the JSON ``meta`` block.
    """
    response = response.replace("<answer>", "", 1)
    response = response.replace("</answer>", "", 1)
    response = response.strip()

    # Add extra info to the response.
    response_with_extra_info = (
        f"Created: {datetime.now().isoformat()}\nMeta:\n{json.dumps(meta, indent=4)}\n\n{response}"
    )
    return response, response_with_extra_info
