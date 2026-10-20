# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data helpers for the anomaly pipeline: caption building, tokenizer/prompt utilities,
image/mask directory pairing, source (conditioning) item construction, and batch collation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn

from anomalygen.configs.texture.constants import VISION_TOKEN_PX

# Single editing-instruction caption shared by training and inference.
CAPTION_TEMPLATE = (
    "Add {defect} to a close-up look of a {texture}. "
    "Defective, Non-conforming, Substandard, Irregular, Damaged, Misaligned, "
    "Incomplete, Malformed, Contaminated, Under-processed, Faulty, Blemished, Expired."
)

# Ceiling on the caption tokens charged to a sample, both for a free-length caption and for the
# length the learnable text prompt pads to. The packed-sequence headroom check is computed against
# it, so going over would make that check under-estimate and the packer would silently discard
# largest-size samples. Lives here so the ceiling sits beside the template it bounds.
CAPTION_TOKEN_BUDGET = 256

# Prompt for the CFG unconditional branch (inference only; training never sees it). It asks for an
# unchanged reconstruction rather than for a "normal product": reconstruction is what an
# image-to-image model does natively, so the branch stays defect-free without inventing content.
# Class-agnostic on purpose — naming a texture or defect would put that concept on the guided axis.
UNCONDITIONAL_CAPTION = (
    "A static, unchanged video showing the exact same contents in all frames. "
    "Nothing is added, removed, or altered; the surface continues uniformly with no marks. "
    "Static, unchanged, identical, unmodified, original, untouched, consistent, stable, "
    "intact, uniform, clean, normal, defect-free."
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
# What a dataset directory — a clean pool, a submask source — is listed by; BMP is taken there too.
DATASET_IMAGE_EXTS = IMAGE_EXTS + (".bmp",)
MASK_FG_THRESHOLD = 128  # Foreground threshold for binarizing a uint8 mask (0–255).


def sample_token_count(input_w: int, input_h: int, caption_tokens: int = CAPTION_TOKEN_BUDGET) -> int:
    """Packed tokens of one sample: two vision items of ``(w/32)*(h/32)`` tokens plus two delimiters
    each, the caption, and one EOS. Mirrors the framework's ``_compute_num_tokens_per_sample``."""
    return 2 * ((input_w // VISION_TOKEN_PX) * (input_h // VISION_TOKEN_PX) + 2) + caption_tokens + 1


# Keys carrying a per-sample tensor whose spatial shape follows that sample's input size, and which must
# therefore survive collation as a plain list rather than being stacked.
RAGGED_KEYS = ("edit_mask",)


def build_caption(defect: str, texture: str) -> str:
    return CAPTION_TEMPLATE.format(
        defect=defect.replace("-", " ").replace("_", " "),
        texture=texture.replace("-", " ").replace("_", " "),
    )


def anomaly_class_ids(anomaly_types: Sequence[Sequence[str]]) -> Dict[str, int]:
    """``{"texture+defect": class_id}``, the id being the pair's position in ``anomaly_types``.

    The one id assignment training, validation and generation share. The recipe builder sorts the
    list and the checkpoint stamps it, so the ids never depend on the order a recipe was written in.
    """
    return {f"{t}+{d}": i for i, (t, d) in enumerate(anomaly_types)}


# Each half must be one safe path segment: an anomaly type becomes a directory name downstream.
# Rejected, never sanitized — as scripts/skill_utility/set_pipeline_vars.sh does for --name/--task.
_ANOMALY_TYPE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\+[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_anomaly_type(anomaly_type: str, *, field: str = "anomaly_type") -> str:
    """Return ``anomaly_type`` if it is one safe ``{texture}+{defect}`` pair, else raise ``ValueError``.

    Values come from a dataset spec or a generation manifest and are joined onto an output directory.
    ``pathlib`` drops the base for an absolute segment and follows ``..`` for a relative one, so the
    check must precede the join. ``field`` names the source key in the error.
    """
    if not isinstance(anomaly_type, str) or not _ANOMALY_TYPE_RE.fullmatch(anomaly_type):
        raise ValueError(
            f"invalid anomaly type in {field}: {anomaly_type!r}. Expected '{{texture}}+{{defect}}', "
            "each half alphanumeric plus '._-'. It becomes a directory name, so a separator, a "
            "leading '.', or an absolute path would write outside the run's output directory."
        )
    return anomaly_type


def caption_for_anomaly_type(anomaly_type: str) -> str:
    """Editing-instruction caption for an ``anomaly_type`` (``"{texture}+{defect}"`` or a bare
    texture). Single source of truth shared by inpaint generation and the guardrail text check,
    so the guarded text is exactly the text the model conditions on. A bare token (no ``"+"``)
    is used as both texture and defect."""
    texture, _, defect = anomaly_type.partition("+")
    return build_caption(defect=defect or texture, texture=texture)


def resolve_word_token_id(tokenizer, word: str) -> int:
    """Resolve a single representative token id for ``word`` (the prompt pad/init word).

    Tries a direct vocabulary lookup first; if that misses (the word is not a standalone
    vocab entry, e.g. a multi-piece BPE word), falls back to the last piece of the encoded
    word. Used to pad the learnable text-prompt region with the ``"anomaly"`` token id so
    the dataset's padded captions and the model-side template init agree.
    """
    tid = tokenizer.convert_tokens_to_ids(word)
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(tid, int) and tid >= 0 and tid != unk:
        return tid
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Cannot resolve a token id for pad/init word {word!r}.")
    return int(ids[-1])


def pad_or_truncate(ids: Sequence[int], length: int, pad_token_id: int) -> List[int]:
    """Right-pad (with ``pad_token_id``) or truncate ``ids`` to exactly ``length`` tokens."""
    out = list(ids[:length])
    if len(out) < length:
        out = out + [int(pad_token_id)] * (length - len(out))
    return out


def build_source_item(
    base: torch.Tensor,
    m3: torch.Tensor,
    background_dropout: bool = False,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Build the source item from ``base`` (clean/target image) and ``m3`` (defect mask, 1 inside).

    The defect region is ALWAYS uniform [-1, 1] noise — that is the single source encoding shared by
    training and inference. ``background_dropout`` only changes what surrounds it: the clean
    background (False) or a constant -1, i.e. black (True), a training-time augmentation that
    occasionally hides the background so the model cannot lean on it as a shortcut. Inference always
    passes False.

    ``seed`` draws the noise from a private generator, so it is reproducible and independent of
    rank, batch composition and guardrail skips. ``None`` (the training path) uses the global RNG,
    i.e. fresh noise every epoch. The draw is on CPU because CUDA's RNG is not arch-invariant.

    Returns a NEW tensor; ``base`` is never mutated.
    """
    if seed is None:
        noise = torch.rand_like(base) * 2.0 - 1.0
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.rand(base.shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
        noise = noise.to(device=base.device, dtype=base.dtype)
    background = torch.full_like(base, -1.0) if background_dropout else base
    return background * (1.0 - m3) + noise * m3


def list_images(d) -> List[Path]:
    """Every image directly under directory ``d`` — ``DATASET_IMAGE_EXTS`` matched on the lower-cased
    suffix, so ``c0.JPG`` is listed like ``c0.jpg`` — sorted by name; ``[]`` when ``d`` is not a
    directory. The ONE listing of a dataset directory (clean pools, submask sources) for AMG and the
    AMP scripts, so every stage sees the same files in the same order.
    """
    d = Path(d)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in DATASET_IMAGE_EXTS)


def require_same_size(image, mask, ident: str = "") -> None:
    """Raise unless a paired image and mask have identical pixel dimensions.

    The crop planner reads its extent from the image and its bbox from the mask, so a mismatched
    pair produces a wrong-scale, misaligned crop rather than an error.
    """
    if image.size != mask.size:
        raise ValueError(
            f"image/mask size mismatch{f' for {ident}' if ident else ''}: image is "
            f"{image.size[0]}x{image.size[1]} but mask is {mask.size[0]}x{mask.size[1]}."
        )


def list_image_mask_pairs(img_dir: str, mask_dir: str, mask_suffix: str = "") -> List[Tuple[str, str]]:
    """List ``(image_path, mask_path)`` pairs for images in ``img_dir``.

    The mask name is the image stem plus ``mask_suffix`` and the image extension, with a
    ``.png`` fallback. The mask path is returned even if it does not exist. Returns an
    empty list when ``img_dir`` is not a directory.
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(img_dir):
        return pairs
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        stem, ext = os.path.splitext(fn)
        mask_path = os.path.join(mask_dir, f"{stem}{mask_suffix}{ext}")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"{stem}{mask_suffix}.png")

        pairs.append((os.path.join(img_dir, fn), mask_path))

    return pairs


def anomalygen_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``custom_collate_fn`` plus pass-through (list) collation for the ragged keys.

    The framework's ``custom_collate_fn`` list-collates a fixed set of keys (``images``,
    ``image_size``, ``text_token_ids``, ...) and sends everything else through ``default_collate``,
    which stacks. Our ``edit_mask`` is not in that set, so under variable-resolution training it
    raises ``RuntimeError: stack expects each tensor to be equal size`` as soon as two samples in a
    batch have different input sizes.

    ``RankPartitionedDataLoader`` forwards ``**dataloader_kwargs`` to ``torch.utils.data.DataLoader``
    and only *defaults* ``collate_fn`` to its own, so supplying ours takes precedence.

    ``model._per_item_pixel_masks`` already accepts either a list of tensors or a stacked
    ``[B,1,H,W]`` tensor, so the consumer side needs no change.
    """
    # All-or-nothing per key: a key present on only some samples is left to ``custom_collate_fn``,
    # which drops it from the batch rather than emitting a partial list.
    ragged = {k: [sample[k] for sample in batch] for k in RAGGED_KEYS if batch and all(k in sample for sample in batch)}
    if not ragged:
        return custom_collate_fn(batch)

    # Shallow-copy so the caller's sample dicts are left intact (they may be reused by the loader).
    stripped = [{k: v for k, v in sample.items() if k not in ragged} for sample in batch]
    collated = custom_collate_fn(stripped)
    collated.update(ragged)
    return collated
