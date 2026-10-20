# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anomaly inpainting training dataset.

Each sample is a two-vision-item image-edit example ``images = [source, target]``, where the
source is the clean image with the defect region substituted by noise — its background dropped to
-1 (black) per-sample with probability ``background_dropout_prob`` — and the target is the clean
image the model learns to produce, plus per-sample ``edit_mask`` and ``anomaly_class_id`` keys.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from PIL import Image

from anomalygen.configs.texture.constants import (
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_RATIO_RANGE,
    DEFAULT_INPUT_SIZE_LADDER,
    DEFAULT_SCALE_RANGE,
    INPUT_SIZE_CAP_HEADROOM,
)
from anomalygen.data.augmentations import (
    RandomHorizontalFlip,
    RandomInstanceDrop,
    RandomOrthogonalRotation,
    RandomRatioCrop,
    RandomRingJitter,
    RandomRotation,
    RandomScaleCrop,
    RandomVerticalFlip,
)
from anomalygen.data.crop_geometry import (
    ceil_input_size,
    mask_bbox_or_none,
    validate_geometry,
)
from anomalygen.data.utils import (
    CAPTION_TOKEN_BUDGET,
    MASK_FG_THRESHOLD,
    anomaly_class_ids,
    build_caption,
    build_source_item,
    list_image_mask_pairs,
    pad_or_truncate,
    require_same_size,
    resolve_word_token_id,
)


def _to_chw(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [3,H,W] in [-1, 1] (VAE convention)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _to_edit_mask(mask: Image.Image) -> torch.Tensor:
    """PIL 'L' mask -> binary float tensor [1,H,W] (1 inside the defect region)."""
    return (torch.from_numpy(np.asarray(mask, dtype=np.float32)) >= MASK_FG_THRESHOLD).float().unsqueeze(0)


def index_samples(dataset_dir: str, anomaly_types: Sequence[Sequence[str]]) -> List[Dict]:
    """Every (image, mask) pair under ``dataset_dir`` for ``anomaly_types``, in the on-disk layout::

    {dataset_dir}/{texture}/anomaly_image/{defect}/<name>.png  +
    {dataset_dir}/{texture}/mask/{defect}/<name>_mask.png

    Raises when a class has no pair: its adapter would never train, while generation would still
    route that defect type to it.
    """
    samples: List[Dict] = []
    n_missing_mask = 0
    empty: List[str] = []
    for texture, defect in anomaly_types:
        img_dir = os.path.join(dataset_dir, texture, "anomaly_image", defect)
        mask_dir = os.path.join(dataset_dir, texture, "mask", defect)
        found = 0
        for img_fn, mask_fn in list_image_mask_pairs(img_dir, mask_dir, mask_suffix="_mask"):
            if not os.path.exists(mask_fn):
                log.warning(f"InpaintingDataset: no mask for {img_fn} (looked for {mask_fn}); skipping.")
                n_missing_mask += 1
                continue
            samples.append(
                {
                    "image": img_fn,
                    "mask": mask_fn,
                    "texture": texture,
                    "defect": defect,
                    "class_name": f"{texture}+{defect}",
                }
            )
            found += 1
        if not found:
            empty.append(f"{texture}+{defect}")

    if n_missing_mask:
        log.warning(f"InpaintingDataset: skipped {n_missing_mask} image(s) with no matching mask on disk.")
    if empty:
        raise FileNotFoundError(
            f"InpaintingDataset: no (image, mask) pairs under {dataset_dir!r} for class(es) {empty}. Every class "
            "in anomaly_types needs {dataset_dir}/{texture}/anomaly_image/{defect}/*.png + "
            "{texture}/mask/{defect}/*_mask.png. Use an ABSOLUTE dataset_path."
        )
    return samples


def scan_masks(samples: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """One decode per pair: check the sizes agree, drop the records whose mask has no defect pixels,
    and take each class's largest bbox side.

    Returns ``(kept_samples, max_bbox_side_per_class)``. Raises when a class is left with no sample:
    dropping its records would leave it training nothing, silently.
    """
    max_bbox: Dict[str, int] = defaultdict(int)
    kept: List[Dict] = []
    for rec in samples:
        with Image.open(rec["image"]) as img, Image.open(rec["mask"]) as mask:
            require_same_size(img, mask, f"{rec['image']} + {rec['mask']}")
            box = mask_bbox_or_none(mask.convert("L"))
        if box is None:
            continue
        kept.append(rec)
        lx, uy, rx, by = box
        max_bbox[rec["class_name"]] = max(max_bbox[rec["class_name"]], rx - lx + 1, by - uy + 1)

    empty = sorted({rec["class_name"] for rec in samples} - set(max_bbox))
    if empty:
        raise RuntimeError(
            f"InpaintingDataset: class(es) {', '.join(empty)} have no non-empty mask — check the masks on disk."
        )
    if len(kept) < len(samples):
        log.warning(
            f"InpaintingDataset: dropped {len(samples) - len(kept)} sample(s) whose mask has no defect pixels "
            f"(>= {MASK_FG_THRESHOLD})."
        )
    return kept, dict(max_bbox)


def class_ladders_from_bboxes(max_bbox: Dict[str, int], ladder: Sequence[int]) -> Dict[str, List[int]]:
    """Per class, ``ladder`` capped at ``ceil(max_bbox * INPUT_SIZE_CAP_HEADROOM)``: a class of small
    defects never draws the largest input size for nothing. CEIL to a size, not nearest: rounding
    down would cap a class below the reference it was asked for and pin its largest defects to the
    framing floor."""
    out: Dict[str, List[int]] = {}
    for cls, side in max_bbox.items():
        cap = ceil_input_size(side * INPUT_SIZE_CAP_HEADROOM, ladder)
        out[cls] = [v for v in ladder if v <= cap]
    return out


class InpaintingDataset(torch.utils.data.Dataset):
    """Anomaly inpainting dataset (noise-substituted source, optional background dropout).

    On-disk layout (one entry per (texture, defect) class), indexed by :func:`index_samples`:
        ``{dataset_dir}/{texture}/anomaly_image/{defect}/<name>.png``  +
        ``{dataset_dir}/{texture}/mask/{defect}/<name>_mask.png``

    Each item: crop the (image, mask) pair onto its input size (see ``_build_crop``), build the source (defect region
    noise-substituted; background dropped to -1 (black) per ``background_dropout_prob``) and the clean
    target, and emit them as the two vision items plus the binary ``edit_mask`` and integer
    ``anomaly_class_id``.
    """

    def __init__(
        self,
        dataset_dir: str,
        anomaly_types: Sequence[Sequence[str]],
        crop_ratio_range: Sequence[float] = DEFAULT_CROP_RATIO_RANGE,
        scale_range: Sequence[float] = DEFAULT_SCALE_RANGE,
        input_size_ladder: Sequence[int] = DEFAULT_INPUT_SIZE_LADDER,
        crop_mode: str = DEFAULT_CROP_MODE,
        background_dropout_prob: float = 0.0,
        inst_aug_prob: float = 0.2,
        ring_jitter_prob: float = 0.0,
        tokenizer_config=None,
        text_prompt_pad_to: int | None = None,
        text_prompt_pad_word: str = "anomaly",
        dataset_name: str = "anomaly",
        seed: int = 1,
        repeat: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.anomaly_types = [tuple(a) for a in anomaly_types]
        # Per-sample probability for augmentations.RandomInstanceDrop.
        self.inst_aug_prob = float(inst_aug_prob)
        if not 0.0 <= self.inst_aug_prob <= 1.0:
            raise ValueError(f"inst_aug_prob must be in [0, 1], got {self.inst_aug_prob}")
        # Per-sample probability for augmentations.RandomRingJitter, plus its band width in ORIGINAL-image
        # pixels (an env knob, not a recipe key). Both are validated here so a bad value fails at
        # dataset build, not inside a worker.
        self.ring_jitter_prob = float(ring_jitter_prob)
        if not 0.0 <= self.ring_jitter_prob <= 1.0:
            raise ValueError(f"ring_jitter_prob must be in [0, 1], got {self.ring_jitter_prob}")
        self.ring_jitter_px = int(os.environ.get("ANOMALYGEN_RING_JITTER_PX", "10"))
        if self.ring_jitter_px < 0:
            raise ValueError(f"ANOMALYGEN_RING_JITTER_PX must be >= 0, got {self.ring_jitter_px}")
        # Per-sample prob the source item's background is dropped to -1 (black; the defect region is noise
        # either way); a training-only augmentation, inference never drops the background.
        self.background_dropout_prob = float(background_dropout_prob)
        if not 0.0 <= self.background_dropout_prob <= 1.0:
            raise ValueError(f"background_dropout_prob must be in [0, 1], got {self.background_dropout_prob}")
        self.tokenizer_config = tokenizer_config
        # When learnable text-prompt tuning is enabled, every caption is padded/truncated to a
        # fixed length so each sample contributes the same number of text positions (the packed
        # text block is then a constant size and the per-defect-type learnable prompt scatters cleanly
        # into the first ``text_prompt_pad_to`` rows). None -> native caption length (feature off).
        self.text_prompt_pad_to = None if text_prompt_pad_to is None else int(text_prompt_pad_to)
        self.text_prompt_pad_word = str(text_prompt_pad_word)
        self._pad_token_id = None  # resolved lazily from the (per-worker) tokenizer
        self.dataset_name = dataset_name
        self.repeat = max(1, int(repeat))
        # Checked here as well as in the recipe builder: this class is also built directly.
        geometry = validate_geometry(crop_mode, input_size_ladder, crop_ratio_range, scale_range)
        self.crop_mode = geometry.crop_mode
        self.input_size_ladder = geometry.input_size_ladder
        self.crop_ratio_range = geometry.crop_ratio_range
        self.scale_range = geometry.scale_range
        self.class_to_id = anomaly_class_ids(self.anomaly_types)
        self._base_seed = int(seed)
        # Set by train.py on a resume; folded into the worker seed so the augmentation stream moves on.
        self.start_iteration = 0
        self._rng = None  # built lazily per worker (see the ``rng`` property)
        # built lazily per worker (binds the per-worker rng; see the ``augmentations`` property)
        self._augmentations = None
        self._tokenizer = None  # built lazily per worker (avoids pickling across workers)
        self.samples: List[Dict] = index_samples(self.dataset_dir, self.anomaly_types)

        # Fail fast: an empty dataset would make the data loader hang with no batch emitted.
        if not self.samples:
            raise FileNotFoundError(
                f"InpaintingDataset found 0 samples under dataset_dir={self.dataset_dir!r} "
                f"(resolved from cwd={os.getcwd()!r}) for anomaly_types={self.anomaly_types}. "
                f"Expected {{dataset_dir}}/{{texture}}/anomaly_image/{{defect}}/*.png + "
                f"{{texture}}/mask/{{defect}}/*_mask.png. Use an ABSOLUTE DATASET_PATH."
            )

        # One decode per pair, at construction rather than in a worker: the size check, the empty-mask
        # drop and the per-class input-size caps all come out of it. Training publishes the caps on
        # the model config (scripts/texture/train.py), so the checkpoint stamps them for generation.
        self.samples, max_bbox = scan_masks(self.samples)
        self.class_input_size_ladders = class_ladders_from_bboxes(max_bbox, self.input_size_ladder)
        log.info(
            "InpaintingDataset input-size caps (max_bbox -> cap): "
            + ", ".join(f"{c}: {max_bbox[c]} -> {self.class_input_size_ladders[c][-1]}" for c in sorted(max_bbox))
        )

        # Group sample indices by class for class-balanced sampling in ``__getitem__``: draw a class
        # uniformly, then a sample within it, giving P(sample) = 1/(num_classes * class_count). This
        # matches a WeightedRandomSampler with weight 1/class_count, done here because the framework
        # builds the DataLoader and takes no external sampler. Fixes class imbalance and, by ignoring
        # the sequential index, the lack of shuffling.
        _pools: Dict[str, List[int]] = defaultdict(list)
        for i, rec in enumerate(self.samples):
            _pools[rec["class_name"]].append(i)
        self._class_pools: List[np.ndarray] = [np.asarray(v, dtype=np.int64) for v in _pools.values()]

        log.info(
            f"InpaintingDataset: {len(self.samples)} samples across {len(self._class_pools)} populated "
            f"classes (virtual length {len(self.samples) * self.repeat} with repeat={self.repeat}); "
            f"class-balanced sampling "
            f"(per-class counts: {{{', '.join(f'{c!r}: {len(p)}' for c, p in zip(_pools, self._class_pools))}}})."
        )

    @property
    def rng(self) -> np.random.Generator:
        """Per-worker NumPy generator, seeded from the experiment seed.

        Built lazily on first access so it is seeded *inside* each dataloader worker (after
        fork) rather than once in the parent. Mixing the base seed with the shard rank and
        worker id yields a stream that is reproducible from the experiment seed yet independent
        across ranks and workers. ``shard_rank`` is set by RankPartitionedDataLoader after
        construction; absent (e.g. num_workers=0, single process) it defaults to 0.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = 0 if info is None else info.id
            shard_rank = getattr(self, "shard_rank", 0)
            seed = (self._base_seed, shard_rank, worker_id)
            # A resumed run continues at start_iteration: folding it in gives the workers a stream the
            # run has not drawn yet rather than its first iterations' again. A fresh run (0) keeps the
            # seed as it is.
            if self.start_iteration:
                seed += (self.start_iteration,)
            self._rng = np.random.default_rng(seed)
        return self._rng

    @property
    def augmentations(self):
        """The read-time augmentation pipeline, in application order.

        Built lazily so every stage binds THIS worker's ``rng`` (see ``rng``), and shares one
        generator so the draw sequence is deterministic per worker. Order matters: ring jitter cuts
        its bands against the full mask, so it must precede the instance drop; both run before any
        geometry so their pixels are flipped and cropped like ordinary content. The crop MUST stay
        last — ``_augment`` peels it off to hand it the per-class input-size cap.

        Stages left at ``p=0`` cost nothing and draw no randomness (see ``augmentations._fires``), so
        the disabled rotation does not perturb any other stage's draws.
        """
        if self._augmentations is None:
            rng = self.rng
            self._augmentations = [
                RandomRingJitter(self.ring_jitter_prob, self.ring_jitter_px, rng=rng),
                RandomInstanceDrop(self.inst_aug_prob, self.ring_jitter_px, rng=rng),
                RandomVerticalFlip(0.5, rng=rng),
                RandomHorizontalFlip(0.5, rng=rng),
                RandomOrthogonalRotation(rng=rng),
                # Off (p=0) — listed so the stage is discoverable and enabling it is a one-word edit.
                # It must sit immediately before the crop, which trims the black corners it leaves.
                RandomRotation(max_angle=20, p=0.0, rng=rng),
                self._build_crop(rng),
            ]
        return self._augmentations

    def _build_crop(self, rng):
        """The crop stage for ``crop_mode``: framing + scale axes, or framing alone."""
        if self.crop_mode == "ratio":
            return RandomRatioCrop(
                crop_ratio_range=self.crop_ratio_range,
                input_size_ladder=self.input_size_ladder,
                rng=rng,
            )
        return RandomScaleCrop(
            crop_ratio_range=self.crop_ratio_range,
            scale_range=self.scale_range,
            input_size_ladder=self.input_size_ladder,
            rng=rng,
        )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            proc = lazy_instantiate(self.tokenizer_config)
            tok = getattr(proc, "tokenizer", proc)
            tok, _ = add_special_tokens(tok)
            self._tokenizer = tok
        return self._tokenizer

    @property
    def pad_token_id(self) -> int:
        """Token id used to pad the learnable text-prompt region (the ``"anomaly"`` word)."""
        if self._pad_token_id is None:
            self._pad_token_id = resolve_word_token_id(self.tokenizer, self.text_prompt_pad_word)
        return self._pad_token_id

    def _tokenize(self, caption: str) -> torch.Tensor:
        ids = tokenize_caption(caption, self.tokenizer, is_video=False, use_system_prompt=False)
        if self.text_prompt_pad_to is not None:
            ids = pad_or_truncate(ids, self.text_prompt_pad_to, self.pad_token_id)
        elif len(ids) > CAPTION_TOKEN_BUDGET:
            # The builder sized max_sequence_length assuming this ceiling; over it, the packer would
            # drop the largest samples with no error. Fail here instead, naming the caption.
            raise ValueError(
                f"caption is {len(ids)} tokens, over the {CAPTION_TOKEN_BUDGET}-token budget the "
                f"packed-sequence headroom check assumes: {caption!r}. Shorten the texture/defect "
                "names, or raise CAPTION_TOKEN_BUDGET and max_sequence_length together."
            )
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        # Virtual length: real samples cycled ``repeat`` times, so the loader's iterator
        # does not exhaust mid-run. __getitem__ maps the index back into the real samples.
        return len(self.samples) * self.repeat

    def _augment(self, rec: Dict) -> Tuple[Image.Image, torch.Tensor]:
        """Read one record and run the augmentation pipeline. Returns the input-sized pair."""
        image = Image.open(rec["image"]).convert("RGB")
        mask = Image.open(rec["mask"]).convert("L")

        # Pipeline order (see the ``augmentations`` property): ring jitter, instance drop, V/H flip,
        # quarter turn, free rotation (off), zoom-crop. Photometric stages run before any geometry so
        # their pixels are flipped and cropped like ordinary content.
        *pre_crop, crop = self.augmentations
        for aug in pre_crop:
            image, mask = aug(image, mask)
        # The crop stage returns the pair ALREADY at input size (it owns the resample decision),
        # so there is no resize here. It raises on a mask an earlier stage emptied; name the record.
        try:
            aug_image, aug_mask = crop(image, mask, self.class_input_size_ladders.get(rec["class_name"]))
        except ValueError as e:
            raise ValueError(f"{rec['mask']}: {e}") from e
        return aug_image, _to_edit_mask(aug_mask)  # [1,H,W]

    def __getitem__(self, idx: int) -> Dict:
        # Class-balanced draw: uniform class, then uniform sample within it. ``idx`` (the framework's
        # sequential counter) is ignored so items are shuffled and every class is equally likely.
        pool = self._class_pools[int(self.rng.integers(len(self._class_pools)))]
        rec = self.samples[int(pool[int(self.rng.integers(len(pool)))])]
        image, edit = self._augment(rec)

        target = _to_chw(image)  # [3,H,W] in [-1,1]  (photometric already applied at read-time)

        # Source item: defect region noise-substituted, background dropped to -1 (black) with prob
        # background_dropout_prob. Guard so prob=0 draws no rng, keeping the augmentation stream
        # identical to background-preserving-only.
        m3 = edit.expand_as(target)
        drop_background = self.background_dropout_prob > 0.0 and self.rng.random() < self.background_dropout_prob
        source = build_source_item(target, m3, drop_background)

        caption = build_caption(defect=rec["defect"], texture=rec["texture"])
        # Derived from the ACTUAL tensor, not from a configured size: under variable-resolution
        # training every sample can have its own input size. One entry PER VISION ITEM — the framework
        # flattens this key across the batch and indexes it per item (``_remove_padding_from_latent``,
        # ``slice_data_batch``). The framework also derives its per-SAMPLE ``data_resolutions`` from
        # this flat list, which mis-indexes it whenever a sample has more than one item; harmless
        # only because ``DEFAULT_SHIFT`` is a scalar, so that lookup is never read.
        h, w = int(target.shape[-2]), int(target.shape[-1])

        return {
            "images": [source, target],  # list of two [3,H,W] tensors → two vision items
            "text_token_ids": self._tokenize(caption),
            "ai_caption": caption,
            "image_size": [
                torch.tensor([h, w, h, w], dtype=torch.float32),
                torch.tensor([h, w, h, w], dtype=torch.float32),
            ],
            "fps": 30.0,
            "num_frames": 2,
            "sequence_plan": SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]),
            "edit_mask": edit,  # [1,H,W]
            "anomaly_class_id": int(self.class_to_id[rec["class_name"]]),
            "dataset_name": self.dataset_name,
        }


def get_inpainting_dataset(**kwargs) -> InpaintingDataset:
    """LazyCall target used by the experiment config."""
    return InpaintingDataset(**kwargs)
