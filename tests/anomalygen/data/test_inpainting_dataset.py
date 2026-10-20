# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the anomaly inpainting TRAINING dataset.

Driven off a tiny on-disk fixture in the layout ``InpaintingDataset._index`` expects, so the whole
read -> augment -> source/target path runs for real. Tokenization is the one thing stubbed out: it
needs the framework's Qwen tokenizer (a download + a GPU-env dependency) and is orthogonal to the
augmentation/source-item behaviour under test.

The augmentations are probabilistic, so every test pins the per-sample probabilities to 0.0/1.0 and
seeds both the dataset's own NumPy generator (``seed=``) and torch's global RNG (the ring jitter is
a torchvision transform).
"""

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from anomalygen.data.augmentations import RandomRatioCrop, RandomScaleCrop
from anomalygen.data.inpainting_dataset import InpaintingDataset
from anomalygen.data.utils import CAPTION_TOKEN_BUDGET

_ANOMALY_TYPES = [["metal", "scratch"]]
_INPUT_SIZE = (64, 64)  # the single ladder size the tests pin, which DOES size the tensor
_BG_VALUE = 128  # flat mid-grey background: any near-black pixel in a sample must come from an aug


def _write_pair(root, blobs, size=64, side=8, bg=_BG_VALUE, name="a"):
    """Write one (anomaly_image, mask) pair in the on-disk layout the dataset indexes.

    ``blobs`` are (row, col) top-left corners of ``side``x``side`` filled mask squares.
    """
    img_dir = root / "metal" / "anomaly_image" / "scratch"
    mask_dir = root / "metal" / "mask" / "scratch"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((size, size, 3), bg, np.uint8), mode="RGB").save(img_dir / f"{name}.png")
    arr = np.zeros((size, size), dtype=np.uint8)
    for r, c in blobs:
        arr[r : r + side, c : c + side] = 255
    Image.fromarray(arr, mode="L").save(mask_dir / f"{name}_mask.png")
    return root


@pytest.fixture
def stub_tokenizer(monkeypatch):
    """Skip real tokenization (needs the framework's Qwen tokenizer); captions aren't under test."""
    monkeypatch.setattr(
        InpaintingDataset, "_tokenize", lambda self, caption: torch.zeros(4, dtype=torch.long), raising=True
    )


def _dataset(root, **kwargs):
    """Dataset over ``root`` with every probabilistic aug off unless the test turns one on."""
    params = dict(
        dataset_dir=str(root),
        anomaly_types=_ANOMALY_TYPES,
        # A one-size ladder at the fixture's own size, with the scale axis pinned to native: every
        # sample is then the whole 64x64 fixture, unresampled and unpadded, so a test can attribute
        # any black pixel to an augmentation rather than to input size padding.
        input_size_ladder=(_INPUT_SIZE[0],),
        crop_ratio_range=(1.5, 2.5),
        scale_range=(1.0, 1.0),
        background_dropout_prob=0.0,
        inst_aug_prob=0.0,
        ring_jitter_prob=0.0,
        seed=1,
    )
    params.update(kwargs)
    return InpaintingDataset(**params)


def _components(edit_mask: torch.Tensor) -> int:
    """Connected-component count of an ``edit_mask`` [1,H,W], background excluded."""
    arr = (edit_mask[0].numpy() > 0).astype(np.uint8)
    return cv2.connectedComponents(arr, connectivity=8)[0] - 1


# --- construction --------------------------------------------------------------------------------


def test_empty_dataset_dir_raises(tmp_path):
    # An empty dataset would hang the loader with no batch emitted, so it must fail at build time.
    with pytest.raises(FileNotFoundError):
        _dataset(tmp_path)


@pytest.mark.parametrize("prob", [-0.1, 1.5])
@pytest.mark.parametrize("field", ["background_dropout_prob", "inst_aug_prob", "ring_jitter_prob"])
def test_per_sample_probabilities_are_range_validated(tmp_path, field, prob):
    # Out-of-range probabilities must fail at dataset build, not silently change the aug rate
    # (p > 1 makes an "optional" aug unconditional; p < 0 makes it dead).
    _write_pair(tmp_path, blobs=[(28, 28)])
    with pytest.raises(ValueError):
        _dataset(tmp_path, **{field: prob})


# --- source item / background dropout ------------------------------------------------------------


def test_background_dropout_blacks_background_and_keeps_defect_noise(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)])
    torch.manual_seed(0)  # the source item's defect noise comes from torch's global RNG
    item = _dataset(tmp_path, background_dropout_prob=1.0)[0]

    source, target = item["images"]
    m3 = item["edit_mask"].expand_as(target) > 0
    # Background dropped to -1 (black) everywhere outside the defect ...
    assert torch.equal(source[~m3], torch.full_like(source[~m3], -1.0))
    # ... while the defect region is noise, not more black.
    assert not torch.all(source[m3] == -1.0)
    assert float(source[m3].max()) > 0.0


def test_no_background_dropout_keeps_target_background(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)])
    torch.manual_seed(0)  # the source item's defect noise comes from torch's global RNG
    item = _dataset(tmp_path, background_dropout_prob=0.0)[0]

    source, target = item["images"]
    m3 = item["edit_mask"].expand_as(target) > 0
    # Source background is the clean target verbatim; only the defect region is substituted.
    assert torch.equal(source[~m3], target[~m3])
    assert not torch.equal(source[m3], target[m3])


# --- instance augmentation -----------------------------------------------------------------------


def test_instance_aug_keeps_one_instance_and_blacks_the_dropped_one(tmp_path, stub_tokenizer):
    # Two well-separated blobs; with inst_aug_prob=1.0 exactly one survives in the mask and the
    # other is blacked out in the IMAGE, so the pair stays self-consistent (no unmarked defect).
    _write_pair(tmp_path, blobs=[(24, 12), (24, 44)])
    two_blob_item = _dataset(tmp_path, inst_aug_prob=0.0)[0]
    assert _components(two_blob_item["edit_mask"]) == 2  # baseline: both instances present

    item = _dataset(tmp_path, inst_aug_prob=1.0)[0]
    assert _components(item["edit_mask"]) == 1
    _, target = item["images"]
    assert float(target.min()) < -0.9  # the dropped blob is black (-1) in the target pixels
    assert float(two_blob_item["images"][1].min()) > -0.9  # ... and was not, without the aug


def test_instance_aug_is_a_noop_on_a_single_instance_mask(tmp_path, stub_tokenizer):
    # Nothing to drop: the mask keeps its one instance and no blackout is painted into the image.
    _write_pair(tmp_path, blobs=[(28, 28)])
    item = _dataset(tmp_path, inst_aug_prob=1.0)[0]

    assert _components(item["edit_mask"]) == 1
    assert float(item["images"][1].min()) > -0.9  # flat mid-grey survives untouched


# --- ring colour jitter --------------------------------------------------------------------------


def test_ring_jitter_never_touches_the_defect_pixels(tmp_path, stub_tokenizer):
    # The whole point of the band is to destroy the colour cue AROUND the defect; recolouring the
    # defect itself would corrupt the thing the model is being taught to draw.
    _write_pair(tmp_path, blobs=[(28, 28)], bg=200)
    ds = _dataset(tmp_path, ring_jitter_prob=1.0)
    image = Image.open(ds.samples[0]["image"]).convert("RGB")
    mask = Image.open(ds.samples[0]["mask"]).convert("L")

    torch.manual_seed(0)  # the jitter draw is a torchvision transform (torch's global RNG)
    jittered, _ = ds.augmentations[0](image, mask)  # [0] is the ring jitter

    before, after = np.asarray(image), np.asarray(jittered)
    inside = np.asarray(mask) > 0
    assert np.array_equal(after[inside], before[inside])  # defect pixels byte-identical
    assert not np.array_equal(after[~inside], before[~inside])  # the surrounding band did change


def test_ring_jitter_sample_is_well_formed(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)], bg=200)
    torch.manual_seed(0)
    item = _dataset(tmp_path, ring_jitter_prob=1.0)[0]

    source, target = item["images"]
    assert source.shape == target.shape == (3, *_INPUT_SIZE)
    assert float(item["edit_mask"].sum()) > 0.0


# --- crop_mode -----------------------------------------------------------------------------------


def test_an_unknown_crop_mode_is_rejected(tmp_path):
    _write_pair(tmp_path, blobs=[(28, 28)])
    with pytest.raises(ValueError, match="crop_mode"):
        _dataset(tmp_path, crop_mode="fixed")


@pytest.mark.parametrize(
    ("crop_mode", "expected"), [("scale", RandomScaleCrop), ("ratio", RandomRatioCrop)], ids=["scale", "ratio"]
)
def test_crop_mode_selects_the_crop_stage(tmp_path, crop_mode, expected):
    # The stage must stay LAST in the pipeline whichever mode is picked: _augment peels it off to
    # hand it the per-class input-size cap.
    _write_pair(tmp_path, blobs=[(28, 28)])
    assert isinstance(_dataset(tmp_path, crop_mode=crop_mode).augmentations[-1], expected)


def test_ratio_mode_emits_the_fixed_input_size(tmp_path, stub_tokenizer):
    # A single-entry ladder under "ratio" is the fixed-model-input-size recipe: every sample lands
    # on that size, so the batch shape is constant.
    _write_pair(tmp_path, blobs=[(28, 28)])
    ds = _dataset(tmp_path, crop_mode="ratio", crop_ratio_range=(1.5, 8.0), input_size_ladder=(32,))
    for _ in range(5):
        assert ds[0]["edit_mask"].shape == (1, 32, 32)


# --- item contract -------------------------------------------------------------------------------


def test_item_shapes_and_class_id(tmp_path, stub_tokenizer):
    _write_pair(tmp_path, blobs=[(28, 28)])
    item = _dataset(tmp_path)[0]

    assert len(item["images"]) == 2  # [source, target] -> two vision items
    assert all(t.shape == (3, *_INPUT_SIZE) for t in item["images"])
    assert item["edit_mask"].shape == (1, *_INPUT_SIZE)
    assert set(torch.unique(item["edit_mask"]).tolist()) <= {0.0, 1.0}  # binary
    assert item["anomaly_class_id"] == 0
    assert item["num_frames"] == 2
    # image_size is published PER VISION ITEM and derived from the actual tensor, since every
    # sample can carry its own input size.
    h, w = _INPUT_SIZE
    assert len(item["image_size"]) == 2
    assert all(s.tolist() == [h, w, h, w] for s in item["image_size"])


@pytest.mark.parametrize("inst_aug_prob", [0.0, 1.0])
def test_edit_mask_is_never_all_zero(tmp_path, stub_tokenizer, inst_aug_prob):
    # An all-zero edit_mask is a degenerate "inpaint nothing" sample whose masked loss has no
    # grad_fn — the crop must always keep the defect in frame, across repeated draws.
    _write_pair(tmp_path, blobs=[(24, 12), (24, 44)])
    ds = _dataset(tmp_path, inst_aug_prob=inst_aug_prob, ring_jitter_prob=1.0)
    torch.manual_seed(0)
    for i in range(10):
        assert float(ds[i]["edit_mask"].sum()) > 0.0


# --- caption token budget ------------------------------------------------------------------------


@pytest.fixture
def fake_tokenize(monkeypatch):
    """Drive the real ``_tokenize`` with a caption->N-ids stub, bypassing the Qwen tokenizer."""

    class _Tok:
        # Only what _tokenize's two branches touch: the pad path resolves a pad id by word.
        def convert_tokens_to_ids(self, word):
            return 0

    def _set(n_tokens):
        monkeypatch.setattr(InpaintingDataset, "tokenizer", property(lambda self: _Tok()), raising=True)
        monkeypatch.setattr(
            "anomalygen.data.inpainting_dataset.tokenize_caption",
            lambda caption, tok, **kw: list(range(n_tokens)),
            raising=True,
        )

    return _set


def test_caption_within_budget_tokenizes(tmp_path, fake_tokenize):
    _write_pair(tmp_path, blobs=[(28, 28)])
    fake_tokenize(CAPTION_TOKEN_BUDGET)  # exactly at the ceiling is still fine
    assert _dataset(tmp_path)._tokenize("short caption").shape == (CAPTION_TOKEN_BUDGET,)


def test_caption_over_budget_raises(tmp_path, fake_tokenize):
    # Over budget the packer would silently discard the largest-size samples, so this must fail loudly.
    _write_pair(tmp_path, blobs=[(28, 28)])
    fake_tokenize(CAPTION_TOKEN_BUDGET + 1)
    with pytest.raises(ValueError, match=f"{CAPTION_TOKEN_BUDGET}-token budget"):
        _dataset(tmp_path)._tokenize("a very long caption")


def test_learnable_text_prompt_pads_to_an_exact_length_instead_of_budgeting(tmp_path, fake_tokenize):
    # With text_prompt_pad_to set the caption length is exact, so the budget does not apply and an
    # over-budget caption is truncated rather than rejected.
    _write_pair(tmp_path, blobs=[(28, 28)])
    fake_tokenize(CAPTION_TOKEN_BUDGET * 2)
    ds = _dataset(tmp_path, text_prompt_pad_to=16)
    assert ds._tokenize("a very long caption").shape == (16,)


# --- per-class input-size cap ---------------------------------------------------------------------


def _write_class_pair(root, texture, defect, blobs, side, size=256, name="a"):
    """One (image, mask) pair under an arbitrary class, so a fixture can hold several classes."""
    img_dir = root / texture / "anomaly_image" / defect
    mask_dir = root / texture / "mask" / defect
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), _BG_VALUE, np.uint8), mode="RGB").save(img_dir / f"{name}.png")
    arr = np.zeros((size, size), dtype=np.uint8)
    for r, c in blobs:
        arr[r : r + side, c : c + side] = 255
    Image.fromarray(arr, mode="L").save(mask_dir / f"{name}_mask.png")


def test_each_class_is_capped_by_its_own_largest_defect(tmp_path, stub_tokenizer):
    # A class of small defects must not draw an input size sized for a large one. The cap CEILS to a
    # size: 40 * 4.0 = 160 is nearer 128 than 256, so rounding to nearest would cap this class below
    # the framing it was asked for and pin its largest defects to the floor.
    _write_class_pair(tmp_path, "metal", "scratch", blobs=[(10, 10)], side=8)
    _write_class_pair(tmp_path, "metal", "dent", blobs=[(10, 10)], side=40)
    ds = _dataset(
        tmp_path,
        anomaly_types=[["metal", "scratch"], ["metal", "dent"]],
        input_size_ladder=(32, 64, 128, 256),
    )

    assert ds.class_input_size_ladders["metal+scratch"] == [32]  # 8 * 4.0 = 32
    assert ds.class_input_size_ladders["metal+dent"] == [32, 64, 128, 256]  # 40 * 4.0 = 160 -> ceil to 256


def test_a_class_whose_masks_are_all_empty_is_reported(tmp_path, stub_tokenizer):
    # Dropping its records would leave the class training nothing, silently.
    _write_class_pair(tmp_path, "metal", "scratch", blobs=[(10, 10)], side=8)
    _write_class_pair(tmp_path, "metal", "dent", blobs=[], side=8)
    with pytest.raises(RuntimeError, match="metal\\+dent"):
        _dataset(tmp_path, anomaly_types=[["metal", "scratch"], ["metal", "dent"]])


def test_a_class_with_no_training_data_is_refused(tmp_path, stub_tokenizer):
    # Its adapter would never train, while generation would still route that defect type to it.
    _write_class_pair(tmp_path, "metal", "scratch", blobs=[(10, 10)], side=8)
    with pytest.raises(FileNotFoundError, match="metal\\+dent"):
        _dataset(tmp_path, anomaly_types=[["metal", "scratch"], ["metal", "dent"]])


def test_a_record_with_an_empty_mask_is_dropped_at_build(tmp_path, stub_tokenizer, loguru_lines):
    # The class ladder pass decodes every mask anyway; a record with nothing to inpaint leaves the
    # sample list there, so __getitem__ is a single draw with no retry.
    _write_pair(tmp_path, blobs=[(20, 20)], name="a")
    _write_pair(tmp_path, blobs=[], name="b")
    ds = _dataset(tmp_path)
    assert [rec["image"].split("/")[-1] for rec in ds.samples] == ["a.png"]
    assert any("dropped 1 sample" in ln for ln in loguru_lines)


def test_a_mask_emptied_by_an_augmentation_names_the_record(tmp_path, stub_tokenizer):
    # An on-disk mask with defect pixels that comes back empty is a bug in the stage chain, not a
    # sample to skip. The error names the record so the stage can be found.
    _write_pair(tmp_path, blobs=[(20, 20)], name="a")
    ds = _dataset(tmp_path)

    def blank(image, mask):
        return image, Image.new("L", mask.size)

    ds._augmentations = [blank, ds.augmentations[-1]]
    with pytest.raises(ValueError, match=r"a_mask\.png.*no defect pixels"):
        ds[0]


def test_dataset_rejects_a_sample_whose_mask_size_differs_from_its_image(tmp_path, stub_tokenizer):
    root = _write_pair(tmp_path, blobs=[(20, 20)])
    arr = np.zeros((128, 128), dtype=np.uint8)
    arr[40:56, 40:56] = 255
    Image.fromarray(arr, mode="L").save(root / "metal" / "mask" / "scratch" / "a_mask.png")

    with pytest.raises(ValueError, match="image/mask size mismatch"):
        _dataset(root)


def test_a_resumed_run_seeds_its_workers_past_the_replay(tmp_path, stub_tokenizer):
    # The worker generators are rebuilt on a resume; without the start iteration in their seed they would
    # draw the run's first iterations again. A fresh run keeps the seed it always had.
    _write_pair(tmp_path, blobs=[(20, 20)], name="a")
    fresh, resumed = _dataset(tmp_path), _dataset(tmp_path)
    resumed.start_iteration = 13000
    assert fresh.rng.random() == np.random.default_rng((fresh._base_seed, 0, 0)).random()
    assert resumed.rng.random() != np.random.default_rng((fresh._base_seed, 0, 0)).random()
