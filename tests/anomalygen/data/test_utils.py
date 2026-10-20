# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for anomalygen.data.utils (caption/tokenizer/path helpers and the source item)."""

import numpy as np
import pytest
import torch
from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from PIL import Image

from anomalygen.data.utils import (
    CAPTION_TEMPLATE,
    RAGGED_KEYS,
    anomaly_class_ids,
    anomalygen_collate_fn,
    build_caption,
    build_source_item,
    list_image_mask_pairs,
    list_images,
    pad_or_truncate,
    require_same_size,
    resolve_word_token_id,
    validate_anomaly_type,
)


@pytest.mark.parametrize(
    "anomaly_type",
    ["wood+scratch", "metal_surface+MT_Blowhole", "passive_component+excess_solder", "Phone+oil", "IC+bridge"],
)
def test_validate_anomaly_type_accepts_the_real_recipe_values(anomaly_type):
    assert validate_anomaly_type(anomaly_type) == anomaly_type


def test_anomaly_class_ids_are_positions_in_the_list():
    assert anomaly_class_ids([["metal", "scratch"], ["wood", "hole"]]) == {"metal+scratch": 0, "wood+hole": 1}


@pytest.mark.parametrize(
    "anomaly_type",
    [
        "../escape+x",
        "wood+../../escape",
        "a/b+c",
        "wood+scratch/../..",
        "..+..",
        "wood+scratch\x00",
    ],
)
def test_validate_anomaly_type_rejects_a_traversing_value(anomaly_type):
    with pytest.raises(ValueError, match="anomaly type"):
        validate_anomaly_type(anomaly_type)


def test_validate_anomaly_type_rejects_an_absolute_value():
    # pathlib.Path(base, seg) DISCARDS base when seg is absolute — the worst case.
    with pytest.raises(ValueError, match="anomaly type"):
        validate_anomaly_type("/tmp/escape+x")


@pytest.mark.parametrize("anomaly_type", ["", "scratch", "+scratch", "wood+", "wood+a+b"])
def test_validate_anomaly_type_rejects_a_malformed_key(anomaly_type):
    with pytest.raises(ValueError, match="anomaly type"):
        validate_anomaly_type(anomaly_type)


def test_validate_anomaly_type_names_the_field_it_rejected():
    with pytest.raises(ValueError, match="defect_type"):
        validate_anomaly_type("../escape+x", field="defect_type")


def test_build_caption_normalises_separators():
    caption = build_caption(defect="missing-solder", texture="pcb_board")
    assert "missing solder" in caption
    assert "pcb board" in caption
    assert caption == CAPTION_TEMPLATE.format(defect="missing solder", texture="pcb board")


class _FakeTokenizer:
    """Minimal tokenizer stub exercising both resolve_word_token_id branches."""

    def __init__(self, vocab, encode_map, unk_token_id=0):
        self._vocab = vocab
        self._encode_map = encode_map
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def encode(self, word, add_special_tokens=False):
        return self._encode_map.get(word, [])


def test_resolve_word_token_id_direct_vocab_hit():
    tok = _FakeTokenizer(vocab={"anomaly": 42}, encode_map={}, unk_token_id=0)
    assert resolve_word_token_id(tok, "anomaly") == 42


def test_resolve_word_token_id_falls_back_to_last_piece():
    # Not a standalone vocab entry (maps to unk) -> use the last BPE piece.
    tok = _FakeTokenizer(vocab={}, encode_map={"anomaly": [11, 12, 13]}, unk_token_id=0)
    assert resolve_word_token_id(tok, "anomaly") == 13


def test_resolve_word_token_id_unresolvable_raises():
    tok = _FakeTokenizer(vocab={}, encode_map={"anomaly": []}, unk_token_id=0)
    with pytest.raises(ValueError):
        resolve_word_token_id(tok, "anomaly")


def test_pad_or_truncate_pads_and_truncates():
    assert pad_or_truncate([1, 2], length=4, pad_token_id=9) == [1, 2, 9, 9]
    assert pad_or_truncate([1, 2, 3, 4, 5], length=3, pad_token_id=9) == [1, 2, 3]
    assert pad_or_truncate([1, 2, 3], length=3, pad_token_id=9) == [1, 2, 3]


def _touch(path, data=b"x"):
    path.write_bytes(data)


def test_list_image_mask_pairs_missing_dir_returns_empty():
    assert list_image_mask_pairs("/definitely/not/a/dir", "/nope") == []


def test_list_image_mask_pairs_pairs_and_fallback(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Two images (sorted), plus a non-image that must be ignored.
    _touch(img_dir / "b.jpg")
    _touch(img_dir / "a.png")
    _touch(img_dir / "notes.txt")

    # a: same-extension mask exists; b: only the .png fallback exists.
    _touch(mask_dir / "a.png")
    _touch(mask_dir / "b.png")

    pairs = list_image_mask_pairs(str(img_dir), str(mask_dir))

    assert [p[0] for p in pairs] == [str(img_dir / "a.png"), str(img_dir / "b.jpg")]
    assert pairs[0][1] == str(mask_dir / "a.png")
    assert pairs[1][1] == str(mask_dir / "b.png")  # jpg image -> .png mask fallback


def test_list_image_mask_pairs_returns_missing_mask_path(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    _touch(img_dir / "x.png")  # no corresponding mask on disk

    pairs = list_image_mask_pairs(str(img_dir), str(mask_dir))
    assert len(pairs) == 1
    # Path is returned even though the mask does not exist (falls back to .png form).
    assert pairs[0][1] == str(mask_dir / "x.png")


def test_list_image_mask_pairs_honours_suffix(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    _touch(img_dir / "a.png")
    _touch(mask_dir / "a_mask.png")

    pairs = list_image_mask_pairs(str(img_dir), str(mask_dir), mask_suffix="_mask")
    assert pairs[0][1] == str(mask_dir / "a_mask.png")


def test_list_images_matches_the_suffix_case_insensitively_and_sorts_by_name(tmp_path):
    """The ONE dataset-directory lister (clean pools, submask sources): png/jpg/jpeg/bmp by lower-cased
    suffix — `c.JPG` is an image — in name order, non-images left out."""
    for name in ("b.png", "a.jpeg", "C.JPG", "d.BMP", "notes.txt", "e.tiff"):
        _touch(tmp_path / name)
    listed = list_images(tmp_path)
    assert [p.name for p in listed] == ["C.JPG", "a.jpeg", "b.png", "d.BMP"], listed
    assert all(p.parent == tmp_path for p in listed)
    assert list_images(str(tmp_path)) == listed  # a str path is taken like a Path


def test_list_images_of_a_missing_dir_or_a_file_is_empty(tmp_path):
    _touch(tmp_path / "a.png")
    assert list_images(tmp_path / "nope") == []
    assert list_images(tmp_path / "a.png") == []


# --- require_same_size --------------------------------------------------------------------------


def _rgb(w, h):
    return Image.fromarray(np.zeros((h, w, 3), np.uint8), mode="RGB")


def _l(w, h):
    return Image.fromarray(np.zeros((h, w), np.uint8), mode="L")


def test_require_same_size_accepts_a_matching_pair():
    assert require_same_size(_rgb(32, 48), _l(32, 48)) is None


@pytest.mark.parametrize("mask_size", [(16, 48), (32, 24), (64, 96), (33, 48)])
def test_require_same_size_rejects_any_disagreement(mask_size):
    # (64, 96) is an exact 2x: a proportional mask is still wrong, the planner compares raw pixels.
    with pytest.raises(ValueError, match="image/mask size mismatch"):
        require_same_size(_rgb(32, 48), _l(*mask_size))


def test_require_same_size_names_the_pair_and_both_sizes():
    with pytest.raises(ValueError) as excinfo:
        require_same_size(_rgb(32, 48), _l(16, 48), "img.png + mask.png")
    msg = str(excinfo.value)
    assert "img.png + mask.png" in msg and "32x48" in msg and "16x48" in msg


def test_require_same_size_transposed_dims_are_rejected():
    # Same pixel count, so any area-based check would pass it.
    with pytest.raises(ValueError, match="image/mask size mismatch"):
        require_same_size(_rgb(64, 32), _l(32, 64))


# --- build_source_item --------------------------------------------------------------------------


def _base_and_mask():
    """A CPU ``([3,1,H,W] base, stride-0 expanded m3)`` pair shaped exactly as inpaint.py builds it."""
    base = torch.linspace(-1.0, 1.0, 3 * 1 * 4 * 4).reshape(3, 1, 4, 4)
    m = torch.zeros(4, 4)
    m[1:3, 1:3] = 1.0
    m3 = m[None, None].expand(base.shape)  # stride-0 view, not contiguous
    return base, m3


def test_build_source_item_keeps_background_and_noises_defect():
    base, m3 = _base_and_mask()
    clone = base.clone()

    out = build_source_item(base, m3, seed=7)

    inside = m3 > 0
    assert torch.equal(out[~inside], base[~inside])  # background is bit-exact clean pixels
    assert (out[inside] >= -1.0).all() and (out[inside] <= 1.0).all()
    assert not torch.any(out[inside] == -1.0)  # noise, not the dropout constant
    assert torch.equal(base, clone)  # base never mutated


def test_build_source_item_background_dropout_blacks_out_background():
    base, m3 = _base_and_mask()
    clone = base.clone()

    out = build_source_item(base, m3, True, seed=7)

    inside = m3 > 0
    assert torch.all(out[~inside] == -1.0)  # background dropped to black
    assert (out[inside] >= -1.0).all() and (out[inside] <= 1.0).all()
    assert not torch.any(out[inside] == -1.0)
    assert torch.equal(base, clone)


def test_build_source_item_same_seed_is_bit_identical():
    base, m3 = _base_and_mask()
    assert torch.equal(build_source_item(base, m3, seed=123), build_source_item(base, m3, seed=123))


def test_build_source_item_different_seed_gives_different_noise():
    base, m3 = _base_and_mask()
    a = build_source_item(base, m3, seed=123)
    b = build_source_item(base, m3, seed=124)
    assert not torch.equal(a, b)


def test_build_source_item_without_seed_stays_stochastic():
    """The training path passes no seed and must draw fresh noise on every call."""
    base, m3 = _base_and_mask()
    assert not torch.equal(build_source_item(base, m3), build_source_item(base, m3))


def test_build_source_item_seed_does_not_advance_global_rng():
    """A seeded draw uses a private generator, so it must not perturb the global RNG stream."""
    base, m3 = _base_and_mask()
    torch.manual_seed(0)
    expected = torch.rand(4)

    torch.manual_seed(0)
    build_source_item(base, m3, seed=99)
    assert torch.equal(torch.rand(4), expected)


# --------------------------------------------------------------------------------------------------
# ragged batch collation — collate_with_ragged
# --------------------------------------------------------------------------------------------------


def _sample(h, w, class_id=0):
    """One dataset item in the shape ``InpaintingDataset.__getitem__`` emits."""
    rng = np.random.default_rng((h, w, class_id))
    img = torch.from_numpy(rng.random((3, h, w), dtype=np.float32))
    mask = torch.zeros(1, h, w)
    mask[:, : h // 2, : w // 2] = 1.0
    return {
        "images": [img, img.clone()],
        "text_token_ids": torch.zeros(4, dtype=torch.long),
        "ai_caption": "a scratch on metal",
        "image_size": [torch.tensor([h, w, h, w], dtype=torch.float32)] * 2,
        "fps": 30.0,
        "num_frames": 2,
        "sequence_plan": SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]),
        "edit_mask": mask,
        "anomaly_class_id": class_id,
        "dataset_name": "anomaly",
    }


def _assert_same_except_ragged(got, ref):
    assert set(got) == set(ref)
    for key in ref:
        if key in RAGGED_KEYS:
            continue
        a, b = got[key], ref[key]
        if isinstance(b, torch.Tensor):
            assert torch.equal(a, b), key
        else:
            assert a == b, key


def test_ragged_batch_keeps_the_mask_as_a_per_sample_list():
    # The case that made this function necessary: two input sizes in one batch, which default_collate
    # cannot stack.
    batch = [_sample(64, 128), _sample(256, 64, class_id=1)]
    out = anomalygen_collate_fn(batch)

    assert isinstance(out["edit_mask"], list) and len(out["edit_mask"]) == 2
    assert [tuple(m.shape) for m in out["edit_mask"]] == [(1, 64, 128), (1, 256, 64)]
    # The masks are the sample tensors themselves, not copies pushed through a transform.
    assert torch.equal(out["edit_mask"][0], batch[0]["edit_mask"])
    # Non-ragged keys still collate normally.
    assert out["anomaly_class_id"].tolist() == [0, 1]
    assert len(out["images"]) == 2


def test_ragged_batch_raises_under_the_framework_collate():
    # Guards the premise: if the framework ever list-collated edit_mask itself this whole module
    # would be dead code.
    batch = [_sample(64, 128), _sample(256, 64)]
    with pytest.raises(RuntimeError):
        custom_collate_fn(batch)


def test_uniform_batch_matches_the_framework_collate_on_every_other_key():
    # A uniform batch must still match the framework's own collate: same keys, same values, only
    # edit_mask differs.
    batch = [_sample(64, 64), _sample(64, 64, class_id=1)]
    out = anomalygen_collate_fn(batch)
    ref = custom_collate_fn([{k: v for k, v in s.items() if k != "edit_mask"} for s in batch])
    ref["edit_mask"] = torch.stack([s["edit_mask"] for s in batch])

    _assert_same_except_ragged(out, ref)
    # Uniform or not, the mask arrives as a list; the model accepts either form.
    assert torch.equal(torch.stack(out["edit_mask"]), ref["edit_mask"])

    # A batch carrying no ragged key at all must be the framework's own result, untouched.
    plain = [{k: v for k, v in s.items() if k != "edit_mask"} for s in batch]
    _assert_same_except_ragged(anomalygen_collate_fn(plain), custom_collate_fn(plain))


def test_input_samples_are_not_mutated():
    # The loader may reuse the sample dicts; stripping the ragged keys must not empty them.
    batch = [_sample(64, 64), _sample(128, 64)]
    anomalygen_collate_fn(batch)
    assert all("edit_mask" in s for s in batch)
