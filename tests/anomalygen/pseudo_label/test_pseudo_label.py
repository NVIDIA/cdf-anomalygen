# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pseudo-labeling COCO conversion + captioner helpers.

The COCO/mask/bbox logic is model-free and tested directly with real inputs (synthetic masks and a
synthetic generate.py output tree). The caption string helpers (placeholder substitution, ``<answer>``
trimming, response formatting) are tested without loading any VLM. The full captioner forward is a
single ``@pytest.mark.gpu`` test gated on a local Cosmos3-Nano checkpoint.
"""

import json
import os
import types

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from transformers import LogitsProcessorList, TopKLogitsWarper

from anomalygen.pseudo_label import (
    DEFAULT_CAPTION_PROMPT_PATH,
    Captioner,
    binary_mask_to_rle,
    coco_decode_rle,
    coco_encode_rle,
    format_response,
    get_bboxes,
    visualize,
)
from anomalygen.pseudo_label import caption as caption_mod
from anomalygen.pseudo_label.caption import _UsableLogitsGuard
from anomalygen.scripts.texture import pseudo_label

_SUBDIRS = ("reconstructed_image", "original_image", "original_mask")


def _square_mask(size, x0, y0, side, fill=255):
    """L-mode mask with a single filled square."""
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[y0 : y0 + side, x0 : x0 + side] = fill
    return Image.fromarray(arr, mode="L")


# --------------------------------------------------------------------------- bbox / RLE


def test_get_bboxes_xywh_and_xyxy():
    mask = _square_mask(64, 10, 20, 15)  # x0=10, y0=20, side=15
    assert get_bboxes(mask, format="xywh") == [(10, 20, 15, 15)]
    assert get_bboxes(mask, format="xyxy") == [(10, 20, 25, 35)]


def test_get_bboxes_empty_mask_is_none():
    empty = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
    assert get_bboxes(empty) == [None]


def test_get_bboxes_rejects_bad_format():
    with pytest.raises(ValueError):
        get_bboxes(_square_mask(16, 1, 1, 4), format="cxcywh")


@pytest.mark.parametrize(
    "pattern",
    [
        np.zeros((8, 8), dtype=bool),
        np.ones((8, 8), dtype=bool),
        np.eye(8, dtype=bool),
        np.array([[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0]], dtype=bool),
    ],
)
def test_rle_roundtrip(pattern):
    coco_rle = coco_encode_rle(binary_mask_to_rle(pattern))
    assert isinstance(coco_rle["counts"], str)  # json-serializable
    np.testing.assert_array_equal(coco_decode_rle(coco_rle).astype(bool), pattern)


def test_binary_mask_to_rle_validates_input():
    with pytest.raises(ValueError):
        binary_mask_to_rle(np.zeros((4, 4), dtype=np.uint8))  # not bool
    with pytest.raises(ValueError):
        binary_mask_to_rle(np.zeros((2, 2, 2), dtype=bool))  # not 2D


# --------------------------------------------------------------------------- annotation dict


def test_compute_annotation_dict_exact_values():
    mask = _square_mask(50, 5, 6, 10)  # 10x10 square at (5, 6)
    ann = pseudo_label.compute_annotation_dict(annotation_id=7, image_id=3, category_id=2, instance_mask=mask)
    assert ann["id"] == 7
    assert ann["image_id"] == 3
    assert ann["category_id"] == 2
    assert ann["bbox"] == (5, 6, 10, 10)
    assert ann["area"] == 100
    assert ann["iscrowd"] == 0
    # Segmentation decodes back to the binary mask.
    np.testing.assert_array_equal(coco_decode_rle(ann["segmentation"]).astype(bool), np.array(mask) > 127)


# --------------------------------------------------------------------------- caption string logic


def _dummy_captioner():
    return Captioner(prompt_data={"system_prompt": "sys {image_type}", "user_prompt": "u {anomaly_type} {bboxes}"})


def test_replace_placeholders():
    cap = _dummy_captioner()
    meta = {"image_type": "wood", "anomaly_type": "scratch", "bboxes": "(1, 2, 3, 4)"}
    assert cap.replace_placeholders(cap.system_prompt, meta) == "sys wood"
    assert cap.replace_placeholders(cap.user_prompt, meta) == "u scratch (1, 2, 3, 4)"


def test_postprocess_response_trims_to_num_bboxes():
    cap = _dummy_captioner()
    resp = "<answer>\ncap\n\n**Anomaly 1:** a\n\n**Anomaly 2:** b\n\n**Anomaly 3:** c\n</answer>"
    out = cap.postprocess_response(resp, num_bboxes=2)
    assert "**Anomaly 1:** a" in out and "**Anomaly 2:** b" in out
    assert "**Anomaly 3:** c" not in out


def test_postprocess_response_without_tags_is_passthrough():
    cap = _dummy_captioner()
    assert cap.postprocess_response("no tags here", num_bboxes=1) == "no tags here"


def test_format_response_strips_tags_and_adds_meta():
    meta = {"anomaly_type": "scratch", "num_bboxes": 1}
    response, response_with_meta = format_response("<answer>\nhello\n</answer>", meta)
    assert response == "hello"
    assert response_with_meta.endswith("hello")
    assert "Meta:" in response_with_meta
    assert json.dumps(meta, indent=4) in response_with_meta


# --------------------------------------------------------------------------- end-to-end (no caption)


def _write_gen_tree(gen_root, keys_per_image):
    """Build a synthetic generate.py output tree; one single-square mask per image."""
    for sub in _SUBDIRS:
        os.makedirs(os.path.join(gen_root, sub), exist_ok=True)
    rows = ["output_filename,anomaly_type"]
    for idx, key in enumerate(keys_per_image):
        name = f"{key}_{idx:05d}.png"
        Image.new("RGB", (64, 64), (120, 120, 120)).save(os.path.join(gen_root, "reconstructed_image", name))
        Image.new("RGB", (64, 64), (10, 10, 10)).save(os.path.join(gen_root, "original_image", name))
        _square_mask(64, 12, 12, 16).save(os.path.join(gen_root, "original_mask", name))
        rows.append(f"{name},{key}")
    with open(os.path.join(gen_root, "texture_ft_generation_result.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


def test_end_to_end_no_caption(tmp_path):
    gen_root = tmp_path / "gen"
    keys = ["wood+scratch", "wood+scratch", "tile+crack"]
    _write_gen_tree(str(gen_root), keys)
    out_dir = tmp_path / "out"

    pseudo_label.main(["--gen_root", str(gen_root), "--output_dir", str(out_dir), "--no_caption"])

    # COCO json: one image per file, one annotation per single-square mask, two categories.
    coco = json.loads((out_dir / "coco_annotations.json").read_text())
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3
    cats = {c["name"]: c["id"] for c in coco["categories"]}
    assert set(cats) == {"tile+crack", "wood+scratch"}
    assert cats["tile+crack"] == 1 and cats["wood+scratch"] == 2  # sorted -> deterministic ids
    # Every annotation's category matches its image's anomaly type.
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_cat = {v: k for k, v in cats.items()}
    for ann in coco["annotations"]:
        assert id_to_cat[ann["category_id"]] == id_to_file[ann["image_id"]].rsplit("_", 1)[0]
        assert ann["area"] == 16 * 16

    # Side outputs.
    assert len(list((out_dir / "images").glob("*.png"))) == 3
    assert len(list((out_dir / "masks").glob("*.png"))) == 3
    assert len(list((out_dir / "visualization").glob("*.png"))) == 3
    assert not (out_dir / "captions").exists()  # --no_caption

    # Classification layout: classes.txt starts with "original", per-class dirs populated.
    classes = (out_dir / "classification" / "classes.txt").read_text().splitlines()
    assert classes[0] == "original"
    assert set(classes[1:]) == {"tile+crack", "wood+scratch"}
    assert len(list((out_dir / "classification" / "original").glob("*.png"))) == 3  # originals
    assert len(list((out_dir / "classification" / "wood+scratch").glob("*.png"))) == 2
    assert len(list((out_dir / "classification" / "tile+crack").glob("*.png"))) == 1


def test_contained_path_refuses_a_segment_that_escapes_the_classification_dir(tmp_path):
    """A class name resolving outside classification/ is refused even if it arrived unvalidated."""
    classification_dir = tmp_path / "classification"
    classification_dir.mkdir()
    with pytest.raises(ValueError, match="outside the classification directory"):
        pseudo_label._contained_path(classification_dir, "../../escape")


def test_contained_path_returns_the_class_subdirectory_for_a_normal_class(tmp_path):
    classification_dir = tmp_path / "classification"
    classification_dir.mkdir()
    assert pseudo_label._contained_path(classification_dir, "wood+scratch") == classification_dir / "wood+scratch"


def test_contained_path_returns_the_path_it_checked_not_the_spelling_it_was_given(tmp_path):
    """The returned path is the resolved one, so nothing re-walks a symlink between check and write."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert pseudo_label._contained_path(link, "wood+scratch") == real / "wood+scratch"


def test_contained_path_refuses_a_filename_that_escapes(tmp_path):
    """The guard covers the whole join, not only the class segment."""
    classification_dir = tmp_path / "classification"
    classification_dir.mkdir()
    with pytest.raises(ValueError, match="outside the classification directory"):
        pseudo_label._contained_path(classification_dir, "wood+scratch", "../../evil.png")


def test_contained_path_returns_the_resolved_write_target(tmp_path):
    classification_dir = tmp_path / "classification"
    classification_dir.mkdir()
    target = pseudo_label._contained_path(classification_dir, "wood+scratch", "img.png")
    assert target == classification_dir / "wood+scratch" / "img.png"


@pytest.mark.parametrize(
    "hostile, escape_relative_to_tmp",
    [
        # out/classification/../../escape+x -> <tmp_path>/escape+x
        ("../../escape+x", "escape+x"),
        # An absolute segment DISCARDS the base, so it lands wherever it says.
        (None, None),
    ],
    ids=["relative-traversal", "absolute"],
)
def test_end_to_end_rejects_a_path_bearing_anomaly_type(tmp_path, hostile, escape_relative_to_tmp):
    """anomaly_type comes from a manifest and becomes a directory name, so a path-bearing value must
    be rejected before any directory is created or any byte written."""
    # Targeted under tmp_path so the test never touches a real system directory.
    if hostile is None:
        escaped = tmp_path / "abs_escape+x"
        hostile = str(escaped)
    else:
        escaped = tmp_path / escape_relative_to_tmp

    gen_root = tmp_path / "gen"
    _write_gen_tree(str(gen_root), ["wood+scratch"])
    # Poison only the anomaly_type column: nothing downstream re-derives the type from the filename.
    csv_path = gen_root / "texture_ft_generation_result.csv"
    name = csv_path.read_text().splitlines()[1].split(",")[0]
    csv_path.write_text(f"output_filename,anomaly_type\n{name},{hostile}\n")
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="anomaly type"):
        pseudo_label.main(["--gen_root", str(gen_root), "--output_dir", str(out_dir), "--no_caption"])

    assert not escaped.exists(), f"pseudo-labeling wrote outside the output dir: {escaped}"


def test_end_to_end_length_mismatch_raises(tmp_path):
    gen_root = tmp_path / "gen"
    _write_gen_tree(str(gen_root), ["wood+scratch", "wood+scratch"])
    # Drop one mask so the counts disagree.
    masks = sorted((gen_root / "original_mask").glob("*.png"))
    masks[0].unlink()
    with pytest.raises(ValueError, match="original images and masks must be the same"):
        pseudo_label.main(["--gen_root", str(gen_root), "--output_dir", str(tmp_path / "out"), "--no_caption"])


def test_end_to_end_multi_instance_and_max_instances(tmp_path):
    """A two-blob mask yields one COCO annotation per connected component (via split_mask_into_instances);
    --max_instances 1 merges them into a single instance."""
    gen_root = tmp_path / "gen"
    for sub in _SUBDIRS:
        os.makedirs(gen_root / sub)
    name = "wood+scratch_00000.png"
    Image.new("RGB", (96, 96), (120, 120, 120)).save(gen_root / "reconstructed_image" / name)
    Image.new("RGB", (96, 96), (10, 10, 10)).save(gen_root / "original_image" / name)
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[8:20, 8:20] = 255  # blob A
    arr[76:88, 76:88] = 255  # blob B
    Image.fromarray(arr, mode="L").save(gen_root / "original_mask" / name)
    (gen_root / "texture_ft_generation_result.csv").write_text(f"output_filename,anomaly_type\n{name},wood+scratch\n")

    # Default max_instances: one annotation per connected component.
    out_multi = tmp_path / "out_multi"
    pseudo_label.main(["--gen_root", str(gen_root), "--output_dir", str(out_multi), "--no_caption"])
    coco = json.loads((out_multi / "coco_annotations.json").read_text())
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert all(ann["area"] == 12 * 12 for ann in coco["annotations"])

    # --max_instances 1: the two blobs collapse into a single instance mask.
    out_one = tmp_path / "out_one"
    pseudo_label.main(
        ["--gen_root", str(gen_root), "--output_dir", str(out_one), "--no_caption", "--max_instances", "1"]
    )
    coco_one = json.loads((out_one / "coco_annotations.json").read_text())
    assert len(coco_one["annotations"]) == 1


def test_caption_failure_is_contained_but_still_fails_the_run(tmp_path, monkeypatch):
    """Captioning has no resume, so one unusable sample must not discard the captions already
    written — and the run must still exit non-zero so a short label set is not mistaken for a
    complete one."""
    gen_root = tmp_path / "gen"
    _write_gen_tree(str(gen_root), ["wood+scratch", "wood+scratch", "tile+crack"])
    out_dir = tmp_path / "out"

    class _CaptionerFailingOnSecond:
        def __init__(self, **_kwargs):
            self.seen = 0

        def generate_caption(self, _ori_path, _mask_path, gen_image_path, _meta):
            self.seen += 1
            if self.seen == 2:
                raise RuntimeError("model produced unusable logits at decode step 7")
            return f"a caption for {gen_image_path.name}"

    monkeypatch.setattr(pseudo_label.pl_caption, "Captioner", _CaptionerFailingOnSecond)

    with pytest.raises(RuntimeError, match="1 of 3 images could not be captioned"):
        pseudo_label.main(["--gen_root", str(gen_root), "--output_dir", str(out_dir)])

    assert len(list((out_dir / "captions").glob("*.txt"))) == 2
    assert len(list((out_dir / "captions_with_meta").glob("*.txt"))) == 2
    # Everything upstream of captioning is untouched by the failure.
    assert len(json.loads((out_dir / "coco_annotations.json").read_text())["images"]) == 3


# --------------------------------------------------------------------------- captioner forward (gated)

_COSMOS3_NANO = os.path.join("checkpoints", "nvidia", "Cosmos3-Nano", "config.json")


_CAPTION_META = {"image_type": "wood", "anomaly_type": "scratch", "bboxes": "(12, 12, 28, 28)", "num_bboxes": 1}


def _caption_triple(tmp_path):
    """The (clean, mask, generated) paths the captioner expects."""
    ori = tmp_path / "ori.png"
    mask = tmp_path / "mask.png"
    gen = tmp_path / "gen.png"
    Image.new("RGB", (64, 64), (10, 10, 10)).save(ori)
    _square_mask(64, 12, 12, 16).save(mask)
    Image.new("RGB", (64, 64), (200, 60, 60)).save(gen)
    return ori, mask, gen


@pytest.mark.gpu
@pytest.mark.skipif(not os.path.exists(_COSMOS3_NANO), reason="requires local checkpoints/nvidia/Cosmos3-Nano")
def test_captioner_end_to_end(tmp_path):
    ori, mask, gen = _caption_triple(tmp_path)
    prompt_data = yaml.safe_load(DEFAULT_CAPTION_PROMPT_PATH.read_text())
    # The cap has to leave room for a caption to finish; a decode that runs into it is rejected as a
    # repetition loop rather than written out, which is what the next test covers.
    captioner = Captioner(prompt_data=prompt_data, max_new_tokens=512)
    response = captioner.generate_caption(ori, mask, gen, _CAPTION_META)
    assert isinstance(response, str) and response.strip()


@pytest.mark.gpu
@pytest.mark.skipif(not os.path.exists(_COSMOS3_NANO), reason="requires local checkpoints/nvidia/Cosmos3-Nano")
def test_captioner_rejects_a_decode_that_runs_into_the_cap(tmp_path):
    """The whole-caption check against the real model rather than a stub: 32 tokens is far too few
    for a caption, so the decode cannot reach a stop token and must not reach the label file."""
    ori, mask, gen = _caption_triple(tmp_path)
    prompt_data = yaml.safe_load(DEFAULT_CAPTION_PROMPT_PATH.read_text())
    captioner = Captioner(prompt_data=prompt_data, max_new_tokens=32)
    with pytest.raises(RuntimeError, match="without emitting a stop token"):
        captioner.generate_caption(ori, mask, gen, _CAPTION_META)


# ``postprocess_response`` must fall back to the raw response when the ``<answer>``/``</answer>`` tags
# are missing or inverted. str.find/rfind return -1 rather than raising, so the pre-fix try/except was
# dead code and a tag-less response produced a corrupted slice instead of the raw text.


def test_caption_missing_both_tags_returns_raw():
    cap = _dummy_captioner()
    # Contains the "\n\n**Anomaly " delimiter so the pre-fix code did not early-return; it instead
    # sliced response[-1 + len(start_tag):-1] and re-wrapped it, corrupting the text.
    raw = "A clean caption sentence.\n\n**Anomaly 1**: a small scratch"
    assert cap.postprocess_response(raw, num_bboxes=1) == raw


def test_caption_missing_end_tag_returns_raw():
    cap = _dummy_captioner()
    raw = "<answer>A clean caption.\n\n**Anomaly 1**: a small scratch"
    assert cap.postprocess_response(raw, num_bboxes=1) == raw


def test_caption_end_before_start_returns_raw():
    cap = _dummy_captioner()
    raw = "</answer> stuff <answer>"
    assert cap.postprocess_response(raw, num_bboxes=1) == raw


def test_caption_wellformed_extracts_and_truncates():
    cap = _dummy_captioner()
    raw = "pre <answer>CAP\n\n**Anomaly 1**: foo\n\n**Anomaly 2**: bar</answer> post"
    out = cap.postprocess_response(raw, num_bboxes=1)
    assert out != raw
    assert "CAP" in out
    assert "**Anomaly 1**: foo" in out
    assert "bar" not in out  # second anomaly dropped (num_bboxes=1)


# ``visualize`` must anchor the label at (0, 0) when an instance has ``bbox=None`` instead of raising a
# NameError (x/y undefined on the first iteration) or silently reusing the previous instance's coords.


def test_visualize_bbox_none_first_instance_no_crash():
    # bbox None on the FIRST instance with a class name: pre-fix -> NameError.
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    out = visualize(img, classes=["defect"], bboxes=[None], instance_masks=[_square_mask(32, 4, 4, 8)])
    assert isinstance(out, Image.Image)


def test_visualize_mixed_bbox_then_none_completes():
    # Second instance bbox None: pre-fix silently reused the first bbox's coords.
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    out = visualize(
        img,
        classes=["a", "b"],
        bboxes=[(2, 2, 10, 10), None],
        instance_masks=[_square_mask(32, 4, 4, 8), _square_mask(32, 4, 4, 8)],
    )
    assert isinstance(out, Image.Image)


def test_visualize_bbox_present_still_draws():
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    out = visualize(img, classes=["a"], bboxes=[(1, 1, 8, 8)], instance_masks=[_square_mask(32, 4, 4, 8)])
    assert isinstance(out, Image.Image)


# --------------------------------------------------------------------------- decoding defaults + logits guard
#
# Two independent properties of the captioner's decoding contract.
#
# 1. Decoding is GREEDY by default. ``argmax`` never reaches ``torch.multinomial``, the operation
#    that aborts on a malformed distribution, and the largest logit survives numerical differences
#    between GPUs and library versions that would shift a sampled draw.
# 2. Unusable logits FAIL LOUDLY on either decoding path. Greedy would otherwise pick an arbitrary
#    index off a nan tensor and emit a quietly wrong caption, which for label data is worse than no
#    caption at all; sampling instead trips a device-side assert inside ``torch.multinomial`` that
#    points at the sampler rather than at the cause. "Unusable" excludes deliberate ``-inf``
#    masking, which stock processors write on purpose — see the guard's docstring.
#
# These run on CPU with a stub model: the point is the decoding contract, not the weights.


class _FakeInputs(dict):
    """Stands in for the processor's BatchFeature: dict-unpackable, ``.to()``-able, has input_ids."""

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, *_args, **_kwargs):
        return self


class _FakeProcessor:
    def apply_chat_template(self, *_a, **_k):
        return "prompt"

    def __call__(self, *_a, **_k):
        return _FakeInputs(torch.zeros((1, 3), dtype=torch.long))

    def batch_decode(self, *_a, **_k):
        return ["decoded"]


_EOS = 151645  # one of Cosmos3-Nano's stop tokens; the exact value only has to be self-consistent here
_PROMPT_LEN = 3  # matches the input_ids _FakeProcessor hands back


class _FakeModel:
    """Records the kwargs ``generate`` was called with, and optionally runs the logits processors."""

    device = "cpu"

    def __init__(self, scores=None, output_ids=None):
        self.calls = []
        self._scores = scores
        self._output_ids = output_ids
        self.generation_config = types.SimpleNamespace(eos_token_id=[_EOS])

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self._scores is not None:
            for proc in kwargs.get("logits_processor") or []:
                proc(torch.zeros((1, _PROMPT_LEN), dtype=torch.long), self._scores)
        if self._output_ids is not None:
            return self._output_ids
        # Two new tokens, well short of any cap under test, ending on a stop token.
        return torch.tensor([[0] * _PROMPT_LEN + [7, _EOS]], dtype=torch.long)


def _generated(new_tokens, *, ends_on_stop):
    """A ``generate`` return of ``new_tokens`` new tokens, terminating on a stop token or not."""
    tokens = [7] * new_tokens
    if ends_on_stop:
        tokens[-1] = _EOS
    return torch.tensor([[0] * _PROMPT_LEN + tokens], dtype=torch.long)


_USE_DEFAULT = object()  # so a test can exercise Captioner's own default rather than restate it


def _stub_captioner(monkeypatch, temperature=_USE_DEFAULT, scores=None, output_ids=None, max_new_tokens=None):
    monkeypatch.setattr(caption_mod, "process_vision_info", lambda _m: ([], []))
    kwargs = {} if temperature is _USE_DEFAULT else {"temperature": temperature}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    cap = Captioner(prompt_data={"system_prompt": "s", "user_prompt": "u"}, **kwargs)
    cap.processor = _FakeProcessor()
    cap.model = _FakeModel(scores=scores, output_ids=output_ids)
    return cap


def test_captioner_default_temperature_is_greedy():
    assert _dummy_captioner().temperature == 0.0


def test_cli_captioner_temperature_defaults_to_zero():
    args = pseudo_label._get_args(["--gen_root", "g", "--output_dir", "o"])
    assert args.captioner_temperature == 0.0


@pytest.mark.parametrize(
    "temperature, expected",
    [
        # 0 must disable sampling outright rather than pass temperature=0 to generate(), which
        # would divide the logits by zero.
        (0.0, {"do_sample": False}),
        (0.7, {"do_sample": True, "temperature": 0.7}),
        # None is the explicit opt-out: defer to the model's own generation_config.
        (None, {}),
    ],
)
def test_generation_kwargs_follow_temperature(monkeypatch, temperature, expected):
    cap = _stub_captioner(monkeypatch, temperature=temperature)
    cap._generate_raw([{"role": "user", "content": []}])
    call = cap.model.calls[0]
    for key, value in expected.items():
        assert call[key] == value
    if not expected:
        assert "do_sample" not in call and "temperature" not in call


def test_default_decoding_never_samples(monkeypatch):
    """The default path must not reach torch.multinomial — that is the operation that asserted."""
    cap = _stub_captioner(monkeypatch)
    cap._generate_raw([{"role": "user", "content": []}])
    assert cap.model.calls[0]["do_sample"] is False


@pytest.mark.parametrize("temperature", [0.0, 0.7, None])
def test_logits_guard_is_installed_on_every_decoding_path(monkeypatch, temperature):
    """Installed regardless of temperature: greedy and sampling fail differently, but both badly."""
    cap = _stub_captioner(monkeypatch, temperature=temperature)
    cap._generate_raw([{"role": "user", "content": []}])
    processors = cap.model.calls[0]["logits_processor"]
    assert [type(p) for p in processors] == [_UsableLogitsGuard]


def test_logits_guard_passes_finite_scores_through():
    scores = torch.randn(1, 16)
    assert torch.equal(_UsableLogitsGuard()(None, scores), scores)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_logits_guard_raises_on_nan_or_positive_inf(bad):
    """These two are never legitimate, whatever else is in the processor chain."""
    scores = torch.randn(1, 16)
    scores[0, 5] = bad
    with pytest.raises(RuntimeError, match="unusable logits"):
        _UsableLogitsGuard("sample_00000.png")(None, scores)


def test_logits_guard_allows_partially_masked_rows():
    """A -inf marks a *masked* token. Only a row with nothing left is a defect."""
    scores = torch.randn(1, 16)
    scores[0, 3:] = float("-inf")
    assert torch.equal(_UsableLogitsGuard("sample_00000.png")(None, scores), scores)


def test_logits_guard_raises_when_a_row_is_fully_masked():
    """Nothing left to pick: argmax returns an arbitrary index and multinomial asserts."""
    scores = torch.randn(2, 16)
    scores[1, :] = float("-inf")
    with pytest.raises(RuntimeError, match="unusable logits"):
        _UsableLogitsGuard("sample_00000.png")(None, scores)


def test_logits_guard_survives_a_real_masking_processor():
    """Order-independence, against transformers rather than a stub.

    ``transformers`` merges custom processors with a ``TODO`` about ordering, so the guard can end
    up either side of top-k/top-p. Running it *after* a real ``TopKLogitsWarper`` — the placement
    that would break a plain ``isfinite`` check, since top-k leaves every non-surviving score at
    ``-inf`` — must still pass the masked scores through untouched.
    """
    scores = torch.randn(1, 32)
    chain = LogitsProcessorList([TopKLogitsWarper(top_k=4), _UsableLogitsGuard("sample_00000.png")])
    out = chain(torch.zeros((1, 3), dtype=torch.long), scores)
    assert int(torch.isneginf(out).sum()) == 28


def test_logits_guard_error_names_the_sample_and_step():
    guard = _UsableLogitsGuard("IC+bridge_00000.png")
    guard(None, torch.zeros(1, 4))  # step 1 is clean
    with pytest.raises(RuntimeError) as excinfo:
        guard(None, torch.tensor([[0.0, float("nan"), 0.0, 0.0]]))
    message = str(excinfo.value)
    assert "IC+bridge_00000.png" in message
    assert "step 2" in message


def test_unusable_logits_abort_generation_greedily(monkeypatch):
    """End of the contract: bad logits raise even when sampling is off, instead of emitting a caption."""
    cap = _stub_captioner(monkeypatch, temperature=0.0, scores=torch.tensor([[0.0, float("nan")]]))
    with pytest.raises(RuntimeError, match="unusable logits"):
        cap._generate_raw([{"role": "user", "content": []}])


# --------------------------------------------------------------------------- whole-caption contract
#
# The logits guard is per decode step, inside generate(). Once generate() returns, nothing between
# there and the label file checks that the caption is *whole*: postprocess_response trims the
# <answer> block but hands an untagged string straight back, so a decode that ran into the token cap
# -- the signature of a repetition loop -- would be written out and counted as a success.


def test_decode_that_hits_the_token_cap_raises(monkeypatch):
    cap = _stub_captioner(monkeypatch, output_ids=_generated(4, ends_on_stop=False), max_new_tokens=4)
    with pytest.raises(RuntimeError, match="without emitting a stop token"):
        cap._generate_raw([{"role": "user", "content": []}], context="IC+bridge_00000.png")


def test_decode_ending_on_a_stop_token_is_whole_even_at_the_cap(monkeypatch):
    """Length alone must not be the signal: a caption may legitimately end on the last allowed token."""
    cap = _stub_captioner(monkeypatch, output_ids=_generated(4, ends_on_stop=True), max_new_tokens=4)
    assert cap._generate_raw([{"role": "user", "content": []}]) == "decoded"


def test_decode_shorter_than_the_cap_is_whole(monkeypatch):
    cap = _stub_captioner(monkeypatch, output_ids=_generated(2, ends_on_stop=False), max_new_tokens=4)
    assert cap._generate_raw([{"role": "user", "content": []}]) == "decoded"


def test_cap_error_names_the_sample(monkeypatch):
    cap = _stub_captioner(monkeypatch, output_ids=_generated(4, ends_on_stop=False), max_new_tokens=4)
    with pytest.raises(RuntimeError) as excinfo:
        cap._generate_raw([{"role": "user", "content": []}], context="IC+bridge_00000.png")
    assert "IC+bridge_00000.png" in str(excinfo.value)


@pytest.mark.parametrize(
    "response, expected_warning",
    [
        ("no tags at all, just prose", "no usable <answer> block"),
        ("<answer>\na caption with no anomaly sections\n</answer>", "no anomaly sections"),
    ],
)
def test_raw_fallback_is_warned_about(monkeypatch, response, expected_warning):
    """The raw fallback stays -- odd-shaped label text beats a corrupted slice -- but it is announced,
    because nothing downstream can tell it apart from a well-formed caption."""
    warnings = []
    monkeypatch.setattr(caption_mod.log, "warning", warnings.append)
    cap = _dummy_captioner()
    assert cap.postprocess_response(response, num_bboxes=1, context="IC+bridge_00000.png") == response
    assert len(warnings) == 1
    assert expected_warning in warnings[0]
    assert "IC+bridge_00000.png" in warnings[0]


def test_well_formed_caption_is_not_warned_about(monkeypatch):
    warnings = []
    monkeypatch.setattr(caption_mod.log, "warning", warnings.append)
    cap = _dummy_captioner()
    cap.postprocess_response("<answer>\ncaption\n\n**Anomaly 1**: a scratch\n</answer>", num_bboxes=1)
    assert warnings == []


# Placement: a model that fits on one GPU stays on one GPU. Sharding it only adds device-to-device
# copies of the hidden state, and a host that corrupts those copies turns them into a handful of
# nan/+inf logits. These tests never touch a GPU: the memory probe and the load are stubbed, so they
# run the same on a CPU-only runner, a single-GPU box and a multi-GPU host.

_WEIGHTS = 1000  # arbitrary unit; the rule is a ratio, not a size
_NEED = int(_WEIGHTS * (1 + caption_mod._PLACEMENT_HEADROOM))  # what one GPU must have free


@pytest.mark.parametrize(
    "free_bytes, expected",
    [
        pytest.param([_NEED], {"": "cuda:0"}, id="single gpu with exactly the headroom"),
        pytest.param([_NEED - 1], "auto", id="single gpu one byte short"),
        pytest.param([_NEED // 2, 10 * _NEED, 2 * _NEED], {"": "cuda:1"}, id="roomiest of several"),
        pytest.param([3 * _NEED, 3 * _NEED, _NEED], {"": "cuda:0"}, id="tie goes to the lowest index"),
        pytest.param([_NEED - 1] * 4, "auto", id="none big enough: split"),
        pytest.param([], "auto", id="no gpu: cpu path"),
    ],
)
def test_device_map_places_on_one_gpu_only_when_it_fits(free_bytes, expected):
    assert caption_mod._device_map(_WEIGHTS, free_bytes) == expected


def test_device_map_defers_to_auto_when_the_size_is_unknown():
    """A hub id has no local weights to measure, so nothing can be promised about fitting."""
    assert caption_mod._device_map(None, [10 * _NEED]) == "auto"


def test_checkpoint_bytes_reads_the_index_metadata(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 12345}}))
    (tmp_path / "shard.safetensors").write_bytes(b"x" * 10)  # ignored while the index has a total
    assert caption_mod._checkpoint_bytes(str(tmp_path)) == 12345


def test_checkpoint_bytes_sums_the_shards_without_an_index(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "a.safetensors").write_bytes(b"x" * 7)
    (tmp_path / "b.safetensors").write_bytes(b"y" * 5)
    (tmp_path / "config.json").write_text("{}")
    assert caption_mod._checkpoint_bytes(str(tmp_path)) == 12


def test_checkpoint_bytes_is_unknown_for_a_hub_id(tmp_path):
    assert caption_mod._checkpoint_bytes("nvidia/Cosmos3-Nano-that-is-not-local") is None
    assert caption_mod._checkpoint_bytes(str(tmp_path)) is None  # a directory with no weights


def _stub_model_load(monkeypatch, tmp_path, free_bytes, hf_device_map=None):
    """Fake the checkpoint, the CUDA memory probe and both ``from_pretrained`` calls.

    ``free_bytes`` has one entry per visible GPU; empty means no CUDA. ``hf_device_map`` is what the
    fake model reports after loading (transformers leaves it unset for a single-device load).
    Returns the kwargs recorded from each model load and the info and warning lines logged.
    """
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": _WEIGHTS}}))
    calls, infos, warnings = [], [], []

    class _Model:
        device = "cuda:0" if free_bytes else "cpu"

    _Model.hf_device_map = hf_device_map

    monkeypatch.setattr(caption_mod, "_resolve_model_path", lambda _id: str(tmp_path))
    monkeypatch.setattr(caption_mod.AutoProcessor, "from_pretrained", staticmethod(lambda _p: "processor"))
    monkeypatch.setattr(
        caption_mod.Cosmos3OmniForConditionalGeneration,
        "from_pretrained",
        staticmethod(lambda path, **kw: (calls.append(kw), _Model())[1]),
    )
    monkeypatch.setattr(caption_mod.torch.cuda, "is_available", lambda: bool(free_bytes))
    monkeypatch.setattr(caption_mod.torch.cuda, "device_count", lambda: len(free_bytes))
    monkeypatch.setattr(caption_mod.device_placement, "free_bytes", lambda: list(free_bytes))
    monkeypatch.setattr(caption_mod.log, "info", infos.append)
    monkeypatch.setattr(caption_mod.log, "warning", warnings.append)
    return calls, infos, warnings


@pytest.mark.parametrize(
    "free_bytes, expected",
    [
        pytest.param([], "auto", id="cpu only"),
        pytest.param([_NEED], {"": "cuda:0"}, id="one gpu that fits"),
        pytest.param([_NEED - 1], "auto", id="one gpu too small"),
        pytest.param([_NEED, 4 * _NEED, 2 * _NEED], {"": "cuda:1"}, id="several gpus: roomiest"),
    ],
)
def test_load_hands_the_placement_to_from_pretrained(monkeypatch, tmp_path, free_bytes, expected):
    calls, infos, warnings = _stub_model_load(monkeypatch, tmp_path, free_bytes)
    cap = _dummy_captioner()
    cap.load()
    assert len(calls) == 1
    assert calls[0]["device_map"] == expected
    assert cap.processor == "processor"
    # The decision is logged once with the versions it has to be triaged against.
    assert len(infos) == 1
    assert f"device_map={expected}" in infos[0] and "transformers" in infos[0] and "torch" in infos[0]
    # A single-device placement is the quiet path.
    assert warnings == []


def test_load_warns_when_the_model_ends_up_sharded(monkeypatch, tmp_path):
    """The split is legitimate when nothing fits, but it is the one placement where a bad
    device-to-device path can reach the logits, so it is announced with the way to tell them apart."""
    sharded = {"model.visual": 0, "model.language_model.layers.0": 0, "model.language_model.layers.1": 1, "lm_head": 1}
    calls, _, warnings = _stub_model_load(monkeypatch, tmp_path, [_NEED - 1, _NEED - 1], hf_device_map=sharded)
    _dummy_captioner().load()
    assert calls[0]["device_map"] == "auto"
    assert len(warnings) == 1
    assert "2 devices" in warnings[0] and "single GPU" in warnings[0]


def test_load_is_idempotent(monkeypatch, tmp_path):
    calls, _, _ = _stub_model_load(monkeypatch, tmp_path, [_NEED])
    cap = _dummy_captioner()
    cap.load()
    cap.load()
    assert len(calls) == 1
