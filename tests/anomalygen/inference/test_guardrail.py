# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generate.py content-safety guardrail adapter.

Pure plumbing (caption, PIL<->numpy, block-record) is tested directly on CPU with real
inputs and no mocks. Behavior that needs the real guardrail models (Qwen3Guard, RetinaFace)
is tagged ``@pytest.mark.gpu`` and auto-skipped when no CUDA device is present.
"""

import os
import pathlib
import types

import nltk
import numpy as np
import pytest
from huggingface_hub import constants as hf_constants
from PIL import Image

import anomalygen.inference.guardrail as guardrail
from anomalygen.data.utils import build_caption, caption_for_anomaly_type
from anomalygen.inference.guardrail import (
    BLOCKED_CSV_HEADER,
    GUARDRAIL_REPO_DIR,
    _dereference_guardrail_nltk_data,
    _frames_to_pil,
    _pil_to_frames,
    _read_guardrail_corpora_for_nltk,
    blocked_row,
    create_guardrail_runners,
    guard_image,
)


def test_caption_for_anomaly_type_texture_plus_defect():
    assert caption_for_anomaly_type("phone_screen+scratch") == build_caption(defect="scratch", texture="phone_screen")


def test_caption_for_anomaly_type_bare_texture():
    # No "+": the bare token is used as both texture and defect (matches inpaint.py).
    assert caption_for_anomaly_type("phone_screen") == build_caption(defect="phone_screen", texture="phone_screen")


def test_pil_to_frames_shape_and_dtype():
    img = Image.new("RGB", (7, 5), (10, 20, 30))  # size = (W=7, H=5)
    frames = _pil_to_frames(img)
    assert frames.shape == (1, 5, 7, 3)  # [1, H, W, C]
    assert frames.dtype == np.uint8


def test_pil_to_frames_returns_writable_array():
    # RetinaFace face-blur writes into the frame in place, so the buffer must be writable.
    frames = _pil_to_frames(Image.new("RGB", (7, 5), (10, 20, 30)))
    assert frames.flags.writeable is True


def test_pil_frames_roundtrip_is_pixel_exact():
    arr = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)  # H=5, W=7
    img = Image.fromarray(arr, mode="RGB")
    back = _frames_to_pil(_pil_to_frames(img))
    assert back.size == img.size
    assert np.array_equal(np.asarray(back), arr)


def test_frames_to_pil_rejects_bad_shape():
    with pytest.raises(ValueError):
        _frames_to_pil(np.zeros((5, 7, 3), dtype=np.uint8))  # missing frame axis (ndim != 4)
    with pytest.raises(ValueError):
        _frames_to_pil(np.zeros((2, 5, 7, 3), dtype=np.uint8))  # multi-frame (shape[0] != 1)


def test_blocked_row_shape_and_fields():
    row = blocked_row(5, 3, "phone_screen+scratch", "img.png", "mask.png", "image", "bad words")
    assert row["sort_key"] == [5, 3]  # (index, output_idx) — keeps per-output rows distinct
    assert set(row["row"]) == set(BLOCKED_CSV_HEADER)
    assert row["row"]["index"] == 5
    assert row["row"]["output_idx"] == 3
    assert row["row"]["guardrail"] == "image"
    assert row["row"]["message"] == "bad words"


def test_nltk_allowlist_gains_the_guardrail_model_dir(tmp_path, monkeypatch):
    """The guardrail's corpora live in the HF cache, which nltk >= 3.10 refuses to read.

    CPU-only on purpose: the end-to-end guardrail test below needs a GPU and is skipped in CI,
    so without this the nltk fix would have no automated coverage at all — and a future nltk
    bump would silently reintroduce the PermissionError it exists to prevent.
    """

    model_dir = tmp_path / GUARDRAIL_REPO_DIR
    model_dir.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(nltk.data, "path", list(nltk.data.path))

    assert _read_guardrail_corpora_for_nltk() == str(model_dir)
    assert str(model_dir) in nltk.data.path

    # Idempotent: a second call must not append a duplicate.
    before = list(nltk.data.path)
    assert _read_guardrail_corpora_for_nltk() == str(model_dir)
    assert nltk.data.path == before


def test_nltk_allowlist_is_a_noop_when_the_model_is_absent(tmp_path, monkeypatch):

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))  # no model dir inside
    monkeypatch.setattr(nltk.data, "path", list(nltk.data.path))
    before = list(nltk.data.path)

    assert _read_guardrail_corpora_for_nltk() is None
    assert nltk.data.path == before


def test_nltk_allowlist_reads_the_env_not_the_frozen_constant(tmp_path, monkeypatch):
    """HF_HUB_CACHE must win over huggingface_hub.constants.

    generate.py sets HF_HUB_CACHE at import time, which can land after huggingface_hub has
    already frozen its constants — reading the constant then points at the wrong cache.
    """

    (tmp_path / GUARDRAIL_REPO_DIR).mkdir()
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", "/nonexistent/frozen/cache")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(nltk.data, "path", list(nltk.data.path))

    assert _read_guardrail_corpora_for_nltk() == str(tmp_path / GUARDRAIL_REPO_DIR)


# --- symlinked HF-cache corpora vs nltk's O_NOFOLLOW open ------------------------------------------
# The HF cache stores every snapshot file as a symlink into blobs/. nltk >= 3.10.3 refuses to open a
# final-component symlink, so the blocklist's punkt/wordnet data must be shadowed by a real copy.

_PUNKT_FILE = "tokenizers/punkt_tab/english/collocations.tab"
_STUB_NLTK = getattr(nltk, "IS_STUB", False)


def _read_like_real_nltk(rel: str) -> bytes:
    """What ``nltk.data.find(rel).open()`` does in nltk >= 3.10.3: open ``rel`` under the first
    ``nltk.data.path`` entry that has it, refusing a final-component symlink. Written out so these tests
    also run where ``nltk`` is the stub, which has neither ``find`` nor the check."""
    for root in nltk.data.path:
        path = os.path.join(root, rel)
        if os.path.lexists(path):
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as f:
                return f.read()
    raise LookupError(rel)


def _fake_hf_guardrail(cache: pathlib.Path, revision: str = "d6d4bfa") -> pathlib.Path:
    """Lay out ``<cache>/models--nvidia--Cosmos-Guardrail1`` the way huggingface_hub does: content in
    ``blobs/``, and every snapshot file a relative symlink into it. Returns the snapshot's nltk_data."""
    model_dir = cache / GUARDRAIL_REPO_DIR
    blob = model_dir / "blobs" / "b38862b8c3126ae4ad3468f3377a7b2708554c17"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"punkt collocations\n")
    nltk_data = model_dir / "snapshots" / revision / "blocklist" / "nltk_data"
    target = nltk_data / _PUNKT_FILE
    target.parent.mkdir(parents=True)
    target.symlink_to(os.path.relpath(blob, target.parent))
    # The rest of the snapshot: model weights, also symlinked, and much larger than the corpora.
    weights_blob = model_dir / "blobs" / "0f3c2a9e7b1d4c6a8e5f0b2d9c7a1e3f5b8d0c2a"
    weights_blob.write_bytes(b"\0" * 4096)
    weights = model_dir / "snapshots" / revision / "face_blur_filter" / "Resnet50_Final.pth"
    weights.parent.mkdir(parents=True)
    weights.symlink_to(os.path.relpath(weights_blob, weights.parent))
    return nltk_data


@pytest.fixture
def isolated_nltk(tmp_path, monkeypatch):
    """A private nltk.data.path and user cache, so the test neither reads nor writes the real ones."""
    monkeypatch.setattr(nltk.data, "path", [])
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return tmp_path


@pytest.mark.skipif(_STUB_NLTK, reason="the real nltk is not installed; the stub reads symlinks")
def test_nltk_refuses_the_symlinked_original(isolated_nltk):
    """The premise of the workaround. When this stops failing, nltk follows symlinks again and
    ``_dereference_guardrail_nltk_data`` can be retired."""
    nltk_data = _fake_hf_guardrail(isolated_nltk / "hub")
    nltk.data.path.append(str(nltk_data))

    with pytest.raises(PermissionError, match="symlink"):
        nltk.data.find(_PUNKT_FILE).open().close()


def test_the_stand_in_refuses_the_symlinked_original_too(isolated_nltk):
    """The premise above, through ``_read_like_real_nltk``, so it holds where the real nltk is absent."""
    nltk.data.path.append(str(_fake_hf_guardrail(isolated_nltk / "hub")))

    with pytest.raises(OSError):
        _read_like_real_nltk(_PUNKT_FILE)


def test_symlinked_nltk_data_is_shadowed_by_a_real_copy(isolated_nltk):
    nltk_data = _fake_hf_guardrail(isolated_nltk / "hub")
    nltk.data.path.append(str(nltk_data))  # what the framework's Blocklist.__init__ does

    copies = _dereference_guardrail_nltk_data(str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR))

    assert len(copies) == 1
    copy = pathlib.Path(copies[0])
    assert nltk.data.path.index(str(copy)) < nltk.data.path.index(str(nltk_data)), nltk.data.path
    assert not any(p.is_symlink() for p in copy.rglob("*")), "the copy still contains symlinks"
    assert copy.is_relative_to(isolated_nltk / "xdg"), "the copy did not land in the user cache"
    assert _read_like_real_nltk(_PUNKT_FILE) == b"punkt collocations\n"


def test_dereference_is_idempotent_and_reuses_the_copy(isolated_nltk):
    nltk_data = _fake_hf_guardrail(isolated_nltk / "hub")
    nltk.data.path.append(str(nltk_data))
    model_dir = str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR)

    first = _dereference_guardrail_nltk_data(model_dir)
    path_after_first = list(nltk.data.path)
    second = _dereference_guardrail_nltk_data(model_dir)

    assert second == [] and nltk.data.path == path_after_first
    # A fresh process (empty nltk.data.path) with the same source finds the same copy: one dir per
    # source, no accumulation in the cache.
    nltk.data.path[:] = [str(nltk_data)]
    assert _dereference_guardrail_nltk_data(model_dir) == first
    assert len(list((isolated_nltk / "xdg" / "anomalygen" / "nltk_data").iterdir())) == 1


def test_dereference_falls_back_to_a_private_temp_dir_when_the_cache_is_unwritable(isolated_nltk, monkeypatch):
    nltk_data = _fake_hf_guardrail(isolated_nltk / "hub")
    nltk.data.path.append(str(nltk_data))
    (isolated_nltk / "xdg").write_text("not a directory")  # mkdir under it fails

    copies = _dereference_guardrail_nltk_data(str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR))

    assert len(copies) == 1
    assert _read_like_real_nltk(_PUNKT_FILE) == b"punkt collocations\n"


def test_a_copy_that_fails_midway_falls_back_and_leaves_no_partial_tree_in_the_cache(isolated_nltk, monkeypatch):
    """A cache that accepts the directory but not the files (ENOSPC, an unreadable blob) is the same case
    as an unwritable cache: fall back to a private temp dir and do not leave `<hash>.tmp-*` behind."""
    nltk_data = _fake_hf_guardrail(isolated_nltk / "hub")
    nltk.data.path.append(str(nltk_data))
    real_copytree = guardrail.shutil.copytree
    cache_root = isolated_nltk / "xdg" / "anomalygen" / "nltk_data"

    def copytree_failing_inside_the_cache(src, dst, *args, **kwargs):  # shutil recurses positionally
        if pathlib.Path(dst).is_relative_to(cache_root):
            raise OSError(28, "No space left on device")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(guardrail.shutil, "copytree", copytree_failing_inside_the_cache)

    copies = _dereference_guardrail_nltk_data(str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR))

    assert len(copies) == 1 and not pathlib.Path(copies[0]).is_relative_to(cache_root)
    assert list(cache_root.iterdir()) == [], "a partial <hash>.tmp-* tree was left in the cache root"
    assert _read_like_real_nltk(_PUNKT_FILE) == b"punkt collocations\n"


def test_a_marker_less_copy_is_reported_not_reused(isolated_nltk):
    """`dst` only ever appears through a rename of a finished tree, so a copy without its marker was
    damaged by hand. Refuse it with a message that names the directory rather than an ENOTEMPTY."""
    nltk_data = _fake_hf_guardrail(isolated_nltk / "hub")
    nltk.data.path.append(str(nltk_data))
    model_dir = str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR)
    (copy,) = _dereference_guardrail_nltk_data(model_dir)
    (pathlib.Path(copy) / guardrail._COMPLETE_MARKER).unlink()
    nltk.data.path[:] = [str(nltk_data)]

    with pytest.raises(RuntimeError, match="without its .complete marker"):
        _dereference_guardrail_nltk_data(model_dir)


def test_dereference_leaves_a_symlink_free_nltk_data_alone(isolated_nltk):
    """A local_dir download (or a hand-materialised tree) has real files; nothing to shadow."""
    nltk_data = isolated_nltk / "hub" / GUARDRAIL_REPO_DIR / "snapshots" / "rev" / "blocklist" / "nltk_data"
    (nltk_data / _PUNKT_FILE).parent.mkdir(parents=True)
    (nltk_data / _PUNKT_FILE).write_bytes(b"real file\n")
    nltk.data.path.append(str(nltk_data))
    before = list(nltk.data.path)

    assert _dereference_guardrail_nltk_data(str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR)) == []
    assert nltk.data.path == before


def test_dereference_copies_the_corpora_but_never_the_model_dir_root(isolated_nltk):
    """``create_guardrail_runners`` puts the model dir itself on nltk.data.path as an allowlist root.
    It is under ``model_dir`` and full of symlinks too, but copying it would duplicate the guardrail's
    weights (gigabytes) into the user cache. Only the ``nltk_data`` directory below it is copied."""
    nltk_data = _fake_hf_guardrail(isolated_nltk / "hub")
    model_dir = str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR)
    assert _read_guardrail_corpora_for_nltk() == model_dir  # what create_guardrail_runners does first
    nltk.data.path.append(str(nltk_data))

    copies = _dereference_guardrail_nltk_data(model_dir)

    assert len(copies) == 1
    copied = {p.name for p in pathlib.Path(copies[0]).rglob("*") if p.is_file()}
    assert "collocations.tab" in copied and "Resnet50_Final.pth" not in copied, copied
    assert len(list((isolated_nltk / "xdg" / "anomalygen" / "nltk_data").iterdir())) == 1
    assert model_dir in nltk.data.path  # the allowlist root is left where it was


def test_dereference_ignores_entries_outside_the_guardrail_model_dir(isolated_nltk):
    """Only the guardrail's own corpora are copied: a user's other nltk_data entries are not touched
    even when they contain symlinks."""
    other = isolated_nltk / "elsewhere" / "nltk_data"
    (other / "tokenizers").mkdir(parents=True)
    (other / "tokenizers" / "x.tab").symlink_to("/nonexistent")
    nltk.data.path.append(str(other))
    (isolated_nltk / "hub" / GUARDRAIL_REPO_DIR).mkdir(parents=True)

    assert _dereference_guardrail_nltk_data(str(isolated_nltk / "hub" / GUARDRAIL_REPO_DIR)) == []
    assert nltk.data.path == [str(other)]


# --- which nltk gets the workarounds ----------------------------------------------------------------


@pytest.fixture
def runners_without_models(monkeypatch):
    """The framework's runner factories, replaced so ``create_guardrail_runners`` loads no model."""
    presets = pytest.importorskip("cosmos_framework.auxiliary.guardrail.common.presets")
    monkeypatch.setattr(presets, "create_text_guardrail_runner", lambda **_: "text")
    monkeypatch.setattr(presets, "create_video_guardrail_runner", lambda **_: _Runner([object()]))
    calls = []
    monkeypatch.setattr(guardrail, "_read_guardrail_corpora_for_nltk", lambda: calls.append("allowlist"))
    return calls


def test_the_real_nltk_gets_the_allowlist_before_and_after_the_text_runner(runners_without_models, monkeypatch):
    monkeypatch.setattr(guardrail, "nltk", types.ModuleType("nltk"))  # the real package has no IS_STUB

    create_guardrail_runners()

    assert runners_without_models == ["allowlist", "allowlist"]


def test_the_stub_needs_no_workaround(runners_without_models, monkeypatch):
    stub = types.ModuleType("nltk")
    stub.IS_STUB = True
    monkeypatch.setattr(guardrail, "nltk", stub)

    create_guardrail_runners()

    assert runners_without_models == []


@pytest.mark.gpu
def test_guardrail_runners_check_and_guard():
    # Exercise the same factory generate.py uses — it also widens nltk's allowlist.
    text_guardrail, video_guardrail = create_guardrail_runners(offload_model_to_cpu=False)

    # A benign templated caption must pass the text guardrail.
    is_safe, _message = text_guardrail.run_safety_check(caption_for_anomaly_type("phone_screen+scratch"))
    assert is_safe is True

    # A plain, faceless image passes through the face-blur postprocessor unchanged in size.
    out, _msg = guard_image(video_guardrail, Image.new("RGB", (64, 48), (127, 127, 127)))
    assert isinstance(out, Image.Image)
    assert out.size == (64, 48)


# --- enforcement capability vs the operator's flag ------------------------------------------------
# A GuardrailRunner with an empty safety_models list always answers "safe": its postprocessors (face
# blur) still run, but nothing can be blocked. timing_summary.json records guardrail_enabled, which
# a reader would take as evidence screening happened, so capability is reported separately.


class _Runner:
    def __init__(self, safety_models):
        self.safety_models = safety_models


def test_runner_with_no_safety_model_is_not_enforcing():
    """The current state of the framework's video preset — face blur only, nothing can deny."""
    assert guardrail.is_enforcing(_Runner([])) is False


def test_runner_with_a_safety_model_is_enforcing():
    assert guardrail.is_enforcing(_Runner([object()])) is True


def test_runner_without_the_attribute_is_not_enforcing():
    """Absence of evidence is reported as not-enforcing, never as enforcing."""

    class _Bare:
        pass

    assert guardrail.is_enforcing(_Bare()) is False
