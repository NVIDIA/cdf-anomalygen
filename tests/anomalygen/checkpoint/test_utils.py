# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the checkpoint integrity checks: the digest sidecars and the geometry stamp."""

import hashlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

import anomalygen.checkpoint.utils as ckpt_utils
from anomalygen.checkpoint.utils import (
    CLASS_LADDERS_KEY,
    GEOMETRY_STAMP_KEY,
    LEGACY_ORIGIN,
    STAMP_ORIGIN_KEY,
    build_geometry_stamp,
    pop_and_verify_geometry_stamp,
    upgrade_legacy_state,
    verify_geometry_stamp,
    verify_manifest,
)


def _write_checkpoint(tmp_path, payload=b"weights"):
    path = tmp_path / "iter_000000100.pt"
    path.write_bytes(payload)
    return str(path)


def test_write_digest_records_a_checkable_sha256sum_line(tmp_path):
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)

    sidecar = tmp_path / "iter_000000100.pt.sha256"
    recorded, name = sidecar.read_text().split()
    assert name == "iter_000000100.pt", "sidecar must name the file, so `sha256sum -c` works by hand"
    assert recorded == ckpt_utils._file_digest(path)


def test_verify_digest_accepts_an_unmodified_checkpoint(tmp_path):
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)

    ckpt_utils.verify_digest(path)  # must not raise


def test_verify_digest_rejects_modified_bytes(tmp_path):
    """The substitution this control exists for: same path, different content."""
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)
    with open(path, "wb") as handle:
        handle.write(b"substituted")

    with pytest.raises(ValueError, match="does not match its recorded digest"):
        ckpt_utils.verify_digest(path)


def test_verify_digest_warns_but_proceeds_when_the_sidecar_is_absent(tmp_path, loguru_lines):
    """Absent is the normal state for a checkpoint written before this change, and for a copy of
    one file. Refusing would break existing runs while buying nothing: an attacker who can replace
    the .pt can replace the .sha256 too."""
    path = _write_checkpoint(tmp_path)

    ckpt_utils.verify_digest(path)  # must not raise

    assert any("no digest beside" in line for line in loguru_lines), "an unverified load must say so"
    assert any("sha256sum" in line for line in loguru_lines), "the warning must name how to record it"


def test_digest_write_is_atomic_leaving_no_tmp_file(tmp_path):
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)

    assert not list(tmp_path.glob("*.tmp")), "a partial write must not survive as a .tmp"


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("empty", ""),
        ("whitespace", "   \n"),
        ("not hex", "not-a-digest  iter_000000100.pt\n"),
        ("truncated", "0" * 32 + "  iter_000000100.pt\n"),
    ],
)
def test_verify_digest_warns_but_proceeds_when_the_sidecar_is_unreadable(tmp_path, loguru_lines, label, content):
    """`sha256sum file > file.sha256` leaves the sidecar empty when sha256sum fails, and the
    recipe this module prints is exactly that. Reading it must not raise, and must not report a
    mismatch — the checkpoint is not what went wrong."""
    path = _write_checkpoint(tmp_path)
    (tmp_path / "iter_000000100.pt.sha256").write_text(content)

    ckpt_utils.verify_digest(path)  # must not raise

    assert any("no readable SHA-256" in line for line in loguru_lines), f"{label} must say so"
    assert not any("does not match" in line for line in loguru_lines), "must not blame the checkpoint"


def test_verify_digest_accepts_an_uppercase_recorded_digest(tmp_path):
    """Hand-written sidecars turn up uppercase; the bytes are what matter, not the casing."""
    path = _write_checkpoint(tmp_path)
    digest = ckpt_utils._file_digest(path)
    (tmp_path / "iter_000000100.pt.sha256").write_text(f"{digest.upper()}  iter_000000100.pt\n")

    ckpt_utils.verify_digest(path)  # must not raise


# --- manifest verification (base + backbone weights) ---------------------------------------------


def _manifest_tree(tmp_path):
    """A checkpoint root with two weights and a sha256sum-format manifest covering both."""
    root = tmp_path / "ckpts"
    (root / "sub").mkdir(parents=True)
    a, b = root / "a.safetensors", root / "sub" / "b.distcp"
    a.write_bytes(b"weight-a")
    b.write_bytes(b"weight-b")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(
        f"{hashlib.sha256(b'weight-a').hexdigest()}  a.safetensors\n"
        f"{hashlib.sha256(b'weight-b').hexdigest()}  sub/b.distcp\n"
    )
    return manifest, root, a


def test_verify_manifest_accepts_an_untouched_tree(tmp_path):
    manifest, root, _ = _manifest_tree(tmp_path)
    assert verify_manifest(manifest, root) == []


def test_verify_manifest_reports_a_modified_weight(tmp_path):
    manifest, root, a = _manifest_tree(tmp_path)
    a.write_bytes(b"tampered")
    failures = verify_manifest(manifest, root)
    assert [path for path, _ in failures] == ["a.safetensors"]
    assert "does not match" in failures[0][1]


def test_verify_manifest_reports_a_missing_weight(tmp_path):
    manifest, root, a = _manifest_tree(tmp_path)
    a.unlink()
    failures = verify_manifest(manifest, root)
    assert [path for path, _ in failures] == ["a.safetensors"]
    assert "missing" in failures[0][1]


def test_verify_manifest_reports_every_failure_not_just_the_first(tmp_path):
    manifest, root, a = _manifest_tree(tmp_path)
    a.write_bytes(b"tampered")
    (root / "sub" / "b.distcp").unlink()
    assert len(verify_manifest(manifest, root)) == 2


def test_verify_manifest_rejects_a_malformed_line(tmp_path):
    manifest, root, _ = _manifest_tree(tmp_path)
    manifest.write_text("not-a-digest  a.safetensors\n")
    with pytest.raises(ValueError, match="not a sha256sum"):
        verify_manifest(manifest, root)


def test_verify_manifest_refuses_a_path_escaping_the_root(tmp_path):
    """The manifest is a file like any other; a relative entry must stay under the checkpoint root."""
    manifest, root, _ = _manifest_tree(tmp_path)
    manifest.write_text(f"{hashlib.sha256(b'x').hexdigest()}  ../outside.bin\n")
    with pytest.raises(ValueError, match="outside the checkpoint root"):
        verify_manifest(manifest, root)


def test_verify_manifest_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_manifest(tmp_path / "nope.sha256", tmp_path)


def test_verify_manifest_only_under_limits_what_is_hashed(tmp_path):
    """A run loads one model size, so the load-time check should not hash the other size's shards."""
    manifest, root, a = _manifest_tree(tmp_path)
    a.write_bytes(b"tampered")  # outside the requested subtree
    assert verify_manifest(manifest, root, only_under="sub") == []
    assert [path for path, _ in verify_manifest(manifest, root, only_under="a.safetensors")] == ["a.safetensors"]


@pytest.mark.parametrize("mode", ["text", "binary"])
def test_verify_manifest_reads_what_coreutils_sha256sum_writes(tmp_path, mode):
    """download_checkpoints.sh records manifests with coreutils `sha256sum` and re-checks them with
    `sha256sum -c`. This parser has to accept exactly what that writes, or the two disagree about a
    file `sha256sum -c` would happily verify."""
    sha256sum = shutil.which("sha256sum")
    if sha256sum is None:
        pytest.skip("coreutils sha256sum not available")
    root = tmp_path / "ckpts"
    root.mkdir()
    (root / "w.safetensors").write_bytes(b"weight")
    flag = ["-b"] if mode == "binary" else []
    out = subprocess.run(
        [sha256sum, *flag, "w.safetensors"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    manifest = tmp_path / "m.sha256"
    manifest.write_text(out)

    assert verify_manifest(manifest, root) == []


# --- geometry stamp -------------------------------------------------------------------------------


def test_the_stamp_records_the_recipe_settings_the_weights_depend_on(stub_model):
    # crop_ratio_range and inference_scale are absent on purpose: both may vary at generation.
    assert build_geometry_stamp(stub_model()) == {
        "crop_mode": "scale",
        "input_size_ladder": [256, 512, 1024, 1536],
        "shift": 1,
        "model_size": "nano",
        "anomaly_types": [["t", "big"], ["t", "small"]],
    }


def test_a_config_without_geometry_cannot_be_stamped():
    # Every checkpoint this pipeline writes is stamped, so a model that cannot be is refused at save time.
    with pytest.raises(ValueError, match="cannot stamp"):
        build_geometry_stamp(SimpleNamespace(config=SimpleNamespace()))


def test_a_matching_stamp_passes(stub_model):
    verify_geometry_stamp(build_geometry_stamp(stub_model()), stub_model(), "iter_1.pt")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(ladder=(256, 512, 1024, 2048)), "input_size_ladder"),
        (dict(shift=5), "shift"),
        (dict(crop_mode="ratio"), "crop_mode"),
        (dict(model_size="edge"), "model_size"),
        (dict(anomaly_types=(("t", "big"), ("t", "hole"))), "anomaly_types"),
    ],
    ids=["ladder", "shift", "crop_mode", "model_size", "anomaly_types"],
)
def test_a_mismatch_raises_and_names_the_field(kwargs, expected, stub_model):
    stamp = build_geometry_stamp(stub_model(**kwargs))
    with pytest.raises(ValueError, match=expected):
        verify_geometry_stamp(stamp, stub_model(), "iter_1.pt")


@pytest.mark.parametrize("kind", ["absent", "partial"])
def test_an_absent_or_partial_stamp_is_refused(kind, stub_model):
    # Both mean a legacy checkpoint. The verifier refuses it; only the inference loader upgrades one
    # first (see upgrade_legacy_state), so the message says what the checkpoint is good for.
    stamp = None
    if kind == "partial":
        stamp = build_geometry_stamp(stub_model())
        del stamp["anomaly_types"]
    with pytest.raises(ValueError, match="legacy checkpoint.*cannot be resumed"):
        verify_geometry_stamp(stamp, stub_model(), "iter_1.pt")


def test_a_model_without_geometry_cannot_verify_a_stamp(stub_model):
    with pytest.raises(ValueError, match="cannot stamp"):
        verify_geometry_stamp(build_geometry_stamp(stub_model()), SimpleNamespace(config=SimpleNamespace()), "x")


def test_pop_and_verify_takes_the_stamp_out_of_the_state(stub_model):
    # The stamp is not a parameter: both load paths must remove it before the state reaches the model.
    state = {"net.w": 0, GEOMETRY_STAMP_KEY: build_geometry_stamp(stub_model())}
    pop_and_verify_geometry_stamp(state, stub_model(), "iter_1.pt")
    assert state == {"net.w": 0}


def test_the_per_class_caps_travel_beside_the_stamp_and_are_handed_back(stub_model):
    # Training's caps are dataset state, not recipe geometry: they ride under their own key, the stamp
    # neither carries nor compares them, and the load path returns them for the caller to attach.
    assert build_geometry_stamp(stub_model(class_ladders={"t+small": [256]})) == build_geometry_stamp(stub_model())
    state = {"net.w": 0, GEOMETRY_STAMP_KEY: build_geometry_stamp(stub_model()), CLASS_LADDERS_KEY: {"t+small": [256]}}
    assert pop_and_verify_geometry_stamp(state, stub_model(), "iter_1.pt") == {"t+small": [256]}
    assert state == {"net.w": 0}
    assert (
        pop_and_verify_geometry_stamp({GEOMETRY_STAMP_KEY: build_geometry_stamp(stub_model())}, stub_model(), "x")
        is None
    )


# --- legacy checkpoints ---------------------------------------------------------------------------
# No stamp, and class ids in the order the recipe listed them. The inference loader upgrades such a
# state in memory: reorder to the sorted ids, then stamp from the recipe.


def _legacy_state(num_classes=2):
    """Class-stacked tensors whose class slice ``c`` is filled with ``c``, so a reorder is visible."""
    stack = lambda shape: torch.stack([torch.full(shape, float(c)) for c in range(num_classes)])  # noqa: E731
    return {
        "net.a.lora_A": stack((2, 3)),
        "net.a.lora_B": stack((3, 2)),
        "net.text_prompt_emb": stack((4, 5)),
        "net.g.lora_A.weight": torch.zeros(2, 3),  # the framework's global LoRA: rank first, not class
    }


def test_upgrade_reorders_classes_from_the_recipe_listing_to_sorted_and_stamps(stub_model):
    # The model sorts to [t,big]=0, [t,small]=1; the recipe listed small first, so listed 0 = small.
    state = _legacy_state()
    upgrade_legacy_state(state, stub_model(), [["t", "small"], ["t", "big"]], "iter_1.pt")

    for key in ("net.a.lora_A", "net.a.lora_B", "net.text_prompt_emb"):
        assert state[key][0].eq(1.0).all() and state[key][1].eq(0.0).all(), key
    assert torch.equal(state["net.g.lora_A.weight"], torch.zeros(2, 3)), "not class-stacked; left alone"
    assert state[GEOMETRY_STAMP_KEY][STAMP_ORIGIN_KEY] == LEGACY_ORIGIN
    # The result is a current state: the ordinary verifier accepts it.
    assert pop_and_verify_geometry_stamp(state, stub_model(), "iter_1.pt") is None
    assert GEOMETRY_STAMP_KEY not in state


def test_upgrade_leaves_tensors_alone_when_the_listing_is_already_sorted(stub_model):
    state, before = _legacy_state(), _legacy_state()
    upgrade_legacy_state(state, stub_model(), [["t", "big"], ["t", "small"]], "iter_1.pt")
    assert all(torch.equal(state[k], before[k]) for k in before)


def test_upgrade_refuses_a_recipe_listing_a_different_class_set(stub_model):
    with pytest.raises(ValueError, match="pass the recipe the checkpoint trained under"):
        upgrade_legacy_state(_legacy_state(), stub_model(), [["t", "big"], ["t", "hole"]], "iter_1.pt")


def test_upgrade_refuses_a_class_count_that_disagrees_with_the_recipe(stub_model):
    with pytest.raises(ValueError, match="holds 3 classes, the recipe lists 2"):
        upgrade_legacy_state(_legacy_state(num_classes=3), stub_model(), [["t", "big"], ["t", "small"]], "x")


def test_upgrade_refuses_a_stamped_state(stub_model):
    # A stamped checkpoint has its own class order; reordering it would corrupt it.
    state = {**_legacy_state(), GEOMETRY_STAMP_KEY: build_geometry_stamp(stub_model())}
    with pytest.raises(ValueError, match="is stamped"):
        upgrade_legacy_state(state, stub_model(), [["t", "big"], ["t", "small"]], "iter_1.pt")
