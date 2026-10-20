# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint integrity checks, shared by the training and inference load paths.

Two things a loaded ``.pt`` cannot tell you on its own: whether its bytes are the ones we wrote
(the digest sidecars) and whether it was trained under the recipe now being paired with it (the
geometry stamp). The digest warns when absent and refuses a mismatch; the stamp refuses both, since
every checkpoint this release writes carries one. A checkpoint with no stamp at all is a legacy one:
the inference loader stamps it from its recipe in memory (:func:`upgrade_legacy_state`); a resume
refuses it.

``weights_only=True`` stops a checkpoint executing code; it does not say the checkpoint is the one
this run wrote. The base weights have assets/checkpoint_manifest.sha256 — this is the equivalent
for what we produce, which is also the file most likely to travel between machines.

A sidecar is not a signature (whoever replaces the .pt can replace the .sha256), so the two cases
differ: a *mismatch* is fatal, since only truncation, a bad sync or a careless swap produces it;
an *absent* digest only warns, because that is every pre-existing checkpoint and every copy of a
single file, and blocking would break those while a real attacker just writes the sidecar too.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from typing import Optional

from cosmos_framework.utils import log

from anomalygen.data.crop_geometry import input_size_spec_from_config

_DIGEST_SUFFIX = ".sha256"

# Key holding the geometry stamp inside the saved model component. Leading underscore so it cannot
# collide with a parameter name; both load paths pop it before the state dict reaches the model.
GEOMETRY_STAMP_KEY = "_anomalygen_geometry"
# Key holding the per-class input-size caps beside the stamp: dataset state, carried but never compared.
CLASS_LADDERS_KEY = "_anomalygen_class_ladders"
# Stamp field set by upgrade_legacy_state, so a stamp synthesised at load can be told from a trained one.
STAMP_ORIGIN_KEY = "origin"
LEGACY_ORIGIN = "legacy checkpoint, stamped from its recipe at load"
# Trained parameters stacked per class (dim 0 is the class id). The framework's global LoRA is a
# ``nn.Linear`` and ends in ``.weight``, so it is not matched.
_CLASS_STACKED = ("lora_A", "lora_B", "text_prompt_emb")


def _file_digest(path: str) -> str:
    """SHA-256 of a file, read in chunks — a checkpoint shard does not fit comfortably in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path, root, *, only_under: str | None = None) -> list[tuple[str, str]]:
    """Check a ``sha256sum``-format manifest against ``root``; return one entry per failure.

    This is the base and backbone counterpart to :func:`verify_digest`: those weights are recorded by
    ``scripts/download_checkpoints.sh`` but were previously only checked at download time, while
    ``checkpoints/`` is a bind mount that anything on the host can rewrite afterwards. Returning the
    failures instead of raising lets the caller decide — preflight reports all of them, the load path
    refuses on the first.

    ``only_under`` limits the check to entries at or below that relative path, so a run can verify the
    model size it is about to load without hashing the other one. A malformed line or an entry
    escaping ``root`` raises: those are defects in the manifest, not in the weights.
    """
    root = os.path.realpath(os.fspath(root))
    with open(manifest_path) as handle:
        lines = handle.read().splitlines()

    failures: list[tuple[str, str]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        # Exactly what coreutils writes: digest, one space, then ' ' (text) or '*' (binary), then the
        # path. Both are accepted by `sha256sum -c`, which download_checkpoints.sh uses on the same
        # files — a parser that took only one of them would disagree with the tool that records them.
        entry = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not entry:
            raise ValueError(f"{manifest_path}:{lineno} is not a sha256sum line: {line!r}")
        digest, rel = entry.group(1), entry.group(2).strip()
        if only_under and rel != only_under and not rel.startswith(only_under.rstrip("/") + "/"):
            continue
        target = os.path.realpath(os.path.join(root, rel))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f"{manifest_path}:{lineno} names {rel!r}, outside the checkpoint root {root}")
        if not os.path.isfile(target):
            failures.append((rel, f"missing: {target}"))
        elif _file_digest(target) != digest.lower():
            failures.append((rel, f"does not match the recorded digest in {os.path.basename(str(manifest_path))}"))
    return failures


def write_digest(path: str) -> None:
    """Record ``<path>.sha256`` in ``sha256sum`` format, so it can be checked by hand."""
    sidecar = path + _DIGEST_SUFFIX
    tmp = sidecar + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(f"{_file_digest(path)}  {os.path.basename(path)}\n")
    # Rename rather than write in place: a redirect truncates first, so an interrupt would leave a
    # half-written digest that fails every later load. Matches download_checkpoints.sh.
    os.replace(tmp, sidecar)


def verify_digest(path: str) -> None:
    """Check ``path`` against its sidecar digest.

    Raises ValueError when the digest is present and disagrees, and propagates the OSError from a
    sidecar that cannot be opened. Warns and returns when it is absent, empty or non-hex — see the
    module docstring for why those cases are treated differently.
    """
    sidecar = path + _DIGEST_SUFFIX
    if not os.path.exists(sidecar):
        log.warning(
            f"Checkpoint integrity: no digest beside {path}, so its bytes are unverified. "
            "Checkpoints written before digest recording have none; ones written from now on do. "
            "To record it for this checkpoint after confirming it is the one you expect:\n"
            f"  ( cd {os.path.dirname(path)!r} && sha256sum {os.path.basename(path)!r} > "
            f"{os.path.basename(path) + _DIGEST_SUFFIX!r} )"
        )
        return
    with open(sidecar) as handle:
        fields = handle.read().split()
    # `sha256sum f > f.sha256` leaves the sidecar empty when sha256sum fails. Warn rather than
    # compare, so a broken sidecar is not reported as a modified checkpoint.
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0].lower()):
        log.warning(
            f"Checkpoint integrity: {sidecar} holds no readable SHA-256, so the bytes of {path} "
            "are unverified. The checkpoint itself may still be fine."
        )
        return
    recorded = fields[0].lower()
    actual = _file_digest(path)
    if actual != recorded:
        raise ValueError(
            f"Checkpoint integrity: {path} does not match its recorded digest.\n"
            f"  recorded {recorded}\n  actual   {actual}\n"
            "The checkpoint's bytes changed after it was written. Treat this as an integrity "
            "problem — do not re-record the digest to make it load."
        )


def resolve_shift(model):
    """The sampler time-shift ``model`` was built with, or ``None`` when its config carries none."""
    config = getattr(model, "config", None)
    shift = getattr(getattr(config, "rectified_flow_training_config", None), "shift", None)
    return shift if isinstance(shift, (int, float)) else None


def _config_field(dec, key: str):
    """``key`` off a ``diffusion_expert_config`` given as an attrs object or a mapping; ``None`` if unset."""
    return dec.get(key) if isinstance(dec, Mapping) else getattr(dec, key, None)


def build_geometry_stamp(model) -> dict:
    """The recipe settings the trained weights are only valid under, read off ``model.config``.

    Saved beside the weights and compared on every load, so a checkpoint cannot be paired with a
    recipe it did not train under. ``crop_ratio_range`` and ``inference_scale`` are left out: both may
    vary at generation. Raises when the config carries no geometry: every checkpoint this pipeline
    writes must be stamped.
    """
    dec = getattr(getattr(model, "config", None), "diffusion_expert_config", None)
    shift = resolve_shift(model)
    model_size = _config_field(dec, "model_size")
    anomaly_types = _config_field(dec, "anomaly_types")
    if dec is None or shift is None or model_size is None or anomaly_types is None:
        raise ValueError(
            "model config carries no geometry (diffusion_expert_config with model_size and anomaly_types, "
            "plus the sampler shift); cannot stamp the checkpoint."
        )
    spec = input_size_spec_from_config(dec)  # the same reader generation plans with
    return {
        "crop_mode": spec.crop_mode,
        "input_size_ladder": list(spec.input_size_ladder),
        "shift": shift,
        "model_size": str(model_size),
        "anomaly_types": [[str(t), str(d)] for t, d in anomaly_types],
    }


def verify_geometry_stamp(stamp, model, source: str) -> None:
    """Refuse ``source`` unless its stamp is complete and equal to the recipe ``model`` was built from.

    No stamp, or one missing a field, means a legacy checkpoint; this verifier does not load it.
    """
    current = build_geometry_stamp(model)
    if not isinstance(stamp, Mapping) or any(k not in stamp for k in current):
        raise ValueError(
            f"{source} carries no complete geometry stamp {list(current)}: a legacy checkpoint. It generates "
            "with its run-dir recipe (the inference loader stamps it at load) but cannot be resumed."
        )
    diffs = [k for k in current if stamp[k] != current[k]]
    if diffs:
        detail = "; ".join(f"{k}: checkpoint {stamp[k]!r} vs recipe {current[k]!r}" for k in diffs)
        raise ValueError(
            f"{source} was trained under a different recipe — {detail}. Pass the recipe it trained under "
            "(the run-dir copy)."
        )


def pop_and_verify_geometry_stamp(state: dict, model, source: str) -> Optional[dict]:
    """Take the stamp and the per-class caps out of a loaded model state (neither is a parameter), check
    the stamp against ``model``, and return the caps, or ``None`` when the checkpoint carries none.

    The caller decides what to do with the caps: generation attaches them; a resume keeps its own dataset's.
    """
    verify_geometry_stamp(state.pop(GEOMETRY_STAMP_KEY, None), model, source)
    caps = state.pop(CLASS_LADDERS_KEY, None)
    return {str(k): list(v) for k, v in caps.items()} if caps else None


def upgrade_legacy_state(state: dict, model, recipe_anomaly_types, source: str) -> None:
    """Give an unstamped (legacy) trained-subset state the stamp and class order the current code expects.

    A legacy checkpoint carries no stamp, so the recipe ``model`` was built from supplies the
    geometry. Its class ids follow ``recipe_anomaly_types``, the order its recipe listed them in; the
    current builder sorts them, so every class-stacked tensor is reordered to match. In memory only:
    the file on disk is left as it is.
    """
    if GEOMETRY_STAMP_KEY in state:
        raise ValueError(f"{source} is stamped; only a legacy (unstamped) checkpoint is upgraded.")
    stamp = build_geometry_stamp(model)
    recipe_order = [[str(t), str(d)] for t, d in recipe_anomaly_types]
    if sorted(recipe_order) != stamp["anomaly_types"]:
        raise ValueError(
            f"{source}: the recipe lists anomaly_types {recipe_order}, but the model was built with "
            f"{stamp['anomaly_types']}; pass the recipe the checkpoint trained under."
        )
    perm = [recipe_order.index(pair) for pair in stamp["anomaly_types"]]  # sorted id -> listed id
    identity = perm == list(range(len(perm)))
    for key, value in list(state.items()):
        if key.rsplit(".", 1)[-1] not in _CLASS_STACKED:
            continue
        if value.shape[0] != len(perm):
            raise ValueError(f"{source}: {key} holds {value.shape[0]} classes, the recipe lists {len(perm)}.")
        if not identity:
            state[key] = value[perm]
    if not identity:
        log.warning(
            f"{source}: classes reordered from the recipe's listing to sorted order (sorted id -> listed id {perm})."
        )
    state[GEOMETRY_STAMP_KEY] = {**stamp, STAMP_ORIGIN_KEY: LEGACY_ORIGIN}
