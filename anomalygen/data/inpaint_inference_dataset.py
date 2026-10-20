# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL-driven dataset for SDG inference, plus the sharded validation-dataloader factory.

Reads a JSONL of testcases, assigns a preset-aware ``index``, fills per-sample generation
defaults, and sorts ascending by mask
instance count for more efficient iterative generation. Also hosts ``get_inpaint_val_dataloader``
(the ``dataloader_val`` LazyCall target used by the texture experiment config) and its private
collate / batch-index helpers. Public surface: ``InpaintInferenceDataset`` and
``get_inpaint_val_dataloader``; everything else is module-private (``_``-prefixed).
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from cosmos_framework.utils import distributed, log
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from anomalygen.configs.texture.constants import (
    DEFAULT_CROP_RATIO,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_NUM_STEPS,
)
from anomalygen.data.crop_geometry import InputSizeSpec, input_size_spec_from_config, plan_for
from anomalygen.data.utils import MASK_FG_THRESHOLD, require_same_size, sample_token_count
from anomalygen.inference.iterative import instance_bboxes

# Noise seed = base_seed + index * SEED_RECORD_STRIDE + output_n * SEED_OUTPUT_STRIDE + instance_j.
# The strides stop the three offsets colliding (testcase 0's 2nd instance vs testcase 1's 1st would
# otherwise share a noise tensor). Room for 64 instances per output, 16 outputs per testcase — past
# ``num_generated_images: 16`` the next testcase's seeds get reused.
SEED_RECORD_STRIDE = 1024
SEED_OUTPUT_STRIDE = 64


class _MaskScan(NamedTuple):
    """What the startup pass reads off one mask: its size and the inclusive bbox of each instance."""

    size: Tuple[int, int]
    bboxes: List[Tuple[int, int, int, int]]


def _sort_records_by_instance_num(
    records: List[dict], resolve_image_path: Callable[[str], str], resolve_mask_path: Callable[[str], str]
) -> Tuple[List[dict], List[_MaskScan]]:
    """Return ``(records, scans)``: the records reordered ascending by instance count (stable; ties keep
    input order) and, per record, what its mask scan found.

    The instances are the ones generation will split the mask into (:func:`instance_bboxes` at the
    record's ``iteration_generation_max_instance``), so the count is the generation depth. The mask is
    decoded once here: the sizes check, the count and the bboxes the validation loader plans its token
    bound from all come off that one read. An instance-free mask is an error, not a zero: generation
    would return the image untouched and save it as a result. AMP repairs its own output
    (``roi_place._verify_and_resize_jsonl``); a hand-written JSONL is not repaired.
    """
    scans = []
    for rec in records:
        img_path, mask_path = resolve_image_path(rec["image_filename"]), resolve_mask_path(rec["mask_filename"])
        max_k = int(rec.get("iteration_generation_max_instance", DEFAULT_MAX_INSTANCES))
        with Image.open(img_path) as img, Image.open(mask_path) as msk:
            require_same_size(img, msk, f"testcase index={rec['index']} ({img_path} + {mask_path})")
            scan = _MaskScan(msk.size, instance_bboxes(msk.convert("L"), max_k))
        if not scan.bboxes:
            raise ValueError(
                f"testcase index={rec['index']}: {mask_path} has no defect pixels "
                f"(a pixel counts as defect at >= {MASK_FG_THRESHOLD})."
            )
        scans.append(scan)

    counts = Counter(len(scan.bboxes) for scan in scans)
    log.info("Instance count distribution across testcases:")
    for k in sorted(counts):
        log.info(f"  {k} instance(s): {counts[k]} testcase(s)")

    paired = sorted(zip(scans, records), key=lambda pair: len(pair[0].bboxes))
    return [rec for _, rec in paired], [scan for scan, _ in paired]


# Per-record keys generation ignores (legacy testcase files carry them), with the reason.
_IGNORED_RECORD_KEYS = {
    "shift": "the sampler shift comes from the recipe",
    "crop_and_paste": "generation always crops and pastes",
}


def _drop_ignored_keys(records: List[dict], input_data_path: str) -> None:
    """Drop the per-record keys generation ignores, warning once per key.

    Dropped rather than kept because ``_val_collate`` keys off the first record and indexes the rest,
    so a key on only some of them raises ``KeyError``.
    """
    for key, reason in _IGNORED_RECORD_KEYS.items():
        carriers = [rec for rec in records if key in rec]
        if not carriers:
            continue
        for rec in carriers:
            del rec[key]
        log.warning(f"{input_data_path}: {len(carriers)} record(s) set {key!r}; dropped, {reason}.")


def _validate_crop_ratio(rec: dict, input_data_path: str) -> None:
    """Normalize a record's ``crop_ratio`` to a float, in place.

    ``crop_ratio`` is the window request: the context to leave around the mask bbox, as a multiple
    of it. Anything but a finite number of at least 1 has no framing to plan from, so it is an
    error rather than a silent substitution.
    """
    where = f"{input_data_path} (index {rec.get('index')})"
    ratio = rec["crop_ratio"]
    if ratio in (None, "none", "None", ""):
        raise ValueError(
            f"crop_ratio must be a number, got {ratio!r} in {where}. Set a ratio "
            f"(e.g. {DEFAULT_CROP_RATIO}) or drop the key to take that default."
        )
    try:
        value = float(ratio)
    except (TypeError, ValueError) as e:
        raise ValueError(f"crop_ratio must be a number, got {ratio!r} in {where}.") from e
    if not (math.isfinite(value) and value >= 1.0):
        raise ValueError(f"crop_ratio must be a finite number >= 1, got {ratio!r} in {where}.")
    rec["crop_ratio"] = value


class InpaintInferenceDataset(Dataset):
    """Reads a JSONL of inpaint testcases, fills defaults, and yields one (record, image, mask) per item."""

    def __init__(
        self,
        input_data_path: str,
        default_guidance: float = DEFAULT_GUIDANCE,
        default_num_steps: int = DEFAULT_NUM_STEPS,
        default_max_instances: int = DEFAULT_MAX_INSTANCES,
        base_dir: Optional[str] = None,
        base_seed: int = 1,
    ) -> None:
        super().__init__()
        # Relative image/mask paths in the JSONL resolve against ``base_dir`` (the repo root) when
        # set, else the current working directory. Training instantiates the val loader under a
        # chdir to the framework checkout, so ``base_dir`` keeps repo-root-relative testcase paths
        # correct; the SDG CLI leaves it None (cwd-relative, unchanged).
        self._base_dir = base_dir
        log.info(f"Loading generation settings from JSONL file: {input_data_path}")
        with open(input_data_path, "r") as f:
            self.input_data = [json.loads(line) for line in f if line.strip()]
        if not self.input_data:
            raise ValueError(f"No testcases found in {input_data_path}.")

        # Before the duplicate check: dropping a key afterwards would let two records that differ
        # only in it pass, then land as the same testcase generated twice.
        _drop_ignored_keys(self.input_data, input_data_path)
        self._raise_on_duplicates(input_data_path)

        # Sample index: honor a preset if present, else default to line position.
        for i, rec in enumerate(self.input_data):
            rec.setdefault("index", i)

        for rec in self.input_data:
            rec.setdefault("guidance", default_guidance)
            rec.setdefault("num_steps", default_num_steps)
            # Own noise per testcase, reproducible run to run since ``index`` is stable. A flat
            # default would give every sample identical noise; an explicit JSONL "seed" wins.
            rec.setdefault("seed", int(base_seed) + int(rec["index"]) * SEED_RECORD_STRIDE)
            rec.setdefault("num_generated_images", 1)
            rec.setdefault("iteration_generation_max_instance", default_max_instances)
            rec.setdefault("crop_ratio", DEFAULT_CROP_RATIO)
            rec.setdefault("poisson_blend", False)
            _validate_crop_ratio(rec, input_data_path)

        self.input_data, self._scans = _sort_records_by_instance_num(self.input_data, self._resolve, self._resolve)

    def _raise_on_duplicates(self, input_data_path: str) -> None:
        encoded = [json.dumps(rec, sort_keys=True) for rec in self.input_data]
        seen: set[str] = set()
        dupes: set[str] = set()
        for s in encoded:
            if s in seen:
                dupes.add(s)
            seen.add(s)
        if dupes:
            raise ValueError(
                f"Found duplicated samples in {input_data_path}. Duplicated samples:\n" + "\n".join(sorted(dupes))
            )

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        base = self._base_dir if self._base_dir is not None else os.getcwd()
        return os.path.abspath(os.path.join(base, path))

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, idx: int) -> dict:
        rec = dict(self.input_data[idx])
        rec["image"] = Image.open(self._resolve(rec["image_filename"])).convert("RGB")
        rec["mask"] = Image.open(self._resolve(rec["mask_filename"])).convert("L")
        return rec

    def planned_tokens(self, input_spec: InputSizeSpec) -> List[int]:
        """Per record, the packed tokens of the heaviest window generation plans for one of its instances.

        Each instance bbox from the startup scan goes through :func:`plan_for`, the call generation
        makes for that instance, so this is what a validation forward holds per record. Only the
        validation loader asks.
        """
        out = []
        for rec, scan in zip(self.input_data, self._scans):
            plans = [
                plan_for(input_spec, box, *scan.size, crop_ratio=rec["crop_ratio"], anomaly_name=rec["anomaly_type"])
                for box in scan.bboxes
            ]
            out.append(max(sample_token_count(plan.input_w, plan.input_h) for plan in plans))
        return out

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Single-sample collate (the SDG loop runs ``batch_size=1``)."""
        if len(batch) != 1:
            raise ValueError(f"InpaintInferenceDataset expects batch_size=1, got {len(batch)}.")
        return batch[0]


def _val_collate(batch: List[dict]) -> dict:
    """Collate per-testcase dicts into a dict of per-key lists, with ``index`` as a ``[B]`` tensor.

    Values stay as per-key lists (no tensor stacking) so PIL images / strings / scalars pass
    through. ``index`` is additionally emitted as a tensor because the framework's per-step
    callbacks infer the batch size from a tensor in the batch (``misc.get_data_batch_size``), and
    our batch is otherwise all lists of str/float/PIL. ``_generate_validation_batch`` coerces indices
    via ``int(x)``, so the tensor works identically to the list form.
    """
    out = {k: [rec[k] for rec in batch] for k in batch[0]}
    out["index"] = torch.tensor([int(rec["index"]) for rec in batch], dtype=torch.long)
    return out


def _build_val_batch_indices(records: List[dict], world_size: int, rank: int, batch_size: int) -> List[List[int]]:
    """Per-rank list of index-lists, each batch homogeneous in ``(guidance, num_steps)``.

    Records are grouped by ``(guidance, num_steps)``, the two per-record sampler parameters, so
    every rank runs the same CFG gate on the same timestep grid. Each group is padded (by repeating
    from the front) to a multiple of ``world_size`` and split contiguously across ranks, so every
    rank yields the same number of same-sized batches (FSDP lockstep). Padding repeats are removed
    later by the ``index`` dedup in ``on_validation_end``.
    """
    groups: Dict[Tuple[float, int], List[int]] = {}
    for i, rec in enumerate(records):
        groups.setdefault((float(rec["guidance"]), int(rec["num_steps"])), []).append(i)

    batch_indices: List[List[int]] = []
    for key in sorted(groups):  # deterministic, identical on every rank
        idxs = groups[key]
        target = ((len(idxs) + world_size - 1) // world_size) * world_size
        padded = (idxs * ((target // len(idxs)) + 1))[:target]
        per = target // world_size
        mine = padded[rank * per : (rank + 1) * per]
        for c in range(0, len(mine), batch_size):
            batch_indices.append(mine[c : c + batch_size])
    return batch_indices


def get_inpaint_val_dataloader(
    input_data_path: str,
    val_batch_size: int,
    default_guidance: float = DEFAULT_GUIDANCE,
    default_num_steps: int = DEFAULT_NUM_STEPS,
    default_max_instances: int = DEFAULT_MAX_INSTANCES,
    num_workers: int = 2,
    base_dir: Optional[str] = None,
    base_seed: int = 1,
    token_budget: Optional[int] = None,
    diffusion_expert_config=None,
) -> DataLoader:
    """LazyCall target for ``dataloader_val``: a sharded, param-homogeneous validation loader.

    Reads world_size/rank from the framework's distributed utils at instantiation time (so it
    also works single-process), builds this rank's batch-index lists, and returns a plain
    ``DataLoader`` whose ``batch_sampler`` is those lists and ``collate_fn`` keeps values as per-key lists.
    ``base_dir`` resolves repo-root-relative testcase paths (training instantiates this under a
    chdir to the framework checkout, so cwd-relative resolution would be wrong).

    ``token_budget`` bounds one validation forward the way the training packer bounds a step: batches
    are cut by count, so the count is shrunk until that many copies of the heaviest window
    ``diffusion_expert_config`` can plan fit the budget. Training builds this loader before the
    per-class caps are known, so the bound is planned against the full ladder, which can only
    over-estimate.
    """
    dataset = InpaintInferenceDataset(
        input_data_path,
        default_guidance=default_guidance,
        default_num_steps=default_num_steps,
        default_max_instances=default_max_instances,
        base_dir=base_dir,
        base_seed=base_seed,
    )
    val_batch_size = int(val_batch_size)
    if token_budget is not None:
        heaviest = max(dataset.planned_tokens(input_size_spec_from_config(diffusion_expert_config)))
        fit = max(1, min(val_batch_size, int(token_budget) // heaviest))
        if fit < val_batch_size:
            log.info(
                f"validation batch size {val_batch_size} -> {fit}: the heaviest planned input is {heaviest} "
                f"tokens against a budget of {token_budget}."
            )
        val_batch_size = fit
    world_size = distributed.get_world_size()
    rank = distributed.get_rank()
    batch_indices = _build_val_batch_indices(dataset.input_data, world_size, rank, val_batch_size)
    return DataLoader(dataset, batch_sampler=batch_indices, collate_fn=_val_collate, num_workers=int(num_workers))
