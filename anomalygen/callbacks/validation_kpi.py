# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic generation + DINOv2 NN/MNN scoring during training, writing per-iteration KPI artifacts.

Generation runs sharded + batched in ``AnomalyGenTextureMoTModel.validation_step`` (fed by the
real validation dataloader) and accumulates onto the model. This callback stages the generation
config on the model in ``on_validation_start``, then in ``on_validation_end`` gathers every rank's
results, dedups by sample ``index``, and (on rank 0) scores NN/MNN (+ FID) and writes valid_kpi.csv.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from PIL import Image

from anomalygen.data.utils import list_image_mask_pairs, require_same_size
from anomalygen.eval.anomaly_quality import compute_anomaly_quality_kpi
from anomalygen.eval.correspondence import (
    DEFAULT_BACKBONE,
    DEFAULT_NN_INST_AGG,
    DEFAULT_NN_LAYER,
    DEFAULT_NN_READOUT,
    DEFAULT_NN_REGION_POLICY,
    compute_correspondence_kpi,
)
from anomalygen.eval.fid import compute_fid_kpi


def _img_to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _mask_to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def _save(img: Image.Image, out_dir: str, subdir: str, filename: str) -> None:
    d = os.path.join(out_dir, subdir)
    os.makedirs(d, exist_ok=True)
    img.save(os.path.join(d, filename), compress_level=1)


def _save_arr(arr: np.ndarray, out_dir: str, subdir: str, filename: str, mode: str = "RGB") -> None:
    """Save a float array in [0,1] (HWC for RGB, HW for L) as a PNG under ``out_dir/subdir``."""
    a = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    img = Image.fromarray(a)
    if mode == "L" and img.mode != "L":
        img = img.convert("L")
    _save(img, out_dir, subdir, filename)


def _flatten_and_dedup(
    gathered_gd: List[Optional[dict]], gathered_si: List[Optional[dict]]
) -> Dict[str, Dict[str, list]]:
    """Merge per-rank generated dicts by anomaly name, then drop duplicate samples by ``index``.

    Ranks may repeat testcases (_build_val_batch_indices pads groups to a multiple of world_size), so the
    same ``index`` can appear more than once — keep the first occurrence. All parallel per-name
    lists are filtered at the same positions.
    """
    flat: Dict[str, Dict[str, list]] = {}
    flat_idx: Dict[str, list] = {}
    for gd, si in zip(gathered_gd, gathered_si):
        for name, d in (gd or {}).items():
            fb = flat.setdefault(name, {})
            for k, lst in d.items():
                fb.setdefault(k, []).extend(lst)
            flat_idx.setdefault(name, []).extend((si or {}).get(name, []))

    seen: set = set()
    out: Dict[str, Dict[str, list]] = {}
    for name, fb in flat.items():
        keep = []
        for pos, ix in enumerate(flat_idx.get(name, [])):
            if ix in seen:
                continue
            seen.add(ix)
            keep.append(pos)
        out[name] = {k: [lst[p] for p in keep] for k, lst in fb.items()}
    return out


class ValidationKPI(Callback):
    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        top_k: int = 3,
        # nn feature-extraction / pooling toggles (see eval.correspondence). Defaults are the
        # validated best setting; override in config to reproduce the old full+final+mean nn.
        nn_layer: int = DEFAULT_NN_LAYER,
        nn_readout: str = DEFAULT_NN_READOUT,
        nn_region_policy: str = DEFAULT_NN_REGION_POLICY,
        nn_inst_agg: str = DEFAULT_NN_INST_AGG,
        # When True, also score FID alongside NN/MNN; if its checkpoint can't load the FID step is skipped (non-fatal).
        compute_fid: bool = True,
        # When True (default), also score the anomaly-quality axes (completeness / precision /
        # boundary_iou) and the aq_nn composite (= completeness + nn_score). These need SAM2;
        # skipped non-fatally if it can't load. Set False to skip that per-validation model load.
        compute_anomaly_quality: bool = True,
        output_subdir: str = "valid",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.top_k = top_k
        self.nn_layer = nn_layer
        self.nn_readout = nn_readout
        self.nn_region_policy = nn_region_policy
        self.nn_inst_agg = nn_inst_agg
        self.compute_fid = compute_fid
        self.compute_anomaly_quality = compute_anomaly_quality
        self.output_subdir = output_subdir
        self._run_dir: Optional[str] = None
        self._out_dir: Optional[str] = None
        # Always the training dataset_dir; resolved in on_train_start.
        self.real_root: Optional[str] = None

    def on_train_start(self, model, iteration: int = 0) -> None:
        cfg = getattr(self, "config", None)
        job = getattr(cfg, "job", None)
        self._run_dir = getattr(job, "path_local", None) or os.getcwd()
        self.real_root = self._infer_dataset_dir(cfg)

        log.info(f"ValidationKPI: real references root = training dataset_dir ({self.real_root}).")

    @staticmethod
    def _infer_dataset_dir(cfg) -> Optional[str]:
        """The training ``dataset_dir`` from the composed config (under ``dataloader_train.dataloader.datasets``)."""
        datasets = cfg.dataloader_train.dataloader.datasets
        for entry in datasets.values():
            dataset = entry["dataset"] if isinstance(entry, dict) else getattr(entry, "dataset", None)
            dataset_dir = (
                dataset.get("dataset_dir") if hasattr(dataset, "get") else getattr(dataset, "dataset_dir", None)
            )
            if dataset_dir:
                return str(dataset_dir)
        return None

    def on_validation_start(self, model, dataloader, iteration: int = 0) -> None:
        out_dir = os.path.join(self._run_dir or os.getcwd(), self.output_subdir, str(iteration))
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        # Reset the per-validation-pass accumulators the model appends to in each validation_step:
        #   validation_generated_images_by_anomaly: "{texture}+{defect}" -> dict of parallel per-sample
        #     lists (reconstructed_image / original_image / original_mask / cropped_* / annotated_image /
        #     mask_cropped_image), one entry per generated testcase;
        #   validation_sample_indices_by_anomaly: same key -> the sample `index` of each appended entry,
        #     used to dedup the batch-padding repeats when results are gathered across ranks below.
        model.validation_generated_images_by_anomaly = {}
        model.validation_sample_indices_by_anomaly = {}
        log.info(f"ValidationKPI[iter={iteration}]: validation started")

    def on_validation_end(self, model, iteration: int = 0) -> Optional[dict]:
        # Generation ran (sharded, batched) in model.validation_step; here we gather every rank's
        # results, dedup the batch-index padding repeats, then score + write on rank 0 only.
        gd = getattr(model, "validation_generated_images_by_anomaly", {})
        si = getattr(model, "validation_sample_indices_by_anomaly", {})
        world = distributed.get_world_size()
        if world > 1:
            gathered_gd: List[Optional[dict]] = [None] * world
            gathered_si: List[Optional[dict]] = [None] * world
            torch.distributed.all_gather_object(gathered_gd, gd)
            torch.distributed.all_gather_object(gathered_si, si)
            distributed.barrier()
        else:
            gathered_gd, gathered_si = [gd], [si]

        kpi = None
        if distributed.is_rank0():
            generated = _flatten_and_dedup(gathered_gd, gathered_si)
            kpi = self._score_and_write(self._out_dir or os.getcwd(), generated, iteration)
        if world > 1:
            distributed.barrier()
        log.info(f"ValidationKPI[iter={iteration}]: validation finished — artifacts in {self._out_dir}")
        return kpi

    @torch.inference_mode()
    def _score_and_write(self, out_dir: str, generated: Dict[str, Dict[str, list]], iteration: int) -> Optional[dict]:
        """Save all gathered arrays as PNGs, score NN/MNN (+FID), and write valid_kpi.csv."""
        if not generated:
            log.warning("ValidationKPI: no testcases generated — skipping KPI.")
            return None

        # Assign a per-name output id in name-sorted order, save every array, and build the composite
        # img_path used for the per-sample KPI rows.
        for name in sorted(generated):
            g = generated[name]
            g["img_path"] = []
            for idx in range(len(g["reconstructed_image"])):
                _save_arr(g["reconstructed_image"][idx], out_dir, "reconstructed_image", f"{name}_{idx:05d}.png")
                _save_arr(g["original_image"][idx], out_dir, "original_image", f"{name}_{idx:05d}.png")
                _save_arr(g["original_mask"][idx], out_dir, "original_mask", f"{name}_{idx:05d}.png", mode="L")
                for sub in ("cropped_image", "cropped_mask", "annotated_image", "mask_cropped_image"):
                    m = "L" if sub == "cropped_mask" else "RGB"
                    for inst_idx, arr in enumerate(g.get(sub, [[]] * (idx + 1))[idx]):
                        _save_arr(arr, out_dir, sub, f"{name}_{idx:05d}_{inst_idx:05d}.png", mode=m)
                g["img_path"].append(os.path.join("reconstructed_image", f"{name}_{idx:05d}.png"))

        real = self._load_real_refs(sorted(generated.keys()), out_dir)
        kpi = compute_correspondence_kpi(
            real,
            generated,
            backbone=self.backbone,
            top_k=self.top_k,
            layer=self.nn_layer,
            readout=self.nn_readout,
            region_policy=self.nn_region_policy,
            inst_agg=self.nn_inst_agg,
        )
        if self.compute_fid:
            self._merge_fid(real, generated, kpi)
        if self.compute_anomaly_quality:
            self._merge_anomaly_quality(real, generated, kpi)
        self._write_kpi_matrix(out_dir, sorted(real.keys()), kpi)

        avg = kpi.get("Average", {})
        log.info(
            f"ValidationKPI[iter={iteration}] nn={avg.get('nn_score')} mnn={avg.get('mnn_score')} "
            f"fid={avg.get('fid')} aq_nn={avg.get('aq_nn')}"
        )
        return kpi

    def _merge_anomaly_quality(self, real: dict, generated: dict, kpi: dict) -> None:
        """Score aq_nn + the geometry axes and fold them into ``kpi``; failures are logged and skipped so NN/FID still write."""
        try:
            aq = compute_anomaly_quality_kpi(real, generated, kpi)
        except Exception as e:  # noqa: BLE001 — a scoring failure must not sink the whole validation
            log.warning(f"ValidationKPI: anomaly_quality computation failed ({e}); writing without aq_nn.")
            return
        for name, vals in aq.items():
            kpi.setdefault(name, {}).update(vals)

    def _merge_fid(self, real: dict, generated: dict, kpi: dict) -> None:
        """Score FID and fold it into ``kpi``; failures are logged and skipped so NN/MNN still write."""
        # FID extracts its own flat, DBSCAN-clustered defect crops via mask_crop_images, using the SAME
        # path for real and generated so the two feature sets are comparable. Each generated entry
        # already carries a differently-shaped ``mask_cropped_image`` (a per-sample list of per-instance
        # crops, kept for the artifact PNGs above); leaving it in place trips mask_crop_images'
        # idempotency guard, which then hands those nested lists to compute_feats ('list' has no astype).
        # Drop that key per anomaly so FID recomputes fresh — matching the standalone evaluate.py path.
        gen_for_fid = {name: {k: v for k, v in d.items() if k != "mask_cropped_image"} for name, d in generated.items()}
        try:
            fid = compute_fid_kpi(real, gen_for_fid)
        except Exception as e:  # noqa: BLE001
            log.warning(f"ValidationKPI: FID computation failed ({e}); writing NN/MNN only.")
            return

        for name, vals in fid.items():
            kpi.setdefault(name, {}).update(vals)

    def _write_kpi_matrix(self, out_dir: str, names: List[str], kpi: Dict) -> None:
        avg = kpi.get("Average", {})
        with open(os.path.join(out_dir, "valid_kpi.csv"), "w", newline="") as f:
            w = csv.writer(f)
            # Averages matrix: one row per KPI type, per-anomaly columns + macro Average.
            w.writerow(["kpi"] + names + ["Average"])
            for kpi_type in sorted(avg.keys()):
                w.writerow([kpi_type] + [kpi[name].get(kpi_type) for name in names] + [avg.get(kpi_type)])
            # Per-sample NN/MNN below a blank separator (FID has no per-sample value); readers stop at the blank.
            self._write_per_sample(w, names, kpi)

    @staticmethod
    def _write_per_sample(w, names: List[str], kpi: Dict) -> None:
        rows = []
        for name in names:
            for entry in kpi.get(name, {}).get("per_sample", []):
                rows.append([name, entry.get("path"), entry.get("nn_score"), entry.get("mnn_score")])
        if not rows:
            return
        w.writerow([])
        w.writerow(["per_sample"])
        w.writerow(["anomaly_type", "sample", "nn_score", "mnn_score"])
        w.writerows(rows)

    def _load_real_refs(self, anomaly_names: List[str], out_dir: str) -> Dict[str, Dict[str, List]]:
        """Real references from ``{texture}/anomaly_image/{defect}`` images paired with ``{texture}/mask/{defect}/<stem>_mask.png``."""
        real: Dict[str, Dict[str, List]] = {}
        for name in anomaly_names:
            texture, _, defect = name.partition("+")
            pairs = list_image_mask_pairs(
                os.path.join(self.real_root, texture, "anomaly_image", defect),
                os.path.join(self.real_root, texture, "mask", defect),
                mask_suffix="_mask",
            )
            imgs, masks = [], []
            for idx, (img_path, mask_path) in enumerate(pairs):
                if not os.path.exists(mask_path):
                    continue
                rimg = Image.open(img_path).convert("RGB")
                rmask = Image.open(mask_path).convert("L")
                require_same_size(rimg, rmask, f"{img_path} + {mask_path}")
                _save(rimg, out_dir, "original_image", f"real_{name}_{idx:05d}.png")
                _save(rmask, out_dir, "original_mask", f"real_{name}_{idx:05d}.png")
                imgs.append(_img_to_float(rimg))
                masks.append(_mask_to_float(rmask))

            real[name] = {"original_image": imgs, "original_mask": masks}
            if not imgs:
                raise RuntimeError(
                    f"ValidationKPI: no real references found for {name} under "
                    f"{os.path.join(self.real_root, texture, 'anomaly_image', defect)} "
                    f"(+ matching {texture}/mask/{defect}/<stem>_mask). KPI scoring cannot run "
                    "without real references — check that the testcase anomaly_type matches the "
                    "training dataset and that the mask files exist."
                )
        return real
