# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone (optionally multi-GPU) synthetic-data-generation over a JSONL of testcases.

Built on this repo's generation backend (``inpaint`` + ``iterative`` + ``crop_paste``). Reads a
JSONL, sorts by mask instance count, splits work round-robin across ranks, generates each sample,
and writes an output tree + ``texture_ft_generation_result.csv`` + ``timing_summary.json``.

Content-safety guardrail (Cosmos 3 parity, on by default; pass ``--no-guardrail`` to disable):
each sample's caption is screened by the text guardrail (Blocklist + Qwen3Guard) before generation,
and every generated composite is passed through the image guardrail (RetinaFace face-blur) before it
is saved. Scope is deliberately the final ``reconstructed_image`` composite only: the input
``original_image``/``original_mask`` are user-provided passthroughs, and the intermediate artifacts
(``annotated_image``, ``cropped_image``, ``cropped_mask``, ``mask_cropped_image``) are internal debug
views — all are written unguarded. A blocked sample is skipped (none of its outputs are written) and
recorded in ``guardrail_blocked.csv``, kept separate from ``texture_ft_generation_result.csv`` so the
downstream pseudo-label/filter steps (which read it by ``output_filename``) never see rows for files
that were not written. The batch keeps going, and ``timing_summary.json`` gains ``guardrail_enabled``
/ ``guardrail_blocked_total`` / ``guardrail_init_seconds`` / ``guardrail_seconds``.

What each guardrail can actually do differs, so the summary reports capability separately from the
flag. ``guardrail_enabled`` is what was asked for; ``text_guardrail_enforcing`` and
``image_guardrail_enforcing`` are what the run could do. Text screening blocks. Image screening
currently **cannot** — the framework preset supplies no image safety model, so only the RetinaFace
face-blur postprocessor runs and no generated image is ever rejected. That is why the field exists:
reading ``guardrail_enabled: true`` alone would suggest image content was screened when it was not.

Run (single GPU):
    python anomalygen/scripts/texture/generate.py \\
        --checkpoint <ckpt> --input_data_path <testcase.jsonl> \\
        --output_dir inference_output --recipe ag_config/exp_texture_ft_phone_screen.yaml

Run (multi GPU):
    torchrun --nproc_per_node=$NPROC anomalygen/scripts/texture/generate.py <same args>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import PIL
import PIL.Image

# Keep Hugging Face model downloads (the content-safety guardrail models, and the base model's
# HF-resolved tokenizer) under the repo's checkpoints/hf. Anchored to the repo root (this file is at
# anomalygen/scripts/texture/generate.py) rather than the CWD, so the pre-fetched cache is found no
# matter where generate.py is launched from. A user-provided HF_HUB_CACHE is respected.
os.environ.setdefault(
    "HF_HUB_CACHE",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "checkpoints", "hf")),
)

# Framework process setup (inference env, grad disabled, distributed init when WORLD_SIZE>1).
# Must run before the heavy imports below, mirroring anomalygen/scripts/texture/train.py.
from cosmos_framework.inference.common.init import init_script  # noqa: E402

# Guarded because it disables autograd process-wide: torchrun and `python generate.py` both run this
# as __main__, so the CLI is unaffected, while importing the module leaves the process alone.
if __name__ == "__main__":
    init_script(training=False)


import torch.distributed as dist  # noqa: E402
from cosmos_framework.utils import distributed, log  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import anomalygen  # noqa: F401,E402  (registration side effects)
from anomalygen.configs.texture.constants import (  # noqa: E402
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_NUM_STEPS,
)
from anomalygen.data.crop_geometry import clamp_crop_ratio, input_size_spec_from_config  # noqa: E402
from anomalygen.data.inpaint_inference_dataset import (  # noqa: E402
    SEED_OUTPUT_STRIDE,
    SEED_RECORD_STRIDE,
    InpaintInferenceDataset,
)
from anomalygen.data.utils import anomaly_class_ids, caption_for_anomaly_type  # noqa: E402
from anomalygen.inference.guardrail import (  # noqa: E402
    BLOCKED_CSV_HEADER,
    blocked_row,
    create_guardrail_runners,
    guard_image,
    is_enforcing,
)
from anomalygen.inference.inpaint import build_inpaint_batch_fn, load_for_inference  # noqa: E402
from anomalygen.inference.iterative import run_iterative_inpaint_batch  # noqa: E402

CSV_HEADER = [
    "output_filename",
    "image_filename",
    "mask_filename",
    "anomaly_type",
    "guidance",
    "num_steps",
    "seed",
    "num_generated_images",
    "crop_ratio",
    "poisson_blend",
    "PSNR",
    "index",
]

# Subdirs written per output, mirroring the other repo's tree. Artifact subdirs come from
# run_iterative_inpaint_batch's artifacts dict (cropped_image / cropped_mask / annotated_image / ...).
_RECON_SUBDIR = "reconstructed_image"
_ORIG_IMAGE_SUBDIR = "original_image"
_ORIG_MASK_SUBDIR = "original_mask"


# --- Distributed work-splitting and result-merging ---------------------------------------------
# Process-group setup is handled by ``init_script`` above, so these are pure helpers: round-robin
# work assignment, the per-sample output plan (stable ids that survive the instance-sort and the
# rank split), and the rank-0 merges.


@dataclass(frozen=True)
class SampleOutputPlan:
    global_order: int  # position in the sorted input
    anomaly_offset: int  # cumulative output count for this anomaly_type before this sample


def _get_rank_work_items(total_items: int, rank: int, world_size: int) -> list[int]:
    """Round-robin item indices for ``rank``: ``[rank, rank+world_size, ...]``."""
    if total_items < 0:
        raise ValueError(f"total_items must be >= 0, got {total_items}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, total_items, world_size))


def _build_sample_output_plans(input_data: list[dict]) -> dict[int, SampleOutputPlan]:
    """Map each sample's ``index`` -> :class:`SampleOutputPlan`.

    Computed over the *full* (sorted) input before the rank split, so every rank derives the
    same per-anomaly_type output offset and ranks write disjoint, collision-free filenames.
    """
    anomaly_offsets: dict[str, int] = defaultdict(int)
    plans: dict[int, SampleOutputPlan] = {}

    for global_order, sample in enumerate(input_data):
        sample_index = int(sample["index"])
        anomaly_type = sample["anomaly_type"]
        num_outputs = int(sample.get("num_generated_images", 1))

        plans[sample_index] = SampleOutputPlan(
            global_order=global_order,
            anomaly_offset=anomaly_offsets[anomaly_type],
        )
        anomaly_offsets[anomaly_type] += num_outputs

    return plans


def _to_json_safe(value):
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _require_collectives(world_size: int) -> bool:
    """True if cross-rank collectives are needed and available; False for single-GPU."""
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if world_size == 1:
        return False
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before using multi-GPU inference collectives.")
    if dist.get_world_size() != world_size:
        raise RuntimeError(
            f"torch.distributed world size does not match the runtime context: {dist.get_world_size()} != {world_size}"
        )
    return True


def gather_from_ranks(obj, world_size: int) -> list:
    """Every rank's ``obj`` as a list in rank order; ``[obj]`` on a single GPU, with no collective.
    JSON-safe copies on both sides, so numpy scalars pickle and land as plain numbers."""
    payload = _to_json_safe(obj)
    if not _require_collectives(world_size):
        return [payload]
    return [_to_json_safe(item) for item in distributed.all_gather_object(payload)]


def _merge_rank_rows(gathered: list[list[dict]]) -> list[list]:
    """Flatten gathered per-rank CSV rows (each ``{"sort_key": [...], "row": [...]}``), ordered by ``sort_key``."""
    merged = [row for rank_rows in gathered for row in rank_rows]
    merged.sort(key=lambda item: tuple(item["sort_key"]))
    return [row["row"] for row in merged]


def _aggregate_rank_timings(rank_timings: list[dict]) -> dict:
    """Collapse per-rank timings into one summary using max wall-time across active ranks."""
    if not rank_timings:
        raise ValueError("rank_timings must not be empty")

    active = [
        t for t in rank_timings if int(t.get("assigned_samples", 0)) > 0 or int(t.get("generated_images", 0)) > 0
    ] or rank_timings

    def max_seconds(key: str, timings: list[dict]) -> float:
        return max(float(t.get(key, 0.0)) for t in timings)

    generated_images_total = sum(int(t.get("generated_images", 0)) for t in rank_timings)
    generation_seconds = max_seconds("generation_seconds", active)

    summary = {
        "aggregation_method": "single_rank" if len(rank_timings) == 1 else "max_rank_wall_time",
        "world_size": max(int(t.get("world_size", 1)) for t in rank_timings),
        "ranks_with_work": sum(int(t.get("assigned_samples", 0)) > 0 for t in rank_timings),
        "assigned_samples_total": sum(int(t.get("assigned_samples", 0)) for t in rank_timings),
        "generated_images_total": generated_images_total,
        "setup_seconds": max_seconds("setup_seconds", active),
        "model_init_seconds": max_seconds("model_init_seconds", active),
        "generation_seconds": generation_seconds,
        "finalize_seconds": max_seconds("finalize_seconds", rank_timings),
        "measured_total_seconds": max_seconds("measured_total_seconds", rank_timings),
        "guardrail_enabled": any(bool(t.get("guardrail_enabled")) for t in rank_timings),
        # `all`, not `any`: enforcement is only true for the batch if every rank that ran had it.
        "text_guardrail_enforcing": all(bool(t.get("text_guardrail_enforcing")) for t in rank_timings),
        "image_guardrail_enforcing": all(bool(t.get("image_guardrail_enforcing")) for t in rank_timings),
        "guardrail_blocked_total": sum(int(t.get("guardrail_blocked", 0)) for t in rank_timings),
        "guardrail_init_seconds": max_seconds("guardrail_init_seconds", active),
        "guardrail_seconds": max_seconds("guardrail_seconds", active),
        "rank_timings": [_to_json_safe(t) for t in rank_timings],
    }
    summary["generation_seconds_per_image"] = (
        generation_seconds / generated_images_total if generated_images_total > 0 else None
    )
    return summary


def _psnr_in_mask(original: PIL.Image.Image, recon: PIL.Image.Image, mask: PIL.Image.Image):
    """PSNR (dB) between original and reconstructed image, restricted to the mask region."""
    orig = np.asarray(original.convert("RGB"), dtype=np.float32)
    rec = np.asarray(recon.convert("RGB").resize(original.size), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(original.size)) >= 127
    if not m.any():
        return None
    mse = float(np.mean((orig[m] - rec[m]) ** 2))
    if mse <= 0.0:
        return 100.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def _save(img: PIL.Image.Image, out_dir: str, subdir: str, filename: str) -> None:
    d = os.path.join(out_dir, subdir)
    os.makedirs(d, exist_ok=True)
    img.save(os.path.join(d, filename), compress_level=1)


# Room the noise-seed strides leave per testcase / per output (see inpaint_inference_dataset:
# seed = base_seed + index * SEED_RECORD_STRIDE + output_n * SEED_OUTPUT_STRIDE + instance_j).
_MAX_OUTPUTS_PER_TESTCASE = SEED_RECORD_STRIDE // SEED_OUTPUT_STRIDE
_MAX_INSTANCES_PER_OUTPUT = SEED_OUTPUT_STRIDE


def _validate_seed_envelope(input_data: list[dict]) -> None:
    """Reject testcases whose output / instance counts would overflow their seed range.

    Both counts come straight from the user's JSONL. Past the envelope the offsets wrap into the
    next output's (or the next testcase's) seeds and two different generations silently share a
    noise draw — invisible in the results, so fail here instead of producing duplicates.
    """
    for rec in input_data:
        num_outputs = int(rec.get("num_generated_images", 1))
        if num_outputs > _MAX_OUTPUTS_PER_TESTCASE:
            raise ValueError(
                f"Testcase index {rec.get('index')} sets num_generated_images={num_outputs}, above the "
                f"{_MAX_OUTPUTS_PER_TESTCASE} the noise seeds have room for (SEED_RECORD_STRIDE // "
                f"SEED_OUTPUT_STRIDE = {SEED_RECORD_STRIDE} // {SEED_OUTPUT_STRIDE}); its extra outputs "
                "would reuse the next testcase's seeds. Split it across JSONL lines with distinct "
                "'seed' values, or widen SEED_RECORD_STRIDE."
            )
        max_instances = int(rec.get("iteration_generation_max_instance", DEFAULT_MAX_INSTANCES))
        if max_instances > _MAX_INSTANCES_PER_OUTPUT:
            raise ValueError(
                f"Testcase index {rec.get('index')} sets iteration_generation_max_instance="
                f"{max_instances}, above the {_MAX_INSTANCES_PER_OUTPUT} the noise seeds have room for "
                f"(SEED_OUTPUT_STRIDE); its later instances would reuse the next output's seeds. "
                "Lower it, or widen SEED_OUTPUT_STRIDE (and SEED_RECORD_STRIDE with it)."
            )


def main(argv=None) -> None:
    t0 = time.time()
    parser = argparse.ArgumentParser(description="AnomalyGen standalone SDG inference over a JSONL")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="fine-tuned run dir / checkpoints dir / iter_<N>.pt",
    )
    parser.add_argument(
        "--base_checkpoint",
        default=None,
        help="base DCP checkpoint warm-started before overlaying the fine-tuned weights "
        "(default: selected from the recipe's model_size)",
    )
    parser.add_argument("--input_data_path", required=True, help="JSONL of testcases")
    parser.add_argument("--output_dir", default="inference_output", help="output directory")
    parser.add_argument(
        "--recipe",
        default="ag_config/exp_texture_ft_phone_screen.yaml",
        help="the recipe the checkpoint trained under (YAML/JSON file or ag_config module): it builds the "
        "model config, whose anomaly_types give the class ids; the checkpoint's stamp must match it",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--guidance", type=float, default=DEFAULT_GUIDANCE, help="fallback guidance when a testcase omits it"
    )
    parser.add_argument(
        "--num_steps", type=int, default=DEFAULT_NUM_STEPS, help="fallback num_steps when a testcase omits it"
    )
    parser.add_argument(
        "--max_instances", type=int, default=DEFAULT_MAX_INSTANCES, help="fallback iteration_generation_max_instance"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=1,
        help="base for the per-testcase noise seeds (a testcase's explicit 'seed' wins); drives "
        "both the latent and source-item noise, so output is independent of rank count, guardrail "
        "skips and batch composition. Same value reproduces, up to GPU kernel nondeterminism; "
        "bump it to re-roll the noise",
    )
    parser.add_argument(
        "--guardrail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the Cosmos content-safety guardrail (text + face-blur) on generated outputs "
        "(default: on; pass --no-guardrail to disable)",
    )
    parser.add_argument(
        "--offload_guardrail_models",
        action="store_true",
        help="offload the guardrail models to CPU between calls to save VRAM",
    )
    args = parser.parse_args(argv)

    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    dataset = InpaintInferenceDataset(
        args.input_data_path,
        default_guidance=args.guidance,
        default_num_steps=args.num_steps,
        default_max_instances=args.max_instances,
        base_seed=args.base_seed,
    )
    # Fail before the model load if any testcase asks for more outputs / instances than its seed
    # range holds, rather than silently generating two samples from the same noise.
    _validate_seed_envelope(dataset.input_data)
    # Plans are built over the full sorted input (before the rank split) so anomaly_offset is
    # global and ranks write disjoint, collision-free filenames into the shared output dir.
    plans = _build_sample_output_plans(dataset.input_data)
    work_items = _get_rank_work_items(len(dataset), rank, world_size)
    loader = DataLoader(
        Subset(dataset, work_items),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=InpaintInferenceDataset.collate_fn,
    )
    log.info(f"[rank {rank}/{world_size}] assigned {len(work_items)}/{len(dataset)} samples.")

    setup_seconds = time.time() - t0
    model_init_start = time.time()
    model = load_for_inference(args.recipe, args.checkpoint, base_checkpoint=args.base_checkpoint)
    model_init_seconds = time.time() - model_init_start
    # Class ids and geometry come off the loaded model config, which the checkpoint's stamp was just
    # checked against. Both are constant for the run, so they are read once here.
    class_ids = anomaly_class_ids(model.config.diffusion_expert_config.anomaly_types)
    input_spec = input_size_spec_from_config(model.config.diffusion_expert_config, class_ladders=model.class_ladders)

    guardrail_init_start = time.time()
    text_guardrail = video_guardrail = None
    if args.guardrail:
        try:
            text_guardrail, video_guardrail = create_guardrail_runners(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize the content-safety guardrail. Ensure HF_TOKEN is set and the "
                "nvidia/Cosmos-Guardrail1 license is accepted (see scripts/download_checkpoints.sh), "
                "or pass --no-guardrail to disable."
            ) from e
    guardrail_init_seconds = time.time() - guardrail_init_start
    # ``guardrail_enabled`` records the flag the operator passed. These record what the run could
    # actually do: a runner with no safety model always answers "safe", so screening was requested
    # but never enforced. Reported separately so a reader of timing_summary.json is not left
    # inferring enforcement from a flag.
    text_guardrail_enforcing = bool(text_guardrail is not None and is_enforcing(text_guardrail))
    image_guardrail_enforcing = bool(video_guardrail is not None and is_enforcing(video_guardrail))

    if distributed.is_rank0():
        os.makedirs(args.output_dir, exist_ok=True)

    gen_start = time.time()
    rows: list[dict] = []
    blocked: list[dict] = []
    guardrail_seconds = 0.0
    # Captions are templated and repeat across samples of the same anomaly_type, and Qwen3Guard runs
    # a 0.6B LLM per call, so cache each caption's verdict to avoid redundant text checks within a run.
    text_verdicts: dict[str, tuple[bool, str]] = {}
    generated_images = 0
    clamped_testcases = 0
    for rec in loader:
        plan = plans[int(rec["index"])]
        anomaly_type = rec["anomaly_type"]
        image, mask = rec["image"], rec["mask"]

        if text_guardrail is not None:
            _g0 = time.time()
            caption = caption_for_anomaly_type(anomaly_type)
            if caption not in text_verdicts:
                text_verdicts[caption] = text_guardrail.run_safety_check(caption)
            is_safe, message = text_verdicts[caption]
            guardrail_seconds += time.time() - _g0
            if not is_safe:
                log.warning(f"[guardrail] blocked text for index {rec['index']} ({anomaly_type}): {message}")
                blocked.append(
                    blocked_row(
                        rec["index"], -1, anomaly_type, rec["image_filename"], rec["mask_filename"], "text", message
                    )
                )
                continue

        # Validated by InpaintInferenceDataset. Clamped into the trained band here, not only inside the
        # planner, so the CSV records the value generation used and out-of-band requests are counted.
        requested_crop_ratio = float(rec["crop_ratio"])
        crop_ratio = clamp_crop_ratio(requested_crop_ratio, input_spec)
        if crop_ratio != requested_crop_ratio:
            clamped_testcases += 1

        # Per-sample generation params (invariant across the num_outputs loop; only seed varies).
        num_outputs = int(rec["num_generated_images"])
        record_seed = int(rec["seed"])
        num_steps = int(rec["num_steps"])
        guidance = float(rec["guidance"])
        max_instances = int(rec["iteration_generation_max_instance"])
        poisson_blend = bool(rec["poisson_blend"])

        # One image per call through the batch orchestrator, the path validation runs too; depth j of
        # the sample is seeded ``seed + j``, so its instances never share a noise draw.
        inpaint_batch = build_inpaint_batch_fn(model, class_ids, num_steps=num_steps, guidance=guidance)
        for i in range(num_outputs):
            output_idx = plan.anomaly_offset + i
            # Strided so the per-instance seed offsets can't reach the next output.
            seed = record_seed + i * SEED_OUTPUT_STRIDE
            composites, all_artifacts = run_iterative_inpaint_batch(
                [image],
                [mask],
                [anomaly_type],
                inpaint_batch,
                seeds=[seed],
                crop_ratios=[crop_ratio],
                poisson_blends=[poisson_blend],
                max_instances_list=[max_instances],
                input_spec=input_spec,
            )
            composite, artifacts = composites[0], all_artifacts[0]

            # Guard only the final composite (the shipped deliverable). original_image/original_mask
            # are user inputs, and the artifacts saved below are internal debug views — left unguarded.
            if video_guardrail is not None:
                _g0 = time.time()
                composite, block_msg = guard_image(video_guardrail, composite)
                guardrail_seconds += time.time() - _g0
                if composite is None:
                    log.warning(
                        f"[guardrail] blocked image for index {rec['index']} ({anomaly_type}) "
                        f"output {output_idx}: {block_msg}"
                    )
                    blocked.append(
                        blocked_row(
                            rec["index"],
                            output_idx,
                            anomaly_type,
                            rec["image_filename"],
                            rec["mask_filename"],
                            "image",
                            block_msg,
                        )
                    )
                    continue

            filename = f"{anomaly_type}_{output_idx:05d}.png"
            _save(composite, args.output_dir, _RECON_SUBDIR, filename)
            _save(image, args.output_dir, _ORIG_IMAGE_SUBDIR, filename)
            _save(mask, args.output_dir, _ORIG_MASK_SUBDIR, filename)
            for subdir, instances in artifacts.items():
                for inst_idx, inst_img in enumerate(instances):
                    _save(inst_img, args.output_dir, subdir, f"{anomaly_type}_{output_idx:05d}_{inst_idx:05d}.png")

            psnr = _psnr_in_mask(image, composite, mask)
            rows.append(
                {
                    "sort_key": [plan.global_order, i],
                    "row": {
                        "output_filename": filename,
                        "image_filename": rec["image_filename"],
                        "mask_filename": rec["mask_filename"],
                        "anomaly_type": anomaly_type,
                        "guidance": rec["guidance"],
                        "num_steps": rec["num_steps"],
                        "seed": seed,
                        "num_generated_images": num_outputs,
                        "crop_ratio": crop_ratio,
                        "poisson_blend": rec["poisson_blend"],
                        "PSNR": psnr,
                        "index": rec["index"],
                    },
                }
            )
            generated_images += 1

    generation_seconds = time.time() - gen_start

    finalize_start = time.time()
    merged_rows = _merge_rank_rows(gather_from_ranks(rows, world_size))
    clamped_total = sum(gather_from_ranks(clamped_testcases, world_size))
    if distributed.is_rank0() and clamped_total:
        lo, hi = input_spec.crop_ratio_range
        log.warning(
            f"{clamped_total} testcase(s) asked for a crop_ratio outside the trained band [{lo:g}, {hi:g}]; "
            "they were generated at the nearest edge, which is the value the CSV records."
        )
    if distributed.is_rank0():
        csv_path = os.path.join(args.output_dir, "texture_ft_generation_result.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            writer.writerows(merged_rows)
        log.info(f"Wrote {len(merged_rows)} rows to {csv_path}.")

    merged_blocked = _merge_rank_rows(gather_from_ranks(blocked, world_size))
    if distributed.is_rank0() and args.guardrail:
        blocked_path = os.path.join(args.output_dir, "guardrail_blocked.csv")
        with open(blocked_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BLOCKED_CSV_HEADER)
            writer.writeheader()
            writer.writerows(merged_blocked)
        log.info(f"Wrote {len(merged_blocked)} blocked rows to {blocked_path}.")

    timing = {
        "rank": rank,
        "world_size": world_size,
        "assigned_samples": len(work_items),
        "generated_images": generated_images,
        "setup_seconds": setup_seconds,
        "model_init_seconds": model_init_seconds,
        "generation_seconds": generation_seconds,
        "finalize_seconds": time.time() - finalize_start,
        "measured_total_seconds": time.time() - t0,
        "guardrail_enabled": bool(args.guardrail),
        "text_guardrail_enforcing": text_guardrail_enforcing,
        "image_guardrail_enforcing": image_guardrail_enforcing,
        "guardrail_blocked": len(blocked),
        "guardrail_init_seconds": guardrail_init_seconds,
        "guardrail_seconds": guardrail_seconds,
    }
    merged_timings = sorted(gather_from_ranks(timing, world_size), key=lambda t: int(t.get("rank", 0)))
    if distributed.is_rank0():
        summary = _aggregate_rank_timings(merged_timings)
        with open(os.path.join(args.output_dir, "timing_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log.info(
            f"SDG done: {summary['generated_images_total']} images across {world_size} rank(s) "
            f"in {summary['generation_seconds']:.1f}s generation wall-time."
        )


if __name__ == "__main__":
    main()
