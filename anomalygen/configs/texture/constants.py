# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defaults for the texture SDG pipeline — the single source of truth.

Kept dependency-free (no anomalygen or framework imports) so every layer — datasets, the generation
core, the SDG CLI, the model config, the callbacks, and the experiment builder — can import it
without pulling in the heavier ``exp_config`` or risking an import cycle.

Sections follow the pipeline: base model -> crop geometry (shared by training and generation) ->
generation sampling -> training runtime.
"""

# --- Base model ----------------------------------------------------------------------------------

# "nano" (Qwen3-VL-8B reasoner) | "edge" (Nemotron-3 Dense VL 2B). Selects the frozen backbone, so
# the two sizes' checkpoints are not interchangeable.
MODEL_SIZES = ("nano", "edge")
DEFAULT_MODEL_SIZE = "nano"

# Frozen base DCP per size, relative to the repo root; see scripts/download_checkpoints.sh.
BASE_CHECKPOINT_PATHS = {
    "nano": "checkpoints/Cosmos3-Nano",
    "edge": "checkpoints/Cosmos3-Edge",
}


# --- Crop geometry (shared by training and generation) ---------------------------------------------

# One vision token covers PATCH_SPATIAL x VAE_SPATIAL_COMPRESSION pixels per axis. Ladder entries are
# multiples of VISION_TOKEN_PX, and a sample's token count follows from it (data.utils.sample_token_count).
PATCH_SPATIAL = 2
VAE_SPATIAL_COMPRESSION = 16
VISION_TOKEN_PX = PATCH_SPATIAL * VAE_SPATIAL_COMPRESSION

# Legal input sizes per axis for variable-resolution training/inference; each axis snaps to one.
# The ceiling bounds a step's activation memory; the floor is the smallest input a sample can train
# at, so small defects snap up to it and gain extra context (see plan_crop).
# Entries must be multiples of VISION_TOKEN_PX.
DEFAULT_INPUT_SIZE_LADDER = (256, 512, 1024, 1536)

# Ceiling on the largest ladder entry, which is what bounds a step's activation memory. Above the
# default largest size of 1536: a ladder that uses 2048 must also raise ``max_sequence_length``, since
# one 2048 sample costs 8453 tokens (see "High-resolution training" in 4-fine-tuning.md).
MAX_INPUT_SIZE = 2048

# Per-class input-size cap = ceil(max_bbox_in_class * this) on the ladder. Stops a class of small
# defects drawing the largest input size for nothing. Recorded per class beside the checkpoint's
# geometry stamp, so generation plans each class against the same cap. A single-size ladder makes it a no-op.
INPUT_SIZE_CAP_HEADROOM = 4.0

# Where the crop window's framing comes from. Both modes then snap the window to the ladder, so
# ``crop_mode`` and ``input_size_ladder`` stay orthogonal.
#
# "scale": two axes — crop_ratio (context) and scale (log-uniform resample). The input size follows
#   the defect's apparent pixel size, so a large defect draws a large input.
# "ratio": one axis — the window is ``max(bbox) * crop_ratio``, so the defect fills 1/ratio of the
#   input whatever its pixel size. ``scale_range`` and ``inference_scale`` are unused.
CROP_MODES = ("scale", "ratio")
DEFAULT_CROP_MODE = "scale"

# FRAMING (crop_ratio): how much context surrounds the defect. The band a testcase's ``crop_ratio``
# is clamped into at generation. The window then snaps to an input size, so the realized ratio can
# land above the band (small defects at the ladder floor) but never under its low end, unless the
# image itself is smaller (see plan_crop).
DEFAULT_CROP_RATIO_RANGE = (1.5, 2.5)

# Lowest crop_ratio a recipe may ask for, and the least context every window keeps (unless the image
# is smaller). Below this the window is too tight for the model to place a coherent defect in it.
MIN_CROP_RATIO = 1.5

# SCALE ("scale" mode only): resampling as augmentation, drawn LOG-uniformly (see RandomScaleCrop);
# teaches defect-size invariance. Widening it pushes training above the size generation asks for,
# leaving the model trained on input sizes its testcases never plan.
DEFAULT_SCALE_RANGE = (0.333, 3.0)


# --- Generation sampling ---------------------------------------------------------------------------

# Sampler time-shift, one value for every input size: 1 measured better on fine-grained defects at
# every input size. Must stay an int: a float takes a dict lookup instead.
DEFAULT_SHIFT = 1

DEFAULT_NUM_STEPS = 35
DEFAULT_GUIDANCE = 3.0

# Apply CFG only while LO < timestep < HI on the num_train_timesteps (0-1000) scale, taking the
# plain conditional path elsewhere. The unconditional branch is re-evaluated on the GUIDED latent
# every step, so once the trajectory is pushed off the normal manifold the anchor follows it there
# and the two branches converge on the same artifact; the window closes before that feedback builds.
# Both ends are exclusive, so the first step (t == 1000) runs unguided.
# Not a recipe key: it describes how sampling is run, not how a model was trained.
DEFAULT_GUIDANCE_INTERVAL = (200.0, 1000.0)

# Search bounds for the per-sample quality-refine sweep. Wider than DEFAULT_GUIDANCE because the
# sweep looks for a per-sample optimum rather than reproducing one value.
DEFAULT_GUIDANCE_RANGE = (1.0, 10.0)

# Per-testcase generation defaults, overridable per record in the testcase JSONL.
DEFAULT_CROP_RATIO = 2.0
DEFAULT_MAX_INSTANCES = 5


# --- Training runtime ------------------------------------------------------------------------------

# Activation checkpointing for the MoT blocks (recipe key ``activation_checkpointing_mode``). Default
# "none": this task packs few enough tokens per step that recomputing costs more time than the
# activations cost memory. Raise it when a recipe runs out of memory; it changes nothing the model
# trains on.
AC_MODES = ("none", "selective", "full")
DEFAULT_AC_MODE = "none"

# torch.compile recompile ceiling. Torch's default of 8 suits one fixed input size; variable ones
# produce many more distinct packed lengths, and past the ceiling torch falls back to eager.
COMPILE_RECOMPILE_LIMIT = 256

# Iterations below this are not eligible for best_checkpoint.txt. The pick is a plain best-of and nn
# swings widely before the adapters settle, so an early spike would otherwise beat a genuinely
# better late checkpoint. Only applied when the run was *planned* to reach past it — a dry run or a
# short fine-tune must still produce a pointer, so there every iteration stays eligible.
CKPT_WARMUP_ITER = 7500
