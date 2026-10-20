<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 4 · Fine-tuning

This is the fourth page of the CDF AnomalyGen texture tutorial series:

1. [Overview & Setup](1-overview.md)
2. [Dataset Preparation](2-dataset-preparation.md)
3. [Agentic Mask Generation](3-agentic-mask-generation.md) (or classic [Auto Mask Placement](3-auto-mask-placement.md))
4. **Fine-tuning** ←
5. [Generation](5-generation.md)
6. [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)
7. [Agentic End-to-End Flow](7-agentic-end-to-end-flow.md)

It assumes you finished [`2-dataset-preparation.md`](2-dataset-preparation.md) — the prepared
`datasets/phone_screen/` — and page 3 ([Agentic Mask Generation](3-agentic-mask-generation.md) or
classic [Auto Mask Placement](3-auto-mask-placement.md)), which built the validation
`datasets/validation_phone_screen/testcase.jsonl`.

Fine-tuning adapts the base Cosmos3 network into an anomaly-inpainting model for *your* defect types.
It trains only a small subset of parameters (per-defect-type LoRA adapters) on top of a frozen
base network, so runs are cheap and checkpoints are tiny. All commands run from the repo root.

## Prerequisites

Work from the environment shell you set up in [`1-overview.md`](1-overview.md) — either the activated
native venv (`source .venv/bin/activate`) or an interactive container shell (`$DOCKER -it bash`). All
commands below run from the repo root and are identical either way.

## The fine-tuning recipe

Open `ag_config/exp_texture_ft_phone_screen.yaml`. The top of the file identifies the run and the builder:

```yaml
task_type: texture_ft                    # selects the exp_config.py builder
experiment: anomalygen_texture_ft        # experiment node name (referenced on the CLI)
job_name: anomalygen_texture_ft          # this run's name (becomes the output folder name)
```

The **required dataset block** wires the fine-tuning recipe to the data you prepared in
[`2-dataset-preparation.md`](2-dataset-preparation.md) and the validation testcases built on page 3:

```yaml
# --- dataset (required) ---
dataset_name: phone_screen
anomaly_types: [[Phone, oil], [Phone, scratch], [Phone, stain]]
dataset_path: datasets/phone_screen
testcase_jsonl: datasets/validation_phone_screen/testcase.jsonl
```

- `dataset_name` — a label; also becomes the `<DATASET>` grouping folder in the output tree
  (see "Output directory structure" below).
- `anomaly_types` — the list of `[texture, defect]` pairs to train on. Each pair must have matching data
  on disk: for `[Phone, oil]` you need `datasets/phone_screen/Phone/anomaly_image/oil/` **and**
  `datasets/phone_screen/Phone/mask/oil/`. The `prepare_phone_screen_defect` script from
  [`2-dataset-preparation.md`](2-dataset-preparation.md) produced exactly
  this layout, so the three pairs above map onto the `oil` / `scratch` / `stain` subfolders.
- `dataset_path` — the prepared dataset root.
- `testcase_jsonl` — the JSONL of validation testcases used for the training-time validation passes.

### Key training knobs

All knobs are optional and live in the fine-tuning recipe. The values below are the **phone_screen
fine-tuning recipe's actual values**, which are also the builder defaults — deleting a key leaves
behaviour unchanged.

| Knob                    | Value   | Meaning                                    |
| ----------------------- | ------- | ------------------------------------------ |
| `max_iter`              | `15000` | total training steps                       |
| `validation_iter`       | `1000`  | validate every N steps                     |
| `save_iter`             | `1000`  | checkpoint every N steps                   |
| `lr`                    | `1e-3`  | AdamW base learning rate                   |
| `per_class_lora_rank`   | `8`     | per-defect-type adapter rank               |
| `per_class_lora_alpha`  | `8`     | per-defect-type adapter scale              |
| `model_size`            | `nano`  | backbone: `nano`, or `edge` (experimental) |

Keep `validation_iter` a multiple of `save_iter`: `best_checkpoint.txt` (see "Output directory
structure") can only name an iteration that was both validated and saved, so with mismatched values it
is never written.

If you change `max_iter`, set `cycle_lengths` to the same value. `cycle_lengths` sets how long the
learning-rate schedule lasts, and training crashes if `max_iter` is larger.

> **Input size is handled for you.** Each sample is cropped around its defect and fitted to an input
> size picked from that defect's size, so there is no image resolution to set. The defaults are
> dataset-independent; generation reads them back off the recipe.
>
> **The augmentation probabilities are deliberately not in the sample recipe.**
> `inst_aug_prob` defaults to `0.2`, so instance drop is on. `background_dropout_prob` and
> `ring_jitter_prob` default to `0.0`, i.e. off. They are settable if you are deliberately running an
> ablation.

**`model_size` is the one to get right up front:** it picks the frozen Cosmos3 backbone (`nano` =
Qwen3-VL-8B, `edge` = Nemotron-3 Dense VL 2B) *and* the base checkpoint generation loads
(`checkpoints/Cosmos3-Nano` / `checkpoints/Cosmos3-Edge`), so adapters trained at one size cannot be
generated with the other. Omitting it falls back to `nano`.

> **⚠ Use `nano` unless you have a specific reason not to.** `edge` is **experimental**: it is
> wired up end to end, but its generation quality has not been validated against `nano` on any
> dataset, and no result in this tutorial was produced with it. Treat `edge` output as unqualified
> until you have benchmarked it on your own data.

## Run the training

```shell
IMAGINAIRE_OUTPUT_ROOT="$PWD/results" \
torchrun --nproc_per_node=1 anomalygen/scripts/texture/train.py \
    --config=cosmos_framework/configs/base/config.py \
    --recipe=ag_config/exp_texture_ft_phone_screen.yaml \
    -- experiment=anomalygen_texture_ft
```

- **`IMAGINAIRE_OUTPUT_ROOT`** sets where runs land (here `./results`).
- **`--nproc_per_node`** is the number of GPUs. Use `--nproc_per_node=8` for an 8-GPU node; the effective
  batch scales with it.
- **`--recipe`** points at your fine-tuning recipe (`ag_config/exp_texture_ft_phone_screen.yaml`) — the
  file you configured above. It supplies the dataset block and the training knobs; swap in
  `--recipe=ag_config/<your_recipe>.yaml` to train on your own data.
- **Quick smoke test:** append config overrides after the bare `--`, e.g.
  `trainer.max_iter=20 trainer.validation_iter=10 checkpoint.save_iter=10`, to confirm the pipeline
  runs end-to-end before committing to a full training. Use a fresh `IMAGINAIRE_OUTPUT_ROOT` — the
  trainer resumes from any checkpoint it finds — and set `save_iter` alongside the other two, or a
  20-iteration run writes no checkpoint at its default of `1000`. A dry run can leave `cycle_lengths` as it is.
- **Run time.** Measured per-iteration times and total wall times (mask generation included) for a
  15000-iteration fine-tune of the three example datasets are in
  [`7-agentic-end-to-end-flow.md`](7-agentic-end-to-end-flow.md#end-to-end-timing).
  `max_iter`, `validation_iter` and `save_iter` count optimizer steps, and each rank packs its own
  samples, so 8 GPUs see 8× the samples per iteration and each step takes longer; to reproduce the
  single-GPU sample budget, divide the iteration counts by the GPU count (≈ 1875 for `max_iter: 15000`),
  and set `cycle_lengths` to the new `max_iter`.

### (Optional) High-resolution training (2048 px)

With our variable model input size mechanism, each sample is fitted to a size drawn from
`input_size_ladder`, which defaults to `[256, 512, 1024, 1536]` — so it tops out at 1536 px. To
train and generate at 2048 px — worth trying when a defect's fine structure is still lost at
1536 px — add two keys in your recipe:

```yaml
input_size_ladder: [256, 512, 1024, 2048]
max_sequence_length: 8704
```

**Both are required.** A step packs at most `max_sequence_length` tokens, and one 2048 px sample
needs **8453 tokens** — more than the default budget of 5120. Raising the ladder alone fails at
startup with an explicit error.

| largest size | tokens/sample | minimum `max_sequence_length` |
| ------------ | ------------- | ----------------------------- |
| 1536 px      | 4869 tokens   | 4870 (default `5120`)         |
| 2048 px      | 8453 tokens   | 8454 (use `8704`)             |

Then train with the expandable allocator, which cuts fragmentation at these sizes:

```shell
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
IMAGINAIRE_OUTPUT_ROOT="$PWD/results" \
torchrun --nproc_per_node=1 anomalygen/scripts/texture/train.py \
    --config=cosmos_framework/configs/base/config.py \
    --recipe=ag_config/<your_recipe>.yaml \
    -- experiment=anomalygen_texture_ft
```

**Notes:**

- **Tokens per sample** are `2 * ((size_px / 32)² + 2) + 257`, if you need a size not listed above.
- **Cost:** A 2048 px sample is ~1.7x the tokens of a 1536 px one, so a step runs slower.
- **Generation must use the same recipe.** The checkpoint holds only the trained weights, so
  `generate.py` takes the geometry from its `--recipe` — and that flag *defaults* to the
  phone_screen example. Point it at the run dir's own copy
  (`results/…/<JOB_NAME>/<your_recipe>.yaml`): the checkpoint's stamp refuses a recipe with a
  different `input_size_ladder`, crop mode, model size or class list, but `crop_ratio_range` is not
  stamped, so a wrong recipe silently changes the framing.

### (Optional) Fixed-input-size training (the legacy v1.1 framing, 512 px)

Try this if the model does not fill the mask region accurately — low `completeness` or
`boundary_iou` in the evaluation KPIs (see
[`6-evaluation-and-refinement.md`](6-evaluation-and-refinement.md)). It restores the legacy (v1.1)
framing and augmentations: every sample cropped around its defect onto one fixed 512 px input, with
that release's augmentation settings. "Legacy" is the word the code and its messages use for this
mechanism and for checkpoints trained with it.

Ready-made recipes ship for the three tutorial datasets. Each is a copy of its base recipe in which
only `job_name` and the "legacy mechanism" block differ:

- `ag_config/exp_texture_ft_phone_screen_legacy.yaml` — or its Python form,
  `--recipe=ag_config.exp_texture_ft_phone_screen_legacy`
- `ag_config/exp_texture_ft_pcb_legacy.yaml`
- `ag_config/exp_texture_ft_magnetic_tile_legacy.yaml`

Point `--recipe` at one of them; the rest of the training command is unchanged:

```shell
IMAGINAIRE_OUTPUT_ROOT="$PWD/results" \
torchrun --nproc_per_node=1 anomalygen/scripts/texture/train.py \
    --config=cosmos_framework/configs/base/config.py \
    --recipe=ag_config/exp_texture_ft_phone_screen_legacy.yaml \
    -- experiment=anomalygen_texture_ft
```

The run lands under `results/anomalygen/phone_screen/anomalygen_texture_ft_legacy/`.

For your own dataset, copy your recipe and add the same block:

```yaml
input_size_ladder: [512]
crop_mode: ratio
crop_ratio_range: [1.5, 8.0]
background_dropout_prob: 0.5
ring_jitter_prob: 0.5
inst_aug_prob: 0.5
```

One other setting has changed since the legacy v1.1 release and is left at its new value in these
recipes: `shift` (5 → 1), while the quality difference it makes is small enough to ignore. Keep `low < high` in
`crop_ratio_range` (with `low` at least 1.5): a pinned band such as `[1.5, 1.5]` gives training a
single framing, and the recipe is refused. The loss split and the interval-gated CFG also changed,
but those are code, not recipe keys.

Generation takes the framing from its `--recipe`, so point `generate.py` at the run dir's own copy
(`results/…/<JOB_NAME>/<your_recipe>.yaml`) — the same requirement as the section above.

**Legacy (v1.1) checkpoints** generate as they are: pass the run dir's own recipe copy as
`--recipe`. Its legacy keys (`image_size`, `model_input_size`, `ratio_range`) are read as the block
above, and the checkpoint, which has no geometry stamp, is stamped from that recipe at load. A
legacy checkpoint cannot be resumed for training.

### Output directory structure

Runs land under `IMAGINAIRE_OUTPUT_ROOT` at:

```text
results/anomalygen/<DATASET>/<JOB_NAME>/
```

where `<DATASET>` and `<JOB_NAME>` come from the recipe's `job.group` / `job.name` — i.e. its
`dataset_name` and `job_name`. For the phone_screen fine-tuning recipe that is
`results/anomalygen/phone_screen/<job_name>/`.

```text
results/anomalygen/phone_screen/<JOB_NAME>/
├── config.yaml                       # fully-resolved config for this run (recipe + builder merged)
├── config.pkl                        # same config, pickled (used on resume)
├── <your_recipe>.yaml                # verbatim copy of the fine-tuning recipe used for this run
├── training_curves.png               # loss + validation-KPI curves over training
├── training_loss.png                 # training-loss curve
├── training_loss.csv                 # per-logged-step loss values
├── stdout.log                        # full training log
├── checkpoints/
│   ├── latest_checkpoint.txt         # filename of the newest checkpoint (used to resume)
│   ├── best_checkpoint.txt           # filename of the best-scoring one (what generation uses)
│   ├── model/                        # THE weights you generate with — per-defect-type LoRA adapters only (tiny)
│   │   ├── iter_000001000.pt         # one file per save_iter step (here every 1000 iters)
│   │   ├── iter_000002000.pt
│   │   └── …                         # up to iter_000015000.pt
│   ├── optim/                        # optimizer state per checkpoint (for resuming; not needed to generate)
│   ├── scheduler/                    # LR-scheduler state per checkpoint
│   └── trainer/                      # trainer/RNG state per checkpoint
└── valid/                            # generated validation samples, one folder per validation pass
    ├── 0/                            # iter 0 (the pre-training pass)
    │   ├── reconstructed_image/      # the generated defect images at this step
    │   ├── original_image/           # the clean inputs
    │   ├── original_mask/            # the input masks
    │   ├── cropped_image/            # crop-and-paste / inspection aids
    │   ├── cropped_mask/
    │   ├── mask_cropped_image/
    │   ├── annotated_image/          # mask overlaid on the result
    │   └── valid_kpi.csv             # per-type KPI for this step (NN/MNN/FID + the aq axes)
    ├── 1000/
    ├── 2000/
    └── …                             # one folder per validation_iter step
```

**Notes:**

- Because the trainer resumes from any checkpoint in this dir (via `latest_checkpoint.txt`), use a
  fresh `job_name` in the recipe (or a new `IMAGINAIRE_OUTPUT_ROOT`) when you want a clean run.
- **The two pointer files do different jobs.** `latest_checkpoint.txt` names the *last* iteration and
  is what resume reads — leave it alone. `best_checkpoint.txt` is written once at train end, names the
  best-scoring iteration by `early_stop_metric` (default `nn`, excluding iterations below 7500 on a
  run longer than that), and is what `--checkpoint <run_dir>` resolves through in
  [`5-generation.md`](5-generation.md). Runs routinely peak mid-training, so the latest is often not
  the best. If it is missing after a completed run, `stdout.log` says why — generation would
  otherwise fall back to the latest checkpoint and only note it in its log.
- **`valid/<step>/reconstructed_image/` is the folder to watch.** These are the actual defect images the
  model synthesized at that `validation_iter` step — open them to see how generation quality improves as
  training progresses. The sibling subfolders (`original_image/`, `original_mask/`, the `cropped_*` /
  `annotated_image/` aids) are just the inputs and inspection views, and `valid_kpi.csv` records the
  per-type scores for that step — so you can eyeball progress without waiting for the end.
- **Anomaly-quality metrics.** Each validation also scores three geometry axes —
  `completeness` (gated-SAM coverage of the worst mask part), `precision` (change kept inside the
  mask), `boundary_iou` — plus the composite `aq_nn = completeness + nn_score`. `training_curves.png`
  plots the **six** per-sample metrics — `nn`, `mnn`, `fid`, `completeness`, `precision`,
  `boundary_iou`; the `aq_nn` composite is written to `valid_kpi.csv` and the training report (and is
  selectable via `early_stop_metric="aq_nn"`) but is **not** drawn on the curve. These need the SAM2
  model, so set `compute_anomaly_quality: false` in the recipe to skip that per-validation model load.
- **The `checkpoints/model/iter_*.pt` files are what you generate with.** They store only the trained
  per-defect-type LoRA parameters (not the frozen base network), so they are tiny.
  The `optim/` `scheduler/` `trainer/` folders exist only to resume training.
- **⚠ Small `/dev/shm`.** If the run aborts with a `/torch_*` shared-memory error, prepend
  `PYTHONPATH="$PWD/shmpatch"` to the command. It stops the dataloader from creating `/dev/shm/torch_*`
  files at all. As a further fallback you can also cut the worker counts, adding
  `dataloader_train.dataloader.num_workers=0 dataloader_val.num_workers=0` after the bare `--`.

## Fine-tuning on your own data

1. Copy the example fine-tuning recipe:

   ```shell
   cp ag_config/exp_texture_ft_phone_screen.yaml ag_config/exp_texture_ft_mydata.yaml
   ```

2. Edit the dataset block — `dataset_name`, `anomaly_types`, `dataset_path`, `testcase_jsonl` — to point
   at your prepared dataset (built the same way as in [`2-dataset-preparation.md`](2-dataset-preparation.md)).
   Make sure each `[texture, defect]` pair has a
   matching `{dataset_path}/{texture}/anomaly_image/{defect}/` and `{dataset_path}/{texture}/mask/{defect}/`.
3. Adjust a couple of common knobs if needed (`max_iter`, `lr`).
4. Train with your fine-tuning recipe:

   ```shell
   IMAGINAIRE_OUTPUT_ROOT="$PWD/results" \
   torchrun --nproc_per_node=1 anomalygen/scripts/texture/train.py \
       --config=cosmos_framework/configs/base/config.py \
       --recipe=ag_config/exp_texture_ft_mydata.yaml \
       -- experiment=anomalygen_texture_ft
   ```

## Next step

Continue to [5 · Generation](5-generation.md).
