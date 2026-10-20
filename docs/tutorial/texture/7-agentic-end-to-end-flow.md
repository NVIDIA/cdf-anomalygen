<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 7 · Agentic End-to-End Flow

This is the seventh page of the CDF AnomalyGen texture tutorial series:

1. [Overview & Setup](1-overview.md)
2. [Dataset Preparation](2-dataset-preparation.md)
3. [Agentic Mask Generation](3-agentic-mask-generation.md) (or classic [Auto Mask Placement](3-auto-mask-placement.md))
4. [Fine-tuning](4-fine-tuning.md)
5. [Generation](5-generation.md)
6. [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)
7. **Agentic End-to-End Flow** ←

Pages 2–6 run the pipeline one stage at a time. This page shows how to run the **whole pipeline from
a single prompt** with an AI coding agent and this repo's `cdf-anomalygen` skill: you describe the run,
and the agent prepares the testcases, fine-tunes (or reuses a checkpoint), generates, evaluates,
refines and pseudo-labels — then hands back the results.

It assumes you have finished [`1-overview.md`](1-overview.md) (environment, checkpoints, a passing
preflight) and have a dataset in the layout of
[`2-dataset-preparation.md`](2-dataset-preparation.md).

All commands are run from the **repo root**.

______________________________________________________________________

## Prerequisites

- The environment shell from [`1-overview.md`](1-overview.md) — either the activated native venv
  (`source .venv/bin/activate`) or an interactive container shell (`$DOCKER -it bash`) — with the
  checkpoints downloaded and `scripts/preflight_env_ckpt.sh` passing.
- An AI coding agent that can load this repo's skills. The pipeline ships as agent skills under
  [`skills/`](../../../skills/); with Claude Code, start `claude` from the repo root and check that
  `cdf-anomalygen` appears in the skill list.
- **Model access for the default mask stage.** The skill's default `mask_backend=amg` (Agentic Mask
  Generation) needs a vision-language model, resolved through the registry
  `~/.anomalygen/endpoints.yaml` set up in
  [`3-agentic-mask-generation.md`](3-agentic-mask-generation.md#prerequisites). If it is missing,
  the agent stops and asks you to supply an API entry, to let it start the self-hosted Gemma server,
  or to set `mask_backend=amp` for the run.

______________________________________________________________________

## What the agent does

The skill encodes every stage's command, the order the stages run in, a **gate** per stage (a step
counts as done only when its output exists on disk, never when its command was merely launched) and
the known failure modes. You pass the run's parameters as `key=value` lines, and the agent asks for
any required input you left out rather than guessing.

The skill numbers its steps by pipeline stage, as the [README](../../../README.md) does; environment
setup (page 1) is the preflight it runs before step 1. Each step feeds the next:

| Step | Stage                     | What it produces                                                                             | Manual page                                                            |
| ---- | ------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1    | Dataset (contract layout) | `dataset_dir` with `anomaly_image/`, `mask/`, `clean_image/` (+ `defect_spec.jsonl` for AMP) | [2](2-dataset-preparation.md)                                          |
| 2    | Mask generation           | validation + generation `testcase.jsonl` of placed defect masks                              | [3](3-agentic-mask-generation.md) (or [AMP](3-auto-mask-placement.md)) |
| 3    | Fine-tuning               | a run dir under `results/anomalygen/<name>/` with per-defect-type LoRA checkpoints           | [4](4-fine-tuning.md)                                                  |
| 4    | Generation (SDG)          | synthetic defect images under `results/generation_<name>/`                                   | [5](5-generation.md)                                                   |
| 5    | Evaluation                | per-type + `Average` KPI JSON (`nn_score`, `mnn_score`, `fid`, anomaly-quality axes)         | [6](6-evaluation-and-refinement.md#evaluation)                         |
| 6    | Quality refinement        | a `searched/` bucket holding each sample's best-scoring render                               | [6](6-evaluation-and-refinement.md#quality-refinement-optional)        |
| 7    | Pseudo-labeling           | COCO annotations, a classification layout, visualizations and optional captions              | [6](6-evaluation-and-refinement.md#pseudo-labeling)                    |

1. **Dataset (contract layout).** Every mode needs `dataset_dir` on disk in the layout the trainer
   expects: `{texture}/anomaly_image/{defect}/`, `{texture}/mask/{defect}/` and
   `{texture}/clean_image/`. The three ready-made `prepare_*` scripts (`pcb`, `magnetic_tile`,
   `phone_screen`) emit it; a custom dataset must match it exactly. The agent confirms the layout
   before moving on (gate **dataset**).
2. **Mask generation.** Builds the `testcase.jsonl` that fine-tuning and generation consume — one
   row per synthetic image, pairing a clean image with a placed defect mask. Two backends produce
   the same artifact: the default **Agentic Mask Generation (AMG)**, a propose → judge → refine loop
   driven by a vision-language model, or the classic **Auto Mask Placement (AMP)**, which places
   training submasks by the routing declared in `defect_spec.jsonl` (`mask_backend=amp`). In `full`
   mode two sets are built — a **validation** set for training-time validation and a **generation**
   set for SDG — each sized by `num_sdg`.
3. **Fine-tuning.** Copies the template recipe, points its dataset block at `dataset_dir` and the
   validation `testcase.jsonl`, and launches training detached through
   `scripts/skill_utility/train_control.sh`. The agent then waits on it (several hours for the
   15000-iteration default — see [End-to-end timing](#end-to-end-timing)) and, once it finishes, takes the **best** checkpoint from `best_checkpoint.txt`
   — not the last one, since runs routinely peak mid-training (gate **train**).
4. **Generation (SDG).** Runs `generate.py` with the best checkpoint, the run dir's own recipe copy
   and the generation `testcase.jsonl`. One image per testcase row, except the samples the
   content-safety guardrail blocks — those are listed in `guardrail_blocked.csv`, so images +
   blocked rows must equal the testcase rows (gate **generate**).
5. **Evaluation.** Scores the batch against the real defect references with `evaluate.py`, producing
   per-type and `Average` blocks. `nn_score` is the primary KPI (higher is better); `mnn_score`,
   `fid` and the anomaly-quality axes (`completeness`, `precision`, `boundary_iou`, `aq_nn`) are
   secondary (gate **eval**).
6. **Quality refinement.** `quality_refine.py run` re-draws `(guidance, crop_ratio)` per sample over
   `num_search_run` rounds of draw → generate → evaluate (round 1 uniform, later rounds per-sample
   Bayesian optimization), then selects each sample's best render into `searched/`. Nothing is
   discarded; the batch stays the same size (gate **refine**).
7. **Pseudo-labeling.** Turns `searched/` into a labeled dataset with `pseudo_label.py`: COCO
   instance annotations, a per-class classification layout, overlay visualizations and, unless
   `--no_caption` is passed, natural-language captions from the Cosmos3-Nano reasoner (gate
   **label**).

______________________________________________________________________

## The end-to-end prompt

Paste the following into your agent session (for Claude Code, started from the repo root with the
built environment active). It runs all seven steps on the `phone_screen` worked example used
throughout this tutorial:

```text
Use AnomalyGen skill with
name=phone_screen
mode=full
dataset_dir=datasets/phone_screen
num_sdg=100
max_iter=15000
validation_iter=1000
save_iter=1000
```

### What each line means

| Line                   | Meaning                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name=phone_screen`    | Experiment label. Names the recipe (`ag_config/exp_texture_ft_<name>.yaml`), the run dir, the logs and the mask-set dirs. It is **never** a dataset location. |
| `mode=full`            | Which steps run: `full` = steps 1 → 7 (train + generate).                                                                                                     |
| `dataset_dir=…`        | The prepared dataset root ([2-dataset-preparation.md](2-dataset-preparation.md)). Required in every mode and never derived from `name`.                       |
| `num_sdg=100`          | How many placed masks / synthetic images to produce, split across the defect types. Sizes both the validation set and the generation set.                     |
| `max_iter=15000`       | Recipe knob: total training steps (the default).                                                                                                              |
| `validation_iter=1000` | Recipe knob: validate every N steps.                                                                                                                          |
| `save_iter=1000`       | Recipe knob: checkpoint every N steps.                                                                                                                        |

The three training knobs are optional and edit the recipe; leaving them out keeps the tested
defaults. The **recipe** is the fine-tuning recipe (page 4): in `full` and `finetune_only` the skill
materialises it at `ag_config/exp_texture_ft_<name>.yaml` from the shipped `phone_screen` template,
editing its dataset block — the path is derived from `name`, so there is nothing to pass. With
`name=phone_screen` that path *is* the shipped example, so the agent edits the tracked file in place;
use another `name` to keep the working tree clean. A `recipe=` path is an input only in
`generation_only` (below).

### Modes

- **`full`** — steps 1 → 7. Required: `name`, `dataset_dir`, `num_sdg`.
- **`finetune_only`** — steps 1 → 3, train only. Required: `name`, `dataset_dir`, `num_sdg`. Here
  `num_sdg` sizes only the validation set.
- **`generation_only`** (a.k.a. `inference_only`) — steps 2 and 4 → 7, generate from an existing
  checkpoint without retraining. Required: `name`, `dataset_dir`, `num_sdg`, `checkpoint` and the
  `recipe` the checkpoint was trained with (use the run dir's own copy — the checkpoint's geometry
  stamp must match it). For example:

  ```text
  Use AnomalyGen skill with
  name=phone_screen
  mode=generation_only
  dataset_dir=datasets/phone_screen
  recipe=results/anomalygen/phone_screen/anomalygen_texture_ft/exp_texture_ft_phone_screen.yaml
  checkpoint=results/anomalygen/phone_screen/anomalygen_texture_ft/checkpoints/model/iter_<best>.pt
  num_sdg=100
  ```

  `iter_<best>.pt` is the iteration named in the run dir's `checkpoints/best_checkpoint.txt` — never
  the last one, since runs routinely peak mid-training (see [4-fine-tuning.md](4-fine-tuning.md)).

### Optional parameters

All have working defaults, so add them only when you want to change something:

| Parameter           | Default | Effect                                                                                                                  |
| ------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `num_gpus`          | `1`     | GPUs for training, generation and each refinement round.                                                                |
| `num_search_run`    | `3`     | Refinement rounds in step 6 (`0` skips the search but still assembles `searched/`).                                     |
| `score`             | `nn`    | Metric the refinement optimizes: `nn`, `mnn`, `completeness`, `precision`, `boundary_iou`, `aq_nn` or `aq_rank`.        |
| `mask_backend`      | `amg`   | `amg` (agentic, default) or `amp` (classic placement by `spatial_dependency` routing).                                  |
| `base_seed`         | `1`     | Seeds the per-testcase noise in generation; keep it fixed within a pipeline run.                                        |
| `per_clean_pairing` | `off`   | AMP only: `on` gives every clean image exactly one record (single defect type).                                         |
| `legacy_fixed_size` | off     | Train with the legacy v1.1 fixed 512 px framing; only when explicitly requested ([4-fine-tuning.md](4-fine-tuning.md)). |

______________________________________________________________________

## Gates between steps

Before step 1 the agent runs `scripts/preflight_env_ckpt.sh` — the same smoke test
[`1-overview.md`](1-overview.md) ends with — and, for the default `mask_backend=amg`, checks that
the model registry resolves (see Prerequisites). Each step then ends with a named gate the agent
must pass before moving on:

| #   | Gate          | Step | Requires                                                                                             |
| --- | ------------- | ---- | ---------------------------------------------------------------------------------------------------- |
| 1   | `dataset`     | 1    | per-pair `clean_image/`, `anomaly_image/{defect}/`, `mask/{defect}/` (+ `defect_spec.jsonl` for AMP) |
| 2   | `amg` / `amp` | 2    | each set's `testcase.jsonl` with the requested row count                                             |
| 3   | `train`       | 3    | training finished **and** `checkpoints/best_checkpoint.txt` exists                                   |
| 4   | `generate`    | 4    | canonical result CSV **and** images + `guardrail_blocked.csv` rows = testcase rows                   |
| 5   | `eval`        | 5    | `<name>_kpi.json` with per-type + `Average` blocks                                                   |
| 6   | `refine`      | 6    | `num_search_run` complete round dirs (images **and** `kpi.json`)                                     |
| 7   | `label`       | 7    | `pseudo_labels/coco_annotations.json` + `classification/classes.txt`                                 |

______________________________________________________________________

## Where the outputs land

With `name=phone_screen`, the run writes into the same directory tree pages 3–6 use by hand; two
names differ from the by-hand commands: the KPI files are `<name>_kpi.json`
(`results/generation_phone_screen/phone_screen_kpi.json` and `searched/phone_screen_kpi.json`, where
page 6 writes `generation_phone_screen_kpi.json` and `searched/kpi.json`), and the pseudo-labels
always go under `searched/pseudo_labels/` (page 6 labels the `filtered/keep` tree in its example).

```text
datasets/validation_phone_screen/testcase.jsonl          # step 2, validation set (fine-tuning)
datasets/generation_phone_screen/testcase.jsonl          # step 2, generation set (SDG)
ag_config/exp_texture_ft_phone_screen.yaml               # step 3, the recipe the run trained with
results/anomalygen/phone_screen/anomalygen_texture_ft/   # step 3, run dir: checkpoints/, valid/, stdout.log
results/generation_phone_screen/                         # step 4, reconstructed_image/ + result CSV
results/generation_phone_screen/phone_screen_kpi.json    # step 5, the KPI
results/generation_phone_screen/rounds/, searched/       # step 6, search rounds + the refined bucket (searched/phone_screen_kpi.json)
results/generation_phone_screen/searched/pseudo_labels/  # step 7, the labeled dataset
results/*_phone_screen.log                               # every step's log
```

Each artifact is described on the page that produces it by hand:
[3-agentic-mask-generation.md](3-agentic-mask-generation.md) or
[3-auto-mask-placement.md](3-auto-mask-placement.md) for the mask sets,
[4-fine-tuning.md](4-fine-tuning.md) for the run dir, [5-generation.md](5-generation.md) for the
generation tree, and [6-evaluation-and-refinement.md](6-evaluation-and-refinement.md) for the KPI,
the refined bucket and the pseudo-labels.

______________________________________________________________________

## End-to-end timing

Measured wall-clock times for the pipeline steps on the three example datasets, on one **RTX PRO 6000
Blackwell (96 GB)** with Agentic Mask Generation driven by `gpt-6-astra`. The run parameters are in the
notes below.

**Fine-tuning** (15000 iterations):

| Dataset         | AMG validation masks | Per-iter time | Total time |
| --------------- | -------------------- | ------------- | ---------- |
| `pcb`           | 1 h 07 min           | 0.73 s        | 4 h 40 min |
| `magnetic_tile` | 0 h 26 min           | 0.75 s        | 3 h 57 min |
| `phone_screen`  | 0 h 28 min           | 0.90 s        | 5 h 15 min |

**Generation** (500 images), without and with quality refinement (3 rounds):

| Dataset         | AMG generation masks | s / image, `generate.py` only | Total time | s / image | NN score | Total time, refined | s / image, refined | NN score, refined |
| --------------- | -------------------- | ----------------------------- | ---------- | --------- | -------- | ------------------- | ------------------ | ----------------- |
| `pcb`           | 1 h 56 min           | 2.3                           | 2 h 15 min | 16.2      | 0.555    | 3 h 27 min          | 24.8               | 0.557             |
| `magnetic_tile` | 0 h 44 min           | 2.3                           | 1 h 03 min | 7.6       | 0.500    | 2 h 17 min          | 16.4               | 0.504             |
| `phone_screen`  | 1 h 04 min           | 5.8                           | 1 h 53 min | 13.5      | 0.592    | 4 h 51 min          | 34.9               | 0.597             |

- Fine-tuning total time includes Agentic Mask Generation (`gpt-6-astra`) for about 50 validation
  masks, 16 validation passes and checkpointing. Per-iter time is the trainer's median training step
  and excludes mask generation, model loading and validation.
- Generation total time includes Agentic Mask Generation (`gpt-6-astra`) for 500 masks; s / image is
  the total time divided by 500 images, and the `generate.py` only column is the generation script's
  own run time per image. The refined columns add three quality-refinement rounds, which is why the
  NN scores improve.
- `magnetic_tile` was generated with `--no-guardrail`: with the text guardrail on, every `MT_Crack`
  sample is blocked ([5-generation.md](5-generation.md#the-content-safety-guardrail)).
- Validation used 48, 50 and 50 masks and generation 500 masks each for `pcb`, `magnetic_tile` and
  `phone_screen`.
- Evaluation took 106 s, 91 s and 407 s and pseudo-labeling 42, 31 and 59 min for `pcb`,
  `magnetic_tile` and `phone_screen`; neither is included in the totals above.
- Timing differs across datasets with the defect resolution of the variable-resolution inputs;
  `phone_screen` has the highest defect resolution, hence the longest per-iter and `generate.py`
  per-image times.
- NN score is the average over the generated images.
- Peak GPU memory is ≈ 74 GiB during fine-tuning and ≈ 48 GiB during generation.

______________________________________________________________________

## Recap — the full pipeline

You now have the complete texture AnomalyGen workflow, by hand and from a single agent prompt:

| Step | Page                                                                                                         | What you produced                                                                                               |
| ---- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| 1    | [Overview & Environment Setup](1-overview.md)                                                                | working env + checkpoints + a passing preflight                                                                 |
| 2    | [Dataset Preparation](2-dataset-preparation.md)                                                              | a dataset dir with `anomaly_image/`, `mask/`, `clean_image/`                                                    |
| 3    | [Agentic Mask Generation](3-agentic-mask-generation.md) (or [Auto Mask Placement](3-auto-mask-placement.md)) | validation + generation `testcase.jsonl` with placed masks                                                      |
| 4    | [Fine-tuning](4-fine-tuning.md)                                                                              | a fine-tuned run under `results/anomalygen/…`                                                                   |
| 5    | [Generation](5-generation.md)                                                                                | a batch of synthetic anomalies at `results/generation_phone_screen/`                                            |
| 6    | [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)                                   | the KPI JSON, a refined `searched/` bucket, an optional `keep/` set, and `pseudo_labels/`                       |
| 7    | Agentic End-to-End Flow ←                                                                                    | all of the above, produced from a single agent prompt                                                           |
