<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 6 · Evaluation, Refinement & Pseudo-labeling

This is the sixth page of the CDF AnomalyGen texture tutorial series:

1. [Overview & Setup](1-overview.md)
2. [Dataset Preparation](2-dataset-preparation.md)
3. [Agentic Mask Generation](3-agentic-mask-generation.md) (or classic [Auto Mask Placement](3-auto-mask-placement.md))
4. [Fine-tuning](4-fine-tuning.md)
5. [Generation](5-generation.md)
6. **Evaluation, Refinement & Pseudo-labeling** ←
7. [Agentic End-to-End Flow](7-agentic-end-to-end-flow.md)

It assumes you finished [`5-generation.md`](5-generation.md) and have a generation output tree at
`results/generation_phone_screen/`.

In this stage you **score** that batch, optionally **refine** each sample's generation params by
searching for a better `(guidance, crop_ratio)`, optionally **filter** the result into a clean
`keep/` set, and finally **pseudo-label** it into a ready-to-train COCO / classification dataset
(with optional captions). All commands run from the repo root.

## Prerequisites

Work from the environment shell you set up in [`1-overview.md`](1-overview.md) — either the activated
native venv (`source .venv/bin/activate`) or an interactive container shell (`$DOCKER -it bash`). All
commands below run from the repo root and are identical either way.

## Evaluation

Three steps act on the **generation output tree** from [`5-generation.md`](5-generation.md), in the
order this page runs them. All three score against your **real defect references** with the same
metrics the training-time validation uses — the NN/MNN (+ FID) correspondence KPI *and* the
anomaly-quality axes.

1. **Evaluation** (`evaluate.py`) — produces *scores*. Per-type and macro-average KPIs, so you can
   judge overall batch quality. Its KPI is also the baseline refinement searches against.
2. **[Quality refinement](#quality-refinement-optional)** (`quality_refine.py`, **optional**) — *improves the
   batch*, by re-generating each sample with better params and keeping the best render.
3. **[Filtering](#filtering-optional)** (`filter.py`, **optional**) — *ranks and splits*, dropping the
   lowest-scoring fraction per type into `keep/` and `drop/` trees.

Refinement comes before filtering because it makes weak samples better rather than discarding them —
filter last, on whatever the best version of the batch turned out to be.

### Shared inputs

`evaluate.py` and `filter.py` take the same two roots plus a type selector.

- **`--gen_root`** — the generation `--output_dir` from [`5-generation.md`](5-generation.md) (e.g.
  `results/generation_phone_screen/`). It must contain `reconstructed_image/` and
  `original_mask/`, named `{texture}+{defect}_{idx}.png`.
- **`--real_root`** — the real training dataset root (e.g.
  `datasets/phone_screen`), which must contain, per type:

  ```text
  datasets/phone_screen/
  └── {texture}/
      ├── anomaly_image/{defect}/<stem>.png
      └── mask/{defect}/<stem>_mask.png
  ```

- **Type selection** — `--recipe results/anomalygen/phone_screen/anomalygen_texture_ft/exp_texture_ft_phone_screen.yaml`
  (the copy saved in the run dir) derives the types to score from the fine-tuning recipe's
  `anomaly_types`. Alternatively pass `--anomaly_types Phone+oil Phone+scratch …` explicitly.

### Evaluate

Score the generated batch and write per-type + macro-average KPIs to a JSON.

```shell
python anomalygen/scripts/texture/evaluate.py \
    --gen_root results/generation_phone_screen \
    --real_root datasets/phone_screen \
    --recipe results/anomalygen/phone_screen/anomalygen_texture_ft/exp_texture_ft_phone_screen.yaml \
    --output_file results/generation_phone_screen/generation_phone_screen_kpi.json
```

**Output** — a JSON at `--output_file` with one block per anomaly type plus an
`Average` block (the macro-average across types):

```json
{
  "Phone+oil":     { "nn_score": <float>, "mnn_score": <float>, "fid": <float>,
                     "completeness": <float>, "precision": <float>, "boundary_iou": <float>,
                     "aq_nn": <float> },
  "Phone+scratch": { "…": "…" },
  "Phone+stain":   { "…": "…" },
  "Average":       { "…": "…" }
}
```

Each per-type block also carries a `per_sample` array — one entry per generated image with `path`,
`nn_score`, `mnn_score` and the per-sample axis scores (elided above); the `Average` block does not.

> **Judge scores against your own runs, not against a fixed target.** `nn_score` / `mnn_score` are
> relative measures whose absolute scale depends on the dataset *and* on the scoring config.

**Reading the KPI:**

- **`nn_score` / `mnn_score`** — nearest-neighbor and mutual-nearest-neighbor cosine similarities
  (DINOv2 features) between each generated defect and its closest real references: **higher is
  better**. These measure how close the synthetic anomalies land to the real ones.
- **`fid`** — an optional set-level Fréchet distance over the defect crops. It needs at least 2 crops
  per type.
- **`completeness` / `precision` / `boundary_iou`** — the anomaly-quality axes: how much of the mask
  the defect actually fills, how much of the change stayed inside it, and how well its boundary
  agrees. **`aq_nn`** is the composite `completeness + nn_score`, the same one training validation
  tracks and `early_stop_metric` can select on.

## Quality refinement (optional)

Refinement raises a batch's score by **re-generating its samples better**, not by discarding any:
`quality_refine.py` searches for a per-sample `(guidance, crop_ratio)` that scores higher than the one
generation used, keeping each mask placement fixed. Every sample ends up at its best-scoring render,
so the batch is the same size it started at.

### The loop

The `run` subcommand drives the whole thing: `(draw → generate → evaluate) × --num_search_run`, then
`select`, then a closing `evaluate` on the assembled bucket.

- **`draw`** writes one round's `testcase.jsonl` by re-drawing `(guidance, crop_ratio)` for **every**
  sample. Round 1 draws uniformly; rounds 2+ use **per-sample Bayesian optimization** — an independent
  Gaussian process over that one sample's own `(guidance, crop_ratio) -> score` history,
  Thompson-sampled. Draws are seeded — under `run`, each round is seeded with its own round number —
  so a round is reproducible.
- **`select`** picks each sample's best-scoring render across the original bucket and the requested rounds,
  into a `searched/` bucket.

```shell
python anomalygen/scripts/texture/quality_refine.py run \
    --base_testcase datasets/generation_phone_screen/testcase.jsonl \
    --original results/generation_phone_screen \
    --original_kpi results/generation_phone_screen/generation_phone_screen_kpi.json \
    --rounds_dir results/generation_phone_screen/rounds \
    --output results/generation_phone_screen/searched \
    --final_kpi results/generation_phone_screen/searched/kpi.json \
    --checkpoint results/anomalygen/phone_screen/anomalygen_texture_ft/checkpoints/model/iter_000015000.pt \
    --recipe results/anomalygen/phone_screen/anomalygen_texture_ft/exp_texture_ft_phone_screen.yaml \
    --real_root datasets/phone_screen \
    --score nn \
    --num_search_run 3 --num_gpus 1
```

- **`--base_testcase`** — the generation `testcase.jsonl` from page 3 (AMG or AMP) that produced
  `--original` (the mask placements the search holds fixed).
- **`--original` / `--original_kpi`** — the [generation](5-generation.md) output tree and its
  [evaluation](#evaluate) KPI JSON: round 0 of the search, and the baseline every round is measured
  against.
- **`--num_search_run`** (default `3`) — how many `draw → generate → evaluate` rounds to run. Each
  round costs one full generation pass over the batch, so this is the main cost dial (measured
  refinement times for the example datasets:
  [`7-agentic-end-to-end-flow.md`](7-agentic-end-to-end-flow.md#end-to-end-timing)).
- **`--score`** (default `nn`) — the metric the search optimizes and selects on: `nn`, `mnn`,
  `completeness`, `precision`, `boundary_iou`, `aq_nn`, or `aq_rank`. `aq_rank` is the optional
  cumulative composite of NN and the three geometry axes described below. The geometry-based scores
  need the SAM2 checkpoint; without it `evaluate.py` falls back to NN/MNN with a warning and `run`
  stops before round 1 for lack of a usable score.
- **`--guidance_range`** (default `1.0 10.0`) and **`--crop_ratio_range`** — the box the search draws
  from. `--crop_ratio_range` defaults to the band `--recipe` trained under (falling back to
  `1.5 2.5`), because generation clamps a testcase's `crop_ratio` into that band; drawing wider just
  repeats its edge. An explicit range is therefore narrowed to that band (with a warning), and one
  entirely outside it is refused. Its `LOW` must be at least `1.5`, the floor a recipe's band has.
- **`--num_gpus`** — GPUs for each round's generation pass.
- **`--output_root`** (default `results`) — `IMAGINAIRE_OUTPUT_ROOT` for each round's `generate.py`.
  It is passed explicitly because the framework otherwise falls back to `/tmp/imaginaire4-output`
  without complaining.
- **`--dry_run`** prints each stage's command without running it.

> **⚠ Rounds must differ from the original only in `(guidance, crop_ratio)`.** Each round runs
> `generate.py` with its defaults (`--base_seed 1`, guardrail on); if you generated `--original` with
> another seed, `--no-guardrail` or a `--base_checkpoint`, pass the same to `run`, which forwards
> them to every round. Otherwise rounds win or lose on noise as well as on params.

`run` **gates each round** — one image per testcase row *and* its `kpi.json` — and exits non-zero
rather than assembling a stale `searched/` from an unfinished round. The scoring knobs it is given
are forwarded to every round's `evaluate`, so each round is scored exactly the way the batch it is
selected against was. `run` has no resume, so repair a failed round in place rather than re-running
`run` (which would redraw every round from 1); see the
[quality-refinement reference](../../../skills/cdf-anomalygen/references/quality-refinement.md).

It also requires the generation CSV to name exactly the reconstructed images in that round. This
check applies to every score mode because both BO history and final selection join KPI rows through
that CSV. `run` passes the current round explicitly so resumed or manually retained later-round
directories cannot leak into the active round's history. When invoking `draw` directly with a
non-standard output path and an existing `--rounds_dir`, pass `--current_round <r>`.
When invoking `select` directly, pass `--max_round <N>` to exclude stale later rounds; without it,
standalone selection intentionally considers every canonical positive-numbered round.

### NN versus cumulative AQ-Rank

With the default `--score nn`, every observation is the sample's absolute NN value. With
`--score aq_rank`, the search does **not** trust the `aq_rank_score` already stored inside each
individual bucket: that value was ranked against that bucket alone. Before each draw it instead:

1. pools Original with all earlier rounds;
2. groups the candidates by anomaly type;
3. recomputes AQ-Rank from NN, completeness, precision, and boundary IoU; and
4. uses those current pooled ranks as the per-sample Bayesian-optimization history.

Final selection recomputes the same ranking over Original plus the rounds generated by the current
run. This mode ranks only candidates with finite values for all four raw inputs. An invalid candidate
is logged and excluded, so it cannot be rewarded when an axis is oriented negatively, while the rest
of the bucket remains usable. A bucket with no eligible candidate or a CSV/KPI mismatch stops the
run. If one sample has no eligible render in any otherwise-valid bucket, selection logs the fallback
and keeps its Original render without treating it as an AQ-Rank win. A per-type pool below five
eligible candidates is explicitly logged and ranks that type by NN only.

AQ-Rank learns each geometry axis's sign from differences **across the pooled samples**, then uses
that sign to choose among Original and round variants **within one sample**. The two relationships can
point in opposite directions, so this compass is an auditable heuristic rather than a guarantee of
within-sample improvement. Each re-ranking logs the pool size plus completeness, precision, and
boundary-IoU AUROC and sign (`+1`, `-1`, or `disabled`). Keep these logs when interpreting a run; a
within-sample orientation would be a new algorithm requiring separate validation.
The current `|AUROC - 0.5| >= 0.05` gate is permissive: it rejects constant or nearly signalless axes,
but it is not a significance test or proof that an enabled sign improves within-sample choices.

NN remains the conservative default. Use AQ-Rank when you explicitly want correspondence and mask
geometry to participate in the search. In a seven-dataset pilot the two objectives selected later
rounds at similar rates, and a one-reviewer blind comparison detected no statistically significant
perceptual difference between their final outputs. Treat both as usable search objectives, not as
proven perceptual equivalents.

### Output

```text
results/generation_phone_screen/
├── rounds/
│   ├── round_1/    # that round's testcase.jsonl + generation tree + kpi.json
│   ├── round_2/
│   └── round_3/
└── searched/                              # the assembled best-of bucket
    ├── reconstructed_image/               # each sample's winning render
    ├── original_image/
    ├── original_mask/
    ├── annotated_image/
    ├── texture_ft_generation_result.csv   # winning params + metrics + selected_by + source
    ├── search_summary.csv                 # per sample: source, score, original_score, improved
    ├── per_sample.csv                     # per-sample nn + mnn table
    └── kpi.json                           # --final_kpi: the refined batch's KPI
```

**`select` always runs**, so `searched/` always exists for pseudo-labeling — with
`--num_search_run 0` it simply clones `--original`.

**Reading the result:** under `--score nn`, compare the `Average.nn_score` in `searched/kpi.json` with
the `generation_phone_screen_kpi.json` from [Evaluate](#evaluate). A `source` column that is all
`original` means no round beat the baseline for any sample — a valid outcome, not a failure.

Under `--score aq_rank`, do not compare AQ-Rank means from separate KPI files: ranks are relative to
the pool in which they were computed. Read `source` and `search_summary.csv`, compare paired raw
metrics, and use human review or downstream performance to judge whether the trade-off is useful.

## Filtering (optional)

Run it when you want a curated subset: it ranks each anomaly type by a per-sample score and splits the
batch into keep/drop trees, dropping the lowest-scoring `--drop_ratio` fraction **per type**.

Point `--gen_root` at the **refined** bucket if you ran refinement, so you curate the best version of
each sample rather than the first one; at `results/generation_phone_screen` if you skipped it.

```shell
python anomalygen/scripts/texture/filter.py \
    --gen_root results/generation_phone_screen/searched \
    --real_root datasets/phone_screen \
    --output_dir results/generation_phone_screen/filtered \
    --recipe results/anomalygen/phone_screen/anomalygen_texture_ft/exp_texture_ft_phone_screen.yaml \
    --drop_ratio 0.3            # keep the top 70% per type; --score defaults to nn
```

- **`--score`** (default `nn`) — the per-sample score to rank on:
  - `nn` / `mnn` — DINOv2 correspondence only. Fast, no extra models; `nn` is the primary KPI and
    needs no per-dataset tuning, hence the default.
  - `completeness` / `precision` / `boundary_iou` / `aq_nn` — one geometry axis on its own, or the
    `completeness + nn_score` composite.
  - `aq_rank` — `nn` and each geometry axis rank-normalised **within the anomaly type**, each axis
    signed `+`/`−` by whether it agrees with `nn`'s own strongest/weakest samples, then summed:
    `aq_rank = rank(nn) + Σ ±rank(axis)`. It jointly considers correspondence and geometry, but signs
    itself from `nn`, so a weak NN anchor can orient an axis incorrectly. It is therefore an
    **opt-in** alternative rather than a human-quality guarantee.
  - The geometry axes run SAM2, so any score using them needs `--real_root` and `original_image/`.
- **`--drop_ratio`** — fraction in `[0, 1]` to drop per type; `0.3` keeps the top 70%. Samples that
  could not be scored are dropped first.

For a fast correspondence-only pass (no SAM), rank on `nn`:

```shell
python anomalygen/scripts/texture/filter.py \
    --gen_root results/generation_phone_screen/searched \
    --real_root datasets/phone_screen \
    --output_dir results/generation_phone_screen/filtered \
    --recipe results/anomalygen/phone_screen/anomalygen_texture_ft/exp_texture_ft_phone_screen.yaml \
    --score nn --drop_ratio 0.3
```

**Output** — under `--output_dir`, a `keep/` and a `drop/` tree, each carrying
the reconstructed image, its mask, and the original image for every routed
sample (matched by basename):

```text
results/generation_phone_screen/filtered/
├── keep/
│   ├── reconstructed_image/   Phone+oil_00000.png …
│   ├── original_mask/         Phone+oil_00000.png …
│   ├── original_image/        Phone+oil_00000.png …
│   └── texture_ft_generation_result.csv   (when the input CSV exists)
├── drop/
│   ├── reconstructed_image/
│   ├── original_mask/
│   ├── original_image/
│   └── texture_ft_generation_result.csv   (when the input CSV exists)
└── texture_ft_generation_result_filtered.csv   (per-sample scores)
```

`texture_ft_generation_result_filtered.csv` lists every sample's `nn_score` and
`mnn_score`; when you filter on an axis or `aq_rank`, it also carries the
`completeness_score` / `precision_score` / `boundary_iou_score` columns and the composite
`aq_rank_score`. When the `texture_ft_generation_result.csv` from generation is present in
`--gen_root`, it is also split into `keep/` and `drop/` copies by output
filename.

## Putting it together

The three stages do different jobs with the same score, and only the first is mandatory:

- **Evaluation *measures*.** It answers "did the fine-tune produce defects that look like the real
  ones, on average and per type?" — and produces the KPI the other two are judged against.
- **Refinement *improves*.** It makes weak samples better by re-generating them with searched params,
  so nothing is discarded. Cost is `--num_search_run` extra generation passes, and the payoff is the
  `Average.nn_score` delta between `searched/kpi.json` and the Evaluate KPI. Skip it if a full extra
  pass over the batch is too expensive, or if that delta comes back near zero on your data.
- **Filtering *discards*.** It turns the same score into a curated set — the `keep/` tree holds the
  highest-scoring synthetic anomalies (image + mask + original), ready to drop into a downstream
  detector/segmentation training set as augmentation. Tune `--drop_ratio` to trade set size for
  average quality, and pick `--score` by how much scrutiny you want: `nn`/`mnn` for a fast
  correspondence-only pass, a single geometry axis to curate on one property, or `aq_rank` for the
  composite of both.

Run them in that order. Refining before filtering means a sample that would have been dropped gets a
chance to be fixed first; filtering first would throw away renders that refinement could have
rescued, and would spend the search's GPU time on a smaller batch for no quality gain.

Whichever you stop at is what you pseudo-label next — `searched/` after refinement,
`filtered/keep/` after filtering, or the raw generation tree if you skipped both.

## Pseudo-labeling

Generation gives you defect **images**; pseudo-labeling turns a generation batch into a **labeled
dataset** ready for downstream training. `pseudo_label.py` consumes the same generation output tree
and emits:

- a **COCO instance dataset** (`coco_annotations.json`) — each defect mask is split into per-instance
  masks and encoded as an RLE segmentation + bounding box, keyed by anomaly-type category;
- a **classification layout** (`classification/`) — the clean originals and every generated image
  sorted into per-class folders with a `classes.txt`;
- **visualizations** (`visualization/`) — each generated image with its instance masks, boxes, and
  class labels overlaid; and
- optional **natural-language captions** (`captions/`, `captions_with_meta/`) — one anomaly
  description per image from the Cosmos3-Nano reasoner.

### Inputs

- **`--gen_root`** — any tree containing `reconstructed_image/`, `original_image/`, `original_mask/`
  and the `texture_ft_generation_result.csv` manifest (used to look up each image's anomaly type).
  Three sensible choices:
  - **`results/generation_phone_screen/searched`** — the refined bucket. Use this if you ran
    [quality refinement](#quality-refinement-optional); it is every sample at its best-scoring render.
  - **`results/generation_phone_screen`** — the raw generation batch, if you skipped refinement.
  - **`results/generation_phone_screen/filtered/keep`** — a filtered `keep/` tree, to label only the
    samples that survived filtering.

### Run pseudo-labeling

```shell
python anomalygen/scripts/texture/pseudo_label.py \
    --gen_root results/generation_phone_screen/filtered/keep \
    --output_dir results/generation_phone_screen/filtered/pseudo_labels
    # add --no_caption to skip the VLM step (fast, no GPU model load)
```

With `--no_caption` you get the COCO dataset, classification layout and visualizations, but no
`captions/` or `captions_with_meta/`.

### Output folder structure

Everything lands under `--output_dir`:

```text
results/generation_phone_screen/filtered/pseudo_labels/
├── coco_annotations.json          # COCO instance annotations (per-instance mask → RLE + bbox)
├── images/                        # the generated defect images
├── masks/                         # the input defect masks
├── visualization/                 # mask + bbox + class-label overlays
├── classification/                # per-class image folders + classes.txt
│   ├── classes.txt                #   "original" + one line per anomaly type
│   ├── original/                  #   the clean input images
│   ├── Phone+oil/                 #   generated images, one folder per anomaly type
│   ├── Phone+scratch/
│   └── Phone+stain/
├── captions/                      # one .txt caption per image (omitted with --no_caption)
└── captions_with_meta/            # same caption prefixed with a timestamp + meta block
```

- **`coco_annotations.json`** — standard COCO `{images, annotations, categories}`. Each mask is split
  into up to `--max_instances` per-instance masks (connected components, matching generation's
  `--max_instances`), and every instance becomes one annotation with an RLE `segmentation`, `bbox`,
  `area`, and `category_id`. Category ids are assigned to the sorted anomaly-type names.
- **`classification/`** — the clean originals under `original/` and each generated image under its
  `{texture}+{defect}/` folder, with `classes.txt` listing `original` followed by the anomaly types —
  a ready-to-use image-classification split.
- **`visualization/`** — the generated image with each instance mask tinted and its box + class label
  drawn on, for eyeballing the labels.
- **`captions/` and `captions_with_meta/`** — the Cosmos3-Nano reasoner is shown the (mask, clean,
  generated) triple and asked to describe the anomaly; `captions_with_meta` prepends a creation
  timestamp and the per-image meta (image/anomaly type, boxes).

### Useful flags

- **`--no_caption`** — skip captioning (no VLM load).
- **`--max_instances`** — max per-instance masks to split each mask into (default 5, matching
  `generate.py`).
- **`--csv_path`** — override the manifest CSV (defaults to `{gen_root}/texture_ft_generation_result.csv`).
- **`--captioner_temperature` / `--captioner_max_tokens` / `--captioner_seed` / `--captioner_prompt_path`**
  — captioner generation knobs; the temperature defaults to `0`, which decodes greedily. Pass
  `>0` to sample at that temperature instead.

## Next step

Continue to [7 · Agentic End-to-End Flow](7-agentic-end-to-end-flow.md), which runs all of the above
from a single agent prompt.
