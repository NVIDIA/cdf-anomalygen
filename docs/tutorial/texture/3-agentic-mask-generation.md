<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 3 · Agentic Mask Generation

This is the third page of the CDF AnomalyGen texture tutorial series:

1. [Overview & Setup](1-overview.md)
2. [Dataset Preparation](2-dataset-preparation.md)
3. **Agentic Mask Generation** ← (or the classic [Auto Mask Placement](3-auto-mask-placement.md))
4. [Fine-tuning](4-fine-tuning.md)
5. [Generation](5-generation.md)
6. [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)
7. [Agentic End-to-End Flow](7-agentic-end-to-end-flow.md)

It assumes you have finished [`2-dataset-preparation.md`](2-dataset-preparation.md): a `dataset_dir`
on disk in the [directory contract](2-dataset-preparation.md#the-directory-contract).

By the end you will have a **`testcase.jsonl`** — one row per placed defect mask, paired with its
clean image — the same deliverable the classic AMP stage produces, consumed by
[fine-tuning](4-fine-tuning.md) and [generation](5-generation.md).

All commands are run from the **repo root**.

______________________________________________________________________

## Prerequisites

Work from the environment shell you set up in [`1-overview.md`](1-overview.md). The stage needs two
things beyond it:

- **Checkpoints** — SAM2 (shared with AMP's `text` route) and DINOv2 (exemplar embeddings),
  provisioned by `scripts/download_checkpoints.sh`.
- **A VLM endpoint** — the agentic loop is driven by a vision-language model. Every model name a
  run uses (`--gen-model` / `--judge-model`) resolves through exactly ONE place: **`endpoints.yaml`**,
  the model registry. Nothing is built into the code and there is no environment variable behind
  it — a name with no entry refuses the run before anything is spent.

Set it up once:

```shell
mkdir -p ~/.anomalygen
cp scripts/agentic_mask_generation_serving/endpoints.example.yaml ~/.anomalygen/endpoints.yaml   # outside the repo — your key never enters version control or a docker build context
```

The template carries both of this page's recipes; every run picks `~/.anomalygen/endpoints.yaml` up
automatically, from any directory (to use another file, pass `--model-registry <path>`). The run's first log lines
confirm where each role landed — `model routing: gen azure/openai/gpt-6-astra -> https://...
(key: set (redacted))` — the key VALUE is never logged.

**Cloud model** (this page's recipe: `azure/openai/gpt-6-astra`): fill in its entry —

```yaml
- model: azure/openai/gpt-6-astra
  base_url: <your gateway's OpenAI-SDK base URL, version segment included, e.g. .../v1>
  key: <your token>
```

The same model straight from OpenAI, with no gateway in between, is one more registry entry and
nothing else — `model: gpt-6-astra` (OpenAI's own id; the gateway's `azure/openai/` prefix names
the provider behind it), `base_url: https://api.openai.com/v1`, `key:` your OpenAI API key — and
you pass `gpt-6-astra` to `--gen-model` / `--judge-model`. The base URL is the same for every
OpenAI model; the model is chosen in the request body. The transport already sends what OpenAI
requires of a reasoning model there — the output cap as `max_completion_tokens` and no sampling
parameters — so no flag changes between the gateway and the official endpoint (the example file
carries the entry, commented out).

**Self-hosted model**: the template's `nvidia/google/gemma-4-31b-it` entry already matches the
shipped recipe (`http://localhost:8902/v1`), so serving is the only step:

```shell
# weights arrive with the other checkpoints: scripts/download_checkpoints.sh -> checkpoints/hf
mkdir -p results   # the script runs vLLM in the foreground, so background it with a log:
CUDA_VISIBLE_DEVICES=<two free GPUs> \
    setsid scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh > results/gemma_server.log 2>&1 &   # serves port 8902
scripts/agentic_mask_generation_serving/serverctl.sh wait      # 0 up · 1 no process (read the log) · 2 still loading
scripts/agentic_mask_generation_serving/serverctl.sh health    # models list AND a real completion
```

Moving or locking that server is the same entry: serve on another port/host and point `base_url`
at it, or start the server with `API_KEY=<secret> scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh` and put
the same value in `key:`. A model with no `key:` sends `EMPTY`, which any unlocked vLLM/SGLang
server accepts. Adding a different self-hosted model is two things: a serving script beside the
gemma recipe, and a registry entry keyed by its served name.

Serving notes:

- **Exit `2` is not failure.** A cold start is 10–15 minutes on one GPU; on a `2`, call `wait`
  again. Restarting on a 2 stacks a second vLLM on the same GPU and OOMs. Only exit `1`
  (no process) means it is not coming — read the server log. (Same when stopping: only `1` means
  the GPU is actually free.)
- **Stop the server before fine-tuning.** Serving holds most of the GPU that training needs: once
  both mask sets are built, run `serverctl.sh stop` and confirm `serverctl.sh health` exits `1`.
- **The knobs you'll touch**: `CUDA_VISIBLE_DEVICES` (required), `TP` (= number of devices named),
  `MAXLEN` (serving window, default 32768), `PORT`. Leave the recipe's other vllm arguments alone.
- **Throughput comes from running several classes at once, not from raising concurrency** — use
  `--max-concurrent-calls 6-8` per run; higher exhausts the KV cache and kills the engine mid-run.

______________________________________________________________________

## What AMG does

Where AMP *places* an existing training submask with hand-declared routing (`spatial_dependency` in
`defect_spec.jsonl`), **Agentic Mask Generation authors the mask**: a VLM agent proposes candidate
masks for each clean image, an independent judge scores them against the dataset's own defect
examples, and a refine loop edits the survivors until they clear an acceptance bar. Its only priors
are the dataset itself — the `(defect image, mask)` exemplar pairs, and a **defect card** distilled
from them at first run (cached) — so a new defect type or product needs no routing configuration and
no code change.

Choosing between the two:

|                  | **AMG (this page — default)**                           | [AMP](3-auto-mask-placement.md) (classic)                                                |
| ---------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Masks are        | authored per clean image, judged before delivery        | training submasks, placed                                                                |
| Per-defect setup | none (card is distilled automatically)                  | `spatial_dependency` routing per defect                                                  |
| Cost             | VLM calls per mask (cloud or self-hosted GPU)           | cheap; VLM only on the `text` route                                                      |
| Best for         | generation batches; new defects/products with no tuning | fast deterministic batches when routing is already declared                              |

Both size a run the same way: `--num-sdg` with `--mode {inference,validation}` (or exact
`--per-defect-counts`), split across the types by the shared allocation rule.

______________________________________________________________________

## Worked example: phone_screen

Build a 15-mask generation batch for the same `phone_screen` dataset the
[AMP page](3-auto-mask-placement.md) uses — `--num-sdg` sizes the whole run and is split across the
declared types exactly the way AMP splits its `--num_sdg` (here: 5 per defect):

```shell
python -m anomalygen.scripts.agentic_mask_generation.generate \
    --dataset-path datasets/phone_screen \
    --anomaly-types Phone+oil Phone+scratch Phone+stain \
    --gen-model azure/openai/gpt-6-astra \
    --judge-model azure/openai/gpt-6-astra \
    --num-sdg 15 \
    --visual-prompt auto --defect-library auto \
    --out-dir datasets/generation_phone_screen
```

`--gen-model` and `--judge-model` are **required** — which model draws and which model judges is
the run's most consequential choice, so it is always stated, never inherited from a hidden default.
This page's two recipes: the cloud model above, or the self-hosted Gemma below.

The first run distills the defect card for each type (a handful of captioning calls, cached under
`--cache-dir`); later runs reuse it. The run is **resumable**: re-running the same command into the
same `--out-dir` produces only the shortfall, and `--overwrite` starts over instead.

```text
datasets/generation_phone_screen/
├── <texture>/mask/<defect>/<clean_stem>_<n>_<j>_mask.png       # the authored masks
├── <texture>/overlay/<defect>/<clean_stem>_<n>_<j>_overlay.png # each mask over its clean, for eyeballing
├── <texture>/_overflow/<defect>/                               # surplus judged masks (+ testcase_overflow.jsonl) — not in testcase.jsonl
├── testcase.jsonl                  # one row per accepted mask ↔ clean image — exactly --num-sdg rows
├── run.json                        # the run's full control plane (settings, knobs)
├── run.log                         # the run log, including every degraded[...] line
└── _runlog/                        # per-run acceptance records + index.jsonl
```

Each `testcase.jsonl` row carries the image, mask, anomaly type, output count, blend flag and instance
cap; generation fills `guidance` (3.0), `num_steps` (35) and `crop_ratio` (2.0) from its defaults unless
a row sets them.

Fine-tuning also needs a **validation** set: run the same command with `--mode validation` and
`--out-dir datasets/validation_phone_screen` — the recipe's `testcase_jsonl` on page 4. Validation
splits `--num-sdg` in proportion to each type's real mask count, and a type whose share would round
to 0 makes the run refuse (a `validation coverage broken` error naming the `--num-sdg` to use).

Count rows, not files: `testcase.jsonl` has exactly `--num-sdg` rows, while the surplus masks the fill
loop trims land in `_overflow/`. Every delivered mask has been judged against the bar; anything the
engine had to degrade or withhold is a named `degraded[...]` line in `run.log` rather than a silent
gap.

To run self-hosted instead (server setup: Prerequisites), point both required model flags at the
served name and turn the in-loop beam on. A smaller
local model needs several distinct-strategy options per refine turn, selected pairwise, where a
strong cloud model's single refine turn is enough:

```shell
python -m anomalygen.scripts.agentic_mask_generation.generate \
    ... \
    --gen-model nvidia/google/gemma-4-31b-it \
    --judge-model nvidia/google/gemma-4-31b-it \
    --beam-multi 4 --beam-pairwise \
    --exemplar-captions --exemplar-visual-prompt --coverage-note \
    --request-timeout 480
```

The three `--exemplar-*` / `--coverage-note` switches open the information channels a smaller
model needs (per-example captions, exemplar structure views, the measured coverage note); each
defaults OFF and costs wall-clock, so leave them off for cloud models. What each one does and
why: the [AMG skill reference](../../../skills/anomalygen/references/agentic-mask-generation.md).

______________________________________________________________________

## A custom dataset

The stage reads the same [directory contract](2-dataset-preparation.md#the-directory-contract) as
the rest of the pipeline — point `--dataset-path` at the dataset root and list your
`<texture>+<defect>` types. No `defect_spec.jsonl` routing is needed for this stage: the defect's
appearance, extent and placement rules are distilled from its own exemplar pairs. For a fine-tuning
**validation** set with per-type counts proportional to the training data, pass
`--mode validation` with `--num-sdg` — the same allocation rule as
[AMP's](3-auto-mask-placement.md), shared by both backends.

______________________________________________________________________

______________________________________________________________________

## Flag reference

`--help` carries the full surface; this is the subset a user reaches for, grouped by
intent. Anything not listed here is an experimenter's dial — changing it means owning the quality
consequences.

### Required

| flag                            | default | takes                      | what it does                                                                                                                   |
| ------------------------------- | ------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--dataset-path`                | —       | path                       | AnomalyGen-format dataset root                                                                                                 |
| `--anomaly-types`               | —       | `<texture>+<defect>` ...   | which types to generate                                                                                                        |
| `--gen-model` / `--judge-model` | —       | model id (a registry name) | who draws / who judges — always stated, never defaulted; they may differ (a weak drawer with a strong judge is a valid recipe) |

### Everyday options

| flag                     | default                                 | takes                             | what it does                                                                                                                                                                                                                                                                                   |
| ------------------------ | --------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--num-sdg`              | off (one pass over every clean image)   | int                               | judge-ACCEPTED masks for the whole run, split across types per `--mode`; required with `--mode validation`                                                                                                                                                                                     |
| `--out-dir`              | `results/amg`                           | path                              | deliverable: masks + `testcase.jsonl` + `run.log` (+ `_runlog/`, `_overflow/`; a trace only where `--trace-dir` points)                                                                                                                                                                        |
| `--mode`                 | `inference`                             | `inference` \| `validation`       | how `--num-sdg` splits: uniform vs proportional to each type's real-mask count                                                                                                                                                                                                                 |
| `--cache-dir`            | `results/_amg_cache/<name>-<path hash>` | path                              | reusable precomputes (vp maps, embeddings, defect cards) — share it across runs, pay once                                                                                                                                                                                                      |
| `--per-defect-counts`    | off                                     | JSON, e.g. `'{"SOP+BRIDGE": 12}'` | exact per-type quotas; its sum outranks `--num-sdg`                                                                                                                                                                                                                                            |
| `--overwrite`            | off                                     | switch                            | **without it, pointing at a populated `--out-dir` RESUMES that run** (same settings enforced, only the shortfall is generated); with it, the directory is deleted and the run starts over                                                                                                      |
| `--max-concurrent-calls` | 32                                      | int                               | global in-flight ceiling; lower it for small-window endpoints (self-hosted recipes use 6-8)                                                                                                                                                                                                    |
| `--request-timeout`      | 180                                     | seconds                           | per-call HTTP timeout; self-hosted serving under load wants 480                                                                                                                                                                                                                                |
| `--model-registry`       | `~/.anomalygen/endpoints.yaml`          | YAML path                         | the ONE place model names become endpoints (`base_url` + `key` per model; nothing is built in) — see `scripts/agentic_mask_generation_serving/endpoints.example.yaml` and Prerequisites; a model in no entry refuses the run. The run's first log lines print where each role's model resolved |

### Quality levers (the local-model recipe)

The five switches default OFF — a strong cloud model doesn't need them, a smaller model does; see
the self-hosted section above and the [AMG skill reference](../../../skills/anomalygen/references/agentic-mask-generation.md) for what each one is. The accept bar is
a number and always in force.

| flag                       | default | takes                 | what it does / cost when on                                                                                                            |
| -------------------------- | ------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--beam-multi`             | off     | int (local recipe: 4) | each refine turn returns N distinct-strategy fix options in one call, best kept                                                        |
| `--beam-pairwise`          | off     | switch                | the selector compares those options A-vs-B instead of scoring each alone                                                               |
| `--exemplar-captions`      | off     | switch                | ~+4-14% wall                                                                                                                           |
| `--exemplar-visual-prompt` | off     | switch                | ~+6-9% wall, one extra image per exemplar                                                                                              |
| `--coverage-note`          | off     | switch                | ~+11-12% wall                                                                                                                          |
| `--accept-threshold`       | 8       | int                   | the accept bar, always on (`semantic >= t`, min `>= t-1`, mean `>= t`); raising it buys little on a judge that inflates its own scores |

### Throughput levers (trade measured quality for speed)

| flag                  | default | effect                                              |
| --------------------- | ------- | --------------------------------------------------- |
| `--beam-prune`        | off     | drop hopeless candidates early                      |
| `--accept-continue`   | off     | stop refining a candidate once it clears the bar    |
| `--judge-cache-order` | off     | board-constant blocks first, for prompt-cache reuse |

Together they cut calls per delivered mask to roughly a third on the Opus 5 measurement (see the
skill reference) — and they are
off by default because that saving is paid in measured quality.

### Situational overrides

| flag               | default      | takes                                                                   | when to reach for it                                                                                                                                                                                            |
| ------------------ | ------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--shape`          | `auto`       | `auto` \| `flexible` \| `fixed`                                         | `auto` = per-type router, with `fill_host=whole` cards forcing flexible; force `flexible` when you suspect the router mis-filed a connection-like type as rigid; `fixed` outranks everything incl. the card ban |
| `--coverage`       | `roundrobin` | `none` \| `localize` \| `propose` \| `both` \| `roundrobin` \| `choose` | how many-masks-per-clean avoid piling on one site: `roundrobin` assigns enumerated regions deterministically; the others tell a turn what is taken, or show the agent the menu                                  |
| `--visual-prompt`  | `auto`       | `sam2` \| `solderlight` \| `normallight` \| `none` \| `auto`            | the structure cue drawn for the model. `auto` reads the type's distilled card and uses none where the card does not say; force a concrete cue to overrule the card                                              |
| `--defect-library` | `auto`       | `none` \| `auto`                                                        | `auto` distills a per-type reference card from the dataset's own examples (one-time VLM cost, cached under `--cache-dir`); `none` runs card-less — and starves `--visual-prompt auto`, which then draws no cue  |
| `--num-examples`   | 3            | int                                                                     | in-context examples per run                                                                                                                                                                                     |

### Diagnostics

| flag           | default | what it does                                                                                                                                                                              |
| -------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--trace-dir`  | off     | keep the full VLM-call audit trail under this directory. The trace answers "what did the model see", but it stores every prompt image — about 1 MB per call, tens of GB for a large batch |
| `--skip-smoke` | off     | skip the model capability probe (images + JSON) that otherwise gates the run                                                                                                              |

## Troubleshooting

- **Runs stall at 0% and `run.log` shows `degraded[api_abort:bad_request]` with a vLLM 400 about the
  model length** — your serving window is too
  small for prompt + the engine's fixed 8000-token agent output budget; serve a larger window
  (`MAXLEN`, default 32768 — the smallest that fits the measured prompts).
- **Self-hosted runs die after repeated transport timeouts** — long generations under load can
  exceed the default 180 s per-call timeout; pass `--request-timeout 480`.
- **Self-hosting but masks are slow and the GPU is idle** — check `--judge-model` names the local
  model too: the two model flags are independent, and a judge left on a cloud name keeps the run's
  most numerous calls off your GPU. The run's opening `model routing:` lines say exactly where each
  role went.

______________________________________________________________________

## Next step

Continue to [4 · Fine-tuning](4-fine-tuning.md).
