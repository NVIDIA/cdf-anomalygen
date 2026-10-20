<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2 · Dataset Preparation

This is the second page of the CDF AnomalyGen texture tutorial series:

1. [Overview & Setup](1-overview.md)
2. **Dataset Preparation** ←
3. [Agentic Mask Generation](3-agentic-mask-generation.md) (or classic [Auto Mask Placement](3-auto-mask-placement.md))
4. [Fine-tuning](4-fine-tuning.md)
5. [Generation](5-generation.md)
6. [Evaluation, Refinement & Pseudo-labeling](6-evaluation-and-refinement.md)
7. [Agentic End-to-End Flow](7-agentic-end-to-end-flow.md)

It assumes you have finished [`1-overview.md`](1-overview.md): the venv is built, Hugging Face
auth is configured, and the base checkpoints are downloaded.

By the end you will have a `dataset_dir` on disk laid out exactly the way the trainer expects. That
directory is the input to page 3 ([Agentic Mask Generation](3-agentic-mask-generation.md), or classic
[Auto Mask Placement](3-auto-mask-placement.md)), and what you later pass to a fine-tuning recipe as
`dataset_path` ([`4-fine-tuning.md`](./4-fine-tuning.md)).

All commands are run from the **repo root**.

______________________________________________________________________

## Prerequisites

Work from the environment shell you set up in [`1-overview.md`](1-overview.md) — either the activated
native venv (`source .venv/bin/activate`) or an interactive container shell (`$DOCKER -it bash`). All
commands below run from the repo root and are identical either way. Most of the prepare scripts pull
from Hugging Face, so have auth configured (`export HF_TOKEN=<your-token>` or `hf auth login`) — see
the per-dataset note below for which ones need it.

______________________________________________________________________

## Ready-made datasets

Three `prepare_*` CLIs turn an upstream source into the layout below. Each takes `--output_dir` (the
convention is `datasets/<name>`) and `--dryrun` (preview without writing).

> **License reminder:** each script prints a notice that **you are responsible for confirming the
> dataset's license is fit for your intended use.** Read it before you build on any upstream data.

| Dataset         | `{texture}` → defects                                                 | Routing | Source                  |
| --------------- | --------------------------------------------------------------------- | ------- | ----------------------- |
| `pcb`           | `IC` → bridge · `passive_component` → excess_solder, missing          | `cad`   | HF (all assets)         |
| `magnetic_tile` | `metal_surface` → MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Uneven | `free`  | public GitHub           |
| `phone_screen`  | `Phone` → oil, scratch, stain                                         | `text`  | Roboflow zip + HF masks |

Each pair becomes a class string `"{texture}+{defect}"` (e.g. `IC+bridge`) — the `defect_type` in
`defect_spec.jsonl` and one `[texture, defect]` entry in the recipe's `anomaly_types`. **`pcb` spans
two textures**, so its three classes are `IC+bridge`, `passive_component+excess_solder` and
`passive_component+missing`.

**PCB** — the full dataset already ships in the expected layout, so the script is a thin
`snapshot_download`:

```shell
python -m anomalygen.scripts.datasets.prepare_pcb_defect --output_dir datasets/pcb
```

**Magnetic tile** — downloaded from public GitHub (no auth); the script curates the reference subset
(5 anomaly images + masks per type, 20 clean images) and converts the anomaly JPEGs to PNG; the
clean images stay JPEG. The raw archive is cached under `datasets/magnetic_tile/.cache/`:

```shell
python -m anomalygen.scripts.datasets.prepare_magnetic_tile_defect --output_dir datasets/magnetic_tile
```

**Phone screen** — masks and `defect_spec.jsonl` come from HF, but the anomaly + clean **images**
come from a Roboflow COCO export, which has no unauthenticated download.

Download the COCO export zip from
<https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen>:

1. Register a free Roboflow account and log in (you may need to set up your Projects / Workspace page
   on first use).
2. Open <https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen>.
3. Click **Fork Dataset** to copy it into your own Workspace.
4. In your Workspace, open the **Mobile screen** project and select **Dataset** in the left bar.
5. At the bottom of the page, tick the check box to the left of **0 selected**, then click
   **Select all 320 media items in your library**. Click **Export** at the bottom-right.
6. Choose **Analyze or experiment**, then **Continue**.
7. Click **Download Anyway**.
8. Select **COCO** format, click **Export As**, and choose **Zip file**.
9. When the export finishes, click **Download**.
10. Save the resulting `Mobile screen.coco.zip` somewhere the script can read it.

Pass that zip to `--zip`. It is required: without it the script would write a dataset with no
images in it, so it refuses and points back here. `--masks_only` asks for that mask-only tree
deliberately.

```shell
python -m anomalygen.scripts.datasets.prepare_phone_screen_defect \
    --output_dir datasets/phone_screen --zip <path/to/downloaded.zip>
```

**HF auth** (`export HF_TOKEN=<token>` or `hf auth login`) is required for `pcb` and `phone_screen`;
`magnetic_tile` needs none.

### Verify the output tree

Each script ends with a `Done: … N anomaly, N mask, N clean; N defect_spec entries` summary.
`phone_screen` and `magnetic_tile` produce the same shape — only the texture folder and defect names
differ; `pcb` adds a second texture, a `cad_mask/` folder per texture and
`semantic_segmentation_labels.json` at the root (see the contract below), and ships JPEG images and
masks. For `phone_screen`:

```text
datasets/phone_screen/
├── Phone/                                  # the {texture} folder
│   ├── anomaly_image/                      # defect images        (from Roboflow --zip)
│   │   ├── oil/       Oil_0001.png  Oil_0021.png  …
│   │   ├── scratch/   Scr_0001.png  Scr_0021.png  …
│   │   └── stain/     Sta_0001.png  Sta_0021.png  …
│   ├── mask/                               # binary defect masks  (from Hugging Face)
│   │   ├── oil/       Oil_0001_mask.png  Oil_0021_mask.png  …   # <stem>_mask.png, pairs 1:1 with anomaly_image
│   │   ├── scratch/   Scr_0001_mask.png  …
│   │   └── stain/     Sta_0001_mask.png  …
│   └── clean_image/   0001.png  0002.png  …    # defect-free screens (from Roboflow --zip)
└── defect_spec.jsonl                       # one JSON line per defect type, AMP only (from Hugging Face)
```

- **`anomaly_image/{defect}/` + `mask/{defect}/`** are the only inputs **training** reads. Each defect
  image is paired with the mask of the same stem plus a `_mask` suffix.
- **`clean_image/`** holds the defect-free images that mask placement and generation paint onto.
- **`defect_spec.jsonl`** describes each defect type (its ROI prompt / spatial dependency). It is read
  only by classic Auto Mask Placement; Agentic Mask Generation ignores it.

That directory is now your `dataset_dir` — the input to page 3, and the `dataset_path` you pass to a
fine-tuning recipe ([`4-fine-tuning.md`](./4-fine-tuning.md)).

______________________________________________________________________

## Organizing a custom dataset

To fine-tune on your own data, you produce the **same layout** the prepare scripts emit — match it
exactly so the trainer can find your images and their masks.

### The directory contract

For a dataset rooted at `{dataset_dir}`, each `(texture, defect)` class needs three things:

```text
{dataset_dir}/{texture}/anomaly_image/{defect}/<name>.png   # the defect image
{dataset_dir}/{texture}/mask/{defect}/<name>_mask.png       # binary mask for that image
{dataset_dir}/{texture}/clean_image/<name>.png              # clean images (as generation inputs)
{dataset_dir}/defect_spec.jsonl                             # AMP only: one JSON object per defect type

# AMP only, and only when a defect uses spatial_dependency: cad or cad_tht
{dataset_dir}/{texture}/cad_mask/<clean_stem>.png           # CAD mask, one per clean_image/ file
{dataset_dir}/semantic_segmentation_labels.json             # label map for those CAD masks
```

What to get right:

- **Every defect image needs a matching mask.** For `anomaly_image/{defect}/<name>.png`, put its mask
  at `mask/{defect}/<name>_mask.png` — the same stem plus a `_mask` suffix, with the image's own
  extension or `.png` (PNG recommended; JPEG masks are lossy). Any image with no matching mask is
  skipped. Images can be `.png`, `.jpg`, or `.jpeg`.
- **Image and mask must be the same pixel size**, and a mask must not be empty: empty masks are
  skipped, and a class whose masks are all empty (or that has no pairs) stops training.
- **In each mask, white marks the defect and black is the background.** Pixels `>= 128` count as
  defect.
- **`clean_image/` is not used for training.** It holds the defect-free images that *generation*
  ([`5-generation.md`](5-generation.md)) paints anomalies onto. Only the `anomaly_image` + `mask` pairs
  are used to fine-tune.
- **Folder names are the defect labels.** The `{texture}` and `{defect}` folder names must match the
  `anomaly_types: [[texture, defect], ...]` you list in the fine-tuning recipe, and the
  `"{texture}+{defect}"` strings (e.g. `Phone+oil`) you use in `defect_spec.jsonl`. The names also
  appear in the training caption, and a defect name the content-safety guardrail flags (e.g. `crack`)
  blocks every generated image of that type unless generation runs with `--no-guardrail` (see
  [5-generation.md](5-generation.md#the-content-safety-guardrail)).
- **`cad_mask/` is keyed by the clean image, not the defect.** It is needed only for defects with
  `spatial_dependency: cad` or `cad_tht`, and holds one mask per `clean_image/` file under the *same
  stem*. `cad_tht` additionally requires each mask to be the **same size** as its clean image, and its
  label file to declare `pad`, `hole`, and a lead class — `pin_tip` or `pin`.

### `defect_spec.jsonl`

One JSON object per defect type, read only by classic Auto Mask Placement. Real rows from
`datasets/phone_screen/defect_spec.jsonl`:

```json
{"defect_type": "Phone+oil", "spatial_dependency": "text", "roi_prompt_defect_location": "the entire mobile phone screen surface, excluding the phone frame and background"}
{"defect_type": "Phone+scratch", "spatial_dependency": "text", "roi_prompt_defect_location": "the entire mobile phone screen surface, excluding the phone frame and background"}
{"defect_type": "Phone+stain", "spatial_dependency": "text", "roi_prompt_defect_location": "the entire mobile phone screen surface, excluding the phone frame and background"}
```

- `defect_type` — the `"{texture}+{defect}"` class string.
- `spatial_dependency` — how the defect region is decided (e.g. `text` = derived from a text ROI prompt).
- `roi_prompt_defect_location` — natural-language description of where the defect may appear.

> **Always set `spatial_dependency` explicitly.** It is optional in the file, but `roi_pair` and
> `roi_place` fall back to *different* defaults when it is missing (`free` and `text` respectively),
> so an omitted field routes the two stages inconsistently. Every row above sets it.

### Concrete example: three textures (`bottle`, `cable`, `screw`)

One dataset can hold several textures at once — each texture is its own top-level folder under
`{dataset_dir}`, with its own defect subfolders. Say your dataset has three textures: `bottle`
(defects `chip`, `stain`), `cable` (defects `cut`, `bent`, `frayed`), and `screw` (defect `scratch`).
The tree becomes:

```text
datasets/mydata/
├── bottle/
│   ├── anomaly_image/
│   │   ├── chip/    0001.png  0002.png  …
│   │   └── stain/   0001.png  0002.png  …
│   ├── mask/
│   │   ├── chip/    0001_mask.png  0002_mask.png  …
│   │   └── stain/   0001_mask.png  0002_mask.png  …
│   └── clean_image/ 0001.png  0002.png  …
├── cable/
│   ├── anomaly_image/
│   │   ├── cut/      0001.png  0002.png  …
│   │   ├── bent/     0001.png  0002.png  …
│   │   └── frayed/   0001.png  0002.png  …
│   ├── mask/
│   │   ├── cut/      0001_mask.png  0002_mask.png  …
│   │   ├── bent/     0001_mask.png  0002_mask.png  …
│   │   └── frayed/   0001_mask.png  0002_mask.png  …
│   └── clean_image/  0001.png  0002.png  …
├── screw/
│   ├── anomaly_image/scratch/   0001.png  0002.png  …
│   ├── mask/scratch/            0001_mask.png  0002_mask.png  …
│   └── clean_image/             0001.png  0002.png  …
└── defect_spec.jsonl
```

`defect_spec.jsonl` — one row per `{texture}+{defect}` class (six rows here):

```json
{"defect_type": "bottle+chip", "spatial_dependency": "text", "roi_prompt_defect_location": "the surface of the bottle"}
{"defect_type": "bottle+stain", "spatial_dependency": "text", "roi_prompt_defect_location": "the surface of the bottle"}
{"defect_type": "cable+cut", "spatial_dependency": "text", "roi_prompt_defect_location": "the cable"}
{"defect_type": "cable+bent", "spatial_dependency": "text", "roi_prompt_defect_location": "the cable"}
{"defect_type": "cable+frayed", "spatial_dependency": "text", "roi_prompt_defect_location": "the cable"}
{"defect_type": "screw+scratch", "spatial_dependency": "text", "roi_prompt_defect_location": "the screw thread"}
```

The matching fine-tuning recipe would then declare every `[texture, defect]` pair:

```yaml
anomaly_types: [[bottle, chip], [bottle, stain], [cable, cut], [cable, bent], [cable, frayed], [screw, scratch]]
dataset_path: /path/to/datasets/mydata
```

Each `[texture, defect]` pair points at `{dataset_path}/{texture}/anomaly_image/{defect}/` and its
`mask/{defect}/` sibling, and produces the class string (`"bottle+chip"`, `"cable+cut"`, …) used in
`defect_spec.jsonl`.

______________________________________________________________________

## Next step

Continue to [3 · Agentic Mask Generation](3-agentic-mask-generation.md), or to the classic
[3 · Auto Mask Placement](3-auto-mask-placement.md).
