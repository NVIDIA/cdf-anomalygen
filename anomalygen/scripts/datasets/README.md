<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataset preparation

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Dataset preparation](#dataset-preparation)
  - [Prerequisites](#prerequisites)
  - [PCB — `prepare_pcb_defect`](#pcb--prepare_pcb_defect)
  - [Metal surface (Magnetic Tile) — `prepare_magnetic_tile_defect`](#metal-surface-magnetic-tile--prepare_magnetic_tile_defect)
  - [Mobile phone screen — `prepare_phone_screen_defect`](#mobile-phone-screen--prepare_phone_screen_defect)

______________________________________________________________________

<!--TOC-->

Each `prepare_*.py` here is a standalone CLI that turns an upstream source into a ready-to-use
`dataset_dir` for AnomalyGen texture fine-tuning. Run them as modules from the repo root:

```shell
python -m anomalygen.scripts.datasets.prepare_<name> --output_dir <output_dir> [flags]
```

`--output_dir` can be anywhere, but the convention is `./datasets/<name>` — e.g. `./datasets/pcb`,
`./datasets/magnetic_tile`, `./datasets/phone_screen` (used in the examples below). Pass the resulting
directory to a recipe as `dataset_path` (see `ag_config/exp_texture_ft_*.yaml`).

| Script                         | Subject                       | Anomaly types                                       | Upstream source                                                                             | Manual step                        |
| ------------------------------ | ----------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------- |
| `prepare_pcb_defect`           | PCB                           | bridge, missing, excess_solder                      | HF `nvidia/Cosmos-AnomalyGen-PCB-Dataset`                                                   | —                                  |
| `prepare_magnetic_tile_defect` | Metal surface / Magnetic Tile | MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Uneven | GitHub `abin24/Magnetic-tile-defect-datasets`                                               | —                                  |
| `prepare_phone_screen_defect`  | Mobile phone screen           | oil, scratch, stain                                 | Roboflow (images) + HF `nvidia/Cosmos-AnomalyGen-Glass-Masks` (masks + `defect_spec.jsonl`) | Download Roboflow zip in a browser |

> The scripts print a reminder that **you are responsible for confirming each dataset's license is
> fit for your intended use.**

## Prerequisites

- The project venv already ships `huggingface_hub` (`requirements.txt`), so no extra install is
  needed for the Hugging Face paths.
- **Hugging Face auth** is required for `prepare_pcb_defect` and `prepare_phone_screen_defect` (masks are always fetched from HF).
  Either export a token or log in once, with read access to the relevant `nvidia/Cosmos-AnomalyGen-*` repos:

  ```shell
  export HF_TOKEN=<your-token>      # one-shot, env-only
  # or
  hf auth login                     # persistent
  ```

## PCB — `prepare_pcb_defect`

The full dataset (anomaly images, masks, clean images, `defect_spec.jsonl`) ships on Hugging Face in
the exact layout the pipeline expects; the script is a thin `snapshot_download` wrapper.

**Run:**

```shell
python -m anomalygen.scripts.datasets.prepare_pcb_defect --output_dir ./datasets/pcb
```

- The HF snapshot is downloaded directly into `./datasets/pcb` and kept there.
- Flags: `--dryrun` (print the resolved HF target and exit).

**Output:**

```text
./datasets/pcb/
  PCB/
    anomaly_image/<TYPE>/
    mask/<TYPE>/
    clean_image/
  defect_spec.jsonl
```

## Metal surface (Magnetic Tile) — `prepare_magnetic_tile_defect`

Downloaded automatically from the public GitHub repo. The script selects a curated subset
(5 anomaly images + masks per type, 20 clean images) matching the reference dataset.

**Run:**

```shell
python -m anomalygen.scripts.datasets.prepare_magnetic_tile_defect --output_dir ./datasets/magnetic_tile
```

- The raw GitHub repo is downloaded + extracted into `./datasets/magnetic_tile/.cache/` and kept there.
- Flags: `--dryrun` (preview the download + curated subset without writing).

**Output:**

```text
./datasets/magnetic_tile/
  metal_surface/
    anomaly_image/<TYPE>/   (jpg -> png)
    mask/<TYPE>/            (<stem>_mask.png)
    clean_image/
  defect_spec.jsonl
```

## Mobile phone screen — `prepare_phone_screen_defect`

Masks + `defect_spec.jsonl` come from Hugging Face; anomaly + clean images come from a Roboflow COCO
export. Roboflow has no unauthenticated programmatic download, so **first download the COCO export zip
in a browser** from <https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen>.

**Prerequisite:**

Follow these steps to download the Roboflow COCO export zip manually:

1. Register a free Roboflow account and log in.
2. Note: You may need to set up your Projects / Workspace page at the first time.
3. Navigate to <https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen>.
4. Click **Fork Dataset** to create your own copy of the dataset in your own Projects / Workspace.
5. In your own Projects / Workspace, enter the **Mobile screen** page and select **Dataset** on the left bar.
6. At the bottom of the page, tick the check box to the left of **0 selected**, then click
   **Select all 320 media items in your library**. Click **Export** at the bottom-right.
7. Choose **Analyze or experiment** and click **Continue**.
8. Click **Download Anyway**.
9. Select **COCO** format, click **Export As** and click **Zip file**.
10. After the export completes, you will see a **Download** button. Click it and the zip file will be downloaded to your local machine.

After downloading the zip file (`Mobile screen.coco.zip`), place or upload it to a location accessible to the script, and pass its path to `--zip`.

**Run:**

```shell
python -m anomalygen.scripts.datasets.prepare_phone_screen_defect --output_dir ./datasets/phone_screen \
    --zip <path/to/downloaded.zip>
```

- Masks + `defect_spec.jsonl` are always fetched from HF; `--zip` also extracts the Roboflow images.
- `--zip` is **required**. Without it the dataset has no clean or anomaly images, and nothing can be
  generated from it — so the script refuses rather than writing one. `--masks_only` asks for exactly
  that incomplete dataset on purpose (for inspecting the masks).
- Flags: `--zip <path>` (required unless `--masks_only`), `--masks_only`, `--dryrun` (preview without writing).

**Output:**

```text
./datasets/phone_screen/
  Phone/
    anomaly_image/{oil,scratch,stain}/    (from Roboflow)
    clean_image/                          (from Roboflow)
    mask/{oil,scratch,stain}/             (from HF)
  defect_spec.jsonl                       (from HF)
```
