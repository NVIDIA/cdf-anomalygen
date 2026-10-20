# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prepare the PCB defect dataset for AnomalyGen training.

Source: Hugging Face dataset repo ``nvidia/Cosmos-AnomalyGen-PCB-Dataset``.

This script is a thin wrapper around ``huggingface_hub.snapshot_download`` plus
a defensive single-wrapper-dir flatten step in case the snapshot lands inside
one outer directory.

Output layout (two textures; JPEG images and ``<stem>_mask.jpg`` masks):
  <output_dir>/
    IC/  passive_component/
      anomaly_image/<TYPE>/
      mask/<TYPE>/
      clean_image/
      cad_mask/                       one CAD mask per clean image, same stem
    defect_spec.jsonl
    semantic_segmentation_labels.json label map for the CAD masks

Usage:
  Step 1 — Authenticate with Hugging Face. Either:
    - export HF_TOKEN=<your-token>          (one-shot, env-only), OR
    - run `hf auth login` once              (persists to ~/.cache/huggingface)
    Your token must have read access to nvidia/Cosmos-AnomalyGen-PCB-Dataset.

  Step 2 — Run the script (from the repo root, in the project venv):
    python3 -m anomalygen.scripts.datasets.prepare_pcb_defect --output_dir <output_dir>

  The HF snapshot is downloaded directly into <output_dir> and kept there.

  Optional:
    --dryrun                Print the resolved HF target and exit

Every prepare_*.py follows the same phases: preflight -> download -> post_process (dryrun previews).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

HF_REPO_ID = "nvidia/Cosmos-AnomalyGen-PCB-Dataset"
# Pinned to a commit, not "main". These images are fine-tuning data, so a mutable revision means a
# later run can train on different content for the same command — an integrity property, not just a
# reproducibility one. snapshot_download verifies each file against the repo's published hash once
# the revision is fixed, so the pin is what turns that check into a meaningful one.
DEFAULT_REVISION = "71bcf9468dcf39ce07cbcacc2165f4a8831bb43e"


def preflight(args) -> None:
    """1. Verify ``huggingface_hub`` is importable and HF auth is available."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: `huggingface_hub` is not installed in the active env.\n"
            "  - It ships in this repo's requirements.txt.\n"
            "  - Otherwise: `pip install 'huggingface_hub>=1.0'`."
        )
    if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache/huggingface/token").is_file():
        sys.exit(
            "ERROR: Hugging Face is not authenticated.\n"
            "  Either:\n"
            "    - export HF_TOKEN=<your-token>, OR\n"
            "    - run `hf auth login` once to persist your token.\n"
            f"  The token needs read access to {HF_REPO_ID}."
        )


def download(args) -> Path:
    """2. Snapshot the HF dataset repo straight into output_dir (it already ships the expected layout)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=DEFAULT_REVISION,
            local_dir=str(dest),
            token=os.environ.get("HF_TOKEN"),
        )
    except RepositoryNotFoundError:
        sys.exit(f"ERROR: Hugging Face repo {HF_REPO_ID} not found or your token lacks read access.")
    except HfHubHTTPError as exc:
        sys.exit(f"ERROR: Hugging Face download failed: {exc}")
    return dest


def post_process(args, raw: Path) -> None:
    """3. Flatten a single wrapper dir if the snapshot nested one, then sanity-check that files landed."""
    children = [p for p in raw.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(children) == 1:
        wrapper = children[0]
        stage = raw / f".flatten.{os.getpid()}"
        wrapper.rename(stage)
        for item in stage.iterdir():
            shutil.move(str(item), str(raw / item.name))
        stage.rmdir()
    if sum(1 for p in raw.rglob("*") if p.is_file()) == 0:
        sys.exit("ERROR: 0 files downloaded.")


def dryrun(args) -> None:
    print(
        f"dry-run: hf download {HF_REPO_ID} --repo-type dataset "
        f"--revision {DEFAULT_REVISION} --local-dir {args.output_dir}"
    )


def main(argv=None) -> None:
    # allow_abbrev=False: argparse otherwise accepts any unambiguous abbreviation of a long option,
    # and these scripts are launched by `usecase://<name>?<p>=...`, where the parameter names come
    # from a config. `?output=` reached --output_dir that way and sent the dataset wherever the
    # config said. staging/usecase.py guards its own flag too, but only the parser can close the
    # class: a flag added here later would reopen it, and every abbreviation of it with it.
    parser = argparse.ArgumentParser(
        description="Download and stage the PCB defect dataset from Hugging Face.", allow_abbrev=False
    )
    parser.add_argument("--output_dir", required=True, help="Destination directory for the prepared dataset")
    parser.add_argument("--dryrun", action="store_true", help="Show what would be downloaded without doing it")
    args = parser.parse_args(argv)

    print("Notice: The user is responsible for checking if the dataset license is fit for the intended purpose.")
    if args.dryrun:
        dryrun(args)
        return
    print("[1/3] preflight: checking Hugging Face auth")
    preflight(args)
    print(f"[2/3] download: fetching {HF_REPO_ID} from Hugging Face")
    raw = download(args)
    print("[3/3] post_process: staging files")
    post_process(args, raw)

    out = Path(args.output_dir)
    spec = out / "defect_spec.jsonl"
    n_spec = sum(1 for _ in spec.open()) if spec.exists() else 0
    print(
        f"Done: {out} -> {len(list(out.glob('*/anomaly_image/*/*')))} anomaly, "
        f"{len(list(out.glob('*/mask/*/*')))} mask, {len(list(out.glob('*/clean_image/*')))} clean; "
        f"{n_spec} defect_spec entries"
    )


if __name__ == "__main__":
    main()
