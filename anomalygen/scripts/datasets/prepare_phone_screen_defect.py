# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prepare the Mobile Phone Screen defect dataset for AnomalyGen training.

Sources:
  - Anomaly + clean images: Roboflow COCO export (single zip).
      Download from: https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen
      The zip contains one split directory (e.g. train/) with images and
      _annotations.coco.json.
  - Masks + defect_spec.jsonl: Hugging Face dataset repo
      nvidia/Cosmos-AnomalyGen-Glass-Masks (always fetched from HF).

Image filename mapping (Roboflow hash removed):
  Oil_0017_jpg.rf.<hash>.jpg  ->  anomaly_image/oil/Oil_0017.png
  Scr_0016_jpg.rf.<hash>.jpg  ->  anomaly_image/scratch/Scr_0016.png
  Sta_0067_jpg.rf.<hash>.jpg  ->  anomaly_image/stain/Sta_0067.png
  0001_png.rf.<hash>.png      ->  clean_image/0001.png

Output layout (masks always from HF; anomaly + clean images require --zip, which is REQUIRED
unless --masks_only says the images are deliberately being left out):
  <output_dir>/
    Phone/
      anomaly_image/
        oil/        Oil_XXXX.png    (from Roboflow)
        scratch/    Scr_XXXX.png    (from Roboflow)
        stain/      Sta_XXXX.png    (from Roboflow)
      clean_image/  XXXX.png        (from Roboflow)
      mask/
        oil/                        (from HF)
        scratch/                    (from HF)
        stain/                      (from HF)
    defect_spec.jsonl               (from HF)

Usage:
  Step 1 — Download the dataset zip from Roboflow (browser required):
    - Download the COCO export zip from https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen

  Step 2 — Authenticate with Hugging Face (masks are always fetched from HF):
    - export HF_TOKEN=<your-token>, OR run `hf auth login` once.

  Step 3 — Run the script (from the repo root, in the project venv):
    python3 -m anomalygen.scripts.datasets.prepare_phone_screen_defect --output_dir <output_dir> \\
        --zip <path/to/downloaded.zip>

  Optional — preview what will be extracted without writing files:
    python3 -m anomalygen.scripts.datasets.prepare_phone_screen_defect --output_dir <output_dir> \\
        --zip <path/to/downloaded.zip> --dryrun

  To pull just the masks/defect_spec (skipping the Roboflow zip) -- a dataset with no images,
  which nothing can generate from, so it has to be asked for explicitly:
    python3 -m anomalygen.scripts.datasets.prepare_phone_screen_defect --output_dir <output_dir> \\
        --masks_only

Every prepare_*.py follows the same phases: preflight -> download -> post_process (dryrun previews).
"""

import argparse
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

HF_MASKS_REPO_ID = "nvidia/Cosmos-AnomalyGen-Glass-Masks"
ROBOFLOW_URL = "https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen"
# The staging spelling, shown in the failure because that is where most callers now arrive from --
# the entrypoint runs this script and the person reading the error is editing a config, not a shell.
USECASE_HINT = "usecase://phone_screen?zip=s3://<bucket>/mobile-screen.zip"
# Pinned to a commit, not "main" — see the rationale in prepare_pcb_defect.py: a mutable revision
# lets the masks this pipeline places change under the same command.
HF_MASKS_DEFAULT_REVISION = "e15b6827eed41d10f812739586d8c057a3b26c12"

# Curated subset: exactly the files shipped in the reference Mobile Phone Screen dataset.
KEEP_ANOMALY = {
    "oil": {"Oil_0001", "Oil_0021", "Oil_0041", "Oil_0061", "Oil_0081"},
    "scratch": {"Scr_0001", "Scr_0021", "Scr_0041", "Scr_0061", "Scr_0081"},
    "stain": {"Sta_0001", "Sta_0021", "Sta_0041", "Sta_0061", "Sta_0081"},
}


def require_images(args) -> None:
    """0. Refuse a run that would write a dataset with no images in it.

    The masks and the defect_spec come from Hugging Face; the images do not exist anywhere this can
    reach, so without --zip this script writes masks and exits 0. That is the worst shape a failure
    can take here: `usecase://phone_screen` fetched cleanly, Phone/clean_image/ was simply absent,
    and what surfaced was an override refusing a path that is "not in the fetched input" or an AMP
    run with no exemplar -- neither of which mentions a zip.

    Checked BEFORE the Hugging Face auth below and before anything is downloaded: a missing zip does
    not need a token to diagnose, and reporting the token first would send someone to fix a
    credential and come back to the real problem on the next run.
    """
    if args.zip or args.masks_only:
        return
    sys.exit(
        "ERROR: this dataset's anomaly and clean images are not downloadable.\n"
        "  They come from a Roboflow COCO export that has to be downloaded by hand:\n"
        f"    {ROBOFLOW_URL}\n"
        "  Then pass it: --zip <path/to/mobile-screen.zip>\n"
        f"    ...or through a use-case: {USECASE_HINT}\n"
        "  To fetch only the masks and defect_spec -- a dataset nothing can generate from -- pass\n"
        "  --masks_only and mean it."
    )


def preflight(args) -> None:
    """1. HF auth (masks are always fetched from HF) + validate --zip exists when given."""
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
            f"  The token needs read access to {HF_MASKS_REPO_ID}."
        )
    if args.zip and not Path(args.zip).exists():
        sys.exit(f"ERROR: {args.zip} not found")


def download(args) -> None:
    """2. Always fetch masks + defect_spec.jsonl from HF straight into output_dir (already the right layout)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=HF_MASKS_REPO_ID,
            repo_type="dataset",
            revision=HF_MASKS_DEFAULT_REVISION,
            local_dir=str(dest),
            token=os.environ.get("HF_TOKEN"),
        )
    except RepositoryNotFoundError:
        sys.exit(f"ERROR: Hugging Face repo {HF_MASKS_REPO_ID} not found or your token lacks read access.")
    except HfHubHTTPError as exc:
        sys.exit(f"ERROR: Hugging Face download failed: {exc}")


def post_process(args, raw) -> None:
    """3. Extract the Roboflow anomaly + clean images from --zip (when given) into <output_dir>/Phone/."""
    if not args.zip:
        return
    zip_path = Path(args.zip)
    phone_dir = Path(args.output_dir) / "Phone"

    entries = []  # (zip_entry, defect_type_or_None, output_stem)
    with zipfile.ZipFile(zip_path) as zf:
        all_names = set(zf.namelist())
        # Roboflow COCO exports split the data into train/valid/test, each with its own manifest —
        # union them so curated files aren't missed. Roboflow renames files as
        # <orig>_<ext>.rf.<hash>.<ext>; strip the hash, route by prefix (Oil_/Scr_/Sta_ = anomaly,
        # else clean), and curate anomalies down to KEEP_ANOMALY.
        manifests = sorted(n for n in all_names if n.endswith("/_annotations.coco.json"))
        if not manifests:
            sys.exit("ERROR: no _annotations.coco.json found in zip.")
        for manifest in manifests:
            split = manifest.rsplit("/_annotations.coco.json", 1)[0]
            for img in json.loads(zf.read(manifest))["images"]:
                fname = img["file_name"]
                zip_entry = f"{split}/{fname}"
                if zip_entry not in all_names:
                    print(f"WARNING: {zip_entry} not in zip, skipping.")
                    continue
                m = re.match(r"^(.+?)_(jpg|png)\.rf\.[A-Za-z0-9]+\.(?:jpg|png)$", fname)
                orig = f"{m.group(1)}.{m.group(2)}" if m else fname
                stem = Path(orig).stem
                if orig.startswith("Oil_"):
                    dtype = "oil"
                elif orig.startswith("Scr_"):
                    dtype = "scratch"
                elif orig.startswith("Sta_"):
                    dtype = "stain"
                else:
                    dtype = None
                if dtype is not None and stem not in KEEP_ANOMALY.get(dtype, set()):
                    continue
                entries.append((zip_entry, dtype, stem))

        # Write them out, re-encoded to png.
        for dtype in ("oil", "scratch", "stain"):
            (phone_dir / "anomaly_image" / dtype).mkdir(parents=True, exist_ok=True)
        (phone_dir / "clean_image").mkdir(parents=True, exist_ok=True)
        for zip_entry, dtype, stem in entries:
            data = zf.read(zip_entry)
            if dtype is None:
                dst = phone_dir / "clean_image" / f"{stem}.png"
            else:
                dst = phone_dir / "anomaly_image" / dtype / f"{stem}.png"
            Image.open(io.BytesIO(data)).convert("RGB").save(dst)


def dryrun(args) -> None:
    if args.zip:
        print(f"dry-run: extract Roboflow anomaly + clean images from {args.zip} into {args.output_dir}/Phone/")
    print(
        f"dry-run: hf download {HF_MASKS_REPO_ID} --repo-type dataset "
        f"--revision {HF_MASKS_DEFAULT_REVISION} --local-dir {args.output_dir}"
    )


def main(argv=None) -> None:
    # allow_abbrev=False: argparse otherwise accepts any unambiguous abbreviation of a long option,
    # and these scripts are launched by `usecase://<name>?<p>=...`, where the parameter names come
    # from a config. `?output=` reached --output_dir that way and sent the dataset wherever the
    # config said. staging/usecase.py guards its own flag too, but only the parser can close the
    # class: a flag added here later would reopen it, and every abbreviation of it with it.
    parser = argparse.ArgumentParser(
        description="Prepare the Mobile Phone Screen defect dataset for AnomalyGen (Roboflow images + HF masks).",
        allow_abbrev=False,
    )
    parser.add_argument("--output_dir", required=True, help="Destination directory for the prepared dataset")
    parser.add_argument(
        "--zip",
        metavar="mobile_screen.zip",
        help="Optional Roboflow zip; when given, its anomaly + clean images are extracted",
    )
    parser.add_argument(
        "--masks_only",
        action="store_true",
        help="Fetch the masks and defect_spec WITHOUT the Roboflow images. The result has no clean "
        "or anomaly images, so nothing can be generated from it; for inspecting the masks.",
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="Show what would be extracted/downloaded without writing files"
    )
    args = parser.parse_args(argv)

    print("Notice: The user is responsible for checking if the dataset license is fit for the intended purpose.")
    if args.dryrun:
        dryrun(args)
        return
    require_images(args)
    print("[1/3] preflight: checking Hugging Face auth + inputs")
    preflight(args)
    print(f"[2/3] download: fetching masks + defect_spec from {HF_MASKS_REPO_ID}")
    raw = download(args)
    if args.zip:
        print(f"[3/3] post_process: extracting curated images from {Path(args.zip).name}")
    else:
        print("[3/3] post_process: skipped (--masks_only; no images in this dataset)")
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
