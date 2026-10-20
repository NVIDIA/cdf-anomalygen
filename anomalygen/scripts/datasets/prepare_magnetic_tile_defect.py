# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prepare the Magnetic Tile defect dataset for AnomalyGen training.

Downloads from: https://github.com/abin24/Magnetic-tile-defect-datasets.

Source layout per defect folder:
  MT_<TYPE>/Imgs/
    exp*_num_*.jpg   <- defect image
    exp*_num_*.png   <- pixel-level mask

  MT_Free/Imgs/
    exp*_num_*.jpg   <- clean (normal) image
    exp*_num_*.png   <- ignored

Output layout:
  <output_dir>/
    metal_surface/
      anomaly_image/<TYPE>/   exp*_num_*.png   (jpg converted to png)
      mask/<TYPE>/            exp*_num_*_mask.png
      clean_image/            exp*_num_*.jpg
    defect_spec.jsonl

Usage:
  The dataset is downloaded automatically from GitHub — no manual step required. The raw repo is
  downloaded + extracted into <output_dir>/.cache/ and kept there.

  Run the script (from the repo root, in the project venv):
    python3 -m anomalygen.scripts.datasets.prepare_magnetic_tile_defect --output_dir <output_dir>

  Optional:
    --dryrun    Show what would be downloaded/written without doing it

Every prepare_*.py follows the same phases: preflight -> download -> post_process (dryrun previews).
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

# Pinned to a commit, not refs/heads/master: a branch archive resolves afresh on every run, so the
# owner of this third-party repo would choose the images we fine-tune on. That is data poisoning,
# not just a reproducibility nicety.
REPO_COMMIT = "cc4eb530b57b9cb7225bf76023873043ffd50655"
REPO_ZIP_URL = f"https://github.com/abin24/Magnetic-tile-defect-datasets./archive/{REPO_COMMIT}.zip"
# The commit pin fixes the content, this fixes the bytes: a raw GitHub zip carries no integrity data
# of its own, unlike a HuggingFace snapshot_download. Caveat as in docker/Dockerfile — GitHub does
# not guarantee byte-stable archives, so re-verify the content before ever refreshing this.
REPO_ZIP_SHA256 = "592b050953d1de4ffa538b93b6b91aee7268d8e59b73d4d9c92cfce187d7ce66"
DEFECT_TYPES = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven"]
CLEAN_TYPE = "MT_Free"

DEFECT_SPEC = [
    {"defect_type": "metal_surface+MT_Blowhole", "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Break", "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Crack", "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Fray", "spatial_dependency": "free", "roi_prompt_defect_location": ""},
    {"defect_type": "metal_surface+MT_Uneven", "spatial_dependency": "free", "roi_prompt_defect_location": ""},
]

# Curated subset: exactly the files shipped in the reference Magnetic Tile dataset.
KEEP_ANOMALY = {
    "MT_Blowhole": {"exp1_num_265077", "exp1_num_4727", "exp2_num_51697", "exp5_num_297552", "exp5_num_7054"},
    "MT_Break": {"exp2_num_116961", "exp2_num_271384", "exp3_num_148977", "exp3_num_26146", "exp5_num_98336"},
    "MT_Crack": {"exp1_num_265613", "exp1_num_276355", "exp1_num_339819", "exp4_num_116553", "exp5_num_116575"},
    "MT_Fray": {"exp1_num_135544", "exp1_num_331149", "exp3_num_135593", "exp3_num_136351", "exp3_num_20409"},
    "MT_Uneven": {"exp0_num_461", "exp5_num_109313", "exp5_num_352540", "exp6_num_139916", "exp6_num_172365"},
}
KEEP_CLEAN = {
    "exp0_num_743",
    "exp1_num_10181",
    "exp1_num_10334",
    "exp1_num_10903",
    "exp1_num_11276",
    "exp1_num_13526",
    "exp1_num_16503",
    "exp1_num_1810",
    "exp1_num_18129",
    "exp1_num_19695",
    "exp1_num_2038",
    "exp1_num_2610",
    "exp1_num_3504",
    "exp1_num_3786",
    "exp1_num_5907",
    "exp1_num_6508",
    "exp1_num_6539",
    "exp1_num_7871",
    "exp1_num_824",
    "exp1_num_855",
}


def preflight(args) -> None:
    """1. No auth or setup needed — the Magnetic Tile dataset is a public GitHub download."""


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a multi-hundred-MB archive is not held in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(args) -> Path:
    """2. Download + extract the raw GitHub repo into a kept <output_dir>/.cache/ subdir; return its root."""
    cache_dir = Path(args.output_dir) / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "magnetic_tile.zip"
    try:
        urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)
    except Exception as e:
        sys.exit(f"Download failed: {e}")

    # Verify before extracting, not after: extraction is what puts attacker-chosen bytes on disk
    # where the curation step below will pick them up.
    digest = _sha256(zip_path)
    if digest != REPO_ZIP_SHA256:
        zip_path.unlink(missing_ok=True)
        sys.exit(
            f"ERROR: {REPO_ZIP_URL}\n"
            f"  expected sha256 {REPO_ZIP_SHA256}\n"
            f"  got      sha256 {digest}\n"
            "The archive at this pinned commit is not the one recorded. Re-verify the content "
            "upstream before updating REPO_ZIP_SHA256 — do not refresh the digest to make this pass."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    zip_path.unlink()

    extracted = sorted(cache_dir.iterdir())
    if not extracted:
        sys.exit("Extraction produced no files.")
    return extracted[0]


def post_process(args, raw: Path) -> None:
    """3. Curate the reference subset from the extracted repo (raw) into <output_dir>/metal_surface/."""
    output_dir = Path(args.output_dir)
    metal_dir = output_dir / "metal_surface"

    # Anomaly images (jpg -> png) + pixel masks, curated to KEEP_ANOMALY.
    for defect_type in DEFECT_TYPES:
        src = raw / defect_type / "Imgs"
        if not src.exists():
            print(f"WARNING: {src} not found, skipping {defect_type}.")
            continue
        keep = KEEP_ANOMALY[defect_type]
        dst_anomaly = metal_dir / "anomaly_image" / defect_type
        dst_mask = metal_dir / "mask" / defect_type
        dst_anomaly.mkdir(parents=True, exist_ok=True)
        dst_mask.mkdir(parents=True, exist_ok=True)
        for jpg in (f for f in sorted(src.glob("*.jpg")) if f.stem in keep):
            Image.open(jpg).convert("RGB").save(dst_anomaly / f"{jpg.stem}.png")
        for png in (f for f in sorted(src.glob("*.png")) if f.stem in keep):
            shutil.copy2(png, dst_mask / f"{png.stem}_mask.png")

    # Clean images, curated to KEEP_CLEAN.
    src_clean = raw / CLEAN_TYPE / "Imgs"
    if src_clean.exists():
        dst_clean = metal_dir / "clean_image"
        dst_clean.mkdir(parents=True, exist_ok=True)
        for jpg in (f for f in sorted(src_clean.glob("*.jpg")) if f.stem in KEEP_CLEAN):
            shutil.copy2(jpg, dst_clean / jpg.name)
    else:
        print(f"WARNING: {src_clean} not found.")

    # defect_spec.jsonl
    with (output_dir / "defect_spec.jsonl").open("w") as f:
        for entry in DEFECT_SPEC:
            f.write(json.dumps(entry) + "\n")


def dryrun(args) -> None:
    cache_dir = Path(args.output_dir) / ".cache"
    print(f"dry-run: download {REPO_ZIP_URL}")
    print(f"         -> extract into {cache_dir}, then write the curated subset:")
    for dt in DEFECT_TYPES:
        print(f"  metal_surface/anomaly_image/{dt}/: {len(KEEP_ANOMALY[dt])} images + masks")
    print(f"  metal_surface/clean_image/: {len(KEEP_CLEAN)} images")
    print(f"  defect_spec.jsonl: {len(DEFECT_SPEC)} entries")


def main(argv=None) -> None:
    # allow_abbrev=False: argparse otherwise accepts any unambiguous abbreviation of a long option,
    # and these scripts are launched by `usecase://<name>?<p>=...`, where the parameter names come
    # from a config. `?output=` reached --output_dir that way and sent the dataset wherever the
    # config said. staging/usecase.py guards its own flag too, but only the parser can close the
    # class: a flag added here later would reopen it, and every abbreviation of it with it.
    parser = argparse.ArgumentParser(
        description="Download and prepare the Magnetic Tile defect dataset for AnomalyGen.",
        allow_abbrev=False,
    )
    parser.add_argument("--output_dir", required=True, help="Destination directory for the prepared dataset")
    parser.add_argument(
        "--dryrun", action="store_true", help="Show what would be downloaded and written without doing it"
    )
    args = parser.parse_args(argv)

    print("Notice: The user is responsible for checking if the dataset license is fit for the intended purpose.")
    if args.dryrun:
        dryrun(args)
        return
    print("[1/3] preflight: no setup needed (public GitHub)")
    preflight(args)
    print("[2/3] download: fetching Magnetic Tile repo from GitHub")
    raw = download(args)
    print("[3/3] post_process: curating reference subset")
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
