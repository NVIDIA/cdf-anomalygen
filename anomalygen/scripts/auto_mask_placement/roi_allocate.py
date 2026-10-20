# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distribute num_sdg across defect types → allocation.json (roi_pair ``--allocation``).

Runs a fail-fast **preflight** over the AMP inputs (dataset_dir, defect_spec)
before allocating: every submask dir, clean-image pool (under dataset_dir), text
``roi_prompt_defect_location``, and cad_mask is checked and all problems are
reported at once (exit 1), so the downstream pair_build/place run can't crash
mid-AMP.

Two modes:

  * --mode inference (default): uniform allocation across defect types.
    base = num_sdg // N; first (num_sdg % N) defects get +1. No floor on
    per-defect count (allows 0 if --per_defect_counts asks for it).

  * --mode validation: proportional to training mask counts (largest-remainder
    rounding). Enforces a KPI floor — every defect must have ≥1 sample so
    per-defect validation metrics stay defined.

Override:

  * --per_defect_counts <JSON> sets exact per-defect counts (e.g.
    '{"IC+bridge": 5, "passive_component+missing": 10}'). Only valid with
    --mode inference. When the sum != --num_sdg, a warning is logged and the
    override sum is used.

Defect types are read from --defect_desc (the defect_spec JSONL), in file order.

Usage::

    python -m anomalygen.scripts.auto_mask_placement.roi_allocate \\
        --num_sdg N --defect_desc datasets/<ds>/defect_spec.jsonl \\
        --dataset_dir datasets/<ds> --output_allocation allocation.json [--mode inference]
"""

import argparse
import json
import pathlib

from cosmos_framework.utils import log
from PIL import Image

from anomalygen import allocation
from anomalygen.auto_mask_placement.tht2roi import THT_CLASSES
from anomalygen.data.utils import list_images, validate_anomaly_type

# Both routes are fed by <TEXTURE>/cad_mask/<clean_stem>.png plus the root label file; they differ
# only in how the parsed classes are turned into regions.
_CAD_LIKE = frozenset({"cad", "cad_tht"})

# The classes cad_tht reads as regions B and C.
_THT_REQUIRED_CLASSES = frozenset({"pad", "hole"})

# D is the lead face, which newer annotations name pin_tip and older ones name pin. Either
# satisfies the requirement -- demanding both would reject the very data this route is for.
_THT_LEAD_CLASSES = frozenset({"pin_tip", "pin"})


def _validate_amp_inputs(dataset_dir, defect_desc):
    """Cross-check the AMP inputs (dataset_dir, defect_desc); return error strings ([] == OK).

    Per defect in defect_desc:
      * <dataset_dir>/<TEXTURE>/mask/<ANOMALY>/ exists (submask source)
      * <dataset_dir>/<TEXTURE>/clean_image/ or <dataset_dir>/<TEXTURE>/ has >=1 image
        (flat <dataset_dir>/ as the final fallback)
      * spatial_dependency=="text": roi_prompt_defect_location present and non-empty
      * spatial_dependency in ("cad", "cad_tht"): semantic_segmentation_labels.json exists,
        <TEXTURE>/cad_mask/ exists, and a cad_mask/<stem>.png exists per clean image
      * spatial_dependency=="cad_tht": the label file declares the THT classes, and the
        defect names are in the published THT enum
    """
    dataset_dir = pathlib.Path(dataset_dir)
    entries = [json.loads(line) for line in pathlib.Path(defect_desc).read_text().splitlines() if line.strip()]

    errors = []
    if any(e.get("spatial_dependency") in _CAD_LIKE for e in entries):
        labels = dataset_dir / "semantic_segmentation_labels.json"
        if not labels.exists():
            errors.append(f"missing {labels} (required for spatial_dependency={'/'.join(sorted(_CAD_LIKE))})")
        elif any(e.get("spatial_dependency") == "cad_tht" for e in entries):
            # A cad_tht run against a `cad` label file (pad/capacitor/ic/solder) parses cleanly and
            # yields zero joints, so every row would be dropped warn-only with nothing to point at.
            # Check the palette here, where the failure can still name what is missing.
            try:
                declared = {info["class"] for info in json.loads(labels.read_text()).values()}
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{labels}: unreadable label file ({exc})")
            else:
                missing = sorted(_THT_REQUIRED_CLASSES - declared)
                if not declared & _THT_LEAD_CLASSES:
                    missing.append(" or ".join(sorted(_THT_LEAD_CLASSES)))
                if missing:
                    errors.append(
                        f"{labels}: spatial_dependency=cad_tht needs the class(es) {missing}; "
                        f"the file declares {sorted(declared)}"
                    )

    for e in entries:
        full = e["defect_type"]
        sd = e.get("spatial_dependency")
        # Rejected here, at the dataset spec, because this value becomes a path segment below and a
        # directory name in the pseudo-label layout several stages later.
        try:
            validate_anomaly_type(full, field="defect_type")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        texture, anomaly = full.split("+", 1)

        submask_dir = dataset_dir / texture / "mask" / anomaly
        if not submask_dir.is_dir():
            errors.append(f"{full}: missing submask source {submask_dir}")

        clean_nested = dataset_dir / texture / "clean_image"
        clean_tex = dataset_dir / texture
        if clean_nested.is_dir():
            clean_pool = clean_nested
        elif clean_tex.is_dir():
            clean_pool = clean_tex
        else:
            clean_pool = dataset_dir
        clean_imgs = list_images(clean_pool)
        if not clean_imgs:
            errors.append(f"{full}: no clean images under {clean_pool}")

        if sd == "text":
            prompt = (e.get("roi_prompt_defect_location") or "").strip()
            if not prompt:
                errors.append(f"{full}: spatial_dependency={sd} requires non-empty roi_prompt_defect_location")
        elif sd in _CAD_LIKE:
            cad_dir = dataset_dir / texture / "cad_mask"
            if not cad_dir.is_dir():
                errors.append(f"{full}: missing {cad_dir}")
                continue
            missing_cad = [p.stem for p in clean_imgs if not (cad_dir / f"{p.stem}.png").exists()]
            if missing_cad:
                errors.append(
                    f"{full}: {len(missing_cad)} clean image(s) without matching cad_mask (first: {missing_cad[0]}.png)"
                )
            # The enum is a module constant, not a spec field, so a typo is caught here rather than
            # silently selecting no ROI rule at placement time.
            if sd == "cad_tht" and anomaly not in THT_CLASSES:
                errors.append(
                    f"{full}: unknown cad_tht defect {anomaly!r}; supported: {', '.join(sorted(THT_CLASSES))}"
                )
            # cad_tht builds the ROI, the AMP canvas and the placed mask at the cad_mask's resolution,
            # so a clean image of a different size would be resampled against that mask downstream.
            # Checked here because it is a property of the dataset, not of one run. Deliberately not
            # applied to `cad`, whose set of accepted inputs must not change.
            if sd == "cad_tht":
                mismatched = []
                for clean in clean_imgs:
                    cad_path = cad_dir / f"{clean.stem}.png"
                    if not cad_path.exists():
                        continue
                    with Image.open(clean) as ci, Image.open(cad_path) as cm:
                        if ci.size != cm.size:
                            mismatched.append(f"{clean.stem} (clean {ci.size} vs cad_mask {cm.size})")
                if mismatched:
                    errors.append(
                        f"{full}: {len(mismatched)} clean image(s) whose cad_mask is a different size "
                        f"(first: {mismatched[0]})"
                    )
        elif sd in (None, "free"):
            pass
        else:
            errors.append(f"{full}: unknown spatial_dependency={sd!r}")

    return errors


def _defect_types_from_spec(defect_desc):
    """Ordered, de-duplicated defect_type list from the defect_spec JSONL."""
    types = []
    for line in pathlib.Path(defect_desc).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        dt = json.loads(line)["defect_type"]
        if dt not in types:
            types.append(dt)
    return types


def _count_masks(dataset_dir, defect_types):
    counts = {}
    for full_type in defect_types:
        texture, anomaly = full_type.split("+", 1)
        d = pathlib.Path(dataset_dir) / texture / "mask" / anomaly
        counts[full_type] = len(list_images(d))
    return counts


def _uniform(num_sdg, defect_types):
    """Uniform allocation: base each, first (num_sdg % N) defects get +1.

    The rule itself lives in ``anomalygen.allocation`` because the AMG backend answers the same
    ``num_sdg`` request and has to answer it identically; this stays as the name AMP's own callers use.
    """
    return allocation.uniform(num_sdg, defect_types)


def _proportional(num_sdg, defect_types, counts):
    """Proportional to mask counts via largest-remainder rounding.

    Delegates for the reason ``_uniform`` does: AMG answers the same ``num_sdg`` request and has to
    answer it identically. This stays as the name AMP's own callers and tests use.
    """
    return allocation.proportional(num_sdg, defect_types, counts)


def _allocate(num_sdg, defect_types, counts, *, mode="inference", override=None):
    """Distribute num_sdg across defect_types.

    mode:
        - "inference": uniform; no per-defect floor.
        - "validation": proportional to mask counts; enforces >=1 per defect.

    override:
        Dict of {defect_type: count}. Only valid with mode="inference". Defect
        types not listed get 0. If sum(override) != num_sdg, a warning is logged
        and the override sum is used as the effective num_sdg.

    The policy itself lives in ``anomalygen.allocation`` so both mask backends share it; this passes
    AMP's logger in, since that module deliberately carries no logging dependency.
    """
    return allocation.allocate(
        num_sdg,
        defect_types,
        counts,
        mode=mode,
        override=override,
        warn=log.warning,
        total_label="num_sdg",
        mode_label="--mode",
        override_label="--per_defect_counts",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_sdg", type=int, required=True)
    parser.add_argument("--defect_desc", type=pathlib.Path, required=True, help="defect_spec JSONL")
    parser.add_argument(
        "--dataset_dir",
        type=pathlib.Path,
        required=True,
        help="Dataset root — scans <dataset_dir>/<TEXTURE>/mask/<ANOMALY>/.",
    )
    parser.add_argument("--output_allocation", type=pathlib.Path, required=True, help="Where to write allocation.json.")
    parser.add_argument(
        "--mode",
        choices=["inference", "validation"],
        default="inference",
        help="inference (default): uniform allocation, no floor. "
        "validation: proportional to mask counts, ≥1 per defect.",
    )
    parser.add_argument(
        "--per_defect_counts",
        default=None,
        help='JSON dict of explicit per-defect counts, e.g. \'{"IC+bridge": 5, "passive_component+missing": 10}\'. '
        "Only valid with --mode inference.",
    )
    args = parser.parse_args(argv)

    errors = _validate_amp_inputs(args.dataset_dir, args.defect_desc)
    if errors:
        log.error(f"AMP input validation failed ({len(errors)} problem(s)):")
        for msg in errors:
            log.error(f"  - {msg}")
        raise SystemExit(1)
    log.info("AMP inputs OK")

    defect_types = _defect_types_from_spec(args.defect_desc)
    counts = _count_masks(args.dataset_dir, defect_types)
    log.info(f"derived per-type mask counts: {counts}")

    override = None
    if args.per_defect_counts is not None:
        override = json.loads(args.per_defect_counts)
        if not isinstance(override, dict):
            raise ValueError("--per_defect_counts must be a JSON object (dict)")

    alloc = _allocate(args.num_sdg, defect_types, counts, mode=args.mode, override=override)
    args.output_allocation.parent.mkdir(parents=True, exist_ok=True)
    args.output_allocation.write_text(json.dumps(alloc, indent=2))
    log.info(f"wrote {args.output_allocation} (mode={args.mode}, sum={sum(alloc.values())})")


if __name__ == "__main__":
    main()
