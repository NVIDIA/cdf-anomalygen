# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pseudo-label generated anomalies into a COCO dataset (+ classification layout + captions).

Consumes a ``generate.py`` output tree and emits, under ``--output_dir``:

  coco_annotations.json                 COCO instance annotations (per-instance mask -> RLE + bbox)
  images/ , masks/                      copied generated images and input masks
  visualization/                        mask + bbox + label overlays
  classification/                       per-class image folders + classes.txt
  captions/ , captions_with_meta/       Cosmos3-reasoner anomaly captions (unless --no_caption)

``--gen_root`` points at the ``generate.py`` output dir; the reconstructed/original/mask subdirs and the
``texture_ft_generation_result.csv`` manifest (for each image's anomaly type) are derived from it.

Run by file path (so ``init_script`` runs before ``import anomalygen``):

    python anomalygen/scripts/texture/pseudo_label.py \
        --gen_root path/to/generate_output \
        --output_dir path/to/pseudo_labels

    python anomalygen/scripts/texture/pseudo_label.py -h
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
from typing import Any, Dict

# Framework process setup (inference env, grad disabled, distributed init when WORLD_SIZE>1).
from cosmos_framework.inference.common.init import init_script
from cosmos_framework.utils import log

# Guarded because it disables autograd process-wide; importing this module must leave it alone.
if __name__ == "__main__":
    init_script(training=False)

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from anomalygen.configs.texture.constants import DEFAULT_MAX_INSTANCES
from anomalygen.data.utils import require_same_size, validate_anomaly_type
from anomalygen.eval.utils import MASK_SUBDIR, ORIG_IMAGE_SUBDIR, RECON_SUBDIR
from anomalygen.inference.iterative import split_mask_into_instances
from anomalygen.pseudo_label import bbox as pl_bbox
from anomalygen.pseudo_label import caption as pl_caption
from anomalygen.pseudo_label import mask as pl_mask
from anomalygen.pseudo_label import utils as pl_utils

_GENERATION_CSV = "texture_ft_generation_result.csv"


def compute_annotation_dict(
    annotation_id: int,
    image_id: int,
    category_id: int,
    instance_mask: Image.Image,
) -> Dict[str, Any]:
    """Build one COCO annotation (RLE segmentation + bbox + area) for a single instance mask."""
    binary_mask = np.array(instance_mask) > 127
    area = int(np.sum(binary_mask))
    uncompressed_rle = pl_mask.binary_mask_to_rle(binary_mask)
    coco_rle = pl_mask.coco_encode_rle(uncompressed_rle)
    # Assert that the RLE encoding is correct.
    np.testing.assert_array_equal(pl_mask.coco_decode_rle(coco_rle), binary_mask)
    bbox = pl_bbox.get_bboxes(instance_mask)[0]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": coco_rle,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }


def _save_coco_annotations(coco_dict: Dict[str, Any], output_path: pathlib.Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)
    log.info(f"COCO JSON saved to {str(output_path)} with {len(coco_dict['annotations'])} annotations.")


def _contained_path(classification_dir: pathlib.Path, *parts: str) -> pathlib.Path:
    """``classification_dir`` joined with ``parts``, resolved and confirmed to be inside it.

    Second of two checks: ``anomaly_type`` is already validated where the manifest is read. Repeated
    at the join because the join is what escapes, so a caller arriving by another route is contained.
    Every part is covered, not just the class segment, and the resolved path is what is returned —
    what gets written is then the path that was checked, rather than a spelling of it walked again.
    """
    root = pathlib.Path(classification_dir).resolve()
    resolved = pathlib.Path(classification_dir, *parts).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"{'/'.join(parts)!r} resolves to {resolved}, outside the classification directory "
            f"{root}. Refusing to write there."
        )
    return resolved


def _get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-labeling script for anomaly generated data.")
    parser.add_argument(
        "--gen_root",
        required=True,
        help="generate.py output dir (reconstructed_image/ + original_image/ + original_mask/ + "
        "texture_ft_generation_result.csv)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="output dir for the COCO dataset, classification layout, and captions"
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="anomaly-type manifest CSV; default {gen_root}/texture_ft_generation_result.csv",
    )
    parser.add_argument("--no_caption", action="store_true", help="skip caption generation")
    parser.add_argument(
        "--max_instances",
        type=int,
        default=DEFAULT_MAX_INSTANCES,
        help="max per-instance masks to split each mask into (connected components, KMeans-merged beyond this); "
        "matches generate.py",
    )
    parser.add_argument(
        "--captioner_prompt_path",
        default=str(pl_caption.DEFAULT_CAPTION_PROMPT_PATH),
        help="captioning prompt config (system_prompt + user_prompt)",
    )
    parser.add_argument(
        "--captioner_temperature",
        type=float,
        default=0.0,
        help="captioner sampling temperature; default 0 decodes greedily, >0 samples at that temperature",
    )
    parser.add_argument("--captioner_max_tokens", type=int, default=4096, help="captioner max_new_tokens")
    parser.add_argument("--captioner_seed", type=int, default=42, help="captioner RNG seed")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _get_args(argv)
    log.info(f"{args}")

    # Resolve the generate.py output layout from --gen_root.
    ori_image_dir = os.path.join(args.gen_root, ORIG_IMAGE_SUBDIR)
    gen_image_dir = os.path.join(args.gen_root, RECON_SUBDIR)
    mask_dir = os.path.join(args.gen_root, MASK_SUBDIR)
    csv_path = args.csv_path or os.path.join(args.gen_root, _GENERATION_CSV)

    # Load images and masks.
    ori_image_paths = sorted(pl_utils.get_image_paths(ori_image_dir))
    gen_image_paths = sorted(pl_utils.get_image_paths(gen_image_dir))
    mask_paths = sorted(pl_utils.get_image_paths(mask_dir))
    if len(ori_image_paths) != len(mask_paths):
        raise ValueError(
            "The number of original images and masks must be the same. "
            f"Found {len(ori_image_paths)} original images and {len(mask_paths)} masks."
        )
    if len(gen_image_paths) != len(mask_paths):
        raise ValueError(
            "The number of generated images and masks must be the same. "
            f"Found {len(gen_image_paths)} generated images and {len(mask_paths)} masks."
        )

    # Map each generated filename to its anomaly type from the generation manifest.
    # The manifest travels between hosts, so neither column is necessarily operator-authored, and
    # this is the one place it is read: output_filename is reduced to a bare name here, and
    # anomaly_type — a directory name under classification/ below — is rejected unless it is safe.
    anomaly_type_map: Dict[str, str] = {}
    with open(csv_path, "r") as csv_file:
        for row in csv.DictReader(csv_file):
            image_filename = pathlib.Path(str(row["output_filename"]).strip()).name
            anomaly_type_map[image_filename] = validate_anomaly_type(str(row["anomaly_type"]).strip())

    # Create the output dirs.
    # By default: (images, masks, visualization)
    # If args.no_caption is false: (captions, captions_with_meta)
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = pathlib.Path(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    output_mask_dir = pathlib.Path(args.output_dir, "masks")
    os.makedirs(output_mask_dir, exist_ok=True)
    visualization_dir = pathlib.Path(args.output_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    if not args.no_caption:
        caption_dir = pathlib.Path(args.output_dir, "captions")
        os.makedirs(caption_dir, exist_ok=True)
        caption_with_meta_dir = pathlib.Path(args.output_dir, "captions_with_meta")
        os.makedirs(caption_with_meta_dir, exist_ok=True)

    # Main loop.
    #   COCO format reference:
    #   https://cocodataset.org/#format-data
    #   https://www.v7labs.com/blog/coco-dataset-guide
    images_list = []
    annotations_list = []
    categories_list = []
    captions_list = []

    # Process the categories. Sorted so category ids are deterministic and match classes.txt.
    category_to_id = {}
    for i, class_name in enumerate(sorted(set(anomaly_type_map.values())), start=1):
        categories_list.append({"id": i, "name": class_name})
        category_to_id[class_name] = i

    # Process the images and annotations.
    annotation_id = 1
    for image_id, (gen_image_path, ori_image_path, mask_path) in tqdm(
        enumerate(zip(gen_image_paths, ori_image_paths, mask_paths), start=1),
        desc="Processing images and annotations",
        total=len(gen_image_paths),
        dynamic_ncols=True,
    ):
        gen_image_filename = gen_image_path.name
        anomaly_type = anomaly_type_map[gen_image_filename]
        generated_image = Image.open(gen_image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # The COCO record's width/height come from the mask, the bboxes from the generated image.
        require_same_size(generated_image, mask, f"{gen_image_path} + {mask_path}")
        instance_masks = split_mask_into_instances(mask, max_k=int(args.max_instances))
        bboxes = pl_bbox.get_bboxes(instance_masks)

        images_list.append(
            {
                "id": image_id,
                "width": mask.width,
                "height": mask.height,
                "file_name": gen_image_filename,
            }
        )

        # Keep only instances with a non-empty bbox, and emit one COCO annotation each.
        new_instance_masks = []
        for bbox, instance_mask in zip(bboxes, instance_masks):
            if bbox is None:
                continue
            new_instance_masks.append(instance_mask)
            annotations_list.append(
                compute_annotation_dict(
                    annotation_id,
                    image_id,
                    category_to_id[anomaly_type],
                    instance_mask,
                )
            )
            annotation_id += 1
        bboxes = pl_bbox.get_bboxes(new_instance_masks)
        instance_masks = new_instance_masks

        # Visualization.
        visualization_image = pl_utils.visualize(
            generated_image,
            [anomaly_type] * len(bboxes),
            bboxes,
            instance_masks,
        )
        visualization_image.save(pathlib.Path(visualization_dir, gen_image_path.name))

        # Save the generated image and input mask.
        generated_image.save(pathlib.Path(image_dir, gen_image_path.name))
        mask.save(pathlib.Path(output_mask_dir, mask_path.name))

        if not args.no_caption:
            # Prepare the data for captioning.
            full_type = anomaly_type_map[gen_image_filename]
            if "+" in full_type:
                image_type, defect_type = full_type.split("+", maxsplit=1)
            else:
                image_type, defect_type = "unknown", full_type
            caption_bboxes = pl_bbox.get_bboxes(instance_masks, format="xyxy")
            bboxes_str = ", ".join(f"({b[0]}, {b[1]}, {b[2]}, {b[3]})" for b in caption_bboxes)
            meta = {
                "image_type": image_type,
                "anomaly_type": defect_type,
                "bboxes": bboxes_str,
                "num_bboxes": len(caption_bboxes),
            }
            captions_list.append((ori_image_path, mask_path, gen_image_path, meta))

    # COCO JSON.
    coco_output_path = pathlib.Path(args.output_dir, "coco_annotations.json")
    _save_coco_annotations(
        {"annotations": annotations_list, "images": images_list, "categories": categories_list},
        coco_output_path,
    )

    # Classification.
    classification_dir = pathlib.Path(args.output_dir, "classification")
    os.makedirs(classification_dir, exist_ok=True)
    # Save the class names to classes.txt.
    classes_txt_path = pathlib.Path(classification_dir, "classes.txt")
    # Each class directory is checked once, here, and reused below. The copies then join a directory
    # that is already known to be inside classification/ with a basename, which cannot carry a
    # separator — so the destination is contained by construction rather than by a repeated check.
    class_dirs = {"original": _contained_path(classification_dir, "original")}
    with open(classes_txt_path, "w") as f:
        f.write("original\n")  # Add original class.
        os.makedirs(class_dirs["original"], exist_ok=True)
        for i, class_name in enumerate(sorted([item["name"] for item in categories_list])):
            class_dirs[class_name] = _contained_path(classification_dir, class_name)
            os.makedirs(class_dirs[class_name], exist_ok=True)
            f.write(class_name)
            if i != len(categories_list) - 1:
                f.write("\n")
    log.info(f"Class names saved to {classes_txt_path} with {len(categories_list) + 1} classes.")
    # copyfile streams; reading each image whole only to write it back cost memory per image.
    for ori_image_path in ori_image_paths:
        shutil.copyfile(ori_image_path, class_dirs["original"] / os.path.basename(ori_image_path))
    for gen_image_path in gen_image_paths:
        anomaly_type = anomaly_type_map[gen_image_path.name]
        shutil.copyfile(gen_image_path, class_dirs[anomaly_type] / os.path.basename(gen_image_path))
    log.info(f"Images are organized to {classification_dir} with {len(ori_image_paths) + len(gen_image_paths)} images.")

    # Captioning.
    if not args.no_caption:
        with open(args.captioner_prompt_path, "r") as f:
            prompt_data: Dict[str, str] = yaml.safe_load(f)
        if "system_prompt" not in prompt_data or "user_prompt" not in prompt_data:
            raise ValueError(
                "The prompt configuration file must contain 'system_prompt' and 'user_prompt' fields. "
                "See anomalygen/pseudo_label/default_caption_prompt.yaml for an example."
            )
        log.info("Initializing the Cosmos3 reasoner captioner...")
        captioner = pl_caption.Captioner(
            prompt_data=prompt_data,
            temperature=args.captioner_temperature,
            max_new_tokens=int(args.captioner_max_tokens),
            seed=int(args.captioner_seed),
        )
        log.info("Successfully initialized the captioner.")

        # Captioning is per sample and has no resume — the captioner raises for exactly one image
        # at a time (an unusable decode step, a decode failure). Each failure is contained and
        # named here, and the run still exits non-zero below, so a short label set is never mistaken
        # for a complete one.
        failed: list[str] = []
        for ori_image_path, ori_mask_path, gen_image_path, meta in tqdm(
            captions_list,
            desc="Processing captions",
            dynamic_ncols=True,
        ):
            try:
                response = captioner.generate_caption(ori_image_path, ori_mask_path, gen_image_path, meta)
            except Exception as exc:
                failed.append(gen_image_path.name)
                log.error(f"Captioning failed for {gen_image_path.name}, leaving it uncaptioned: {exc}")
                continue
            response, response_with_meta = pl_caption.format_response(response, meta)
            (caption_dir / f"{gen_image_path.stem}.txt").write_text(response)
            (caption_with_meta_dir / f"{gen_image_path.stem}.txt").write_text(response_with_meta)
        written = len(captions_list) - len(failed)
        log.info(f"Captions are saved to {caption_dir} and {caption_with_meta_dir} with {written} captions.")
        if failed:
            raise RuntimeError(
                f"{len(failed)} of {len(captions_list)} images could not be captioned: {', '.join(failed)}. "
                f"Everything else in {args.output_dir} is complete; re-run with --gen_root limited to "
                "those images to fill the gaps."
            )


if __name__ == "__main__":
    main()
