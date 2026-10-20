# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-path discovery and mask/bbox overlay visualization for pseudo-labeling."""

from __future__ import annotations

import pathlib
from typing import List, Union

import numpy as np
from PIL import Image, ImageDraw

_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def get_image_paths(image_dir: Union[str, List[str]]) -> List[pathlib.Path]:
    """Recursively collect image paths (by extension) under one or more directories."""
    if isinstance(image_dir, (str, pathlib.Path)):
        image_dir = [image_dir]
    if not isinstance(image_dir, (tuple, list)):
        raise ValueError("image_dir should be a string or a list of strings.")
    image_dir = sorted(image_dir)
    image_paths = []
    for d in image_dir:
        image_paths.extend([p for p in pathlib.Path(d).rglob("**/*") if p.suffix.lower() in _IMG_EXTENSIONS])
    return image_paths


def visualize(
    image: Image.Image,
    classes,
    bboxes,
    instance_masks: List[Image.Image],
    mask_alpha: float = 0.6,
) -> Image.Image:
    """Overlay each instance mask (tinted), its bbox (red), and its class label onto ``image``."""

    def textsize(text, font):
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    for class_name, bbox, mask in zip(classes, bboxes, instance_masks):
        # Mask.
        mask_array = np.array(mask)
        color = np.array([30 / 255, 144 / 255, 1.0, mask_alpha])
        mask_array = np.expand_dims(mask_array, axis=-1) * color.reshape(1, 1, -1)
        mask = Image.fromarray((mask_array).astype(np.uint8), mode="RGBA")
        alpha_channel = mask.split()[-1]
        image.paste(mask, mask=alpha_channel)
        # Bbox.
        if bbox is not None:
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        else:
            # No bbox for this instance: anchor the label at the top-left instead
            # of crashing (x/y are undefined on the first iteration) or silently
            # reusing the previous instance's coordinates.
            x, y = 0, 0
        # Class name.
        if class_name is not None:
            _, text_height = textsize(class_name, font=None)
            draw.text((x, y - text_height), class_name, fill="white")
    return image
