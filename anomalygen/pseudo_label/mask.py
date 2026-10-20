# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""COCO RLE (de)serialization for pseudo-labeling.

Instance splitting is not defined here — it reuses
:func:`anomalygen.inference.iterative.split_mask_into_instances`, the same splitter the generation
pipeline uses, so the annotated instances match the ones that were generated.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pycocotools import mask as mask_utils


def binary_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Convert a binary mask to uncompressed RLE (column-major run lengths).

    Reference: https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
    """
    if binary_mask.dtype != np.bool_:
        raise ValueError("Input binary_mask must be of type np.bool_")
    if len(binary_mask.shape) != 2:
        raise ValueError("Input binary_mask must be a 2D array")

    rle = {"counts": [], "size": list(binary_mask.shape)}
    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.nonzero(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))
    # Note that the odd counts are always the numbers of zeros.
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))
    rle["counts"] = lengths.tolist()
    return rle


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    """Compress an uncompressed RLE into COCO's byte-encoded form (``counts`` as a UTF-8 str)."""
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json.
    return rle


def coco_decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Decode a COCO RLE back into a contiguous binary mask array."""
    return np.ascontiguousarray(mask_utils.decode(rle))
