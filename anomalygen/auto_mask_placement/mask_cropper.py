# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mask Cropper module for Automatic Mask Placement.

This module handles mask cropping operations.
"""

import numpy as np


class MaskCropper:
    """Handles mask cropping operations"""

    @staticmethod
    def crop_mask_by_max_dimensions(mask: np.ndarray) -> np.ndarray:
        """Crop mask to its maximum height and width bounding box"""
        # Find bounding box of the mask
        rows, cols = np.nonzero(mask > 127)
        if len(rows) == 0 or len(cols) == 0:
            return mask

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Crop to bounding box
        cropped_mask = mask[min_row : max_row + 1, min_col : max_col + 1]
        return cropped_mask
