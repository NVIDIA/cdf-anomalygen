# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mask Placer module for Automatic Mask Placement.

This module handles the placement of masks into ROI regions.
It is responsible only for placement logic, not augmentation.
"""

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.auto_mask_placement.data_types import AlignmentPoint


@dataclass
class PlacementResult:
    """Result of a single mask placement attempt"""

    success: bool
    final_mask: Optional[np.ndarray] = None
    pixel_count: int = 0


class MaskPlacer:
    """Handles the placement of masks into ROI regions.

    This class is responsible only for:
    - Calculating alignment positions
    - Placing masks in the correct location
    - Applying ROI clipping
    - Saving debug visualizations

    Augmentation should be done externally before calling place_mask().
    """

    def __init__(
        self, image_width: int, image_height: int, roi_alignment_point: AlignmentPoint = AlignmentPoint.RANDOM
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.roi_alignment_point = roi_alignment_point

    def place_mask(
        self, augmented_mask: np.ndarray, fixed_point: Tuple[int, int], roi_mask: np.ndarray
    ) -> PlacementResult:
        """
        Place an augmented mask into an ROI region.

        Args:
            augmented_mask: The already-augmented mask to place
            fixed_point: The alignment point in the augmented mask
            roi_mask: The ROI mask defining the valid region

        Returns:
            PlacementResult containing success status and final mask
        """
        # Place the mask in full image space
        full_aligned_mask = self._place_in_image_space(augmented_mask, fixed_point, roi_mask)

        # Apply ROI clipping
        final_mask = cv2.bitwise_and(full_aligned_mask, roi_mask)

        # Check if placement was successful
        pixel_count = np.sum(final_mask > 127)
        success = pixel_count > 0

        return PlacementResult(success=success, final_mask=final_mask, pixel_count=pixel_count)

    def calculate_alignment_position(
        self, roi_mask: np.ndarray, mask_h: int, mask_w: int, fixed_point: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate placement position based on ROI alignment point and fixed point"""
        # Get ROI bounding box
        y_coords, x_coords = np.nonzero(roi_mask > 127)
        if len(x_coords) == 0 or len(y_coords) == 0:
            # Fallback to center if no valid coordinates
            return self.image_width // 2 - mask_w // 2, self.image_height // 2 - mask_h // 2

        roi_min_x, roi_max_x = np.min(x_coords), np.max(x_coords)
        roi_min_y, roi_max_y = np.min(y_coords), np.max(y_coords)
        roi_center_x = (roi_min_x + roi_max_x) // 2
        roi_center_y = (roi_min_y + roi_max_y) // 2

        # Calculate the target alignment point in ROI
        target_x, target_y = self._get_target_point(
            roi_min_x, roi_max_x, roi_min_y, roi_max_y, roi_center_x, roi_center_y
        )

        # Calculate final position by offsetting from target point
        final_x = target_x - fixed_point[0]
        final_y = target_y - fixed_point[1]

        return final_x, final_y

    def _get_target_point(
        self, roi_min_x: int, roi_max_x: int, roi_min_y: int, roi_max_y: int, roi_center_x: int, roi_center_y: int
    ) -> Tuple[int, int]:
        """Get target alignment point based on roi_alignment_point"""
        if self.roi_alignment_point == AlignmentPoint.CENTER:
            return roi_center_x, roi_center_y
        elif self.roi_alignment_point == AlignmentPoint.TOP_LEFT:
            return roi_min_x, roi_min_y
        elif self.roi_alignment_point == AlignmentPoint.TOP_RIGHT:
            return roi_max_x, roi_min_y
        elif self.roi_alignment_point == AlignmentPoint.BOTTOM_LEFT:
            return roi_min_x, roi_max_y
        elif self.roi_alignment_point == AlignmentPoint.BOTTOM_RIGHT:
            return roi_max_x, roi_max_y
        elif self.roi_alignment_point == AlignmentPoint.TOP_CENTER:
            return roi_center_x, roi_min_y
        elif self.roi_alignment_point == AlignmentPoint.BOTTOM_CENTER:
            return roi_center_x, roi_max_y
        elif self.roi_alignment_point == AlignmentPoint.LEFT_CENTER:
            return roi_min_x, roi_center_y
        elif self.roi_alignment_point == AlignmentPoint.RIGHT_CENTER:
            return roi_max_x, roi_center_y
        elif self.roi_alignment_point == AlignmentPoint.RANDOM:
            return random.randint(roi_min_x, roi_max_x), random.randint(roi_min_y, roi_max_y)
        else:
            return roi_center_x, roi_center_y

    def _place_in_image_space(
        self, augmented_mask: np.ndarray, fixed_point: Tuple[int, int], roi_mask: np.ndarray
    ) -> np.ndarray:
        """Place the augmented mask in the full image space"""
        full_aligned_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        mask_h_full, mask_w_full = augmented_mask.shape

        # Calculate placement position
        start_x, start_y = self.calculate_alignment_position(roi_mask, mask_h_full, mask_w_full, fixed_point)

        # Target region in full image (may be partially outside)
        target_x_end = start_x + mask_w_full
        target_y_end = start_y + mask_h_full

        # Clamp to image boundaries
        img_x_start = max(0, start_x)
        img_y_start = max(0, start_y)
        img_x_end = min(self.image_width, target_x_end)
        img_y_end = min(self.image_height, target_y_end)

        # Calculate offset in source mask (for negative start positions)
        mask_x_offset = img_x_start - start_x
        mask_y_offset = img_y_start - start_y

        # Calculate dimensions to copy
        copy_width = img_x_end - img_x_start
        copy_height = img_y_end - img_y_start

        # Place the mask if there's a valid region
        if copy_width > 0 and copy_height > 0:
            mask_x_end = mask_x_offset + copy_width
            mask_y_end = mask_y_offset + copy_height

            full_aligned_mask[img_y_start:img_y_end, img_x_start:img_x_end] = augmented_mask[
                mask_y_offset:mask_y_end, mask_x_offset:mask_x_end
            ]

            log.debug(
                f"  Placed mask: target=({start_x},{start_y}), "
                f"image_region=({img_x_start},{img_y_start})-({img_x_end},{img_y_end}), "
                f"size={copy_width}x{copy_height}"
            )

        return full_aligned_mask

    @staticmethod
    def log_mask_info(augmented_mask: np.ndarray, fixed_point: Tuple[int, int]):
        """Log information about the augmented mask"""
        mask_h_full, mask_w_full = augmented_mask.shape

        y_coords_mask, x_coords_mask = np.nonzero(augmented_mask > 127)
        if len(x_coords_mask) > 0 and len(y_coords_mask) > 0:
            mask_min_x = np.min(x_coords_mask)
            mask_max_x = np.max(x_coords_mask)
            mask_min_y = np.min(y_coords_mask)
            mask_max_y = np.max(y_coords_mask)

            mask_w_actual = mask_max_x - mask_min_x + 1
            mask_h_actual = mask_max_y - mask_min_y + 1

            log.info(
                f"  Augmented mask: full={mask_w_full}x{mask_h_full}, white region={mask_w_actual}x{mask_h_actual}, fixed_point={fixed_point}"
            )
        else:
            log.warning(f"  Warning: No white pixels found in augmented mask")

    @staticmethod
    def save_augmented_mask(augmented_mask: np.ndarray, fixed_point: Tuple[int, int], output_path: str):
        """Save augmented mask with fixed point visualization"""
        augmented_mask_vis = augmented_mask.copy()

        fp_x, fp_y = fixed_point
        h, w = augmented_mask_vis.shape

        # Draw horizontal line
        start_x_fp = max(0, fp_x - 5)
        end_x_fp = min(w, fp_x + 6)
        if start_x_fp < end_x_fp and 0 <= fp_y < h:
            augmented_mask_vis[fp_y, start_x_fp:end_x_fp] = 128

        # Draw vertical line
        start_y_fp = max(0, fp_y - 5)
        end_y_fp = min(h, fp_y + 6)
        if start_y_fp < end_y_fp and 0 <= fp_x < w:
            augmented_mask_vis[start_y_fp:end_y_fp, fp_x] = 128

        # Draw center point
        if 0 <= fp_x < w and 0 <= fp_y < h:
            augmented_mask_vis[fp_y, fp_x] = 64

        Image.fromarray(augmented_mask_vis).save(output_path)
        log.info(f"  Saved augmented mask: {output_path}")
