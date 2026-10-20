# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mask Augmentor module for Automatic Mask Placement.

This module handles mask augmentation operations including:
- Shifting
- Scaling
- Rotation
- Flipping
- Shearing
- Morphological operations
"""

import random
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import shift

from anomalygen.auto_mask_placement.config import AugmentationParams
from anomalygen.auto_mask_placement.data_types import AlignmentPoint


class MaskAugmentor:
    """Handles mask augmentation operations"""

    def __init__(
        self,
        params: AugmentationParams,
        roi_alignment_point: AlignmentPoint = AlignmentPoint.RANDOM,
        submask_alignment_point: AlignmentPoint = None,
    ):
        self.params = params
        self.roi_alignment_point = roi_alignment_point
        # Use separate alignment mode for submask augmentation, default to same as ROI alignment mode
        self.submask_alignment_point = submask_alignment_point or roi_alignment_point

    def get_submask_fixed_point(self, h: int, w: int) -> Tuple[int, int]:
        """Get the fixed point for submask augmentation based on submask alignment mode"""
        if self.submask_alignment_point == AlignmentPoint.CENTER:
            return (w // 2, h // 2)
        elif self.submask_alignment_point == AlignmentPoint.TOP_LEFT:
            return (0, 0)
        elif self.submask_alignment_point == AlignmentPoint.TOP_RIGHT:
            return (w - 1, 0)
        elif self.submask_alignment_point == AlignmentPoint.BOTTOM_LEFT:
            return (0, h - 1)
        elif self.submask_alignment_point == AlignmentPoint.BOTTOM_RIGHT:
            return (w - 1, h - 1)
        elif self.submask_alignment_point == AlignmentPoint.TOP_CENTER:
            return (w // 2, 0)
        elif self.submask_alignment_point == AlignmentPoint.BOTTOM_CENTER:
            return (w // 2, h - 1)
        elif self.submask_alignment_point == AlignmentPoint.LEFT_CENTER:
            return (0, h // 2)
        elif self.submask_alignment_point == AlignmentPoint.RIGHT_CENTER:
            return (w - 1, h // 2)
        else:  # RANDOM or default
            return (w // 2, h // 2)

    def _get_roi_alignment_position(self, roi_w: int, roi_h: int) -> tuple:
        """Get ROI alignment position based on roi_alignment_point"""
        if self.roi_alignment_point == AlignmentPoint.CENTER:
            return roi_w / 2, roi_h / 2
        elif self.roi_alignment_point == AlignmentPoint.TOP_LEFT:
            return 0, 0
        elif self.roi_alignment_point == AlignmentPoint.TOP_RIGHT:
            return roi_w, 0
        elif self.roi_alignment_point == AlignmentPoint.BOTTOM_LEFT:
            return 0, roi_h
        elif self.roi_alignment_point == AlignmentPoint.BOTTOM_RIGHT:
            return roi_w, roi_h
        elif self.roi_alignment_point == AlignmentPoint.TOP_CENTER:
            return roi_w / 2, 0
        elif self.roi_alignment_point == AlignmentPoint.BOTTOM_CENTER:
            return roi_w / 2, roi_h
        elif self.roi_alignment_point == AlignmentPoint.LEFT_CENTER:
            return 0, roi_h / 2
        elif self.roi_alignment_point == AlignmentPoint.RIGHT_CENTER:
            return roi_w, roi_h / 2
        else:
            return roi_w / 2, roi_h / 2

    def augment_mask(self, mask: np.ndarray, strict_alignment: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Apply random augmentations to the mask and return fixed point position"""
        augmented_mask = mask.copy()
        h, w = mask.shape
        fixed_point = self.get_submask_fixed_point(h, w)  # Get initial submask fixed point

        # Random shifting - separate X and Y probabilities
        # Use individual probabilities if specified, otherwise fall back to combined probability
        shift_x_prob = self.params.shift_x_probability
        shift_y_prob = self.params.shift_y_probability

        apply_shift_x = random.random() < shift_x_prob
        apply_shift_y = random.random() < shift_y_prob

        if apply_shift_x or apply_shift_y:
            shift_x = random.randint(self.params.shift_x_range[0], self.params.shift_x_range[1]) if apply_shift_x else 0
            shift_y = random.randint(self.params.shift_y_range[0], self.params.shift_y_range[1]) if apply_shift_y else 0
            # Calculate required padding to avoid cropping - only pad where needed
            pad_top = max(0, -shift_y)  # Pad top if shifting up
            pad_bottom = max(0, shift_y)  # Pad bottom if shifting down
            pad_left = max(0, -shift_x)  # Pad left if shifting left
            pad_right = max(0, shift_x)  # Pad right if shifting right

            # Pad the array only where needed
            padded = np.pad(
                augmented_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0
            )
            # Apply shift to padded array
            augmented_mask = shift(padded, shift=(shift_y, shift_x), mode="constant", cval=0)
            # Update fixed point position to account for padding and shifting
            fixed_point = (fixed_point[0] - min(0, shift_x), fixed_point[1] - min(0, shift_y))

        # Random scaling (resize) - separate X and Y probabilities
        scale_x_prob = self.params.scale_x_probability
        scale_y_prob = self.params.scale_y_probability

        apply_scale_x = random.random() < scale_x_prob
        apply_scale_y = random.random() < scale_y_prob

        if apply_scale_x or apply_scale_y:
            h, w = augmented_mask.shape

            if self.params.scale_fixed_ratio:
                # Preserve aspect ratio - use single scale factor (only if both X and Y scaling are enabled)
                if apply_scale_x and apply_scale_y:
                    scale = random.uniform(
                        max(self.params.scale_x_range[0], self.params.scale_y_range[0]),
                        min(self.params.scale_x_range[1], self.params.scale_y_range[1]),
                    )
                    new_w, new_h = int(w * scale), int(h * scale)
                elif apply_scale_x:
                    scale_x = random.uniform(self.params.scale_x_range[0], self.params.scale_x_range[1])
                    new_w, new_h = int(w * scale_x), h
                else:  # apply_scale_y
                    scale_y = random.uniform(self.params.scale_y_range[0], self.params.scale_y_range[1])
                    new_w, new_h = w, int(h * scale_y)
            else:
                # Separate X and Y scaling
                scale_x = (
                    random.uniform(self.params.scale_x_range[0], self.params.scale_x_range[1]) if apply_scale_x else 1.0
                )
                scale_y = (
                    random.uniform(self.params.scale_y_range[0], self.params.scale_y_range[1]) if apply_scale_y else 1.0
                )
                new_w, new_h = int(w * scale_x), int(h * scale_y)

            if new_h > 0 and new_w > 0:
                augmented_mask = cv2.resize(augmented_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                # Update fixed point position based on scaling
                scale_factor_x = new_w / w
                scale_factor_y = new_h / h
                fixed_point = (int(fixed_point[0] * scale_factor_x), int(fixed_point[1] * scale_factor_y))

        # Random flip - independent X and Y
        h, w = augmented_mask.shape

        # Horizontal flip (X direction)
        if random.random() < self.params.flip_x_probability:
            augmented_mask = np.fliplr(augmented_mask)
            if strict_alignment:
                fixed_point = (w - 1 - fixed_point[0], fixed_point[1])

        # Vertical flip (Y direction)
        if random.random() < self.params.flip_y_probability:
            augmented_mask = np.flipud(augmented_mask)
            if strict_alignment:
                fixed_point = (fixed_point[0], h - 1 - fixed_point[1])

        # Random rotation (fixed at alignment point)
        if random.random() < self.params.rotation_probability:
            h, w = augmented_mask.shape

            angle = random.uniform(self.params.rotation_range[0], self.params.rotation_range[1])
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)

            translate_to_fixed = np.array(
                [[1, 0, -fixed_point[0]], [0, 1, -fixed_point[1]], [0, 0, 1]], dtype=np.float32
            )

            translate_back = np.array([[1, 0, fixed_point[0]], [0, 1, fixed_point[1]], [0, 0, 1]], dtype=np.float32)

            final_matrix = np.dot(translate_back, np.dot(rotation_matrix, translate_to_fixed))

            h, w = augmented_mask.shape
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T

            transformed_corners = final_matrix @ corners
            transformed_corners = transformed_corners[:2, :]

            min_x = int(np.floor(np.min(transformed_corners[0])))
            max_x = int(np.ceil(np.max(transformed_corners[0])))
            min_y = int(np.floor(np.min(transformed_corners[1])))
            max_y = int(np.ceil(np.max(transformed_corners[1])))

            new_w = max_x - min_x
            new_h = max_y - min_y

            adjust_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

            adjusted_affine_matrix = (adjust_matrix @ final_matrix)[:2, :]

            augmented_mask = cv2.warpAffine(
                augmented_mask,
                adjusted_affine_matrix,
                (new_w, new_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            fixed_point = (fixed_point[0] - min_x, fixed_point[1] - min_y)

        # Random shear (fixed at alignment point) - separate X and Y probabilities
        shear_x_prob = self.params.shear_x_probability
        shear_y_prob = self.params.shear_y_probability

        apply_shear_x = random.random() < shear_x_prob
        apply_shear_y = random.random() < shear_y_prob

        if apply_shear_x or apply_shear_y:
            h, w = augmented_mask.shape

            shear_angle_x = (
                random.uniform(self.params.shear_x_range[0], self.params.shear_x_range[1]) if apply_shear_x else 0.0
            )
            shear_angle_y = (
                random.uniform(self.params.shear_y_range[0], self.params.shear_y_range[1]) if apply_shear_y else 0.0
            )

            shear_x = np.tan(np.radians(shear_angle_x))
            shear_y = np.tan(np.radians(shear_angle_y))

            shear_matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float32)

            translate_to_fixed = np.array(
                [[1, 0, -fixed_point[0]], [0, 1, -fixed_point[1]], [0, 0, 1]], dtype=np.float32
            )

            translate_back = np.array([[1, 0, fixed_point[0]], [0, 1, fixed_point[1]], [0, 0, 1]], dtype=np.float32)

            final_matrix = np.dot(translate_back, np.dot(shear_matrix, translate_to_fixed))

            h, w = augmented_mask.shape
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T

            transformed_corners = final_matrix @ corners
            transformed_corners = transformed_corners[:2, :]

            min_x = int(np.floor(np.min(transformed_corners[0])))
            max_x = int(np.ceil(np.max(transformed_corners[0])))
            min_y = int(np.floor(np.min(transformed_corners[1])))
            max_y = int(np.ceil(np.max(transformed_corners[1])))

            new_w = max_x - min_x
            new_h = max_y - min_y

            adjust_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

            adjusted_affine_matrix = (adjust_matrix @ final_matrix)[:2, :]

            augmented_mask = cv2.warpAffine(
                augmented_mask,
                adjusted_affine_matrix,
                (new_w, new_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            fixed_point = (fixed_point[0] - min_x, fixed_point[1] - min_y)

        # Random morphological operation
        if random.random() < self.params.morph_probability:
            morph_op = random.choice(self.params.morph_operations)
            kernel = np.ones((self.params.kernel_size, self.params.kernel_size), np.uint8)

            if morph_op == "dilate":
                augmented_mask = cv2.dilate(augmented_mask, kernel, iterations=1)
            elif morph_op == "erode":
                augmented_mask = cv2.erode(augmented_mask, kernel, iterations=1)
            elif morph_op == "open":
                augmented_mask = cv2.morphologyEx(augmented_mask, cv2.MORPH_OPEN, kernel)
            elif morph_op == "close":
                augmented_mask = cv2.morphologyEx(augmented_mask, cv2.MORPH_CLOSE, kernel)

        # Ensure binary mask
        augmented_mask = (augmented_mask > 127).astype(np.uint8) * 255

        return augmented_mask, fixed_point

    def calculate_dynamic_shift_range(
        self, roi_mask: np.ndarray, submask: np.ndarray, buffer_factor: float = 1.0
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate dynamic shift range considering alignment points and ROI boundaries"""
        y_coords, x_coords = np.nonzero(roi_mask > 127)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return ((-1, 1), (-1, 1))

        roi_min_x, roi_max_x = np.min(x_coords), np.max(x_coords)
        roi_min_y, roi_max_y = np.min(y_coords), np.max(y_coords)
        roi_w = roi_max_x - roi_min_x + 1
        roi_h = roi_max_y - roi_min_y + 1

        submask_h, submask_w = submask.shape

        roi_target_x, roi_target_y = self._get_roi_alignment_position(roi_w, roi_h)
        submask_ref_x, submask_ref_y = self.get_submask_fixed_point(submask_h, submask_w)

        aligned_start_x = roi_target_x - submask_ref_x
        aligned_start_y = roi_target_y - submask_ref_y
        aligned_end_x = aligned_start_x + submask_w
        aligned_end_y = aligned_start_y + submask_h

        max_shift_left = max(0, int(aligned_start_x * buffer_factor))
        max_shift_right = max(0, int((roi_w - aligned_end_x) * buffer_factor))
        max_shift_up = max(0, int(aligned_start_y * buffer_factor))
        max_shift_down = max(0, int((roi_h - aligned_end_y) * buffer_factor))

        max_shift_left = max(1, max_shift_left)
        max_shift_right = max(1, max_shift_right)
        max_shift_up = max(1, max_shift_up)
        max_shift_down = max(1, max_shift_down)

        shift_x_range = (-max_shift_left, max_shift_right)
        shift_y_range = (-max_shift_up, max_shift_down)

        return (shift_x_range, shift_y_range)

    def calculate_dynamic_rotation_range(self, roi_mask: np.ndarray, submask: np.ndarray) -> Tuple[float, float]:
        """Calculate dynamic rotation range considering alignment-based overflow"""
        y_coords, x_coords = np.nonzero(roi_mask > 127)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return (-15.0, 15.0)

        roi_min_x, roi_max_x = np.min(x_coords), np.max(x_coords)
        roi_min_y, roi_max_y = np.min(y_coords), np.max(y_coords)
        roi_w = roi_max_x - roi_min_x + 1
        roi_h = roi_max_y - roi_min_y + 1

        submask_h, submask_w = submask.shape

        max_allowed_overflow_w = submask_w * 0.2
        max_allowed_overflow_h = submask_h * 0.2

        roi_target_x, roi_target_y = self._get_roi_alignment_position(roi_w, roi_h)
        submask_ref_x, submask_ref_y = self.get_submask_fixed_point(submask_h, submask_w)

        space_left = roi_target_x + max_allowed_overflow_w
        space_right = roi_w - roi_target_x + max_allowed_overflow_w
        space_top = roi_target_y + max_allowed_overflow_h
        space_bottom = roi_h - roi_target_y + max_allowed_overflow_h

        corners = [
            (0 - submask_ref_x, 0 - submask_ref_y),
            (submask_w - submask_ref_x, 0 - submask_ref_y),
            (submask_w - submask_ref_x, submask_h - submask_ref_y),
            (0 - submask_ref_x, submask_h - submask_ref_y),
        ]

        max_positive_angle = self._find_max_angle_simple(
            corners, space_left, space_right, space_top, space_bottom, positive=True
        )
        max_negative_angle = self._find_max_angle_simple(
            corners, space_left, space_right, space_top, space_bottom, positive=False
        )

        return (-max_negative_angle, max_positive_angle)

    def _find_max_angle_simple(
        self,
        corners: list,
        space_left: float,
        space_right: float,
        space_top: float,
        space_bottom: float,
        positive: bool,
    ) -> float:
        """Find maximum rotation angle using simple rotation formula"""
        max_angle = 45.0

        for corner_x, corner_y in corners:
            for test_angle_deg in range(5, 50, 5):
                theta = np.radians(test_angle_deg if positive else -test_angle_deg)

                rotated_x = corner_x * np.cos(theta) - corner_y * np.sin(theta)
                rotated_y = corner_x * np.sin(theta) + corner_y * np.cos(theta)

                if (
                    rotated_x < -space_left
                    or rotated_x > space_right
                    or rotated_y < -space_top
                    or rotated_y > space_bottom
                ):
                    max_angle = min(max_angle, max(5.0, test_angle_deg - 5))
                    break

        return max_angle

    def calculate_dynamic_shear_range(
        self, roi_mask: np.ndarray, submask: np.ndarray
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate dynamic shear range considering alignment-based overflow"""
        y_coords, x_coords = np.nonzero(roi_mask > 127)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return ((-5.0, 5.0), (-5.0, 5.0))

        roi_min_x, roi_max_x = np.min(x_coords), np.max(x_coords)
        roi_min_y, roi_max_y = np.min(y_coords), np.max(y_coords)
        roi_w = roi_max_x - roi_min_x + 1
        roi_h = roi_max_y - roi_min_y + 1

        submask_h, submask_w = submask.shape

        max_allowed_overflow_w = submask_w * 0.2
        max_allowed_overflow_h = submask_h * 0.2

        roi_target_x, roi_target_y = self._get_roi_alignment_position(roi_w, roi_h)
        submask_ref_x, submask_ref_y = self.get_submask_fixed_point(submask_h, submask_w)

        space_left = roi_target_x + max_allowed_overflow_w
        space_right = roi_w - roi_target_x + max_allowed_overflow_w
        space_top = roi_target_y + max_allowed_overflow_h
        space_bottom = roi_h - roi_target_y + max_allowed_overflow_h

        corners = [
            (0 - submask_ref_x, 0 - submask_ref_y),
            (submask_w - submask_ref_x, 0 - submask_ref_y),
            (submask_w - submask_ref_x, submask_h - submask_ref_y),
            (0 - submask_ref_x, submask_h - submask_ref_y),
        ]

        max_positive_shear_x = self._find_max_shear_simple(corners, space_left, space_right, axis="x", positive=True)
        max_negative_shear_x = self._find_max_shear_simple(corners, space_left, space_right, axis="x", positive=False)

        max_positive_shear_y = self._find_max_shear_simple(corners, space_top, space_bottom, axis="y", positive=True)
        max_negative_shear_y = self._find_max_shear_simple(corners, space_top, space_bottom, axis="y", positive=False)

        shear_x_range = (-max_negative_shear_x, max_positive_shear_x)
        shear_y_range = (-max_negative_shear_y, max_positive_shear_y)

        return (shear_x_range, shear_y_range)

    def _find_max_shear_simple(
        self, corners: list, space_min: float, space_max: float, axis: str, positive: bool
    ) -> float:
        """Find maximum shear angle in degrees for specified axis"""
        max_shear_angle = 30.0

        for corner_x, corner_y in corners:
            for test_angle_deg in range(2, 31, 2):
                angle_rad = np.radians(test_angle_deg if positive else -test_angle_deg)
                shear_factor = np.tan(angle_rad)

                if axis == "x":
                    transformed_coord = corner_x + shear_factor * corner_y
                else:
                    transformed_coord = corner_y + shear_factor * corner_x

                if transformed_coord < -space_min or transformed_coord > space_max:
                    max_shear_angle = min(max_shear_angle, max(2.0, test_angle_deg - 2))
                    break

        return max_shear_angle
