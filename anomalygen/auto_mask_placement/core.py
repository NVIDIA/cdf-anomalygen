# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core module for Automatic Mask Placement.

This module contains the main AutoMaskPlacement class that implements
the workflow for mask placement within ROI regions.
"""

import json
import os
import random
from dataclasses import replace
from typing import List, Tuple

import cv2
import numpy as np
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.auto_mask_placement.config import AugmentationParams
from anomalygen.auto_mask_placement.data_types import ROI, AlignmentPoint, BoundingBox
from anomalygen.auto_mask_placement.mask_augmentor import MaskAugmentor
from anomalygen.auto_mask_placement.mask_cropper import MaskCropper
from anomalygen.auto_mask_placement.mask_placer import MaskPlacer, PlacementResult
from anomalygen.auto_mask_placement.roi_separator import ROISeparator
from anomalygen.auto_mask_placement.utils import validate_binary_mask


class AutoMaskPlacement:
    """Main class implementing the automatic mask placement workflow

    Workflow:
    1. Define ROIs by giving bounding boxes to define legal region and illegal region to present submask
    2. Give a submask image
    3. Give a number N to determine the number of submask should be synthetically added to the ROI later
    4. Use a mask separator to separate different ROIs and store each ROI for later use
    5. Crop the submask by cropping the mask out from the max height and weight
    6. Randomly add augmentation like resize, Shifting, Rotation, Morphological operation, Resize, Flip, Affine transform
    7. Align the cropped submask with a selected ROI center
    8. Dot product with the selected ROI again to make sure that the mask is only presented in the legal region
    9. If N > 1, go to step 6 to generate another one
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        augmentation_params: AugmentationParams = None,
        roi_alignment_point: AlignmentPoint = AlignmentPoint.RANDOM,
        submask_alignment_point: AlignmentPoint = None,
        strict_alignment: bool = False,
        random_seed: int = None,
        max_retry_per_mask: int = 10,
        separate_rois: bool = True,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.roi_separator = ROISeparator(image_width, image_height)
        self.mask_cropper = MaskCropper()
        self.random_seed = random_seed
        self.max_retry_per_mask = max_retry_per_mask
        self.separate_rois = separate_rois
        self.roi_alignment_point = roi_alignment_point

        # Set random seed for reproducible results
        if random_seed is not None:
            self.set_random_seed(random_seed)
            log.info(f"Random seed set to {random_seed} for reproducible results")

        # Handle strict alignment mode
        original_params = augmentation_params or AugmentationParams()
        adjusted_params = self._adjust_params_for_strict_alignment(
            original_params, roi_alignment_point, strict_alignment
        )

        # Create augmentor and placer (no dependency between them)
        self.mask_augmentor = MaskAugmentor(adjusted_params, roi_alignment_point, submask_alignment_point)
        self.mask_placer = MaskPlacer(image_width, image_height, roi_alignment_point)

    def _adjust_params_for_strict_alignment(
        self, original_params: AugmentationParams, roi_alignment_point: AlignmentPoint, strict_alignment: bool
    ) -> AugmentationParams:
        """Adjust augmentation parameters based on strict alignment mode"""
        if not strict_alignment:
            return original_params

        flip_x_prob = original_params.flip_x_probability
        flip_y_prob = original_params.flip_y_probability

        if roi_alignment_point == AlignmentPoint.CENTER:
            pass
        elif roi_alignment_point in [AlignmentPoint.TOP_CENTER, AlignmentPoint.BOTTOM_CENTER]:
            flip_y_prob = 0.0
        elif roi_alignment_point in [AlignmentPoint.LEFT_CENTER, AlignmentPoint.RIGHT_CENTER]:
            flip_x_prob = 0.0
        elif roi_alignment_point in [
            AlignmentPoint.TOP_LEFT,
            AlignmentPoint.TOP_RIGHT,
            AlignmentPoint.BOTTOM_LEFT,
            AlignmentPoint.BOTTOM_RIGHT,
        ]:
            flip_x_prob = 0.0
            flip_y_prob = 0.0
        elif roi_alignment_point == AlignmentPoint.RANDOM:
            flip_x_prob = 0.0
            flip_y_prob = 0.0

        adjusted_params = replace(
            original_params,
            shift_x_probability=0.0,
            shift_y_probability=0.0,
            flip_x_probability=flip_x_prob,
            flip_y_probability=flip_y_prob,
        )

        # Log what's disabled
        disabled = ["shifting"]
        if flip_x_prob <= 0.0 and original_params.flip_x_probability > 0.0:
            disabled.append("X-flipping")
        if flip_y_prob <= 0.0 and original_params.flip_y_probability > 0.0:
            disabled.append("Y-flipping")

        log.info(
            f"Strict alignment mode enabled for ROI alignment '{roi_alignment_point.value}' - {', '.join(disabled)} disabled to prevent boundary clipping"
        )

        return adjusted_params

    def set_random_seed(self, seed: int):
        """Set random seed for both random and numpy.random to ensure reproducible results"""
        random.seed(seed)
        np.random.seed(seed)
        log.info(f"Set random seed to {seed} for reproducible results")

    def add_roi(self, bbox: BoundingBox, is_legal: bool, roi_id: str = None) -> ROI:
        """Add a ROI region (Step 1)"""
        return self.roi_separator.add_roi(bbox, is_legal, roi_id)

    def load_rois_from_json(self, json_path: str):
        """Load ROI regions from JSON file"""
        with open(json_path, "r") as f:
            data = json.load(f)

        legal_count = 0
        illegal_count = 0

        for roi_data in data["rois"]:
            bbox = BoundingBox(x=roi_data["x"], y=roi_data["y"], width=roi_data["width"], height=roi_data["height"])
            is_legal = roi_data["is_legal"]
            self.add_roi(bbox, is_legal, roi_data.get("id"))

            if is_legal:
                legal_count += 1
            else:
                illegal_count += 1

        self.roi_separator.create_separated_roi_masks_from_json()

        if illegal_count > 0:
            log.info(f"Loaded {legal_count} legal ROIs and {illegal_count} illegal ROIs from {json_path}")
        else:
            log.info(f"Loaded {legal_count} legal ROIs from {json_path}")

    def load_combined_rois(
        self,
        json_path: str = None,
        roi_image_paths: List[str] = None,
        illegal_image_paths: List[str] = None,
        min_area: int = 10,
    ):
        """Load ROIs from multiple sources with unified approach"""
        roi_sources_loaded = []

        all_legal_rois = []
        all_legal_masks = []
        combined_illegal_roi = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        # Load JSON ROIs if provided
        if json_path:
            log.info(f"Loading JSON ROIs from: {json_path}")
            self.load_rois_from_json(json_path)

            if hasattr(self.roi_separator, "original_legal_masks") and self.roi_separator.original_legal_masks:
                all_legal_masks.extend(self.roi_separator.original_legal_masks)

            if (
                hasattr(self.roi_separator, "illegal_regions_mask")
                and self.roi_separator.illegal_regions_mask is not None
            ):
                combined_illegal_roi = cv2.bitwise_or(combined_illegal_roi, self.roi_separator.illegal_regions_mask)

            json_legal_count = len([roi for roi in self.roi_separator.rois if roi.is_legal])
            json_illegal_count = len([roi for roi in self.roi_separator.rois if not roi.is_legal])

            if json_illegal_count > 0:
                roi_sources_loaded.append(f"JSON: {json_path} ({json_legal_count} legal, {json_illegal_count} illegal)")
            else:
                roi_sources_loaded.append(f"JSON: {json_path} ({json_legal_count} legal)")

        # Load legal ROI images if provided
        if roi_image_paths:
            log.info(f"Loading legal ROI images: {roi_image_paths}")
            total_image_rois = 0
            for roi_image_path in roi_image_paths:
                if os.path.exists(roi_image_path):
                    roi_img = cv2.imread(roi_image_path, cv2.IMREAD_GRAYSCALE)
                    if roi_img is not None:
                        roi_img = validate_binary_mask(roi_img, f"Legal ROI image: {os.path.basename(roi_image_path)}")
                        roi_img = cv2.resize(roi_img, (self.image_width, self.image_height))

                        if self.separate_rois:
                            separated_masks = self.roi_separator.separate_connected_regions(roi_img, min_area)
                            for mask in separated_masks:
                                all_legal_masks.append(mask)
                                total_image_rois += 1
                            log.info(f"Loaded {len(separated_masks)} legal ROIs from: {roi_image_path} (separated)")
                        else:
                            all_legal_masks.append(roi_img)
                            total_image_rois += 1
                            log.info(f"Loaded 1 legal ROI from: {roi_image_path} (whole image, separation disabled)")
                    else:
                        log.warning(f"Could not load ROI image: {roi_image_path}")
                else:
                    log.warning(f"ROI image not found: {roi_image_path}")

            if len(roi_image_paths) == 1:
                roi_sources_loaded.append(f"Legal images: 1 file ({total_image_rois} ROIs)")
            else:
                roi_sources_loaded.append(f"Legal images: {len(roi_image_paths)} files ({total_image_rois} ROIs)")

        # Load illegal ROI images if provided
        if illegal_image_paths:
            log.info(f"Loading illegal ROI images: {illegal_image_paths}")
            for illegal_image_path in illegal_image_paths:
                if os.path.exists(illegal_image_path):
                    illegal_img = cv2.imread(illegal_image_path, cv2.IMREAD_GRAYSCALE)
                    if illegal_img is not None:
                        illegal_img = validate_binary_mask(
                            illegal_img, f"Illegal ROI image: {os.path.basename(illegal_image_path)}"
                        )
                        illegal_img = cv2.resize(illegal_img, (self.image_width, self.image_height))
                        combined_illegal_roi = cv2.bitwise_or(combined_illegal_roi, illegal_img)
                        log.info(f"Loaded illegal ROI from: {illegal_image_path}")
                    else:
                        log.warning(f"Could not load illegal ROI image: {illegal_image_path}")
                else:
                    log.warning(f"Illegal ROI image not found: {illegal_image_path}")

            roi_sources_loaded.append(f"Illegal images: {len(illegal_image_paths)} files")

        if all_legal_masks:
            self.roi_separator.original_legal_masks = all_legal_masks.copy()

        separated_roi_masks = []
        skipped_count = 0

        if np.any(combined_illegal_roi):
            log.info("Applying illegal ROIs to all legal ROIs...")
            for mask in all_legal_masks:
                final_mask = cv2.bitwise_and(mask, cv2.bitwise_not(combined_illegal_roi))

                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    area = sum(cv2.contourArea(contour) for contour in contours)
                else:
                    area = 0

                if area < min_area:
                    log.info(f"Skipping legal ROI after illegal processing (area: {area})")
                    skipped_count += 1
                    continue

                separated_roi_masks.append(final_mask)
        else:
            separated_roi_masks = all_legal_masks.copy()

        if skipped_count > 0:
            log.info(f"Filtered out {skipped_count} legal ROIs that became too small after illegal ROI processing")

        self.roi_separator.separated_roi_masks = separated_roi_masks
        self.roi_separator.illegal_regions_mask = combined_illegal_roi

        log.info(f"ROI sources loaded: {', '.join(roi_sources_loaded)}")
        log.info(
            f"Total legal ROIs loaded: {len(all_legal_rois)} (JSON) + {len(all_legal_masks) - len(all_legal_rois)} (images)"
        )
        log.info(
            f"Final separated ROI masks: {len(separated_roi_masks)} (after illegal ROI processing and area filtering)"
        )

        if np.any(combined_illegal_roi):
            illegal_pixels = np.sum(combined_illegal_roi > 127)
            log.info(f"Illegal ROIs applied: {illegal_pixels} pixels from JSON and/or images")

        return self.roi_separator

    def process_submask(
        self,
        submask_path: str,
        n_instances: int,
        output_dir: str = "output",
        save_cropped_submask: bool = False,
        save_augmented_masks: bool = False,
        strict_alignment: bool = False,
    ) -> List[dict]:
        """Process submask and generate a single image with N ROI regions containing masks

        Args:
            submask_path: Path to the submask image file
            n_instances: Number of mask instances to generate
            output_dir: Output directory for generated masks
            save_cropped_submask: Whether to save the cropped submask
            save_augmented_masks: Whether to save augmented masks
            strict_alignment: Whether to use strict alignment mode

        Returns:
            List of dictionaries containing output paths and metadata
        """
        # Load and validate submask
        submask = Image.open(submask_path).convert("L")
        original_size = (submask.width, submask.height)
        submask_array = np.array(submask)
        submask_array = validate_binary_mask(submask_array, f"Submask: {os.path.basename(submask_path)}")

        # Resize submask if dimensions don't match (similar to ROI image handling)
        if submask_array.shape != (self.image_height, self.image_width):
            log.info(
                f"Resizing submask from {original_size[0]}x{original_size[1]} to {self.image_width}x{self.image_height}"
            )
            submask_resized = cv2.resize(
                submask_array, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST
            )
            submask_array = validate_binary_mask(submask_resized, f"Resized submask")
        else:
            log.info(f"Submask size matches image size: {self.image_width}x{self.image_height}")

        # Get available ROIs
        available_rois = self.roi_separator.get_separated_roi_masks()
        if not available_rois:
            log.error("Error: No ROI regions available for mask placement")
            return []

        # Crop submask to its bounding box
        cropped_mask = self.mask_cropper.crop_mask_by_max_dimensions(submask_array)

        # Save cropped submask if requested
        if save_cropped_submask:
            cropped_submask_path = os.path.join(output_dir, "cropped_submask.png")
            Image.fromarray(cropped_mask).save(cropped_submask_path)
            log.info(f"Saved cropped submask: {cropped_submask_path}")

        # Select ROI indices for placement
        selected_roi_indices = self._select_roi_indices(n_instances, len(available_rois))
        log.info(f"Selected {n_instances} ROI regions from {len(available_rois)} available regions")

        # Prepare augmented masks directory
        augmented_masks_dir = None
        if save_augmented_masks:
            augmented_masks_dir = os.path.join(output_dir, "augmented_masks")
            os.makedirs(augmented_masks_dir, exist_ok=True)

        # Place masks in selected ROIs
        combined_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        for i in range(n_instances):
            roi_index = selected_roi_indices[i]
            selected_roi_mask = available_rois[roi_index]

            log.info(f"Placing mask {i + 1}/{n_instances} in ROI region {roi_index + 1}")

            result = self._place_mask_with_retry(
                cropped_mask=cropped_mask,
                roi_mask=selected_roi_mask,
                strict_alignment=strict_alignment,
                save_augmented_masks=save_augmented_masks,
                augmented_masks_dir=augmented_masks_dir,
                mask_index=i,
            )

            if result.success:
                combined_mask = cv2.bitwise_or(combined_mask, result.final_mask)

        # Save the combined result
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"auto_placed_mask_with_{n_instances}_rois_seed_{self.random_seed}.png")
        Image.fromarray(combined_mask).save(output_path)

        log.info(f"Generated single image with {n_instances} ROI regions: {output_path}")

        # Return result as list of dictionaries for consistency
        return [{"output_path": output_path, "n_instances": n_instances, "seed": self.random_seed}]

    def _place_mask_with_retry(
        self,
        cropped_mask: np.ndarray,
        roi_mask: np.ndarray,
        strict_alignment: bool,
        save_augmented_masks: bool,
        augmented_masks_dir: str,
        mask_index: int,
    ) -> PlacementResult:
        """Place a mask with augmentation and retry logic"""
        roi_w, roi_h = self._get_roi_dimensions(roi_mask)
        submask_h, submask_w = cropped_mask.shape

        for attempt in range(self.max_retry_per_mask):
            if attempt > 0:
                log.warning(f"  Retry attempt {attempt}/{self.max_retry_per_mask - 1} for mask {mask_index + 1}")

            # Calculate and log ROI-specific augmentation parameters
            roi_specific_params = self._calculate_roi_specific_params(
                roi_mask, cropped_mask, attempt, roi_w, roi_h, submask_w, submask_h
            )

            # Create ROI-specific augmentor and apply augmentation
            roi_specific_augmentor = MaskAugmentor(
                roi_specific_params,
                self.mask_augmentor.roi_alignment_point,
                self.mask_augmentor.submask_alignment_point,
            )
            augmented_mask, fixed_point = roi_specific_augmentor.augment_mask(cropped_mask, strict_alignment)

            # Log augmented mask info
            MaskPlacer.log_mask_info(augmented_mask, fixed_point)

            # Place the augmented mask using MaskPlacer
            result = self.mask_placer.place_mask(augmented_mask, fixed_point, roi_mask)

            # Save augmented mask if requested
            if save_augmented_masks and augmented_masks_dir:
                suffix = f"_attempt_{attempt + 1}" if attempt > 0 else ""
                output_path = os.path.join(augmented_masks_dir, f"augmented_mask_{mask_index + 1}{suffix}.png")
                MaskPlacer.save_augmented_mask(augmented_mask, fixed_point, output_path)

            if result.success:
                if attempt > 0:
                    log.info(
                        f"  Placement successful on attempt {attempt + 1} (final mask has {result.pixel_count} pixels)"
                    )
                else:
                    log.info(f"  Placement successful (final mask has {result.pixel_count} pixels)")
                return result
            else:
                log.warning(f"  Placement failed on attempt {attempt + 1}: mask disappeared after ROI clipping")

        log.error(
            f"  Failed to place mask {mask_index + 1} after {self.max_retry_per_mask} attempts. This ROI will be skipped."
        )
        return PlacementResult(success=False)

    def _get_roi_dimensions(self, roi_mask: np.ndarray) -> Tuple[int, int]:
        """Get ROI width and height from mask"""
        y_coords, x_coords = np.nonzero(roi_mask > 127)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return roi_mask.shape[1], roi_mask.shape[0]

        roi_min_x, roi_max_x = np.min(x_coords), np.max(x_coords)
        roi_min_y, roi_max_y = np.min(y_coords), np.max(y_coords)
        return roi_max_x - roi_min_x + 1, roi_max_y - roi_min_y + 1

    def _calculate_roi_specific_params(
        self,
        roi_mask: np.ndarray,
        cropped_mask: np.ndarray,
        attempt: int,
        roi_w: int,
        roi_h: int,
        submask_w: int,
        submask_h: int,
    ) -> AugmentationParams:
        """Calculate ROI-specific augmentation parameters"""
        # Calculate dynamic shift range if enabled
        if self.mask_augmentor.params.use_dynamic_shift_range:
            shift_x_range, shift_y_range = self.mask_augmentor.calculate_dynamic_shift_range(roi_mask, cropped_mask)
            if attempt == 0:
                log.info(
                    f"  Shift range: x={shift_x_range}, y={shift_y_range} (dynamic, ROI: {roi_w}x{roi_h}, Submask: {submask_w}x{submask_h})"
                )
        else:
            shift_x_range = self.mask_augmentor.params.shift_x_range
            shift_y_range = self.mask_augmentor.params.shift_y_range
            if attempt == 0:
                log.info(f"  Shift range: x={shift_x_range}, y={shift_y_range} (fixed)")

        # Calculate dynamic rotation range if enabled
        if self.mask_augmentor.params.use_dynamic_rotation_range:
            rotation_range = self.mask_augmentor.calculate_dynamic_rotation_range(roi_mask, cropped_mask)
            if attempt == 0:
                log.info(
                    f"  Rotation range: {rotation_range}° (dynamic, ROI: {roi_w}x{roi_h}, Submask: {submask_w}x{submask_h})"
                )
        else:
            rotation_range = self.mask_augmentor.params.rotation_range
            if attempt == 0:
                log.info(f"  Rotation range: {rotation_range}° (fixed)")

        # Calculate dynamic shear range if enabled
        if self.mask_augmentor.params.use_dynamic_shear_range:
            shear_x_range, shear_y_range = self.mask_augmentor.calculate_dynamic_shear_range(roi_mask, cropped_mask)
            if attempt == 0:
                log.info(
                    f"  Shear range: x={shear_x_range}°, y={shear_y_range}° (dynamic, ROI: {roi_w}x{roi_h}, Submask: {submask_w}x{submask_h})"
                )
        else:
            shear_x_range = self.mask_augmentor.params.shear_x_range
            shear_y_range = self.mask_augmentor.params.shear_y_range
            if attempt == 0:
                log.info(f"  Shear range: x={shear_x_range}°, y={shear_y_range}° (fixed)")

        return replace(
            self.mask_augmentor.params,
            shift_x_range=shift_x_range,
            shift_y_range=shift_y_range,
            rotation_range=rotation_range,
            shear_x_range=shear_x_range,
            shear_y_range=shear_y_range,
        )

    def _select_roi_indices(self, n_instances: int, n_available: int) -> List[int]:
        """Select ROI indices for mask placement"""
        if n_instances <= n_available:
            return random.sample(range(n_available), n_instances)
        else:
            indices = list(range(n_available))
            remaining = n_instances - n_available
            for _ in range(remaining):
                indices.append(random.randint(0, n_available - 1))
            return indices

    def visualize_rois(self, output_path: str = "roi_visualization.png"):
        """Create a visualization of all ROI regions"""
        vis_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        separated_masks = self.roi_separator.get_separated_roi_masks()

        for i, mask in enumerate(separated_masks):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

            y_coords, x_coords = np.nonzero(mask > 127)
            if len(x_coords) > 0 and len(y_coords) > 0:
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))

                label = f"ROI {i + 1}"
                min_side = min(self.image_width, self.image_height)
                font_scale = min_side / 10.0 / 100.0
                thickness = max(1, int(min_side / 200))
                cv2.putText(
                    vis_image, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness
                )

        cv2.imwrite(output_path, vis_image)
        log.info(f"ROI visualization saved to: {output_path}")

    def create_roi_binary_images(self, output_dir: str = "roi_binaries"):
        """Create binary images from actual processed ROI data for visualization"""
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.roi_separator, "original_legal_masks") and self.roi_separator.original_legal_masks is not None:
            all_legal_masks = self.roi_separator.original_legal_masks
            legal_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            for mask in all_legal_masks:
                legal_mask = cv2.bitwise_or(legal_mask, mask)
        else:
            raise ValueError("Original legal masks not found")

        if hasattr(self.roi_separator, "illegal_regions_mask") and self.roi_separator.illegal_regions_mask is not None:
            illegal_mask = self.roi_separator.illegal_regions_mask
        else:
            illegal_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        combined_mask = legal_mask.copy()
        if np.any(illegal_mask > 127):
            combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(illegal_mask))

        Image.fromarray(legal_mask).save(os.path.join(output_dir, "legal_rois_binary.png"))
        Image.fromarray(illegal_mask).save(os.path.join(output_dir, "illegal_rois_binary.png"))
        Image.fromarray(combined_mask).save(os.path.join(output_dir, "combined_rois_binary.png"))

        log.info(f"ROI binary images saved to '{output_dir}':")
        log.info(f"  - legal_rois_binary.png (white = legal ROIs only)")
        log.info(f"  - illegal_rois_binary.png (white = illegal ROIs only)")
        log.info(f"  - combined_rois_binary.png (white = legal ROIs with illegal holes)")

    def save_separated_roi_masks(self, output_dir: str = "separated_roi_masks"):
        """Save separated ROI masks created by CV algorithms"""
        os.makedirs(output_dir, exist_ok=True)

        separated_masks = self.roi_separator.get_separated_roi_masks()

        for i, mask in enumerate(separated_masks):
            output_path = os.path.join(output_dir, f"separated_roi_{i + 1:03d}.png")
            Image.fromarray(mask).save(output_path)

        log.info(f"Saved {len(separated_masks)} separated ROI masks to '{output_dir}'")
        return separated_masks
