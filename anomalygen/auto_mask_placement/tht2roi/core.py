# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""THT ROI generation: parse a semantic mask into joints, then build one defect's ROI candidates."""

from anomalygen.auto_mask_placement.cad2roi.parser import CADParser
from anomalygen.auto_mask_placement.tht2roi.defects.bridge import get_bridge_candidates
from anomalygen.auto_mask_placement.tht2roi.defects.ring import get_ring_candidates
from anomalygen.auto_mask_placement.tht2roi.regions import joints_from_components
from anomalygen.auto_mask_placement.tht2roi.spec import BRIDGE, THT_CLASSES


class THTToROIGenerator:
    """Generate THT ROI candidates from a colour-coded semantic mask.

    The parser is ``CADParser`` unchanged: the THT annotation is the same kind of artefact the `cad`
    route already consumes -- a colour-coded PNG plus a label JSON -- and it decodes to one component
    per ``pad`` / ``hole`` / ``pin`` class with no THT-specific handling. Only the layer above it,
    which turns components into regions and regions into ROIs, is new.

    Unlike ``CADToROIGenerator`` this does not compute every defect's candidates and discard the rest;
    the caller names the one defect it is placing.
    """

    DEFECT_TYPES = THT_CLASSES

    def __init__(self, label_path: str, min_area_abs: int = 5, fragment_ratio: float = 0.1):
        self.parser = CADParser(label_path, min_area_abs=min_area_abs, fragment_ratio=fragment_ratio)

    def parse_joints(self, mask_path: str):
        """``(joints, (H, W))`` for one semantic mask."""
        components, img_shape = self.parser.parse(mask_path)
        return joints_from_components(components, img_shape), img_shape

    def generate_candidates(self, joints, tht_defect: str):
        """ROI candidates for one defect class.

        For ``bridge`` these are the ROI OVERLAY only, not a confinement: a bridge must span pad to
        crop edge, so clipping it to a region would guarantee it never reaches the edge. Its placer
        takes the joints themselves and tests against the pad ring and the frame.
        """
        if tht_defect == BRIDGE:
            return get_bridge_candidates(joints)
        return get_ring_candidates(joints, tht_defect)
