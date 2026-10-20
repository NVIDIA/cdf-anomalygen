# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""THT (through-hole) ROI generation for ``spatial_dependency: cad_tht``.

The delivered annotation is the same artefact shape the ``cad`` route consumes -- a colour-coded
semantic mask plus ``semantic_segmentation_labels.json`` -- but with ``pad`` / ``hole`` / ``pin``
classes that are pixel-disjoint and therefore already the placement regions B / C / D.

Public API:
- ``THTToROIGenerator``: parse a semantic mask into joints, then build one defect's ROI candidates.
- ``THTJoint`` / ``joints_from_components`` / ``roi_for``: the parsed geometry and its ROI rules.
- ``get_ring_candidates`` / ``RingMaskPlacer``: the six ring-confined classes.
- ``get_bridge_candidates`` / ``THTBridgeMaskPlacer``: bridge, which is fitted rather than confined.
  (Prefixed because ``cad2roi`` exports a differently-shaped ``BridgeMaskPlacer``.)
- ``THT_ROI_SPEC`` / ``THT_ROI_INNER`` / ``roi_rule``: the published per-class ROI table.
"""

from anomalygen.auto_mask_placement.tht2roi.core import THTToROIGenerator
from anomalygen.auto_mask_placement.tht2roi.defects.bridge import (
    THTBridgeMaskPlacer,
    bridge_ok,
    fit_bridge,
    get_bridge_candidates,
)
from anomalygen.auto_mask_placement.tht2roi.defects.ring import RingMaskPlacer, get_ring_candidates
from anomalygen.auto_mask_placement.tht2roi.regions import (
    THTJoint,
    joints_carrying,
    joints_from_components,
    roi_for,
)
from anomalygen.auto_mask_placement.tht2roi.spec import (
    BRIDGE,
    RING_CLASSES,
    THT_CLASSES,
    THT_ROI_INNER,
    THT_ROI_SPEC,
    roi_rule,
)

__all__ = [
    "THTToROIGenerator",
    "THTJoint",
    "joints_from_components",
    "roi_for",
    "get_ring_candidates",
    "RingMaskPlacer",
    "get_bridge_candidates",
    "THTBridgeMaskPlacer",
    "bridge_ok",
    "fit_bridge",
    "joints_carrying",
    "THT_ROI_SPEC",
    "THT_ROI_INNER",
    "THT_CLASSES",
    "RING_CLASSES",
    "BRIDGE",
    "roi_rule",
]
