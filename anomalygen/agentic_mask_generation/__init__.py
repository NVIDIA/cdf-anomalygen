# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic defect-mask generation — a plugin module for cosmos3-anomalygen.

Given a clean object image + a defect type + a few real examples (AnomalyGen format), an LLM agent
proposes, refines, ranks, and judge-accepts defect *masks* (the "where + what shape" of a defect).
Those masks then feed the inpainting model. The public surface is intentionally small — this module
imports only ``Config``, so the modules that pull SAM2 (``sam2``, ``visual_prompts``) never load with
it.
"""

from anomalygen.agentic_mask_generation.config import Config

__all__ = ["Config"]
