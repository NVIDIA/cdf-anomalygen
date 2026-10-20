# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I2I anomaly-image-generation plugin built on ``cosmos_framework``.

Importing the package registers the reusable library groups (model group, ckpt_type,
callbacks) — when the training stack is available.
"""

from anomalygen import transformers_patch  # noqa: F401  (must precede the framework imports below)

try:
    from anomalygen import register  # noqa: F401  (registration side effects)
except ImportError:
    # Registration pulls the cosmos_framework training stack (transformer_engine / megatron / apex / …),
    # which isn't installed in lightweight CPU-only environments (e.g. unit-test CI). It's only needed
    # for the training / inference launch path, so degrade gracefully when those deps are absent.
    import logging

    logging.getLogger(__name__).debug("anomalygen registration skipped: training deps unavailable")
