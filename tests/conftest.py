# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration.

Provides the ``gpu`` marker: tag any test that needs a CUDA GPU with
``@pytest.mark.gpu`` and it is skipped automatically when no CUDA device is
available — so the suite stays green on CPU-only / no-dedicated-GPU runners.

Also provides ``loguru_lines``, for asserting on what a module logged, and restores autograd around
every test (importing an inference entry point turns it off process-wide).
"""

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a CUDA GPU; skipped when no CUDA device is available")


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires a CUDA GPU (no CUDA device available)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _autograd_enabled():
    """Run every test with autograd on, whatever a previously imported module did to it.

    The framework's ``init_script`` disables autograd process-wide. The pipeline scripts guard that
    call behind ``__main__`` so importing them is inert, but pytest imports every test module during
    collection, so one unguarded call anywhere would silently turn gradients off session-wide.
    """
    import torch

    was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        yield
    finally:
        torch.set_grad_enabled(was_enabled)


@pytest.fixture
def loguru_lines():
    """Collect what the code under test logs, as a list of rendered lines.

    The project logs through cosmos_framework's loguru wrapper, whose sink is bound at import and
    bypasses both caplog (stdlib logging) and capfd. Attaching a sink reads the real output instead
    of replacing the module's logging functions.
    """
    from cosmos_framework.utils import log

    lines = []
    sink_id = log.logger.add(lambda m: lines.append(str(m)), level="INFO")
    yield lines
    log.logger.remove(sink_id)
