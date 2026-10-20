#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# env_setup_ci_cpu.sh — build a lightweight CPU-only venv that can *import* anomalygen and run the
# unit tests. GPU-only tests (model inference) are auto-skipped by tests/conftest.py's `gpu` marker
# when no CUDA device is present, so no CUDA toolchain / GPU is required.
#
# Unlike scripts/env_setup.sh (which source-compiles flash-attn / transformer-engine / apex /
# natten against CUDA), this installs only CPU-wheel deps:
#   - anomalygen/__init__ skips ConfigStore registration when the training stack is absent, and
#   - roi_generation lazily imports the cradio backbone (apex) only on the GPU path,
# so the package imports without any CUDA-compiled extension.
#
# Run from anywhere:
#   bash scripts/env_setup_ci_cpu.sh [VENV_DIR]        (default: .venv-cpu)

set -euo pipefail

# Navigate to repo root. requirements.txt and assets/ are read as paths relative to root.
cd "$(dirname "$0")/.."
VENV_DIR="${1:-.venv-cpu}"

echo "=== create py3.13 venv at ${VENV_DIR} ==="
# Patch-pinned to match scripts/env_setup.sh — see the note there.
uv venv --python 3.13.16 "${VENV_DIR}"
export VIRTUAL_ENV="$PWD/${VENV_DIR}"

echo "=== torch + torchvision (CPU wheels) ==="
# --no-build on the wheel-only steps: nothing there may run an sdist's setup.py. requirements.txt
# (sam2 from git) and the editable installs are source builds and cannot carry it.
uv pip install --no-build torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu

echo "=== triton (CPU torch doesn't bundle it) ==="
# Matches what the CUDA torch above requires, so CI and the image agree.
uv pip install --no-build triton==3.7.1

echo "=== Python deps ==="
uv pip install -r requirements.txt

echo "=== force headless OpenCV ==="
uv pip uninstall opencv-python 2>/dev/null || true
uv pip install --no-build --reinstall-package opencv-python-headless opencv-python-headless==5.0.0.93

echo "=== cosmos-framework (pinned git commit, --no-deps) ==="
uv pip install -r requirements-nodeps.txt --no-deps
# Import-only stub so cosmos-framework's action-dataset modules import without the
# real `lerobot`.
uv pip install -e assets/lerobot_stub --no-deps
# Stand-in for nltk: the guardrail blocklist's tokenizer and noun lemmatizer (assets/nltk_stub).
uv pip install -e assets/nltk_stub --no-deps

echo "=== anomalygen (editable, --no-deps) ==="
uv pip install -e . --no-deps

echo "=== Python dev deps ==="
uv pip install --no-build -r requirements-dev.txt

echo ""
echo "=== DONE — CI CPU venv at $(realpath "${VENV_DIR}") ==="
