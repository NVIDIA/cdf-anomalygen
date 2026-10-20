#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# env_setup.sh — build the py3.13 + cu132 venv that runs both
# cdf-anomalygen and cosmos-framework inference.
#
# Run from anywhere:
#   bash scripts/env_setup.sh [MAX_JOBS] [NVCC_THREADS]
#
# Build parallelism for the source-compiled extensions
#   MAX_JOBS      parallel compile jobs           (default 16)
#   NVCC_THREADS  threads per nvcc invocation      (default 1)
# Examples:
#   bash scripts/env_setup.sh 8 2
#   MAX_JOBS=8 bash scripts/env_setup.sh

set -euo pipefail

MAX_JOBS="${1:-${MAX_JOBS:-16}}"
NVCC_THREADS="${2:-${NVCC_THREADS:-1}}"
export MAX_JOBS NVCC_THREADS
echo "Build parallelism: MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS}"

# Navigate to repo root. requirements.txt and assets/ are read as paths relative to root.
cd "$(dirname "$0")/.."
[[ -f requirements.txt ]] || { echo "ERROR: requirements.txt not found in $(pwd)" >&2; exit 1; }
[[ -f requirements-nodeps.txt ]] || { echo "ERROR: requirements-nodeps.txt not found in $(pwd)" >&2; exit 1; }

echo "=== Stage 1/15: create py3.13 venv ==="
# Patch-pinned, not bare 3.13: 3.13.15 and earlier carry known CVEs.
PYTHON_VERSION=3.13.16
uv venv --clear --python "${PYTHON_VERSION}"
source .venv/bin/activate
python --version

# X.Y form for the site-packages paths below — site-packages lives under
# lib/python3.13/, so interpolating the full X.Y.Z would build
# lib/python3.13.16/, a path that matches nothing.
PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

echo "=== Stage 2/15: basic build tools ==="
# --no-build on the wheel-only stages: nothing there may run an sdist's setup.py. The stages that
# compile CUDA extensions from source below cannot carry it.
uv pip install --no-build packaging wheel ninja cmake

echo "=== Stage 3/15: torch 2.13 + CUDA 13.2 runtime ==="
uv pip install --no-build torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132

echo "=== Stage 4/15: CUDA 13.2 nvcc toolchain ==="
CUDA_PIN='13.2.*'
uv pip install --no-build \
    "nvidia-cuda-nvcc==${CUDA_PIN}" \
    "nvidia-cuda-cccl==${CUDA_PIN}" \
    "nvidia-cuda-crt==${CUDA_PIN}" \
    "nvidia-nvvm==${CUDA_PIN}"

export CUDA_HOME="$VIRTUAL_ENV/lib/python${PY_VER}/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"

echo "=== Stage 5/15: symlink nvidia/* unversioned .so libraries ==="
NV_SITE="$VIRTUAL_ENV/lib/python${PY_VER}/site-packages/nvidia"
for libdir in "$NV_SITE"/*/lib; do
  ( cd "$libdir" && \
    for f in lib*.so.*; do
      base=$(echo "$f" | sed -E 's/\.so\.[0-9.]+$/.so/')
      if [[ "$f" != "$base" ]] && [[ ! -e "$base" ]]; then ln -s "$f" "$base"; fi
    done )
done

export CUDNN_HOME="$NV_SITE/cudnn"
export CUDNN_PATH="$CUDNN_HOME"
export CPATH="$NV_SITE/cudnn/include:$NV_SITE/nccl/include:$NV_SITE/cusparselt/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$NV_SITE/cudnn/lib:$NV_SITE/nccl/lib:$NV_SITE/cusparselt/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$NV_SITE/cudnn/lib:$NV_SITE/nccl/lib:$NV_SITE/cusparselt/lib:${LD_LIBRARY_PATH:-}"

echo "=== Stage 6/15: flash-attn 2.8.3 ==="
uv pip install \
    --no-cache-dir --no-build-isolation \
    flash-attn==2.8.3

echo "=== Stage 7/15: transformer-engine 2.15 ==="
NVTE_FRAMEWORK=pytorch uv pip install \
    --no-cache-dir --no-build-isolation \
    "transformer-engine[pytorch]==2.15.*"

echo "=== Stage 8/15: cuda_profiler_api.h shim for apex ==="
# Apex's csrc/megatron/scaled_upper_triang_masked_softmax_cuda.cu includes
# <cuda_profiler_api.h>. The nvidia-cuda-runtime lacks this one, so the include
# fails to resolve and the apex build below dies.
#
# This shim is only the two public prototypes, which is all apex needs to
# compile; the symbols themselves resolve at link time from libcudart, which
# the wheel does ship.
if [[ -f "$CUDA_HOME/include/cuda_profiler_api.h" ]]; then
    echo "cuda_profiler_api.h already present in $CUDA_HOME/include — leaving it alone"
else
    cat > "$CUDA_HOME/include/cuda_profiler_api.h" <<'CUDA_PROFILER_API_H'
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal stand-in for the CUDA Runtime API profiler control header, provided
 * because the cu13 Python wheels omit it. Declarations only.
 */

#if !defined(__CUDA_PROFILER_API_H__)
#define __CUDA_PROFILER_API_H__

#include "driver_types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

extern __host__ cudaError_t CUDARTAPI cudaProfilerStart(void);
extern __host__ cudaError_t CUDARTAPI cudaProfilerStop(void);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !__CUDA_PROFILER_API_H__ */
CUDA_PROFILER_API_H
    echo "wrote cuda_profiler_api.h shim to $CUDA_HOME/include"
fi

echo "=== Stage 9/15: apex ==="
uv pip install \
    --force-reinstall -v --disable-pip-version-check \
    --no-cache-dir --no-build-isolation \
    --config-settings='--build-option=--cpp_ext' \
    --config-settings='--build-option=--cuda_ext' \
    git+https://github.com/NVIDIA/apex.git@becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39

echo "=== Stage 10/15: flash-attn-3-nv ==="
# Do NOT set FLASH_ATTENTION_DISABLE_SM80: it drops the sm80 kernel sources, losing Ampere support.
FLASH_ATTENTION_FORCE_BUILD=TRUE \
    uv pip install \
        --no-cache-dir --no-build-isolation \
        git+https://github.com/alihassanijr/flash_attn_3_nv.git@c603389369b8662a5cf94e5f0096ba739b99ffdd

echo "=== Stage 11/15: natten ==="
# natten reads its OWN arch var (NATTEN_CUDA_ARCH, ";"-separated), not TORCH_CUDA_ARCH_LIST.
# Both are unset by default, which is deliberate: this is the developer venv, so natten and apex
# autodetect the local GPU and build that one arch — fastest, and all this machine can run.
# NOTE this does not apply to every extension: flash-attn and flash-attn-3-nv ignore autodetect and
# use their own hardcoded arch lists (FLASH_ATTN_CUDA_ARCHS defaults to 80;90;100;120), so they
# build multi-arch regardless — which is most of the build time.
# (`:-` is required: bare ${TORCH_CUDA_ARCH_LIST} would abort under `set -u`.) Export
# TORCH_CUDA_ARCH_LIST beforehand for a portable multi-arch venv; docker/build_wheels.sh does that
# for the published wheels. Autodetect needs a visible GPU — without one natten silently builds
# kernel-less.
NATTEN_ARCH="$(printf %s "${TORCH_CUDA_ARCH_LIST:-}" | tr " ," ";;" | tr -s ";")"
NATTEN_ARCH="${NATTEN_ARCH#;}"; NATTEN_ARCH="${NATTEN_ARCH%;}"
uv pip install --no-build cmake
CUDAToolkit_ROOT="$CUDA_HOME" NATTEN_CUDA_ARCH="$NATTEN_ARCH" uv pip install \
    --no-build-isolation \
    --extra-index-url https://whl.natten.org \
    natten==0.21.7

echo "=== Stage 12/15: vllm (AMG local-serving VLM) ==="
# Mirrors the vLLM stanza in docker/Dockerfile; keep in step with its ARGs. Installed ahead of
# requirements.txt so its protobuf and setuptools pins land above vllm's caps.
VLLM_VER=0.29.0
uv pip install --no-build "vllm==${VLLM_VER}" numpy==2.4.6 lark==1.2.2
uv pip uninstall torchaudio pycountry PyNvVideoCodec
# Import-only stand-in for pycountry (LGPL), which vllm's API server reaches through mistral-common.
uv pip install -e assets/pycountry_stub --no-deps

echo "=== Stage 13/15: Python deps (cosmos-anomalygen + cosmos-framework) ==="
uv pip install -r requirements.txt
# Guard against resolver drift in what requirements.txt does not pin (the torch stack the compiled
# wheels above target, and vllm's caps).
VLLM_VER="${VLLM_VER}" python - <<'PYEOF'
import importlib.metadata as md
import os
want = {"torch": "2.13.0+cu132", "torchvision": "0.28.0+cu132", "vllm": os.environ["VLLM_VER"],
        "numpy": "2.4.6", "lark": "1.2.2"}
for name, version in want.items():
    got = md.version(name)
    assert got == version, f"{name}: installed {got}, pinned {version}"
for gone in ("torchaudio", "pycountry", "pynvvideocodec", "nltk"):
    try:
        md.version(gone)
        raise AssertionError(f"{gone} is installed — removed from this env, so a resolve brought it back")
    except md.PackageNotFoundError:
        pass
print("vllm stack OK:", ", ".join(f"{k} {v}" for k, v in want.items()))
PYEOF

echo "=== Stage 14/15: cosmos-framework ==="
uv pip install -r requirements-nodeps.txt --no-deps
# Import-only stub so cosmos-framework's action-dataset modules import without the
# real `lerobot`.
uv pip install -e assets/lerobot_stub --no-deps
# Stand-in for nltk: the guardrail blocklist's tokenizer and noun lemmatizer (assets/nltk_stub).
uv pip install -e assets/nltk_stub --no-deps

echo "=== Stage 15/15: anomalygen ==="
uv pip install -e . --no-deps

echo
echo "=== DONE — venv ready at $(realpath .venv) ==="
echo "activate with:  source .venv/bin/activate"
