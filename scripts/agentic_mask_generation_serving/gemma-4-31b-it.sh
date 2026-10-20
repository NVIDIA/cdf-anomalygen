#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Serve google/gemma-4-31b-it for AMG, FP8, sharded over two GPUs.
#
#   CUDA_VISIBLE_DEVICES=1,6 scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh
#   CUDA_VISIBLE_DEVICES=0 TP=1 MAXLEN=65536 scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh
#
# CUDA_VISIBLE_DEVICES is REQUIRED, not defaulted: which GPUs are safe to load is a property of the
# host, not of this recipe, and a default here would be wrong on every machine but one.
#
# TP must equal the number of devices CUDA_VISIBLE_DEVICES names, and MAXLEN is the serving window
# every request must fit prompt + max_tokens inside. Both are host properties for the same reason
# the device list is, so both are overridable; the defaults are the two-GPU recipe.
set -eu
: "${CUDA_VISIBLE_DEVICES:?set it explicitly, e.g. CUDA_VISIBLE_DEVICES=1,6}"
VLLM_BIN=${VLLM_BIN:-vllm}
PORT=${PORT:-8902}
# Optional bearer lock: set API_KEY to make the server reject other callers. Clients then present
# the same value via their endpoints.yaml entry's `key:`. Unset = the server accepts any bearer,
# and the client default EMPTY works.
API_KEY=${API_KEY:-}
TP=${TP:-2}
MAXLEN=${MAXLEN:-32768}
# Which weights this recipe serves is a row in the weights catalogue, named by group: the catalogue
# is what provisions them and what every preflight checks, so the repo id is declared once, there.
# The entrypoint reads GROUP off this line the way it reads TP and PORT.
GROUP=${GROUP:-gemma}
# shellcheck source=scripts/weights_catalogue.sh
source "$(dirname "${BASH_SOURCE[0]}")/../weights_catalogue.sh"
MODEL="$(w_location "$GROUP")"
# The served name is what --gen-model/--judge-model must be given, and what the model registry keys on.
SERVED=${SERVED:-nvidia/google/gemma-4-31b-it}
# Weights live in checkpoints/hf with every other checkpoint (scripts/download_checkpoints.sh), so the
# repo mount carries them into a container; the recipe runs from the repo root. Override to serve
# from another cache.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PWD/checkpoints/hf}

[[ -n "${CUDA_HOME:-}" ]] && export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# FlashInfer's kernels are JIT-compiled at startup and need nvcc, which the image does not ship. Three
# paths reach them: the top-k/top-p sampler, and at TP>1 the allreduce+RMSNorm fusion pass (on by default
# on Hopper and datacenter Blackwell) and the FlashInfer all-reduce communicator. vLLM's own kernels
# need nothing. Override on a host with a toolkit to opt back in.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"pass_config":{"fuse_allreduce_rms":false}}'}
# Leave VLLM_USE_DEEP_GEMM alone. Its default of 1 is the fast Hopper FP8 kernel; an earlier note here
# recommended 0 and was wrong.
# fuse_allreduce_rms off: with TP>1 vLLM otherwise JIT-compiles FlashInfer's mnnvl all-reduce fusion at
# startup, which fails on a host without a matching CUDA toolkit (and buys nothing on one node). Every
# server we have run successfully carried this flag; the shipped recipe was missing it.

exec "$VLLM_BIN" serve "$MODEL" ${API_KEY:+--api-key "$API_KEY"} \
  --served-model-name "$SERVED" \
  --quantization fp8 --port "$PORT" \
  --tensor-parallel-size "$TP" --disable-custom-all-reduce \
  --compilation-config "$COMPILATION_CONFIG" \
  --max-model-len "$MAXLEN" \
  --limit-mm-per-prompt '{"image":64}' \
  --gpu-memory-utilization 0.9
