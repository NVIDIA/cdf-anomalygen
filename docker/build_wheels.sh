#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# build_wheels.sh — build the heavy native wheels, and (optionally) upload them
# to the project's GitLab PyPI registry so subsequent Docker / CI builds can
# `pip install` instead of recompiling from source.
#
# This script is the SINGLE SOURCE OF TRUTH for the wheel build. It has two
# modes, and the file is laid out in that order:
#
#   MODE A  (no flag)        The orchestration around it, on a developer host:
#                            start the devel container -> run MODE B inside it
#                            -> upload the wheels to the GitLab registry.
#
#   MODE B  --in-container   The compile itself: 12 steps, /wheels as output.
#                            Runs inside a CUDA devel image. Invoked both by
#                            MODE A and by docker/Dockerfile's `wheelbuilder`
#                            stage, so the recipe and its pins exist in exactly
#                            one place and both Docker paths build identically.
#
# The build environment mirrors scripts/env_setup.sh. Keep the pins below in
# sync with env_setup.sh.
#
# Wheels produced:
#   flash-attn                2.8.3      (source-compiled)
#   transformer-engine        2.15       (wrapper)
#   transformer-engine-cu13   2.15       (prebuilt CUDA-13 backend, mirrored)
#   transformer-engine-torch  2.15       (PyTorch backend, source-compiled)
#   apex                      git becbb77 (--cpp_ext --cuda_ext)
#   flash-attn-3-nv           git c603389 (Hopper FA3)
#   natten                    0.21.7      (compiled; no prebuilt for this combo)
#   opencv-python-headless    5.0.0.93    (GitHub tag `93`, WITH_FFMPEG=OFF)
#
# These wheels are architecture-specific (they carry compiled CUDA extensions),
# so they are built and staged per-arch under wheels/<arch>/. The script builds
# for the HOST architecture (x86_64 or aarch64) — run it on an arm64 host to
# produce the aarch64 set.
#
# Usage:
#   GITLAB_TOKEN=<token-with-api-scope> ./docker/build_wheels.sh
#   ./docker/build_wheels.sh --no-upload        # build only, no token needed
#   ./docker/build_wheels.sh --in-container     # compile only; run INSIDE a CUDA devel image
#
# Required env (unless --no-upload):
#   GITLAB_TOKEN     project or personal access token with `api` scope
# Optional env:
#   BUILD_IMAGE      CUDA devel image MODE A compiles in. Must ship nvcc and
#                    match TORCH_VERSION's CUDA. default: the 13.2.1 devel image
#                    (same as docker/Dockerfile's CUDA_DEVEL_BASE).
#   GITLAB_HOST      GitLab instance hosting the package registry. No default;
#                    required unless --no-upload.
#   GITLAB_PROJECT   project path or numeric id on $GITLAB_HOST. No default;
#                    required unless --no-upload.
#                    Both are resolved into REPO_URL before the upload step.
#   MAX_JOBS         parallel compile jobs. default: 16 (matches env_setup.sh).
#                    flash-attn / transformer-engine are memory-hungry — lower
#                    this if the build OOMs.
#   NVCC_THREADS     threads per nvcc invocation. default: 1.
#   TORCH_CUDA_ARCH_LIST
#                    GPU compute-capability list the kernels are built for,
#                    space-separated. This is the GPU arch list — the two
#                    per-arch lists we publish differ only because of which
#                    machines exist:
#                      amd64  "8.0 8.6 9.0 10.0 12.0"
#                             A100 / A10,A40,RTX30 / H100 / B100,B200 /
#                             RTX PRO,RTX50. 8.6 also covers Ada (8.9), and 10.0
#                             covers 10.3, by minor-version cubin compatibility.
#                      arm64  "9.0 10.0 10.3 12.0"
#                             GH200 / GB200 / GB300 / workstation Blackwell.
#                             10.3 is explicit for GB300.
#                    MODE A: unset = auto-derived from
#                    torch.cuda.get_arch_list() — which is why the container
#                    runs with --gpus.
#                    Dockerfile stage 1: a docker build NEVER sees a GPU, so
#                    nothing can be derived; it uses the ARG default instead
#                    (the amd64 list), overridable with --build-arg.
# =============================================================================

set -euo pipefail

usage() { awk 'NR>=5 && /^#/ {print; next} NR>=5 {exit}' "$0"; }

# ---- pins (keep in sync with scripts/env_setup.sh) --------------------------
# Defaults, overridable from the environment so the Dockerfile can pass its ARGs
# through. Both Docker paths (this orchestrator and Dockerfile's wheelbuilder)
# read them from here, so the build recipe is pinned in one file.
# Full X.Y.Z — pinned so the wheels are built on a known CPython patch. PY_VER is
# the X.Y form, and the two are NOT interchangeable: the venv's site-packages
# lives under lib/python3.13/, so path construction must use PY_VER.
: "${PYTHON_VERSION:=3.13.16}"
PY_VER="${PYTHON_VERSION%.*}"
: "${TORCH_VERSION:=2.13.0+cu132}"
: "${TORCHVISION_VERSION:=0.28.0+cu132}"
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu132}"
: "${FLASH_ATTN_VERSION:=2.8.3}"
: "${TE_VERSION:=2.15.0}"   # exact, not 2.15.* — must equal TE_VER in docker/Dockerfile
: "${APEX_SHA:=becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39}"
: "${FA3_SHA:=c603389369b8662a5cf94e5f0096ba739b99ffdd}"
: "${NATTEN_VERSION:=0.21.7}"
: "${OPENCV_TAG:=93}"   # = opencv-python-headless 5.0.0.93
: "${MAX_JOBS:=16}"
: "${NVCC_THREADS:=1}"
# Compile environment for MODE A. Must be a CUDA *devel* image (nvcc + headers)
# and must match the CUDA that TORCH_VERSION was built against — 13.2 here.
# docker/Dockerfile pins the same image as its CUDA_DEVEL_BASE.
: "${BUILD_IMAGE:=nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04}"
# uv, pinned by tag AND digest to its official multi-arch image — the same pin
# docker/Dockerfile uses, so every build path gets a byte-identical binary.
# MODE A extracts it from here and bind-mounts it into the build container.
: "${UV_IMAGE:=ghcr.io/astral-sh/uv:0.12.22@sha256:f513a91fc62fe7c17567eee97230dd198e43edb8a9fbecca843714a4358fe1bc}"
# Version-pinned installer, used only by MODE B's fallback when no uv is present
# (a hand-run --in-container). Keep in sync with UV_IMAGE's tag.
: "${UV_VERSION:=0.12.22}"
export PYTHON_VERSION TORCH_VERSION TORCHVISION_VERSION TORCH_INDEX_URL \
    FLASH_ATTN_VERSION TE_VERSION APEX_SHA FA3_SHA NATTEN_VERSION OPENCV_TAG \
    MAX_JOBS NVCC_THREADS UV_VERSION

# ---- registry (MODE A only; unused by the compile) --------------------------
# Where the prebuilt wheels are published, and where a Docker build pulls them
# from instead of recompiling. Both come from the environment — CI resolves the
# same values from $CI_SERVER_HOST and $CI_PROJECT_ID, so a manual upload lands
# in the registry the image build reads. They are validated, and REPO_URL built
# from them, in MODE A below; the compile never touches them.
GITLAB_HOST="${GITLAB_HOST:-}"
GITLAB_PROJECT="${GITLAB_PROJECT:-}"

# =============================================================================
# MODE B — compile (--in-container)
#
# Runs as root inside a CUDA devel image and writes finished wheels to /wheels.
# Two callers, and they must stay interchangeable:
#   * MODE A below, via `docker run` on a developer machine.
#   * docker/Dockerfile's `wheelbuilder` stage, for a standalone image build.
# Steps 1-5 build the compile environment; steps 6-11 produce one wheel each.
# Every wheel step is skippable (see `have`), so a re-run only builds what is
# missing after a failure part-way through.
# =============================================================================
build_in_container() {
    set -x

    # ---- Step 1/12: OS toolchain (compilers, cmake/ninja, git) --------------
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential cmake ninja-build \
        libssl-dev libffi-dev

    # ---- Step 2/12: uv, then the py3.13 venv + build tooling ----------------
    # uv provides the pinned Python (same as scripts/env_setup.sh). Normally it
    # is already on PATH — MODE A bind-mounts the digest-pinned binary, and the
    # Dockerfile COPYs it from the same image — so this installs only when the
    # script is run --in-container by hand.
    if ! command -v uv >/dev/null 2>&1; then
        # -L follows redirects and the result is piped straight to a shell, so pin the scheme on
        # the redirect too: without it one Location: http://... is arbitrary code over plaintext.
        curl -LsSf --proto '=https' --proto-redir '=https' "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
        export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
    fi

    mkdir -p /wheels /tmp/build && cd /tmp/build

    uv venv --clear --python "$PYTHON_VERSION"
    source .venv/bin/activate
    uv pip install --no-build pip packaging wheel ninja cmake scikit-build

    # ---- Step 3/12: torch — fixes the CUDA/ABI every wheel links against ----
    uv pip install torch=="$TORCH_VERSION" torchvision=="$TORCHVISION_VERSION" \
        --index-url "$TORCH_INDEX_URL"

    # ---- Step 4/12: GPU arch list the kernels are compiled for --------------
    # Auto-derive TORCH_CUDA_ARCH_LIST (unless set) from the pinned torch
    # build-arch set. Relies on --gpus: get_arch_list() returns [] without a
    # visible GPU, and an empty arch crashes apex and yields a kernel-less
    # natten.
    if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        TORCH_CUDA_ARCH_LIST="$(python - <<'PYEOF'
import re

import torch

# "sm_90" -> 9.0, "sm_100" -> 10.0: last digit is minor. Strip the "a"/"f" arch-conditional
# suffix; passing it to float() would raise and take out the whole derivation.
caps = set()
for arch in torch.cuda.get_arch_list():
    m = re.fullmatch(r"sm_(\d+)(\d)[a-z]*", arch)
    if m:
        caps.add(f"{m.group(1)}.{m.group(2)}")
print(" ".join(sorted((c for c in caps if float(c) >= 8.0), key=float)))
PYEOF
)"
    fi
    [[ -n "$TORCH_CUDA_ARCH_LIST" ]] || {
        echo "ERROR: TORCH_CUDA_ARCH_LIST is empty and could not be derived:" >&2
        echo "       torch.cuda.get_arch_list() returns [] when no GPU is visible." >&2
        echo "       MODE A  -> ensure 'docker run --gpus all' (nvidia-container-toolkit)." >&2
        echo "       Dockerfile stage 1 -> a docker build NEVER sees a GPU; you must pass" >&2
        echo "       --build-arg TORCH_CUDA_ARCH_LIST='8.0 8.6 9.0 10.0 12.0' (or your arches)." >&2
        exit 1
    }
    export TORCH_CUDA_ARCH_LIST
    echo "==> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

    # ---- Step 5/12: CUDA toolchain + torch's bundled CUDA libs --------------
    # CUDA 13.2 toolkit comes from the devel base image.
    export CUDA_HOME="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"

    # The +cu132 torch wheel ships its CUDA runtime libs under site-packages/
    # nvidia as lib*.so.N only; create the unversioned .so symlinks the
    # linker needs.
    NV_SITE="$VIRTUAL_ENV/lib/python$PY_VER/site-packages/nvidia"
    for libdir in "$NV_SITE"/*/lib; do
        ( cd "$libdir" && for f in lib*.so.*; do
            base=$(echo "$f" | sed -E 's/\.so\.[0-9.]+$/.so/')
            if [[ "$f" != "$base" ]] && [[ ! -e "$base" ]]; then ln -s "$f" "$base"; fi
        done )
    done
    export CUDNN_HOME="$NV_SITE/cudnn" CUDNN_PATH="$NV_SITE/cudnn"
    export CPATH="$NV_SITE/cudnn/include:$NV_SITE/nccl/include:$NV_SITE/cusparselt/include:${CPATH:-}"
    export LIBRARY_PATH="$CUDA_HOME/lib64:$NV_SITE/cudnn/lib:$NV_SITE/nccl/lib:$NV_SITE/cusparselt/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NV_SITE/cudnn/lib:$NV_SITE/nccl/lib:$NV_SITE/cusparselt/lib:${LD_LIBRARY_PATH:-}"

    # ---- Steps 6-11/12: one wheel per step, into /wheels --------------------
    #
    # WHICH TOOL TO USE — decided by the verb, not by preference.
    #
    #   python -m pip wheel     build a .whl into /wheels   <- the shipped artifact
    #   python -m pip download  mirror an upstream prebuilt .whl, no compile
    #   uv pip install          put a package in the venv so a LATER step can
    #                           compile against it — never produces an artifact
    #
    # Idempotent: skip any wheel already present so a re-run only builds
    # what is missing (wheel dist-names normalize "-" to "_").
    #
    # Keyed on a STAMP, not just the filename. /wheels is bind-mounted and persists
    # across runs (MODE A), and the filename does not always encode what changed:
    # apex is always apex-0.1-*.whl and flash-attn-3-nv always flash_attn_3_nv-1.0.3-*
    # whatever APEX_SHA / FA3_SHA say. Globbing on the name alone therefore skips the
    # build after a pin bump and silently ships the stale wheel. The stamp records the
    # pin each wheel was built from, so a bump misses and rebuilds.
    stamp_file() { local name="$1"; echo "/wheels/.stamp-${name}"; }
    have() {  # have <dist name> <pin>
        local name="$1" pin="$2"
        ls /wheels/"${name}"-*.whl >/dev/null 2>&1 || return 1
        [[ "$(cat "$(stamp_file "$name")" 2>/dev/null)" = "$pin" ]]
    }
    mark() { local name="$1" pin="$2"; echo "$pin" > "$(stamp_file "$name")"; }

    # ---- Step 6/12: flash-attn (FA2) ----------------------------------------
    if have flash_attn "$FLASH_ATTN_VERSION"; then echo "skip flash-attn (already built)"; else
        python -m pip wheel --no-build-isolation --no-deps -w /wheels \
            flash-attn=="$FLASH_ATTN_VERSION"
        mark flash_attn "$FLASH_ATTN_VERSION"
    fi

    # ---- Step 7/12: transformer-engine — 3 distributions --------------------
    # wrapper + prebuilt CUDA-13 backend (mirror) + source PyTorch backend.
    # NOTE: the prebuilt backend is transformer-engine-cu13 for CUDA 13;
    #       adjust if a future TE release renames it.
    if have transformer_engine_torch "$TE_VERSION"; then echo "skip transformer-engine (already built)"; else
        python -m pip wheel --no-build-isolation --no-deps -w /wheels \
            transformer-engine=="$TE_VERSION"
        uv pip install --no-deps \
            transformer-engine=="$TE_VERSION" transformer-engine-cu13=="$TE_VERSION"
        python -m pip download --no-deps --only-binary=:all: -d /wheels \
            transformer-engine-cu13=="$TE_VERSION"
        NVTE_FRAMEWORK=pytorch python -m pip wheel --no-build-isolation --no-deps \
            -w /wheels transformer-engine-torch=="$TE_VERSION"
        mark transformer_engine_torch "$TE_VERSION"
    fi

    # ---- Step 8/12: apex (pinned SHA, cpp_ext + cuda_ext) -------------------
    if have apex "$APEX_SHA"; then echo "skip apex (already built)"; else
        python -m pip wheel -v --no-build-isolation --no-deps \
            --disable-pip-version-check --no-cache-dir \
            --config-settings='--build-option=--cpp_ext' \
            --config-settings='--build-option=--cuda_ext' \
            -w /wheels "apex @ git+https://github.com/NVIDIA/apex.git@$APEX_SHA"
        mark apex "$APEX_SHA"
    fi

    # ---- Step 9/12: flash-attn-3-nv (FA3, pinned SHA) -----------------------
    if have flash_attn_3_nv "$FA3_SHA"; then echo "skip flash-attn-3-nv (already built)"; else
        FLASH_ATTENTION_FORCE_BUILD=TRUE \
            python -m pip wheel --no-build-isolation --no-deps -w /wheels \
            "git+https://github.com/alihassanijr/flash_attn_3_nv.git@$FA3_SHA"
        mark flash_attn_3_nv "$FA3_SHA"
    fi

    # ---- Step 10/12: natten -------------------------------------------------
    # natten reads its OWN arch var NATTEN_CUDA_ARCH, NOT TORCH_CUDA_ARCH_LIST;
    # with it unset it compiles the libnatten CUDA kernels only when
    # torch.cuda.is_available() holds AT BUILD TIME, and silently produces a
    # kernel-less pure (py3-none-any) wheel otherwise.
    if have natten "$NATTEN_VERSION"; then echo "skip natten (already built)"; else
        NATTEN_ARCH="$(printf %s "$TORCH_CUDA_ARCH_LIST" | tr " ," ";;" | tr -s ";")"
        NATTEN_ARCH="${NATTEN_ARCH#;}"; NATTEN_ARCH="${NATTEN_ARCH%;}"
        CUDAToolkit_ROOT="$CUDA_HOME" NATTEN_CUDA_ARCH="$NATTEN_ARCH" \
          NATTEN_N_WORKERS="$MAX_JOBS" \
            python -m pip wheel --no-build-isolation --no-deps -w /wheels \
            "natten==$NATTEN_VERSION"
        # Guard: a kernel-ful natten is a platform wheel; a kernel-less one is
        # py3-none-any. Fail loudly so a broken wheel never reaches the registry.
        if ls /wheels/natten-*-py3-none-any.whl >/dev/null 2>&1; then
            echo "ERROR: natten built WITHOUT CUDA kernels (py3-none-any). Set TORCH_CUDA_ARCH_LIST (=> NATTEN_CUDA_ARCH), or ensure torch.cuda.is_available() at build time." >&2
            # Delete it, or `have natten` sees it next run, skips the build and ships it.
            rm -f /wheels/natten-*-py3-none-any.whl
            exit 1
        fi
        mark natten "$NATTEN_VERSION"
    fi

    # ---- Step 11/12: opencv-python-headless (tag, WITH_FFMPEG=OFF) ----------
    # Built here rather than taken from PyPI so the wheel carries no bundled
    # FFmpeg — the PyPI headless wheel does, and it is the licence-heavy part.
    if have opencv_python_headless "$OPENCV_TAG"; then echo "skip opencv (already built)"; else
        git clone --recursive --shallow-submodules --depth 1 --branch "$OPENCV_TAG" \
            https://github.com/opencv/opencv-python.git /tmp/opencv-python
        cd /tmp/opencv-python
        ENABLE_HEADLESS=1 CMAKE_ARGS="-DWITH_FFMPEG=OFF" MAKEFLAGS="-j$MAX_JOBS" \
            python -m pip wheel --no-build-isolation --no-deps -w /wheels .
        cd /
        rm -rf /tmp/opencv-python
        mark opencv_python_headless "$OPENCV_TAG"
    fi

    # ---- Step 12/12: hand /wheels back to the invoking host user ------------
    # HOST_UID is set only by MODE A's `docker run`; a Dockerfile build has no
    # host user to hand back to, and /wheels is consumed by the next stage.
    if [[ -n "${HOST_UID:-}" ]]; then
        chown -R "$HOST_UID":"${HOST_GID:-$HOST_UID}" /wheels
    fi
    set +x
    echo "==> Built wheels:"
    ls -1 /wheels
}

# =============================================================================
# Entry point — dispatch to MODE B, or fall through to MODE A below
# =============================================================================
NO_UPLOAD=0
IN_CONTAINER=0
for arg in "$@"; do
    case "$arg" in
        -h|--help)      usage; exit 0 ;;
        --no-upload)    NO_UPLOAD=1 ;;
        --in-container) IN_CONTAINER=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$IN_CONTAINER" -eq 1 ]]; then
    build_in_container
    exit 0
fi

# =============================================================================
# MODE A — orchestrate (default)
#
# 1. Resolve the target CPU arch and the per-arch staging dir.
# 2. Run MODE B in the CUDA devel container (this same file, bind-mounted).
# 3. Upload whatever is not already in the registry.
# Nothing here compiles anything itself — it only sets up and moves artifacts.
# =============================================================================
if [[ "$NO_UPLOAD" -eq 0 ]]; then
    : "${GITLAB_HOST:?set GITLAB_HOST (registry host), or pass --no-upload}"
    : "${GITLAB_PROJECT:?set GITLAB_PROJECT (path or numeric id), or pass --no-upload}"
    : "${GITLAB_TOKEN:?set GITLAB_TOKEN (token with api scope), or pass --no-upload}"
    # GitLab's API accepts either a numeric id or a URL-encoded path.
    REPO_URL="https://${GITLAB_HOST}/api/v4/projects/${GITLAB_PROJECT//\//%2F}/packages/pypi"
fi
# ---- MODE A step 1: target architecture -------------------------------------
# Always the host arch: the nvidia/cuda base image and torch's +cu132 wheels
# ship amd64 + arm64 variants, so a native build on either host Just Works.
HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    aarch64|arm64) ARCH="aarch64"; DOCKER_PLATFORM="linux/arm64" ;;
    x86_64|amd64)  ARCH="x86_64";  DOCKER_PLATFORM="linux/amd64" ;;
    *) echo "Unsupported architecture: $HOST_ARCH" >&2; exit 2 ;;
esac

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Per-arch staging dir: keeps an x86_64 and an aarch64 wheel set separate so the
# idempotency check can't mistake one arch's wheels for the other's.
WHEELS_DIR="$REPO_ROOT/wheels/$ARCH"
mkdir -p "$WHEELS_DIR"

# ---- MODE A step 2: compile, by running MODE B in the devel container -------
# apt-get needs root, so the build container runs as root and chown's /wheels
# back to the host user at the end (rather than --user, which would break apt).
echo "==> Building $ARCH wheels in $BUILD_IMAGE -> $WHEELS_DIR"
# --platform: pin the base-image variant to the target arch.
# --gpus all: lets torch.cuda.get_arch_list() initialize CUDA and return the
# pinned torch's full build-arch set, which the container uses to auto-derive
# TORCH_CUDA_ARCH_LIST (it returns [] without a visible GPU); requires
# nvidia-container-toolkit. This script is bind-mounted in and re-invoked with
# --in-container, so the compile recipe lives in exactly one place
# (docker/Dockerfile's wheelbuilder runs the same file).
# Extract the digest-pinned uv so the container uses the exact same binary the
# Dockerfile COPYs.
UV_STAGE="$(mktemp -d)"
trap 'rm -rf "$UV_STAGE"' EXIT
uv_cid="$(docker create --platform "$DOCKER_PLATFORM" "$UV_IMAGE")"
docker cp "$uv_cid:/uv" "$UV_STAGE/uv" >/dev/null
docker rm -f "$uv_cid" >/dev/null
chmod 755 "$UV_STAGE/uv"
echo "==> uv pinned from $UV_IMAGE"

docker run --rm \
    --platform "$DOCKER_PLATFORM" \
    --gpus all \
    -v "$WHEELS_DIR:/wheels" \
    -v "$SCRIPT_PATH:/tmp/build_wheels.sh:ro" \
    -v "$UV_STAGE/uv:/usr/local/bin/uv:ro" \
    -e MAX_JOBS -e NVCC_THREADS -e TORCH_CUDA_ARCH_LIST \
    -e PYTHON_VERSION -e TORCH_VERSION -e TORCHVISION_VERSION -e TORCH_INDEX_URL \
    -e FLASH_ATTN_VERSION -e TE_VERSION -e APEX_SHA -e FA3_SHA \
    -e NATTEN_VERSION -e OPENCV_TAG \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "$BUILD_IMAGE" bash /tmp/build_wheels.sh --in-container

echo "==> Built wheels:"
ls -lh "$WHEELS_DIR"/*.whl

# ---- MODE A step 3: upload --------------------------------------------------
if [[ "$NO_UPLOAD" -eq 1 ]]; then
    echo "==> --no-upload set; wheels left in $WHEELS_DIR"
    exit 0
fi

echo "==> Uploading $ARCH wheels to ${REPO_URL}"
docker run --rm \
    --platform "$DOCKER_PLATFORM" \
    --user "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e HOME=/tmp \
    -v "$WHEELS_DIR:/wheels:ro" \
    -e REPO_URL="$REPO_URL" \
    -e TWINE_USERNAME=gitlab-ci-token \
    -e TWINE_PASSWORD="$GITLAB_TOKEN" \
    python:3.13-slim bash -euxc '
        pip install --no-cache-dir twine
        # Credentials come from TWINE_USERNAME/TWINE_PASSWORD env, so the token
        # never appears on the command line (safe under `set -x`).
        #
        # Idempotent upload WITHOUT parsing twine errors (GitLab rejects a
        # duplicate with an opaque "400 Bad Request", and its --skip-existing is
        # unsupported under --repository-url): ask the registry which files it
        # already has, then upload only the rest. The arch-independent
        # transformer-engine wrapper (py3-none-any) is the usual repeat across
        # arches. A failed query, or any genuinely failed upload, aborts (set -e).
        #
        # NOTE: replacing a file means DELETE via the API first, then waiting for
        # the listing to drop it — GitLab frees the name asynchronously and an
        # immediate re-upload of the same filename fails with an opaque 400.
        python3 - >/tmp/existing.txt <<"PYEOF"
import json, os, urllib.request
base = os.environ["REPO_URL"].split("/packages/pypi")[0]
hdr = {"PRIVATE-TOKEN": os.environ["TWINE_PASSWORD"]}
def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdr)))
names, page = set(), 1
while True:
    pkgs = get(base + "/packages?package_type=pypi&per_page=100&page=" + str(page))
    if not pkgs:
        break
    for p in pkgs:
        for f in get(base + "/packages/" + str(p["id"]) + "/package_files"):
            names.add(f["file_name"])
    page += 1
print("\n".join(sorted(names)))
PYEOF
        for whl in /wheels/*.whl; do
            b="$(basename "$whl")"
            if grep -qxF "$b" /tmp/existing.txt; then
                echo "skip (already in registry): $b"
            else
                python -m twine upload --repository-url "$REPO_URL" "$whl"
                echo "uploaded: $b"
            fi
        done
    '

# The wheel's version ends in the opencv-python tag it was built from (5.0.0.93 <- tag 93).
OPENCV_WHEEL_VERSION="$(basename "$(ls "$WHEELS_DIR"/opencv_python_headless-*."${OPENCV_TAG}"-*.whl | head -1)" | cut -d- -f2)"
echo "==> Uploaded. Consume via:"
echo "    pip install --extra-index-url '${REPO_URL}/simple' \\"
echo "      flash-attn==${FLASH_ATTN_VERSION} \\"
echo "      transformer-engine==${TE_VERSION} transformer-engine-torch==${TE_VERSION} \\"
echo "      apex natten==${NATTEN_VERSION} \\"
echo "      opencv-python-headless==${OPENCV_WHEEL_VERSION}    # FFmpeg-free build, opencv-python tag ${OPENCV_TAG}"
