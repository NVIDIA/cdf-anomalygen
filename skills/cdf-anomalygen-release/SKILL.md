---
name: cdf-anomalygen-release
description: >-
  Use when building the cdf-anomalygen Docker image (develop / product /
  air-gapped) from docker/Dockerfile — build the chosen target from source and
  run it. Not for training or synthetic defect-image generation (SDG).
license: Apache-2.0
compatibility: >-
  Requires Docker 23+ (BuildKit) + nvidia-container-toolkit and a CUDA GPU. The
  build compiles the heavy CUDA extensions from source and takes >1 h.
metadata:
  owner: NVIDIA
  service: docker
  version: 1.2.0
  author: NVIDIA <nvidia@nvidia.com>
  tags:
      - physical-ai
      - docker
      - release
      - air-gapped
  languages: [shell]
  frameworks: [docker]
---

# Skill: cdf-anomalygen-release

## Purpose

Build and run the **cdf-anomalygen** container. `docker/Dockerfile` reproduces
`scripts/env_setup.sh` as a two-stage image (compile heavy wheels → slim non-root runtime).
This is a **release/build** workflow — not training or inference
(use the `cdf-anomalygen` skill for those). Build targets, GPU-arch lists, and troubleshooting
detail live in `docker/README.md`. Run every command from the repo root.

> **Check whether a build is needed at all.** Prebuilt images are published on
> [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cdf-anomalygen):
> `docker pull nvcr.io/nvidia/cdf-anomalygen:1.2.0` needs no login and no compile.
> Build only when the user wants a modified image, a target NGC does not publish, or an
> air-gapped bundle. **A build from source takes >1 h** — say so before starting one.

## When to Use This Skill

Use this skill when the user wants to build or ship the anomalygen container —
build a **develop** / **product** / **air-gapped** image, push it to a registry, or
run it. E.g. "build the anomalygen product container", "build an air-gapped develop
image".

Do not use it to run training or SDG generation — that's the `cdf-anomalygen` skill.

## Prerequisites

- **Docker 23+** (BuildKit on by default) + **nvidia-container-toolkit** + a CUDA GPU.
- Enough headroom for the source compile: the heavy extensions are memory-hungry and the
  build needs tens of GB of disk for intermediate layers.
- The repo **`.venv`** (`scripts/env_setup.sh`: py3.13, torch 2.13.0+cu132) is **optional** —
  it is used only to auto-derive `TORCH_CUDA_ARCH_LIST` below, via
  `torch.cuda.get_arch_list()`. Without it the documented default is used.
- **Air-gapped only:** the model checkpoints under `./checkpoints/`. `scripts/download_checkpoints.sh`
  populates this directory.

## Targets

`docker/Dockerfile` exposes four targets (`--target`, default `develop`):

| Target              | What                                                                          |
| ------------------- | ----------------------------------------------------------------------------- |
| `develop` (default) | non-root interactive image + dev tooling (`pytest`, `ruff`, `pre-commit`)     |
| `product`           | non-root, app code **read-only**, `ANOMALYGEN_PRODUCT_MODE=1`, no dev tooling |
| `airgapped-develop` | `develop` + the checkpoints **baked in** for offline runs                     |
| `airgapped-product` | `product` + the checkpoints **baked in** for offline runs                     |

## Instructions

**Default flow: Step 1, then stop with the built image.** Push (Step 2) and run
(Step 3) are opt-in — only do them when the user explicitly asks to publish or run the image.

Set the shared variables once (arm64 host → use the commented values):

```shell
export TARGET=develop                       # develop | product | airgapped-develop | airgapped-product
export IMAGE=cdf-anomalygen:cuda-13.2.1-${TARGET}-ubuntu24.04-amd64
# tag = <base-cuda>-<target>-<ubuntu>-<arch>; arm64 host → …-arm64. Set TARGET *first*: it feeds both
# the tag and --target below, so they cannot disagree. Hardcoding "develop" here and then building
# --target product silently ships a product image labelled develop.
export PLATFORM=linux/amd64                 # arm64 host: linux/arm64
# GPU arch (>=8.0), derived from .venv torch when present; forwarded to the build as a --build-arg.
# The `:-` matters: get_arch_list() is [] with no GPU (and the command fails outright with no .venv),
# and a bare --build-arg forwards that empty value, overriding the Dockerfile default.
export TORCH_CUDA_ARCH_LIST="$(.venv/bin/python -c "import torch; print(' '.join(c for c in sorted({f'{a[3:-1]}.{a[-1]}' for a in torch.cuda.get_arch_list() if a.startswith('sm_')}, key=float) if float(c) >= 8.0))" 2>/dev/null)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 9.0 10.0 12.0}"  # arm64: '9.0 10.0 10.3 12.0'
```

### Step 1 — build the image

Stage 1 compiles flash-attn / transformer-engine / apex / flash-attn-3-nv / natten from
source. **This takes >1 h** — it is not a hang. `TORCH_CUDA_ARCH_LIST` is *required* on
this path: `docker build` never sees a GPU, so nothing can be autodetected.

**Standard** (`develop` / `product`):

```shell
docker build --platform "$PLATFORM" -f docker/Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST \
    --target "$TARGET" -t "$IMAGE" .        # TARGET from the shared vars; tag and target stay in sync
```

**Air-gapped** (`airgapped-develop` / `airgapped-product`) — needs `buildx` + a named
`ckpts` context (which bypasses `.dockerignore`). The image is self-contained, so save it
to `./results/` as a gzipped bundle for offline transfer and **report that path to the user**:

```shell
# --load is required: buildx's default driver keeps the result in its own build cache, so without it
# the tag never reaches the local daemon and the `docker save` below fails with "reference does not
# exist". (Single-platform builds only — --load cannot export a multi-arch manifest.)
docker buildx build --platform "$PLATFORM" -f docker/Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST \
    --build-context ckpts=./checkpoints \
    --target "$TARGET" -t "$IMAGE" --load . # set TARGET=airgapped-develop|airgapped-product first

mkdir -p results
BUNDLE="results/$(basename "$IMAGE" | tr ':' '_').tar.gz"
docker save "$IMAGE" | gzip > "$BUNDLE"
echo "air-gapped bundle: $BUNDLE"      # tell the user this location
```

### Step 2 — push (only when the user asks to publish; skip by default)

Ask the user which registry to publish to — there is no default. Tag the local image for
that registry, authenticate, then push:

```shell
export REGISTRY=<your-registry>          # e.g. registry.example.com or an org path
docker tag "$IMAGE" "$REGISTRY/$IMAGE"
docker login "$REGISTRY"                 # credentials per your registry; --password-stdin in CI
docker push "$REGISTRY/$IMAGE"
```

### Step 3 — run (only when the user asks to run the container; skip by default)

`docker run` invocation, the `--user` / `-e USER` pairing that makes `getpass.getuser()` resolve,
`--shm-size`, `HF_TOKEN`, and loading an air-gapped bundle:
[`running.md`](references/running.md).

## Verification

Run these **after Step 1 has produced an image** — they check that image, and the smoke test needs the
Step 3 `docker run` invocation, so they are not a standalone entry point.

1. Image exists: `docker images "$IMAGE"`.
2. **product only** — guardrails hold: `ANOMALYGEN_PRODUCT_MODE=1` is set, it runs as
   the non-root `nvidia` user, and the app code is read-only. If any fails, the
   image is **not** productized.
3. Smoke test — in the Step 3 container (repo + `checkpoints/` mounted; baked in for
   air-gapped), a 20-iteration dry run finishes without error.

   > **The output root must be fresh, or this check is worthless.** A reused root resumes a leftover
   > checkpoint and finishes instantly, which looks identical to success. The `date +%s` suffix
   > below is what makes it fresh — keep it, and confirm the run actually logs 20 iterations.

```shell
IMAGINAIRE_OUTPUT_ROOT="$PWD/results/dryrun-$(date +%s)" torchrun --nproc_per_node=1 \
    anomalygen/scripts/texture/train.py --config=cosmos_framework/configs/base/config.py \
    --recipe=ag_config/exp_texture_ft_phone_screen.yaml \
    -- experiment=anomalygen_texture_ft trainer.max_iter=20 trainer.validation_iter=10 checkpoint.save_iter=10
```

## Examples

**"Build the anomalygen product container"** — shared variables, then Step 1 with the product target;
stop there (no push unless asked):

```shell
export TARGET=product   # re-export IMAGE so the tag follows the target
export IMAGE=cdf-anomalygen:cuda-13.2.1-${TARGET}-ubuntu24.04-amd64
docker build --platform "$PLATFORM" -f docker/Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST \
    --target "$TARGET" -t "$IMAGE" .
```

**"Build an air-gapped develop image"** — Step 1's `buildx` variant with the `ckpts` named context,
then load the saved bundle on the offline host. The checkpoints must already be under
`./checkpoints/`.

## Troubleshooting

- **"TORCH_CUDA_ARCH_LIST is empty and could not be derived"** → something overrode the Dockerfile
  default with an empty value: a bare `--build-arg` forwards an empty env var, and `get_arch_list()`
  is `[]` with no GPU. The shared vars guard this with `:-`; otherwise set it by hand (amd64
  `8.0 8.6 9.0 10.0 12.0`, arm64 `9.0 10.0 10.3 12.0`).
- **Build seems stuck for an hour** → that is the source compile of flash-attn / TE / apex, which is
  expected. Don't kill it; watch the layer output instead.
- **Build OOM** (flash-attn / TE / apex) → lower the job count with
  `--build-arg MAX_JOBS=…` (default 16).
- **Missing `--shm-size` (min `16g`)** → "Bus error" / silent hangs: the dataloader
  uses `/dev/shm` and Docker's 64 MB default is too small. Always pass it.
- **`failed to find stage "…"`** → bad `--target`; valid: `develop`, `product`,
  `airgapped-develop`, `airgapped-product`.
- **Don't build inside a product runtime** (`ANOMALYGEN_PRODUCT_MODE=1`) — build from
  a clean clone or a develop container.
