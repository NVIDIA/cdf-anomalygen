# Docker

Container build for **CDF AnomalyGen**. Reproduces `scripts/env_setup.sh`
(py3.13, torch 2.13.0+cu132, CUDA 13.2) in an image.

> **You may not need to build at all** — *if* this repo's `VERSION` is already published on
> [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cdf-anomalygen). Pulling needs
> no registry login:
>
> ```shell
> docker pull nvcr.io/nvidia/cdf-anomalygen:1.2.0
> ```
>
> If the pull fails (*manifest unknown*, *not found* or *pull access denied*) or that tag is
> not on the [tag list](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cdf-anomalygen/tags) yet,
> build from this checkout — an older tag was built from a different source revision and will
> not match this tree. Also build when you want a modified image, a target NGC does not
> publish, or an air-gapped bundle. **A source build takes > 1 h.**

| File              | Purpose                                                       |
| ----------------- | ------------------------------------------------------------- |
| `Dockerfile`      | Two-stage image build (compile → slim runtime).               |
| `build_wheels.sh` | The compile recipe stage 1 runs; also runnable standalone.    |

The compile-heavy dependencies are **flash-attn, transformer-engine, apex,
flash-attn-3-nv, natten**, and an **FFmpeg-free opencv-python-headless**.
Compiling them from source is what makes the build take > 1 h.

## 1. `Dockerfile` — build the image

**Stage 1 `wheelbuilder`** (`...-cudnn-devel`): compiles the heavy wheels by running
`docker/build_wheels.sh --in-container`, so the pins live in one file and a standalone
`build_wheels.sh` run produces the same wheels the image gets.

**Stage 2 `runtime`** (`...-base`, no compiler): creates the py3.13 venv and
installs the heavy wheels from stage 1, the pip requirements, cosmos-framework
(pinned in `requirements-nodeps.txt`), and the in-repo packages, as a non-root
`nvidia` user.

**Targets** (`--target`, default `develop`):

- `develop` — non-root interactive image + dev/test tooling (`pytest`, `ruff`,
  `pre-commit`); the default.
- `product` — non-root image + app code locked read-only + `ANOMALYGEN_PRODUCT_MODE=1`,
  **no dev tooling**.
- `airgapped-develop` / `airgapped-product` — the above with the model's
  checkpoints **baked in** for offline runs. Populate `./checkpoints` with
  `scripts/download_checkpoints.sh` first, then supply it via a named `ckpts`
  context (bypasses `.dockerignore`) with `buildx` (see below). These two also
  bake a ~24 MB environment for the framework's `uv run`-ed HF CLI and point
  `IMAGINAIRE_HF_CLI_ENVIRONMENT` at it — without that the CLI reaches for a
  package index on every checkpoint resolve. See the `airgapped-base` stage in
  `docker/Dockerfile` for the mechanism.
  They also set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so the baked,
  revision-pinned checkpoints are the only thing a model can load: some framework
  code calls `from_pretrained()` without a revision, and without these a cache
  miss there would silently resolve a mutable branch over the network instead of
  failing. `product` and `develop` keep network access — a `develop` user may
  legitimately run `scripts/download_checkpoints.sh` inside the container — and
  can opt in per run with `docker run -e HF_HUB_OFFLINE=1 ...`.

Shared setup (used by both build types below):

```shell
export TAG=cuda-13.2.1-develop-ubuntu24.04-amd64         # or -arm64
export IMAGE=cdf-anomalygen:$TAG

# Target platform + GPU arch list (see the table below). The commands forward the
# arch list from the environment with a bare `--build-arg TORCH_CUDA_ARCH_LIST`.
export PLATFORM=linux/amd64
export TORCH_CUDA_ARCH_LIST='8.0 8.6 9.0 10.0 12.0'   # amd64; see table below
# For an arm64 image instead:
#   export PLATFORM=linux/arm64
#   export TORCH_CUDA_ARCH_LIST='9.0 10.0 10.3 12.0'
```

### `TORCH_CUDA_ARCH_LIST` per CPU arch

These are the lists the published wheels are built with. The **GPU** arch list is
independent of the **CPU** arch — the two differ only because of which machines exist.

| CPU arch  | `TORCH_CUDA_ARCH_LIST`  | GPUs covered                                              |
| --------- | ----------------------- | --------------------------------------------------------- |
| **amd64** | `8.0 8.6 9.0 10.0 12.0` | A100 · A10/A40/RTX 30 · H100 · B100/B200 · RTX PRO/RTX 50 |
| **arm64** | `9.0 10.0 10.3 12.0`    | GH200 · GB200 · GB300 · workstation Blackwell             |

- **Gaps are deliberate, not omissions.** A cubin runs on any *higher minor* revision
  of the same major, so `8.6` also covers Ada (`8.9`) and `10.0` covers `10.3`.
- **arm64 lists no Ampere** — NVIDIA's Grace-based systems pair with Hopper and
  Blackwell only — and lists `10.3` explicitly because GB300 is a Grace part.
- **It is required, not optional.** `docker build` gets no GPU, and
  `torch.cuda.get_arch_list()` returns `[]` without one, so nothing can be
  autodetected. The Dockerfile therefore defaults it to the amd64 row rather than
  leaving it empty, so a bare `docker build` works; override for arm64 or a
  narrower set.

### Standard image — `develop` / `product`

```shell
export TARGET=develop  # or: product
docker build --platform "$PLATFORM" -f docker/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST --target "$TARGET" -t "$IMAGE" .
```

### Air-gapped image — `airgapped-develop` / `airgapped-product`

Bakes the model's checkpoints in, so it needs `buildx` and a named `ckpts`
context (which bypasses `.dockerignore`):

```shell
export TARGET=airgapped-product  # or: airgapped-develop
docker buildx build --platform "$PLATFORM" -f docker/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST --build-context ckpts=./checkpoints --target "$TARGET" --load -t "$IMAGE" .
```

## 2. Publish (optional)

To push to your own registry, tag the image for it and authenticate first:

```shell
export REGISTRY=<your-registry>
docker tag "$IMAGE" "$REGISTRY/$IMAGE"
docker login "$REGISTRY"
docker push "$REGISTRY/$IMAGE"
```

## 3. Run

Use the image as a ready venv and bind-mount the repo over `/workspace/cdf-anomalygen`,
so it runs the current code/config plus the `datasets/` and base `checkpoints/`.
`--user` + the `/etc/passwd` mount keep outputs host-owned and let `getpass.getuser()`
resolve; `--shm-size` feeds the dataloader workers.

Set `IMAGE` to whichever image you are using — the one built above, or a pulled
`nvcr.io/nvidia/cdf-anomalygen:<tag>`.

> **The whole-tree bind mount below is a `develop` workflow.** It mounts the host checkout
> *over* `/workspace/cdf-anomalygen`, which is where the image's own application code lives —
> so the code that runs is the host copy, and the `product` target's read-only application tree
> is shadowed rather than enforced. That is the right trade for development, where running the
> edited checkout is the whole point. If you chose `product` or `airgapped-product` *for* that
> immutability, use the recipe in "Running a product image" below instead.

```shell
DOCKER="docker run --rm --gpus all --shm-size=16g \
  --user $(id -u):$(id -g) -e USER=$(id -un) -e HOME=/tmp -e HF_TOKEN \
  -v $PWD:/workspace/cdf-anomalygen \
  -w /workspace/cdf-anomalygen $IMAGE"

# interactive shell
$DOCKER -it bash

# dry-run smoke test — reference recipe + hydra overrides (20 iters)
$DOCKER bash -c '
  IMAGINAIRE_OUTPUT_ROOT="$PWD/results/dryrun" \
  torchrun --nproc_per_node=1 anomalygen/scripts/texture/train.py \
    --config=cosmos_framework/configs/base/config.py \
    --recipe=ag_config/exp_texture_ft_phone_screen.yaml \
    -- experiment=anomalygen_texture_ft \
       trainer.max_iter=20 trainer.validation_iter=10 checkpoint.save_iter=10'
```

- **Overrides** go after `--` (hydra). Drop the `trainer.*`/`checkpoint.*` overrides
  for a full fine-tune (schedule then comes from the recipe).
- Point `IMAGINAIRE_OUTPUT_ROOT` at a **fresh** dir — the trainer *resumes* from any
  checkpoint already present in the output dir.

### Running a product image

The `product` and `airgapped-product` targets own their application code: it is `chown root` and
`chmod a-w`, so nothing inside the container can modify what runs. Keeping that property means
mounting **data, not the tree** — bind-mount only the directories the pipeline reads and writes,
and let the code come from the image:

```shell
DOCKER_PRODUCT="docker run --rm --gpus all --shm-size=16g \
  --user $(id -u):$(id -g) -e USER=$(id -un) -e HOME=/tmp -e HF_TOKEN \
  -v $PWD/datasets:/workspace/cdf-anomalygen/datasets:ro \
  -v $PWD/checkpoints:/workspace/cdf-anomalygen/checkpoints:ro \
  -v $PWD/results:/workspace/cdf-anomalygen/results \
  -v $PWD/ag_config:/workspace/cdf-anomalygen/ag_config:ro \
  -w /workspace/cdf-anomalygen $IMAGE"

# Verify the property actually holds — this must fail:
$DOCKER_PRODUCT sh -c 'touch anomalygen/__init__.py' && echo "UNEXPECTED: app code is writable"
```

Notes:

- `checkpoints/` is mounted `:ro` so the weights the manifest gate verified cannot be swapped by a
  later step in the same run. Drop `:ro` only if you intend to run `download_checkpoints.sh`
  inside the container.
- `results/` is the one writable mount — it is where the run's outputs belong.
- `ag_config/` is mounted read-only so a recipe can be supplied without making the code tree
  writable. Omit it to use only the recipes baked into the image.
- The air-gapped targets additionally set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so a
  cache miss fails loudly instead of silently resolving a mutable branch over the network.

## Troubleshooting

- **Build looks stuck for an hour** — that is stage 1 compiling flash-attn /
  transformer-engine / apex from source, which is expected. Watch the layer output
  rather than killing it.
- **`flash-attn` / `transformer-engine` / `apex` build OOMs** — lower the job
  count (`--build-arg MAX_JOBS=...`, default 16).
- **Driver / CUDA mismatch** — the image needs a host driver supporting its
  CUDA version. Run `nvidia-smi` on the host to confirm.
- **BuildKit required** — the Dockerfile uses BuildKit-only features (the
  `# syntax` directive, `RUN <<EOF` heredocs, `--build-context`). Docker 23+ has
  BuildKit on by default; on older Docker `export DOCKER_BUILDKIT=1`, or use
  `docker buildx build`.
- **`failed to find stage "..."`** — an unknown `--target` name. Valid targets:
  `develop` (default), `product`, `airgapped-develop`, `airgapped-product`.
