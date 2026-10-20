# lerobot stub

An **import-only stub** for the `lerobot` package. It exposes just
`lerobot.datasets.video_utils.decode_video_frames` (which raises if ever called) — enough to
satisfy imports, nothing functional.

## Why it exists

`cosmos_framework`'s `make_config()` unconditionally imports its droid action-policy config, which
transitively does `from lerobot.datasets.video_utils import decode_video_frames` at module load.
So building **any** experiment config requires `lerobot` to be importable — even though the
anomaly-inpainting recipes never touch action datasets.

Installing the real `lerobot` is not an option: it would **downgrade `torch` off the CUDA 13.2
stack** (and pull a CUDA 12.8 wheel set), breaking flash-attn / apex / natten. The stub satisfies
the import without disturbing the environment.

## How it's wired

`docker/Dockerfile` installs it with the other stubs in one layer, before cosmos-framework;
`scripts/env_setup.sh` and `scripts/env_setup_ci_cpu.sh` install it editable alongside the framework:

```shell
uv pip install -e assets/lerobot_stub --no-deps
```

If you ever genuinely need the real `lerobot` (e.g. to load action-dataset video), install it in a
separate, torch-compatible environment instead of here.
