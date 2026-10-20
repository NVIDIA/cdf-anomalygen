# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training entry point: register the anomalygen plugin, compose the experiment from the
ConfigStore, then instantiate and train via the framework trainer.

``cosmos_framework`` has no plugin auto-discovery, so the plugin must be registered before
the experiment is composed. Importing ``anomalygen`` registers the reusable library groups
(model / ckpt_type / callbacks). ``--recipe`` then registers the experiment node — it accepts
either a Python recipe module in ``ag_config/`` (imported for its ``cs.store``) or a path to a
``.yaml`` / ``.yml`` / ``.json`` recipe file (fed to its builder via ``anomalygen.configs.loader``).
We then follow the framework's ConfigStore launch path: ``make_config()`` builds the base
``Config`` and registers the default groups, ``override()`` hydra-composes the chosen
``experiment=...`` plus any ``key=value`` overrides on top, and ``trainer.train()`` runs.

Usage (single host):
    torchrun --nproc_per_node=$NPROC anomalygen/scripts/texture/train.py \
        --config=cosmos_framework/configs/base/config.py \
        --recipe=ag_config/exp_texture_ft_phone_screen.yaml -- \
        experiment=anomalygen_texture_ft trainer.max_iter=200
"""

from __future__ import annotations

import os

# Keep the base model's HF-resolved caption tokenizer under the repo's checkpoints/hf, anchored to
# the repo root rather than the CWD, mirroring generate.py. A user-provided HF_HUB_CACHE is respected.
os.environ.setdefault(
    "HF_HUB_CACHE",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "checkpoints", "hf")),
)

# Framework process setup (training env, distributed error handling). Mirrors
# cosmos_framework.scripts._train; must run before the heavy imports below.
from cosmos_framework.inference.common.init import init_script

init_script(training=True, env={"COSMOS_TRAINING": "1"}, default_env={"COSMOS_VERBOSE": "1"})

import argparse
import contextlib
import importlib

import hydra
from cosmos_framework.inference.common.config import ROOT_DIR  # noqa: E402
from cosmos_framework.utils import distributed, log  # noqa: E402
from cosmos_framework.utils.config_helper import get_config_module, override  # noqa: E402
from cosmos_framework.utils.flags import INTERNAL  # noqa: E402

import anomalygen  # noqa: F401,E402  (registers experiment / model / ckpt_type / callbacks)
from anomalygen.configs.loader import register_recipe


def launch(config, dry_run: bool = False) -> None:
    # Distributed must be up before config.validate() (it syncs a buffer across ranks) and
    # before the trainer constructor, which uses rank-0 guards and a barrier.
    distributed.init()
    config.validate()
    config.freeze()  # type: ignore[attr-defined]

    if dry_run:
        log.info("Dry run — config validated, skipping training.")
        return

    # Persist the console log to the run dir as stdout.log. The framework's trainer writes this
    # file only when its INTERNAL flag is set, so add the sink here otherwise, and skip it when
    # the framework already writes it. Added before the trainer/model build so it captures the
    # (slow, one-time) torch.compile warmup. loguru's sink filters to rank 0.
    if not INTERNAL and distributed.is_rank0():
        os.makedirs(config.job.path_local, exist_ok=True)
        log.init_loguru_file(f"{config.job.path_local}/stdout.log")

    # Instantiate with cwd at cosmos_framework's parent dir: the model build reads packaged
    # resources by "cosmos_framework/..."-prefixed paths relative to it (e.g. the Qwen3-VL
    # config JSON and the tokenizers/ tree). ROOT_DIR is the package dir, so its parent is
    # site-packages for the pip-installed framework.
    # Our dataset / checkpoint / VAE paths are absolute and unaffected by this cwd.
    with contextlib.chdir(ROOT_DIR.parent):
        trainer = config.trainer.type(config)
        model = hydra.utils.instantiate(config.model)
        dataloader_train = hydra.utils.instantiate(config.dataloader_train)
        dataloader_val = hydra.utils.instantiate(config.dataloader_val)

    # The dataset scanned the training masks; hand the model the per-class input-size caps it applies,
    # so the checkpoint carries them and generation plans each class against the sizes it trained on.
    # (PackingDataLoader -> RankPartitionedDataLoader -> InpaintingDataset, as exp_config wires them.)
    dataset = dataloader_train.dataloader_list[0].dataset
    model.class_ladders = dataset.class_input_size_ladders
    # A resume restarts the dataloader workers; seeding them from the iteration it continues at gives
    # them a stream the run has not drawn yet, not a replay of its first iterations. The checkpointer
    # restores every other generator itself.
    dataset.start_iteration = trainer.checkpointer.resume_iteration()

    trainer.train(model, dataloader_train, dataloader_val)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="AnomalyGen I2I anomaly-inpainting training")
    parser.add_argument("--config", required=True, help="Path to the Python config module (make_config lives here).")
    parser.add_argument(
        "--recipe",
        default="ag_config/exp_texture_ft_phone_screen.yaml",
        help="Experiment recipe: an importable Python module (Cosmos recipes) or a path to a .yaml/.yml/.json recipe "
        "file.",
    )
    parser.add_argument("--dryrun", action="store_true", help="Validate the composed config without training.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help='Overrides after a "--", e.g. -- experiment=anomalygen_texture_ft'
    )
    args = parser.parse_args(argv)

    register_recipe(args.recipe)

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)

    launch(config, dry_run=args.dryrun)


if __name__ == "__main__":
    main()
