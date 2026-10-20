# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic experiment-recipe loader for ``--recipe`` (Python module or YAML/JSON file).

``register_recipe(recipe)`` accepts either:

- an **importable Python module** (e.g. ``ag_config.exp_texture_ft_phone_screen``) whose import-time
  ``cs.store(...)`` registers the experiment node; or
- a path to a **``.yaml`` / ``.yml`` / ``.json`` file**: a mapping of builder kwargs plus two meta
  keys — ``task_type`` (a friendly name from ``_TASK_TYPE``) and ``experiment`` (the ConfigStore name
  to register under). The selected builder is called with the remaining kwargs (``recipe_path`` is
  injected if the builder accepts it) and its result is stored as the experiment node.

Either way it returns the registered experiment name. To expose a new experiment type to YAML/JSON
recipes, add one entry to ``_TASK_TYPE`` below.

A legacy ``texture_ft`` file recipe (``image_size`` / ``model_input_size`` / ``ratio_range`` keys)
is rewritten into the current spelling before the builder runs; see ``configs.texture.legacy``.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Optional

from cosmos_framework.utils import log
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from anomalygen.configs.texture.legacy import is_legacy_recipe, translate_legacy_recipe

_RECIPE_FILE_SUFFIXES = (".yaml", ".yml", ".json")
# Importing a module recipe runs it. --recipe comes from argv, so this is not a privilege
# boundary today, but a future caller sourcing it from a file or an API would inherit an
# arbitrary-import primitive; pinning the namespaces keeps that from being silent.
_RECIPE_MODULE_PREFIXES = ("ag_config.", "anomalygen.configs.")
_TASK_TYPE: dict[str, str] = {
    "texture_ft": "anomalygen.configs.texture.exp_config.build_anomalygen_texture_ft_experiment",
}


def register_recipe(recipe: str, experiment: Optional[str] = None, *, legacy: bool = False) -> str:
    """Register the experiment node named by ``--recipe`` and return its experiment name.

    ``recipe`` is a YAML/JSON file or a Python module inside an allowed namespace. ``experiment``
    picks the node when a module stores more than one; a file recipe names its own. ``legacy`` reads
    a file recipe as legacy even when no key marks it so: a legacy checkpoint's recipe may rely on
    the legacy defaults throughout. A module recipe cannot be read that way.
    """
    if recipe.endswith(_RECIPE_FILE_SUFFIXES):
        return _register_from_file(recipe, legacy=legacy)
    if legacy:
        raise ValueError(
            f"Recipe module {recipe!r} cannot be read as a legacy recipe; pass the run dir's YAML recipe copy."
        )
    if not recipe.startswith(_RECIPE_MODULE_PREFIXES):
        raise ValueError(
            f"Recipe module {recipe!r} is outside the allowed namespaces "
            f"{list(_RECIPE_MODULE_PREFIXES)}; pass a recipe file "
            f"({', '.join(_RECIPE_FILE_SUFFIXES)}) or a module under one of them."
        )
    return _register_from_module(recipe, experiment)


def read_experiment_field(experiment: str, path: str):
    """``model.config.<path>`` off the registered ``experiment`` node, or ``None`` when it is unset."""
    node = ConfigStore.instance().repo["experiment"][f"{experiment}.yaml"].node
    return OmegaConf.select(node, f"model.config.{path}")


def _register_from_module(module: str, experiment: Optional[str] = None) -> str:
    """Run ``module`` for its ``cs.store`` side effect and return the name of the node it stored.

    The module picks the name itself, so the node is found as the ``experiment`` entry the run added
    or replaced; when it stores several (or imports a sibling that stores one), ``experiment`` says
    which. A module already imported is reloaded: a second import is a no-op with nothing to find,
    and reloading rebuilds the node under the current environment.
    """
    repo = ConfigStore.instance().repo
    before = dict(repo.get("experiment", {}))  # by reference, so a replaced entry still shows as new
    loaded = sys.modules.get(module)
    if loaded is None:
        importlib.import_module(module)
    else:
        importlib.reload(loaded)
    stored = repo.get("experiment", {})
    fresh = sorted(k.removesuffix(".yaml") for k, v in stored.items() if before.get(k) is not v)
    if experiment is not None:
        if f"{experiment}.yaml" not in stored:
            raise ValueError(f"Recipe module {module!r} did not store experiment {experiment!r}; it stored {fresh}.")
        return experiment
    if len(fresh) != 1:
        raise ValueError(
            f"Recipe module {module!r} must store exactly one experiment node when run; "
            f"found {len(fresh)}: {fresh}. Pass experiment= to pick one."
        )
    return fresh[0]


def _register_from_file(path: str, legacy: bool = False) -> str:
    cfg_path = Path(path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {cfg_path}")
    data = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Recipe file {cfg_path} must be a mapping of builder kwargs, got {type(data).__name__}.")
    missing = [k for k in ("task_type", "experiment") if k not in data]
    if missing:
        raise ValueError(
            f"Recipe file {cfg_path} is missing required key(s) {missing}; it must define "
            "'task_type' (one of " + ", ".join(sorted(_TASK_TYPE)) + ") and 'experiment' (ConfigStore node name)."
        )
    task_type = str(data.pop("task_type"))
    target = _TASK_TYPE.get(task_type)
    if target is None:
        raise ValueError(f"Unknown task_type {task_type!r} in {cfg_path}; choose one of {sorted(_TASK_TYPE)}.")
    builder = _import_object(target)
    name = str(data.pop("experiment"))
    if task_type == "texture_ft" and (legacy or is_legacy_recipe(data)):
        data = translate_legacy_recipe(data)
        log.warning(
            f"Recipe {cfg_path} is read as a legacy recipe: fixed {data['input_size_ladder'][0]} px input, "
            f"crop_mode=ratio, crop_ratio_range={data['crop_ratio_range']}, shift={data['shift']}."
        )
    params = inspect.signature(builder).parameters
    # Hand the builder the recipe's own path (for TrainingReport's snapshot) if it accepts one.
    if "recipe_path" in params:
        data.setdefault("recipe_path", str(cfg_path))
    # Name a key the builder does not take rather than letting the call die with a bare TypeError.
    unknown = sorted(set(data) - set(params))
    if unknown and not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        raise ValueError(f"Recipe file {cfg_path} has key(s) {task_type!r} does not take: {', '.join(unknown)}.")
    node = builder(**data)
    ConfigStore.instance().store(group="experiment", package="_global_", name=name, node=node)
    return name


def _import_object(dotted: str):
    """Import ``module.attr`` from a dotted path."""
    module_name, _, attr = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"Builder target must be a dotted 'module.function' path, got {dotted!r}.")
    return getattr(importlib.import_module(module_name), attr)
