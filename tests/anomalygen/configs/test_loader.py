# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the experiment-recipe loader (public API used by train.py / generate.py)."""

import sys

import pytest
import yaml
from hydra.core.config_store import ConfigStore

from anomalygen.configs import loader
from anomalygen.configs.loader import _import_object, read_experiment_field, register_recipe


def test_import_object_resolves_dotted_path():
    import os

    assert _import_object("os.path.join") is os.path.join


def test_import_object_requires_module_qualifier():
    with pytest.raises(ValueError):
        _import_object("join")  # no 'module.attr' form


def test_register_recipe_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        register_recipe("/no/such/recipe.yaml")


def test_register_recipe_non_mapping_raises(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]))
    with pytest.raises(ValueError):
        register_recipe(str(path))


def test_register_recipe_missing_required_keys_raises(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"task_type": "texture_ft"}))  # 'experiment' missing
    with pytest.raises(ValueError):
        register_recipe(str(path))


def test_register_recipe_unknown_task_type_raises(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"task_type": "bogus", "experiment": "e"}))
    with pytest.raises(ValueError):
        register_recipe(str(path))


def _texture_recipe(**extra):
    return {
        "task_type": "texture_ft",
        "experiment": "unittest_texture_ft",
        "dataset_name": "unit_test_ds",
        "anomaly_types": [["metal", "scratch"], ["wood", "hole"]],
        "dataset_path": "data/train",  # absolutized as a string; need not exist
        "testcase_jsonl": "tests/example.jsonl",
        **extra,
    }


def test_register_recipe_file_happy_path_registers_real_experiment(tmp_path):
    # The real texture_ft builder imports the cosmos_framework model stack, which pulls in triton
    # (a CUDA dep absent from the CPU-only test env). Skip there; this runs in the full/GPU env.
    pytest.importorskip("triton")

    # Drive the real texture_ft builder through a YAML recipe (config assembly only — no model
    # build or I/O), then read the registered node back from the ConfigStore.
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(_texture_recipe()))

    name = register_recipe(str(path))
    assert name == "unittest_texture_ft"

    cs = ConfigStore.instance()
    assert "unittest_texture_ft.yaml" in cs.list("experiment")

    node = cs.load("experiment/unittest_texture_ft.yaml").node
    # The builder accepts recipe_path, so the loader injects the recipe's own resolved path.
    assert node["trainer"]["callbacks"]["training_report"]["recipe_path"] == str(path.resolve())
    # num_classes is derived from the two supplied anomaly_types pairs.
    assert node["model"]["config"]["diffusion_expert_config"]["anomaly_num_classes"] == 2


# --- legacy recipes ---------------------------------------------------------------------------------
# A legacy file recipe is rewritten into current keys before the builder sees it (configs/texture/legacy.py).


@pytest.fixture
def builder_kwargs(monkeypatch):
    """Stand the texture_ft builder in for one that records its kwargs; the node it stores is removed after."""
    seen = {}

    def _fake_builder(**kwargs):
        seen.update(kwargs)
        return {"model": {"config": {}}}

    monkeypatch.setattr(loader, "_import_object", lambda dotted: _fake_builder)
    yield seen
    ConfigStore.instance().repo.get("experiment", {}).pop("unittest_texture_ft.yaml", None)


def _write(tmp_path, **recipe):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(_texture_recipe(**recipe)))
    return str(path)


def test_a_legacy_recipe_reaches_the_builder_in_current_keys(tmp_path, builder_kwargs, loguru_lines):
    register_recipe(_write(tmp_path, image_size=[512, 512], ratio_range=[1.5, 8.0], model_input_size=512, shift=5.0))
    assert builder_kwargs["input_size_ladder"] == [512]
    assert builder_kwargs["crop_mode"] == "ratio"
    assert builder_kwargs["crop_ratio_range"] == [1.5, 8.0]
    assert builder_kwargs["shift"] == 5
    assert not {"image_size", "ratio_range", "model_input_size"} & set(builder_kwargs)
    assert any("read as a legacy recipe" in line for line in loguru_lines), "the rewrite must be visible in the log"


def test_a_current_recipe_is_handed_over_untouched(tmp_path, builder_kwargs):
    register_recipe(_write(tmp_path))
    assert "crop_mode" not in builder_kwargs and "shift" not in builder_kwargs


def test_legacy_forces_the_legacy_reading_of_a_recipe_with_no_marking_key(tmp_path, builder_kwargs):
    # A legacy checkpoint's recipe may rely on every legacy default; the caller knows, the file cannot show it.
    register_recipe(_write(tmp_path), legacy=True)
    assert builder_kwargs["input_size_ladder"] == [512] and builder_kwargs["shift"] == 5


def test_a_module_recipe_cannot_be_read_as_legacy():
    with pytest.raises(ValueError, match="cannot be read as a legacy recipe"):
        register_recipe("json", legacy=True)


# --- keys the builder does not take -----------------------------------------------------------------
# Checked against the builder's signature before it is called, so a typo is named instead of a bare TypeError.


def test_an_unknown_recipe_key_is_named(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(_texture_recipe(learnig_rate=0.1)))
    with pytest.raises(ValueError, match="does not take: learnig_rate"):
        register_recipe(str(path))


# --- module recipes ---------------------------------------------------------------------------------
# Importing a module recipe runs it, so the namespace is pinned even though --recipe is argv-only.


@pytest.mark.parametrize("recipe", ["os", "antigravity", "evil_pkg.payload", "..sneaky"])
def test_module_recipe_outside_the_allowed_namespaces_is_refused(recipe):
    with pytest.raises(ValueError, match="outside the allowed namespaces"):
        register_recipe(recipe)


@pytest.fixture
def module_recipe(tmp_path, monkeypatch):
    """Write a throwaway module recipe that is importable and inside the allowed namespace.

    The module is dropped from ``sys.modules`` afterwards and the nodes it ``stores`` are removed
    from the global ConfigStore, so a test leaves the process as it found it.
    """
    written = []

    def _write(module, body, stores=()):
        (tmp_path / f"{module}.py").write_text(body)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(loader, "_RECIPE_MODULE_PREFIXES", (module,))
        written.append((module, tuple(stores)))
        return module

    yield _write

    repo = ConfigStore.instance().repo.get("experiment", {})
    for module, stores in written:
        sys.modules.pop(module, None)
        for name in stores:
            repo.pop(f"{name}.yaml", None)


def _stores(experiment, band=(1.5, 2.5)):
    """Body of a module recipe that registers one experiment node carrying ``band``."""
    node = {"model": {"config": {"diffusion_expert_config": {"crop_ratio_range": list(band)}}}}
    return (
        "from hydra.core.config_store import ConfigStore\n"
        f"ConfigStore.instance().store(group='experiment', package='_global_', name={experiment!r}, node={node!r})\n"
    )


def test_a_module_recipe_returns_the_experiment_it_stored(module_recipe):
    module = module_recipe("unit_recipe_a", _stores("unit_exp_a"), stores=["unit_exp_a"])
    assert register_recipe(module) == "unit_exp_a"


def test_a_module_recipe_registered_twice_still_returns_its_name(module_recipe):
    # A second plain import would be a no-op with no node to find, so the module is reloaded and the
    # node rebuilt. Without that, a second caller in the same process silently got nothing.
    module = module_recipe("unit_recipe_b", _stores("unit_exp_b"), stores=["unit_exp_b"])
    repo = ConfigStore.instance().repo["experiment"]
    assert register_recipe(module) == "unit_exp_b"
    first = repo["unit_exp_b.yaml"]
    assert register_recipe(module) == "unit_exp_b"
    assert repo["unit_exp_b.yaml"] is not first


def test_a_module_recipe_that_stores_nothing_is_rejected(module_recipe):
    module = module_recipe("unit_recipe_c", "x = 1\n")
    with pytest.raises(ValueError, match="exactly one experiment node"):
        register_recipe(module)


def test_a_module_recipe_storing_two_nodes_needs_experiment_to_pick_one(module_recipe):
    # A module that stores two variants (or imports a sibling that stores one) cannot name itself;
    # the caller's explicit experiment= picks the node, which is what load_for_inference passes.
    module = module_recipe(
        "unit_recipe_e", _stores("unit_exp_e1") + _stores("unit_exp_e2"), stores=["unit_exp_e1", "unit_exp_e2"]
    )
    with pytest.raises(ValueError, match="exactly one experiment node.*Pass experiment="):
        register_recipe(module)
    assert register_recipe(module, experiment="unit_exp_e2") == "unit_exp_e2"
    with pytest.raises(ValueError, match="did not store experiment 'unit_exp_e3'"):
        register_recipe(module, experiment="unit_exp_e3")


def test_read_experiment_field_reads_the_registered_node(module_recipe):
    module = module_recipe("unit_recipe_d", _stores("unit_exp_d", band=(1.5, 8.0)), stores=["unit_exp_d"])
    name = register_recipe(module)
    assert list(read_experiment_field(name, "diffusion_expert_config.crop_ratio_range")) == [1.5, 8.0]
    assert read_experiment_field(name, "diffusion_expert_config.model_size") is None


def test_file_recipes_are_unaffected_by_the_namespace_check(tmp_path):
    """The allowlist must gate module recipes only; file recipes take the other branch."""
    missing = tmp_path / "no_such_recipe.yaml"
    with pytest.raises(FileNotFoundError):
        register_recipe(str(missing))
