# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the AnomalyGen generation pipeline from one YAML config.

    python -m anomalygen.scripts.run_pipeline --config run.yaml

Six stages, each writing under ``output``. Exactly one of the two mask stages runs; every stage can
be skipped, so a run can resume from whatever is already on disk::

    amp       roi_place_pipeline          -> <output>/amp/testcase.jsonl
    amg       agentic_mask_generation     -> <output>/amg/testcase.jsonl   (mutually exclusive with amp)
    generate  generate.py                 -> <output>/sdg/reconstructed_image/
    evaluate  evaluate.py                 -> <output>/sdg/kpi.json
    refine    quality_refine.py           -> <output>/searched/
    export    (here)                      -> <output>/export/<clean_stem>.png

The stages are the ones ``skills/cdf-anomalygen/SKILL.md`` documents as Steps 2 and 4-6; this drives
them from a config file instead of six shell invocations, and adds the export step, which has no
CLI of its own.

ROUTE-AGNOSTIC
No route is decided here. Which one a defect takes is a property of its ``defect_spec.jsonl``
row, and the only reason this file reads ``spatial_dependency`` at all is to predict which models
a stage will load, which is provisioning rather than routing. The two knobs a THT run happens to
need -- ``per_defect_counts``
to name one class and ``amp_per_clean_pairing`` to give every clean image exactly one output -- are useful
to any dataset whose clean images are work items rather than a pool of backdrops.

WHY EXPORT EXISTS
``generate.py`` names outputs ``<anomaly_type>_<index>.png``, which loses the identity of the crop
they came from. A flow that composites results back onto a board needs the source name. The
generation CSV keeps ``image_filename`` per row, so the mapping is recoverable, and this stage
applies it. Two outputs claiming one source name is an error, not something to disambiguate
silently: it means the batch does not have the one-output-per-crop shape the caller asked for.

TWO MASK BACKENDS, ONE ROW SHAPE
``amp`` and ``amg`` write the same six-column ``testcase.jsonl``; the tail stages cannot tell them
apart. ``amp`` also writes the sampling knobs per row and ``amg`` does not, so the ``amg`` stage
back-fills the configured ones afterwards -- otherwise ``generate.py`` would fall back to the shared
defaults and the config's ``guidance`` would be ignored without a word.

WHERE THE MODEL ENDPOINT COMES FROM
Two ways, chosen by the config alone. ``amg_model_registry`` names a file that says where a model
is already serving -- a cloud API, or a shared server the orchestrator started. ``amg_serve`` names
a recipe under ``scripts/agentic_mask_generation_serving/`` for this run to start itself: the
server takes the first ``tp`` visible GPUs, the amg stage runs against it, and it is stopped before
generate starts, so the pod needs ``max(tp, num_gpus)`` GPUs rather than their sum. Nothing is
inferred from a model's name or URL -- a ``localhost`` registry entry may well be a server the
operator started by hand, and starting a second copy would only OOM the first. A SIGTERM (an
orchestrator cancelling the job) is turned into an exception before any stage runs, so a server this
run started is still stopped on the way out rather than orphaned on its GPUs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import shutil
import signal
import sys
import uuid
from dataclasses import MISSING, asdict, dataclass, field, fields

import yaml
from cosmos_framework.utils import log

from anomalygen import catalogue, childproc, serving
from anomalygen.configs.texture.constants import DEFAULT_MODEL_SIZE, MODEL_SIZES

_DEFECT_SPEC_FILE = "defect_spec.jsonl"

# Anchored to this file, not the CWD: under an orchestrator (Airflow's DockerOperator, a k8s Job)
# the working directory is whatever the image declares, and a relative "anomalygen/scripts/..." then
# resolves to nothing. Every weight path in the codebase is anchored the same way.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TEXTURE = pathlib.Path(__file__).resolve().parent / "texture"

# The two mask-placement backends are mutually exclusive: both write a testcase.jsonl, and
# generate must never have to guess which one it is reading.
_MASK_STAGES = ("amp", "amg")
STAGES = ("prepare", "amp", "amg", "generate", "evaluate", "refine", "export", "publish")
# The default run: one mask stage (the classic one) and no staging. The staging stages need
# locations the config may not carry, and amg needs a model endpoint, so a run that does not ask
# for them must not get them.
DEFAULT_STAGES = tuple(s for s in STAGES if s not in ("prepare", "amg", "publish"))

# Written by generate.py and, with the same name and columns, by quality_refine.py, so export
# reads whichever bucket it is pointed at.
_GEN_CSV = "texture_ft_generation_result.csv"
_BLOCKED_CSV = "guardrail_blocked.csv"

# Stage -> the stages whose output it reads. A tuple inside the tuple is a group of alternatives: any
# one of them satisfies the dependency. A skipped dependency is fine when its artifact is already on
# disk, which is what makes a resume possible; the artifact check below is the real gate.
_DEPENDS_ON = {
    "generate": (_MASK_STAGES,),
    "evaluate": ("generate",),
    "refine": (_MASK_STAGES, "generate", "evaluate"),
    "export": ("generate",),
    "publish": ("export",),
}


def _skipped_dependencies(stage, selected) -> list:
    """The dependencies of ``stage`` that are not selected, an alternatives group rendered ``a|b``."""
    skipped = []
    for dep in _DEPENDS_ON.get(stage, ()):
        alternatives = dep if isinstance(dep, tuple) else (dep,)
        if not any(a in selected for a in alternatives):
            skipped.append("|".join(alternatives))
    return skipped


_REQUIRED_FOR = {  # stages that cannot run without these keys
    "generate": ("checkpoint",),
    "refine": ("checkpoint",),
    "amg": ("amg_gen_model", "amg_judge_model"),
}


@dataclass
class Config:
    """One run's settings. Field defaults ARE the config defaults, and a field without one is
    required -- so the key list, the defaults and what is mandatory cannot drift apart.

    ``defect_counts`` is deliberately a property rather than a field: it is derived from
    ``per_defect_counts``, so it cannot go stale, and ``asdict`` records what the run was asked
    for rather than a value that is really an implementation detail of the amp stage.
    """

    input: str  # dataset root, in the Step 1 layout
    output: str  # every stage writes under here
    recipe: str

    checkpoint: str | None = None  # required by the generate and refine stages
    base_checkpoint: str | None = None  # None -> generate.py picks it from the recipe's model_size

    num_sdg: int | None = None  # required unless the amp stage is skipped
    per_defect_counts: dict | None = None  # None -> every type in defect_spec, allocated by roi_allocate
    mode: str = "inference"  # inference | validation
    amp_per_clean_pairing: str = "off"  # on | off

    guidance: float | None = None  # None -> the shared texture constants
    num_steps: int | None = None
    crop_ratio: float | None = None

    # Vary BOTH per task when a run emits one image: roi_pair keys its submask shuffle on
    # (seed, defect_type) alone and a one-record run always takes submask 0 of it, so a fan-out at a
    # fixed seed gives every task the same base mask and, with index 0, the same noise. Key them on
    # the trigger as well as the work item if re-triggering one item should differ.
    # See ag_config/pipeline/example.yaml.
    amp_seed: int = 42  # placement
    base_seed: int = 1  # diffusion noise -- a different roll of the dice, not a different placement

    num_search_run: int = 3
    num_gpus: int = 1
    guardrail: bool = True
    output_naming: str = "source"  # source | indexed

    # ── amg backend (read only when `amg` is in stages) ──
    amg_gen_model: str | None = None  # registry name of the model that proposes / refines
    amg_judge_model: str | None = None  # registry name of the model that scores
    amg_model_registry: str | None = None  # the amg CLI's --model-registry file: [{model, base_url, key}]
    # None -> <output>/amg_cache; point every run of one dataset at a shared dir to reuse the
    # distilled defect library
    amg_cache_dir: str | None = None
    # A recipe under scripts/agentic_mask_generation_serving/ (e.g. gemma-4-31b-it): run_pipeline
    # starts that server before the amg stage, waits for it, points the CLI at it and stops it when
    # the stage ends. None -> the registry names an endpoint that is already serving.
    amg_serve: str | None = None
    # A small local model needs the in-loop beam and the three information channels; a strong cloud
    # model does not. All off by default so a cloud run's prompts stay byte-identical to the CLI's.
    amg_beam_multi: int = 0  # 0 = off; the Gemma recipe uses 4
    amg_beam_pairwise: bool = False
    amg_exemplar_captions: bool = False
    amg_exemplar_visual_prompt: bool = False
    amg_coverage_note: bool = False

    # Sources and sinks, read by the staging stages and by nothing else. Never put a credential
    # here: main() writes this config verbatim into the output directory, which is the tree publish
    # uploads. Credentials come from the environment (see anomalygen.staging.store).
    #
    #   fetch   s3:// msc:// hf:// ngc://, or a local path already on the node
    #   publish s3:// msc:// only -- see store.py for why a registry is never an upload destination
    staging: dict | None = None

    stages: list = field(default_factory=lambda: list(DEFAULT_STAGES))

    @property
    def defect_counts(self) -> dict | None:
        """``{type: count}`` for ``--per_defect_counts``, validated at load time."""
        return normalise_per_defect_counts(self.per_defect_counts)


def _field_names() -> set:
    return {f.name for f in fields(Config)}


def _required_names() -> list:
    """The fields with no default -- the dataclass's own notion of mandatory."""
    return [f.name for f in fields(Config) if f.default is MISSING and f.default_factory is MISSING]


def _resolve_recipe(name, where=""):
    """The serving recipe ``amg_serve`` names, or a SystemExit that says what is wrong with the name.

    The one place a recipe is turned from a config value into a Recipe. load_config calls it
    eagerly so a bad value fails before any stage; _serve_recipe calls it defensively for a Config
    built without load_config (a test, a direct caller), so both fail the same way -- and both name
    the bad value and what is available, where the parser's own KeyError would name neither.
    """
    prefix = f"{where}: " if where else ""
    if name not in serving.recipe_names():
        raise SystemExit(f"{prefix}amg_serve {name!r} is not a serving recipe; available: {serving.recipe_names()}")
    try:
        return serving.recipe(name)
    except RuntimeError as e:
        raise SystemExit(f"{prefix}amg_serve {name}: {e}") from None


def _serve_recipe(cfg):
    """The serving recipe this run starts, or None: unset, or no amg stage to serve."""
    if not cfg.amg_serve or "amg" not in cfg.stages:
        return None
    return _resolve_recipe(cfg.amg_serve)


def _effective_registry(cfg, paths) -> pathlib.Path:
    """The registry the CLI receives: only the models THIS run names, plus the served model.

    Written beside ``amg/``, not inside it: the CLI treats a non-empty out-dir as a run to resume,
    and refuses when it finds no ``run.json`` there. The served entry wins a clash: a config that
    both starts a server and names the same model elsewhere has asked for the server.

    Only ``amg_gen_model`` / ``amg_judge_model`` survive from the user's registry, not every entry
    in it: the resolved config is handed over with the output tree, and a key for a model this run
    never calls has no business riding along in a file the tree ships. World- and group-unreadable
    for the same reason -- it can carry a live API key.

    The served entry's key is whatever ``API_KEY`` the environment holds: the recipe script locks
    the server with that same variable, so a run that inherits it must present it, and one that does
    not gets the open-server default the CLI already uses.
    """
    recipe = _serve_recipe(cfg)
    if not recipe:
        return pathlib.Path(cfg.amg_model_registry)
    wanted = {cfg.amg_gen_model, cfg.amg_judge_model}
    entries = list(_read_registry(cfg.amg_model_registry).values()) if cfg.amg_model_registry else []
    entries = [e for e in entries if e.get("model") != recipe.served and e.get("model") in wanted]
    key = os.environ.get("API_KEY") or "EMPTY"
    entries.append({"model": recipe.served, "base_url": f"http://localhost:{recipe.port}/v1", "key": key})
    target = pathlib.Path(cfg.output) / "amg_endpoints.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(entries, sort_keys=False))
    os.chmod(target, 0o600)
    return target


# ── config ────────────────────────────────────────────────────────────────────


def load_config(path) -> Config:
    """Read and validate the YAML config.

    Unknown keys are rejected rather than ignored. A typo in an optional key would otherwise leave
    the default silently in force, and the run would look correct while doing something else --
    ``guidence: 3.0`` is the whole failure.
    """
    raw = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level, got {type(raw).__name__}")

    unknown = sorted(set(raw) - _field_names())
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) {unknown}. Known keys: {sorted(_field_names())}")

    # Checked before constructing, because a missing field would otherwise surface as argparse-style
    # noise about a keyword argument rather than as the name of the key to add.
    missing = [k for k in _required_names() if not raw.get(k)]
    if missing:
        raise SystemExit(f"{path}: missing required key(s) {missing}")
    cfg = Config(**raw)

    if isinstance(cfg.stages, str):
        cfg.stages = [cfg.stages]
    bad = [s for s in cfg.stages if s not in STAGES]
    if bad:
        raise SystemExit(f"{path}: unknown stage(s) {bad}. Known stages, in run order: {list(STAGES)}")
    cfg.stages = [s for s in STAGES if s in cfg.stages]  # canonical order, deduplicated

    selected_mask = [s for s in _MASK_STAGES if s in cfg.stages]
    if len(selected_mask) > 1:
        raise SystemExit(
            f"{path}: stages lists both mask backends {selected_mask}; pick one. Both write a "
            "testcase.jsonl, and generate must not have to guess which one to read."
        )

    if cfg.amg_serve:
        recipe = _resolve_recipe(cfg.amg_serve, where=path)
        if "amg" not in cfg.stages:
            log.info(f"{path}: amg_serve is set but the amg stage is not selected, so no server is started")
        else:
            cfg.amg_gen_model = cfg.amg_gen_model or recipe.served
            cfg.amg_judge_model = cfg.amg_judge_model or recipe.served
    if "amg" in cfg.stages and not cfg.amg_model_registry and not _serve_recipe(cfg):
        raise SystemExit(
            f"{path}: stage 'amg' requires amg_model_registry (an endpoint that is already serving) "
            "or amg_serve (a recipe for run_pipeline to start)."
        )

    # Checked here and not in the stage: publish runs LAST, so a destination typo found at stage
    # time costs the whole generation. This is the same reason check_weights runs before any stage.
    if "publish" in cfg.stages:
        destination = _publish_destination(cfg)
        if not destination:
            raise SystemExit(
                f"{path}: stage 'publish' requires staging.publish.output (an object-store prefix). "
                "Drop 'publish' from stages to keep the results local."
            )
        from anomalygen.staging import check_uri

        try:
            check_uri(destination)
        except ValueError as exc:
            raise SystemExit(f"{path}: staging.publish.output — {exc}") from exc

    _check_fetch_spec(cfg, path)
    _warn_unused_fetch(cfg, path)

    for stage, keys in _REQUIRED_FOR.items():
        if stage in cfg.stages:
            absent = [k for k in keys if not getattr(cfg, k)]
            if absent:
                raise SystemExit(f"{path}: stage '{stage}' requires {absent}")

    if cfg.output_naming not in ("source", "indexed"):
        raise SystemExit(f"{path}: output_naming must be 'source' or 'indexed', got {cfg.output_naming!r}")
    cfg.amp_per_clean_pairing = _on_off(cfg.amp_per_clean_pairing, "amp_per_clean_pairing", path)
    if "amg" in cfg.stages:
        if cfg.amp_per_clean_pairing == "on":
            raise SystemExit(
                f"{path}: amp_per_clean_pairing: on is an amp pairing rule; amg has no pairing step and "
                "places by model judgment, so it cannot promise one output per clean image. Drop the key."
            )
        if "amp_seed" in raw:
            log.info(f"{path}: amp_seed is set but the amg stage does not use it (placement is by model judgment)")
    if cfg.mode not in ("inference", "validation"):
        raise SystemExit(f"{path}: mode must be 'inference' or 'validation', got {cfg.mode!r}")

    counts = normalise_per_defect_counts(cfg.per_defect_counts)  # raises here rather than mid-stage

    # roi_allocate rejects --per_defect_counts outside --mode inference, four stage-frames deep.
    if counts is not None and cfg.mode != "inference":
        raise SystemExit(
            f"{path}: per_defect_counts selects specific types, which roi_allocate accepts only "
            f"with mode: inference (got mode: {cfg.mode}). A validation set is allocated "
            "proportionally over every type in defect_spec.jsonl, so drop per_defect_counts."
        )
    # An explicit mapping already states the total, so num_sdg may be omitted. When both are given
    # and disagree, roi_allocate uses the mapping's sum -- so num_sdg is adopted here rather than
    # enforced, to keep this entrypoint's rule the same as the CLI's. Two things make the override
    # safe: it is announced BEFORE any stage runs, not from inside a stage's log, and cfg is
    # normalised to the effective total, so the resolved config records the count actually
    # generated instead of the one that was asked for.
    if counts is not None:
        total = sum(counts.values())
        if cfg.num_sdg not in (None, total):
            log.warning(
                f"{path}: num_sdg is {cfg.num_sdg} but per_defect_counts sums to {total}; "
                f"the counts win, so this run generates {total} image(s), not {cfg.num_sdg}."
            )
        cfg.num_sdg = total

    if cfg.num_sdg is not None:
        try:
            cfg.num_sdg = int(cfg.num_sdg)
        except ValueError:
            raise SystemExit(f"{path}: num_sdg must be an integer, got {cfg.num_sdg!r}") from None

    try:
        cfg.num_gpus = int(cfg.num_gpus)
    except (TypeError, ValueError):
        raise SystemExit(f"{path}: num_gpus must be an integer, got {cfg.num_gpus!r}") from None

    selected_mask_stage = next((s for s in _MASK_STAGES if s in cfg.stages), None)
    if selected_mask_stage and not cfg.num_sdg:
        raise SystemExit(f"{path}: stage '{selected_mask_stage}' requires num_sdg")

    # A path staging.fetch will provide does not exist YET, and saying so would make every remote
    # run fail at load. The rest is still checked here, and the deferred ones are re-checked the
    # moment prepare has run -- so nothing is merely trusted, it is only checked later.
    deferred = set(_fetch_spec(cfg)) - {OVERRIDE_KEY}  # not a config field; it names paths inside one
    _check_paths(cfg, path, deferred=deferred if "prepare" in cfg.stages else frozenset())
    return cfg


def _check_paths(cfg, path, deferred=frozenset()) -> None:
    """Fail on a path that does not exist, before a stage pays to find out.

    A typo'd checkpoint costs a model load (~25 s) to discover; a typo'd defect type costs a whole
    AMP run and reports success, because roi_allocate only warns about a type it does not recognise
    and then allocates it zero images.

    ``deferred`` names keys a later stage will provide, which are skipped here and checked once it
    has. The message is the same either way, so a fetch that landed the wrong layout reads as the
    missing path it is rather than as a staging error.
    """
    # base_checkpoint is here because it is fetchable: prepare can write it, and a fetch that landed
    # the wrong layout is otherwise reported by the weights preflight as a missing WEIGHT rather than
    # as the path it is.
    for key in ("input", "recipe", "checkpoint", "base_checkpoint", "amg_model_registry"):
        value = getattr(cfg, key)
        if key in deferred:
            continue
        if key in ("checkpoint", "base_checkpoint") and not any(s in cfg.stages for s in ("generate", "refine")):
            continue  # not needed by the selected stages, so not this run's problem
        if key == "amg_model_registry" and "amg" not in cfg.stages:
            continue
        if value and not pathlib.Path(value).exists():
            raise SystemExit(f"{path}: {key} does not exist: {value}")

    if "input" in deferred:
        return  # both branches below read the dataset prepare has yet to fetch; re-checked after it
    if "amp" in cfg.stages:
        spec = pathlib.Path(cfg.input) / _DEFECT_SPEC_FILE
        if not spec.is_file():
            raise SystemExit(f"{path}: input has no defect_spec.jsonl: {spec}")
        if not cfg.defect_counts:
            return
        known = {json.loads(line)["defect_type"] for line in spec.read_text().splitlines() if line.strip()}
        unknown = sorted(set(cfg.defect_counts) - known)
        if unknown:
            raise SystemExit(f"{path}: per_defect_counts names {unknown}, not in {spec}. Available: {sorted(known)}")
    elif "amg" in cfg.stages and cfg.defect_counts:
        # amg reads the mask pairs, not the spec, so the types it can place are the ones on disk.
        known = set(_mask_types_on_disk(cfg.input))
        unknown = sorted(set(cfg.defect_counts) - known)
        if unknown:
            raise SystemExit(
                f"{path}: per_defect_counts names {unknown}, which have no {cfg.input}/<texture>/mask/<defect>/ "
                f"directory. Available: {sorted(known)}"
            )


def _on_off(value, key, path) -> str:
    """Accept ``on``/``off`` from YAML, which parses both as booleans (YAML 1.1).

    ``amp_per_clean_pairing: on`` is the natural spelling and the one the CLI takes, but yaml.safe_load
    hands back ``True`` for it. Rejecting that would make the config demand quotes for no reason a
    reader could see.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    if value in ("on", "off"):
        return value
    raise SystemExit(f"{path}: {key} must be 'on' or 'off', got {value!r}")


def _mask_types_on_disk(input_dir) -> list:
    """Every ``<texture>+<defect>`` that has a ``<input>/<texture>/mask/<defect>/`` directory.

    The amg backend's notion of "the types in this dataset": it reads the mask pairs directly and
    never consults defect_spec.jsonl, so this is the list its ``--anomaly-types`` is drawn from.
    """
    root = pathlib.Path(input_dir)
    types = []
    for mask_dir in sorted(root.glob("*/mask/*")):
        if mask_dir.is_dir():
            types.append(f"{mask_dir.parents[1].name}+{mask_dir.name}")
    return types


def normalise_per_defect_counts(per_defect_counts):
    """``{type: count}`` for ``--per_defect_counts``, or None for "every type in defect_spec".

    The same model roi_allocate has, deliberately: it offers no other type selection -- a type
    absent from the mapping is allocated zero -- so a second, friendlier key here (a list of types
    to split num_sdg between) would be a concept with nothing under it, and one more pair of config
    values that could disagree.
    """
    if per_defect_counts is None:
        return None
    if not isinstance(per_defect_counts, dict):
        raise SystemExit(
            f"per_defect_counts must be a mapping of defect type to count, got {type(per_defect_counts).__name__}"
        )
    if not per_defect_counts:
        raise SystemExit("per_defect_counts is empty; omit it to use every type in defect_spec.jsonl")
    return {str(k): int(v) for k, v in per_defect_counts.items()}


# ── weights ───────────────────────────────────────────────────────────────────

# Every weight below is resolved from a path baked into the code, not from a flag: DINOv2 and
# C-RADIO have no override at all (``anomalygen/models/vision_encoder/dinov2.py``,
# ``anomalygen/eval/fid.py``), and the base DCP and VAE resolve against the repo root
# (``configs/texture/exp_config.py``). So under an orchestrator the weights are MOUNTED at
# ``<repo>/checkpoints`` rather than passed in, and what this file can usefully do is say so early.
#
# A check, never a download. ``scripts/download_checkpoints.sh`` is the provisioning tool and belongs
# to image build or volume setup: it writes unconditionally (HF ``refs/main``, and an ``rm -rf`` of
# the Edge ``trees/``), which fails outright on a read-only mount and races other containers on a
# shared one; it re-hashes ~30 GB against the integrity manifests on every invocation; and when
# something IS absent it starts a ≈ 140 GB pull inside what should be a fast-failing task.


def _hf_cache() -> pathlib.Path:
    """The HF cache generate.py will use: an inherited ``HF_HUB_CACHE``, else ``<repo>/checkpoints/hf``.

    Mirrors the ``os.environ.setdefault`` at the top of generate.py, so this checks the directory the
    run will actually read rather than a second guess at it.
    """
    env = os.environ.get("HF_HUB_CACHE")
    return pathlib.Path(env) if env else _REPO_ROOT / "checkpoints" / "hf"


def _hf_missing(repo) -> list:
    """The probe files absent from every snapshot of ``repo``.

    Any snapshot rather than a pinned revision: the revision lives in the framework's checkpoint
    registry, and importing that here would pull torch into a check whose whole point is to run
    before anything heavy does.
    """
    entry = _hf_cache() / f"models--{repo.replace('/', '--')}" / "snapshots"
    probes = catalogue.hf_probes(repo)
    if not probes:
        raise SystemExit(
            f"no probe files are known for HF repo {repo!r}. Add it to scripts/weights_catalogue.sh, "
            "which is what provisions it and what both preflights check."
        )
    snapshots = sorted(entry.iterdir()) if entry.is_dir() else []
    best = list(probes)
    for snap in snapshots:
        absent = [f for f in probes if not (snap / f).exists()]
        if not absent:
            return []
        best = min(best, absent, key=len)
    return [f"{repo}: {f}" for f in best]


def _model_size(recipe) -> str:
    """The recipe's ``model_size``, lower-cased and checked. Unreadable -> the shared default.

    Normalised HERE because this is the only place the value is produced, and every reader of it
    wants the same spelling. ``model_size`` is free text in a recipe, and the readers disagreed about
    what to do with it: ``catalogue.base_group`` lower-cased, the caption-tokenizer branch compared
    ``size == "edge"`` raw. So ``model_size: Nano`` preflighted as base-nano, provisioned 30 GB, and
    then failed hours later inside generate.py -- which checks the same value against MODEL_SIZES and
    will not have it -- while ``model_size: Edge`` asked for the Edge base checkpoint and the *nano*
    caption tokenizer, a combination nothing would have caught.

    Checked against MODEL_SIZES rather than against the catalogue's base-<size> rows: generate.py is
    what ultimately rejects it, so failing on its list is what makes this preflight and not a second
    opinion. A ``.py`` recipe is a module, not something to parse here, and guessing wrong would only
    make this name the wrong base directory -- so it falls back rather than failing.
    """
    path = pathlib.Path(recipe)
    if path.suffix.lower() not in (".yaml", ".yml", ".json"):
        return DEFAULT_MODEL_SIZE
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return DEFAULT_MODEL_SIZE
    raw = loaded.get("model_size")
    if raw is None:
        return DEFAULT_MODEL_SIZE
    size = str(raw).strip().lower()
    if size not in MODEL_SIZES:
        raise SystemExit(
            f"{path}: model_size is {raw!r}, which is not one of {list(MODEL_SIZES)}. generate.py "
            "rejects it too, but only after the mask stage has run and the weights have been "
            "provisioned."
        )
    return size


def base_checkpoint(cfg) -> pathlib.Path:
    """The base DCP this run loads, ABSOLUTE, and the one thing both the preflight and the CLI use.

    The two used to resolve it against different roots. This preflight anchored on the repository;
    generate.py, given no ``--base_checkpoint``, falls back to the relative
    ``BASE_CHECKPOINT_PATHS[size]`` and then ``os.path.abspath``s it -- against the WORKING
    DIRECTORY. Under an orchestrator whose cwd is not the repository root, which is the case this
    entrypoint exists for, the preflight passed on <repo>/checkpoints/Cosmos3-Nano and generate
    failed seconds later on <cwd>/checkpoints/Cosmos3-Nano -- or, if some other tree happened to sit
    there, loaded that one without a word. So the stages pass this value explicitly and the loader
    never reaches its own default.

    A ``base_checkpoint:`` the CONFIG names resolves against the working directory, like every other
    path a config gives -- ``input``, ``recipe``, ``checkpoint`` are all read that way, and so is the
    destination ``prepare`` fetches this one to. It was anchored on the repository here instead,
    which put the fetch and the loader on different roots: prepare downloaded the DCP to
    <cwd>/checkpoints/ft and generate looked for <repo>/checkpoints/ft. Usually that surfaces as the
    weights preflight reporting missing what prepare has just said it fetched; where a tree happens
    to sit at the repo path, it loads that one instead and says nothing.

    Only the DEFAULT is repo-anchored, because it is not a config value at all -- it is the
    catalogue's own location for the base checkpoint this repository provisions.
    """
    if cfg.base_checkpoint:
        return pathlib.Path(cfg.base_checkpoint).resolve()
    group = catalogue.base_group(_model_size(cfg.recipe))
    return _REPO_ROOT / "checkpoints" / catalogue.entry(group).location


def _text_routed(cfg) -> bool:
    """Whether any defect this run will place takes the text2roi route.

    Mirrors roi_place's own reading of the spec, including its default: a row with no
    ``spatial_dependency`` is **text**-routed (``roi_place.py``), not free. Defaulting the other way
    would let a spec that does reach text2roi pass this check.

    Only the selected types count -- ``per_defect_counts`` naming one cad_tht class means the text
    rows in the same spec are allocated zero and never placed.
    """
    spec = pathlib.Path(cfg.input) / _DEFECT_SPEC_FILE
    if not spec.is_file():
        return False
    selected = cfg.defect_counts
    for line in spec.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if selected is not None and not selected.get(entry["defect_type"]):
            continue
        if entry.get("spatial_dependency", "text") == "text":
            return True
    return False


def _needed(label, group, location=None) -> tuple:
    """One catalogue row as ``required_weights`` returns it: ``(label, path-or-repo-id, group)``.

    The label is the only thing this file still decides. Where the weight lives, what counts as it
    being there and which provisioning block writes it are the catalogue's, so the check cannot
    describe a location the provisioner does not write to -- which is how the guardrail probes and
    the ``base-<Size>`` group name came apart.
    """
    w = catalogue.entry(group, location)
    # _REPO_ROOT read here rather than folded into a module constant: the preflight tests point it
    # at a tmp tree, and a constant built at import time would have them checking the real one.
    return (label, w.location if w.kind == "hf" else _REPO_ROOT / "checkpoints" / w.location, group)


def required_weights(cfg) -> list:
    """``[(label, path-or-marker, group)]`` the SELECTED stages will load.

    Only what this run needs: an amp-only run loads no model at all, and an evaluate-only run has no
    use for the 30 GB base checkpoint. Demanding the full tree would make a legitimate partial run
    fail on weights it never opens.

    ``group`` names the ``download_checkpoints.sh`` block that provisions the weight, so prepare can
    ask for those blocks instead of all eleven -- Gemma alone is 62.6 GB and no stage here opens it.
    A wrong or missing mapping cannot corrupt anything: the provisioning re-checks the same list
    afterwards and fails naming what is still absent.
    """
    needed, size = [], _model_size(cfg.recipe)
    generating = {"generate", "refine"} & set(cfg.stages)
    scoring = {"evaluate", "refine"} & set(cfg.stages)

    # text2roi derives its ROI with a grounding VLM and SAM2; cad, cad_tht and free need neither, so
    # a THT run must not be asked for 30 GB of VLM it never opens. roi_place loads both unguarded, so
    # a text-routed run without them dies seconds in -- exactly what this check exists to pre-empt.
    if "amp" in cfg.stages and _text_routed(cfg):
        needed.append(_needed("text2roi grounding VLM", "cosmos-vlm"))
        needed.append(_needed("text2roi segmentation (SAM2.1)", "sam2"))

    # amg segments each candidate with SAM2 and ranks exemplars with DINOv2; both load from the repo
    # tree, unguarded, the same way roi_place's do.
    if "amg" in cfg.stages:
        needed.append(_needed("AMG segmentation (SAM2.1)", "sam2"))
        needed.append(_needed("AMG exemplar embedding (DINOv2)", "dinov2"))
        # The one stage that does open the served model: only when this run starts the server
        # itself. The recipe script names the catalogue group it serves from, so what is preflighted
        # here is by construction what the script will load.
        recipe = _serve_recipe(cfg)
        if recipe:
            needed.append(_needed(f"{recipe.script} weights", recipe.group))

    if generating:
        # base-<size> even when cfg.base_checkpoint points at a fine-tuned copy: the script cannot
        # produce one of those either way, and naming the group keeps the failure identical to what
        # a full provisioning run would report rather than inventing a quieter one. The size is
        # looked up rather than interpolated -- see catalogue.base_group for what that used to cost.
        needed.append((f"base checkpoint ({size})", base_checkpoint(cfg), catalogue.base_group(size)))
        needed.append(_needed("Wan2.2 VAE", "wan-vae"))
        if size == "edge":
            needed.append(_needed("caption tokenizer", "edge-processor"))
        else:
            needed.append(_needed("caption tokenizer", "qwen-tokenizer"))
        if cfg.guardrail:
            needed.append(_needed("guardrail (blocklist + face-blur)", "guardrail", "nvidia/Cosmos-Guardrail1"))
            needed.append(_needed("guardrail (text safety)", "guardrail", "Qwen/Qwen3Guard-Gen-0.6B"))
    if scoring:
        needed.append(_needed("DINOv2 (NN/MNN)", "dinov2"))
        needed.append(_needed("C-RADIO v3 (FID)", "cradio"))
        # evaluate.py imports anomaly_quality, which segments each generated defect with SAM2 for the
        # aq_* axes. That import tolerates a missing checkpoint and logs a warning, so requiring it
        # here does turn a soft dependency hard -- deliberately: a KPI quietly one column short is
        # worse to receive than a preflight naming an 857 MB file to provision.
        needed.append(_needed("SAM2.1 (anomaly-quality axes)", "sam2"))
    return needed


def _present(target) -> bool:
    """Whether a weight path is really there: a file, or a directory with something IN it.

    The catalogue defines its ``dir`` kind as "a non-empty directory" and both shell readers test
    ``ls -A``; this side tested ``exists()`` alone. A k8s subPath volume is created empty by the
    kubelet, so ``checkpoints/Cosmos3-Nano`` mounted that way passed the preflight, ``missing_groups``
    left ``base-nano`` out, prepare provisioned nothing, and generate.py died in
    ``load_for_inference`` after the mask stage had run.

    Read off the disk rather than off the row's ``kind``, because the disk answers the same question:
    a ``file`` row is never a directory, so the branch it takes is the one it would have been given.
    """
    if target.is_dir():
        return any(target.iterdir())
    return target.exists()


def missing_weights(cfg) -> list:
    """The weights the selected stages need that are not on disk, as ``label -> location`` lines.

    Split out of :func:`check_weights` so the prepare stage can ask the same question without
    catching the exception that answers it -- provisioning and the preflight then cannot disagree
    about what "present" means.
    """
    missing = []
    for label, target, _ in required_weights(cfg):
        if isinstance(target, str):  # an HF-cache entry, addressed by repo id
            missing += [f"{label} -> {m}" for m in _hf_missing(target)]
        elif not _present(target):
            missing.append(f"{label} -> {target}")
    return missing


def missing_groups(cfg) -> list:
    """The ``download_checkpoints.sh`` groups that would provision what :func:`missing_weights` found.

    Deduplicated and ordered as the weights are, so the argument list reads the same way the preflight
    does. Empty when nothing is missing -- which the caller must NOT pass to the script, since no
    arguments there means every group.
    """
    groups = []
    for label, target, group in required_weights(cfg):
        absent = _hf_missing(target) if isinstance(target, str) else ([] if _present(target) else [target])
        if absent and group not in groups:
            groups.append(group)
    return groups


def check_weights(cfg) -> None:
    """Fail before any stage when a weight the run needs is absent.

    Every missing entry at once, not the first: an operator wiring up a volume mount wants one list,
    not a failure per round trip through a 20-minute stage.
    """
    missing = missing_weights(cfg)
    if missing:
        raise SystemExit(
            f"{len(missing)} weight(s) the selected stages need are missing:\n"
            + "\n".join(f"    {m}" for m in missing)
            + f"\nHF cache checked: {_hf_cache()}"
            + "\nProvision them with scripts/download_checkpoints.sh (image build or volume setup),"
            " or mount an existing checkpoints tree at <repo>/checkpoints."
        )


def _read_registry(path) -> dict:
    """``{model: entry}`` from the amg CLI's registry file. Stricter than the CLI: this requires
    ``base_url`` on every entry up front, where the CLI checks it lazily, only for the models a run
    actually names.

    Re-implemented rather than imported: ``agentic_mask_generation.generate`` pulls in the whole
    engine and torch, and this check exists to run before anything heavy does.
    """
    entries = yaml.safe_load(pathlib.Path(path).read_text()) or []
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: expected a YAML list of model entries, got {type(entries).__name__}")
    registry = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("model"):
            raise SystemExit(f"{path}: entry {i} is missing its 'model' name")
        if not entry.get("base_url"):
            raise SystemExit(f"{path}: entry {i} ({entry['model']}) is missing its 'base_url'")
        registry[str(entry["model"])] = entry
    return registry


def check_amg(cfg) -> None:
    """Fail before the amg stage when its model names cannot resolve.

    The amg CLI refuses a name with no registry entry too, but only after the subprocess has
    started; this names the fix from the config's point of view, and lets --dry_run answer it.
    Endpoint liveness is deliberately not probed here: the CLI's capability smoke does that over
    the real transport before spending.
    """
    if "amg" not in cfg.stages:
        return
    recipe = _serve_recipe(cfg)
    registry = _read_registry(cfg.amg_model_registry) if cfg.amg_model_registry else {}
    if recipe:
        registry.setdefault(
            recipe.served, {"model": recipe.served, "base_url": "http://localhost", "served_by": "amg_serve"}
        )
    for key in ("amg_gen_model", "amg_judge_model"):
        name = getattr(cfg, key)
        if name not in registry:
            if cfg.amg_model_registry:
                where = cfg.amg_model_registry
            elif recipe:
                where = f"no registry (amg_serve only serves {recipe.served})"
            else:
                where = "no registry"
            raise SystemExit(
                f"{key} {name!r} has no entry in {where}; it declares "
                f"{sorted(registry)}. Add an entry {{model: {name}, base_url: ..., key: ...}}."
            )


def gpus_required(cfg) -> int:
    """GPUs the pod must expose. Sequential: the server is stopped before generate starts, so the
    requirement is the larger consumer, not the sum."""
    recipe = _serve_recipe(cfg)
    return max(recipe.tp if recipe else 0, cfg.num_gpus)


def check_gpus(cfg) -> None:
    """Fail before any stage when fewer GPUs are visible than this config needs.

    A CPU-only preflight host cannot answer the question -- the real pod will -- so no visible GPU is
    a warning, not a refusal.
    """
    visible = serving.visible_gpus()
    need = gpus_required(cfg)
    if not visible:
        log.warning(f"no GPU is visible here; this config needs {need} at run time (not checked on this host)")
        return
    if len(visible) < need:
        recipe = _serve_recipe(cfg)
        term = (
            f"amg_serve {cfg.amg_serve} (tp {recipe.tp})"
            if recipe and recipe.tp >= cfg.num_gpus
            else f"num_gpus {cfg.num_gpus}"
        )
        raise SystemExit(
            f"{len(visible)} GPU(s) visible ({','.join(visible)}) but this config needs {need}, set by {term}."
        )


# ── paths ─────────────────────────────────────────────────────────────────────


def _testcase_path(cfg, out) -> pathlib.Path:
    """Which backend's testcase.jsonl this run reads.

    By selection when a mask stage runs; otherwise by what is on disk, because a resumed tail was
    handed an output tree and has to find the artifact in it. Two on disk is ambiguous and is said
    so, rather than resolved by a preference nobody stated.
    """
    candidates = {s: out / s / "testcase.jsonl" for s in _MASK_STAGES}
    for stage in _MASK_STAGES:
        if stage in cfg.stages:
            return candidates[stage]
    present = [s for s, p in candidates.items() if p.exists()]
    if len(present) > 1:
        raise SystemExit(
            f"{out} holds a testcase.jsonl from both {present}; name the mask stage in `stages` so "
            "the run reads the one you mean."
        )
    return candidates[present[0]] if present else candidates["amp"]


def layout(cfg) -> dict:
    out = pathlib.Path(cfg.output)
    return {
        # One id per invocation, so a bucket can say which run filled it. Random rather than a
        # counter or a timestamp: nothing here is allowed to depend on two runs being ORDERED, only
        # on their being told apart -- clocks go backwards and a resumed tail is not "later" in any
        # sense the output tree can check.
        "run": uuid.uuid4().hex[:12],
        "amp": out / "amp",
        "amg": out / "amg",
        # the amg CLI's reusable precompute; kept inside the run tree by default, or wherever
        # amg_cache_dir points so several runs of one dataset can share the distilled defect library
        "amg_cache": pathlib.Path(cfg.amg_cache_dir) if cfg.amg_cache_dir else out / "amg_cache",
        "sdg": out / "sdg",
        "rounds": out / "rounds",
        "searched": out / "searched",
        "export": out / "export",
        "logs": out / "logs",
        "imaginaire": out / "imaginaire",
        "testcase": _testcase_path(cfg, out),
        "kpi": out / "sdg" / "kpi.json",
        "final_kpi": out / "searched" / "kpi.json",
        "defect_spec": pathlib.Path(cfg.input) / _DEFECT_SPEC_FILE,
    }


# ── whose files are these ─────────────────────────────────────────────────────
#
# A second run into one output directory used to mix two batches in every bucket, and no stage could
# tell its own output from the last one's. mtime cannot answer it: generate overwrites
# <anomaly_type>_<nnnnn>.png in place, and an in-place overwrite leaves the parent directory's mtime
# untouched -- only adding or removing an entry changes it, which is why comparing the two buckets'
# mtimes read as "not newer" for exactly the re-run it was meant to catch.
#
# So each bucket says who wrote it. ``source`` records the marker of the bucket it was DERIVED from,
# which is what makes the searched/-vs-sdg/ question exact rather than an ordering guess: searched/
# is current when the sdg/ it was assembled from is the sdg/ that is there now.
RUN_STAMP = ".anomalygen_run"


def _stamp(bucket, run, source=None) -> None:
    """Record that ``run`` wrote ``bucket``, and which bucket's contents it was built from."""
    bucket = pathlib.Path(bucket)
    bucket.mkdir(parents=True, exist_ok=True)
    payload = {"run": run} | ({"source": source} if source else {})
    (bucket / RUN_STAMP).write_text(json.dumps(payload) + "\n")


def _stamp_of(bucket) -> dict:
    """``bucket``'s run marker, or ``{}`` when it has none -- a tree older than this check."""
    try:
        return json.loads((pathlib.Path(bucket) / RUN_STAMP).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def take_bucket(bucket, run) -> None:
    """Empty ``bucket`` so that what it holds afterwards is this run's, and record that it is.

    Only for a bucket the calling stage REWRITES WHOLE. generate writes one file per testcase row
    and refine writes one directory per round, so a shorter second run leaves the first run's extra
    files behind -- and evaluate scores the directory rather than the CSV
    (``anomalygen/eval/utils.py``), so the KPI silently averages in images this run did not produce.
    quality_refine's round check counts the same stale files and stops the search with advice that
    cannot work, since generate.py overwrites but never removes.

    NOT for ``amp/``: the cad and text routes treat an existing ``<stem>__seed<i>.png`` as a cache
    hit and report CACHED, so that directory is a deliberate resume point and emptying it would turn
    every re-run into a full re-placement. Its own staleness is handled by reporting it instead --
    see :func:`require_full_testcase`.
    """
    bucket = pathlib.Path(bucket)
    if bucket.exists() and _stamp_of(bucket).get("run") != run:
        shutil.rmtree(bucket)
    _stamp(bucket, run)


def gen_root(cfg, paths) -> pathlib.Path:
    """Which bucket export reads: the refined one when refinement produced it, for THIS run.

    By SELECTION first. Picking on disk state alone shipped the wrong run's images: a first run with
    refine leaves searched/, a second run in the same output with a different amp_seed and stages
    [amp, generate, export] rewrites sdg/ -- and export still read searched/, exporting the first
    run's pictures under the second run's recorded config.

    A resumed tail that did not select refine still has to find it, so disk state remains the
    fallback -- but it is read off the run markers, not off mtimes. The first version of this
    compared directory mtimes and was dead on arrival: generate overwrites its images in place, and
    an in-place overwrite does not change the parent directory's mtime, so sdg/ was never "newer"
    for the very case this guards. The test that covered it aged the directories with os.utime,
    which a real write never does.

    searched/ records the sdg/ marker it was assembled from, so the question is exact: it is current
    when that marker is the one sdg/ carries now. Unknown is refused rather than guessed -- a tree
    written before this check, or assembled by hand, cannot be shown to belong to this sdg/, and
    reading it would ship images the run did not produce.
    """
    refined = paths["searched"] / "reconstructed_image"
    if "refine" in cfg.stages:
        return paths["searched"]
    if not refined.is_dir():
        return paths["sdg"]

    built_from = _stamp_of(paths["searched"]).get("source")
    current = _stamp_of(paths["sdg"]).get("run")
    if built_from and current and built_from == current:
        return paths["searched"]

    why = (
        f"was assembled from a different generate run ({built_from}, but {paths['sdg'].name} now holds {current})"
        if built_from and current
        else "carries no run marker, so which generate run produced it cannot be established"
    )
    raise SystemExit(
        f"{paths['searched']} {why}. Exporting it would ship images this run did not produce. "
        f"Re-run with 'refine' in stages to rebuild it from the current {paths['sdg'].name}, or "
        f"delete {paths['searched']} to export what generate produced."
    )


# ── stage running ─────────────────────────────────────────────────────────────


def run(cmd, log_path, env=None) -> None:
    """Run a stage, streaming to console and log. Raises SystemExit with the stage's own exit code.

    Through ``childproc.stream`` so a cancelled run does not leave the stage behind: SIGTERM becomes
    SystemExit here, which unwinds out of the output loop, and a torchrun left running keeps its GPUs
    and writes into a closed pipe. See that module for why the whole process group is stopped.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("$ " + " ".join(str(c) for c in cmd))
    with open(log_path, "w") as fp:

        def echo(line):
            sys.stdout.write(line)
            fp.write(line)

        returncode = childproc.stream(cmd, echo, env=env)
    if returncode != 0:
        raise SystemExit(f"stage failed (exit {returncode}); see {log_path}")


def require(path, stage, produced_by) -> None:
    """A skipped stage is only safe when the artifact it would have written is already there."""
    if not pathlib.Path(path).exists():
        raise SystemExit(
            f"stage '{stage}' needs {path}, which is missing. "
            f"Run the '{produced_by}' stage, or point 'output' at a run that already has it."
        )


def stage_amp(cfg, paths) -> None:
    cmd = [
        sys.executable,
        "-m",
        "anomalygen.scripts.auto_mask_placement.roi_place_pipeline",
        "--num_sdg",
        cfg.num_sdg,
        "--mode",
        cfg.mode,
        "--defect_desc",
        paths["defect_spec"],
        "--dataset_dir",
        cfg.input,
        "--output_dir",
        paths["amp"],
        "--seed",
        cfg.amp_seed,
        "--per_clean_pairing",
        cfg.amp_per_clean_pairing,
    ]
    if cfg.defect_counts:
        cmd += ["--per_defect_counts", json.dumps(cfg.defect_counts)]
    # The sampling knobs are written into each testcase row here, not passed to generate.py.
    passthrough = []
    for key, flag in (("guidance", "--guidance"), ("num_steps", "--num_steps"), ("crop_ratio", "--crop_ratio")):
        value = getattr(cfg, key)
        if value is not None:
            passthrough += [flag, value]
    if passthrough:
        cmd += ["--", *passthrough]
    run(cmd, paths["logs"] / "amp.log")
    require_full_testcase(cfg, paths, "amp", _amp_shortfall_remedy(paths))
    refuse_source_name_collisions(cfg, paths)


_AMG_CLI = "anomalygen.scripts.agentic_mask_generation.generate"


def stage_amg(cfg, paths) -> None:
    """Place masks with the agentic backend, then make its rows look like amp's.

    The CLI's own preflight (dataset layout, model registry, capability smoke) runs inside the
    subprocess and fails it before spending; this adds only what the entrypoint can see that the
    CLI cannot: the run's total, and the sampling knobs the config asked for.
    """
    types = sorted(cfg.defect_counts) if cfg.defect_counts else _mask_types_on_disk(cfg.input)
    if not types:
        raise SystemExit(f"amg: no <texture>/mask/<defect>/ directories under {cfg.input}; nothing to place")
    cmd = [
        sys.executable,
        "-m",
        _AMG_CLI,
        "--dataset-path",
        cfg.input,
        "--anomaly-types",
        *types,
        "--gen-model",
        cfg.amg_gen_model,
        "--judge-model",
        cfg.amg_judge_model,
        "--model-registry",
        _effective_registry(cfg, paths),
        "--num-sdg",
        cfg.num_sdg,
        "--mode",
        cfg.mode,
        "--out-dir",
        paths["amg"],
        # The CLI's default is CWD-relative (results/_amg_cache), which under an orchestrator lands
        # outside the run tree; inside it, the precompute travels with the run it served.
        "--cache-dir",
        paths["amg_cache"],
        "--visual-prompt",
        "auto",
        "--defect-library",
        "auto",
    ]
    if cfg.defect_counts:
        cmd += ["--per-defect-counts", json.dumps(dict(sorted(cfg.defect_counts.items())))]
    if cfg.amg_beam_multi > 0:
        cmd += ["--beam-multi", cfg.amg_beam_multi]
    for key, flag in (
        ("amg_beam_pairwise", "--beam-pairwise"),
        ("amg_exemplar_captions", "--exemplar-captions"),
        ("amg_exemplar_visual_prompt", "--exemplar-visual-prompt"),
        ("amg_coverage_note", "--coverage-note"),
    ):
        if getattr(cfg, key):
            cmd.append(flag)

    recipe = _serve_recipe(cfg)
    if recipe:
        visible = serving.visible_gpus()
        if len(visible) < recipe.tp:
            log.warning(f"amg_serve {cfg.amg_serve}: only {len(visible)} GPU(s) visible, recipe wants tp={recipe.tp}")
        gpus = visible[: recipe.tp]
        with serving.Server(recipe, gpus, paths["logs"] / "serve.log") as server:
            if not server.started and {"generate", "refine"} & set(cfg.stages):
                # A reused server (started by hand, or left over from a prior run) is not this
                # run's to stop, so its GPUs stay claimed until the reused process exits -- for
                # the WHOLE run, not just the amg stage. A later stage that also wants GPUs then
                # sees fewer than `visible`.
                log.warning(
                    f"amg: reusing the server already on :{recipe.port} -- it keeps its GPUs for "
                    "the whole run, so later stages see fewer than the visible total"
                )
            _run_amg_cli(cfg, paths, cmd)
    else:
        _run_amg_cli(cfg, paths, cmd)


def require_full_testcase(cfg, paths, stage, remedy) -> None:
    """Fail when the mask stage wrote fewer rows than the run asked for.

    Both backends exit 0 having placed fewer masks than requested -- roi_place drops a sample it
    cannot place, the amg CLI stops short when it degrades -- and every stage after this one then
    succeeds on the short batch. A caller compositing the results back onto a board is the first
    thing to notice, which is the one place this must not surface.

    Existence is not completion either: both write incrementally, so an interrupted run leaves a
    short file that reads as a valid one.
    """
    testcase = paths["testcase"]
    log_path = paths["logs"] / f"{stage}.log"
    if not testcase.exists():
        raise SystemExit(f"{stage}: the stage exited 0 but wrote no {testcase}; see {log_path}")
    n_rows = sum(1 for line in testcase.read_text().splitlines() if line.strip())
    if n_rows == cfg.num_sdg:
        return
    if n_rows > cfg.num_sdg:
        # A different failure with a different cause, and it used to borrow the shortfall's words:
        # "a dropped sample is a placement that could not be made" is simply false when the count is
        # OVER. roi_place builds the testcase by globbing <amp>/<stem>/<type>/*__seed*.png, and a
        # second run with another amp_seed draws other submasks, whose masks land beside the first
        # run's under different names. Not emptied here on purpose: the cad and text routes read an
        # existing seed file as a cache hit (CACHED), so that directory is a resume point.
        why = (
            " A different amp_seed draws different submasks, and their masks are written beside the "
            "earlier ones rather than over them."
            if stage == "amp"
            else ""
        )
        raise SystemExit(
            f"{stage}: {testcase} has {n_rows} row(s) but the run asked for {cfg.num_sdg} -- MORE, "
            f"not fewer, so nothing was dropped. {paths[stage]} still holds masks from an earlier "
            f"run and the testcase is built from everything in there.{why} Point 'output' at a "
            f"fresh directory, or delete {paths[stage]} and re-run the mask stage."
        )
    raise SystemExit(f"{stage}: {testcase} has {n_rows} row(s) but the run asked for {cfg.num_sdg}. {remedy}")


# What roi_place calls a placement that happened, taken from its own arithmetic rather than guessed:
# it counts `s["status"] in ("OK", "CACHED")` as placed. The rest -- FAILED, NO_CANDIDATES,
# NO_DETECTION, ROI_ONLY, SIZE_MISMATCH -- are the ones worth naming when a run came up short.
# Listing every status instead buries the reason under "7x OK".
_AMP_PLACED = ("OK", "CACHED")


def _amp_shortfall_remedy(paths) -> str:
    """What to tell an operator whose amp run placed fewer masks than it was asked for.

    Read off the SEEDS, not off the sample. Every route returns the sample as "OK" once placement
    was attempted at all -- ``_process_cad_tht_sample`` ends on a literal ``"status": "OK"`` however
    its seeds went -- so the commonest shortfall of all, a placer that returned None for every seed,
    was summarised as "no unplaced sample": the message told the operator there was no cause in the
    one run that had one. The per-seed entries carry the real outcome (FAILED, PARTIAL), and a
    sample that never got as far as seeds still has only its own status to give, so both are counted
    and the sample-level one is used only where there are no seeds to ask.
    """
    summary = paths["amp"] / "summary.json"
    try:
        results = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return f"See {summary} for what happened to each sample."
    entries = results.values() if isinstance(results, dict) else results
    counts = collections.Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        seeds = entry.get("seeds") or {}
        if seeds:
            counts.update(s.get("status", "?") for s in seeds.values() if isinstance(s, dict))
        else:
            counts[entry.get("status", "?")] += 1
    short = ", ".join(f"{n}x {status}" for status, n in counts.most_common() if status not in _AMP_PLACED)
    return (
        f"roi_place ended on: {short or 'no unplaced sample or seed'}. Every sample's outcome is in "
        f"{summary}. A dropped sample is a placement that could not be made, not a transient "
        "failure -- re-running the same config reproduces it."
    )


def _run_amg_cli(cfg, paths, cmd) -> None:
    """The CLI call, the row-count gate and the back-fill -- everything that must happen while the
    server (if any) is still up, so a gate failure stops it on the way out."""
    run(cmd, paths["logs"] / "amg.log")
    require_full_testcase(
        cfg,
        paths,
        "amg",
        "Re-run the 'amg' stage into the same output; the CLI resumes and produces only the "
        "shortfall. If a re-run does not move the count, the CLI stopped short on purpose — see "
        f"the degraded[...] lines in {paths['amg'] / 'run.log'}.",
    )
    rewritten = _backfill_sampling(paths["testcase"], cfg)
    if rewritten:
        log.info(f"amg: wrote the configured sampling knobs into {rewritten} testcase row(s)")
    refuse_source_name_collisions(cfg, paths)


def _backfill_sampling(testcase, cfg) -> int:
    """Write the config's guidance / num_steps / crop_ratio into every row; return rows rewritten.

    amp writes these per row through roi_place; amg does not, and generate.py would fall back to
    the shared defaults, silently ignoring the config. Only the knobs actually set are written, so
    an unset one still takes generate.py's default the way it does for amp. Atomic: a crash
    mid-rewrite must not leave a half-written testcase that the row-count gate above already passed.
    """
    knobs = {k: getattr(cfg, k) for k in ("guidance", "num_steps", "crop_ratio") if getattr(cfg, k) is not None}
    if not knobs:
        return 0
    testcase = pathlib.Path(testcase)
    rows = [json.loads(line) for line in testcase.read_text().splitlines() if line.strip()]
    for row in rows:
        row.update(knobs)
    tmp = testcase.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fp:
        fp.write("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, testcase)
    return len(rows)


def stage_generate(cfg, paths) -> None:
    require(paths["testcase"], "generate", "amp|amg")
    # generate writes one image per testcase row and never removes one, so a shorter second run into
    # the same output leaves the first run's extras in place -- and evaluate scores the directory,
    # not the CSV, so the KPI quietly averages them in.
    take_bucket(paths["sdg"], paths["run"])
    cmd = [
        *childproc.torchrun(cfg.num_gpus, _TEXTURE / "generate.py"),
        "--checkpoint",
        cfg.checkpoint,
        "--recipe",
        cfg.recipe,
        "--input_data_path",
        paths["testcase"],
        "--output_dir",
        paths["sdg"],
        "--base_seed",
        cfg.base_seed,
    ]
    # Always passed, never left to generate.py's own default: that default is relative and gets
    # os.path.abspath'd against the WORKING DIRECTORY, which is not the repository root under an
    # orchestrator. See base_checkpoint().
    cmd += ["--base_checkpoint", base_checkpoint(cfg)]
    cmd.append("--guardrail" if cfg.guardrail else "--no-guardrail")
    # Torchrun's own variables are stripped, not inherited: launched from inside an outer torchrun
    # -- which an orchestrator may well do -- the child would see two rendezvous configurations and
    # report a world-size mismatch. quality_refine has filtered them for its rounds all along; this
    # stage did not.
    env = childproc.without_torchrun_vars()
    # Precautionary, and for symmetry with the refine stage, which passes the same directory as
    # quality_refine's --output_root and so sets this for the generate.py children it launches.
    # Generation writes its images to --output_dir, and nothing in generate.py was found to read
    # this root, so the two paths agreeing matters more here than what lands under it: unset, the
    # framework resolves job paths to /tmp/imaginaire4-output rather than failing, which would put
    # anything that did appear outside the run tree and outside this machine's view of the run.
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(paths["imaginaire"])
    run(cmd, paths["logs"] / "generate.log", env=env)


def stage_evaluate(cfg, paths) -> None:
    require(paths["sdg"] / "reconstructed_image", "evaluate", "generate")
    run(
        [
            sys.executable,
            _TEXTURE / "evaluate.py",
            "--gen_root",
            paths["sdg"],
            "--real_root",
            cfg.input,
            "--recipe",
            cfg.recipe,
            "--output_file",
            paths["kpi"],
        ],
        paths["logs"] / "evaluate.log",
    )


def stage_refine(cfg, paths) -> None:
    require(paths["testcase"], "refine", "amp|amg")
    require(paths["sdg"] / "reconstructed_image", "refine", "generate")
    require(paths["kpi"], "refine", "evaluate")
    # Same reason as generate, and one worse: quality_refine's completeness check counts the pngs in
    # a round directory against the testcase rows, so a previous run's leftovers make every round
    # read as over-full and stop the search with advice -- "re-run this round's generate.py (it
    # overwrites)" -- that cannot clear them.
    take_bucket(paths["rounds"], paths["run"])
    run(
        [
            sys.executable,
            _TEXTURE / "quality_refine.py",
            "run",
            "--base_testcase",
            paths["testcase"],
            "--original",
            paths["sdg"],
            "--original_kpi",
            paths["kpi"],
            "--rounds_dir",
            paths["rounds"],
            "--output",
            paths["searched"],
            "--final_kpi",
            paths["final_kpi"],
            "--checkpoint",
            cfg.checkpoint,
            "--recipe",
            cfg.recipe,
            "--real_root",
            cfg.input,
            "--num_search_run",
            cfg.num_search_run,
            "--num_gpus",
            cfg.num_gpus,
            "--output_root",
            paths["imaginaire"],
            # Same sampling setup as the generate stage, so a round is comparable to the bucket it
            # is selected against rather than differing in how it was produced -- including the base
            # checkpoint, absolute for the reason base_checkpoint() gives.
            "--base_checkpoint",
            base_checkpoint(cfg),
            "--base_seed",
            cfg.base_seed,
            "--guardrail" if cfg.guardrail else "--no-guardrail",
        ],
        paths["logs"] / "refine.log",
    )
    # Recorded after the stage succeeded, and naming the sdg/ marker rather than this run's: what
    # export has to decide later is whether searched/ still belongs to the sdg/ beside it, and a
    # resumed [refine, export] tail assembles a fresh searched/ from an EARLIER run's sdg/. Stamping
    # it with this run's id would make that legitimate pairing look like a mismatch.
    _stamp(paths["searched"], paths["run"], source=_stamp_of(paths["sdg"]).get("run"))


# ── export ────────────────────────────────────────────────────────────────────


def read_generation_rows(gen_dir, produced_by) -> list:
    """The rows of whichever bucket export was pointed at.

    ``produced_by`` names the stage that writes THAT bucket, because ``require``'s advice is
    otherwise wrong for the refined one: a half-finished refine leaves ``searched/`` without a CSV,
    and telling the caller to run ``generate`` sends them somewhere that cannot produce it.
    """
    csv_path = pathlib.Path(gen_dir) / _GEN_CSV
    require(csv_path, "export", produced_by)
    with open(csv_path, newline="") as fp:
        return list(csv.DictReader(fp))


def report_blocked(gen_dir) -> None:
    """Name the sources the guardrail dropped.

    Blocked images are absent from the generation CSV, so export simply writes fewer files than the
    testcase has rows. For a composite-back-onto-the-board flow that gap is the whole result -- some
    joints get no image -- and it must not be something the caller discovers by counting files.
    """
    blocked_path = pathlib.Path(gen_dir) / _BLOCKED_CSV
    if not blocked_path.is_file():
        return
    with open(blocked_path, newline="") as fp:
        blocked = list(csv.DictReader(fp))
    if not blocked:
        return
    names = sorted({pathlib.Path(r.get("image_filename", "?")).name for r in blocked})
    log.warning(f"export: {len(blocked)} image(s) blocked by the guardrail and NOT exported: {names[:10]}")


def refuse_source_name_collisions(cfg, paths) -> None:
    """Fail on a source-naming collision as soon as the mask stage has made one decidable.

    export raises on the same condition, but by then generate, evaluate and refine have run -- and
    the collision was already determined by the testcase rows, which name the clean image each
    output will be called after. Checking here costs a file read and saves the batch.

    Counts per clean image rather than rows: one row asking for several images collides with itself
    just as two rows over one clean image do.
    """
    if cfg.output_naming != "source":
        return
    testcase = paths["testcase"]
    if not testcase.is_file():
        return
    claims = collections.Counter()
    for line in testcase.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        claims[pathlib.Path(row["image_filename"]).stem] += int(row.get("num_generated_images", 1) or 1)
    clashes = {stem: n for stem, n in claims.items() if n > 1}
    if not clashes:
        return
    shown = "\n".join(f"    {stem}  <- {n} outputs" for stem, n in sorted(clashes.items())[:5])
    raise SystemExit(
        f"output_naming: source gives every output its clean image's name, and {len(clashes)} clean "
        f"image(s) would be claimed more than once:\n{shown}"
        + ("\n    ..." if len(clashes) > 5 else "")
        + "\nUse output_naming: indexed, or place one output per clean image "
        "(amp_per_clean_pairing: on, one defect type)."
    )


def plan_export(rows, naming) -> dict:
    """``{target_name: source_name}``. Raises when two outputs claim one target.

    A collision means the batch is not one-output-per-crop: several defect types over the same
    cleans, or several images per testcase. Renaming past it would silently drop generated images
    and leave the caller compositing whichever happened to be written last.
    """
    if naming == "indexed":
        return {r["output_filename"]: r["output_filename"] for r in rows}

    claims = {}
    for r in rows:
        target = pathlib.Path(r["image_filename"]).stem + pathlib.Path(r["output_filename"]).suffix
        claims.setdefault(target, []).append(r["output_filename"])

    clashes = {t: srcs for t, srcs in claims.items() if len(srcs) > 1}
    if clashes:
        lines = [f"    {t}  <- {', '.join(sorted(s))}" for t, s in sorted(clashes.items())[:5]]
        raise SystemExit(
            f"{len(clashes)} source image(s) map to more than one output:\n"
            + "\n".join(lines)
            + ("\n    ..." if len(clashes) > 5 else "")
            + "\nUse output_naming: indexed, or generate one defect type per run."
        )
    return {t: srcs[0] for t, srcs in claims.items()}


def stage_export(cfg, paths) -> None:
    src_root = gen_root(cfg, paths)
    rows = read_generation_rows(src_root, "refine" if src_root == paths["searched"] else "generate")
    report_blocked(src_root)
    mapping = plan_export(rows, cfg.output_naming)

    dst = paths["export"]
    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    for target, source in sorted(mapping.items()):
        src = src_root / "reconstructed_image" / source
        if not src.exists():
            raise SystemExit(f"export: {src} is missing though the CSV lists it")
        shutil.copy2(src, dst / target)
        written += 1
    log.info(f"export: {written} image(s) from {src_root.name}/reconstructed_image -> {dst}")

    manifest = dst / "export_manifest.json"
    inputs_by_output = {r["output_filename"]: r["image_filename"] for r in rows}
    manifest.write_text(
        json.dumps(
            {
                "source_bucket": str(src_root),
                "images": mapping,
                "input_images": {target: inputs_by_output[source] for target, source in mapping.items()},
            },
            indent=2,
        )
        + "\n"
    )
    log.info(f"export: mapping recorded in {manifest}")


# ── main ──────────────────────────────────────────────────────────────────────


def _fetch_spec(cfg) -> dict:
    """``staging.fetch`` as ``{config field: sources}``, validated."""
    block = cfg.staging or {}
    if not isinstance(block, dict):
        raise SystemExit(f"staging must be a mapping, got {type(block).__name__}.")
    spec = block.get("fetch") or {}
    if not isinstance(spec, dict):
        raise SystemExit(f"staging.fetch must be a mapping, got {type(spec).__name__}.")
    return spec


# The only keys a fetch may write to. Every other field is either not a path -- `staging.fetch.num_sdg`
# reached `Path(40)` and died with a TypeError -- or is a path this run WRITES: fetch replaces its
# destination, so `staging.fetch.output` deleted the output tree after the resolved config and the
# logs had been written into it. Being a config key is not enough to be a destination.
_FETCHABLE = ("input", "recipe", "checkpoint", "base_checkpoint", "amg_model_registry")

# The one key under `staging.fetch` that is not a config field. It names paths INSIDE `input` --
# `<TEXTURE>/clean_image`, `<TEXTURE>/cad_mask`, `defect_spec.jsonl` -- and each is replaced by the
# sources given for it. Named for its target rather than just `override`: it sits among keys that
# ARE config fields, where a bare `override` reads like one more of them and says nothing about what
# it overrides.
OVERRIDE_KEY = "input_override"


def _check_fetch_spec(cfg, path) -> None:
    """Reject a fetch block before a run starts, not after it has downloaded half of one.

    The keys are this config's own field names and the destination is whatever that field already
    says, so there is no second copy of a path to keep in step -- and a key that is not a field, or
    one whose field is unset, has nowhere to put what it downloads.
    """
    from anomalygen.staging import check_source, usecase

    spec = _fetch_spec(cfg)
    if spec.get(OVERRIDE_KEY):
        _check_override_spec(spec, path)
    for key, sources in spec.items():
        if key == OVERRIDE_KEY:
            continue
        if key not in _FETCHABLE:
            known = f"Fetchable keys: {sorted(_FETCHABLE)}"
            why = (
                " That key names an output of this run, not an input."
                if key in _field_names()
                else " It is not a config key at all."
            )
            raise SystemExit(f"{path}: staging.fetch.{key} is not something a fetch can write to.{why} {known}")
        if not getattr(cfg, key, None):
            raise SystemExit(f"{path}: staging.fetch.{key} has no destination — set '{key}' as well.")
        urls = [sources] if isinstance(sources, str) else list(sources or [])
        if not urls:
            raise SystemExit(f"{path}: staging.fetch.{key} lists no sources.")
        for url in urls:
            # A use-case script writes a DATASET -- a tree of textures and a defect_spec -- so it can
            # only fill a directory. The other fetchable keys each name one file, and the mismatch
            # would otherwise surface from inside the copy as an IsADirectoryError naming neither the
            # key nor the use-case.
            if usecase.is_usecase(url) and key != "input":
                raise SystemExit(
                    f"{path}: staging.fetch.{key} is a single file, and {url} prepares a whole "
                    f"dataset. A use-case belongs on 'input'; give {key} the file it needs."
                )
            try:
                check_source(url)
            except ValueError as exc:
                raise SystemExit(f"{path}: staging.fetch.{key} — {exc}") from exc


def _warn_unused_fetch(cfg, path) -> None:
    """Say so when a fetch block is configured and ``prepare`` is not selected to run it.

    `prepare` is the stage that fetches, so a config carrying `staging.fetch` without it downloads
    nothing and silently runs on whatever is already at those paths. For `input_override` that is
    the exact outcome the key exists to prevent -- the images the config named are not there, the
    dataset's own are, and the run succeeds.

    A WARNING AND NOT A REFUSAL. Re-running later stages over an existing output, with the same
    config file and a shorter `stages`, is how a run resumes -- example.yaml documents it -- and that
    config legitimately still carries the fetch block that filled those paths the first time.
    Refusing it would break the resume this pipeline is built for; what was missing is that the run
    never said which of the two it was doing.
    """
    spec = _fetch_spec(cfg)
    if not spec or "prepare" in cfg.stages:
        return
    keys = sorted(spec)
    log.warning(
        f"{path}: staging.fetch is set ({keys}) but 'prepare' is not in stages, so NOTHING WILL BE "
        f"FETCHED this run -- every path is read as it already is on disk. That is right for a "
        f"resume, and wrong if this run was meant to bring its own inputs."
        + (
            " input_override in particular: the images it names will not replace anything, and the "
            "run will generate from whatever the dataset already holds."
            if OVERRIDE_KEY in spec
            else ""
        )
    )


def _check_override_spec(spec, path) -> None:
    """Reject a ``staging.fetch.input_override`` block at config time.

    The paths themselves cannot be checked until the input has been fetched -- that is what they name
    -- but the shape, the sources and the one structural requirement can be, and a run that will
    refuse an override is not one worth starting.
    """
    from anomalygen.staging import check_override

    if "input" not in spec:
        raise SystemExit(
            f"{path}: staging.fetch.input_override names paths inside the dataset, and this config does "
            "not fetch one. Add staging.fetch.input, or drop the override and point 'input' at a "
            "dataset that already has the files you want."
        )
    try:
        check_override(spec[OVERRIDE_KEY])
    except ValueError as exc:
        raise SystemExit(f"{path}: staging.fetch.input_override — {exc}") from exc


_DOWNLOAD_CHECKPOINTS = pathlib.Path("scripts") / "download_checkpoints.sh"


def stage_prepare(cfg, paths) -> None:
    """Populate the local paths the later stages read. A wrapper; the work is in the package."""
    from anomalygen.staging import ensure_base_weights, fetch

    spec = _fetch_spec(cfg)
    if not spec:
        log.info("prepare: staging.fetch is empty — nothing to fetch")
    # In config-field order rather than mapping order, so a log reads the same way run to run. Read
    # off `fields(Config)` and NOT `_field_names()`, which returns a set: str hashing is randomised
    # per process, so that really did give a different order on every run. Measured across four
    # interpreters, four different orders.
    #
    # It is not only the log. fetch REPLACES its destination, so nested destinations -- `input:
    # /work/data` with `recipe: /work/data/recipe.yaml` -- depend on the order. Fetching input last
    # would take the recipe with it; the foreign-destination guard refuses that rather than losing
    # it, which means the same config succeeded or failed at random depending on the hash seed.
    for key in [f.name for f in fields(Config) if f.name in spec]:
        # The override goes to the fetch of the tree it addresses, so it is applied to the merge
        # before that merge is moved into place -- see staging.fetch.
        fetch(spec[key], getattr(cfg, key), override=spec.get(OVERRIDE_KEY) if key == "input" else None)

    # After the fetches, not before: deciding WHICH weights this run needs means reading the recipe
    # and the defect_spec, and those are two of the things the fetches just brought in.
    #
    # Only what the selected stages will open, which is what makes this cheap for the common case --
    # a cad_tht amp-only run needs no model at all, so this returns without touching the network.
    ensure_base_weights(
        _REPO_ROOT / _DOWNLOAD_CHECKPOINTS,
        _REPO_ROOT / "checkpoints",
        lambda: missing_weights(cfg),
        groups=lambda: missing_groups(cfg),
        hf_cache=_hf_cache(),
    )


def check_publish_destination(cfg) -> None:
    """Fail on an unreachable publish destination before the run, not after it.

    publish is the last stage, so a typo in the bucket used to cost everything ahead of it -- amp,
    generate and export all ran, and the batch they produced was then stranded on local disk inside
    a container about to be discarded.

    Not in load_config, deliberately: this makes a network call, and load_config is what --dry_run
    and every test rely on being offline and instant. It sits with check_weights instead, which is
    the same kind of check -- everything that has to be true before a stage is worth starting.
    """
    destination = _publish_destination(cfg)
    if "publish" not in cfg.stages or not destination:
        return
    from anomalygen.staging import check_reachable

    try:
        check_reachable(destination)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"staging.publish.output — {exc}") from exc


def _publish_destination(cfg):
    """``staging.publish.output``, or None when the config does not ask for an upload."""
    block = cfg.staging or {}
    if not isinstance(block, dict):
        raise SystemExit(f"staging must be a mapping, got {type(block).__name__}.")
    publish = block.get("publish") or {}
    if not isinstance(publish, dict):
        raise SystemExit(f"staging.publish must be a mapping, got {type(publish).__name__}.")
    return publish.get("output")


def stage_publish(cfg, paths) -> None:
    """Upload the export bucket. A wrapper, like every other stage -- the work is in the package."""
    from anomalygen.staging import MANIFEST_NAME, publish

    # Named rather than the directory: the manifest is what publish reads, and a bucket whose images
    # are present but whose manifest is not is exactly the state a half-finished export leaves.
    require(paths["export"] / MANIFEST_NAME, "publish", "export")
    published = publish(paths["export"], _publish_destination(cfg))
    # Recorded beside the results so a re-run can see what the last one shipped without listing the
    # remote prefix, which may not be readable from wherever the question is being asked.
    (pathlib.Path(cfg.output) / "publish_manifest.json").write_text(json.dumps(published, indent=2) + "\n")


def _exit_on_signal(signum, frame) -> None:
    """Turn a terminating signal into an exception. k8s and Airflow cancel a job with SIGTERM, a
    dropped ssh session sends SIGHUP, an operator sends SIGINT; Python's default for the first two
    ends the interpreter without unwinding anything, which skips ``childproc.stream``'s ``finally``
    and ``Server.__exit__`` -- orphaning a torchrun or a 2-GPU server. Every stage child now runs in
    its own session, so it will not see the terminal's SIGHUP itself; this process has to pass the
    intent on. Only an exception unwinds those blocks, so this is what lets a cancelled run stop
    what it started rather than leaving it on the dead pod's GPUs."""
    raise SystemExit(128 + signum)


# What a cancelled or disconnected run may be told with. Installed in main() before the first stage.
_TERMINATING = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


_RUNNERS = {
    "prepare": stage_prepare,
    "amp": stage_amp,
    "amg": stage_amg,
    "generate": stage_generate,
    "evaluate": stage_evaluate,
    "refine": stage_refine,
    "export": stage_export,
    "publish": stage_publish,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="pipeline config YAML")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate the config and the weights the selected stages need, print the plan, run nothing",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = layout(cfg)

    log.info(f"stages: {' -> '.join(cfg.stages)}")
    for stage in cfg.stages:
        skipped = _skipped_dependencies(stage, cfg.stages)
        if skipped:
            log.info(f"  {stage}: reads output of skipped stage(s) {skipped} — expecting them on disk")
    # Before the plan is printed, so --dry_run doubles as an orchestrator's preflight task: it answers
    # "is this container wired up to run" without starting a stage.
    #
    # Unless prepare is selected. These checks read what prepare provisions -- the weights themselves,
    # the recipe and defect_spec they consult to decide WHICH weights this run needs, and the amg
    # registry file when it is fetched -- so running them first would report a container that is
    # wired up correctly as broken. With prepare they run immediately after, below, which is still
    # before any stage that loads a model or calls a model endpoint.
    check_gpus(cfg)
    if "prepare" not in cfg.stages:
        check_weights(cfg)
        check_amg(cfg)
    check_publish_destination(cfg)
    if args.dry_run:
        recipe = _serve_recipe(cfg)
        log.info(f"gpus_required: {gpus_required(cfg)} (visible: {','.join(serving.visible_gpus()) or 'none'})")
        log.info(f"service: {recipe.script + f' (tp {recipe.tp}, port {recipe.port})' if recipe else 'none'}")
        log.info(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return

    for sig in _TERMINATING:
        signal.signal(sig, _exit_on_signal)

    paths["logs"].mkdir(parents=True, exist_ok=True)
    (pathlib.Path(cfg.output) / "pipeline_config.resolved.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=True))
    for stage in cfg.stages:
        log.info(f"=== {stage} ===")
        _RUNNERS[stage](cfg, paths)
        if stage == "prepare":
            # Now that the downloads have landed, everything deferred at load time is checkable.
            _check_paths(cfg, args.config)
            check_weights(cfg)
            check_amg(cfg)
    log.success(f"done -> {cfg.output}")


if __name__ == "__main__":
    main()
