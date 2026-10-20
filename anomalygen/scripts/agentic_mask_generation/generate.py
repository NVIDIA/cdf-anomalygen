# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry for agentic defect-mask generation.

AMG places every defect type it is given. A per-type VLM router decides only HOW the shape is handled:
``fixed`` stamps a REAL training-mask shape and moves it rigidly, ``flexible`` synthesizes the shape and
refines it freely.

Output is AMP-compatible: the deliverable is a ``testcase.jsonl`` (the schema the texture generation
step consumes via ``--input_data_path``) with binary ``{0,255}`` masks against the clean images in
``<dataset>/<texture>/clean_image/``.

Fully driven by CLI args; anything not passed falls back to the ``Config`` defaults. Endpoints and
secrets are read ONLY here, and from exactly one place: the model registry file
(``endpoints.yaml`` / ``--model-registry`` — nothing is built in), then injected into the
transport — ``test_the_engine_reads_no_environment``.

    python -m anomalygen.scripts.agentic_mask_generation.generate \
        --dataset-path /data/my_boards --anomaly-types SOP+INSUFFICIENT_SOLDER \
        --gen-model azure/openai/gpt-6-astra --judge-model azure/openai/gpt-6-astra \
        --num-sdg 20 --visual-prompt auto --defect-library auto --out-dir results/amg/run1
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import numpy as np
from cosmos_framework.utils import log
from cosmos_framework.utils.log import init_loguru_file
from PIL import Image
from tqdm import tqdm

from anomalygen import allocation
from anomalygen.agentic_mask_generation.card_distill import build_cards_dir, card_variant, observation_for, read_flags
from anomalygen.agentic_mask_generation.config import Config
from anomalygen.agentic_mask_generation.defects import from_dir
from anomalygen.agentic_mask_generation.exemplars import DatasetError, load_exemplars, preflight
from anomalygen.agentic_mask_generation.imaging import Views
from anomalygen.agentic_mask_generation.judging import component_count
from anomalygen.agentic_mask_generation.loop import run_beam
from anomalygen.agentic_mask_generation.ops import CV_KINDS, canonical_kind
from anomalygen.agentic_mask_generation.regions import enumerate_regions
from anomalygen.agentic_mask_generation.utils import atomic_write
from anomalygen.agentic_mask_generation.visual_prompts import annotate_exemplar, by_name
from anomalygen.agentic_mask_generation.vlm import (
    ANTHROPIC_BASE,
    BAD_REQUEST,
    MAX_TOKENS_ROUTE,
    QUOTA_EXHAUSTED,
    TransportUnavailable,
    Vlm,
    classify_exception,
    discard_staged_masks,
    get_api_ms,
    get_call_count,
    get_failed_count,
    get_usage,
    set_trace_ctx,
)
from anomalygen.data.utils import list_images, validate_anomaly_type

# Where --model-registry looks when not passed. Under $HOME, never under the repo: the product
# images chmod the checkout read-only (a live registry there could not even be created), a
# credential file inside a build context wants to end up in an image layer, and one registry then
# serves every checkout and worktree regardless of the launch directory. There are NO endpoints
# in this codebase — not even for the shipped local-serving recipes — because an endpoint hardcoded
# here is configuration the user's endpoints.yaml doesn't show; the template beside the serving
# scripts (REGISTRY_TEMPLATE) carries the known-good entries instead.
# A literal `~`, expanded only when the default is actually consulted (`_load_registry`'s
# expanduser): resolving $HOME at import time made a HOME-less arbitrary-uid container die before
# argparse ran — even a run that passed --model-registry never got to say so.
DEFAULT_REGISTRY = "~/.anomalygen/endpoints.yaml"
REGISTRY_TEMPLATE = "scripts/agentic_mask_generation_serving/endpoints.example.yaml"


# --------------------------------------------------------------------------------------------------
# The model registry — THE one way a model name becomes an endpoint
# --------------------------------------------------------------------------------------------------
def _load_registry(path: str | None) -> dict[str, dict]:
    """{served_model: spec}, read from the ONE registry file — nothing is built in.

    The YAML (--model-registry; when not passed, DEFAULT_REGISTRY is picked up if present) is a
    list of entries:

        - model: nvidia/google/gemma-4-31b-it     # the name the SERVER calls itself
          base_url: http://localhost:8902/v1      # where that name lives
          key: EMPTY                              # the bearer; EMPTY (or omitted) for a keyless
          #                                       # local server. The registry lives outside the
          #                                       # repo and only whether a key is SET is logged.

    This registry is the whole routing story: a model name not found in it refuses the run before
    anything is spent (`_endpoints`) — there is no environment fallback and no built-in entry to
    silently reroute it. The template (REGISTRY_TEMPLATE) carries the known-good starting entries."""
    explicit = path is not None
    path = os.path.expanduser(path or DEFAULT_REGISTRY)
    reg: dict[str, dict] = {}
    if not os.path.exists(path):
        if explicit:
            raise SystemExit(f"--model-registry {path}: no such file")
        return reg  # no file, empty registry: every model name will refuse with the copy-the-example hint
    import yaml

    entries = yaml.safe_load(Path(path).read_text()) or []
    if not isinstance(entries, list):
        raise SystemExit(f"--model-registry {path}: expected a YAML LIST of model entries")
    for e in entries:
        m = (e or {}).get("model")
        if not m:
            raise SystemExit(f"--model-registry {path}: an entry is missing its 'model' name")
        reg[m] = dict(e)
    return reg


def _resolve_endpoint(model: str, spec: dict) -> tuple[str, str, str]:
    """(base_url, key, key_source) for one registry entry. `key` defaults to EMPTY — vLLM/SGLang
    accept any bearer when their --api-key is unset. The SOURCE string is what logs and run records
    may carry — never the key itself."""
    base = spec.get("base_url")
    if not base:
        raise SystemExit(f"model registry entry for {model}: no base_url")
    key = str(spec.get("key") or "EMPTY")
    return base, key, ("EMPTY" if key == "EMPTY" else "set (redacted)")


def _endpoints(models: list[str], registry: dict[str, dict]) -> dict[str, tuple[str, str]]:
    """{served_model: (base_url, key)} for every model this run will call. A name the registry does
    not know refuses the run HERE — before the out-dir is touched or a token is spent — because the
    registry is the only routing there is, and 'which URL did my prompts post at' must never be
    answered by a fallback."""
    out: dict[str, tuple[str, str]] = {}
    for m in dict.fromkeys(m for m in models if m):  # de-dup, keep order; drop the unset ("") roles
        spec = registry.get(m)
        if spec is None:
            raise SystemExit(
                f"model {m!r} is not in the model registry, so there is nowhere to send its calls. "
                f"Add an entry (model + base_url + key) to {DEFAULT_REGISTRY} — start from "
                f"{REGISTRY_TEMPLATE} — or pass --model-registry. "
                f"Registered: {', '.join(sorted(registry)) or 'nothing (no registry file found)'}."
            )
        base, key, _src = _resolve_endpoint(m, spec)
        out[m] = (base, key)
    return out


def _log_routing(roles: dict[str, str], registry: dict[str, dict]) -> None:
    """One line per role saying where its model RESOLVES — the answer to 'which URL does this name
    reach', printed before anything is spent. Key presence only, never key values."""
    for role, m in roles.items():
        if not m:
            continue
        base, _key, src = _resolve_endpoint(m, registry[m])
        log.info(f"model routing: {role} {m} -> {base} (key: {src})")


def _build_config(args) -> Config:
    """CLI args -> Config. Only args the user actually passed override the Config defaults."""
    # Each token becomes `<out-dir>/<texture>/mask/<defect>`, so it is held to the shape the AMP scripts
    # hold their `defect_type` to — one safe path segment per half — and a separator, a `..` or an
    # absolute path is refused here, before the out-dir is touched.
    try:
        for t in args.anomaly_types:
            validate_anomaly_type(t, field="--anomaly-types")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    types = tuple(tuple(t.split("+", 1)) for t in args.anomaly_types)
    # `--mode` names how a TOTAL is split, so `--mode validation` without `--num-sdg` is a request
    # with a hole where the total should be. Refused at parse time rather than at allocation, where
    # the hole would surface as a zero-share coverage error blaming the dataset. (The two sizing
    # flags themselves may coexist: when `--per-defect-counts` disagrees with `--num-sdg`,
    # allocation warns and the override sum wins — exact counts ARE the total.)
    if args.mode == "validation" and args.num_sdg is None:
        raise SystemExit("--mode validation needs --num-sdg: there is no total to split proportionally.")
    # Both passes write the same `<clean stem>_<reuse>_<j>` filenames into the same directory, so the
    # second overwrites the first while testcase.jsonl carries both rows. Refused rather than
    # de-duplicated — which of the two the user meant to drop is not this code's guess to make.
    # a repeated anomaly type is refused before it can overwrite its own masks.
    dupes = sorted({"+".join(t) for t in types if types.count(t) > 1})
    if dupes:
        raise SystemExit(
            f"--anomaly-types repeats {', '.join(dupes)}. Both passes write the same mask filenames "
            f"into the same directory, so the second overwrites the first and testcase.jsonl claims "
            f"twice the masks that exist. List each type once."
        )
    # ONE IDIOM: every flag that maps to a Config field is listed here, and `None` means "not passed,
    # keep the Config default" — which is why every one of them declares `default=None`, including the
    # `--no-*` switches, which are `store_false` and so are None / False, never True. Both halves of
    # the contract — a passed flag reaches its field, an omitted one keeps the Config default — hold
    # for every row. A missing row is silent — the flag still parses, `run.json` still records the
    # DEFAULT, and the run that was asked for never happened.
    overrides = {
        "gen_model": args.gen_model,
        "judge_model": args.judge_model,
        "num_examples": args.num_examples,
        "num_candidates": args.num_candidates,
        "beam_width": args.beam_width,
        "max_refine_rounds": args.max_refine_rounds,
        "accept_threshold": args.accept_threshold,
        "final_judge_samples": args.final_judge_samples,
        "image_max_dim": args.image_max_dim,
        "max_concurrent_calls": args.max_concurrent_calls,
        "visual_prompt": args.visual_prompt,
        "coverage_feedback": args.coverage,
        "slot_regions": args.slot_regions,
        "shape_policy": args.shape_policy,
        # The in-loop beam and the three throughput levers. Same idiom as every row above — the flag
        # carries `default=None`, so `None` here means "not typed" and the Config default stands. A
        # truthiness test (`if args.x:`) cannot tell a switch that was passed False from one that was
        # never passed at all.
        "beam_multi": args.beam_multi,
        "beam_pairwise": args.beam_pairwise,
        "beam_prune": args.beam_prune,
        "accept_continue": args.accept_continue,
        "judge_cache_order": args.judge_cache_order,
        "exemplar_captions": args.exemplar_captions,
        "exemplar_vp": args.exemplar_vp,
        "coverage_note": args.coverage_note,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    # `allowed_cv_ops` is the one flag whose value is not the string that was typed.
    if args.allowed_cv_ops:
        # `canonical_kind` first, so the flag takes the same spellings the engine accepts from the
        # model (`polygon` is `add_polygon`, `stroke` is `brush`); an unknown name is refused HERE
        # rather than KeyError-ing from inside the engine on every beam, and the refusal names the op
        # AS TYPED. an unknown allowed cv op is refused by the driver not by the engine on every beam.
        asked = tuple(x.strip() for x in args.allowed_cv_ops.split(",") if x.strip())
        unknown = [x for x in asked if canonical_kind(x) not in CV_KINDS]
        if unknown:
            raise SystemExit(
                f"--allowed-cv-ops names {', '.join(repr(u) for u in unknown)}, which is not a CV op. "
                f"Valid ops: {', '.join(sorted(CV_KINDS))}"
            )
        overrides["allowed_cv_ops"] = tuple(dict.fromkeys(canonical_kind(x) for x in asked))
    return Config(anomaly_types=types, dataset_path=args.dataset_path, **overrides)


# --------------------------------------------------------------------------------------------------
# Capability smoke — can each model do the job at all, asked before the run pays to find out
# --------------------------------------------------------------------------------------------------
# AMG asks one thing of a model on every call: look at images, answer in JSON. A model that cannot do
# either produces a run that fails everywhere at once — every region call, every beam, every verdict —
# after the card distillation and the region enumeration have been paid for, and the log then reports
# the symptom (`api_abort:bad_request`, `call_gave_up`) rather than the cause. One call per model up
# front turns all of that into a named refusal that costs nothing.
_SMOKE_SYSTEM = (
    "Reply with strict JSON and nothing else: no prose, no markdown fence. "
    'Answer exactly {"n_images": <integer>} — how many images this message contains.'
)
# TWO images, not one: with a single image `1` is the answer a model that cannot see any of them
# guesses, so the probe would pass on the guess and the batch would go on to discover the truth one
# beam at a time. Two costs the same single call and no likely guess is right.
_SMOKE_COLOURS = ((20, 90, 200), (200, 90, 20))
_SMOKE_N = len(_SMOKE_COLOURS)


def _smoke_blocks() -> list:
    """The probe payload: `_SMOKE_N` tiny images and the question only sight answers.

    GENERATED, never read off the dataset: the probe runs before the run touches its inputs, and an
    answer that depended on which board came first would be a second thing that can go wrong here.
    64x64, so the whole check is a rounding error against one beam.
    """
    return ["How many images are in this message?", *(Image.new("RGB", (64, 64), c) for c in _SMOKE_COLOURS)]


def _is_bare_anthropic_host(base: str) -> bool:
    """True when ``base`` is the Anthropic API host with no path, i.e. ANTHROPIC_BASE minus its /v1."""
    parts = urlparse(base)
    return parts.netloc == "api.anthropic.com" and parts.path.rstrip("/") == ""


def _smoke_reason(e: BaseException, base: str = "") -> str:
    """What the probe proved the model could not do, in the terms the user has to act on.

    Read off the FAILURE the transport classified rather than off the exception type, because the
    things that go wrong here take different fixes and a traceback tells them apart for nobody: a
    base url one segment short, a credential the endpoint will not take, a model that is not
    serving vision, and a model that will not answer in JSON. ``base`` is the endpoint this model's
    calls resolved to, for the first of those.
    """
    f = getattr(e, "failure", None)
    if f is None:  # a reply ARRIVED and was not the object that was asked for
        return f"it did not answer with the JSON it was asked for ({type(e).__name__}: {e})"
    # The one misconfiguration that wears another failure's clothes: api.anthropic.com WITHOUT the
    # /v1 segment is not ANTHROPIC_BASE, so the transport speaks OpenAI /chat/completions at it and
    # the API answers with a 4xx — which would be diagnosed below as bad credentials or a model not
    # serving images, when the real fix is one missing path segment.
    if f.kind == BAD_REQUEST and _is_bare_anthropic_host(base):
        return (
            f"the base url {base!r} is api.anthropic.com without the /v1 segment, so the request "
            f"went out as /chat/completions — a route the Anthropic API does not serve. Set the "
            f"model's registry entry to base_url: {ANTHROPIC_BASE}"
        )
    if f.status in (401, 403):
        return f"the endpoint refused the credentials (HTTP {f.status}) — check the key and the base url"
    if f.kind == QUOTA_EXHAUSTED:
        return (
            "the account behind this key has no quota or credits left (HTTP 429 insufficient_quota) — "
            "not a rate limit, so retrying cannot help; add credits or use another key"
        )
    if f.kind == BAD_REQUEST:
        return f"the endpoint rejected the request outright, which is what a model not serving images does: {f.detail}"
    return f"the endpoint did not answer ({f.kind}): {f.detail}"


def _smoke_failed(model: str, reason: str) -> SystemExit:
    """Book the deviation and BUILD the abort — the caller raises it.

    Returned rather than raised so the caller can `raise ... from None`: a chained transport traceback
    is noise in front of a message whose whole job is to name which of three things to fix.
    """
    log.warning(f"degraded[model_smoke_failed]: {model}: {reason}")
    return SystemExit(
        f"{model} failed the capability probe, so the run stopped before it spent anything: {reason}. "
        f"AMG needs a model that accepts images and answers in JSON — fix the endpoint, name a "
        f"different model with --gen-model/--judge-model, or pass --skip-smoke to run against it anyway."
    )


def _smoke_models(vlm, models: list[str], bases: "dict[str, str] | None" = None) -> None:
    """ONE probe per DISTINCT model, down the transport that model will really use.

    Per model rather than per run, because `--gen-model` and `--judge-model` can name different
    models on different endpoints and either one being unusable sinks the run; de-duplicated, because
    the usual case names one model twice and the second probe would answer the same question twice.

    `schema="smoke"` puts the probe under the same one retry every boundary gets, so a success is one
    call and a model that needed a second ask to drop the prose costs two.
    """
    for m in dict.fromkeys(x for x in models if x):
        try:
            out = vlm.call_json(_SMOKE_SYSTEM, _smoke_blocks(), model=m, max_tokens=MAX_TOKENS_ROUTE, schema="smoke")
        except (Exception, TransportUnavailable) as e:  # noqa: BLE001 — every kind is a failed probe
            raise _smoke_failed(m, _smoke_reason(e, (bases or {}).get(m, ""))) from None
        n = out["n_images"]  # an integer: the schema above is what makes that true
        if n != _SMOKE_N:
            raise _smoke_failed(
                m, f"it was sent {_SMOKE_N} images and reported {n}, so it is not reading the images it is sent"
            )
        log.info(f"model smoke: {m!r} took {_SMOKE_N} images and answered in JSON")


# --------------------------------------------------------------------------------------------------
# Dataset IO (AMP-compatible)
# --------------------------------------------------------------------------------------------------
def _list_cleans(dataset_path, texture) -> list[Path]:
    """Clean-image pool, AMP's resolution order: ``<ds>/<tex>/clean_image/`` → ``<tex>/clean/`` →
    flat ``<tex>/``. First non-empty wins."""
    base = Path(dataset_path) / texture
    for sub in ("clean_image", "clean", "."):
        d = base / sub
        imgs = list_images(d)
        if imgs:
            # Two cleans differing only by extension write the same `<stem>_<reuse>_<j>` mask file,
            # so refuse rather than pick — the same failure a repeated `--anomaly-types` is refused
            # for, reached by the data rather than by a typo.
            by_stem: dict[str, list[Path]] = {}
            for q in imgs:
                by_stem.setdefault(q.stem, []).append(q)
            clash = {k: v for k, v in by_stem.items() if len(v) > 1}
            if clash:
                shown = "; ".join(f"{k}: {', '.join(q.name for q in v)}" for k, v in sorted(clash.items())[:3])
                raise SystemExit(
                    f"{d} holds {len(clash)} clean image(s) whose names differ only by extension "
                    f"({shown}). Every mask is filed under the clean's STEM, so these would write "
                    f"the same mask file and testcase.jsonl would claim twice the masks that "
                    f"exist, half of them naming the wrong board. Rename or remove one of each."
                )
            return imgs
    return []


def _iter_max(dataset_path, texture, defect) -> int:
    """AMP's rule for ``iteration_generation_max_instance``: 1 iff EVERY training submask for the type
    is multi-instance, else 5. Read from the real masks under ``<tex>/mask/<defect>/``."""
    mdir = Path(dataset_path) / texture / "mask" / defect
    masks = list_images(mdir)
    if not masks:
        return 5
    counts = [component_count(Image.open(p).convert("L")) for p in masks]
    return 1 if counts and all(c > 1 for c in counts) else 5


def _binarize(mask: Image.Image) -> Image.Image:
    """AMP mask contract: grayscale L, strictly ``{0, 255}``, white = defect."""
    a = (np.asarray(mask.convert("L")) > 127).astype("uint8") * 255
    return Image.fromarray(a, "L")


# --------------------------------------------------------------------------------------------------
# Routers (VLM)
# --------------------------------------------------------------------------------------------------
_VP_CHOICES = ("solderlight", "normallight", "sam2", "none")


def _vp_from_card(defects, defect_type, *, vp_dir, views=None):
    """The visual prompt named by the type's distilled card, or None if the card cannot say.

    The card writes the prompt's own name, so there is nothing to translate here. It belongs to the
    card because the thing it turns on — how the boards were lit — is a property of the capture rig and
    therefore of the type; deciding it per clean costs a call per board and is unstable, since one
    dataset's boards can answer differently.
    """
    card = defects.card(defect_type) if defects else None
    if not card:
        return None
    name = read_flags(card).get("visual_prompt", "")
    return by_name(name, vp_dir=vp_dir, views=views) if name in _VP_CHOICES else None


def _as_bool(v) -> bool:
    """JSON-ish truthiness. The model is asked for a boolean but may answer with a string, and bool("false")
    is True — which would put a whole defect type on the rigid-stamp path on a wording accident."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _card_forbids_stamp(defects, texture, defect) -> bool:
    """True when this type's card judges ``fill_host=whole`` — one mark covers ALL of its host
    structure(s), so the mask IS whole host instance(s). Any other value, an absent flag or an
    unreadable card is False: routing proceeds exactly as with no card."""
    try:
        card = defects.card(f"{texture}+{defect}") if defects else None
        flags = read_flags(card or "")
        return flags.get("fill_host") == "whole"
    except Exception:  # noqa: BLE001 — no flags, so routing proceeds exactly as with no card
        return False


def _route_card_context(card: str | None) -> str:
    """The route call's card block: the three geometry flags and the card's own shape prose.

    The card already knows what the images alone may not show — "one mass spanning two lead joints"
    is a sites_spanned judgment distilled from the type's whole annotation pool — so the router
    reads it too. "" when there is no card (or it says nothing useful): the route call is then the
    image-only ask, exactly as with no library."""
    if not card:
        return ""
    # Value-neutral glosses, one per flag, rendered only for the flags this card actually carries.
    gloss = {
        "sites_spanned": "how many separate structures ONE mark lies across",
        "fill_host": "how much of its host structure one mark covers",
        "hosts_presented": "how the scene presents that host",
    }
    flags = read_flags(card)
    present = [k for k in gloss if k in flags]
    shape = card.split("## typical shape", 1)
    prose = " ".join(shape[1].split("##", 1)[0].split()) if len(shape) > 1 else ""
    if not present and not prose:
        return ""
    out = "Reference card for this type (distilled from its real annotation pool):"
    if present:
        out += (
            " "
            + ", ".join(f"{k}={flags[k]}" for k in present)
            + " ("
            + "; ".join(f"{k}: {gloss[k]}" for k in present)
            + ")."
        )
    if prose:
        out += f" Typical shape: {prose}"
    return out


def _route_shape(vlm, dataset_path, texture, defect, *, model, card=None, clean_path=None, emb_cache_dir=None) -> bool:
    """Per-type shape classifier: ONE VLM call → ``fixed_shape``.

    True → the shape is CLEARLY intrinsic to the defect, so AMG stamps a real example shape and
    refines placement with RIGID ops only. False → AMG synthesizes the shape and refines it freely.
    The ask is deliberately asymmetric: ``fixed`` narrows the whole type's vocabulary to stamps, so
    it is answered only on a clear case, and every doubt goes to the full agentic path — the same
    direction the ``except`` fails to.

    ``card`` is the type's distilled card ("" / None routes image-only); ``clean_path`` is a
    reference board for DINOv2-similar exemplar selection (None falls back to first-k), sharing
    ``emb_cache_dir`` with the main loop's own per-clean ranking so the embeddings are paid once."""
    try:
        exs = load_exemplars(
            dataset_path, texture, defect, k=3, sampling="similar", clean_path=clean_path, emb_cache_dir=emb_cache_dir
        )
        blocks: list = []
        card_ctx = _route_card_context(card)
        if card_ctx:
            blocks.append(card_ctx)
        blocks.append("Example (defect image, its mask — white = defect region) pairs for this type:")
        for e in exs:
            blocks += [e.defect, e.mask]
        system = (
            f"Decide how the mask SHAPE for defect type '{texture}+{defect}' should be produced, from "
            "the real (defect image, mask) example pairs shown"
            + (" and the type's reference card" if card_ctx else "")
            + ". "
            'Answer "fixed_shape": true ONLY when the shape is CLEARLY intrinsic to the defect '
            "itself: rigid, consistent across the examples, and reusable VERBATIM on any board "
            "without reading that board's own structures (target-independent). "
            'Answer "fixed_shape": false on ANY doubt, ANY variability across the examples, or ANY '
            "dependence on the target's structures — a defect that connects/bridges/spans separate "
            "structures, or whose size/outline follows the board's own pads, gaps or features, is "
            "false even when the example masks look alike (boards that look alike make masks that "
            "look alike; that is not a rigid shape). "
            'Output JSON {"fixed_shape": true|false, "why": "<one sentence>"}.'
        )
        # This answer governs every beam the type runs, so keep the evidence for it.
        set_trace_ctx(f"{texture}_{defect}_shaperoute")
        try:
            out = vlm.call_json(system, blocks, model=model, max_tokens=MAX_TOKENS_ROUTE)
        finally:
            set_trace_ctx("")
        fixed = _as_bool(out.get("fixed_shape"))
        log.info(f"{texture}+{defect}: shape route -> {'fixed' if fixed else 'flexible'} ({out.get('why', '')})")
        return fixed
    except Exception as e:
        log.warning(f"degraded[shape_route_failed]: [{texture}+{defect}] {type(e).__name__}: {e}")
        return False


def _resolve_shape_mode(vlm, cfg, texture, defect, defects=None, *, clean_path=None, emb_cache_dir=None) -> str:
    """A type's ``shape_mode`` under ``cfg.shape_policy``. ``fixed`` and ``flexible`` set every type
    without asking the router — ``fixed`` outranks the card ban below, so the full fixed vocabulary
    (stamp included) stays on offer. ``auto`` routes per type, under ONE rule above the router: a
    card judging ``fill_host=whole`` says one mark of this defect covers ALL of its host
    structure(s) — the mask IS whole host instance(s) — so a stamped example shape can only collide
    with this board's own structures: the type is forced flexible and the router is not asked — and
    since flexible mode still offers ``stamp`` as an op, `BeamContext.build` withholds the op itself
    from every beam of the type (under ``flexible`` policy too). Every other flag value (and no
    card) leaves the decision to the router, which then reads the same card as context; ``defects``
    is the card library, and ``clean_path``/``emb_cache_dir`` are the router's exemplar-similarity
    reference (see :func:`_route_shape`)."""
    if cfg.shape_policy == "fixed":
        return "fixed"
    if cfg.shape_policy == "flexible":
        return "flexible"
    if _card_forbids_stamp(defects, texture, defect):
        log.info(
            f"{texture}+{defect}: card fill_host=whole — the mask is whole host instance(s), so "
            f"stamp is forbidden under --shape auto: forcing flexible and withholding the stamp op"
        )
        return "flexible"
    try:
        card = defects.card(f"{texture}+{defect}") if defects else None
    except Exception:  # noqa: BLE001 — an unreadable card routes exactly as no card
        card = None
    fixed = _route_shape(
        vlm,
        cfg.dataset_path,
        texture,
        defect,
        model=cfg.gen_model,
        card=card,
        clean_path=clean_path,
        emb_cache_dir=emb_cache_dir,
    )
    return "fixed" if fixed else "flexible"


# --------------------------------------------------------------------------------------------------
# Visual prompts — which cue a (clean, type) pair is shown, and its one-time precompute
# --------------------------------------------------------------------------------------------------
@dataclass
class VisualPrompts:
    """The visual prompt each ``(clean, type)`` pair gets, resolved once and remembered.

    Keyed by BOTH: the card answers per type, so a clean-only key would serve the first type's answer
    to the second.
    """

    routing: bool  # --visual-prompt auto: the type's own card names the prompt
    fixed: Any  # the prompt named on the command line; None when `routing`
    vp_dir: str
    views: Views
    library: str  # the --defect-library setting, named in the no-card warning below
    # The distilled cards, which only the routing path reads. Set by the caller once the card library
    # has been loaded, which happens after this object exists.
    defects: Any = None
    cache: dict = field(default_factory=dict)

    def for_clean(self, cp, dtype):
        k = (str(cp), dtype)
        vp = self.cache.get(k)
        if vp is None:
            vp = self.fixed
            if self.routing:
                # The card, or nothing. A type whose card cannot name a prompt gets none: a hint
                # guessed at from one board is worse than no hint.
                vp = _vp_from_card(self.defects, dtype, vp_dir=self.vp_dir, views=self.views)
                if vp is None:
                    # `NoneVP.exists()` is True and the snapshot still records
                    # `visual_prompt: "auto"`, so a run with no cue is otherwise indistinguishable
                    # from one with a cue — and it is the DEFAULT path, since `--defect-library`
                    # defaults to "none" and `auto` then yields no prompt for any type.
                    log.warning(
                        f"degraded[vp_auto_no_card]: [{dtype}] --visual-prompt auto found no "
                        f"card to route from (--defect-library is "
                        f"{self.library!r}), so this run has NO "
                        f"injected structure cue — pass `--defect-library auto` to get one"
                    )
                    vp = by_name("none")
            self.cache[k] = vp
        # THE INJECTED PROMPT ONLY: `context.build` asks `layers()` of the instance it was handed,
        # so an artifact belonging to a prompt that was not injected is unreachable, and precomputing
        # it costs GPU seconds and cache megabytes per clean that no run can read.
        # the driver precomputes the injected visual prompt and nothing else.
        if not vp.exists(str(cp)):
            log.info(f"precomputing {vp.name!r} maps for {Path(cp).name} ...")
            try:
                vp.precompute(str(cp))
            except Exception as e:  # noqa: BLE001 — a failed cue must not sink the run
                # The sole record that a beam ran with no cue and no layer at all —
                # a failed precompute is booked now that it costs the run its only cue.
                log.warning(
                    f"degraded[vp_precompute_failed]: {vp.name}: {Path(cp).name}: "
                    f"{type(e).__name__}: {e} — its cue and layer will be absent"
                )
        return vp


# --------------------------------------------------------------------------------------------------
# Placement (retry-to-fill)
# --------------------------------------------------------------------------------------------------
class _UsageLedger:
    """What the beams have spent, as a RUNNING TOTAL: every beam's token/call/time row is folded in
    once, by `_run` below, as the beam lands — plus, on a multi-type run, one `prep`-tagged row per
    type, because prep (shape routing, region enumeration) then happens on a per-type driver thread
    whose counters nothing else reads. `_run_usage` reads the totals, so the usage term of a run.json
    write costs the same at the five-thousandth mask as at the first (the per-mask rows are still
    serialized on every write) — and run.json is written on every accepted mask.

    Module-level, not thread-local: the beams run on workers and the driver reports. Locked, because
    the workers land concurrently and each total is a read-modify-write.
    """

    def __init__(self) -> None:
        self.tokens: dict = {}  # usage key -> total
        self.calls = 0
        self.failed: dict = {}  # failure kind -> attempts that did not answer
        self.api_ms = 0.0
        self.beams = 0  # a `prep` row is spend, not a beam
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        """Fold one `{usage, api_calls, failed, api_ms[, prep]}` row into the totals."""
        with self._lock:
            for k, v in row["usage"].items():
                if isinstance(v, (int, float)):
                    self.tokens[k] = self.tokens.get(k, 0) + v
            self.calls += row["api_calls"]
            for k, v in (row.get("failed") or {}).items():
                self.failed[k] = self.failed.get(k, 0) + v
            self.api_ms += row["api_ms"]
            if "prep" not in row:
                self.beams += 1

    def total(self) -> dict:
        """One consistent snapshot of the totals, taken under the lock."""
        with self._lock:
            return {
                "tokens": dict(self.tokens),
                "calls": self.calls,
                "failed": dict(self.failed),
                "api_ms": self.api_ms,
                "beams": self.beams,
            }


_RUN_USAGE = _UsageLedger()


def _run_usage() -> dict:
    """What the run has spent: the beams' running total plus the driver's own calls (card
    distillation; also type routing and region enumeration when they run on this thread — a
    multi-type run books those per type into `_RUN_USAGE` instead, as `prep` rows).

    Read off `_RUN_USAGE` rather than off this thread: `get_usage()` is thread-local and the beams
    run on workers, so the calling thread's own counters hold only the driver's calls.

    From the BEAMS, and not from the delivered rows: a trimmed mask's row is removed from the
    deliverable, but the tokens that produced it were still spent, so summing what survived
    undercounts by exactly the overshoot.
    """
    spent = _RUN_USAGE.total()
    tot = {
        **get_usage(),
        "api_ms": round(get_api_ms() + spent["api_ms"], 1),
        "calls": get_call_count() + spent["calls"],
        # BY KIND. `rate_limited` was refused before inference and is almost certainly
        # free; `server_error` means the server did not finish. Only `transport` is even a
        # candidate for having been billed, and whether it was is the provider's policy.
        "failed_attempts": get_failed_count(),
    }
    for k, v in spent["tokens"].items():
        tot[k] = tot.get(k, 0) + v
    for k, v in spent["failed"].items():
        tot["failed_attempts"][k] = tot["failed_attempts"].get(k, 0) + v
    tot["beams"] = spent["beams"]
    return tot


def _config_snapshot(cfg) -> dict:
    """Every Config field, JSON-shaped, for the run record.

    Derived from `dataclasses.fields` rather than hand-listed, so a new field cannot be left out;
    tuples become lists, the only conversion needed, since Config is policy-only by construction.
    """

    def _plain(v):
        if isinstance(v, tuple):
            return [_plain(x) for x in v]
        return v

    return {f.name: _plain(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}


def _engine_constants() -> dict:
    """The module-level numbers that shape a call but are NOT run policy.

    Properties of the ANSWER being asked for (a two-field classification vs a seven-section card)
    rather than of the run, so they stay module constants — still recorded, so two runs that differ in
    one are distinguishable.
    """
    from anomalygen.agentic_mask_generation import imaging as I
    from anomalygen.agentic_mask_generation import vlm as V

    names = (
        "MAX_TOKENS_ROUTE",
        "MAX_TOKENS_REGIONS",
        "MAX_TOKENS_CAPTION",
        "MAX_TOKENS_CARD",
        "MAX_TOKENS_JUDGE",
        "MAX_TOKENS_AGENT",
    )
    out = {n.lower(): getattr(V, n) for n in names}
    out.update(
        {
            n.lower(): getattr(I, n)
            for n in ("ZOOM_MAGNIFY", "MASK_PAD", "REGION_PAD", "ZOOM_FLOOR", "MONTAGE_COLS", "MONTAGE_PANEL")
        }
    )
    return out


# --------------------------------------------------------------------------------------------------
# Resume — carry on a run that stopped short, in the directory it stopped in
# --------------------------------------------------------------------------------------------------
_MASK_SUFFIX = "_mask.png"
_RUNLOG_SUFFIX = "_run.log"


def _read_jsonl(path: Path) -> list[dict]:
    """Every JSON object in a manifest. A line that will not parse is skipped, not raised on:
    `atomic_write` rules out a half-written FILE, but nothing rules out a hand-edited one, and a
    manifest that cannot be read at all would turn a resumable run into a lost one."""
    out: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _split_indexed(name: str, suffix: str, n: int) -> "tuple[str, list[int]] | None":
    """A run filename -> (clean stem, its `n` trailing indices). None when it is not one of ours.

    Read from the RIGHT, always. The stem is whatever the dataset named the board and may carry
    underscores and digits of its own — `panel_2_top.png` is an ordinary filename — so only the last
    `n` components belong to the run: `<stem>_<reuse>_<j>_mask.png`, `<stem>_<reuse>_run.log`.
    """
    if not name.endswith(suffix):
        return None
    parts = name[: -len(suffix)].rsplit("_", n)
    if len(parts) != n + 1 or not all(p.isdigit() for p in parts[1:]):
        return None
    return parts[0], [int(p) for p in parts[1:]]


# The flags that act on the PROCESS rather than on what it hands over, so they are kept out of the argv
# the run records and compares: `--overwrite` is consumed the moment the out-dir is (re)opened, and
# `--skip-smoke` only decides whether the capability probe runs. Matched down to the shortest prefix
# argparse resolves to each one alone (`--ov`, `--sk`); a shorter prefix is ambiguous with `--out-dir`
# or with the other `--s` flags (`--shape`, `--slot-regions`), so the parser rejects
# it before anyone gets here.
_PROCESS_FLAGS = ("--overwrite", "--skip-smoke")


def _is_process_flag(a: str) -> bool:
    """True for a `_PROCESS_FLAGS` entry or any argparse abbreviation that resolves to it."""
    return len(a) >= 4 and any(f.startswith(a) for f in _PROCESS_FLAGS)


def _argv_pairs(tokens) -> list:
    """argv regrouped as sorted `(flag, values)` pairs: a flag is a `--` token, its values are every
    token up to the next flag (so `-1` is a value), and `--flag=value` is the pair `--flag value`."""
    pairs: list = []
    for a in tokens:
        if a.startswith("--"):
            flag, eq, inline = a.partition("=")
            pairs.append((flag, (inline,) if eq else ()))
        elif pairs:
            pairs[-1] = (pairs[-1][0], (*pairs[-1][1], a))
        else:
            pairs.append(("", (a,)))
    return sorted(pairs)


def _control_plane_diff(prev: dict, cfg, argv: list[str]) -> list[str]:
    """How this run's settings differ from the ones recorded in an out-dir's `run.json`.

    Empty means the two runs are the same run, and the second may continue the first. Anything else
    is two configurations sharing one deliverable, which is the thing nothing downstream can undo:
    `testcase.jsonl` carries no per-row record of the settings that produced it.

    BOTH halves of the control plane: `config` is every `Config` field, `argv` covers the flags that
    reach a run without landing in one. argv is compared as a MULTISET OF (flag, values) PAIRS: the
    flags may come in any order, but each value stays bound to its flag and a multi-value flag keeps
    the order of its values, so `--max-rounds 3 --request-timeout 480` is not the same run as
    `--max-rounds 480 --request-timeout 3`. `argv[0]` is dropped — it is how the program was invoked,
    not what it was asked to do.
    test_a_run_with_different_settings_is_refused_and_the_refusal_names_the_field,
    test_swapped_flag_values_are_refused_and_reordered_flags_are_not.

    Only the fields THIS build declares are compared, and a field present on one side alone is not a
    difference: a key the record has and `Config` does not is one this build dropped, a key `Config`
    has and the record does not is one it gained. Neither is two configurations sharing a
    deliverable, which is the only thing this function is for. Comparing the UNION instead made
    adding a single `Config` field — the most ordinary change there is — refuse every out-dir written
    before it, under a message asserting the settings differ when the shared ones are identical, and
    the documented way out of that refusal is the one that destroys the delivered masks.
    a record from a build with other fields still resumes.
    """
    diffs: list[str] = []
    now, before = _config_snapshot(cfg), (prev.get("config") or {})
    for k in sorted(now):
        if k in before and before[k] != now[k]:
            diffs.append(f"config.{k}: was {before[k]!r}, now {now[k]!r}")
    # Schema drift is worth knowing but is not grounds for refusal, so it goes on the record instead.
    drift = sorted(set(before) - set(now)) + sorted(set(now) - set(before))
    if drift:
        log.warning(
            f"degraded[resume_schema_drift]: this build and the recorded run declare "
            f"different Config fields ({', '.join(drift)}); the fields they share matched, "
            f"so the run continues"
        )
    was = _argv_pairs((prev.get("argv") or [])[1:])
    is_ = _argv_pairs(list(argv)[1:])
    if was != is_:
        gone = [" ".join((f, *v)) for f, v in was if (f, v) not in is_]
        new = [" ".join((f, *v)) for f, v in is_ if (f, v) not in was]
        diffs.append(f"argv: dropped {gone or '(nothing)'}, added {new or '(nothing)'}")
    return diffs


class _Resume:
    """What an earlier process left in an out-dir, and what this one therefore still owes.

    Read back from the DELIVERABLES — the manifest, the mask files, the per-beam run logs — because
    those are what survives a restart.
    """

    def __init__(
        self, rows, overflow_rows, summary, delivered, next_reuse, orphans, missing, prior, prior_targets=None
    ):
        self.rows, self.overflow_rows, self.summary = rows, overflow_rows, summary
        # The per-type split the earlier process resolved. Under `--mode validation` the split is
        # weighted by the dataset's own mask counts, which are read at run time and appear in neither
        # argv nor Config — so `_control_plane_diff` cannot see a dataset that grew between the two
        # runs. Recorded and compared instead, the way AMP materializes `allocation.json` rather than
        # recomputing it in each downstream step.
        self.prior_targets = prior_targets
        self.delivered = delivered  # dtype -> masks already in testcase.jsonl AND on disk
        # dtype -> clean stem -> the first `reuse` index no file or run log has claimed: the counter
        # restarts at 0 in every process, so without this a resume writes over the first run's masks.
        # the reuse counter is seeded from the files and from the run logs.
        self.next_reuse = next_reuse
        self.orphans, self.missing = orphans, missing
        self.prior = prior  # what the earlier process(es) delivered and spent

    def target_for(self, dtype: str, want: "int | None") -> "int | None":
        """This run's target for one type: the SHORTFALL, not the original count.

        ``want`` is that type's own target — its share of ``--num-sdg``, or its exact
        ``--per-defect-counts`` entry. Either way the shortfall is per type, which is
        what makes an allocated run resumable: re-deriving the split from the same total and the same
        declared order reproduces this type's `want` exactly.
        """
        if want is None:
            return None
        return max(0, int(want) - int(self.delivered.get(dtype, 0)))

    def deviations(self, out: Path) -> list[tuple[str, str]]:
        """The `(kind, detail)` pairs this resume owes the record, each emitted by the caller as one
        `degraded[KIND]: detail` line — including the two ways the manifest and the disk can disagree,
        each with what was DONE about it."""
        lines = [
            (
                "resumed_run",
                f"{out} already holds {len(self.rows)} delivered mask(s) from "
                f"{self.prior['runs']} earlier process(es); this run produces only the shortfall, "
                f"and its `usage` block counts only what THIS process spends",
            )
        ]
        if self.missing:
            lines.append(
                (
                    "resume_row_without_mask",
                    f"{self.missing} row(s) in testcase.jsonl name a mask file that is not on disk — the "
                    f"rows are dropped and those masks count as NOT delivered, so the shortfall covers "
                    f"them. A manifest may not name a file that does not exist",
                )
            )
        n_orphan = sum(self.orphans.values())
        if n_orphan:
            lines.append(
                (
                    "resume_mask_without_row",
                    f"{n_orphan} mask file(s) under {out} have no row in testcase.jsonl "
                    f"({', '.join(f'{k}: {v}' for k, v in sorted(self.orphans.items()) if v)}) — left "
                    f"where they are and NOT counted as delivered (there is no verdict, no clean and no "
                    f"iteration cap recorded for them, and inventing a row would put an unjudged mask in "
                    f"the deliverable). Their filenames stay reserved, so this run cannot write over them",
                )
            )
        return lines


def _plan_resume(out: Path, cfg, argv: list[str]) -> _Resume:
    """Read a stopped run back out of `out` — or refuse to run into it. Reads only; writes nothing.

    AUTO-DETECTED rather than put behind a `--resume` flag, whose forgotten form is the exact failure
    this prevents. test_a_resumed_run_makes_up_the_shortfall_without_touching_the_first_runs_masks.
    """
    try:
        prev = json.loads((out / "run.json").read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"{out} is not empty and holds no readable run.json ({type(e).__name__}), so there is "
            f"nothing to check these settings against and no way to know which masks are already "
            f"there. Choose a new --out-dir, or pass --overwrite to replace this one."
        ) from None
    diffs = _control_plane_diff(prev, cfg, argv)
    if diffs:
        raise SystemExit(
            f"{out} holds a run with DIFFERENT settings, so continuing it would put two "
            f"configurations in one deliverable:\n  "
            + "\n  ".join(diffs)
            + "\nRun this configuration in a new --out-dir, or pass --overwrite to replace that one."
        )

    rows, missing = [], 0
    for r in _read_jsonl(out / "testcase.jsonl"):
        # THE MANIFEST IS ONLY TRUE IF THE FILE IS THERE. A row whose mask has been moved or deleted
        # is dropped and re-earned: keeping it would hand the next stage a path that does not open.
        if r.get("mask_filename") and Path(r["mask_filename"]).exists():
            rows.append(r)
        else:
            missing += 1
    overflow_rows = [
        r
        for r in _read_jsonl(out / "testcase_overflow.jsonl")
        if r.get("mask_filename") and Path(r["mask_filename"]).exists()
    ]
    delivered: dict = {}
    for r in rows:
        delivered[r.get("anomaly_type", "")] = delivered.get(r.get("anomaly_type", ""), 0) + 1
    claimed = {r["mask_filename"] for r in (*rows, *overflow_rows)}
    # `results` is run.json's per-mask record; keep exactly the rows whose mask survived above, so
    # the two halves of the record cannot disagree about which masks this directory holds.
    kept = {(r.get("anomaly_type"), Path(r["mask_filename"]).name) for r in rows}
    summary = [s for s in (prev.get("results") or []) if (s.get("type"), s.get("mask")) in kept]

    next_reuse: dict = {}
    orphans: dict = {}
    for texture, defect in cfg.anomaly_types:
        dtype = f"{texture}+{defect}"
        highest: dict = {}  # clean stem -> the largest `reuse` anything on disk has claimed
        n_orphan = 0
        for sub in (out / texture / "mask" / defect, out / texture / "_overflow" / defect):
            for p in sorted(sub.glob("*" + _MASK_SUFFIX)):
                hit = _split_indexed(p.name, _MASK_SUFFIX, 2)
                if hit is None:
                    continue
                stem, (reuse, _j) = hit
                highest[stem] = max(highest.get(stem, -1), reuse)
                if str(p.resolve()) not in claimed:
                    n_orphan += 1
        # The run logs too, and they matter on their own: a beam that accepted nothing leaves a log
        # and no mask, and the `reuse` it claimed is spent.
        for p in sorted((out / "_runlog" / dtype).glob("*" + _RUNLOG_SUFFIX)):
            hit = _split_indexed(p.name, _RUNLOG_SUFFIX, 1)
            if hit is not None:
                stem, (reuse,) = hit
                highest[stem] = max(highest.get(stem, -1), reuse)
        next_reuse[dtype] = {k: v + 1 for k, v in highest.items()}
        orphans[dtype] = n_orphan

    prior = {
        "runs": int((prev.get("resumed") or {}).get("runs", 1)) + 1,
        "delivered_before": len(rows),
        "usage_before": prev.get("usage") or {},
    }
    return _Resume(
        rows,
        overflow_rows,
        summary,
        delivered,
        next_reuse,
        orphans,
        missing,
        prior,
        prior_targets=prev.get("targets"),
    )


def _open_out_dir(out: Path, cfg, argv: list[str], *, overwrite: bool) -> "_Resume | None":
    """Decide what `out` is for this run, and leave it ready to be written into.

    One question, three answers. A POPULATED OUT-DIR IS A RUN TO CONTINUE, as long as it is THIS run:
    `_plan_resume` checks the whole control plane and reads back which `reuse` indices are spent, and
    a different configuration is refused by name. `--overwrite` says start over instead. An empty or
    absent one is a fresh run and needs only the `mkdir`.

    THIS IS THE FIRST THING TO TOUCH `out`, and it reads it before it writes it — a caller that
    creates run.log or a mask directory first hands the resume check its own preparation to read.
    Returns what the earlier process left, or None when there is no earlier process.
    test_a_resumed_run_makes_up_the_shortfall_without_touching_the_first_runs_masks,
    test_a_run_with_different_settings_is_refused_and_the_refusal_names_the_field.
    """
    resume = None
    if out.exists() and any(out.iterdir()) and not overwrite:
        resume = _plan_resume(out, cfg, argv)
    if overwrite and out.exists():
        # SAY WHAT IS BEING DESTROYED, before destroying it — an unconditional rmtree of whatever
        # --out-dir names, and the last moment the count exists (run.log is created after the `mkdir`
        # below, INSIDE the directory). stderr, not the logger, for the same reason.
        # overwrite names what it destroys before destroying it.
        n = sum(1 for _ in out.rglob("*_mask.png"))
        print(f"--overwrite: deleting {out} ({n} mask{'' if n == 1 else 's'})", file=sys.stderr)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return resume


def _next_trace_seq(trace_dir: Path) -> int:
    """The first call number free under an existing trace; 0 when there is nothing there.

    Every traced call writes `<trace>/<ctx>/<NNNNN>_<kind>/`, so a second process numbering from zero
    files its first calls into the first process's directories — and the tracer reports nothing when
    a write fails, by design, so the loss is silent. The DIRECTORIES are what collide, so they are
    what is counted: `index.jsonl` is missing a row for any call whose process died between the two
    writes.
    """
    if not trace_dir.is_dir():
        return 0
    highest = -1
    for d in (*trace_dir.glob("[0-9]*_*"), *trace_dir.glob("*/[0-9]*_*")):
        head = d.name.split("_", 1)[0]
        if head.isdigit() and d.is_dir():
            highest = max(highest, int(head))
    return highest + 1


def _failure_sig(e: BaseException) -> str:
    """What makes two beam failures THE SAME failure, for the repeated-failure guard below.

    The IDENTITY of the cause, not the sentence the server wrote about it: the classified `Failure`'s
    kind and status are the same on every occurrence, while its detail embeds `body[:200]` and a proxy
    echoes a request id into that. Below the transport it is COARSER — `classify_exception` folds
    every ValueError/KeyError/IndexError into `unparseable`, so two different KeyErrors share one
    signature. one cause wearing a fresh request id each beam still trips the repeated failure guard.
    """
    f = getattr(e, "failure", None)  # CallFailed carries the Failure the transport classified
    if f is None:
        f = classify_exception(e)  # a raise from anywhere else, classified the same way
    kind = getattr(f, "kind", "") or "unclassified"
    status = getattr(f, "status", None)
    return f"{type(e).__name__}/{kind}" + (f"/{status}" if status else "")


# --------------------------------------------------------------------------------------------------
# Ledger attribution — which type a degraded[...] line belongs to, once types interleave
# --------------------------------------------------------------------------------------------------
# Some `degraded[...]` emitters name their type in the detail and many engine-side ones do not; once
# the cross-type scheduler interleaves types, an untagged line's position in run.log says nothing
# about its type, so the tag has to. The tag is a thread-local (the `set_trace_ctx` pattern): set
# wherever a thread starts working for one type, injected into the DETAIL by the loguru patcher below.
_LEDGER = threading.local()


def _set_ledger_type(dtype: str) -> None:
    """Every `degraded[...]` line this thread logs from now on belongs to `dtype`; "" = none."""
    _LEDGER.dtype = dtype


_DEGRADED_LINE = re.compile(r"^(degraded\[[^\]]*\]: )(.*)", re.S)


def _tag_degraded(record: dict) -> None:
    """Loguru patcher: prefix ``[dtype]`` onto the DETAIL of a ``degraded[KIND]: detail`` line.

    Into the detail, never the kind: the readers (the test fixture, any run.log bucketer) split on
    ``]: `` and count kinds, so the shape they parse is preserved. A detail that already leads with
    this type's own tag is left alone — several emitters tag themselves.
    """
    tag = getattr(_LEDGER, "dtype", "")
    if not tag:
        return
    m = _DEGRADED_LINE.match(record["message"])
    if m and not m.group(2).startswith(f"[{tag}]"):
        record["message"] = f"{m.group(1)}[{tag}] {m.group(2)}"


def _install_ledger_tagger() -> None:
    """Install the patcher — multi-type runs only, so a single-type run's lines are byte-identical
    to what they always were. Reaches the patcher list exactly where the framework's own log module
    builds it (`cosmos_framework/utils/log.py` rebuilds `logger._options` with its path patcher);
    appending there reaches every `opt()`/`bind()` clone, which share the list."""
    patchers = log.logger._options[-2]
    if _tag_degraded not in patchers:
        patchers.append(_tag_degraded)


@dataclass(eq=False)
class Quota:
    """The retry-to-fill state for ONE defect type, and the decisions taken against it.

    Every number the decisions read is a field here rather than a variable a closure reaches back
    into. The loop that ACTS on them is the run-level :class:`_Scheduler`, which deals one shared
    pool's slots across every type's quota at once: :meth:`want_inflight` says how many beams this
    type wants, :meth:`next_task` which task is next, :meth:`collect` books one finished beam, and
    the scheduler-owned lifecycle fields say how it ended.

    The beam itself is NOT here. ``submit`` takes ``(clean, reuse)`` and returns a running
    :class:`~concurrent.futures.Future` of ``(masks, rec)``; what a beam IS — which models, which
    exemplars, which region — belongs to the caller, and none of it changes how many to keep in
    flight.
    """

    # A beam that accepts nothing still costs a full round-trip budget, so a configuration that can
    # never accept (a wrong model id, a dead judge, a defect the bar rejects outright) would otherwise
    # resubmit beams for as long as the process lives. Stop once this many consecutive beams have come
    # back empty: enough that an unlucky streak on a hard type rides through, few enough that a broken
    # run ends in minutes instead of burning quota indefinitely.
    DRY_LIMIT: ClassVar[int] = 60
    # Identical beam failures with nothing accepted yet: that many in a row is the configuration,
    # not the data. Distinct from the transport's exhaustion guard, which ends the run on the FIRST
    # call that runs out of attempts; what is left here is every beam failure that is not the
    # endpoint being gone, and nothing else counts those.
    SAME_FAILURE_LIMIT: ClassVar[int] = 5

    dtype: str
    cleans: list  # the clean images this type draws from, in round-robin order
    target_n: "int | None"  # accepted masks wanted; None = a single pass, one beam per clean
    pool_size: int
    beam_width: int  # the PRIOR for masks-per-beam, before any beam has reported
    max_rounds: "int | None"  # --max-rounds, as a per-clean multiplier -> an attempt cap
    reuse_start: dict  # clean STEM -> the first `reuse` index no earlier run has claimed
    submit: Callable  # (clean, reuse) -> Future of (masks, rec)
    on_accept: "Callable | None"  # called on THIS thread for every accepted mask, as it arrives
    runlog_dir: Path
    idx: Any  # the open index.jsonl handle one row per beam is appended to
    bar: Any  # tqdm over ACCEPTED masks, or None on a single pass
    # The accepted masks, as (clean, reuse, j, AcceptedMask). Supplied by the caller because the
    # coverage feedback reads the same list this loop appends to.
    accepted: list = field(default_factory=list)

    attempts: int = 0
    yield_attempts: int = 0  # beams that produced a real MEASUREMENT of yield (see `collect`)
    warned: bool = False  # the low-accept-rate warning is said once per type
    dry_streak: int = 0  # consecutive beams that accepted nothing
    # The classified cause of the last beam failure and how many in a row have carried it. The CAUSE,
    # not the message: a proxy echoes a request id into the detail, so six beams refused by one rate
    # limit would otherwise read as six different failures.
    fail_kind: "str | None" = None
    fail_run: int = 0
    inflight: set = field(default_factory=set)

    # Lifecycle, owned by the scheduler. `abandoned` is what holds the pinned no-drain-on-guard
    # rule under a SHARED pool: a failed type's trailing beams complete into the scheduler, which
    # must skip them — nothing is booked behind the repeated-failure guard — while their spend is
    # still recorded by the worker-side `finally` in `_run`.
    abandoned: bool = False
    # What settled this quota: None for a met (or honestly short) target, the exception for an
    # abandoned one. The thread waiting on `settled` re-raises it, which is how a type failure
    # reaches the per-type handler and an endpoint stop reaches `main`'s.
    outcome: "BaseException | None" = None
    settled: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.order = {str(cp): i for i, cp in enumerate(self.cleans)}  # stable distinct-first tiebreak
        # Where each clean's `reuse` numbering picks up. 0 on a fresh out-dir; past whatever a previous
        # process into this same out-dir already wrote, on a resumed one — the masks it produced are
        # named after this counter and nothing else keeps two runs' names apart.
        self.reuse_ctr = {str(cp): int(self.reuse_start.get(cp.stem, 0)) for cp in self.cleans}
        self.max_attempts = (self.max_rounds * len(self.cleans)) if self.max_rounds else None
        # A single pass (`target_n is None`) is a fixed task list — one beam per clean, reuse 0 —
        # metered through the same dealer as everything else.
        self.pending = [(cp, 0) for cp in self.cleans] if self.target_n is None else []
        self.t0 = time.time()

    # ----------------------------------------------------------------------------------------------
    def next_task(self) -> tuple:
        """The next (clean, reuse) to launch — distinct-first: the clean run the fewest times so far."""
        cp = min(self.cleans, key=lambda c: (self.reuse_ctr[str(c)], self.order[str(c)]))
        r = self.reuse_ctr[str(cp)]
        self.reuse_ctr[str(cp)] += 1
        return cp, r

    def target_inflight(self) -> int:
        """How many beams to keep in flight = remaining masks / masks-per-beam.

        Before any data, assume a beam yields ~``beam_width`` masks (its batch), NOT 1 — else a cold
        start with pool_size >> target fires `target` beams at once, each returns a batch, and the run
        overshoots ~beam_width-fold before a single beam reports back. Once beams complete, the
        observed yield replaces the estimate.
        """
        need = self.target_n - len(self.accepted)
        if need <= 0:
            return 0
        # SHRUNK TOWARD THE PRIOR, not a plain running mean: the estimate sits in the DENOMINATOR, so
        # one unlucky beam does not nudge the launch target, it multiplies it. The weight is a stated
        # prior, not a fitted one. Overshoot past the target is BATCH ATOMICITY — a beam returns its
        # whole batch and the surplus is trimmed aside, not deleted.
        # test_a_dead_endpoint_stops_the_run_and_what_the_pool_had_already_earned_is_kept.
        _PRIOR_BEAMS = 3.0  # how many beams of evidence the prior is worth
        prior = float(max(1, self.beam_width))
        yield_per_beam = (_PRIOR_BEAMS * prior + len(self.accepted)) / (_PRIOR_BEAMS + self.yield_attempts)
        return min(self.pool_size, max(1, math.ceil(need / max(yield_per_beam, 0.15))))

    def want_inflight(self) -> int:
        """The total in-flight beams this type wants right now — the fill gates as one number.

        The dealer (:func:`plan`) tops the quota up toward this and never past it, so these are
        exactly the per-launch gates: a single pass wants every not-yet-launched task; a dry streak
        wants nothing more; ``--max-rounds`` caps total launches at ``max_attempts``.
        """
        if self.target_n is None:
            return len(self.inflight) + len(self.pending)
        if self.dry_streak >= self.DRY_LIMIT:
            return 0
        want = self.target_inflight()
        if self.max_attempts:
            want = min(want, max(0, self.max_attempts - self.attempts))
        return want

    def collect(self, fut) -> None:
        """Book one finished beam: its masks, its record, and what it says about this configuration."""
        try:
            masks, rec = fut.result()
        except Exception as e:  # one bad beam must not sink the whole type/run — log, count, continue
            self.attempts += 1  # counts for the bar and the dry-streak guard...
            # ...but NOT for the yield estimate: feeding a raised beam in drags it toward the 0.15
            # floor, which multiplies the in-flight target by up to 6.7x.
            self.dry_streak += 1
            # COUNTED by cause, REPORTED in full — see `_failure_sig`.
            detail = f"{type(e).__name__}: {e}"[:200]
            sig = _failure_sig(e)
            self.fail_run = self.fail_run + 1 if self.fail_kind == sig else 1
            self.fail_kind = sig
            log.warning(
                f"degraded[beam_failed]: [{self.dtype}] {detail} — the beam's spend is counted, its masks are not"
            )
            log.warning(f"[{self.dtype}] a beam failed ({detail}); skipping it, continuing")
            if self.fail_run >= self.SAME_FAILURE_LIMIT and not self.accepted:
                raise RuntimeError(
                    f"{self.SAME_FAILURE_LIMIT} beams in a row failed with the same error ({sig}) and "
                    f"none has ever succeeded — this is the configuration, not the data. "
                    f"Last: {detail}"
                ) from e
            return
        self.attempts += 1
        # A beam MEASURES this configuration's yield if it got a verdict on ANY candidate, including
        # one that judged five and honestly rejected all five. `judged` is the test, not `scores`,
        # which holds ACCEPTED masks only.
        # a beam reports verdicts it got not candidates it sent.
        if int(rec.get("judged") or 0) > 0:
            self.yield_attempts += 1
        self.fail_kind, self.fail_run = None, 0
        self.dry_streak = 0 if masks else self.dry_streak + 1
        if self.dry_streak == self.DRY_LIMIT:
            log.warning(
                f"degraded[target_short:dry_streak]: [{self.dtype}] delivered "
                f"{len(self.accepted)} of {self.target_n} — the target was abandoned, not met"
            )
            log.warning(
                f"[{self.dtype}] {self.DRY_LIMIT} consecutive beams accepted nothing — "
                f"stopping this type at {len(self.accepted)}/{self.target_n}. Check the "
                f"models, the bar and the data."
            )
        self.accepted.extend(masks)
        if self.on_accept:  # write each mask to disk AS it is accepted — incremental + crash-safe.
            for _cp, _reuse, _j, _m in masks:  # this runs on the DRIVER's thread, so no race
                self.on_accept(_cp, _reuse, _j, _m)
        # The run log is the beam's LAST write: a resume reads a log as the beam having happened (its
        # `reuse` is spent, and a one-pass run counts the clean as done), so it may exist only once every
        # mask it accounts for is on disk with its manifest row.
        atomic_write(
            self.runlog_dir / f"{rec['stem']}_{rec['reuse']}_run.log",
            lambda t: t.write_text(json.dumps(rec, indent=2)),
        )
        self.idx.write(
            json.dumps(
                {
                    "ts": rec["ts"],
                    "clean": rec["clean"],
                    "reuse": rec["reuse"],
                    "accepted": rec["accepted"],
                    "best_score": max(rec["scores"], default=0),
                    "api_calls": rec["api_calls"],
                    "log": f"{rec['stem']}_{rec['reuse']}_run.log",
                }
            )
            + "\n"
        )
        self.idx.flush()
        if self.bar is None:
            return
        self.bar.update(len(masks))  # update(0) on a barren beam still refreshes the postfix below
        rate = len(self.accepted) / max(1, self.attempts)
        remaining = max(0, self.target_n - len(self.accepted))
        eta = (
            (remaining / max(rate, 1e-3)) * ((time.time() - self.t0) / max(1, self.attempts)) / 60 if remaining else 0.0
        )
        self.bar.set_postfix_str(
            f"{self.attempts} runs · {len(self.inflight)} in-flight · rate {rate:.0%} · ETA ~{eta:.0f}m"
        )
        if remaining and rate < 0.2 and self.attempts >= len(self.cleans) and not self.warned:
            self.warned = True
            log.warning(f"[{self.dtype}] low accept rate ({rate:.0%}) — filling to {self.target_n} may take a while")

    def drain(self, futs) -> None:
        """Book what the beams still in the pool earned, on the way out of an ENDPOINT stop.

        A finished beam has ALREADY PAID for its masks and `collect` is the only thing that writes
        them, so leaving the pool without collecting loses every one. The wait costs nothing extra:
        leaving the `with` calls `ThreadPoolExecutor.shutdown(wait=True)` anyway.
        test_a_dead_endpoint_stops_the_run_and_what_the_pool_had_already_earned_is_kept.
        """
        wait(futs)
        for f in futs:
            try:
                self.collect(f)
            except (Exception, TransportUnavailable) as e:  # noqa: BLE001 — a stop is already in flight
                # EXPECTED: every beam still in the pool was talking to the endpoint that has just
                # stopped answering, and that must not cost the finished ones their masks.
                log.warning(
                    f"degraded[beam_lost_on_stop]: [{self.dtype}] {type(e).__name__}: {e} — "
                    f"a beam in flight when the run stopped could not be booked"
                )


def plan(wants: list, free: int) -> list:
    """Deficit dealer: each of the `free` pool slots goes to the hungriest quota — the one holding
    the fewest beams among those that want more — with the given order (= `--anomaly-types` order)
    breaking ties, until slots or hunger run out.

    Each `wants` row is ``(quota, want, inflight)``: ``want`` the total in-flight beams that quota's
    own gates allow right now (:meth:`Quota.want_inflight`), ``inflight`` what it already holds.
    PURE — no scheduler state enters — so the fairness properties are testable the way `decide()` is.

    Fairness, both directions, falls out of the least-held rule. A type whose rejections inflate its
    `want` (the 0.15-yield floor in `target_inflight`) can never take a slot while any hungry type
    holds fewer beams than it does. And a hungry type can never be starved: holding the least, it
    takes the very next slot. A fixed-order pass does NOT have that property — slots usually free up
    one at a time, and a pass truncated at one slot restarts at the first-declared type every time,
    which starves a type admitted while the pool was full.
    """
    deals: list = []
    want = {q: w for q, w, _held in wants}
    held = {q: n for q, _w, n in wants}
    order = [q for q, _w, _n in wants]
    while len(deals) < free:
        hungry = [q for q in order if held[q] < want[q]]
        if not hungry:
            break
        q = min(hungry, key=lambda h: held[h])  # `min` is stable: ties fall to declaration order
        deals.append(q)
        held[q] += 1
    return deals


class _Scheduler:
    """ONE fill loop for every type: a shared worker pool, a single wait loop, one writer thread.

    Everything arrives on one queue — finished beams, freshly prepped quotas, writer-thread work,
    per-type completion — and is consumed by exactly one thread, the one that calls :meth:`pump`.
    So every booking, every `emit`/flush and every trim happens on one thread, with no lock around
    the deliverables. Beams keep flying while the pump is busy; only their booking waits.

    With one type this degenerates to one quota, dealing = "top up to `want_inflight`", the pump =
    wait -> collect -> refill —
    `test_the_single_type_run_is_bit_identical_to_the_pre_scheduler_code` holds it at the byte
    level.
    """

    def __init__(self, ex, pool_size: int, n_types: int, deliv=None):
        self.ex = ex  # the run-level pool every quota's `submit` posts into
        self.pool_size = pool_size
        self.multi = n_types > 1
        self._n_types = n_types
        self._deliv = deliv  # flushed once per finished type (multi-type runs; None otherwise)
        self._msgs: queue.SimpleQueue = queue.SimpleQueue()
        # Guards the dead-vs-enqueue race: `close` flushes the queue and marks the scheduler dead
        # under this lock, so a message posted from another thread is either flushed there or
        # refused here — never silently parked on a queue nobody pumps.
        self._post_lock = threading.Lock()
        self._quotas: list[Quota] = []  # admission order == declaration order (prep is serial)
        self._owner: dict = {}  # future -> its quota
        self._live: set = set()  # every launched, not-yet-consumed future — abandoned types included
        self._finished = 0
        self._writer = threading.get_ident()  # the ONLY thread that books beams and writes deliverables
        self._bar = None  # one aggregate tqdm across every type (multi-type runs)
        self._bar_quotas: list[Quota] = []
        self.dead: "BaseException | None" = None

    # ---- callable from any thread ----------------------------------------------------------------
    def admit(self, q: Quota) -> None:
        """A prepped quota enters the dealing set; the pump deals it its first beams."""
        with self._post_lock:
            if self.dead is not None:
                raise self.dead
            self._msgs.put(("admit", q))

    def type_finished(self) -> None:
        self._msgs.put(("finished", None))

    def fatal(self, e: BaseException) -> None:
        """An endpoint stop found off the pump thread (a type's prep): the pump re-raises it."""
        self._msgs.put(("fatal", e))

    def on_writer(self, fn):
        """Run `fn` on the writer thread and hand back its result.

        Every current caller is a type's driver thread asking for its trim, so the work is queued to
        the pump — which is what keeps the trim's deliverable rewrites on the one writing thread
        while another type's beams are being booked. The inline branch keeps the call safe from the
        writer thread itself.
        """
        if threading.get_ident() == self._writer:
            return fn()
        box: list = []
        done = threading.Event()
        with self._post_lock:
            if self.dead is not None:
                raise self.dead
            self._msgs.put(("call", (fn, box, done)))
        done.wait()
        verdict, payload = box[0]
        if verdict == "err":
            raise payload
        return payload

    # ---- the pump: runs on the writer thread only -------------------------------------------------
    def pump(self, until: "Quota | None" = None) -> None:
        """Consume messages until every type has finished (`until` None) or that quota settles.

        Raises what stops the RUN: `TransportUnavailable` — after ONE drain across every quota, so
        each books what its finished beams already earned — reaches `main`'s handler unchanged.
        What stops one TYPE never leaves here; it settles into that quota's `outcome`.
        """
        try:
            while not (until.settled.is_set() if until is not None else self._finished >= self._n_types):
                kind, payload = self._msgs.get()
                if kind == "future":
                    self._book(payload)
                elif kind == "admit":
                    self._quotas.append(payload)
                    self._attach_bar(payload)
                    self._deal()
                    # A quota admitted already wanting nothing would otherwise never settle:
                    # `_check_done` runs only when a beam books, and a quota that launches no
                    # beams books none — the driver then waits on `settled` forever. The driver's
                    # own want<=0 early return makes this unreachable today; checking at admission
                    # is what makes the deadlock impossible rather than merely avoided.
                    self._check_done(payload)
                elif kind == "call":
                    fn, box, done = payload
                    try:
                        box.append(("ok", fn()))
                    except Exception as e:  # noqa: BLE001 — the failure belongs to the requesting type
                        box.append(("err", e))
                    finally:
                        if not box:  # a BaseException is passing through the pump; unblock the waiter
                            box.append(("err", RuntimeError("the scheduler stopped mid-request")))
                        done.set()
                elif kind == "finished":
                    self._finished += 1
                    if self._deliv is not None:
                        self._deliv.flush()  # crash-safe: deliverables reflect every completed type
                elif kind == "fatal":
                    raise payload
        except TransportUnavailable:
            # THE ENDPOINT IS GONE — one loop, so one drain: EVERY quota books what its finished
            # beams already earned (strictly better than the per-type pools, where only the running
            # type had beams to save), then the stop reaches `main`'s handler unchanged.
            #
            # THIS EXCEPTION ONLY, not `BaseException`: the repeated-failure abort claims nothing
            # was ever accepted, and a KeyboardInterrupt asks for the pool to stop, not be waited on.
            # test_a_dead_endpoint_stops_the_run_and_what_the_pool_had_already_earned_is_kept,
            # test_the_broken_configuration_abort_does_NOT_book_the_beams_behind_it.
            self._drain_all()
            raise

    def _book(self, fut) -> None:
        """Route one finished beam to its quota — the wait->collect->refill step, shared-pool form."""
        q = self._owner.pop(fut)
        self._live.discard(fut)
        q.inflight.discard(fut)
        if q.abandoned:
            # A failed type's trailing beam: deliberately unbooked — the guard's abort claims
            # nothing was ever accepted (test_the_broken_configuration_abort_does_NOT_book_the_
            # beams_behind_it) — and its spend is already in `_RUN_USAGE` from the worker `finally`.
            # The freed slot is still DEALT before the return: the abandoned type's trailing beams
            # may hold every remaining slot, and a slot nothing re-deals never rejoins the pool —
            # the other types would sit starved behind a type that is already over.
            self._deal()
            for other in self._quotas:
                self._check_done(other)
            return
        _set_ledger_type(q.dtype if self.multi else "")
        try:
            q.collect(fut)
        except Exception as e:
            # The per-type diagnosis (the repeated-failure guard, or anything else `collect` can
            # raise): THIS type is over, un-drained per the pin above; every other quota carries on.
            q.abandoned = True
            self._settle(q, e)
        finally:
            _set_ledger_type("")
        self._deal()
        for other in self._quotas:
            self._check_done(other)

    def _deal(self) -> None:
        """Hand the free slots out by deficit round-robin and launch the dealt beams."""
        free = self.pool_size - len(self._live)
        active = [q for q in self._quotas if not (q.abandoned or q.settled.is_set())]
        for q in plan([(q, q.want_inflight(), len(q.inflight)) for q in active], free):
            cp, reuse = q.pending.pop(0) if q.target_n is None else q.next_task()
            fut = q.submit(cp, reuse)
            q.inflight.add(fut)
            self._live.add(fut)
            self._owner[fut] = q
            fut.add_done_callback(lambda f: self._msgs.put(("future", f)))

    def _check_done(self, q: Quota) -> None:
        """A quota is done when nothing is in flight and its own gates launch nothing more."""
        if q.settled.is_set() or q.abandoned or q.inflight or q.want_inflight() > 0:
            return
        self._settle(q, None)

    def _settle(self, q: Quota, outcome: "BaseException | None") -> None:
        q.outcome = outcome
        q.settled.set()

    def _drain_all(self) -> None:
        for q in self._quotas:
            if q.abandoned:
                continue  # its beams are unbooked by decision, not by this stop
            _set_ledger_type(q.dtype if self.multi else "")
            try:
                q.drain(set(q.inflight))
            finally:
                _set_ledger_type("")

    def _attach_bar(self, q: Quota) -> None:
        """Multi-type progress is ONE bar over every type's target, per-type counts in the postfix;
        a single-type run keeps its own bar (built by `_run_type`)."""
        if not self.multi or not q.target_n:
            return
        if self._bar is None:
            self._bar = tqdm(total=0, desc="all types", unit="mask")
        self._bar_quotas.append(q)
        self._bar.total = (self._bar.total or 0) + q.target_n
        self._bar.refresh()
        q.bar = _BarView(self)

    def close(self, exc: "BaseException | None" = None) -> None:
        """Wake every thread still parked on this scheduler; idempotent, run on every exit path.

        Runs queued writer work FIRST — this thread is the writer, and the queued work is a finished
        type's trim, which the stop must not strand: a trim the stop swallows leaves that type
        over-delivered in testcase.jsonl, and a resume then sees no shortfall, so nothing ever cuts
        it back. Then settles unsettled quotas with `exc` (their waiters re-raise it) and refuses
        everything posted afterwards.
        """
        with self._post_lock:
            self.dead = self.dead or exc or RuntimeError("the scheduler is finished")
            leftovers = []
            while True:
                try:
                    leftovers.append(self._msgs.get_nowait())
                except queue.Empty:
                    break
        for kind, payload in leftovers:
            if kind == "admit":
                self._quotas.append(payload)
            elif kind == "call":
                fn, box, done = payload
                try:
                    box.append(("ok", fn()))
                except Exception as e:  # noqa: BLE001 — the failure belongs to the requesting type
                    box.append(("err", e))
                finally:
                    done.set()
        for q in self._quotas:
            if not q.settled.is_set():
                self._settle(q, self.dead)
        if self._bar is not None:
            self._bar.close()


class _BarView:
    """A quota's handle on the run-wide bar: updates feed the one total, and the postfix renders
    every type's progress instead of the calling type's own stats."""

    def __init__(self, sched: _Scheduler):
        self._sched = sched

    def update(self, n: int) -> None:
        self._sched._bar.update(n)

    def set_postfix_str(self, _own: str) -> None:
        s = self._sched
        s._bar.set_postfix_str(
            " · ".join(f"{q.dtype} {min(len(q.accepted), q.target_n)}/{q.target_n}" for q in s._bar_quotas)
        )


def _run_type(
    vlm,
    cfg,
    texture,
    defect,
    cleans,
    vp_for,
    defects,
    *,
    region_cache_dir=None,
    shape_mode,
    target_n,
    max_rounds,
    pool_size,
    runlog_dir,
    emb_cache_dir=None,
    on_accept=None,
    reuse_start=None,
    scheduler=None,
    prep_done=None,
):
    """Produce judge-accepted masks for one defect type, writing a per-type run log.

    ``target_n`` set  → CONTINUOUS retry-to-fill: keep beams in flight, each drawing the next
    ``(clean, reuse)`` from a distinct-first round-robin over the cleans (reuse = which repetition
    of that clean), so the concurrency budget saturates even with fewer cleans than slots.
    ``target_n`` None → a single pass: one beam per clean (reuse 0).

    ``scheduler`` None → this call owns the run: it opens its own ``pool_size`` pool and pumps the
    scheduler itself, so one type behaves exactly as it always has. ``scheduler`` given → the run is
    multi-type: this call is the type's PREP stage (visual prompts, exemplars, regions — serial by
    the caller's turnstile), it admits its quota into the shared pool, calls ``prep_done`` so the
    next type may start prepping, and waits for the scheduler to settle the quota.

    The ``Vlm`` permit is the API cap, one per base_url, and the shared pool is the only beam-level
    fan-out under it — judging is serial, so a beam holds at most one permit at a time (per-call
    acquire, no deadlock).

    RUN LOG (auto): each beam writes ``<runlog_dir>/<clean>_<reuse>_run.log`` and appends a row to
    ``<runlog_dir>/index.jsonl`` as it completes; ``reuse_start`` (clean STEM -> first free index) is
    what keeps a resumed run off the previous one's ``<stem>_<reuse>_<j>_mask.png`` filenames.

    Returns ``(accepted, attempts)`` where ``accepted`` is a list of ``(clean_path, reuse, AcceptedMask)``."""
    type_cfg = replace(cfg, shape_mode=shape_mode)
    dtype = f"{texture}+{defect}"
    # Multi-type runs prefix per-type identity where a single-type run never carried one: trace
    # groups and the ledger tag. Gated on the DECLARED types, so a single-type run carries no prefix
    # at all.
    multi = len(cfg.anomaly_types) > 1
    runlog_dir.mkdir(parents=True, exist_ok=True)
    # Resolve visual prompts AND exemplars up front (serial) so parallel workers don't race the vp
    # cache / map precompute or the DINOv2 'similar' backbone — the routing VLM call, any precompute,
    # and the (per-clean, reuse-invariant) exemplar ranking each happen exactly once per clean here.
    vps = {str(cp): vp_for(cp, dtype) for cp in cleans}
    exs = {
        str(cp): load_exemplars(
            cfg.dataset_path,
            texture,
            defect,
            k=cfg.num_examples,
            sampling="similar",
            clean_path=str(cp),
            emb_cache_dir=emb_cache_dir,
        )
        for cp in cleans
    }
    # Attach each exemplar's own distill-time caption (host / extent / shape), so every prompt that
    # shows an exemplar also SAYS what its real annotation covers. The card's synthesized flags can
    # mis-vote against their own evidence (a pool captioned 10/16 'covers all of one host' has
    # shipped as fill part); the per-image captions ARE that evidence, and here they travel with
    # the images they describe. `defect_library` sits beside the embedding cache by construction.
    _lib = str(Path(emb_cache_dir).parent / "defect_library") if emb_cache_dir else None
    if _lib:
        for cp in cleans:
            for ex in exs[str(cp)]:
                if type_cfg.exemplar_captions and not ex.observation and ex.defect_path:
                    ex.observation = observation_for(texture, defect, cfg.gen_model, _lib, ex.defect_path, mask=ex.mask)
                if type_cfg.exemplar_vp and ex.annotated is None:
                    # Exemplar visual parity: the example in the candidate's modality. Same serial
                    # section as the vp precompute above, for the same GPU-race reason; content-hash
                    # caching makes it one pass per UNIQUE exemplar image.
                    ex.annotated = annotate_exemplar(vps[str(cp)], ex)
    # The accepted masks live here rather than inside the fill loop because `_covered` reads them:
    # the coverage feedback has to see the sites THIS clean already carries, and that is the same
    # list the loop appends to.
    accepted: list = []

    # Whether one accepted mask of this type spans several sites. Read once: it decides how many regions
    # a beam is handed, and run_beam separately reads it to decide whether the region binds at all.
    _multi_type = read_flags(defects.card(dtype) or "").get("distribution_kind", "") == "multi"
    # Regions handed to one beam: one per candidate slot, never more than the beam has slots.
    _slot_k = max(1, min(int(getattr(type_cfg, "slot_regions", 1)), int(getattr(type_cfg, "num_candidates", 1))))

    # ROUND-ROBIN (--coverage roundrobin): enumerate each clean's VALID REGIONS ONCE (inference-legal, no
    # GT, DEFECT-AGNOSTIC), then hand each run a distinct region so accepted defects spread across the sites.
    # The region COUNT gates whether we can split at all (>=2 spread, ==1 confine-only, 0 unbounded — see below).
    regions_by_clean, region_ctr = {}, {}
    if type_cfg.coverage_feedback in ("roundrobin", "choose"):
        try:
            for cp in cleans:
                # Own trace context, so the region call's prompt/images/answer land under
                # <trace>/<stem>_regions/ instead of being mixed into whichever beam ran last.
                # Type-prefixed only when types interleave: two defects on one texture share stems.
                set_trace_ctx(f"{dtype}_{cp.stem}_regions" if multi else f"{cp.stem}_regions")
                regions_by_clean[str(cp)] = enumerate_regions(
                    vlm, str(cp), dtype, exs[str(cp)], type_cfg.gen_model, cache_dir=region_cache_dir
                )
                # The site counter continues past what earlier processes into this out-dir handed out:
                # every beam they completed on this clean (`reuse_start` counts the ones that left a mask
                # or a run log) advanced it by the same step `_target_region` uses.
                region_ctr[str(cp)] = int((reuse_start or {}).get(cp.stem, 0)) * (1 if _multi_type else _slot_k)
        finally:
            set_trace_ctx("")  # a raise here must not leave later calls tagged as this clean's regions
        log.info(
            f"{dtype}: enumerated valid-regions/clean = "
            + ", ".join(f"{cp.stem}:{len(regions_by_clean[str(cp)])}" for cp in cleans)
        )
        # The regions are re-enumerated every run and the model does not have to agree with itself, so
        # persist the boxes it returned: this is what a later run is compared against, and what a drawn
        # ROI is checked for coverage against.
        rdir = Path(runlog_dir).parent / "_regions"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"{dtype}.json").write_text(
            json.dumps({Path(k).name: v for k, v in regions_by_clean.items()}, indent=1)
        )

    def _covered(cp):  # binary masks already accepted on THIS clean (coverage feedback); None unless on
        if type_cfg.coverage_feedback not in ("localize", "propose", "both", "choose"):
            return None
        return [m.mask for (c, _r, _j, m) in accepted if str(c) == str(cp)]

    def _target_region(cp):
        """A distinct valid region for EVERY candidate slot of the next beam, not one for the whole beam.

        One region per beam caps the sites a run can touch at the number of beams it launches, so
        improving yield would delete sites. `run_beam` honours a per-slot list — each candidate slot
        gets its own region; that this builds one is not pinned anywhere.
        """
        regs = regions_by_clean.get(str(cp)) or []
        if type_cfg.coverage_feedback != "roundrobin" or not regs:
            return None  # off, or NO discrete valid local site -> the whole board is in play (nothing to split)
        if len(regs) == 1:
            return regs[0]  # single valid region -> confine every run to it, but no spread
        # A multi type gets ONE region — the beam anchors on it rather than being bounded by it, so
        # handing it `_slot_k` would advance this counter by `_slot_k` for one site used.
        if _multi_type:
            i = region_ctr[str(cp)]
            region_ctr[str(cp)] += 1
            return regs[i % len(regs)]
        i = region_ctr[str(cp)]
        region_ctr[str(cp)] += _slot_k
        return [regs[(i + j) % len(regs)] for j in range(_slot_k)]

    def _run(cp, reuse, covered=None, target_region=None):  # one beam on one clean (its `reuse`-th run)
        # Group this beam's --trace-dir entries (serial judging = 1 thread); dtype-prefixed only when
        # types interleave. The ledger tag makes this worker's degraded[...] lines attributable too.
        set_trace_ctx(f"{dtype}_{cp.stem}_r{reuse}" if multi else f"{cp.stem}_r{reuse}")
        _set_ledger_type(dtype if multi else "")
        # Snapshot here, delta in `finally`: a beam that RAISES has still spent its tokens — it can
        # die in the final judge after twenty calls, and `_collect` treats that as routine.
        _u0, _c0, _a0 = get_usage(), get_call_count(), get_api_ms()
        _f0 = get_failed_count()
        try:
            res = run_beam(
                str(cp),
                dtype,
                vlm=vlm,
                visual_prompt=vps[str(cp)],
                defects=defects,
                cfg=type_cfg,
                exemplars=exs[str(cp)],
                covered=covered,
                target_region=target_region,
                regions=regions_by_clean.get(str(cp)),
                profile=True,
            )  # per-call API durations -> summed-API-time/mask throughput
        finally:
            # `finally`, on THIS beam's own worker thread, and read from the thread-local counters
            # rather than from `res`, which does not exist on the raise path.
            _u1 = get_usage()
            _RUN_USAGE.append(
                {
                    "usage": {k: _u1.get(k, 0) - _u0.get(k, 0) for k in _u1},
                    "api_calls": get_call_count() - _c0,
                    "failed": {k: v - _f0.get(k, 0) for k, v in get_failed_count().items()},
                    "api_ms": get_api_ms() - _a0,
                }
            )
            # ...and DROP anything this beam staged and never sent. The tracer keeps pending masks
            # in `threading.local()`, drained by the next traced call; the last refine round stages
            # every surviving candidate, so a beam that ends without a final-judge call leaves them
            # on a POOLED thread and the next `(clean, reuse)` files board A's masks under board
            # B's propose call. Same reason the usage delta is here: the beam may have raised.
            discard_staged_masks()
            _set_ledger_type("")  # the pooled thread's next beam may belong to another type
        passed = [m for m in res.masks if m.verdict.passed]
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": dtype,
            "clean": cp.name,
            "clean_path": str(cp.resolve()),
            "stem": cp.stem,
            "reuse": reuse,
            "shape_mode": shape_mode,
            "visual_prompt": getattr(vps[str(cp)], "name", "?"),
            "accepted": len(passed),
            "scores": [int(m.verdict.score) for m in passed],
            "masks": [f"{cp.stem}_{reuse}_{j}_mask.png" for j in range(len(passed))],
            # The edits that produced each ACCEPTED mask, aligned with `masks` above. `applied` below
            # says which edits happened; this says which of them ended up in something that passed —
            # the difference between a record of activity and a record of what works.
            "trails": [list(getattr(m, "trail", ()) or []) for m in passed],
            # getattr, like every other field here: a run log is an observation, and no observation
            # should be able to fail the run it is observing.
            "dims": [{d: int(v) for d, v in (getattr(m.verdict, "scores", None) or {}).items()} for m in passed],
            # WHY each of those numbers, in the judge's own words — merged across the consensus
            # samples, so every dimension carries a comment even when one sample omitted it. Aligned
            # with `masks` like `dims` and `trails`; the per-mask copy in run.json's `results` is
            # keyed by mask NAME instead, which is the join a board wants. This record is the one
            # that covers a mask the trim later moves under `_overflow`.
            "reasons": [dict(getattr(m.verdict, "reasons", None) or {}) for m in passed],
            "feedback": [str(getattr(m.verdict, "feedback", "") or "") for m in passed],
            # per round, per candidate: what the engine APPLIED (the trace holds what was ASKED for,
            # so together they show whether an op survived the vocabulary filter or the destroy guard)
            "applied": getattr(res, "trace", None),
            # Candidates the judge never returned a verdict on (rate-limited, or the call gave up).
            # A beam with masks=0 and every candidate unjudged is not a measurement of this
            # configuration's yield — it is the endpoint refusing — and the fill loop must not read
            # it as one.
            "unjudged": getattr(res, "unjudged", 0),
            "judged": getattr(res, "judged", 0),  # candidates that DID get a verdict
            "api_calls": getattr(res, "api_calls", None),
            "usage": dict(getattr(res, "usage", None) or {}),
            "api_ms": getattr(res, "api_ms", 0.0),  # pure HTTP wait (network + server inference)
            "non_api_ms": getattr(res, "non_api_ms", 0.0),  # LOCAL: image-encode/view-build/CV/parse + GIL-wait
            "wall_ms": getattr(res, "wall_ms", 0.0),  # api + non_api (this beam's serial wall time)
            # Where the API time went, per call site, for the sites this beam REACHED; `{}` when it
            # did not profile. SECONDS — the engine rounds to 0.1 s, so the `_s` suffix carries the
            # unit rather than the value being converted to match its `_ms` neighbours.
            "phase_times_s": dict(getattr(res, "phase_times", None) or {}),
        }
        # carry each mask's within-beam index j so its on-disk name stays f"{stem}_{reuse}_{j}" even
        # after trimming/reordering — i.e. the surviving files still line up with the run log's `masks`.
        return [(cp, reuse, j, m) for j, m in enumerate(passed)], rec

    def _submit(cp, reuse):
        """One beam, launched. The coverage feedback and the region assignment are read HERE, on the
        scheduler's thread at submit time, because both advance state: `_target_region` walks the site
        counter, and `_covered` reads the masks accepted so far."""
        return sched.ex.submit(_run, cp, reuse, _covered(cp), _target_region(cp))

    def _quota(idx, bar):
        return Quota(
            dtype=dtype,
            cleans=cleans,
            target_n=target_n,
            pool_size=pool_size,
            beam_width=cfg.beam_width,
            max_rounds=max_rounds,
            reuse_start=dict(reuse_start or {}),
            submit=_submit,
            on_accept=on_accept,
            runlog_dir=runlog_dir,
            idx=idx,
            bar=bar,
            accepted=accepted,
        )

    if scheduler is None:
        # This call owns the run: pool, scheduler and pump all live here, on this thread — one
        # quota topped up to `want_inflight`, wait -> collect -> refill until it settles.
        with ThreadPoolExecutor(max_workers=pool_size) as ex, open(runlog_dir / "index.jsonl", "a") as idx:
            sched = _Scheduler(ex, pool_size, n_types=1)
            bar = tqdm(total=target_n, desc=dtype, unit="mask") if target_n else None
            q = _quota(idx, bar)
            sched.admit(q)
            sched.pump(until=q)
            if bar is not None:
                bar.close()
    else:
        sched = scheduler
        with open(runlog_dir / "index.jsonl", "a") as idx:
            q = _quota(idx, None)  # the shared scheduler attaches its aggregate bar
            sched.admit(q)  # enters the dealing set; other types' beams are already flying
            if prep_done:
                prep_done()  # the next type may start ITS prep now — prep is serial, fills are not
            q.settled.wait()
    if q.outcome is not None:
        raise q.outcome
    # A `--max-rounds` exhaustion is a SHORTFALL, not an empty type — both otherwise render as a
    # short testcase.jsonl and exit 0.
    if target_n and len(accepted) < target_n and q.max_attempts and q.attempts >= q.max_attempts:
        log.warning(
            f"degraded[target_short:max_rounds]: [{dtype}] delivered {len(accepted)} of "
            f"{target_n} — the --max-rounds cap of {q.max_attempts} attempts was reached"
        )
    return accepted, q.attempts


def _trim_to_target(accepted: list, n: int) -> list:
    """Cut an over-generated ``accepted`` list (each beam returns a BATCH, so the fill overshoots) down
    to exactly ``n``, keeping the highest-judged masks but round-robin ACROSS cleans — so the trim
    preserves board diversity (the point of cycling cleans) instead of collapsing to one board."""
    by_clean: dict = {}
    for item in accepted:
        by_clean.setdefault(str(item[0]), []).append(item)
    for lst in by_clean.values():  # best-judged first within each clean (overall score, then dim-sum tiebreak)
        lst.sort(
            key=lambda t: (int(t[3].verdict.score), sum(int(v) for v in t[3].verdict.scores.values())), reverse=True
        )
    out: list = []
    queues = list(by_clean.values())
    while len(out) < n and any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
                if len(out) >= n:
                    break
    return out


# --------------------------------------------------------------------------------------------------
# The deliverables — the files this run hands over, and the lists they are written from
# --------------------------------------------------------------------------------------------------
def _jsonl(rows: list) -> Callable[[Path], None]:
    """A writer of one JSON object per line, for :func:`atomic_write`."""
    return lambda t: t.write_text("".join(json.dumps(r) + "\n" for r in rows))


@dataclass
class Deliverables:
    """What the run hands over: ``testcase.jsonl``, its overflow and ``run.json``.

    Every file is rewritten in full, never appended to, so :meth:`flush` is safe to call after every
    accepted mask and after every type. Two of the three are a projection of the lists below;
    ``run.json`` is those lists plus the control plane and what the run has spent so far, which
    :meth:`record` reads from `cfg`, the transport and `_RUN_USAGE`.

    On a resume the lists START FROM WHAT IS ALREADY THERE, because a rewrite from empty would
    SHRINK the deliverable.
    test_a_resumed_run_makes_up_the_shortfall_without_touching_the_first_runs_masks.
    """

    out: Path
    cfg: Any
    argv: list[str]
    endpoint_report: Callable[[], dict]  # per-endpoint calls/refusals/waiting, read at write time
    resume: "_Resume | None" = None

    # Filled by the run, not by the caller: `__post_init__` decides what they start from.
    testcase_rows: list[dict] = field(default_factory=list, init=False)
    # accepted past the target; run-level, so a later type cannot truncate an earlier one's manifest
    overflow_rows: list[dict] = field(default_factory=list, init=False)
    summary: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.testcase = self.out / "testcase.jsonl"
        if self.resume:
            self.testcase_rows = list(self.resume.rows)
            self.overflow_rows = list(self.resume.overflow_rows)
            self.summary = list(self.resume.summary)

    def record(self) -> dict:
        """``run.json``: the whole control plane, what the run cost, and one row per delivered mask."""
        return {
            # THE WHOLE CONTROL PLANE: two runs differing only in `slot_regions` have to be tellable
            # apart afterwards, and a directory name is not that. Policy only by construction, so
            # this cannot leak a key.
            "config": _config_snapshot(self.cfg),
            # ...and the COMMAND, because several flags reach a run without landing in a
            # `Config` field — among them `--max-rounds`, `--defect-library`, `--request-timeout`,
            # `--cache-dir`, `--trace-dir`. And
            # `config.shape_mode` is worse than missing: it is the dataclass DEFAULT on every run,
            # since the routed value is applied per type via a local `replace(cfg, ...)` and recorded
            # in each `results` row (`config.shape_policy` is the run-level policy it was resolved
            # under).
            "argv": list(self.argv),
            "engine_constants": _engine_constants(),
            # ...and these four ALSO stay at the top level: every downstream reader (the KPI scorer,
            # the HTML builders, any log bucketer) indexes them there.
            "dataset_path": self.cfg.dataset_path,
            "gen_model": self.cfg.gen_model,
            "judge_model": self.cfg.judge_model,
            # The resolved per-type split, so a resume can be held to the SAME one — see
            # `_Resume.prior_targets`. `None` while a run is one-pass, which has no split.
            "targets": getattr(self, "targets", None),
            "n_masks": len(self.testcase_rows),
            # DELIVERED vs PRODUCED. The fill loop overshoots and trims, so a run that hands over 40
            # masks may have produced 84, and dividing a run's cost by one of those answers a different
            # question from dividing it by the other. Both are recorded so a cost-per-mask number says
            # which divisor it used: `produced` is what the beams accepted, `trimmed` is what was moved
            # aside for being past the target (kept on disk under _overflow, not deleted).
            "produced": len(self.testcase_rows) + len(self.overflow_rows),
            "trimmed": len(self.overflow_rows),
            # Per-endpoint calls / refusals / waiting. Written whether or not anything went wrong: an
            # endpoint having a bad afternoon is otherwise invisible, and it shows up in the API-time
            # numbers looking like a slow model.
            "endpoints": self.endpoint_report(),
            # WHAT THE RUN COST, in tokens: API seconds INCLUDE permit wait, so they are a function
            # of concurrency rather than of the work. Summed from the beams — `get_usage()` is
            # thread-local and the beams ran on workers.
            "usage": _run_usage(),
            # WHAT THIS PROCESS INHERITED, kept BESIDE `usage` rather than added into it: a
            # cost-per-mask dividing one process's tokens by two processes' masks is worse than
            # either number. test_a_resumed_run_makes_up_the_shortfall_without_touching_the_first_runs_masks.
            **({"resumed": self.resume.prior} if self.resume else {}),
            "results": self.summary,
        }

    def flush(self) -> None:
        """Persist every deliverable from whatever has completed so far. Called at the start of the run
        (the control plane on disk before the first model call), after EVERY accepted mask, inside
        `TypeOutput.trim`, after EVERY type and at the end whether the run finishes or the endpoint
        stops answering — so a later type crashing never loses the earlier types' results.

        Every write is atomic (:func:`atomic_write`): rewriting in place empties the manifest first
        and fills it second, so the crash this exists to survive is the one most likely to happen
        WHILE it runs. a failed write leaves the previous file intact. run.json goes first: a kill
        between the two writes then leaves a `results` row the manifest lacks, which a resume drops
        (`_plan_resume` keeps results only for manifest rows), never a manifest row the results lack."""
        atomic_write(self.out / "run.json", lambda t: t.write_text(json.dumps(self.record(), indent=2)))
        atomic_write(self.testcase, _jsonl(self.testcase_rows))
        # WRITTEN OR REMOVED, never left standing: the file's existence is the signal that this
        # out-dir HAS overflow, and a stale one names files that are not on disk.
        # the overflow manifest is removed when there is no overflow left.
        ov = self.out / "testcase_overflow.jsonl"
        if self.overflow_rows:
            atomic_write(ov, _jsonl(self.overflow_rows))
        else:
            ov.unlink(missing_ok=True)
        # flush owns the out-dir's manifests: a file it does not write, it removes.
        (self.out / "amp_todo.json").unlink(missing_ok=True)


@dataclass
class TypeOutput:
    """Where ONE defect type's accepted masks land, and what becomes of the ones past the target.

    CONSTRUCTING IT CREATES ITS TWO DIRECTORIES — ``<out>/<texture>/mask/<defect>`` and
    ``.../overlay/<defect>`` — so a type that goes on to accept nothing still leaves the shape of
    where its masks would have gone.

    ``emitted`` — mask name -> the four things written for it — is what :meth:`emit` fills and
    :meth:`trim` reads, and nothing else touches it.
    """

    deliv: Deliverables
    dataset_path: str
    texture: str
    defect: str
    shape_mode: str

    def __post_init__(self) -> None:
        self.dtype = f"{self.texture}+{self.defect}"
        self.mask_dir = self.deliv.out / self.texture / "mask" / self.defect
        self.overlay_dir = self.deliv.out / self.texture / "overlay" / self.defect
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.iter_max = _iter_max(self.dataset_path, self.texture, self.defect)
        self.emitted: dict = {}  # mask name -> (mask_path, overlay_path, row, summ)

    def emit(self, cp, reuse, j, m) -> None:
        """Write ONE accepted mask to disk immediately, as the fill loop books it."""
        stem = f"{cp.stem}_{reuse}_{j}"  # j = within-beam index; name matches the run log's `masks`
        mpath = self.mask_dir / f"{stem}_mask.png"
        _binarize(m.mask).save(mpath)
        ovp = self.overlay_dir / f"{stem}_overlay.png"
        if m.overlay is not None:
            m.overlay.convert("RGB").save(ovp)
        row = {
            "image_filename": str(cp.resolve()),
            "mask_filename": str(mpath.resolve()),
            "anomaly_type": self.dtype,
            "num_generated_images": 1,
            "poisson_blend": False,
            "iteration_generation_max_instance": self.iter_max,
        }
        # The mask's content hash is the join to the tracer's copy under <trace>/_masks/<sha>.png:
        # `j` is assigned here, at emit time, and appears nowhere in the trace. Empty when the file
        # cannot be read, because a record of a mask must not be able to fail the run.
        try:
            sha = hashlib.md5(mpath.read_bytes(), usedforsecurity=False).hexdigest()[:16]
        except OSError:
            sha = ""
        summ = {
            "clean": cp.name,
            "type": self.dtype,
            "reuse": reuse,
            "mask": mpath.name,
            "sha": sha,
            "judge": m.verdict.score,
            "shape_mode": self.shape_mode,
            "dims": {k: int(v) for k, v in (m.verdict.scores or {}).items()},  # per-dim final-judge scores
            # ...and WHY each of those numbers, keyed by the mask FILE. `getattr`, because a
            # record of a mask must not be able to fail the run that produced it.
            "reasons": dict(getattr(m.verdict, "reasons", None) or {}),
            "feedback": str(getattr(m.verdict, "feedback", "") or ""),
        }
        self.deliv.testcase_rows.append(row)
        self.deliv.summary.append(summ)
        self.emitted[mpath.name] = (mpath, ovp, row, summ)
        self.deliv.flush()  # incremental: testcase.jsonl + run.json reflect every accepted mask, live

    def trim(self, accepted: list, target_n: int, runlog_dir: Path) -> list:
        """Cut this type's over-generated batch back to ``target_n`` and return what is kept.

        Over-generated masks are MOVED aside, not deleted: they passed the same bar and only landed
        past the target, and nothing else can recover them.
        test_the_trim_leaves_the_deliverable_naming_only_files_that_are_there.

        Everything the move implies happens here too: the surplus rows leave ``testcase_rows`` and
        ``summary`` for ``overflow_rows``, the deliverables are flushed, and each per-beam run log
        under ``runlog_dir`` is rewritten so it no longer names a file that has been moved.
        """
        kept = _trim_to_target(accepted, target_n)
        keep = {f"{cp.stem}_{reuse}_{j}_mask.png" for cp, reuse, j, m in kept}
        over_m = self.deliv.out / self.texture / "_overflow" / self.defect
        over_o = self.deliv.out / self.texture / "_overflow_overlay" / self.defect
        for name, (mpath, ovp, row, summ) in list(self.emitted.items()):
            if name not in keep:
                over_m.mkdir(parents=True, exist_ok=True)
                # Drop the deliverable rows FIRST, while `row` is still the dict that is in the
                # list: rebind it to the moved copy before the membership test and the test
                # compares a dict that differs by mask_filename, so testcase.jsonl keeps a row
                # pointing at a file that has just been moved away.
                if row in self.deliv.testcase_rows:
                    self.deliv.testcase_rows.remove(row)
                if summ in self.deliv.summary:
                    self.deliv.summary.remove(summ)
                # TWO MOVES, TWO OUTCOMES: the mask and its overlay can fail separately, and one
                # `try` around both records a path for a file that did not move.
                dest = over_m / mpath.name
                try:
                    mpath.replace(dest)
                except OSError as e:
                    log.warning(
                        f"degraded[overflow_move_failed]: {mpath.name}: "
                        f"{type(e).__name__}: {e} — past the target and not movable "
                        f"under _overflow, so it stays where it is"
                    )
                    dest = mpath
                if ovp.exists():
                    try:
                        over_o.mkdir(parents=True, exist_ok=True)
                        ovp.replace(over_o / ovp.name)
                    except OSError as e:
                        log.warning(
                            f"degraded[overflow_move_failed]: {ovp.name} (overlay): "
                            f"{type(e).__name__}: {e} — left beside its mask"
                        )
                self.deliv.overflow_rows.append({**row, "mask_filename": str(dest.resolve())})
        n_over = len(self.emitted) - len(keep)
        if n_over > 0:
            # A SEPARATE manifest, written by `Deliverables.flush` from the run-level list:
            # testcase.jsonl stays exactly the N that were asked for, so nothing downstream picks
            # these up by accident. Promote by appending from the other file.
            log.info(
                f"{self.dtype}: {n_over} accepted masks past the target kept in {over_m} (+ testcase_overflow.jsonl)"
            )
        self.deliv.flush()
        # The run logs still name every mask their beam produced; record which the trim removed,
        # so they stay truthful without pointing at files that are gone.
        for lg in sorted(runlog_dir.glob("*_run.log")):
            try:
                rec = json.loads(lg.read_text())
            except Exception:  # noqa: BLE001 — a damaged log must not fail the run
                continue
            names = rec.get("masks") or []
            dropped = [n for n in names if n not in keep]
            if not dropped:
                continue
            rec["masks"] = [n for n in names if n in keep]
            rec["trimmed_masks"] = dropped
            lg.write_text(json.dumps(rec, indent=2))
        return kept


def _cache_root(cache_dir_arg: str | None, dataset_path: str) -> Path:
    """Where the reusable, dataset-derived precomputes live (cards, embeddings, vp maps).

    Keyed by the dataset's directory name PLUS a hash of its full resolved path — the name alone
    collides (`/data/2025q1/pcb` and `/data/2025q3/pcb`), and `build_cards_dir` skips existing
    cards, so the collision's shape is the second dataset silently judged against the first one's
    library. Two paths never share a cache; the price is that ONE dataset seen from two mount
    points distills twice (a re-spend, not a wrong answer). Rooted under results/, never inside
    the dataset: datasets/ is mounted read-only in the product containers."""
    if cache_dir_arg:
        return Path(cache_dir_arg)
    rp = Path(dataset_path).resolve()
    digest = hashlib.sha1(str(rp).encode(), usedforsecurity=False).hexdigest()[:8]
    return Path("results") / "_amg_cache" / f"{rp.name or 'dataset'}-{digest}"


# --------------------------------------------------------------------------------------------------
# Defect cards
# --------------------------------------------------------------------------------------------------
def _load_defects(args, cfg, vlm, cache_dir):
    if args.defect_library == "none":
        return from_dir(None)
    # "auto": distill a per-type reference doc from the dataset's own real examples (cached, reused across runs)
    lib = str(Path(cache_dir) / "defect_library")
    build_cards_dir(vlm, cfg.dataset_path, cfg.anomaly_types, lib, model=cfg.gen_model)
    # The SAME `card_variant` the distiller named the file with: a reader that disagrees with the
    # writer finds no file and hands the run an empty card library.
    # each model gets its own card and the writer and reader agree on the name.
    return from_dir(lib, variant=card_variant(cfg.gen_model))


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic defect-mask generation (CLI-driven; no recipe files).")
    # required
    p.add_argument(
        "--dataset-path", required=True, help="AnomalyGen dataset root (<texture>/clean_image|anomaly_image|mask)"
    )
    p.add_argument(
        "--anomaly-types",
        required=True,
        nargs="+",
        metavar="TEXTURE+DEFECT",
        help="one or more 'texture+defect' pairs, e.g. SOP+INSUFFICIENT_SOLDER",
    )
    # entry I/O (not engine policy)
    p.add_argument("--out-dir", default="results/amg", help="where masks + testcase.jsonl land")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="dir for REUSABLE artifacts shared across runs — visual-prompt maps, DINOv2 "
        "embeddings, and the distilled defect library (subdirs visual_prompts/ dinov2_emb/ "
        "defect_library/). Default results/_amg_cache/<name>-<path hash>; precompute happens once, not per --out-dir.",
    )
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        help="skip the capability probe — one call per distinct model (two tiny images, a "
        "JSON answer) that runs before the first card is distilled. It is what turns a model "
        "that cannot take images or will not answer in JSON into a named refusal costing "
        "nothing; skipping it moves that same failure into the middle of the batch",
    )
    # placement scope
    p.add_argument(
        "--num-sdg",
        type=int,
        default=None,
        help="target judge-ACCEPTED masks for the WHOLE run, split across the types the way AMP splits "
        "its --num_sdg (how the split is made depends on --mode). Delivers exactly this many rows. "
        "Omit both this and --per-defect-counts for one pass over every clean image",
    )
    p.add_argument(
        "--mode",
        choices=("inference", "validation"),
        default="inference",
        help="how --num-sdg is split, matching AMP's --mode. 'inference' (the generation set): "
        "uniform, and a type may get 0 when the total is below the type count. 'validation' (the set "
        "Step 3 validates against): proportional to each type's real mask count, and a zero share is "
        "REFUSED — validation scores per type, so a type with no samples drops silently out of the KPI "
        "that picks the best checkpoint",
    )
    p.add_argument(
        "--per-defect-counts",
        type=str,
        default=None,
        help="exact masks per type as JSON, e.g. '{\"SOP+BRIDGE\": 12}'; types omitted get 0. "
        "--mode inference only, like AMP's --per_defect_counts. Its sum wins over --num-sdg",
    )
    p.add_argument("--max-rounds", type=int, default=None, help="cap fill rounds per type (safety; default unlimited)")
    p.add_argument(
        "--shape",
        dest="shape_policy",
        choices=["flexible", "auto", "fixed"],
        default=None,
        help="shape policy for every type (default auto). 'auto' asks the per-type shape router, "
        "except that a card judging fill_host=whole forces flexible and withholds the stamp op "
        "(the mask IS whole host instance(s), so a pasted example shape can only collide with the "
        "board's own structures); 'flexible' makes every type flexible (the same card still "
        "withholds stamp); 'fixed' makes every type fixed (stamp + rigid refine) and outranks "
        "the card ban, so stamp stays on offer",
    )
    # Config overrides (omit any to use the quality-first Config default)
    p.add_argument(
        "--visual-prompt",
        default=None,
        choices=[*_VP_CHOICES, "auto"],
        help="'auto' = take it from the type's distilled card (its `visual_prompt` flag), "
        "and use none where the card does not say",
    )
    p.add_argument(
        "--defect-library",
        default="auto",
        choices=["none", "auto"],
        help="'auto' distills a per-type reference doc from the dataset's own real examples "
        "(cached under <cache-dir>/defect_library) to ground the proposer/judge; 'none' skips it",
    )
    # REQUIRED, deliberately: which model draws and which model judges is the run's most
    # consequential choice, so it is always stated, never inherited from a default nobody sees.
    p.add_argument("--gen-model", required=True, help="model that proposes/refines (registry name)")
    p.add_argument("--judge-model", required=True, help="model that scores; may differ from --gen-model")
    p.add_argument("--num-candidates", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=None)
    p.add_argument("--max-refine-rounds", type=int, default=None)
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--final-judge-samples", type=int, default=None)
    p.add_argument("--accept-threshold", type=int, default=None)
    p.add_argument(
        "--image-max-dim",
        type=int,
        default=None,
        help="cap every image's longest side (px) before sending; token-cost lever (default 1024)",
    )
    p.add_argument("--max-concurrent-calls", type=int, default=None)
    p.add_argument(
        "--model-registry",
        default=None,
        help="YAML mapping model names to their endpoints (model + base_url + key) — the ONLY "
        "place endpoints are configured; a model with no entry refuses the run. "
        f"Default: {DEFAULT_REGISTRY}. Start from {REGISTRY_TEMPLATE}.",
    )
    p.add_argument(
        "--exemplar-captions",
        dest="exemplar_captions",
        action="store_true",
        default=None,
        help="attach each exemplar's own distill-time caption (host / extent / shape) next to its "
        "images in propose/refine/judge — the per-image evidence the card summarizes, carried with "
        "the images it describes. Cache-only, no new calls.",
    )
    p.add_argument(
        "--exemplar-visual-prompt",
        dest="exemplar_vp",
        action="store_true",
        default=None,
        help="visual parity for the in-context examples: also show each exemplar as the injected "
        "vp's painted map (sam2 / solderlight) with its REAL mask outlined. One extra image per "
        "exemplar; map painting is content-hash cached.",
    )
    p.add_argument(
        "--coverage-note",
        dest="coverage_note",
        action="store_true",
        default=None,
        help="inject the measured STRUCTURE-COVERAGE note (rank-paired coverage bands, grouped "
        "footprint, off-structure mass — candidate and real annotations measured with ONE yardstick) "
        "at judge, refine and the pairwise selector.",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        # `%%`, not `%`: argparse runs every help string through `%` formatting to expand
        # `%(default)s`, so a literal percent sign makes `--help` itself raise ValueError.
        help="per-call HTTP timeout (s). DEFAULT 180. A cap sized to the distribution of "
        "SUCCESSFUL calls (median 41 s, p90 64 s) is survivorship-biased — that sample cannot "
        "contain the calls that would have timed out. At 90 s transport failures run "
        "8.4-12.8%% against 1.1%% at 180 s, which is 3.5x more waiting, not less",
    )

    p.add_argument(
        "--allowed-cv-ops",
        default=None,
        help="comma-separated ops the agent may use (default: all). Restricts BOTH turn 1 "
        "and refine, so the listed ops are the only way to reach the mask — which is "
        "how a small vocabulary is tested for whether it is expressive enough. Include "
        "at least one op that draws from nothing (disk, brush, add_polygon, stamp, ...) "
        "or turn 1 has no vocabulary; the run says so if you do not. "
        "e.g. 'disk,brush,erase,conform,snap'",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="START OVER: delete an existing --out-dir. Without it, pointing at a "
        "populated out-dir CONTINUES that run and produces only the shortfall (the "
        "same settings are required; different ones are refused, by name)",
    )
    p.add_argument(
        "--beam-multi",
        type=int,
        default=None,
        metavar="K",
        help="IN-LOOP BEAM: ask for K distinct-strategy fix options per candidate in ONE "
        "refine call, render them all and keep the best (Config beam_multi; 0/unset = "
        "off). Was AMG_BEAM_MULTI.",
    )
    # `store_true` + `default=None`, like the `--no-*` switches below and for the same reason: the
    # overrides table reads None as "not typed", and a plain `store_true` would hand Config a False
    # nobody asked for on every run that never mentioned the flag.
    p.add_argument(
        "--beam-pairwise",
        action="store_true",
        default=None,
        help="the beam's selector compares options A-vs-B (king-of-the-hill, K-1 SERIAL "
        "calls) instead of scoring each alone in parallel. A weak judge compares more "
        "reliably than it scores, and a pairwise winner cannot tie.",
    )
    p.add_argument(
        "--beam-prune",
        action="store_true",
        default=None,
        help="THROUGHPUT: refine only the top-beam_width survivors each round (the set the "
        "final stage already selects) — fewer LLM calls/mask, GT mean held; deterministic",
    )
    p.add_argument(
        "--accept-continue",
        action="store_true",
        default=None,
        help="THROUGHPUT (opt-in): bank a candidate the in-loop judge already passes and "
        "skip its final judge, dropping it from the beam — fewer calls, but it accepts "
        "on ONE verdict instead of the final consensus, a small mean risk on hard defects",
    )
    p.add_argument(
        "--judge-cache-order",
        action="store_true",
        default=None,
        help="THROUGHPUT: reorder the judge prompt static-blocks-first so a self-hosted vLLM "
        "reuses the prefix KV across a board's judge calls (prefill lever; opt-in, and it "
        "changes the prompt bytes, so the model's output can shift)",
    )
    p.add_argument(
        "--slot-regions",
        type=int,
        default=None,
        help="valid regions assigned to ONE beam, one per candidate slot (default 2). Raising "
        "it decouples the sites a run covers from the number of beams it launches, at "
        "the cost of each candidate getting a smaller share of the proposer's attention; "
        "past 2 that cost shows up as mask outside the ROI",
    )
    p.add_argument(
        "--coverage",
        default=None,
        choices=["none", "localize", "propose", "both", "roundrobin", "choose"],
        help="feed already-placed sites so it targets an UNCOVERED valid site (localize/propose/both), "
        "'roundrobin' = enumerate the valid regions once per clean and assign each run a distinct "
        "region (deterministic, but distinct sites can never exceed the number of runs), or "
        "'choose' = show the agent the enumerated sites with the taken ones marked and let it "
        "pick. DEFAULT: roundrobin (Config.coverage_feedback), not none",
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        help="keep every API call's full IO under this directory — the audit trail that answers "
        "what the model saw. Off unless given: it stores every prompt image, about 1 MB per call, "
        "tens of GB for a large batch. One directory per beam (dtype-prefixed on a multi-type run) "
        "plus a global index.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = _args()
    cfg = _build_config(args)
    registry = _load_registry(args.model_registry)
    # Resolve EVERY model this run will call — gen, judge, and the beam selector if one is named —
    # above every side effect, so an unroutable name refuses the run before the out-dir is touched.
    endpoints = _endpoints([cfg.gen_model, cfg.judge_model, cfg.beam_select], registry)
    # PREFLIGHT — every exemplar pair and clean image this run may touch, read once, before the run has
    # any side effect at all. Global rather than per type, so a broken file in type 7 of 12 is found
    # before six types have been paid for. Pure I/O over a bounded set (a few hundred pairs, seconds at
    # 16 threads), noise against a run measured in minutes.
    #
    # It runs HERE, above everything, and not next to the first model call, because two side effects sit
    # between: `--overwrite` rmtree's the previous run's deliverables, and `init_loguru_file` creates
    # run.log — which makes the out-dir non-empty, so a refusal would make the retry-after-fixing demand
    # `--overwrite`, i.e. demand the delete. A gate that destroys the last run's results and then
    # refuses to start is worse than no gate. The report is held and logged once run.log exists.
    try:
        _preflight = preflight(
            cfg.dataset_path, cfg.anomaly_types, {t: _list_cleans(cfg.dataset_path, t) for t, _d in cfg.anomaly_types}
        )
    except DatasetError as e:
        raise SystemExit(str(e)) from None
    out = Path(args.out_dir)
    # `--overwrite` and `--skip-smoke` act on this process, not on the deliverable — the first is
    # consumed the moment the out-dir is (re)opened, the second decides whether the probe runs — so
    # neither belongs in the control plane, and both are stripped from the argv the run records and
    # compares. A run born with either is continued by re-running WITHOUT it (keeping `--overwrite`
    # would wipe the delivered masks again, and dropping a recorded flag would be refused as a
    # control-plane change), and the probe's own remedy — the same command plus `--skip-smoke` — is
    # accepted into the refused run's out-dir.
    argv = [a for a in sys.argv if not _is_process_flag(a)]
    resume = _open_out_dir(out, cfg, argv, overwrite=args.overwrite)
    # Every warning and error into a file next to the deliverables: the framework's logger otherwise
    # has only a stdout sink, so a run's warnings survive only if the caller kept the redirect.
    init_loguru_file(str(out / "run.log"))
    log.info(str(_preflight))  # checked before any side effect; recorded now that run.log exists
    if resume is not None:
        # ON THE RECORD: nothing else says a deliverable was produced by two processes.
        # the manifest and the disk disagreeing is decided and recorded.
        for kind, detail in resume.deviations(out):
            log.warning(f"degraded[{kind}]: {detail}")
    _log_routing({"gen": cfg.gen_model, "judge": cfg.judge_model, "beam-select": cfg.beam_select}, registry)
    trace_dir = args.trace_dir
    # The transport's default (key, base) pair is the GEN model's endpoint — every model this run
    # names is in `endpoints`, so the default is only a formality of the constructor.
    gen_base, gen_key = endpoints[cfg.gen_model]
    vlm = Vlm(
        gen_key,
        gen_base,
        concurrency=cfg.max_concurrent_calls,
        img_maxdim=cfg.image_max_dim,
        trace_dir=trace_dir,
        # Numbering picks up past whatever is already in that directory. A resumed run shares
        # the trace with the run it continues, and both number their calls per process, so
        # starting at 0 would file this run's first calls into the last run's directories.
        trace_seq_start=_next_trace_seq(Path(trace_dir)) if trace_dir else 0,
        endpoints=endpoints,
        # 180 s per call — see `--request-timeout`'s help for the measurement.
        timeout=(args.request_timeout if args.request_timeout else 180.0),
    )
    # All reusable, dataset-derived artifacts live under ONE cache (shared across runs); --out-dir holds
    # only this run's outputs. Keyed by the dataset's NAME so cards can't collide across datasets, but
    # rooted under results/ — every other stage's write target — never inside the dataset: the product
    # containers mount datasets/ read-only, and a run that had already paid the smoke probe died on it.
    cache_dir = _cache_root(args.cache_dir, cfg.dataset_path)
    vp_dir = str(cache_dir / "visual_prompts")
    emb_cache_dir = str(cache_dir / "dinov2_emb")
    # The delivery cap travels WITH the visual prompt, because the prompt is what draws grids on its
    # own artifacts and those grids are read by the judge. Built here because this is where cfg is.
    views = Views(cfg.image_max_dim or 0)
    routing_vp = cfg.visual_prompt == "auto"
    prompts = VisualPrompts(
        routing=routing_vp,
        fixed=None if routing_vp else by_name(cfg.visual_prompt, vp_dir=vp_dir, views=views),
        vp_dir=vp_dir,
        views=views,
        library=args.defect_library,
    )
    deliv = Deliverables(out, cfg, argv, vlm.endpoint_report, resume)
    multi = len(cfg.anomaly_types) > 1
    # Each type's ORIGINAL target, before any resume shortfall. `--num-sdg` splits the run's total the
    # way AMP splits its own, so the row count lands on the number that was asked for instead of
    # `ceil(total / types) * types`. With neither sizing flag the run is one pass per clean image,
    # which `None` carries through untouched.
    #
    # Read off `args`, deliberately NOT a Config field. A resumed run must still pass the same total —
    # a different split over the same out-dir would mix two deliverables — and `_control_plane_diff`
    # already enforces that through argv, which it compares as (flag, values) pairs and refuses on
    # any difference. Adding a Config field would buy nothing there and would change `run.json` for every
    # run, including the single-type one the scheduler's null-case fixture pins byte-for-byte.
    if args.num_sdg is not None or args.per_defect_counts is not None:
        dtypes = [f"{t}+{d}" for t, d in cfg.anomaly_types]
        try:
            override = json.loads(args.per_defect_counts) if args.per_defect_counts else None
        except json.JSONDecodeError as e:
            raise SystemExit(f"--per-defect-counts is not valid JSON: {e}") from None
        try:
            # The weights are the per-type mask counts preflight already counted on its way past every
            # exemplar pair — `inference` ignores them, `validation` is proportional to them.
            targets = allocation.allocate(
                args.num_sdg if args.num_sdg is not None else 0,
                dtypes,
                _preflight.per_type,
                mode=args.mode,
                override=override,
                warn=log.warning,
                total_label="--num-sdg",
                mode_label="--mode",
                override_label="--per-defect-counts",
            )
        except ValueError as e:
            raise SystemExit(str(e)) from None
        log.info(
            f"--mode {args.mode}, --num-sdg {args.num_sdg} across {len(dtypes)} type(s): "
            + ", ".join(f"{k} {v}" for k, v in targets.items())
        )
    else:
        # Neither sizing flag: one pass per clean image, which `None` is how `_run_type` spells.
        targets = {f"{t}+{d}": None for t, d in cfg.anomaly_types}
    # Held to the earlier process's split, not just to the same flags. The comparison lives HERE and
    # not in `_control_plane_diff` because the split is not known when the out-dir is opened: under
    # `--mode validation` it depends on mask counts read from the dataset a moment later.
    if resume is not None and resume.prior_targets and resume.prior_targets != targets:
        moved = [
            f"{k}: was {resume.prior_targets.get(k)}, now {targets.get(k)}"
            for k in sorted(set(resume.prior_targets) | set(targets))
            if resume.prior_targets.get(k) != targets.get(k)
        ]
        raise SystemExit(
            f"{out} holds a run whose per-type split DIFFERS from this one, so continuing it would "
            f"put two allocations in one deliverable:\n  "
            + "\n  ".join(moved)
            + "\nThe flags match, so the dataset's mask counts changed under a proportional "
            "(--mode validation) split. Run this in a new --out-dir, or pass --overwrite."
        )
    deliv.targets = targets
    if multi:
        # Types interleave in run.log from here on, so position stops attributing the untagged
        # degraded[...] lines; the thread-local tag takes over. Single-type runs stay untouched.
        _install_ledger_tagger()

    def _process_type(texture: str, defect: str, sched: "_Scheduler | None" = None, prep_done=None) -> None:
        dtype = f"{texture}+{defect}"
        if multi:
            _set_ledger_type(dtype)  # prep runs on this thread; its degraded lines belong to the type
        cleans = _list_cleans(cfg.dataset_path, texture)
        if not cleans:
            log.warning(
                f"degraded[type_no_cleans]: {dtype} has no clean images under "
                f"{texture}/clean_image — the type is skipped and delivers nothing"
            )
            log.warning(f"{dtype}: no clean images under {texture}/clean_image — skipping")
            return
        # The FIRST clean is the router's similarity reference — the per-clean 'similar' ranking the
        # fill loop runs later reuses these embeddings from the same cache, so the route pays nothing
        # the run would not pay anyway.
        shape_mode = _resolve_shape_mode(
            vlm, cfg, texture, defect, defects, clean_path=str(cleans[0]), emb_cache_dir=emb_cache_dir
        )
        # THE SHORTFALL, not the target; `reuse_start` is the other half, saying what the new masks
        # may be CALLED. a resumed run trims to the shortfall and not to the original target.
        want = targets[dtype]
        # A type allocated NOTHING never reaches the pool: its quota would want no beams and hold
        # none, so nothing would ever complete into the scheduler to settle it and the driver would
        # wait on `settled` forever while the pump waited on the driver. Spreading fewer samples than
        # types is a legitimate request (`--num-sdg 2` over three types is 1/1/0), so this is an
        # answer, not an error — and it is the same early return the resume path already takes when a
        # type turns out to owe nothing. `<=`, not `==`: allocation refuses negative counts at the
        # CLI, so a below-zero want here is a bug upstream — and the one wrong answer to a bug is
        # the deadlock above, so it takes the same exit. The pump's admission `_check_done` is the
        # second layer of the same guarantee. None is not a count at all — the no-target one-pass
        # run — and passes through to the quota, whose target_n=None means one beam per clean.
        if want is not None and want <= 0:
            log.info(f"{dtype}: allocated 0 masks — nothing to produce for this type")
            return
        target_n, reuse_start = want, {}
        if resume:
            reuse_start = resume.next_reuse.get(dtype, {})
            target_n = resume.target_for(dtype, want)
            if target_n == 0:
                log.info(
                    f"{dtype}: {resume.delivered.get(dtype, 0)}/{want} already delivered — nothing left to produce"
                )
                return
            if target_n is None:  # one pass per clean: the cleans that already had theirs are done
                cleans = [c for c in cleans if c.stem not in reuse_start]
                if not cleans:
                    log.info(f"{dtype}: every clean has had its pass — nothing left to produce")
                    return
        log.info(f"{dtype}: shape={shape_mode}, {len(cleans)} cleans, target={target_n if target_n else 'one-pass'}")
        runlog_dir = out / "_runlog" / dtype
        tout = TypeOutput(deliv, cfg.dataset_path, texture, defect, shape_mode)
        accepted, attempts = _run_type(
            vlm,
            cfg,
            texture,
            defect,
            cleans,
            prompts.for_clean,
            defects,
            shape_mode=shape_mode,
            target_n=target_n,
            max_rounds=args.max_rounds,
            pool_size=cfg.max_concurrent_calls,
            runlog_dir=runlog_dir,
            emb_cache_dir=emb_cache_dir,
            region_cache_dir=str(cache_dir / "regions"),
            on_accept=tout.emit,
            reuse_start=reuse_start,
            scheduler=sched,
            prep_done=prep_done,
        )
        generated = len(accepted)
        # Trimmed against the SHORTFALL: `accepted` holds only what THIS process produced. On the
        # WRITER thread always — under a shared scheduler this code runs on the type's own driver
        # thread while another type's beams are being booked, and the trim rewrites the same
        # deliverable lists the booking appends to.
        # a resumed run trims to the shortfall and not to the original target.
        if target_n and generated > target_n:  # beams return batches -> trim to exactly N

            def _trim():
                _set_ledger_type(dtype if multi else "")
                try:
                    return tout.trim(accepted, target_n, runlog_dir)
                finally:
                    _set_ledger_type("")

            accepted = sched.on_writer(_trim) if sched is not None else _trim()
        trimmed = f" (trimmed from {generated})" if generated != len(accepted) else ""
        log.info(
            f"{dtype}: {len(accepted)} masks in {attempts} runs{trimmed} (run log: {runlog_dir}) "
            f"· {vlm.endpoint_summary()}"
        )

    failed = []

    def _one_type(texture: str, defect: str, sched: "_Scheduler | None" = None, prep_done=None) -> None:
        try:
            _process_type(texture, defect, sched, prep_done)
        except Exception as e:  # one type's failure must NOT sink the others — log, persist, continue
            _set_ledger_type("")  # the record below names its type itself
            failed.append(f"{texture}+{defect} ({type(e).__name__}: {e})")
            # ON THE LEDGER, like `type_no_cleans` beside it: the traceback below is a stack, not
            # a record, and a reader counts `degraded[...]` kinds.
            log.warning(
                f"degraded[type_failed]: {texture}+{defect}: {type(e).__name__}: {e} — "
                f"the type is abandoned and delivers nothing; the run continues with the "
                f"others and exits non-zero"
            )
            log.exception(f"{texture}+{defect}: FAILED; skipping this type, continuing")
        finally:
            _set_ledger_type("")

    # The control plane on disk BEFORE the first model call, the capability probe included: run.json is
    # what a later process compares against, so a run that stops at the probe or during its first type
    # would otherwise leave a non-empty directory with no record, which only `--overwrite` can enter.
    deliv.flush()

    # THE CAPABILITY GATE: on the transport that will do the work, and above every model call the run
    # makes — the card distillation directly below, the region enumeration and every beam. It sits below
    # `_open_out_dir` because the transport it probes is built from the trace sequence that directory
    # holds; below `init_loguru_file` so the refusal lands in run.log with the rest of the ledger; and
    # below the flush above so a refused run's out-dir is a run to continue — the refusal recommends
    # `--skip-smoke`, and that retry resumes here instead of being refused as a non-empty directory
    # without a record. Nothing between the transport's construction and this point calls the endpoint:
    # the split is arithmetic over preflight's counts, and the flush is a write.
    if not args.skip_smoke:
        # Each model's RESOLVED base rides along, so the probe's refusal can name a base that is
        # itself the problem (api.anthropic.com short of its /v1 segment).
        _bases = {m: endpoints[m][0] for m in (cfg.gen_model, cfg.judge_model) if m}
        _smoke_models(vlm, [cfg.gen_model, cfg.judge_model], _bases)

    try:
        # THE RUN'S FIRST MODEL CALLS: `--defect-library auto` distills a card per type from the
        # endpoint, so they sit inside the guard for the same reason every beam does — the record is
        # written and the directory stays resumable.
        # a dead endpoint in the defect library leaves a directory that can be resumed.
        defects = _load_defects(args, cfg, vlm, cache_dir)
        prompts.defects = defects  # only `--visual-prompt auto` reads them, and only from here on
        if not multi:
            for texture, defect in cfg.anomaly_types:
                _one_type(texture, defect)
                deliv.flush()  # crash-safe: deliverables reflect the completed type
        else:
            # EVERY TYPE AGAINST ONE POOL. A driver thread per type runs the (serial) prep and then
            # waits; the MAIN thread pumps the scheduler and is the only thread that books beams or
            # writes deliverables. The gates are the prep turnstile: type i+1's prep starts when
            # type i's quota is admitted — or when its driver exits without one — so prep stays
            # one-type-at-a-time (the vp-cache / DINOv2 invariant) while every admitted type's
            # beams share the pool.
            with ThreadPoolExecutor(max_workers=cfg.max_concurrent_calls) as ex:
                sched = _Scheduler(ex, cfg.max_concurrent_calls, n_types=len(cfg.anomaly_types), deliv=deliv)
                gates = [threading.Event() for _ in cfg.anomaly_types]
                gates[0].set()

                def _driver(i: int, texture: str, defect: str) -> None:
                    release = gates[i + 1].set if i + 1 < len(gates) else (lambda: None)
                    try:
                        gates[i].wait()
                        if sched.dead is None:  # the run is already stopping; start no more prep
                            with contextlib.suppress(Exception):  # the run is unwinding; main holds the cause
                                _one_type(texture, defect, sched, prep_done=release)
                    except TransportUnavailable as e:
                        # BaseException by design, so it sailed past `_one_type`'s per-type handler;
                        # handed to the pump, whose own raise is what reaches `main`'s handler.
                        sched.fatal(e)
                    finally:
                        # Prep ran on THIS thread, so its calls live in THIS thread's counters —
                        # book them into `_RUN_USAGE`, or the type's routing and region calls
                        # vanish from run.json. The thread does nothing else, so its totals ARE the
                        # prep delta.
                        spent, calls = get_usage(), get_call_count()
                        ms, fails = get_api_ms(), get_failed_count()
                        if calls or ms or fails or any(spent.values()):
                            _RUN_USAGE.append(
                                {
                                    "usage": spent,
                                    "api_calls": calls,
                                    "failed": fails,
                                    "api_ms": ms,
                                    "prep": f"{texture}+{defect}",
                                }
                            )
                        release()  # idempotent: normally freed at admission, here on every other path
                        sched.type_finished()

                threads = [
                    threading.Thread(target=_driver, args=(i, t, d), name=f"amg-prep-{t}+{d}")
                    for i, (t, d) in enumerate(cfg.anomaly_types)
                ]
                for th in threads:
                    th.start()
                try:
                    sched.pump()
                finally:
                    # On every path: wake whatever still waits on the scheduler, then join, so a
                    # raise out of `pump` cannot leave a driver thread parked forever.
                    sched.close(sys.exception())
                    for th in threads:
                        th.join()
    except TransportUnavailable as e:
        # THE ENDPOINT IS GONE, so there is no next type to try. Keep what is paid for, say why the
        # run is short, and leave with a status. a dead endpoint stops the run and the run says so.
        deliv.flush()
        # Two stops arrive here. An endpoint that stopped answering is worth re-running as is; an
        # account with no quota left (QUOTA_EXHAUSTED) is not, and saying "stopped answering" would
        # send the operator to retry the one thing that cannot help.
        if e.failure.kind == QUOTA_EXHAUSTED:
            what = (
                f"the account behind {e.endpoint or 'the endpoint'} has no quota or credits left "
                f"(HTTP 429 insufficient_quota) — not a rate limit, so retrying cannot help; add credits "
                f"or use another key, then re-run"
            )
        else:
            what = f"the endpoint stopped answering and the run stopped with it: {e}. Re-run"
        log.warning(
            f"degraded[run_stopped:{e.failure.kind}]: {e} — {len(deliv.testcase_rows)} mask(s) "
            f"delivered before the run stopped; they are kept, and re-running "
            f"the same command (without --overwrite) into {out} continues from here"
        )
        raise SystemExit(
            f"{what} the same command with the same --out-dir (without --overwrite) to produce the "
            f"rest: {len(deliv.testcase_rows)} mask(s) are on disk and recorded in {deliv.testcase}."
        ) from None

    log.info(f"wrote {deliv.testcase} ({len(deliv.testcase_rows)} rows across {len(cfg.anomaly_types)} types)")
    if failed:
        # Exiting 0 after writing an empty testcase.jsonl is how a missing checkpoint gets mistaken for
        # "this type has nothing to generate". The deliverable is still written; the status is not a lie.
        raise SystemExit(f"{len(failed)}/{len(cfg.anomaly_types)} types FAILED: " + "; ".join(failed))


if __name__ == "__main__":
    main()
