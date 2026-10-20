# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The beam-search mask-generation engine — propose -> refine -> rank -> final-judge -> accept.

One VLM conversation drives a run: the proposer generates K diverse candidate masks ONCE, then each
round scores them all in ONE call and refines EVERY one — the loop never prunes. ``beam_width`` sizes
the final consensus judge, which alone decides acceptance; self-scores never do.

All knobs come from a :class:`~anomalygen.agentic_mask_generation.config.Config` and nothing is read
from the environment (``test_the_engine_reads_no_environment``). Results are in-memory dataclasses and this
engine writes no files. The public entry point is :func:`run_beam`.
"""

from __future__ import annotations

import collections
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.agentic_mask_generation import vlm as vlm_knobs
from anomalygen.agentic_mask_generation.config import DIMS
from anomalygen.agentic_mask_generation.context import BeamContext, seed_instruction
from anomalygen.agentic_mask_generation.imaging import (
    binarize,
    mask_geometry,
    overlay_mask,
)
from anomalygen.agentic_mask_generation.judging import (
    AcceptBar,
    AdaptiveStop,
    fill_coverage_note,
    gate_only_dims,
    judge_consensus,
)


def _score_num(v) -> float:
    """A model-reported score as a comparable number — 8, "8", "9/10" all parse; the rest score 0.

    Finite only: json.loads accepts NaN/1e999, and round() on either raises."""
    try:
        f = float(str(v).split("/")[0])
    except (ValueError, TypeError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _model_int(v, kind: str) -> int | None:
    """A model-reported candidate id as an exact int, or None after a `degraded[kind]` line saying why.

    Exact, never a guess: `int(True) == 1` would land on candidate 1 and `int(2.7) == 2` on candidate
    2, so a bool or a non-integer number is unplaceable. A numeric id inside a string is accepted,
    since `int("0")` is exact — it is exempt from the `i != v` check only because `"0" != 0` in
    Python. OverflowError: json.loads turns `1e999` into inf, and int(inf) raises it."""
    if isinstance(v, bool):
        log.warning(f"degraded[{kind}]: boolean id {v!r}")
        return None
    try:
        i = int(v)
    except (TypeError, ValueError, OverflowError):
        log.warning(f"degraded[{kind}]: non-numeric id {v!r}")
        return None
    if i != v and not isinstance(v, str):
        log.warning(f"degraded[{kind}]: non-integer id {v!r}")
        return None
    return i


def _self_score(sample, k) -> float:
    """One option's model-reported "self" score, as a comparable number whatever the model sent.

    The ask says integer, and a weak model still answers "8", "9/10" or prose. These feed a `max()`
    across options, where one string among ints is a TypeError — and an exception here is counted
    as a beam failure, five of which abandon the whole type as a configuration problem. Whatever
    arrives: parse what parses, score the rest 0 (that option is simply never preferred)."""
    v = sample.get(k, {}).get("self") if isinstance(sample.get(k), dict) else None
    return _score_num(v)


def _fill_note(ctx, mask) -> list:
    """The measured coverage line as an extra judge/refine block — [] when switched off
    (cfg.coverage_note) or unmeasurable, so its absence changes nothing."""
    if not getattr(ctx, "coverage_note", False):
        return []
    note = fill_coverage_note(ctx.clean, mask, getattr(ctx, "flags", None), getattr(ctx, "exemplars", None))
    return [note] if note else []


from anomalygen.agentic_mask_generation.ops import (
    RIGID_CV_KINDS,
    Applied,
    Refused,
    apply_cv_sequence,
    filter_to_vocabulary,
    record_label,
)
from anomalygen.agentic_mask_generation.prompts import assert_no_leak
from anomalygen.agentic_mask_generation.regions import ex_dist_summary
from anomalygen.agentic_mask_generation.vlm import (
    MAX_TOKENS_ROUTE,
    USAGE_KEYS,
    CallFailed,
    SchemaInvalid,
    get_api_ms,
    get_call_count,
    get_usage,
    stage_trace_mask,
)


# --------------------------------------------------------------------------------------------------
# Returned dataclasses.
# --------------------------------------------------------------------------------------------------
@dataclass
class Verdict:
    """The authoritative judge's verdict on one mask (scores + whether it clears acceptance)."""

    scores: dict
    score: int
    passed: bool
    reasons: dict = field(default_factory=dict)
    feedback: str = ""


@dataclass
class AcceptedMask:
    """A returned candidate mask (PIL 'L'), its clean-image overlay, and its judge verdict."""

    mask: Any
    overlay: Any
    verdict: Verdict
    board: str = ""
    # The edits that produced this mask, oldest first — one per round that APPLIED one, so at most
    # `max_refine_rounds - 1` (the last round breaks before it edits, and at 1 round the trail is
    # empty); the slot invariant that lets a trail be traced to the mask it ended as is pinned by
    # a round that gives up judges the mask its own trail describes. Empty for a warm-start seed
    # or a re-judged keep-best candidate.
    trail: tuple = ()


@dataclass
class RunResult:
    """The outcome of one :func:`run_beam` call for a (clean image, defect type)."""

    masks: list
    api_calls: int = 0
    usage: dict = field(default_factory=dict)
    # Seconds per VLM call site — localize, propose, refine_assess, inloop_judge, final_judge — for
    # the sites this beam REACHED only; `None` (not `{}`) when the caller did not ask to profile.
    # the two records a finished beam is read by are on the result.
    phase_times: dict | None = None
    trace: list | None = None
    api_ms: float = 0.0  # summed HTTP round-trip time for this beam (pure API wait)
    non_api_ms: float = 0.0  # wall - api = all LOCAL cost (image encode/view-build/CV/parse + GIL-wait)
    wall_ms: float = 0.0  # total run_beam wall time on this beam's (single) thread
    # Candidates the final judge never answered on — NOT rejections; a caller sizing its work from the
    # observed yield must not read them as evidence this configuration produces few masks.
    unjudged: int = 0
    # Candidates a judge RETURNED A VERDICT ON, not candidates that reached one — `len(judged)` is the
    # wrong number. a beam reports verdicts it got not candidates it sent.
    judged: int = 0


# --------------------------------------------------------------------------------------------------
# Pure helpers.
# --------------------------------------------------------------------------------------------------
def _ptime(phase_times: dict | None, phase: str, fn, *a, **k):
    """Time a VLM call under a phase label (localize/propose/refine_assess/inloop_judge/final_judge)
    when profiling. ``phase_times`` is the per-run accumulator (or ``None`` to disable); the base
    phase (before ':') aggregates, so the per-round and per-candidate suffixes the refine and in-loop
    labels carry collapse into one bucket per phase.

    No lock: each run_beam owns its own dict and every call that writes it runs on that beam's thread
    (the final judge is a serial comprehension, not a fan-out)."""
    if phase_times is None:
        return fn(*a, **k)
    t0 = time.perf_counter()
    try:
        return fn(*a, **k)
    finally:
        base = phase.split(":")[0]
        phase_times[base] = phase_times.get(base, 0.0) + (time.perf_counter() - t0)


def _rigidize_fix(f: dict, *, dropped: list | None = None) -> dict:
    """FIXED-SHAPE: constrain one refine fix to rigid, shape-preserving edits — every ``"ops"`` entry
    that NAMES an op is limited to ``RIGID_CV_KINDS``, DERIVED from each op's declared ``rigid``.

    ``stamp`` PAINTS, so it unions a second copy into the mask, and declares ``rigid = False``; the
    derived set and the record the engine publishes are pinned to agree by
    the fixed shape record names the ops the engine actually KEEPS.

    ``dropped`` collects what was removed: an emptied ``"ops"`` list falls through to ``"kept"``,
    which is byte-identical to the model answering {"keep": true}.
    """
    g = dict(f)
    if g.get("ops"):
        # Through `filter_to_vocabulary` so aliases resolve and a `kind` arriving as a list is
        # DROPPED rather than raising `unhashable type` — out here, above the `try`. A kind-less op
        # rides through: `op_no_kind` is emitted downstream off the list this returns
        # (a misnamed op is named on a FIXED SHAPE run too).
        keep, dropped_kinds, no_kind = filter_to_vocabulary(g["ops"], RIGID_CV_KINDS)
        if dropped is not None:
            dropped.extend(dropped_kinds)
        g["ops"] = keep + no_kind
    return g


def _rigid_turn1(gen_specs: list, ctx) -> list:
    """FIXED-SHAPE, turn 1: the proposer's op sequences narrowed to what a shape-preserving run can
    draw from nothing — ``stamp`` only, however many of them a candidate asked for.

    A post-filter — nothing upstream knows the run is rigid — so it drops, records, and falls back to
    one stamp rather than leaving an EMPTY spec
    (fixed shape drops synthesized geometry and still returns a candidate). The refine half
    of this mode is :func:`_rigidize_fix`; the rules differ.
    """
    out: list = []
    for _i, spec in enumerate(gen_specs):
        # `{"stamp"}` and not `RIGID_CV_KINDS`: turn 1 has no mask, so the rigid ops have nothing
        # to move — the shape has to come out of the library first.
        keep, _drop, _nk = filter_to_vocabulary(spec, frozenset({"stamp"}))
        if _drop:
            log.warning(
                f"degraded[rigid_dropped]: candidate {_i} asked for "
                f"{'+'.join(_drop)} on a FIXED-SHAPE run, which "
                f"takes stamps only — dropped"
            )
        if not keep and ctx.shapes:
            # Keep what this mode can still honour — the model's own centre and a per-candidate
            # shape index — and require the centre to be USABLE, not merely two-long.
            # the fixed shape fallback does not collapse the beam to one mask,
            # the fixed shape fallback only claims a centre it could use.
            _at = next(
                (
                    list(o["center"])
                    for o in spec
                    if isinstance(o, dict)
                    and isinstance(o.get("center"), (list, tuple))
                    and len(o["center"]) == 2
                    and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0
                        for v in o["center"]
                    )
                ),
                None,
            )
            _gave = _at is not None
            _at = _at if _gave else [0.5, 0.5]
            _sid = _i % len(ctx.shapes)
            keep = [{"kind": "stamp", "shape_id": _sid, "center": list(_at)}]
            log.warning(
                f"degraded[rigid_fallback]: candidate {_i} emitted no stamp; "
                f"substituted shape_id {_sid} at {_at} — "
                f"{'its own centre' if _gave else 'the canvas centre; it named no usable one'}"
            )
        # Appended AFTER the fallback reads `keep`, so a candidate holding only malformed ops still
        # gets the substitution; they ride along only so the downstream filter can NAME them.
        out.append(keep + _nk)
    return out


def _as_seed_mask(_m, ctx):
    """A warm-start seed, as this beam's kind of mask at this beam's size — and say what that cost.

    Two records, not one: `seed_reduced` measures loss as a fraction of the canvas (which NEAREST
    preserves) and `seed_resized` names both sizes, which is also what makes an upscale visible —
    the seed record measures the door and the door is the RESIZE.
    """
    _raw = np.asarray(_m.convert("L") if isinstance(_m, Image.Image) else _m)
    _a, _notes = binarize(_raw)
    for _n in _notes:
        log.warning(f"degraded[seed_renormalized]: {_n}")
    _out = Image.fromarray((_a.astype("uint8") * 255), "L").resize((ctx.W, ctx.H), Image.NEAREST)
    if _raw.shape[:2] != (ctx.H, ctx.W):
        log.warning(
            f"degraded[seed_resized]: a seed arrived at "
            f"{_raw.shape[1]}x{_raw.shape[0]} and this beam is {ctx.W}x{ctx.H} — "
            f"resized with NEAREST, which neither smooths nor interpolates"
        )
    # RELATIVE AREA, after the resize: NEAREST preserves it to rounding, and the 2% floor is that
    # rounding — a 1-px hatching at 960x720 keeps none of itself through a 4x NEAREST downsample.
    _in = float(_a.sum()) / max(1, _a.size)
    _out_frac = float(np.count_nonzero(np.asarray(_out) > 127)) / max(1, ctx.W * ctx.H)
    if _in > 0 and _out_frac < _in * 0.98:
        log.warning(
            f"degraded[seed_reduced]: a seed covering {_in * 100:.2f}% of its "
            f"canvas covers {_out_frac * 100:.2f}% of this beam's — the mask did "
            f"not survive the way in, which is not a resolution change"
        )
    return _out


def _initial_candidates(seed_masks, *, ctx, conv, blocks, phase_times) -> tuple[list, dict]:
    """The beam's starting candidates and the rubric they are judged against — ``(masks, definition)``.

    A warm start makes no propose call and authors no rubric, so it returns the same EMPTY definition
    the propose-failure path falls back to; a cold start asks for ``ctx.num_candidates`` and renders
    each op sequence onto a blank canvas.

    ``masks`` is never empty: both arms substitute one blank canvas rather than return nothing, since
    the refine loop is index-parallel over this list and a beam with no slots cannot be refined.
    """
    K = ctx.num_candidates
    if seed_masks is not None:
        masks = [_as_seed_mask(_m, ctx) for _m in seed_masks][: max(K, 1)] or [Image.new("L", (ctx.W, ctx.H), 0)]
        # The only place an empty warm start is booked: no propose-side record covers the seeding
        # arm. a blank warm start is on the record.
        _blank_seeds = sum(1 for _m in masks if not np.any(np.asarray(_m) > 127))
        if _blank_seeds:
            log.warning(
                f"degraded[seed_empty]: {_blank_seeds} of {len(masks)} warm-start seeds "
                f"have no pixels — this beam will refine a blank, and nothing upstream of "
                f"here reports it"
            )
        return masks, {"summary": ""}

    # A call that gave up is already booked as `call_gave_up:propose`; this flag keeps the
    # short-candidate guard from booking the same event twice.
    # a propose call that gave up is not also booked as a short reply.
    _proposer_answered = True
    try:
        out = _ptime(
            phase_times,
            "propose",
            _ask_json_retry,
            conv,
            blocks,
            vlm_knobs.MAX_TOKENS_AGENT,
            f"propose {K} candidates",
            schema="propose",
            kind="propose",
        )
        # `is not None`, not `bool(out)`: `{}`, `[]`, `null` and `0` are replies that ARRIVED (a
        # top-level JSON array is a routine format slip) and must reach the guard below.
        _proposer_answered = out is not None
        out = out or {}
        defn = out.get("definition")
        definition = defn if isinstance(defn, dict) else {"summary": str(defn or "")}
        gen_specs = [c.get("ops", []) for c in out.get("candidates", []) if isinstance(c, dict)][:K]
        if ctx.rigid_only:
            gen_specs = _rigid_turn1(gen_specs, ctx)
        if ctx.siting.hard and ctx.siting.crops:  # each candidate's crop-frame coords -> ITS OWN region's box
            gen_specs = [ctx.siting.to_board(spec, i, ctx.W, ctx.H) for i, spec in enumerate(gen_specs)]
    except Exception:
        # _proposer_answered is deliberately NOT cleared here. Landing in this handler means a
        # reply arrived and blew up on the way through parsing — `out.get` on a list, a spec that
        # is not a dict, an `ops` that is not a list reaching the site remap — which is a reply the
        # guard below SHOULD report, not one it should excuse.
        definition, gen_specs = {"summary": ""}, []
    # ONE PATH. Turn 1 applies its op sequence to a blank canvas with the same
    # `apply_cv_sequence` refine uses, so there is no second renderer — and no second copy of the
    # vocabulary or of the per-op field list — to keep in step.
    _why: list = []
    _blank = Image.new("L", (ctx.W, ctx.H), 0)
    masks = []
    for _ci, p in enumerate(gen_specs):
        # Turn 1 is filtered by the run's vocabulary exactly as a refine fix is, one bucket per
        # kind. the restriction reaches turn 1 too and says when it leaves nothing,
        # one vocabulary filter serves both turns and buckets the same way.
        _kept, _wd, _nokind = filter_to_vocabulary(p or [], ctx.allowed)
        for _wk in _wd:
            log.warning(
                f"degraded[op_withdrawn:{record_label(_wk)}]: candidate {_ci} drew "
                f"with {_wk}, which this run does not offer — dropped"
            )
        # A kind-less op (`{"op":"polygon"}`, the field misnamed) falls through BOTH filters
        # above and is named here. an op with no kind is named on both turns.
        if _nokind:
            _saw = sorted({k for o in _nokind for k in o if isinstance(k, str)})[:4]
            log.warning(
                f"degraded[op_no_kind]: candidate {_ci} sent {len(_nokind)} op(s) with "
                f'no usable "kind" — keys were {_saw}; every op is identified by '
                f'"kind", and a null one names nothing'
            )
            # ONE ENTRY PER OP, with no count and no keys in it: the reporter below prefixes
            # `{n}x` from a Counter over these strings.
            _why.extend('op sent with no "kind"' for _ in _nokind)
        _oc: list = []
        masks.append(apply_cv_sequence(_kept, _blank, ctx.clean, layers=ctx.layers, outcomes=_oc, shapes=ctx.shapes))
        _why.extend(r.reason for _k, r in _oc if isinstance(r, Refused))
    # Counted from what RENDERED, not from what the reply contained — a non-empty spec can still
    # draw nothing, and each cause comes from `apply_cv_sequence`'s outcomes. Downstream of the
    # px->norm and per-region remaps, so a candidate lost to a degenerate crop is counted here.
    # candidates that render nothing are counted as short,
    # a primitive that draws nothing says why.
    _usable = sum(1 for m in masks if m.getbbox())  # PIL 'L'; getbbox() is None when all black
    if _proposer_answered and _usable < K:
        _causes = ", ".join(
            f"{n}x {w}" for w, n in sorted(collections.Counter(_why).items(), key=lambda kv: -kv[1])[:4]
        )
        log.warning(
            f"degraded[proposer_short]: asked for {K} candidates, "
            f"{_usable} of {len(masks)} rendered anything" + (f" — {_causes}" if _causes else "")
        )
    return (masks or [Image.new("L", (ctx.W, ctx.H), 0)]), definition


def _fix_target(f: dict, n: int) -> int | None:
    """Which candidate this fix is for, or None if it cannot be placed.

    An unplaceable fix is dropped and RECORDED — otherwise it is indistinguishable from `keep`, the
    same code path. The id is read by `_model_int`, the one rule `_assess_index` reads by too.
    """
    i = _model_int(f.get("id", -1), "fix_unplaceable")
    if i is None:
        return None
    if not 0 <= i < n:
        log.warning(f"degraded[fix_unplaceable]: id {i} outside 0..{n - 1}")
        return None
    return i


def _assess_index(assess: list, n: int) -> dict:
    """Score entries keyed by the candidate each belongs to — the read half of what _fix_target does
    for the write half. Ids are read from `id`, `candidate` or `index`, coerced, and fall back to
    position only when no entry is numbered at all.

    ALL OR NOTHING WHEN THE TARGET IS AMBIGUOUS: a partial placement misattributes verdicts
    silently, which zeros-plus-a-note does not. Every rejection below is a case where WHICH candidate
    an entry belongs to cannot be known; a repeated id and a short reply that names candidate 0 are
    not, so they keep their scores — a short numbered reply is never read as a shifted one.
    """
    _KEYS = ("id", "candidate", "index")
    if not all(isinstance(x, dict) for x in assess):
        # a prose entry ("C0 looks good") among the objects: every position below it shifts by one
        log.warning("degraded[assess_unplaceable]: the assess list holds something that is not an object")
        return {}
    _named = [any(k in x for k in _KEYS) for x in assess]
    if any(_named) and not all(_named):
        # Half-numbered: position next to explicit ids is a guess.
        log.warning("degraded[assess_unplaceable]: some entries carry an id and some do not")
        return {}
    if not any(_named) and len(assess) != n:
        # Purely positional is a fair reading only when the reply is COMPLETE.
        log.warning(f"degraded[assess_unplaceable]: {len(assess)} unnumbered entries for {n} candidates")
        return {}
    out: dict = {}
    for pos, x in enumerate(assess):
        i = _model_int(x.get("id", x.get("candidate", x.get("index", pos))), "assess_unplaceable")
        if i is None:
            return {}
        if not 0 <= i < n:
            log.warning(f"degraded[assess_unplaceable]: id {i} outside 0..{n - 1} — whole assess discarded")
            return {}
        if i in out:
            # Unambiguous — the model meant candidate `i` both times — so keep the first rather
            # than zeroing the whole round.
            log.warning(f"degraded[assess_duplicate_id]: id {i} scored twice — keeping the first")
            continue
        out[i] = x
    if len(out) < n and 0 not in out and any(_named):
        # A short NUMBERED reply that never names candidate 0 reads both 0-based and 1-based, so it
        # is discarded; naming candidate 0 makes it unambiguous and it keeps its scores. The range
        # check above only refuses the COMPLETE shift, whose last id runs off the end.
        log.warning(
            f"degraded[assess_unplaceable]: {len(out)} numbered entries for {n} candidates "
            f"and none is candidate 0 — 0-based and 1-based readings disagree on every "
            f"score, whole assess discarded"
        )
        return {}
    return out


def _not_destroyed(old, new) -> bool:
    """A refine step may change a mask freely, but a fix that renders it (near-)empty or collapses it
    to a sliver is a mechanical failure — a bad spec or an over-aggressive op — never an intended edit.
    Model-free and defect-agnostic; growth is NOT capped, since union-style ops legitimately grow a lot.

    Destruction is relative to the INPUT, so an already-empty candidate cannot be emptied
    (a fix where nothing ran is not reported as one that destroyed the mask). `nm is m` is not
    an equivalent test: a `translate` with no `dx` and no `dy` APPLIES (both default to 0.0) and
    returns a new object."""
    na = int((np.asarray(new) > 127).sum())
    oa = int((np.asarray(old) > 127).sum())
    if na == 0:
        return oa == 0
    return not (oa > 0 and na < 0.05 * oa)


def _ask_json_retry(conv, blocks, max_tokens, what, schema=None, *, kind):
    """One retry so a transient parse hiccup doesn't void the whole run — a failed assess call
    `break`s the refine loop, and a cold-start beam that breaks on round 0 reaches the final judge
    with nothing to judge.
    ask_json appends the user turn BEFORE calling the API, so on an API failure the dangling user
    message must be dropped before re-sending the same blocks; on a parse failure the assistant reply
    is already in history, so a small corrective turn re-asks for the JSON — except an EMPTY reply,
    which ask_json does not append, so it is re-sent like an API failure. Final failure -> logged
    to stdout, returns None (caller decides how to degrade).

    ``what`` is the sentence a person reads; ``kind`` is the `degraded[KIND]` bucket — keyword-only
    with no default, so a new call site states its own."""
    try:
        return conv.ask_json(blocks, max_tokens=max_tokens, schema=schema)
    except CallFailed as e:
        # NO RETRY: the transport already spent the attempts it judged worthwhile.
        #
        # Pop first: ask_json appends the user turn BEFORE the call, and /messages requires
        # alternation — this same conv is carried into every later round.
        if conv.messages and conv.messages[-1]["role"] == "user":
            conv.messages.pop()
        log.warning(f"degraded[call_gave_up:{kind}]: {e.failure.kind}: {e.failure.detail}")
        return None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"  # a reply ARRIVED and could not be used
        log.warning(f"[agent] {what}: attempt 1 failed ({err}); retrying once")
        try:
            if conv.messages and conv.messages[-1]["role"] == "user":
                conv.messages.pop()  # no assistant turn kept (API died, or an empty reply): drop the ask, resend as-is
                return conv.ask_json(blocks, max_tokens=max_tokens, schema=schema)
            # The reply arrived and was unusable. SAY WHICH: telling a model its JSON was invalid when
            # the JSON was fine and a field was missing sends it to re-format instead of to answer.
            fix = (
                f"Your previous reply was valid JSON but did not match what was asked: {e.detail}. "
                "Send it again as ONE JSON object with every required field present and in range, "
                "no prose."
                if isinstance(e, SchemaInvalid)
                else "Your previous reply was not valid JSON. Repeat it as ONE valid JSON object — "
                "same schema, no prose."
            )
            return conv.ask_json([fix], max_tokens=max_tokens, schema=schema)
        except Exception as e2:
            if conv.messages and conv.messages[-1]["role"] == "user":
                conv.messages.pop()  # keep history alternating for any later turns
            err2 = f"{type(e2).__name__}: {e2}"
            log.warning(f"degraded[call_gave_up:{kind}]: {err} -> {err2}")
            return None


def _avg4(a: dict) -> int:
    try:
        return round(sum(_score_num(a.get(k, 0)) for k in DIMS) / len(DIMS))
    except Exception:
        return 0


def _last_edits_notes(op_outcomes: list, t: int) -> list[str]:
    """What the previous round's ops DID, as turn-leading notes for the next ask.

    Two channels, neither of which the agent can see from the mask: a REFUSED op left the candidate
    untouched, and an APPLIED one may have had its value clamped on the way in
    (`Applied.adjustments`). Returns the notes in the order they should be inserted at the FRONT of
    the ask, or [] when nothing happened.

    Round-stamped, because the note stays in the conversation after it is sent —
    the notes are round stamped and adjustments reach the agent.
    """
    notes: list[str] = []
    _ref = [(c, r) for c, _k, r in op_outcomes if isinstance(r, Refused)]
    _adj = [(c, k, a) for c, k, r in op_outcomes if isinstance(r, Applied) and r.adjustments for a in r.adjustments]
    if _adj:
        _atxt = "; ".join(dict.fromkeys(f"candidate {c}: {k} — {a}" for c, k, a in _adj))[:300]
        notes.insert(
            0,
            f"ROUND {t} — ADJUSTED: {len(_adj)} of your value(s) were outside what the op "
            f"accepts and were changed. {_atxt}",
        )
    if _ref:
        # De-duplicated on the TEXT, then truncated on an ENTRY boundary — a mid-word cut is not a
        # report.
        _uniq, _by = [], {}
        for cand, r in _ref:
            _by.setdefault(f"{r.reason} — {r.correction}", []).append(cand)
        for text, cands in _by.items():
            _uniq.append(f"candidate {', '.join(str(c) for c in sorted(set(cands)))}: {text}")
        _keep, _n = [], 0
        for entry in _uniq:  # whole entries only, never a partial one
            if _n + len(entry) > 600 and _keep:
                break
            # The FIRST entry is bounded too (`and _keep` exempts it from the loop's cap): a refusal
            # embeds the model's own VALUE verbatim, so `disk{radius:"z"*5000}` makes a 5,144-char note.
            _keep.append(entry if len(entry) <= 600 else entry[:597] + "...")
            _n += len(entry) + 2
        # the header counts the WORLD (every refusal), `(+N more)` counts what is not shown.
        # Reporting only what is shown tells the agent that the hidden failed edits worked.
        _hidden = sum(len(_by[e.split(": ", 1)[1]]) for e in _uniq[len(_keep) :])
        _more = f" (+{_hidden} more not shown)" if _hidden else ""
        notes.insert(
            0,
            f"ROUND {t} — YOUR LAST EDITS: {len(_ref)} of {len(op_outcomes)} op(s) did NOT run, "
            f"so those candidates are unchanged. " + "; ".join(_keep) + _more,
        )
    return notes


def _apply_fixes(masks: list, fixes: dict, ctx, t: int, live=None) -> tuple[list, list, list]:
    """One refine round applied: every candidate's mask after its fix, what was done to it,
    and what the ops said about it — ``(new_masks, new_actions, op_outcomes)``.

    INDEX-PARALLEL with ``masks`` on every branch — a round can lose an EDIT, never a candidate.
    Only the applied branch is covered: no test reaches the `except` below, so that append is
    unguarded.

    ``live`` names the candidates this round still refines, in order, for ``beam_prune`` /
    ``accept_continue``; ``None`` (the default, and the only value on the delivery path) means every
    one of them. The returns are parallel to ``live``, but every INDEX in a log line and in
    ``op_outcomes`` is the candidate's own number — the ``Ck`` the model was shown — because a
    renumbered report names the wrong candidate to the only reader who cannot check.

    ``new_actions`` is the OFFLINE record (it reaches `RunResult.trace` and `AcceptedMask.trail`)
    and is never sent to a model; ``op_outcomes`` is the other direction — what the next turn
    tells the agent about the edits it asked for.
    """
    op_outcomes: list = []
    new_masks, new_actions = [], []
    for k in range(len(masks)) if live is None else live:
        m = masks[k]
        f = fixes.get(k)
        if f is not None and ctx.rigid_only:  # strip non-rigid ops before applying
            _rd: list = []
            f = _rigidize_fix(f, dropped=_rd)
            if _rd:
                # Recorded here because the rewrite precedes the withdrawal log below; the line
                # INTERPOLATES `RIGID_CV_KINDS` so the set cannot drift out from under it.
                # the fixed shape record names the ops the engine actually KEEPS.
                log.warning(
                    f"degraded[rigid_dropped]: round {t} candidate {k} asked for "
                    f"{'+'.join(dict.fromkeys(_rd))} on a FIXED-SHAPE run, which takes "
                    f"{'/'.join(sorted(RIGID_CV_KINDS))} only — dropped"
                )
        try:
            if f is None or f.get("keep"):
                new_masks.append(m)
                new_actions.append("kept")
            elif f.get("ops"):
                # `ctx.allowed` is the EFFECTIVE set and always concrete: an empty-means-everything
                # sentinel cannot tell "nothing restricted" from "nothing survived the layer
                # withdrawal" —
                # a restriction the layer withdrawal empties does not become no restriction.
                ops, _wd, _nokind = filter_to_vocabulary(f["ops"], ctx.allowed)
                # One bucket per (candidate, kind), not per op: a run's log is counted by LABEL, so
                # collapsing the kind into the detail makes the ops uncountable.
                # one vocabulary filter serves both turns and buckets the same way.
                for _wk in _wd:
                    log.warning(
                        f"degraded[op_withdrawn:{record_label(_wk)}]: round {t} "
                        f"candidate {k} asked for {_wk}, which this run does not "
                        f"offer — dropped"
                    )
                # ...and the SAME no-kind case turn 1 has —
                # an op with no kind is named on both turns.
                if _nokind:
                    _ks = sorted({kk for _o in _nokind for kk in _o if isinstance(kk, str)})[:4]
                    log.warning(
                        f"degraded[op_no_kind]: round {t} candidate {k} sent "
                        f'{len(_nokind)} op(s) with no usable "kind" — keys were {_ks}; '
                        f'every op is identified by "kind", and a null one '
                        f"names nothing"
                    )
                _oc: list = []
                nm = (
                    apply_cv_sequence(ops, m, ctx.clean, layers=ctx.layers, outcomes=_oc, shapes=ctx.shapes)
                    if ops
                    else m
                )
                # WITH the candidate index: an accumulator pooled over the round names nobody.
                op_outcomes.extend((k, _kind, _r) for _kind, _r in _oc)
                if _not_destroyed(m, nm):
                    new_masks.append(nm)
                    # What RAN, not what was asked for: a bare "CV[snap_layer]" reads the same
                    # whether the op moved the mask or was refused.
                    _ran = [kk for kk, r in _oc if isinstance(r, Applied)]
                    _no = [kk for kk, r in _oc if isinstance(r, Refused)]
                    new_actions.append(
                        ("CV[" + "+".join(_ran) + "]" if _ran else "CV[none applied]")
                        + (" refused[" + "+".join(dict.fromkeys(_no)) + "]" if _no else "")
                    )
                else:
                    new_masks.append(m)
                    # EMPTIED and COLLAPSED are reported as different failures: "drew nothing" is
                    # false for a collapse, where every op applied and one drew real area.
                    # A fix where nothing ran must not be reported as one that destroyed the
                    # mask; no test reaches either arm below.
                    _new_px = int((np.asarray(nm) > 127).sum())
                    _old_px = int((np.asarray(m) > 127).sum())
                    _refused = [f"{kk}: {r.reason}" for kk, r in _oc if isinstance(r, Refused)]
                    if _new_px == 0:
                        _what = "emptied the mask"
                        _why = (
                            _refused
                            or [f"{kk} applied but drew nothing" for kk, r in _oc if isinstance(r, Applied)]
                            or ["no op ran"]
                        )
                    else:
                        # The ops that RAN, first: a refusal left the mask untouched, so listing
                        # refusals ahead of them hides the op that actually collapsed it.
                        _what = f"collapsed the mask to {_new_px}px from {_old_px}px, under the 5% floor"
                        _ran = [f"{kk} applied" for kk, r in _oc if isinstance(r, Applied)]
                        _why = _ran + _refused or ["no op ran"]
                    log.warning(
                        f"degraded[redraw_empty]: round {t} candidate {k} {_what} "
                        f"and was refused — {'; '.join(dict.fromkeys(_why))}"
                    )
                    _ops_asked = "+".join(dict.fromkeys(kk for kk, _r in _oc)) or "the edit"
                    new_actions.append(
                        f"kept(guard: op emptied/collapsed — {_ops_asked} "
                        + (
                            f"left the mask EMPTY"
                            if _new_px == 0
                            else f"collapsed the mask to {_new_px}px from {_old_px}px"
                        )
                        + "; candidate kept UNCHANGED. Ask for a gentler edit, or none)"
                    )
            else:
                new_masks.append(m)
                new_actions.append("kept")
        except Exception as e:
            # `kept(err)` marks it in this beam's action log; this counts it where the run reports.
            log.warning(f"degraded[fix_apply_failed]: {type(e).__name__}: {e}")
            new_masks.append(m)
            new_actions.append("kept(err)")
    return new_masks, new_actions, op_outcomes


def _final_selection(
    judge,
    ctx,
    *,
    masks,
    scored,
    pool,
    seed_orig,
    applied_log,
    bar,
    beam,
    board,
    slots=None,
    slot_log=None,
    banked=(),
) -> tuple[list, int, int]:
    """Everything the beam earned, judged and accepted — ``(accepted_masks, unjudged, judged)``.

    ``judge`` is the authoritative consensus call, taking ``(selfsc, mask, slot)`` and returning
    ``(mask, score, subs, reasons, feedback, slot, unjudged)``.

    Three pools, deduped against ONE ``seen`` set so no mask is judged twice: the last round's top
    ``beam`` candidates, the original warm-start seeds (``seed_orig``, empty on a cold start), and
    ``pool`` — the in-loop judge's per-round bests.

    ``banked`` is ``cfg.accept_continue``'s output — ``(mask, score, subs, reasons, slot)`` the IN-LOOP
    judge already cleared the bar on. They are emitted as passed WITHOUT a final-judge call (that
    bypass IS the lever), so they are deduped FIRST and every later pool sees their geometry.

    ``slots``/``slot_log`` carry candidate IDENTITY, which stops being the list POSITION as soon as
    ``beam_prune``/``accept_continue`` drop a candidate mid-loop: ``slots[i]`` is the stable id of
    ``masks[i]``, and ``slot_log[r]`` names the candidate each action in ``applied_log[r]`` belongs to.
    Both ``None`` means nothing was ever dropped, so position IS identity and the trail reads exactly as
    it did before either lever existed.

    Neither count is ``len()`` of anything returned — see :class:`RunResult`: a beam reports
    verdicts it got, not candidates it sent.
    """
    seen = set()
    cands = []
    # BANKED FIRST: `seen` is what stops a mask being judged twice, and these are already accepted —
    # letting a later pool claim the same geometry would pay for the very final-judge call this lever
    # exists to skip.
    banked_out = []
    for _m, _jsc, _subs, _reasons, _slot in banked:
        g = mask_geometry(_m)
        sig = (g.get("area_ratio"), tuple(g.get("centroid") or []))
        if sig in seen:
            continue
        seen.add(sig)
        banked_out.append((_m, _jsc, _subs, _reasons, _slot))
    # The mask is resolved from `masks[k]` HERE, not snapshotted when the round scored it: the scores
    # can be one round older than the masks (a round whose assess call gives up `break`s before it
    # scores anything). a round that gives up judges the mask its own trail describes.
    for selfsc, k, _su in sorted(scored, key=lambda x: -x[0])[:beam]:  # pool 1 of 3 — see the docstring
        m = masks[k]
        g = mask_geometry(m)
        sig = (g.get("area_ratio"), tuple(g.get("centroid") or []))
        if sig in seen:
            continue
        seen.add(sig)
        cands.append((selfsc, m, slots[k] if slots is not None else k))
    # WARM-START: the refine may degrade a good seed, so the ORIGINAL accepted seeds stay in the
    # pool (keep-best). Empty on a cold start, where every candidate came out of the proposer.
    for _sm in seed_orig:
        g = mask_geometry(_sm)
        sig = (g.get("area_ratio"), tuple(g.get("centroid") or []))
        if sig in seen:
            continue
        seen.add(sig)
        cands.append((8, _sm, -1))  # a carried-over seed has no trail in THIS beam

    # SERIAL over candidates — the fill loop parallelizes at the BEAM level (one clean concurrency knob),
    # so the final judge stays single-threaded on this beam's thread (simpler + clean --trace-dir
    # grouping). Candidates do NOT share a prompt — both overlay images carry the mask, and so does the
    # measured-geometry text when two candidates differ in shape — so the cache is warmed within one
    # candidate's own samples (judging.py), not across them.
    cres = [judge(cm) for cm in cands]

    def _trail(slot):
        """The edits this candidate went through, oldest first. Empty for a warm-start seed, which was
        produced by an earlier beam and has no history in this one.

        `slot` is the candidate's STABLE id. With no lever dropping candidates that is also its
        position, and `slot_log` is None, so this reads positionally; once a round can drop a
        candidate, position and identity diverge and only `slot_log` says which action belonged to
        whom."""
        if slot is None or slot < 0:
            return ()
        if slot_log is None:
            return tuple(r["actions"][slot] for r in applied_log if slot < len(r.get("actions") or []))
        out = []
        for r, ids in zip(applied_log, slot_log):
            acts = r.get("actions") or []
            if slot in ids and ids.index(slot) < len(acts):
                out.append(acts[ids.index(slot)])
        return tuple(out)

    unjudged = sum(1 for c in cres if c[6])
    n_judged = 0
    judged = []
    # The banked masks, carrying the in-loop verdict they were accepted on. `passed=True` is not a
    # re-derivation: `accept_continue` only banks what `bar.clears` already said yes to.
    for _m, _jsc, _subs, _reasons, _slot in banked_out:
        judged.append((_m, _jsc, _subs, True, _reasons, "", _trail(_slot)))
    for m, jsc, subs, reasons, jfb, slot, _uj in cres:
        # An unjudged candidate is still recorded but cannot pass (`bar.clears({})` is False), so
        # `len(judged)` counts candidates that REACHED a judge — which is the other question.
        if not _uj:
            n_judged += 1
        passed = bar.clears(subs)
        judged.append((m, jsc, subs, passed, reasons, jfb, _trail(slot)))
    # GLOBAL KEEP-BEST: an earlier round's strong candidate is admitted on the FINAL judge's verdict,
    # never the in-loop one's — the bar is calibrated on a multi-sample consensus and the in-loop
    # judge takes ONE sample, so admitting on it lets candidates in on noise.
    # the keep best pool is admitted on the final judges verdict.
    #
    # The in-loop score still decides WHICH pool candidates are worth the call; the ones that are get
    # re-judged, with an empty trail, having no recorded edit history in this beam.
    _pool_cands = []
    for m, jsc, subs, reasons in pool:
        g = mask_geometry(m)
        sig = (g.get("area_ratio"), tuple(g.get("centroid") or []))
        if sig in seen:
            continue
        seen.add(sig)
        if bar.clears(subs):  # pre-filter on the GUIDANCE score, before paying for a judge call
            _pool_cands.append((jsc, m, None))
    for m, jsc, subs, reasons, jfb, _slot, _uj in [judge(cm) for cm in _pool_cands]:
        if _uj:
            unjudged += 1
            continue
        n_judged += 1
        passed = bar.clears(subs)
        judged.append((m, jsc, subs, passed, reasons, jfb, ()))
    # emit ONLY judge-accepted masks — a rejected run yields 0 finals (the caller retries to fill)
    returned = [x for x in judged if x[3]]
    returned.sort(key=lambda x: -x[1])

    return (
        [
            AcceptedMask(
                mask=m,
                overlay=overlay_mask(ctx.clean, m),
                verdict=Verdict(
                    scores={d: subs.get(d, 0) for d in DIMS}, score=jsc, passed=passed, reasons=reasons, feedback=jfb
                ),
                board=board,
                trail=trail,
            )
            for (m, jsc, subs, passed, reasons, jfb, trail) in returned
        ],
        unjudged,
        n_judged,
    )


def _refine_ask(
    ctx, masks, *, t, last, review_blocks, last_area, judge_note, op_outcomes, board, mem_note="", work_region=None
) -> list:
    """One refine turn, assembled: the views of every candidate and the instruction that goes with
    them. Builds images and text and makes no model call, so a caller can render a turn and read it.

    Everything the round CONSUMES is an argument and none of it is written back; the caller clears
    them once the turn is built.

    ``work_region`` is the beam-level localized box; a candidate with an assigned crop uses that
    crop's box instead. Either way each candidate's zoom is widened to contain it, so the agent
    refines against the whole host feature rather than only where its mask already is.

    ``mem_note`` is the in-loop beam's cross-round memory (empty on every other path, so the default
    turn is unchanged). Like ``judge_note`` and the op-outcome notes it leads the turn, and like them
    it is exempt from the leak guard below — see ``theirs``.

    The returned list is already through :func:`~anomalygen.agentic_mask_generation.prompts.assert_no_leak`.
    """
    overlays = [overlay_mask(ctx.clean, m) for m in masks]
    # tiles fitted and gridded at TILE size — see Views.montage for the three stacked shrinks
    montage = ctx.views.montage(overlays, [f"C{k}" for k in range(len(masks))])
    clean_gv = ctx.clean_grid

    ask: list[Any] = list(review_blocks)
    # Blocks this engine did not write — the card, the visual prompt, the distilled shape sentence —
    # exempted from the leak guard below BY IDENTITY: the notes below are `insert(0, ...)`ed after this
    # list is built, so an index-based exemption could not survive the shift.
    theirs: list = list(review_blocks) + ([ctx.shape_line] if ctx.shape_line else [])
    ask += [
        ctx.prompts.refine_frame(ctx.siting.hard),
        *([ctx.shape_line] if ctx.shape_line else []),
        "CLEAN image (no mask) — the geometry REFERENCE (judge fit against THIS, not the examples):",
        clean_gv,
        f"Your {len(masks)} current candidates C0..C{len(masks) - 1} (red mask on clean):",
        montage,
        ex_dist_summary(ctx.exemplars) + " For any multi-instance candidate, compare its OWN count/"
        "spacing to these and CLOSE the gap (add instances / tighten clustering) until it clearly "
        "matches — leave margin so distribution is unambiguous.",
    ]
    for k, (m, ov) in enumerate(zip(masks, overlays)):
        # `k` is the current-round POSITION, consistent with the C{k} labels this turn shows (crop
        # anchoring is approximate after a prune/bank renumbering) — see `frame_for`.
        _reg = ctx.siting.frame_for(k) if ctx.siting.crops else work_region
        z, zb = ctx.views.zoom_around_mask(ov, m, region=_reg)
        if z is not None:
            z = ctx.views.grid_fullcoords(z, zb)
            _bb = (mask_geometry(m) or {}).get("bbox")
            _sz = ""
            if isinstance(_bb, (list, tuple)) and len(_bb) == 4:
                _sz = ctx.prompts.zoom_units(
                    (ctx.W, ctx.H),
                    ((_bb[2] - _bb[0]) * ctx.W, (_bb[3] - _bb[1]) * ctx.H),
                    (zb[2] - zb[0]) * ctx.W if zb else ctx.W,
                )
            _was = last_area[k] if k < len(last_area) else None
            if _was is not None:
                _now = int(np.count_nonzero(np.asarray(m) > 127))
                _sz += f" Your last edit took this candidate from {_was} to {_now} px ({_now / max(_was, 1):.1f}x)."
            ask += [f"C{k} zoom{_sz}:", z]
        # The same measured coverage line the judge reads — the refiner must see the number it will
        # be judged by, or the judge's expectation is unactionable. Empty = unmeasurable = absent.
        _fn = _fill_note(ctx, m)
        if _fn:
            ask.append(f"C{k}: {_fn[0]}")
    # The `clear`-led redraw hatch, gated on BOTH the vocabulary and the shape mode: `rigid_only`
    # does not narrow `ctx.allowed`, so a vocabulary-only gate would offer a restart no op could
    # perform. The hatch is shut by the experimenter (withholding `clear`) and by nobody else —
    # that rule drives the vocabulary gate; the `rigid_only` half is unguarded.
    ask.append(ctx.prompts.refine_ask(len(masks), last=last, restart="clear" in ctx.allowed and not ctx.rigid_only))
    if ctx.rigid_only and not last:
        # Rendered from the registry intersected with the run, like every other offer here: a
        # written-out triple would offer whichever of them this beam does not have.
        _rigid = sorted(RIGID_CV_KINDS & ctx.allowed)
        if not _rigid and not t:
            # Nothing to describe, and no fix this run can survive: `_rigidize_fix` keeps only
            # these ops. Said ONCE per beam — a property of the configuration, not of the round.
            log.warning(
                f"degraded[no_rigid_op]: [{ctx.defect_type}] {board} is FIXED-SHAPE and "
                f"allows none of {'/'.join(sorted(RIGID_CV_KINDS))}, so no fix can "
                f"survive and every refine round will keep the mask it was given"
            )
        if _rigid:
            ask.append(
                'FIXED-SHAPE: every fix MUST be rigid — "ops" from {'
                + ", ".join(_rigid)
                + "} ONLY; anything else is dropped. Preserve the "
                "stamped shape exactly; adjust position/orientation only."
                # Only when the run HAS it — a warning about an unreachable tool is a
                # leaked op with extra steps.
                + (
                    ' Note "stamp" PAINTS a second copy rather than moving the one you have.'
                    if "stamp" in ctx.allowed
                    else ""
                )
            )
    if judge_note:
        ask.insert(0, judge_note)
        theirs.append(judge_note)  # the in-loop judge's own words — see `theirs` above
    if mem_note:
        # The MODEL's own last-round scores and chosen fix, quoted back. A REPORT, not an offer, on
        # the same footing as the op-outcome notes below: it can name an op the model asked for and
        # the run does not offer, and guarding it would let one reply kill the beam.
        ask.insert(0, mem_note)
        theirs.append(mem_note)
    # OP OUTCOMES from the round just finished, capped, and to the GENERATOR only —
    # a refused op is told to the agent and never to the judge.
    for _note in reversed(_last_edits_notes(op_outcomes, t)):
        ask.insert(0, _note)
        # A REPORT, not an offer: each note quotes the registry's own `signature()` text and the
        # model's rejected value verbatim, so guarding it would let a reply kill the beam.
        theirs.append(_note)
    # The same guard `propose_ask` and the system prompt run, over what this function OFFERS — none
    # of it comes from `ops_catalogue`, so nothing but this reads it. Unguarded: neutering this call
    # leaves the whole suite green, so no test covers the refine turn's leak check.
    _theirs = {id(b) for b in theirs}
    assert_no_leak(
        "\n".join(b for b in ask if isinstance(b, str) and id(b) not in _theirs), tuple(ctx.allowed), "refine_ask"
    )
    return ask


# --------------------------------------------------------------------------------------------------
# THE IN-LOOP BEAM (`beam_multi` / `beam_samples`, both 0/1 = off).
#
# One lever: re-sample the refine turn, render every result, and let a judge keep the best ONE per
# candidate.
# --------------------------------------------------------------------------------------------------
def _multi_ask(k: int) -> str:
    """The MULTI override: K materially-different revisions per candidate, in ONE generation.

    ``k`` is a COUNT (`Config.beam_multi` is an int for exactly this reason): rendered from a bool it
    would read "propose 1 MATERIALLY DIFFERENT candidate revisions" and the lever would do nothing.
    """
    return (
        "OVERRIDE the fix format: instead of ONE fix per candidate, propose " + str(int(k)) + " MATERIALLY "
        "DIFFERENT candidate revisions for EACH candidate — genuinely distinct STRATEGIES (e.g. a small "
        "positional/size adjust vs a full redraw vs growing to reach), NOT minor numeric variations of "
        'one idea. Output {"assess":[...],"fixes":[{"id":0,"options":[FIX, FIX, ...]}, ...one per '
        'candidate]} where each FIX is a normal fix object PLUS an integer field "self": 0-10 = your OWN '
        "honest quality score for THAT option (how well it fixes the defect). Give genuinely DIFFERENT "
        "self scores across the options; do not tie them all."
    )


def _multi_options(v, n: int) -> dict:
    """``{candidate index: [option fix, ...]}`` from a MULTI reply.

    A candidate whose entry carries no usable ``options`` list falls back to the entry ITSELF as its
    single option, so a model that ignored the override still gets its one fix applied rather than
    dropped. Placement uses the same `_fix_target` the greedy path does, so an unplaceable id is
    recorded once and skips ONE candidate, never the round.
    """
    out: dict = {}
    for f in (v or {}).get("fixes") or []:
        if not isinstance(f, dict):
            continue
        fid = _fix_target(f, n)
        if fid is None:
            continue
        opts = f.get("options")
        out[fid] = [o for o in opts if isinstance(o, dict)] if isinstance(opts, list) and opts else [f]
    return out


def _beam_memory_note(scored, assess_ix: dict, labels: dict, live: list | None = None) -> str:
    """A compact record of THIS round's own reasoning, replayed into the next round.

    ``scored``'s ids are THIS round's positions, but the note is read NEXT round, beside a montage
    ``live`` has relabelled — so each entry is labelled by its next-round position, and a candidate a
    lever took out of the beam is dropped (there is no C to hang it on any more).

    The merged assess+options call runs on a side conversation, so the main conv never sees the
    round's own scores; without this the beam loses the round-to-round memory the plain path gets for
    free by keeping the reply in history.

    ``labels`` is per candidate, and is the ACTION LABEL `_apply_fixes` produced for the option the
    model scored highest ITSELF — so the note quotes what that option actually DID (after the
    vocabulary filter and the destruction guard), not the kinds the reply asked for. Re-deriving the
    label here by reading `kind` out of the raw fix would be both a second reading of a reply the
    renderer has already read and an op-kind lookup in a module that is not allowed one (one
    vocabulary filter serves both turns and buckets the same way).

    `_dimstr`, NOT `_dims`: a display string must never carry the name of a live dimension set. Bound
    to one in a scope that also builds judges, a single round of beam memory would rebind it to a
    34-character string for the rest of that scope — every judge after it would iterate CHARACTERS,
    every candidate would score 0, and the selector's max() would fall through to candidate #0 on a
    total tie, silently. The engine reads the module-level DIMS and threads no dimension set through
    the beam; the name is kept distinct regardless.
    """
    pos = {k: i for i, k in enumerate(live)} if live is not None else None
    notes = []
    for _sc, k, subs in scored:
        if pos is not None and k not in pos:
            continue
        a = assess_ix.get(k, {})
        _dimstr = " ".join(f"{d[:4]}{int(subs.get(d, 0))}" for d in DIMS)
        reason = str(a.get("reason", ""))[:100]
        notes.append(
            f"C{pos[k] if pos is not None else k}[{_dimstr}] reason:{reason} -> your top fix:{labels.get(k, 'keep')}"
        )
    if not notes:  # every scored candidate left the beam: nothing next round could act on
        return ""
    return (
        "YOUR OWN self-assessment LAST round (continuity; the board images below are the current "
        "truth): " + " || ".join(notes)
    )


def _beam_pairwise_winner(options, *, vlm, ctx, defect_type, sel, phase_times) -> int:
    """King-of-the-hill over one candidate's rendered options — K-1 sequential A-vs-B calls.

    A weak judge compares two masks more reliably than it scores one alone, and the prompt explicitly
    counters the aesthetic bias (crude-but-correct BEATS pretty-but-wrong).

    `overlay_mask` is the module-level import from `anomalygen.agentic_mask_generation.imaging` — an
    absolute package import, never a bare `from judging import overlay_mask`, which resolves only
    while PYTHONPATH points INSIDE the package and raises ModuleNotFoundError on every beam that
    enters this branch otherwise.

    `judge_vis=True`, and this is the ONLY call site in the package that asks for it: a comparison is
    decided on where the boundary IS, so the selector draws the boundary it asks about (a 2px contour)
    and lightens the fill to 35% to keep the structure under it readable, where the flat fill every
    other overlay uses hides exactly what is being compared.
    """

    def _b_beats_a(ma, mb) -> bool:
        try:
            blocks: list[Any] = [f"Defect type: {defect_type}"]
            if ctx.judge_docs:
                blocks.append("Defect knowledge:\n" + ctx.judge_docs)
            blocks += [
                "Two candidate defect masks (red overlay) on the SAME clean board. Decide which is a "
                "MORE CORRECT instance of THIS defect — correct structure / placement / extent / "
                "connectivity — NOT which looks tidier or more natural. A crude shape that is "
                "structurally correct BEATS a pretty shape that is structurally wrong.",
                "CANDIDATE A:",
                overlay_mask(ctx.clean, ma, judge_vis=True),
                # the same measured coverage every other role reads — a comparison that cannot see
                # the numbers re-introduces the eyeballing the note exists to replace
                *_fill_note(ctx, ma),
                "CANDIDATE B:",
                overlay_mask(ctx.clean, mb, judge_vis=True),
                *_fill_note(ctx, mb),
                'Answer ONLY compact JSON {"winner":"A"} or {"winner":"B"}.',
            ]
            conv = vlm.conversation(ctx.judge_system, model=sel, pin_first_images=True)
            out = _ptime(phase_times, "beam_select", conv.ask_json, blocks, max_tokens=400) or {}
            return str(out.get("winner", "A")).strip().upper().startswith("B")
        except Exception as e:  # noqa: BLE001 — a lost comparison keeps the champion, never sinks the beam
            log.warning(f"degraded[beam_select_failed]: pairwise compare: {type(e).__name__}: {e}")
            return False

    champ = 0
    for i in range(1, len(options)):
        if _b_beats_a(options[champ][1], options[i][1]):
            champ = i
    return champ


def _beam_round(
    masks,
    fixes,
    *,
    ctx,
    t,
    live,
    ask,
    v,
    merged,
    beam_snap,
    vlm,
    cfg,
    gen_model,
    beam_multi,
    beam_samples,
    beam_sel,
    defect_type,
    definition,
    phase_times,
    work_region=None,
) -> tuple[list, list, list, dict]:
    """One refine round done as a BEAM — ``(new_masks, new_actions, op_outcomes, memory_labels)``.

    ``work_region`` is the beam-level localized box, handed to the selector judge so its zoomed view
    contains the whole work region (a candidate with an assigned crop uses that crop's box instead).

    Two ways to get more than one fix per candidate, and the config picks exactly one:
    ``beam_multi`` asks for K distinct STRATEGIES in a single generation, ``beam_samples`` re-asks the
    identical turn N times at the endpoint's own sampling. Either way each option is rendered and a selector keeps
    the best ONE per candidate.

    EVERY option goes through :func:`_apply_fixes`, the one place a fix becomes a mask — so an option
    gets the vocabulary filter, the fixed-shape rewrite, the op-outcome record, the destruction guard
    and the action label that the plain path gets, and the round appends to ``applied_log`` like any
    other. Rendering an option inline and skipping that append would hand every mask the beam accepts
    an EMPTY trail.
    """
    n = len(masks)
    byk: dict = {}
    if beam_multi:
        if merged:
            vm = v  # the merged assess+options call already carries them — no second generation
        else:
            side = vlm.conversation(ctx.proposer_system, model=gen_model, pin_first_images=True)
            side.messages = list(beam_snap)
            vm = _ptime(
                phase_times,
                f"beam_multi:r{t}",
                _ask_json_retry,
                side,
                ask + [_multi_ask(beam_multi)],
                vlm_knobs.MAX_TOKENS_AGENT,
                f"beam multi r{t}",
                schema="assess",
                kind="assess",
            )
        byk = _multi_options(vm, n)
        width = max((len(o) for o in byk.values()), default=0)
        # Option j of every candidate becomes one whole-round fix set, so a candidate that got fewer
        # options than its neighbour simply has no fix in the later sets and is carried forward.
        samples = [{k: byk[k][j] for k in byk if j < len(byk[k])} for j in range(width)] or [fixes]
    else:
        samples = [fixes]  # sample 1 = the greedy proposal already in hand
        nmore = max(0, beam_samples - 1)
        if nmore:

            def _one_sample(i):
                c = vlm.conversation(ctx.proposer_system, model=gen_model, pin_first_images=True)
                c.messages = list(beam_snap)
                out = _ask_json_retry(
                    c, ask, vlm_knobs.MAX_TOKENS_AGENT, f"beam sample r{t}#{i}", schema="assess", kind="assess"
                )
                return {k: opts[0] for k, opts in _multi_options(out, n).items()}

            # FAN-OUT #1 of 2. This is the one place the engine breaks "one beam holds one API call at
            # a time" — see the note at the selector fan-out below for exactly what that costs.
            with ThreadPoolExecutor(max_workers=min(nmore, max(1, int(cfg.beam_workers)))) as ex:
                samples.extend(ex.map(_one_sample, range(nmore)))

    # Every option applied, through the ONE renderer. `rendered[j]` is (masks, actions, outcomes) for
    # option j, each parallel to `live`.
    rendered = [_apply_fixes(masks, s, ctx, t, live=live) for s in samples]

    _jex = ctx.exemplars[: int(cfg.beam_judge_max_exemplars)] if cfg.beam_judge_max_exemplars else ctx.exemplars
    _jxb = [] if cfg.beam_judge_no_vp else ctx.judge_vp_blocks

    def _beam_judge(nm, reg) -> float:
        try:
            jv = _ptime(
                phase_times,
                "beam_select",
                judge_consensus,
                vlm,
                _jex,
                ctx.clean,
                nm,
                defect_type,
                model=beam_sel,
                definition=definition,
                samples=max(1, int(cfg.beam_judge_samples)),
                extra_blocks=_jxb + _fill_note(ctx, nm),
                defect_docs=ctx.judge_docs,
                judge_system=ctx.judge_system,
                cache_order=cfg.judge_cache_order,
                work_region=reg,
            )
            return float(jv.get("score", 0)) if isinstance(jv, dict) else -1e9
        except Exception as e:  # noqa: BLE001 — a lost score ranks last; it must not sink the beam
            log.warning(f"degraded[beam_select_failed]: round {t}: {type(e).__name__}: {e}")
            return -1e9

    new_masks, new_actions, op_outcomes = [], [], []
    # What the model's OWN top-scored option did, per candidate — the memory note's material. Read off
    # the rendered labels, so it says what ran rather than what was asked for.
    mem_labels: dict = {}
    for pos, k in enumerate(live):
        options = [(r[1][pos], r[0][pos]) for r in rendered]  # (action label, mask)
        # this candidate's work region; `k` is a current-round position, like the refine views
        _reg = ctx.siting.frame_for(k) if ctx.siting.crops else work_region
        _selfs = [_self_score(s, k) for s in samples]
        mem_labels[k] = options[max(range(len(options)), key=lambda i: _selfs[i])][0]
        if beam_sel and cfg.beam_pairwise:
            bi = _beam_pairwise_winner(
                options, vlm=vlm, ctx=ctx, defect_type=defect_type, sel=beam_sel, phase_times=phase_times
            )
        elif beam_sel and len(options) > 1:
            # FAN-OUT #2 of 2, and the reason `beam_workers` exists. It breaks "one beam holds one API
            # call at a time", the invariant `Quota.target_inflight` and the constant endpoint permit
            # were sized against. What that costs, exactly: the permit is taken PER CALL and released
            # before any backoff, and a beam waiting here holds none of its own, so total in-flight work
            # is still capped by the permit and no call can be starved by one that is waiting for it.
            # What degrades is ACCOUNTING — the launch policy's "one beam = one call" estimate, and the
            # thread-local counters behind `api_calls`/`usage`/`api_ms`, which these worker calls bump
            # on their own threads and this beam therefore under-reports. Numbers, not masks.
            with ThreadPoolExecutor(max_workers=min(max(1, int(cfg.beam_workers)), len(options))) as ex:
                scores = list(ex.map(lambda o, _r=_reg: _beam_judge(o[1], _r), options))
            bi = max(range(len(options)), key=lambda i: scores[i])
        else:
            # No selector configured at all (`beam_select` unset AND `judge_model` empty), or a single
            # option to choose from: keep the first, which is the greedy proposal.
            bi = 0
        new_masks.append(options[bi][1])
        new_actions.append(f"beam[{options[bi][0]}]")
        # Only the WINNING option's outcomes reach the agent: reporting the refusals of options that
        # were discarded would ask it to correct edits its candidate never received.
        op_outcomes.extend(o for o in rendered[bi][2] if o[0] == k)
    return new_masks, new_actions, op_outcomes, mem_labels


def run_beam(
    clean_path,
    defect_type,
    *,
    vlm,
    visual_prompt,
    defects,
    cfg,
    exemplars,
    seed_masks=None,
    profile=False,
    prompts=None,
    covered=None,
    target_region=None,
    regions=None,
) -> RunResult:
    """Beam-search mask generation for one (clean image, defect type).

    See the module docstring for the loop; the ``[:beam_width]`` slice at the final judge is all
    ``beam_width`` controls in this engine, and a mask is accepted iff ``AcceptBar.clears`` says so.
    Returns every passing mask (empty when none pass — the caller retries to fill) as
    :class:`AcceptedMask` objects inside a :class:`RunResult`. ``exemplars`` is supplied by the caller
    (the engine loads no data). Set ``seed_masks`` to warm-start from prior accepted masks (PROPOSE
    skipped). Set ``profile`` to fill :attr:`RunResult.phase_times` with the seconds spent in each
    VLM call site; it is ``None`` otherwise.
    Pass ``prompts`` (a :class:`~anomalygen.agentic_mask_generation.prompts.PromptSet`) to override the
    prompt text for a quick test; by default it resolves ``cfg.prompt_set`` via
    :func:`~anomalygen.agentic_mask_generation.prompts.by_name`.
    """
    _calls0 = get_call_count()
    _api_ms0 = get_api_ms()  # API vs non-API split: wall - (summed HTTP ms) = this beam's local CPU cost
    _wall0 = time.perf_counter()
    # Per-run token/cache usage = this THREAD's delta. The counters are thread-local, so the delta is
    # complete only because every call in a beam runs on the beam's own thread (judging is serial).
    # THE ONE EXCEPTION is the in-loop beam (`beam_multi`/`beam_samples`, both off by default), which
    # fans work out over `beam_workers` threads; those calls bump the WORKER's counters and this delta
    # does not see them. Under-count, never over-count — see the note at the fan-out below.
    _usage0 = get_usage()
    phase_times: dict | None = {} if profile else None
    # The clock starts BEFORE `build`: its local cost is exactly what `non_api_ms` is defined as, and
    # it makes no model call, so it does not touch `api_calls`, `usage` or `api_ms`.
    ctx = BeamContext.build(
        clean_path,
        defect_type,
        visual_prompt=visual_prompt,
        defects=defects,
        cfg=cfg,
        exemplars=exemplars,
        prompts=prompts,
        covered=covered,
        target_region=target_region,
        regions=regions,
    )
    # MODEL TIERING: a fast gen_model drives generation AND the in-loop guidance judge (which has no
    # model of its own); acceptance is decided on judge_model.
    gen_model = cfg.gen_model
    judge_model = cfg.judge_model
    beam = cfg.beam_width
    max_rounds = cfg.max_refine_rounds
    final_judge_samples = cfg.final_judge_samples
    board = Path(clean_path).stem
    # THE IN-LOOP BEAM, read once. `beam_multi` is a COUNT of options per candidate and `beam_samples`
    # a count of whole re-samples of the turn; either above its off value arms the branch below.
    beam_multi = int(cfg.beam_multi or 0)
    beam_samples = int(cfg.beam_samples or 1)
    # "" means the beam's own judge. Resolved to a model HERE, once: the selector keeps whichever
    # option it scores highest, and an empty selector would keep candidate #0 unjudged.
    beam_sel = cfg.beam_select or judge_model
    # The acceptance decision, built once — the threshold and the gate-only dimensions the bar keeps
    # out of its mean; every accept test below is this object.
    # both paths into the bar score the same dimensions.
    bar = AcceptBar(thr=cfg.accept_threshold, gate_only=gate_only_dims(ctx.flags))
    # Shared by the FINAL judge and (under accept_continue) the banking in-loop judge below.
    adaptive_fj = AdaptiveStop(bar)
    conv = vlm.conversation(ctx.proposer_system, model=gen_model, pin_first_images=True)

    # ---- Localize the work region, so propose gets a MAGNIFIED view of where the defect goes ----
    # (a crop alone loses the relational context bridging needs, so the global view is kept too).
    work_zoom, wz_box = ctx.siting.primary()  # an assigned region IS the work region — no localize call
    if not ctx.siting.regions:
        # Built OUTSIDE the try: the handler below is for a failed CALL, and a beam whose own views
        # cannot be built would be misbooked as `localize_failed` and raise at `propose_context`
        # anyway.
        loc_blocks = ctx.localize_blocks()
        try:
            loc = _ptime(
                phase_times,
                "localize",
                vlm.call_json,
                ctx.proposer_system,
                loc_blocks,
                model=gen_model,
                max_tokens=MAX_TOKENS_ROUTE,
                schema="localize",
            )
            wr = loc.get("work_region", {})
            work_zoom, wz_box = ctx.views.zoom_to_box(
                ctx.clean, (wr.get("x0"), wr.get("y0"), wr.get("x1"), wr.get("y1"))
            )
            if work_zoom is None:
                # zoom_to_box RETURNS (None, None) on a degenerate box rather than raising, so a
                # reply with no usable work_region never reaches the handler below — without this
                # line the beam proposes against the full image with nothing said.
                log.warning(f"degraded[localize_empty]: {Path(clean_path).name}: no usable work_region")
            else:
                work_zoom = ctx.views.grid_fullcoords(work_zoom, wz_box)
        except Exception as e:  # noqa: BLE001 — a lost work zoom must not sink the beam, but SAY so
            # LABELLED and countable, a sibling of `localize_empty`: both end with the beam proposing
            # against the full image, and a bug in the two lines above would otherwise be
            # indistinguishable from a timed-out call.
            # a beam that lost its work zoom says it proposed against the whole board.
            log.warning(
                f"degraded[localize_failed]: [{defect_type}] localize failed on "
                f"{Path(clean_path).name} ({type(e).__name__}: {e}) — proposing against the "
                f"full image instead"
            )
            work_zoom, wz_box = None, None

    # ---- Propose K diverse candidates ----
    # The context (examples + clean + visual prompt + card + crops) is kept apart from the instruction
    # because the warm start replays exactly it into the first refine turn.
    context_blocks = ctx.propose_context(work_zoom)
    blocks = context_blocks + ctx.propose_ask()

    seeding = seed_masks is not None
    masks, definition = _initial_candidates(seed_masks, ctx=ctx, conv=conv, blocks=blocks, phase_times=phase_times)
    # WARM START: keep the ORIGINAL accepted seeds so final selection cannot discard them. They are
    # exactly the masks the beam starts from, read before the first round edits any of them.
    _seed_orig = list(masks) if seeding else []

    # What the engine APPLIED per round, per candidate — the other half of what the trace holds, which
    # is what the model ASKED for.
    applied_log: list[dict] = []
    last_area: list[int] = []  # per-candidate area BEFORE the current round's edit
    # (overall, candidate index, sub-scores) — NO MASK in it: the mask is resolved from `masks` after
    # the loop. a round that gives up judges the mask its own trail describes.
    last_round_scored: list[tuple] = []
    judge_note = ""  # authoritative judge verdict on last round's best, fed into the NEXT round
    judge_bk, judge_steer = None, ""  # its pieces, so a lever's renumbering can rebuild the label
    # What the OPS said about the edits they were asked for — left to infer a refusal from an
    # unchanged pixel count, the agent reads it as "too timid".
    op_outcomes: list = []
    judged_pool: list[tuple] = []  # GLOBAL keep-best: every round's authoritatively-judged best
    review_blocks: list[Any] = []
    if seeding:  # no PROPOSE call happened -> replay the full context into the FIRST refine turn
        review_blocks = list(context_blocks) + [seed_instruction()]
    mem_note = ""  # the in-loop beam's cross-round memory; "" on every other path
    # CANDIDATE IDENTITY. `slots[i]` is the stable id of `masks[i]` and `slot_log[r]` names the
    # candidate each action in `applied_log[r]` belongs to. Position IS identity until `beam_prune` or
    # `accept_continue` drops a candidate mid-loop; `_dropped_any` stays False until one does, and
    # `_final_selection` then reads the trail exactly as it did before either lever existed.
    slots = list(range(len(masks)))
    slot_log: list[list[int]] = []
    banked: list[tuple] = []  # accept_continue's output — (mask, score, subs, reasons, slot)
    _dropped_any = False

    for t in range(max_rounds):
        last = t == max_rounds - 1
        ask = _refine_ask(
            ctx,
            masks,
            t=t,
            last=last,
            review_blocks=review_blocks,
            last_area=last_area,
            judge_note=judge_note,
            op_outcomes=op_outcomes,
            board=board,
            mem_note=mem_note,
            work_region=wz_box,
        )
        # Consumed by the turn just built, all four; left standing, each would be sent again next
        # round as if it were new.
        review_blocks, judge_note, op_outcomes, mem_note = [], "", [], ""
        # ---- THE IN-LOOP BEAM (off by default) ----
        # Snapshot the conversation BEFORE the ask, so every re-sample asks the SAME round-t prompt
        # instead of one that has already been answered once.
        beam_on = bool(beam_multi or beam_samples > 1) and not last
        beam_snap = list(conv.messages) if beam_on else None
        # MERGED: a MULTI reply already carries `assess`, so a separate assess call would re-ask for
        # it. The merged ask runs on a SIDE conversation so the verbose K-option reply never
        # accumulates in the main conv — accumulated, it would grow every later round's prompt. The
        # main conv stays frozen at propose; the masks are re-shown fresh in `ask` every round anyway.
        # `beam_legacy_assess` keeps the separate assess call as well.
        merged = beam_on and bool(beam_multi) and not cfg.beam_legacy_assess
        if merged:
            side = vlm.conversation(ctx.proposer_system, model=gen_model, pin_first_images=True)
            side.messages = list(beam_snap or [])
            v = _ptime(
                phase_times,
                f"beam_merged:r{t}",
                _ask_json_retry,
                side,
                ask + [_multi_ask(beam_multi)],
                vlm_knobs.MAX_TOKENS_AGENT,
                f"round {t} merged",
                # The `assess` schema constrains `assess` only, so the MULTI `fixes` shape (an
                # `options` list per candidate) validates unchanged — the merged call is checked
                # exactly as tightly as the plain one.
                schema="assess",
                kind="assess",
            )
        else:
            v = _ptime(
                phase_times,
                f"refine_assess:r{t}",
                _ask_json_retry,
                conv,
                ask,
                vlm_knobs.MAX_TOKENS_AGENT,
                f"round {t} assess",
                schema="assess",
                kind="assess",
            )
        if v is None:
            break  # both attempts failed — logged above; keep whatever beam we have so far
        assess = v.get("assess") or []
        if not isinstance(assess, list):
            # A dict is TRUTHY, so `not assess` never saw this one, and iterating a dict yields its
            # str keys — every isinstance(x, dict) test below fails and every candidate scores 0.
            log.warning(f"degraded[assess_missing]: round {t}: assess came back as {type(assess).__name__}, not a list")
            assess = []
        elif not assess:
            # Every candidate then scores 0 and the round's "best" is decided by sort-tie order —
            # the search silently becomes random on the arm being measured.
            log.warning(f"degraded[assess_missing]: round {t}: the reply carried no assess")
        _ax = _assess_index(assess, len(masks))
        scored = []
        for k in range(len(masks)):
            a = _ax.get(k, {})
            subs = {kk: round(_score_num(a.get(kk, 0))) for kk in DIMS}
            scored.append((_avg4(a), k, subs))
        scored.sort(key=lambda x: -x[0])
        last_round_scored = scored
        # PRUNE (cfg.beam_prune): only the top-`beam` survivors go into the next round's refine — the
        # same slice the final stage already selects, so nothing that COULD be accepted is lost, and
        # the round after this one generates and in-loop-judges over fewer candidates. `None` = keep
        # every candidate, which is the default.
        keep_set = {k for _sc, k, _su in scored[:beam]} if cfg.beam_prune else None
        bo, bk, bsubs = scored[0]
        bm = masks[bk]

        # JUDGE-IN-THE-LOOP: an authoritative verdict on this round's best, recorded AND fed into the
        # next round so refinement steers on the judge's weakest dimension rather than the self-score.
        # Skipped on the last round: its verdict only steers the NEXT round, and its target — this
        # round's best — is in the `[:beam]` slice the final stage judges.
        rec_subs, rec_score, rec_reasons, rfb = bsubs, bo, {}, ""
        jv = None  # the authoritative in-loop verdict; stays None on a skip or a failure — gates accept_continue
        if not last:
            try:
                jv = _ptime(
                    phase_times,
                    f"inloop_judge:r{t}:c{bk}",
                    judge_consensus,
                    vlm,
                    ctx.exemplars,
                    ctx.clean,
                    bm,
                    defect_type,
                    # A verdict with BANKING AUTHORITY (accept_continue) is the final judge
                    # called early — same model, same consensus samples, same adaptive stop,
                    # inherited from the final judge's own settings rather than restated. A verdict
                    # that only STEERS stays on the gen side at 1 sample: the mask it scores still
                    # faces the real final judge, so consensus precision buys nothing there.
                    model=(judge_model if cfg.accept_continue else gen_model),
                    definition=definition,
                    samples=(final_judge_samples if cfg.accept_continue else 1),
                    adaptive=(adaptive_fj if cfg.accept_continue else None),
                    extra_blocks=ctx.judge_vp_blocks + _fill_note(ctx, bm),
                    defect_docs=ctx.judge_docs,
                    judge_system=ctx.judge_system,
                    cache_order=cfg.judge_cache_order,
                    # `bk` is a current-round position, like the refine views this verdict steers
                    work_region=(ctx.siting.frame_for(bk) if ctx.siting.crops else wz_box),
                )
            except Exception as e:  # noqa: BLE001 — a lost verdict must not sink the beam, but SAY so
                # A lost in-loop verdict costs the round its steer and its pool entry — a distinct
                # kind from `final_judge_failed` because the two losses cost different things.
                # an in loop judge that fails is recorded.
                log.warning(f"degraded[inloop_judge_failed]: round {t}: {type(e).__name__}: {e}")
                jv = None
            if jv is not None:
                # Rebuilt over DIMS, like the final judge's: a mean over a different dimension set is
                # not comparable to the threshold it is tested against.
                # both paths into the bar score the same dimensions.
                rec_subs = {kk: int(jv.get(kk, 0)) for kk in DIMS}
                rec_score = int(jv["score"])
                rec_reasons = jv.get("reasons") or {}
                rfb = jv.get("feedback", "")
                judged_pool.append((bm, rec_score, rec_subs, rec_reasons))  # global keep-best
                weak = sorted(DIMS, key=lambda k: rec_subs.get(k, 0))[:2]
                judge_bk, judge_steer = bk, ctx.prompts.weak_dim_steer(rec_subs, weak, "propose", rfb)
                judge_note = (
                    f"INDEPENDENT JUDGE (authoritative) on your current BEST candidate C{judge_bk}: {judge_steer}"
                )
        # ACCEPT-AND-CONTINUE (cfg.accept_continue): the in-loop judge already cleared the bar on this
        # round's best, so BANK it now — it is emitted as passed without a final judge — and drop it
        # from the beam so later rounds refine only the not-yet-good candidates. Gated on `jv is not
        # None`, i.e. only when that judge ACTUALLY RAN this round: a self-assessment must never
        # accept. The same `bar` every other accept test uses, and nothing else: the deterministic
        # veto is prompt text, not a second gate here.
        acc_idx = None
        if cfg.accept_continue and jv is not None and bar.clears(rec_subs):
            banked.append((bm, rec_score, rec_subs, rec_reasons, slots[bk]))
            acc_idx = bk
        if last:
            break
        # ---- REFINE. By default EVERY candidate, in this one response (no pruning — salvage the weak
        # ones); `beam_prune`/`accept_continue` narrow that to `live` below. ----
        # A MERGED reply's `fixes` are the MULTI options wrapper, parsed in the beam branch — the
        # greedy parse here would read each `{"id":k,"options":[...]}` as a fix with no ops and turn
        # every candidate into a `keep`.
        fixes = {}
        if not merged:
            for f in v.get("fixes", []) or []:
                if not isinstance(f, dict):
                    continue
                _fid = _fix_target(f, len(masks))  # an unplaceable fix skips ONE fix, never the run
                if _fid is not None:
                    fixes[_fid] = f
            if not fixes:
                # ROUTINE, not a failure — it is how a model says "no change". Counted because the RATE
                # is the finding: a refine reply that edits nothing is recorded.
                log.warning(f"degraded[fixes_no_op]: round {t}: no placeable fix — every candidate carried forward")

        # `live` is the candidates this round actually refines, in list order. Both levers express
        # "drop from the beam" HERE, in one guard, so nothing downstream has to know which one fired.
        live = [k for k in range(len(masks)) if k != acc_idx and (keep_set is None or k in keep_set)]
        # `None` when nothing was dropped, so the default path takes `_apply_fixes`'s original
        # every-candidate loop rather than an equivalent-looking list of every index.
        _live_arg = None if len(live) == len(masks) else live
        if _live_arg is not None:
            _dropped_any = True
        if not live:
            # NOTHING LEFT TO REFINE — the levers have taken every candidate. `beam_prune` narrows
            # the beam to `beam_width` and `accept_continue` banks its best, so a beam already down
            # to `beam_width` loses its LAST candidate the round that best clears the bar; a
            # `beam_width` of 0 empties it outright.
            #
            # THE RULE THIS KEEPS: `masks` is never empty. `_initial_candidates` guarantees it at the
            # start and this is the only place a round can take it away, so every stage downstream —
            # `_assess_index`'s id range, `scored[0]`, `_final_selection`'s `masks[k]` — may go on
            # assuming a candidate exists. Without it the next round asked a model to assess an empty
            # list, booked every reply as `assess_unplaceable: id 0 outside 0..-1`, and raised
            # IndexError on `scored[0]`; the driver's repeated-failure guard then abandoned the whole
            # defect type after five such beams.
            #
            # BREAKING HERE, not at the top of the next round, is what keeps the trail readable:
            # `masks`, `slots` and `last_round_scored` are still this round's and still agree, so the
            # final stage judges the candidates the beam actually holds. The banked masks are already
            # in `banked` and are this beam's real output.
            # a beam that banks every candidate finishes and delivers.
            break

        if beam_on and beam_snap is not None:
            new_masks, new_actions, op_outcomes, mem_labels = _beam_round(
                masks,
                fixes,
                ctx=ctx,
                t=t,
                live=live,
                ask=ask,
                v=v,
                merged=merged,
                beam_snap=beam_snap,
                vlm=vlm,
                cfg=cfg,
                gen_model=gen_model,
                beam_multi=beam_multi,
                beam_samples=beam_samples,
                beam_sel=beam_sel,
                defect_type=defect_type,
                definition=definition,
                phase_times=phase_times,
                work_region=wz_box,
            )
            if merged and not cfg.beam_no_memory:
                # The merged call ran on a side conversation, so the main conv never saw this round's
                # own scores; without this the beam loses the round-to-round memory the plain path
                # gets free by keeping its reply in history.
                mem_note = _beam_memory_note(scored, _ax, mem_labels, live=live)
        else:
            new_masks, new_actions, op_outcomes = _apply_fixes(masks, fixes, ctx, t, live=_live_arg)
        # What each edit did to its candidate's area: an agent that cannot see the size of its own
        # last edit cannot tell a 10 px stroke from a 200 px one, and will repeat the mistake.
        last_area = [int(np.count_nonzero(np.asarray(masks[k]) > 127)) for k in live]
        # No per-round emptiness record: `_not_destroyed` cannot let a fix empty a candidate that had
        # pixels, so emptiness at round >= 1 is inherited and is recorded at its source.
        # a blank warm start is on the record.
        applied_log.append({"round": t, "actions": list(new_actions)})
        # WHOSE action each entry is. Only consulted once a lever has actually dropped somebody —
        # until then position is identity and the trail reads exactly as it always did.
        slot_log.append([slots[k] for k in live])
        # Stage every surviving candidate under this round, so a finished mask can be replayed edit
        # by edit against `actions`. Labelled by SLOT — the same id `slot_log` and the trail name —
        # not by its position in `new_masks`, which `live` has renumbered once a lever dropped one.
        for _k, _m in enumerate(new_masks):
            stage_trace_mask(f"r{t}_c{slots[live[_k]]}_after", _m, vlm)
        masks = new_masks
        slots = [slots[k] for k in live]
        if _live_arg is not None:
            # RENUMBERED, so the ids in hand no longer name what they did. `last_round_scored` holds
            # THIS round's ids — positions in the list just replaced — and the loop can still leave
            # without scoring again: an assess call that gives up `break`s, and `_final_selection`
            # then resolves `masks[k]` and `slots[k]` from exactly these ids. Carried over unchanged
            # they read the wrong candidate, or run off the end of the shorter list (IndexError).
            # Dropping the entries that did not survive is the levers' own contract — a pruned or
            # banked candidate is not eligible for the final judge, and a banked one is in `banked`.
            # a pruned beam that gives up judges the candidates it still has.
            _pos = {k: i for i, k in enumerate(live)}
            last_round_scored = [(_sc, _pos[_k], _su) for _sc, _k, _su in last_round_scored if _k in _pos]
            # The notes the NEXT round reads name candidates by THIS round's positions, but they sit
            # beside a montage relabelled by the new ones — remapped here, exactly like
            # `last_round_scored` above (`mem_note` is built through `live` already). A best that a
            # lever took out of the beam keeps its steer but loses its label: C{old} would point at a
            # different survivor, or off the montage entirely.
            op_outcomes = [(_pos[_c], _ki, _re) for _c, _ki, _re in op_outcomes if _c in _pos]
            if judge_note:
                _lbl = (
                    f"your current BEST candidate C{_pos[judge_bk]}"
                    if judge_bk in _pos
                    else "last round's BEST candidate (no longer in the beam)"
                )
                judge_note = f"INDEPENDENT JUDGE (authoritative) on {_lbl}: {judge_steer}"

    # ---- Final: judge the last round's top-`beam` candidates (deduped) and return EVERY one that
    # clears the bar, not capped at `beam`. This `[:beam]` slice is `beam_width`'s only use HERE — the
    # driver reads it again as its masks-per-beam yield prior. ----

    def _final_judge(cm):
        selfsc, m, _slot = cm
        try:
            verdict = _ptime(
                phase_times,
                "final_judge",
                judge_consensus,
                vlm,
                ctx.exemplars,
                ctx.clean,
                m,
                defect_type,
                model=judge_model,
                definition=definition,
                samples=final_judge_samples,
                adaptive=adaptive_fj,
                extra_blocks=ctx.judge_vp_blocks + _fill_note(ctx, m),
                defect_docs=ctx.judge_docs,
                judge_system=ctx.judge_system,
                cache_order=cfg.judge_cache_order,
                # `_slot` is the stable id = the candidate index the crop assignment was made
                # by; a keep-best pool entry (None) or a warm-start seed (-1) has no slot in this
                # beam and takes the beam-level box.
                work_region=(
                    ctx.siting.frame_for(_slot)
                    if (ctx.siting.crops and isinstance(_slot, int) and _slot >= 0)
                    else wz_box
                ),
            )
            return (
                m,
                int(verdict["score"]),
                {kk: int(verdict.get(kk, 0)) for kk in DIMS},
                verdict.get("reasons") or {},
                verdict.get("feedback", ""),
                _slot,
                False,
            )
        except Exception as e:  # noqa: BLE001
            # UNJUDGED, not rejected: empty dims fail the bar, so a rate-limited judge would
            # otherwise read as a judge that looked and said no.
            # a candidate the final judge never answered on is booked as a lost verdict.
            log.warning(
                f"degraded[final_judge_failed]: [{defect_type}] final judge failed on a "
                f"candidate ({type(e).__name__}: {e}) — counting it as UNJUDGED, not as a "
                f"rejection"
            )
            return (m, selfsc, {}, {}, "", _slot, True)

    masks_out, unjudged, n_judged = _final_selection(
        _final_judge,
        ctx,
        masks=masks,
        scored=last_round_scored,
        pool=judged_pool,
        seed_orig=_seed_orig,
        applied_log=applied_log,
        bar=bar,
        beam=beam,
        board=board,
        # Only handed over once a lever has ACTUALLY dropped a candidate. Until then position is
        # identity and `_final_selection` takes the positional trail read it always had, so no
        # default-path behaviour depends on this bookkeeping existing.
        slots=slots if _dropped_any else None,
        slot_log=slot_log if _dropped_any else None,
        banked=banked,
    )
    _usage1 = get_usage()
    _wall_ms = round((time.perf_counter() - _wall0) * 1000, 1)
    _api_ms = round(get_api_ms() - _api_ms0, 1)
    return RunResult(
        masks=masks_out,
        api_calls=get_call_count() - _calls0,
        unjudged=unjudged,
        judged=n_judged,
        usage={k: _usage1[k] - _usage0.get(k, 0) for k in USAGE_KEYS},
        phase_times=({k: round(v, 1) for k, v in phase_times.items()} if phase_times is not None else None),
        trace=applied_log,
        api_ms=_api_ms,
        non_api_ms=round(_wall_ms - _api_ms, 1),
        wall_ms=_wall_ms,
    )
