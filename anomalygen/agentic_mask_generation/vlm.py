# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin VLM transport: text + images in, JSON out (with retry).

``httpx`` directly, no vendor SDK. EXACTLY TWO ROUTING LAYERS, and they answer different questions:

- **The base picks the envelope** (`_route_for`). ``ANTHROPIC_BASE`` — the public Anthropic API,
  the only base constant in the code — serves exactly one protocol, the native ``/messages`` wire
  format with ``x-api-key`` + ``anthropic-version`` auth, so every model on it takes that route.
  ANY other base is an OpenAI-compatible endpoint and is spoken to at ``/chat/completions`` with a
  Bearer header, whatever the model on it is called.
- **The model name picks the cache markers** (`_is_claude`). A claude-named model carries
  ``cache_control`` breakpoints — natively on ``/messages``, and embedded in OpenAI block form on
  chat (the LiteLLM/OpenRouter gateway convention, which forwards them to Anthropic backends).
  Every other model gets the clean payload, byte-identical to a plain OpenAI request.

The route suffix is appended to a ``base_url`` that already carries its version segment
(``.../v1``), exactly like the OpenAI SDK's ``base_url``. Nothing is read from the environment — it
all comes in through ``Vlm(...)``.

Call/usage counters are THREAD-LOCAL: a run's ``get_call_count()`` / ``get_usage()`` delta is
complete only because every API call a beam makes happens on the beam's own thread.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.agentic_mask_generation.utils import atomic_write

# Per-call answer ceilings, sized by how long the answer can legitimately be. `max_tokens` is a
# CEILING billed per token actually generated, so headroom costs nothing when unused, while an answer
# cut off mid-JSON is unparseable and wastes the entire call, images included.
MAX_TOKENS_ROUTE = 2000  # a two-field classification — and the localize box, which shares it
MAX_TOKENS_REGIONS = 8000  # a list of boxes, one per valid site on the board
MAX_TOKENS_CAPTION = 4000  # a structured description of one real defect image
MAX_TOKENS_CARD = 8000  # the synthesized defect card (7 sections)
MAX_TOKENS_JUDGE = 12000  # six dimensions, each with its own written reason
# A request must fit prompt + max_tokens inside the SERVING window. 8000 is a FIXED constant like
# every other per-call budget here — no env, no registry field — and it is field-validated on both
# sides: the largest measured agent reply (beam-multi's K-plans-in-one-call included) is ~1.7k
# tokens, 4.7x headroom, measured on cloud Opus and local Gemma. On the shipped 32k local serving
# window it leaves 24.7k for the prompt against a measured max of ~23.4k — tight; a deployment
# whose prompts outgrow that serves a LARGER window (MAXLEN), it does not shrink this budget.
MAX_TOKENS_AGENT = 8000  # a plan plus a full primitive spec for every candidate


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _keep_constant_as_text(token: str) -> str:
    """`NaN` / `Infinity` / `-Infinity` arrive as TEXT, not as floats.

    They are Python's extension to JSON and `json` accepts them by default, which puts a value through
    the schema that the schema CANNOT reject: `minimum`/`maximum` are `<`/`>` comparisons and every
    comparison against NaN is False, so `{"x0": NaN, "y0": 0, "x1": 1, "y1": 1}` satisfies `regions`
    and `enumerate_regions` caches it to disk — a bad board replayed on every later run.

    Kept as the token string rather than raised on, so the reply fails as a number that is not a
    number and takes the retry every other schema failure takes.
    """
    return token


# One decoder for every parse in this module, so no path can quietly accept what another rejects.
_DECODER = json.JSONDecoder(parse_constant=_keep_constant_as_text)


def _img_label(caption: str) -> str:
    """Short filesystem slug from an image's preceding caption text, so trace images are self-describing
    (e.g. 'CLEAN image (coordinate grid):' -> 'clean_image_coordinate_grid', 'C0 zoom:' -> 'c0_zoom')."""
    return re.sub(r"[^a-z0-9]+", "_", caption.lower()).strip("_")[:44].strip("_")


# --- per-thread accounting (module-level; not config) ---
_local = threading.local()
USAGE_KEYS = ("in_tok", "out_tok", "cache_w", "cache_r")
CALL_LOG: list[dict] = []  # per-process timing probe: {kind, ms, chars, ...} in call order
_CALL_LOG_MAX = 5000


def get_call_count() -> int:
    return getattr(_local, "count", 0)


def get_failed_count() -> dict:
    """{failure kind: attempts}. Only `transport` is even a candidate for having cost anything."""
    return dict(getattr(_local, "failed", None) or {})


def _bump_failed(kind: str) -> None:
    """Attempts that did not answer, BY KIND — because the kinds do not cost the same. A `transport`
    timeout is the only one where the model may have generated the output and the client simply
    stopped waiting; whether that is billed is the provider's policy."""
    d = getattr(_local, "failed", None)
    if d is None:
        d = _local.failed = {}
    d[kind] = d.get(kind, 0) + 1


def _bump() -> None:
    _local.count = getattr(_local, "count", 0) + 1


def get_api_ms() -> float:
    """THIS thread's accumulated HTTP round-trip ms. A run takes a before/after delta and subtracts it
    from its wall time, splitting the run into API time and everything local (image encoding, view
    building, CV, JSON parse, plus GIL wait under concurrency)."""
    return getattr(_local, "api_ms", 0.0)


def _acc_api_ms(ms: float) -> None:
    _local.api_ms = getattr(_local, "api_ms", 0.0) + (ms or 0.0)


def set_trace_ctx(ctx: str) -> None:
    """Tag this thread's ``--trace-dir`` entries with a beam/run id, so a PARALLEL trace groups each
    beam's calls under ``<trace_dir>/<ctx>/``. Judging is serial, so every call in a beam runs on the
    beam's own thread and setting this once per beam is enough (no propagation)."""
    _local.trace_ctx = ctx


def stage_trace_mask(name: str, mask: "Image.Image", vlm: "Vlm") -> None:
    """Stage a candidate BINARY mask to be dumped into the NEXT traced call's dir — content-addressed
    and symlinked like the sent images, so downstream analysis reads the real mask a judge scored
    rather than guessing it off the red overlay the model was shown. `vlm` is the instance whose
    upcoming call on this thread claims the buffer: a no-op unless it runs with --trace-dir, so an
    untraced Vlm never buffers masks, and a fresh worker thread that stages before its first call
    still traces."""
    if mask is None or not getattr(vlm, "_trace_dir", None):
        return
    buf = getattr(_local, "pending_masks", None)
    if buf is None:
        buf = _local.pending_masks = []
    buf.append((name, mask))


def _acc_usage(in_tok, out_tok, cache_w, cache_r) -> None:
    u = getattr(_local, "usage", None)
    if u is None:
        u = dict.fromkeys(USAGE_KEYS, 0)
        _local.usage = u
    for k, v in zip(USAGE_KEYS, (in_tok, out_tok, cache_w, cache_r)):
        if v:
            u[k] += int(v)


def get_usage() -> dict:
    """Snapshot (copy) of THIS thread's accumulated usage. Callers take before/after deltas."""
    u = getattr(_local, "usage", None)
    return dict(u) if u else dict.fromkeys(USAGE_KEYS, 0)


def discard_staged_masks() -> None:
    """Drop anything staged on THIS thread that no call will ever send.

    PUBLIC because the driver needs it: a beam that ends without a final-judge call leaves its last
    round's candidates pending on a pooled thread, and the next beam files them under its own call.
    """
    _local.pending_masks, _local.raw_io = [], None


def _log_call(entry: dict) -> None:
    CALL_LOG.append(entry)
    if len(CALL_LOG) > _CALL_LOG_MAX:
        del CALL_LOG[: len(CALL_LOG) - _CALL_LOG_MAX]


# A COMPLETE JSON string, escape-aware. Scanned left to right, whatever quote follows the last of
# these opens a string the reply never closed. (Counting quotes cannot do this: `"he said \"go\""`
# holds four, two of them escaped.)
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
# A numeric literal running to the very end of the fragment. JSON gives a number no terminator, so
# nothing here says whether `1` is the value or the first digit of `10`.
_TRAILING_NUMBER = re.compile(r"-?\d[\d.eE+-]*$|-$")
# The `"key":` that introduces the value at the end, with the comma that separates it from the field
# before — so a dropped value takes its key with it and the field is ABSENT rather than wrong.
_KEY_BEFORE_END = re.compile(r',?\s*"(?:[^"\\]|\\.)*"\s*:\s*$')


def _dangling_value(frag: str) -> int:
    """Where the value the reply was cut INSIDE starts, or -1 when it does not end inside one."""
    end = 0
    for m in _JSON_STRING.finditer(frag):
        end = m.end()
    q = frag.find('"', end)  # a quote past the last complete string opens an unclosed one
    if q >= 0:
        return q
    m = _TRAILING_NUMBER.search(frag)
    return m.start() if m else -1


def _missing_closers(frag: str) -> str:
    """The closers `frag` still owes: an open string is closed first (the caller has already cut a
    string left dangling mid-escape), then every open `{`/`[`, innermost first. One escape-aware
    scan, because brackets inside string values are text, not structure (counting cannot tell these
    apart: a reason mentioning `}` pays for a closer the object is still owed)."""
    stack: list[str] = []
    in_str = escaped = False
    for ch in frag:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    return ('"' if in_str else "") + "".join(reversed(stack))


def _repair_truncated_json(frag: str) -> tuple[str, str]:
    """Best-effort salvage of a JSON object cut off by max_tokens, and WHAT the salvage dropped.

    Drops the value the reply was cut inside (with its key), then a dangling comma, then balances
    braces/brackets, so a truncated judge/beam reply still yields the fields that arrived whole.

    A VALUE CUT MID-TOKEN IS NOT A VALUE, which is why the KEY goes with it. A verdict cut one
    character into ``"semantic": 10`` repairs to the valid, different number 1 — schema-valid, all
    six dimensions present, averaged into the consensus as a real sample. Dropped, the dimension is
    MISSING and `_parse_judge_out` re-asks instead. Same for a string: a prefix of an op name
    (`erase_polygon` -> `erase`) is another op.

    A truncated ``true``/``false``/``null`` needs nothing here: no prefix of one is a JSON value, so
    the decode this feeds raises and the reply is re-asked.
    """
    frag = frag.rstrip()
    dropped = ""
    cut = _dangling_value(frag)
    if cut >= 0:
        key = _KEY_BEFORE_END.search(frag[:cut])
        cut = key.start() if key else cut
        dropped = frag[cut:].lstrip(", ")[:80]
        frag = frag[:cut]
    frag = frag.rstrip().rstrip(",")
    return frag + _missing_closers(frag), dropped


# --------------------------------------------------------------------------------------------------
# REPLY SHAPES — what each kind of call is allowed to come back as. The retry trigger is "did not
# match the declared shape", not "did not parse", so a reply that parses but is missing the field the
# engine reads is re-asked rather than becoming a default in the loop.
#
# The constraints are deliberately loose: an over-strict rule costs real money and says nothing while
# it does, so each one only pins what the engine actually reads. Rendering catches what a schema
# cannot (three identical polygon points are schema-valid and draw nothing), and the judge catches
# semantics.
_NUM01 = {"type": "number", "minimum": 0, "maximum": 1}
_SCORE = {"type": "integer", "minimum": 0, "maximum": 10}
_BOX = {
    "type": "object",
    "required": ["x0", "y0", "x1", "y1"],
    "properties": {k: _NUM01 for k in ("x0", "y0", "x1", "y1")},
}
# Regions only: weak VLMs answer the normalized ask in their trained 0-1000 (or pixel) frame; the
# frame is repaired in enumerate_regions, so here the box just has to be numeric and non-negative.
_BOX_ANYFRAME = {
    "type": "object",
    "required": ["x0", "y0", "x1", "y1"],
    "properties": {k: {"type": "number", "minimum": 0} for k in ("x0", "y0", "x1", "y1")},
}

REPLY_SCHEMAS: dict[str, dict] = {
    # The DRIVER's pre-run capability probe, and the only entry here no engine boundary asks for. It
    # is a schema rather than a `.get` in the driver because that is what makes the one retry
    # `call_json` owns apply to it: a model whose first reply wraps the answer in prose gets the same
    # second ask every other boundary gets, so the probe fails on the capability and not on a habit.
    # `integer`, not `number`: a count answered as 2.0 is a model formatting a count, and "an integer"
    # is what the ask says, so the reply is re-asked rather than rounded.
    "smoke": {
        "type": "object",
        "required": ["n_images"],
        "properties": {"n_images": {"type": "integer", "minimum": 0}},
    },
    # Region enumeration writes its answer to a CONTENT-KEYED DISK CACHE, so an unchecked box is not a
    # bad beam, it is a bad board replayed on every later run.
    "regions": {
        "type": "object",
        "required": ["regions"],
        "properties": {"regions": {"type": "array", "items": _BOX_ANYFRAME}},
    },
    "localize": {"type": "object", "required": ["work_region"], "properties": {"work_region": _BOX}},
    # All six dimensions required, `distribution` included: `_parse_judge_out` raises without it, and
    # a schema that permits what the parser refuses turns a retryable reply into a lost sample.
    "judge": {
        "type": "object",
        "required": ["roi", "placement", "shape", "fill", "distribution", "semantic"],
        "properties": {d: _SCORE for d in ("roi", "placement", "shape", "fill", "distribution", "semantic")},
    },
    # `ops` typed: the loop iterates it, so a candidate whose `ops` is null is a shape miss to re-ask,
    # not a reply for the loop to trip over.
    "propose": {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "object", "required": ["ops"], "properties": {"ops": {"type": "array"}}},
            }
        },
    },
    # A duplicate id is deliberately NOT here: the loop RECOVERS from that one rather than rejecting
    # it. `id` OR the aliases `_assess_index` accepts, because requiring `id` alone would reject a
    # reply the loop can read and make that recovery unreachable for the weaker models it is for.
    "assess": {
        "type": "object",
        "required": ["assess"],
        "properties": {
            "assess": {
                "type": "array",
                "items": {
                    "type": "object",
                    "anyOf": [{"required": ["id"]}, {"required": ["candidate"]}, {"required": ["index"]}],
                    "properties": {k: {"type": "integer", "minimum": 0} for k in ("id", "candidate", "index")},
                },
            }
        },
    },
}
_VALIDATORS: dict[str, Any] = {}


class SchemaInvalid(ValueError):
    """A reply that PARSED and is not the shape that was asked for — a distinct type because it is
    the one reply failure a second ask can fix."""

    def __init__(self, category: str, detail: str):
        self.category, self.detail = category, detail
        super().__init__(f"{category} reply does not match its schema: {detail}")


def validate_reply(obj: Any, category: str | None) -> Any:
    """Return `obj`, or raise :class:`SchemaInvalid`.

    `category=None` means "this call declares no shape". A category that is not None and not in the
    table raises, because a typo in a `schema=` literal would otherwise disable validation for that
    call site silently.
    """
    if category is None:
        return obj
    if category not in REPLY_SCHEMAS:
        raise KeyError(f"no reply schema named {category!r}; have {sorted(REPLY_SCHEMAS)}")
    v = _VALIDATORS.get(category)
    if v is None:
        import jsonschema  # local: keeps a heavy import off the module path of every consumer

        v = _VALIDATORS[category] = jsonschema.Draft202012Validator(REPLY_SCHEMAS[category])
    errs = sorted(v.iter_errors(obj), key=lambda e: list(e.path))
    if errs:
        raise SchemaInvalid(category, "; ".join(f"{e.json_path}: {e.message}" for e in errs[:3])[:300])
    return obj


# --------------------------------------------------------------------------------------------------
# THE FAILURE POLICY — wait and try again within a per-call budget (`CallBudget`: `_MAX_ATTEMPTS`
# per kind of outage, `_RATE_LIMIT_MAX_TOTAL_WAIT` of waiting on rate limits), then stop the RUN
# (`TransportUnavailable`) rather than the beam, because one more request buys the same refusal at
# the same price and a stopped run can be resumed. The endpoint keeps a permit and a SHARED backoff window, so one refusal
# quiets it for every thread instead of each waiting thread posting its own fresh request into an
# endpoint that has just said no.
# --------------------------------------------------------------------------------------------------
_MAX_ATTEMPTS = 3  # per KIND of outage, within one call (see CallBudget)
# A 429 is the one refusal that is neither the request's fault nor the endpoint's: the caller is over
# a per-minute budget that refills on its own, so it is budgeted in TIME rather than in attempts.
# Three tries fifteen seconds apart all land inside the same closed window, and an attempt count
# cannot be sized to an outage of unknown length. Measured 2026-09-30: an Azure deployment shared by
# every run through the gateway returned 429 for 20-30 minutes at a stretch, and every pod that hit
# three in a row on one call lost 20 minutes of work. A call therefore keeps re-asking, with a wait
# that doubles up to two minutes, until it has spent 30 minutes waiting on rate limits -- long
# enough to outlast the stretch that was measured -- and only then does the run stop. A
# `Retry-After` the server sends is taken as-is up to the same two-minute cap.
_RATE_LIMIT_MAX_WAIT = 120.0
_RATE_LIMIT_MAX_TOTAL_WAIT = 1800.0
# A hard ceiling on requests, so the loop's end never rests on a float adding up: with a zero
# wait the time budget would never fill and the call would hammer the endpoint forever (a hang,
# not a failure, in CI). 1800 s of doubling waits is ~20 requests; 100 is far above any real path.
_MAX_RATE_LIMIT_REQUESTS = 100

# --- what went wrong -------------------------------------------------------------------------
RATE_LIMITED = "rate_limited"  # 429 — OUR rate; refused before inference, so wait and re-ask
QUOTA_EXHAUSTED = "quota_exhausted"  # 429 `insufficient_quota` — the ACCOUNT has no credits; waiting fixes nothing
SERVER_ERROR = "server_error"  # 5xx — their trouble; wait and re-ask
TRANSPORT = "transport"  # socket/timeout — never reached the model
BAD_REQUEST = "bad_request"  # 4xx — the request is wrong; retrying it is just paying twice
UNPARSEABLE = "unparseable"  # a reply arrived and was not the JSON that was asked for


@dataclass
class CallBudget:
    """What ONE logical call has spent so far, by kind, and whether it may spend more.

    Kinds are budgeted separately, so a 5xx met in the middle of a throttling streak neither ends
    the call on the outage's count nor buys the throttle extra tries: a 429, 429, 503 sequence is
    one outage attempt and two rate-limit waits, and the call goes on. Gateways do interleave the
    two (a proxy's 502/504 while the upstream throttles; LiteLLM's 503 "no deployments available"
    while every deployment cools down).
    """

    attempts: dict = field(default_factory=dict)  # kind -> requests that met this kind
    rate_limit_waited: float = 0.0  # seconds this call has spent waiting on 429s

    def record(self, f: Failure) -> int:
        """Book one failed request; returns how many of this kind the call has now met."""
        self.attempts[f.kind] = self.attempts.get(f.kind, 0) + 1
        return self.attempts[f.kind]

    def allows_another(self, f: Failure) -> bool:
        if f.kind == RATE_LIMITED:
            return (
                self.rate_limit_waited < _RATE_LIMIT_MAX_TOTAL_WAIT
                and self.attempts[RATE_LIMITED] < _MAX_RATE_LIMIT_REQUESTS
            )
        return self.attempts.get(f.kind, 0) < _MAX_ATTEMPTS

    def spent(self, act: "Retry") -> None:
        if act.reason == RATE_LIMITED:
            self.rate_limit_waited += act.after

    @property
    def requests(self) -> int:
        return sum(self.attempts.values())

    def describe(self, f: Failure) -> str:
        """The budget line for the retry log: attempts for an outage, time for a rate limit."""
        if f.kind == RATE_LIMITED:
            return f"{self.rate_limit_waited:.0f}s of {_RATE_LIMIT_MAX_TOTAL_WAIT:.0f}s waited"
        return f"attempt {self.attempts.get(f.kind, 0)} of {_MAX_ATTEMPTS}"


@dataclass(frozen=True)
class Failure:
    kind: str
    detail: str
    status: Optional[int] = None
    retry_after: Optional[float] = None


@dataclass(frozen=True)
class Retry:
    after: float
    reason: str


@dataclass(frozen=True)
class Abort:
    reason: str


class CallFailed(RuntimeError):
    """A model call that could not be completed, and that the caller may carry on past. Raised for a
    request the endpoint REJECTED — a fault in one call, not in the endpoint. Carries the Failure so
    a caller with a local fallback can decide on the KIND rather than on `except Exception`.
    """

    def __init__(self, failure: Failure, attempts: int):
        self.failure, self.attempts = failure, attempts
        super().__init__(f"{failure.kind} after {attempts} attempt(s): {failure.detail}")


class TransportUnavailable(BaseException):  # NOSONAR — must pass every `except Exception`; see below
    """Every attempt on one call failed: the endpoint is not answering, and the RUN stops.

    A `BaseException` deliberately. Every layer between a call and the driver catches `Exception` so
    one bad beam cannot sink a type — `_ask_json_retry`, the judge's per-sample guard, the region
    enumerator, the type router, the driver's per-beam and per-type handlers — and this has to pass
    through all of them unaltered. A run stopped here is RESUMABLE: the driver reads back what was
    delivered and asks only for the shortfall.
    """

    def __init__(self, failure: Failure, attempts: int, endpoint: str = ""):
        self.failure, self.attempts, self.endpoint = failure, attempts, endpoint
        super().__init__(f"{endpoint or 'endpoint'}: {failure.kind} on all {attempts} attempts — {failure.detail}")


def classify_status(status: int, body: str, retry_after: Optional[str]) -> Optional[Failure]:
    """A response -> Failure, or None when it is fine. The only place HTTP codes are interpreted."""
    if status < 400:
        return None
    ra: Optional[float] = None
    if retry_after:
        try:
            ra = float(retry_after)
        except ValueError:
            ra = None
    if status == 429:
        # Two different things arrive as 429. A rate limit (`rate_limit_exceeded`, `too_many_requests`,
        # or no code at all) is a per-minute budget that refills, so waiting works. `insufficient_quota`
        # is OpenAI's code for an account with no credits left: waiting does nothing, and retries would
        # only hide the real message. The code sits in `error.code` / `error.type` of the OpenAI error
        # envelope, which Azure, vLLM and a LiteLLM gateway all speak -- and a gateway wraps the
        # upstream envelope inside its own `error.message` as TEXT, so a substring test on the body is
        # the one check that reaches both the top-level field and the wrapped one. It is an
        # OpenAI-envelope code: the native Anthropic route (`_post_anthropic`) never emits it, so an
        # out-of-credit Anthropic key still classifies by status alone (its 429 as RATE_LIMITED, its
        # 400 "credit balance is too low" as BAD_REQUEST).
        if "insufficient_quota" in body:
            return Failure(QUOTA_EXHAUSTED, f"HTTP 429 insufficient_quota: {body[:300]}", status, ra)
        return Failure(RATE_LIMITED, f"HTTP 429: {body[:200]}", status, ra)
    if status >= 500:
        return Failure(SERVER_ERROR, f"HTTP {status}: {body[:200]}", status, ra)
    return Failure(BAD_REQUEST, f"HTTP {status}: {body[:300]}", status, ra)


def classify_exception(exc: BaseException) -> Optional[Failure]:
    """An exception raised while calling or parsing -> Failure, or None if it is OURS.

    None means "not a call failure": a bug in the code that built the request would otherwise be
    asked three times, push a backoff window onto every other thread on that endpoint, stop the run,
    and lose the traceback saying where the bug is. It propagates instead.
    """
    import httpx

    if isinstance(exc, httpx.HTTPError):
        return Failure(TRANSPORT, f"{type(exc).__name__}: {exc}")
    if isinstance(exc, (ValueError, KeyError, IndexError)):
        return Failure(UNPARSEABLE, f"{type(exc).__name__}: {exc}")
    return None


def decide(f: Failure, attempt: int, *, base: float, max_wait: float) -> Retry | Abort:
    """(failure, which attempt it was) -> wait this long, or stop now. Pure: no clock, no state, no I/O.

    For an outage (5xx, a dropped socket) a short step, not an escalation: `base` seconds before the
    second attempt, twice that before the third. A refusal that seconds do not clear is an outage,
    and the run stops instead.

    For a rate limit the step DOUBLES -- base, 2x, 4x ... -- up to `_RATE_LIMIT_MAX_WAIT`, because
    the budget it ran into refills per minute and a wait shorter than that only re-asks into the
    same closed window; how long the call keeps this up is `CallBudget`'s decision, not this one's.

    `attempt` is the number of failures OF THIS KIND the call has met, so a 5xx that interrupts a
    throttling streak does not reset the 429 step, nor the other way round.

    A server asking for longer than the step gets it, up to the applicable cap. Unbounded, an
    outsized `Retry-After` parks every thread on that endpoint with nothing logged, and the run
    looks hung for exactly as long as the header says.
    """
    if f.kind == BAD_REQUEST:
        return Abort(f"the request itself was rejected and will be rejected again — {f.detail}")
    if f.kind == QUOTA_EXHAUSTED:
        return Abort(f"the account behind this endpoint has no quota left; waiting cannot fix that — {f.detail}")
    if f.kind == RATE_LIMITED:
        step, cap = base * (2 ** max(0, attempt - 1)), max(max_wait, _RATE_LIMIT_MAX_WAIT)
    else:
        step, cap = base * max(1, attempt), max_wait
    return Retry(after=min(max(step, f.retry_after or 0.0), cap), reason=f.kind)


class Endpoint:
    """One endpoint's gate: a FIXED concurrency permit and a SHARED backoff window.

    Used as a context manager for the permit. A caller that is refused releases the permit FIRST and
    then waits — holding it across a backoff pins `live` at the limit for the whole wait, so no new
    work can start.

    The window is the shared part: one refusal quiets the endpoint for every thread until it expires,
    so a rate limit costs one wait rather than one per thread that happened to be mid-call.
    """

    def __init__(self, limit: int, *, base_wait: float = 5.0, max_wait: float = 60.0, name: str = ""):
        self.name = name
        self._limit = max(1, int(limit))
        self._live = 0
        self._until = 0.0
        self._base, self._max_wait = base_wait, max_wait
        self._cv = threading.Condition()
        self.calls = self.refusals = 0
        self.cache_w = self.cache_r = 0  # prompt-cache write/read token TOTALS, across every reply
        self.waited = 0.0

    def __enter__(self):
        with self._cv:
            while True:
                # ONE clock read per iteration, used by both the predicate and the timeout. Two reads
                # let the window expire between them — the predicate says "blocked", the timeout says
                # "the window has passed, wait forever" — and the thread parks on a notify that may
                # never come, holding no permit, so nothing wakes it at the tail of a type.
                now = time.monotonic()
                waiting_for_window = now < self._until
                if not (self._live >= self._limit or waiting_for_window):
                    break
                self._cv.wait(timeout=max(0.05, self._until - now) if waiting_for_window else None)
            self._live += 1
            self.calls += 1
        return self

    def __exit__(self, *exc):
        with self._cv:
            self._live -= 1
            self._cv.notify()
        return False

    def observe_failure(self, f: Failure, attempt: int, more: bool) -> Retry | Abort:
        """Decide on one failure and, when another attempt will follow, open the shared window.

        `attempt` comes from the caller's own loop (failures of this kind on ONE call), because the
        wait belongs to that call's tries; an endpoint-level counter would make it belong to
        whichever other thread failed first. `more` says whether the caller WILL try again — the
        caller's `CallBudget` knows, this endpoint does not, so it is not guessed here.

        The LAST attempt opens no window and adds nothing to `waited`, because nobody is going to
        wait: opening one holds every other thread on this endpoint for a step at the exact moment
        the run is trying to stop, and books that step into the figure `run.json` publishes as time
        this run spent waiting.
        """
        with self._cv:
            self.refusals += 1
            act = decide(f, attempt, base=self._base, max_wait=self._max_wait)
            if isinstance(act, Retry) and more:
                self._until = max(self._until, time.monotonic() + act.after)
                self.waited += act.after
                self._cv.notify_all()
            return act

    def observe_usage(self, cache_w: int | None, cache_r: int | None) -> None:
        """Accumulate one reply's prompt-cache usage into the endpoint's totals.

        The routes have already normalized the envelope's names — Anthropic's
        `cache_creation_input_tokens`/`cache_read_input_tokens` or OpenAI's
        `prompt_tokens_details.cached_tokens` — down to the (cache_w, cache_r) pair, so this books
        numbers whichever convention the endpoint spoke. None or 0 (an endpoint that does not
        cache, or a cold call) books nothing, and the report still says 0 rather than dropping
        the key: the per-call figures reach the trace either way, but the trace is opt-in and the
        run record is not — these totals are the one place a run's cache usage lands
        unconditionally.

        Called where the routes' 5-tuple is unpacked (`_call_json_once`, `Conversation.ask_json`)
        — NOT inside `_with_policy`, which is shape-agnostic about what `send` returns and is
        tested that way (stub endpoints, string-returning sends).
        """
        with self._cv:
            if cache_w:
                self.cache_w += int(cache_w)
            if cache_r:
                self.cache_r += int(cache_r)

    def report(self) -> dict:
        with self._cv:
            return {
                "calls": self.calls,
                "refusals": self.refusals,
                "waited_s": round(self.waited, 1),
                "limit": self._limit,
                # ALWAYS present, always ints — 0 when the endpoint never reported cache usage —
                # so every run.json carries the same `endpoints` schema and a reader can sum
                # across runs without guarding each key.
                "cache_write_tokens": self.cache_w,
                "cache_read_tokens": self.cache_r,
            }


def summarize(endpoints: dict) -> str:
    """One line per endpoint for the run's final log — printed whether or not anything went wrong,
    because an endpoint having a bad afternoon is otherwise invisible and lands in the API-time
    numbers looking like a slow model."""
    if not endpoints:
        return "endpoints: none"
    out = []
    for n, e in sorted(endpoints.items()):
        r = e.report()
        out.append(
            f"{n or 'default'}: {r['calls']} calls, {r['refusals']} refused, "
            f"{r['waited_s']}s waited, cache {r['cache_write_tokens']}w/{r['cache_read_tokens']}r tok, "
            f"concurrency {r['limit']}"
        )
    return "endpoints: " + " · ".join(out)


def log_retry(name: str, f: Failure, act: Retry, budget: str) -> None:
    """Say out loud that a call is waiting: a run backing off is otherwise indistinguishable from a
    run that has hung."""
    log.warning(f"{name or 'endpoint'}: {f.kind} ({budget}) — waiting {act.after:.0f}s · {f.detail[:120]}")


def _extract_json(text: str, schema: "str | None" = None) -> dict[str, Any]:
    """The reply's JSON object. With `schema`, the object the CALLER ASKED FOR — not just the first one.

    A reply can hold more than one top-level object: `AGENT_SYSTEM` frames its turn as picking an
    action, so an assess reply often writes `{"action":"cv","ops":[...]}` and THEN `{"assess":[...]}`.

    Only ever a fallback — with no schema, or when the first object satisfies it, the first object is
    what comes back.
    """
    cleaned = _FENCE.sub("", text).strip()
    _local.last_reply_repaired = False  # names THIS parse; the salvage below raises it
    try:
        obj = _DECODER.decode(cleaned)
    except json.JSONDecodeError:
        pass
    else:
        return _accept_or_look_further(obj, cleaned, schema)
    # Two different ways a reply can be unusable, and they need opposite handling.
    s = cleaned.find("{")
    if s < 0:
        raise json.JSONDecodeError("no object in reply", cleaned, 0)
    body = cleaned[s:]
    try:
        obj, end = _DECODER.raw_decode(body)
    except json.JSONDecodeError as e:
        # UNCLOSED: the object did not parse. Salvage what is there so a judge or beam sample still
        # yields its scores instead of being dropped whole.
        #
        # Two causes get separate kinds so a missing character does not read as a budget problem: a
        # reply cut off by max_tokens lost an answer, one that closed every brace but the outermost
        # lost a character. Both are REPAIR rather than failure kinds — a verdict that arrives whole
        # after a brace is added is not the run drifting off the configuration being measured.
        unbalanced = body.count("{") - body.count("}")
        kind = (
            "json_repaired:unclosed"
            if unbalanced > 0 and not e.msg.startswith("Unterminated")
            else "json_repaired:truncated"
        )
        repaired, dropped = _repair_truncated_json(body)
        out = _DECODER.decode(repaired)  # raises if it is past repairing
        # A salvaged reply is schema-valid and SHORTER than what the model wrote; a caller that caches
        # its answer watches for this (`watch_reply_repair`) and skips the write.
        _local.last_reply_repaired = True
        # WHAT WAS LOST, not just how much: a count of characters and braces cannot say that the
        # sample now has no `semantic`.
        log.warning(
            f"degraded[{kind}]: {len(body)} chars, {unbalanced} brace(s) unclosed"
            + (f"; dropped {dropped} — cut mid-value, so that field is absent" if dropped else "")
        )
        # Validated like the other two returns; skipped, repaired replies become the one population
        # nothing checks.
        return _accept_or_look_further(out, body, schema)
    # CLOSED EARLY: `raw_decode` SUCCEEDED, which is what makes this one dangerous — the object
    # parses, it just is not the whole answer. Prose after it is fine; `, "next_key":` means the
    # model put a brace where it did not belong and the rest is still to come. Taking the first
    # object there turns a seven-section card into a one-section one that still looks finished, and
    # that is what gets cached as the type's reference knowledge.
    for _ in range(4):
        rest = body[end:].lstrip()
        if not (rest.startswith(",") and rest[1:].lstrip().startswith('"')):
            break
        close = body.rfind("}", 0, end)
        if close < 0:
            break
        body = body[:close] + body[close + 1 :]
        try:
            obj, end = _DECODER.raw_decode(body)
        except json.JSONDecodeError as e:
            # The text past the splice does not parse either, so there is no whole answer here — and
            # `obj` is the object that closed early, the very reading this loop exists to reject.
            log.warning(
                f"degraded[json_splice_failed]: {len(body)} chars; the object closes early at {close} and the "
                f"text after the splice does not parse ({e.msg}, char {e.pos})"
            )
            raise
    # `body`, not `cleaned`: the loop above may have SPLICED OUT a premature brace, and rescanning the
    # original text would hand back the very reading that splice exists to reject.
    return _accept_or_look_further(obj, body, schema)


def _accept_or_look_further(obj: Any, text: str, schema: "str | None") -> Any:
    """`obj` if it satisfies `schema`; else the first LATER object in `text` that does; else raise.

    The argument that differs between `_extract_json`'s three returns is WHICH TEXT gets rescanned,
    which is the one thing an inlined copy gets wrong silently.
    """
    try:
        return validate_reply(obj, schema)
    except SchemaInvalid:
        alt = _later_object_matching(text, schema)
        if alt is not None:
            return alt
        raise


def _extract_traced(text: str, schema: "str | None") -> "tuple[Any, dict]":
    """The reply's object AND how it came back — ok, unparseable, or schema-invalid — from ONE parse.

    ONE parse, not two, because `_extract_json` is not side-effect free: it books
    `degraded[json_repaired:*]` when it salvages a reply, and a run is read by COUNTING those out of
    run.log, so a second parse would double-count every salvage.

    The failure is RETURNED rather than raised, because the trace has to be written before it
    surfaces — a rejection that erases its own evidence is how a too-strict constraint hides.
    """
    try:
        obj = _extract_json(text, schema)
    except SchemaInvalid as e:
        meta = {"reply": "schema_invalid", "reply_detail": e.detail}
        return e, ({**meta, "reply_repaired": True} if _local.last_reply_repaired else meta)
    except Exception as e:  # noqa: BLE001 — a trace field must never be what fails a call
        return e, {"reply": "unparseable", "reply_detail": f"{type(e).__name__}: {e}"[:200]}
    return obj, ({"reply": "ok", "reply_repaired": True} if _local.last_reply_repaired else {"reply": "ok"})


@dataclass
class ReplyRepairWatch:
    repaired: bool = False


@contextlib.contextmanager
def watch_reply_repair():
    """Whether a reply parsed inside the block was salvaged by `_repair_truncated_json`: read `.repaired`
    after the block.

    Such a reply is schema-valid and SHORTER than what the model wrote, so a caller that caches its
    answer skips the write. The flag is per thread and names the LAST parse — a re-asked call's final
    reply — and is cleared on entry, so a call that parsed nothing reads False.
    """
    _local.last_reply_repaired = False
    w = ReplyRepairWatch()
    try:
        yield w
    finally:
        w.repaired = bool(getattr(_local, "last_reply_repaired", False))


def _later_object_matching(cleaned: str, schema: "str | None") -> "dict[str, Any] | None":
    """Scan on past the first object for one that DOES satisfy `schema`. None if there is no such thing.

    Bounded on purpose: at most 8 objects, and only ones that decode cleanly — this is a salvage path,
    not a second parser, and a reply that needs more than eight tries is a reply worth re-asking for.
    """
    if not schema:
        return None
    pos, found = 0, 0
    while found < 8:
        i = cleaned.find("{", pos)
        if i < 0:
            return None
        try:
            obj, end = _DECODER.raw_decode(cleaned[i:])
        except json.JSONDecodeError:
            pos = i + 1
            continue
        found += 1
        pos = i + end
        try:
            return validate_reply(obj, schema)
        except SchemaInvalid:
            continue
    return None


# The public Anthropic API — the ONLY base constant in the code, and the one base that speaks the
# native protocol. A property of the world, not deployment config (a module constant reads no
# environment, so the zero-env-reads engine invariant holds): api.anthropic.com serves exactly one
# protocol, the native /messages wire format, so the base alone decides the envelope there whatever
# the model is called — only Claude models live at this base, and a wrong name fails at the
# capability smoke before the run spends.
ANTHROPIC_BASE = "https://api.anthropic.com/v1"


def _is_claude(model: str) -> bool:
    """A Claude-family id — LAYER 2 of the exactly-two routing layers: the model NAME picks the
    CACHE MARKERS, never the envelope. A claude-named model carries `cache_control` breakpoints
    whatever base it resolves to — natively on /messages, and embedded in OpenAI block form on
    /chat/completions (the LiteLLM/OpenRouter gateway convention: the gateway forwards the markers
    to an Anthropic backend, so prompt caching survives the chat envelope). Any other model gets
    the clean payload an OpenAI-compatible endpoint expects."""
    m = model.lower()
    return "claude" in m or "anthropic" in m


def sent_png_bytes(img: "Image.Image", max_dim: int, max_b64_len: int) -> bytes:
    """The EXACT PNG bytes the transport sends for one image. Convert to RGB, optionally thumbnail the
    long side to ``max_dim`` (0 = native), then shrink + re-encode until the BASE64 length is under
    ``max_b64_len`` (gateways cap the base64 payload, not the raw PNG). The single source of truth for
    what a delivered image is; ``_img_block`` is its one caller, and the tracer decodes what that
    produced rather than re-encoding, so "what the API saw" cannot drift between them."""
    img = img.convert("RGB")
    if max_dim and max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw)
    while len(b64) > max_b64_len and min(img.size) > 64:
        scale = max(0.4, (max_b64_len / len(b64)) ** 0.5 * 0.9)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        b64 = base64.b64encode(raw)
    return raw


def _store_blob(data: bytes, shared: Path) -> str:
    """Put bytes in the content-addressed store under ``shared``, ONCE, and return their key.

    THE one implementation, for images, masks and the sanitized payload alike: a second copy of the
    hashing is a second answer to "which file is this", and `_trace_call` swallows every exception,
    so it would surface only as a `sha` naming a file that does not exist.
    """
    h = hashlib.md5(data, usedforsecurity=False).hexdigest()[:16]
    dst = shared / f"{h}.png"
    if not dst.exists():
        # The name IS the digest, so a concurrent writer races to byte-identical content — but a
        # reader must never open the half of it that had landed.
        atomic_write(dst, lambda t: t.write_bytes(data))
    return h


def _link_to_store(link: Path, target: Path) -> None:
    """Point ``link`` at the stored copy, RELATIVELY, so a trace directory survives being moved or
    copied whole. An existing link is left alone: it was made from the same content hash, so it
    already names these bytes."""
    try:
        link.symlink_to(os.path.relpath(target, link.parent))
    except FileExistsError:
        pass


class _Refusal(Exception):
    """A response the endpoint declined. Carries the classified Failure to the policy layer, so the
    routes never decide anything — they post, classify, and hand it up."""

    def __init__(self, failure):
        self.failure = failure
        super().__init__(failure.detail)


class Vlm:
    """VLM transport client. Construct once from ``Config`` (via the entry) and pass into the engine.

    Holds one `Endpoint` per base_url (concurrency permit + shared backoff window) and the image
    downscale cap. Concurrency is capped per ENDPOINT because every beam sharing one shares its
    rate limit with it. One pooled ``httpx.Client`` across threads.
    """

    def __init__(
        self,
        key: str,
        base_url: str,
        *,
        concurrency: int = 16,
        img_maxdim: Optional[int] = None,
        qwen_thinking: bool = False,
        timeout: float = 180.0,
        max_image_bytes: int = 4_500_000,
        trace_dir: Optional[str] = None,
        trace_seq_start: int = 0,
        endpoints: Optional[dict[str, tuple[str, str]]] = None,
    ) -> None:
        self._key = key
        self._base_url = base_url
        # Per-model endpoint overrides: {model_name: (base_url, key)}; a model not in the map falls
        # back to (base_url, key) above. This is how --gen-model / --judge-model can point at
        # different backends (a local GLM vs Claude on a cloud gateway).
        self._endpoints = dict(endpoints or {})
        # Health per ENDPOINT, not per process: two models on one base share a limit because the
        # endpoint shares one, and a local GLM does not belong under a cloud gateway's backoff.
        self._concurrency = max(1, int(concurrency))
        self._healths: dict[str, Endpoint] = {}
        self._health_lock = threading.Lock()
        self._img_maxdim = img_maxdim
        # HARD cap on a single image's encoded size — gateways commonly reject images over 5 MB, and
        # the default 4.5 MB leaves margin under it. `img_maxdim` is only a quality/token preference;
        # every image is shrunk here whatever it is, so a request cannot be rejected for image size.
        self._max_image_bytes = int(max_image_bytes)
        self._qwen_thinking = qwen_thinking
        # httpx defaults to max_connections=100, which would silently cap concurrency below the permit
        # limit for a large max_concurrent_calls. Size the pool to the requested concurrency so the
        # endpoint's permit is the only concurrency limiter.
        _pool = max(16, int(concurrency) + 8)
        self._http = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=_pool, max_keepalive_connections=_pool),
            # Ambient env (HTTPS_PROXY / SSL_CERT_FILE) must not reroute calls — the endpoints
            # registry is the ONE source of where a request goes.
            trust_env=False,
        )
        # Full-IO trace: when set, EVERY call dumps its system + text + images + response +
        # timing/tokens to a per-call subdir (and a root index.jsonl).
        self._trace_dir = trace_dir
        # Call numbering starts where the CALLER says, because a traced call writes
        # <trace_dir>/<ctx>/<seq>_<kind>/: a process pointed at a trace directory that already holds
        # calls would otherwise file its first one on top of the first one there, and `_trace_call`
        # reports nothing. The driver hands in the next free number when it resumes into a trace.
        self._trace_seq = max(0, int(trace_seq_start))
        self._trace_lock = threading.Lock()

    @property
    def image_max_dim(self) -> int:
        """The longest side every image is shrunk to before it is sent; 0 = native resolution.

        PUBLIC because `Views.from_vlm` reads it: the views draw grids at the size the image will be
        DELIVERED at. A property, so a rename of the field behind it raises here instead of quietly
        handing every judge, region and card view a cap of 0.
        """
        return int(self._img_maxdim or 0)

    def _sanitize_payload(self, obj: Any, shared: Path) -> Any:
        """The request body as sent, with image base64 swapped for a reference to the stored bytes.

        Everything else is verbatim — the full `messages` array, the history and the `cache_control`
        markers — so the file can answer what the API actually received. Only the base64 is replaced,
        because a six-turn beam repeats the same images every turn: written out in full they are
        hundreds of megabytes per run, byte-identical to the copy already in _images/.
        """
        if isinstance(obj, dict):
            src = obj.get("source")
            if obj.get("type") == "image" and isinstance(src, dict) and src.get("data"):
                try:
                    data = base64.b64decode(src["data"])
                except Exception:
                    return {k: self._sanitize_payload(v, shared) for k, v in obj.items()}
                out = {"type": "image", "sha": _store_blob(data, shared), "bytes": len(data)}
                try:
                    out["size"] = list(Image.open(io.BytesIO(data)).size)
                except Exception:
                    pass
                if obj.get("cache_control"):
                    out["cache_control"] = obj["cache_control"]
                return out
            # data: URI form used by the chat envelope
            url = (obj.get("image_url") or {}).get("url") if isinstance(obj.get("image_url"), dict) else None
            if url and url.startswith("data:") and "," in url:
                try:
                    data = base64.b64decode(url.split(",", 1)[1])
                    out = {"type": "image_url", "sha": _store_blob(data, shared), "bytes": len(data)}
                    if obj.get("cache_control"):  # a claude-on-gateway marker rides image blocks too
                        out["cache_control"] = obj["cache_control"]
                    return out
                except Exception:
                    pass
            return {k: self._sanitize_payload(v, shared) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_payload(v, shared) for v in obj]
        return obj

    def _trace_call(self, kind: str, system: str, content: list[Any], text: str, meta: dict) -> None:
        """Dump one call's full IO — the EXACTLY-sent request — under ``trace_dir`` (no-op when unset).
        ``content`` is the transport content (post ``_blocks_to_content``): its image base64 is byte-
        identical to what the API received (both envelopes send these bytes — the chat one only
        rewraps them as a data: URI), so we DECODE it rather than re-encode the pre-downscale
        original. Best-effort."""
        # Cleared HERE, above the early return and outside the try. Inside the try, a symlink failure
        # (ENOSPC, a filesystem without symlinks) leaves it pending for the NEXT traced call to claim
        # — a request.json naming someone else's candidate mask. Inside `if self._trace_dir`, every
        # pool thread pins its last request body, base64 images included, for the executor's life.
        pending = getattr(_local, "pending_masks", None) or []
        wire = getattr(_local, "raw_io", None)  # the route's exact (payload, response) pair, if captured
        _local.pending_masks, _local.raw_io = [], None
        if not self._trace_dir:
            return
        try:
            with self._trace_lock:
                seq = self._trace_seq
                self._trace_seq += 1
            ctx = getattr(_local, "trace_ctx", "")  # per-beam group (set once per run; serial judging)
            root = Path(self._trace_dir)
            d = (root / ctx if ctx else root) / f"{seq:05d}_{kind}"
            d.mkdir(parents=True, exist_ok=True)
            shared = root / "_images"  # content-addressed store: identical images kept ONCE, symlinked in
            req: list[dict] = []
            n_img = 0
            last_cap = ""  # the text block right before an image is its caption -> used for the filename
            for b in content:
                if not isinstance(b, dict):  # defensive: a raw string
                    last_cap = str(b)
                    req.append({"text": str(b)})
                    continue
                if b.get("type") == "image":
                    data = base64.b64decode(b.get("source", {}).get("data", ""))  # EXACT bytes sent
                    h = _store_blob(data, shared)  # stored ONCE (dedup across every call + beam)
                    try:
                        wh = list(Image.open(io.BytesIO(data)).size)  # the SENT size (post-downscale)
                    except Exception:
                        wh = None
                    name = f"{n_img:02d}_{_img_label(last_cap) or 'image'}.png"  # e.g. 02_clean_image_grid.png
                    _link_to_store(d / name, shared / f"{h}.png")
                    req.append({"image": name, "of": last_cap[:70], "sha": h, "size": wh})
                    n_img += 1
                else:  # text block
                    last_cap = b.get("text", "")
                    req.append({"text": last_cap})
            # flush any staged binary candidate masks into this call's dir (content-addressed like images)
            masks_dir = root / "_masks"
            for i, (mname, m) in enumerate(pending):
                try:
                    buf = io.BytesIO()
                    m.convert("L").save(buf, "PNG")
                    h = _store_blob(buf.getvalue(), masks_dir)  # ONCE (the judge samples reuse it)
                    nm = f"mask_{i:02d}_{_img_label(mname) or 'mask'}.png"
                    _link_to_store(d / nm, masks_dir / f"{h}.png")
                    req.append({"mask": nm, "of": mname, "sha": h})
                except Exception:
                    pass
            (d / "request.json").write_text(
                json.dumps({"kind": kind, "model": meta.get("model"), "system": system, "blocks": req}, indent=1)
            )
            (d / "response.txt").write_text(text)
            # The wire pair, when the route captured it. request.json above is this call's NEW blocks
            # only — for a conversation, one turn rather than the accumulated history that was
            # billed — so what the API really received has to be read from here.
            if wire:
                try:
                    (d / "payload.json").write_text(
                        json.dumps(self._sanitize_payload(wire["payload"], shared), indent=1)
                    )
                    (d / "response_raw.json").write_text(json.dumps(wire["response"], indent=1))
                except Exception:  # a trace must never be able to fail the call it is recording
                    pass
            (d / "meta.json").write_text(json.dumps({**meta, "thread": threading.get_ident()}, indent=1))
            with self._trace_lock, open(Path(self._trace_dir) / "index.jsonl", "a") as fp:
                fp.write(
                    json.dumps(
                        {
                            "seq": seq,
                            "ctx": ctx,
                            "kind": kind,
                            "ms": meta.get("ms"),
                            "encode_ms": meta.get("encode_ms"),  # LOCAL image-encode CPU (not in ms)
                            "in_tok": meta.get("in_tok"),
                            "out_tok": meta.get("out_tok"),
                            "cache_r": meta.get("cache_r"),
                            "n_images": n_img,
                            "resp_chars": len(text),
                            "model": meta.get("model"),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass  # tracing must never break a run

    # --- content encoding ---
    def _img_block(self, img: Image.Image) -> dict[str, Any]:
        # The encoded payload is memoized ON the image object, keyed by (maxdim, max_bytes): the same
        # view images are re-sent across every refine round and judge sample. Safe only because view
        # images are never mutated after they are built — anything that changes gets a fresh object.
        key = (self._img_maxdim or 0, self._max_image_bytes)
        cached = getattr(img, "_amg_b64", None)
        if cached is not None and cached[0] == key:
            data = cached[1]
        else:
            data = base64.b64encode(sent_png_bytes(img, self._img_maxdim or 0, self._max_image_bytes)).decode()
            try:
                img._amg_b64 = (key, data)
            except Exception:
                pass  # a rare image proxy may reject attributes — just skip the cache
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    def _blocks_to_content(self, blocks: list[Any]) -> list[dict[str, Any]]:
        return [self._img_block(b) if isinstance(b, Image.Image) else {"type": "text", "text": str(b)} for b in blocks]

    # --- transports ---
    def _endpoint_for(self, model: str) -> tuple[str, str]:
        """(base_url, key) for ``model`` — a per-model override if registered, else the default pair."""
        return self._endpoints.get(model, (self._base_url, self._key))

    def _route_for(self, model: str) -> str:
        """Which envelope this call speaks — ``anthropic`` (native /messages) or ``chat``.

        LAYER 1 of the exactly-two routing layers: THE BASE PICKS THE ENVELOPE. ``ANTHROPIC_BASE``
        is the public Anthropic API, which serves exactly one protocol — the native ``/messages``
        wire format — so every model on it takes the Anthropic route whatever it is called (only
        Claude models live there; a wrong name fails at the capability smoke pre-spend). ANY other
        base — per-model override included — is an OpenAI-compatible endpoint, and every model on
        it speaks ``/chat/completions`` whatever it is called: OpenRouter-style endpoints serve ids
        like ``anthropic/claude-*`` over plain chat, and a name rule here would speak the Anthropic
        wire format at an endpoint that does not have it. The model name decides only the cache
        markers (layer 2, `_is_claude`), never the envelope. Bases compare ``rstrip('/')``-
        normalized, so a trailing slash cannot flip the protocol.
        """
        base, _ = self._endpoint_for(model)
        return "anthropic" if base.rstrip("/") == ANTHROPIC_BASE else "chat"

    def _health_for(self, model: str) -> Endpoint:
        base, _ = self._endpoint_for(model)
        with self._health_lock:
            h = self._healths.get(base)
            if h is None:
                h = self._healths[base] = Endpoint(self._concurrency, name=base)
            return h

    def endpoint_report(self) -> dict:
        """Per-endpoint calls / refusals / waiting / prompt-cache totals, for the run record."""
        with self._health_lock:
            return {n: e.report() for n, e in self._healths.items()}

    def endpoint_summary(self) -> str:
        """One line about how the endpoints behaved, for a run's closing log."""
        with self._health_lock:
            return summarize(self._healths)

    def _with_policy(self, model: str, send):
        """Run ONE call under the endpoint's policy: permit, attempt, classify, decide, wait, repeat —
        until the call's `CallBudget` is spent, and then the run is over.

        The permit is held for the attempt and released before any waiting.

        This is the ONLY HTTP-level retry loop: the routes post, classify and raise, and nothing
        below them retries. Two refusals end the RUN rather than the call: a spent budget (the
        endpoint is not answering) and `QUOTA_EXHAUSTED` (nothing this run sends will succeed until
        the account is funded). Both leave the run resumable. `BAD_REQUEST` ends only the CALL, as
        one request's fault, and the caller carries on past it.

        Outage kinds are counted separately (`_MAX_ATTEMPTS` EACH), so an endpoint flapping between
        a 5xx, a dropped socket and an unparseable reply can cost one call up to three times three
        requests before it stops — still bounded, and each kind's step starts over.
        """
        ep = self._health_for(model)
        budget = CallBudget()
        last = None
        while True:
            with ep:
                # EVERY ATTEMPT counts, not just the one that worked: a call the endpoint refused
                # twice before answering cost three round-trips.
                _bump()
                try:
                    out = send()
                except _Refusal as r:
                    f = r.failure
                except Exception as e:  # noqa: BLE001 — classified below, or re-raised if it is ours
                    f = classify_exception(e)
                    if f is None:
                        # our bug, not a refusal — but it still ends the call, so the staged trace
                        # state must not survive into the next one
                        discard_staged_masks()
                        raise
                else:
                    return out
                _bump_failed(f.kind)  # by KIND: a 429 and a timeout do not cost the same
                nth = budget.record(f)
                more = budget.allows_another(f)
                # Register the backoff window while still holding the permit. Released first, the
                # notify lets waiting threads into the endpoint that has just declined — up to one
                # fresh request per refusal.
                act = ep.observe_failure(f, nth, more)
            last = f
            if isinstance(act, Abort):
                discard_staged_masks()
                if f.kind == QUOTA_EXHAUSTED:
                    log.warning(f"degraded[api_gave_up:{f.kind}]: {act.reason}")
                    raise TransportUnavailable(f, budget.requests, ep.name)
                log.warning(f"degraded[api_abort:{f.kind}]: {act.reason}")
                raise CallFailed(f, budget.requests)
            if not more:
                break
            budget.spent(act)
            log_retry(ep.name, f, act, budget.describe(f))
            time.sleep(act.after)  # OUTSIDE the permit: waiting must never hold one
        discard_staged_masks()  # BEFORE the record: this path raises, and a staged mask must not outlive it
        log.warning(
            f"degraded[api_gave_up:{last.kind}]: {ep.name or 'endpoint'} still refusing after "
            f"{budget.requests} requests ({budget.describe(last)}), so the run stops — {last.detail}"
        )
        raise TransportUnavailable(last, budget.requests, ep.name)

    def _post_anthropic(self, model, system, messages, max_tokens):
        """ONE /messages round-trip. No retry and no backoff here: the route posts, classifies
        the response, and raises _Refusal for the policy layer (_with_policy) to decide on."""
        base, key = self._endpoint_for(model)
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        # No sampling parameters on either route: every call takes the endpoint's own default, so a
        # request is the same for every model (Astra rejects `temperature` outright with HTTP 400).
        # /messages is only ever spoken at ANTHROPIC_BASE (layer 1), and Anthropic's real endpoint
        # authenticates API keys via `x-api-key` — its `authorization: Bearer` slot is OAuth
        # tokens, so a key sent there is a 401 blamed on the user's key.
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        url = base.rstrip("/") + "/messages"
        r = self._http.post(url, headers=headers, json=body)
        f = classify_status(r.status_code, r.text, r.headers.get("retry-after"))
        if f:
            raise _Refusal(f)
        d = r.json()
        # The EXACT wire pair for the tracer: response.txt keeps only the text blocks, so anything
        # returned in another block kind (thinking, tool_use) is billed but invisible there.
        _local.raw_io = {"payload": body, "response": d, "status": r.status_code}
        text = "".join(p.get("text", "") for p in d.get("content", []) if p.get("type") == "text")
        u = d.get("usage") or {}
        return (
            text,
            u.get("input_tokens"),
            u.get("output_tokens"),
            u.get("cache_creation_input_tokens"),
            u.get("cache_read_input_tokens"),
        )

    def _to_openai_messages(self, system, messages, mark_cache: bool):
        """The shared Anthropic-shaped history in OpenAI chat form.

        ``mark_cache`` embeds the SAME ``cache_control`` markers the /messages path carries, in
        OpenAI block form — the system message's content becomes a text block carrying the marker,
        and a content block that was marked upstream (cache_prefix, the pinned first user turn)
        keeps its marker on the converted block. This is the LiteLLM/OpenRouter gateway convention
        for prompt caching over chat. With ``mark_cache`` False the payload is the plain OpenAI
        request, marker-free end to end — an endpoint that does not know the convention never sees
        an unknown field.
        """
        sys_content: Any = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}] if mark_cache else system
        )
        out: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                blocks = []
                for b in c:
                    if b.get("type") == "image":
                        src = b["source"]
                        nb: dict[str, Any] = {
                            "type": "image_url",
                            "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"},
                        }
                    else:
                        nb = {"type": "text", "text": b.get("text", "")}
                    if mark_cache and b.get("cache_control"):
                        nb["cache_control"] = b["cache_control"]
                    blocks.append(nb)
                c = blocks if m["role"] == "user" else "".join(x.get("text", "") for x in blocks)
            out.append({"role": m["role"], "content": c})
        return out

    def _post_openai(self, model, system, messages, max_tokens):
        """One /chat/completions round-trip. Returns (text, in_tok, out_tok, cache_w, cache_r).

        LAYER 2 applies here: a claude-named model gets the cache markers embedded in OpenAI block
        form (`_to_openai_messages`); anything else gets the plain payload. The cache usage fields
        come back in either the gateway's Anthropic-passthrough names (`cache_creation_input_tokens`
        / `cache_read_input_tokens`) or OpenAI's (`prompt_tokens_details.cached_tokens`); absent on
        an endpoint that does not cache, they are None and the accounting books nothing. One
        semantics caveat for readers of the runlog: OpenAI's `cached_tokens` is a SUBSET of
        `prompt_tokens`, while Anthropic's `cache_read_input_tokens` is DISJOINT from
        `input_tokens` — so cache_r's relation to in_tok differs by which names the endpoint used."""
        base, key = self._endpoint_for(model)
        ml = model.lower()
        # Reasoning models (e.g. nemotron '*-reasoning') spend tokens on a hidden reasoning stream
        # that counts against the output cap, and the serving endpoint exposes no toggle to turn it
        # off — so a budget sized for the answer alone can be consumed entirely by reasoning, leaving
        # empty content. The headroom is free: the cap is billed per token actually generated.
        effective_max = max_tokens + 16384 if "reasoning" in ml else max_tokens
        # The cap goes out as `max_completion_tokens`, the chat API's current name for it, and the
        # ONLY name OpenAI's own endpoint accepts for a reasoning model such as GPT-6 Astra (it
        # answers `max_tokens` with HTTP 400). vLLM reads the two names into the same field
        # (`max_tokens` is its deprecated spelling) and LiteLLM gateways translate it for every
        # backend, so one payload serves the official endpoint, a gateway and a local server alike.
        payload: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": effective_max,
            "messages": self._to_openai_messages(system, messages, mark_cache=_is_claude(model)),
        }
        # qwen and GLM emit a long <think> stream by default, which costs latency and puts prose in front
        # of the JSON this pipeline parses. Their vLLM chat templates honour
        # chat_template_kwargs.enable_thinking=False, which is the only way to suppress it.
        if ("qwen" in ml or "glm" in ml) and not self._qwen_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        url = base.rstrip("/") + "/chat/completions"
        headers = {"authorization": f"Bearer {key}"}
        r = self._http.post(url, headers=headers, json=payload)
        f = classify_status(r.status_code, r.text, r.headers.get("retry-after"))
        if f:
            raise _Refusal(f)
        d = r.json()
        _local.raw_io = {"payload": payload, "response": d, "status": r.status_code}
        # No choice, or a choice with no message, is an EMPTY reply: one that arrived and cannot be
        # used, which the JSON layer re-asks once. Read as a refusal it would be retried under the
        # endpoint's policy instead, opening a backoff window on every thread sharing the endpoint.
        choices = d.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        u = d.get("usage") or {}
        cw = u.get("cache_creation_input_tokens")
        cr = u.get("cache_read_input_tokens")
        if cr is None:
            cr = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        return msg.get("content") or "", u.get("prompt_tokens"), u.get("completion_tokens"), cw, cr

    # --- public API ---
    def call_json(
        self,
        system: str,
        blocks: list[Any],
        *,
        model: str,
        max_tokens: int = 1500,
        cache_prefix: int = 0,
        schema: Optional[str] = None,
    ) -> dict[str, Any]:
        """One-shot JSON call, retried once on an unusable REPLY, with the give-up recorded.

        The transport retries 429s and 5xxs; this layer owns the failure above them — a reply that
        ARRIVED and could not be used, which would otherwise become a default in the caller's
        `except` (no regions, no verdict, a candidate scored minus infinity). Conversations get the
        same policy in `loop._ask_json_retry`.
        """
        try:
            return self._call_json_once(
                system, blocks, model=model, max_tokens=max_tokens, cache_prefix=cache_prefix, schema=schema
            )
        except CallFailed:
            # Already classified by the transport's policy; retrying here would be a second, unaware
            # retry loop — a 400 asked twice, a rate-limit give-up paid for six times.
            raise
        except Exception as e:
            # A reply ARRIVED and was not the JSON asked for: the one failure a second ask can fix.
            # Anything `classify_exception` does not recognise is a bug in OUR request-building.
            # `SchemaInvalid` is named explicitly because it is our check, not a transport failure.
            if not isinstance(e, SchemaInvalid) and classify_exception(e) is None:
                raise
            err = f"{type(e).__name__}: {e}"
            try:
                return self._call_json_once(
                    system, blocks, model=model, max_tokens=max_tokens, cache_prefix=cache_prefix, schema=schema
                )
            except Exception as e2:
                log.warning(f"degraded[call_gave_up:call_json]: {err} -> {type(e2).__name__}: {e2}")
                raise

    def _call_json_once(
        self,
        system: str,
        blocks: list[Any],
        *,
        model: str,
        max_tokens: int = 1500,
        cache_prefix: int = 0,
        schema: Optional[str] = None,
    ) -> dict[str, Any]:
        """``blocks`` mixes PIL images and strings; returns the parsed JSON dict.

        ``cache_prefix`` > 0 attaches a prompt-cache marker to content block ``cache_prefix - 1`` so
        the prefix ``system + blocks[0:cache_prefix]`` is cached — on the native /messages envelope,
        and for claude-named models on chat, where it rides in OpenAI block form (layer 2). A
        non-claude model's chat payload stays clean, so for it the ask is ignored."""
        _e0 = time.perf_counter()
        content = self._blocks_to_content(blocks)
        encode_ms = round((time.perf_counter() - _e0) * 1000)  # LOCAL client CPU: PNG + base64 image encoding
        t0 = time.perf_counter()
        route = self._route_for(model)  # LAYER 1: the RESOLVED base decides the envelope
        # LAYER 2: the model NAME decides the cache markers. The marker is attached to the shared
        # Anthropic-shaped block either way; on the chat envelope `_to_openai_messages` carries it
        # into OpenAI block form, and a non-claude model never gets one, so its payload stays clean.
        if cache_prefix > 0 and content and (route == "anthropic" or _is_claude(model)):
            content[min(cache_prefix, len(content)) - 1]["cache_control"] = {"type": "ephemeral"}
        if route == "anthropic":
            text, in_tok, out_tok, cw, cr = self._with_policy(
                model, lambda: self._post_anthropic(model, system, [{"role": "user", "content": content}], max_tokens)
            )
        else:
            text, in_tok, out_tok, cw, cr = self._with_policy(
                model, lambda: self._post_openai(model, system, [{"role": "user", "content": content}], max_tokens)
            )
        _acc_usage(in_tok, out_tok, cw, cr)  # _bump fires per ATTEMPT, inside _with_policy
        # Per-ENDPOINT cache totals (thread-locals above are per-BEAM): run.json's `endpoints` block
        # carries what the per-call trace already knew, envelope-independent because both routes
        # normalized the names into (cw, cr) already.
        self._health_for(model).observe_usage(cw, cr)
        http_ms = round((time.perf_counter() - t0) * 1000)
        _acc_api_ms(http_ms)
        rec = {
            "kind": "call_json",
            "ms": http_ms,  # round-trip + any permit wait / backoff this call served
            "encode_ms": encode_ms,  # LOCAL image-encode CPU (client; NOT in ms)
            "chars": len(text),
            "in_tok": in_tok,
            "out_tok": out_tok,
            "cache_w": cw,
            "cache_r": cr,
        }
        _log_call(rec)
        obj, outcome = _extract_traced(text, schema)
        self._trace_call("call_json", system, content, text, {**rec, "model": model, "schema": schema or "", **outcome})
        # Surfaced after the trace, so a schema-rejected reply is still on disk to be looked at.
        if isinstance(obj, BaseException):
            raise obj
        return obj

    def conversation(
        self,
        system: str,
        *,
        model: str,
        prune_old_images: bool = True,
        pin_first_images: bool = False,
    ) -> "Conversation":
        return Conversation(
            self, system, model=model, prune_old_images=prune_old_images, pin_first_images=pin_first_images
        )


class Conversation:
    """A stateful multi-turn conversation: context accumulates across turns, ONE API call each."""

    def __init__(
        self,
        vlm: Vlm,
        system: str,
        *,
        model: str,
        prune_old_images: bool = True,
        pin_first_images: bool = False,
    ) -> None:
        self._vlm = vlm
        self.system = system
        self.model = model
        self.messages: list[dict[str, Any]] = []
        self.last_text: str = ""
        # Keep only the LATEST turn's images (full TEXT history always stays) so context is bounded.
        self.prune_old_images = prune_old_images
        # Keep the FIRST user turn's images (exemplars + clean + visual prompts) for the whole
        # conversation — refining "like the examples show" is impossible if they vanish after turn 1.
        self.pin_first_images = pin_first_images

    def _prune(self) -> None:
        first_user = next((i for i, m in enumerate(self.messages) if m["role"] == "user"), None)
        for i, msg in enumerate(self.messages[:-1]):
            if self.pin_first_images and i == first_user:
                continue
            if msg["role"] != "user" or not isinstance(msg.get("content"), list):
                continue
            kept = [b for b in msg["content"] if b.get("type") != "image"]
            n_img = len(msg["content"]) - len(kept)
            if n_img:
                kept.append(
                    {"type": "text", "text": f"[{n_img} image(s) from an earlier turn omitted to save context]"}
                )
                msg["content"] = kept

    def ask_json(
        self, blocks: list[Any], max_tokens: int = 1200, cache_prefix: int = 0, schema: Optional[str] = None
    ) -> dict[str, Any]:
        _e0 = time.perf_counter()
        content = self._vlm._blocks_to_content(blocks)  # transport content; its image base64 is what's sent
        encode_ms = round((time.perf_counter() - _e0) * 1000)  # LOCAL client CPU: PNG + base64 image encoding
        self.messages.append({"role": "user", "content": content})
        if self.prune_old_images:
            self._prune()
        route = self._vlm._route_for(self.model)  # LAYER 1: the RESOLVED base decides the envelope
        # LAYER 2: the model NAME decides the cache markers — a claude-named model carries them on
        # either envelope (the chat conversion embeds them in OpenAI block form).
        marked = route == "anthropic" or _is_claude(self.model)
        # PROMPT CACHE only honoured when pinned: `_prune` rewrites earlier turns' images to a text
        # placeholder (skipping the first user turn only when pinned), so an unpinned marked turn-1
        # gets mutated next turn — every write cache-busted before it is read. Pinned, the default
        # marker goes at the END of the first user turn, so the exemplars, clean image and visual
        # prompts are read from cache on every refine round. `cache_prefix` still overrides.
        if self.pin_first_images and marked:
            first_user = next((m for m in self.messages if m["role"] == "user"), None)
            if first_user and isinstance(first_user.get("content"), list) and first_user["content"]:
                c = first_user["content"]
                at = min(cache_prefix, len(c)) - 1 if cache_prefix > 0 else len(c) - 1
                c[at]["cache_control"] = {"type": "ephemeral"}
        # HTTP round-trip PLUS whatever the endpoint's policy made this call wait for: a permit and,
        # after a refusal, a backoff.
        t0 = time.perf_counter()
        if route == "anthropic":
            text, in_tok, out_tok, cw, cr = self._vlm._with_policy(
                self.model, lambda: self._vlm._post_anthropic(self.model, self.system, self.messages, max_tokens)
            )
        else:
            text, in_tok, out_tok, cw, cr = self._vlm._with_policy(
                self.model, lambda: self._vlm._post_openai(self.model, self.system, self.messages, max_tokens)
            )
        _acc_usage(in_tok, out_tok, cw, cr)  # _bump fires per ATTEMPT, inside _with_policy
        # Per-ENDPOINT cache totals, exactly as in `_call_json_once` — a conversation turn is the
        # OTHER place a reply's usage is unpacked, and forgetting it here would under-report every
        # refine round.
        self._vlm._health_for(self.model).observe_usage(cw, cr)
        # An empty (or whitespace) reply is not appended: /messages rejects an empty text block, so
        # one in the history fails every later request on this conversation. The trailing message is
        # then the user's ask, which `loop._ask_json_retry` pops and re-sends.
        if text.strip():
            self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        self.last_text = text
        http_ms = round((time.perf_counter() - t0) * 1000)
        _acc_api_ms(http_ms)
        rec = {
            "kind": "conversation",
            "ms": http_ms,  # round-trip + any permit wait / backoff this call served
            "encode_ms": encode_ms,  # LOCAL image-encode CPU (client; NOT in ms)
            "chars": len(text),
            "in_tok": in_tok,
            "out_tok": out_tok,
            "cache_w": cw,
            "cache_r": cr,
        }
        _log_call(rec)
        obj, outcome = _extract_traced(text, schema)
        # trace this turn's NEW blocks (reading the numbered dirs in order replays the conversation)
        self._vlm._trace_call(
            "conversation", self.system, content, text, {**rec, "model": self.model, "schema": schema or "", **outcome}
        )
        # Surfaced AFTER the assistant turn is appended and traced: a reply with text in it happened,
        # so it is part of the conversation, and the retry above re-asks within that history.
        if isinstance(obj, BaseException):
            raise obj
        return obj
