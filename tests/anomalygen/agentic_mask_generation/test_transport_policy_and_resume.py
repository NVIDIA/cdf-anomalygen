# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The endpoint-stop and the resume, tested against each other running for real: a dying endpoint
drains what was already earned, a broken configuration books nothing, a stopped run continued with
the same command makes up exactly the shortfall — and a DIFFERENT command is refused by name."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from anomalygen.agentic_mask_generation import vlm as V
from anomalygen.agentic_mask_generation.config import Config
from anomalygen.agentic_mask_generation.defects import from_dir
from anomalygen.agentic_mask_generation.exemplars import PreflightReport
from anomalygen.scripts.agentic_mask_generation import generate as G


def _mask(n: int) -> Image.Image:
    """A mask whose PIXELS say which beam made it, so "unchanged bytes" is a real comparison."""
    m = Image.new("L", (32, 32), 0)
    m.paste(255, (2, 2, 4 + (n % 20), 4 + (n % 20)))
    return m


def _acc(n, score=9):
    v = SimpleNamespace(passed=True, score=score, scores={"roi": 8})
    return SimpleNamespace(mask=_mask(n), overlay=None, verdict=v)


def _res(masks, judged=None):
    return SimpleNamespace(masks=masks, judged=len(masks) if judged is None else judged, api_calls=1)


def _rows(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


_ONE_REGION = [[0.1, 0.1, 0.3, 0.3]]  # a single valid site: every beam is confined to it, nothing to walk
_FOUR_REGIONS = [[0.1, 0.1, 0.3, 0.3], [0.5, 0.1, 0.7, 0.3], [0.1, 0.5, 0.3, 0.7], [0.5, 0.5, 0.7, 0.7]]


def _drive(monkeypatch, tmp_path, *, out, argv=(), beam=None, concurrency=1, regions=None):
    """Run the REAL `generate.main()` into `out`, against stubs at the boundary only (transport,
    dataset gate, regions, beam); two calls with the same `out` ARE the two processes."""
    ds = tmp_path / "ds"
    (ds / "T" / "clean_image").mkdir(parents=True, exist_ok=True)
    # digit/underscore-bearing stems, the shape real boards have: the resume's reuse counter must
    # parse `board_2_v0_1_0_mask.png` from the RIGHT (`_split_indexed`), or the byte checks below fail
    for i in range(2):
        Image.new("RGB", (32, 32), (30 + 10 * i, 40, 50)).save(ds / "T" / "clean_image" / f"board_2_v{i}.png")
    made = []
    inner = beam or (lambda clean, dtype, **kw: _res([_acc(len(made))]))

    def _beam(clean, dtype, **kw):  # every beam ATTEMPTED, including one that raises
        made.append((Path(clean).stem, dtype))
        return inner(clean, dtype, **kw)

    def _pf(ds, ts, *a, **k):
        return PreflightReport(len(ts), len(ts), 0, 0.0, per_type={f"{t}+{d}": 1 for t, d in ts})

    class _Vlm:
        image_max_dim = 0

        def __init__(self, key, base, **kw):
            pass

        def call_json(self, system, blocks, **k):  # the capability probe, answered by counting
            return {"n_images": sum(isinstance(b, Image.Image) for b in blocks)}

        def endpoint_report(self):
            return {}

        def endpoint_summary(self):
            return "ok"

    monkeypatch.setattr(G, "preflight", _pf)
    monkeypatch.setattr(G, "Vlm", _Vlm)
    monkeypatch.setattr(G, "_route_shape", lambda *a, **k: False)
    monkeypatch.setattr(G, "run_beam", _beam)
    monkeypatch.setattr(G, "_load_defects", lambda *a, **k: from_dir(None))
    monkeypatch.setattr(G, "load_exemplars", lambda *a, **k: [])
    monkeypatch.setattr(G, "enumerate_regions", lambda *a, **k: [list(r) for r in (regions or _ONE_REGION)])
    monkeypatch.setattr(G, "init_loguru_file", lambda p: None)
    reg = tmp_path / "endpoints.yaml"  # the default models must resolve in the ONE registry
    reg.write_text("- model: azure/openai/gpt-6-astra\n  base_url: https://llm-gateway.example/v1\n  key: test-key\n")
    args = ["generate", "--dataset-path", str(ds), "--anomaly-types", "T+D", "--out-dir", str(out)]
    args += ["--cache-dir", str(tmp_path / "cache"), "--visual-prompt", "none", "--model-registry", str(reg)]
    args += ["--gen-model", "azure/openai/gpt-6-astra", "--judge-model", "azure/openai/gpt-6-astra"]
    args += ["--max-concurrent-calls", str(concurrency), *argv]
    monkeypatch.setattr(sys, "argv", args)
    exit_ = None
    try:
        G.main()
    except SystemExit as e:
        exit_ = e
    run = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else None
    rows = _rows(out / "testcase.jsonl")
    return SimpleNamespace(out=out, exit=exit_, beams=made, rows=rows, run=run)


def _masks_on_disk(out: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(out.rglob("*_mask.png"))}


def _stopping_beam(after: int):
    """A beam that delivers `after` masks and then finds the endpoint gone."""
    n = [0]

    def beam(clean, dtype, **kw):
        n[0] += 1
        if n[0] > after:
            raise V.TransportUnavailable(V.Failure(V.TRANSPORT, "ReadTimeout: the socket closed"), 3, "https://proxy")
        return _res([_acc(n[0])])

    return beam


def test_a_dead_endpoint_stops_the_run_and_what_the_pool_had_already_earned_is_kept(
    monkeypatch, tmp_path, degradations
):
    """Exhaustion reaches the DRIVER and ends the run; every beam that had ALREADY ACCEPTED a mask —
    including the ones still in flight when the endpoint died — delivers it, on disk and in the manifest.
    MUTANT this kills: removing the drain from the scheduler's stop path (rows read 0 while two beams accepted)."""
    seen = threading.Lock()
    n = [0]

    def beam(clean, dtype, **kw):
        with seen:
            n[0] += 1
            mine = n[0]
        if mine == 1:  # the FIRST beam raises at once; the sleepers are reliably still in flight
            raise V.TransportUnavailable(V.Failure(V.TRANSPORT, "ReadTimeout: the socket closed"), 3, "https://proxy")
        time.sleep(0.3)
        return _res([_acc(mine)])

    # 4 slots, target 8: the launch estimate opens 3 beams at once (8 / the beam_width prior of 3),
    # so one raises and two are mid-flight when the stop unwinds the pool.
    did = _drive(monkeypatch, tmp_path, out=tmp_path / "out", beam=beam, concurrency=4, argv=["--num-sdg", "8"])

    assert did.exit is not None and did.exit.code, "a run that could not finish exited 0"
    assert "endpoint stopped answering" in str(did.exit.code), did.exit.code
    assert len(did.beams) == 3, f"{len(did.beams)} beams ran against a dead endpoint; the fill loop should open 3"
    detail = degradations.first("run_stopped:transport")
    assert "ReadTimeout" in detail and "continues from here" in detail, detail
    # WHAT WAS PAID FOR IS KEPT — even by the beams nobody had collected when the stop unwound the pool
    assert len(did.rows) == 2, f"{len(did.rows)} row(s): beams in flight at the stop were discarded un-booked"
    assert all(Path(r["mask_filename"]).exists() for r in did.rows), did.rows
    assert did.run["n_masks"] == 2, did.run["n_masks"]


def test_the_broken_configuration_abort_does_NOT_book_the_beams_behind_it(monkeypatch, tmp_path):
    """Draining is for the ENDPOINT stop only: the repeated-failure guard claims nothing was ever
    accepted, so nothing may be booked behind it past the type's target."""
    seen = threading.Lock()
    n = [0]

    def beam(clean, dtype, **kw):
        with seen:
            n[0] += 1
            mine = n[0]
        if mine <= 5:  # one cause, five times, with nothing accepted -> the configuration
            raise V.CallFailed(V.Failure(V.TRANSPORT, "ReadTimeout: timed out after 180s"), 3)
        time.sleep(0.3)  # still in flight when the guard above fires
        return _res([_acc(mine)])

    did = _drive(monkeypatch, tmp_path, out=tmp_path / "out", beam=beam, concurrency=4, argv=["--num-sdg", "8"])

    assert did.exit is not None and "this is the configuration" in str(did.exit.code), did.exit
    assert did.rows == [], "the repeated-failure abort booked the beams behind it, past a target it says was never met"


class _ScriptedWire:
    """Stand-in for the pooled `httpx.Client`: answers each POST with the next scripted (status, body)."""

    def __init__(self, replies):
        self.replies, self.posts = list(replies), []

    def post(self, url, headers=None, json=None, **kw):
        self.posts.append(json)
        status, body = self.replies.pop(0)
        if isinstance(body, dict):  # the envelope itself
            d = body
        elif url.endswith("/messages"):  # the assistant's content, in the envelope of the route posted to
            d = {"content": [{"type": "text", "text": body}], "usage": {}}
        else:
            d = {"choices": [{"message": {"content": body}}], "usage": {}}
        return SimpleNamespace(status_code=status, headers={}, text=str(body), json=lambda: d)


def test_a_refused_attempt_is_booked_under_the_permit_waited_outside_it_and_never_re_asked_above(
    monkeypatch, degradations
):
    """`_with_policy`'s documented order — permit, attempt, classify, decide, wait, repeat — on the REAL
    `Vlm`: every attempt counts, the failure is booked BY KIND and the shared window registered while
    the permit is still HELD, the permit is released BEFORE the wait, a 400 aborts on its first attempt,
    and the third failure stops the run opening no window for a wait nobody takes. Above the transport,
    `call_json`'s one re-ask is for a reply that ARRIVED unusable — never for a refusal.
    MUTANT this kills: `ep.observe_failure(...)` moved below the `with ep:` block (booked with the
    permit already released), or `time.sleep(act.after)` moved into it (waited holding the permit)."""
    base = "https://llm-gateway.example/v1"
    vlm = V.Vlm("their-key", base, trace_dir=None)
    ep = vlm._healths[base] = V.Endpoint(1, base_wait=0.1, name="gw")  # a short step: the ORDER is the claim
    seen: list[tuple] = []
    real_observe = V.Endpoint.observe_failure
    monkeypatch.setattr(
        V.Endpoint,
        "observe_failure",
        lambda self, f, attempt, more=None: (
            seen.append(("booked", self._live, f.kind)) or real_observe(self, f, attempt, more)
        ),
    )
    monkeypatch.setattr(V.time, "sleep", lambda s: seen.append(("waited", ep._live, s)))
    V._local.count, V._local.failed = 0, {}

    def ask(*replies):
        vlm._http = wire = _ScriptedWire(replies)
        return vlm.call_json("sys", ["hi"], model="m", schema="smoke"), wire

    # 429 then 200: both attempts counted, ONE refusal booked by kind with the permit held, the wait outside it
    out, wire = ask((429, ""), (200, '{"n_images": 2}'))
    assert out == {"n_images": 2} and len(wire.posts) == 2
    assert V.get_call_count() == 2 and V.get_failed_count() == {V.RATE_LIMITED: 1}
    assert seen == [("booked", 1, V.RATE_LIMITED), ("waited", 0, 0.1)], seen
    assert (ep.report()["calls"], ep.report()["refusals"], ep.report()["waited_s"]) == (2, 1, 0.1), ep.report()
    # a 400: aborted on the FIRST attempt — booked, no window, no second request from either layer
    del seen[:]
    with pytest.raises(V.CallFailed) as cf:
        ask((400, "this model does not support image input"))
    assert (cf.value.attempts, cf.value.failure.kind) == (1, V.BAD_REQUEST), cf.value
    assert len(vlm._http.posts) == 1 and seen == [("booked", 1, V.BAD_REQUEST)], seen
    assert "api_abort:bad_request" in degradations and ep.report()["waited_s"] == 0.1
    # three 5xx: the RUN stops; the last attempt opens no window (0.1 + 0.2 waited, nothing for the third)
    del seen[:]
    with pytest.raises(V.TransportUnavailable) as tu:
        ask((503, ""), (503, ""), (503, ""))
    assert tu.value.attempts == 3 and len(vlm._http.posts) == 3, tu.value
    assert [s for s in seen if s[0] == "waited"] == [("waited", 0, 0.1), ("waited", 0, 0.2)], seen
    assert ep.report()["waited_s"] == 0.4 and "api_gave_up:server_error" in degradations
    # a reply that ARRIVED unusable is the JSON layer's failure: re-asked once, never booked as a refusal
    del seen[:]
    out, wire = ask((200, "two, I think"), (200, '{"n_images": 2}'))
    assert out == {"n_images": 2} and len(wire.posts) == 2 and seen == [], seen
    assert V.get_failed_count() == {V.RATE_LIMITED: 1, V.BAD_REQUEST: 1, V.SERVER_ERROR: 3}


@pytest.mark.parametrize(
    "base, envelope",
    [
        ("https://llm-gateway.example/v1", {"choices": [], "usage": {}}),
        ("https://llm-gateway.example/v1", {"usage": {}}),
        ("https://llm-gateway.example/v1", {"choices": [{"finish_reason": "stop"}], "usage": {}}),
        (V.ANTHROPIC_BASE, {"content": [], "usage": {}}),
    ],
    ids=["chat:no_choice", "chat:no_choices_key", "chat:choice_without_message", "messages:no_content"],
)
def test_a_200_whose_envelope_carries_no_reply_is_re_asked_once_above_and_never_booked_as_a_refusal(
    monkeypatch, degradations, base, envelope
):
    """A 200 with nothing in it is a reply that ARRIVED and cannot be used — `call_json`'s one re-ask,
    on either envelope. Booked as a refusal it would be retried under the endpoint's policy instead:
    `_MAX_ATTEMPTS` requests for one call, a backoff window on every thread sharing the endpoint,
    and the run stopped when the third one is empty too.
    MUTANT this kills: `_post_openai` hard-indexing `d["choices"][0]["message"]`."""
    vlm = V.Vlm("their-key", base, trace_dir=None)
    ep = vlm._healths[base] = V.Endpoint(1, base_wait=0.1, name="gw")
    monkeypatch.setattr(V.time, "sleep", lambda s: pytest.fail(f"the transport waited {s}s on a reply that arrived"))
    V._local.count, V._local.failed = 0, {}

    def ask(*replies):
        vlm._http = wire = _ScriptedWire(replies)
        return vlm.call_json("sys", ["hi"], model="m", schema="smoke"), wire

    # empty then good: answered on the second ask, two POSTs, nothing booked against the endpoint
    out, wire = ask((200, envelope), (200, '{"n_images": 2}'))
    assert out == {"n_images": 2} and len(wire.posts) == 2 and V.get_call_count() == 2
    assert V.get_failed_count() == {}, "an empty reply was booked as a transport failure"
    assert (ep.report()["refusals"], ep.report()["waited_s"], ep._until) == (0, 0.0, 0.0), ep.report()
    # empty every time: the JSON layer gives up after its ONE re-ask — two POSTs, and the run goes on
    with pytest.raises(json.JSONDecodeError):
        ask((200, envelope), (200, envelope), (200, envelope))
    assert len(vlm._http.posts) == 2 and "call_gave_up:call_json" in degradations
    assert V.get_failed_count() == {} and ep.report()["refusals"] == 0, ep.report()


def test_a_resumed_run_makes_up_the_shortfall_without_touching_the_first_runs_masks(
    monkeypatch, tmp_path, degradations
):
    """The whole point: stop at 2 of 6, run the same command again, end with 6 — the first run's
    BYTES unchanged, no name used twice.
    MUTANT this kills: `reuse_ctr = {str(cp): 0 for cp in cleans}` — the second process numbers from
    0 again and writes `c0_0_0_mask.png` over the first process's file."""
    out = tmp_path / "out"
    first = _drive(monkeypatch, tmp_path, out=out, beam=_stopping_beam(2), argv=["--num-sdg", "6"])
    assert len(first.rows) == 2 and first.exit is not None
    before = _masks_on_disk(out)
    assert len(before) == 2

    second = _drive(monkeypatch, tmp_path, out=out, argv=["--num-sdg", "6"])
    assert second.exit is None, second.exit
    # ONLY THE SHORTFALL: four more beams, not six.
    assert len(second.beams) == 4, second.beams
    assert len(second.rows) == 6, second.rows
    assert second.run["n_masks"] == 6

    after = _masks_on_disk(out)
    assert len(after) == 6, sorted(after)
    for name, data in before.items():
        assert name in after, f"the resumed run deleted {name}"
        assert after[name] == data, f"the resumed run wrote over {name}"
    # every row names a file that exists, and no two rows name one file
    paths = [r["mask_filename"] for r in second.rows]
    assert len(set(paths)) == 6 and all(Path(p).exists() for p in paths)
    # ...and the run says it was resumed, with what it inherited kept apart from what it spent
    assert second.run["resumed"]["runs"] == 2 and second.run["resumed"]["delivered_before"] == 2
    assert "resumed_run" in degradations

    # ...and a THIRD run into a directory that already has its six does nothing at all
    third = _drive(monkeypatch, tmp_path, out=out, argv=["--num-sdg", "6"])
    assert third.exit is None and third.beams == [], third.beams
    assert len(third.rows) == 6 and _masks_on_disk(out) == after


R0, R1, R2, R3 = _FOUR_REGIONS


@pytest.mark.parametrize(
    "argv, multi, n_first, n_more, first, resumed",
    [
        # one site per beam: three beams per clean take the first three sites, the two the resume
        # adds take the fourth and then wrap
        (["--slot-regions", "1"], False, 3, 2, [[R0], [R1], [R2]], [[R3], [R0]]),
        # two sites per beam (the default): the resumed beam starts at the third site, not the first
        ([], False, 1, 1, [[R0, R1]], [[R2, R3]]),
        # a multi type takes ONE site per beam whatever --slot-regions says, so the resume steps by
        # one — not by the slot count, which would skip the fourth site
        ([], True, 3, 2, [R0, R1, R2], [R3, R0]),
    ],
    ids=["k1", "k2", "multi"],
)
def test_a_resumed_run_continues_the_site_round_robin_where_the_first_run_left_it(
    monkeypatch, tmp_path, argv, multi, n_first, n_more, first, resumed
):
    """--coverage roundrobin hands each beam on a clean the next valid site(s); the first process has
    already handed out the first `n_first` of them, so the resume must continue from there.
    MUTANT this kills: `region_ctr[str(cp)] = 0` — the resumed beams cover the first sites again."""
    if multi:
        monkeypatch.setattr(G, "read_flags", lambda card: {"distribution_kind": "multi"})
    out = tmp_path / "out"
    handed: dict = {}  # clean stem -> the target_region of every beam launched on it, in launch order
    n = [0]
    dies_after = [2 * n_first]  # two cleans, `n_first` beams each, then the endpoint dies; None = healthy

    def beam(clean, dtype, **kw):
        handed.setdefault(Path(clean).stem, []).append(kw["target_region"])
        n[0] += 1
        if dies_after[0] is not None and n[0] > dies_after[0]:
            raise V.TransportUnavailable(V.Failure(V.TRANSPORT, "ReadTimeout: the socket closed"), 3, "https://proxy")
        return _res([_acc(n[0])])

    total = ["--num-sdg", str(2 * (n_first + n_more))]
    one = _drive(monkeypatch, tmp_path, out=out, argv=[*total, *argv], beam=beam, regions=_FOUR_REGIONS)
    assert one.exit is not None and len(one.rows) == 2 * n_first, (one.exit, one.rows)
    stems = sorted(handed)
    assert len(stems) == 2
    assert {s: handed[s][:n_first] for s in stems} == {s: first for s in stems}, handed

    handed.clear()
    dies_after[0] = None
    two = _drive(monkeypatch, tmp_path, out=out, argv=[*total, *argv], beam=beam, regions=_FOUR_REGIONS)
    assert two.exit is None and len(two.rows) == 2 * (n_first + n_more), (two.exit, two.rows)
    assert handed == {s: resumed for s in stems}, handed


def test_a_manifest_line_that_will_not_parse_is_skipped_and_the_rows_around_it_are_kept(tmp_path):
    """The manifest is only ever written whole (`atomic_write`), so a line that will not parse is a
    hand-edited one — and raising on it would turn a resumable directory into a lost one."""
    p = tmp_path / "testcase.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
    assert G._read_jsonl(p) == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize(
    "changed, names",
    [
        (["--accept-threshold", "9"], "config.accept_threshold"),
        # a flag with no Config field of its own — the half a config snapshot cannot see
        (["--max-rounds", "3"], "argv"),
    ],
)
def test_a_run_with_different_settings_is_refused_and_the_refusal_names_the_field(
    monkeypatch, tmp_path, changed, names
):
    """Continuing is only safe for the SAME run; anything else is two configurations silently blended
    into one deliverable, refused by name and without touching the first run's masks.
    MUTANT this kills: gutting `_control_plane_diff` — every other kept resume reuses an identical command."""
    out = tmp_path / "out"
    first = _drive(monkeypatch, tmp_path, out=out, argv=["--num-sdg", "2"])
    assert first.exit is None and len(first.rows) == 2
    before = _masks_on_disk(out)

    # the SAME target, so the only thing that differs is the flag under test
    again = _drive(monkeypatch, tmp_path, out=out, argv=["--num-sdg", "2", *changed])
    assert again.exit is not None, "a different configuration ran into a finished run's directory"
    said = str(again.exit.code)
    assert names in said, said
    assert "--overwrite" in said, "the refusal does not say how to proceed"
    assert again.beams == [], "the refused run still ran beams"
    assert _masks_on_disk(out) == before, "the refused run touched the first run's masks"


def test_swapped_flag_values_are_refused_and_reordered_flags_are_not(monkeypatch, tmp_path):
    """The argv half binds each value to its flag: `--max-rounds 3 --request-timeout 480` and
    `--max-rounds 480 --request-timeout 3` are two configurations, the same pairs in another order
    are one. MUTANT this kills: comparing argv as a sorted multiset of loose tokens — both spellings
    sort to the same four tokens, so the swapped run resumes into the first run's out-dir."""
    out = tmp_path / "out"
    same = ["--num-sdg", "2", "--max-rounds", "3", "--request-timeout", "480"]
    first = _drive(monkeypatch, tmp_path, out=out, argv=same)
    assert first.exit is None and len(first.rows) == 2
    before = _masks_on_disk(out)

    swapped = _drive(
        monkeypatch, tmp_path, out=out, argv=["--num-sdg", "2", "--max-rounds", "480", "--request-timeout", "3"]
    )
    assert swapped.exit is not None, "swapped flag values ran into a finished run's directory"
    said = str(swapped.exit.code)
    assert "--max-rounds" in said and "--request-timeout" in said, said
    assert "--overwrite" in said, "the refusal does not say how to proceed"
    assert swapped.beams == [], "the refused run still ran beams"
    assert _masks_on_disk(out) == before, "the refused run touched the first run's masks"

    reordered = _drive(
        monkeypatch, tmp_path, out=out, argv=["--request-timeout", "480", "--num-sdg", "2", "--max-rounds", "3"]
    )
    assert reordered.exit is None, reordered.exit
    assert reordered.beams == [] and len(reordered.rows) == 2 and _masks_on_disk(out) == before


@pytest.mark.parametrize(
    "was, now, names",
    [
        # the same pairs in another order are one command
        (["--max-rounds", "3", "--request-timeout", "480"], ["--request-timeout", "480", "--max-rounds", "3"], []),
        # `--flag=value` is the pair `--flag value`
        (["--max-rounds", "3"], ["--max-rounds=3"], []),
        # a boolean flag carries no value, wherever it sits
        (["--beam-prune", "--max-rounds", "3"], ["--max-rounds", "3", "--beam-prune"], []),
        # a negative number is a value bound to its flag, not a flag
        (["--a", "-1", "--b", "2"], ["--b", "2", "--a", "-1"], []),
        (["--a", "-1", "--b", "2"], ["--a", "2", "--b", "-1"], ["--a", "--b"]),
        # a multi-value flag keeps the order of its values
        (["--anomaly-types", "A+B", "C+D"], ["--anomaly-types", "C+D", "A+B"], ["--anomaly-types"]),
        # a boolean flag on one side only
        (["--max-rounds", "3"], ["--max-rounds", "3", "--beam-prune"], ["--beam-prune"]),
    ],
)
def test_argv_is_compared_as_flag_value_pairs(was, now, names):
    """`_control_plane_diff`'s argv half, on the shapes argparse accepts: refused iff a (flag, values)
    pair changed, and the refusal names the flag."""
    cfg = Config(anomaly_types=(("T", "D"),), dataset_path="ds")
    prev = {"config": G._config_snapshot(cfg), "argv": ["generate", *was]}
    diffs = G._control_plane_diff(prev, cfg, ["generate", *now])
    assert bool(diffs) == bool(names), diffs
    for flag in names:
        assert flag in " ".join(diffs), diffs


@pytest.mark.parametrize("spelled", ["--skip-smoke", "--sk"], ids=["full", "abbreviated"])
def test_skip_smoke_is_not_part_of_the_recorded_command_so_a_plain_re_run_resumes(monkeypatch, tmp_path, spelled):
    """`--skip-smoke` decides whether the probe runs and shapes nothing the run hands over, so — like
    `--overwrite`, and under every abbreviation argparse resolves to it — it is stripped from the argv
    the run records: a run started with it is continued by the same command WITHOUT it, and no mask
    is touched. MUTANT this kills: the strip matching `--overwrite` alone — the re-run is refused as
    `argv: dropped --skip-smoke`."""
    out = tmp_path / "out"
    first = _drive(monkeypatch, tmp_path, out=out, argv=[spelled, "--num-sdg", "2"])
    assert first.exit is None and len(first.rows) == 2, (first.exit, first.rows)
    assert not any(a.startswith("--sk") for a in first.run["argv"]), first.run["argv"]
    before = _masks_on_disk(out)

    again = _drive(monkeypatch, tmp_path, out=out, argv=["--num-sdg", "2"])
    assert again.exit is None, again.exit
    assert again.beams == [] and len(again.rows) == 2 and _masks_on_disk(out) == before


def test_the_chat_payload_caps_output_as_max_completion_tokens_and_the_anthropic_body_as_max_tokens():
    """The two envelopes name the output cap the way their endpoint requires. On `/chat/completions`
    it is `max_completion_tokens` — the only name OpenAI's own endpoint accepts for a reasoning model
    such as GPT-6 Astra (`max_tokens` is HTTP 400 there), the same field to vLLM, translated per
    backend by a LiteLLM gateway — and the payload carries no sampling parameter, whatever the model.
    On the native `/messages` route the cap is Anthropic's `max_tokens`, unchanged. MUTANT this kills:
    the chat payload naming `max_tokens` again (Astra direct fails every call), or a `temperature`
    reappearing in either body."""
    V._local.count, V._local.failed = 0, {}
    chat = V.Vlm("k", "https://llm-gateway.example/v1", trace_dir=None)
    chat._http = wire = _ScriptedWire([(200, '{"n_images": 1}')])
    chat.call_json("sys", ["hi"], model="azure/openai/gpt-6-astra", max_tokens=1234, schema="smoke")
    (p,) = wire.posts
    assert p["max_completion_tokens"] == 1234 and "max_tokens" not in p, p
    assert "temperature" not in p, p

    native = V.Vlm("k", V.ANTHROPIC_BASE, trace_dir=None)
    native._http = wire = _ScriptedWire([(200, '{"n_images": 1}')])
    native.call_json("sys", ["hi"], model="claude-opus-5", max_tokens=1234, schema="smoke")
    (b,) = wire.posts
    assert b["max_tokens"] == 1234 and "max_completion_tokens" not in b, b
    assert "temperature" not in b, b


def test_a_rate_limit_is_waited_out_with_a_doubling_step_and_an_outage_is_not():
    """`decide` on the two refusals that ARE worth another request. A 429 is a per-minute budget that
    refills, so its wait doubles -- base, 2x, 4x ... -- and is capped at `_RATE_LIMIT_MAX_WAIT`, with a
    `Retry-After` honoured up to the same cap. A 5xx keeps the short linear step and the endpoint's
    own `max_wait`. MUTANT this kills: a 429 back on the linear step (three tries inside one closed
    window), or the cap applied to `Retry-After` before the step."""
    f429 = V.classify_status(429, '{"error":{"message":"exceeded token rate limit","type":"too_many_requests"}}', None)
    assert f429.kind == V.RATE_LIMITED
    waits = [V.decide(f429, a, base=5.0, max_wait=60.0).after for a in range(1, 9)]
    assert waits == [5, 10, 20, 40, 80, 120, 120, 120], waits
    told = V.classify_status(429, "", "37")
    assert V.decide(told, 1, base=5.0, max_wait=60.0).after == 37  # the server's number wins over the step
    assert V.decide(V.classify_status(429, "", "900"), 1, base=5.0, max_wait=60.0).after == 120  # ... up to the cap
    f503 = V.classify_status(503, "", None)
    assert [V.decide(f503, a, base=5.0, max_wait=60.0).after for a in (1, 2, 3)] == [5, 10, 15]


def test_the_call_budget_is_time_for_a_rate_limit_and_attempts_per_kind_for_an_outage():
    """`CallBudget` is what decides whether ONE call may ask again. A rate limit is budgeted in TIME
    waited (`_RATE_LIMIT_MAX_TOTAL_WAIT`), because an attempt count cannot be sized to an outage of
    unknown length -- eight attempts at the 120 s cap is fourteen minutes, and the stretch this was
    written for lasted twenty to thirty. Outages are counted per KIND, so a 5xx inside a throttling
    streak neither ends the call on the outage's count nor buys the throttle extra tries. MUTANT
    this kills: a single attempt counter judged on the latest failure's kind."""
    f429, f503 = V.classify_status(429, "", None), V.classify_status(503, "", None)
    b = V.CallBudget()
    for _ in range(14):  # 14 x 120 s = 28 min: still inside the 30 min budget
        b.record(f429)
        assert b.allows_another(f429)
        b.spent(V.Retry(after=120.0, reason=V.RATE_LIMITED))
    b.spent(V.Retry(after=120.0, reason=V.RATE_LIMITED))  # 30 min waited: the budget is spent
    assert not b.allows_another(f429)
    # a 5xx met in the middle of that streak is judged on ITS OWN count, not on the 429s
    b2 = V.CallBudget()
    assert [b2.record(f) for f in (f429, f429, f503)] == [1, 2, 1] and b2.allows_another(f503)
    b2.record(f503)
    assert b2.allows_another(f503) and not (b2.record(f503) < 3 or b2.allows_another(f503))
    assert b2.requests == 5 and b2.attempts == {V.RATE_LIMITED: 2, V.SERVER_ERROR: 3}


@pytest.mark.parametrize(
    "body",
    [
        '{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}',
        # a LiteLLM gateway wraps the upstream envelope as TEXT inside its own message
        '{"error":{"message":"litellm.RateLimitError: OpenAIException - {\\"error\\": {\\"code\\": \\"insufficient_quota\\"}}","type":null}}',
    ],
)
def test_an_account_with_no_quota_left_stops_the_run_on_the_first_request(body, monkeypatch, degradations):
    """The other thing that arrives as 429: `insufficient_quota` means the account has no credits.
    Waiting for a budget that will never refill only hides the message, and carrying on past the
    call -- the `CallFailed` path every caller degrades through -- would submit beam after beam,
    each buying one refused request, until a type died of `dry_streak`. So it ends the RUN, after
    exactly one request, resumable like any other stop, and the reason names the fix. MUTANT this
    kills: every 429 treated as RATE_LIMITED, or QUOTA_EXHAUSTED raised as CallFailed."""
    f = V.classify_status(429, body, None)
    assert f.kind == V.QUOTA_EXHAUSTED and isinstance(V.decide(f, 1, base=5.0, max_wait=60.0), V.Abort)
    assert V.classify_status(429, '{"error":{"code":"rate_limit_exceeded"}}', None).kind == V.RATE_LIMITED
    base = "https://llm-gateway.example/v1"
    vlm = V.Vlm("their-key", base, trace_dir=None)
    vlm._healths[base] = V.Endpoint(1, base_wait=0.01, name="gw")
    monkeypatch.setattr(V.time, "sleep", lambda s: None)
    V._local.count, V._local.failed = 0, {}
    vlm._http = wire = _ScriptedWire([(429, body)] * 3)
    with pytest.raises(V.TransportUnavailable) as tu:
        vlm.call_json("sys", ["hi"], model="m", schema="smoke")
    assert len(wire.posts) == 1 and tu.value.attempts == 1 and tu.value.failure.kind == V.QUOTA_EXHAUSTED
    assert "no quota left" in degradations.first("api_gave_up:quota_exhausted")  # the reason names the fix


def test_rate_limits_no_longer_end_the_run_and_a_5xx_among_them_does_not_either(monkeypatch, degradations):
    """On the real `Vlm`. Five 429s then a 200 is a SUCCESSFUL call (six requests, five waits that
    double), where three 429s used to stop the run. 429, 429, 503, 200 also succeeds: the outage is
    one attempt of its own kind, not the third of the call. A pure outage still stops after three,
    and a rate-limit streak that outlasts the time budget stops too -- with the request count the
    call actually made. MUTANT this kills: `_with_policy` counting to `_MAX_ATTEMPTS` for every kind,
    or judging the whole call on its latest failure."""
    base = "https://llm-gateway.example/v1"
    vlm = V.Vlm("their-key", base, trace_dir=None)
    ep = vlm._healths[base] = V.Endpoint(1, base_wait=0.01, name="gw")
    slept: list[float] = []
    monkeypatch.setattr(V.time, "sleep", lambda s: slept.append(round(s, 3)))
    V._local.count, V._local.failed = 0, {}
    vlm._http = wire = _ScriptedWire([(429, "")] * 5 + [(200, '{"n_images": 4}')])
    assert vlm.call_json("sys", ["hi"], model="m", schema="smoke") == {"n_images": 4}
    assert len(wire.posts) == 6 and slept == [0.01, 0.02, 0.04, 0.08, 0.16], (len(wire.posts), slept)
    assert V.get_failed_count() == {V.RATE_LIMITED: 5} and ep.report()["refusals"] == 5
    # an outage in the middle of the streak: its own first attempt, and the 429 step is not reset
    del slept[:]
    vlm._http = wire = _ScriptedWire([(429, ""), (429, ""), (503, ""), (200, '{"n_images": 1}')])
    assert vlm.call_json("sys", ["hi"], model="m", schema="smoke") == {"n_images": 1}
    assert len(wire.posts) == 4 and slept == [0.01, 0.02, 0.01], slept  # 429 step 1, 429 step 2, 503 step 1
    # a pure outage keeps its short budget
    vlm._http = wire = _ScriptedWire([(503, "")] * 3)
    with pytest.raises(V.TransportUnavailable) as tu:
        vlm.call_json("sys", ["hi"], model="m", schema="smoke")
    assert tu.value.attempts == 3 and len(wire.posts) == 3
    # a rate-limit streak that outlasts the TIME budget: bound the waits so the test is instant.
    # Waits 0.01 + 0.02 + 0.04 = 0.07 >= 0.05 after the third refusal, so the fourth request is the
    # last: 4 posts, attempts == 4, and the stop is booked as a rate limit.
    monkeypatch.setattr(V, "_RATE_LIMIT_MAX_TOTAL_WAIT", 0.05)
    vlm._http = wire = _ScriptedWire([(429, "")] * 50)
    with pytest.raises(V.TransportUnavailable) as tu:
        vlm.call_json("sys", ["hi"], model="m", schema="smoke")
    assert tu.value.failure.kind == V.RATE_LIMITED and len(wire.posts) == tu.value.attempts == 4, len(wire.posts)
    assert "api_gave_up:rate_limited" in degradations
    # and the REQUEST ceiling ends the loop even when no wait ever accrues (the float never fills)
    monkeypatch.setattr(V, "_RATE_LIMIT_MAX_TOTAL_WAIT", 1e9)
    monkeypatch.setattr(V, "_MAX_RATE_LIMIT_REQUESTS", 6)
    vlm._healths[base] = V.Endpoint(1, base_wait=0.0, name="gw")  # a zero step: every wait is 0 s
    vlm._http = wire = _ScriptedWire([(429, "")] * 50)
    with pytest.raises(V.TransportUnavailable) as tu:
        vlm.call_json("sys", ["hi"], model="m", schema="smoke")
    assert len(wire.posts) == tu.value.attempts == 6, len(wire.posts)
