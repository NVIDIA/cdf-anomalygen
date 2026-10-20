# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI-safe unit tests for agentic_mask_generation — no VLM, no network, no external data.

Every drive below runs the REAL engine (`run_beam`, `generate.main`) against scripted stubs and
asserts on observable behaviour: files written, manifests trimmed, ledger lines parsed back."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from anomalygen.agentic_mask_generation import loop as L
from anomalygen.agentic_mask_generation import ops as O
from anomalygen.agentic_mask_generation import regions as R
from anomalygen.agentic_mask_generation import vlm as V
from anomalygen.agentic_mask_generation.card_distill import read_flags
from anomalygen.agentic_mask_generation.config import DIMS, Config
from anomalygen.agentic_mask_generation.defects import from_dir
from anomalygen.agentic_mask_generation.exemplars import Exemplar, patch_histogram

_D8 = {d: 8 for d in DIMS}
_DISK = {"kind": "disk", "center": [0.5, 0.5], "radius": 0.12}


def test_the_engine_reads_no_environment():
    """Everything reaches the engine via `Config` or explicit arguments — never the environment —
    so a run's recorded configuration determines its behaviour and two "identical" arms cannot
    differ by a shell variable no run log shows. The driver (`scripts/.../generate.py`) is the one
    place AMG_* is read. `config.py` and `vlm.py` cite this test by name."""
    import ast

    import anomalygen.agentic_mask_generation as pkg

    offenders = []
    for path in sorted(Path(pkg.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr in {"environ", "environb", "getenv"}
            ):
                offenders.append(f"{path.name}:{node.lineno}: os.{node.attr}")
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                offenders.append(f"{path.name}:{node.lineno}: from os import {', '.join(a.name for a in node.names)}")
    assert not offenders, f"the engine read the environment: {offenders}"


def test_flags_read_current_and_legacy_spellings():
    """fill_host reads as exactly part/whole/n_a — any other spelling reads as NO flag — and a card
    already in the field (fill_target / roi_kind front-matter with region-valued answers) reads back
    as the current flags, so an existing library behaves identically without being re-distilled."""
    for v in ("part", "whole", "n_a"):
        assert read_flags(f"---\nfill_host: {v}\n---\n# card")["fill_host"] == v
    for absent in ("total", "WHOLE", "unknown", ""):
        assert "fill_host" not in read_flags(f"---\nfill_host: {absent}\n---\n# card")
    assert read_flags("# a card with no front-matter") == {}
    legacy = read_flags("---\nfill_target: whole\nroi_kind: multi_region\nhost_relation: match\n---\n# card")
    assert legacy == {"fill_host": "whole", "hosts_presented": "multi_host"}
    assert read_flags("---\nroi_kind: single_region\n---\n# card")["hosts_presented"] == "single_host"


def test_the_shape_policy_decides_the_router_the_mode_and_the_card_ban(monkeypatch, tmp_path):
    """--shape auto: a card judging fill_host=whole (ANY sites_spanned; the legacy fill_target
    spelling included) forces flexible without asking the router; every other card (none, and an
    unreadable one, included) goes to the router. --shape flexible/fixed decide every type without
    the router, and fixed OUTRANKS the card ban."""

    from anomalygen.scripts.agentic_mask_generation import generate as G

    asked = []
    monkeypatch.setattr(G, "_route_shape", lambda *a, **k: asked.append(a) or True)

    def cfgp(policy):
        return SimpleNamespace(shape_policy=policy, dataset_path=str(tmp_path), gen_model="m")

    def lib(fm):
        return SimpleNamespace(card=lambda t: f"---\n{fm}\n---" if fm else None)

    banned = (
        "fill_host: whole\nsites_spanned: one",  # one whole host instance
        "fill_host: whole\nsites_spanned: two_or_more",  # several whole host instances — still whole
        "fill_host: whole",  # the ban reads fill_host alone
        "fill_target: whole",  # a card already in the field, legacy spelling
    )
    for fm in banned:
        assert G._resolve_shape_mode(None, cfgp("auto"), "T", "D", lib(fm)) == "flexible", fm
    assert asked == [], "the router was asked about a type the card already decided"
    unbanned = (
        "fill_host: part\nsites_spanned: one",  # part of one host — a stamped shape is legitimate
        "fill_host: n_a",  # no discrete host at all
        "",  # no card
    )
    for fm in unbanned:
        assert G._resolve_shape_mode(None, cfgp("auto"), "T", "D", lib(fm)) == "fixed", fm
    assert len(asked) == len(unbanned), "an unbanned combination did not reach the router"

    def _raises(t):
        raise OSError("unreadable card")

    assert G._resolve_shape_mode(None, cfgp("auto"), "T", "D", SimpleNamespace(card=_raises)) == "fixed"
    assert len(asked) == len(unbanned) + 1, "an unreadable card did not fail safe to the router"

    del asked[:]
    for fm in (*banned, *unbanned):
        assert G._resolve_shape_mode(None, cfgp("flexible"), "T", "D", lib(fm)) == "flexible", fm
        assert G._resolve_shape_mode(None, cfgp("fixed"), "T", "D", lib(fm)) == "fixed", fm
    assert asked == [], "a forced policy still asked the router"


def test_the_shape_router_reads_the_card_and_asks_conservatively(monkeypatch, tmp_path):
    """The route call carries the card's three geometry flags and its typical-shape prose (no card →
    the image-only ask), selects its exemplars DINOv2-similar to the reference clean, answers fixed
    only on a clearly intrinsic shape, and fails safe to flexible on an unparseable answer or a
    failed call."""
    from anomalygen.scripts.agentic_mask_generation import generate as G

    seen = {}

    def _load(*a, **k):
        seen["load"] = k
        return []

    monkeypatch.setattr(G, "load_exemplars", _load)

    class _V:
        def __init__(self, out):
            self._out = out

        def call_json(self, system, blocks, model=None, max_tokens=0, **k):
            seen["system"], seen["blocks"] = system, blocks
            if isinstance(self._out, Exception):
                raise self._out
            return self._out

    card = (
        "---\nhosts_presented: multi_host\nfill_host: part\nsites_spanned: two_or_more\n---\n"
        "# c\n## typical shape\nAn elongated lobe with a narrow neck.\n"
    )
    routed = G._route_shape(
        _V({"fixed_shape": "false", "why": "spans two joints"}),
        str(tmp_path),
        "T",
        "D",
        model="m",
        card=card,
        clean_path="ref.png",
        emb_cache_dir=str(tmp_path),
    )
    assert routed is False
    assert seen["load"]["sampling"] == "similar" and seen["load"]["clean_path"] == "ref.png"
    joined = "\n".join(b for b in seen["blocks"] if isinstance(b, str))
    assert "sites_spanned=two_or_more" in joined and "An elongated lobe with a narrow neck." in joined
    assert "ONLY when the shape is CLEARLY intrinsic" in seen["system"] and '"why"' in seen["system"]
    assert G._route_shape(_V({"fixed_shape": True, "why": "a die-cut logo"}), str(tmp_path), "T", "D", model="m")
    assert "reference card" not in seen["system"], "a card-less route still claimed a card"
    assert not any("Reference card" in b for b in seen["blocks"] if isinstance(b, str))
    for bad in ({"fixed_shape": "maybe"}, {}, RuntimeError("endpoint down")):
        assert G._route_shape(_V(bad), str(tmp_path), "T", "D", model="m", card=card) is False, bad


def test_a_failed_route_verdicts_flexible_and_is_ledgered(monkeypatch, tmp_path, degradations):
    """A route whose internals raise still verdicts flexible (the fail-safe), and the failure is a
    `degraded[shape_route_failed]` line naming the type and the exception — so a broken router is
    distinguishable from a deliberate flexible answer in the run's own record."""
    from anomalygen.scripts.agentic_mask_generation import generate as G

    def _boom(*a, **k):
        raise RuntimeError("exemplar store offline")

    monkeypatch.setattr(G, "load_exemplars", _boom)
    assert G._route_shape(None, str(tmp_path), "T", "D", model="m") is False
    assert degradations.first("shape_route_failed") == "[T+D] RuntimeError: exemplar store offline"


def test_the_leak_guard_polices_only_words_the_engine_wrote(tmp_path):
    """PROVENANCE: a card and a distill caption may say anything — "a dark ring", "moisture can
    erode the pad" — without failing a single turn, and the exempt text is still SENT (exemption ≠
    deletion). The restricted-vocabulary arm is what exercises the exemption; the fixed-shape arm
    pins that the same text still rides the rigid turn. The guard is not neutered: an engine block
    doctored to offer a withheld op still raises, in both halves of the propose turn and in the
    judge blocks."""
    from anomalygen.agentic_mask_generation import context as C
    from anomalygen.agentic_mask_generation import judging as J
    from anomalygen.agentic_mask_generation.context import BeamContext
    from anomalygen.agentic_mask_generation.imaging import Views

    clean_p = tmp_path / "clean.png"
    Image.fromarray(np.full((96, 96, 3), 60, np.uint8)).save(clean_p)
    m = np.zeros((96, 96), np.uint8)
    m[30:50, 30:50] = 255
    card = (
        "---\nfill_host: part\nsites_spanned: two_or_more\n---\n# T+D\n"
        "## typical shape\nA dark ring; moisture can erode the pad outline.\n"
    )
    exm = [
        Exemplar(
            defect=Image.fromarray(np.full((96, 96, 3), 60, np.uint8)),
            mask=Image.fromarray(m, "L"),
            observation="flux can erode the fillet here",
        )
    ]

    def ctx(**kw):
        return BeamContext.build(
            str(clean_p),
            "T+D",
            visual_prompt=_Cue(),
            defects=_Cue(card=card),
            cfg=_cfg(tmp_path, restate_shape_target=True, **kw),
            exemplars=exm,
        )

    # restricted vocabulary (no erode/ring), then the rigid fixed-shape vocabulary
    for kw in ({"allowed_cv_ops": ("disk", "translate", "clear")}, {"shape_policy": "fixed", "shape_mode": "fixed"}):
        c = ctx(**kw)
        sent = "\n".join(b for b in c.propose_context(None) + c.propose_ask() if isinstance(b, str))
        assert "can erode the pad outline" in sent, kw  # the card block rode the turn
        assert "can erode the fillet" in sent, kw  # the caption rode it too
        turn = L._refine_ask(
            c,
            [Image.fromarray(m, "L")],
            t=0,
            last=False,
            review_blocks=[],
            last_area=[],
            judge_note="judge: you can erode the overshoot",
            op_outcomes=[],
            board="b",
        )
        assert any(isinstance(b, str) and "can erode the pad outline" in b for b in turn), kw
    for order in (False, True):  # cache_order reorders the groups; the exemption must survive it
        jb = J._build_judge_blocks(
            exm,
            Image.open(clean_p),
            Image.fromarray(m, "L"),
            "T+D",
            defect_docs=card,
            views=Views(0),
            cache_order=order,
        )
        assert any(isinstance(b, str) and "can erode the fillet" in b for b in jb), order

    # the counter-pins: an ENGINE block offering a withheld op still raises, on every surface
    mk = pytest.MonkeyPatch()
    c = ctx(allowed_cv_ops=("disk", "translate"))
    try:
        mk.setattr(C, "describe_geometry", lambda g: "you can erode it")
        with pytest.raises(AssertionError, match="propose_context"):
            c.propose_context(None)
    finally:
        mk.undo()
    try:
        mk.setattr(C, "ex_dist_summary", lambda ex: "- erode: shrink it inward")
        with pytest.raises(AssertionError, match="propose_ask"):
            c.propose_ask()
    finally:
        mk.undo()
    try:
        mk.setattr(J, "structural_facts", lambda *a: '"kind":"erode"')
        with pytest.raises(AssertionError, match="judge_blocks"):
            J._build_judge_blocks(exm, Image.open(clean_p), Image.fromarray(m, "L"), "T+D", views=Views(0))
    finally:
        mk.undo()


def test_mixed_block_frames_offer_no_op_with_neutral_filler():
    """The engine FRAMING of each runtime-exempt block is itself op-free. At runtime the whole
    rendered block is foreign (one card or caption must not kill a beam), which would hide a frame
    regression, so the templates are pinned here with neutral filler."""
    from anomalygen.agentic_mask_generation.card_distill import exemplar_note
    from anomalygen.agentic_mask_generation.context import _CARD_CAPTION
    from anomalygen.agentic_mask_generation.judging import _DEFECT_DOCS_JUDGE_CAPTION
    from anomalygen.agentic_mask_generation.prompts import leaked_ops, shape_target, weak_dim_steer

    assert leaked_ops(_CARD_CAPTION, ()) == []
    assert leaked_ops(_DEFECT_DOCS_JUDGE_CAPTION, ()) == []
    assert leaked_ops(exemplar_note(1, "neutral words"), ()) == []
    card = "## typical shape\nan irregular elongated patch\n## typical size / extent\nabout one pad-pitch wide\n"
    assert leaked_ops(shape_target(card), ()) == []
    assert leaked_ops(weak_dim_steer({d: 8 for d in DIMS}, ("fill",), "refine", "neutral words"), ()) == []


def test_a_whole_host_card_withholds_the_stamp_op_except_under_shape_fixed(tmp_path):
    """The op-level half of the ban: flexible mode still offers `stamp`, so a fill_host=whole card
    withdraws the op itself — no stamp on any prompt surface, under auto and flexible policy alike
    and for ANY sites_spanned — while a part-host type keeps the menu. Under --shape fixed the CLI
    outranks the ban: stamp stays offered for the SAME card and the rigid-only beam is satisfiable
    (real shapes to stamp, the menu and the stamp-only instruction both present)."""
    from anomalygen.agentic_mask_generation.context import BeamContext

    clean_p = tmp_path / "clean.png"
    Image.fromarray(np.full((64, 64, 3), 60, np.uint8)).save(clean_p)
    m = np.zeros((64, 64), np.uint8)
    m[20:40, 20:40] = 255
    exm = [Exemplar(defect=Image.fromarray(np.full((64, 64, 3), 60, np.uint8)), mask=Image.fromarray(m, "L"))]

    def ctx(fm, **cfg_kw):
        return BeamContext.build(
            str(clean_p),
            "T+D",
            visual_prompt=_Cue(),
            defects=_Cue(card=f"---\n{fm}\n---"),
            cfg=_cfg(tmp_path, **cfg_kw),
            exemplars=exm,
        )

    for policy in ("auto", "flexible"):
        for sites in ("one", "two_or_more"):
            blocked = ctx(f"fill_host: whole\nsites_spanned: {sites}", shape_policy=policy)
            assert "stamp" not in blocked.allowed, (policy, sites)
            text = (
                blocked.proposer_system
                + blocked.judge_system
                + "\n".join(b for b in blocked.propose_ask() if isinstance(b, str))
            )
            assert "stamp" not in text.lower(), f"a banned type's prompts still mention stamping ({policy}, {sites})"
    for policy in ("auto", "flexible"):
        offered = ctx("fill_host: part\nsites_spanned: one", shape_policy=policy)
        assert "stamp" in offered.allowed, policy
        assert any("STAMP" in b for b in offered.propose_ask() if isinstance(b, str)), "the menu is gone for everyone"

    fixed = ctx("fill_host: whole\nsites_spanned: one", shape_policy="fixed", shape_mode="fixed")
    assert fixed.rigid_only and "stamp" in fixed.allowed and fixed.shapes
    ask = "\n".join(b for b in fixed.propose_ask() if isinstance(b, str))
    assert "Available STAMP shapes" in ask and "FIXED-SHAPE DEFECT" in ask


def test_the_shape_flag_lands_in_the_run_record_and_the_force_flags_are_gone(monkeypatch, tmp_path):
    """--shape fixed reaches Config.shape_policy — run.json records the POLICY, beside the per-type
    resolved shape_mode in each results row — and the retired --force-*-shape spellings are refused
    by argparse."""
    did = drive_generate(monkeypatch, tmp_path, accepted=1, argv=["--shape", "fixed", "--num-sdg", "1"])
    assert did.exit is None, did.exit
    rec = did.json("run.json")
    assert rec["config"]["shape_policy"] == "fixed"
    assert {r["shape_mode"] for r in rec["results"]} == {"fixed"}, rec["results"]
    for retired in ("--force-fixed-shape", "--force-flexible-shape"):
        did = drive_generate(monkeypatch, tmp_path / retired.strip("-"), argv=[retired])
        assert did.exit is not None and did.exit.code == 2, f"argparse accepted {retired}"


def _v(n, **over):
    """A verdict in the shape `judge_consensus` returns, scoring `n` on every dimension."""
    return {"score": n, **{d: n for d in DIMS}, "reasons": {}, "feedback": "", **over}


def _cfg(tmp, **kw):
    """A small one-beam Config over `tmp`, overridable per keyword."""
    base = {"max_refine_rounds": 2, "num_candidates": 1, "beam_width": 1, "final_judge_samples": 1}
    return Config(anomaly_types=(("board", "spike"),), dataset_path=str(tmp), **{**base, **kw})


class _Cue:
    """Visual-prompt / defects stub; without a label map the layer ops are withdrawn."""

    def __init__(self, card="", layers=None):
        self._card, self._layers = card, layers

    def build(self, *a, **k):
        return []

    def build_for_judge(self, *a, **k):
        return []

    def layers(self, *a, **k):
        return {} if self._layers is None else self._layers

    def card(self, *a, **k):
        return self._card

    def docs(self, *a, **k):
        return []


class _Conv:
    """A conversation whose `ask_json` defers to `ask(conv, blocks, max_tokens, schema)`."""

    system, model = "s", "m"

    def __init__(self, ask):
        self.messages = []
        self._ask = ask

    def ask_json(self, blocks, max_tokens=1200, cache_prefix=0, schema=None):
        return self._ask(self, blocks, max_tokens, schema)


class _SVlm:
    """A scripted transport: `ask` answers conversations, `call(blocks, max_tokens, schema)` answers
    `call_json`."""

    image_max_dim = 0  # `Views.from_vlm` reads the delivered-size cap off the transport

    def __init__(self, ask=None, call=None):
        self._ask, self._call = ask, call

    def conversation(self, system, model=None, pin_first_images=False):
        return _Conv(self._ask)

    def call_json(self, system, blocks, model=None, max_tokens=1200, cache_prefix=0, schema=None):
        return self._call(blocks, max_tokens, schema)


class _Wire:
    """A scripted HTTP client under the REAL `Vlm` (`vlm._http = _Wire([...])`): each post answers with the
    next reply text in a /chat/completions envelope, and keeps the payload it was sent."""

    def __init__(self, replies):
        self.replies, self.posts = list(replies), []

    def post(self, url, headers=None, json=None, **kw):
        self.posts.append(json)
        d = {"choices": [{"message": {"content": self.replies.pop(0)}}], "usage": {}}
        return SimpleNamespace(status_code=200, headers={}, text="", json=lambda: d)


class _DriverRun:
    """What one `generate.main()` actually DID: side effects in order, and the files left behind."""

    def __init__(self, out):
        self.out = out
        self.order: list[str] = []
        self.exit: SystemExit | None = None

    def _did(self, what):
        self.order.append(what)

    def json(self, name):
        return json.loads((self.out / name).read_text())

    def lines(self, name):
        p = self.out / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def drive_generate(monkeypatch, tmp_path, *, argv=(), types=("T+D",), accepted=0, clean_ext=".png"):
    """Run the REAL `generate.main()` end to end against recording stubs at the model/dataset/beam
    boundary, and report what it did. The clean pool is read by the real dataset IO; `clean_ext` is
    the extension its two cleans are written under."""
    import sys

    from anomalygen.agentic_mask_generation.exemplars import PreflightReport
    from anomalygen.scripts.agentic_mask_generation import generate as G

    ds = tmp_path / "ds"
    for t in types:
        d = ds / t.split("+")[0] / "clean_image"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            Image.new("RGB", (8, 8)).save(d / f"c{i}{clean_ext}")
    out = tmp_path / "out"
    did = _DriverRun(out)

    def _preflight(dataset_path, anomaly_types, cleans):
        did._did("preflight")
        per_type = {f"{t}+{d}": 1 for t, d in anomaly_types}  # weights the proportional validation split
        n = len(anomaly_types)
        return PreflightReport(types=n, pairs=n, cleans=len(cleans or ()), seconds=0.0, per_type=per_type)

    class _Vlm:
        def __init__(self, key, base, **kw):
            did._did("Vlm")

        def call_json(self, system, blocks, **k):
            # the capability probe: answer by COUNTING the images handed over
            did._did("smoke")
            return {"n_images": sum(isinstance(b, Image.Image) for b in blocks)}

        def endpoint_report(self):
            return {"https://e": {"calls": 1, "refusals": 0}}

        def endpoint_summary(self):
            return "endpoints: ok"

    def _run_type(vlm, cfg, texture, defect, cleans, vp_for, defects, **kw):
        did._did("_run_type")
        made = []
        for j in range(accepted):
            v = SimpleNamespace(passed=True, score=9 - j, scores={"roi": 8, "semantic": 9 - j})
            m = SimpleNamespace(mask=Image.new("L", (8, 8), 255), overlay=None, verdict=v)
            item = (cleans[j % len(cleans)], 0, j, m)
            made.append(item)
            if kw.get("on_accept"):
                kw["on_accept"](*item)
        return made, len(made)

    # `_run_usage` starts from THIS thread's counters; zero them so earlier tests don't bleed in
    V._local.count, V._local.failed, V._local.usage, V._local.api_ms = 0, {}, None, 0.0

    # A CLAUDE id here, on purpose, while the other three CLI fixtures use the shipped default
    # (azure/openai/gpt-6-astra): a claude-named model is the one that carries cache markers on the
    # chat envelope -- image-block marker forwarding, the sanitiser's marker copy, the cache_prefix
    # marker -- and this is the fixture that drives exemplar images and cache_prefix through the CLI.
    # On Astra `_is_claude` is false and none of that code is reached.
    reg = tmp_path / "endpoints.yaml"  # the default models must resolve in the ONE registry
    reg.write_text(
        "- model: azure/anthropic/claude-opus-5\n  base_url: https://llm-gateway.example/v1\n  key: test-key\n"
    )
    monkeypatch.setattr(G, "preflight", _preflight)
    monkeypatch.setattr(G, "Vlm", _Vlm)
    monkeypatch.setattr(G, "_route_shape", lambda *a, **k: False)
    monkeypatch.setattr(G, "_run_type", _run_type)
    monkeypatch.setattr(G, "_load_defects", lambda *a, **k: did._did("_load_defects") or from_dir(None))
    monkeypatch.setattr(G, "run_beam", lambda *a, **k: did._did("run_beam"))
    monkeypatch.setattr(G, "init_loguru_file", lambda p: None)
    monkeypatch.setattr(G, "_RUN_USAGE", G._UsageLedger())  # run usage is the BEAMS' running total
    args = ["generate", "--dataset-path", str(ds), "--anomaly-types", *types, "--out-dir", str(out)]
    args += ["--model-registry", str(reg)]
    args += ["--gen-model", "azure/anthropic/claude-opus-5", "--judge-model", "azure/anthropic/claude-opus-5"]
    monkeypatch.setattr(sys, "argv", args + ["--visual-prompt", "none", *argv])
    try:
        G.main()
    except SystemExit as e:
        did.exit = e
    return did


class _BeamRun:
    """What one `run_beam` did: the turns sent, the judge calls made, what came back."""

    def __init__(self):
        self.turns: list = []  # every `blocks` list handed to conv.ask_json, in call order
        self.budgets: list = []  # ("ask" | "call", schema, max_tokens) of every transport call, in call order
        self.judge: list = []  # (args, kwargs) of every judge_consensus call
        self.res = None


def drive_beam(tmp_path, script, *, layers=None, judge_score=9, target_region=None, **cfg):
    """Run the REAL `run_beam` against a scripted transport: `script(n, blocks) -> dict` answers every
    `conv.ask_json` (n=0 propose, then refine rounds); the judge scores `judge_score` on every
    dimension (the default 9 clears the bar) — or raises it, when it is an exception. Pass `layers`
    (a label map) to keep the layer ops; with the default `{}` stub they are withdrawn. Pass
    `target_region` (a normalized box) to run the beam hard-sited on it — the proposer then authors in
    the crop's frame. Further keywords override `_cfg`'s Config."""
    W, H = 240, 180
    got = _BeamRun()
    clean_p = tmp_path / "board.png"
    Image.fromarray(np.full((H, W, 3), 40, np.uint8)).save(clean_p)
    ex_mask = np.zeros((H, W), np.uint8)
    ex_mask[80:100, 100:140] = 255

    def _ask(conv, blocks, max_tokens, schema):
        got.turns.append(list(blocks))
        got.budgets.append(("ask", schema, max_tokens))
        return script(len(got.turns) - 1, blocks)

    def _call(blocks, max_tokens, schema):
        got.budgets.append(("call", schema, max_tokens))
        return {"work_region": {"x0": 0.2, "y0": 0.2, "x1": 0.8, "y1": 0.8}}

    vlm = _SVlm(ask=_ask, call=_call)

    def _judge(*a, **kw):
        got.judge.append((a, kw))
        if isinstance(judge_score, BaseException):
            raise judge_score
        return _v(judge_score)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(L, "judge_consensus", _judge)
    cue = _Cue(card="", layers=layers)
    exm = [Exemplar(defect=Image.fromarray(np.full((H, W, 3), 60, np.uint8)), mask=Image.fromarray(ex_mask))]
    try:
        got.res = L.run_beam(
            str(clean_p),
            "board+spike",
            vlm=vlm,
            visual_prompt=cue,
            defects=cue,
            cfg=_cfg(tmp_path, **cfg),
            exemplars=exm,
            target_region=target_region,
        )
    finally:
        monkey.undo()
    return got


def test_the_accept_bar_gates_on_semantic_weakest_dim_and_mean_with_gate_only_out_of_the_mean():
    """The whole acceptance rule at thr=8: semantic >= thr, every dim >= thr-1, mean >= thr.
    MUTANT this kills: in `AcceptBar.clears`, each `>=` -> `>` and `thr - 1` -> `thr`."""
    from anomalygen.agentic_mask_generation.judging import AcceptBar

    bar, ten = AcceptBar(thr=8), {d: 10 for d in DIMS}
    assert not bar.clears({})  # UNJUDGED (every judge sample dropped) cannot pass
    assert bar.clears(_D8)  # exactly at the bar on every dimension
    assert not bar.clears({**ten, "semantic": 7})  # semantic has no thr-1 allowance
    assert not bar.clears({**ten, "shape": 6})  # the weakest dimension may not fall below thr-1
    assert bar.clears({**ten, "shape": 7})  # ...at thr-1 it is tolerated while the mean holds
    assert not bar.clears({**_D8, "shape": 7, "fill": 7})  # every dim >= thr-1 but the mean sags under thr
    low_dist = {**_D8, "distribution": 7}  # one dim at thr-1, the rest exactly at the bar
    assert AcceptBar(thr=8, gate_only=("distribution",)).clears(low_dist)  # out of the mean, which holds at 8
    assert not bar.clears(low_dist)  # in the mean it drags it below thr
    assert not AcceptBar(thr=8, gate_only=("distribution",)).clears({**ten, "distribution": 6})  # still vetoes


def test_the_adaptive_stop_needs_every_gate_clear_by_its_margin_and_reads_the_bars_gate_only():
    """Stop-accept only with all three gates clear by accept_margin; stop-reject only with reject_margin
    set and a gate failed by it.
    MUTANT this kills: `reached` handing `_gates` a local `()` instead of `self.bar.gate_only`."""
    from anomalygen.agentic_mask_generation.judging import AcceptBar, AdaptiveStop

    stop = AdaptiveStop(AcceptBar(thr=8))  # accept_margin 1.0, reject_margin 1.5
    assert not stop.reached([])
    assert stop.reached([_v(9)])  # every gate clear by exactly the margin
    assert not stop.reached([_v(9), _v(9, semantic=8)])  # semantic mean 8.5: above the bar, under the margin
    assert not stop.reached([_v(10, shape=7)])  # weakest 7 under its 8, and not failed by 1.5: no stop either way
    assert stop.reached([_v(10, semantic=6)])  # stop-REJECT: the semantic gate failed by >= 1.5
    assert not AdaptiveStop(AcceptBar(thr=8), reject_margin=None).reached([_v(10, semantic=6)])
    on_the_gate = [_v(9, distribution=8)]  # inside the mean it drags it under 9; outside, every gate clears
    assert AdaptiveStop(AcceptBar(thr=8, gate_only=("distribution",))).reached(on_the_gate)
    assert not AdaptiveStop(AcceptBar(thr=8)).reached(on_the_gate)


def test_the_otsu_segmentation_runs_once_per_image_object(monkeypatch):
    """The same clean image is segmented for every candidate across a beam; the Otsu computation
    must run once per image OBJECT, not once per call. A distinct image gets its own computation,
    and a memo entry dies with its image."""
    import gc

    from anomalygen.agentic_mask_generation import judging as J

    calls = {"n": 0}
    real = J.cv2.threshold

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(J.cv2, "threshold", counting)
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).convert("RGB")
    first = J._otsu_structure_labels(img)
    J._otsu_structure_labels(img)
    assert calls["n"] == 1, "the SAME image object was segmented twice"
    other = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).convert("RGB")
    J._otsu_structure_labels(other)
    assert calls["n"] == 2, "a different image must be segmented on its own"
    lab, keep = first
    assert lab.shape == (64, 64) and isinstance(keep, list)  # the memo hands back the real result
    other_key = id(other)
    del other
    gc.collect()
    assert other_key not in J._OTSU_MEMO, "a memo entry outlived its image"


def test_every_component_count_the_judge_reads_drops_specks_at_the_same_floor():
    """`_shape_stats` and `_component_text` render into ONE judge prompt, and `component_count`
    feeds the candidate-vs-exemplar count next to them. A speck sitting between two different
    floors would make the same prompt say "3 region(s)" and "2 SEPARATE regions"; every site
    reads the one module floor, so a speck is in or out for all of them."""
    from anomalygen.agentic_mask_generation import judging as J

    a = np.zeros((200, 200), np.uint8)  # size 40000: floor 0.0002 -> 8 px
    a[20:40, 20:40] = 255
    a[120:140, 120:140] = 255
    a[170:175, 80:82] = 255  # a 10 px speck = 0.00025 of the image, above the floor
    assert int((a[170:175, 80:82] > 0).sum()) == 10 == 0.00025 * a.size
    mask = Image.fromarray(a, "L")
    assert J.MIN_COMPONENT_FRAC == 0.0002
    assert J.component_count(mask) == 3
    assert J._shape_stats(mask).startswith("3 region(s)"), J._shape_stats(mask)
    assert "3 SEPARATE regions" in J._component_text(mask), J._component_text(mask)


def test_a_board_structure_and_a_mask_region_have_their_own_floors():
    """`_otsu_structure_labels` floors the BOARD's bright blobs at `MIN_STRUCTURE_FRAC`, coarser than
    the mask-region floor: a bright speck the region counts would keep is not a structure a mask is
    measured against."""
    from anomalygen.agentic_mask_generation import judging as J

    a = np.zeros((200, 200), np.uint8)  # 40000 px: region floor 8 px, structure floor 32 px
    a[20:50, 20:50] = 255
    a[150:154, 100:105] = 255  # 20 px = 0.0005 of the image: a region, not a structure
    assert J.MIN_STRUCTURE_FRAC == 4 * J.MIN_COMPONENT_FRAC
    _lab, keep = J._otsu_structure_labels(Image.fromarray(a, "L").convert("RGB"))
    assert len(keep) == 1 and J.component_count(Image.fromarray(a, "L")) == 2


def test_a_beam_whose_every_verdict_is_below_the_bar_delivers_nothing(tmp_path):
    """The REAL `run_beam` returns zero masks on sub-bar consensus scores; at the bar it delivers.
    MUTANT this kills: `AcceptBar.clears` body -> `return True` — no other suite path fails it."""
    base = {"definition": {"summary": "d"}, "candidates": [{"ops": [dict(_DISK)]}], "assess": [{"id": 0, **_D8}]}
    below = drive_beam(tmp_path, lambda n, b: dict(base), judge_score=7)
    assert below.res.judged >= 1 and below.res.unjudged == 0, "the candidates never reached the final judge"
    assert below.res.masks == [], "a sub-bar verdict was delivered anyway"
    at_bar = drive_beam(tmp_path, lambda n, b: dict(base), judge_score=8)
    assert at_bar.res.masks, "the discriminating sibling: at the bar the same beam must deliver"
    assert all(m.verdict.passed for m in at_bar.res.masks)


def test_every_agent_turn_and_judge_sample_hands_the_transport_its_own_fixed_budget(tmp_path):
    """The per-call ceilings are constants sized per answer, and every call site hands its own over:
    each agent turn (propose, then every refine assess) goes out under MAX_TOKENS_AGENT, the localize
    box under MAX_TOKENS_ROUTE, and every judge sample under MAX_TOKENS_JUDGE (12000, imported from
    its one owner) — read off the transport boundary through the REAL `run_beam` and `judge_consensus`.
    MUTANT this kills: a refine call site handing `_ask_json_retry` a literal or the judge's cap
    instead of `vlm_knobs.MAX_TOKENS_AGENT`; `judging.py` redefining MAX_TOKENS_JUDGE locally."""
    from anomalygen.agentic_mask_generation import judging as J

    base = {"definition": {"summary": "d"}, "candidates": [{"ops": [dict(_DISK)]}], "assess": [{"id": 0, **_D8}]}
    got = drive_beam(tmp_path, lambda n, b: dict(base))
    asks = [(schema, mt) for surface, schema, mt in got.budgets if surface == "ask"]
    assert [s for s, _ in asks] == ["propose", "assess", "assess"], asks  # every agent turn this beam made
    assert {mt for _, mt in asks} == {V.MAX_TOKENS_AGENT} == {8000}, asks
    assert ("call", "localize", V.MAX_TOKENS_ROUTE) in got.budgets, got.budgets

    seen = []

    def call(blocks, max_tokens, schema):
        seen.append((schema, max_tokens))
        return {d: 8 for d in DIMS}

    m = np.zeros((48, 64), np.uint8)
    m[10:20, 20:40] = 255
    clean, mask = Image.new("RGB", (64, 48), (40, 40, 40)), Image.fromarray(m)
    J.judge_consensus(_SVlm(call=call), [], clean, mask, "board+spike", model="m", judge_system="s", samples=2)
    assert seen == [("judge", V.MAX_TOKENS_JUDGE)] * 2 and V.MAX_TOKENS_JUDGE == 12000, seen


def test_accept_continue_banks_the_in_loop_judges_clearing_verdict_and_never_a_self_score(tmp_path, degradations):
    """Under `accept_continue` the in-loop judge is the final judge called early — judge_model, the
    final judge's samples, its adaptive stop — and a round's best that clears the bar is BANKED: the
    beam stops refining it and delivers it as passed with no final-judge call. OFF (the default) the
    same verdict only steers (gen_model, one sample) and the mask still faces the final judge. The
    bank is gated on the judge having ANSWERED: a self-assessment at the bar banks nothing.
    MUTANT this kills: dropping `jv is not None` from the bank condition (a beam whose judge is down
    banks on its own 8s); `banked=banked` withheld from `_final_selection` (the mask falls into the
    keep-best pool and pays the final-judge call the lever exists to skip)."""
    models = {"gen_model": "drawer", "judge_model": "scorer", "final_judge_samples": 2}
    base = {"definition": {"summary": "d"}, "candidates": [{"ops": [dict(_DISK)]}], "assess": [{"id": 0, **_D8}]}

    on = drive_beam(tmp_path, lambda n, b: dict(base), judge_score=9, accept_continue=True, **models)
    assert len(on.judge) == 1, "the banked mask faced a judge again"  # ONE call: the banking verdict itself
    (early,) = on.judge  # ...under the final judge's own settings
    assert (early[1]["model"], early[1]["samples"]) == ("scorer", 2) and early[1]["adaptive"] is not None, early[1]
    assert len(on.turns) == 2, "the beam went on refining a candidate it had banked"  # propose + the banking round
    assert [(m.verdict.passed, m.verdict.score) for m in on.res.masks] == [(True, 9)], on.res.masks

    off = drive_beam(tmp_path, lambda n, b: dict(base), judge_score=9, **models)
    steer, final = off.judge  # the steer on the gen side, then the real final judge
    assert (steer[1]["model"], steer[1]["samples"], steer[1]["adaptive"]) == ("drawer", 1, None), steer[1]
    assert (final[1]["model"], final[1]["samples"]) == ("scorer", 2) and final[1]["adaptive"] is not None, final[1]
    assert len(off.turns) == 3 and [m.verdict.passed for m in off.res.masks] == [True]

    # the judge never answers while the self-assessment sits at the bar (8s): nothing may be banked
    down = RuntimeError("judge offline")
    silent = drive_beam(tmp_path, lambda n, b: dict(base), judge_score=down, accept_continue=True, **models)
    assert "inloop_judge_failed" in degradations
    assert silent.res.masks == [] and len(silent.turns) == 3, "a self-assessment banked a mask"


def test_a_staged_trace_mask_is_labelled_with_its_slot_not_its_position(tmp_path, monkeypatch):
    """`r{t}_c{k}_after` names a candidate the way every other record does — by its STABLE slot, the
    `Ck` the model was shown — so once `beam_prune` drops one, the traced mask and the run log's
    `slot_log` / `AcceptedMask.trail` agree on whose mask it is.
    MUTANT this kills: the label built from `enumerate(new_masks)`'s position instead of `slots[live[_k]]`."""
    staged: list = []
    monkeypatch.setattr(L, "stage_trace_mask", lambda name, mask, vlm: staged.append((name, np.asarray(mask) > 127)))
    disks = [{**_DISK, "center": [x, 0.5]} for x in (0.25, 0.5, 0.75)]  # three distinct candidates
    rank = {0: 5, 1: 6, 2: 9}  # round-0 self-scores: pruning to width 2 keeps slots {1, 2} and drops 0

    def script(n, blocks):
        if n == 0:
            return {"definition": {"summary": "d"}, "candidates": [{"ops": [d]} for d in disks]}
        if n == 1:  # round 0 over C0..C2; slot 1 is edited and slot 2 is not, so their trails differ
            return {
                "assess": [{"id": k, **{d: s for d in DIMS}} for k, s in rank.items()],
                "fixes": [{"id": 1, "ops": [{"kind": "translate", "dx": 0.05, "dy": 0.0}]}],
            }
        return {"assess": [{"id": k, **_D8} for k in range(2)]}  # rounds 1-2 over the two survivors: no drop

    got = drive_beam(tmp_path, script, num_candidates=3, beam_width=2, beam_prune=True, max_refine_rounds=3)
    labels = [name for name, _m in staged]
    assert labels == ["r0_c1_after", "r0_c2_after", "r1_c1_after", "r1_c2_after"], labels
    # the same two masks read from the run log's side: each trail is its slot's edits
    by_trail = {m.trail: np.asarray(m.mask) > 127 for m in got.res.masks}
    last = dict(staged[2:])
    assert np.array_equal(last["r1_c1_after"], by_trail[("CV[translate]", "kept")])
    assert np.array_equal(last["r1_c2_after"], by_trail[("kept", "kept")])


def test_a_clean_pool_of_upper_case_extensions_is_listed_and_delivered(monkeypatch, tmp_path):
    """A camera writes `c0.JPG`; the clean pool is listed by lower-cased suffix, so that dataset
    delivers rows instead of an empty testcase.jsonl with exit 0 and a `type_no_cleans` line.
    MUTANT this kills: a case-sensitive `glob("*.jpg")` in `_list_cleans`."""
    did = drive_generate(monkeypatch, tmp_path, accepted=1, clean_ext=".JPG")
    assert did.exit is None, did.exit
    assert "_run_type" in did.order, f"the type was skipped as having no cleans: {did.order}"
    rows = did.lines("testcase.jsonl")
    assert len(rows) == 1 and Path(rows[0]["image_filename"]).suffix == ".JPG", rows


def test_an_anomaly_type_that_is_a_path_is_refused_before_anything_is_written(monkeypatch, tmp_path):
    """Each `--anomaly-types` token becomes `<out-dir>/<texture>/mask/<defect>`, so a defect half
    carrying `..` would file masks outside `--out-dir`. The token is held to the one validator the AMP
    scripts hold their `defect_type` to, and refused in its words — before the out-dir exists.
    MUTANT this kills: `_build_config` checking only that the token splits on one `+`."""
    did = drive_generate(monkeypatch, tmp_path, types=("T+../../../escaped",), accepted=1)
    assert did.exit is not None, "a path-bearing anomaly type ran"
    said = str(did.exit.code)
    assert "invalid anomaly type in --anomaly-types" in said and "T+../../../escaped" in said, said
    assert not did.out.exists(), sorted(p.name for p in did.out.iterdir())
    assert not (tmp_path / "escaped").exists(), "masks were filed OUTSIDE --out-dir"


def test_the_trim_leaves_the_deliverable_naming_only_files_that_are_there(monkeypatch, tmp_path):
    """testcase.jsonl is exactly the N asked for, every path in it exists, trimmed masks are MOVED
    (run-level overflow manifest), and run.json agrees with both files."""
    did = drive_generate(monkeypatch, tmp_path, types=("T+D", "U+E"), accepted=3, argv=["--num-sdg", "2"])
    assert did.exit is None, did.exit

    kept = did.lines("testcase.jsonl")
    over = did.lines("testcase_overflow.jsonl")
    assert len(kept) == 2, f"the trim did not cut each type to its target: {kept}"
    assert len(over) == 4, f"the overflow manifest is per-type again: {over}"

    for r in kept + over:
        assert Path(r["mask_filename"]).is_file(), f"names a file that is not there: {r}"
    assert not ({r["mask_filename"] for r in kept} & {r["mask_filename"] for r in over}), "a mask is in both manifests"
    assert all("_overflow" in r["mask_filename"] for r in over), over
    assert {r["anomaly_type"] for r in kept} == {"T+D", "U+E"}, kept

    rec = did.json("run.json")
    assert (rec["n_masks"], rec["produced"], rec["trimmed"]) == (2, 6, 4), rec
    assert len(rec["results"]) == 2, "a trimmed mask's summary row survived in run.json"
    assert {r["judge"] for r in rec["results"]} == {9}, rec["results"]


def test_the_run_usage_is_a_running_total_that_never_re_reads_a_booked_row(monkeypatch):
    """`_run_usage()` is read on every accepted mask (run.json is rewritten per accept), so it must
    cost the same at the five-thousandth mask as at the first: each row is folded into the totals as
    it is booked and the summary reads the totals — exactly what a from-scratch re-sum of the same
    rows gives (tokens by key, calls, failed attempts by kind, api_ms, and `beams` counting the beam
    rows only), without touching a booked row again.
    MUTANTS this kills: a booking that skips or half-does the fold; a `prep` row counted as a beam;
    a summary that walks the rows again — every read of a row after its booking is counted."""
    from anomalygen.scripts.agentic_mask_generation import generate as G

    class _Row(dict):
        """A booked row that counts every read of itself."""

        reads = 0

        def __getitem__(self, k):
            type(self).reads += 1
            return super().__getitem__(k)

        def get(self, k, default=None):
            type(self).reads += 1
            return super().get(k, default)

    V._local.count, V._local.failed, V._local.usage, V._local.api_ms = 0, {}, None, 0.0
    led = type(G._RUN_USAGE)()  # a fresh ledger
    monkeypatch.setattr(G, "_RUN_USAGE", led)
    rows = [
        {
            "usage": {"in_tok": 100 * i, "out_tok": 7, "cache_r": i},
            "api_calls": 3 + i,
            "failed": {"transport": i % 2},
            "api_ms": 12.5 * i,
        }
        for i in range(1, 6)
    ]
    rows.append(
        {
            "usage": {"in_tok": 50, "out_tok": 0},
            "api_calls": 2,
            "failed": {"rate_limited": 1},
            "api_ms": 3.25,
            "prep": "T+D",
        }
    )
    for row in rows:
        led.append(_Row(row))
    booked = _Row.reads

    expected = {**dict.fromkeys(V.USAGE_KEYS, 0), "api_ms": 0.0, "calls": 0, "failed_attempts": {}}
    for row in rows:  # the from-scratch re-sum
        for k, v in row["usage"].items():
            expected[k] += v
        expected["calls"] += row["api_calls"]
        for k, v in row["failed"].items():
            expected["failed_attempts"][k] = expected["failed_attempts"].get(k, 0) + v
        expected["api_ms"] += row["api_ms"]
    expected["api_ms"] = round(expected["api_ms"], 1)
    expected["beams"] = sum("prep" not in r for r in rows)

    assert G._run_usage() == expected
    assert G._run_usage() == expected  # a total is kept, not consumed
    assert _Row.reads == booked, (
        f"the summary read the booked rows again: {_Row.reads - booked} reads over {len(rows)} rows"
    )


def test_a_beam_records_each_mask_as_it_lands_and_writes_its_run_log_last(monkeypatch, tmp_path):
    """One beam, two accepted masks, through the real `Quota.collect`: every accept rewrites run.json
    before the next arrives (a crash after the first leaves its `results` row on disk), and the beam's
    run log is the LAST thing written — so a crash inside the beam leaves masks without a log, never a
    log without its masks, which a one-pass resume would take as the clean already done.
    MUTANTS this kills: `emit` without `flush`; the run log written before `on_accept`."""
    import io
    from concurrent.futures import Future

    from anomalygen.scripts.agentic_mask_generation import generate as G

    order: list[str] = []  # every atomic_write target, interleaved with every emit, in order
    real = G.atomic_write

    def _recording(path, write):
        real(path, write)
        order.append(Path(path).name)

    monkeypatch.setattr(G, "atomic_write", _recording)
    cfg, out, clean = _cfg(tmp_path), tmp_path / "out", tmp_path / "c0.png"
    deliv = G.Deliverables(out, cfg, ["generate"], dict)
    tout = G.TypeOutput(deliv, cfg.dataset_path, "board", "spike", "flexible")
    recorded: list[list] = []  # run.json's `results` masks, read back after each emit

    def emit(cp, reuse, j, m):
        tout.emit(cp, reuse, j, m)
        run = out / "run.json"
        recorded.append([r["mask"] for r in json.loads(run.read_text())["results"]] if run.exists() else [])
        order.append(f"emit:{j}")

    (out / "_runlog").mkdir()
    q = G.Quota("board+spike", [clean], 4, 1, 2, None, {}, None, emit, out / "_runlog", io.StringIO(), None)
    verdict = SimpleNamespace(score=9, scores={"roi": 8})
    masks = [
        (clean, 0, j, SimpleNamespace(mask=Image.new("L", (8, 8), 255), overlay=None, verdict=verdict)) for j in (0, 1)
    ]
    rec = {
        "ts": "t",
        "clean": "c0.png",
        "stem": "c0",
        "reuse": 0,
        "judged": 2,
        "accepted": 2,
        "scores": [9, 9],
        "api_calls": 3,
    }
    fut: Future = Future()
    fut.set_result((masks, rec))
    q.collect(fut)

    assert recorded == [["c0_0_0_mask.png"], ["c0_0_0_mask.png", "c0_0_1_mask.png"]], recorded
    assert order.count("c0_0_run.log") == 1 and order[-1] == "c0_0_run.log", order


def test_a_degradation_line_is_the_shape_its_readers_parse(tmp_path, degradations):
    """Real degradations, provoked in the real code, come out of the real logger as (kind, detail)
    pairs the reader can split — covering every SHAPE the format takes (plain and composite kinds,
    exception details, details carrying their own colons)."""

    from anomalygen.agentic_mask_generation.imaging import Views

    full = dict(_D8)
    board = np.full((48, 64, 3), 40, np.uint8)
    clean_p = tmp_path / "board.png"
    Image.fromarray(board).save(clean_p)

    # --- fire them -------------------------------------------------------------------------------
    box, ex0 = {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6}, [Exemplar(defect=None, mask=None)]
    R._widen_to_examples([box], ex0)  # exemplar_unreadable
    R.enumerate_regions(SimpleNamespace(image_max_dim=0), str(tmp_path / "gone.png"), "spike", [], model="m")
    big_box = [{"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9}]  # (the call above: regions_unavailable)
    R.Siting.build(None, big_box, [], want_hard=True, views=Views(512))  # region_crop_failed
    V._extract_json('{"a": 1, "b": {"c": 2}')  # json_repaired:unclosed
    V._extract_json('{"a": 1, "b": "half of a str')  # json_repaired:truncated
    O.apply_op({"kind": "dilate", "px": 3}, None)  # op_failed:dilate
    L._fix_target({"id": "C0"}, 3)  # fix_unplaceable
    L._assess_index([{"id": True, **full}], 2)  # assess_unplaceable
    L._assess_index([{"id": 0, **full}, {"id": 0, **full}], 2)  # assess_duplicate_id

    class _ConvJunk:
        def __init__(self):
            self.messages = []

        def ask_json(self, blocks, max_tokens=1200, schema=None):
            raise ValueError("no JSON object in reply")

    L._ask_json_retry(_ConvJunk(), ["x"], 100, "propose x3", kind="propose")  # call_gave_up:propose

    # a real dataset directory with an unreadable mask — the detail carries a PATH, with its own colons
    img = tmp_path / "board" / "anomaly_image" / "spike"
    msk = tmp_path / "board" / "mask" / "spike"
    img.mkdir(parents=True)
    msk.mkdir(parents=True)
    Image.fromarray(board).save(img / "000.png")
    (msk / "000_mask.png").write_text("not a png at all")
    patch_histogram(str(tmp_path), "board", "spike")  # real_mask_unreadable

    # the beam's own three, through the harness
    base = {"definition": {"summary": "d"}, "candidates": [{"ops": [dict(_DISK)]}], "assess": [{"id": 0, **full}]}
    empty_cand = {**base, "candidates": [{"ops": [{"kind": "polygon", "points": []}]}]}
    drive_beam(tmp_path, lambda n, b: dict(empty_cand))  # proposer_short
    drive_beam(tmp_path, lambda n, b: dict(base))  # fixes_no_op (no `fixes` key at all)
    fx = [{"id": 0, "ops": [{"kind": "clear"}, {"kind": "disk", "center": "nope", "radius": 0.1}]}]
    drive_beam(tmp_path, lambda n, b: {**base, "fixes": fx}, layers={"sam2": np.zeros((180, 240), np.int32)})

    # --- read them back THROUGH THE READER ---------------------------------------------------------
    rows = degradations.rows
    assert len(rows) >= 14, f"only {len(rows)} lines parsed: {degradations}"
    got = dict(rows)
    kinds = (
        "exemplar_unreadable regions_unavailable region_crop_failed json_repaired:unclosed json_repaired:truncated "
        "op_failed:dilate fix_unplaceable assess_unplaceable assess_duplicate_id call_gave_up:propose "
        "real_mask_unreadable proposer_short fixes_no_op"
    )
    for kind in kinds.split():
        assert kind in got, f"{kind} never reached the reader; saw {sorted(got)}"
        # a kind with no detail is a row the counter tallies and nobody can act on
        assert got[kind].strip(), f"degraded[{kind}] carried an empty detail"

    # the composite kinds keep BOTH halves — `op_failed` alone does not say which op
    assert got["op_failed:dilate"].startswith("AttributeError"), got["op_failed:dilate"]
    # a detail with its own ": " does not truncate the row: the reader splits on the FIRST `]: ` only
    assert "_mask.png" in got["real_mask_unreadable"] and ":" in got["real_mask_unreadable"]
    # ...and the fixture's own composite guard is live
    with pytest.raises(AssertionError, match="PREFIX of a composite kind"):
        assert "op_failed" in degradations


def test_a_fix_with_a_boolean_or_overflowing_id_is_unplaceable_not_misplaced(degradations):
    """`int(True) == 1`, so a bool id would land the fix on candidate 1; `1e999` parses to inf and
    `int(inf)` raises OverflowError, which is neither TypeError nor ValueError. Both are recorded as
    unplaceable, the way `_assess_index` records them for the read half."""
    assert L._fix_target({"id": True}, 3) is None
    assert "True" in degradations.first("fix_unplaceable"), degradations
    assert L._fix_target({"id": 1e999}, 3) is None
    assert degradations.kinds()["fix_unplaceable"] == 2, degradations


def test_a_model_id_is_read_by_one_exact_rule(degradations):
    """`_fix_target` and `_assess_index` read ids through `_model_int`: an exact integer — plain, in
    a string, or as an integer-valued float — places; a bool, a non-integer number, an overflow, an
    unparsable string and None do not, each named in the ledger under the caller's kind."""
    table = [(3, 3), ("3", 3), (3.0, 3), (2.7, None), (True, None), (1e999, None), ("x", None), (None, None)]
    assert [L._model_int(v, "id_unplaceable") for v, _ in table] == [want for _, want in table]
    assert [d for k, d in degradations.rows if k == "id_unplaceable"] == [
        "non-integer id 2.7",
        "boolean id True",
        "non-numeric id inf",
        "non-numeric id 'x'",
        "non-numeric id None",
    ]
    # a fix for "candidate 2.7" is a guess either way round, so both halves refuse it
    assert L._fix_target({"id": 2.7}, 3) is None and L._assess_index([{"id": 2.7}], 1) == {}


@pytest.mark.parametrize("rot", [False, True], ids=["landscape", "portrait"])
def test_a_pad_scale_region_is_widened_per_axis_to_the_examples_own_extents(rot):
    """A wide-thin region grows its SHORT axis to the examples' short extent (component scale) and
    keeps it when the examples are themselves thin — per-axis long/short targets, never shrinking.
    ``rot`` swaps the axes of region and exemplar alike: the result must transpose with them.
    Widening is UNCONDITIONAL — `Siting.build` takes no card flag, so every assigned region is
    restored to the examples' measured scale; an enumeration already at that scale passes through
    unchanged, which is what makes the unconditional pass safe."""
    from anomalygen.agentic_mask_generation.imaging import Views

    def _ex(wf, hf):
        a = np.zeros((200, 200), np.uint8)
        a[: int(200 * hf), : int(200 * wf)] = 255
        return Exemplar(defect=Image.new("RGB", (8, 8)), mask=Image.fromarray(a))

    def _sw(x, y):  # (x-value, y-value) under the rotation
        return (y, x) if rot else (x, y)

    def _sited(box, ex):  # the widen as the engine runs it: every Siting.build, no flag in sight
        sit = R.Siting.build(Image.new("RGB", (64, 48)), [dict(box)], [ex], want_hard=False, views=Views(0))
        return sit.regions[0]

    pad_box = dict(zip(("x0", "y0", "x1", "y1"), [*_sw(0.3, 0.45), *_sw(0.7, 0.55)]))  # 0.40 x 0.10

    # component-scale exemplar (long 0.5, short 0.16): the region's short axis is widened to hold it
    out = _sited(pad_box, _ex(*_sw(0.5, 0.16)))
    w, h = out["x1"] - out["x0"], out["y1"] - out["y0"]
    assert min(w, h) == pytest.approx(0.16 * 1.25), (w, h)  # short axis reaches the examples' short extent
    assert max(w, h) == pytest.approx(0.5 * 1.25), (w, h)
    assert (w > h) == (not rot), "the region's long axis moved to the other side under rotation"

    # elongated exemplar (long 0.6, short 0.05): the short axis already suffices and is KEPT — the
    # long-axis factor must not square the box, and no axis ever shrinks
    (out,) = R._widen_to_examples([dict(pad_box)], [_ex(*_sw(0.6, 0.05))])
    w, h = out["x1"] - out["x0"], out["y1"] - out["y0"]
    assert min(w, h) == pytest.approx(0.10), (w, h)  # the assigned short side, unshrunk and unsquared
    assert max(w, h) == pytest.approx(0.6 * 1.25), (w, h)

    # an enumeration already at (or past) the examples' scale comes back the same box — never-shrink
    # is what makes the unconditional widen a no-op instead of a resize
    big_box = dict(zip(("x0", "y0", "x1", "y1"), [*_sw(0.1, 0.2), *_sw(0.9, 0.5)]))  # 0.80 x 0.30
    out = _sited(big_box, _ex(*_sw(0.5, 0.16)))
    assert out == pytest.approx(big_box), out


def test_the_zoom_window_contains_the_work_region_it_was_given():
    """`zoom_around_mask(region=...)` widens its window to CONTAIN the work-region box, so the judge
    sees the whole host feature; without one the window still hugs the mask."""
    from anomalygen.agentic_mask_generation.imaging import zoom_around_mask

    img = Image.new("RGB", (400, 400))
    m = np.zeros((400, 400), np.uint8)
    m[300:340, 180:220] = 255  # a pad-scale mask at the foot of a tall component
    mask = Image.fromarray(m)
    wr = (0.4, 0.05, 0.6, 0.95)  # the tall work region that component occupies
    crop, box = zoom_around_mask(img, mask, region=wr)
    assert crop is not None
    assert box[0] <= wr[0] and box[1] <= wr[1] and box[2] >= wr[2] and box[3] >= wr[3], box
    _, free = zoom_around_mask(img, mask)  # no region: the mask-hugging window, which misses the host
    assert free[1] > wr[1] and free[3] < wr[3], free


@pytest.mark.parametrize("seeded", [False, True], ids=["pool_slot_none", "seed_slot_minus1"])
def test_a_candidate_with_no_slot_of_its_own_still_reaches_the_final_judge(tmp_path, seeded):
    """Keep-best pool entries carry slot None and warm-start seeds -1; with assigned crops present
    the final judge anchors their zoom to the beam-level box instead of losing them UNJUDGED."""
    W, H = 240, 180
    clean_p = tmp_path / "board.png"
    Image.fromarray(np.full((H, W, 3), 40, np.uint8)).save(clean_p)
    ex_mask = np.zeros((H, W), np.uint8)
    ex_mask[80:100, 100:140] = 255
    exm = [Exemplar(defect=Image.fromarray(np.full((H, W, 3), 60, np.uint8)), mask=Image.fromarray(ex_mask))]
    # every refine round MOVES the candidate, so the pooled/seeded original differs from the final
    # mask and is judged as its own entry — the slot-None / slot=-1 path
    moved = {
        "definition": {"summary": "d"},
        "candidates": [{"ops": [dict(_DISK)]}],
        "assess": [{"id": 0, **_D8}],
        "fixes": [{"id": 0, "ops": [{"kind": "disk", "center": [0.3, 0.3], "radius": 0.08}]}],
    }
    vlm = _SVlm(ask=lambda conv, blocks, mt, sc: dict(moved))
    calls = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(L, "judge_consensus", lambda *a, **kw: calls.append(kw) or _v(9))
    seed = None
    if seeded:
        sm = np.zeros((H, W), np.uint8)
        sm[20:50, 30:70] = 255
        seed = [Image.fromarray(sm)]
    try:
        res = L.run_beam(
            str(clean_p),
            "board+spike",
            vlm=vlm,
            visual_prompt=_Cue(),
            defects=_Cue(),
            cfg=_cfg(tmp_path),
            exemplars=exm,
            seed_masks=seed,
            target_region={"x0": 0.4, "y0": 0.4, "x1": 0.9, "y1": 0.9},
        )
    finally:
        monkey.undo()
    assert res.unjudged == 0, f"a slot-less candidate was lost to the judge: {res.unjudged} unjudged"
    assert res.judged >= 2, "the original (pool/seed) geometry never reached the final judge"


def test_the_frame_repair_decides_by_band_and_a_guess_is_never_cached(tmp_path, degradations):
    """m<=1 untouched; m<=2 clamped in place; m<=1000 divided by 1000; beyond that by the image
    size, each repair named by its band — and only an unrepaired reply reaches the disk cache."""
    from anomalygen.agentic_mask_generation.regions import _repair_frame, enumerate_regions

    box = {"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.4}
    regs, repaired = _repair_frame([dict(box)], (96, 96), "t")
    assert (regs, repaired) == ([box], False)

    regs, repaired = _repair_frame([dict(box), {"x0": 0.58, "y0": 1.065, "x1": 0.64, "y1": 1.157}], (96, 96), "t")
    assert repaired and regs[0] == box, "clamping must not move in-range boxes"
    assert regs[1] == {"x0": 0.58, "y0": 1.0, "x1": 0.64, "y1": 1.0}

    regs, repaired = _repair_frame([{"x0": 0.5, "y0": 0.5, "x1": 2.0, "y1": 1.0}], (96, 96), "t")
    assert repaired and regs[0]["x1"] == 1.0 and regs[0]["x0"] == 0.5, "m=2.0 belongs to the clamp band"

    regs, repaired = _repair_frame([{"x0": 0.5, "y0": 0.5, "x1": 2.001, "y1": 1.0}], (96, 96), "t")
    assert repaired and regs[0]["x1"] == pytest.approx(0.002001), "m>2.0 is the 0-1000 convention"

    regs, repaired = _repair_frame([{"x0": 583, "y0": 630, "x1": 640, "y1": 723}], (96, 96), "t")
    assert repaired and regs[0]["x0"] == pytest.approx(0.583)

    regs, repaired = _repair_frame([{"x0": 500, "y0": 250, "x1": 1500, "y1": 500}], (1000, 500), "t")
    assert repaired and regs[0] == {"x0": 0.5, "y0": 0.5, "x1": 1.0, "y1": 1.0}
    # the ledger names the BAND, not the divisor: a 1000 px wide image's pixel frame is not the 0-1000 convention
    labels = [d.rsplit(" -> ", 1)[1] for k, d in degradations.rows if k == "regions_frame_repaired"]
    assert labels == ["overshoot clamped"] * 2 + ["0-1000 convention"] * 2 + ["pixel frame"], labels

    class _V:
        image_max_dim = 0

        def call_json(self, system, blocks, **kw):
            return {"regions": self.regions}

    v = _V()
    clean = tmp_path / "c.png"
    Image.fromarray(np.full((32, 32, 3), 60, np.uint8)).save(clean)
    cache = tmp_path / "regions"
    cache.mkdir()
    v.regions = [{"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 1.157}]
    out = enumerate_regions(v, str(clean), "T+D", [], "m", cache_dir=str(cache))
    assert out and not list(cache.glob("*.json")), "a repaired reply must not be cached"
    v.regions = [{"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.4}]
    out = enumerate_regions(v, str(clean), "T+D", [], "m", cache_dir=str(cache))
    assert out and len(list(cache.glob("*.json"))) == 1, "an unrepaired reply is cached, as a whole reply always is"


def test_a_region_reply_salvaged_by_the_json_repair_serves_this_run_and_is_never_cached(tmp_path, degradations):
    """A region reply cut off at an array boundary is repaired into a schema-valid SHORTER list. The
    sites that arrived whole serve THIS run; the partial list is not written as the board's sites for
    every later run. A whole reply on the same thread is cached, as a whole reply always is.
    MUTANT this kills: the cache decision reading `_repair_frame`'s flag alone."""
    from anomalygen.agentic_mask_generation.regions import enumerate_regions

    b1 = {"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.4}
    b2 = {"x0": 0.5, "y0": 0.5, "x1": 0.8, "y1": 0.8}
    b3 = {"x0": 0.6, "y0": 0.1, "x1": 0.9, "y1": 0.4}
    whole = json.dumps({"regions": [b1, b2, b3]})
    cut = whole[: whole.index(json.dumps(b3))]  # ends `}, ` — the boundary before the third box
    vlm = V.Vlm("their-key", "https://llm-gateway.example/v1", trace_dir=None)
    vlm._http = wire = _Wire([cut, whole])
    clean = tmp_path / "c.png"
    Image.fromarray(np.full((32, 32, 3), 60, np.uint8)).save(clean)
    cache = tmp_path / "regions"
    cache.mkdir()
    out = enumerate_regions(vlm, str(clean), "T+D", [], "m", cache_dir=str(cache))
    assert out == [b1, b2] and len(wire.posts) == 1, (out, len(wire.posts))  # salvaged, not re-asked
    assert "json_repaired:unclosed" in degradations, degradations
    assert not list(cache.glob("*.json")), "the salvaged list was cached as this board's sites"
    out = enumerate_regions(vlm, str(clean), "T+D", [], "m", cache_dir=str(cache))
    assert out == [b1, b2, b3] and len(wire.posts) == 2, (out, len(wire.posts))
    assert len(list(cache.glob("*.json"))) == 1, "a whole reply is cached, as a whole reply always is"


def test_a_size_field_sent_as_a_string_is_remapped_like_the_number_it_holds(degradations):
    """The op layer reads a size field with `float()`, so `"0.2"` is 0.2 to it; the crop-frame remap
    must read it the same way, or the string skips the scaling and is drawn at the CROP fraction of
    the BOARD — 5x on a one-fifth crop, `Applied` with no adjustment and no degraded line. A size
    that holds no number is left as authored and named, for the op layer to refuse by name."""
    box = (0.4, 0.4, 0.6, 0.6)  # a crop one fifth of the board on each side
    disk = {"kind": "disk", "center": [0.5, 0.5], "radius": 0.2}
    (want,) = R._remap_primitives([[disk]], box, 100, 100)
    (got,) = R._remap_primitives([[{**disk, "radius": "0.2"}]], box, 100, 100)
    assert want[0]["radius"] == pytest.approx(0.04)
    assert got == want, f"a string size skipped the remap: {got[0]}"
    assert "remap_coord_unusable" not in degradations, "a string holding a number is no degradation"

    (got,) = R._remap_primitives([[{**disk, "radius": "big"}]], box, 100, 100)
    assert got[0]["radius"] == "big", "a size holding no number is left as authored, not invented"
    assert "disk.radius='big'" in degradations.first("remap_coord_unusable")
    r = O.apply_op(got[0], Image.new("L", (100, 100), 0))
    assert isinstance(r, O.Refused) and "radius" in r.reason, r


def test_a_non_finite_point_coordinate_is_refused_by_name_not_clamped_to_the_border(degradations):
    """A scalar `Infinity` is refused as not finite. A point is a LIST, so its coordinates cross
    `float()` without that check: `_px` clamps inf to the border and the disk is drawn there as
    `Applied` with no adjustment, and NaN reaches PIL as a generic op_failed; the cross-cutting
    `target` point is parsed on its own and a non-finite one is "nearest" to region 1 by default.
    Every one of them is refused naming the parameter; a finite point draws the reference ellipse."""
    from PIL import ImageDraw

    blank = Image.new("L", (100, 100), 0)
    for bad in (["Infinity", 0.5], ["NaN", 0.5]):
        r = O.apply_op({**_DISK, "center": bad}, blank)
        assert isinstance(r, O.Refused) and "center" in r.reason and "finite" in r.reason, r
    r = O.apply_op({"kind": "polygon", "points": [[0.1, 0.1], [0.9, 0.1], ["Infinity", 0.9]]}, blank)
    assert isinstance(r, O.Refused) and "points" in r.reason and "finite" in r.reason, r

    r = O.apply_op(dict(_DISK), blank)
    want = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(want).ellipse([38, 38, 62, 62], fill=255)  # center (50, 50), radius 12 px
    assert isinstance(r, O.Applied) and r.adjustments == (), r
    assert r.mask.tobytes() == want.tobytes(), "a finite center draws the reference ellipse"
    one_region = r.mask
    for bad in (["Infinity", 0.5], ["NaN", 0.5]):
        t = O.apply_op({"kind": "dilate", "frac": 0.02, "target": bad}, one_region)
        assert isinstance(t, O.Refused) and "target" in t.reason, t
    assert not any(k.startswith("op_failed") for k in degradations.kinds()), degradations


def test_a_candidate_whose_ops_is_null_is_re_asked_by_the_schema_and_degrades_on_a_hard_site(tmp_path, degradations):
    """`{"ops": null}` is not an op list: the propose schema rejects it, so the reply is re-asked like any
    other shape miss rather than reaching the loop. One that reaches the loop anyway is mapped onto its
    assigned site INSIDE the guard around the propose reply, so a hard-sited beam books it as a short
    proposer and goes on, instead of raising out of `run_beam` and being booked as a failed beam.
    MUTANT this kills: `ops` untyped in the schema; the site remap outside the propose `try`."""
    null_ops = {"candidates": [{"ops": None}]}
    with pytest.raises(V.SchemaInvalid, match="ops"):
        V._extract_json(json.dumps(null_ops), "propose")
    V.validate_reply({"candidates": [{"ops": [dict(_DISK)]}]}, "propose")

    good = {"definition": {"summary": "d"}, "candidates": [{"ops": [dict(_DISK)]}], "assess": [{"id": 0, **_D8}]}
    site = {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7}
    got = drive_beam(tmp_path, lambda n, b: dict(null_ops) if n == 0 else dict(good), target_region=site)
    assert got.res is not None and len(got.turns) >= 2, "the beam did not go on past the propose turn"
    assert "proposer_short" in degradations, degradations


def test_the_propose_ask_json_template_is_brace_balanced(tmp_path):
    """MUTANT this kills: the template's last fragment losing its f-prefix — doubled braces reach
    the model and the schema it is shown ends at depth -2."""
    from anomalygen.agentic_mask_generation.context import BeamContext

    clean_p = tmp_path / "clean.png"
    Image.fromarray(np.full((96, 96, 3), 60, np.uint8)).save(clean_p)
    c = BeamContext.build(
        str(clean_p), "T+D", visual_prompt=_Cue(), defects=_Cue(card=None), cfg=_cfg(tmp_path), exemplars=[]
    )
    ask = "\n".join(b for b in c.propose_ask() if isinstance(b, str))
    template = ask[ask.index("Output JSON") : ask.index("with EXACTLY")]
    depth = 0
    for ch in template:
        depth += ch == "{"
        depth -= ch == "}"
        assert depth >= 0, f"brace depth goes negative in: …{template[-80:]}"
    assert depth == 0, f"template ends at depth {depth}"


def test_a_models_self_score_is_a_number_whatever_it_sent():
    """The "self" field feeds a max() across options, so ONE string among ints was a TypeError —
    counted as a beam failure, five of which abandon the type as a configuration problem.
    MUTANT this kills: comparing the raw field again in `_selfs`."""
    zoo = [
        ({"self": 8}, 8.0),  # the integer the ask demands — value preserved
        ({"self": "8"}, 8.0),  # quoted by a weak model
        ({"self": "9/10"}, 9.0),  # chatty variant
        ({"self": "high"}, 0.0),  # unparseable: scored 0, never preferred
        ({"self": None}, 0.0),
        ({}, 0.0),  # field absent
    ]
    for payload, want in zoo:
        assert L._self_score({0: payload}, 0) == want, (payload, want)
    assert L._self_score({0: "not-a-dict"}, 0) == 0.0
    assert L._self_score({}, 0) == 0.0  # candidate absent from the sample
    # the exact downstream use: an argmax over MIXED replies must not raise and must pick the best
    samples = [{0: {"self": "9/10"}}, {0: {"self": 7}}, {0: {"self": "high"}}]
    scores = [L._self_score(s, 0) for s in samples]
    assert max(range(3), key=lambda i: scores[i]) == 0, scores


def test_a_caption_is_matched_to_its_image_not_its_position(tmp_path):
    """`_observe` skips a failed caption, so after one skip a positional reader silently hands every
    later exemplar its neighbour's caption. The model name is deliberately hostile (a `+`, over 80
    chars): the reader must find the writer's cache file through the ONE `slug`, not a re-encoding.
    MUTANTS this kills: `observation_for` matching by index; either side renaming the cache alone."""
    from anomalygen.agentic_mask_generation import card_distill as CD
    from anomalygen.agentic_mask_generation.exemplars import load_exemplars
    from anomalygen.agentic_mask_generation.utils import slug

    model = "prov/gemma+fp8-" + "x" * 80  # '+' kept by slug, replaced by a naive re-encode; cap bites

    ds, lib = tmp_path / "ds", tmp_path / "lib"
    img_dir, msk_dir = ds / "tex" / "anomaly_image" / "br", ds / "tex" / "mask" / "br"
    img_dir.mkdir(parents=True), msk_dir.mkdir(parents=True)
    m = np.zeros((32, 32), np.uint8)
    m[8:20, 8:20] = 255
    for stem in ("a", "b", "c"):
        Image.fromarray(np.full((32, 32, 3), 90, np.uint8)).save(img_dir / f"{stem}.png")
        Image.fromarray(m).save(msk_dir / f"{stem}_mask.png")

    stems = iter("abc")  # exemplars load in filename order, so the n-th call captions the n-th stem

    def call(blocks, max_tokens, schema):
        s = next(stems)
        if s == "b":
            raise RuntimeError("caption model hiccup")
        return {"host": f"host-of-{s}", "extent": "part_of_one"}

    (lib / "tex+br").mkdir(parents=True)
    returned = CD._observe(
        _SVlm(call=call), load_exemplars(str(ds), "tex", "br", k=100), "tex/br", model, lib / "tex+br"
    )
    assert all(set(o) == set(CD._CAPTION_KEYS) for o in returned), returned  # filename never reaches the synthesis

    got = {s: CD.observation_for("tex", "br", model, str(lib), str(img_dir / f"{s}.png")) for s in "abc"}
    assert "host-of-a" in got["a"] and "host-of-c" in got["c"], got
    assert got["b"] == "", got  # never captioned: silence, not a neighbour's caption

    # a cache without filenames (written before the field) is stale: unreadable here, re-captioned there
    fp = lib / "tex+br" / f"observations.{slug(model)}.json"
    assert fp.is_file(), list((lib / "tex+br").iterdir())
    blob = json.loads(fp.read_text())
    assert [o["image"] for o in blob["obs"]] == ["a.png", "c.png"], blob["obs"]
    blob["obs"] = [{"host": f"host-of-{s}"} for s in "abc"]
    fp.write_text(json.dumps(blob))
    assert CD.observation_for("tex", "br", model, str(lib), str(img_dir / "b.png")) == ""
    stems2 = iter("abc")
    redo = CD._observe(
        _SVlm(call=lambda *a: {"host": f"re-{next(stems2)}"}),
        load_exemplars(str(ds), "tex", "br", k=100),
        "tex/br",
        model,
        lib / "tex+br",
    )
    assert [o["host"] for o in redo] == ["re-a", "re-b", "re-c"], redo

    # a valid cache is served without a model call, stripped exactly like a fresh caption
    hit = CD._observe(
        _SVlm(call=lambda *a: pytest.fail("cache hit must not caption")), [], "tex/br", model, lib / "tex+br"
    )
    assert [o["host"] for o in hit] == ["re-a", "re-b", "re-c"] and all(set(o) == set(CD._CAPTION_KEYS) for o in hit)


def _card_dataset(tmp_path, stems="abc"):
    """A tiny (tex, br) dataset with one annotated defect image per stem; returns (dataset, library)."""
    ds, lib = tmp_path / "ds", tmp_path / "lib"
    img_dir, msk_dir = ds / "tex" / "anomaly_image" / "br", ds / "tex" / "mask" / "br"
    img_dir.mkdir(parents=True), msk_dir.mkdir(parents=True)
    m = np.zeros((32, 32), np.uint8)
    m[8:20, 8:20] = 255
    for stem in stems:
        Image.fromarray(np.full((32, 32, 3), 90, np.uint8)).save(img_dir / f"{stem}.png")
        Image.fromarray(m).save(msk_dir / f"{stem}_mask.png")
    return ds, lib


def _caption_json(stem):
    from anomalygen.agentic_mask_generation import card_distill as CD

    return json.dumps(
        {k: (f"host-of-{stem}" if k == "host" else "part_of_one" if k == "extent" else "t") for k in CD._CAPTION_KEYS}
    )


def test_a_caption_cut_off_by_the_budget_is_skipped_and_never_cached(tmp_path, degradations):
    """A caption reply cut before its tail fields is salvaged into an object whose remaining fields are
    blank. It is skipped like a failed caption — a row of blanks would weaken the synthesis quietly —
    and above all NOT written into the observation cache, which the ask stamp and the filename check
    cannot see through. The other captions serve the run and are cached.
    MUTANT this kills: the caption call unwatched (the salvaged entry appended and cached)."""
    from anomalygen.agentic_mask_generation import card_distill as CD
    from anomalygen.agentic_mask_generation.exemplars import load_exemplars

    ds, lib = _card_dataset(tmp_path)
    whole_b = _caption_json("b")
    cut_b = whole_b[: whole_b.index('"boundary"')]  # ends `, ` — cut at the field boundary before "boundary"
    vlm = V.Vlm("their-key", "https://llm-gateway.example/v1", trace_dir=None)
    vlm._http = wire = _Wire([_caption_json("a"), cut_b, _caption_json("c")])
    (lib / "tex+br").mkdir(parents=True)
    got = CD._observe(vlm, load_exemplars(str(ds), "tex", "br", k=100), "tex/br", "m", lib / "tex+br")
    assert [o["host"] for o in got] == ["host-of-a", "host-of-c"], got
    assert len(wire.posts) == 3, "a salvaged caption is skipped, not re-asked"
    assert "json_repaired:unclosed" in degradations and "caption_repaired" in degradations, degradations
    blob = json.loads((lib / "tex+br" / "observations.m.json").read_text())
    assert [o["image"] for o in blob["obs"]] == ["a.png", "c.png"], blob["obs"]


def test_a_card_cut_off_by_the_budget_is_re_asked_and_never_written(tmp_path, degradations):
    """The synthesis schema puts the flags first, so a card cut off after its third section is salvaged
    with every flag and three sections — enough for `_faults` — and would land as the type's reference
    knowledge. A salvaged card is a fault: the informed re-ask fires, and after the attempts run out no
    card is written; a whole card is written, as a whole card always is.
    MUTANT this kills: the synthesis call unwatched (the salvaged card passes `_faults` and is written)."""
    from anomalygen.agentic_mask_generation import card_distill as CD

    ds, lib = _card_dataset(tmp_path)
    flags = {
        "host_structure": "one lead",
        "hosts_presented": "multi_host",
        "fill_host": "part",
        "boundary_contact": "outer",
        "sites_spanned": "one",
        "distribution_kind": "single",
        "visual_prompt": "none",
        "why": "w",
    }
    whole = json.dumps({"flags": flags, **{s: f"{s} text" for s in CD.SECTIONS}})
    cut = whole[: whole.index(json.dumps(CD.SECTIONS[3]))]  # flags + three sections survive the repair
    assert not CD._faults(V._extract_json(cut)), "the premise: the salvaged card alone passes _faults"
    vlm = V.Vlm("their-key", "https://llm-gateway.example/v1", trace_dir=None)
    vlm._http = wire = _Wire([_caption_json(s) for s in "abc"] + [cut, cut, cut])
    dest = lib / "tex+br" / f"{CD.card_variant('m')}.md"
    CD.build_cards_dir(vlm, str(ds), [("tex", "br")], str(lib), model="m")
    assert not dest.exists(), "a card cut off by the budget was written as the type's reference knowledge"
    assert len(wire.posts) == 6 and "card_failed" in degradations, (len(wire.posts), degradations)
    assert "rejected" in json.dumps(wire.posts[4]) and "cut off" in json.dumps(wire.posts[4]), (
        "the re-ask names the fault"
    )
    wire.replies = [whole]  # the captions are cached; a whole synthesis is written
    CD.build_cards_dir(vlm, str(ds), [("tex", "br")], str(lib), model="m")
    assert dest.exists() and len(wire.posts) == 7, (dest.exists(), len(wire.posts))


def test_a_thin_annotation_survives_the_exemplar_mask_thumbnail(tmp_path):
    """Every consumer reads an exemplar mask at `> 127` (`overlay_mask`, `outline_mask`, the caption's
    zoom, the geometry note). An interpolating thumbnail averages a 1-px scratch on a 2048 annotation
    into grey below that bar at 640, so the region is gone (no zoom, an empty ring, a caption about an
    undrawn region) while preflight, which reads the full-resolution file, sees nothing wrong.
    `resize_mask` is NEAREST: values stay 0/255 and the area is unbiased. Its limitation is real and
    NOT what is pinned — a 1-px line lying between two of its samples is lost by NEAREST as well; real
    exemplar corpora hold no component narrower than the downscale factor. The pinned case is the
    survivable one: the line sits on a column NEAREST samples, read off Pillow itself rather than
    derived by hand. The annotated view (`annotate_exemplar`) resamples the same file, so the two
    renderings of one exemplar must agree on where the annotation is.
    MUTANT this kills: `mask_img.thumbnail(...)` with the default filter in place of `resize_mask`."""
    from anomalygen.agentic_mask_generation import visual_prompts as VP
    from anomalygen.agentic_mask_generation.exemplars import load_exemplars

    # Which 2048-wide source column does Pillow's NEAREST 2048 -> 640 read for destination column 100?
    # An "I" image whose pixel VALUE is its column index answers that exactly.
    probe = Image.fromarray(np.tile(np.arange(2048, dtype=np.int32), (4, 1)), "I")
    dst = 100
    col = int(np.asarray(probe.resize((640, 4), Image.NEAREST))[0, dst])

    ds = tmp_path / "ds"
    img_dir, msk_dir = ds / "tex" / "anomaly_image" / "scratch", ds / "tex" / "mask" / "scratch"
    img_dir.mkdir(parents=True), msk_dir.mkdir(parents=True)
    m = np.zeros((2048, 2048), np.uint8)
    m[256:1792, col] = 255  # a 1-px-wide, 1536-px-long vertical scratch
    Image.fromarray(np.full((2048, 2048, 3), 90, np.uint8)).save(img_dir / "a.png")
    Image.fromarray(m).save(msk_dir / "a_mask.png")

    (ex,) = load_exemplars(str(ds), "tex", "scratch", k=1, max_side=640)
    assert ex.defect.size == ex.mask.size == (640, 640)
    on = np.asarray(ex.mask) > 127
    assert on.any(), "the thumbnail lost the annotation"
    length = 1536 * 640 // 2048  # the scratch's length at 640, one pixel wide
    assert abs(int(on.sum()) - length) <= 2, on.sum()
    assert set(np.where(on)[1].tolist()) == {dst}, set(np.where(on)[1].tolist())
    assert set(np.unique(np.asarray(ex.mask))) <= {0, 255}, "an exemplar mask is 0/255, never grey"

    class _MapVP:  # the painted map is the defect image's own size; the outline is drawn on it
        def precompute(self, path):
            pass

        def painted_map(self, path):
            return Image.new("RGB", (640, 640), (40, 40, 40))

    annotated = VP.annotate_exemplar(_MapVP(), ex, max_side=640)
    white = (np.asarray(annotated) == 255).all(axis=2)
    assert white.any() == on.any(), "the exemplar's mask and its annotated view disagree on the region"
    outline_cols = set(np.where(white.any(axis=0))[0].tolist())
    assert outline_cols & {dst - 1, dst, dst + 1}, (
        f"the outline is not drawn around column {dst}: {sorted(outline_cols)[:8]}"
    )


def test_a_chatty_assess_score_is_coerced_not_a_beam_crash(tmp_path):
    """Assess dims arrive as "9/10", "8" or prose from weak models; `int()` on them was a ValueError
    inside the round loop — a beam failure, five of which abandon the type (the self-score's failure
    class, on the sibling field). Chatty parses to its number, prose scores 0, the beam completes.
    MUTANT this kills: `int(a.get(kk, 0))` restored in the `subs` comprehension or inside `_avg4`."""
    chatty = {
        "definition": {"summary": "d"},
        "candidates": [{"ops": [dict(_DISK)]}],
        "assess": [{"id": 0, "roi": "9/10", **{d: "no numbers here" for d in DIMS if d != "roi"}}],
    }
    got = drive_beam(tmp_path, lambda n, b: dict(chatty), judge_score=9)
    assert got.res.masks, "a coercible assess reply still crashed the beam"
    assert all(m.verdict.passed for m in got.res.masks)
    assert L._avg4({d: "9/10" for d in DIMS}) == 9
    assert L._avg4({**{d: 9 for d in DIMS}, "roi": "looks right"}) == 8  # ONE prose dim scores 0, not the entry
    assert L._avg4("not a dict") == 0
    assert L._score_num(float("nan")) == 0.0 and L._score_num(float("inf")) == 0.0  # json accepts NaN/1e999


def test_a_reply_that_closes_early_and_cannot_be_spliced_is_re_asked_not_kept(tmp_path, degradations):
    """A `}` where the model meant to go on with `, "key": ...` is spliced out and the text re-read;
    when the re-read does not parse either, the reply raises as unparseable (one re-ask above) rather
    than returning the object that closed early — a shorter answer that looks finished, and that the
    caller keeps on disk for every later run.
    MUTANT this kills: the splice loop's `except json.JSONDecodeError: break`."""
    from anomalygen.agentic_mask_generation.regions import enumerate_regions

    box = {"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.4}
    cut = json.dumps({"regions": [box]}) + ', "why": "one pad per site, and the'
    # a splice that re-reads cleanly is the answer the model meant, with nothing to report
    assert V._extract_json('{"summary": "s"}, "sections": {"a": 1}}') == {"summary": "s", "sections": {"a": 1}}
    assert not degradations.rows, degradations
    # a splice that does not: raised, and said
    with pytest.raises(json.JSONDecodeError):
        V._extract_json(cut, "regions")
    assert "json_splice_failed" in degradations, degradations
    degradations.clear()

    # through `call_json` into a caller that caches: re-asked once, given up on, and NOTHING written
    vlm = V.Vlm("their-key", "https://llm-gateway.example/v1", trace_dir=None)
    vlm._http = wire = _Wire([cut, cut])
    clean = tmp_path / "c.png"
    Image.fromarray(np.full((32, 32, 3), 60, np.uint8)).save(clean)
    cache = tmp_path / "regions"
    cache.mkdir()
    out = enumerate_regions(vlm, str(clean), "T+D", [], "m", cache_dir=str(cache))
    assert out == [] and len(wire.posts) == 2, (out, len(wire.posts))
    assert not list(cache.glob("*.json")), "the object that closed early was cached as this board's sites"
    assert degradations.kinds()["json_splice_failed"] == 2 and "call_gave_up:call_json" in degradations, degradations


def test_a_truncated_reply_is_repaired_by_structure_not_brace_counts():
    """The repair balanced a cut-off reply with count(), which reads braces inside STRING VALUES as
    structure and appends every `]` before any `}` — a reply whose text mentioned `}`, or one cut
    inside nested arrays, was "repaired" into JSON that cannot parse, and the whole reply dropped.
    MUTANT this kills: count()-based balancing restored in `_repair_truncated_json`."""
    # a `}` inside the last complete string value is text; the value must come back intact
    obj = V._extract_json('{"reason": "brace } inside prose", "scores": {"roi": 8')
    assert obj == {"reason": "brace } inside prose", "scores": {}}, obj
    # an unterminated-string tail: the cut field is DROPPED with its key (absent, so the caller
    # re-asks) and the `}` already delivered inside a value does not pay for the missing closer
    obj = V._extract_json('{"msg": "ok }", "tail": "half of a str')
    assert obj == {"msg": "ok }"}, obj
    # cut inside nested arrays: closers come back innermost-first, `]` and `}` interleaved
    obj = V._extract_json('{"ops": [{"kind": "polygon", "points": [[0.1, 0.2], [0.3')
    assert obj["ops"][0]["points"] == [[0.1, 0.2], []], obj
    # the scanner's own contract: an open string is closed before the open scopes are
    assert V._missing_closers('{"a": [1, "half') == '"]}'


def test_rectangle_rotates_the_same_direction_as_every_other_shape():
    """`Rectangle.draw` built its rotation from the y-up textbook matrix (x'=x cosθ-y sinθ,
    y'=x sinθ+y cosθ) applied directly to y-DOWN pixel coordinates, which flips the visual sense of
    the turn: for the same +angle, Rectangle rotated clockwise on screen while every other shape that
    rotates (Ellipse via PIL's `Image.rotate`, `rotate_mask`'s `cv2.getRotationMatrix2D`) turns
    counter-clockwise. A model correcting "rotate the rectangle +15 the way you just rotated the
    ellipse" would get the opposite turn with no error — same op vocabulary, opposite geometry.
    MUTANT this kills: the sign of the y-term in `Rectangle.draw`'s corner formula flipped back."""
    m = np.zeros((101, 101), np.uint8)
    ctx = None
    # a wide, thin bar through the center — asymmetric enough that rotation direction is unambiguous
    p = O.Rectangle().check_input({"center": [0.5, 0.5], "rx": 0.3, "ry": 0.01, "angle": 30.0})
    out = O.Rectangle().apply(m, p, ctx)
    ys, xs = np.where(out.mask > 0)
    rightmost_y = ys[np.argmax(xs)]
    # PIL's rotate(+30) on the same right-of-center geometry moves it UP (smaller y) — see the
    # module docstring on `Rectangle.draw`. The broken formula moved it down (rightmost_y > 50).
    assert rightmost_y < 50, (rightmost_y, "rectangle's rightmost point should move UP for +angle")


def test_fill_holes_fills_only_what_the_outside_cannot_reach():
    """`fill_holes` flooded from the FIRST zero border pixel, so a mask that splits the background
    (an edge-to-edge stroke) had the whole far-side background filled in as a "hole" — Applied, no
    adjustment, 192 px quietly became thousands. The zero-ring pad connects every border-touching
    background component, so one flood reaches them all. MUTANT this kills: seeding the flood on one
    unpadded border pixel (or restoring the seed-is-None early return)."""
    fh = O.FillHoles()
    p = fh.check_input({})
    stripe = np.zeros((64, 64), np.uint8)
    stripe[:, 30:33] = 255  # edge-to-edge: both sides touch the border, neither is a hole
    assert int(np.count_nonzero(fh.apply(stripe, p, None).mask)) == 192
    ring = np.zeros((64, 64), np.uint8)
    ring[20:40, 20:40] = 255
    ring[26:34, 26:34] = 0  # a genuine enclosed hole: still filled
    assert int(np.count_nonzero(fh.apply(ring, p, None).mask)) == 400
    frame = np.full((64, 64), 255, np.uint8)
    frame[10:54, 10:54] = 0  # the mask owns EVERY border pixel; its interior gap is still enclosed
    assert int(np.count_nonzero(fh.apply(frame, p, None).mask)) == 64 * 64


def test_an_empty_reply_leaves_no_assistant_turn_and_the_retry_resends_the_ask_once():
    """A turn answered with nothing (or whitespace) appends no assistant turn: on the /messages route an
    empty text block fails every later request on that conversation with a 400, for the rest of the
    beam. The trailing message is then the user's ask, which `_ask_json_retry` pops and re-sends, so the
    next request carries that ask exactly once; a reply that parses is appended, as any parsed reply is.
    MUTANT this kills: the assistant turn appended before the reply is looked at."""
    vlm = V.Vlm("their-key", "https://llm-gateway.example/v1", trace_dir=None)
    vlm._http = wire = _Wire(["", "", '{"a": 1}', "  \n", '{"b": 2}'])
    bare = vlm.conversation("s", model="m")
    with pytest.raises(json.JSONDecodeError):
        bare.ask_json(["q"])
    # the ask stays, for the caller to pop or re-send
    assert [m["role"] for m in bare.messages] == ["user"], bare.messages

    conv = vlm.conversation("s", model="m")  # the beam's path: two turns, each answered with nothing first
    assert L._ask_json_retry(conv, ["q1"], 100, "ask q1", kind="propose") == {"a": 1}
    assert L._ask_json_retry(conv, ["q2"], 100, "ask q2", kind="assess") == {"b": 2}
    assert [m["role"] for m in conv.messages] == ["user", "assistant", "user", "assistant"], conv.messages
    assert all(b["text"].strip() for m in conv.messages for b in m["content"]), conv.messages
    sent = [m for m in wire.posts[-1]["messages"] if m["role"] != "system"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"], sent  # each ask once, no empty text
    assert len(wire.posts) == 5


def test_the_notes_a_round_reads_name_candidates_by_its_own_montage(tmp_path):
    """The judge steer (`C{bk}`) and the op-outcome notes (`candidate {c}`) are built with THIS
    round's positions but read NEXT round, after `beam_prune` has renumbered the montage — so a
    5-kept-to-3 beam opened on "your BEST candidate C4" while the board showed C0..C2. Both notes are
    remapped through `live`, exactly like `last_round_scored`. MUTANT this kills: sending either note
    through unremapped."""
    disks = [{**_DISK, "center": [x, 0.5]} for x in (0.25, 0.5, 0.75)]
    rank = {0: 5, 1: 6, 2: 9}  # prune to width 2 keeps old {1, 2}: old 1 -> C0, old 2 (the best) -> C1

    def script(n, blocks):
        if n == 0:
            return {"definition": {"summary": "d"}, "candidates": [{"ops": [d]} for d in disks]}
        if n == 1:  # round 0: old 1 gets an out-of-range edit, so the next round carries an edits note
            return {
                "assess": [{"id": k, **{d: s for d in DIMS}} for k, s in rank.items()],
                "fixes": [{"id": 1, "ops": [{"kind": "translate", "dx": 5.0, "dy": 0.0}]}],
            }
        return {"assess": [{"id": k, **_D8} for k in range(2)]}

    got = drive_beam(tmp_path, script, num_candidates=3, beam_width=2, beam_prune=True, max_refine_rounds=3)
    texts = [b for b in got.turns[2] if isinstance(b, str)]  # round 1's ask, over the renumbered pair
    steer = next(b for b in texts if b.startswith("INDEPENDENT JUDGE"))
    assert "C1" in steer and "C2" not in steer, steer  # old best C2 is C1 on the montage this ask shows
    edits = next(b for b in texts if "YOUR LAST EDITS" in b or "ADJUSTED" in b)
    assert "candidate 0" in edits and "candidate 1" not in edits and "candidate 2" not in edits, edits


def test_the_beam_memory_note_speaks_in_next_round_positions():
    """Same renumbering, third note: the merged beam's memory is built from this round's `scored` ids
    but read beside next round's montage. Entries are relabelled through `live`; a candidate a lever
    removed is dropped, and a note with nobody left says nothing. MUTANT this kills:
    `_beam_memory_note` labelling `C{k}` with the unmapped id."""
    scored = [(9, 2, {d: 9 for d in DIMS}), (6, 1, {d: 6 for d in DIMS}), (5, 0, {d: 5 for d in DIMS})]
    note = L._beam_memory_note(scored, {}, {}, live=[1, 2])
    assert "C0[" in note and "C1[" in note and "C2[" not in note, note
    assert L._beam_memory_note(scored, {}, {}, live=[]) == ""
    legacy = L._beam_memory_note(scored, {}, {})  # no lever fired: labels are the ids, unchanged
    assert "C0[" in legacy and "C1[" in legacy and "C2[" in legacy
