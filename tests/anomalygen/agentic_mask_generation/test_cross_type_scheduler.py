# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cross-type fill scheduler, held to multi-type fill correctness. Everything runs the REAL
`generate.main()` with boundary stubs."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from anomalygen.agentic_mask_generation import vlm as V

# ---- The harness: real main(), boundary stubs, per-type beam control --------------------------------
_REGIONS = [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]  # two, so the round-robin site counter really walks


def _mask(stem: str, reuse: int, j: int) -> Image.Image:
    """Pixels are a pure function of (clean, reuse, j), so byte equality is about the CODE, not timing."""
    m = Image.new("L", (32, 32), 0)
    side = 3 + (int(stem[-1]) * 7 + reuse * 3 + j) % 20
    m.paste(255, (2, 2, 2 + side, 2 + side))
    return m


def _beam_result(clean, n_masks: int, reuse: int = 0, judged: "int | None" = None):
    """`judged` defaults to the mask count; pass it explicitly for a beam that judged and rejected all."""
    stem = Path(clean).stem
    masks = [
        SimpleNamespace(
            mask=_mask(stem, reuse, j),
            overlay=None,
            verdict=SimpleNamespace(
                passed=True, score=9 - (j % 3), scores={"roi": 8, "semantic": 7 + j}, reasons={}, feedback=""
            ),
        )
        for j in range(n_masks)
    ]
    judged = n_masks if judged is None else judged
    return SimpleNamespace(masks=masks, judged=judged, api_calls=3, usage={"in_tok": 11, "out_tok": 3}, api_ms=12.5)


def _drive(monkeypatch, tmp, *, types=("T+D",), cleans=2, concurrency=1, n=None, argv=(), **kw):
    """Run the REAL `generate.main()` against boundary stubs (kw: out, beam, regions, per_type) and
    return what happened plus the rows/manifests read back off disk."""
    from anomalygen.agentic_mask_generation.defects import from_dir
    from anomalygen.agentic_mask_generation.exemplars import PreflightReport
    from anomalygen.scripts.agentic_mask_generation import generate as G

    # both survive across tests in one process, so they are zeroed like every other driver harness does
    V._local.count, V._local.failed, V._local.usage, V._local.api_ms = 0, {}, None, 0.0
    monkeypatch.setattr(G, "_RUN_USAGE", G._UsageLedger())
    ds = tmp / "ds"
    for t in types:
        d = ds / t.split("+")[0] / "clean_image"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(cleans):
            Image.new("RGB", (32, 32), (30 + 10 * i, 40, 50)).save(d / f"c{i}.png")
    made: list = []
    inner = kw.get("beam") or (lambda clean, dtype, **k: _beam_result(clean, 2))

    def _beam(clean, dtype, **k):
        made.append((Path(clean).stem, dtype))
        return inner(clean, dtype, **k)

    class _Vlm:
        image_max_dim = 0
        endpoint_report = staticmethod(dict)

        def __init__(self, *a, **k):
            pass

        def call_json(self, system, blocks, **k):
            return {"n_images": sum(isinstance(b, Image.Image) for b in blocks)}

        def endpoint_summary(self):
            return "ok"

    def _pf(ds_, ts, *a, **k):  # a real PreflightReport (main reads per_type off it); equal weights -> even split
        pt = kw.get("per_type") or {f"{t}+{d}": 1 for t, d in ts}
        return PreflightReport(types=len(ts), pairs=len(ts), cleans=0, seconds=0.0, per_type=pt)

    monkeypatch.setattr(G, "preflight", _pf)
    monkeypatch.setattr(G, "Vlm", _Vlm)
    monkeypatch.setattr(G, "_route_shape", lambda *a, **k: False)
    monkeypatch.setattr(G, "run_beam", _beam)
    monkeypatch.setattr(G, "_load_defects", lambda *a, **k: from_dir(None))
    monkeypatch.setattr(G, "load_exemplars", lambda *a, **k: [])
    monkeypatch.setattr(G, "enumerate_regions", kw.get("regions") or (lambda *a, **k: [list(r) for r in _REGIONS]))
    monkeypatch.setattr(G, "init_loguru_file", lambda p: None)
    reg = tmp / "endpoints.yaml"  # every model a test names must resolve in the ONE registry
    reg.write_text("- model: azure/openai/gpt-6-astra\n  base_url: https://llm-gateway.example/v1\n  key: test-key\n")
    out = kw.get("out") or tmp / "out"
    cmd = ["generate", "--dataset-path", str(ds), "--anomaly-types", *types, "--out-dir", str(out)]
    cmd += ["--cache-dir", str(tmp / "cache"), "--visual-prompt", "none", "--max-concurrent-calls", str(concurrency)]
    cmd += ["--model-registry", str(reg)]
    cmd += ["--gen-model", "azure/openai/gpt-6-astra", "--judge-model", "azure/openai/gpt-6-astra"]
    cmd += ([] if n is None else ["--num-sdg", str(n)]) + list(argv)  # order pinned by the fixture
    monkeypatch.setattr(sys, "argv", cmd)
    exit_ = None
    try:
        G.main()
    except SystemExit as e:
        exit_ = e
    tc = out / "testcase.jsonl"
    rows = [json.loads(x) for x in tc.read_text().splitlines() if x.strip()] if tc.exists() else []
    run = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else None
    return SimpleNamespace(out=out, ds=ds, exit=exit_, beams=made, rows=rows, run=run)


# ---- Multi-type fill correctness ---------------------------------------------------------------------
def test_two_types_both_fill_at_concurrency_one(monkeypatch, tmp_path):
    """ONE slot, two hungry types: both fill to target — the state a done-check bug reads as finished.
    MUTANT this kills: dropping the `want_inflight() > 0` guard from `_check_done` (exit 0, one type EMPTY)."""
    t2_prepped = threading.Event()

    def regions(vlm, clean, dtype, exs, model, cache_dir=None):
        if dtype == "T2+D":
            t2_prepped.set()
        return [list(r) for r in _REGIONS]

    def beam(clean, dtype, **kw):
        if dtype == "T1+D":
            t2_prepped.wait(timeout=10)
            time.sleep(0.2)  # the admission message behind that prep reaches the pump
        return _beam_result(clean, 1)  # 2 beams per type, 1 slot for all

    did = _drive(monkeypatch, tmp_path, types=("T1+D", "T2+D"), concurrency=1, beam=beam, regions=regions, n=4)
    assert did.exit is None, did.exit
    by = {t: sum(1 for r in did.rows if r["anomaly_type"] == t) for t in ("T1+D", "T2+D")}
    assert by == {"T1+D": 2, "T2+D": 2}, by
    assert all(Path(r["mask_filename"]).exists() for r in did.rows)


# ---- One-pass mode (no target at all) ------------------------------------------------------------------
def test_a_run_with_no_target_makes_one_pass_over_the_cleans(monkeypatch, tmp_path):
    """MUTANT this kills: the zero-want admission guard comparing None <= 0 — a TypeError that
    abandoned every type of a run started without --num-sdg / --per-defect-counts."""
    did = _drive(monkeypatch, tmp_path)
    assert did.exit is None, did.exit
    assert len(did.rows) > 0 and all(Path(r["mask_filename"]).exists() for r in did.rows)


# ---- --overwrite is consumed, not recorded --------------------------------------------------------------
def test_an_overwrite_run_is_continuable_without_the_flag(monkeypatch, tmp_path):
    """MUTANT this kills: recording --overwrite into run.json's argv — the re-run then either wipes
    the delivered masks (flag kept) or is refused as a control-plane change (flag dropped)."""
    did = _drive(monkeypatch, tmp_path, n=4, argv=("--overwrite",))
    assert did.exit is None, did.exit
    assert "--overwrite" not in did.run["argv"]
    again = _drive(monkeypatch, tmp_path, n=4)  # same out-dir, flag dropped: resume, not refusal
    assert again.exit is None, again.exit
    assert len(again.rows) == len(did.rows)


# ---- The agent token budget is a CONSTANT, like every other per-call budget --------------------------
def test_the_agent_token_budget_is_the_constant_and_is_recorded(monkeypatch, tmp_path):
    """8000, fixed — no env, no registry field, no per-model override — and run.json records it.
    MUTANT this kills: an override path quietly reintroduced somewhere between vlm.py and main()."""
    did = _drive(monkeypatch, tmp_path, n=2)
    assert did.exit is None and did.run["engine_constants"]["max_tokens_agent"] == V.MAX_TOKENS_AGENT == 8000
