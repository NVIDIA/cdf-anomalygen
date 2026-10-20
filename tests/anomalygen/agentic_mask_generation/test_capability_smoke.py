# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-run capability probe (a refused model stops the run ABOVE every spend) and the two-layer
endpoint dispatch (the base picks the envelope, the model name picks only the cache markers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from anomalygen.agentic_mask_generation import vlm as V

_SPEND = ("_load_defects", "enumerate_regions", "run_beam")
# A generic gateway base — deliberately NOT vlm.ANTHROPIC_BASE, so every row that uses it exercises
# the "any other base" leg of the dispatch.
_GATEWAY = "https://llm-gateway.example/v1"
_EPH = {"type": "ephemeral"}


def _counting(system, blocks, **kw):
    """Answers the probe by COUNTING the images it was handed, so a constant reply cannot pass."""
    return {"n_images": sum(isinstance(b, Image.Image) for b in blocks)}


def _refusing(kind, detail, status=None):
    """A model the endpoint will not run this request against — the transport's own failure kinds."""

    def call_json(system, blocks, **kw):
        raise V.CallFailed(V.Failure(kind, detail, status), 1)

    return call_json


class _Wire:
    """Stand-in for the pooled `httpx.Client`: records every POST and answers it with one 200 shaped
    for the route it was sent to."""

    def __init__(self):
        self.posts: list = []

    def post(self, url, headers=None, json=None, **kw):
        self.posts.append(SimpleNamespace(url=url, headers=dict(headers or {}), payload=json))
        body = '{"n_images": 2}'
        d = (
            {"content": [{"type": "text", "text": body}], "usage": {}}
            if url.endswith("/messages")
            else {"choices": [{"message": {"content": body}}], "usage": {}}
        )
        return SimpleNamespace(status_code=200, headers={}, text="", json=lambda: d)


def _vlm(base, key="their-key"):
    """The shipped `Vlm` over a recording client: routing, headers and payload building all real."""
    wire = _Wire()
    vlm = V.Vlm(key, base, trace_dir=None)
    vlm._http = wire
    return vlm, wire


def _drive(monkeypatch, tmp_path, *, call_json=_counting, extra=(), run_log=False):
    """Run the REAL `generate.main()` and record WHAT IT REACHED, in order — stubs only at the
    boundary. `extra` is appended to the command line; two calls with the same `tmp_path` are two
    processes over one out-dir. `run_log=True` has the logger stub create `run.log` the way the real
    `init_loguru_file` does (without the sink), so the out-dir is non-empty the way a real run leaves it."""
    from anomalygen.agentic_mask_generation.defects import from_dir
    from anomalygen.agentic_mask_generation.exemplars import PreflightReport
    from anomalygen.scripts.agentic_mask_generation import generate as G

    ds = tmp_path / "ds"
    (ds / "T" / "clean_image").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (40, 60, 80)).save(ds / "T" / "clean_image" / "c0.png")
    out = tmp_path / "out"
    order: list[str] = []
    probes: list[tuple] = []
    traces: list = []  # the `trace_dir` every constructed transport was handed

    class _Vlm:
        image_max_dim = 0

        def __init__(self, key, base, **kw):
            traces.append(kw.get("trace_dir"))

        def call_json(self, system, blocks, **kw):
            order.append("smoke")
            probes.append((kw.get("model"), sum(isinstance(b, Image.Image) for b in blocks)))
            return call_json(system, blocks, **kw)

        def endpoint_report(self):
            return {}

        def endpoint_summary(self):
            return "ok"

    def _beam(clean, dtype, **kw):
        order.append("run_beam")
        verdict = SimpleNamespace(passed=True, score=9, scores={"roi": 8})
        mask = SimpleNamespace(mask=Image.new("L", (32, 32), 255), overlay=None, verdict=verdict)
        return SimpleNamespace(masks=[mask], judged=1, api_calls=1)

    def _defects(*a, **k):
        order.append("_load_defects")
        return from_dir(None)

    def _regions(*a, **k):
        order.append("enumerate_regions")
        return [[0.1, 0.1, 0.3, 0.3]]

    def _preflight(ds, ts, *a, **k):
        return PreflightReport(
            types=len(ts), pairs=len(ts), cleans=0, seconds=0.0, per_type={f"{t}+{d}": 1 for t, d in ts}
        )

    monkeypatch.setattr(G, "preflight", _preflight)
    monkeypatch.setattr(G, "Vlm", _Vlm)
    monkeypatch.setattr(G, "_route_shape", lambda *a, **k: False)
    monkeypatch.setattr(G, "run_beam", _beam)
    monkeypatch.setattr(G, "_load_defects", _defects)
    monkeypatch.setattr(G, "load_exemplars", lambda *a, **k: [])
    monkeypatch.setattr(G, "enumerate_regions", _regions)
    monkeypatch.setattr(G, "init_loguru_file", (lambda p: Path(p).touch()) if run_log else (lambda p: None))
    reg = tmp_path / "endpoints.yaml"  # the default models must resolve in the ONE registry
    reg.write_text(f"- model: azure/openai/gpt-6-astra\n  base_url: {_GATEWAY}\n  key: test-key\n")
    head = ["generate", "--dataset-path", str(ds), "--anomaly-types", "T+D", "--out-dir", str(out)]
    tail = f"--cache-dir {tmp_path / 'cache'} --visual-prompt none --num-sdg 1".split()
    tail += ["--model-registry", str(reg)]
    tail += ["--gen-model", "azure/openai/gpt-6-astra", "--judge-model", "azure/openai/gpt-6-astra"]
    monkeypatch.setattr(sys, "argv", head + tail + list(extra))
    exit_ = None
    try:
        G.main()
    except SystemExit as e:
        exit_ = e
    tc = out / "testcase.jsonl"
    rows = [json.loads(x) for x in tc.read_text().splitlines() if x.strip()] if tc.exists() else []
    return SimpleNamespace(exit=exit_, order=order, probes=probes, rows=rows, traces=traces, out=out)


def test_a_model_that_takes_images_and_answers_in_json_is_probed_once_and_the_run_proceeds(monkeypatch, tmp_path):
    """Pass-through: the probe runs FIRST and ONCE, carrying >1 image, and the run proceeds to deliver."""
    did = _drive(monkeypatch, tmp_path)

    assert did.exit is None, did.exit
    assert did.order[0] == "smoke", did.order
    for later in _SPEND:
        assert later in did.order, f"{later} never ran, so its absence in the refusal test proves nothing"
        assert did.order.index("smoke") < did.order.index(later), f"the probe ran after {later}: {did.order}"
    assert did.order.count("smoke") == 1, f"one model was probed {did.order.count('smoke')} times"
    assert len(did.rows) == 1, did.rows
    ((_model, n_images),) = did.probes
    assert n_images > 1, f"the probe sent {n_images} image(s), so a blind guess of 1 would pass it"


@pytest.mark.parametrize(
    "call_json, says",
    [
        (_refusing(V.BAD_REQUEST, "HTTP 400: this model does not support image input", 400), "not serving images"),
        (lambda system, blocks, **kw: {"n_images": 0}, "reported 0"),
    ],
    ids=["endpoint-refuses", "wrong-answer"],
)
def test_a_failed_probe_stops_the_run_above_every_spend_and_says_what_failed(
    monkeypatch, tmp_path, degradations, call_json, says
):
    """A failed probe stops ABOVE every spend and says — on the exit and the ledger — what failed and
    the `--skip-smoke` escape. MUTANT this kills: deleting the `_smoke_models(...)` call from `main`."""
    did = _drive(monkeypatch, tmp_path, call_json=call_json)

    assert did.order == ["smoke"], f"a run whose model failed the probe still reached: {did.order[1:]}"
    assert did.exit is not None and did.exit.code, did.exit
    assert not did.rows, "the refused run delivered masks anyway"
    assert says in str(did.exit.code), did.exit.code
    assert "--skip-smoke" in str(did.exit.code)
    assert says in degradations.first("model_smoke_failed")


def test_a_refused_probe_leaves_a_resumable_out_dir_and_the_skip_smoke_retry_continues_it(monkeypatch, tmp_path):
    """The refusal names `--skip-smoke` as the way to run anyway, so that retry — the same command plus
    the flag, into the same out-dir, WITHOUT `--overwrite` — has to be accepted. Two things make it so:
    the control plane (`run.json`) is on disk before the probe, so the refused run's out-dir is a run
    to continue rather than a non-empty directory without a record; and `--skip-smoke` is not part of
    the command the run records, so adding it is not a control-plane change.
    MUTANTS this kills: the first `deliv.flush()` below the probe (no run.json after the refusal);
    `--skip-smoke` left in the recorded argv (the retry refused as `argv: added --skip-smoke`)."""
    refused = _drive(monkeypatch, tmp_path, call_json=_refusing(V.BAD_REQUEST, "no image input", 400), run_log=True)
    assert refused.exit is not None and "--skip-smoke" in str(refused.exit.code), refused.exit
    assert (refused.out / "run.log").exists(), "the harness control: the refusal left the out-dir non-empty"
    assert (refused.out / "run.json").exists(), sorted(p.name for p in refused.out.iterdir())

    retry = _drive(monkeypatch, tmp_path, extra=["--skip-smoke"], run_log=True)
    assert retry.exit is None, retry.exit
    assert "smoke" not in retry.order and len(retry.rows) == 1, (retry.order, retry.rows)
    recorded = json.loads((retry.out / "run.json").read_text())["argv"]
    assert "--skip-smoke" not in recorded, recorded


def test_the_trace_is_opt_in(monkeypatch, tmp_path):
    """A run with default flags hands the transport no trace dir and leaves no trace directory in
    the out-dir; `--trace-dir` is the one way to have every call's IO kept.
    MUTANT this kills: a default of `<out-dir>/trace` in place of `None`."""
    plain = _drive(monkeypatch, tmp_path)
    assert plain.exit is None and plain.traces == [None], plain.traces
    assert not (plain.out / "trace").exists(), "a default run left a trace directory behind"
    kept = _drive(monkeypatch, tmp_path / "again", extra=["--trace-dir", str(tmp_path / "kept")])
    assert kept.exit is None and kept.traces == [str(tmp_path / "kept")], kept.traces


def test_the_entry_refuses_an_unregistered_model_and_names_the_registry():
    """The registry is the ONLY routing: a model name it does not know refuses the run, the refusal
    names the model and points at endpoints.yaml, and there is no environment fallback behind it.
    MUTANT this kills: `_endpoints` routing an unknown name to some default base instead of refusing."""
    from anomalygen.scripts.agentic_mask_generation import generate as G

    registry = {
        "their-vlm": {"model": "their-vlm", "base_url": _GATEWAY}
    }  # a literal file's worth — nothing is built in
    with pytest.raises(SystemExit) as refusal:
        G._endpoints(["azure/anthropic/claude-nonesuch"], registry)
    said = str(refusal.value)
    assert "azure/anthropic/claude-nonesuch" in said and "endpoints.yaml" in said, said
    # ... and the EMPTY registry (no file at all) refuses too, saying so, instead of inventing a base
    with pytest.raises(SystemExit) as bare:
        G._endpoints(["their-vlm"], {})
    assert "no registry file found" in str(bare.value), bare.value

    # a registered model resolves to exactly its entry — base_url verbatim, key defaulting to EMPTY
    assert G._endpoints(["their-vlm"], registry)["their-vlm"] == (_GATEWAY, "EMPTY")
    # a literal key comes back verbatim, and its logged SOURCE never carries the value
    base, key, src = G._resolve_endpoint("m", {"base_url": _GATEWAY, "key": "sk-secret"})
    assert (base, key) == (_GATEWAY, "sk-secret") and "sk-secret" not in src, (base, key, src)


@pytest.mark.parametrize("absent", ["--gen-model", "--judge-model"])
def test_a_command_that_does_not_name_a_models_role_is_refused_by_the_parser(monkeypatch, capsys, absent):
    """`--gen-model` and `--judge-model` are REQUIRED: which model draws and which judges is always
    stated, never inherited from a default nobody sees. A command missing either is refused by the
    parser itself — a usage error naming the flag, before the registry, the out-dir or a token — and
    the control, the same command with both, parses. MUTANT this kills: `required=True` swapped for a
    `default=<model>` on either flag."""
    from anomalygen.scripts.agentic_mask_generation import generate as G

    full = ["generate", "--dataset-path", "ds", "--anomaly-types", "T+D", "--gen-model", "m", "--judge-model", "m"]
    monkeypatch.setattr(sys, "argv", full)
    both = G._args()
    assert (both.gen_model, both.judge_model) == ("m", "m")
    at = full.index(absent)
    monkeypatch.setattr(sys, "argv", full[:at] + full[at + 2 :])
    with pytest.raises(SystemExit) as refusal:
        G._args()
    assert refusal.value.code == 2, refusal.value.code  # argparse's usage error, not a run that started
    said = capsys.readouterr().err
    assert "required" in said and absent in said, said


def test_the_pooled_client_does_not_trust_ambient_proxy_env(monkeypatch):
    """With HTTPS_PROXY set, the Vlm's client mounts no proxy — the registry stays the one routing source.
    MUTANT this kills: dropping `trust_env=False` from the `httpx.Client(...)` construction."""
    import httpx

    monkeypatch.setenv("HTTPS_PROXY", "http://ambient-proxy.example:3128")
    # the control: a client left trusting env DOES mount the proxy, so the assertion below discriminates
    assert httpx.Client(trust_env=True)._mounts, "httpx stopped reading proxy env; this pin proves nothing"
    client = V.Vlm("their-key", _GATEWAY, trace_dir=None)._http
    assert client.trust_env is False and not client._mounts


def test_mask_staging_follows_the_staging_vlms_trace_dir(tmp_path):
    """A traced Vlm flipped a module-wide latch nothing reset, so every LATER untraced Vlm's calls
    kept buffering staged masks; a per-thread flag armed at dispatch was no better — the judge
    fan-out stages BEFORE the first call on a fresh worker thread, and the mask vanished from the
    trace. Staging is gated by the Vlm handed to `stage_trace_mask` — the instance whose next call
    claims the buffer — so it needs no call history at all.
    MUTANT this kills: a process-wide latch restored, or the gate read off thread state."""
    import threading

    mask = Image.new("L", (8, 8), 255)
    traced = V.Vlm("their-key", _GATEWAY, trace_dir=str(tmp_path / "trace"))
    traced._http = _Wire()
    seen = {}

    def _judge_order():  # a fresh judge worker: stage FIRST, call second
        V.stage_trace_mask("candidate_mask", mask, traced)
        seen["staged"] = bool(getattr(V._local, "pending_masks", None))
        traced.call_json("sys", ["hi"], model="m")
        seen["claimed"] = not getattr(V._local, "pending_masks", None)

    th = threading.Thread(target=_judge_order)
    th.start()
    th.join()
    assert seen == {"staged": True, "claimed": True}, seen
    assert list((tmp_path / "trace").rglob("*candidate_mask*")), "the claimed mask never reached the trace dir"

    untraced = V.Vlm("their-key", _GATEWAY, trace_dir=None)
    untraced._http = _Wire()
    V.stage_trace_mask("candidate_mask", mask, untraced)
    assert not getattr(V._local, "pending_masks", None), "an untraced Vlm buffered a staged mask"


@pytest.mark.parametrize(
    "base, model, path, marked",
    [
        # the ONE native-protocol base routes to /messages whatever the model is called
        ("https://api.anthropic.com/v1", "claude-opus-5", "/messages", True),
        # any other base speaks OpenAI chat, even claude-named; the NAME picks only the cache markers
        (_GATEWAY, "azure/anthropic/claude-opus-5", "/chat/completions", True),
        (_GATEWAY, "their-own-vlm-7b", "/chat/completions", False),
    ],
)
def test_which_route_a_model_name_takes_and_against_which_base(base, model, path, marked):
    """EXACTLY TWO LAYERS: the BASE picks the envelope (ANTHROPIC_BASE = /messages + x-api-key, any
    other = chat + Bearer), the NAME picks only the cache markers, which ride the chosen envelope.
    MUTANT this kills: routing by model name to /messages on a base that only serves chat."""
    vlm, wire = _vlm(base)

    assert vlm.call_json("sys", ["hi"], model=model, schema="smoke") == {"n_images": 2}
    (post,) = wire.posts
    assert post.url == base.rstrip("/") + path, post.url
    if base.rstrip("/") == V.ANTHROPIC_BASE:
        # Anthropic's real endpoint authenticates API keys via x-api-key; its Bearer slot is OAuth.
        assert post.headers["x-api-key"] == "their-key", post.headers
        assert "authorization" not in post.headers, post.headers
        assert post.payload["system"][0]["cache_control"] == _EPH, post.payload["system"]
    else:
        assert post.headers["authorization"] == "Bearer their-key", post.headers
        assert "x-api-key" not in post.headers, post.headers
        sys_msg = post.payload["messages"][0]
        assert sys_msg["role"] == "system", post.payload["messages"]
        if marked:  # claude-named on a gateway: the system marker rides the chat payload in block form
            assert sys_msg["content"] == [{"type": "text", "text": "sys", "cache_control": _EPH}], sys_msg
        else:  # non-claude: the plain OpenAI request, marker-free end to end
            assert sys_msg["content"] == "sys", sys_msg
            assert "cache_control" not in json.dumps(post.payload), post.payload
    assert post.payload["model"] == model, post.payload


def test_every_default_base_carries_its_version_segment():
    """Every routing test sets its base explicitly, so the SHIPPED bases' version segments are
    pinned only here: the one base constant in the code, and every entry in the template users
    copy. MUTANT this kills: an endpoints.example.yaml base_url losing its version segment."""
    import pathlib

    import yaml

    from anomalygen.scripts.agentic_mask_generation import generate as G

    example = (
        pathlib.Path(G.__file__).parents[3] / "scripts" / "agentic_mask_generation_serving" / "endpoints.example.yaml"
    )
    bases = {"vlm.ANTHROPIC_BASE": V.ANTHROPIC_BASE}
    bases |= {e["model"]: e["base_url"] for e in yaml.safe_load(example.read_text())}
    for name, base in bases.items():
        path = base.split("://", 1)[1].rstrip("/")
        assert "/" in path, f"{name}: {base} has no path — the version segment is gone"
        last = path.rsplit("/", 1)[1]
        assert last.startswith("v") and last[1].isdigit(), f"{name}: {base} does not end in a version segment"


@pytest.mark.parametrize(
    "base, bare",
    [
        ("https://api.anthropic.com", True),  # the misconfiguration the diagnosis names: /v1 missing
        ("https://api.anthropic.com/", True),  # a trailing slash is still no path
        ("http://api.anthropic.com", True),  # the scheme is not what makes it bare
        (V.ANTHROPIC_BASE, False),  # the shipped base, /v1 present
        (_GATEWAY, False),  # any other host, whatever its path
    ],
)
def test_the_bare_anthropic_host_diagnosis_fires_only_without_the_version_segment(base, bare):
    """The smoke-probe diagnosis for a 4xx from api.anthropic.com WITHOUT /v1 keys on this predicate.
    MUTANT this kills: matching the host regardless of path, which would misdiagnose a correctly
    configured ANTHROPIC_BASE as a missing segment."""
    from anomalygen.scripts.agentic_mask_generation import generate as G

    assert G._is_bare_anthropic_host(base) is bare


def test_the_default_registry_lives_outside_the_repo():
    """The product images chmod the checkout read-only, so a repo-relative default registry cannot
    even be CREATED there — and a credential file inside the tree is one COPY away from a published
    image layer. The default must be absolute and outside the repo. MUTANT this kills: pointing
    DEFAULT_REGISTRY back into the checkout (e.g. beside the template)."""
    import os
    import pathlib

    from anomalygen.scripts.agentic_mask_generation import generate as G

    # a literal `~`, never resolved at import (a HOME-less container must reach argparse);
    # expanded it must be absolute
    assert G.DEFAULT_REGISTRY.startswith("~"), G.DEFAULT_REGISTRY
    assert os.path.isabs(os.path.expanduser(G.DEFAULT_REGISTRY)), G.DEFAULT_REGISTRY
    repo = pathlib.Path(G.__file__).parents[3].resolve()
    assert not pathlib.Path(os.path.expanduser(G.DEFAULT_REGISTRY)).resolve().is_relative_to(repo), G.DEFAULT_REGISTRY
    # the template it is copied from stays in the tree, as documentation and as the probe's anchor
    assert (repo / G.REGISTRY_TEMPLATE).is_file(), G.REGISTRY_TEMPLATE


def test_the_default_cache_never_writes_into_the_dataset():
    """The default cache lived at <dataset>/_amg_cache — inside the INPUT, which the product
    containers mount read-only, so a run paid the smoke probe and then died writing its first card.
    The default is rooted under results/, stays keyed per dataset (two datasets never share cards),
    and an explicit --cache-dir still wins. MUTANT this kills: defaulting back under dataset_path."""
    import pathlib

    from anomalygen.scripts.agentic_mask_generation import generate as G

    root = G._cache_root(None, "datasets/nvpcb")
    assert not str(root.resolve()).startswith(str(pathlib.Path("datasets/nvpcb").resolve())), root
    assert str(root).startswith("results"), root
    assert G._cache_root(None, "datasets/nvpcb") != G._cache_root(None, "datasets/uc1pega")
    # the colliding shape: SAME leaf, different parents — the name alone would share one library
    assert G._cache_root(None, "data/2025q1/pcb") != G._cache_root(None, "data/2025q3/pcb")
    assert G._cache_root(None, "datasets/nvpcb") == G._cache_root(None, "datasets/nvpcb")  # stable key
    assert G._cache_root("/tmp/b3/cache", "datasets/nvpcb") == pathlib.Path("/tmp/b3/cache")
