# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The control plane for ``agentic_mask_generation``: one flat, frozen dataclass.

Policy only — no secrets and no environment reads, so a ``Config`` is always safe to snapshot
into a run record. Defaults are quality-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# The six shared scoring dimensions, used by both the gen self-assessment and the authoritative judge.
DIMS = ("roi", "placement", "shape", "fill", "distribution", "semantic")


@dataclass(frozen=True)
class Config:
    """The single, flat, typed control plane. Policy only — no secrets."""

    # --- dataset (required) ---
    anomaly_types: tuple[tuple[str, str], ...]  # (texture, defect) pairs, e.g. ("Phone", "oil")
    dataset_path: str

    # --- models (switch model = change the id; vlm routes by id) ---
    # NO defaults: --gen-model and --judge-model are required at the CLI, so which model draws and
    # which judges is always stated on the command line, never inherited from a value nobody sees.
    # Generation sets the ceiling — the judge only selects among the candidates generation produced —
    # so gen_model is the consequential of the two.
    gen_model: str = ""  # propose / refine / rank + in-loop steering judge
    judge_model: str = ""  # authoritative final judge

    # --- in-context examples ---
    num_examples: int = 3  # how many real examples to USE (pool size itself is data-driven)

    # --- weak-model information channels (all OFF by default: the base prompts stay byte-identical;
    # a local/smaller model turns them on — same contract as the in-loop beam flags) ---
    # Attach each exemplar's own distill-time caption (host / extent / shape) next to its images.
    exemplar_captions: bool = False
    # Also show each exemplar as the injected vp's painted map with its REAL mask outlined.
    exemplar_vp: bool = False
    # Inject the measured STRUCTURE-COVERAGE note (rank-paired bands + footprint + off-structure
    # mass) at judge / refine / pairwise.
    coverage_note: bool = False

    # --- beam-search engine ---
    num_candidates: int = 5  # candidate masks proposed ONCE per beam, then refined in place every round
    # How many of the LAST round's candidates go to the final consensus judge. The refine loop prunes
    # nothing — every candidate is refined every round — so this sizes final-judge cost, not refine cost.
    beam_width: int = 3
    max_refine_rounds: int = 5
    # The accept bar, in the order `AcceptBar` tests it: semantic >= this, no dimension below
    # this - 1, and the mean >= this — NOT "every dimension >= this".
    accept_threshold: int = 8
    final_judge_samples: int = 3
    # Regions handed to ONE beam, one per candidate slot. More of them decouples the sites a run covers
    # from the number of beams it launches. Swept on a multi-site class: 2 is the knee —
    # it materially lifts how many sites a run covers with spill and ground truth statistically
    # unmoved, while 3 starts leaking outside the ROI and 5 leaks badly for no further coverage.
    slot_regions: int = 2
    # The defect card is sent once, at propose, then buried under rounds of images and scores; this
    # repeats its "typical shape" sentence each refine turn. In a controlled arm it substantially lifts
    # shape coverage against the real defects, with acceptance rate and ground truth unchanged.
    restate_shape_target: bool = True
    # DISTRIBUTION / coverage feedback: spread accepted defects across a board's valid sites by telling
    # a turn which locations already carry one on THIS clean. "none" = off; the other five are two
    # different mechanisms sharing one field:
    #   "localize" | "propose" | "both"  — which TURN is told what is already covered.
    #   "roundrobin"                     — enumerate each clean's valid regions once and ASSIGN each
    #                                      candidate slot a distinct one (the engine picks the site).
    #   "choose"                         — enumerate the same regions and show the AGENT the menu with
    #                                      the taken ones marked, so it picks (goes to the propose turn).
    # coverage_mode = how the covered state is shown.
    coverage_feedback: str = "roundrobin"  # enumerate the valid regions per clean; each run gets a distinct one
    coverage_mode: str = "text"  # "text" (normalized bbox list) | "visual" (covered masks overlaid on clean)
    # Downscale every image's longest side to this before sending. Image tokens scale with AREA, so
    # 1024 cuts tokens well below native on high-res boards while keeping fine structure legible. A
    # pure TOKEN-COST lever — the 5 MB/image gateway limit is a separate hard byte cap in the
    # transport. None = native resolution.
    image_max_dim: Optional[int] = 1024
    # Global API concurrency: the `Vlm` permit AND the fill-loop beam count, 1:1 because every call a
    # beam makes is serial, so a beam IS one thread and one request in flight. A FIXED ceiling for the
    # whole run — the permit never moves, and a refusal is answered by waiting on a shared backoff
    # window, so setting it too high is paid for in waiting rather than in a collapse.
    # 32 is a ceiling this run will not exceed, not a measured limit of any endpoint. An endpoint with
    # a known one wants its own value (`--max-concurrent-calls`; the qwen3-vl serving recipe sets 8).
    max_concurrent_calls: int = 32

    # Restrict the vocabulary to these op kinds (empty = the full CV_KINDS). The restriction reaches
    # turn 1 as well as refine, so the listed ops are the only way to reach the mask — which is how a
    # small vocabulary can be tested for whether it is expressive enough.
    allowed_cv_ops: tuple[str, ...] = ()

    # --- pluggable-by-name ---
    # "auto" reads each type's distilled card (its `visual_prompt` flag) and uses none where the card
    # does not say — quality-first, like every other default here. The card exists because
    # `--defect-library` also defaults to auto; a run that turns the library off falls back to no cue.
    visual_prompt: str = "auto"  # "sam2" | "solderlight" | "normallight" | "none" | "auto"
    prompt_set: str = "default_v1"

    # --- placement scope -----------------------------------------------------------------------
    # The run-level shape POLICY (`--shape`). "auto" asks the per-type VLM shape router, under one
    # rule above it: a card judging `fill_host=whole` forces flexible — the mask IS whole host
    # instance(s), so a stamped example shape can only collide with the target board's own
    # structures — and the stamp op is withheld from that type's beams (`BeamContext.build`).
    # "flexible" makes every type flexible (the same card still withholds stamp); "fixed" makes
    # every type fixed and OUTRANKS the card ban, so the full fixed vocabulary — stamp included —
    # stays on offer (a fixed-shape beam's only vocabulary is stamps, so withdrawing the op would
    # make it unsatisfiable). An absent, unreadable or otherwise-valued card bans nothing anywhere.
    shape_policy: str = "auto"
    # The per-type RESOLVED mode the driver applies via `replace()`: "flexible" = synthesize + freely
    # refine the shape; "fixed" = stamp a REAL training-mask shape and refine placement with RIGID
    # ops only (translate/rotate/flip — never deform it).
    shape_mode: str = "flexible"
    # --- the in-loop beam (all OFF by default) -----------------------------------------------------
    # RE-SAMPLE the refine turn and let a judge keep the best result.
    #
    # Every lever here is a Config field, never an environment variable read inside the engine
    # (`test_the_engine_reads_no_environment`): a run's recorded configuration determines its behaviour,
    # and a shell variable nobody wrote down means two "identical" arms can differ by something no run
    # log will show.
    #
    # K distinct-strategy options per candidate, asked for in ONE call (0 = off). An int, not a bool:
    # the refine prompt renders `int(beam_multi)` as "propose K options", so a literal True would
    # degenerate to "propose 1" and lose the whole point of the lever. 0 is falsy, so the off value
    # reads as off wherever the lever is tested for truth.
    beam_multi: int = 0
    beam_samples: int = 1  # independent samples of the same refine turn (1 = the one proposal only)
    # Concurrency for the fan-outs INSIDE one beam (option re-sampling, and the selector judge over one
    # candidate's rendered options). See the note at each consumer in loop.py: this is the one place the
    # engine breaks "one beam holds one API call at a time".
    beam_workers: int = 6
    beam_no_memory: bool = False  # drop the cross-round memory the multi-option ask normally carries
    # Also run the separate per-round self-assess call alongside the multi-option ask.
    beam_legacy_assess: bool = False
    # Model that picks among the sampled candidates. "" = the beam's own judge (`judge_model`). The
    # fallback is load-bearing: the selector is what keeps one of the K rendered options rather than
    # candidate #0, so an empty selector must resolve to a model, never to "no selection".
    beam_select: str = ""
    beam_pairwise: bool = False  # selector compares candidates A-vs-B rather than scoring each alone
    beam_judge_samples: int = 1  # consensus samples for that selector judge
    beam_judge_max_exemplars: int = 0  # cap its exemplar images (0 = no cap; some endpoints cap at 5)
    beam_judge_no_vp: bool = False  # drop the visual prompt from that judge's context

    # --- algorithmic throughput levers (fewer LLM calls per accepted mask; GT dual-track verified) ---
    # PRUNE the refine beam to beam_width. Every round re-generates AND re-judges all num_candidates,
    # yet only the top-beam_width survivors are ever eligible for the final stage (the final judge takes
    # scored[:beam_width]). Carrying only the top-beam_width forward each round shrinks the next round's
    # generation + in-loop judging with no change to what CAN be accepted — the survivors the design
    # already selects for. Measured effect: about a third fewer LLM calls per accepted
    # mask, ground-truth mean unchanged. OFF keeps the salvage-every-candidate behaviour.
    beam_prune: bool = False
    # Reorder the JUDGE prompt so the blocks CONSTANT across every candidate on a board (definition,
    # exemplars, defect docs, visual prompt, clean view) form a stable leading PREFIX and the per-candidate
    # blocks (structural facts, measured geometry, global/zoom overlays) come last — so vLLM prefix-caching
    # reuses the static prefix's KV across a board's many judge calls instead of re-prefilling it each time
    # (the win is on the self-hosted air-gapped path; Claude caches per-candidate only). Reordering changes
    # the prompt bytes (the model's output can shift), so it is opt-in; OFF is byte-identical to before.
    judge_cache_order: bool = False
    # ACCEPT-AND-CONTINUE: when the authoritative in-loop judge already clears the final-judge bar on a
    # round's best candidate, BANK it immediately (it bypasses the final judge) and drop it from the beam,
    # so later rounds refine only the not-yet-good candidates. Attacks both the refine loop (fewer live
    # candidates) and the final judge (banked masks skip it): measured as roughly a quarter fewer LLM calls per mask. It accepts
    # on a single in-loop-judge verdict rather than the final judge's consensus, so it can admit a mask the
    # fuller final vote would have rejected — a small mean risk on hard defects. Opt-in; GT-gate adoption.
    accept_continue: bool = False
