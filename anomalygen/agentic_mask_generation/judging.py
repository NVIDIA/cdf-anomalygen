# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General (domain-agnostic) VLM judge: score a candidate mask 0-10 by understanding the defect,
never by GT-match.

Everything is passed in and nothing is read from the environment (`test_the_engine_reads_no_environment`).
Alongside the VLM calls it computes deterministic structural facts (region counts, spanned
structures, spacing, geometry) from the pixels, so the model's claims rest on exact numbers.
"""

from __future__ import annotations

import statistics
import weakref
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.agentic_mask_generation.card_distill import exemplar_note
from anomalygen.agentic_mask_generation.config import DIMS
from anomalygen.agentic_mask_generation.imaging import (
    Views,
    overlay_mask,
    resize_mask,
)
from anomalygen.agentic_mask_generation.prompts import assert_names_no_op
from anomalygen.agentic_mask_generation.visual_prompts import EXEMPLAR_VP_CAPTION

# MAX_TOKENS_JUDGE is the output cap for one judge call, IMPORTED from its one owner and never
# redefined here — a local copy shadows the import and wins silently
# (no module redefines a shared token ceiling). A too-small cap costs MASKS, not tokens: a
# reply cut mid-sentence may not salvage into JSON, the sample is dropped, and a candidate whose
# samples all truncated is booked UNJUDGED, no verdict at all rather than a rejection.
from anomalygen.agentic_mask_generation.vlm import MAX_TOKENS_JUDGE, Vlm, stage_trace_mask

_NO_REASON = "(no reason given)"

# Prepended to injected ``defect_docs``; the card text itself is passed in, only the framing is here.
_DEFECT_DOCS_JUDGE_CAPTION = (
    "AUTHORITATIVE DEFECT KNOWLEDGE (reference for scoring the SEMANTIC dimension — does the mask read "
    "as this defect's real mechanism/appearance? Use as context, not a hard rule):"
)

# The one floor, as a fraction of the image area, below which a connected component is not a region.
# Every count the judge is shown reads it — `component_count`, `nn_spacing`, `_shape_stats`,
# `_component_text` — so one prompt cannot carry two region counts that disagree on a speck.
MIN_COMPONENT_FRAC = 0.0002
# The floor for the BOARD's Otsu structures (`_otsu_structure_labels`) — a different population from
# the mask's regions: the bright side of a photo carries specks (glints, texture) no mask could be
# measured against, so this floor is four times coarser, and a speck is never a structure the
# candidate "misses" in a coverage rank or a structures-spanned count.
MIN_STRUCTURE_FRAC = 0.0008


def _dim_line(d: dict[str, Any], key: str, label: str) -> str:
    v = d.get(key, "")
    if isinstance(v, dict):  # {"target": ..., "reference": ...}
        tgt = v.get("target", "")
        ref = v.get("reference", "")
        return f"- {label}: {tgt}" + (f"   [reference: {ref}]" if ref else "")
    return f"- {label}: {v}"


def _render_definition(d: dict[str, Any] | None) -> str:
    """Render the proposer's own 6-dimension rubric, so it and the judge score against ONE set of
    per-dimension targets + references."""
    if not d:
        return ""
    lines = [
        "AGREED DEFINITION & RUBRIC for this defect (the SAME items the generator and judge use — "
        "follow / score against THIS, do not re-interpret):",
        f"- image understanding (what the structures in this image ARE): {d.get('image_understanding', d.get('summary', ''))}",
        f"- defect mechanism (physical cause — why/where it occurs): {d.get('mechanism', '')}",
        _dim_line(d, "roi", "ROI (legal region — right structure type, avoid excluded)"),
        _dim_line(d, "placement", "Placement (precise alignment on the local feature; right structures/number)"),
        _dim_line(d, "shape", "Shape (form/topology, hug real outline)"),
        _dim_line(d, "fill", "Fill (how much of the target feature it should cover)"),
        _dim_line(d, "distribution", "Distribution (count/spread/irregularity; multi-instance)"),
        _dim_line(d, "semantic", "Semantically correct (true instance under the mechanism)"),
    ]
    return "\n".join(lines)


# How a dimension that cannot fail is kept from being a free pass: on a type whose card says one
# occurrence makes one region, `distribution` reads 10 on every well-formed candidate, and the bar
# takes the MEAN, so that automatic 10 lifts every candidate by its share. Naming it here keeps it in
# `min` and out of the mean (both paths into the bar score the same dimensions). Do NOT drop
# the dimension from the type instead: it is what catches a mask that fragmented into three pieces,
# and if it reads 10 on every accepted mask the WORDING is what to fix
# (`prompts._distribution_expectation`).
GATE_ONLY = ("distribution",)


def gate_only_dims(flags: dict | None = None) -> tuple[str, ...]:
    """Which dimensions VETO without averaging, for this type. Derived from the same card flag that
    words the prompt, so the scoring rule and the sentence the judge reads cannot disagree."""
    return GATE_ONLY if (flags or {}).get("distribution_kind") == "single" else ()


def _gates(scores: dict, gate_only: tuple = ()) -> tuple[float, float, float]:
    """The three numbers the acceptance bar is made of, in the order it tests them: the semantic score,
    the weakest dimension, and the mean. ONE definition, read by both the bar and the early stop;
    `gate_only` names dimensions that count toward the WEAKEST but not the MEAN."""
    vals = list(scores.values())
    avg = [v for k, v in scores.items() if k not in gate_only] or vals
    return float(scores.get("semantic", 0)), float(min(vals)), sum(avg) / len(avg)


@dataclass(frozen=True)
class AcceptBar:
    """Whether a mask is accepted: the judge's verdict, read against the score bar.

    Calibrated to ground-truth-achievable scores: the calibration probe showed even REAL example masks
    score ~7 on the geometric dims (shape/fill/placement), so requiring ALL six >= thr is stricter than
    ground truth and flips on +/-1 noise. Hence the asymmetric rule in :meth:`clears`.
    """

    thr: int
    # dimensions that VETO without averaging (`gate_only_dims`), carried here so the bar and the early
    # stop read one set.
    gate_only: tuple = ()

    def clears(self, scores: dict) -> bool:
        """The whole acceptance decision — no pixels read, no API. False on an empty dict, which is how
        an UNJUDGED candidate (every judge sample dropped) is kept out without a special case."""
        if not scores:
            return False
        sem, mn, av = _gates({k: int(v) for k, v in scores.items()}, self.gate_only)
        return sem >= self.thr and mn >= self.thr - 1 and av >= self.thr


@dataclass(frozen=True)
class AdaptiveStop:
    """When the final judge may stop sampling early, defined RELATIVE to the bar it is anticipating.

    Asymmetric margins on the RAW per-dim means so far: stop-ACCEPT when all three of the bar's gates
    clear by >= accept_margin, stop-REJECT (only when reject_margin is not None) when one is failed by
    >= reject_margin. reject > accept because an early accept admits a mask already well clear of the
    bar, where an early reject throws away a good mask and its beam on one low noisy sample.
    """

    bar: AcceptBar
    accept_margin: float = 1.0  # stop-ACCEPT once every gate is cleared by at least this
    reject_margin: float | None = 1.5  # stop-REJECT once a gate is failed by at least this

    def reached(self, verdicts: list) -> bool:
        if not verdicts:
            return False
        md = {d: sum(int(v.get(d, 0)) for v in verdicts) / len(verdicts) for d in DIMS}
        # `self.bar.gate_only`, never a local one: `_gates` is one definition only if both readers
        # hand it the same arguments.
        sem, mn, av = _gates(md, self.bar.gate_only)
        thr, Ma, Mr = self.bar.thr, self.accept_margin, self.reject_margin
        if sem >= thr + Ma and mn >= (thr - 1) + Ma and av >= thr + Ma:
            return True
        return Mr is not None and (sem <= thr - Mr or mn <= (thr - 1) - Mr or av <= thr - Mr)


def judge_consensus(
    vlm: Vlm,
    exemplars: list,
    clean: Image.Image,
    mask: Image.Image,
    defect_type: str,
    *,
    model: str,
    judge_system: str,
    definition: dict[str, Any] | None = None,
    samples: int = 3,
    extra_blocks: list | None = None,
    defect_docs: str | None = None,
    adaptive: "AdaptiveStop | None" = None,
    cache_order: bool = False,
    work_region=None,
) -> dict[str, Any]:
    """Run the judge `samples` times and AVERAGE the sub-scores to cut its ±1-2 noise, so the
    optimization signal (keep-best / CV-amount search) is stable instead of chasing variance.
    Reasons/feedback/correction are taken from the sample closest to the averaged overall.

    ``cache_order`` fronts the board-constant blocks so a prefix cache can span a board's candidates
    — see :func:`_build_judge_blocks`. It changes the prompt BYTES, so it is opt-in.

    ``work_region`` is the candidate's normalized work-region box: the zoomed view is widened to
    contain it, so the judge sees the whole host feature and not just where the mask already is."""
    # DETERMINISTIC in (clean, mask, exemplars, definition, ...), so all `samples` send byte-identical
    # bytes: build once (overlay/zoom/grids + structural CV + ~10 image encodes), loop only the call.
    blocks = _build_judge_blocks(
        exemplars,
        clean,
        mask,
        defect_type,
        definition=definition,
        extra_blocks=extra_blocks,
        defect_docs=defect_docs,
        views=Views.from_vlm(vlm),
        cache_order=cache_order,
        work_region=work_region,
    )
    cache_prefix = len(blocks) if samples > 1 else (1 if definition else 0)  # samples>1 -> cache whole prompt
    # SERIAL samples — the beam-level fill loop owns ALL the parallelism, so every call in a beam runs
    # on the beam's own thread (which is what keeps per-beam --trace-dir grouping clean). Costs
    # nothing: sample #0 warms the prompt cache and the rest cache-READ it, serial reads still hit.
    verdicts = []
    for i in range(max(1, samples)):
        try:
            stage_trace_mask("candidate_mask", mask, vlm)  # dump the exact binary candidate into this judge call
            out = vlm.call_json(
                judge_system,
                blocks,
                model=model,
                max_tokens=MAX_TOKENS_JUDGE,
                cache_prefix=cache_prefix,
                schema="judge",
            )
            verdicts.append(_parse_judge_out(out))
        except Exception as e:
            # The only trace a dropped sample leaves: a model that omits a score key turns every
            # candidate UNJUDGED and leaves run.json otherwise clean. Labelled per EXCEPTION KIND —
            # half rate limits and half transport is a different diagnosis from all of either.
            log.warning(f"degraded[judge_sample_dropped:{type(e).__name__}]: {e}")
        if adaptive and 0 < len(verdicts) < samples and adaptive.reached(verdicts):
            break  # accept/reject decision already confident -> skip the remaining samples
    if not verdicts:
        raise RuntimeError("all judge samples failed")
    avg = {k: round(sum(v.get(k, 0) for v in verdicts) / len(verdicts)) for k in DIMS}
    score = round(sum(avg.values()) / len(DIMS))
    rep = min(verdicts, key=lambda v: abs(v["score"] - score))  # representative sample for text
    return {**rep, **avg, "reasons": _merge_reasons(rep, verdicts), "score": score}


def _merge_reasons(rep: dict[str, Any], verdicts: list[dict[str, Any]]) -> dict[str, str]:
    """Each dimension's reason from ``rep`` first, else from the first sample that gave one, so EVERY
    dimension has a non-empty comment even if ``rep`` omitted one."""
    merged = {}
    for k in DIMS:
        for v in [rep, *verdicts]:
            rk = (v.get("reasons") or {}).get(k, "")
            if rk and rk != _NO_REASON:
                merged[k] = rk
                break
        merged.setdefault(k, _NO_REASON)
    return merged


def component_count(mask: Image.Image, min_frac: float = MIN_COMPONENT_FRAC) -> int:
    """Number of connected regions in a binary mask (regions below min_frac of area ignored)."""
    a = (np.asarray(mask.convert("L")) > 127).astype(np.uint8)
    if not a.any():
        return 0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(a)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_frac * a.size)


# Keyed on the image OBJECT's identity: the same clean image is segmented for every candidate
# across a beam and lives as long as the beam. PIL images define a content ``__eq__`` with no
# ``__hash__``, so a WeakKeyDictionary cannot hold them; each entry instead pairs the value with a
# weakref whose callback evicts it when the image dies, and a hit counts only while that weakref
# still points at the caller's image (a reused id reads as a miss). Unsynchronized on purpose — a
# racing thread at worst recomputes the same value.
_OTSU_MEMO: dict[int, tuple[weakref.ref, tuple]] = {}


def _otsu_structure_labels(image: Image.Image):
    """Crude, domain-agnostic structure segmentation: Otsu-split bright vs dark, label the BRIGHT
    blobs. No SAM/CAD, no domain rules. Memoized per image object in ``_OTSU_MEMO``."""
    key = id(image)
    hit = _OTSU_MEMO.get(key)
    if hit is not None and hit[0]() is image:
        return hit[1]
    g = np.asarray(image.convert("L"))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _c = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8))
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_STRUCTURE_FRAC * g.size]
    _OTSU_MEMO[key] = (weakref.ref(image, lambda _r, _k=key: _OTSU_MEMO.pop(_k, None)), (lab, keep))
    return lab, keep


def structures_spanned(mask: Image.Image, image: Image.Image) -> int:
    """How many distinct image structures (bright Otsu blobs) the mask overlaps."""
    a = np.asarray(mask.convert("L")) > 127
    if not a.any():
        return 0
    lab, keep = _otsu_structure_labels(image)
    if lab.shape != a.shape:
        a = np.asarray(resize_mask(mask, image.size)) > 127
    return sum(1 for i in keep if (a & (lab == i)).sum() > 0.05 * (lab == i).sum())


def _structure_coverages(image: Image.Image, mask: Image.Image) -> list[float]:
    """Per-structure coverage of ``mask`` against ``image``'s own Otsu structures — the same
    structures and graze rule ``structures_spanned`` counts by. [] when nothing qualifies."""
    a = np.asarray(mask.convert("L")) > 127
    if not a.any():
        return []
    lab, keep = _otsu_structure_labels(image)
    if lab.shape != a.shape:
        a = np.asarray(resize_mask(mask, image.size)) > 127
    covs = []
    for i in keep:
        s = lab == i
        hit = int((a & s).sum())
        if hit > 0.05 * int(s.sum()):
            covs.append(hit / int(s.sum()))
    return covs


def _footprint_coverage(image: Image.Image, mask: Image.Image) -> float | None:
    """Coverage of the CONVEX HULL of the structures the mask touches — the grouped footprint.

    Otsu fragments a component into its bright parts (two terminations of a resistor; a pin and its
    pad) and sees the body/gap between them as background, so per-structure coverage is blind to a
    mask that covers every part yet skips the middle. The hull of the touched parts contains that
    middle by construction — pure geometry, no semantics, any defect type. None when fewer than one
    structure qualifies (nothing to group) so absence never reads as a verdict."""
    a = np.asarray(mask.convert("L")) > 127
    if not a.any():
        return None
    lab, keep = _otsu_structure_labels(image)
    if lab.shape != a.shape:
        a = np.asarray(resize_mask(mask, image.size)) > 127
    hit = np.zeros_like(a)
    n = 0
    for i in keep:
        st = lab == i
        if int((a & st).sum()) > 0.05 * int(st.sum()):
            hit |= st
            n += 1
    if n == 0:
        return None
    pts = cv2.findNonZero(hit.astype(np.uint8))
    hull = cv2.convexHull(pts)
    fp = np.zeros(a.shape, np.uint8)
    cv2.fillConvexPoly(fp, hull, 1)
    denom = int(fp.sum())
    if denom == 0:
        return None
    return int((a & (fp > 0)).sum()) / denom


def _off_structure_share(image: Image.Image, mask: Image.Image) -> float | None:
    """The share of ``mask`` that lies on NO structure (background), same structures as above.
    The per-structure ranks are blind to this mass — a mask covering its structure fully AND a
    swath of background reads perfect there — so it is measured on its own. None on an empty mask."""
    a = np.asarray(mask.convert("L")) > 127
    if not a.any():
        return None
    lab, keep = _otsu_structure_labels(image)
    if lab.shape != a.shape:
        a = np.asarray(resize_mask(mask, image.size)) > 127
    on = np.zeros_like(a)
    for i in keep:
        on |= lab == i
    return 1.0 - int((a & on).sum()) / int(a.sum())


def fill_coverage_note(clean: Image.Image, mask: Image.Image, flags: dict | None, exemplars: list | None = None) -> str:
    """One pixel-computed sentence placing the candidate's structure coverage NEXT TO the coverage
    the type's own real annotations show, for every role that reasons about the mask.

    The reference band is measured from the exemplars with the same yardstick (each exemplar mask
    against its own image's structures), so it is scale-free — absolute example sizes never enter,
    only the dimensionless covering ratio, which does transfer across boards. With a band the note
    needs no flag semantics ("whole" = a band near 1, a partial type = a lower band), so a card
    whose fill_host was mis-judged cannot silence or misdirect it; the flag only adds wording.

    INFORMATION, not a rule: nothing rejects, gates or loops. Silent ("") whenever it cannot be
    measured honestly — an empty mask, no qualifying structure under the candidate, or no
    measurable exemplar reference."""
    covs = _structure_coverages(clean, mask)
    if not covs:
        return ""
    # Per-exemplar per-structure coverage lists, each sorted most-covered first — the reference is
    # paired with the candidate BY RANK (main structure vs main structure, graze vs graze), because
    # structure IDENTITY does not transfer across boards but the annotation's importance ORDER does.
    ex_sorted = []
    for ex in exemplars or []:
        ec = _structure_coverages(ex.defect, ex.mask)
        if ec:
            ex_sorted.append(sorted(ec, reverse=True))
    if not ex_sorted:
        return ""
    cand = sorted(covs, reverse=True)
    k_ref = max(len(e) for e in ex_sorted)
    pairs = []
    for k in range(min(len(cand), k_ref)):
        vals = [e[k] for e in ex_sorted if len(e) > k]
        pairs.append(f"its #{k + 1} at {cand[k]:.0%} (real #{k + 1}: {min(vals):.0%}-{max(vals):.0%})")
    pct = "; ".join(pairs)
    extra_line = ""
    if len(cand) > k_ref:
        extra_line = (
            f" It also touches {len(cand) - k_ref} more structure(s) at "
            + ", ".join(f"{v:.0%}" for v in cand[k_ref:])
            + " — contacts beyond what any real annotation of this type shows."
        )
    fp = _footprint_coverage(clean, mask)
    ex_fp = [v for v in (_footprint_coverage(ex.defect, ex.mask) for ex in exemplars or []) if v is not None]
    fp_line = ""
    if fp is not None and ex_fp:
        fp_line = (
            f" FOOTPRINT (the hull of the structures it touches, gaps included): this candidate "
            f"covers {fp:.0%} of it; the real annotations cover {min(ex_fp):.0%}-{max(ex_fp):.0%} "
            "of theirs — a mask that covers the parts but skips the middle shows up here."
        )
    off_line = ""
    c_off = _off_structure_share(clean, mask)
    ex_off = [v for v in (_off_structure_share(ex.defect, ex.mask) for ex in exemplars or []) if v is not None]
    if c_off is not None and ex_off:
        off_line = (
            f" OFF-STRUCTURE MASS: {c_off:.0%} of this candidate lies on no structure at all "
            f"(background); the real annotations of this type carry {min(ex_off):.0%}-{max(ex_off):.0%} "
            "off-structure. Clearly above that band, the mask is claiming background the annotation "
            "convention never does — growth must land ON the structures, not around them."
        )
    tail = ""
    if (flags or {}).get("fill_host") == "whole":
        tail = (
            " (its card marks fill_host: whole — one occurrence covers ALL of its host structure; "
            "under-coverage is the failure, 'too big' is not.)"
        )
    return (
        "STRUCTURE-COVERAGE (computed from the pixels with ONE yardstick — each mask against its "
        "own image's structures, a scale-free ratio — and PAIRED BY RANK: the candidate's k-th "
        "most-covered structure against the real annotations' k-th, so main structures compare to "
        "main structures and grazes to grazes): "
        f"{pct}." + extra_line + " A rank clearly BELOW its real range under-fills that structure — grow it, and score "
        "fill accordingly; clearly ABOVE it overshoots — trim it. Inside the range, coverage is "
        "right; judge the other dimensions." + fp_line + off_line + tail
    )


def nn_spacing(mask: Image.Image) -> float | None:
    """Median nearest-neighbour distance between connected regions, normalized by short side — a
    clustering signal (small = tightly clustered, large = spread/uniform). None if <2 regions."""
    a = (np.asarray(mask.convert("L")) > 127).astype(np.uint8)
    H, W = a.shape
    n, _l, st, ct = cv2.connectedComponentsWithStats(a)
    pts = np.array([ct[i] for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_FRAC * a.size])
    if len(pts) < 2:
        return None
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    np.fill_diagonal(D, 1e9)
    return round(float(np.median(D.min(1)) / min(H, W)), 3)


def structural_facts(clean: Image.Image, mask: Image.Image, exemplars) -> str:
    """Region count and structures spanned, for the candidate AND the examples, computed from the
    pixels. Handed to the judge as approximate, not exact: `structures_spanned` is an Otsu bright-blob
    split, and on finely segmented or low-resolution images it counts one real structure as several."""
    cand_n = component_count(mask)
    cand_span = structures_spanned(mask, clean)
    ex_n = [component_count(ex.mask) for ex in exemplars]
    ex_span = [structures_spanned(ex.mask, ex.defect) for ex in exemplars]
    return (
        "STRUCTURAL FACTS (computed from the pixels; a bright-blob split that can count one real "
        "structure as several on finely segmented or low-resolution images, so treat the numbers as "
        "approximate):\n"
        f"  CANDIDATE: {cand_n} connected region(s); overlaps {cand_span} distinct image structure(s).\n"
        f"  EXAMPLES: {ex_n} region(s) respectively; span {ex_span} structure(s) respectively.\n"
        "Read the spans PER REGION, not as one total: a connection is a single region lying across "
        "the structures it joins, and a mask may legitimately hold several such regions — how many "
        "is the distribution dimension's question. What is wrong is a REGION that reaches fewer "
        "structures than the examples show one mark of this defect reaching: a lone blob where the "
        "examples join two → low shape/semantic. "
        "(Do NOT compare absolute SIZE to the examples — image scale differs; judge coverage against "
        "the actual feature in THIS image instead.)"
        + (
            f"\n  CLUSTERING (multi-instance) — median nearest-neighbour spacing (small=tightly "
            f"clustered, large=spread/uniform): candidate={nn_spacing(mask)}; "
            f"examples={[nn_spacing(e.mask) for e in exemplars]}. Match the examples' spacing "
            "character (read it — do NOT assume a fixed pattern)."
            if (cand_n > 2 or any(c > 2 for c in ex_n))
            else ""
        )
    )


def _shape_stats(mask: Image.Image) -> str:
    """Objective per-region geometry (count + median width/height/aspect) so the judge grounds its
    shape/scale claims on NUMBERS — this is what stops confabulations like calling a wide oval 'round'."""
    a = (np.asarray(mask.convert("L")) > 127).astype(np.uint8)
    H, W = a.shape
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(a)
    comps = [
        (stats[i, cv2.CC_STAT_WIDTH] / W, stats[i, cv2.CC_STAT_HEIGHT] / H)
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_FRAC * a.size
    ]
    if not comps:
        return "empty"
    ws = [w for w, _ in comps]
    hs = [h for _, h in comps]
    ars = [w / max(h, 1e-6) for w, h in comps]
    return (
        f"{len(comps)} region(s); per-region median w={statistics.median(ws):.3f} "
        f"h={statistics.median(hs):.3f} aspect(w/h)={statistics.median(ars):.2f} "
        f"(>1 = wider than tall, <1 = taller than wide)"
    )


def _component_text(mask: Image.Image, max_components: int = 6) -> str:
    """List the candidate mask's separate regions (centroids) so the judge can assess EACH. Only when
    there are a FEW — a scattered pattern is judged holistically, not per-region."""
    a = (np.asarray(mask.convert("L")) > 127).astype(np.uint8)
    H, W = a.shape
    n, lab, stats, cent = cv2.connectedComponentsWithStats(a)
    comps = [
        (round(float(cent[i][0]) / W, 3), round(float(cent[i][1]) / H, 3))
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_FRAC * a.size
    ]
    if not (2 <= len(comps) <= max_components):
        return ""
    # No per-region CORRECTION is asked for: wiring the judge's per-region fix list into the next
    # refine turn is A/B-neutral on quality (+0.39 SE) and costs +22% input tokens — the ask buys an
    # `ops` array in the output, which enters the next turn's input, which the agent answers with more
    # ops. The centroid list and "assess each independently" stay: they change what the scores mean.
    return (
        f"This candidate mask has {len(comps)} SEPARATE regions, centroids (x,y): {comps}. "
        "Assess EACH region independently. Score the dimensions over the whole set."
    )


def _build_judge_blocks(
    exemplars: list,
    clean: Image.Image,
    mask: Image.Image,
    defect_type: str,
    *,
    definition: dict[str, Any] | None = None,
    extra_blocks: list | None = None,
    defect_docs: str | None = None,
    views: Views,
    cache_order: bool = False,
    work_region=None,
) -> list[Any]:
    """Build the judge prompt blocks — DETERMINISTIC in (clean, mask, exemplars, definition, ...), which
    is what lets ``judge_consensus`` pay the overlay/zoom/grid CV + ~10 image encodes once for all N
    samples.

    Every view goes through ``views`` so it is gridded at the size it is DELIVERED at: a grid drawn at
    native resolution and then shrunk is unreadable, and reading it is the judge's MANDATORY FIRST STEP.
    ``views`` is REQUIRED and has no default, so a forgotten argument raises instead of silently sending
    native (omitting the delivery cap raises instead of returning a native view); ``Views(0)`` is
    how a caller asks for native on purpose.

    ``cache_order`` puts every BOARD-CONSTANT group ahead of every CANDIDATE-VARYING one, so a board's
    many judge calls share a stable token prefix. OFF is the original interleaved order, byte for byte.
    """
    overlay_full = overlay_mask(clean, mask)
    # the zoom window contains ``work_region`` when the caller has one — see zoom_around_mask
    zoom, zbox = views.zoom_around_mask(overlay_full, mask, region=work_region)
    if zoom is not None:
        zoom = views.grid_fullcoords(zoom, zbox)  # grid labelled in full-image coords for precise reads
    global_view = views.grid(overlay_full)
    clean_view = views.grid(clean)

    comp_text = _component_text(mask)  # per-component centroids (only when few regions)
    cand_stats = _shape_stats(mask)
    facts = structural_facts(clean, mask, exemplars)  # deterministic connectivity / #-structures

    # Built as GROUPS, each tagged with whether this module wrote it, because `cache_order` reorders
    # them and the guard below excuses blocks BY INDEX — an index recorded while appending would point
    # at a different block once the order changed. Tagging the group instead makes foreignness travel
    # with the text, so a block added to any group here is covered unless its group says otherwise.
    _g_def: tuple[list, bool] = ([_render_definition(definition)] if definition else [], True)
    _ex_blocks: list[Any] = []
    # The distill captions are the MODEL's own words riding inside an engine group, so they are
    # exempted BY IDENTITY (the refine turn's `theirs` rule) rather than by group: the group's
    # framing stays policed, and the exemption survives `cache_order`'s reordering.
    _ex_theirs: list = []
    for i, ex in enumerate(exemplars, 1):
        _ex_blocks += [f"EXAMPLE {i} defect:", ex.defect, f"EXAMPLE {i} mask:", ex.mask]
        if getattr(ex, "observation", ""):
            _note = exemplar_note(i, ex.observation)
            _ex_blocks.append(_note)
            _ex_theirs.append(_note)
        if getattr(ex, "annotated", None) is not None:  # exemplar visual parity
            _ex_blocks += [f"EXAMPLE {i} {EXEMPLAR_VP_CAPTION}", ex.annotated]
    _g_ex: tuple[list, bool] = (_ex_blocks, False)
    _g_facts: tuple[list, bool] = ([facts], False)
    _g_cand: tuple[list, bool] = (
        [
            "CANDIDATE mask measured geometry (so you describe ITS OWN shape accurately and don't "
            f"mis-call it — e.g. calling a wider-than-tall region 'round' is WRONG): {cand_stats}.\n"
            "REFERENCE FOR GEOMETRY = THE CLEAN IMAGE, NOT THE EXAMPLES. The examples are on different "
            "images at different sizes/orientations, so absolute shape does NOT transfer — use them "
            "ONLY to learn what the defect IS and what FRACTION of its host feature it should cover. "
            "Judge the candidate's geometric FIT against the ACTUAL feature it sits on in THIS clean "
            "image: it must (a) ALIGN to that real feature, (b) cover the right PORTION of it (fill it "
            "as the defect implies — e.g. a partial-fill defect should sit within and fill part of the "
            "feature, not a thin sliver), and (c) NOT extend beyond the feature into adjacent gaps or "
            "structures. Overshooting the feature boundary AND under-filling it are BOTH faults.",
        ],
        False,
    )
    _g_type: tuple[list, bool] = ([f"Defect type: {defect_type}"], True)  # a real type is named `open`
    _g_clean: tuple[list, bool] = (
        [
            "CLEAN image — no mask, same framing + grid. Use it to see the STRUCTURES under the red "
            "overlay (which structure the mask sits ON, and which it should avoid):",
            clean_view,
        ],
        False,
    )
    _g_global: tuple[list, bool] = (
        [
            "GLOBAL view — whole clean image, candidate mask in red" + " (normalized coord grid shown):",
            global_view,
        ],
        False,
    )
    # caller-composed judge visual-prompt blocks (str | PIL.Image), and the distilled defect knowledge
    # that grounds SEMANTIC — both the caller's words, neither guarded.
    _g_vp: tuple[list, bool] = (list(extra_blocks) if extra_blocks else [], True)
    _g_docs: tuple[list, bool] = ([_DEFECT_DOCS_JUDGE_CAPTION + "\n" + defect_docs] if defect_docs else [], True)

    if cache_order:
        # Every board-CONSTANT group first, so a board's many judge calls share a token prefix; the
        # groups that depend on THIS candidate's mask (facts, its measured geometry, the overlays)
        # trail. `zoom` and `comp_text` below are candidate-varying too and already trail both orders.
        _groups = [_g_def, _g_ex, _g_docs, _g_vp, _g_type, _g_clean, _g_facts, _g_cand, _g_global]
    else:
        # The ORIGINAL interleaved order, block for block — this is the arm the delivery path runs.
        _groups = [_g_def, _g_facts, _g_ex, _g_cand, _g_type, _g_clean, _g_global, _g_vp, _g_docs]

    blocks: list[Any] = []
    # Indices of blocks this module did NOT write, so the guard at the bottom skips them.
    foreign: set[int] = set()
    for _bs, _is_foreign in _groups:
        if _is_foreign:
            foreign.update(range(len(blocks), len(blocks) + len(_bs)))
        blocks += _bs
    if zoom is not None:
        # The worked example states the OFFSET, not the op that would apply it: naming `translate`
        # hands the judge an entry from the proposer's vocabulary, which the guard below refuses.
        blocks += [
            "ZOOMED view — crop around the candidate mask in red, with a coordinate grid labelled "
            "in FULL-IMAGE normalized coords. READ precise positions off this grid (mask centre vs "
            "the target feature) to set correction amounts accurately, e.g. mask at x≈0.11, feature "
            "at x≈0.14 → it is 0.03 too far LEFT:",
            zoom,
        ]
    if comp_text:
        blocks.append(comp_text)
    # THE SAME GUARD THE JUDGE SYSTEM PROMPT GETS, on the text sent in the same call: these blocks are
    # the judge's other outgoing surface, so a guard on the system prompt alone says nothing about what
    # the judge was told. OURS ONLY — `definition` is the PROPOSER's, `extra_blocks`/`defect_docs` the
    # caller's, the distill captions the model's, and a defect type really is called `open`; a guard
    # that fired on words it does not own would be deleted by the first run that hit one
    # (the judge guard does not police words it did not write).
    _their_ids = {id(b) for b in _ex_theirs}
    assert_names_no_op(
        "\n".join(
            b for i, b in enumerate(blocks) if isinstance(b, str) and i not in foreign and id(b) not in _their_ids
        ),
        "judge_blocks",
    )
    # BLOCK ORDER constrains what the caller's `cache_prefix` can cover. In the DEFAULT order `facts`
    # sits at blocks[1] and depends on the candidate MASK, so the only stable leading block across
    # candidates is the definition and the image-heavy exemplar blocks sit behind a varying one; the
    # prefix that pays is then the WHOLE prompt across consensus samples of ONE candidate, which is
    # byte-identical, so samples 2..N cache-read everything (images included) that sample 1 wrote.
    # `cache_order` is what extends a shared prefix ACROSS candidates — at the cost of new bytes.
    return blocks


def _parse_judge_out(out: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw judge JSON to int scores over `DIMS`, guarantee a reason per dimension, and set
    the deterministic avg."""
    # A dimension the judge never wrote is NOT a zero: a truncated reply is salvaged into a partial
    # object, so zeroing the missing dims averages a cut-off sample into the consensus — and with the
    # adaptive stop, a truncated FIRST sample gives semantic=0, trips stop-REJECT and the other two
    # are never taken. The output template puts free-text `observed` BEFORE the scores, so verbose replies
    # truncate exactly there (a verdict cut mid number loses the dimension instead of gaining a different one).
    absent = [k for k in DIMS if k not in out]
    if absent:
        raise ValueError(f"judge verdict is missing {', '.join(absent)} — incomplete, not zero")
    subs = [int(out.get(k, 0)) for k in DIMS]
    for k, v in zip(DIMS, subs):
        out[k] = v
    # guarantee EVERY dimension has a reason key (so the dashboard never shows a score with no comment)
    r = out.get("reasons") if isinstance(out.get("reasons"), dict) else {}
    out["reasons"] = {k: str(r.get(k, "") or _NO_REASON) for k in DIMS}
    # Overall = AVERAGE of the six sub-scores, computed deterministically so it can never contradict
    # them. Acceptance never reads it: `AcceptBar` re-derives its gates from the sub-scores, so a weak
    # dimension cannot be diluted away by this average.
    out["score"] = round(sum(subs) / len(subs))
    return out
