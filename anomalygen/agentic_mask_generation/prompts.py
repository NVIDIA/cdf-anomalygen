# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable prompt-sets for the agentic mask-generation engine.

A :class:`PromptSet` bundles the prompt text the engine needs — the proposer/agent system prompt, the
judge system prompt, and the few per-turn lines around them — behind a tiny
:class:`typing.Protocol`. :class:`DefaultPrompts` holds the production text; the engine reaches it
through :func:`by_name` (keyed on ``cfg.prompt_set``) or an explicit ``run_beam(prompts=...)``.

Iterate on prompts without touching the engine: subclass :class:`DefaultPrompts`, override one
method, decorate it ``@register("myv2")``, and select it via ``cfg.prompt_set="myv2"`` — or pass a
``PromptSet`` instance straight into ``run_beam`` for a throwaway A/B —
the refine ask is reachable from a prompt set without editing the engine.
"""

import re
from typing import Protocol

from anomalygen.agentic_mask_generation.config import DIMS
from anomalygen.agentic_mask_generation.ops import CROSS_CUTTING, CV_KINDS, OPS

# --------------------------------------------------------------------------------------------------
# The default production prompt text, as module constants. The `__NAME__` spans are placeholders
# filled in by `DefaultPrompts` below, from the ops and card flags this run actually has.
# --------------------------------------------------------------------------------------------------
AGENT_SYSTEM = """\
PURPOSE. You produce a binary defect MASK on a CLEAN image, by analogy to in-context examples.
The mask is NOT a defect image — it is a region that will later CONDITION a downstream
mask-guided anomaly generator. So your job is to choose a placement/shape that is both VALID
(functionally consistent with what the defect IS) and PLAUSIBLE (where such a defect could really
occur on THIS image). There is no single right answer — many placements are valid. You are given
(defect image, mask) example pairs as the only guidance; the clean image's own "correct" mask is
NOT available to you.

HOW THIS LOOP WORKS (the framework — play by these rules).
- Turn 1 you study the examples and PROPOSE a definition + an initial mask spec.
- Each later turn you SEE your current mask rendered on the clean image — a GLOBAL view plus a
  real ZOOMED-IN crop around the mask — and you pick ONE next action (below).
- Rendering and CV edits are DETERMINISTIC local operations (no model call, no cost) — prefer a
  cheap CV edit over redrawing whenever the mask is only slightly off.
- It is a stateful conversation: you remember every prior attempt, score and decision — build on
  them, never repeat a move that already failed. The loop keeps your BEST mask and runs for a
  limited number of turns.
- IMPORTANT — your self-scores do NOT decide acceptance. An INDEPENDENT judge re-scores the mask by
  consensus and that score is authoritative. It is given the DEFINITION you write below — your
  image understanding, the mechanism, and the per-dimension target and reference — and told to score
  against it. That definition is the only thing you control about how the mask is judged, so write
  it as the rubric it becomes, not as a note to yourself. Inflating your own scores will NOT get a mask
  accepted — it only wastes turns. So score yourself HONESTLY and strictly; that is how you
  actually converge.

The mask is drawn by applying a SEQUENCE of ops to a blank canvas. EVERY op MUST include its full
coordinates — an op without them draws NOTHING (empty mask):
__INIT_OPS__

__STAMP_ADVICE__

Understand the defect by combining the EXAMPLES with your OWN understanding of what the structures
in the image ARE and how this kind of defect physically occurs (use your domain knowledge freely —
just don't CONTRADICT what the examples show). __RELPOS__

THE SHARED RUBRIC — SIX dimensions, used IDENTICALLY at every stage: you write a per-dimension
target for them in your definition, you score them each turn, and the independent judge scores the
SAME six. Judge each against the REFERENCE noted:
- ROI (legal region — coarse): the mask sits on the KIND of structure the EXAMPLE masks sit on and
  AVOIDS the structures they never cover. REF: examples → checked against THIS clean image.
- PLACEMENT (precise position — fine): precisely ALIGNED on the specific local feature the examples
  sit on, and — for a connection-type defect — actually JOINING the correct structures and the right
  NUMBER of them, each such mark a CONTINUOUS region, the way the EXAMPLES show. STUDY the examples to see
  WHERE on the structures the defect sits and how it joins them — do not assume a spot; reproduce
  the relationship the examples show. REF: the examples' relative placement + this clean image.
  __SITES_EXPECTATION__
- SHAPE (form/topology): form, connectivity (one connected region vs separate), elongation and
  orientation; HUGS the real feature's outline, not a crude geometric blob. REF: example masks'
  form + the actual feature outline in the clean image.
- FILL (coverage of the target feature): covers the RIGHT PORTION of the actual feature in THIS
  image — does NOT overshoot beyond it into gaps/adjacent structures, and does NOT under-fill it.
  REF: the actual feature in the clean image (+ how much of it the examples cover).
- DISTRIBUTION (multi-instance only): READ the distribution CHARACTER from the EXAMPLE masks and
  reproduce IT — how many instances, and how they are spaced (tightly clustered? loosely spread?
  uniform? grouped near certain features?). Different defects distribute differently, so do NOT
  assume any fixed pattern — match what THESE examples actually show. REF: the example masks (+ the
  measured stats provided). Single-instance is NOT a free pass: it scores whether the mask IS one
  connected region, and a candidate that fragmented into pieces FAILS this dimension.
  __DISTRIBUTION_EXPECTATION__
- SEMANTICALLY CORRECT (gating): FIRST understand what the structures in THIS image ARE and the
  PHYSICAL CAUSE / MECHANISM of the defect (from the examples + your reading of the image), THEN
  make sure the mask is a TRUE instance under that mechanism, ON THE RELEVANT STRUCTURE — place it
  where the defect actually occurs (on the main relevant structure the examples show it on), and do
  NOT float it on an unrelated/secondary feature elsewhere just because that spot superficially
  resembles one. If the defect joins structures, it must actually join the right number of them.
  REF: your causal understanding (examples + image) + the clean image.

Score all six (roi, placement, shape, fill, distribution, semantic) 0-10 each turn, strictly and honestly.

HOW THIS RUNS. You do NOT work on one mask a turn at a time. You propose SEVERAL candidates at
once, and every refine round you score ALL of them and send a fix for EACH — they are refined in
parallel and the strongest survive. So spend turn 1 on genuinely DIFFERENT placements rather than
variations of one idea: a candidate you do not explore is one nobody else will.

Each refine round, for EVERY candidate, send exactly one of:
- {"id":k,"ops":[{"kind":..., <params>, "target":[x,y] optional}]} — edit candidate k. Use this for
  any position/size/orientation/thickness fix. kinds and their params (normalized units):
__OPS__
  "target":[x,y] applies the op ONLY to the connected region nearest that point — use it to fix
  ONE bad region in a multi-region mask and leave the correct ones untouched.
__RESTART_ADVICE__- {"id":k,"keep":true} — candidate k is already as good as you can make it this round.

You do NOT decide acceptance. An INDEPENDENT judge scores every surviving candidate on the same six
dimensions and a fixed bar decides — semantic is a hard gate, no dimension may fall far below the
bar, and the mean must clear it. Your own scores decide which candidates get refined and judged, so
score them strictly and honestly: inflating one wastes the round it wins, and deflating a good one
throws it away before the judge ever sees it.
"""

# Relative-placement guidance, always applied: it makes the agent place the defect RELATIVE to its
# host structure in THIS image rather than at the examples' absolute coordinates.
_RELPOS_OURS = (
    "Apply it to THIS clean image, which may be oriented, scaled, OR POSITIONED/FRAMED differently — "
    "the SAME host structure (the band/feature the defect sits on) can appear at a DIFFERENT absolute "
    "position here than in the examples. So FIRST locate that host structure in THIS image, then place "
    "the defect RELATIVE to it (same relationship the examples show, e.g. same edge/end alignment). "
    "Match the RELATIONSHIP, NOT absolute coords — never reproduce the examples' absolute position/Y "
    "just because that is where the defect sat in their frame."
)

JUDGE_SYSTEM = """\
You are a strict inspector for synthetic-defect MASK placement. You score whether a candidate
mask is a VALID and PLAUSIBLE place/shape to synthesize a given defect on a clean image. There
is NO single correct answer — many placements can be valid. Judge by the defect's nature, as
given by the shared DEFINITION and the examples, NOT by matching any one example mask. USE your
own understanding of what the structures are and how this defect physically arises — just don't
contradict what the definition and examples actually show.

You MUST reason from BOTH of these, together, before scoring:

(1) SEMANTIC / FUNCTIONAL meaning of the defect. Understand what physically makes a mask VALID
    — the condition that defines the defect, using the examples AND your knowledge of how such a
    defect occurs. A mask that satisfies that condition is valid even if its fine position differs
    from the examples. For instance, if the examples show the defect CONNECTING two structures,
    then any mask that actually spans those structures is valid — and the exact spot along them
    may not matter. Do NOT invent extra positional requirements that aren't needed for the defect
    to be what the examples show it is.

(2) THE USER'S EXAMPLES. Study the provided (defect, mask) pairs CAREFULLY and concretely
    before judging — their exact position, shape, and RELATIVE relationship to local structures
    (which structures the mask touches, how it spans them, its proportion). Ground your reasons
    in what you ACTUALLY SEE in these masks. Never assert a criterion (e.g. "should be on the
    outer edge") unless the example masks actually show it — if the examples sit on the inner
    side, then inner is correct.

Judge the candidate against BOTH: is it FUNCTIONALLY valid (1) AND consistent with the
appearance the examples actually show (2)? Apply to THIS input's own structures (it may be
oriented/scaled differently). Do NOT penalize fine differences that neither break the defect's
meaning (1) nor contradict the examples (2).

You are given:
- The shared DEFINITION written by the proposer: image understanding, the defect's mechanism, and
  the per-dimension target + reference for roi / placement / shape / fill / distribution / semantic.
  Score AGAINST it (do not silently re-interpret the task).
- EXAMPLE pairs (defect image + mask) showing valid placements.
- A GLOBAL view: the whole clean image with the candidate mask in red — judge whether it is on
  the RIGHT REGION / structure.
- A ZOOMED view: a crop around the candidate mask in red — judge (a) whether it PRECISELY sits
  on the right local feature (not adjacent/empty/wrong areas), and (b) COVERAGE: whether the
  red region covers the appropriate EXTENT of that feature, as the examples do (not a small
  corner, not spilling outside). Both fine misplacement and under/over-coverage are only
  visible here — weight heavily.
- The defect type name.

MANDATORY FIRST STEP — do NOT skip, do NOT rely on assumptions. READ the zoom's coordinate
grid and write "observed": the mask region's coordinate(s); the nearby structures' coordinates;
and which DIRECTION is toward vs away from the body/centre (so inner/outer is grounded in what
you SEE, not a prior). All placement reasoning MUST be consistent with these read coordinates —
e.g. if the mask is already at the outer edge per the grid, do NOT say "move further out".

MANDATORY ON-FEATURE CHECK (in the ZOOM, this image only — do NOT compare brightness across images):
look at what is DIRECTLY UNDER the red in the zoom. Does the red actually OVERLAP the feature the
defect occurs on (the same visible structure the EXAMPLE masks cover), or does it sit BESIDE / just
off it (on the adjacent dark area, gap, or a different part of the structure)? State this in
"observed". If the red is NOT on the feature itself, PLACEMENT and SEMANTIC are LOW (≤4) — a mask
next to the feature is not on it, no matter how tidy.

MANDATORY HOST-REGION GROUNDING (do this BEFORE scoring placement) — read it from the examples
combined with your understanding of what the structures are:
- Look at the pixels UNDER each EXAMPLE mask. Name the kind of surface/structure the defect sits
  ON, described purely by what you SEE (e.g. "small bright features", "the elongated structures",
  "the flat background between parts"), and just as importantly name the structures the example
  masks NEVER cover (the EXCLUSIONS — e.g. a large uniform region that stays clear).
- Now look at the clean image under the CANDIDATE mask. Does it sit on the SAME kind of
  structure, and does it AVOID the excluded structures?
- If the candidate sits on a structure the examples never mark (e.g. on a large body/region that
  every example keeps clear), "roi" is LOW (≤3) no matter how tidy it looks — this is the
  single most common failure. For multi-instance defects, EVERY instance must be on a valid host
  structure; instances landing on an excluded structure each cost "roi".
Write this region check into "observed" too.

Score SIX dimensions (each 0-10) — the SAME rubric the generator used and that is written into
the shared DEFINITION. Score EACH against what the DEFINITION says for it AND the reference noted:
- "roi" (Region Of Interest = legal region): is the mask in a VALID region for this defect — on
  the KIND of structure, AND on the relevant PART of the scene, that the examples show this
  defect occurring on (the main relevant structure), AVOIDING the structures they never
  cover? REFERENCE: the example (defect,mask) pairs, checked against THIS clean image's structures.
  A mask on an excluded structure, OR floating on an UNRELATED / secondary feature elsewhere (not
  the main component the defect belongs to) just because that spot superficially resembles one, is
  LOW (≤3) however tidy — the main relevant component must be where the defect is placed.
- "placement" (precise position within the ROI): is it precisely positioned/ALIGNED on the
  SPECIFIC local feature (e.g. on the feature itself, not the adjacent gap), and — for a
  connection-type defect — does it relate to the CORRECT structures (the right ones, the right
  NUMBER)? This is the FINE position check, distinct from being in the right region (roi).
  REFERENCE: the examples' relative placement + THIS clean image's local features (read off the zoom).
  __SITES_EXPECTATION__
- "shape" (form / topology): judge by FUNCTIONAL PLAUSIBILITY, not strict resemblance. The
  examples often VARY in exact shape/size, so do NOT require the candidate to match any one of
  them — give ≥8 to ANY believable defect form that has the right topology (connected vs separate)
  and roughly hugs the real feature. Only score LOW (≤5) for a clearly wrong form: wrong topology
  (e.g. disjoint when it must connect), or a crude geometric blob that ignores the feature entirely.
  A slightly rounded / blocky / not-pixel-perfect outline is FINE (≥7) — a real defect mask itself
  is rarely a perfect hug. For a CONNECTION/elongated defect, a solid blob/capsule that spans the
  structures with the right axis/orientation is a GOOD shape → ≥8; do NOT withhold ≥8 just because
  it is a smooth blob rather than tracing an exact contour (real defect masks are often simple blobs).
  REFERENCE: the defect's form character from the examples (allowing their variance) + the feature.
- "fill" (how well it covers the target feature): give ≥8 if the mask sits ON the feature and
  covers a REASONABLE portion of THAT feature. Minor spill or being a bit large/small is FINE (≥7).
  PENALIZE: gross OVERSHOOT far beyond the feature, OR covering only a tiny sliver OF THE FEATURE
  (gross under-coverage of the feature it sits on) → fill LOW. For a CONNECTION, "the feature" is
  the two structures PLUS the gap between them: a mask that covers the gap and reaches both
  structures is good fill → ≥8 (do not demand more). REFERENCE: the ACTUAL feature in the clean
  image (NOT the examples' absolute size — image scale differs).
- "distribution" (multi-instance only): does the candidate REPRODUCE the distribution CHARACTER the
  EXAMPLE masks actually show — read it from them: the count, and the SPACING (tightly clustered vs
  loosely spread vs uniform vs grouped near features)? Match whatever pattern THESE examples have
  (the measured stats + masks tell you) — do NOT assume one. If the candidate's count or spacing
  clearly departs from the examples' (e.g. far fewer, or uniform when the examples cluster), score
  LOW. A connection defect is one CONNECTED blob per occurrence — but a mask may legitimately hold
  several occurrences, so do not read "it is a connection" as "there should be only one".
  __DISTRIBUTION_EXPECTATION__
- "semantic" (semantically correct — the GATING sanity check): FIRST understand what the
  structures in THIS image actually ARE and the PHYSICAL CAUSE / MECHANISM of this defect (from the
  examples + your comprehension of the image + your knowledge of how this defect forms); THEN judge
  whether the mask is a TRUE instance of the defect under that mechanism, ON THE RELEVANT STRUCTURE.
  CRITICAL: the defect must be placed where it would actually occur on the MAIN relevant component
  (the main relevant structure), as the examples show — NOT floating on some unrelated
  or secondary feature elsewhere in the image just because that spot superficially looks similar. If
  the main component that should carry this defect does NOT actually have it where the mask is (the
  mask is on the wrong part of the scene), semantic is LOW (≤3) even if the patch looks tidy. Other
  checks: a connection-type defect must ACTUALLY connect the structures it shorts (same NUMBER the
  examples do — two separate dots are not a connection); a surface defect must lie on the relevant
  surface. REFERENCE: your causal understanding (examples + image) + the clean image. Score STRICTLY
  and INDEPENDENTLY — a mask can be tidy on shape/fill yet be semantically wrong; NOT the average.

The overall score is computed automatically as the AVERAGE of the six sub-scores, so focus on
scoring each dimension honestly and consistently with its reason — do not set an overall yourself.

For EVERY one of the SIX dimensions you MUST give a non-empty "reason" explaining its score,
grounded in what you read/understand — never leave any dimension with a score but no reason. This
per-dimension analysis is shown to the user.

Output JSON only (no overall score — it is computed from the six sub-scores):
{"observed": "<read off the zoom grid: mask at (x,y); structures at ...; what the structures ARE and the defect's mechanism>",
 "roi": <0-10>, "placement": <0-10>, "shape": <0-10>, "fill": <0-10>, "distribution": <0-10>, "semantic": <0-10>,
 "reasons": {"roi":"<full reason>","placement":"...","shape":"...","fill":"...","distribution":"...","semantic":"<mechanism-based: what the structures are, the defect cause, and whether the mask is a true instance>"},
 "feedback": "<as detailed as needed: name the worst dimension(s) and exactly how to fix it>"}
"""


class PromptSet(Protocol):
    """Every method the engine calls on a prompt-set. Any object exposing these is a drop-in.

    Nothing enforces this at runtime — not `runtime_checkable`, `BeamContext` annotates its `prompts`
    field `Any`, and no type-checker runs here — so a method added to `DefaultPrompts` has to be added
    here by hand. the refine ask is reachable from a prompt set without editing the engine.
    """

    name: str

    def proposer_system(self, allowed_ops=None, distribution_kind: str = "", layers=(), sites_spanned: str = "") -> str:
        """The proposer/agent system prompt (studies the examples, proposes + refines masks)."""
        ...

    def judge_system(self, allowed_ops=None, distribution_kind: str = "", sites_spanned: str = "") -> str:
        """The independent judge system prompt (scores a candidate mask 0-10 on the shared rubric)."""
        ...

    def shape_target(self, card) -> str:
        """The shape the card asks for, restated for the turn that has to hit it."""
        ...

    def refine_frame(self, from_crop) -> str:
        """Which frame a refine turn's numbers are read in — the assigned crop, or the full board."""
        ...

    def zoom_units(self, board, mask_px, view_w) -> str:
        """What one unit means in the zoom view a judge is shown."""
        ...

    def refine_ask(self, n, *, last, restart) -> str:
        """The rubric a refine turn scores on, and the JSON its answer must arrive in."""
        ...

    def weak_dim_steer(self, dims, weak, verb, feedback=None) -> str:
        """An authoritative verdict on the round's best, worded as the next round's instruction."""
        ...


# Op names that are ORDINARY ENGLISH in the text this guard reads, where a bare mention is a word and
# not an offer. Seven of them are measured — these exact sentences render into the system prompts this
# package ships: "image scale differs", "a smooth blob", "the mean must clear it", "a solid
# blob/capsule", "the outer disk minus the inner one", "which a round brush cannot make", "a crisp
# rectangle through to a rounded blob". `close`, `open` and `arc` are here for the plainer reason that
# they are ordinary words ("the mask is close to the pad", "an open gap") before they are op names.
# For all ten `leaked_ops` wants a syntactic marker — or the imperative form below, which is the one
# prose shape that IS an offer.
#
# AN EXCEPTION LIST, and the polarity is the point: every op NOT named here is matched bare, so a new
# op joins the day it is registered — an op offered in prose is seen for every op in the registry.
# A STALE entry is the silent direction: the set is only ever read for ops that exist, so a name left
# here after its op is deleted is never consulted and nothing fails.
_ENGLISH_WORD_OPS = frozenset(
    # "ring" earned its place in the field: a distilled card describing a "circular ring structure"
    # (SEM contact pads) tripped the bare-mention rule on every judge call and zeroed the class.
    {"arc", "brush", "capsule", "clear", "close", "disk", "open", "rectangle", "ring", "scale", "smooth"}
)

# The one prose form that offers an op without any syntactic marker: the name used as an IMPERATIVE
# VERB with an object — "rotate it by 15 degrees", "You may scale the mask up", "smooth its outline",
# "then clear anything outside the pad". BOTH halves are required. The cue alone matches "the coarse
# 0-10 scale — keep editing"; the object alone matches "the mean must clear it", a sentence about the
# accept bar that AGENT_SYSTEM really contains. Together they separate the four defeats above from
# every ordinary-English use in this package's own text.
#
# The NAME is matched case-sensitively, like every marker above it: the engine writes an op name in
# lower case everywhere it emits one — JSON kinds, catalogue heads, `ops.OPS` keys — so a capitalised
# occurrence is emphasis on the English word. "compare its OWN count/spacing to these and CLOSE the
# gap" (regions.ex_dist_summary) is that, and it is not an offer of `close{px}`. The cue and object
# words around it are matched either way, since those are ordinary words wherever they fall.
_DIRECTIVE_CUE = (
    r"(?:^|(?<=[.;:,\n])|(?i:\b(?:and|or|then|also|use|using|prefer|apply|try|never|"
    r"first|next|instead|may|can|you\s+must|you\s+should)\s+))\s*"
)
_VERB_OBJECT = (
    r"(?i:(?=\s+(?:it|its|the|this|these|each|every|all|any|anything|them|their|your|"
    r"an?)\b))"
)
# A GENERATED catalogue entry — `translate{dx,dy} — slide it; ...`. The imperative rule does not read
# these lines: an entry is documentation of the op that heads it, written in the imperative because
# that is how an instruction reads ("scale_layer{...} — FIT: scale each region up to ..."), and the
# op it actually offers is already decided by the markers. Bare matching still applies here, so an
# entry that names a DIFFERENT withheld op is still caught.
_ENTRY_LINE = re.compile(r"\b(?:%s)\s*\{" % "|".join(sorted(CV_KINDS, key=len, reverse=True)))


def _plain_cv_guide(allowed=None) -> str:
    """The plain-language guide, cut down to the ops this run may actually use.

    Filtering is by BLOCK — an op's entry, or a whole prose paragraph — because a wrapped sentence cut
    line by line leaves a fragment that still reads as advice. Whatever survives is passed through the
    same guard that asserts on the finished prompt, so the guide cannot leak whatever else is written
    into it later.
    """
    # `is None`, not falsiness: `()` is a REAL answer — "this run allows nothing" — and it is
    # reachable, e.g. `--allowed-cv-ops snap_layer,fill_layer` on a board with no guidance layer,
    # where the layer withdrawal empties the set. Reading it as "no restriction" here would offer
    # the whole guide to a run that may use none of it.
    if allowed is None:
        return _PLAIN_CV_GUIDE
    if not allowed:
        return ""
    blocks, cur = [], []
    for line in _PLAIN_CV_GUIDE.splitlines():
        m = re.match(r"\s*-\s*([a-z_]+)", line)
        starts = (m and m.group(1) in CV_KINDS) or not line.strip()  # an op entry, or a blank line
        if starts and cur:
            blocks.append(cur)
            cur = []
        cur.append(line)
    blocks.append(cur)
    kept = [b for b in blocks if not leaked_ops("\n".join(b), allowed)]

    def head(b):
        return next((l for l in b if l.strip()), "")

    out = []
    for i, b in enumerate(kept):  # a section heading with no entry left under it
        if re.match(r"\s*\d\)", head(b)):
            nxt = next((head(x) for x in kept[i + 1 :] if head(x)), "")
            if not re.match(r"\s*-", nxt):
                continue
        out.extend(b)
    return "\n".join(out)


def leaked_ops(text: str, allowed) -> list[str]:
    """CV ops NAMED as ops in ``text`` that ``allowed`` does not contain.

    An op is "named as an op" when it appears with its parameter brace (``translate{dx,dy}`` or
    ``translate {dx,dy}``), as a catalogue bullet (``- conform:``), as a JSON kind
    (``"kind":"translate"``), quoted on its own (``"conform"``) — which is how every catalogue and
    instruction refers to one — or, in prose, as an imperative verb taking an object ("rotate it by 15
    degrees"). A BARE mention counts too, for every op except the handful of names this package's own
    prompts use as English words (see :data:`_ENGLISH_WORD_OPS`); the stamp primitive's own
    ``"scale":`` field is excluded by the quote form, so this can be asserted on.

    WHICH WAY IT ERRS: towards missing a bare mention, and only of the names in that list — "a clear
    region" is prose, "clear the candidate" is an offer and the imperative form catches it.
    ordinary english is not read as an offer,
    an op offered in prose is seen for every op in the registry.
    """
    keep = None if allowed is None else set(allowed)
    if keep is None:
        return []
    # The lines that are NOT generated catalogue entries; the imperative rule reads only these.
    prose = "\n".join(ln for ln in text.splitlines() if not _ENTRY_LINE.search(ln))
    out = []
    for k in sorted(CV_KINDS - keep):
        pat = r"\b%s\s*\{|^\s*[-*]\s*%s\s*[:{]|\"kind\"\s*:\s*\"%s\"|\"%s\"(?!\s*:)" % (k, k, k, k)
        if k not in _ENGLISH_WORD_OPS:
            pat += r"|\b%s\b" % k
        if re.search(pat, text, re.M):
            out.append(k)
        elif re.search(_DIRECTIVE_CUE + k + _VERB_OBJECT, prose, re.M):
            out.append(k)
    return out


def assert_no_leak(text: str, allowed, where: str) -> str:
    """Refuse to send a prompt that offers a tool the run cannot use — a silently-leaked op looks
    exactly like an honoured restriction.
    no turn the model is actually SENT names a withheld op."""
    bad = leaked_ops(text, allowed)
    if bad:
        raise AssertionError(f"{where} names unavailable ops {bad}; build the text from ops_catalogue()")
    return text


def assert_names_no_op(text: str, where: str) -> str:
    """Refuse to send JUDGE text that names an op — ANY op, whatever the run allows.

    The judge scores and explains; the proposer decides how to fix. The op vocabulary is not a thing
    the judge has less of on a restricted run, it is a thing the judge does not have, and the EMPTY
    allowed-set is that invariant written down — asked "does this name an op the run does not offer",
    an unrestricted run answers no to every op.
    the judge guard is the empty allowed set not the runs.
    """
    return assert_no_leak(text, (), where)


_BRUSH_NORM = (
    'Give "path":[[x,y],...] in image coords [0,1] — the same frame as every other op — and\n'
    '    thickness as "width_rel", a fraction of THE MASK YOU ARE EDITING (0.3 = about a third\n'
    "    as thick as it is, 1.0 = as thick as it is). Thickness is deliberately not a fraction\n"
    "    of the image: a defect covers a fraction of a percent of a board, so an image-relative\n"
    "    stroke is far too coarse to edit one with."
)


def zoom_units(board: tuple[int, int], mask_px: tuple[float, float], view_w: float) -> str:
    """The unit half of a candidate's zoom caption.

    A magnified view invites the agent to size an edit against what it can see, which is the view, not
    the board — so the caption has to say which frame each number is in, and quote a worked thickness."""
    W, H = board
    pw, ph = mask_px
    head = f" — magnified view of a {view_w:.0f} px-wide part of the {W}x{H} px board; the grid is in IMAGE coords."
    return (
        f"{head} This mask spans {pw / W * 100:.1f}% x {ph / H * 100:.1f}% of the board, so a stroke "
        'of "width_rel": 0.3 is about a third of its thickness — that field is a fraction of THIS '
        "mask, not of the board or of this view."
    )


def shape_target(card: str | None) -> str:
    """The card's own words for what this defect LOOKS like, to restate while the mask is being shaped.

    The card is sent once, in the propose turn, and then several rounds of images and scores are
    stacked on top of it. Without this line the form the run is aiming at is stated at the moment the
    mask is created and never again while it is being reshaped, and the mask drifts towards a tidy
    blob.
    """
    if not card:
        return ""
    body = card.split("## typical shape", 1)
    if len(body) < 2:
        return ""
    text = body[1].split("##", 1)[0].strip().replace("\n", " ")
    if not text:
        return ""
    out = (
        "SHAPE TARGET for this defect, from its reference card — shape the mask TOWARDS this, not "
        f"towards a tidy blob: {' '.join(text.split())}"
    )
    # The size section and the fill_host flag get the same restatement the shape prose already
    # earned, for the same reason: stated only at propose, the EXTENT the run is aiming at is
    # forgotten while the mask is reshaped, and the mask drifts small. Card-derived text only —
    # nothing here is specific to any defect.
    size = card.split("## typical size / extent", 1)
    if len(size) > 1:
        stext = " ".join(size[1].split("##", 1)[0].split())
        if stext:
            out += f" SIZE/EXTENT TARGET, same card: {stext}"
    m = re.search(r"^fill_(?:host|target):\s*(\w+)", card, re.M)
    if m and m.group(1) == "whole":
        out += (
            " (fill_host: whole — one occurrence covers ALL of its host structure; "
            "under-coverage is the failure, 'too big' is not.)"
        )
    elif m and m.group(1) == "part":
        out += (
            " (fill_host: part — it covers only PART of one host structure; overshoot and under-fill are both wrong.)"
        )
    return out


def refine_frame(from_crop: bool) -> str:
    """The coordinate frame for a refine turn, stated outright.

    The propose turn hands the agent a CROP and calls it the entire canvas; refine then shows the
    whole board using the same field names. Unannounced, that change applies crop-unit strokes as
    board fractions — a ~10x overshoot that destroys a candidate on its first edit.
    """
    head = "COORDINATE FRAME for this turn: the FULL image, normalised [0,1]. Every grid below is labelled in it."
    if not from_crop:
        return head
    return (
        head + " This is a CHANGE from the propose turn, which gave you one region's crop and asked "
        "for coordinates inside that crop. Those numbers do NOT carry over: re-read every position "
        "and every stroke width off the grids in front of you now."
    )


# The scoring instruction every refine turn opens with — the same six dimensions the system prompt
# defines and the judge scores, restated in one line beside the candidates being scored.
_REFINE_RUBRIC = (
    "Score EACH candidate on the SHARED 6 dimensions 0-10, each against your "
    "definition target+reference: roi (legal KIND of structure, avoid excluded — coarse), "
    "placement (precisely aligned on the local feature, not the gap; right structures/number "
    "— fine), shape (form/topology, hug real outline), fill (right portion of the ACTUAL "
    "feature in this clean image — no overshoot/under-fill), distribution (count/spread vs "
    "examples; single→full), semantic (understand structures+mechanism, then is it a TRUE "
    "instance — connection connects the right NUMBER, etc.). Geometry judged vs the CLEAN "
    "image, not the examples' absolute shape."
)


def refine_ask(n: int, *, last: bool, restart: bool) -> str:
    """What a refine turn ASKS FOR: the rubric, and the JSON the round's answer must arrive in.

    ``n`` is how many candidates the round holds, ``last`` marks the round that only assesses (the
    final judge scores everything afterwards, so a fix would never be applied), and ``restart`` offers
    the `clear`-led redraw — which is an OP like any other, so a run that withheld it, or a
    fixed-shape run whose fixes are filtered down to rigid edits, must not be offered the hatch.

    `reason` is asked for and NOTHING reads it, deliberately: asking a model to state its reason can
    change the SCORES it gives even when the text is discarded, so dropping the ask is a measured
    prompt change rather than a cleanup.
    The length asked for and the length read are the same constant.
    """
    if last:
        return (
            _REFINE_RUBRIC + " This is the FINAL round — just assess. Output JSON "
            '{"assess":[{"id":0,"roi":0,"placement":0,"shape":0,"fill":0,"distribution":0,'
            '"semantic":0}, ...]} for every candidate. Scores only — no "reason" field on '
            "this round, it is not read."
        )
    return (
        _REFINE_RUBRIC + " Then in the SAME response output a FIX for EVERY candidate (do NOT "
        "discard the weak ones — many are salvageable). Each fix targets that candidate's "
        "weakest dimensions. A fix is ONE of: "
        '{"id":k,"ops":[<cv ops from the catalogue you were given>]} (refine candidate k '
        "via CV — preferred when it is roughly right), OR "
        + (
            '{"id":k,"ops":[{"kind":"clear"}, <ops that draw a new mask>]} (start '
            "candidate k over — only if fundamentally wrong), OR "
            if restart
            else ""
        )
        + '{"id":k,"keep":true} (already good, leave unchanged). Output JSON '
        '{"assess":[{"id":0,"roi":0,"placement":0,"shape":0,"fill":0,"distribution":0,"semantic":0,'
        f'"reason":""}}, ...for ALL {n}],"fixes":[{{"id":0,...}}, ...one per candidate]}}. '
        'Keep each "reason" to at most 100 characters — one clause naming the weakest '
        "dimension and why. Only the first 100 characters are ever read."
    )


def weak_dim_steer(dims: dict, weak, verb: str, feedback: str | None = None) -> str:
    """An authoritative verdict on the round's best candidate, turned into the next round's steer.

    ``verb`` names what the agent is about to do with the candidates the steer is about.

    ``feedback`` is the judge's own sentence, and it is included whenever it is not None — including
    when it is empty, which is what a judge that returned no feedback field says. The caller that has
    no judge at all passes None and the segment is absent.
    """
    return (
        ", ".join(f"{k} {dims[k]}" for k in DIMS)
        + f". Weakest: {', '.join(weak)}. "
        + ("" if feedback is None else f"Judge feedback: {feedback} ")
        + f"When you {verb} the NEXT candidates you MUST specifically fix these weakest "
        "dimensions, choosing from the ops you have: shape/fill → hug the real feature; "
        "placement → move onto it; roi → move to the legal "
        "structure; distribution → match the examples' count/spread. Build on this best, "
        "do not regress the dimensions that are already high."
    )


_ALL_LAYER_NAMES = ("normallight", "sam2", "solderlight")  # every visual prompt that carries one


_RESTART_ADVICE = """- {"id":k,"ops":[{"kind":"clear"}, <ops that draw a new mask>]} — start candidate k OVER. `clear`
  empties the mask and the ops after it draw from a blank canvas, so this THROWS AWAY every edit
  that candidate has accumulated. Use it ONLY when the mask is FUNDAMENTALLY wrong (wrong
  region/structure, wrong topology, wrong distribution) — never for a position/size/shape offset,
  which is what the ops above are for. IMPORTANT: if your edits are making the mask geometrically
  CLOSER but the integer score has not moved, that is just the coarse 0-10 scale — keep editing
  (try a different op or amount); do NOT start over because the number plateaued.
"""

_STAMP_ADVICE = """PREFER A STAMP for high quality. A geometric primitive can only APPROXIMATE the defect's real form,
so it caps shape/fill/semantic around the borderline. A STAMP pastes the REAL example shape, which
scores clearly higher (and for multi-instance defects, stamping a 'full' example mask reproduces its
real count + clustering at once). DEFAULT to a stamp of the example shape that best matches, placed
on the right structure and scaled/rotated to fit THIS image, then refine it; use geometric
shape ops only for genuinely simple shapes where no example shape fits. Aim for CLEAR margin (each
dimension solidly ≥9), not just passing. Example: {"ops":[{"kind":"stamp","shape_id":0,"center":[0.12,0.45],"scale":1.0}]}"""


def _init_kinds(allowed=None):
    """The ops turn 1 can offer: those that need no existing mask, intersected with what this run
    allows. Derived from `requires`, so a new op joins the right turn by declaring what it needs."""
    init = {k for k, sp in OPS.items() if "mask" not in sp.requires}
    return init if allowed is None else (init & set(allowed))


def _units_line(ops, *, init: bool = False) -> str:
    """The size fields and the unit the MODEL writes them in, collected from `Param.unit`.

    NOT from `ops.FIELD_SCALE`, which answers a different question (how a field transforms between
    the crop frame and the board frame). They disagree for `scale`, which remaps by the short-side
    ratio but is DECLARED as a multiplier of the stamp's native size.
    """
    by: dict = {}
    for sp in ops.values():
        for prm in sp.params:
            if prm.edit_only and init:
                continue
            if prm.type in ("number", "int") and prm.unit:
                by.setdefault(prm.unit.split(",")[-1].strip(), set()).add(prm.name)
    parts = [
        f"{'/'.join(sorted(names))}: {unit}"
        for unit, names in sorted(by.items())
        if len(names) > 1 or "fraction" in unit or "multiplier" in unit
    ]
    # DROPPED when no listed op has a size, rather than emitted as a dangling "Sizes: ." — reachable
    # with `--allowed-cv-ops add_region`, whose only params are a layer name and a point.
    #
    # NO FRAME SENTENCE: which canvas the coordinates are measured against is a per-beam fact owned by
    # `Siting` and stated once, by `context.propose_context`.
    return "Sizes: " + "; ".join(parts) + "." if parts else ""


def ops_catalogue(allowed=None, layers=(), *, voice: str = "edit") -> str:
    """The op catalogue to show the model.

    Built from `ops.OPS` against the ops this run actually has, ONE ENTRY PER OP (a group is only a
    heading), so a restricted run is never shown an op it cannot use —
    no turn the model is actually SENT names a withheld op.

    ``allowed`` is None when the run restricts NOTHING, and a set — possibly EMPTY — when it restricts.
    An empty set must render as an empty catalogue: reading `()` as "everything" is a leak of the
    whole vocabulary
    (a restriction the layer withdrawal empties does not become no restriction).

    ``layers`` names the guidance layers actually available for THIS clean — one, under the rule that
    a layer op may only use the injected visual prompt's own map, so the layer-op entries can name it
    rather than guessing. Empty ``layers`` means the caller does not know (tests, the leak guard) and
    every possible name is listed; the beam always knows.
    """
    names = " | ".join(f'"{n}"' for n in (sorted(layers) or _ALL_LAYER_NAMES))
    keep = None if allowed is None else set(allowed)
    init = voice == "init"
    listed = {k: sp for k, sp in OPS.items() if keep is None or k in keep}
    # Turn 1 authors from nothing, so it needs the units up front; refine has them from the turn it
    # is refining and from `cv_api`. Rendered from the ops ACTUALLY listed, so a restricted run is
    # not told the units of a tool it does not have.
    out = ["  " + _units_line(listed, init=True)] if (init and listed) else []
    seen = None
    for group, op, desc in (
        (sp.group, sp.kind, (sp.init_prompt or sp.prompt) if init else sp.prompt) for sp in OPS.values()
    ):
        if keep is not None and op not in keep:
            continue
        if group != seen:
            out.append(f"  {group}:")
            # On the SECTION, not on whichever stroke op happens to be listed last. Refine only:
            # `_BRUSH_NORM` teaches `width_rel`, a fraction of THE MASK BEING EDITED, which is a
            # fraction of nothing on turn 1's blank canvas — the param declares `edit_only`.
            if group.startswith("FREEFORM") and not init:
                out.append("    " + _BRUSH_NORM)
            seen = group
        out.append(f"    {desc}")
    # The CROSS-CUTTING parameters, plus two facts about the ops that no per-op line can carry: the
    # sign conventions, and what rotate/scale/stretch pivot on. Taught here because this catalogue is
    # the only op documentation the model is ever shown.
    if not init:
        out.append("")
        for _name, _how in sorted(CROSS_CUTTING.items()):
            out.append(
                f"    {_name}:{_how.split(' — ')[0]} — {_how.split(' — ', 1)[1]}. "
                f"Optional on ANY op; omit it and the op edits the whole mask."
            )
        # NAMED OPS, so these follow the same restriction every catalogue line does. Written
        # unconditionally they advertised `rotate`/`scale`/`stretch` on a run that withheld them —
        # and `scale` is one of the names `leaked_ops` will not match bare, so the guard cannot be
        # what keeps this sentence honest.
        _sign = {"translate": "+dx = RIGHT, +dy = DOWN", "rotate": "+angle = COUNTER-CLOCKWISE"}
        _signs = [_sign[n] for n in ("translate", "rotate") if keep is None or n in keep]
        if _signs:
            out.append("    Signs: " + ", ".join(_signs) + ".")
        _pivot = [n for n in ("rotate", "scale", "stretch") if keep is None or n in keep]
        if _pivot:
            out.append(
                f"    {'/'.join(_pivot)} pivot on the mask's own CENTROID, so with "
                f"target:[x,y] each region turns about its own centre."
            )
    return "\n".join(out).replace("__LAYERS__", names)


def _distribution_expectation(kind: str, *, drawing: bool, can_stamp: bool = False) -> str:
    """What the type's ``distribution_kind`` card flag means, worded for whichever side is asking.

    ONE function on purpose: the two sides need different wording but must never disagree about how
    many regions the type has, and adding a third kind forces both branches to be written at once.

    ``can_stamp`` gates the one sentence here that names an op, and defaults to FALSE because a caller
    that forgets loses a hint where the other default would leak one. The leak guard cannot catch this
    one: the sentence writes the name in CAPITALS and ``leaked_ops`` matches case-sensitively, which
    is what keeps ordinary English like "CLOSE the gap" from reading as an offer of ``close{px}``.
    the multi region advice does not name an op the run withheld.
    """
    if kind == "multi":
        return (
            "THIS TYPE IS MULTI-REGION: one real occurrence USUALLY produces several separate "
            "regions. The reference card's count/distribution section gives the measured counts "
            "and spacing and the examples show the spread — match BOTH. Too few and too many are "
            "equally wrong, and so is spreading the regions wider apart than the examples do: a "
            "typical count clustered the way the examples cluster, not one blob and not a scatter "
            "across the whole image. "
            + (
                (
                    "Usually several does not mean always several — if the card records single-patch "
                    "cases, they are part of the distribution too. "
                    + (
                        "Easiest way to get count and spacing right at once is to STAMP a 'full' "
                        "example mask that already carries several patches; otherwise place several "
                        "shape ops on separate, comparable sites at the examples' spacing."
                        if can_stamp
                        else "Place several shape ops on separate, comparable sites at the examples' spacing."
                    )
                )
                if drawing
                else "So a single blob where the real ones run to several is WRONG on this dimension "
                "and scores LOW — but so is a candidate carrying more regions, or flung further "
                "apart, than the examples ever show. Score the MATCH to the examples, in both "
                "directions; do not simply reward more."
            )
        )
    if kind == "single":
        return "THIS TYPE MAKES ONE REGION per occurrence: " + (
            "draw a single connected region and do not fragment it into pieces."
            if drawing
            else "so this dimension scores exactly one thing — whether the candidate IS one "
            "region. ONE connected region: 10. If the block above reports SEPARATE regions, "
            "the candidate has fragmented and this dimension FAILS: 2 pieces scores 5 or "
            "less, 3 or more scores 3 or less, however good each piece looks on its own. "
            "This is not a formality — a defect that should be one mark and arrives as three "
            "is the wrong defect, and this is the only dimension that can say so."
        )
    return "Read from the examples whether this type is multi-region; single region in the examples " + (
        "-> one region is enough."
        if drawing
        else "-> score whether the candidate matches that: one connected region is 10, and a "
        "candidate that fragmented into several is marked down here, not excused."
    )


def _sites_expectation(sites_spanned: str, *, drawing: bool) -> str:
    """What the type's ``sites_spanned`` card flag means, worded for whichever side is asking.

    ONE function, on the same terms as :func:`_distribution_expectation`, and asked of EACH SEPARATE
    MARK rather than each occurrence: `_distribution_expectation` spends the word "occurrence" on the
    real-world EVENT, and per mark the two flags are answerable at once — which is what a type reading
    ``two_or_more`` here and ``multi`` there needs.
    how many structures one occurrence covers reaches BOTH prompts.
    """
    if sites_spanned == "two_or_more":
        return (
            "THIS TYPE IS A CONNECTION: one mark of it lies across TWO OR MORE separate "
            "structures, so a mark resting on a single structure is not this defect at all, "
            "however well positioned and formed. "
            + (
                "Every separate mark you draw must therefore reach onto BOTH (or all) of the "
                "structures it joins and cover the ground between them — read off the examples "
                "which structures an example mark joins and how far onto each one it reaches. "
                "This says nothing about HOW MANY marks belong in the mask, which is the "
                "DISTRIBUTION dimension's question: a mask may hold several, and each of them "
                "must lie across two or more structures."
                if drawing
                else "A region that touches only one structure fails BOTH this dimension and semantic, "
                "however tidy it is — it is not an instance of this type. The measured spans you "
                "were given are for the WHOLE mask, not per region: on a single-region candidate, "
                "a span of 1 means the connection was never made, and on a candidate holding "
                "several regions, read the zoom to check that EACH of them reaches two structures "
                "rather than taking the total as a pass. How many regions the mask should hold is "
                "the DISTRIBUTION dimension's question, not this one — several regions, each of "
                "them a connection, is correct here."
            )
        )
    if sites_spanned == "one":
        return (
            "One mark of this type sits on a SINGLE structure: it belongs to one of them and "
            "stays within it, and a mark reaching across onto a neighbouring structure has "
            "become a different defect. "
            + (
                "Where the examples show several marks they sit on several structures, one mark "
                "each, so keep every mark you draw inside the bounds of the one structure it sits "
                "on."
                if drawing
                else "So a region lying across two or more distinct structures has grown onto a "
                "neighbour it must not touch, and is marked DOWN here. The measured spans you "
                "were given are for the WHOLE mask, not per region: a candidate whose three "
                "regions sit on three structures overlaps three structures and that is RIGHT for "
                "this type. A span above one is wrong only when a SINGLE region is what reaches "
                "across — read the zoom to tell the two apart."
            )
        )
    return (
        "Read from the examples whether one mark of this defect lies across two or more "
        "structures or sits on a single one"
        + (
            " — it is what makes the defect the defect, so reproduce the relationship the example "
            "masks show, mark for mark."
            if drawing
            else " — and hold the candidate to what you see, region by region: a defect the examples "
            "show joining structures and the candidate joins none, or one the examples keep on a "
            "single structure and the candidate has grown onto its neighbour, both fail here."
        )
    )


class DefaultPrompts:
    """The production prompt-set (``default_v1``). Subclass and override a single method to
    experiment; the base returns the defaults."""

    name = "default_v1"

    def zoom_units(self, board, mask_px, view_w) -> str:
        return zoom_units(board, mask_px, view_w)

    def refine_frame(self, from_crop) -> str:
        return refine_frame(from_crop)

    def refine_ask(self, n, *, last, restart) -> str:
        return refine_ask(n, last=last, restart=restart)

    def weak_dim_steer(self, dims, weak, verb, feedback=None) -> str:
        return weak_dim_steer(dims, weak, verb, feedback)

    def shape_target(self, card) -> str:
        return shape_target(card)

    def proposer_system(self, allowed_ops=None, distribution_kind: str = "", layers=(), sites_spanned: str = "") -> str:
        """The proposer system prompt. ``allowed_ops`` selects which refine ops the catalogue lists:
        ``None`` restricts NOTHING, a set restricts to its members, and an EMPTY set offers no op at
        all (see :func:`ops_catalogue`). ``layers`` names the guidance layers this clean actually has.

        ``distribution_kind`` and ``sites_spanned`` are the SAME two card flags the judge is given,
        and each must be passed to BOTH or neither —
        how many structures one occurrence covers reaches BOTH prompts."""
        return assert_no_leak(
            AGENT_SYSTEM.replace("__RELPOS__", _RELPOS_OURS)
            # TWO faces of one registry: `__INIT_OPS__` is what turn 1 can draw from
            # nothing, `__OPS__` is everything refine can reach for — both INTERSECTED
            # with the run's restriction, so `--allowed-cv-ops` closes turn 1 as well.
            # the restriction reaches turn 1 too and says when it leaves nothing.
            .replace("__INIT_OPS__", ops_catalogue(_init_kinds(allowed_ops), layers, voice="init"))
            # Advice about a tool this run does not have is a leaked op with extra steps.
            .replace("__STAMP_ADVICE__", _STAMP_ADVICE if "stamp" in _init_kinds(allowed_ops) else "")
            # `clear` is an op like any other and a run can withdraw it, so the restart
            # bullet goes with it —
            # the redraw hatch is shut by the experimenter and by nobody else.
            .replace(
                "__RESTART_ADVICE__",
                _RESTART_ADVICE if "clear" in (allowed_ops if allowed_ops is not None else CV_KINDS) else "",
            )
            .replace("__OPS__", ops_catalogue(allowed_ops, layers))
            .replace(
                "__DISTRIBUTION_EXPECTATION__",
                _distribution_expectation(
                    distribution_kind, drawing=True, can_stamp="stamp" in _init_kinds(allowed_ops)
                ),
            )
            .replace("__SITES_EXPECTATION__", _sites_expectation(sites_spanned, drawing=True)),
            # `allowed_ops` ALONE: widening it to a union with whatever the beam happens to have
            # makes the guard vacuous.
            # turn ones instruction is checked against the same vocabulary as the catalogue.
            None if allowed_ops is None else tuple(allowed_ops),
            "proposer_system",
        )

    def judge_system(self, allowed_ops=None, distribution_kind: str = "", sites_spanned: str = "") -> str:
        """The judge system prompt — the ONE shared rubric used identically by propose / generate /
        judge. It names no ops: the judge scores and explains, the proposer decides how to fix.

        ``allowed_ops`` is accepted because the :class:`PromptSet` protocol passes it and a subclass
        may render an op-aware judge; it is NOT what this text is checked against
        (the judge guard is the empty allowed set not the runs).

        ``distribution_kind`` and ``sites_spanned`` come from the type's distilled card, which made
        both calls from up to a hundred real masks: left to decide for itself, the judge always
        decides single and scores the dimension 10. They say what the type REQUIRES, so the MEASURED
        spans the judge is handed have something to be checked against.
        how many structures one occurrence covers reaches BOTH prompts."""
        del allowed_ops
        return assert_names_no_op(
            JUDGE_SYSTEM.replace(
                "__DISTRIBUTION_EXPECTATION__", _distribution_expectation(distribution_kind, drawing=False)
            ).replace("__SITES_EXPECTATION__", _sites_expectation(sites_spanned, drawing=False)),
            "judge_system",
        )


_REGISTRY: dict[str, type] = {"default_v1": DefaultPrompts}


def register(name: str):
    """Decorator: register an alternative :class:`PromptSet` class under ``name`` (for A/B / experiments).

    Usage::

        @register("myv2")
        class MyPrompts(DefaultPrompts):
            def proposer_system(self) -> str: ...

    then select it with ``cfg.prompt_set="myv2"``.
    """

    def _decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return _decorator


def by_name(name: str) -> PromptSet:
    """Construct a :class:`PromptSet` by registry ``name``; an unknown name falls back to
    :class:`DefaultPrompts`. This is what lets ``cfg.prompt_set`` switch prompt-sets by string."""
    return _REGISTRY.get(name, DefaultPrompts)()


# Plain-language restatement of the edit (CV) operations, appended to the proposer prompt. The default
# op list names each op in CV terms ("GrabCut-snap", "morph gradient", "approxPolyDP"), which a model
# only follows if it already knows those algorithms; here each op is indexed by the PROBLEM it fixes
# instead. Placement comes first because a mask on the wrong structure is unusable however well it is
# drawn, so that is the correction to reach for before size or shape.
_PLAIN_CV_GUIDE = """
------------------------------------------------------------------------
PLAIN GUIDE TO THE EDIT OPERATIONS — choose the op by the PROBLEM you see.
The mask MUST sit exactly on the same kind of structure the EXAMPLE masks sit on: look at what is
directly UNDER each example mask, then place yours on the same kind of spot in the clean image. The
most common mistake is the mask being in the wrong PLACE, so fix placement first.

1) WRONG PLACE (mask is on bare background, beside the target, or hanging half off it):
   - translate {dx,dy}: slide the whole mask. +dx moves it RIGHT, +dy moves it DOWN; the values are
     fractions of the image (0.1 = one tenth of the width). Read the gap in the zoom and move by that
     much. This is your main placement fix.
   - snap: delete every mask pixel that is NOT on the bright/real feature and keep only the pixels
     actually sitting on the structure. Use when the mask spills over onto empty background.
   - conform: shrink-wrap the mask onto the real shape in the photo beneath it (like a vacuum bag
     around the object). Use when the mask is a loose blob covering more than the real feature.

2) WRONG SIZE:
   - scale {factor}: bigger (factor>1) or smaller (factor<1); shape unchanged.
   - dilate {px}: grow the outline outward. erode {px}: shrink it inward, by px PIXELS. Erode to
     pull an oversized mask off its neighbours; dilate to thicken a too-thin one.

3) WRONG SHAPE or ANGLE:
   - rotate {angle}: spin it (degrees, + is counter-clockwise).
   - stretch {sx,sy}: lengthen or thin along one axis (sy=2 doubles the height).
   - smooth {px}: round off a jagged edge; here px is the WIDTH of the blur kernel in pixels and
     NOT a radius, so it reaches half as far as the same number does on a grow/shrink op (frac is
     that width as a fraction of the short side). convex: fill in dents to make a clean blob.

4) STRAY SPECKS or HOLES:
   - keep_largest: delete little stray pieces and keep the main one.
   - fill_holes: fill enclosed gaps inside the mask. close {px}: close small gaps+holes. open {px}:
     remove small specks.

WORKFLOW when a mask is off: FIRST translate it onto the correct structure, THEN snap/conform to hug
the real shape, THEN fix size/shape. Never accept a mask that is not sitting on the correct structure
- a mask on the wrong spot is wrong no matter how good its shape is.
------------------------------------------------------------------------
"""


@register("plain_cv")
class PlainCVPrompts(DefaultPrompts):
    """default_v1 + a plain-language CV-operations guide appended to the proposer prompt. For models
    that need the ops described by the problem each one fixes rather than by their CV names."""

    name = "plain_cv"

    def proposer_system(self, allowed_ops=None, distribution_kind: str = "", layers=(), sites_spanned: str = "") -> str:
        # THROUGH THE GUARD, again: `super()` checked text that does not yet contain the guide.
        # a registered prompt set cannot append an offer the guard does not read.
        return assert_no_leak(
            super().proposer_system(allowed_ops, distribution_kind, layers, sites_spanned)
            + "\n"
            + _plain_cv_guide(None if allowed_ops is None else tuple(allowed_ops)),
            None if allowed_ops is None else tuple(allowed_ops),
            "plain_cv.proposer_system",
        )
