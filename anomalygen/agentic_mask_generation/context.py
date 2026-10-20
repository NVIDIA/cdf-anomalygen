# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one beam SHOWS the model — derived once, from that beam's own inputs.

Every value is computed once in `BeamContext.build`, from that beam's own inputs, then read unchanged
for the rest of it. The system prompts, the op vocabulary's rendering and the site wording are not
here — `prompts.py`, `ops.py` and `regions.py` own those; this is the assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosmos_framework.utils import log
from PIL import Image

from anomalygen.agentic_mask_generation.card_distill import exemplar_note, read_flags
from anomalygen.agentic_mask_generation.imaging import (
    Views,
    describe_geometry,
    describe_shapes,
    mask_geometry,
)
from anomalygen.agentic_mask_generation.ops import CV_KINDS, OPS, extract_shapes, unavailable
from anomalygen.agentic_mask_generation.prompts import assert_no_leak, by_name
from anomalygen.agentic_mask_generation.regions import (
    Siting,
    coverage_blocks,
    ex_dist_summary,
    site_menu,
)
from anomalygen.agentic_mask_generation.visual_prompts import EXEMPLAR_VP_CAPTION

# Framing for the distilled card in the PROPOSE prompt; the judge's counterpart is judging._DEFECT_DOCS_JUDGE_CAPTION.
_CARD_CAPTION = (
    "AUTHORITATIVE DEFECT KNOWLEDGE (curated from inspection references — use it to inform your "
    "understanding and to write an accurate definition; weigh it together with the actual images):"
)


def seed_instruction() -> str:
    """The warm start's opening line — what to do with masks it did not draw; REPLACES `propose_ask`."""
    return (
        "^ Above: the real defect examples, the clean board, and any visual prompt. "
        "Below are your CURRENT candidate mask(s) — each is a KNOWN-GOOD mask from a "
        "prior ACCEPTED run on this board. "
        "Do NOT start over: ASSESS each against the 6-dimension rubric and make only "
        "SMALL, valid refinements or nearby variations (keep what already works, fix "
        "only real weaknesses)."
    )


# `eq=False`: the fields are PIL images and dicts, so a generated `__eq__` compares boards pixel by
# pixel and the `__hash__` frozen+eq brings raises on the first dict.
@dataclass(frozen=True, eq=False)
class BeamContext:
    """One beam's constants, and the opening turns built from them."""

    defect_type: str
    clean: Any  # the clean board, RGB
    exemplars: tuple
    prompts: Any  # the PromptSet in force for this run
    num_candidates: int
    target_region: Any  # what the caller ASSIGNED, which is not the same as what could be cropped

    allowed: frozenset  # the op vocabulary — turn 1 AND refine — AFTER the withdrawal in `build`
    layers: Any  # the INJECTED visual prompt's own structure maps, keyed by name
    rigid_only: bool  # fixed-shape mode: refine may only reposition stamps

    card: Any
    flags: dict

    siting: Siting
    shapes: list
    clean_grid: Any

    vp_blocks: Any  # proposer visual prompt (str | PIL blocks)
    judge_vp_blocks: Any
    judge_docs: Any
    cov_blocks: Any
    # The delivery cap as an OBJECT, not a number, so an image is built at the size it is sent at —
    # every entry point that builds a view declares the cap with no default, and the cap that is
    # passed is the size the images come out at.
    views: Any
    cov_localize: bool
    cov_propose: bool

    proposer_system: str
    judge_system: str
    shape_line: str  # the card's shape sentence, restated on every refine turn
    # Whether the measured STRUCTURE-COVERAGE note is injected at judge/refine/pairwise
    # (cfg.coverage_note). Defaulted OFF: a caller that does not ask for the note gets the base context.
    coverage_note: bool = False

    @property
    def W(self) -> int:
        return self.clean.size[0]

    @property
    def H(self) -> int:
        return self.clean.size[1]

    @property
    def multi(self) -> bool:
        """Does one occurrence of this type usually produce several separate regions?"""
        return self.flags.get("distribution_kind", "") == "multi"

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        clean_path,
        defect_type,
        *,
        visual_prompt,
        defects,
        cfg,
        exemplars,
        prompts=None,
        covered=None,
        target_region=None,
        regions=None,
    ) -> "BeamContext":
        """Read the card, site the defect, build the views, and settle the vocabulary — in that order.

        The order is load-bearing: the layer withdrawal must precede `proposer_system`, which is
        rendered from what it leaves.
        """
        prompts = prompts or by_name(cfg.prompt_set)
        clean = Image.open(clean_path).convert("RGB")
        W, H = clean.size
        # ALWAYS CONCRETE: `_allowed` is the EFFECTIVE set of ops, so empty means empty and not
        # "nothing was restricted" — a restriction the layer withdrawal empties does not become no restriction.
        _asked = tuple(cfg.allowed_cv_ops or ())
        _allowed = set(_asked) if _asked else set(CV_KINDS)
        # The redraw hatch needs no flag: "start over" is an `ops` list opening with `clear`, so
        # withholding `clear` closes it — the redraw hatch is shut by the experimenter and by nobody else.
        card = defects.card(defect_type)
        flags = read_flags(card or "")
        # fill_host=whole: one mark covers ALL of its host structure(s), so the mask IS whole host
        # instance(s) and pasting an example shape can only collide with this board's own structures
        # — `stamp` is withdrawn, composing with (never widening) the run's own restriction. The
        # driver forces flexible shape mode on the same rule under `--shape auto`; this covers the
        # flexible path, where `stamp` is otherwise still on offer. NEVER under `--shape fixed`:
        # the CLI outranks the card ban there, and a fixed-shape beam's only vocabulary is stamps,
        # so withdrawing the op would leave every candidate empty.
        if cfg.shape_policy != "fixed" and flags.get("fill_host") == "whole":
            _allowed.discard("stamp")
        # WHERE this beam may place the defect, and in whose [0,1] frame the proposer authors.
        views = Views(cfg.image_max_dim or 0)
        siting = Siting.build(clean, target_region, exemplars, want_hard=True, views=views)
        # COVERAGE feedback: steer localize/propose away from sites already placed on THIS clean.
        # `coverage_feedback` carries TWO decisions in one string — the driver reads it to decide
        # whether to ENUMERATE regions and whether to assign one; this reads it to decide which TURN
        # hears about coverage.
        _cov_fb = cfg.coverage_feedback
        cov_localize = _cov_fb in ("localize", "both")
        # `choose` goes to PROPOSE, not localize: it decides WHERE the defect sits, and under `choose`
        # there is no engine assignment for it to contradict.
        cov_propose = _cov_fb in ("propose", "both", "choose")
        cov_blocks = coverage_blocks(clean, covered, cfg.coverage_mode, views) if (cov_localize or cov_propose) else []
        if _cov_fb == "choose":  # the agent picks the site; the menu replaces the engine's assignment
            cov_blocks = site_menu(regions, covered, W, H) + cov_blocks

        shapes = extract_shapes(exemplars)  # stamp library: real shapes from the example masks
        # ONE grid object for the whole beam, so _img_block's per-object cache encodes it once.
        clean_grid = views.grid(clean)

        # judge_docs / judge_vp_blocks are passed into judge_consensus as defect_docs / extra_blocks.
        judge_docs = defects.docs(defect_type)
        # The card's flags reach BOTH prompts from this one read, and one side alone makes a
        # connection type unsolvable — how many structures one occurrence covers reaches BOTH prompts.
        judge_system = prompts.judge_system(
            _allowed, flags.get("distribution_kind", ""), sites_spanned=flags.get("sites_spanned", "")
        )
        judge_vp_blocks = visual_prompt.build_for_judge(clean_path, defect_type)
        vp_blocks = visual_prompt.build(clean_path, defect_type)  # proposer visual prompt (str | PIL blocks)

        # Asked of the INJECTED prompt itself. Reading layer artifacts off disk instead would hand
        # the engine whatever an earlier run left in the shared, content-keyed artifact cache.
        layers = visual_prompt.layers(clean_path)

        # An op whose REQUIREMENT this beam cannot meet is withdrawn, from the registry's own
        # declarations rather than a hand-kept list.
        _have = {"clean"} | ({"layer"} if layers else set()) | ({"shapes"} if shapes else set())
        _gone = unavailable(_have)
        _hidden = {o for o in _gone if o in _allowed}
        if _hidden:
            # ONE line per beam, deliberately NOT `op_withdrawn:<kind>` — that label means the model
            # asked for an op the RUN does not offer, a different event. See
            # ops a board cannot carry are booked as withheld from that beam.
            log.warning(
                f"degraded[ops_unavailable]: [{defect_type}] {Path(clean_path).name} has no "
                f"{'/'.join(sorted(_have ^ {'clean', 'layer', 'shapes'}))} for this beam — "
                f"withholding {', '.join(sorted(_hidden))}"
            )
        _allowed = set(_allowed) - _gone
        if not layers:
            if not _allowed:
                # `--allowed-cv-ops` that is entirely layer ops, on a layerless board.
                log.warning(
                    f"[{defect_type}] every op this run allows needs a guidance layer and "
                    f"{Path(clean_path).name} has none — this beam can apply no CV op"
                )

        # `--allowed-cv-ops dilate,erode` leaves turn 1 no op that can draw from a blank canvas: a real
        # configuration, REPORTED rather than rendered as an empty catalogue under a sentence saying the
        # mask is drawn by applying ops to one — the restriction reaches turn 1 too and says when it leaves nothing.
        if not {k for k in _allowed if "mask" not in OPS[k].requires}:
            log.warning(
                f"degraded[no_init_op]: [{defect_type}] {Path(clean_path).name} — no allowed "
                f"op can draw from a blank canvas ({', '.join(sorted(_allowed)) or 'none'}), "
                f"so turn 1 has no vocabulary and every candidate will be empty"
            )
        proposer_system = prompts.proposer_system(
            _allowed,
            flags.get("distribution_kind", ""),
            tuple(layers),  # a layer op is only ever offered under a name this clean has
            sites_spanned=flags.get("sites_spanned", ""),
        )
        return cls(
            defect_type=defect_type,
            clean=clean,
            exemplars=tuple(exemplars),
            prompts=prompts,
            num_candidates=cfg.num_candidates,
            target_region=target_region,
            allowed=frozenset(_allowed),
            layers=layers,
            # FIXED-SHAPE: candidates must be STAMPS of real training-mask shapes, and refine may
            # only reposition/orient them (RIGID_CV_KINDS).
            rigid_only=cfg.shape_mode == "fixed",
            card=card,
            flags=flags,
            siting=siting,
            shapes=shapes,
            clean_grid=clean_grid,
            vp_blocks=vp_blocks,
            judge_vp_blocks=judge_vp_blocks,
            judge_docs=judge_docs,
            cov_blocks=cov_blocks,
            views=views,
            cov_localize=cov_localize,
            cov_propose=cov_propose,
            proposer_system=proposer_system,
            judge_system=judge_system,
            shape_line=(
                prompts.shape_target(card)  # the card itself is turn 1 only
                if cfg.restate_shape_target
                else ""
            ),
            coverage_note=cfg.coverage_note,
        )

    # --------------------------------------- the turns ----------------------------------------------
    def _example_blocks(self, with_geometry: bool, theirs: list | None = None) -> list:
        """The (defect, mask, zoomed defect) triple per example; propose adds each mask's geometry.

        ``theirs`` collects the blocks this engine did not write — here the distill-time caption,
        which is the MODEL's own reading of a real example — for the caller's leak guard to exempt
        (see ``propose_context``). The rendered block is exempted whole, engine framing included:
        one caption must not be able to kill the beam."""
        blocks: list[Any] = []
        for i, ex in enumerate(self.exemplars, 1):
            cap = f"EXAMPLE {i} mask:"
            if with_geometry:
                cap = f"EXAMPLE {i} mask (geometry: {describe_geometry(mask_geometry(ex.mask))}):"
            blocks += [f"EXAMPLE {i} defect:", ex.defect, cap, ex.mask]
            if getattr(ex, "observation", ""):
                _note = exemplar_note(i, ex.observation)
                blocks.append(_note)
                if theirs is not None:
                    theirs.append(_note)
            if getattr(ex, "annotated", None) is not None:  # exemplar visual parity
                blocks += [f"EXAMPLE {i} {EXEMPLAR_VP_CAPTION}", ex.annotated]
            ez, _ = self.views.zoom_around_mask(ex.defect, ex.mask)  # magnify the real defect region
            if ez is not None:
                blocks += [
                    f"EXAMPLE {i} defect ZOOM (real defect region magnified — read its fine structure/appearance):",
                    ez,
                ]
        return blocks

    def localize_blocks(self) -> list:
        """Turn 0: find the WORK REGION, so propose gets a MAGNIFIED view of where the defect goes.

        The global view is kept beside the crop — a crop alone lost the relational context bridging
        needs. Built on demand: a beam ASSIGNED a region never asks.
        """
        blocks = self._example_blocks(with_geometry=False)
        blocks += [
            f"Defect type: {self.defect_type}",
            "CLEAN image (coordinate grid):",
            self.clean_grid,
        ]
        if self.vp_blocks:  # ground localization in the visual prompt too (same full+grid map the proposer sees)
            blocks += list(self.vp_blocks)
        if self.cov_localize and self.cov_blocks:  # steer the work region to an uncovered valid site
            blocks += list(self.cov_blocks)
        blocks.append(
            "From the examples (and the visual prompt above, if any), identify in THIS clean image the WORK "
            "REGION: the tightest bounding box (normalized 0-1 coords) that contains the structure(s) this "
            "defect sits on, plus their immediate surroundings (so relationships between adjacent structures "
            'stay visible). Output JSON {"work_region":{"x0":0.0,"y0":0.0,"x1":1.0,"y1":1.0},"structures":""}.'
        )
        return blocks

    def propose_context(self, work_zoom) -> list:
        """Everything the propose turn SHOWS: examples, clean board, visual prompt, card, crop(s).

        Returned apart from the instruction below it because the warm start replays exactly this into
        the FIRST refine turn, where no propose call happens.

        This is the guard's OTHER half of the propose turn (``propose_ask`` checks the instruction):
        the engine-authored framing here is policed against the run's vocabulary, and the blocks this
        engine did not write — the card, the exemplar captions, the visual prompt's own words, the
        defect type's name — are exempted BY IDENTITY, ``theirs``, on the refine turn's rule: the
        guard does not police words it did not write, and one card or caption must not be able to
        kill the beam (exemption ≠ deletion — the exempt text is still sent).
        """
        # THE SAME PREDICATE `run_beam` USES to remap a candidate's coordinates: a beam that wanted
        # hard siting and got no crop must not be told its canvas is a crop.
        hard_region = bool(self.siting.hard and self.siting.crops)
        theirs: list = []
        blocks = self._example_blocks(with_geometry=True, theirs=theirs)
        clean_cap = (
            "CLEAN input image — GLOBAL view for CONTEXT ONLY. Do NOT read coordinates from it; your "
            "coordinates are relative to the magnified crop shown below:"
            if hard_region
            else "CLEAN input image (coordinate grid; place masks here) — GLOBAL view, keeps the full "
            "layout and relationships between structures visible:"
        )
        # THE FRAME, stated ONCE and by the party that decides it (`Siting`, via `hard_region`): no
        # other block, the op catalogue included, may name a frame — a model cannot follow two
        # answers to "relative to what", and which it picks decides where its mask lands.
        blocks.append(
            "COORDINATE FRAME — read this before any number below. Your canvas is the ASSIGNED CROP "
            "shown in this turn: every coordinate AND every size is relative to that crop, (0,0) its "
            "top-left and (1,1) its bottom-right. It is not the whole board, and you cannot address "
            "anything outside it."
            if hard_region
            else "COORDINATE FRAME — read this before any number below. Your canvas is the FULL image: "
            "every coordinate and every size is relative to it, (0,0) its top-left and (1,1) its "
            "bottom-right."
        )
        _type_line = f"Defect type: {self.defect_type}"  # a real defect type is named `open`
        theirs.append(_type_line)
        blocks += [_type_line, clean_cap, self.clean_grid]
        if self.card:  # distilled defect knowledge -> seed the definition
            _card_block = _CARD_CAPTION + "\n" + self.card
            theirs.append(_card_block)
            blocks += [_card_block]
        if self.vp_blocks:  # same visual prompt the localize call saw (built once above)
            # The injected prompt's own words, on the judge guard's rule for `extra_blocks`.
            theirs += [b for b in self.vp_blocks if isinstance(b, str)]
            blocks += list(self.vp_blocks)
        if work_zoom is not None:
            wz_cap = (
                "MAGNIFIED view of the ASSIGNED REGION — THIS crop is your ENTIRE canvas. Author EVERY "
                "primitive coordinate in THIS crop's [0,1] frame: (0,0)=its top-left, (1,1)=its "
                "bottom-right, per the crop grid. Place the defect on the valid structure you see here; "
                "you cannot address anything outside this crop:"
                if hard_region
                else "MAGNIFIED view of the work region (same image, zoomed; grid labelled in FULL-IMAGE "
                "normalized coords). READ the fine structure here — individual structures, gaps, and "
                "exact boundaries — to place each candidate PRECISELY on the right feature (the global "
                "view above is for context, this is for precision):"
            )
            if hard_region and len(self.siting.crops) > 1:
                # The assignment is WRITTEN OUT: there are more candidates than crops, so "candidate k's
                # canvas is crop k" names no canvas for most of them. From `Siting`, which also maps back.
                blocks += [
                    wz_cap.split(":")[0] + " — ONE CANVAS PER CANDIDATE, ASSIGNED ROUND-ROBIN over "
                    f"the {len(self.siting.crops)} crops below: "
                    f"{self.siting.assignment_text(self.num_candidates)}. Author each "
                    "candidate's primitive coordinates in ITS crop's [0,1] frame. The crops are "
                    "different sites on the same board, so the same numbers mean different places in "
                    "different crops:"
                ]
                for _pos, (_i, _z, _b) in enumerate(self.siting.crops):
                    blocks += [f"{self.siting.canvas_label(_pos)}:", _z]
            else:
                blocks += [wz_cap, work_zoom]
        if (
            self.target_region is not None and not hard_region
        ):  # SOFT targeting — a text instruction (the model can drift)
            # No multi-region wording here on purpose: a multi type's crop is widened to hold a whole
            # example, so it takes the HARD branch below, where that instruction lives.
            blocks.append(
                "TARGET REGION (assigned): the magnified work-region view shows the SINGLE valid site "
                "this defect is assigned to. EVERY candidate MUST be placed on THAT region's structure "
                "— never on another site or structure. This guarantees the dataset spreads across all "
                "sites; a valid placement on the assigned region is the ONLY goal."
            )
        elif hard_region:  # HARD targeting — the crop IS the region; coords are crop-relative so drift is impossible
            blocks.append(
                "HARD-CONSTRAINED PLACEMENT: each candidate's canvas IS the crop it was assigned above, "
                "and its coordinates are that crop's [0,1] frame, so every candidate lands on its assigned "
                "site by construction — you do not choose sites. Candidates sharing a crop must still "
                "DIFFER: place them on different structures within it, or at different extents. Make the "
                "best placement of this defect on the valid structure you see in each crop."
                if len(self.siting.crops) > 1
                else "HARD-CONSTRAINED PLACEMENT: your canvas IS the assigned region's crop above. All "
                "coordinates are crop-relative [0,1], so every candidate lands in THIS region by "
                "construction — you don't need to (and can't) choose a site. Just make the best "
                "placement of this defect on the valid structure you see in the crop."
                + (
                    " This crop was widened to hold a WHOLE example of this defect, so it takes in several "
                    "comparable sites: use them. The occurrences go on the sites inside this crop, at the "
                    "examples' spacing — the crop's size is the scale reference for how big each one should "
                    "be."
                    if self.multi
                    else ""
                )
            )
        if (
            self.cov_propose and self.cov_blocks
        ):  # steer placement to an uncovered valid site (relocation is a proposer job)
            blocks += list(self.cov_blocks)
        # The refine turn's guard, on this half of the propose turn: engine text only, `theirs`
        # exempt by id() — everything the engine wrote here is code-determined, so a trip is a real
        # engine bug, while a card word must never be able to fail the type.
        _theirs = {id(b) for b in theirs}
        assert_no_leak(
            "\n".join(b for b in blocks if isinstance(b, str) and id(b) not in _theirs),
            tuple(self.allowed),
            "propose_context",
        )
        return blocks

    def propose_ask(self) -> list:
        """Turn 1's own instruction: the stamp menu, the measured example distribution, and the ask.

        Sent only on the propose call: the warm start replays `propose_context` and asks for a
        refinement instead, so a seed is never told to start over.
        """
        K = self.num_candidates
        # `stamp` is withdrawn when this beam's exemplars yield no shapes, and the whole turn-1 call
        # must agree — turn ones instruction is checked against the same vocabulary as the catalogue.
        _stamps = "stamp" in self.allowed and bool(self.shapes)
        # "sized and oriented", not "scale/rotate": those are the STAMP's own fields, but the leak
        # guard reads op NAMES, so the op spelling offered `scale`/`rotate` to a run that has neither.
        _prefer = (
            "PREFER STAMPS of the listed example shapes (place on the right structure, sized and "
            "oriented to fit) over geometric approximations — real shapes give clearer margin on "
            "shape/fill/semantic, and a 'full' stamp reproduces a multi-instance pattern's real "
            "count+clustering at once. "
            if _stamps
            else "This run has no usable example shapes, so build each mask from the drawing ops in your "
            "catalogue and match the example masks' outline and distribution by eye. "
        )
        blocks = [
            *(
                [
                    f"Available STAMP shapes (real shapes from the example masks; use kind='stamp' "
                    f"shape_id to paste one when the defect shape is irregular): "
                    f"{describe_shapes(self.shapes)}"
                ]
                if _stamps
                else []
            ),
            ex_dist_summary(self.exemplars),
            f"FIRST understand what the structures in THIS image ARE and the defect's physical "
            "CAUSE/MECHANISM (from the examples + reading the image), and write the SHARED 6-dimension "
            "RUBRIC (per-dimension target + reference) — it is reused verbatim by self-assess and the "
            f"independent judge. THEN propose {K} DIVERSE candidate masks (genuinely DIFFERENT valid "
            "placements/shapes — explore the space, NOT near-duplicates), each consistent with the rubric. "
            + _prefer
            + 'Output JSON {"definition":{'
            '"image_understanding":"","mechanism":"",'
            '"roi":{"target":"","reference":"examples→this clean image"},'
            '"placement":{"target":"","reference":"examples relative placement + this clean image local features"},'
            '"shape":{"target":"","reference":"example masks + real outline"},'
            '"fill":{"target":"","reference":"actual feature in clean image"},'
            '"distribution":{"target":"","reference":"example masks distribution"},'
            '"semantic":{"target":"","reference":"causal understanding + clean image"}},'
            f'"candidates":[{{"plan":"detailed: (a) which structure(s) + grid coords, (b) precise '
            "placement aligned to real feature boundaries (no spill into gaps/background), (c) shape "
            'choice + why it matches the example masks, (d) count/spacing if multi-instance",'
            f'"ops":[...],"rationale":""}}]}} '
            f"with EXACTLY {K} candidates. Use the MAGNIFIED view to make each plan concrete.",
        ]
        if self.rigid_only and not _stamps:
            # FIXED-SHAPE with no shape library is unsatisfiable — stamps only, nothing to stamp —
            # so every candidate comes back empty for the whole run.
            log.warning(
                f"degraded[fixed_shape_no_library]: [{self.defect_type}] this run is FIXED-SHAPE "
                f"and has no usable example shapes — every candidate will be empty"
            )
        # ON THE SAME CONDITION as the stamp menu above: this paragraph is nothing but stamp
        # instructions, so on a beam with no stamp it is a leaked op start to finish. It names no
        # ALTERNATIVE either — listing the ops not to use advertised whichever ones were withdrawn.
        if self.rigid_only and _stamps:
            blocks.append(
                "FIXED-SHAPE DEFECT: the shape is rigid and must be preserved EXACTLY. Every candidate's "
                '"ops" MUST be STAMP(s) — {"kind":"stamp","shape_id":<id>,"center":[x,y],'
                '"angle":deg} reusing a listed real example shape. Do NOT draw or synthesize a shape '
                "of your own. Vary candidates by WHICH shape, WHERE (center) and orientation "
                "(angle) only — never by deforming the outline."
            )
        # The guard on the TURN, not just on the system prompt it rides with — this is the
        # instruction half; `propose_context` guards the other half, where the foreign (card /
        # caption / visual-prompt) blocks live and are exempted. Every block HERE is engine-authored,
        # so nothing is exempt.
        assert_no_leak("\n".join(b for b in blocks if isinstance(b, str)), tuple(self.allowed), "propose_ask")
        return blocks
