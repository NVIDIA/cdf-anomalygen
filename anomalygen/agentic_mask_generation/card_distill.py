# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-native defect cards: distill a TEXT-ONLY reference card from a defect's own real examples.

The VLM captions each real ``(defect image, mask)`` of one ``(texture, defect)`` — the annotated
region outlined RED — into a structured observation, then one synthesis pass turns those into a
markdown card. Same imaging modality as deployment, so appearance language is aligned by
construction. Transport and exemplars are injected; the caller decides where to persist the card
(see :func:`build_cards_dir`).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from cosmos_framework.utils import log

from anomalygen.agentic_mask_generation.exemplars import count_pairs, load_exemplars, patch_histogram
from anomalygen.agentic_mask_generation.imaging import (
    Views,
    outline_mask,
)
from anomalygen.agentic_mask_generation.utils import atomic_write, slug
from anomalygen.agentic_mask_generation.vlm import (
    MAX_TOKENS_CAPTION,
    MAX_TOKENS_CARD,
    set_trace_ctx,
    watch_reply_repair,
)

# --- captioning (per-image) ---
CAPTION_SYS = (
    "You are a senior visual-inspection engineer writing precise observations of real "
    "defect photos. The defect's annotated region is outlined RED. Describe ONLY what "
    "is visible; never invent."
)
CAPTION_ASK = (
    "Defect type: %s. Describe this real defect precisely as JSON:\n"
    '{"host": "<the ONE structure the red region sits on, as a short noun phrase — this is what the '
    "extent and boundary answers below are measured against. Name the SMALLEST structure that still "
    "contains it: 'the solder fillet on one lead', 'one chip-resistor end termination', 'the "
    "ground band on the part face'. If there is no discrete structure under it, say 'none "
    "(continuous surface)'>\",\n"
    ' "location": "<which part of that host it sits on, and what is around it>",\n'
    ' "appearance": "<colors/brightness/texture AS RENDERED IN THIS IMAGING>",\n'
    ' "shape_size": "<shape and size relative to the host structure>",\n'
    ' "extent": "<EXACTLY one of: part_of_one | all_of_one | across_two | no_host — measured '
    "against THE HOST YOU NAMED. part_of_one = the red region covers only PART of that host. "
    "all_of_one = it covers essentially the WHOLE host. across_two = it spans or joins "
    "TWO OR MORE separate structures, bridging the gap between them. no_host = there is no "
    'discrete structure under it, just a continuous surface it could sit anywhere on>",\n'
    ' "boundary": "<EXACTLY one of: outer | inner | interior | any. Where the red region sits '
    "relative to the OUTLINE OF THE HOST YOU NAMED — the same structure, not a bigger or smaller one. "
    "outer = it grows OUTWARD past the outline. "
    "inner = it touches the outline and eats INWARD, its body inside the structure. interior = it "
    "lies wholly inside and does NOT touch the outline. any = it does neither consistently. This is "
    "the structure's own inside and outside; no parent or assembly direction is involved>\",\n"
    ' "root_cause": "<one-line manufacturing root-cause hypothesis consistent with the '
    'location and appearance>"}'
)
# No patch COUNT is asked for here: it is measured off the masks (exemplars.patch_histogram). A model
# asked to count answers a different question — defect EVENTS in a photo, not patches drawn.
_CAPTION_KEYS = ("host", "location", "appearance", "shape_size", "extent", "boundary", "root_cause")

# --- synthesis (one pass over all observations) ---
SYNTH_SYS = (
    "You compile inspection knowledge. Given many per-image observations of ONE defect "
    "type (all from the same imaging modality), synthesize a concise, accurate reference "
    "card grounded ONLY in these observations. Report PATTERNS with rough frequencies "
    "('most', 'about half', 'rare'); never crown a single feature as the only criterion "
    "when observations vary."
)
# Type-level geometry, decided once by the synthesis pass, which sees every observation at once.
# FREE_TEXT marks a flag that can only be required to be present and short — but three of the others
# are measured against it, so leaving it implicit is worse than not being able to check it.
FREE_TEXT = "<free text>"

FLAGS = {
    "host_structure": (
        FREE_TEXT,
        "the ONE structure this defect sits on, as a short noun phrase — the reference "
        "that fill_host, boundary_contact and sites_spanned are all measured against. "
        "Name the SMALLEST structure that still contains the defect ('the solder fillet "
        "on one lead', 'one chip-resistor end termination', 'the ground band on the part "
        "face'), or 'none (continuous surface)' when there is no discrete host. Take it "
        "from the per-image 'host' fields; if those name different things say so in "
        "'why' rather than picking one silently — a type whose own examples disagree "
        "about their host is one that needs a human, and the three flags below cannot "
        "mean anything until it is settled",
    ),
    "hosts_presented": (
        "multi_host",
        "single_host",
        "free",
        "how the scene presents this defect's host_structure — asked as: if you drew the "
        "legal area by hand, would it come out as SEVERAL separate pieces, ONE piece, or "
        "the whole image? multi_host = repeated host structures the defect can occur at "
        "(pads, leads, pins) · single_host = one bounded host with no separate sites "
        "inside it (a screen's glass, a vial's body) · free = no meaningful constraint",
    ),
    "fill_host": (
        "part",
        "whole",
        "n_a",
        "how much of its ONE host_structure a single mark of this defect covers. "
        "part = it covers PART of one (a scratch on a screen's glass, a starved fillet on "
        "one lead), so both overshoot and under-fill are wrong · whole = it covers ALL of "
        "one (a missing component, masked as the part's entire footprint; a stain over a "
        "whole face), so under-fill is the failure and 'too big' is not a thing · n_a = "
        "ONLY when there is no discrete host at all (host_structure 'none (continuous "
        "surface)') — any named host, repeated or not, takes part or whole",
    ),
    "boundary_contact": (
        "outer",
        "inner",
        "interior",
        "any",
        "where the defect sits relative to the OUTLINE OF host_structure — that same "
        "structure, not a larger or smaller one, because the answer flips with the "
        "choice. Read it straight off the annotated region in the examples. "
        "outer = it grows OUTWARD past the outline (excess solder, a spike, a burr, a "
        "bridge leaving the pad) · inner = it touches the outline and eats INWARD, its "
        "body inside the structure (a notch, an edge chip, a starved fillet receding "
        "from the edge) · interior = it lies wholly inside and does NOT touch the "
        "outline (a void, a pit, a bubble) — touching the edge would make it a "
        "different defect · any = no such constraint, it may sit anywhere on or in the "
        "structure. This is about the structure's own inside and outside; no parent or "
        "assembly direction is involved",
    ),
    "sites_spanned": (
        "one",
        "two_or_more",
        "how many host_structures ONE occurrence covers — count instances of the host YOU "
        "settled above, not pieces inside one: a mark enclosing both pads of one two-pad "
        "footprint spans ONE host. two_or_more is what "
        "makes a defect a CONNECTION — a bridge is one blob lying across two pads, and a "
        "mask that sits on only one of them is not that defect at all. Distinct from "
        "distribution_kind, which counts separate BLOBS: a bridge is one blob over two "
        "sites; three starved leads are three blobs over three sites",
    ),
    "distribution_kind": (
        "single",
        "multi",
        "how many separate, non-touching patches a typical ANNOTATED MASK of this type "
        "contains — read it off the MEASURED histogram given above, not from the "
        "observations. "
        "This is about the mask, not about the event: a board can show one bridging "
        "'incident' and still be annotated as three separate patches, and it is the "
        "MASK the judge scores. Answer `multi` if a substantial share of the examples "
        "carry more than one patch; `single` only if essentially all of them are one, "
        "in which case this dimension has nothing to score and is dropped",
    ),
    # Not a geometry flag: it picks the VISUAL PROMPT, and belongs to the TYPE because how the boards
    # were lit is a property of the capture rig. The values ARE the `visual_prompts` names.
    "visual_prompt": (
        "solderlight",
        "normallight",
        "sam2",
        "none",
        "which precomputed hint would most help someone PLACE this defect. The two solder "
        "options differ only by how the board was LIT, judged from the CLOSE detail of "
        "one joint rather than from how colourful the board looks overall: solderlight = "
        "tri-colour ring light, where a single fillet shows a GRADIENT across its own "
        "curve (red where flat, blue where steep) because colour encodes surface slope · "
        "normallight = ordinary illumination, where every material keeps ONE flat colour "
        "of its own and solder reads as small bright metallic specks. Not solder "
        "photographs? Then choose between: sam2 = the defect sits on a discrete structure "
        "with a findable outline (a component, a lead, a panel, a bottle), so having "
        "every structure segmented helps · none = there is no such structure to sit on "
        "(open texture, uniform surface, free-floating defects), where segmentation would "
        "only add noise",
    ),
}

SECTIONS = [
    "visual appearance (in this imaging)",
    "host structure & location patterns",
    "typical shape",
    "typical size / extent",
    "count / distribution",
    "root cause analysis",
    "look-alikes / cautions",
]
_MAX_WORDS = 400  # synthesis word budget


_ABSENT = "unknown"  # `_flag_block` writes it for a flag with no usable answer; `read_flags` drops it


def _one_line(v: Any) -> str:
    r"""``v`` as ONE front-matter line, by `read_flags`'s own notion of where a line ends.

    `str.splitlines()` — what the reader uses — ends a line on `\r`, `\v`, `\f`, `\x1c`-`\x1e`,
    `\x85`, U+2028 and U+2029 as well as on `\n`, and `str.split()` counts every one as whitespace.
    So collapsing whitespace runs stops a model-authored value from reaching the reader as a SECOND
    line, to be parsed there as a flag of its own.
    """
    return " ".join(str(v).split())


def _flag_value(flags: dict[str, Any], k: str) -> str:
    """What flag ``k`` will actually CARRY once written into a card and read back out; '' for nothing.

    ONE reading for `_faults`, `_flag_block` and `read_flags`, which must agree; a free-text answer of
    literally ``unknown`` is no answer, and fails the gate here rather than at the next check.
    """
    spec = FLAGS[k]
    v = _one_line(flags.get(k, ""))  # one line: this is YAML front-matter
    if spec[0] is FREE_TEXT:
        return "" if v == _ABSENT else v
    return v if v in spec[:-1] else ""


# Cards already in the field carry these older spellings in their front-matter; the reader maps them
# onto the current flags so an existing library behaves identically without being re-distilled. The
# writer (`_flag_block`) emits only the current spellings.
_LEGACY_KEYS = {"fill_target": "fill_host", "roi_kind": "hosts_presented"}
_LEGACY_VALUES = {"hosts_presented": {"multi_region": "multi_host", "single_region": "single_host"}}


def read_flags(card_text: str) -> dict[str, str]:
    """The geometry flags out of a card's front-matter, or {} when it carries none.

    A missing or 'unknown' flag means "carry on as before" downstream: a card without front-matter is
    still a good card, and a geometry the model could not judge is not one to guess at.
    """
    if not card_text.startswith("---"):
        return {}
    end = card_text.find("\n---", 3)
    if end < 0:
        return {}
    out = {}
    for line in card_text[3:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            k = _LEGACY_KEYS.get(k, k)
            if k not in FLAGS:
                continue
            v = _LEGACY_VALUES.get(k, {}).get(v, v)
            if FLAGS[k][0] is FREE_TEXT:
                if v and v != _ABSENT:
                    out[k] = v
            elif v in FLAGS[k][:-1]:
                out[k] = v
    return out


def _flag_block(card: dict[str, Any]) -> list[str]:
    """The type's geometry as machine-readable front-matter, with the reason it was judged that way.

    A wrong flag changes the pipeline silently, so the justification is stored beside the value."""
    got = card.get("flags") if isinstance(card.get("flags"), dict) else {}
    out = ["---"]
    for k in FLAGS:
        out.append(f"{k}: {_flag_value(got, k) or _ABSENT}")
    why = _one_line(got.get("why", ""))
    if why:
        out.append(f"flags_why: {why}")
    out.append("---")
    return out


def exemplar_note(i: int, observation: str) -> str:
    """The distill caption, framed for a prompt — ONE spelling for the proposer and the judge sides.

    The rendered block is exempt from the leak guards at runtime (the caption is the MODEL's own
    words, and one caption must not be able to kill a beam), so the FRAME is policed by a template
    test with neutral filler instead — one function keeps that pin honest for both call sites."""
    return f"EXAMPLE {i} annotation notes (captioned from THIS example): {observation}"


def observation_for(texture: str, defect: str, model: str, cache_dir, defect_path: str, mask=None) -> str:
    """The distill pass's caption of ONE exemplar image, rendered as a short text line.

    Matched by the filename stored with each observation — `_observe` skips failed captions, so a
    positional match hands out the neighbour's caption after a skip. "" whenever anything is
    missing: the caption is an extra, never a gate."""
    import json as _json
    from pathlib import Path as _Path

    try:
        # `slug`, same as the writer `_observe` — a second character class here misses its file.
        fp = _Path(cache_dir) / f"{texture}+{defect}" / f"observations.{slug(str(model))}.json"
        if not fp.is_file():
            return ""
        obs = _json.load(open(fp)).get("obs") or []
        o = next((e for e in obs if e.get("image") == _Path(defect_path).name), None)
        if o is None:
            return ""  # this image was never captioned
        # `across_two` is ambiguous between one patch SPANNING two structures (a connection) and a
        # mask whose SEVERAL patches sit on one structure each (two filled fillets) — and the wrong
        # reading redraws the defect (a spanning gloss on a two-patch example pulled masks across
        # the component body). The example's own mask disambiguates: its patch count is measured
        # here and the gloss says which geometry this example actually shows.
        n_patch = 0
        if mask is not None:
            import numpy as _np

            _a = (_np.asarray(mask.convert("L")) > 127).astype("uint8")
            if _a.any():
                import cv2 as _cv2

                n_patch = int(_cv2.connectedComponents(_a)[0]) - 1
        extent_gloss = {
            "all_of_one": "the mark covers ALL of one host structure",
            "part_of_one": "the mark covers only PART of one host structure",
            "no_host": "the mark sits on no particular structure",
        }
        ext = str(o.get("extent", "")).strip()
        bits = []
        if o.get("host"):
            bits.append(f"host: {o['host']}")
        if ext == "across_two":
            if n_patch > 1:
                bits.append(
                    f"extent: across_two — this example's mask holds {n_patch} SEPARATE patches, "
                    "each sitting on (and filling) its own host structure; no patch spans between them"
                )
            elif n_patch == 1:
                bits.append(
                    "extent: across_two — ONE patch lying across two or more host structures "
                    "(it reaches onto each of them and covers the ground between)"
                )
            else:
                bits.append("extent: across_two")
        elif ext:
            bits.append(f"extent: {ext}" + (f" ({extent_gloss[ext]})" if ext in extent_gloss else ""))
        if o.get("shape_size"):
            bits.append(f"shape/size: {o['shape_size']}")
        if o.get("boundary"):
            bits.append(f"boundary: {o['boundary']}")
        return "; ".join(bits)
    except Exception:  # noqa: BLE001 — a caption is an extra; a failure must not cost the run
        return ""


def _card_markdown(label: str, n_obs: int, card: dict[str, Any]) -> str:
    """Assemble the final markdown card from the synthesized per-section dict."""
    lines = [
        *_flag_block(card),
        f"# {label} — dataset-distilled defect card",
        f"> Source: {n_obs} real defect images from the AnomalyGen dataset, captioned and synthesized "
        "by VLM. TEXT ONLY. Same imaging modality as deployment — appearance language aligned by "
        "construction.",
        "",
    ]
    for s in SECTIONS:
        lines += [f"## {s}", str(card.get(s, "")).strip(), ""]
    return "\n".join(lines)


def distill_card(
    vlm: Any,
    dataset_path: str,
    texture: str,
    defect: str,
    *,
    model: str,
    n_examples: int = 100,
    max_tokens: int = MAX_TOKENS_CARD,
    obs_cache_dir: Optional[Path] = None,
) -> str:
    """Caption the real examples of (texture, defect) via ``vlm``, then synthesize a text card.

    Distillation is a ONE-TIME op, so it captions up to ``n_examples`` rather than a small sample:
    ``len(examples) + 1`` calls, and a type with more logs a warning and distills only the first
    ``n_examples``. The card is NOT written here; ``obs_cache_dir`` is a DIRECTORY the captions may
    be kept in, named by `_observe`.
    """
    label = f"{texture}/{defect}"
    total = count_pairs(dataset_path, texture, defect)
    if total > n_examples:
        log.warning(
            f"card_distill: {label} has {total} real examples but the distill cap is {n_examples} "
            f"— distilling only the first {n_examples}; {total - n_examples} NOT distilled"
        )
    exemplars = load_exemplars(dataset_path, texture, defect, k=n_examples)
    # The card reaches every propose and judge prompt, so its captions and synthesis are traced too.
    set_trace_ctx(f"{texture}_{defect}_card")
    try:
        obs = _observe(vlm, exemplars, label, model, obs_cache_dir)
        return _synthesize(vlm, obs, label, model, max_tokens, patches=patch_histogram(dataset_path, texture, defect))
    finally:
        set_trace_ctx("")  # every path resets: a raised caption must not tag the rest of the run


def ask_stamp(*parts: str) -> str:
    """Fingerprint of the QUESTION a cached answer answers; shared with `regions`' site cache.

    Version a cache on the ask, not on a key list, and a reworded question re-asks by itself.
    ``parts`` are joined with a separator that cannot occur in a prompt, so no regrouping collides.
    """
    return hashlib.sha1("\x00".join(parts).encode(), usedforsecurity=False).hexdigest()[:12]


def _strip(obs: list[dict]) -> list[dict[str, str]]:
    """An observation list as the synthesis reads it: the caption keys only, each a string. The
    filename an entry carries is for `observation_for`, not the synthesis."""
    return [{k: str(o.get(k, "")) for k in _CAPTION_KEYS} for o in obs]


def _observe(vlm, exemplars, label, model, cache_dir: Optional[Path]):
    """One structured observation per real defect image — the expensive half, so it is cached.

    Captioning is ``len(exemplars)`` VLM calls against the synthesis's one, and an unusable card is
    almost always the synthesis. The cache file carries the MODEL: ``_amg_cache`` is shared by every
    arm and these are one model's words about one image, so one name would serve a comparison arm
    the other's captions. The ask is stamped inside the entry (`ask_stamp`), the other half of the
    same key.
    """
    cache = (Path(cache_dir) / f"observations.{slug(str(model))}.json") if cache_dir else None
    if cache and cache.is_file():
        try:
            blob = json.loads(cache.read_text())
            # Stamped on the ASK, not on `_CAPTION_KEYS`: a key added to that list but never to the
            # ask leaves the stamp matching, and stale observations are served with the field blank.
            blob = blob if isinstance(blob, dict) else {}
            obs = blob.get("obs") or []
            # An entry without its image's filename cannot be matched back by `observation_for`.
            if blob.get("ask") == ask_stamp(CAPTION_ASK) and obs and all(o.get("image") for o in obs):
                log.info(f"card_distill: {label} reusing {len(obs)} cached observations ({cache})")
                return _strip(obs)
            log.info(f"card_distill: {label} observation cache predates the current caption schema — re-captioning")
        except (json.JSONDecodeError, OSError) as e:
            # A COST event, not a correctness one — the answer is unchanged, the spend is not. See
            # an unreadable observation cache says every caption is being paid for again.
            log.warning(
                f"degraded[obs_cache_invalid]: {label}: {cache.name}: "
                f"{type(e).__name__}: {e} — re-captioning every example"
            )

    obs: list[dict[str, str]] = []
    for ex in exemplars:
        cap_blocks: list[Any] = [CAPTION_ASK % label, outline_mask(ex.defect, ex.mask)]
        ez, _ = Views.from_vlm(vlm).zoom_around_mask(ex.defect, ex.mask)  # magnify the real defect
        if ez is not None:
            cap_blocks += [
                "Same defect, ZOOMED to the annotated region — describe its fine structure/appearance from THIS:",
                ez,
            ]
        try:
            with watch_reply_repair() as parse:
                r = vlm.call_json(CAPTION_SYS, cap_blocks, model=model, max_tokens=MAX_TOKENS_CAPTION)
            if parse.repaired:
                # Cut off by the budget: the fields that arrived are whole and the rest would be blank
                # — a row that weakens the synthesis quietly and, cached, is served on every later run.
                # Skipped like a failed caption; the cache's ask stamp cannot tell a partial row apart.
                log.warning(
                    f"degraded[caption_repaired]: {Path(ex.defect_path).name if ex.defect_path else label}: "
                    "reply cut off before its end — caption skipped, not cached"
                )
                continue
            # A weak model answers this ask with a JSON ARRAY rather than the object it was asked
            # for. In measurement a small minority of caption replies are arrays, and every observed
            # array CONTAINED the object — so unwrap it and keep the sample. Falling back to
            # None rather than {} is deliberate: an empty dict would append a row of empty strings,
            # quietly weakening the synthesis instead of loudly losing one caption.
            if isinstance(r, list):
                r = next((x for x in r if isinstance(x, dict)), None)
            # READING the reply is inside the guard: no schema is declared, so a reply that is
            # neither an object nor an array carrying one raises HERE, and escaping would lose every
            # caption already paid for (cache written only after the loop).
            entry = {k: str(r.get(k, "")) for k in _CAPTION_KEYS}
            # A skipped caption shifts every later index, so each entry names its image instead.
            entry["image"] = Path(ex.defect_path).name if ex.defect_path else ""
            obs.append(entry)
        except Exception as e:  # noqa: BLE001 — one bad caption must not sink the whole card
            # ...but the card is then synthesized from fewer real examples than it claims.
            log.warning(f"degraded[caption_failed]: {type(e).__name__}: {e}")
    if cache and obs:
        atomic_write(
            cache,
            lambda t: t.write_text(
                json.dumps({"ask": ask_stamp(CAPTION_ASK), "keys": list(_CAPTION_KEYS), "obs": obs}, indent=1)
            ),
        )
    return _strip(obs)


def _faults(card: dict[str, Any]) -> list[str]:
    """What is wrong with a synthesized card, in the words the retry will be given.

    A missing SECTION is visible — the card reads thin. A missing FLAG is not: it is written
    `unknown`, dropped by `read_flags`, read everywhere as "carry on as before", and the type quietly
    loses its host layout, its fill rule or its visual prompt. That silence is why this check exists.
    """
    got = card.get("flags") if isinstance(card.get("flags"), dict) else {}
    # Through `_flag_value`, so "answered" means the value SURVIVES being written and read back.
    bad = [k for k in FLAGS if not _flag_value(got, k)]
    filled = [s for s in SECTIONS if str(card.get(s, "")).strip()]
    out = []
    if bad:
        out.append(
            "these flags are missing or not one of their allowed values: "
            + "; ".join(
                f"{k} (allowed: {f'a short phrase, not empty and not {_ABSENT!r}' if FLAGS[k][0] is FREE_TEXT else ', '.join(FLAGS[k][:-1])})"
                for k in bad
            )
        )
    if len(filled) < max(2, len(SECTIONS) // 2):
        out.append(
            f"only {len(filled)} of {len(SECTIONS)} sections have any text in them ({', '.join(filled) or 'none'})"
        )
    return out


def _synthesize(vlm, obs, label, model, max_tokens, attempts: int = 3, patches=None):
    """Turn the observations into a card, insisting it is complete; the retry is TOLD what was wrong,
    because a silent re-ask is just a second roll of the same dice.
    """
    blocks: list[Any] = [f"Defect type: {label}. {len(obs)} per-image observations follow as JSON lines:"]
    if patches:
        _n = sum(patches.values())
        _multi = sum(v for k, v in patches.items() if k > 1)
        blocks.append(
            "MEASURED from this type's real annotation masks — how many separate, non-touching patches "
            f"each one contains: {patches} (patches: how many masks). {_multi} of {_n} carry more than "
            "one. This is a count of the annotations themselves, not an impression, and it is the "
            "evidence for distribution_kind — use it rather than your sense of how many defect EVENTS "
            "an image shows, because what gets scored is a MASK."
        )
    # The extent captions are the direct evidence fill_host and sites_spanned summarize — and the
    # synthesis has shipped flags that contradict their own majority (a pool captioned 10/16
    # 'all_of_one' went out as fill part, and the run drew slivers). The tally is computed here, on
    # the same terms as the patch histogram above: measured context the synthesis must reconcile
    # with, never an override — a justified minority call stays available, but it has to SAY so.
    _votes = Counter(str(o.get("extent", "")).strip() for o in obs if str(o.get("extent", "")).strip())
    if _votes:
        blocks.append(
            "MEASURED tally of the per-image 'extent' captions above: "
            + ", ".join(f"{k} x{v}" for k, v in _votes.most_common())
            + ". These captions are the evidence FILL_HOST summarizes: a fill_host that contradicts "
            "their majority is usually a synthesis slip rather than a judgment call — if you write "
            "one anyway, name in flags_why which observations overrule the majority. For "
            "sites_spanned they are only raw material: the captions vote at whatever granularity "
            "each captioner picked, and sites_spanned counts instances of the host YOU settled — "
            "a two-land footprint reads across_two land-by-land yet spans ONE footprint-level host, "
            "so translate the votes to your host before counting, do not tally them as-is."
        )
    blocks += [json.dumps(o) for o in obs]
    schema: dict[str, Any] = {
        "flags": {k: (spec[0] if spec[0] is FREE_TEXT else "|".join(spec[:-1])) for k, spec in FLAGS.items()}
    }
    schema["flags"]["why"] = "<one line: what in the observations decided each, with rough counts>"
    schema.update(
        {s: "<text>" for s in SECTIONS}
    )  # flags FIRST: a truncated parse loses prose, not the machine-readable half
    blocks.append(
        "Synthesize the reference card as JSON: "
        + json.dumps(schema)
        + f" — the sections total under {_MAX_WORDS} words. "
        + "Each flag is named after the judging dimension it configures, and describes the defect "
        + "TYPE rather than any one image — judge from the observations as a whole and pick the value "
        + "that fits MOST of them. Settle host_structure FIRST: fill_host, boundary_contact and "
        + "sites_spanned are all measured against it, and answered against different structures they "
        + "contradict each other silently. sites_spanned also depends on hosts_presented — it asks about "
        + "one structure among several, so answer one unless hosts_presented is multi_host. "
        + " · ".join(f"{k}: {spec[-1]}" for k, spec in FLAGS.items())
        + ". The per-image 'extent' fields are the evidence for fill_host and sites_spanned and the "
        + "measured patch histogram is the evidence for distribution_kind (a substantial minority with "
        + "more than one patch is enough for `multi`) and the "
        + "'boundary' fields are the evidence for boundary_contact — count them ('across_two' makes "
        + "sites_spanned two_or_more only where the two things spanned are two instances of YOUR "
        + "host_structure; across two pieces inside one host is still one host), and note in 'why' if "
        + "they disagree, because a type whose "
        + "examples genuinely differ is worth knowing about rather than averaging away. "
        + "EVERY flag must be answered with one of its "
        + "allowed values; there is no 'unknown' and leaving one out is not an option."
    )
    faults: list[str] = []
    for i in range(attempts):
        ask = list(blocks)
        if faults:  # say what was wrong, and that the observations have not changed
            ask.append(
                "Your previous answer for this same type was rejected: "
                + "; and ".join(faults)
                + ". The observations above are unchanged — answer again, complete this time."
            )
        with watch_reply_repair() as parse:
            card = vlm.call_json(SYNTH_SYS, ask, model=model, max_tokens=max_tokens)
        faults = _faults(card)
        if parse.repaired:
            # Flags come first, so a card cut off by the budget keeps every flag and its leading
            # sections and reads complete to `_faults`; it is still not the card that was written.
            faults.append(
                f"the answer was cut off before its end — keep the sections under {_MAX_WORDS} words in total"
            )
        if not faults:
            return _card_markdown(label, len(obs), card)
        log.warning(f"card_distill: {label} synthesis attempt {i + 1}/{attempts} rejected — " + "; ".join(faults))
    raise ValueError(
        f"{attempts} synthesis attempts all incomplete ({'; '.join(faults)}) — refusing to "
        "write a card that would look finished and quietly drop a flag"
    )


def card_variant(model: str) -> str:
    """The card-file stem: what :func:`build_cards_dir` WRITES and what `defects.LocalCards` READS.

    ONE definition across the two modules, and the model is in the stem because the library at
    the per-dataset `_amg_cache/defect_library` is shared by every arm — both halves are pinned by
    each model gets its own card and the writer and reader agree on the name.
    """
    return f"card_dataset.{slug(model)}"


def build_cards_dir(
    vlm: Any,
    dataset_path: str,
    anomaly_types: Iterable[tuple[str, str]],
    out_dir: str,
    *,
    model: str,
) -> str:
    """Distill a card for every ``(texture, defect)`` in ``anomaly_types`` and write
    ``<out_dir>/<texture>+<defect>/<card_variant(model)>.md`` ; return ``out_dir``. Skips a card that
    already exists FOR THIS MODEL. Keyed by the ``"texture+defect"`` class string (AnomalyGen's
    ``class_to_id`` convention), which is what `defects.LocalCards` looks up by.
    """
    root = Path(out_dir)
    for texture, defect in anomaly_types:
        dest = root / f"{texture}+{defect}" / f"{card_variant(model)}.md"
        if dest.exists():
            log.info(f"card_distill: skip {texture}+{defect} (exists: {dest})")
            continue
        try:
            card = distill_card(vlm, dataset_path, texture, defect, model=model, obs_cache_dir=dest.parent)
        except Exception as e:  # noqa: BLE001 — keep distilling the remaining types on any single failure
            # This type is then proposed and judged with NO card and every geometry flag unset — see
            # a type whose card could not be distilled is named as running without one.
            log.warning(
                f"degraded[card_failed]: {texture}+{defect}: {type(e).__name__}: {e} — this "
                f"type is generated and judged with no card"
            )
            continue
        # Atomically: a card takes minutes to distill and is read by whatever run happens to be going.
        atomic_write(dest, lambda t: t.write_text(card))
        log.info(f"card_distill: wrote {dest} ({len(card.split())} words)")
    return out_dir
