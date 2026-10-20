# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ring-confined THT placement: one candidate factory and one placer for six of the seven classes.

WHY ONE MODULE AND NOT SIX
``bright``, ``dry_contact``, ``hidden``, ``interrupted``, ``less_solder`` and ``trapped`` share a
single mechanism -- place the submask in the ROI and keep as much of the annotation as the clip
allows. They differ only in the two values ``spec.roi_rule()`` returns, so six modules would be six
copies of one function differing by the string ``"BC"`` or ``"C"``. ``bridge`` is the class with a
genuinely different mechanism, and it gets its own module.

WHAT IS BEING MAXIMISED, AND WHY IT IS AREA AND NOT POSITION
AMP aligns a submask's centre to the ROI's BOUNDING BOX centre. For a pad-shaped ROI that is the pad
centre; on an ANNULUS it is the hole, so a centred mask lands where there is no ROI and the clip
takes most of it. An earlier version swept the submask's centre over the ROI's own pixels and broke
ties outward. It scattered the mask correctly but kept too little of it: measured against the
reference batch, 46% of the annotation survived where the reference kept 70%, and since the mask
bbox is what sizes the generation crop, a shrunken mask also moves the model off its trained scale.

So the rule is read off the ROI's SHAPE, not off a class name or a radius:

  * ROI has no pixel at its own centroid (an annulus) -> translate by up to ``shift_frac`` of the
    ROI's equivalent radius, which is where a ring defect can actually sit;
  * otherwise -> centre-align strictly.

Draws stop as soon as ``target_frac`` of the annotation survives, and a best draw that keeps less
than half falls back to the other alignment mode. Nothing here reads the annotation's source pad.

The choice reaches AMP as a shift range (or ``strict_alignment`` for the centred case), so AMP itself
is untouched. Selection is made on the mask AMP actually returned, never on the requested offset, so
it stays correct however AMP orders shift, scale and rotation internally.
"""

import os
import tempfile
from dataclasses import replace
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cosmos_framework.utils import log
from PIL import Image

from anomalygen.auto_mask_placement.cad2roi.mask_utils import save_temp_mask
from anomalygen.auto_mask_placement.cad2roi.parser import ROICandidate
from anomalygen.auto_mask_placement.config import AugmentationParams
from anomalygen.auto_mask_placement.core import AutoMaskPlacement
from anomalygen.auto_mask_placement.data_types import AlignmentPoint
from anomalygen.auto_mask_placement.tht2roi.regions import joints_carrying, roi_for
from anomalygen.auto_mask_placement.tht2roi.spec import roi_rule

# The default augmentation for THT AMP.
# The three use_dynamic_* flags at the bottom are load-bearing, not boilerplate.
# AugmentationParams.from_config_file turns a dynamic range off whenever the matching explicit range
# appears in the file; constructing the dataclass directly does NOT. Omit them and every range set
# here is silently replaced by AMP's own per-ROI calculation -- nothing raises, and the only signal
# is "(dynamic)" rather than "(fixed)" in the placement log.
THT_AUGMENTATION = AugmentationParams(
    shift_x_probability=0.0,
    shift_y_probability=0.0,
    shift_x_range=(0, 0),
    shift_y_range=(0, 0),
    rotation_probability=0.5,
    rotation_range=(-15, 15),
    shear_x_probability=0.0,
    shear_y_probability=0.0,
    shear_x_range=(0, 0),
    shear_y_range=(0, 0),
    # No shrinking. A base mask's size relative to its own crop is the defect's physical scale, so a
    # range below 1.0 would fabricate a defect smaller than any that was observed. The upper bound is
    # there because clipping to a ring ROI cuts part of the mask away; a little growth lets more of it
    # survive that clip without inventing size.
    #
    # The RANGE is not here, though the probability is: _draw fixes the scale per draw from SCALES,
    # because a range sampled inside AMP is a number this module never learns, and it has to know it
    # to say what share of a draw survived.
    scale_x_probability=1.0,
    scale_y_probability=1.0,
    scale_x_range=(1.0, 1.0),
    scale_y_range=(1.0, 1.0),
    scale_fixed_ratio=True,
    flip_x_probability=0.4,
    flip_y_probability=0.4,
    morph_probability=0.0,
    use_dynamic_shift_range=False,
    use_dynamic_rotation_range=False,
    use_dynamic_shear_range=False,
)


class RingPlacementError(RuntimeError):
    """A placement violated a structural invariant (as opposed to simply not landing)."""


N_DRAWS = 16  # placement draws per ROI
TARGET_FRAC = 0.95  # stop drawing once this share of the DRAW survives the clip

# The scales a draw may be made at, swept in order rather than sampled, so every draw's pre-clip
# area is a number this module knows. It used to be a random range handed to AMP, which made
# retention unmeasurable -- see _best_draw for what that cost.
SCALES = (1.0, 1.1, 1.2)

# Below this share of the draw surviving the clip, the placement is recorded as SMALL rather than
# passed off as a placement. 0.5 is what the free and text routes use as min_area_ratio and what
# cad2roi's LessSolderMaskPlacer._check_min_area enforces; cad_tht had no floor at all, so a
# placement keeping 2% of the annotation was recorded OK and inpainted as a full defect.
MIN_RETENTION = 0.5
SHIFT_FRAC = 0.70  # translation range, as a fraction of the ROI's equivalent radius
MIN_ROI_PX = 20  # below this a ROI cannot hold a defect; the sample is skipped rather than forced


def _is_annulus(roi_bool: np.ndarray) -> bool:
    """True when the ROI has no pixel where AMP would centre a mask on it.

    AMP centre-aligns, so such a ROI would park a centred mask on the hole and the clip would take
    it. The test is on the ROI's shape and uses no class name, so it transfers to any region rule.

    Measured at the BOUNDING-BOX centre, which is the point AMP actually uses
    (``MaskAugmentor._get_roi_alignment_position`` returns ``roi_w / 2, roi_h / 2`` over the ROI's
    extent). This asked the pixel CENTROID instead. The two agree for a symmetric ring and part
    company for anything else -- a crescent left by a partly clipped joint can have its centroid on
    ROI pixels while the bbox centre sits in the hole, so the ROI was called solid, the mask was
    centred straight onto the hole, and the whole 16-draw sweep was spent before the retention
    fallback rescued it in the other alignment mode.
    """
    ys, xs = np.nonzero(roi_bool)
    cy = min(int((ys.min() + ys.max() + 1) // 2), roi_bool.shape[0] - 1)
    cx = min(int((xs.min() + xs.max() + 1) // 2), roi_bool.shape[1] - 1)
    return not roi_bool[cy, cx]


def _annotation_area(submask_path: str, w: int, h: int) -> int:
    """The submask's area once resized to the canvas -- what a placement is measured against.

    Nearest-neighbour to match what AMP does internally, so "kept 638/910" compares like with like.
    """
    src = np.array(Image.open(submask_path).convert("L"))
    if src.shape != (h, w):
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_NEAREST)
    return int((src > 127).sum())


def get_ring_candidates(joints, tht_defect: str, min_roi_px: int = MIN_ROI_PX) -> List[ROICandidate]:
    """One ROI candidate per joint: the joint's region letters, with its inside cut away.

    The whole placement policy for a ring class lives in the candidate's shape -- by the time the
    placer sees it, "bright uses B+C with the inner half removed" is already a mask. That is what
    keeps AMP unmodified.
    """
    letters, inner_k = roi_rule(tht_defect)
    candidates = []
    # A joint must carry every region its rule names; a clipped one would otherwise contribute a
    # comfortably large "B+C" candidate that is really just B, and n_instances would pick between it
    # and the intact joint at random.
    for joint in joints_carrying(joints, letters, tht_defect):
        roi = roi_for(joint, letters, inner_k)
        area = int((roi > 127).sum())
        if area < min_roi_px:
            log.warning(f"  {tht_defect}: ROI only {area} px after the {inner_k:.2f} inner cut, skipping joint")
            continue
        ys, xs = np.nonzero(roi > 127)
        candidates.append(
            ROICandidate(
                mask=roi,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                centroid=(float(xs.mean()), float(ys.mean())),
                area=area,
                source_classes=list(letters),
                component_ids=[],
            )
        )
    return candidates


class RingMaskPlacer:
    """Place a defect submask into a ring ROI by sweeping it and selecting outward."""

    def __init__(
        self,
        augmentation: AugmentationParams = None,
        n_draws: int = N_DRAWS,
        target_frac: float = TARGET_FRAC,
        shift_frac: float = SHIFT_FRAC,
    ):
        self.aug = augmentation or THT_AUGMENTATION
        self.n_draws = n_draws
        self.target_frac = target_frac
        self.shift_frac = shift_frac

    def _draw(self, roi_path, submask_path, shift, w, h, seed, work, scale=1.0) -> Optional[np.ndarray]:
        """One AMP placement at a FIXED ``scale``. ``shift`` is the +/- translation range, 0 = centred.

        A shift of 0 goes through ``strict_alignment``, which force-zeroes the shift probabilities in
        core.py -- so "centred" and "translated" are mutually exclusive by construction rather than
        by leaving a range at (0, 0) and trusting it.
        """
        strict = shift == 0
        aug = replace(self.aug, scale_x_range=(scale, scale), scale_y_range=(scale, scale))
        aug = (
            aug
            if strict
            else replace(
                aug,
                shift_x_probability=1.0,
                shift_y_probability=1.0,
                shift_x_range=(-shift, shift),
                shift_y_range=(-shift, shift),
                use_dynamic_shift_range=False,
            )
        )
        amp = AutoMaskPlacement(
            image_width=w,
            image_height=h,
            augmentation_params=aug,
            roi_alignment_point=AlignmentPoint.CENTER,
            submask_alignment_point=AlignmentPoint.CENTER,
            strict_alignment=strict,
            random_seed=seed,
            max_retry_per_mask=8,
            separate_rois=False,
        )
        amp.load_combined_rois(roi_image_paths=[roi_path])
        try:
            res = amp.process_submask(
                submask_path=submask_path, n_instances=1, output_dir=work, strict_alignment=strict
            )
        except Exception as exc:  # a single bad draw must not lose the other 15
            log.warning(f"  AMP draw failed: {exc}")
            return None
        if not res:
            return None
        # Read the path AMP reported rather than globbing the directory: draws share `work`, so a
        # glob would be ambiguous about which output belongs to this draw.
        return np.array(Image.open(res[0]["output_path"]).convert("L"))

    def place_on_roi(
        self, submask_path: str, roi_candidate: ROICandidate, img_shape: tuple, seed: int = 42
    ) -> Tuple[Optional[np.ndarray], str]:
        """Place the submask into one ROI, keeping as much of the annotation as the clip allows.

        Returns ``(mask, note)``; ``mask`` is None when nothing landed, which the caller reports as a
        skipped sample. There is deliberately no "use the ROI itself" fallback: the ROI is the whole
        legal region, and shipping it as a defect mask would paint the entire solder ring.

        WHETHER TO TRANSLATE is read off the ROI, not the class: AMP centre-aligns, so a ROI with no
        pixel at its own centroid would park the mask on the hole and clip it away. Those get a
        translation range of SHIFT_FRAC of the ROI's equivalent radius; a ROI that does cover its own
        centroid is placed centred.

        The draw stops as soon as TARGET_FRAC of the annotation survives, and falls back to the other
        alignment mode when the best draw keeps less than half of it -- the case that guards is a
        compact annotation on an annulus ROI, where centring parks it on the hole.
        """
        h, w = img_shape
        roi_bool = roi_candidate.mask > 127
        ys, xs = np.nonzero(roi_bool)
        if len(ys) < 4:
            return None, "EMPTY_ROI"

        target = _annotation_area(submask_path, w, h)
        radius = float(np.sqrt(roi_bool.sum() / np.pi))
        shift = int(round(self.shift_frac * radius)) if _is_annulus(roi_bool) else 0

        roi_path = save_temp_mask(roi_candidate.mask)
        try:
            with tempfile.TemporaryDirectory() as work:
                best = self._best_draw(roi_path, submask_path, shift, w, h, seed, work, target)
                note = "centred" if shift == 0 else f"shift +/-{shift}px"
                # Whatever the ROI's shape suggested, a draw that keeps less than half of itself is
                # worth one attempt at the other mode -- the shape prior is a prior, not a proof.
                if best is None or best[1] < MIN_RETENTION * best[2]:
                    other = self._best_draw(
                        roi_path,
                        submask_path,
                        0 if shift else int(round(self.shift_frac * radius)),
                        w,
                        h,
                        seed,
                        work,
                        target,
                    )
                    if other is not None and (best is None or other[1] / other[2] > best[1] / best[2]):
                        best, note = other, note + "+rescued"
        finally:
            os.unlink(roi_path)

        if best is None:
            return None, "NO_PLACEMENT"
        mask, kept, drawn = best
        retention = kept / drawn
        # Below the floor the mask is still returned -- the free and text routes write an undersized
        # placement too -- but it is labelled, so the caller records SMALL instead of OK and the run
        # does not inpaint a sliver as a full defect on the strength of a log line nobody read.
        label = "SMALL" if retention < MIN_RETENTION else note
        return mask, f"{label}, kept {kept}/{drawn:.0f}px ({retention:.0%} of the draw)"

    def _best_draw(self, roi_path, submask_path, shift, w, h, seed, work, target):
        """The best-RETAINED on-ROI draw of ``n_draws``, stopping once TARGET_FRAC of one survives.

        Returns ``(mask, kept, drawn)`` -- what survived the clip, and what the draw held before it.

        MEASURED AGAINST THE DRAW, NOT THE ANNOTATION, which is the whole of this function's history.
        AMP was given a scale RANGE, so every draw carried 1.0 to 1.44x the annotation's pixels
        before clipping while ``target`` stayed the raw annotation area. Picking the largest survivor
        therefore picked the largest SCALE: on the fixture here a 1.2x draw kept 238 px of the 323 it
        had (74%) and beat a 1.0x draw that kept 186 of 224 (83%), reported it as "kept 238/224px",
        and tripped the early stop first -- so the loop ended on the worse placement and never made
        the remaining draws. A metric that can exceed 100% is not measuring retention.
        """
        best = None
        for att in range(self.n_draws):
            scale = SCALES[att % len(SCALES)]
            drawn = target * scale * scale
            placed = self._draw(roi_path, submask_path, shift, w, h, seed + att, work, scale=scale)
            if placed is None:
                continue
            kept = int((placed > 127).sum())
            if kept < 4:
                continue
            if best is None or kept / drawn > best[1] / best[2]:
                best = (placed.copy(), kept, drawn)
            if kept >= self.target_frac * drawn:
                break
        return best

    def place_all(
        self,
        submask_path: str,
        candidates: List[ROICandidate],
        img_shape: tuple,
        n_instances: int = 1,
        n_seeds: int = 1,
        base_seed: int = 42,
    ) -> dict:
        """For each seed, place the submask on ``n_instances`` of the joints in this crop.

        Returns ``{seed_index: (n_requested, n_placed, mask or None, undersized)}``.

        The fourth element is what makes a low-retention placement visible. The other routes record
        an undersized placement as SMALL rather than OK and still write it; cad_tht had no floor, so
        a placement keeping 2% of its draw went into testcase.jsonl as OK and was inpainted as a
        full defect. Both THT placers carry it and ``_process_cad_tht_sample`` is their only caller,
        so this is a wider tuple than cad2roi's rather than a break of a shared contract.

        ``n_instances`` is a count, never -1. One joint carries one defect: the generated crop is
        composited back onto the board against a single joint id, so a second defect on the same crop
        would have nowhere to be recorded.
        """
        if not candidates:
            return {}
        h, w = img_shape
        n_total = len(candidates)
        n_sel = min(max(1, n_instances), n_total)
        results = {}

        for seed_i in range(n_seeds):
            rng = np.random.RandomState(base_seed + seed_i)
            roi_indices = rng.choice(n_total, size=n_sel, replace=False)

            combined = np.zeros((h, w), dtype=np.uint8)
            n_ok = 0
            small = False
            for ri in roi_indices:
                # Every (seed, ROI) pair gets its OWN block of n_draws consecutive seeds, because
                # place_on_roi spends seed..seed+n_draws-1 on its sweep. A small per-ROI offset would
                # let neighbouring blocks overlap -- with a stride of 1 per seed and 16 draws each,
                # consecutive seeds would share 15 of their 16 draw seeds.
                block = (seed_i * n_total + int(ri)) * self.n_draws
                mask, note = self.place_on_roi(submask_path, candidates[ri], img_shape, seed=base_seed + block)
                if mask is None:
                    log.warning(f"  ring placement failed on ROI {ri}: {note}")
                    continue
                # Unreachable while the ROI and the AMP canvas are the same size, which they are by
                # construction: both come from the cad_mask. AMP resizes a ROI image to the canvas
                # (core.py) AFTER binarising it, so once the two sizes diverge the bilinear resize
                # leaves grey edge values, the bitwise_and clip stops being exact, and the placed mask
                # spills a pixel or two outside the region. This catches that the moment someone
                # decouples the two, rather than letting the region rules quietly stop being enforced.
                if ((mask > 127) & ~(candidates[ri].mask > 127)).any():
                    raise RingPlacementError(
                        f"placed mask escaped its ROI (ROI {candidates[ri].mask.shape} vs canvas {img_shape}) -- "
                        "the ROI and the AMP canvas must be the same size"
                    )
                combined = cv2.bitwise_or(combined, mask)
                if note.startswith("SMALL"):
                    # The note carries the arithmetic -- "kept 366/1673px (22% of the draw)" -- and
                    # it was being dropped on the floor: place_all logged a note only when the
                    # placement returned None, and an undersized one returns a mask. So summary.json
                    # said SMALL and nothing anywhere said how small, which leaves an operator unable
                    # to tell a placement that missed the floor by a point from one that kept a
                    # twentieth of itself. Those want different answers -- the first is a threshold
                    # to argue about, the second is a joint the geometry cannot serve.
                    log.warning(f"  ring ROI {ri}: {note}")
                    small = True
                n_ok += 1

            results[seed_i + 1] = (n_sel, n_ok, combined if combined.sum() > 0 else None, small)
            log.info(f"  ring seed {seed_i + 1}: requested={n_sel}, placed={n_ok}")
        return results
