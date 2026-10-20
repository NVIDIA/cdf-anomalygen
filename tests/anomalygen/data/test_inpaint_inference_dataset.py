# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SDG inference dataset + the validation dataloader helpers.

Covers the JSONL-driven ``InpaintInferenceDataset`` (public API used by generate.py) and the
validation-loader helpers it hosts: ``_val_collate``, ``_build_val_batch_indices``, and the public
``get_inpaint_val_dataloader`` factory.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from anomalygen.configs.texture.constants import (
    DEFAULT_CROP_RATIO,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_NUM_STEPS,
)
from anomalygen.data.crop_geometry import input_size_spec_from_config, plan_for
from anomalygen.data.inpaint_inference_dataset import (
    SEED_OUTPUT_STRIDE,
    SEED_RECORD_STRIDE,
    InpaintInferenceDataset,
    _build_val_batch_indices,
    _sort_records_by_instance_num,
    _val_collate,
    get_inpaint_val_dataloader,
)
from anomalygen.data.utils import MASK_FG_THRESHOLD, sample_token_count


def _save_image(path, size=(32, 32)):
    Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8), mode="RGB").save(path)


def _save_mask(path, blobs, size=32):
    """Save an L-mode mask with a filled 5x5 square at each (row, col) top-left in ``blobs``."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for r, c in blobs:
        arr[r : r + 5, c : c + 5] = 255
    Image.fromarray(arr, mode="L").save(path)


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records))


# --- module-level helpers ------------------------------------------------------------------------


def test_sort_records_by_instance_num(tmp_path):
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    _save_mask(one, blobs=[(2, 2)])
    _save_mask(two, blobs=[(2, 2), (20, 20)])

    img = tmp_path / "img.png"
    _save_image(img, size=(32, 32))
    records = [
        {"index": 0, "image_filename": str(img), "mask_filename": str(two)},
        {"index": 1, "image_filename": str(img), "mask_filename": str(one)},
    ]
    ordered, scans = _sort_records_by_instance_num(records, lambda p: p, lambda p: p)
    # Ascending by instance count: the single-blob record comes first. Each scan carries the mask
    # size and the instance bboxes the validation loader plans its token bound from.
    assert [rec["mask_filename"] for rec in ordered] == [str(one), str(two)]
    assert [len(scan.bboxes) for scan in scans] == [1, 2]
    assert scans[0].size == (32, 32) and scans[0].bboxes == [(2, 2, 6, 6)]


def test_sort_counts_the_instances_generation_splits_not_the_components(tmp_path):
    # Past iteration_generation_max_instance the splitter clusters components, so generation runs
    # at the cluster count; the sort and the token bound must see that count, not the raw one.
    img, msk = tmp_path / "img.png", tmp_path / "m.png"
    _save_image(img)
    _save_mask(msk, blobs=[(2, 2), (2, 25), (25, 2)])
    records = [
        {"index": 0, "image_filename": str(img), "mask_filename": str(msk), "iteration_generation_max_instance": 2}
    ]
    _, scans = _sort_records_by_instance_num(records, lambda p: p, lambda p: p)
    assert len(scans[0].bboxes) == 2


def test_sort_records_by_instance_num_rejects_a_mismatched_pair(tmp_path):
    """The size check rides along with the decode this pass already does."""
    img, msk = tmp_path / "img.png", tmp_path / "m.png"
    _save_image(img, size=(64, 64))
    _save_mask(msk, blobs=[(2, 2)], size=32)
    records = [{"index": 0, "image_filename": str(img), "mask_filename": str(msk)}]
    with pytest.raises(ValueError):
        _sort_records_by_instance_num(records, lambda p: p, lambda p: p)


def test_sort_records_by_instance_num_rejects_a_mask_with_no_instances(tmp_path):
    """A mask whose pixels all sit under the foreground cut has nothing to inpaint, so generation would
    return the input untouched and save it as a result. Refuse it up front, naming the threshold."""
    img, msk = tmp_path / "img.png", tmp_path / "m.png"
    _save_image(img, size=(32, 32))
    arr = np.zeros((32, 32), np.uint8)
    arr[2:7, 2:7] = 1  # a blob, but under the cut
    Image.fromarray(arr, mode="L").save(msk)
    records = [{"index": 7, "image_filename": str(img), "mask_filename": str(msk)}]
    with pytest.raises(ValueError, match=rf"index=7.*no defect pixels.*>= {MASK_FG_THRESHOLD}"):
        _sort_records_by_instance_num(records, lambda p: p, lambda p: p)


# --- InpaintInferenceDataset ---------------------------------------------------------------------


def _make_dataset(tmp_path):
    img = tmp_path / "img.png"
    m_one = tmp_path / "m_one.png"
    m_two = tmp_path / "m_two.png"
    _save_image(img)
    _save_mask(m_one, blobs=[(2, 2)])
    _save_mask(m_two, blobs=[(2, 2), (20, 20)])

    jsonl = tmp_path / "cases.jsonl"
    # Deliberately list the two-instance case first so we can observe the ascending sort.
    _write_jsonl(
        jsonl,
        [
            {"image_filename": str(img), "mask_filename": str(m_two), "anomaly_type": "t+d"},
            {"image_filename": str(img), "mask_filename": str(m_one), "anomaly_type": "t+d"},
        ],
    )
    return jsonl, m_one, m_two


def test_dataset_fills_defaults_and_sorts(tmp_path):
    jsonl, m_one, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(
        jsonl.as_posix(),
        default_guidance=DEFAULT_GUIDANCE,
        default_num_steps=DEFAULT_NUM_STEPS,
        default_max_instances=DEFAULT_MAX_INSTANCES,
    )

    assert len(ds) == 2
    # Sorted ascending by instance count: single-blob mask first.
    assert ds.input_data[0]["mask_filename"] == str(m_one)

    rec = ds.input_data[0]
    assert rec["guidance"] == DEFAULT_GUIDANCE
    assert rec["num_steps"] == DEFAULT_NUM_STEPS
    # Seed is derived per testcase (base_seed + index * stride), not flat, so every testcase gets
    # its own noise. ``rec`` is the single-blob mask, which was line 1 of the JSONL before the sort.
    assert rec["index"] == 1
    assert rec["seed"] == 1 + 1 * SEED_RECORD_STRIDE
    assert rec["num_generated_images"] == 1
    assert rec["iteration_generation_max_instance"] == DEFAULT_MAX_INSTANCES
    assert rec["crop_ratio"] == DEFAULT_CROP_RATIO
    assert rec["poisson_blend"] is False
    # index defaults to original line position (assigned before the sort).
    assert set(r["index"] for r in ds.input_data) == {0, 1}


@pytest.mark.parametrize(
    "bad_ratio, message",
    [
        (None, "must be a number"),
        ("none", "must be a number"),
        ("None", "must be a number"),
        ("", "must be a number"),
        ("wide", "must be a number"),
        ("nan", "must be a finite number >= 1"),
        ("inf", "must be a finite number >= 1"),
        (0, "must be a finite number >= 1"),
        (-1, "must be a finite number >= 1"),
        (0.5, "must be a finite number >= 1"),
    ],
)
def test_dataset_rejects_a_crop_ratio_that_cannot_frame_a_window(tmp_path, bad_ratio, message):
    # crop_ratio is the only window request. A non-number has no meaning, nan slips through the band
    # clamp, and a ratio under 1 asks for a window smaller than the defect: fail loudly rather than
    # silently substituting some other framing.
    img, mask = tmp_path / "img.png", tmp_path / "m.png"
    _save_image(img)
    _save_mask(mask, blobs=[(2, 2)])
    jsonl = tmp_path / "cases.jsonl"
    _write_jsonl(
        jsonl,
        [{"image_filename": str(img), "mask_filename": str(mask), "anomaly_type": "t+d", "crop_ratio": bad_ratio}],
    )
    with pytest.raises(ValueError, match=f"crop_ratio {message}"):
        InpaintInferenceDataset(
            jsonl.as_posix(),
            default_guidance=DEFAULT_GUIDANCE,
            default_num_steps=DEFAULT_NUM_STEPS,
            default_max_instances=DEFAULT_MAX_INSTANCES,
        )


def test_dataset_getitem_loads_pil_pair(tmp_path):
    jsonl, _, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(jsonl.as_posix())
    item = ds[0]
    assert isinstance(item["image"], Image.Image) and item["image"].mode == "RGB"
    assert isinstance(item["mask"], Image.Image) and item["mask"].mode == "L"


def test_dataset_collate_fn_single_sample(tmp_path):
    jsonl, _, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(jsonl.as_posix())
    sample = ds[0]
    assert InpaintInferenceDataset.collate_fn([sample]) is sample
    with pytest.raises(ValueError):
        InpaintInferenceDataset.collate_fn([sample, sample])


def test_dataset_rejects_duplicates(tmp_path):
    img = tmp_path / "img.png"
    mask = tmp_path / "m.png"
    _save_image(img)
    _save_mask(mask, blobs=[(2, 2)])
    rec = {"image_filename": str(img), "mask_filename": str(mask), "anomaly_type": "t+d"}
    jsonl = tmp_path / "dup.jsonl"
    _write_jsonl(jsonl, [rec, dict(rec)])  # identical records
    with pytest.raises(ValueError):
        InpaintInferenceDataset(jsonl.as_posix())


def test_dataset_empty_jsonl_raises(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    with pytest.raises(ValueError):
        InpaintInferenceDataset(jsonl.as_posix())


def test_dataset_base_dir_resolves_relative_testcase_paths(tmp_path):
    # Repo-root-relative image/mask paths must resolve against base_dir, not cwd (training
    # instantiates the loader under a chdir to the framework checkout). The files exist ONLY under
    # base_dir, so cwd-relative resolution (cwd = repo root during pytest) would FileNotFoundError.
    (tmp_path / "imgs").mkdir()
    _save_image(tmp_path / "imgs" / "img.png")
    _save_mask(tmp_path / "imgs" / "msk.png", blobs=[(2, 2)])
    jsonl = tmp_path / "tc.jsonl"
    _write_jsonl(
        jsonl,
        [{"image_filename": "imgs/img.png", "mask_filename": "imgs/msk.png", "anomaly_type": "t+d"}],
    )
    ds = InpaintInferenceDataset(jsonl.as_posix(), base_dir=str(tmp_path))
    item = ds[0]
    assert item["image"].size == (32, 32) and item["mask"].mode == "L"


# --- noise-seed envelope -------------------------------------------------------------------------
#
# A generation's seed is assembled from three offsets applied in three different places:
#   dataset:   base_seed + index * SEED_RECORD_STRIDE      (per testcase)
#   generate:  + output_n * SEED_OUTPUT_STRIDE             (per requested output image)
#   iterative: + instance_j                                (per defect instance, batched depth)
# The strides give each testcase room for SEED_RECORD_STRIDE // SEED_OUTPUT_STRIDE outputs and each
# output room for SEED_OUTPUT_STRIDE instances. Those limits are the whole reason the strides exist,
# so they are pinned here.

_MAX_OUTPUTS_PER_TESTCASE = SEED_RECORD_STRIDE // SEED_OUTPUT_STRIDE  # 16
_MAX_INSTANCES_PER_OUTPUT = SEED_OUTPUT_STRIDE  # 64


def _seed_for(base_seed, index, output_n, instance_j):
    """Seed a given (testcase, output, instance) triple generates with, per the layout above."""
    return base_seed + index * SEED_RECORD_STRIDE + output_n * SEED_OUTPUT_STRIDE + instance_j


def test_dataset_base_seed_propagates_into_derived_seeds(tmp_path):
    # --base_seed is the run-level re-roll knob: same value -> bit-identical noise, bump it for a
    # fresh draw. It must reach every testcase's derived seed.
    jsonl, _, _ = _make_dataset(tmp_path)
    ds = InpaintInferenceDataset(jsonl.as_posix(), base_seed=9000)
    by_index = {r["index"]: r["seed"] for r in ds.input_data}
    assert by_index == {0: 9000, 1: 9000 + SEED_RECORD_STRIDE}


def test_dataset_explicit_seed_wins_over_derived(tmp_path):
    # A testcase that pins its own seed is reproducing a specific generation; the derivation must
    # not overwrite it (setdefault), and it must not leak into its neighbours.
    img = tmp_path / "img.png"
    m_one = tmp_path / "m_one.png"
    m_two = tmp_path / "m_two.png"
    _save_image(img)
    _save_mask(m_one, blobs=[(2, 2)])
    _save_mask(m_two, blobs=[(2, 2), (20, 20)])
    jsonl = tmp_path / "cases.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"image_filename": str(img), "mask_filename": str(m_one), "anomaly_type": "t+d", "seed": 4242},
            {"image_filename": str(img), "mask_filename": str(m_two), "anomaly_type": "t+d"},
        ],
    )
    ds = InpaintInferenceDataset(jsonl.as_posix(), base_seed=7)
    by_index = {r["index"]: r["seed"] for r in ds.input_data}
    assert by_index[0] == 4242  # explicit
    assert by_index[1] == 7 + SEED_RECORD_STRIDE  # derived, unaffected


def test_seeds_are_unique_within_the_documented_envelope():
    # Inside the envelope no two (testcase, output, instance) triples may share a noise draw —
    # that is exactly what the two strides buy.
    seeds = [
        _seed_for(1, index, n, j)
        for index in range(32)
        for n in range(_MAX_OUTPUTS_PER_TESTCASE)
        for j in range(_MAX_INSTANCES_PER_OUTPUT)
    ]
    assert len(set(seeds)) == len(seeds)


def test_seed_collides_past_the_outputs_per_testcase_limit():
    # CHARACTERISATION, not an endorsement: request more than _MAX_OUTPUTS_PER_TESTCASE images and
    # a testcase's extra outputs reuse the NEXT testcase's seeds (both 1025 below) — silently
    # duplicated noise across two different testcases. Guarded in generate.py; if this test starts
    # failing because the strides moved, the guard's limits must move with them.
    assert _seed_for(1, 0, _MAX_OUTPUTS_PER_TESTCASE, 0) == _seed_for(1, 1, 0, 0)


def test_seed_collides_past_the_instances_per_output_limit():
    # Same story one level down: past _MAX_INSTANCES_PER_OUTPUT instances, an output's later
    # instances reuse the next output's seeds (both 65).
    assert _seed_for(1, 0, 0, _MAX_INSTANCES_PER_OUTPUT) == _seed_for(1, 0, 1, 0)


def test_generate_rejects_records_outside_the_seed_envelope():
    # The two counts come straight from user JSONL, so the collisions characterised above are
    # reachable by configuration alone; generate.py must refuse them up front rather than emit
    # duplicate noise. Imported lazily: importing generate.py runs the framework's init_script.
    # init_script also turns autograd off for the whole interpreter, which would break any later
    # test that backpropagates, so restore the flag.
    grad_was_enabled = torch.is_grad_enabled()
    try:
        from anomalygen.scripts.texture.generate import _validate_seed_envelope
    finally:
        torch.set_grad_enabled(grad_was_enabled)

    ok = [{"index": 0, "num_generated_images": _MAX_OUTPUTS_PER_TESTCASE, "iteration_generation_max_instance": 1}]
    assert _validate_seed_envelope(ok) is None  # at the limit is fine

    with pytest.raises(ValueError, match="num_generated_images"):
        _validate_seed_envelope([{"index": 3, "num_generated_images": _MAX_OUTPUTS_PER_TESTCASE + 1}])
    with pytest.raises(ValueError, match="iteration_generation_max_instance"):
        _validate_seed_envelope([{"index": 3, "iteration_generation_max_instance": _MAX_INSTANCES_PER_OUTPUT + 1}])


# --- _val_collate --------------------------------------------------------------------------------


def test_val_collate_index_is_size_inferable_tensor():
    # The framework infers validation batch size from a tensor in the batch; _val_collate must emit
    # ``index`` as a [B] tensor while other fields stay per-key lists.
    batch = [{"index": 3, "anomaly_type": "t+d"}, {"index": 7, "anomaly_type": "t+d"}]
    out = _val_collate(batch)
    assert isinstance(out["index"], torch.Tensor)
    assert out["index"].tolist() == [3, 7]
    assert out["anomaly_type"] == ["t+d", "t+d"]


# --- _build_val_batch_indices ------------------------------------------------------------------------


def _recs(guidances):
    # index = position in the list; num_steps is always present on a dataset record (a default).
    return [{"index": i, "guidance": g, "num_steps": DEFAULT_NUM_STEPS} for i, g in enumerate(guidances)]


def test_plan_batches_are_guidance_homogeneous():
    recs = _recs([6.0, 6.0, 7.0, 7.0])
    plan = _build_val_batch_indices(recs, world_size=1, rank=0, batch_size=4)
    for batch in plan:
        assert len({recs[i]["guidance"] for i in batch}) == 1  # one guidance per batch


def test_plan_equal_batch_count_across_ranks_and_full_coverage():
    recs = _recs([6.0] * 15)
    world = 4
    plans = [_build_val_batch_indices(recs, world_size=world, rank=r, batch_size=4) for r in range(world)]
    counts = [len(p) for p in plans]
    assert len(set(counts)) == 1  # every rank runs the same number of batches (FSDP lockstep)
    seen = set()
    for p in plans:
        for batch in p:
            seen.update(batch)
    assert seen == set(range(15))  # union (post padding-dedup) covers every record


def test_plan_small_group_pads_to_world_multiple():
    recs = _recs([6.0])  # single record, 8 ranks
    plans = [_build_val_batch_indices(recs, world_size=8, rank=r, batch_size=4) for r in range(8)]
    assert all(len(p) == 1 and p[0] == [0] for p in plans)  # each rank gets the (repeated) sole record


# --- get_inpaint_val_dataloader ------------------------------------------------------------------


def test_get_inpaint_val_dataloader_builds_batches(tmp_path):
    img = tmp_path / "img.png"
    msk = tmp_path / "msk.png"
    _save_image(img)
    _save_mask(msk, blobs=[(8, 8)])
    jsonl = tmp_path / "testcase.jsonl"
    # Distinct index per record so they aren't rejected as duplicates.
    _write_jsonl(
        jsonl,
        [
            {
                "index": i,
                "image_filename": str(img),
                "mask_filename": str(msk),
                "anomaly_type": "t+d",
                "guidance": DEFAULT_GUIDANCE,
                "num_steps": DEFAULT_NUM_STEPS,
                "iteration_generation_max_instance": 1,
            }
            for i in range(5)
        ],
    )
    dl = get_inpaint_val_dataloader(str(jsonl), val_batch_size=2, num_workers=0)
    batches = list(dl)
    # 5 records, batch_size 2 -> 3 batches (2, 2, 1); collate yields dict-of-lists incl loaded PIL.
    assert sum(len(b["anomaly_type"]) for b in batches) == 5
    assert "image" in batches[0] and "index" in batches[0]
    assert all(len(b["anomaly_type"]) <= 2 for b in batches)


# The default planner geometry, as a model's diffusion_expert_config would carry it.
_DEC = SimpleNamespace(
    input_size_ladder=(256, 512, 1024, 1536), crop_mode="scale", crop_ratio_range=(1.5, 2.5), inference_scale=1.0
)


def _val_records(img, msk, n):
    return [
        {
            "index": i,
            "image_filename": str(img),
            "mask_filename": str(msk),
            "anomaly_type": "t+d",
            "guidance": DEFAULT_GUIDANCE,
            "num_steps": DEFAULT_NUM_STEPS,
            "iteration_generation_max_instance": 1,
        }
        for i in range(n)
    ]


def test_the_validation_batch_is_cut_to_the_token_budget(tmp_path):
    # Batches are cut by count. A batch of defects planned at the top of the ladder would pack many
    # times the tokens of a training step into one forward, so the count shrinks until the heaviest
    # possible forward fits the budget: one 1536 px input is already most of it.
    img, msk = tmp_path / "img.png", tmp_path / "msk.png"
    _save_image(img, size=(2048, 2048))
    arr = np.zeros((2048, 2048), np.uint8)
    arr[200:1200, 200:1200] = 255  # a 1000 px defect plans at 1536
    Image.fromarray(arr, mode="L").save(msk)
    jsonl = tmp_path / "testcase.jsonl"
    _write_jsonl(jsonl, _val_records(img, msk, 4))
    dl = get_inpaint_val_dataloader(
        str(jsonl), val_batch_size=4, num_workers=0, token_budget=5120, diffusion_expert_config=_DEC
    )
    assert [len(b) for b in dl.batch_sampler] == [1, 1, 1, 1]


def test_small_defects_keep_the_requested_validation_batch_size(tmp_path):
    img, msk = tmp_path / "img.png", tmp_path / "msk.png"
    _save_image(img)
    _save_mask(msk, blobs=[(8, 8)])
    jsonl = tmp_path / "testcase.jsonl"
    _write_jsonl(jsonl, _val_records(img, msk, 4))
    dl = get_inpaint_val_dataloader(
        str(jsonl), val_batch_size=4, num_workers=0, token_budget=5120, diffusion_expert_config=_DEC
    )
    assert [len(b) for b in dl.batch_sampler] == [4]


def test_the_token_bound_comes_from_the_instances_not_the_union_bbox(tmp_path):
    """A union bbox can plan a LIGHTER window than one of its instances: the shared scale shrinks on
    the wide axis and the other axis snaps down with it. The bound must come from the instances
    generation will actually cut, through the call generation makes."""
    img, msk = tmp_path / "img.png", tmp_path / "msk.png"
    _save_image(img, size=(4000, 4000))
    arr = np.zeros((4000, 4000), np.uint8)
    arr[1000:2000, 500:1500] = 255  # a 1000 px square instance
    arr[1000:1010, 3000:3510] = 255  # a small far-off instance: the union bbox is 3010 x 1000
    Image.fromarray(arr, mode="L").save(msk)
    jsonl = tmp_path / "testcase.jsonl"
    _write_jsonl(
        jsonl, [{"image_filename": str(img), "mask_filename": str(msk), "anomaly_type": "t+d", "crop_ratio": 2.5}]
    )
    spec = input_size_spec_from_config(_DEC)

    def tokens(bbox):
        plan = plan_for(spec, bbox, 4000, 4000, crop_ratio=2.5, anomaly_name="t+d")
        return sample_token_count(plan.input_w, plan.input_h)

    assert tokens((500, 1000, 3509, 1999)) < tokens((500, 1000, 1499, 1999))  # the union under-counts
    assert InpaintInferenceDataset(str(jsonl)).planned_tokens(spec) == [tokens((500, 1000, 1499, 1999))]


def test_dataset_rejects_a_testcase_whose_mask_size_differs_from_its_image(tmp_path):
    # AMP repairs its own output (roi_place._verify_and_resize_jsonl); a hand-written JSONL is not.
    img, mask = tmp_path / "img.png", tmp_path / "m.png"
    _save_image(img, size=(32, 32))
    _save_mask(mask, blobs=[(2, 2)], size=64)
    jsonl = tmp_path / "cases.jsonl"
    _write_jsonl(jsonl, [{"image_filename": str(img), "mask_filename": str(mask), "anomaly_type": "t+d"}])

    with pytest.raises(ValueError, match="image/mask size mismatch"):
        InpaintInferenceDataset(jsonl.as_posix())


def test_dataset_accepts_a_matching_pair(tmp_path):
    img, mask = tmp_path / "img.png", tmp_path / "m.png"
    _save_image(img, size=(32, 32))
    _save_mask(mask, blobs=[(2, 2)], size=32)
    jsonl = tmp_path / "cases.jsonl"
    _write_jsonl(jsonl, [{"image_filename": str(img), "mask_filename": str(mask), "anomaly_type": "t+d"}])

    assert len(InpaintInferenceDataset(jsonl.as_posix())) == 1


# --- removed per-record keys ---------------------------------------------------------------------


def _jsonl_with_num_steps(tmp_path, first, second) -> str:
    """A two-record testcase file differing only in ``num_steps``."""
    img, m_one, m_two = tmp_path / "i.png", tmp_path / "m1.png", tmp_path / "m2.png"
    _save_image(img)
    _save_mask(m_one, blobs=[(2, 2)])
    _save_mask(m_two, blobs=[(2, 2), (20, 20)])
    base = {"image_filename": str(img), "anomaly_type": "t+d"}
    jsonl = tmp_path / "steps.jsonl"
    _write_jsonl(
        jsonl,
        [
            {**base, "mask_filename": str(m_one), "num_steps": first},
            {**base, "mask_filename": str(m_two), "num_steps": second},
        ],
    )
    return jsonl.as_posix()


def _dataset_with(tmp_path, extra_first=None, extra_second=None, same_mask=False):
    """The two-record fixture, with extra keys merged into each record.

    ``same_mask`` points both records at the same mask, so they differ only by ``extra_*``.
    """
    img, m_one, m_two = tmp_path / "i.png", tmp_path / "m1.png", tmp_path / "m2.png"
    _save_image(img)
    _save_mask(m_one, blobs=[(2, 2)])
    _save_mask(m_two, blobs=[(2, 2), (20, 20)])
    base = {"image_filename": str(img), "anomaly_type": "t+d"}
    second_mask = m_one if same_mask else m_two
    jsonl = tmp_path / "tc.jsonl"
    _write_jsonl(
        jsonl,
        [
            {**base, "mask_filename": str(m_one), **(extra_first or {})},
            {**base, "mask_filename": str(second_mask), **(extra_second or {})},
        ],
    )
    return InpaintInferenceDataset(
        jsonl.as_posix(),
        default_guidance=DEFAULT_GUIDANCE,
        default_num_steps=DEFAULT_NUM_STEPS,
        default_max_instances=DEFAULT_MAX_INSTANCES,
    )


_IGNORED = pytest.mark.parametrize("extra", [{"shift": 5.0}, {"crop_and_paste": True}], ids=["shift", "crop_and_paste"])


@_IGNORED
def test_an_ignored_per_record_key_is_dropped_not_carried(tmp_path, extra, loguru_lines):
    """Shift comes from the recipe and crop-and-paste is always on, so both are ignored — and must
    not linger: ``_val_collate`` keys off the first record and indexes the rest, so a key on only
    some of them raises ``KeyError``. Dropping is said out loud."""
    (key,) = extra
    ds = _dataset_with(tmp_path, extra_first=extra)
    assert all(key not in rec for rec in ds.input_data)
    assert any(f"set {key!r}" in ln for ln in loguru_lines)


@_IGNORED
def test_a_partially_present_ignored_key_still_collates(tmp_path, extra):
    """A key on one record only would KeyError in _val_collate if it survived."""
    (key,) = extra
    ds = _dataset_with(tmp_path, extra_first=extra)
    batch = _val_collate(list(ds.input_data))
    assert key not in batch


def test_records_differing_only_in_shift_are_caught_as_duplicates(tmp_path):
    """The drop has to run before the duplicate check, not after.

    ``shift`` is stripped, so two records distinguished only by it are the same testcase. Dropping
    after the check would let them through and generate it twice.
    """
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _dataset_with(tmp_path, extra_first={"shift": 5.0}, extra_second={"shift": 7.0}, same_mask=True)


def test_mixed_num_steps_are_batched_apart_for_validation(tmp_path):
    """A batch mixing step counts would put the ranks on different timestep grids and desynchronise
    the CFG gate, so the grouping keys on num_steps as well as guidance."""
    jsonl = _jsonl_with_num_steps(tmp_path, 20, 35)
    ds = InpaintInferenceDataset(jsonl, default_guidance=DEFAULT_GUIDANCE, default_num_steps=DEFAULT_NUM_STEPS)
    batches = _build_val_batch_indices(ds.input_data, world_size=1, rank=0, batch_size=8)
    assert sorted(ds.input_data[b[0]]["num_steps"] for b in batches) == [20, 35]
    assert all(len({ds.input_data[i]["num_steps"] for i in b}) == 1 for b in batches)
