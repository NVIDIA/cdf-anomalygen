# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-checkpoint selection in TrainingReport (pure Python; no model/GPU/distributed)."""

import os
from types import SimpleNamespace

from anomalygen.callbacks.training_report import TrainingReport

# Summary block written by ValidationKPI, followed by the blank line + per-sample section.
_CSV = """kpi,t+a,t+b,Average
fid,{fid_a},{fid_b},{fid}
mnn_score,{mnn_a},{mnn_b},{mnn}
nn_score,{nn_a},{nn_b},{nn}

per_sample
anomaly_type,sample,nn_score,mnn_score
t+a,reconstructed_image/t+a_00000.png,{nn_a},{mnn_a}
"""


def _write_valid(run_dir, iteration, nn, mnn=0.5, fid=100.0, nn_a=None):
    """Write valid/<iteration>/valid_kpi.csv. ``nn_a`` (column 1) defaults to a decoy that
    peaks at a different iteration than the Average column."""
    d = os.path.join(run_dir, "valid", str(iteration))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "valid_kpi.csv"), "w", newline="") as f:
        f.write(
            _CSV.format(
                nn=nn,
                mnn=mnn,
                fid=fid,
                nn_a=nn if nn_a is None else nn_a,
                nn_b=nn,
                mnn_a=mnn,
                mnn_b=mnn,
                fid_a=fid,
                fid_b=fid,
            )
        )


def _write_ckpt(run_dir, iteration):
    d = os.path.join(run_dir, "checkpoints", "model")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f"iter_{iteration:09}.pt"), "w").close()


def _callback(run_dir, **kwargs):
    cb = TrainingReport(**kwargs)
    cb.config = SimpleNamespace(job=SimpleNamespace(path_local=str(run_dir)), trainer=SimpleNamespace(max_iter=5000))
    return cb


def _pointer(run_dir):
    path = os.path.join(run_dir, "checkpoints", "best_checkpoint.txt")
    return open(path).read().strip() if os.path.isfile(path) else None


def test_picks_peak_not_latest(tmp_path):
    """The peak nn iteration wins even when a later iteration exists (the real runs all peaked mid-training)."""
    run = str(tmp_path)
    for it, nn in [(1000, 0.6498), (2000, 0.6622), (3000, 0.6494), (5000, 0.6579)]:
        _write_valid(run, it, nn)
        _write_ckpt(run, it)

    _callback(run)._write_best_checkpoint()
    assert _pointer(run) == "iter_000002000.pt"


def test_reads_average_not_first_defect_column(tmp_path):
    """Column 1 is the first defect type; selecting on it picks a different (wrong) iteration."""
    run = str(tmp_path)
    # Average peaks at 2000; the decoy column 1 peaks at 4000.
    for it, nn, nn_a in [(2000, 0.60, 0.10), (4000, 0.50, 0.99)]:
        _write_valid(run, it, nn, nn_a=nn_a)
        _write_ckpt(run, it)

    _callback(run)._write_best_checkpoint()
    assert _pointer(run) == "iter_000002000.pt"


def test_skips_iterations_without_a_checkpoint(tmp_path):
    """run_validation_on_start scores iteration 0, but no checkpoint is saved for it."""
    run = str(tmp_path)
    _write_valid(run, 0, 0.99)  # best score, but no iter_000000000.pt on disk
    _write_valid(run, 1000, 0.42)
    _write_ckpt(run, 1000)

    _callback(run)._write_best_checkpoint()
    assert _pointer(run) == "iter_000001000.pt"


def test_honors_min_direction_for_fid(tmp_path):
    """fid is lower-better; METRIC_SPECS supplies the direction."""
    run = str(tmp_path)
    for it, fid in [(1000, 900.0), (2000, 800.0), (3000, 850.0)]:
        _write_valid(run, it, nn=0.5, fid=fid)
        _write_ckpt(run, it)

    _callback(run, best_metric="fid")._write_best_checkpoint()
    assert _pointer(run) == "iter_000002000.pt"


def test_no_validation_data_writes_nothing(tmp_path):
    """A run with no valid/ tree must not crash or emit a dangling pointer."""
    run = str(tmp_path)
    _callback(run)._write_best_checkpoint()
    assert _pointer(run) is None


def test_no_checkpoints_writes_nothing(tmp_path):
    """Scores exist but no checkpoint was ever saved — emit nothing rather than a dangling path."""
    run = str(tmp_path)
    _write_valid(run, 1000, 0.5)
    _callback(run)._write_best_checkpoint()
    assert _pointer(run) is None


def test_scored_but_uncheckpointed_iterations_are_explained(tmp_path, loguru_lines):
    """The usual cause — validation_iter not a multiple of save_iter — is a recipe bug. Returning
    silently makes it indistinguishable from 'training never validated' at the train gate."""
    warnings = loguru_lines
    run = str(tmp_path)
    _write_valid(run, 1500, 0.5)  # validated at 1500...
    _write_ckpt(run, 2000)  # ...but only 2000 was checkpointed
    _callback(run)._write_best_checkpoint()
    assert _pointer(run) is None
    assert any("save_iter" in w for w in warnings), warnings


def test_no_validation_at_all_is_explained_without_blaming_save_iter(tmp_path, loguru_lines):
    """No validation ran. Still worth saying — a missing pointer silently demotes inference to
    latest_checkpoint.txt — but it must not blame save_iter, which is a different cause."""
    warnings = loguru_lines
    run = str(tmp_path)
    _callback(run)._write_best_checkpoint()
    assert _pointer(run) is None
    assert any("no validation results" in w for w in warnings), warnings
    assert not any("save_iter" in w for w in warnings), warnings


def test_metric_absent_from_validation_is_explained(tmp_path, loguru_lines):
    """Validation ran but never recorded best_metric (e.g. an axis this run does not score). The
    cause is the metric, not the cadence, so the message must not point at validation_iter."""
    warnings = loguru_lines
    run = str(tmp_path)
    _write_valid(run, 1000, 0.5)
    _write_ckpt(run, 1000)
    _callback(run, best_metric="completeness")._write_best_checkpoint()
    assert _pointer(run) is None
    assert any("none recorded 'completeness'" in w for w in warnings), warnings


def test_recipe_wires_best_metric_from_early_stop_metric(tmp_path):
    """best_checkpoint.txt must be selected on the metric the run was actually monitored on."""
    from anomalygen.configs.texture.exp_config import build_anomalygen_texture_ft_experiment

    exp = build_anomalygen_texture_ft_experiment(
        dataset_name="t",
        dataset_path=str(tmp_path),
        anomaly_types=[["t", "a"]],
        testcase_jsonl=str(tmp_path / "testcase.jsonl"),
        early_stop_metric="fid",
    )
    assert exp.trainer.callbacks["training_report"].best_metric == "fid"


# ---------------------------------------------------------------------------
# warm-up guard — early nn spikes must not win a long run
# ---------------------------------------------------------------------------
def test_warmup_excludes_early_peak_on_a_long_run(tmp_path):
    """nn swings early, and the pick is a plain best-of, so on a full-length run an early spike would
    otherwise beat a settled later checkpoint."""
    run = str(tmp_path)
    for it, nn in [(2000, 0.90), (8000, 0.70), (12000, 0.75)]:
        _write_valid(run, it, nn)
        _write_ckpt(run, it)

    cb = _callback(run)
    cb.config.trainer.max_iter = 15000
    cb._write_best_checkpoint()

    assert _pointer(run) == "iter_000012000.pt"  # best at/after the warm-up, not the 2000 spike


def test_warmup_not_applied_to_a_short_run(tmp_path):
    """A run planned at or below the warm-up (dry runs, quick smoke tests) must still get a pointer,
    picked from everything it produced."""
    run = str(tmp_path)
    for it, nn in [(1000, 0.90), (4000, 0.70)]:
        _write_valid(run, it, nn)
        _write_ckpt(run, it)

    cb = _callback(run)
    cb.config.trainer.max_iter = 5000
    cb._write_best_checkpoint()

    assert _pointer(run) == "iter_000001000.pt"


def test_warmup_falls_back_when_nothing_reaches_it(tmp_path, loguru_lines):
    """Early stopping (or a crash) can end a long run before the warm-up. Writing no pointer at all
    would silently demote inference to latest_checkpoint.txt, so fall back and say so."""
    run = str(tmp_path)
    for it, nn in [(1000, 0.60), (3000, 0.80)]:
        _write_valid(run, it, nn)
        _write_ckpt(run, it)

    cb = _callback(run)
    cb.config.trainer.max_iter = 15000
    cb._write_best_checkpoint()

    assert _pointer(run) == "iter_000003000.pt"
    assert any("warm-up" in w for w in loguru_lines), loguru_lines
