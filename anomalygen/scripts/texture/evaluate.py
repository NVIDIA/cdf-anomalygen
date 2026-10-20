# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone KPI harness over SDG output: score NN/MNN + FID, report per-type + Average.

Consumes the layouts produced by ``generate.py`` and used by the in-training ``ValidationKPI``
(no reshaping), around :func:`anomalygen.eval.correspondence.compute_correspondence_kpi`:

  --gen_root   generate.py output:      {gen}/reconstructed_image/{texture}+{defect}_{idx}.png
                                        {gen}/original_mask/{texture}+{defect}_{idx}.png
  --real_root  training dataset root:   {texture}/anomaly_image/{defect}/<stem>.png
                                        {texture}/mask/{defect}/<stem>_mask.png

Anomaly types to score come from --anomaly_types, else --recipe's ``anomaly_types``, else are
inferred from the generated filenames.

The SDG-output loaders live in :mod:`anomalygen.eval.utils` (shared with ``filter.py``).
"""

from __future__ import annotations

import argparse
import json
import os

# Framework process setup (inference env, grad disabled, distributed init when WORLD_SIZE>1).
from cosmos_framework.inference.common.init import init_script
from cosmos_framework.utils import log

# Guarded because it disables autograd process-wide; importing this module must leave it alone.
if __name__ == "__main__":
    init_script(training=False)

from anomalygen.eval.anomaly_quality import augment_with_quality, compute_anomaly_quality_kpi
from anomalygen.eval.correspondence import add_nn_scoring_args, compute_correspondence_kpi, nn_scoring_kwargs
from anomalygen.eval.fid import compute_fid_kpi
from anomalygen.eval.utils import RECON_SUBDIR, load_generated, load_real, resolve_anomaly_types


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="AnomalyGen KPI over SDG output: NN/MNN (+FID) correspondence plus the "
        "anomaly-quality axes (completeness / precision / boundary_iou) and the aq_nn composite — "
        "the same metrics training validation records."
    )
    parser.add_argument(
        "--gen_root", required=True, help="generate.py output dir (reconstructed_image/ + original_mask/)"
    )
    parser.add_argument(
        "--real_root", required=True, help="real dataset root ({texture}/anomaly_image/{defect} + mask/{defect})"
    )
    parser.add_argument(
        "--anomaly_types",
        nargs="+",
        default=None,
        help="keys 'texture+defect' to score; default: derive from --recipe, else infer from --gen_root",
    )
    parser.add_argument(
        "--recipe", default=None, help="recipe YAML/JSON to derive anomaly_types from its 'anomaly_types'"
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="best-matching real refs to average per sample (-1 = use all)"
    )
    parser.add_argument(
        "--fid_crop_size",
        type=int,
        default=512,
        help="resolution the FID backbone (C-RADIO-V3) runs at; smaller trades fidelity for speed",
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=None,
        help="resize loaded images/masks to this square side; default None keeps native resolution",
    )
    parser.add_argument("--output_file", default="kpi.json", help="output KPI JSON file path")
    add_nn_scoring_args(parser)  # nn/mnn knobs (default = validated zoom/12/worst25/min; override to restore old)
    args = parser.parse_args(argv)

    recon_dir = os.path.join(args.gen_root, RECON_SUBDIR)
    anomaly_types = resolve_anomaly_types(args.anomaly_types, args.recipe, recon_dir)

    # with_original_image is required, not optional: the quality axes diff the render against the
    # pre-edit clean image, and compute_anomaly_quality_kpi zips the four lists together — without it
    # ``original_image`` is empty, the zip yields nothing, and every run silently scores NN/MNN only.
    generated = load_generated(
        args.gen_root, anomaly_types, target_size=args.model_input_size, with_original_image=True
    )
    if not generated:
        raise RuntimeError(f"No generated images found under {recon_dir} for {anomaly_types}.")
    # Only score (and require real refs for) types that actually have generated images.
    real = load_real(args.real_root, list(generated.keys()), target_size=args.model_input_size)

    kpi = compute_correspondence_kpi(real, generated, top_k=args.top_k, **nn_scoring_kwargs(args))
    # FID (C-RADIO backbone) is heavier and can fail — it needs the checkpoint and >=2 defect crops
    # per type — so compute it but skip gracefully on error, mirroring the in-training ValidationKPI.
    try:
        for name, vals in compute_fid_kpi(real, generated, crop_size=args.fid_crop_size).items():
            kpi.setdefault(name, {}).update(vals)
    except Exception as e:
        log.warning(f"FID computation failed ({e}); writing the KPI without FID.")

    # Anomaly-quality axes, always scored so a Step 5 KPI carries the same metrics training validation
    # records and downstream rankers never have to recompute them. It loads SAM2, so like FID above a
    # failure degrades to NN/MNN(+FID) rather than sinking the whole evaluation.
    try:
        aq = compute_anomaly_quality_kpi(real, generated, kpi)
    except Exception as e:
        log.warning(f"anomaly_quality computation failed ({e}); writing the KPI without the quality axes.")
        aq = {}
    if aq:
        # Per-sample first: this folds each axis (+ aq_nn_score / bucket-local aq_rank_score) into the
        # existing rows, so downstream rankers read one row per sample rather than joining twice.
        # quality_refine's aq_rank mode ignores that local rank and recomputes from these raw inputs
        # over Original + observed rounds; filter.py intentionally ranks this one bucket as written.
        augment_with_quality(kpi, aq)
        for name, vals in aq.items():
            # per_sample_axes is already merged into per_sample above; keeping it would duplicate
            # every axis value in the JSON.
            kpi.setdefault(name, {}).update({k: v for k, v in vals.items() if k != "per_sample_axes"})

    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(kpi, f, indent=2)
    log.info(f"Scored {len(generated)} anomaly type(s); wrote {args.output_file}.")
    log.info(f"Average KPI:\n{json.dumps(kpi.get('Average', {}), indent=2)}")


if __name__ == "__main__":
    main()
