# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The weights catalogue: what anomalygen can load, where each one lands, and how to tell it is
# really there. Sourced, never executed -- it declares and defines, and does nothing.
#
# WHAT IT DOES NOT COVER
# The PROVISIONING side: this file, run_pipeline's weights preflight, and preflight_env_ckpt.sh --
# everything that downloads a weight or checks that one arrived. The loaders that open them still
# name their own paths (eval/fid.py, models/vision_encoder/dinov2.py, agentic_mask_generation/sam2.py,
# auto_mask_placement/roi_generation/model.py, eval/anomaly_quality.py, configs/texture/exp_config.py,
# pseudo_label/caption.py, inference/guardrail.py), and only the two base checkpoints are pinned to
# this table by a test. Editing a row therefore moves provisioning and both preflights and leaves the
# loader where it was, which surfaces hours later at load. Saying so here rather than claiming
# "every reader", which this file used to.
#
# WHY IT IS ITS OWN FILE
# Four places knew this and none of them owned it: download_checkpoints.sh (its group names, its
# output paths, its hub-cache probes), run_pipeline.py (_HF_REPO_FILES and required_weights),
# serving.py (a recipe's hf_probe) and preflight_env_ckpt.sh. Three docstrings warned about the
# drift, which is a good sign the structure was wrong, and the copies had already drifted twice:
#
#   run_pipeline derived its group name as `base-<model_size>` straight off the recipe field, so a
#   recipe saying `model_size: Nano` had prepare ask for `base-Nano` -- rejected as an unknown
#   group, 30 GB not provisioned, seconds after the preflight had named it as missing.
#
#   the guardrail skip guard tested `blocklist` alone while the entrypoint preflight demanded
#   `face_blur_filter/Resnet50_Final.pth` as well, so a download interrupted between the two left a
#   tree the provisioner reported as present and the preflight then refused to run against.
#
# Both are gone by construction now: there is one row per weight and every reader reads it.
#
# WHAT A ROW SAYS
#
#   group|kind|location|probe ...
#
# `group` is the download_checkpoints.sh block that provisions it, and the name a caller passes to
# ask for a subset. Several rows may share one -- the guardrail is two repositories.
#
# `kind` is what "present" means. Each reader implements the same three:
#
#   dir   a non-empty directory under CKPT_DIR
#   file  a file under CKPT_DIR
#   hf    a hub-cache entry under CKPT_DIR/hf, addressed by repo id rather than by path, so the
#         probes ARE the test
#
# `location` is relative to CKPT_DIR for dir and file, and is the repository id for hf.
#
# `probe` is what the loader actually opens. Never config.json: hf links each file into the snapshot
# as it completes, so the small metadata is present within seconds of a pull that may never finish,
# and a config-only probe green-lights an interrupted download. For dir and file the location is
# itself the probe and the field is empty.
#
# Revisions are deliberately NOT here. Which commit to fetch is the downloader's business, and two
# of them are read out of the framework's own registry at run time; what this file has to agree on
# with its readers is where a weight lands and how to recognise it.

WEIGHTS=(
  "base-nano|dir|Cosmos3-Nano|"
  "base-edge|dir|Cosmos3-Edge|"
  "dinov2|dir|facebook/dinov2-large|"
  "cradio|file|nvidia/C-RADIO-V3/model.safetensors|"
  "wan-vae|file|wan2pt2/Wan2.2_VAE.pth|"
  "sam2|file|facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt|"
  "cosmos-vlm|file|nvidia/Cosmos3-Nano/config.json|"
  "guardrail|hf|nvidia/Cosmos-Guardrail1|blocklist face_blur_filter/Resnet50_Final.pth"
  "guardrail|hf|Qwen/Qwen3Guard-Gen-0.6B|model.safetensors"
  "qwen-tokenizer|hf|Qwen/Qwen3-VL-8B-Instruct|tokenizer.json"
  "edge-processor|hf|nvidia/Cosmos3-Edge|processor_config.json"
  "gemma|hf|google/gemma-4-31b-it|model-00001-of-00002.safetensors model-00002-of-00002.safetensors"
)

# Derived, so the catalogue above is the only list to keep in order. Not named GROUPS: bash keeps a
# special array under that name holding the caller's group ids, and assigning to it is silently
# ignored -- the catalogue would read as "2524 1002 983" and reject every real group name.
KNOWN_GROUPS=()
for _w_row in "${WEIGHTS[@]}"; do
  IFS='|' read -r _w_group _ <<<"${_w_row}"
  [[ " ${KNOWN_GROUPS[*]} " == *" ${_w_group} "* ]] || KNOWN_GROUPS+=("${_w_group}")
done
unset _w_row _w_group

w_location() {  # w_location <group> -- the location of the single row in <group>
  local row g k loc found=""
  for row in "${WEIGHTS[@]}"; do
    IFS='|' read -r g k loc _ <<<"${row}"
    if [[ "${g}" == "$1" ]]; then
      # Refused rather than resolved: a caller asking a two-row group for "the" location wants one
      # answer, and silently returning the first would point half the readers at half the weights.
      [[ -z "${found}" ]] || { echo "ERROR: group '$1' has more than one catalogue row" >&2; return 1; }
      found="${loc}"
    fi
  done
  [[ -n "${found}" ]] || { echo "ERROR: no catalogue row for group '$1'" >&2; return 1; }
  printf '%s' "${found}"
}

w_repo() {  # w_repo <group> -- the repository id of a hub-cache group
  # Separate from w_location, and refusing a dir/file group, because the two are not the same kind of
  # string and reading one as the other is a mistake that looks like it works. For an `hf` row the
  # location IS the repository id; for the others it is a path under CKPT_DIR that may merely
  # resemble one -- cradio is the counterexample in this very table, `nvidia/C-RADIO-V3/...` on disk
  # against `nvidia/C-RADIOv3-B` on the Hub. Deriving a repo id with `dirname` of a location happened
  # to be right for three groups and would have been silently wrong for the next one added.
  local row g k loc
  for row in "${WEIGHTS[@]}"; do
    IFS='|' read -r g k loc _ <<<"${row}"
    if [[ "${g}" == "$1" ]]; then
      [[ "${k}" == hf ]] || { echo "ERROR: group '$1' is ${k}-addressed; it has a path, not a repo id" >&2; return 1; }
      printf '%s' "${loc}"
      return 0
    fi
  done
  echo "ERROR: no catalogue row for group '$1'" >&2
  return 1
}

w_probes() {  # w_probes <hf repo id> -- the probe files recorded for that hub-cache entry
  local row g k loc p
  for row in "${WEIGHTS[@]}"; do
    IFS='|' read -r g k loc p <<<"${row}"
    if [[ "${k}" == hf && "${loc}" == "$1" ]]; then printf '%s' "${p}"; return 0; fi
  done
  echo "ERROR: no hub-cache row in the catalogue for '$1'" >&2
  return 1
}
