#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the checkpoints anomalygen needs into ./checkpoints :
#   - base models -> checkpoints/Cosmos3-Nano, checkpoints/Cosmos3-Edge (DCP, framework converter)
#   - DINOv2      -> checkpoints/facebook/dinov2-large    (KPI correspondence backbone, NN/MNN)
#   - C-RADIO v3  -> checkpoints/nvidia/C-RADIO-V3        (KPI FID backbone)
#   - SAM2.1      -> checkpoints/facebook/sam2.1-hiera-large  (ROI-generation segmentation)
#   - Cosmos3-Nano-> checkpoints/nvidia/Cosmos3-Nano          (text2roi grounding VLM, default)
#   - Cosmos-Guardrail1 -> checkpoints/hf         (generate.py guardrail: blocklist + face-blur)
#   - Qwen3Guard-Gen-0.6B -> checkpoints/hf       (generate.py guardrail: text safety LLM)
#   - Qwen3-VL-8B-Instruct tokenizer -> checkpoints/hf   (nano caption tokenizer)
#   - Cosmos3-Edge processor      -> checkpoints/hf   (edge caption processor)
#   - Gemma (google/gemma-4-31b-it) -> checkpoints/hf (AMG local-serving VLM)
#
# Prerequisites: the uv environment is active (see README.md "Installation" / "Usage")
# and you are authenticated to Hugging Face (HF_TOKEN exported or `uvx hf auth login`),
# with both base model licenses accepted on their HF pages.
#
# Idempotent: existing outputs are skipped.
#
# With no arguments every group above is fetched -- what the README, the skills and the image build
# all invoke. Naming groups narrows it to those (`--list` prints the names):
#
#   bash scripts/download_checkpoints.sh base-edge wan-vae edge-processor guardrail
#
# Narrowing is opt-in, never inferred: which weights a run opens is the caller's knowledge, and a
# script that guessed would leave someone's pipeline missing a checkpoint at minute forty.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"

# The weights catalogue: the group names below, the output paths and the hub-cache probes all come
# from it, and so do run_pipeline's preflight and preflight_env_ckpt.sh. See its header for why.
# shellcheck source=scripts/weights_catalogue.sh
source "${REPO_ROOT}/scripts/weights_catalogue.sh"

# The catalogue locations this file needs, resolved up front so every later use is a string.
# Resolved into variables of their own rather than inline: wrapping the lookup --
# `X="$(dirname "$(w_location sam2)")"` -- defeats `set -e`, because w_location fails and prints but
# dirname is handed the empty string, answers ".", and exits 0. The assignment then succeeds and a
# typo'd group name becomes `hf download .` rather than an error. Stripping with ${var%/*} below
# also drops two subprocesses per path.
DINOV2_LOCATION="$(w_location dinov2)"
CRADIO_LOCATION="$(w_location cradio)"
WAN_VAE_LOCATION="$(w_location wan-vae)"
SAM2_LOCATION="$(w_location sam2)"
COSMOS_VLM_LOCATION="$(w_location cosmos-vlm)"


# ---- group selection ---------------------------------------------------------
# KNOWN_GROUPS is the catalogue's, one name per download block below. A `cad_tht` generate run opens
# neither Gemma (62.6 GB) nor the KPI backbones nor one of the two base checkpoints -- better than
# half the ≈ 140 GB tree -- and making the smallest useful task pay for the largest one is the cost
# this exists to remove.
SELECTED=()
for arg in "$@"; do
  case "${arg}" in
    --list) printf '%s\n' "${KNOWN_GROUPS[@]}"; exit 0 ;;
    -h|--help)
      echo "usage: $(basename "$0") [group ...]   (no arguments: every group)"
      printf '  %s\n' "${KNOWN_GROUPS[@]}"
      exit 0 ;;
    -*)
      # Separated from the group check below so the message is true: an unknown FLAG reported as an
      # "unknown group" sends the reader to the catalogue to look for something that was never there.
      echo "ERROR: unknown option '${arg}'. The only options are --list and --help; every other" >&2
      echo "argument is a group name ($(basename "$0") --list)." >&2
      exit 1 ;;
    *) ;;  # a group name: checked against the catalogue below, which is where the catalogue is
  esac
  # Reject an unknown name rather than fetching nothing under it: a typo would otherwise read as a
  # successful run that provisioned less than asked, and be found only when a load fails.
  if [[ " ${KNOWN_GROUPS[*]} " != *" ${arg} "* ]]; then
    echo "ERROR: unknown group '${arg}'. Known groups:" >&2
    printf '  %s\n' "${KNOWN_GROUPS[@]}" >&2
    exit 1
  fi
  SELECTED+=("${arg}")
done

want() {  # want <group> -- true when this run should fetch <group>; no selection means all of them
  if [[ ${#SELECTED[@]} -eq 0 ]]; then return 0; fi
  local g
  for g in "${SELECTED[@]}"; do
    if [[ "${g}" == "$1" ]]; then return 0; fi
  done
  return 1
}

BASE_CHECKPOINT_NAMES=()
if want base-nano; then BASE_CHECKPOINT_NAMES+=(Cosmos3-Nano); fi
if want base-edge; then BASE_CHECKPOINT_NAMES+=(Cosmos3-Edge); fi
# Edge's framework default is main; use the revision the converted manifest was recorded against.
# Bump the two together: the Edge digests in assets/checkpoint_manifest_converted.sha256 are this
# revision's generator weights run through the converter.
EDGE_BASE_REV="a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
# The downloader is pinned too, not just what it downloads: `hf@latest` would resolve and run the
# newest PyPI release on every invocation, with HF_TOKEN in the environment and write access to the
# manifests below — an unverified executable fronting every pin in this file. 1.26.x is pinned on
# its own (requirements.txt pins huggingface-hub separately) and goes with the Edge trees/ removal
# below; bump them deliberately.
HF_CLI="hf==${HF_CLI_VERSION:-1.26.0}"
DINOV2_REPO="${DINOV2_REPO:-facebook/dinov2-large}"
# Pinned like every other repo below: an unpinned "main" means a later run can fetch different
# bytes for the same command, and the digest recorded in assets/checkpoint_manifest.sha256 would
# then be the only thing that noticed. Bump the revision and the manifest together.
DINOV2_REV="${DINOV2_REV:-47b73eefe95e8d44ec3623f8890bd894b6ea2d6c}"
# C-RADIO v3 ViT-B FID backbone. The HF repo is nvidia/C-RADIOv3-B but the code resolves the
# checkpoint under checkpoints/nvidia/C-RADIO-V3 (matching upstream), so download just the weight
# file into that fixed directory.
CRADIO_REPO="${CRADIO_REPO:-nvidia/C-RADIOv3-B}"
CRADIO_FILE="${CRADIO_LOCATION##*/}"
CRADIO_REV="${CRADIO_REV:-44653a0482cf460bb4f12595fc3cc3dfecc403d1}"
# Wan2.2 VAE the tokenizer encodes/decodes with. Shipped inside the Wan2.2-TI2V-5B repo.
WAN_VAE_REPO="${WAN_VAE_REPO:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_VAE_FILE="${WAN_VAE_LOCATION##*/}"
WAN_VAE_REV="${WAN_VAE_REV:-921dbaf3f1674a56f47e83fb80a34bac8a8f203e}"
# SAM2.1 hiera-large weights for ROI-generation segmentation. The code resolves the checkpoint
# under checkpoints/facebook/sam2.1-hiera-large, so download just the weight file into that dir.
SAM2_REPO="${SAM2_REPO:-facebook/sam2.1-hiera-large}"
SAM2_FILE="${SAM2_LOCATION##*/}"
SAM2_REV="${SAM2_REV:-665f8e2ad61cf5f53d65644ff27c8ee525124610}"
# text2roi grounding VLMs (HF transformers format), resolved from checkpoints/<org>/<name> by
# anomalygen.auto_mask_placement.text2roi.Text2BoxDetector. Cosmos3-Nano is the default.
# For Cosmos3-Nano the reasoner shares the transformer/ + vision_encoder/ weights, so the
# generation-only vae/sound_tokenizer/scheduler dirs are skipped.
COSMOS_VLM_REPO="${COSMOS_VLM_REPO:-nvidia/Cosmos3-Nano}"
# Pinned like every other repo here, and doubly so: the Hub manifest below records this model's
# shards, so an unpinned fetch would move the snapshot out from under the recorded digests.
COSMOS_VLM_REV="${COSMOS_VLM_REV:-411f42a8fdfb8c5b2583cb8786e0938f49796eaa}"
# Content-safety guardrail models used by anomalygen/scripts/texture/generate.py.
# The framework resolves these from the HF cache at runtime, so we only warm the cache below rather
# than writing into checkpoints/.
# Named here rather than read with w_location: the guardrail group holds two rows, and asking it
# for "the" location is refused. Both ids are still checked against the catalogue -- hf_cached
# looks their probes up by repo id and fails for one it does not carry.
GUARDRAIL_REPO="nvidia/Cosmos-Guardrail1"
QWEN_GUARD_REPO="Qwen/Qwen3Guard-Gen-0.6B"
# The framework loads Qwen3Guard unpinned via from_pretrained() at runtime, so pinning here makes only
# the *download* deterministic; full runtime reproducibility also needs an upstream pin. Pinned to the
# repo HEAD at authoring time; override QWEN_GUARD_REV to bump.
QWEN_GUARD_REV="${QWEN_GUARD_REV:-fada3b2f655b89601929198343c94cd2f64d93cc}"
# Caption tokenizer for training. NANO_MODEL_CONFIG wires vlm_config.tokenizer to
# create_qwen2_tokenizer_with_download(pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
# config_variant="hf"), which resolves straight from the HF cache at model-construction time.
# Without it a run with no Hub access dies before iteration 1 (TypeError: expected str, bytes or
# os.PathLike object, not NoneType — transformers' cached_file returns None when offline), so the
# air-gapped images need it baked in. Only *.json/*.txt: the 8B weights (~17 GB) are never loaded
# here, the model itself comes from checkpoints/Cosmos3-Nano.
QWEN_VLM_REPO="$(w_repo qwen-tokenizer)"
# Caption processor for model_size: edge — the Edge counterpart of the Qwen tokenizer above.
# EDGE_MODEL_CONFIG resolves it through CheckpointDirHf at model-build time, so without this
# prefetch every rank pulls 8.6 GB, and an air-gapped run fails inside load_for_inference.
EDGE_VLM_REPO="$(w_repo edge-processor)"
# Gemma VLM that scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh serves for AMG (Step 2 mask placement).
# vLLM resolves it from the HF hub cache by repo id — the recipe defaults HF_HUB_CACHE to
# checkpoints/hf — so this is a hub-cache entry like the guardrails below. apache-2.0, ungated on
# HF: no licence acceptance or token needed, unlike the base models. Pinned to one revision so
# every install serves the same weights; override GEMMA_REV to bump — and bump the Hub manifest
# with it, like DINOV2_REV above: its gemma digests live under snapshots/<rev>.
GEMMA_REPO="$(w_repo gemma)"
GEMMA_REV="${GEMMA_REV:-842da3794eaa0b77d5f08bae87a17459d91ff475}"

# Every output path is the catalogue's, so the directory this writes to is by construction the one
# run_pipeline's preflight and preflight_env_ckpt.sh look in.
#
# Each location is resolved into a variable of its own before anything is done to it. Wrapping the
# lookup -- `X="$(dirname "$(w_location sam2)")"` -- defeats `set -e`: w_location fails and prints,
# but dirname is handed the empty string, answers ".", and exits 0, so the assignment succeeds and a
# typo'd group name becomes `hf download .` instead of an error. Stripping with ${var%/*} also drops
# the two subprocesses.
DINOV2_OUT="${CKPT_DIR}/${DINOV2_LOCATION}"
CRADIO_OUT="${CKPT_DIR}/${CRADIO_LOCATION%/*}"
WAN_VAE_OUT="${CKPT_DIR}/${WAN_VAE_LOCATION}"
SAM2_OUT="${CKPT_DIR}/${SAM2_LOCATION%/*}"
COSMOS_VLM_OUT="${CKPT_DIR}/${COSMOS_VLM_LOCATION%/*}"

mkdir -p "${CKPT_DIR}"

# ---- base models (download + convert to DCP) ---------------------------------
if ! python -c "import cosmos_framework" 2>/dev/null; then
  echo "ERROR: cosmos_framework not importable. Activate the venv first:" >&2
  echo "  source ${REPO_ROOT}/.venv/bin/activate && export LD_LIBRARY_PATH=" >&2
  exit 1
fi

# The converter pulls each repo from the Hub before writing the DCP, deliberately BEFORE the
# HF_HUB_CACHE export below, so those one-time source snapshots land in the user's default cache
# rather than bloating checkpoints/hf (the runtime cache the air-gapped images bake in). The DCP
# holds tensors only — no tokenizer — and this loop is skipped when it already exists, so the
# processor prefetch at the end cannot rely on this pull.
#
# Both sizes by default — a checkout that can only run one is half-installed, and a
# `model_size: edge` recipe would otherwise fail deep inside load_for_inference instead of here.
# Selecting one size is for a caller that knows its recipe, not for a general checkout.
# A failure is collected rather than fatal so one unaccepted licence does not also take out the
# ungated DINOv2 / C-RADIO / SAM2 / guardrail pulls below and read as eight broken downloads.
failed_sizes=()
convert_base_checkpoint() {
  local base_name="$1" base_out="$2" checkpoint_path="$1"
  if [[ "${base_name}" == Cosmos3-Edge ]]; then
    checkpoint_path="$(uvx "${HF_CLI}" download --format=json --repo-type model "${EDGE_VLM_REPO}" \
      --revision "${EDGE_BASE_REV}" | python -c 'import json, sys; print(json.load(sys.stdin)["path"])')" || return "$?"
  fi
  python -m cosmos_framework.scripts.convert_model_to_dcp \
    -o "${base_out}" \
    --checkpoint-path "${checkpoint_path}"
}

# CONVERT ASIDE, THEN RENAME. The presence check below is "the directory exists and is not empty",
# so a conversion that stops midway leaves a tree that every later run reports as already present
# while load_for_inference fails on the missing .metadata -- with nothing in this script to repair
# it. The rm -rf on the failure branch handles a converter that EXITS non-zero; it cannot handle the
# process being killed, which is what a cancelled orchestrator task does. Converting into a scratch
# directory and renaming means the real path is only ever absent or complete: rename(2) is atomic,
# and both paths are under CKPT_DIR so it stays within one filesystem.
#
# The scratch name is fixed rather than mktemp so an interrupted run's leftovers are found and
# removed by the next one instead of accumulating 30 GB at a time.
for base_name in "${BASE_CHECKPOINT_NAMES[@]}"; do
  base_out="${CKPT_DIR}/${base_name}"
  base_tmp="${CKPT_DIR}/.${base_name}.converting"
  if [[ -d "${base_out}" ]] && [[ -n "$(ls -A "${base_out}" 2>/dev/null)" ]]; then
    echo "[skip] base checkpoint already present: ${base_out}"
  else
    echo "[download+convert] ${base_name} -> ${base_out}"
    rm -rf "${base_tmp}"
    if convert_base_checkpoint "${base_name}" "${base_tmp}"; then
      rm -rf "${base_out}"
      mv "${base_tmp}" "${base_out}"
    else
      echo "[FAIL] ${base_name} — continuing with the remaining downloads" >&2
      failed_sizes+=("${base_name}")
      rm -rf "${base_tmp}"
    fi
  fi
done

# ---- DINOv2 (KPI backbone) ---------------------------------------------------
if ! want dinov2; then
  echo "[skip] DINOv2 — not selected"
elif [[ -d "${DINOV2_OUT}" ]] && [[ -n "$(ls -A "${DINOV2_OUT}" 2>/dev/null)" ]]; then
  echo "[skip] DINOv2 already present: ${DINOV2_OUT}"
else
  echo "[download] ${DINOV2_REPO} -> ${DINOV2_OUT}"
  uvx "${HF_CLI}" download --repo-type model "${DINOV2_REPO}" --revision "${DINOV2_REV}" --local-dir "${DINOV2_OUT}"
fi

# ---- C-RADIO v3 (KPI FID backbone) -------------------------------------------
if ! want cradio; then
  echo "[skip] C-RADIO v3 — not selected"
elif [[ -f "${CRADIO_OUT}/${CRADIO_FILE}" ]]; then
  echo "[skip] C-RADIO v3 already present: ${CRADIO_OUT}/${CRADIO_FILE}"
else
  echo "[download] ${CRADIO_REPO}:${CRADIO_FILE} -> ${CRADIO_OUT}"
  mkdir -p "${CRADIO_OUT}"
  uvx "${HF_CLI}" download --repo-type model "${CRADIO_REPO}" "${CRADIO_FILE}" --revision "${CRADIO_REV}" \
    --local-dir "${CRADIO_OUT}"
fi

# ---- Wan2.2 VAE (tokenizer) --------------------------------------------------
if ! want wan-vae; then
  echo "[skip] Wan2.2 VAE — not selected"
elif [[ -f "${WAN_VAE_OUT}" ]]; then
  echo "[skip] Wan2.2 VAE already present: ${WAN_VAE_OUT}"
else
  echo "[download] ${WAN_VAE_REPO}:${WAN_VAE_FILE} -> ${WAN_VAE_OUT}"
  mkdir -p "$(dirname "${WAN_VAE_OUT}")"
  uvx "${HF_CLI}" download --repo-type model "${WAN_VAE_REPO}" "${WAN_VAE_FILE}" --revision "${WAN_VAE_REV}" \
    --local-dir "$(dirname "${WAN_VAE_OUT}")"
fi

# ---- SAM2.1 hiera-large (ROI-generation segmentation) ------------------------
if ! want sam2; then
  echo "[skip] SAM2.1 — not selected"
elif [[ -f "${SAM2_OUT}/${SAM2_FILE}" ]]; then
  echo "[skip] SAM2.1 already present: ${SAM2_OUT}/${SAM2_FILE}"
else
  echo "[download] ${SAM2_REPO}:${SAM2_FILE} -> ${SAM2_OUT}"
  mkdir -p "${SAM2_OUT}"
  uvx "${HF_CLI}" download --repo-type model "${SAM2_REPO}" "${SAM2_FILE}" --revision "${SAM2_REV}" \
    --local-dir "${SAM2_OUT}"
fi

# ---- Cosmos3-Nano (reasoner only) ---
if ! want cosmos-vlm; then
  echo "[skip] Cosmos3-Nano VLM — not selected"
elif [[ -f "${COSMOS_VLM_OUT}/config.json" ]]; then
  echo "[skip] Cosmos3-Nano VLM already present: ${COSMOS_VLM_OUT}"
else
  echo "[download] ${COSMOS_VLM_REPO}@${COSMOS_VLM_REV} -> ${COSMOS_VLM_OUT} (~30GB; excludes generation-only vae/sound_tokenizer/scheduler)"
  uvx "${HF_CLI}" download --repo-type model "${COSMOS_VLM_REPO}" --revision "${COSMOS_VLM_REV}" \
    --local-dir "${COSMOS_VLM_OUT}" \
    --exclude "sound_tokenizer/*" --exclude "vae/*" --exclude "scheduler/*" --exclude "assets/*" --exclude "images/*"
fi

# ---- hub cache (shared by the four hub-cache groups below) -------------------
# Lifted out of the guardrail block that used to own it, and deliberately no further: all four
# groups below read it, so leaving it inside one would make `download_checkpoints.sh gemma` resolve
# against the user's default cache -- but exporting it any earlier would redirect the --local-dir
# downloads above too. Its position relative to those and to the base conversion is unchanged.
# These are resolved from the HF hub cache at runtime (the classes give no local-dir option), so we
# point that cache under ${CKPT_DIR}/hf to keep the checkout self-contained.
export HF_HUB_CACHE="${CKPT_DIR}/hf"

# Skip guard for the four hub-cache entries. The six --local-dir entries above test an output path;
# these land as content-addressed blobs, so the equivalent test is "does this revision's snapshot
# hold a file the loader actually reads" — the same probe preflight_env_ckpt.sh checks for.
#
# Guards the download ONLY. The refs/main writes and the trees/ removal below stay unconditional:
# they are what make an offline resolve work, they are what preflight asserts, and skipping them on
# a warm cache would silently re-break air-gapped runs. They cost a 40-byte write and an rm.
# The revision may be a sha (snapshots/<sha>) or a branch name, which hf records as refs/<branch>
# -> sha rather than a directory. Resolve the ref when the direct path is absent, like preflight.
# Sets HF_SNAPSHOT_DIR to the snapshot it matched, so the skip message names the directory that was
# actually checked rather than the shared cache root — all four entries live under the same root, so
# printing that alone says nothing about which one was found.
hf_snapshot_has() {  # hf_snapshot_has <repo id> <revision> <probe path in snapshot>
  local dir="${HF_HUB_CACHE}/models--${1//\//--}" sha="$2"
  HF_SNAPSHOT_DIR=""
  if [[ ! -e "${dir}/snapshots/${sha}/${3}" ]]; then
    [[ -s "${dir}/refs/${2}" ]] || return 1
    sha="$(cat "${dir}/refs/${2}")"
    [[ -e "${dir}/snapshots/${sha}/${3}" ]] || return 1
  fi
  HF_SNAPSHOT_DIR="${dir}/snapshots/${sha}"
}

hf_cached() {  # hf_cached <repo id> <revision> -- true when EVERY catalogue probe is in its snapshot
  # Which files count is the catalogue's answer, not this block's. Asked here they drifted: the
  # guardrail guard tested `blocklist` alone while run_pipeline's preflight also wanted
  # face_blur_filter/Resnet50_Final.pth, so a pull interrupted between the two left a tree this
  # script skipped as present and the entrypoint then refused to run against, with no way forward
  # short of deleting the cache by hand. w_probes fails loudly for a repo the catalogue does not
  # know, which is what keeps the repo ids above from drifting away from it.
  # A repo the catalogue does not carry is FATAL, not "not cached". Returning 1 made it look like a
  # cold cache, so an id this file still lets the environment override -- a mirror or a fork -- would
  # re-download on every invocation and break the air-gapped re-run that used to skip.
  local repo="$1" rev="$2" probe probes
  if ! probes="$(w_probes "${repo}")"; then
    echo "ERROR: ${repo} is not in scripts/weights_catalogue.sh, so there is nothing that says what" >&2
    echo "       a complete download of it looks like. Add a row for it." >&2
    exit 1
  fi
  for probe in ${probes}; do
    hf_snapshot_has "${repo}" "${rev}" "${probe}" || return 1
  done
  return 0
}

# ---- Cosmos-Guardrail1 + Qwen3Guard (content-safety guardrail for generate.py) ----
# Read the framework's pinned revision (single source of truth — no hardcoded hash, no drift), then
# download with hf like every other entry above.
if ! want guardrail; then
  echo "[skip] Cosmos-Guardrail1 + Qwen3Guard — not selected"
else
  GUARDRAIL_REV="${GUARDRAIL_REV:-$(python -c "from cosmos_framework.auxiliary.guardrail.common.core import GUARDRAIL1_CHECKPOINT as c; print(c.revision)")}"
  if hf_cached "${GUARDRAIL_REPO}" "${GUARDRAIL_REV}"; then
    echo "[skip] Cosmos-Guardrail1 already present: ${HF_SNAPSHOT_DIR}"
  else
    echo "[download] ${GUARDRAIL_REPO}@${GUARDRAIL_REV} -> ${HF_HUB_CACHE} (blocklist word-lists + RetinaFace face-blur weights)"
    uvx "${HF_CLI}" download --repo-type model "${GUARDRAIL_REPO}" --revision "${GUARDRAIL_REV}"
  fi

  if hf_cached "${QWEN_GUARD_REPO}" "${QWEN_GUARD_REV}"; then
    echo "[skip] Qwen3Guard already present: ${HF_SNAPSHOT_DIR}"
  else
    echo "[download] ${QWEN_GUARD_REPO}@${QWEN_GUARD_REV} -> ${HF_HUB_CACHE} (text-safety LLM)"
    uvx "${HF_CLI}" download --repo-type model "${QWEN_GUARD_REPO}" --revision "${QWEN_GUARD_REV}"
  fi
  # Qwen3Guard.__init__ calls from_pretrained() with no revision, so an offline resolve goes through
  # refs/main — which `hf download --revision <sha>` does not write. Without this the guardrail is
  # unusable air-gapped despite the snapshot being present. Online runs rewrite it anyway.
  QWEN_GUARD_CACHE_DIR="${HF_HUB_CACHE}/models--${QWEN_GUARD_REPO//\//--}"
  mkdir -p "${QWEN_GUARD_CACHE_DIR}/refs"
  printf '%s' "${QWEN_GUARD_REV}" > "${QWEN_GUARD_CACHE_DIR}/refs/main"
fi

# ---- Qwen3-VL-8B-Instruct tokenizer (training caption tokenizer) ----
# Revision comes from the framework's own registry rather than a hardcoded hash, so this cannot
# drift from the pin create_qwen2_tokenizer_with_download resolves. _CHECKPOINTS is private; if an
# upstream refactor moves it this fails loudly here instead of silently fetching a different commit.
if ! want qwen-tokenizer; then
  echo "[skip] Qwen3-VL tokenizer — not selected"
else
  QWEN_VLM_REV="${QWEN_VLM_REV:-$(python -c "
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS
register_checkpoints()
print(next(c.hf.revision for c in _CHECKPOINTS.values()
           if getattr(c, 'hf', None) and c.hf.repository == '${QWEN_VLM_REPO}'))")}"
  if hf_cached "${QWEN_VLM_REPO}" "${QWEN_VLM_REV}"; then
    echo "[skip] Qwen3-VL tokenizer already present: ${HF_SNAPSHOT_DIR}"
  else
    echo "[download] ${QWEN_VLM_REPO}@${QWEN_VLM_REV} -> ${HF_HUB_CACHE} (training caption tokenizer, *.json/*.txt only)"
    uvx "${HF_CLI}" download --repo-type model "${QWEN_VLM_REPO}" --revision "${QWEN_VLM_REV}" \
      --include "*.json" --include "*.txt"
  fi
  # create_qwen2_tokenizer_with_download calls Qwen2Tokenizer.from_pretrained(<repo id>) with no
  # revision, so an offline resolve maps "main" -> sha through refs/main. `hf download --revision
  # <sha>` writes the snapshot but never that ref, so without this line the cache is unusable offline:
  # every file resolves to None and tokenization_qwen2.py does open(None). Pin + ref, not one or the
  # other. 40 bytes, no trailing newline — matches what huggingface_hub itself writes.
  QWEN_VLM_CACHE_DIR="${HF_HUB_CACHE}/models--${QWEN_VLM_REPO//\//--}"
  mkdir -p "${QWEN_VLM_CACHE_DIR}/refs"
  printf '%s' "${QWEN_VLM_REV}" > "${QWEN_VLM_CACHE_DIR}/refs/main"
fi

# ---- Cosmos3-Edge processor (model_size: edge caption processor) ----
# Repo id and revision are read from the model config, not hardcoded, so this cannot drift from what
# build_processor_lazy asks for; an upstream move fails loudly here instead of warming the wrong entry.
#
# The revision it gives is `main` -- the same unpinned branch the base checkpoint was on until
# EDGE_BASE_REV pinned it, and the same failure waiting to happen, only quieter. These ~28 MB of
# config, tokenizer and chat template are in no integrity manifest, so an upstream edit to the chat
# template or the tokenizer would change every caption this model produces and nothing would report
# anything. (The 2026-08-26 move that broke the converted-weight digests did NOT touch them: every
# file the processor reads is byte-identical across it. The exposure is real, it just has not fired.)
#
# So a branch is replaced with EDGE_BASE_REV, which makes the processor and the DCP it feeds one
# commit. A sha is followed as given: the runtime resolves whatever this config says, so overriding
# a real pin would download one revision and load another.
if ! want edge-processor; then
  echo "[skip] Cosmos3-Edge processor — not selected"
else
  EDGE_VLM_REV="${EDGE_VLM_REV:-$(python -c "
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
tok = EDGE_MODEL_CONFIG['vlm_config']['tokenizer']
assert tok['repository'] == '${EDGE_VLM_REPO}', tok['repository']
print(tok['revision'])")}"
  EDGE_VLM_REF=""  # the branch the runtime resolves through, when it is not asking for a sha
  if [[ ! "${EDGE_VLM_REV}" =~ ^[0-9a-f]{40}$ ]]; then
    EDGE_VLM_REF="${EDGE_VLM_REV}"
    EDGE_VLM_REV="${EDGE_BASE_REV}"
  fi
  # ~28 MB of an 8.6 GB repo: the processor reads only the config / tokenizer / chat-template files.
  # transformer/, vae/ and vision_encoder/ hold the generation weights, which are never read here —
  # the model loads those from the DCP in checkpoints/Cosmos3-Edge.
  #
  # CAVEAT: this warms the cache but cannot cap it. The tokenizer node passes no include patterns, so
  # on a networked node the first edge run still pulls the remaining ~8.5 GB. What the restriction
  # buys is a small air-gapped image and a working offline resolve.
  if hf_cached "${EDGE_VLM_REPO}" "${EDGE_VLM_REV}"; then
    echo "[skip] Cosmos3-Edge processor already present: ${HF_SNAPSHOT_DIR}"
  else
    echo "[download] ${EDGE_VLM_REPO}@${EDGE_VLM_REV} -> ${HF_HUB_CACHE} (edge caption processor, no generation weights)"
    uvx "${HF_CLI}" download --repo-type model "${EDGE_VLM_REPO}" --revision "${EDGE_VLM_REV}" \
      --exclude "transformer/*" --exclude "vae/*" --exclude "vision_encoder/*"
  fi
  # Drop hf's repo-tree manifest. NOT cosmetic: hf 1.26 (the pinned HF_CLI) writes trees/<sha>.json, and the
  # framework's own pinned hf 1.25.1 then compares it against the snapshot, sees the weight dirs we
  # skipped, and dies with "Incomplete snapshot available". Absent, it returns the snapshot as-is.
  # Verified both ways offline. Only the offline path is affected; preflight asserts it stays gone.
  EDGE_VLM_CACHE_DIR="${HF_HUB_CACHE}/models--${EDGE_VLM_REPO//\//--}"
  # hf records refs/<branch> only when it downloads BY that branch. This now downloads by sha, and
  # build_processor_lazy still passes the branch through, so an offline resolve needs the mapping
  # written here -- exactly the Qwen blocks' reason, and what preflight checks for this entry.
  if [[ -n "${EDGE_VLM_REF}" ]]; then
    mkdir -p "${EDGE_VLM_CACHE_DIR}/refs"
    printf '%s' "${EDGE_VLM_REV}" > "${EDGE_VLM_CACHE_DIR}/refs/${EDGE_VLM_REF}"
  fi
  rm -rf "${EDGE_VLM_CACHE_DIR}/trees"
fi

# ---- Gemma (AMG local-serving VLM, scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh) ----
# Last on purpose: at ~62.6GB it is far larger than every other hub-cache entry, so everything
# small lands before it starts and an interruption costs only this entry. The probe is BOTH weight
# shards, not config.json (preflight probes the same): hf links each file into the snapshot as it
# completes, so the small metadata is "present" within seconds of a pull starting, and a
# config-only probe would [skip] a truncated snapshot after an interruption. With the shards as
# the probe, a re-run after an interruption re-enters the download and hf fetches only what is
# still missing.
if ! want gemma; then
  echo "[skip] Gemma — not selected"
else
  if hf_cached "${GEMMA_REPO}" "${GEMMA_REV}"; then
    echo "[skip] Gemma already present: ${HF_SNAPSHOT_DIR}"
  else
    echo "[download] ${GEMMA_REPO}@${GEMMA_REV} -> ${HF_HUB_CACHE} (~62.6GB; AMG local-serving VLM)"
    uvx "${HF_CLI}" download --repo-type model "${GEMMA_REPO}" --revision "${GEMMA_REV}"
  fi
  # vllm serve gets the bare repo id, no revision, so an offline resolve goes through refs/main —
  # which `hf download --revision <sha>` never writes. Unconditional, like the Qwen blocks above.
  GEMMA_CACHE_DIR="${HF_HUB_CACHE}/models--${GEMMA_REPO//\//--}"
  mkdir -p "${GEMMA_CACHE_DIR}/refs"
  printf '%s' "${GEMMA_REV}" > "${GEMMA_CACHE_DIR}/refs/main"
  # Drop hf's repo-tree manifest, like the Edge block above. NOT cosmetic even though this snapshot
  # is complete: the air-gapped images bake the cache root-owned, hf 1.26 reads trees/<sha>.json on
  # every offline resolve, and a non-root uid cannot — so every vllm start warns "Ignoring corrupted
  # tree cache file ... Permission denied" about a file the resolve never needed. Absent, the resolve
  # is silent. Unconditional, and preflight asserts it stays gone.
  rm -rf "${GEMMA_CACHE_DIR}/trees"
fi

echo ""
echo "Done. Checkpoints in ${CKPT_DIR}:"
for base_name in "${BASE_CHECKPOINT_NAMES[@]}"; do
  echo "  base       : ${CKPT_DIR}/${base_name}"
done
if want dinov2; then echo "  dinov2     : ${DINOV2_OUT}"; fi
if want cradio; then echo "  cradio     : ${CRADIO_OUT}/${CRADIO_FILE}"; fi
if want wan-vae; then echo "  wan vae    : ${WAN_VAE_OUT}"; fi
if want sam2; then echo "  sam2       : ${SAM2_OUT}/${SAM2_FILE}"; fi
if want cosmos-vlm; then echo "  cosmos vlm : ${COSMOS_VLM_OUT}"; fi
if want guardrail; then echo "  guardrail  : ${GUARDRAIL_REPO} (${HF_HUB_CACHE})"; fi
if want guardrail; then echo "  qwen guard : ${QWEN_GUARD_REPO} (${HF_HUB_CACHE})"; fi
if want qwen-tokenizer; then echo "  qwen vlm tk: ${QWEN_VLM_REPO} (${HF_HUB_CACHE})"; fi
if want edge-processor; then echo "  edge proc  : ${EDGE_VLM_REPO} (${HF_HUB_CACHE})"; fi
if want gemma; then echo "  gemma      : ${GEMMA_REPO} (${HF_HUB_CACHE})"; fi
if [[ ${#SELECTED[@]} -gt 0 ]]; then
  echo ""
  echo "Only the groups named on the command line were fetched: ${SELECTED[*]}"
fi
echo ""
echo "These paths are hardcoded in the anomalygen code; no env vars to set."

# --- checkpoint integrity -----------------------------------------------------
# Pinning a revision fixes which commit is fetched, not which bytes land on disk, and says nothing
# about the tree afterwards: checkpoints/ is bind-mounted into the container, so anything with write
# access to the host path can swap a weight file between download and load.
#
# So: record a digest the first time, and check against it on every later run. Skipped when a
# download failed, because a partial tree would record partial digests as the reference.
#
# The weights that carry model behaviour, not the whole tree: these are ~90% of the bytes, and the
# rest is config and tokenizer JSON a tamper would have to change these to exploit. All seven Nano
# shards, though — a checkpoint loads every one, so recording a single shard leaves six unverified.
#
# `sha256sum -c` does the work, so there is no manifest format of our own and a reviewer can re-run
# the same command by hand. Paths are relative to CKPT_DIR so the manifest is portable.
#
# Two manifests, because the two groups of files have different truth conditions and therefore
# different remedies when they mismatch:
#
#   hub       Downloaded, and every repo is revision-pinned above. HuggingFace publishes the same
#             sha256 as each file's LFS oid, so these digests are upstream's bytes, independently
#             checkable, and only change if a pin changed. A mismatch here is an integrity problem.
#   converted Produced locally by convert_model_to_dcp, so the bytes depend on the torch version,
#             the cosmos-framework pin, and the shard layout of the machine that ran the conversion.
#             A bump to any of those changes them while the upstream model is identical, so a
#             mismatch here is usually a stale record rather than tampering.
#
# Re-recording is "delete the manifest and re-run", so keeping them apart is what makes that safe: a
# torch bump discards the converted digests and leaves the verifiable Hub ones in place. One file
# would make the only remedy throw away both groups, turning the gate into a habit of deleting it.
#
# The Hub manifest is committed, so a fresh clone verifies rather than records. Its digests are
# HuggingFace's published LFS oids, re-derivable from the Hub without trusting this checkout.
HUB_MANIFEST="${REPO_ROOT}/assets/checkpoint_manifest.sha256"
# Grouped exactly as the downloads above are, so a selective run verifies what it fetched and says
# nothing about what it did not -- listing an absent weight would report a tamper for a file this
# run was never asked to provision.
HUB_WEIGHTS=()
if want cosmos-vlm; then HUB_WEIGHTS+=(
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00001-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00002-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00003-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00004-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00005-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00006-of-00007.safetensors"
  "nvidia/Cosmos3-Nano/transformer/diffusion_pytorch_model-00007-of-00007.safetensors"
); fi
if want dinov2; then HUB_WEIGHTS+=("facebook/dinov2-large/pytorch_model.bin"); fi
if want cradio; then HUB_WEIGHTS+=("nvidia/C-RADIO-V3/model.safetensors"); fi
if want wan-vae; then HUB_WEIGHTS+=("wan2pt2/Wan2.2_VAE.pth"); fi
if want sam2; then HUB_WEIGHTS+=("facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt"); fi
if want gemma; then HUB_WEIGHTS+=(
  # Gemma's shards are hub-cache snapshot paths, not --local-dir ones, because that is where the
  # download above puts the bytes; their digests are HuggingFace's published LFS oids like every
  # other entry. Built from the pins so a GEMMA_REV bump that forgets the Hub manifest fails the
  # coverage gate below instead of silently unverifying the new snapshot.
  "hf/models--${GEMMA_REPO//\//--}/snapshots/${GEMMA_REV}/model-00001-of-00002.safetensors"
  "hf/models--${GEMMA_REPO//\//--}/snapshots/${GEMMA_REV}/model-00002-of-00002.safetensors"
); fi
# Edge is here and not in HUB_WEIGHTS because the base models are never downloaded in Hub weight
# format — convert_model_to_dcp writes a DCP, so these shards are their only local copy. Nano is
# in both lists because text2roi and the captioner load a second, Hub-format copy of it.
# All shards, not one: a DCP loads every shard, and hashing the 8 takes ~30 s.
CONVERTED_MANIFEST="${REPO_ROOT}/assets/checkpoint_manifest_converted.sha256"
CONVERTED_WEIGHTS=()
if want base-nano; then CONVERTED_WEIGHTS+=(
  "Cosmos3-Nano/model/__0_0.distcp"
  "Cosmos3-Nano/model/__0_1.distcp"
  "Cosmos3-Nano/model/__0_2.distcp"
  "Cosmos3-Nano/model/__0_3.distcp"
  "Cosmos3-Nano/model/__0_4.distcp"
  "Cosmos3-Nano/model/__0_5.distcp"
); fi
if want base-edge; then CONVERTED_WEIGHTS+=(
  "Cosmos3-Edge/model/__0_0.distcp"
  "Cosmos3-Edge/model/__0_1.distcp"
); fi

verify_or_record() {  # verify_or_record <manifest path> <label> <mismatch guidance> <files...>
  local manifest="$1" label="$2" guidance="$3"
  shift 3
  # A selective run can leave one of the two lists empty -- `... gemma` touches no converted weight.
  # sha256sum with no operands would read stdin and hang, so return before it can.
  if [[ $# -eq 0 ]]; then return 0; fi
  local rel="${manifest#"${REPO_ROOT}/"}"
  if [[ -f "${manifest}" ]]; then
    echo "Verifying ${label} weights against ${rel} ..."
    # `sha256sum -c` only checks the lines it is given, so a weight added to the list above but
    # absent from an existing manifest would be silently unverified — the check would still pass
    # and report OK. Fail on under-coverage instead, naming the remedy, so extending the list is
    # never a no-op.
    local uncovered=()
    local want
    for want in "$@"; do
      grep -qF -- "  ${want}" "${manifest}" || uncovered+=("${want}")
    done
    if [[ ${#uncovered[@]} -gt 0 ]]; then
      echo "" >&2
      echo "ERROR: ${rel} does not cover ${#uncovered[@]} requested ${label} weight(s):" >&2
      printf '  %s\n' "${uncovered[@]}" >&2
      echo "The list in this script grew without the manifest growing with it. Re-record with:" >&2
      echo "  rm ${rel} && bash scripts/download_checkpoints.sh" >&2
      echo "then confirm the new digests against the Hub before committing them." >&2
      exit 1
    fi
    # `sha256sum -c` checks every line of the file it is given, not the files named above -- those
    # only drive the coverage gate. On a full run that is the stronger check and stays: a manifest
    # line with no entry in the list is still verified. A subset run has to narrow it, or it would
    # report every group it was never asked to fetch as a missing weight.
    local checked="${manifest}" subset=""
    if [[ ${#SELECTED[@]} -gt 0 ]]; then
      subset="$(mktemp)"
      local f
      for f in "$@"; do grep -F -- "  ${f}" "${manifest}" >> "${subset}"; done
      checked="${subset}"
    fi
    if ( cd "${CKPT_DIR}" && sha256sum -c --quiet "${checked}" ); then
      echo "${label} weights OK ($(grep -c . "${checked}") files)."
      if [[ -n "${subset}" ]]; then rm -f "${subset}"; fi
    else
      if [[ -n "${subset}" ]]; then rm -f "${subset}"; fi
      echo "" >&2
      echo "ERROR: ${label} weights do not match ${rel}." >&2
      echo "Each line above is a weight whose bytes differ from what was recorded, or that is gone." >&2
      echo "${guidance}" >&2
      exit 1
    fi
  elif [[ ${#SELECTED[@]} -gt 0 ]]; then
    # A subset run must not author the reference. The digests it could record cover only the groups
    # it fetched, and the coverage gate above would then fail every later full run against its own
    # complete weight list -- turning "provision one model quickly" into a broken checkout. Both
    # manifests are committed, so this branch is for maintainers re-recording, which is a full run.
    echo "Not recording ${rel}: this run fetched a subset. Re-run with no group arguments." >&2
  else
    mkdir -p "$(dirname "${manifest}")"
    # Write-then-rename: a redirect truncates first, so a kill between truncate and write would
    # leave a half-written manifest behind. rename(2) is atomic, so the file is either the old
    # content or the complete new one.
    ( cd "${CKPT_DIR}" && sha256sum "$@" ) > "${manifest}.tmp"
    mv "${manifest}.tmp" "${manifest}"
    echo "Recorded ${label} weights -> ${rel} ($(grep -c . "${manifest}") files)."
  fi
}

if [[ ${#failed_sizes[@]} -eq 0 ]]; then
  echo ""
  verify_or_record "${HUB_MANIFEST}" "Hub-sourced" \
    "These are revision-pinned downloads whose digests match HuggingFace's published LFS oids, so
they should never change unless a pin above did. Treat a mismatch as an integrity problem, not a
stale manifest — re-record only after confirming the new bytes are the ones upstream publishes." \
    "${HUB_WEIGHTS[@]}"
  verify_or_record "${CONVERTED_MANIFEST}" "Locally-converted" \
    "These are produced by convert_model_to_dcp on this machine, so a torch or cosmos-framework bump
legitimately changes them — as does a bump of EDGE_BASE_REV, or a DCP converted before the current
pin from older Hub weights. To keep the weights you have, re-record by deleting
${CONVERTED_MANIFEST#"${REPO_ROOT}/"} and re-running; to pick up the pinned Hub weights instead,
also delete the stale checkpoints/<Cosmos3-Nano|Cosmos3-Edge> directory first. The Hub-sourced
manifest is separate and stays in place." \
    "${CONVERTED_WEIGHTS[@]}"
fi

# Reported last so it is the final thing on screen, and after the summary so the user can see which
# downloads DID succeed — the point of not aborting on the first gated 403.
if [[ ${#failed_sizes[@]} -gt 0 ]]; then
  echo "" >&2
  echo "ERROR: base model size(s) not downloaded: ${failed_sizes[*]}" >&2
  echo "Everything else above succeeded. This is almost always an unaccepted model licence:" >&2
  for base_name in "${failed_sizes[@]}"; do
    echo "  https://huggingface.co/nvidia/${base_name}   <- open, accept, then re-run this script" >&2
  done
  echo "Both sizes are required: a recipe's model_size selects the base checkpoint, so a" >&2
  echo "missing size fails at model load instead of here. Re-runs skip what is already present." >&2
  exit 1
fi
