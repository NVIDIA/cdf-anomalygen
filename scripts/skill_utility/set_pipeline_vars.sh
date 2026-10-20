# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# set_pipeline_vars.sh — validate the AnomalyGen pipeline's inputs and set the shared shell variables the
# steps refer to (${DATASET_DIR}, ${GEN_DIR}, ${OUT}, ${RUN_DIR}, ...).
#
# SOURCE it; do not execute it — an executed copy sets the variables in a child shell that then exits:
#
#   source scripts/skill_utility/set_pipeline_vars.sh --name phone_screen \
#       --dataset_dir datasets/phone_screen --num_sdg 25
#   source scripts/skill_utility/set_pipeline_vars.sh --name uc1pega-opus5-e2e \
#       --dataset_dir datasets/uc1pega --num_sdg 40 --num_gpus 8 --num_search_run 5
#
# The second example is the normal shape, not an odd one: the experiment label and the dataset name
# differ, which is why one is never derived from the other.
#
# Re-source it in every shell that needs the variables. Sourcing is idempotent and cheap (no I/O), and
# a fresh shell — a new terminal, a new tool call, a new CI step — does not inherit them.
#
# Options (defaults match the Inputs table in skills/cdf-anomalygen/SKILL.md):
#   --name             required  EXPERIMENT LABEL ONLY. It names the recipe, the run dir, the logs and
#                      the derived validation_*/generation_* mask-set dirs, and says nothing about where
#                      the dataset lives — that is --dataset_dir
#   --dataset_dir      required  dataset root in the Step 1 contract layout. It comes from the caller and
#                      is NEVER derived from --name: labels are routinely 'uc1pega-opus5-e2e' while the
#                      dataset is 'datasets/uc1pega', so a "datasets/${NAME}" default would be wrong in
#                      the normal case rather than the exceptional one. Existence is NOT checked here —
#                      that is the Step 1 dataset gate's job, and this script stays free of I/O
#   --num_sdg          required  samples to generate; in finetune_only it sizes the validation mask set
#   --num_gpus         1         --nproc_per_node for fine-tuning and generation
#   --num_search_run   3         Step 6 search rounds; 0 = clone-only
#   --task             texture_ft  recipe task_type; picks ${SCRIPTS} and the experiment name
#   --job_name         anomalygen_${TASK}  MUST match the recipe's job_name — it is the last segment of
#                      ${RUN_DIR}, so a recipe that renames job_name (e.g. to escape a resumed
#                      checkpoint) needs the same value here or Steps 4-7 read the wrong run dir
#
# Returns 0 on success, 64 on a usage/validation error, and prints what it set. Each printed line
# carries its provenance, so a log shows whether a value was asked for or fell back:
#
#   NUM_GPUS       = 8          (set)       <- passed on the command line
#   NUM_SEARCH_RUN = 3          (default)   <- the script's default
#   RUN_DIR        = results/…  (derived)   <- computed from the above
#
# A (set) marker on a value that differs from the documented default is the flag to look for when
# auditing a run: it means the caller overrode the skill, not that the skill chose it.
#
# ${LOG_DIR} is where every step's log belongs. Logs must never land in the repo root — a stray
# ./foo.log is invisible next to the run it came from and outlives the run that wrote it.

if [[ "${BASH_SOURCE[0]}" = "$0" ]]; then
  printf 'set_pipeline_vars.sh: source this script, do not execute it:\n' >&2
  printf '  source %s --name <label> --dataset_dir <dir> --num_sdg <n>\n' "$0" >&2
  exit 64
fi

_ag_vars() {
  local name="" dataset_dir="" num_sdg=""
  local num_gpus=1 num_search_run=3 task=texture_ft job_name=""

  # Provenance per variable, recorded as the options are parsed — by the time the values are printed a
  # defaulted and an explicitly-passed one are indistinguishable. Unlisted variables read 'derived'.
  local -A prov=()

  local opt
  while [[ $# -gt 0 ]]; do
    opt="$1"
    case "$opt" in
      --name)           name="${2-}";           prov[NAME]=set;           shift 2 || return 64 ;;
      --dataset_dir)    dataset_dir="${2-}";    prov[DATASET_DIR]=set;    shift 2 || return 64 ;;
      --num_sdg)        num_sdg="${2-}";        prov[NUM_SDG]=set;        shift 2 || return 64 ;;
      --num_gpus)       num_gpus="${2-}";       prov[NUM_GPUS]=set;       shift 2 || return 64 ;;
      --num_search_run) num_search_run="${2-}"; prov[NUM_SEARCH_RUN]=set; shift 2 || return 64 ;;
      --task)           task="${2-}";           prov[TASK]=set;           shift 2 || return 64 ;;
      --job_name)       job_name="${2-}";       prov[JOB_NAME]=set;       shift 2 || return 64 ;;
      -h|--help)
        awk '/^#!/ {next} /SPDX-/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
        return 64 ;;
      *)
        printf 'set_pipeline_vars.sh: unknown option %q (see --help)\n' "$opt" >&2
        return 64 ;;
    esac
  done

  # ${NAME} and ${TASK} reach shell command lines and path expansions below, so validate before use —
  # reject, never sanitize. A silently rewritten name would point every later step at the wrong dirs.
  # ${DATASET_DIR} is exempt from this charset by design: it is a path, and a legitimate dataset root
  # can live anywhere. It gets the protection paths get instead — quoted at every expansion downstream.
  local ident='^[A-Za-z0-9][A-Za-z0-9._-]*$'
  if ! [[ $name =~ $ident ]]; then
    printf 'set_pipeline_vars.sh: invalid --name %q: letters/digits/._- only, must start alphanumeric\n' \
      "$name" >&2
    return 64
  fi
  if ! [[ $task =~ $ident ]]; then
    printf 'set_pipeline_vars.sh: invalid --task %q: letters/digits/._- only, must start alphanumeric\n' \
      "$task" >&2
    return 64
  fi
  job_name="${job_name:-anomalygen_${task}}"
  if ! [[ $job_name =~ $ident ]]; then
    printf 'set_pipeline_vars.sh: invalid --job_name %q: letters/digits/._- only, must start alphanumeric\n' \
      "$job_name" >&2
    return 64
  fi

  # Required with no fallback: --dataset_dir is never derived from --name (see the header — the label
  # equals the dataset name only in the exceptional case), and a guessed "datasets/${NAME}" would hand
  # Step 5's and Step 6's --real_root a wrong reference on every run where they differ. A missing value
  # must stop the caller, not be guessed at.
  if [[ -z $dataset_dir ]]; then
    printf 'set_pipeline_vars.sh: --dataset_dir is required; it is never derived from --name (%q is only an experiment label)\n' \
      "$name" >&2
    return 64
  fi

  local v val
  for v in num_sdg num_gpus num_search_run; do
    val="${!v}"
    if ! [[ $val =~ ^[0-9]+$ ]]; then
      printf 'set_pipeline_vars.sh: invalid --%s %q: must be a non-negative integer\n' "$v" "$val" >&2
      return 64
    fi
  done
  if [[ "$num_gpus" -lt 1 ]]; then
    printf 'set_pipeline_vars.sh: invalid --num_gpus %q: need at least 1\n' "$num_gpus" >&2
    return 64
  fi

  # ${TASK} and ${SCRIPTS} are a pair, not a derivation: the task_type is what a recipe declares
  # (_TASK_TYPE in anomalygen/configs/loader.py) and ${SCRIPTS} is where that family's train/generate/
  # evaluate live. A new task family adds one line here. Failing loudly beats slicing a path out of the
  # task name and handing every later step a directory that does not exist.
  case "$task" in
    texture_ft) SCRIPTS="anomalygen/scripts/texture" ;;
    *)
      printf 'set_pipeline_vars.sh: unknown --task %q — add its SCRIPTS dir to the case in %s\n' \
        "$task" "${BASH_SOURCE[0]}" >&2
      return 64 ;;
  esac

  NAME="$name"
  TASK="$task"
  NUM_SDG="$num_sdg"
  NUM_GPUS="$num_gpus"
  NUM_SEARCH_RUN="$num_search_run"
  DATASET_DIR="$dataset_dir"
  # Path derivation only — no existence check for either of these. A wrong DATASET_DIR is the Step 1
  # dataset gate's to catch one step later; DEFECT_SPEC has a single consumer, the amp backend, and
  # under the default amg the file may legitimately not exist and nothing reads it.
  DEFECT_SPEC="${DATASET_DIR}/defect_spec.jsonl"
  RECIPE="ag_config/exp_${TASK}_${NAME}.yaml"
  VAL_DIR="datasets/validation_${NAME}"
  GEN_DIR="datasets/generation_${NAME}"
  OUT="results/generation_${NAME}"
  JOB_NAME="$job_name"
  RUN_DIR="results/anomalygen/${NAME}/${JOB_NAME}"
  LOG_DIR="results"

  # Anything the caller did not pass is either a script default or computed from one. Naming which is
  # which is the point: 'set' next to a non-default value is what an audit is looking for.
  local d
  for d in TASK NUM_GPUS NUM_SEARCH_RUN LOG_DIR; do prov[$d]="${prov[$d]:-default}"; done

  for v in NAME TASK SCRIPTS NUM_SDG NUM_GPUS NUM_SEARCH_RUN DATASET_DIR DEFECT_SPEC RECIPE \
           VAL_DIR GEN_DIR OUT JOB_NAME RUN_DIR LOG_DIR; do
    printf '%-14s = %-42s (%s)\n' "$v" "${!v}" "${prov[$v]:-derived}"
  done
}

# Clean up on both paths — 'return 64' would otherwise leave the helper defined in the caller's shell.
_ag_vars "$@"
_ag_rc=$?
unset -f _ag_vars
if [[ "$_ag_rc" -ne 0 ]]; then
  unset _ag_rc
  return 64
fi
unset _ag_rc
