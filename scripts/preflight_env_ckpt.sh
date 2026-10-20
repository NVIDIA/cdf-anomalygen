#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# preflight_env_ckpt.sh — preflight the anomalygen runtime before running the pipeline. Checks, in order:
#   1. GPU / driver       (nvidia-smi lists the NVIDIA driver + visible GPUs)
#   2. Python             (>= 3.10)
#   3. CUDA / torch       (torch imports and can use CUDA)
#   4. Python deps        (the core packages the pipeline imports)
#   5. Hugging Face auth  (only a failure when checkpoints turn out to be missing — it is needed to
#                          fetch them, not to run once they are on disk)
#   6. Model checkpoints  (under ./checkpoints: both base sizes, KPI/ROI models, HF cache entries)
#
# Read-only; exits non-zero if any check fails (doubles as a CI/preflight gate) and prints a fix step
# for each failing group. Read WHICH group failed before reacting: it exits 1 on any failure, so
# chaining the downloader off that exit code would fire a checkpoint download for a torch/CUDA
# problem. Run scripts/download_checkpoints.sh only when the checkpoint group is what failed.
#
# Run from anywhere (with the uv env active — source .venv/bin/activate):
#   bash scripts/preflight_env_ckpt.sh
#
# Env overrides:
#   CKPT_DIR   checkpoint root (default: <repo>/checkpoints) — same override as download_checkpoints.sh
#   PYTHON     python interpreter to probe (default: python)
#
# No BASE_CHECKPOINT_NAME override — keep the list below in step with download_checkpoints.sh's.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"

# Every location and probe below comes from the catalogue, which is also what download_checkpoints.sh
# provisions from -- so this cannot assert a path the provisioner does not write. Asked separately
# they drifted: this file wanted face_blur_filter/Resnet50_Final.pth from the guardrail snapshot and
# the provisioner's skip guard tested only `blocklist`.
# shellcheck source=scripts/weights_catalogue.sh
source "${REPO_ROOT}/scripts/weights_catalogue.sh"
# Resolved up front and each one CHECKED with `|| exit`, because this file deliberately runs without
# `set -e`: it collects every failure and reports them together, so aborting on the first would
# defeat it. That makes a bare `X="$(w_location ...)"` silent here in a way it is not in
# download_checkpoints.sh -- the lookup fails, prints, and leaves X empty, and
# `check_weights_dir "${CKPT_DIR}/"` then finds the first >=100MB file anywhere under checkpoints/
# and prints [pass] for a weight that is missing. A false green is the one answer a preflight must
# never give, so these exit rather than collect.
#
# `|| exit` on the ASSIGNMENT, which carries the substitution's status. Wrapping the lookup in a
# helper that exits does not work: the exit runs in the command substitution's subshell and the
# script carries on with an empty value -- the same silent failure in a different disguise.
BASE_NANO_NAME="$(w_location base-nano)" || exit 2
BASE_EDGE_NAME="$(w_location base-edge)" || exit 2
BASE_CHECKPOINT_NAMES=("${BASE_NANO_NAME}" "${BASE_EDGE_NAME}")
DINOV2_LOCATION="$(w_location dinov2)" || exit 2
CRADIO_LOCATION="$(w_location cradio)" || exit 2
WAN_VAE_LOCATION="$(w_location wan-vae)" || exit 2
SAM2_LOCATION="$(w_location sam2)" || exit 2
COSMOS_VLM_LOCATION="$(w_location cosmos-vlm)" || exit 2
QWEN_TOKENIZER_REPO="$(w_repo qwen-tokenizer)" || exit 2
EDGE_PROCESSOR_REPO="$(w_repo edge-processor)" || exit 2
GEMMA_REPO_ID="$(w_repo gemma)" || exit 2
PY="${PYTHON:-python}"

fail=0
ckpt_missing=0
ok()  { local msg="$1"; printf '[pass] %s\n' "$msg"; return 0; }
bad() { local msg="$1"; printf '[fail] %s\n' "$msg"; fail=$((fail + 1)); return 0; }
hdr() { local title="$1"; printf '\n=== %s ===\n' "$title"; return 0; }

# --- 1. GPU / NVIDIA driver (nvidia-smi) --------------------------------------
hdr "GPU / NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi not on PATH — no NVIDIA driver / GPU visible"
else
  gpus="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
  if [[ -z "$gpus" ]]; then
    bad "nvidia-smi found but reported no GPUs"
  else
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    ndev="$(printf '%s\n' "$gpus" | grep -c .)"
    ok "NVIDIA driver ${drv}, ${ndev} GPU(s)"
    i=0
    while IFS= read -r g; do
      ok "GPU${i}: ${g}"
      i=$((i + 1))
    done <<< "$gpus"
  fi
fi

# --- 2. Python ----------------------------------------------------------------
hdr "Python (need >= 3.10)"
if ! command -v "$PY" >/dev/null 2>&1; then
  bad "no '$PY' on PATH — activate the env: source ${REPO_ROOT}/.venv/bin/activate"
else
  pyver="$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"
  if "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    ok "Python ${pyver} ($(command -v "$PY"))"
  else
    bad "Python ${pyver:-unknown} — need >= 3.10"
  fi
fi

# --- 3 + 4. CUDA / torch + Python deps (one interpreter, so torch loads once) --
hdr "CUDA / torch + Python deps"
"$PY" - <<'PY'
import importlib, sys

try:
    import importlib.metadata as _md
    _pkgdist = _md.packages_distributions()  # top-level import name -> [dist names]
except Exception:  # noqa: BLE001
    _md, _pkgdist = None, {}

n_fail = 0
def line(passed, msg):
    print(("[pass] " if passed else "[fail] ") + msg)

def version(name, mod):
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    if _md is not None:
        try:
            return _md.version(name)  # dist name usually matches the import name
        except Exception:  # noqa: BLE001
            pass
        for dist in _pkgdist.get(name, []):  # fall back to the import-name -> dist map
            try:
                return _md.version(dist)
            except Exception:  # noqa: BLE001
                pass
    return "unknown"

try:
    import torch
    avail = torch.cuda.is_available()
    ndev = torch.cuda.device_count() if avail else 0
    line(True, f"torch {torch.__version__}")
    line(avail, f"CUDA available={avail}, torch sees {ndev} device(s)")
    if not avail:
        n_fail += 1
except Exception as exc:  # noqa: BLE001
    line(False, f"torch import failed: {exc}")
    n_fail += 1

# Core packages the pipeline imports: framework + this package + transformers, and the
# compiled CUDA extensions built by scripts/env_setup.sh (attention / TE / apex / natten).
for name in (
    "cosmos_framework", "anomalygen", "transformers",
    "flash_attn", "flash_attn_3_nv", "transformer_engine", "apex", "natten",
):
    try:
        mod = importlib.import_module(name)
        line(True, f"{name} {version(name, mod)}")
    except Exception as exc:  # noqa: BLE001
        line(False, f"{name}: {exc}")
        n_fail += 1

sys.exit(min(n_fail, 250))  # exit code carries the failure count
PY
fail=$((fail + $?))

# --- 5. Hugging Face auth -----------------------------------------------------
# Checked offline (env var, else the token the CLI stores at login) so the preflight stays fast and
# usable air-gapped — `hf auth whoami` would need network. The *verdict* is deferred to section 6:
# auth is only needed to fetch checkpoints, so failing here when they are already on disk would make
# the documented `preflight || download_checkpoints.sh` chain start a download nothing was waiting for.
hdr "Hugging Face auth (needed only to download checkpoints)"
hf_auth_ok=0
hf_token_file="${HF_TOKEN_PATH:-${HF_HOME:-${HOME:-}/.cache/huggingface}/token}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  ok "HF_TOKEN is set"; hf_auth_ok=1
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  ok "HUGGING_FACE_HUB_TOKEN is set"; hf_auth_ok=1
elif [[ -s "${hf_token_file}" ]]; then
  ok "stored login token: ${hf_token_file}"; hf_auth_ok=1
else
  printf '[warn] %s\n' "not authenticated — harmless if every checkpoint below is already present;"
  printf '[warn] %s\n' "  a failure only if any must be downloaded (see the next section)."
fi

# --- 6. Model checkpoints -----------------------------------------------------
hdr "Model checkpoints (under ${CKPT_DIR})"
# Weight files are size-checked, not just presence-checked: an interrupted HF pull commonly writes the
# small metadata (config.json) and dies before the weights, and a presence-only gate green-lights that
# — then the model load fails much later instead of here. `min_mb` is a floor well under the real size,
# enough to catch a truncated or zero-byte file without pinning an exact version.
check_dir() {  # check_dir <dir> <label>: non-empty directory
  local dir="$1" label="$2"
  if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
    ok "$label"
  else
    bad "$label — missing: $dir"; ckpt_missing=$((ckpt_missing + 1))
  fi
  return 0
}
check_file() {  # check_file <path> <label> [min_mb]
  local path="$1" label="$2" min_mb="${3:-0}" size_mb
  if [[ ! -f "$path" ]]; then bad "$label — missing: $path"; ckpt_missing=$((ckpt_missing + 1)); return; fi
  size_mb=$(( $(wc -c < "$path" 2>/dev/null || echo 0) / 1048576 ))
  if [[ "$size_mb" -lt "$min_mb" ]]; then
    bad "$label — truncated: $path is ${size_mb}MB, expected >= ${min_mb}MB (re-download)"
    ckpt_missing=$((ckpt_missing + 1))
  else
    ok "$label"
  fi
}
# A weights dir is only as good as its largest file — catches a config-only partial pull.
check_weights_dir() {  # check_weights_dir <dir> <label> <min_mb>
  local dir="$1" label="$2" min_mb="$3" biggest
  if [[ ! -d "$dir" ]]; then bad "$label — missing: $dir"; ckpt_missing=$((ckpt_missing + 1)); return; fi
  biggest=$(find "$dir" -type f -printf '%s\n' 2>/dev/null | sort -rn | head -1)
  biggest=$(( ${biggest:-0} / 1048576 ))
  if [[ "$biggest" -lt "$min_mb" ]]; then
    bad "$label — no weight file >= ${min_mb}MB under $dir (largest ${biggest}MB; partial download?)"
    ckpt_missing=$((ckpt_missing + 1))
  else
    ok "$label"
  fi
}

# HF hub-cache entries are resolved by repo id, not path, so a snapshot alone is not enough:
# refs/<rev> must map branch -> sha (a sha-pinned `hf download` never writes it), and <probe> must
# exist in that snapshot. Pass an empty <ref> for entries the framework resolves by pinned sha.
# The probes are the catalogue's, and all of them are required -- a hub-cache entry is complete or
# it is not. w_probes fails for a repo the catalogue does not carry, which is what stops a repo id
# typed here from quietly checking nothing.
check_hf_cache() {  # check_hf_cache <repo id> <ref name, or "" to skip> <label>
  local repo="$1" ref="$2" label="$3" sha probe probes
  local dir="${CKPT_DIR}/hf/models--${repo//\//--}"
  if ! probes="$(w_probes "$repo")"; then
    bad "$label — not in scripts/weights_catalogue.sh, so there is nothing to check it against"
    ckpt_missing=$((ckpt_missing + 1)); return
  fi
  if [[ ! -d "$dir" ]]; then bad "$label — missing: $dir"; ckpt_missing=$((ckpt_missing + 1)); return; fi
  if [[ -n "$ref" ]]; then
    if [[ ! -s "${dir}/refs/${ref}" ]]; then
      bad "$label — no refs/${ref} under $dir; an offline resolve of revision '${ref}' cannot find the snapshot"
      ckpt_missing=$((ckpt_missing + 1)); return
    fi
    sha="$(cat "${dir}/refs/${ref}")"
  else
    sha="$(ls -1 "${dir}/snapshots" 2>/dev/null | head -1)"
  fi
  for probe in ${probes}; do
    if [[ -z "$sha" ]] || [[ ! -e "${dir}/snapshots/${sha}/${probe}" ]]; then
      bad "$label — ${probe} missing from snapshot ${sha:-<none>} under $dir (partial download?)"
      ckpt_missing=$((ckpt_missing + 1)); return
    fi
  done
  ok "$label"
}

# Both sizes: model_size picks the DCP, so checking nano alone let an edge run die at model load.
for base_name in "${BASE_CHECKPOINT_NAMES[@]}"; do
  check_weights_dir "${CKPT_DIR}/${base_name}" "${base_name} (Base model)" 100
done
# The minimum sizes are this file's own: the catalogue says where a weight is and what proves it is
# there, and a floor well under the real size is a stricter test on top of that, not a second answer.
check_weights_dir "${CKPT_DIR}/${DINOV2_LOCATION}" "DINOv2 (KPI model)" 100
check_file "${CKPT_DIR}/${CRADIO_LOCATION}" "C-RADIO v3 (KPI model)" 100
check_file "${CKPT_DIR}/${WAN_VAE_LOCATION}" "Wan2.2 VAE (tokenizer)" 100
check_file "${CKPT_DIR}/${SAM2_LOCATION}" "SAM2.1 (ROI segmentation)" 100
check_file "${CKPT_DIR}/${COSMOS_VLM_LOCATION}" "Cosmos3-Nano HF config" 0
check_weights_dir "${CKPT_DIR}/${COSMOS_VLM_LOCATION%/*}" "Cosmos3-Nano HF weights (Text2ROI, Caption)" 100

# --- HF hub cache (${CKPT_DIR}/hf) --------------------------------------------
# Guardrail: resolved at the sha the framework itself pins (GUARDRAIL1_CHECKPOINT.revision), so no ref.
check_hf_cache nvidia/Cosmos-Guardrail1 "" "Cosmos-Guardrail1 (generate.py guardrail)"
# Qwen3Guard: the framework calls from_pretrained() unpinned, i.e. by branch, so refs/main is required.
check_hf_cache Qwen/Qwen3Guard-Gen-0.6B main "Qwen3Guard-Gen-0.6B (text safety)"
# Caption tokenizer for model_size: nano. Weights are deliberately absent — probe a tokenizer file.
check_hf_cache "${QWEN_TOKENIZER_REPO}" main "Qwen3-VL tokenizer (nano caption)"
# Caption processor for model_size: edge, resolved from this cache while the model is being built.
check_hf_cache "${EDGE_PROCESSOR_REPO}" main "Cosmos3-Edge processor (edge caption)"
# That entry is partial (no generation weights), which only resolves offline while no tree manifest
# claims more files than are on disk. hf >= 1.26 writes one; the downloader removes it, a hand-run
# `hf download` puts it back and silently re-breaks air-gapped edge.
if [[ -d "${CKPT_DIR}/hf/models--nvidia--Cosmos3-Edge/trees" ]]; then
  bad "Cosmos3-Edge processor — repo tree manifest present; an offline resolve will report an incomplete snapshot. Re-run scripts/download_checkpoints.sh (or: rm -rf ${CKPT_DIR}/hf/models--nvidia--Cosmos3-Edge/trees)"
  ckpt_missing=$((ckpt_missing + 1))
else
  ok "Cosmos3-Edge processor — no tree manifest (partial snapshot resolves offline)"
fi
# Gemma: scripts/agentic_mask_generation_serving/gemma-4-31b-it.sh serves it via vllm by bare repo id (no revision), so
# an offline resolve needs refs/main; probe BOTH weight shards, not config.json — hf links each file
# into the snapshot as it completes, so the small metadata is "present" within seconds of a pull
# starting and a config-only probe green-lights an interrupted download.
GEMMA_REPO="${GEMMA_REPO_ID}"
check_hf_cache "$GEMMA_REPO" main "Gemma-4-31b-it (AMG serving VLM, both shards)"
# The air-gapped images bake this cache root-owned and hf 1.26 reads trees/<sha>.json on every
# offline resolve, which a non-root uid cannot — every vllm start then warns "Ignoring corrupted
# tree cache file". The downloader removes it; a hand-run `hf download` puts it back.
if [[ -d "${CKPT_DIR}/hf/models--${GEMMA_REPO//\//--}/trees" ]]; then
  bad "Gemma-4-31b-it — repo tree manifest present; a baked, non-root offline resolve warns on every serve. Re-run scripts/download_checkpoints.sh (or: rm -rf ${CKPT_DIR}/hf/models--${GEMMA_REPO//\//--}/trees)"
  ckpt_missing=$((ckpt_missing + 1))
else
  ok "Gemma-4-31b-it — no tree manifest (offline resolve is silent)"
fi

# --- 7. checkpoint integrity --------------------------------------------------
# Pinning a revision fixes which commit was fetched, not which bytes are on disk now: checkpoints/ is
# a bind mount, so anything with write access can swap a weight between download and load. The two
# manifests were previously only compared inside download_checkpoints.sh, i.e. at download time and
# nowhere else. Checked here so the documented pre-run gate is also the integrity gate.
#
# Only when the weights are present — a missing checkpoint is already reported above, and reporting
# it twice as an integrity failure would point at the wrong fix step.
hdr "Checkpoint integrity"
if [[ "$ckpt_missing" -gt 0 ]]; then
  note_skip="${ckpt_missing} checkpoint(s) missing above — integrity check skipped until they are present"
  printf '[skip] %s\n' "$note_skip"
else
  for manifest_rel in assets/checkpoint_manifest.sha256 assets/checkpoint_manifest_converted.sha256; do
    manifest="${REPO_ROOT}/${manifest_rel}"
    if [[ ! -f "$manifest" ]]; then
      bad "${manifest_rel} — absent; run scripts/download_checkpoints.sh to record it"
      ckpt_missing=$((ckpt_missing + 1))
      continue
    fi
    # One implementation, shared with the opt-in load-time check in anomalygen/inference/inpaint.py,
    # so the two cannot drift. Failures are printed one per line by the helper.
    if out=$("$PY" -c '
import sys
from anomalygen.checkpoint.utils import verify_manifest
failures = verify_manifest(sys.argv[1], sys.argv[2])
for rel, why in failures:
    print(f"{rel}: {why}")
sys.exit(1 if failures else 0)
' "$manifest" "$CKPT_DIR" 2>&1); then
      ok "${manifest_rel} — all recorded weights match"
    else
      bad "${manifest_rel} — recorded weights do not match:"
      printf '        %s\n' "$out"
      ckpt_missing=$((ckpt_missing + 1))
    fi
  done
fi

# The section-5 auth verdict, resolvable only now that we know whether anything must be fetched.
hf_auth_missing=0
if [[ "$ckpt_missing" -gt 0 ]] && [[ "$hf_auth_ok" -eq 0 ]]; then
  bad "Hugging Face auth — required to download the ${ckpt_missing} missing checkpoint(s) above"
  hf_auth_missing=1
fi

# --- summary ------------------------------------------------------------------
if [[ "$fail" -eq 0 ]]; then
  printf '\nAll checks passed.\n'
  exit 0
fi

# Auth is subtracted alongside the checkpoints: it can only fail when they are missing, and the
# checkpoint fix step below already tells you to authenticate first. Counting it as an env failure
# would wrongly advise rebuilding the venv.
env_fail=$((fail - ckpt_missing - hf_auth_missing))
printf '\n%d check(s) failed. Fix steps:\n' "$fail"
if [[ "$env_fail" -gt 0 ]]; then
  printf '  - env / Python / deps (%d): rebuild the venv, then activate it —\n' "$env_fail"
  printf '      bash scripts/env_setup.sh\n'
  printf '      source .venv/bin/activate\n'
fi
if [[ "$ckpt_missing" -gt 0 ]]; then
  printf '  - missing checkpoints (%d): authenticate to Hugging Face, then fetch —\n' "$ckpt_missing"
  printf '      export HF_TOKEN=<token>       # or: hf auth login  (accept the model HF license)\n'
  printf '      bash scripts/download_checkpoints.sh\n'
fi
exit 1
