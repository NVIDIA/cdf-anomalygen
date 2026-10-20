#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# train_control.sh — launch an AnomalyGen fine-tune that outlives the shell, and wait for it in slices.
#
# A fine-tune runs for hours at the 15000 default (dataset-dependent; measured times are in
# docs/tutorial/texture/7-agentic-end-to-end-flow.md): far longer than one foreground command, so it is
# launched detached with a PID file and a log, then polled.
#
#   scripts/skill_utility/train_control.sh start --name phone_screen --recipe ag_config/exp_texture_ft_phone_screen.yaml
#   scripts/skill_utility/train_control.sh wait  --name phone_screen        # repeat until it stops exiting 2
#   scripts/skill_utility/train_control.sh stop  --name phone_screen
#
# 'wait' blocks for --timeout seconds (default 570, just under a 10-minute tool budget) and reports the
# run's state as its EXIT CODE, so "still going" is never confused with "finished":
#
#   0  finished    the log holds 'Done with training'      -> go to Step 4
#   1  failed      the log holds a traceback, or the process died without the marker
#   2  running     the slice elapsed with the run healthy  -> call 'wait' again, as many times as it takes
#
# Exit 2 is the normal case for most of a run: at ~9.5 min a slice, the 15000 default takes dozens of them.
# A checkpoint appearing is NOT the finish line — the trainer writes one every save_iter and all but the
# last are intermediate. Gate on exit 0.
#
# Options:
#   start  --name <n> --recipe <path> [--num_gpus 1] [--scripts <dir>] [--task texture_ft]
#          [-- <hydra overrides>]   appended after experiment=, e.g. for a two-minute smoke test:
#          -- trainer.max_iter=20 trainer.validation_iter=10 checkpoint.save_iter=10
#   wait   --name <n> [--timeout 570] [--interval 30]
#   stop   --name <n>
#
# Paths are relative to the repo root; run it from there. Logs and PID files land in ./results.
set -uo pipefail

usage() {
  awk '/^#!/ {next} /SPDX-/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}" >&2
  exit 64
}

die() {
  local msg="$1"
  printf 'train_control.sh: %s\n' "$msg" >&2
  exit 64
}

name="" recipe="" num_gpus=1 scripts_dir="" task=texture_ft timeout_s=570 interval=30
overrides=()
cmd="${1-}"
case "$cmd" in "" | -h | --help) usage ;; *) ;; esac  # anything else: the dispatch below decides
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)     name="${2-}";        shift 2 || usage ;;
    --recipe)   recipe="${2-}";      shift 2 || usage ;;
    --num_gpus) num_gpus="${2-}";    shift 2 || usage ;;
    --scripts)  scripts_dir="${2-}"; shift 2 || usage ;;
    --task)     task="${2-}";        shift 2 || usage ;;
    --timeout)  timeout_s="${2-}";   shift 2 || usage ;;
    --interval) interval="${2-}";    shift 2 || usage ;;
    -h|--help)  usage ;;
    --)         shift; overrides=("$@"); break ;;
    *)          die "unknown option '$1' (see --help)" ;;
  esac
done

# ${name} and ${task} are spliced into paths and a command line below — reject, never sanitize.
ident='^[A-Za-z0-9][A-Za-z0-9._-]*$'
[[ $name =~ $ident ]] || die "invalid --name '${name}': letters/digits/._- only, must start alphanumeric"
[[ $task =~ $ident ]] || die "invalid --task '${task}': letters/digits/._- only, must start alphanumeric"
for v in num_gpus timeout_s interval; do
  [[ ${!v} =~ ^[0-9]+$ ]] || die "invalid --${v} '${!v}': must be a non-negative integer"
done

LOG="results/train_${name}.log"
PIDFILE="results/train_${name}.pid"

case "$cmd" in
  start)
    [[ -n "$recipe" ]] || die "start needs --recipe"
    [[ -f "$recipe" ]] || die "recipe not found: ${recipe}"
    scripts_dir="${scripts_dir:-anomalygen/scripts/texture}"
    [[ -f "${scripts_dir}/train.py" ]] || die "no train.py under --scripts '${scripts_dir}'"
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      die "a run for '${name}' is already alive (PID $(cat "$PIDFILE")); 'stop' it or pick another --name"
    fi
    mkdir -p results || die "cannot create ./results"

    # ${LOG} is keyed on --name alone, and 'wait' greps it from byte 0. Appending a second run onto
    # the first one's markers would make 'wait' return the PREVIOUS run's verdict within a second of
    # start: a finished run 1 yields a false FINISHED, which clears gate train (best_checkpoint.txt
    # still holds run 1's value until on_train_end) and sends Step 4 generating from the old
    # checkpoint — no error, plausible numbers, wrong model. Rotate so the live log is exactly one
    # run; the gate in verification.md greps this same path and stays correct for free.
    if [[ -s "$LOG" ]]; then
      rotated="${LOG}.$(date +%Y%m%d-%H%M%S)"
      mv "$LOG" "$rotated" || die "cannot rotate ${LOG}"
      printf 'rotated previous log -> %s\n' "$rotated"
    fi
    : > "$LOG" || die "cannot create ${LOG}"

    # setsid puts training in its own session. A plain '... &' already survives this script exiting —
    # it reparents to init either way — but stays in our session, so a harness that cleans up with
    # 'pkill -s <sid>' takes the run with it. That is what has reaped long runs here before.
    # The inner shell records its own PID and then execs, so ${PIDFILE} holds the real torchrun PID:
    # '$!' would name the setsid wrapper, which exits immediately. Every ${...} below is expanded by
    # THIS shell and passed as an argument, so the single-quoted inner script stays free of expansions.
    setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ \
      "$PIDFILE" \
      env IMAGINAIRE_OUTPUT_ROOT="$PWD/results" \
      torchrun --nproc_per_node="${num_gpus}" "${scripts_dir}/train.py" \
        --config=cosmos_framework/configs/base/config.py \
        --recipe="${recipe}" \
        -- experiment="anomalygen_${task}" "${overrides[@]}" \
      >> "$LOG" 2>&1 </dev/null &

    # The PID file is written by the detached shell, so it appears a moment after we return.
    for _ in $(seq 1 50); do
      [[ -s "$PIDFILE" ]] && break
      sleep 0.1
    done
    [[ -s "$PIDFILE" ]] || die "training did not start — see ${LOG}"
    printf 'started: PID %s, recipe %s, log %s\n' "$(cat "$PIDFILE")" "$recipe" "$LOG"
    printf "wait for it with: scripts/skill_utility/train_control.sh wait --name %s\n" "$name"
    ;;

  wait)
    [[ -f "$LOG" ]] || die "no log at ${LOG} — was 'start' run for '${name}'?"
    [[ -s "$PIDFILE" ]] || die "no PID file at ${PIDFILE} — was 'start' run for '${name}'?"
    pid="$(cat "$PIDFILE")"
    status=2
    SECONDS=0
    while :; do
      if grep -q "Done with training" "$LOG"; then status=0; break; fi
      if grep -qE "Traceback|Error executing job|Killed" "$LOG"; then status=1; break; fi
      if ! kill -0 "$pid" 2>/dev/null; then
        # It may have written the marker between the grep above and this check — re-read before
        # calling a clean finish a crash.
        if grep -q "Done with training" "$LOG"; then status=0; else status=1; fi
        break
      fi
      [[ "$SECONDS" -ge "$timeout_s" ]] && break     # still healthy, slice is up -> exit 2
      sleep "$interval"
    done

    # Both terminal verdicts are decided from the log, which the trainer writes *before* torchrun
    # finishes tearing down. Let the process go before returning: Step 4 generates on the same GPUs
    # and would otherwise race training's memory, and 'start' refuses a live PID, so returning early
    # makes the documented stop -> start resume path fail with "already alive". The markers remain
    # the verdict — a trainer that never exits still reports what the log says.
    if [[ "$status" -ne 2 ]]; then
      for _ in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
    fi
    tail -3 "$LOG"
    case "$status" in
      0) printf 'train: FINISHED (%s) — run the train gate, then Step 4\n' "$LOG" ;;
      1) printf 'train: FAILED (%s) — read the log; do not proceed\n' "$LOG" >&2 ;;
      2) printf 'train: RUNNING after %ss — call wait again\n' "$SECONDS" ;;
      # Unreachable: the loop above sets only 0, 1 or 2. Here so a later edit that adds a
      # verdict cannot exit silently with a status the caller has no message for.
      *) printf 'train: unknown status %s (%s)\n' "$status" "$LOG" >&2 ;;
    esac
    exit "$status"
    ;;

  stop)
    [[ -s "$PIDFILE" ]] || die "no PID file at ${PIDFILE}"
    pid="$(cat "$PIDFILE")"
    kill "$pid" 2>/dev/null || die "no live process for PID ${pid}"
    printf 'sent SIGTERM to %s; relaunching resumes from checkpoints/latest_checkpoint.txt\n' "$pid"
    ;;

  *) die "unknown command '${cmd}' (expected start, wait or stop)" ;;
esac
