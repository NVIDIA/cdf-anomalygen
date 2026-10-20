#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Control a locally-served model:
#   serverctl.sh {stop|health|wait} [PORT] [SERVED_MODEL_NAME] [--timeout SECONDS]
#
# `stop` matches the exact `--port N` argument rather than the model name. A broad `pkill -f gemma`
# also matches the shell running it, this script, and any editor with the recipe open; it has taken
# down a healthy server mid-run more than once.
#
# `wait` and `health` report the server's state as their EXIT CODE, the same three-way contract as
# scripts/skill_utility/train_control.sh and for the same reason: a cold start runs far longer than one
# foreground command, so "still coming up" must never be confused with "failed".
#
#   0  up       /v1/models answered                     -> `health`, then run Step 2 against it
#   1  dead     no `vllm serve --port N` process at all  -> read the server log; it is not coming
#   2  loading  process alive, socket not accepting yet  -> call again, as many times as it takes
#
# `health` adds two more, so a caller can tell "restart it" from "not yours":
#
#   3  other    serving, but not SERVED_MODEL_NAME       -> free the port or use another
#   4  dead engine  HTTP up, completion fails            -> the EngineCore died; restart the server
#
# and both `health` and `wait` report one more, immediately rather than after a timeout:
#
#   5  locked   the server answers 401                    -> it was started with an API_KEY this
#                                                            caller does not have (or has wrong)
#
# A server started with API_KEY set rejects unauthenticated probes with 401. Export the same API_KEY
# to this script and every probe carries it; without it, 401 is its own answer above, not "loading".
# Probes go straight to localhost (`--noproxy '*'`, a corporate http_proxy would swallow them) and
# are bounded (`--max-time`), so a wedged HTTP layer cannot hang the caller.
#
# --timeout defaults to 570s, just under a 10-minute tool budget, so a cold start takes two or three
# calls rather than one that outlives its caller. Everything before the socket binds — loading weights,
# torch.compile, CUDA-graph capture, chat-template detection — is invisible to curl: a measured
# single-GPU gemma-4-31b-it start first answered /v1/models about 14 minutes in, and ~3.5 of those
# minutes came AFTER the log said `Supported tasks: ['generate']`. Exit 2 throughout is correct.
#
# The distinction that makes this worth having is the process check. A bare failed curl reads exactly
# the same whether the server died or has not finished loading, and acting on the wrong reading is
# expensive in both directions: restarting a live server puts two vLLMs on one GPU at
# --gpu-memory-utilization 0.9, and waiting on a dead one waits forever.
set -u

CMD=${1:?stop|health|wait}
shift || true

TIMEOUT=570
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --timeout)
      TIMEOUT=${2:?--timeout needs a value in seconds}
      shift 2 ;;
    *)
      POS+=("$1")
      shift ;;
  esac
done
API_KEY=${API_KEY:-}
PORT=${POS[0]:-8902}
MODEL=${POS[1]:-nvidia/google/gemma-4-31b-it}

# Rejected, not coerced: `$((SECONDS + abc))` reads an unset name as 0, so a typo'd --timeout would
# silently turn `wait` into a single non-blocking probe that reports "still loading" and never waits.
case "$TIMEOUT" in
  '' | *[!0-9]*)
    echo "serverctl.sh: invalid --timeout '${TIMEOUT}': whole seconds only" >&2
    exit 64 ;;
esac

# The same pattern `stop` kills, so all three subcommands agree on what "this server" means.
#
# Only processes whose --port N is OUR localhost:N count. `pgrep -f` also sees a vLLM running inside
# a container on the same host, bound on the container's address: waiting on it never ends, and
# `stop` would kill it from outside. Two facts about a pid decide it, both from /proc:
#
#   /proc/<pid>/ns/net     readable only for our own uid. Readable and equal to ours -> ours (this
#                          includes a --net=host container, which IS on our port). Readable and
#                          different -> another network namespace, not ours.
#   /proc/<pid>/status     readable for every pid. Its NSpid line lists one pid per PID namespace
#                          the process is in; a container's process shows two or more.
#
# When the namespace link is unreadable the process belongs to another uid: a container's (nested
# NSpid, not ours) or another operator's server on this host (single NSpid, ours -- waiting on it,
# or refusing to start a duplicate on its GPUs, is exactly what this filter exists to do).
server_pids() {
  local mine pid theirs nspid
  mine=$(readlink /proc/self/ns/net)
  for pid in $(pgrep -f "vllm serve .*--port ${PORT}"); do
    if theirs=$(readlink /proc/"$pid"/ns/net 2>/dev/null); then
      [ "$theirs" = "$mine" ] && echo "$pid"
    else
      nspid=$(awk '/^NSpid:/{print NF-1}' /proc/"$pid"/status 2>/dev/null)
      [ "${nspid:-1}" -le 1 ] && echo "$pid"
    fi
  done
}
is_running() { [ -n "$(server_pids)" ]; }
PROBE_TIMEOUT=${PROBE_TIMEOUT:-5}            # /v1/models: a bound socket answers in milliseconds
COMPLETION_TIMEOUT=${COMPLETION_TIMEOUT:-90} # one 8-token completion on a loaded 31B model
# Every probe: silent, straight to localhost, bounded, and carrying the bearer when there is one.
# Callers append their own --max-time to override the default; curl takes the last one given.
CURL=(curl -s --noproxy '*' --max-time "${PROBE_TIMEOUT}" ${API_KEY:+-H "Authorization: Bearer ${API_KEY}"})
probe() { "${CURL[@]}" -f "$@"; }
# The HTTP status of /v1/models: 200 up, 401 locked, 000 nothing answered (or the probe timed out).
models_status() { "${CURL[@]}" -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/v1/models" 2>/dev/null || echo 000; }
models_up() { [ "$(models_status)" = "200" ]; }
locked() { echo "LOCKED (:${PORT} answers 401 — started with an API_KEY this caller does not have)"; exit 5; }

case "$CMD" in
  stop)
    pids=$(server_pids)
    if [ -n "$pids" ]; then
      echo "stopping: $pids"; kill $pids; sleep 8; kill -9 $pids 2>/dev/null || true
    else
      echo "no server on :${PORT}"
    fi
    ;;
  health)
    # Split the failed-curl case before reporting anything: DOWN, LOADING and LOCKED are different
    # answers and only some of them are problems.
    [ "$(models_status)" = "401" ] && locked
    if ! models_up; then
      if is_running; then
        echo "LOADING (process alive on :${PORT}, socket not accepting yet — see the server log)"
        exit 2
      fi
      echo "DOWN (no 'vllm serve --port ${PORT}' process)"
      exit 1
    fi
    probe "http://localhost:${PORT}/v1/models" | grep -q "$MODEL" \
      && echo "OK model listed" \
      || { echo "SERVING on :${PORT}, but '${MODEL}' is not the served name"; exit 3; }
    # /v1/models answering is NOT the same as the engine being alive: an EngineCore that died on a KV
    # cache exhaustion keeps the HTTP layer up and 500s every completion. Check a real completion.
    probe --max-time "${COMPLETION_TIMEOUT}" "http://localhost:${PORT}/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with: pong\"}],\"max_tokens\":8}" \
      | grep -qi "pong\|assistant" && echo "OK completion" || { echo "completion FAILED (engine dead?)"; exit 4; }
    ;;
  wait)
    end=$((SECONDS + TIMEOUT))
    misses=0
    while [ "$SECONDS" -lt "$end" ]; do
      case "$(models_status)" in
        200) echo "up after ${SECONDS}s"; exit 0 ;;
        401) locked ;;
      esac
      # Checked every pass, not once at the start: a server that dies mid-load must not be waited on
      # for the rest of the slice, and every later slice after that.
      #
      # TWO consecutive misses, not one. The recipe reaches `exec vllm serve ...` only after its env
      # setup, so for a moment after launch the process list holds the wrapper script and not yet a
      # cmdline this pattern matches — and `wait` is documented to be called immediately after
      # backgrounding it. One poll apart is enough to tell that window from a process that is genuinely
      # gone, and costs 10s in the case that really never started, against 1200s before.
      if is_running; then
        misses=0
      else
        misses=$((misses + 1))
        [ "$misses" -ge 2 ] && {
          echo "no 'vllm serve --port ${PORT}' process — it died or was never started"; exit 1; }
      fi
      sleep 10
    done
    echo "still loading after ${TIMEOUT}s — call wait again"
    exit 2 ;;
  *) echo "usage: $0 {stop|health|wait} [PORT] [MODEL] [--timeout SECONDS]"; exit 64 ;;
esac
