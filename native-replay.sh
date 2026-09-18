#!/usr/bin/env bash
# Native-speed replay of battery tasks against a launcher variant (no sanitizer): crash-or-clean bisect arm.
#   bash native-replay.sh <launcher.ps1 basename> <tag> [tasks] [budget]
# Launches via the ps1 (logs -> ik-serve-8099.{out,err}.log), waits for health, runs the battery runner on the
# tasks (sentinel on), then stops the server and archives the logs as replay-<tag>.{out,err}.log.
LAUNCHER=${1:?launcher}; TAG=${2:?tag}; TASKS=${3:-caller-omitted-refactor-01,duplicate-finalization-01}; BUDGET=${4:-15m}
cd /d/AI/ik_llama-qwen4exp || exit 1
KEY=$(grep -oE 'sk-lm-[A-Za-z0-9]+' /d/AI/llama-swap/config.yaml | head -1)
ALIVE=$(powershell.exe -NoProfile -Command "(Get-Process llama-server,compute-sanitizer -ErrorAction SilentlyContinue|Measure-Object).Count" 2>/dev/null | tr -d '\r')
[ "${ALIVE:-0}" -eq 0 ] || { echo "[replay $TAG] ABORT: $ALIVE llama-server/sanitizer alive"; exit 2; }
echo "[replay $TAG] launcher=$LAUNCHER tasks=$TASKS budget=$BUDGET spec=$(grep -oE 'spec-type [a-z-]+' "$LAUNCHER" | paste -sd, ) graphs-off=$(grep -c "GGML_CUDA_DISABLE_GRAPHS = '1'" "$LAUNCHER") $(date +%T)"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/$LAUNCHER" > "replay-$TAG-launcher.out" 2>&1
OK=0; for i in $(seq 1 60); do sleep 5; curl -s -m 3 http://127.0.0.1:8099/health 2>/dev/null | grep -q '"ok"' && { OK=1; break; }; done
[ "$OK" -eq 1 ] || { echo "[replay $TAG] server not healthy in 300 s"; exit 3; }
echo "[replay $TAG] healthy after ~$((i*5)) s"
# pre-warm: 3 requests (JIT-compiles PTX-only builds, fills the ngram table, warms graphs) so the battery's
# quiet-box sentinel measures steady state, not first-launch cost
[ -f bench-body.json ] || node make-bench-body.mjs
for w in 1 2 3; do curl -s -o /dev/null -m 900 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' --data-binary @bench-body.json http://127.0.0.1:8099/v1/chat/completions; done
echo "[replay $TAG] pre-warm done $(date +%T)"
cd /d/Projects/longspear || exit 1
export LONGSPEAR_API_KEY="$KEY"
# lift undici's 300 s response-headers timeout (slow diagnostic servers: per-op sync, sanitizer)
export NODE_OPTIONS="--require D:/Projects/longspear/lab/battery/preload-undici-timeout.cjs"
ID="replay-$TAG-$(date +%Y%m%d-%H%M)"
# RECORD=1: route the harness through the recording proxy (:8098 -> :8099) so the request sequence becomes a
# harness-free reproducer (lab/battery/record-proxy.mjs + replay-recorded.mjs); log = sessions/repro/<ID>.jsonl
EP=""; PPID_REC=""
if [ "${RECORD:-0}" = "1" ]; then
  mkdir -p sessions/repro
  node lab/battery/record-proxy.mjs --listen 8098 --target http://127.0.0.1:8099 --out "sessions/repro/$ID.jsonl" > "sessions/repro/$ID.proxy.log" 2>&1 &
  PPID_REC=$!; sleep 2; EP="--endpoint http://127.0.0.1:8098/v1"
  echo "[replay $TAG] recording proxy pid=$PPID_REC -> sessions/repro/$ID.jsonl"
fi
env -u LONGSPEAR_BATTERY_AGENT_OPTS npm run battery -- --tasks "$TASKS" --budget "$BUDGET" --max-turns 48 --run-id "$ID" --stop-on-infra --agent longspear --controller $EP > "/d/AI/ik_llama-qwen4exp/replay-$TAG-battery.out" 2>&1
RC=$?
[ -n "$PPID_REC" ] && { kill "$PPID_REC" 2>/dev/null; echo "[replay $TAG] recorded $(grep -c '"body"' "sessions/repro/$ID.jsonl" 2>/dev/null) requests"; }
cd /d/AI/ik_llama-qwen4exp || exit 1
CE=$(grep -cE "CUDA error" ik-serve-8099.err.log)
# the error STRING matters: an out-of-memory is an engine/config failure, not the driver launch-failure class
ERRS=$(grep -m1 -oE "CUDA error: [a-z ]+" ik-serve-8099.err.log)
TS=$(ls /d/Projects/longspear/lab/battery/results/$ID/ 2>/dev/null | grep -c events.jsonl)
echo "[replay $TAG] battery rc=$RC cuda_errors=$CE first_error='${ERRS}' tasks_started=$TS $(date +%T)"
grep -E "INFRA:|stop-on-infra" "replay-$TAG-battery.out" | head -2 | cut -c1-200
grep -oE "'(PASS|FAIL)' *│ '[A-Za-z/]+' *│ '[0-9.]+s' *│ [0-9]+ *│ '[a-z-]+'" "replay-$TAG-battery.out" | head -4
[ "$CE" -gt 0 ] && { C=$(grep -n -m1 "CUDA error" ik-serve-8099.err.log | cut -d: -f1); echo "[replay $TAG] CRASH context:"; sed -n "$((C-3)),$((C+2))p" ik-serve-8099.err.log | cut -c1-160; grep -E "kv cache rm" ik-serve-8099.out.log | grep -oE "id_task=[0-9]+ p0=[0-9]+" | tail -2 | paste -sd' '; }
cp ik-serve-8099.err.log "replay-$TAG.err.log"; cp ik-serve-8099.out.log "replay-$TAG.out.log"
powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1; sleep 5
if [ "$CE" -gt 0 ] && echo "$ERRS" | grep -q "out of memory"; then V=OOM-VOID; elif [ "$CE" -gt 0 ]; then V=CRASH; elif [ "${TS:-0}" -ge 2 ]; then V=CLEAN; else V=INVALID; fi
echo "[replay $TAG] done $(date +%T) verdict=$V first_error='${ERRS}' (tasks_started=$TS)"
