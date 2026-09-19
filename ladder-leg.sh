#!/usr/bin/env bash
# One expert-placement ladder leg: bash ladder-leg.sh <launcher.ps1> <tag>
# 1) launch variant, health, record model/KV buffer sizes + post-load VRAM; 2) 32K prefill smoke (256 gen) under a
# 1 Hz memory trace -> high-water mark; 3) stop; 4) bench-decode.sh (self-contained: launch, 8 warm-ups, 5 runs, stop).
LAUNCHER=${1:?launcher}; TAG=${2:?tag}
cd /d/AI/ik_llama-qwen4exp || exit 1
ALIVE=$(powershell.exe -NoProfile -Command "(Get-Process llama-server,compute-sanitizer -ErrorAction SilentlyContinue|Measure-Object).Count" 2>/dev/null | tr -d '\r')
[ "${ALIVE:-0}" -eq 0 ] || { echo "[leg $TAG] ABORT: $ALIVE llama-server alive"; exit 2; }
echo "[leg $TAG] launcher=$LAUNCHER $(grep -oE '(-ncmoe [0-9]+|-c [0-9]+)' "$LAUNCHER" | paste -sd' ') $(date +%T)"
bash mem-trace.sh "$TAG-smoke" 1500 > /dev/null 2>&1 &
MPID=$!
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/$LAUNCHER" > "leg-$TAG-launcher.out" 2>&1
OK=0; for i in $(seq 1 60); do sleep 5; curl -s -m 3 http://127.0.0.1:8099/health 2>/dev/null | grep -q '"ok"' && { OK=1; break; }; done
if [ "$OK" -ne 1 ]; then echo "[leg $TAG] server not healthy in 300 s (OOM at load?)"; tail -3 ik-serve-8099.err.log | cut -c1-160; kill $MPID 2>/dev/null; exit 3; fi
echo "[leg $TAG] loaded: $(grep -oE 'CUDA0 model buffer size = +[0-9.]+ MiB' ik-serve-8099.err.log | tail -1) | $(grep -oE 'CUDA0 KV buffer size = +[0-9.]+ MiB' ik-serve-8099.err.log | tail -1) | vram_after_load=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ') MiB"
MAXTOK=256 bash smoke-heavy-prefill.sh 32768 > "leg-$TAG-smoke.out" 2>&1
SRC=$?
echo "[leg $TAG] smoke rc=$SRC cuda_errors=$(grep -cE 'CUDA error' ik-serve-8099.err.log) $(grep -oE 'prompt eval time = +[0-9.]+ ms / +[0-9]+ tokens' ik-serve-8099.err.log | tail -1)"
powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 6
kill $MPID 2>/dev/null
echo "[leg $TAG] smoke high-water: $(awk 'NR>0 && $2>m{m=$2} END{print m+0}' "mem-trace-$TAG-smoke.log") MiB of 32607"
cp ik-serve-8099.err.log "leg-$TAG.err.log"
[ "$SRC" -eq 0 ] || { echo "[leg $TAG] smoke FAILED — bench skipped"; exit 4; }
bash mem-trace.sh "$TAG-bench" 1500 > /dev/null 2>&1 &
MPID=$!
bash bench-decode.sh "$LAUNCHER" "$TAG" 5 > "bench-$TAG.out" 2>&1
kill $MPID 2>/dev/null
echo "[leg $TAG] bench high-water: $(awk '$2>m{m=$2} END{print m+0}' "mem-trace-$TAG-bench.log") MiB | $(paste -sd' ' "bench-$TAG.summary.txt" 2>/dev/null | cut -c1-220)"
echo "[leg $TAG] done $(date +%T)"
