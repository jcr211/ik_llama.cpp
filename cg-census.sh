#!/usr/bin/env bash
# CUDA-graph census for a build: bash cg-census.sh <launcher.ps1> <tag>
# Derives a variant launcher with LONGSPEAR_CG_DEBUG=1 (honest: never DEBUG2) and LONGSPEAR_CG_STATS=1 (G1 builds),
# runs bench-decode (8 warm-ups + 5 runs), then summarizes the [cg] lines: passes, upd= per step, vetoes by reason,
# revive transitions, and the LONGSPEAR_CG_STATS summary block if the build emits one.
LAUNCHER=${1:?launcher}; TAG=${2:?tag}; MODE=${3:-stats}   # stats = CG_STATS only (timings valid); debug = + CG_DEBUG per-split lines (timings VOID)
cd /d/AI/ik_llama-qwen4exp || exit 1
V="launch-cgcensus-$TAG-8099.ps1"
if [ "$MODE" = "debug" ]; then
  sed -E "s/^Remove-Item Env:LONGSPEAR_CG_DEBUG .*/\$env:LONGSPEAR_CG_DEBUG = '1'; \$env:LONGSPEAR_CG_STATS = '1'/" "$LAUNCHER" > "$V"
  grep -q "LONGSPEAR_CG_DEBUG = '1'" "$V" || { echo "[census] could not set CG_DEBUG in $V"; exit 2; }
else
  sed -E "s/^Remove-Item Env:LONGSPEAR_CG_DEBUG .*/Remove-Item Env:LONGSPEAR_CG_DEBUG -ErrorAction SilentlyContinue; \$env:LONGSPEAR_CG_STATS = '1'/" "$LAUNCHER" > "$V"
  grep -q "LONGSPEAR_CG_STATS = '1'" "$V" || { echo "[census] could not set CG_STATS in $V"; exit 2; }
fi
# the census owns the box for ~6 min: stop whatever llama-server is up (the standing one is relaunched at the end)
powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 8
bash bench-decode.sh "$V" "cg-$TAG" 5 > "bench-cg-$TAG.out" 2>&1
cp ik-serve-8099.err.log "cg-$TAG.err.log"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/launch-standing-8099.ps1" > "cg-$TAG-restore.out" 2>&1
echo "[census $TAG] bench: $(paste -sd' ' "bench-cg-$TAG.summary.txt" 2>/dev/null | cut -c1-160)"
echo "[census $TAG] cg pass lines: $(grep -c '^\[cg\] pass' "cg-$TAG.err.log")  upd=1: $(grep -c '^\[cg\] pass.*upd=1' "cg-$TAG.err.log")  upd=0: $(grep -c '^\[cg\] pass.*upd=0' "cg-$TAG.err.log")"
echo "[census $TAG] vetoes by reason:"; grep -oE '^\[cg\] veto[^ ]* [^ ]+' "cg-$TAG.err.log" | sort | uniq -c | sort -rn | head -8
echo "[census $TAG] revive/variant lines: $(grep -cE '^\[cg\] (revive|variant)' "cg-$TAG.err.log")"
grep -nE 'LONGSPEAR_CG_STATS|\[cg-stats\]' "cg-$TAG.err.log" | head -30 | cut -c1-200
