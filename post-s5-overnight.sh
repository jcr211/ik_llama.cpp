#!/usr/bin/env bash
# Overnight box program after the S5 campaign: analyzer -> cache-truth probe (on/off) -> expert-placement ladder
# (37@192K, 36@192K, 36@160K, 36@150K) -> restore the standing :8099. Each GPU step is self-guarded; the box is never
# left without a server at the end. Log: post-s5-overnight.out
cd /d/AI/ik_llama-qwen4exp || exit 1
echo "[overnight] waiting for the S5 chain to finish $(date +%T)"
for i in $(seq 1 720); do grep -q "campaign done" s5-chain-gen4.out 2>/dev/null && break; sleep 30; done
grep -q "campaign done" s5-chain-gen4.out || { echo "[overnight] ABORT: campaign not done after 6 h"; exit 2; }
echo "[overnight] campaign done $(date +%T): $(grep 'campaign done' s5-chain-gen4.out | cut -c1-120)"
sleep 20
# 1. analyzer on the four rows (CPU)
cd /d/Projects/longspear || exit 1
node scripts/analyze-r3.mjs --campaign s5 --files lab/battery/results/s5-P-1-20260918-1330.json,lab/battery/results/s5-P-2-20260918-1330.json,lab/battery/results/s5-S-1-20260918-1857.json,lab/battery/results/s5-S-2-20260918-1857.json > sessions/s5-gen4-20260918/analysis.txt 2>&1
echo "[overnight] analyzer rc=$? -> sessions/s5-gen4-20260918/analysis.txt; verdict lines:"; grep -iE "verdict|decision|non-regress|regress|adopt|mismatch" sessions/s5-gen4-20260918/analysis.txt | head -8 | cut -c1-160
cd /d/AI/ik_llama-qwen4exp || exit 1
powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 8
# 2. cache-truth probe: prompt cache ON (default 8192) vs OFF (--cache-ram 0)
bash prefix-cache-probe.sh launch-standing-8099.ps1 cacheon > probe-cacheon.out 2>&1; grep -E "^\[probe" probe-cacheon.out | cut -c1-200
sleep 8
bash prefix-cache-probe.sh launch-cacheoff-8099.ps1 cacheoff > probe-cacheoff.out 2>&1; grep -E "^\[probe" probe-cacheoff.out | cut -c1-200
sleep 8
# 3. expert-placement ladder (each leg self-guards; an OOM at load or in the smoke is reported and the ladder continues)
for leg in "launch-ncmoe37-8099.ps1 ncmoe37" "launch-ncmoe36-8099.ps1 ncmoe36" "launch-ncmoe36-160k-8099.ps1 ncmoe36-160k" "launch-ncmoe36-150k-8099.ps1 ncmoe36-150k"; do
  set -- $leg
  bash ladder-leg.sh "$1" "$2" 2>&1 | grep -E "^\[leg" | cut -c1-260
  powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 8
done
# 4. restore the standing server
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/launch-standing-8099.ps1" > restore-standing.out 2>&1
for i in $(seq 1 60); do sleep 5; curl -s -m 3 http://127.0.0.1:8099/health 2>/dev/null | grep -q '"ok"' && { echo "[overnight] standing :8099 restored $(date +%T)"; break; }; done
echo "[overnight] done $(date +%T) replays=$(nvidia-smi -q | grep -m1 'Replays Since Reset' | grep -oE '[0-9]+$')"
