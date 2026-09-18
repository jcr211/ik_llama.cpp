#!/usr/bin/env bash
# S5 campaign on driver 616.56 / PCIe Gen 4: wait for bench-decode to finish -> relaunch standing :8099 with a
# short heavy-prefill smoke -> PCIe replay telemetry at 1 Hz -> run-s5.sh P 2 (stamp A) -> run-s5.sh S 2 (stamp B).
# Arms are frozen per stamp (run-s5.sh), so P and S get distinct stamps; analyze with
#   node scripts/analyze-r3.mjs --campaign s5 --files <P-rows,S-rows>
cd /d/AI/ik_llama-qwen4exp || exit 1
echo "[s5-chain] waiting for bench-gen4 to finish $(date +%T)"
for i in $(seq 1 240); do [ -f bench-gen4.summary.txt ] && break; sleep 5; done
[ -f bench-gen4.summary.txt ] || { echo "[s5-chain] ABORT: bench never finished"; exit 2; }
echo "[s5-chain] bench summary: $(paste -sd' ' bench-gen4.summary.txt | cut -c1-200)"
for i in $(seq 1 24); do
  ALIVE=$(powershell.exe -NoProfile -Command "(Get-Process llama-server,compute-sanitizer -ErrorAction SilentlyContinue|Measure-Object).Count" 2>/dev/null | tr -d '\r')
  [ "${ALIVE:-0}" -eq 0 ] && break; sleep 5
done
bash relaunch-and-smoke.sh 4096 > s5-gen4-relaunch.out 2>&1 || { echo "[s5-chain] ABORT: relaunch/smoke failed rc=$?"; tail -5 s5-gen4-relaunch.out; exit 2; }
echo "[s5-chain] :8099 relaunched + smoked $(date +%T); replays now: $(nvidia-smi -q | grep -m1 'Replays Since Reset' | grep -oE '[0-9]+$')"
bash pcie-telemetry.sh s5-gen4 36000 > /dev/null 2>&1 &
TPID=$!
export LONGSPEAR_API_KEY=$(grep -oE 'sk-lm-[A-Za-z0-9]+' /d/AI/llama-swap/config.yaml | head -1)
cd /d/Projects/longspear || exit 1
mkdir -p sessions/s5-gen4-20260918
STAMP_P=$(date +%Y%m%d-%H%M)
echo "[s5-chain] P 2 stamp=$STAMP_P $(date +%T)"
S5_STAMP=$STAMP_P bash scripts/run-s5.sh P 2 > "sessions/s5-gen4-20260918/P-$STAMP_P.log" 2>&1
echo "[s5-chain] P done rc=$? $(date +%T) replays: $(nvidia-smi -q | grep -m1 'Replays Since Reset' | grep -oE '[0-9]+$')"
sleep 30
STAMP_S=$(date +%Y%m%d-%H%M)
echo "[s5-chain] S 2 stamp=$STAMP_S $(date +%T)"
S5_STAMP=$STAMP_S bash scripts/run-s5.sh S 2 > "sessions/s5-gen4-20260918/S-$STAMP_S.log" 2>&1
echo "[s5-chain] S done rc=$? $(date +%T) replays: $(nvidia-smi -q | grep -m1 'Replays Since Reset' | grep -oE '[0-9]+$')"
kill "$TPID" 2>/dev/null
echo "[s5-chain] campaign done $(date +%T): P=$STAMP_P S=$STAMP_S; analyze: node scripts/analyze-r3.mjs --campaign s5 --files <P,S>"
