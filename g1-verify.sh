#!/usr/bin/env bash
# G1 verification chain on build-g1 (box must be free of batteries): stats census (timings valid) on G1 and on
# the production binary, a second bench leg each (N=2), then the 4-task production replay on G1 under PCIe
# telemetry, then the standing server is restored. Log: g1-verify.out
cd /d/AI/ik_llama-qwen4exp || exit 1
echo "[g1v] start $(date +%T)"
powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 8
bash cg-census.sh launch-g1-8099.ps1 g1a stats 2>&1 | grep -E "^\[census|cg-stats\] total|cg-stats\] begin" | cut -c1-240
sleep 8; powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 8
bash cg-census.sh launch-standing-8099.ps1 prod38s stats 2>&1 | grep -E "^\[census|cg-stats\] total|cg-stats\] begin" | cut -c1-240
sleep 8; powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 8
bash bench-decode.sh launch-g1-8099.ps1 g1b 5 > bench-g1b.out 2>&1; echo "[g1v] g1b: $(paste -sd' ' bench-g1b.summary.txt 2>/dev/null | cut -c1-200)"; sleep 8
bash bench-decode.sh launch-standing-8099.ps1 ctl-g1 5 > bench-ctl-g1.out 2>&1; echo "[g1v] ctl-g1: $(paste -sd' ' bench-ctl-g1.summary.txt 2>/dev/null | cut -c1-200)"; sleep 8
bash pcie-telemetry.sh g1-replay 7200 > /dev/null 2>&1 &
TPID=$!
bash native-replay.sh launch-g1-8099.ps1 g1 caller-omitted-refactor-01,duplicate-finalization-01,HB-C-03,HB-C-05 20m > matrix-g1.out 2>&1
kill $TPID 2>/dev/null
grep -E "verdict|CRASH|battery rc" matrix-g1.out | cut -c1-200
echo "[g1v] replay replays: $(awk 'NR>1{if($2>m)m=$2} END{print m+0}' pcie-telemetry-g1-replay.log)  g1 stats at exit: $(grep -E 'cg-stats\] total passes' replay-g1.err.log | tail -1 | cut -c1-200)"
sleep 8
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/launch-standing-8099.ps1" > restore-standing-g1.out 2>&1
for i in $(seq 1 60); do sleep 5; curl -s -m 3 http://127.0.0.1:8099/health 2>/dev/null | grep -q '"ok"' && { echo "[g1v] standing :8099 restored $(date +%T)"; break; }; done
echo "[g1v] done $(date +%T)"
