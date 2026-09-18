#!/usr/bin/env bash
# Same-day PCIe Gen 5 vs Gen 4 A/B of the standing production binary (bench-decode.sh, fixed prompt, N runs).
#   bash bench-pcie-ab.sh <tag>        e.g.  bash bench-pcie-ab.sh gen4-d1   (then James flips BIOS, reboots)  bash bench-pcie-ab.sh gen5-d1
# Records the link state and the replay counter before/after so each leg is self-describing; refuses a busy box.
TAG=${1:?tag like gen4-d1 / gen5-d1}; N=${2:-5}
cd /d/AI/ik_llama-qwen4exp || exit 1
ALIVE=$(powershell.exe -NoProfile -Command "(Get-Process llama-server,compute-sanitizer -ErrorAction SilentlyContinue|Measure-Object).Count" 2>/dev/null | tr -d '\r')
[ "${ALIVE:-0}" -eq 0 ] || { echo "[ab $TAG] ABORT: $ALIVE llama-server/sanitizer alive"; exit 2; }
LINK=$(nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.width.max,driver_version --format=csv,noheader | tr -d ' ')
R0=$(nvidia-smi -q | grep -m1 "Replays Since Reset" | grep -oE "[0-9]+$")
echo "[ab $TAG] link(hostmax,gpumax,width,driver)=$LINK replays_before=$R0 $(date +%T)"
bash pcie-telemetry.sh "ab-$TAG" 1800 > /dev/null 2>&1 &
TPID=$!
bash bench-decode.sh launch-standing-8099.ps1 "ab-$TAG" "$N" > "bench-ab-$TAG.out" 2>&1
RC=$?
kill "$TPID" 2>/dev/null
R1=$(nvidia-smi -q | grep -m1 "Replays Since Reset" | grep -oE "[0-9]+$")
echo "[ab $TAG] bench rc=$RC replays_after=$R1 (delta $((R1-R0))) $(date +%T)"
echo "[ab $TAG] $(paste -sd' ' "bench-ab-$TAG.summary.txt" 2>/dev/null | cut -c1-300)"
