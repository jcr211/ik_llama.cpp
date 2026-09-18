#!/usr/bin/env bash
# 1 Hz GPU telemetry with the PCIe replay counter, for correlating link errors with load and with the crash.
#   bash pcie-telemetry.sh <tag> [seconds]   -> pcie-telemetry-<tag>.log  (time, replays, gen, width, power, temp, sm, mem, util, tx/rx KB/s)
TAG=${1:?tag}; SECS=${2:-7200}
OUT=/d/AI/ik_llama-qwen4exp/pcie-telemetry-$TAG.log
echo "time replays link_gen link_width power_w temp_c sm_mhz mem_mhz util_pct tx_kbs rx_kbs" > "$OUT"
for i in $(seq 1 "$SECS"); do
  R=$(nvidia-smi -q 2>/dev/null | grep -m1 "Replays Since Reset" | grep -oE "[0-9]+$")
  Q=$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')
  T=$(nvidia-smi -q 2>/dev/null | grep -E "Tx Throughput|Rx Throughput" | grep -oE "[0-9]+ KB/s" | grep -oE "[0-9]+" | paste -sd' ')
  echo "$(date +%H:%M:%S) ${R:-NA} $Q ${T:-NA NA}" >> "$OUT"
  sleep 1
done
