#!/usr/bin/env bash
# 1 Hz GPU memory trace: bash mem-trace.sh <tag> [seconds]  -> mem-trace-<tag>.log (time used_mib); prints the high-water mark at the end
TAG=${1:?tag}; SECS=${2:-1800}
OUT=/d/AI/ik_llama-qwen4exp/mem-trace-$TAG.log; : > "$OUT"
for i in $(seq 1 "$SECS"); do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> "$OUT"; sleep 1; done
