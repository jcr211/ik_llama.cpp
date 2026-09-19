#!/usr/bin/env bash
# Derive launcher variants from the standing launcher (same dir; the API key stays inside the files, never printed).
cd /d/AI/ik_llama-qwen4exp || exit 1
S=launch-standing-8099.ps1
sed -E 's/-ncmoe 38/-ncmoe 37/' "$S" > launch-ncmoe37-8099.ps1
sed -E 's/-ncmoe 38/-ncmoe 36/' "$S" > launch-ncmoe36-8099.ps1
sed -E 's/-ncmoe 38/-ncmoe 36/; s/-c 196608/-c 163840/' "$S" > launch-ncmoe36-160k-8099.ps1
sed -E 's/-ncmoe 38/-ncmoe 36/; s/-c 196608/-c 147456/' "$S" > launch-ncmoe36-150k-8099.ps1
sed -E 's/-rtr -muge/-rtr -muge --cache-ram 0/' "$S" > launch-cacheoff-8099.ps1
for f in launch-ncmoe37-8099.ps1 launch-ncmoe36-8099.ps1 launch-ncmoe36-160k-8099.ps1 launch-ncmoe36-150k-8099.ps1 launch-cacheoff-8099.ps1; do
  printf "%s: %s\n" "$f" "$(grep -oE '(-ncmoe [0-9]+|-c [0-9]+|--cache-ram [0-9]+)' "$f" | paste -sd' ')"
done
