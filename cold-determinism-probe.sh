#!/usr/bin/env bash
# Cold-run determinism falsifier: the SAME prompt (probe-A.json, temp 0, 256 tokens) sent to a FRESH server
# three times (full restart between runs, so no slot/prefix reuse). Identical outputs = deterministic engine;
# differing outputs = run-to-run nondeterminism (reduction order), independent of prefix reuse.
#   bash cold-determinism-probe.sh <launcher.ps1> <tag>      (ends with the STANDING server relaunched)
LAUNCHER=${1:-launch-standing-8099.ps1}; TAG=${2:-cold}
cd /d/AI/ik_llama-qwen4exp || exit 1
KEY=$(grep -oE 'sk-lm-[A-Za-z0-9]+' /d/AI/llama-swap/config.yaml | head -1)
[ -f probe-A.json ] || { echo "[cold] probe-A.json missing (run prefix-cache-probe.sh once)"; exit 2; }
stop() { powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1; sleep 8; }
launch() { powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/$1" > "cold-$TAG-launcher.out" 2>&1; for i in $(seq 1 60); do sleep 5; curl -s -m 3 http://127.0.0.1:8099/health 2>/dev/null | grep -q '"ok"' && return 0; done; return 1; }
for r in 1 2 3; do
  stop; launch "$LAUNCHER" || { echo "[cold] run$r: server not healthy"; exit 3; }
  curl -s -m 900 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' --data-binary @probe-A.json http://127.0.0.1:8099/v1/chat/completions > "cold-$TAG-run$r.json"
  echo "[cold] run$r: $(grep -oE 'prompt eval time = +[0-9.]+ ms / +[0-9]+ tokens' ik-serve-8099.err.log | tail -1) cuda_errors=$(grep -cE 'CUDA error' ik-serve-8099.err.log)"
done
node -e 'const fs=require("fs");const t=i=>JSON.parse(fs.readFileSync("cold-'"$TAG"'-run"+i+".json")).choices[0].message.content;const a=t(1),b=t(2),c=t(3);
const eq=(x,y)=>x===y;let n=0;for(let i=0;i<Math.min(a.length,b.length)&&a[i]===b[i];i++)n++;
console.log("[cold] identity 1v2:"+(eq(a,b)?"IDENTICAL":"DIFFERENT")+" 1v3:"+(eq(a,c)?"IDENTICAL":"DIFFERENT")+" 2v3:"+(eq(b,c)?"IDENTICAL":"DIFFERENT")+" lengths "+a.length+"/"+b.length+"/"+c.length+" first divergence 1v2 at char "+n)'
stop; launch launch-standing-8099.ps1 && echo "[cold] standing :8099 restored $(date +%T)"
