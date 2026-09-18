#!/usr/bin/env bash
# P-lite probe: does the server's RAM prompt cache remove the frozen-prefix re-prefill across "tasks"?
#   bash prefix-cache-probe.sh <launcher.ps1> <tag> [prefix-text-file]
# Sends 3 chat completions sharing one long system prefix (~20K tokens): prefix+A, prefix+B, prefix+A again.
# Reads the server's own "prompt eval time" per request and any [mtp] invalidate lines; saves completions
# 1 and 3 for a greedy-identity diff. Box must be free (no battery, no other consumer).
LAUNCHER=${1:?launcher basename}; TAG=${2:?tag}; PREFIX=${3:-/d/Projects/longspear/PRODUCT.md}
cd /d/AI/ik_llama-qwen4exp || exit 1
KEY=$(grep -oE 'sk-lm-[A-Za-z0-9]+' /d/AI/llama-swap/config.yaml | head -1)
ALIVE=$(powershell.exe -NoProfile -Command "(Get-Process llama-server,compute-sanitizer -ErrorAction SilentlyContinue|Measure-Object).Count" 2>/dev/null | tr -d '\r')
[ "${ALIVE:-0}" -eq 0 ] || { echo "[probe $TAG] ABORT: $ALIVE llama-server alive"; exit 2; }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/$LAUNCHER" > "probe-$TAG-launcher.out" 2>&1
for i in $(seq 1 60); do sleep 5; curl -s -m 3 http://127.0.0.1:8099/health 2>/dev/null | grep -q '"ok"' && break; done
echo "[probe $TAG] healthy; cache line: $(grep -m1 -oE 'prompt cache is [a-z]+[^\r]{0,40}' ik-serve-8099.out.log ik-serve-8099.err.log 2>/dev/null | head -1)"
# build a ~20K-token system prefix by repeating the prefix file (the model tokenizes ~4 chars/token on prose)
node -e '
const fs=require("fs"); const src=fs.readFileSync(process.argv[1],"utf8"); let p=""; while(p.length<80000) p+=src+"\n";
const mk=(task,name)=>fs.writeFileSync(name, JSON.stringify({model:"qwen3.8-flash-next", temperature:0, max_tokens:256, stream:false,
  messages:[{role:"system",content:"You are a careful engineer. Reference document follows.\n\n"+p},{role:"user",content:task}]}));
mk("Task A: list the three layers named in the document, one line each.","probe-A.json");
mk("Task B: name the two pillars of the north star in one sentence each.","probe-B.json");
' "$PREFIX"
LOGLINES0=$(wc -l < ik-serve-8099.err.log)
for r in 1 2 3; do
  body=probe-A.json; [ "$r" = 2 ] && body=probe-B.json
  curl -s -m 900 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' --data-binary @$body http://127.0.0.1:8099/v1/chat/completions > "probe-$TAG-run$r.json"
  echo "[probe $TAG] run$r ($body): $(grep -oE 'prompt eval time = +[0-9.]+ ms / +[0-9]+ tokens' ik-serve-8099.err.log | tail -1)  mtp: $(tail -n +$LOGLINES0 ik-serve-8099.err.log | grep -c '\[mtp\] companion invalidated')  cuda_errors: $(grep -cE 'CUDA error' ik-serve-8099.err.log)"
done
node -e 'const fs=require("fs");const a=JSON.parse(fs.readFileSync("probe-'"$TAG"'-run1.json")).choices[0].message.content,b=JSON.parse(fs.readFileSync("probe-'"$TAG"'-run3.json")).choices[0].message.content;console.log("[probe] run1 vs run3 greedy identity:", a===b?"IDENTICAL":"DIFFERENT ("+a.length+" vs "+b.length+" chars)")'
powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1
echo "[probe $TAG] done $(date +%T)"
