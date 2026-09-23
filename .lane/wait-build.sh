#!/usr/bin/env bash
# Poll the SL-1 build status for up to $1 seconds (default 540); print the status and the log tail.
ROOT=/d/AI/worktrees/sl1-spec-ckpt
LIMIT=${1:-540}
waited=0
while [ "$waited" -lt "$LIMIT" ]; do
    st=$(cat "$ROOT/.lane/build.status" 2>/dev/null)
    if [ "$st" != "running" ]; then
        break
    fi
    sleep 10
    waited=$((waited + 10))
done
echo "status=$(cat "$ROOT/.lane/build.status" 2>/dev/null) waited=${waited}s"
grep -c "\] Building\|\] Linking" "$ROOT/.lane/build.log" 2>/dev/null | sed 's/^/ninja steps logged: /'
grep -E "error C|error:|FAILED:|BUILD_PERSTEP|PREEMPTED|\[run-build\]" "$ROOT/.lane/build.log" | tail -15
tail -2 "$ROOT/.lane/build.log"
