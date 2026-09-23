#!/usr/bin/env bash
# SL-1 build wrapper: runs build-perstep.cmd into build-sl1 and aborts it the moment a llama-server
# process appears (a GPU window took the box; a compile must not load the CPU during a measurement).
# Usage: bash .lane/run-build.sh [server]
set -u
ROOT=/d/AI/worktrees/sl1-spec-ckpt
LOG=$ROOT/.lane/build.log
STATUS=$ROOT/.lane/build.status
ARG=${1:-}

: > "$LOG"
echo "running" > "$STATUS"
echo "[run-build] start $(date '+%F %T') arg=$ARG" >> "$LOG"

cmd.exe //d //c "D:\\AI\\worktrees\\sl1-spec-ckpt\\build-perstep.cmd $ARG" >> "$LOG" 2>&1 &
BPID=$!

preempted=0
while kill -0 "$BPID" 2>/dev/null; do
    if tasklist //NH //FI "IMAGENAME eq llama-server.exe" 2>/dev/null | grep -qi "llama-server"; then
        echo "[run-build] PREEMPTED $(date '+%F %T'): a llama-server started; killing the build" >> "$LOG"
        pwsh -NoProfile -File "$ROOT/.lane/kill-build.ps1" >> "$LOG" 2>&1
        preempted=1
        break
    fi
    sleep 5
done

wait "$BPID"
rc=$?
echo "[run-build] end $(date '+%F %T') rc=$rc preempted=$preempted" >> "$LOG"
if [ "$preempted" = 1 ]; then
    echo "preempted" > "$STATUS"
elif [ "$rc" = 0 ]; then
    echo "ok" > "$STATUS"
else
    echo "fail rc=$rc" > "$STATUS"
fi
exit "$rc"
