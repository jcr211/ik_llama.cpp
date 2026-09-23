#!/usr/bin/env bash
# Compile every SL-1 commit in order (llama-server; plus the SL-1 tests once they exist) in build-sl1,
# then return to the branch head. Stops, and kills its build, if a llama-server process appears.
# Output: .lane/per-commit-build.log (one "commit rc" line per commit) and .lane/per-commit.status.
# Run a COPY of this script (and of kill-build.ps1, via KILL=...) from outside the worktree: the checkouts
# remove the tracked .lane files of later commits while the loop runs.
set -u
KILL=${KILL:-/d/AI/worktrees/sl1-spec-ckpt/.lane/kill-build.ps1}
ROOT=/d/AI/worktrees/sl1-spec-ckpt
LOG=$ROOT/.lane/per-commit-build.log
STATUS=$ROOT/.lane/per-commit.status
ONE='C:\Users\jcrog\AppData\Local\Temp\claude\D--Projects-longspear\659ff728-69bf-4414-8821-d837656d1943\scratchpad\build-one.cmd'
BRANCH=lane/sl1-spec-ckpt
BASE=d583c220

cd "$ROOT" || exit 2
: > "$LOG"
echo "running" > "$STATUS"
overall=0
for c in $(git rev-list --reverse "$BASE..$BRANCH"); do
    subj=$(git log -1 --format=%s "$c")
    if ! git diff --name-only "$c~1" "$c" | grep -qE '\.(c|cpp|h|cu|cuh)$|CMakeLists'; then
        echo "$c skip (no C/C++ change) $subj" >> "$LOG"
        continue
    fi
    git -c advice.detachedHead=false checkout -q "$c" || { echo "$c checkout failed" >> "$LOG"; overall=1; break; }
    targets=llama-server
    if [ -f tests/test-ple-perstep.cpp ]; then
        targets="llama-server test-ple-perstep test-iqk-moe-chunks test-spec-ckpt-sampler"
    fi
    for t in $targets; do
        cmd.exe //d //c "$ONE $t" > "$ROOT/.lane/per-commit-$t.log" 2>&1 &
        bp=$!
        while kill -0 "$bp" 2>/dev/null; do
            if tasklist //NH //FI "IMAGENAME eq llama-server.exe" 2>/dev/null | grep -qi "llama-server"; then
                echo "PREEMPTED at $c: a llama-server started" >> "$LOG"
                pwsh -NoProfile -File "$KILL" >> "$LOG" 2>&1
                wait "$bp"
                git checkout -q "$BRANCH"
                echo "preempted" > "$STATUS"
                exit 3
            fi
            sleep 5
        done
        wait "$bp"
        rc=$?
        echo "$c $t rc=$rc $subj" >> "$LOG"
        if [ "$rc" != 0 ]; then
            overall=1
            grep -E "error C|error LNK|FAILED:" "$ROOT/.lane/per-commit-$t.log" | head -5 >> "$LOG"
        fi
    done
done
git checkout -q "$BRANCH"
echo "back on $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)" >> "$LOG"
if [ "$overall" = 0 ]; then echo "ok" > "$STATUS"; else echo "fail" > "$STATUS"; fi
exit "$overall"
