#!/usr/bin/env bash
# W-SL1 auto-stop chain (GPU-WINDOW.md §7). The coordinator runs it inside the granted GPU window, after the
# justify file is committed and the four launchers are copied into the fork dir; the lane never runs it live.
#   bash w-sl1-chain.sh <stamp>
# Steps, each followed by its gate; the first miss stops the chain:
#   0  VRAM preflight: crosscheck launcher (the larger config), one >= 131072-token request; STOP if the
#      1 Hz high-water > 32351 MiB, on a CUDA error or a PCIe replay increment.
#   1  probe: A2 + crosscheck native replay, stopped by a watcher once every j in 0..M-2 has >= 20 crosscheck
#      rows, or at PROBE_CAP verify rounds; gate exit 4 (a j value short) = INCONCLUSIVE: no arm runs.
#   2  arms P0-1 A2-1 A0-1 A2-2 P0-2 A0-2, a row gate after each, the pair gate after A2-1 and P0-2;
#      both pairs < +8 % -> KILLED at once; pairs of opposite sign -> preregistered third rep P0-3, A2-3
#      (then >= 2 of 3 pairs at +8 % -> candidate, else KILLED); both >= +8 % -> candidate; otherwise HOLD.
#   3  only for a candidate: bench-decode P0/A2 x 2, stepcost gate (a K=2..5 regression > 3 % blocks promotion).
#   4  fidelity (v2 statistical gate) is a manual step; the chain prints the reminder.
# On exit the chain stops telemetry and any llama-server and relaunches the standing server (RESTORE=0 skips).
# SIM=<dir> runs the decision logic on fixture logs (w-sl1-chain-selftest.sh): no server, no GPU, no restore.
set -u
STAMP=${1:?stamp}
LANE=/d/AI/worktrees/sl1-spec-ckpt/.lane
GATE="$LANE/sl1-gate.sh"
FORK=${FORK:-/d/AI/ik_llama-qwen4exp}
TASKS=${TASKS:-caller-omitted-refactor-01,duplicate-finalization-01}
BUDGET=${BUDGET:-20m}
M=5
PROBE_CAP=${PROBE_CAP:-600}
VRAM_LIMIT_MIB=32351
KILL_PCT=8
SIM=${SIM:-}
RESTORE=${RESTORE:-1}
[ -n "$SIM" ] && RESTORE=0

P=sl1-$STAMP
LOG=$FORK/w-sl1-$STAMP.chain.log
VERDICT="ABORTED"
TELEMETRY_PIDS=""

say() { echo "[w-sl1 $(date +%T)] $*" | tee -a "$LOG"; }

# ---- box operations (fixtures under SIM) -------------------------------------------------------------

telemetry_start() { # $1 tag; pcie always, mem trace when $2 = mem
    [ -n "$SIM" ] && { cp "$SIM/pcie.log" "$FORK/pcie-telemetry-$1.log" 2>/dev/null; return; }
    bash "$FORK/pcie-telemetry.sh" "$1" 14400 > /dev/null 2>&1 &
    TELEMETRY_PIDS="$TELEMETRY_PIDS $!"
    if [ "${2:-}" = mem ]; then
        bash "$FORK/mem-trace.sh" "$1" 14400 > /dev/null 2>&1 &
        TELEMETRY_PIDS="$TELEMETRY_PIDS $!"
    fi
}

telemetry_stop() {
    for p in $TELEMETRY_PIDS; do kill "$p" 2>/dev/null; done
    TELEMETRY_PIDS=""
    [ -z "$SIM" ] && sleep 2
}

servers_alive() {
    [ -n "$SIM" ] && { echo 0; return; }
    powershell.exe -NoProfile -Command "(Get-Process llama-server -ErrorAction SilentlyContinue|Measure-Object).Count" 2>/dev/null | tr -d '\r'
}

stop_server() {
    [ -n "$SIM" ] && return
    powershell.exe -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue" > /dev/null 2>&1
    for i in $(seq 1 24); do [ "$(servers_alive)" = 0 ] && return; sleep 5; done
}

wait_healthy() {
    [ -n "$SIM" ] && return 0
    for i in $(seq 1 60); do
        sleep 5
        curl -s -m 3 http://127.0.0.1:8099/health 2>/dev/null | grep -q '"ok"' && return 0
    done
    return 1
}

run_replay() { # $1 launcher basename, $2 tag -> replay-<tag>.err.log in FORK; prints the native-replay verdict
    if [ -n "$SIM" ]; then
        cp "$SIM/${2#$P-}.err.log" "$FORK/replay-$2.err.log" 2>/dev/null || { echo "verdict=INVALID"; return; }
        echo "verdict=CLEAN"; return
    fi
    bash "$FORK/native-replay.sh" "$1" "$2" "$TASKS" "$BUDGET" > "$FORK/w-sl1-$STAMP-$2.replay.out" 2>&1
    grep -oE "verdict=[A-Z-]+" "$FORK/w-sl1-$STAMP-$2.replay.out" | tail -1
}

on_exit() {
    telemetry_stop
    if [ "$RESTORE" = 1 ]; then
        say "restore: stopping any llama-server and relaunching launch-standing-8099.ps1"
        stop_server
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/launch-standing-8099.ps1" > "$FORK/w-sl1-$STAMP-restore.out" 2>&1
        wait_healthy && say "restore: standing server healthy" || say "restore: standing server NOT healthy in 300 s"
    fi
    say "VERDICT $VERDICT (log $LOG)"
}
trap on_exit EXIT

stop_chain() { # $1 verdict, rest = reason
    VERDICT=$1; shift
    say "STOP: $*"
    exit 1
}

gate() { # runs the gate, logs its output, returns its exit code
    local rc
    bash "$GATE" "$@" > "$FORK/w-sl1-$STAMP-gate.tmp" 2>&1
    rc=$?
    sed 's/^/    /' "$FORK/w-sl1-$STAMP-gate.tmp" | tee -a "$LOG"
    return $rc
}

# ---- preconditions ---------------------------------------------------------------------------------

: >> "$LOG"
say "start stamp=$STAMP sim=${SIM:-no} tasks=$TASKS budget=$BUDGET M=$M probe_cap=$PROBE_CAP"
if [ -z "$SIM" ]; then
    ls /d/Projects/longspear/bench/gpu-justify/*-spec-perstep.md > /dev/null 2>&1 || stop_chain ABORTED "no bench/gpu-justify/<date>-spec-perstep.md"
    for l in launch-perstep-8099.ps1 launch-perstep-p0-8099.ps1 launch-perstep-a0-8099.ps1 launch-perstep-xcheck-8099.ps1; do
        [ -f "$FORK/$l" ] || stop_chain ABORTED "launcher $l not copied into $FORK"
    done
    [ -f "$FORK/native-replay.sh" ] && [ -f "$FORK/bench-decode.sh" ] || stop_chain ABORTED "native-replay.sh / bench-decode.sh missing in $FORK"
    stop_server
    [ "$(servers_alive)" = 0 ] || stop_chain ABORTED "a llama-server is still alive"
fi

# ---- step 0: VRAM preflight --------------------------------------------------------------------------

say "step 0: VRAM preflight (crosscheck config, >= 131072-token request)"
telemetry_start "$P-preflight" mem
if [ -n "$SIM" ]; then
    cp "$SIM/mem-trace.log" "$FORK/mem-trace-$P-preflight.log"
    cp "$SIM/preflight.err.log" "$FORK/$P-preflight.err.log"
    LR=0
else
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/AI/ik_llama-qwen4exp/launch-perstep-xcheck-8099.ps1" > "$FORK/w-sl1-$STAMP-preflight-launch.out" 2>&1
    wait_healthy || stop_chain VOID "preflight server not healthy in 300 s"
    node "$LANE/sl1-long-request.mjs" --min 131072 --max 180000 --n 256 2>&1 | tee -a "$LOG"
    LR=${PIPESTATUS[0]}
    cp "$FORK/ik-serve-8099.err.log" "$FORK/$P-preflight.err.log"; cp "$FORK/ik-serve-8099.out.log" "$FORK/$P-preflight.out.log"
    stop_server
fi
telemetry_stop
HW=$(awk '$2 ~ /^[0-9]+$/ { if ($2 > m) m = $2 } END { print m + 0 }' "$FORK/mem-trace-$P-preflight.log")
say "preflight: long request rc=$LR, VRAM high-water ${HW} MiB (limit $VRAM_LIMIT_MIB)"
grep -E "per_step_alloc|checkpoint_alloc_shadows|fixed recurrent checkpoint mode|checkpoint capacity" "$FORK/$P-preflight.err.log" | sed 's/^/    /' | tee -a "$LOG"
[ "$LR" = 0 ] || stop_chain VOID "the long request did not complete (rc=$LR)"
[ "$HW" -gt 0 ] && [ "$HW" -le "$VRAM_LIMIT_MIB" ] || stop_chain KILLED "VRAM high-water $HW MiB > $VRAM_LIMIT_MIB (or no sample)"
[ "$(grep -c "CUDA error" "$FORK/$P-preflight.err.log")" = 0 ] || stop_chain KILLED "CUDA error during the preflight"
gate pcie "$FORK/pcie-telemetry-$P-preflight.log" || stop_chain VOID "PCIe replay increment during the preflight"

# ---- step 1: probe ---------------------------------------------------------------------------------

say "step 1: probe (A2 + crosscheck) until every j in 0..$((M-2)) has >= 20 crosscheck rows or $PROBE_CAP verify rounds"
telemetry_start "$P-probe"
if [ -n "$SIM" ]; then
    V=$(run_replay launch-perstep-xcheck-8099.ps1 "$P-probe")
    say "probe watcher (sim): $(bash "$GATE" jcount "$FORK/replay-$P-probe.err.log" "$M")"
else
    # the watcher reads the live server log: clear the preflight's lines first (its copy is saved above)
    : > "$FORK/ik-serve-8099.err.log"
    run_replay launch-perstep-xcheck-8099.ps1 "$P-probe" > "$FORK/w-sl1-$STAMP-probe.verdict" &
    RP=$!
    sleep 60
    while kill -0 "$RP" 2>/dev/null; do
        JC=$(bash "$GATE" jcount "$FORK/ik-serve-8099.err.log" "$M")
        R=$(echo "$JC" | grep -oE "rounds=[0-9]+" | cut -d= -f2); S=$(echo "$JC" | grep -oE "short=[0-9]+" | cut -d= -f2)
        if [ "${S:-1}" -eq 0 ] || [ "${R:-0}" -ge "$PROBE_CAP" ]; then
            say "probe watcher: target reached ($JC); stopping the probe server"
            stop_server
            break
        fi
        sleep 10
    done
    wait "$RP"
    V=$(cat "$FORK/w-sl1-$STAMP-probe.verdict")
fi
telemetry_stop
say "probe: native replay $V"
case "$V" in *CRASH*|*OOM*) stop_chain KILLED "probe $V" ;; esac
gate probe "$FORK/replay-$P-probe.err.log" "$M"
rc=$?
[ "$rc" = 4 ] && stop_chain INCONCLUSIVE "probe short of 20 crosscheck rows for some j at the cap: not a pass, no arm runs"
[ "$rc" = 0 ] || stop_chain KILLED "probe mechanism check missed (gate exit $rc)"
gate pcie "$FORK/pcie-telemetry-$P-probe.log" || stop_chain VOID "PCIe replay increment during the probe"

# ---- step 2: arms ------------------------------------------------------------------------------------

launcher_for() {
    case "$1" in
        P0) echo launch-perstep-p0-8099.ps1 ;;
        A2) echo launch-perstep-8099.ps1 ;;
        A0) echo launch-perstep-a0-8099.ps1 ;;
    esac
}

run_row() { # $1 arm, $2 rep
    local arm=$1 rep=$2 tag="$P-$1-$2" v
    say "row $arm-$rep"
    telemetry_start "$tag"
    v=$(run_replay "$(launcher_for "$arm")" "$tag")
    telemetry_stop
    say "row $arm-$rep: native replay $v"
    case "$v" in
        *CLEAN*) ;;
        *CRASH*|*OOM*) stop_chain KILLED "row $arm-$rep $v" ;;
        *) stop_chain VOID "row $arm-$rep $v (infra: fewer than 2 tasks started)" ;;
    esac
    gate row "$arm" "$FORK/replay-$tag.err.log" "$FORK/pcie-telemetry-$tag.log" || stop_chain KILLED "row $arm-$rep gate"
}

pair_gain() { # $1 P0 rep, $2 A2 rep -> sets GAIN; stops on a pair miss
    gate pair "$FORK/replay-$P-P0-$1.err.log" "$FORK/replay-$P-A2-$2.err.log" "$M" || stop_chain KILLED "pair P0-$1/A2-$2 (acceptance or drafts per verify)"
    GAIN=$(grep -oE "^GAIN_PCT=[-0-9.NA]+" "$FORK/w-sl1-$STAMP-gate.tmp" | cut -d= -f2)
    say "pair P0-$1/A2-$2: gain ${GAIN}%"
}

ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a != "NA" && a >= b) }'; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a != "NA" && a < b) }'; }

run_row P0 1
run_row A2 1
pair_gain 1 1; G1=$GAIN
run_row A0 1
run_row A2 2
run_row P0 2
pair_gain 2 2; G2=$GAIN
if lt "$G1" "$KILL_PCT" && lt "$G2" "$KILL_PCT"; then
    stop_chain KILLED "both A2 reps below +$KILL_PCT % (${G1}%, ${G2}%)"
fi
run_row A0 2

DECISION=""
if { lt "$G1" 0 && ge "$G2" 0; } || { ge "$G1" 0 && lt "$G2" 0; }; then
    say "reps disagree in direction (${G1}%, ${G2}%): preregistered third rep P0-3, A2-3"
    run_row P0 3
    run_row A2 3
    pair_gain 3 3; G3=$GAIN
    N_OK=0
    for g in "$G1" "$G2" "$G3"; do ge "$g" "$KILL_PCT" && N_OK=$((N_OK + 1)); done
    if [ "$N_OK" -ge 2 ]; then DECISION=CANDIDATE; else stop_chain KILLED "third rep: $N_OK of 3 pairs at +$KILL_PCT % (${G1}%, ${G2}%, ${G3}%)"; fi
elif ge "$G1" "$KILL_PCT" && ge "$G2" "$KILL_PCT"; then
    DECISION=CANDIDATE
else
    VERDICT="HOLD"
    say "HOLD: gains ${G1}% and ${G2}% meet neither the kill nor the promotion rule; step 3 cannot change a decision, skipped"
    exit 0
fi
say "arms: promotion candidate (gains ${G1}% ${G2}% ${G3:-}); mechanism counters in the row gates above"

# ---- step 3: fixed-context legs ------------------------------------------------------------------------

say "step 3: bench-decode P0/A2 x 2 (per-K step cost)"
for rep in 1 2; do
    for arm in P0 A2; do
        tag="$P-$arm-b$rep"
        telemetry_start "$tag"
        if [ -n "$SIM" ]; then
            cp "$SIM/$arm-b$rep.err.log" "$FORK/bench-$tag.err.log"
        else
            bash "$FORK/bench-decode.sh" "$(launcher_for "$arm")" "$tag" 5 > "$FORK/w-sl1-$STAMP-$tag.bench.out" 2>&1
            grep -E "MEDIAN|STEPCOST" "$FORK/w-sl1-$STAMP-$tag.bench.out" | sed 's/^/    /' | tee -a "$LOG"
        fi
        telemetry_stop
        [ "$(grep -c "CUDA error" "$FORK/bench-$tag.err.log" 2>/dev/null)" = 0 ] || stop_chain KILLED "CUDA error in $tag"
        gate pcie "$FORK/pcie-telemetry-$tag.log" || stop_chain VOID "PCIe replay increment in $tag"
    done
done
gate stepcost "$FORK/bench-$P-P0-b1.err.log,$FORK/bench-$P-P0-b2.err.log" "$FORK/bench-$P-A2-b1.err.log,$FORK/bench-$P-A2-b2.err.log"
rc=$?
if [ "$rc" = 3 ]; then
    VERDICT="CANDIDATE-BLOCKED"
    say "step 3: K=2..5 regression > 3 %: reported, promotion blocked"
    exit 0
fi
[ "$rc" = 0 ] || stop_chain VOID "stepcost gate error ($rc)"

VERDICT="CANDIDATE"
say "step 4 (manual): v2 statistical fidelity gate, A2 vs P0 on this binary, >= 12 prompts x 64 tokens (GPU-WINDOW.md §6); promotion then goes through the battery rule"
exit 0
