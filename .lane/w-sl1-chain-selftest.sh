#!/usr/bin/env bash
# Offline self-test of w-sl1-chain.sh's decision logic (SIM mode: fixture logs, no server, no GPU).
# One line per scenario: expected verdict, expected rows run / not run. Exit 1 on a mismatch.
set -u
CHAIN=/d/AI/worktrees/sl1-spec-ckpt/.lane/w-sl1-chain.sh
fail=0

round() { # mode K k_prop accepted restore_result redecode_n mtp_skip clamp xcheck
    echo "[spec-host] slot=0 mode=$1 K=$2 k_prop=$3 accepted=$4 restore_result=$5 redecode_n=$6 ckpt_init_us=12 ckpt_save_us=30 cells_copy_us=0 shadow_copy_us=0 sync_us=0 sampler_init_us=5 sampler_clone_us=9 restore_us=400 redecode_us=0 draft_host_us=2900 sample_us=700 mtp_skip=$7 clamp=$8 xcheck=$9"
}
vt() { echo "[vt] K=$1 n_kv=4000 mtp_op=0 us=60000 build=100 compute=$2 logits=10 embd=10 reused=1 nodes=3000 splits=40"; }
evalt() { echo "       eval time = $1 ms / 1400 tokens (   28.57 ms per token,    35.00 tokens per second)"; }
ms_for_gain() { awk -v g="$1" 'BEGIN { printf "%.2f", 50000/(1 + g/100) }'; }

probe_fixture() { # $1 file, $2 rows per j
    : > "$1"
    for i in $(seq 0 $((4 * $2 - 1))); do
        j=$((i % 4))
        round per-step 5 5 $j direct 0 0 0 1 >> "$1"
        echo "[ckpt-xcheck] j=$j comp=gdn_s n_bitequal=700000 n=786432 relL2=3.1e-04 max_layer_relL2=9e-04" >> "$1"
        echo "[ckpt-xcheck] j=$j comp=ple_tail n_bitequal=92160 n=92160 relL2=0 max_layer_relL2=0" >> "$1"
        vt 5 60000 >> "$1"
    done
    evalt 40000.00 >> "$1"
    echo "[ple-hist] set seq=0 next_pos=4000 n_prev=2 site=spec-replay" >> "$1"
}
arm_fixture() { # $1 file, $2 kind (P0|A2|A0|A2FB), $3 gain % (A2)
    : > "$1"
    for i in $(seq 1 100); do
        case "$2" in
            P0)   round gpu-fallback 5 5 2 replay 3 0 0 0 >> "$1" ;;
            A2)   round per-step 5 5 2 direct 0 0 0 0 >> "$1" ;;
            A2FB) round gpu-fallback 5 5 2 replay 3 0 0 0 >> "$1" ;;
            A0)   ;;
        esac
        vt 5 60000 >> "$1"
    done
    case "$2" in P0|A0|A2FB) evalt 50000.00 >> "$1" ;; A2) evalt "$(ms_for_gain "$3")" >> "$1" ;; esac
    echo "[ple-hist] set seq=0 next_pos=4000 n_prev=2 site=server-resume" >> "$1"
    [ "${RESET_IN:-}" = "$2" ] && echo "[ple-hist] reset seq=0 pos=4012 next_pos=4015" >> "$1"
    return 0
}
bench_fixture() { # $1 file, $2 compute
    : > "$1"
    for k in 2 3 4 5; do for i in 1 2 3; do vt $k "$2" >> "$1"; done; done
}

# $1 scenario name; env: G1 G2 G3 (A2 gains), PROBE_ROWS, HW, A2_1 (A2|A2FB), BENCH_A2
scenario() {
    local name=$1 want_verdict=$2 want_rows=$3 absent_rows=$4
    local S; S=$(mktemp -d); local F; F=$(mktemp -d)
    echo "time replays link_gen link_width power_w temp_c sm_mhz mem_mhz util_pct tx_kbs rx_kbs" > "$S/pcie.log"
    echo "12:00:00 3 4 16 400 60 2800 14000 90 100 100" >> "$S/pcie.log"
    echo "12:00:01 3 4 16 400 60 2800 14000 90 100 100" >> "$S/pcie.log"
    printf "12:00:00 1200\n12:00:01 %s\n12:00:02 1300\n" "${HW:-31000}" > "$S/mem-trace.log"
    echo "per_step_alloc:      CUDA0 per-step buffer =   454.52 MiB (max_tokens=5)" > "$S/preflight.err.log"
    probe_fixture "$S/probe.err.log" "${PROBE_ROWS:-30}"
    arm_fixture "$S/P0-1.err.log" P0;  arm_fixture "$S/P0-2.err.log" P0;  arm_fixture "$S/P0-3.err.log" P0
    arm_fixture "$S/A0-1.err.log" A0;  arm_fixture "$S/A0-2.err.log" A0
    arm_fixture "$S/A2-1.err.log" "${A2_1:-A2}" "${G1:-25}"
    arm_fixture "$S/A2-2.err.log" A2 "${G2:-25}"
    arm_fixture "$S/A2-3.err.log" A2 "${G3:-25}"
    for r in 1 2; do bench_fixture "$S/P0-b$r.err.log" 50000; bench_fixture "$S/A2-b$r.err.log" "${BENCH_A2:-50500}"; done

    SIM=$S FORK=$F bash "$CHAIN" "t$name" > "$F/chain.out" 2>&1
    local verdict; verdict=$(grep -oE "VERDICT [A-Z-]+" "$F/chain.out" | tail -1 | cut -d' ' -f2)
    local ok=1
    [ "$verdict" = "$want_verdict" ] || ok=0
    for r in $want_rows; do [ -f "$F/replay-sl1-t$name-$r.err.log" ] || ok=0; done
    for r in $absent_rows; do [ -f "$F/replay-sl1-t$name-$r.err.log" ] && ok=0; done
    grep -qE "command not found|syntax error|unbound variable|No such file" "$F/chain.out" && { ok=0; grep -E "command not found|syntax error|unbound variable|No such file" "$F/chain.out" | head -3; }
    if [ "$ok" = 1 ]; then echo "ok   $name: $verdict"; else echo "BAD  $name: verdict=$verdict want=$want_verdict"; tail -25 "$F/chain.out"; fail=1; fi
    rm -rf "$S" "$F"
}

scenario pass            CANDIDATE          "probe P0-1 A2-1 A0-1 A2-2 P0-2 A0-2" "P0-3 A2-3"
G1=3  G2=5  scenario both-below-8       KILLED        "P0-1 A2-1 A0-1 A2-2 P0-2" "A0-2 P0-3"
G1=20 G2=-5 G3=15 scenario disagree-third-ok  CANDIDATE     "A0-2 P0-3 A2-3" ""
G1=20 G2=-5 G3=2  scenario disagree-third-low KILLED        "P0-3 A2-3" ""
G1=12 G2=5  scenario hold               HOLD          "A0-2" "P0-3"
PROBE_ROWS=15 scenario probe-short      INCONCLUSIVE  "probe" "P0-1"
HW=32500 scenario vram-over             KILLED        "" "probe P0-1"
A2_1=A2FB scenario a2-without-per-step  KILLED        "P0-1 A2-1" "A0-1"
BENCH_A2=52500 scenario stepcost-regression CANDIDATE-BLOCKED "A0-2" ""
RESET_IN=P0 scenario p0-ple-hist-reset  KILLED        "P0-1" "A2-1"
exit $fail
