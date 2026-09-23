#!/usr/bin/env bash
# Offline self-test of sl1-gate.sh on synthetic telemetry (no GPU): a passing probe, a probe whose PLE
# tail diverges, a probe that replays, a row, and a pair. Prints one line per case; exit 1 on a mismatch.
set -u
DIR=$(mktemp -d)
GATE=/d/AI/worktrees/sl1-spec-ckpt/.lane/sl1-gate.sh
fail=0

mk_round() { # $1 mode $2 K $3 accepted $4 restore_result $5 redecode_n $6 mtp_skip $7 clamp $8 xcheck
    echo "[spec-host] slot=0 mode=$1 K=$2 accepted=$3 restore_result=$4 redecode_n=$5 ckpt_init_us=12 ckpt_save_us=30 cells_copy_us=0 shadow_copy_us=0 sync_us=0 sampler_init_us=5 sampler_clone_us=9 restore_us=400 redecode_us=0 draft_host_us=2900 sample_us=700 mtp_skip=$6 clamp=$7 xcheck=$8"
}

mk_probe() { # $1 file, $2 ple mode (equal|diverge), $3 replay (0|1)
    : > "$1"
    for i in $(seq 0 119); do
        j=$((i % 4)); K=5
        if [ "$3" = 1 ] && [ "$i" -lt 10 ]; then
            mk_round per-step $K $j replay $((j+1)) 0 0 0 >> "$1"
        else
            mk_round per-step $K $j direct 0 0 0 1 >> "$1"
        fi
        echo "[ckpt-xcheck] j=$j comp=gdn_s n_bitequal=700000 n=786432 relL2=3.1e-04 max_layer_relL2=9e-04" >> "$1"
        echo "[ckpt-xcheck] j=$j comp=gdn_conv n_bitequal=30720 n=30720 relL2=0 max_layer_relL2=0" >> "$1"
        if [ "$2" = equal ]; then
            echo "[ckpt-xcheck] j=$j comp=ple_tail n_bitequal=92160 n=92160 relL2=0 max_layer_relL2=0" >> "$1"
        else
            echo "[ckpt-xcheck] j=$j comp=ple_tail n_bitequal=100 n=92160 relL2=8.0e-01 max_layer_relL2=8.0e-01" >> "$1"
        fi
        echo "[vt] K=$K n_kv=4000 mtp_op=0 us=60000 build=100 compute=50000 logits=10 embd=10 reused=1 nodes=3000 splits=40" >> "$1"
    done
    echo "       eval time =   40000.00 ms /  1400 tokens (   28.57 ms per token,    35.00 tokens per second)" >> "$1"
}

expect() { # $1 expected exit, $2 description, rest = command
    local want=$1 what=$2; shift 2
    "$@" > "$DIR/out.txt" 2>&1
    local got=$?
    if [ "$got" = "$want" ]; then echo "ok   $what (exit $got)"; else echo "BAD  $what (exit $got, want $want)"; cat "$DIR/out.txt"; fail=1; fi
}

mk_probe "$DIR/pass.err.log" equal 0
expect 0 "probe: per-step, bit-equal PLE tail, j 0..3 x30" bash "$GATE" probe "$DIR/pass.err.log" 5
mk_probe "$DIR/ple.err.log" diverge 0
expect 1 "probe: PLE tail diverges (relL2 0.8)" bash "$GATE" probe "$DIR/ple.err.log" 5
mk_probe "$DIR/replay.err.log" equal 1
expect 1 "probe: 10 of 120 rejected rounds replayed" bash "$GATE" probe "$DIR/replay.err.log" 5
expect 1 "probe: M=9 needs j up to 7" bash "$GATE" probe "$DIR/pass.err.log" 9

# row: an A2 row with no replays passes; a gpu-fallback row has no per-step check
expect 0 "row: A2 with 0 replay calls" bash "$GATE" row "$DIR/pass.err.log"

# pair: P0 gpu-fallback vs A2
: > "$DIR/p0.err.log"
for i in $(seq 1 100); do mk_round gpu-fallback 5 2 replay 3 0 0 0 >> "$DIR/p0.err.log"; done
echo "       eval time =   50000.00 ms /  1400 tokens (   35.71 ms per token,    28.00 tokens per second)" >> "$DIR/p0.err.log"
: > "$DIR/a2.err.log"
for i in $(seq 1 100); do mk_round per-step 5 2 direct 0 0 0 0 >> "$DIR/a2.err.log"; done
echo "       eval time =   40000.00 ms /  1400 tokens (   28.57 ms per token,    35.00 tokens per second)" >> "$DIR/a2.err.log"
expect 0 "pair: same acceptance and drafts, +25% tok/s" bash "$GATE" pair "$DIR/p0.err.log" "$DIR/a2.err.log"
: > "$DIR/a2low.err.log"
for i in $(seq 1 100); do mk_round per-step 3 1 direct 0 0 0 0 >> "$DIR/a2low.err.log"; done
expect 1 "pair: A2 drafts per verify 2 vs 4 (< 90%)" bash "$GATE" pair "$DIR/p0.err.log" "$DIR/a2low.err.log"

if [ -n "${KEEP:-}" ]; then
    bash "$GATE" probe "$DIR/pass.err.log" 5
    bash "$GATE" pair "$DIR/p0.err.log" "$DIR/a2.err.log"
fi
rm -rf "$DIR"
exit $fail
