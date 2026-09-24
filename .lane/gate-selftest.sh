#!/usr/bin/env bash
# Offline self-test of sl1-gate.sh on synthetic telemetry (no GPU). One line per case; exit 1 on a mismatch.
# KEEP=1 also prints the full gate output of the passing probe and the clamp-confound pair.
set -u
DIR=$(mktemp -d)
GATE=/d/AI/worktrees/sl1-spec-ckpt/.lane/sl1-gate.sh
fail=0

# $1 mode $2 K $3 k_prop $4 accepted $5 restore_result $6 redecode_n $7 mtp_skip $8 clamp $9 xcheck
round() {
    echo "[spec-host] slot=0 mode=$1 K=$2 k_prop=$3 accepted=$4 restore_result=$5 redecode_n=$6 ckpt_init_us=12 ckpt_save_us=30 cells_copy_us=0 shadow_copy_us=0 sync_us=0 sampler_init_us=5 sampler_clone_us=9 restore_us=400 redecode_us=0 draft_host_us=2900 sample_us=700 mtp_skip=$7 clamp=$8 xcheck=$9"
}
vt() { echo "[vt] K=$1 n_kv=4000 mtp_op=0 us=60000 build=100 compute=$2 logits=10 embd=10 reused=1 nodes=3000 splits=40"; }
evalt() { echo "       eval time = $1 ms / $2 tokens (   28.57 ms per token,    35.00 tokens per second)"; }

# probe: 4 * $2 rejected rounds with j = i % 4; $3 ple (equal|diverge); $4 replayed rounds; $5 failed rounds; $6 mtp_skip
probe_log() {
    local f=$1 n=$2 ple=$3 nrep=$4 nfail=$5 skip=$6
    : > "$f"
    for i in $(seq 0 $((n - 1))); do
        j=$((i % 4))
        if [ "$i" -lt "$nrep" ]; then round per-step 5 5 $j replay $((j + 1)) 0 0 0 >> "$f"
        elif [ "$i" -lt $((nrep + nfail)) ]; then round per-step 5 5 $j failed 0 0 0 0 >> "$f"
        else round per-step 5 5 $j direct 0 "$([ "$i" -lt $((nrep + nfail + skip)) ] && echo 1 || echo 0)" 0 1 >> "$f"
        fi
        echo "[ckpt-xcheck] j=$j comp=gdn_s n_bitequal=700000 n=786432 relL2=3.1e-04 max_layer_relL2=9e-04" >> "$f"
        echo "[ckpt-xcheck] j=$j comp=gdn_conv n_bitequal=30720 n=30720 relL2=0 max_layer_relL2=0" >> "$f"
        if [ "$ple" = equal ]; then
            echo "[ckpt-xcheck] j=$j comp=ple_tail n_bitequal=92160 n=92160 relL2=0 max_layer_relL2=0" >> "$f"
        else
            echo "[ckpt-xcheck] j=$j comp=ple_tail n_bitequal=100 n=92160 relL2=3.0e-01 max_layer_relL2=3.0e-01" >> "$f"
        fi
        vt 5 60000 >> "$f"
    done
    evalt 40000.00 1400 >> "$f"
    echo "[ple-hist] set seq=0 next_pos=4000 n_prev=2 site=spec-replay" >> "$f"
}

# arm row: 100 rounds. $2 arm kind (a2|p0|a0|a2-fallback), $3 mtp_skip rounds
row_log() {
    local f=$1 kind=$2 skip=${3:-0}
    : > "$f"
    for i in $(seq 1 100); do
        case "$kind" in
            a2)          round per-step 5 5 2 direct 0 "$([ "$i" -le "$skip" ] && echo 1 || echo 0)" 0 0 >> "$f" ;;
            a2-fallback) round gpu-fallback 5 5 2 replay 3 0 0 0 >> "$f" ;;
            p0)          round gpu-fallback 5 5 2 replay 3 0 0 0 >> "$f" ;;
            a0)          ;;
        esac
        vt 5 60000 >> "$f"
    done
    evalt 40000.00 1400 >> "$f"
    echo "[ple-hist] set seq=0 next_pos=4000 n_prev=2 site=server-resume" >> "$f"
}

pcie_log() { # $1 file, then counter samples
    local f=$1; shift
    echo "time replays link_gen link_width power_w temp_c sm_mhz mem_mhz util_pct tx_kbs rx_kbs" > "$f"
    for v in "$@"; do echo "12:00:00 $v 4 16 400 60 2800 14000 90 100 100" >> "$f"; done
}

expect() { # $1 expected exit, $2 description, rest = command
    local want=$1 what=$2; shift 2
    "$@" > "$DIR/out.txt" 2>&1
    local got=$?
    if [ "$got" = "$want" ]; then echo "ok   $what (exit $got)"; else echo "BAD  $what (exit $got, want $want)"; cat "$DIR/out.txt"; fail=1; fi
}

# ---- probe
probe_log "$DIR/pass.err.log" 120 equal 0 0 0
expect 0 "probe: per-step, bit-equal PLE tail, every j >= 20" bash "$GATE" probe "$DIR/pass.err.log" 5
probe_log "$DIR/ple.err.log" 120 diverge 0 0 0
expect 1 "probe: PLE tail relL2 0.3 (the B1 signature) STOPs" bash "$GATE" probe "$DIR/ple.err.log" 5
probe_log "$DIR/replay.err.log" 120 equal 10 0 0
expect 1 "probe: 10 of 120 rejected rounds needed a replay" bash "$GATE" probe "$DIR/replay.err.log" 5
probe_log "$DIR/failed.err.log" 120 equal 0 1 0
expect 1 "probe: one failed restore STOPs (N2)" bash "$GATE" probe "$DIR/failed.err.log" 5
probe_log "$DIR/skip.err.log" 120 equal 0 0 20
expect 1 "probe: mtp_skip 20 of 120 rejected rounds STOPs" bash "$GATE" probe "$DIR/skip.err.log" 5
probe_log "$DIR/short.err.log" 60 equal 0 0 0
expect 4 "probe: 15 rows per j is INCONCLUSIVE, not a pass (AMEND 1)" bash "$GATE" probe "$DIR/short.err.log" 5
expect 1 "probe: a mechanism miss outranks a short j histogram" bash "$GATE" probe "$DIR/ple.err.log" 9
bash "$GATE" jcount "$DIR/short.err.log" 5 > "$DIR/jc.txt"
if grep -q "rounds=60 short=4" "$DIR/jc.txt"; then echo "ok   jcount: 60 rounds, 4 values of j short"; else echo "BAD  jcount: $(cat "$DIR/jc.txt")"; fail=1; fi

# ---- rows
pcie_log "$DIR/pcie-flat.log" 7 7 7 7
pcie_log "$DIR/pcie-inc.log" 7 7 8 8
pcie_log "$DIR/pcie-na.log" NA NA
pcie_log "$DIR/pcie-empty.log"
row_log "$DIR/a2.err.log" a2 0
expect 0 "row A2: per-step, no replay; PCIe counter 7 at row start, no increment" bash "$GATE" row A2 "$DIR/a2.err.log" "$DIR/pcie-flat.log"
expect 1 "row A2: PCIe counter 7 -> 8 STOPs" bash "$GATE" row A2 "$DIR/a2.err.log" "$DIR/pcie-inc.log"
expect 1 "row A2: PCIe log with only NA samples STOPs (N1)" bash "$GATE" row A2 "$DIR/a2.err.log" "$DIR/pcie-na.log"
expect 1 "row A2: header-only PCIe log STOPs (N1)" bash "$GATE" row A2 "$DIR/a2.err.log" "$DIR/pcie-empty.log"
row_log "$DIR/a2fb.err.log" a2-fallback
expect 1 "row A2: a row that ran gpu-fallback fails instead of skipping (N2)" bash "$GATE" row A2 "$DIR/a2fb.err.log" "$DIR/pcie-flat.log"
row_log "$DIR/a2skip.err.log" a2 30
expect 1 "row A2: mtp_skip 30 of 100 rejected rounds STOPs (S2)" bash "$GATE" row A2 "$DIR/a2skip.err.log" "$DIR/pcie-flat.log"
row_log "$DIR/p0.err.log" p0
expect 0 "row P0: gpu-fallback rounds pass" bash "$GATE" row P0 "$DIR/p0.err.log" "$DIR/pcie-flat.log"
expect 1 "row P0: an A2 log under the P0 label STOPs" bash "$GATE" row P0 "$DIR/a2.err.log" "$DIR/pcie-flat.log"
row_log "$DIR/a0.err.log" a0
expect 0 "row A0: no verify rounds" bash "$GATE" row A0 "$DIR/a0.err.log" "$DIR/pcie-flat.log"
cp "$DIR/a2.err.log" "$DIR/a2reset.err.log"
echo "[ple-hist] reset seq=0 pos=4012 next_pos=4015" >> "$DIR/a2reset.err.log"
expect 1 "row A2: one mid-sequence [ple-hist] reset STOPs (B1)" bash "$GATE" row A2 "$DIR/a2reset.err.log" "$DIR/pcie-flat.log"
grep -v "\[ple-hist\]" "$DIR/p0.err.log" > "$DIR/p0nolog.err.log"
expect 1 "row P0: no [ple-hist] lines (log not live) STOPs" bash "$GATE" row P0 "$DIR/p0nolog.err.log" "$DIR/pcie-flat.log"
cp "$DIR/pass.err.log" "$DIR/probereset.err.log"
echo "[ple-hist] reset seq=0 pos=900 next_pos=903" >> "$DIR/probereset.err.log"
expect 1 "probe: a mid-sequence [ple-hist] reset STOPs (B1)" bash "$GATE" probe "$DIR/probereset.err.log" 5
expect 1 "row A0: verify rounds under A0 STOP" bash "$GATE" row A0 "$DIR/p0.err.log" "$DIR/pcie-flat.log"

# ---- pair: the clamp confound (S1). P0 verifies 12% ngram drafts of 16 (K=17); A2 clamps them to 4 (K=5)
: > "$DIR/p0c.err.log"; : > "$DIR/a2c.err.log"
for i in $(seq 1 100); do
    if [ $((i % 8)) -eq 0 ]; then
        round gpu-fallback 17 17 10 replay 11 0 0 0 >> "$DIR/p0c.err.log"
        round per-step 5 17 4 none 0 0 12 0 >> "$DIR/a2c.err.log"
    else
        round gpu-fallback 5 5 2 replay 3 0 0 0 >> "$DIR/p0c.err.log"
        round per-step 5 5 2 direct 0 0 0 0 >> "$DIR/a2c.err.log"
    fi
done
evalt 50000.00 1400 >> "$DIR/p0c.err.log"
evalt 40000.00 1400 >> "$DIR/a2c.err.log"
expect 0 "pair: clamped K=17 rounds do not false-kill (pre-clamp drafts, fit-round acceptance)" bash "$GATE" pair "$DIR/p0c.err.log" "$DIR/a2c.err.log" 5
: > "$DIR/a2low.err.log"
for i in $(seq 1 100); do round per-step 3 3 1 direct 0 0 0 0 >> "$DIR/a2low.err.log"; done
evalt 40000.00 1400 >> "$DIR/a2low.err.log"
expect 1 "pair: A2 proposes 2 drafts per verify vs P0's 5.4 STOPs" bash "$GATE" pair "$DIR/p0c.err.log" "$DIR/a2low.err.log" 5
: > "$DIR/a2acc.err.log"
for i in $(seq 1 100); do round per-step 5 5 1 direct 0 0 0 0 >> "$DIR/a2acc.err.log"; done
evalt 40000.00 1400 >> "$DIR/a2acc.err.log"
expect 1 "pair: A2 fit-round acceptance 25% vs P0 50% STOPs" bash "$GATE" pair "$DIR/p0c.err.log" "$DIR/a2acc.err.log" 5
bash "$GATE" pair "$DIR/p0c.err.log" "$DIR/a2c.err.log" 5 | grep -q "^GAIN_PCT=25.00" && echo "ok   pair: GAIN_PCT=25.00 printed for the chain" || { echo "BAD  pair: no GAIN_PCT=25.00"; fail=1; }

# ---- stepcost
: > "$DIR/b-p0.err.log"; : > "$DIR/b-a2ok.err.log"; : > "$DIR/b-a2bad.err.log"
for k in 2 3 4 5; do
    for i in 1 2 3; do vt $k 50000 >> "$DIR/b-p0.err.log"; vt $k 51000 >> "$DIR/b-a2ok.err.log"; vt $k 52500 >> "$DIR/b-a2bad.err.log"; done
done
expect 0 "stepcost: A2 +2% at K=2..5 passes" bash "$GATE" stepcost "$DIR/b-p0.err.log" "$DIR/b-a2ok.err.log"
expect 3 "stepcost: A2 +5% is reported and blocks promotion (not a STOP)" bash "$GATE" stepcost "$DIR/b-p0.err.log" "$DIR/b-a2bad.err.log"

if [ -n "${KEEP:-}" ]; then
    bash "$GATE" probe "$DIR/pass.err.log" 5
    bash "$GATE" pair "$DIR/p0c.err.log" "$DIR/a2c.err.log" 5
fi
rm -rf "$DIR"
exit $fail
