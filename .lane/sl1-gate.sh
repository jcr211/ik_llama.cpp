#!/usr/bin/env bash
# SL-1 W-SL1 mechanism gates, read from the server's own telemetry lines.
# Exit 0 = PASS, 1 = STOP (a criterion missed: stop the campaign), 2 = usage or no data,
# 3 = REPORT-ONLY miss that blocks promotion (stepcost), 4 = INCONCLUSIVE (probe short of its j target).
#   bash sl1-gate.sh probe    <err.log> [M=5]                      step 1: A2 + crosscheck probe
#   bash sl1-gate.sh row      <P0|A2|A0> <err.log> <pcie.log>      after every arm row
#   bash sl1-gate.sh pair     <p0.err.log> <a2.err.log> [M=5]      A2 vs its P0 (acceptance, drafts, gain)
#   bash sl1-gate.sh jcount   <err.log> [M=5]                      probe watcher: rounds and j rows short of 20
#   bash sl1-gate.sh stepcost <p0 bench err logs, comma-separated> <a2 ...>   step 3: per-K step cost K=2..5
# Lines read: [spec-host] (LONGSPEAR_SPEC_HOST_TIMING), [ckpt-xcheck] (LONGSPEAR_SPEC_CKPT_CROSSCHECK),
# [vt] (LONGSPEAR_VERIFY_TIMING, standing env), the server's "eval time" lines (err or out log), "CUDA error",
# and pcie-telemetry.sh's replay counter (column 2).
# redecode_n counts replays the restore itself required; a crosscheck round's diagnostic replay is marked
# xcheck=1 and is not in redecode_n. K is the verified batch, k_prop the drafter's proposal before the
# capacity clamp (k_prop = K + clamp).
# Clamp confound (S1): drafts per verify is compared on the PRE-clamp proposal (k_prop - 1, both arms);
# acceptance is compared only on rounds whose proposal fits the capacity (k_prop <= M, both arms). Rounds
# with k_prop > M are reported separately: A2's accepted/(M-1) vs P0's min(accepted, M-1)/(M-1), i.e. how
# much of the same first M-1 drafts each arm got accepted.
set -u
MODE=${1:-}

field() { # $1 = key; prints the value of key=value from stdin lines, one per line
    grep -oE "(^| )$1=[^ ]+" | sed -E "s/^ ?$1=//"
}

eval_tps() { # aggregate decode tok/s = sum(tokens) / sum(ms) over "eval time" lines (not prompt eval)
    local err=$1 out=${1%.err.log}.out.log
    cat "$err" "$out" 2>/dev/null | grep -E "^ *eval time =" | \
        awk '{ got_ms = 0; got_tok = 0
               for (i = 1; i <= NF; i++) {
                   if ($i == "=" && !got_ms) { ms += $(i+1); got_ms = 1 }
                   if ($i == "tokens" && !got_tok) { tok += $(i-1); got_tok = 1 }
               } }
             END { if (ms > 0) printf "%.3f %d %.1f\n", tok/(ms/1000.0), tok, ms; else print "0 0 0" }'
}

row_metrics() { # $1 = err log, $2 = M; prints key=value metrics of one row
    local err=$1 m=${2:-5}
    grep -F "[spec-host]" "$err" | awk -v M="$m" '
        function val(k,   i, n, a) { for (i = 1; i <= NF; i++) { n = split($i, a, "="); if (n == 2 && a[1] == k) return a[2] } return "" }
        {
            rounds++; K = val("K") + 0; acc = val("accepted") + 0; cl = val("clamp") + 0
            kp = val("k_prop"); kp = (kp == "") ? K + cl : kp + 0
            drafted += K - 1; drafted_prop += kp - 1; accepted += acc
            if (kp <= M) { fit_drafted += K - 1; fit_acc += acc }
            else {
                over++
                if (cl > 0) { clamp_rounds++; clamp_acc += acc; clamp_drafted += K - 1 }
                else { cap_acc += (acc < M - 1 ? acc : M - 1); cap_drafted += M - 1 }
            }
            rr = val("restore_result"); if (rr != "none") rej++
            if (rr == "direct") direct++; if (rr == "replay") replay++; if (rr == "failed") failed++
            rn = val("redecode_n") + 0; if (rn > 0) replay_calls++; redecode_tokens += rn
            md = val("mode"); if (md == "per-step") per_step++; else if (md == "gpu-fallback") fallback++; else other_mode++
            mtp_skip += val("mtp_skip") + 0
            xcheck += val("xcheck") + 0
            h_init += val("ckpt_init_us"); h_save += val("ckpt_save_us"); h_sinit += val("sampler_init_us")
            h_sclone += val("sampler_clone_us"); h_restore += val("restore_us"); h_redecode += val("redecode_us")
            h_draft += val("draft_host_us"); h_sample += val("sample_us")
        }
        END {
            if (rounds == 0) { print "rounds=0 rejected=0 failed=0 replay_calls=0 per_step=0 fallback=0 other_mode=0 mtp_skip=0"; exit }
            printf "rounds=%d rejected=%d direct=%d replay=%d failed=%d replay_calls=%d redecode_tokens=%d per_step=%d fallback=%d other_mode=%d\n", rounds, rej, direct, replay, failed, replay_calls, redecode_tokens, per_step, fallback, other_mode
            printf "drafts_per_verify=%.4f drafts_per_verify_prop=%.4f acceptance_all=%.4f acceptance_fit=%s over_cap_rounds=%d\n", drafted/rounds, drafted_prop/rounds, (drafted > 0 ? accepted/drafted : 0), (fit_drafted > 0 ? sprintf("%.4f", fit_acc/fit_drafted) : "n/a"), over
            printf "clamp_rounds=%d clamp_acceptance=%s p0_first_m1_acceptance=%s mtp_skip=%d mtp_skip_per_rejected=%s xcheck_rounds=%d\n", clamp_rounds, (clamp_drafted > 0 ? sprintf("%.4f", clamp_acc/clamp_drafted) : "n/a"), (cap_drafted > 0 ? sprintf("%.4f", cap_acc/cap_drafted) : "n/a"), mtp_skip, (rej > 0 ? sprintf("%.4f", mtp_skip/rej) : "n/a"), xcheck
            printf "host_us_per_verify ckpt_init=%.0f ckpt_save=%.0f sampler_init=%.0f sampler_clone=%.0f restore=%.0f redecode=%.0f draft=%.0f sample=%.0f\n", h_init/rounds, h_save/rounds, h_sinit/rounds, h_sclone/rounds, h_restore/rounds, h_redecode/rounds, h_draft/rounds, h_sample/rounds
        }'
    echo "main_passes=$(grep -F "[vt]" "$err" | grep -c " mtp_op=0 ")"
    echo "cuda_errors=$(grep -c "CUDA error" "$err")"
    echo "decode_tps_tokens_ms=$(eval_tps "$err")"
}

metric() { # $1 = metrics text, $2 = key
    echo "$1" | field "$2" | head -1
}

pcie_delta() { # replay-counter increase over the row: first to last numeric sample; NA when none
    awk 'NR > 1 && $2 ~ /^[0-9]+$/ { if (!n++) f = $2; l = $2 } END { if (n) print l - f; else print "NA" }' "$1"
}

jcount() { # $1 = err log, $2 = M; prints "rounds=<[spec-host] lines> short=<j values under 20 rows> hist=..."
    local err=$1 m=${2:-5}
    local rounds hist short
    rounds=$(grep -cF "[spec-host]" "$err")
    hist=$(grep -F "[ckpt-xcheck]" "$err" | grep "comp=gdn_s" | field j | sort -n | uniq -c | awk '{ printf "j%s=%s,", $2, $1 }')
    short=$(grep -F "[ckpt-xcheck]" "$err" | grep "comp=gdn_s" | field j | sort -n | uniq -c | awk -v m="$m" '{ c[$2] = $1 } END { bad = 0; for (j = 0; j <= m - 2; j++) if (c[j] + 0 < 20) bad++; print bad }')
    echo "rounds=$rounds short=$short hist=${hist:-none}"
}

pass=1
check() { # $1 = condition result (0 = ok), $2 = description
    if [ "$1" -eq 0 ]; then echo "PASS  $2"; else echo "STOP  $2"; pass=0; fi
}
ok_if() { # $1 = awk boolean expression over the -v variables that follow; prints 0 when true
    local expr=$1; shift
    awk "$@" "BEGIN { print ($expr) ? 0 : 1 }"
}

case "$MODE" in
jcount)
    ERR=${2:?err log}; M=${3:-5}
    [ -f "$ERR" ] || { echo "rounds=0 short=$((M-1)) hist=none"; exit 0; }
    jcount "$ERR" "$M"
    exit 0
    ;;
probe)
    ERR=${2:?err log}; M=${3:-5}
    [ -f "$ERR" ] || { echo "no log $ERR"; exit 2; }
    MET=$(row_metrics "$ERR" "$M"); echo "$MET"
    R=$(metric "$MET" rounds); [ "${R:-0}" -gt 0 ] || { echo "no [spec-host] lines"; exit 2; }
    REJ=$(metric "$MET" rejected); PS=$(metric "$MET" per_step); RC=$(metric "$MET" replay_calls)
    DIR=$(metric "$MET" direct); SKIP=$(metric "$MET" mtp_skip); FAILED=$(metric "$MET" failed)
    check "$(ok_if 'p == r' -v p="$PS" -v r="$R")" "mode=per-step on every round ($PS of $R)"
    check "$(ok_if 'f == 0' -v f="$FAILED")" "no failed restore ($FAILED)"
    check "$(ok_if 'j > 0 && c <= 0.01*j' -v c="$RC" -v j="$REJ")" "required replays (redecode_n > 0) on <= 1% of rejected rounds ($RC of $REJ; direct=$DIR)"
    check "$(ok_if 's <= 0.10*j' -v s="$SKIP" -v j="$REJ")" "mtp_skip <= 10% of rejected rounds ($SKIP vs $REJ; crosscheck rounds commit the companion the per-step way)"
    check "$(ok_if 'c == 0' -v c="$(metric "$MET" cuda_errors)")" "no CUDA error"
    # crosscheck: PLE tail bit-equal on >= 99% of rows, else its median relL2 within 10x the GDN-S median and max <= 0.05
    XC=$(grep -F "[ckpt-xcheck]" "$ERR")
    NX=$(echo "$XC" | grep -c "comp=ple_tail")
    if [ "$NX" -eq 0 ]; then
        check 1 "crosscheck produced ple_tail rows (0)"
    else
        BEQ=$(echo "$XC" | grep "comp=ple_tail" | awk '{ nb = ""; n = ""; for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == "n_bitequal") nb = a[2]; if (a[1] == "n") n = a[2] } if (nb == n) c++ } END { print c + 0 }')
        GMED=$(echo "$XC" | grep "comp=gdn_s" | field relL2 | sort -g | awk '{ v[NR] = $1 } END { if (NR == 0) print 0; else print v[int((NR + 1)/2)] }')
        PMAX=$(echo "$XC" | grep "comp=ple_tail" | field relL2 | sort -g | tail -1)
        PMED=$(echo "$XC" | grep "comp=ple_tail" | field relL2 | sort -g | awk '{ v[NR] = $1 } END { print v[int((NR + 1)/2)] }')
        echo "xcheck ple_tail rows=$NX bit_equal=$BEQ ple_median_relL2=$PMED ple_max_relL2=$PMAX gdn_s_median_relL2=$GMED"
        check "$(ok_if 'b >= 0.99*n || (pm <= 10*g && px <= 0.05)' -v b="$BEQ" -v n="$NX" -v pm="$PMED" -v px="$PMAX" -v g="$GMED")" "PLE tail bit-equal on >= 99% of rows ($BEQ/$NX) or median relL2 <= 10x GDN-S median and max <= 0.05"
    fi
    JC=$(jcount "$ERR" "$M"); echo "j coverage: $JC"
    SHORT=$(echo "$JC" | field short)
    if [ "$pass" -eq 1 ] && [ "${SHORT:-1}" -gt 0 ]; then
        echo "INCONCLUSIVE  $SHORT value(s) of j in 0..$((M-2)) have fewer than 20 crosscheck rows: not a pass, no arm runs"
        echo "RESULT INCONCLUSIVE"; exit 4
    fi
    [ "${SHORT:-1}" -eq 0 ] && echo "PASS  every j in 0..$((M-2)) has >= 20 crosscheck rows"
    ;;
row)
    ARM=${2:?arm P0|A2|A0}; ERR=${3:?err log}; PCIE=${4:-}; M=5
    [ -f "$ERR" ] || { echo "no log $ERR"; exit 2; }
    MET=$(row_metrics "$ERR" "$M"); echo "$MET"
    R=$(metric "$MET" rounds); MP=$(metric "$MET" main_passes); RC=$(metric "$MET" replay_calls)
    REJ=$(metric "$MET" rejected); FAILED=$(metric "$MET" failed); SKIP=$(metric "$MET" mtp_skip)
    TOK=$(echo "$MET" | grep -oE "^decode_tps_tokens_ms=[^ ]+ [0-9]+" | awk '{ print $2 }')
    check "$(ok_if 't > 0' -v t="${TOK:-0}")" "the row decoded tokens (${TOK:-0})"
    case "$ARM" in
    A2)
        check "$(ok_if 'r > 0 && p == r' -v r="$R" -v p="$(metric "$MET" per_step)")" "A2 ran per-step on every verify round ($(metric "$MET" per_step) of $R)"
        check "$(ok_if 'f == 0' -v f="$FAILED")" "no failed restore ($FAILED)"
        check "$(ok_if 'm > 0 && c <= 0.01*m' -v c="${RC:-0}" -v m="$MP")" "A2 restore-required replays <= 1% of main passes (${RC:-0} of $MP)"
        check "$(ok_if 's <= 0.10*j' -v s="$SKIP" -v j="$REJ")" "mtp_skip <= 10% of rejected rounds ($SKIP vs $REJ)"
        ;;
    P0)
        check "$(ok_if 'r > 0 && p == r' -v r="$R" -v p="$(metric "$MET" fallback)")" "P0 ran gpu-fallback on every verify round ($(metric "$MET" fallback) of $R)"
        check "$(ok_if 'f == 0' -v f="$FAILED")" "no failed restore ($FAILED)"
        ;;
    A0)
        check "$(ok_if 'r == 0' -v r="$R")" "A0 ran no speculation ($R verify rounds)"
        ;;
    *)
        echo "unknown arm $ARM"; exit 2
        ;;
    esac
    check "$(ok_if 'c == 0' -v c="$(metric "$MET" cuda_errors)")" "no CUDA error"
    if [ -n "$PCIE" ] && [ -f "$PCIE" ]; then
        D=$(pcie_delta "$PCIE")
        check "$(ok_if 'd != "NA" && d == 0' -v d="$D")" "no PCIe replay increment over the row (delta $D; NA = no numeric sample)"
    else
        check 1 "PCIe telemetry log for the row (missing: ${PCIE:-none})"
    fi
    ;;
pair)
    P0=${2:?p0 err log}; A2=${3:?a2 err log}; M=${4:-5}
    MP0=$(row_metrics "$P0" "$M"); MA2=$(row_metrics "$A2" "$M")
    AP=$(metric "$MP0" acceptance_fit); AA=$(metric "$MA2" acceptance_fit)
    DP=$(metric "$MP0" drafts_per_verify_prop); DA=$(metric "$MA2" drafts_per_verify_prop)
    TP=$(echo "$MP0" | grep -oE "^decode_tps_tokens_ms=[0-9.]+" | cut -d= -f2)
    TA=$(echo "$MA2" | grep -oE "^decode_tps_tokens_ms=[0-9.]+" | cut -d= -f2)
    echo "P0 acceptance_fit=$AP drafts_per_verify_prop=$DP decode_tps=$TP over_cap_rounds=$(metric "$MP0" over_cap_rounds) first_m1_acceptance=$(metric "$MP0" p0_first_m1_acceptance) mtp_skip_per_rejected=$(metric "$MP0" mtp_skip_per_rejected)"
    echo "A2 acceptance_fit=$AA drafts_per_verify_prop=$DA decode_tps=$TA clamp_rounds=$(metric "$MA2" clamp_rounds) clamp_acceptance=$(metric "$MA2" clamp_acceptance) mtp_skip_per_rejected=$(metric "$MA2" mtp_skip_per_rejected)"
    check "$(ok_if 'a != "n/a" && p != "n/a" && a >= p - 0.03' -v a="$AA" -v p="$AP")" "A2 acceptance >= P0 - 3 pts (rounds with k_prop <= $M in both arms)"
    check "$(ok_if 'p > 0 && a >= 0.90*p' -v a="$DA" -v p="$DP")" "A2 drafts per verify >= 90% of P0 (pre-clamp proposal k_prop - 1 in both arms)"
    GAIN=$(awk -v a="${TA:-0}" -v p="${TP:-0}" 'BEGIN { if (p > 0) printf "%.2f", 100*(a/p - 1); else print "NA" }')
    echo "GAIN_PCT=$GAIN (A2 vs P0 aggregate decode tok/s; kill after both A2 reps if < +8 in both)"
    ;;
stepcost)
    P0S=${2:?p0 bench err logs}; A2S=${3:?a2 bench err logs}
    percost() { # comma-separated logs -> "K mean_compute" for K=2..5 over main-model [vt] lines
        echo "$1" | tr ',' '\n' | while read -r f; do [ -f "$f" ] && grep -F "[vt]" "$f"; done | \
            awk 'function val(k,   i, n, a) { for (i = 1; i <= NF; i++) { n = split($i, a, "="); if (n == 2 && a[1] == k) return a[2] } return "" }
                 val("mtp_op") == "0" { k = val("K") + 0; if (k >= 2 && k <= 5) { n[k]++; c[k] += val("compute") } }
                 END { for (k = 2; k <= 5; k++) printf "%d %s\n", k, (n[k] > 0 ? sprintf("%.1f", c[k]/n[k]) : "NA") }'
    }
    percost "$P0S" > /tmp/sl1-stepcost-p0.$$; percost "$A2S" > /tmp/sl1-stepcost-a2.$$
    worst=$(join /tmp/sl1-stepcost-p0.$$ /tmp/sl1-stepcost-a2.$$ | awk '{ if ($2 != "NA" && $3 != "NA" && $2 > 0) { r = 100*($3/$2 - 1); printf "K=%d P0=%s A2=%s %+.2f%%\n", $1, $2, $3, r > "/dev/stderr"; if (r > w) w = r } } END { printf "%.2f\n", w + 0 }')
    rm -f /tmp/sl1-stepcost-p0.$$ /tmp/sl1-stepcost-a2.$$
    echo "STEPCOST_WORST_PCT=$worst"
    if awk -v w="$worst" 'BEGIN { exit !(w > 3) }'; then
        echo "REPORT  K=2..5 regression > 3% (worst $worst%): reported, blocks promotion"; echo "RESULT BLOCKS-PROMOTION"; exit 3
    fi
    echo "PASS  K=2..5 step cost within 3% of P0 (worst $worst%)"
    ;;
*)
    sed -n '2,21p' "$0"; exit 2
    ;;
esac

[ "$pass" -eq 1 ] && { echo "RESULT PASS"; exit 0; } || { echo "RESULT STOP"; exit 1; }
