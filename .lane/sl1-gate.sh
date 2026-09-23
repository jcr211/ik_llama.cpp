#!/usr/bin/env bash
# SL-1 W-SL1 mechanism gates, read from the server's own telemetry lines. Exit 0 = PASS, 1 = STOP (a
# criterion missed: stop the campaign), 2 = usage or no data.
#   bash sl1-gate.sh probe <err.log> [M=5]        step 1: A2 + crosscheck probe
#   bash sl1-gate.sh row   <err.log> [pcie.log]   after every arm row: metrics + the per-row kills
#   bash sl1-gate.sh pair  <p0.err.log> <a2.err.log>   after each A2 row: A2 vs its P0 (acceptance, drafts)
# Lines read: [spec-host] (LONGSPEAR_SPEC_HOST_TIMING), [ckpt-xcheck] (LONGSPEAR_SPEC_CKPT_CROSSCHECK),
# [vt] (LONGSPEAR_VERIFY_TIMING, standing env), the server's "eval time" lines (err or out log), and
# "CUDA error". redecode_n counts replays the restore itself required; a crosscheck round's diagnostic
# replay is marked xcheck=1 and is not in redecode_n.
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

row_metrics() { # prints key=value metrics of one arm row
    local err=$1
    grep -F "[spec-host]" "$err" | awk '
        function val(k,   i, n, a) { for (i = 1; i <= NF; i++) { n = split($i, a, "="); if (n == 2 && a[1] == k) return a[2] } return "" }
        {
            rounds++; K = val("K") + 0; acc = val("accepted") + 0
            drafted += K - 1; accepted += acc
            rr = val("restore_result"); if (rr != "none") rej++
            if (rr == "direct") direct++; if (rr == "replay") replay++; if (rr == "failed") failed++
            rn = val("redecode_n") + 0; if (rn > 0) replay_calls++; redecode_tokens += rn
            if (val("mode") != "per-step") not_per_step++
            mtp_skip += val("mtp_skip") + 0
            cl = val("clamp") + 0; if (cl > 0) { clamp_rounds++; clamp_drafted += K - 1; clamp_acc += acc }
            xcheck += val("xcheck") + 0
            h_init += val("ckpt_init_us"); h_save += val("ckpt_save_us"); h_sinit += val("sampler_init_us")
            h_sclone += val("sampler_clone_us"); h_restore += val("restore_us"); h_redecode += val("redecode_us")
            h_draft += val("draft_host_us"); h_sample += val("sample_us")
        }
        END {
            if (rounds == 0) { print "rounds=0"; exit }
            printf "rounds=%d rejected=%d direct=%d replay=%d failed=%d replay_calls=%d redecode_tokens=%d not_per_step=%d\n", rounds, rej, direct, replay, failed, replay_calls, redecode_tokens, not_per_step
            printf "drafts_per_verify=%.4f acceptance=%.4f mtp_skip=%d clamp_rounds=%d clamp_acceptance=%s xcheck_rounds=%d\n", drafted/rounds, (drafted > 0 ? accepted/drafted : 0), mtp_skip, clamp_rounds, (clamp_drafted > 0 ? sprintf("%.4f", clamp_acc/clamp_drafted) : "n/a"), xcheck
            printf "host_us_per_verify ckpt_init=%.0f ckpt_save=%.0f sampler_init=%.0f sampler_clone=%.0f restore=%.0f redecode=%.0f draft=%.0f sample=%.0f\n", h_init/rounds, h_save/rounds, h_sinit/rounds, h_sclone/rounds, h_restore/rounds, h_redecode/rounds, h_draft/rounds, h_sample/rounds
        }'
    local main_passes
    main_passes=$(grep -F "[vt]" "$err" | grep -c " mtp_op=0 ")
    echo "main_passes=$main_passes"
    echo "cuda_errors=$(grep -c "CUDA error" "$err")"
    echo "decode_tps_tokens_ms=$(eval_tps "$err")"
}

metric() { # $1 = metrics text, $2 = key
    echo "$1" | field "$2" | head -1
}

pass=1
check() { # $1 = PASS|STOP condition result (0 = ok), $2 = description
    if [ "$1" -eq 0 ]; then echo "PASS  $2"; else echo "STOP  $2"; pass=0; fi
}

case "$MODE" in
probe)
    ERR=${2:?err log}; M=${3:-5}
    [ -f "$ERR" ] || { echo "no log $ERR"; exit 2; }
    MET=$(row_metrics "$ERR"); echo "$MET"
    R=$(metric "$MET" rounds); [ "${R:-0}" -gt 0 ] || { echo "no [spec-host] lines"; exit 2; }
    REJ=$(metric "$MET" rejected); NPS=$(metric "$MET" not_per_step); RC=$(metric "$MET" replay_calls)
    DIR=$(metric "$MET" direct); SKIP=$(metric "$MET" mtp_skip)
    check "$([ "$NPS" -eq 0 ] && echo 0 || echo 1)" "mode=per-step on every round ($NPS of $R not)"
    check "$(awk -v r="$RC" -v j="$REJ" 'BEGIN { print (j > 0 && r <= 0.01*j) ? 0 : 1 }')" "redecode_n=0 on >= 99% of rejected rounds ($RC of $REJ replayed; direct=$DIR)"
    check "$(awk -v s="$SKIP" -v j="$REJ" 'BEGIN { print (s <= 0.10*j) ? 0 : 1 }')" "mtp_skip <= 10% of rejected rounds ($SKIP vs $REJ)"
    check "$([ "$(metric "$MET" cuda_errors)" -eq 0 ] && echo 0 || echo 1)" "no CUDA error"
    # crosscheck: PLE tail bit-equal on >= 99% of rows, else its relL2 within 10x the GDN-S median and <= 0.05
    grep -F "[ckpt-xcheck]" "$ERR" > "$ERR.xcheck.tmp"
    NX=$(grep -c "comp=ple_tail" "$ERR.xcheck.tmp")
    if [ "$NX" -eq 0 ]; then
        check 1 "crosscheck produced ple_tail rows (0)"
    else
        BEQ=$(grep "comp=ple_tail" "$ERR.xcheck.tmp" | awk '{ nb = ""; n = ""; for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == "n_bitequal") nb = a[2]; if (a[1] == "n") n = a[2] } if (nb == n) c++ } END { print c + 0 }')
        GMED=$(grep "comp=gdn_s" "$ERR.xcheck.tmp" | field relL2 | sort -g | awk '{ v[NR] = $1 } END { if (NR == 0) print 0; else print v[int((NR + 1)/2)] }')
        PMAX=$(grep "comp=ple_tail" "$ERR.xcheck.tmp" | field relL2 | sort -g | tail -1)
        PMED=$(grep "comp=ple_tail" "$ERR.xcheck.tmp" | field relL2 | sort -g | awk '{ v[NR] = $1 } END { print v[int((NR + 1)/2)] }')
        echo "xcheck ple_tail rows=$NX bit_equal=$BEQ ple_median_relL2=$PMED ple_max_relL2=$PMAX gdn_s_median_relL2=$GMED"
        check "$(awk -v b="$BEQ" -v n="$NX" -v pm="$PMED" -v px="$PMAX" -v g="$GMED" 'BEGIN { print (b >= 0.99*n || (pm <= 10*g && px <= 0.05)) ? 0 : 1 }')" "PLE tail bit-equal on >= 99% of rows ($BEQ/$NX) or relL2 within 10x GDN-S median and <= 0.05"
        # every accepted step j in 0..M-2 exercised >= 20 times
        HIST=$(grep "comp=gdn_s" "$ERR.xcheck.tmp" | field j | sort -n | uniq -c | awk '{ printf "j%s=%s ", $2, $1 }')
        echo "j histogram: $HIST"
        MISSING=$(grep "comp=gdn_s" "$ERR.xcheck.tmp" | field j | sort -n | uniq -c | awk -v m="$M" '{ c[$2] = $1 } END { bad = 0; for (j = 0; j <= m - 2; j++) if (c[j] + 0 < 20) bad++; print bad }')
        check "$([ "$MISSING" -eq 0 ] && echo 0 || echo 1)" "j histogram covers every j in 0..$((M-2)) at least 20x ($MISSING short)"
    fi
    rm -f "$ERR.xcheck.tmp"
    ;;
row)
    ERR=${2:?err log}; PCIE=${3:-}
    [ -f "$ERR" ] || { echo "no log $ERR"; exit 2; }
    MET=$(row_metrics "$ERR"); echo "$MET"
    MP=$(metric "$MET" main_passes); RC=$(metric "$MET" replay_calls)
    if [ "$(metric "$MET" rounds)" != "0" ] && grep -qF "mode=per-step" "$ERR"; then
        check "$(awk -v r="${RC:-0}" -v m="$MP" 'BEGIN { print (m > 0 && r <= 0.01*m) ? 0 : 1 }')" "A2 replay passes <= 1% of main passes (${RC:-0} of $MP)"
    fi
    check "$([ "$(metric "$MET" cuda_errors)" -eq 0 ] && echo 0 || echo 1)" "no CUDA error"
    if [ -n "$PCIE" ] && [ -f "$PCIE" ]; then
        # pcie-telemetry.sh: column 2 is the "Replays Since Reset" counter
        D=$(awk 'NR == 1 { f = $2 } { l = $2 } END { print l - f }' "$PCIE")
        check "$([ "${D:-0}" -eq 0 ] && echo 0 || echo 1)" "no PCIe replay increment ($D)"
    fi
    ;;
pair)
    P0=${2:?p0 err log}; A2=${3:?a2 err log}
    MP0=$(row_metrics "$P0"); MA2=$(row_metrics "$A2")
    AP=$(metric "$MP0" acceptance); AA=$(metric "$MA2" acceptance)
    DP=$(metric "$MP0" drafts_per_verify); DA=$(metric "$MA2" drafts_per_verify)
    TP=$(metric "$MP0" decode_tps_tokens_ms); TA=$(metric "$MA2" decode_tps_tokens_ms)
    echo "P0 acceptance=$AP drafts_per_verify=$DP decode_tps=$TP"
    echo "A2 acceptance=$AA drafts_per_verify=$DA decode_tps=$TA"
    check "$(awk -v a="$AA" -v p="$AP" 'BEGIN { print (a >= p - 0.03) ? 0 : 1 }')" "A2 acceptance >= P0 - 3 pts"
    check "$(awk -v a="$DA" -v p="$DP" 'BEGIN { print (p > 0 && a >= 0.90*p) ? 0 : 1 }')" "A2 drafts per verify >= 90% of P0"
    echo "gain=$(awk -v a="${TA%% *}" -v p="${TP%% *}" 'BEGIN { if (p > 0) printf "%+.2f%%", 100*(a/p - 1); else print "n/a" }') (kill after both A2 reps if < +8% in both)"
    ;;
*)
    sed -n '2,12p' "$0"; exit 2
    ;;
esac

[ "$pass" -eq 1 ] && { echo "RESULT PASS"; exit 0; } || { echo "RESULT STOP"; exit 1; }
