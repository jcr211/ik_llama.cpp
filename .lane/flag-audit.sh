#!/usr/bin/env bash
# SV2-E1 flag audit: every added line that reads a State-OS v2 flag, and every added call site whose
# behaviour depends on one. Run from the worktree root: bash .lane/flag-audit.sh
cd "$(dirname "$0")/.." || exit 1
BASE=d583c220
echo "== env flags read (added lines) =="
git diff "$BASE"..HEAD -U0 -- src common include examples | grep -n -E '^\+.*(getenv|LONGSPEAR_STATEOS_[A-Z_]+)' | grep -v '^\S*+\s*//'
echo
echo "== gates: stateos_div_log() / stateos_tail_snapshot() / stateos_tail_xcheck() / common_speculative_tail_snapshot_enabled() =="
for f in examples/server/server-context.cpp common/speculative.cpp; do
  grep -n -E 'stateos_div_log\(\)|stateos_tail_snapshot\(\)|stateos_tail_xcheck\(\)|common_speculative_tail_snapshot_enabled\(\)' "$f" | sed "s|^|$f:|"
done
echo
echo "== writers of the only state the flags create (shadow_pos set, tail checkpoints, [stateos-div]/[ckpt-xcheck] output) =="
grep -n -E 'shadow_pos = root_pos|STATEOS_ORIGIN_TAIL;|fprintf\(stderr, "\[(stateos-div|ckpt-xcheck)\]|fprintf\(stderr, "%s chosen_origin' src/llama.cpp examples/server/server-context.cpp
echo
echo "== new aborts/asserts on added lines (must be empty) =="
git diff "$BASE"..HEAD -U0 -- src common include examples tests | grep -n -E '^\+.*(GGML_ABORT|GGML_ASSERT|abort\(|std::terminate)' || echo "(none)"
