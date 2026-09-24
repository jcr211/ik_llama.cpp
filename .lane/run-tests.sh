#!/usr/bin/env bash
# Run the SL-1 CPU tests on build-sl1 and print one exit-code line per test (logs in .lane/test-*.log).
ROOT=/d/AI/worktrees/sl1-spec-ckpt
BIN=$ROOT/build-sl1/bin
VOCAB=$ROOT/models/ggml-vocab-qwen2.gguf
fail=0
run() { # $1 = test name, rest = args
    local t=$1; shift
    "$BIN/$t.exe" "$@" > "$ROOT/.lane/$t.log" 2>&1
    local rc=$?
    echo "$t exit=$rc :: $(grep -E "all OK|FAIL|failed" "$ROOT/.lane/$t.log" | tail -1)"
    [ "$rc" = 0 ] || fail=1
}
run test-ple-perstep
run test-iqk-moe-chunks
run test-spec-ckpt-sampler "$VOCAB"
run test-spec-ckpt-clamp "$VOCAB"
exit $fail
