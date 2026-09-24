#!/usr/bin/env bash
# SV2-E1 unit tests with the GPU hidden. Run: bash .lane/run-unit-tests.sh
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=-1
fail=0
for t in test-partial-state test-stateos-tail test-ple-hist; do
  build-stateos-tail/bin/$t.exe > ".lane/$t.out" 2>&1
  rc=$?
  echo "$t rc=$rc :: $(tail -n 1 ".lane/$t.out")"
  [ "$rc" -eq 0 ] || fail=1
done
node --test tools/stateos-tail-tools.test.mjs > .lane/node-test.txt 2>&1
rc=$?
echo "node tools tests rc=$rc :: $(grep -E '^. (pass|fail) ' .lane/node-test.txt | tr '\n' ' ')"
[ "$rc" -eq 0 ] || fail=1
exit $fail
