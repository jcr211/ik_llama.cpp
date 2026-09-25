# Lane 1 progress (State-OS v1 engine: keyed header + refusal + one-op restore + companion)

Checklist:
- [x] pure header/container module (examples/server/stateos-header.*)
- [x] llama API: append-to-file, load-file-range, layout descriptor
- [x] common/speculative: MTP companion ctx + warmed-heads get/set
- [x] server SLOT_SAVE: keyed container (TOKS, MAIN, CKPT, COMP), atomic tmp+rename
- [x] server SLOT_RESTORE: verify-before-load, 409 refusals, legacy refusal, one-op restore
- [x] slot errors reach the HTTP handler (legacy result queue)
- [x] unit tests (tests/test-stateos-header.cpp) registered in CTest
- [x] .lane/GPU-VERIFY.md (+ gpu-verify-l1.ps1)
- [x] build in build-stateos-l1 + run non-GPU tests (build-l1.cmd and test-l1.cmd exit 0; 156 checks, ctest 2/2)
- [x] GET /props "stateos" capability (coordinator contract addition) + unit test + GPU-VERIFY check
- [ ] .lane/REPORT.md (the subagent file guard refused the write; the report text went to the coordinator by message)

Log (ET, approximate):
- 17:50 pure module stateos-header.{h,cpp} (sha256, H/S/I header, verify, container scan, checkpoint codec)
- 17:53 llama API (append/range-load/layout desc) + speculative companion accessors
- 17:57 server save/restore + /list + legacy-queue slot errors + tests/test-stateos-header.cpp
- 18:00 GPU-VERIFY.md + gpu-verify-l1.ps1; missing file = 409 state_missing (harness lane contract)
- 18:05 fingerprint moved to stateos-model.cpp + unit-tested on vocab GGUFs; build/test scripts
- 18:09 build started (coordinator: box free for builds)
- 18:18 build 1 failed in the test TU (tree is C++20: path::u8string type); fixed with stateos_path helpers
- 18:28 build 2 green; /props capability added; vocab-only GGUF fingerprint clamp; build 3+4 green, tests green 18:31
- 18:40 fix round (review FIX-FIRST @ c8a9261d): M1 empty-slot restore (llama early return + erase-equivalent restore),
  M2 zero-size MAIN/COMP refused + nread==0 failure, M3 GPU script (empty round trip, MAIN tamper, COMP tamper),
  SHOULDs (companion partial-cell clear, startup fingerprint, /props only with --slot-save-path, try/catch, register
  before post), model_fingerprint_v2 (16 windows/shard, all shards, shard count); P3 extras: section cap 16,
  TOKS bounded before read, checkpoint position sanity, MoveFileExW replace (no delete-first), replace-mode JSON dump
- 18:57 build exit 0, test-l1 exit 0 (195 checks / 0 failures, ctest 2/2); no compiler of another lane was running
- 19:17 fix round 2 (re-review ALLOW): N1 split.count=0, N2 KV<->tokens save refusal + restore failure, N3 guarded
  startup identity, N4 script guards/verdicts; build exit 0, test-l1 exit 0 (202 checks / 0 failures, ctest 2/2)
- 22:55 fix round 3: merged lane/ple-hist-rewind (1b3352ed; CMake conflict kept both); restore reaches the
  server-resume choke point; GPU script sets PLE env + counts [ple-hist] resets per restore round; build exit 0,
  test-l1 exit 0 (202 / 0, ctest 3/3 incl. test-ple-hist)
- 01:20 (09-24) lane/stateos-lane1-f11: F11 items 1-7 committed one per item; build-f11 exit 0; test-f11 exit 0
  (242 / 0, ctest 4/4 incl. test-stateos-layout)
- 01:35 F11 review round: P1 CKPT bound, P2-1 startup cvec, P2-2 adapter generation, P2-3 reserved suffix, P3 size
  check + fallback sha; build-f11 exit 0; test-f11 exit 0 (264 / 0, ctest 4/4). Open (noted): lora-init-without-apply
  and path-spelling over-refusal, flush retry + POSIX dir fsync, a 7c77724b-anchored layout golden, fmt buffer assert
- 01:50 F11-2 review round: P2-A/B cached effective_model + handlers forward to the task (c5d66f76), P3-1 rename
  validation + trailing dot/space (91579f16), P3-2 ckpt_ok hard gate (82c7b1b1), P3-4 system-prompt generation
  (9729d763), P3-3 recovery documented in REPORT.md; build-f11 exit 0; test-f11 exit 0 (285 / 0, ctest 4/4)
- 01:58 F11-3 review round: P2-1 lora_live for --lora-init-without-apply (0acc6624), P3-1 SET_LORA keeps unknown
  (5eab5279), P3-2 restore refuses on unknown (e3e4b7ab), P3-3 UPSTREAM-CODE FIX combined cvec sizing (8347edf7), P3-4
  system-prompt recovery documented; P3-5 left as follow-ups; build-f11 exit 0; test-f11 exit 0 (291 / 0, ctest 4/4)
- 2026-09-25: Clean PLE round-2 merge delta and bounded build script; external index lock blocked commit.
- 2026-09-25: Provisional build green; 4/4 CTest and 291/0 checks; exact served cache was pending.
- 2026-09-25: Provisional receipt done; script guards missing cache; external index lock denied commit.
- 2026-09-25: Served-cache build 6f6b75c8 green; version/hash attested; 4/4 CTest, 291/0 direct checks.
- 2026-09-25: Receipt commit attempt denied by external worktree index.lock; coordinator must force-add and commit.
