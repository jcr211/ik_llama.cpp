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
- [x] build in build-stateos-l1 + run non-GPU tests (build-l1.cmd exit 0; test-l1.cmd exit 0: 156 checks / 0 failures, ctest 2/2)
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
