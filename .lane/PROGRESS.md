# Lane 1 progress (State-OS v1 engine: keyed header + refusal + one-op restore + companion)

Checklist:
- [ ] pure header/container module (examples/server/stateos-header.*)
- [ ] llama API: append-to-file, load-file-range, layout descriptor
- [ ] common/speculative: MTP companion ctx + warmed-heads get/set
- [ ] server SLOT_SAVE: keyed container (TOKS, MAIN, CKPT, COMP), atomic tmp+rename
- [ ] server SLOT_RESTORE: verify-before-load, 409 refusals, legacy refusal, one-op restore
- [ ] slot errors reach the HTTP handler (legacy result queue)
- [ ] unit tests (tests/test-stateos-header.cpp) registered in CTest
- [ ] .lane/GPU-VERIFY.md
- [ ] build after 23:15 ET in build-stateos-l1 + run non-GPU tests
- [ ] .lane/REPORT.md

Log:
- 17:50 ET pure module stateos-header.{h,cpp} written (sha256, H/S/I header, verify, container scan, checkpoint codec)
- 17:53 ET llama API (append/range-load/layout desc) + speculative companion accessors committed
- 17:57 ET server save/restore + /list + legacy-queue slot errors + tests/test-stateos-header.cpp committed
- 18:10 ET writing .lane/gpu-verify-l1.ps1 + GPU-VERIFY.md (no GPU use; coordinator runs it)
- 18:05 ET recipe committed; missing file = 409 state_missing (harness lane contract); fingerprint moved to stateos-model.cpp + tested
- 18:15 ET build-l1.cmd / test-l1.cmd staged for 23:15 (no compile before then)
