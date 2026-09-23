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
