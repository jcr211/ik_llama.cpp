# stateos-lane1 — REPORT (from the lane's final message, 2026-09-23)

Base d583c220 (lane 0). Commits 5cfb3c85, a8967890, 2555da57, 8b6b16d2, b401d045, 1fef747d, ce691681, 6330f41c
(+ progress). No GPU used. Build: `.lane/build-l1.cmd` exit 0 (`build-stateos-l1`, Ninja, CUDA arch 120,
AVX512, tests on). Tests: `.lane/test-l1.cmd` exit 0: test-stateos-header 156 checks, 0 failures; ctest 2/2.

Container v1 (one file per save): 'LSOS' | v1 | header_len | header text, then sections TOKS (int32 ids), MAIN
(llama seq state, streamed), CKPT (checkpoints with pos_max < pos_next), optional COMP (companion sub-header + state),
END at EOF. Written to .tmp and renamed. Header lines `H|S|I key=value`; the server's class wins (a file can't demote
a hard field); unknown hard → refuse, unknown soft → warn.
Hard: model_fingerprint_v2 (see the fix round below), n_ctx, cache_type_k/v, rope,
kv_layout_version, system_prompt_sha256, kv_geometry, n_tokens + token_sha256 (recomputed from TOKS).
Soft: build. Info: saved_unix, slot_id.
Restore: verify fully (structure, header, TOKS hash, identity, n_tokens ≤ n_ctx, vocab range, CKPT, COMP verify) →
409 with slot_untouched:true and type state_refused (+ refused_field, refused[]) / state_legacy_unkeyed /
state_corrupt / state_missing. Then ONE destructive op: bounded MAIN range-load → MTP invalidate from 0 + warmed-heads
reset (lane 0 only invalidated from pos_next) → clear checkpoints + server_cached_prompt → cache_tokens from TOKS →
COMP load or "skipped: reason" → reinstall checkpoints. A MAIN failure mid-load clears the slot (500,
slot_untouched:false). Success adds a `stateos` object (token_sha256, bytes, companion, warnings, checkpoints_restored).
Also: a pre-existing bug fixed (SLOT_* task errors never reached HTTP → hung requests; now send_slot_error);
`GET /props` `stateos: {version:1, keyed_header:true, companion}`; /list shows State-OS files; new llama APIs
llama_state_seq_append_to_file / load_file_range / layout_desc; common MTP companion accessors.
Not verified (GPU): greedy identity at 4K/32K, live 409s, companion, bytes at 32K/190K, /props on the patched server
→ `.lane/GPU-VERIFY.md` + `.lane/gpu-verify-l1.ps1` (port 8101, auto-stop, standing :8099 relaunched in finally).
Known limits: the new llama APIs are exercised only via the server; report-only non-identical components (kp_l,
sampler, ngram_mod, hidden-state cache → the first post-restore step drafts nothing); corruption inside a structurally
valid MAIN is caught after seq_rm (slot cleared, 500), so a content checksum is v2; save answers 501 for media slots.

## Fix round after the Opus review (FIX-FIRST @ c8a9261d)

- M1 (P0): restoring a state saved from an empty slot aborted the server: `read_kv_cache_meta` indexed
  `batch.pos[-1]` / `cells[head - 1]` for 0 cells. It now returns right after its `seq_rm` when `cell_count == 0`
  (`src/llama.cpp`). The server restores an empty-state file as an erase: the loader is not called, the answer is 200
  with `stateos.empty: true`. Saving an empty slot stays allowed.
- M2 (P1): a zero-size MAIN is refused with 409 `state_corrupt` (`section:MAIN`) before anything is touched
  (`stateos_check_sections`, unit-tested). `nread == 0` counts as a failure. A zero-size COMP state is skipped, and a COMP
  load must return non-zero.
- M3: `gpu-verify-l1.ps1` and `GPU-VERIFY.md` gain three steps. (a) An empty-slot round trip: 200, the server stays up,
  and the next request re-prefills correctly. (b) A MAIN tamper (`cell_count` + 1 in a well-formed container): 500 with
  `slot_untouched:false`, the server stays up, and the re-prefill is correct. (c) Leg B, a COMP sub-header tamper: 200
  with the companion skipped.
- SHOULDs:
  - A failed COMP load now drops the companion's partial cells.
  - The fingerprint is computed once, right after the model load, and only with `--slot-save-path`. On failure it
    fails open with a warning: `/props` omits `stateos`, and save/restore answer 500.
  - `/props.stateos` appears only when `/slots` is served and the model identity exists.
  - Save/restore are wrapped in try/catch and answer 500. `slot_untouched` is derived from a `destroyed` flag, and a
    catch after destruction clears the slot.
  - The `/slots` handlers register the waiting id before posting (this fixes the lost-fast-409 race).
- P3 items also done:
  - The section table is capped at 16 sections.
  - TOKS is bounded by the slot `n_ctx` before it is read.
  - Checkpoint positions are validated (ordered, non-negative, no `pos_max_prompt + 1` overflow).
  - The save commits via `MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)` with retries. The old state is never deleted
    first, and the temp suffix is now `.stateos.tmp`.
  - Slot success bodies use the replace-mode JSON dump, so non-UTF-8 soft values cannot turn a successful restore into
    a 500.
  - `stateos.kv_pos_max` is reported on save and restore, so the GPU run shows the KV↔TOKS invariant.
- Coordinator ruling on identity: the header field is now `model_fingerprint_v2`. For EVERY shard of a split GGUF
  (named `<prefix>-0000i-of-0000N.gguf`, count read from `split.count`) it hashes: the GGUF header bytes
  `[0, data_offset)`, the file size, and 16 evenly spaced 64 KiB windows of tensor data (the whole region when it is
  ≤ 1 MiB). The shard count is hashed too. A missing shard or a misnamed split refuses to produce an identity.
- **Residual, documented:** an edit that keeps every header byte and every file size identical, and changes only tensor
  bytes outside all sampled windows, is NOT detected. Examples: in-place GGUF surgery of same-shape weights, or a
  fine-tune saved with copied metadata. Restoring onto such a model continues on a KV computed with other weights. That
  is a silent quality failure, not a crash, because the llama reader still checks shapes and types. The unit test
  demonstrates both sides on a 4 MiB tensor: a flipped byte inside a window changes the fingerprint, and a flipped byte
  in a gap does not. A whole-file sha256 would cost minutes per start on the ~180 GB model, so v1 accepts this residual
  (coordinator ruling).
- Not done, noted: F9 is not enforced; it is report-only via `kv_pos_max` until the GPU run shows the invariant holds
  on legitimate flows. F11 is left open: runtime LoRA / control-vector / override-kv identity, a golden layout-descriptor
  test, `/list` vocab-range and exposure, the state_missing naming for stat/permission errors, and
  FlushFileBuffers/stale-`.tmp` cleanup.
- Build: `.lane/build-l1.cmd` exit 0. Before it, a process check showed no nvcc, cl, cmake or ninja from another
  lane. The lane files have no diagnostics.
- Tests: `.lane/test-l1.cmd` exit 0. `test-stateos-header` runs 195 checks with 0 failures, and ctest passes 2/2
  (`test-speculative-params` too). The new checks cover section checks, the `/props` predicate, file replace, the
  fingerprint windows, real tensor data and split shards.
