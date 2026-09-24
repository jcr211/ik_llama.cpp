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

## Fix round 2 (re-review of ec0b962a = ALLOW; N1/N2 before merge, N3/N4 cheap)

- N1: `llama-gguf-split --merge` writes `split.count = 0`. The fingerprint now treats 0 as a single file, like the
  loader, instead of refusing it (which disabled State-OS on merged GGUFs). Unit case: a tensor GGUF with
  `split.count = 0` fingerprints. Negative values of a signed type are still refused.
- N2: KV↔tokens.
  - Save refuses with 409 `state_inconsistent` (`slot_untouched: true`, no file written) when the slot lists tokens
    but holds no KV cells.
  - Restore answers 500 with `slot_untouched: false` when a MAIN loads cleanly but leaves no cells under a non-empty
    TOKS. That became reachable once the empty-sequence early return landed. The slot is cleared and re-prefills.
  - Pure helper `stateos_kv_consistent`, unit-tested. The exact `pos_max == n_tokens - 1` relation stays
    report-only (`kv_pos_max`).
- N3: `stateos_init_identity` is wrapped in try/catch. A throw fails open with a warning: `/props` omits `stateos`,
  and save/restore answer 500.
- N4 (script):
  - Step (c) runs only when the 32K save reported `companion: saved`, and inside its own try. Otherwise it records
    FAIL with the reason and continues to the 190K measurement.
  - Steps (a) and (b) now report `verdict` = PASS / INCONCLUSIVE / FAIL. The mechanism conditions must always hold;
    an output-only difference while `identity_4k.restored_runs_agree` is false is INCONCLUSIVE, the same rule as the
    identity legs.
- Not done (P3, follow-ups): N4 (iii) COMP payload-tamper step, N4 (iv) RAM prompt-cache request, N5 nits, N6
  cosmetic, and the open parts of F6/F10/F11.
- Build: before it, no nvcc, cl, cmake or ninja from another lane was running. `.lane/build-l1.cmd` exit 0; no
  diagnostics in lane files.
- Tests: `.lane/test-l1.cmd` exit 0. `test-stateos-header` runs 202 checks with 0 failures; ctest 2/2.

## Fix round 3 (merge of lane/ple-hist-rewind)

- Merged `lane/ple-hist-rewind` (`b72eb38d`, `424cbe84`) as `1b3352ed`. The only conflict was `tests/CMakeLists.txt`,
  resolved by keeping both test registrations. `src/llama.cpp`, `include/llama.h`, `common/speculative.*` and
  `server-context.cpp` auto-merged into separate hunks.
- **The State-OS restore reaches the PLE choke point.** `stateos_slot_restore` installs `cache_tokens` from TOKS.
  On the next request, `batch_pending_prompt` keeps the common prefix and computes `p0 > 0`. On that first prompt
  batch (`n_prompt_tokens_processed == 0`) and before any decode, it calls
  `common_ple_history_set(..., system_tokens + cache_tokens[n_past - n_hist .. n_past), p0, "server-resume")`. So no
  extra `llama_ple_history_set` call is needed in the restore. An empty-state or failed restore leaves `n_past = 0`
  and `p0 = 0`, where the position-0 EOS convention applies. The gate is `LONGSPEAR_PLE_HIST_REWIND=1`; off, the
  behaviour is unchanged.
- **GPU script:**
  - Every 8101 server gets `LONGSPEAR_PLE_HIST_REWIND=1` and `LONGSPEAR_PLE_HIST_LOG=1` (plus `VERIFY_TIMING` and
    `CG_REVIVE`). They are recorded as `<name> env: ...` in `launch-args.txt` and removed before the standing
    relaunch.
  - Every restore round counts the `[ple-hist] reset` lines at pos > 0 (expect 0) and the `site=server-resume` sets:
    identity rounds, the empty round trip, the MAIN tamper and the no-companion restore.
  - Summary in `results.json → ple_hist`. `GPU-VERIFY.md` has a new section on this.
- `build-l1.cmd` also builds `test-ple-hist`, and `test-l1.cmd` runs it. The test script prints
  `CUDA_VISIBLE_DEVICES=[-1]`, a non-empty value.
- Build: a process check showed no nvcc, cl, cmake or ninja from another lane. `.lane/build-l1.cmd` exit 0 (120 steps).
- Tests: `.lane/test-l1.cmd` exit 0. `test-stateos-header` runs 202 checks with 0 failures; ctest 3/3
  (`test-speculative-params`, `test-stateos-header`, `test-ple-hist`).

## Follow-ups (branch lane/stateos-lane1-f11 from 7c77724b; build-stateos-f11)

Worktree `D:/AI/worktrees/stateos-lane1-f11`, one commit per item. The frozen acceptance worktree and its build were
not touched. No GPU was used.

1. `0c1bebea` — new hard field **`effective_model`**. It is `"none"`, or a length-prefixed sha256 over the active
   runtime items in load order: LoRA adapters (path, scale), applied control vectors, `--override-kv` entries, and
   expert-count overrides. A state saved under one adapter set is refused on another. Files written by 7c77724b lack
   the field, so this build refuses them as `<missing>`. That is fail closed: re-save them.
2. `14c84ffe` — the layout descriptor is rendered by `src/llama-state-layout.h`. The output text is byte-identical to
   before, so existing `kv_geometry` digests still match. `tests/test-stateos-layout.cpp` pins two goldens and checks
   that every field reaches the text. The BUMP RULE is documented in the header, and `write_kv_cache` points to it.
3. `4e1f27ca` — `/list` no longer decodes State-OS tokens to text. Those entries show `prompt: null`,
   `stateos.prompt_redacted`, `token_count` and `token_sha256`. Legacy entries get a vocabulary-range check (prompt null
   plus an error instead of calling the detokenizer), and the body uses the replace-mode JSON dump.
4. `a3ff750d` — a file that exists but cannot be stat'ed or opened, or is not a regular file, now answers 409
   `state_unreadable`. `state_missing` means only "no such file".
5. `842e16f6` — the save flushes to disk (`FlushFileBuffers`, or `fsync` on POSIX) before the commit rename. At startup
   with `--slot-save-path`, leftover `*.stateos.tmp` files older than 1 h are removed; only that suffix, only regular
   files, not recursive.
6. `b29a940c` — the CKPT section is bounded before it is read: `ctx_checkpoints_n` records, each at most one partial
   state of this context (the full MAIN size when the SWA window is compacted). An over-budget section is skipped,
   and the restore proceeds without checkpoints (`stateos.checkpoints: "skipped: …"`). An oversized record after
   decoding is corrupt.
7. `9436d471` — N5/N6 nits:
   - an exception after the save's commit now answers a minimal success, not "save failed";
   - the replace retries only lock, sharing and access errors, and never sleeps after the final attempt;
   - tokens > n_ctx is `state_refused`;
   - the empty-restore reply no longer mentions a companion the server does not have.

Build: before every build a process check showed no nvcc, cl, cmake or ninja from another lane.
`.lane/build-f11.cmd` exit 0 on every item.
Tests: `.lane/test-f11.cmd` exit 0 with `CUDA_VISIBLE_DEVICES=-1`: `test-stateos-header` 242 checks, 0 failures;
ctest 4/4 (`test-speculative-params`, `test-stateos-header`, `test-ple-hist`, `test-stateos-layout`).
Not verified on GPU: `effective_model` on a live server, and the /list, CKPT-skip and flush paths. The GPU script
tampers `effective_model` among the hard fields.

### F11 review round (REVIEW-F11 = FIX-FIRST)

- `4d44c638` **P1:** the CKPT per-record bound no longer depends on what the target slot holds at restore time.
  - A checkpoint is a fixed partial state plus 8 B per cell of its source conversation, so the old bound refused valid
    files as `state_corrupt`, or skipped their checkpoints, after a short conversation.
  - The bound is now `stateos_ckpt_record_bound` = the current partial size + 8 B × slot `n_ctx` (the MAIN size when the
    SWA window is compacted). It is overflow-safe and unit-tested: a 4K-conversation checkpoint against a 10-cell slot
    and an empty one, and a full list of 32 long checkpoints within budget.
  - `gpu-verify-l1.ps1` now asserts, after every identity round, `checkpoints_restored == checkpoints_saved` and
    status `restored` (`identity_*.ckpt_ok`).
  - This defect was only on this branch: 7c77724b does not contain `b29a940c`.
- `1e94d160` **P2-1:** `effective_model` now covers startup control vectors (`--control-vector`,
  `--control-vector-scaled`, `--control-vector-layer-range`, all in `params.control_vectors`).
- `e2b76a3a` **P2-2, honest stamp.** Design: a server-wide adapter generation, bumped on SET_LORA and on every
  control-vector apply. Each slot records the generation its KV was built under.
  - When the stamp is set: a prompt that starts from an empty KV, erase or clear, and a verified restore.
  - A RAM prompt-cache load sets it to "unknown" unless the adapter set never changed since startup.
  - A save whose slot generation differs answers 409 `state_adapters_changed`.
  - Why this over clearing every slot on an adapter change: it is equally correct for State-OS and does not change
    serving behaviour outside it. The predicate is unit-tested.
- `fea27876` **P2-3:** names ending (case-insensitively) in `.stateos.tmp` are refused with 409
  `state_name_reserved` for `/slots` save and restore, and as a `/rename_prompt` destination. The startup cleanup
  therefore never meets a committed state.
- `60f9c9a4` **P3s:**
  - Before the flush and rename, the finished temp file must be exactly `stateos_container_size(header, sections)`
    bytes, so a temp deleted and recreated mid-save cannot replace the last good state.
  - The committed-save fallback reply carries `stateos.token_sha256`.
- **7c77724b files:** states saved by tonight's acceptance binary (7c77724b) have no `effective_model` field. This
  branch refuses them with 409 `state_refused`, `refused_field: effective_model`, `saved=<missing>`. This is
  deliberate: that build could not know the adapter set. Their `kv_geometry` still matches, because the layout
  renderer is byte-identical. Re-save with this build.
- **Left as follow-ups (review P3):**
  - `effective_model` over-refuses harmless cases: `--lora-init-without-apply`, and LoRA paths spelled differently.
  - The flush has no sharing-violation retry, and POSIX has no directory fsync after the rename.
  - One layout golden taken from a real 7c77724b descriptor.
  - `llama_state_layout_fmt`'s 511-byte fragment limit (assert or grow it).
- Build: before every build a process check showed no nvcc, cl, cmake or ninja from another lane.
  `.lane/build-f11.cmd` exit 0.
- Tests: `.lane/test-f11.cmd` exit 0 with `CUDA_VISIBLE_DEVICES=-1`: `test-stateos-header` 264 checks, 0 failures;
  ctest 4/4.

### F11-2 review round (REVIEW-F11-2 = FIX-FIRST)

- `c5d66f76` **P2-A + P2-B, cached stamp.** `effective_model` is now a cached string (`stateos_effective_cur`).
  - Only the main loop writes it: at startup, after SET_LORA, and before every return of
    `apply_control_vectors_internal`. A failed apply writes the sentinel `unknown`.
  - Save and restore read the cached string, never the live scales.
  - The startup `cvec-startup` parts are dropped from the stamp after the first runtime control-vector change, which
    replaces the startup vectors.
  - `/lora-adapters` and `/control-vectors/apply` no longer mutate anything on the HTTP thread. They forward the request
    in the task, and the task applies it all-or-nothing (`stateos_apply_scales`, which validates every id first). A bad
    id answers 400 and leaves the scales, the generation and the cached stamp unchanged.
  - Unit tests cover the helpers: the cvec parts with startup live and dropped, and a bad-id request leaving the
    scales untouched.
  - **Failed apply.** While the stamp is `unknown`, every save is refused with 409 `state_adapters_unknown`, and every
    restore is refused outright with 409 `state_adapters_unknown` and `slot_untouched`, whatever the file's header says
    (F11-3 P3-2; before that it was an `effective_model` mismatch, which a hand-made `effective_model=unknown` header
    could pass). Only a later successful
    `/control-vectors/apply` clears it; a `/lora-adapters` call keeps it (F11-3 P3-1).
- `91579f16` **P3-1:** `/rename_prompt` runs `fs_validate_filename` on both names (400). The reserved-suffix check now
  strips Windows trailing dots and spaces first, so `x.stateos.tmp.` and `x.stateos.tmp ` are refused too. It applies to
  save and restore names and to both rename names.
- `82c7b1b1` **P3-2:** `ckpt_ok` is a hard check in `gpu-verify-l1.ps1`: a dropped checkpoint fails that identity leg
  (`fail_reason`), and the 4K KILL message names it.
- `9729d763` **P3-4:** `system_prompt_update` records the generation its seq-0 KV was computed under. A slot that
  starts from empty copies that KV in, so it is stamped with `stateos_slot_start_gen`:
  - the current generation when there is no system prompt, or when the system prompt was computed under the current
    set;
  - otherwise `-1` (unknown), and the save is refused with `state_adapters_changed`. Unit-tested.
- **P3-3, recovery for "unsaveable after an adapter change".** A slot whose KV predates the current adapter set answers
  409 `state_adapters_changed`. The same happens for a RAM prompt-cache load after any adapter change, and for a start
  from a stale system prompt (see P3-4, and its own recovery below). To recover (not the stale-system-prompt case):
  1. Erase the slot: `POST /slots/{id}?action=erase`. This clears the KV and stamps the current generation.
  2. Re-prefill the conversation with a normal request, then save again.
  - A request whose prompt diverges at token 0 also re-prefills from empty and has the same effect.
  - If the stamp itself is `unknown` (failed control-vector apply), first make it known with a successful
    `/control-vectors/apply` (an empty array, which disables steering, also counts). A `/lora-adapters` call does not
    clear `unknown`.
  - **Stale legacy system prompt (F11-3 P3-4).** Erase and re-prefill do not help here: the next fresh start still
    copies the system KV computed under the old adapter set, so it is stamped unknown again. On transformer models,
    erase also drops the slot's copy of the system KV, and the trim path does not copy it back (an upstream legacy
    bug). The recovery is to set the system prompt again: send a completion request that carries the legacy
    `"system_prompt"` field (the same text is fine). That runs `system_prompt_update`, which recomputes the system KV
    under the current adapter set, and it releases and clears every slot. Then re-prefill the conversation and save.
- Build: before every build a process check showed no nvcc, cl, cmake or ninja from another lane.
  `.lane/build-f11.cmd` exit 0.
- Tests: `.lane/test-f11.cmd` exit 0 with `CUDA_VISIBLE_DEVICES=-1`: `test-stateos-header` 285 checks, 0 failures;
  ctest 4/4.
- Not verified on GPU: the handler forwarding, the failed-apply sentinel, and the system-prompt stamp.

### F11-3 review round (REVIEW-F11-3 = FIX-FIRST)

- `0acc6624` **P2-1:** `--lora-init-without-apply` loads the adapters at their scales but never applies them.
  `stateos_lora_live` (false at load under that flag, set by SET_LORA after `llama_lora_adapters_apply`) now gates the
  `lora` lines, built by the pure `stateos_lora_parts` (same text as before, so existing stamps are unchanged).
  Unit-tested.
- `5eab5279` **P3-1:** SET_LORA refreshes with `apply_ok = (stamp != unknown)`, so a `/lora-adapters` call keeps the
  `unknown` a failed control-vector apply left. Only a successful `/control-vectors/apply` clears it. The save refusal
  text and the recovery note above say so.
- `e3e4b7ab` **P3-2:** restore answers 409 `state_adapters_unknown` with `slot_untouched` while the stamp is `unknown`,
  before any identity comparison, whatever the file's header says.
- `8347edf7` **P3-3, UPSTREAM-CODE FIX** (upstream 17d101863, not State-OS logic, its own commit):
  - `apply_control_vectors_internal` now sizes the combined vector to n_embd × (n_layer − 1), zero-filled, and
    accumulates each vector over the common length only (`stateos_cvec_accumulate`, unit-tested).
  - This removes the heap overflow (a shorter vector loaded first, then a full one). It also removes the stale layers
    a shorter vector left from an earlier apply, so the applied steering no longer depends on history.
  - No change for the usual all-layer files.
- **P3-4:** the recovery for a stale legacy system prompt is to set the system prompt again, not erase + re-prefill
  (see the recovery note above).
- **P3-5, left as follow-ups:**
  - `/delete_prompt` has no `fs_validate_filename` or reserved-name check. It fails closed.
  - 8.3 short names can alias a `*.stateos.tmp` as rename's old name. This fails closed too.
  - `GET /lora-adapters` and `GET /control-vectors` read on HTTP threads. This is older code, out of scope.
- Build: before every build a process check showed no nvcc, cl, cmake or ninja from another lane.
  `.lane/build-f11.cmd` exit 0.
- Tests: `.lane/test-f11.cmd` exit 0 with `CUDA_VISIBLE_DEVICES=-1`: `test-stateos-header` 291 checks, 0 failures;
  ctest 4/4.
- Not verified on GPU: none of this round (no GPU).
