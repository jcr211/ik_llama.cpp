# SV2-E1 lane progress (lane/stateos-tail, base fix/dsa-inv-sum-overrun @ d583c220)

Order: `D:/Projects/longspear/docs/drafts/stateos-v2-merged-plan-20260924.md` §7 ORDER SV2-E1
(amendments §1 M1/M3/M4/M8/Q1, §2, V1, V2). Draft C1: `docs/drafts/stateos-v2-design-20260924.md`.

## Status
- [x] worktree created (`git worktree add -b lane/stateos-tail ... d583c220`)
- [x] commit 0 code-reading note (this file)
- [x] commit 1 telemetry (4c0258c8) — built (full Release build, llama-server)
- [x] commit 2 helper refactor + test (b0d09d9d) — built, test-partial-state 14/14
- [x] commit 3 capped/shadow writer + tests (f8fa7e91) — built, test-partial-state 41/41
- [x] census tool written early; reproduces the order's numbers on ik-serve-8099.err.log (.lane/census-prod.txt)
- [x] commit 4 tail snapshot (865ef7e7) — built
- [x] commit 5 xcheck (0f919909) — built
- [x] commit 6 tests + tools + build script + launchers (a01586b5)
- [x] build: Release, all three targets, BUILD_STATEOS_TAIL_OK (flags are runtime env: one build covers both
      states); test-partial-state 41/41, test-stateos-tail 65/65, node tools tests 6/6
- [x] flag audit: `.lane/flag-audit.sh` -> `.lane/flag-audit.txt`; census: `.lane/census-prod.txt`
- [x] merged lane/ple-hist-rewind (b54ba500; conflict only in tests/CMakeLists.txt). Code check:
      - tail restore (flag on) and the flag-off restore both end in apply_checkpoint -> keep_first(n_past)
        -> the server-resume choke point in batch_pending_prompt (n_prompt_tokens_processed == 0, p0 > 0),
        which rebuilds the history from cache_tokens[n_past - n_hist, n_past) at p0. Covered, no extra call.
      - xcheck replays (tail -> X, ref -> X) decode inside apply_checkpoint, outside the choke point:
        stateos_decode_cached now calls common_ple_history_set(site=stateos-xcheck) before each replay.
        The xcheck ends by restoring the ref checkpoint, which then passes the choke point. Covered.
      - tail snapshot now also skips with cause=system-prompt (positions offset by system tokens).
- [x] rebuilt; unit tests with CUDA_VISIBLE_DEVICES=-1: test-partial-state 41/41, test-stateos-tail 65/65,
      test-ple-hist all passed, node tools 7/7
- [x] launchers set LONGSPEAR_PLE_HIST_REWIND=1 + LONGSPEAR_PLE_HIST_LOG=1 in every arm; census has
      `--check step1|step2|step3` (exit 2 = stop) incl. `[ple-hist] reset` at pos > 0 == 0 and "armed"

- [x] fix round 1 (cross-family review of be896293: Grok SAFE w/ findings, Opus NOT SAFE): items 1-7
      done; rebuilt (slot-file change); tests with CUDA_VISIBLE_DEVICES=-1 green; node tools 12/12
- [x] fix round 3 (re-review of cd34defb): gate B built once from A0, one traffic-defined step-3 class +
      T1 mechanism check, VOID on traffic mismatch / insufficient-sample, id count == usage, restore
      binding by order + content; node tools 14/14 (tools only, no engine change, no rebuild)
- [x] fix round 4 (review of 322dbfa7): strict tail restore + xcheck refusal (Opus repro test), mislaunch
      VOID in steps 2/3, step-3 floor rule, CUDA errors STOP in every arm, A1 request-A = determinism,
      erase no-op documented; node tools 19/19 (tools only)

- [x] fix round 5 (final check of 1f0389b7): launcher flags sidecar, mislaunch from recorded flags only,
      A1 determinism before drops, hard tail-integrity misses, C parse errors drop, stale comments;
      node tools 20/20 (tools + launcher only)

## GPU-window assumptions (W-SV2)
- Every arm (P0, T1, A0, A1, C, A5, A3, probe) launches via launch-stateos-tail-8099.ps1, so all run
  with LONGSPEAR_PLE_HIST_REWIND=1 and LONGSPEAR_PLE_HIST_LOG=1; arms differ only in the tail lever
  (and the benign-arm ExtraArgs).
- Chain auto-stop after each step/row: `node tools/stateos-div-census.mjs --check stepN <log>`
  (step3: `--check step3 --p0 <P0 log> <T1 log>`). Mechanism checks in every step: at least one
  `[ple-hist] set` line (repair armed), zero `[ple-hist] reset` at pos > 0 (no unrepaired rewind), zero
  CUDA error lines; plus the step's own criteria from merged plan section 4.
- The xcheck relL2 is only meaningful with the repair on in both paths (it is: the tail replay, the ref
  replay and the final resume all set the history).
- PCIe replay counter before/after stays the coordinator's (not in the log).
- Launcher: `-LogStem` is mandatory and must be new per arm (an existing .err.log is refused, never
  truncated). A BOX-LOCK.json refuses the launch unless its `owner` equals `-LockOwner`, default
  `W-SV2 chain`: the chain writes exactly that owner string into its own lock.
- MISLAUNCH (coordinator ruling, round 5): decided ONLY from the launcher's recorded flags, never from the
  lever's own output. Before starting the server the launcher writes `<LogStem>.flags` next to the log: one
  `[stateos-flags]` line with the effective LONGSPEAR_STATEOS_DIV_LOG, _TAIL_SNAPSHOT, _TAIL_XCHECK and
  LONGSPEAR_PLE_HIST_REWIND, _LOG (0/1). The census and the gate read it (a `[stateos-flags]` header line in
  the log also counts). VOID ("mislaunched") = the recorded flags differ from the step's required set; no
  record = VOID ("flags unknown"). Required sets (PLE_HIST_REWIND=1 and PLE_HIST_LOG=1 in all):
  step 1 DIV_LOG only; step 2 DIV_LOG + TAIL_SNAPSHOT + TAIL_XCHECK; step 3 P0 DIV_LOG only, T1 DIV_LOG +
  TAIL_SNAPSHOT; step 4 C DIV_LOG + TAIL_SNAPSHOT, all other arms DIV_LOG only. With the right flags
  recorded, every tail failure (verify failure, tail never chosen, sha mismatch, writer refusal) is STOP.
- Tail-integrity misses (restore-failed / verify-failed / rewind-refused after a tail choice, tail sha
  mismatch) are hard: STOP even when a VOID check also fails (steps 2 and 3), like CUDA errors.
- Step 1 denominator: restore-branch decisions MINUS new-conversation resets (outcome reset:no-checkpoint
  with common prefix < 64). Both counts are printed.
- Steps 2 and 3: zero outcomes restore-failed / verify-failed / rewind-refused after a tail choice, and
  zero tail sha mismatches. `reset:xcheck-flag-off` after a tail choice is a hit, not a miss. Step 2's
  hit-rate denominator = last-token divergences with a tail available + tails eligible at release that the
  writer did not produce, and step 2 also requires tail_skip cause refused / order / cache-short /
  size-mismatch == 0 each.
- Census `--check` exit codes: 0 PASS, 2 STOP (a miss: the lever's kill), 3 VOID (traffic mismatch or
  protocol error: the measurement did not answer; NOT a C1 kill).
- Step 3 rule (coordinator rulings on Opus S2/N3 and the Grok re-review):
  - ONE eligibility class in both runs, a property of the TRAFFIC: last-token divergences whose previous
    generation ended with a drafted round that accepted >= 1 draft, computed identically from each run's
    own [stateos-div] lines. (n_acc >= 1 is exact: shadow_pos = root - 1, last cached = root + n_acc, so
    shadow_pos <= last cached - 2 iff n_acc >= 1; the root position is n_past_pre_spec,
    server-context.cpp:5467-5468, recorded as root - 1 at llama.cpp:10144; spec_pos_base is root + 1.)
  - Traffic sanity, VOID when violated: each run has >= 5 eligible events, T1's eligible count is within
    0.5x-2x P0's.
  - Protocol, VOID ("mislaunched"): P0's or T1's recorded flags differ from their required sets (see
    MISLAUNCH above), or are missing.
  - Mechanism, STOP when violated: in T1 a tail was written AND chosen (outcome restored, reason tail) on
    >= 90 % of eligible events.
  - Effect (coordinator protocol ruling on the step-3 floor), STOP when violated: a tail restore
    necessarily re-prefills the final round's accepted drafts (the shadow sits at root - 1), so a
    tail-served event's gap is n_acc. Over eligible events, T1's mean gap <= floor + 0.10 x (P0's mean
    gap - floor), where floor = the mean n_acc (prev_n_acc) of those events' final rounds, computed from
    each run's own lines (floor_T1 on the left, floor_P0 inside the bracket). That is >= 90 % of the
    ACHIEVABLE reduction. If P0's mean gap <= floor_P0 x 1.5 (nothing meaningful to save), step 3 is VOID
    ("gap too small to measure").
  - REPORT-ONLY: the raw per-event ratio T1 mean gap / P0 mean gap, and the all-last-token-events ratio.
  - A CUDA error line in the run's log is STOP even when a VOID check also fails (every step).
- Step 4 (gate driver):
  - Run A0 FIRST. Request B is built ONCE from A0's request-A output; every arm sends its own request A
    (its cache, and C's tail, exist), then A0's B tokens (sha recorded per row).
  - Every arm launched with -DivLog (C with -Tail -DivLog). Token ids come from /v1/completions logprobs
    and their count must equal usage.completion_tokens (a UTF-8-split token has no logprobs entry).
  - Drops: B sha differs across arms; C's request-A output differs from A0's; C's response could not be
    parsed (e.g. a UTF-8-split id-count mismatch in a drifted continuation; counted and reported); the
    request-B restore line of A0, A1 or C is missing or lacks tail_dist=1, or C's is not a strict tail
    restore (chosen_origin=tail, outcome=restored, reason=tail) (benign arms reach the same B from their
    own cache and are not constrained).
    Restore lines bind by request order AND content (n_past == forced index, cache-window marked token ==
    g_A0[G-2], prompt-window marked token == X). Dropped counts in gate.json.
  - Engagement: C needs >= 20 tail restores on its request-B lines; a tail restore is exactly
    chosen_origin=tail, outcome=restored, reason=tail (restored:xcheck-flag-off does NOT count: that server
    continued on the flag-off state). Otherwise `not-engaged`: VOID when tails were available on < 20 B
    requests (no eligible prompts), STOP when they were available and not used.
  - Pre-score refusals: CUDA error lines in ANY arm's log = cuda-errors (STOP: stops the window); an arm
    whose recorded flags differ from its required set, or with no flags record, = mislaunched (VOID); an
    HTTP or connection error on a C request where A0's row succeeded = shell-errors (STOP).
  - A1's request A differing from A0's is spec-on nondeterminism = void-determinism, counted BEFORE any
    drop (so it is not hidden when C's request A differs too).
  - Status: compatible-at-horizon PASS (exit 0); shellWorse, not-engaged:tails-not-used, cuda-errors,
    shell-errors STOP (2); insufficient-sample, void-determinism, mislaunched,
    not-engaged:no-eligible-prompts VOID (3) - the gate did not answer, not a C1 kill.
  - Step 2 VOID ("mislaunched") when its recorded flags are not DIV_LOG + TAIL_SNAPSHOT + TAIL_XCHECK (+ PLE).
  - Step-3 floors (coordinator ruling): floor_T1 on the left, floor_P0 inside the bracket, each from its
    own run's lines.
  - The slot erase before each prompt is a NO-OP in W-SV2 (/slots/:id exists only with --slot-save-path,
    which the launcher does not pass), so request A of prompt k+1 runs on prompt k's cache. Harmless: request
    A never chooses a tail (a tail sits at <= last cached - 2 of the previous generation, A diverges near the
    prompt start), A0/A1/C send the same request sequence and take the same restore path for A, and the
    tail writer's extra eviction cannot change C's list below the 32-checkpoint cap.
  - Forced token X differs from the original by id and by text, raw and with ' ', '\n', '\r' deleted, and
    keeps text after that deletion.
- Report-only: with the tail flag on, the tail buffer (~113 MiB) is allocated before the eviction loop,
  so the process transiently holds 33 checkpoints (+113 MiB host peak) during release.
- Slot files (--slot-save-path, not used by W-SV2): tail snapshots are not persisted (no origin/sha in the
  format, so a reloaded tail could not be revalidated).
- REPORT.md: blocked by a hook for subagents; the report is returned as text to the coordinator

## Commit 0 — code-reading note (base d583c220; line numbers are the base file's)

### (a) PARTIAL_ONLY payload layout and the single-sequence reader
Writer: `llama_data_write::write_kv_cache` (src/llama.cpp:10599-10653) for a non-openPangu arch:
1. Cell selection 10609-10631: every cell with `has_seq_id(seq_id)` (or non-empty for seq −1), grouped into
   contiguous `[first, second)` ranges; `cell_count` = number of such cells.
2. Only if `kv_self.any_compacted()` (SWA compaction; qwen4exp does not compact): the "SWAC" header
   10640-10647 (magic, size_swa, live rows, pos_base_swa, head_swa).
3. `u32 cell_count` 10649.
4. `write_kv_cache_meta` 10273-10291: per selected cell `i32 pos` + `u32 n_seq_id`; for a single sequence
   `n_seq_id = 0` and no ids follow ⇒ **8 B per cell** (this is M1's 7.99 B/cell growth).
5. `write_kv_cache_data` 10336+: `u32 v_state`, `u32 n_layer`; per layer K header `i32 type` + `u64 row size`
   (PARTIAL_ONLY: `has_k_cache` is false unless the layer is compacted ⇒ `-1, 0`, no rows, 10361-10377);
   V headers the same way (12 B/layer for v_state 0 or 1, none for v_state 2, 10396-10474);
   `u32 qnext_state` 10476; if set, per layer `i32 s type`, `u64 s row size`, `u32 s_rows`, then
   `s_rows × row size` bytes read from `s_l[il]` at row `cells[seq_id].src` (10484-10515); then
   `u32 dsa_indexer_state` and (PARTIAL_ONLY) no kr rows (10517-10538); DSV4/openPangu blocks are
   arch-specific and not taken for qwen4exp.
So a checkpoint = small headers + 4 + 8·cell_count + the recurrent rows. The rows are the only
state; the metadata is what re-tags attention cells on restore.

Reader: `llama_data_read::read_kv_cache_meta` single-sequence branch (10751-10802): refuses
`cell_count > kv size`, then `llama_kv_cache_seq_rm(kv, dest, −1, −1)` (10767) untags every cell of the
sequence, then `llama_kv_cache_find_slot` over a synthetic batch of the saved positions (10769-10792)
re-tags `cell_count` contiguous cells starting at `head` (0 after the full seq_rm when the sequence starts at
cell 0). The attention K/V bytes are NOT rewritten: they are still in those cells. This only works if
cell index == position for the sequence (V1: defrag off, `defrag_thold = −1` default at llama.cpp:8075,
used at 7222). Hence the tail writer must emit metadata for exactly the cells with `pos ≤ shadow_pos`:
restoring more cells would make `seq_pos_max` = cache end ≠ checkpoint `pos_max` and
`verify_restored_checkpoint` (server-context.cpp:3656-3668) would force a full re-prefill.
The recurrent rows are read back into row `seq_id` (11241-11266) while the writer reads row
`cells[seq_id].src` — identical for the single-slot server (src of cell 0 is 0).

### (b) Checkpoint-list order invariant and the reverse search
- `create_checkpoint` (server-context.cpp:3840-3875) only appends when
  `cache_tokens.n_tokens() > checkpoints.back().n_tokens` (3850), so the list is ascending in `n_tokens`
  (and, text-only, in `pos_max`). Eviction (3854-3865) removes interior entries (variance) or the front.
- `apply_checkpoint` (3686-3793) searches `rbegin..rend` for the first entry with
  `pos_max < pos_min_thold = max(0, pos_next − 1)` (3699-3705): the newest usable one, which is only
  correct if the list is ascending. It restores via `llama_state_seq_set_data(PARTIAL_ONLY)`, resumes at
  `max(pos_min + 1, pos_max)` = `pos_max + 1` for hybrid caches (public `seq_pos_min` returns `pos_max`
  for hybrid/recurrent caches, llama.cpp:10126-10131), then erases every entry with `pos_max > pos_min_thold`
  (3781-3792). A tail snapshot therefore has to be appended before the release checkpoint, with
  `shadow_pos` > the current `back().pos_max` (M4).

### (c) What the gpu-fallback shadow holds
- Save: server-context.cpp:5023-5051 calls `common_speculative_before_draft` only for slots with a
  non-empty `i_batch_dft`; `n_past_pre_spec` = the position of the root token (5027-5028).
  `common_speculative_checkpoint_save` (common/speculative.cpp:2478-2516) stores `ckpt.n_past = root_pos`
  and calls `llama_spec_ckpt_save` → `llama_kv_cache::checkpoint_save` (llama.cpp:2143-2189): copies every
  `s_l[il]` into the full shadow (`ggml_dup_tensor`, 2085) and snapshots `cells/head/used`. The shadow
  = recurrent state after position `root_pos − 1` (the verify batch starts with the root at `root_pos`;
  the replay re-adds `ckpt.sampled` at `ckpt.n_past`, speculative.cpp:2597).
- Full accept: `common_speculative_commit` (2653-2722) runs `seq_rm(pos_base + n_acc, −1)` (2719) and
  `checkpoint_discard` → `checkpoint_delete` only clears `saved` (llama.cpp:2239-2241). Shadow keeps
  the pre-round bytes = state at `root_pos − 1`.
- Partial accept: `llama_spec_ckpt_restore_ex` GPU_FALLBACK (10030-10039) → `checkpoint_restore`
  (2191-2237) copies shadow → live and `cells = cells_snapshot`, then `seq_rm(root_pos, −1)`; the replay
  re-decodes root + accepted drafts (speculative.cpp:2593-2616). The shadow is only read: still the state
  at `root_pos − 1`.
- Root-only round, three code paths:
  (i) draft shorter than `min_usable_draft` (server-context.cpp:3607-3614): `i_batch_dft` cleared, no
      save (5024-5026 skip), the token goes through the ordinary sample path (4821-4903);
  (ii) save failed → `make_root_only` (4977-5021) — `_save_at` sets `shadow_pos = −1` before the copy;
  (iii) DFlash target-only (`spec_target_only`, 3616-3618): `i_batch_dft = {root}` so a save DOES run with
      `max_tokens = 1` (not in production).
  In (i) the shadow is not overwritten, so it still holds an OLDER round's pre-state: a valid snapshot of
  an older position of the same append-only history (Q1) — eligible with a larger gap, as long as
  `shadow_pos` was recorded at save time.
- Stop-cut final round: 4357-4365 push all accepted drafts to `cache_tokens` and commit (4367) BEFORE
  the `process_token` loop (4391-4421) finds the stop at `ids[i]`; the cache then extends past what the
  client saw. Shadow = `root_pos − 1`; the next prompt diverges at or before the first token after the stop.
- Commit failure (4367-4388): `slot.release()` directly (no release checkpoint). The failure comes from
  a failed restore (speculative.cpp:2548-2552) or a failed re-decode (2610-2616); the live state is
  undefined, so `shadow_pos` must be invalidated there.

### (d) The bonus token
`ids.back()` becomes `slot.sampled` (4364) and is NOT pushed to `cache_tokens` (the loop 4361-4363 stops at
`ids.end() − 1`). It is decoded only as the next round's root (3604). When generation stops it is never
decoded; the last CACHED token is the last accepted draft, or the root of a round with 0 accepted drafts.

### (e) `n_past_offset`
`get_common_prefix` (server-common.cpp:1597) is a fuzzy text-level match, so the cache index
(`prefix.first`) and prompt index (`prefix.second`) can differ; 4021-4023 sets
`n_past_offset = n_past_prompt − n_past`. `server_prompt_checkpoint_update` (21-35) stores
`pos_*_prompt = pos_* + offset`; `apply_checkpoint` (3738-3743) maps back with
`size_up_to_pos(min(pos_next(n_past_prompt), pos_max_prompt + 1))`. Reset to 0 at 2135, 4033, 4045, 4120.
For text-only caches `pos_next(n) == n` and `size_up_to_pos(p) == min(p, size)` (server-common.cpp:1173,
1217), so the tail's `n_tokens = shadow_pos + 1` and `pos_max_prompt = shadow_pos + n_past_offset`.

### Conflicts / refinements vs the order (recorded, following the order's intent)
1. "Speculative generation creates no interval checkpoints" is literally true of
   `speculative_decoding_accept`, but root-only rounds of kind (i) run the ordinary sample path, which calls
   `create_checkpoint_at_interval` (4876) when `n_decoded > 1`. So `gen-interval` checkpoints CAN appear
   during a speculative generation; the eligibility rule `shadow_pos > back().pos_max` handles them.
2. "`cache_end − 2`" is read as `seq_pos_max − 2` (last cached position − 2). With
   `shadow_pos = root − 1` and last cached `= root + n_acc`, this is exactly "the final round accepted ≥ 1
   draft", i.e. the snapshot can serve a last-token divergence (design C1: `D ≥ root + 1`).
3. `apply_checkpoint` never calls `seq_rm` before `set_data` for non-openPangu; the reader's
   `seq_rm(−1,−1)` does it. So "`seq_rm` with `p0 ≤ shadow_pos`" invalidation also fires on every
   `set_data`, which is what the order wants anyway.
4. The per-step (SL-1) path writes `kv.cells[seq_id].pos` (llama.cpp:10024); out of scope (V2: PER_STEP
   never records `shadow_pos`).

## Open items from the coordinator
- PLE n-gram history (coordinator 09-23, from the SL-1 review): qwen4exp keeps host-side per-sequence
  `lctx.ple_hist` that no save/restore covers; after ANY rewind (tail restore, flag-off restore, xcheck
  replay) the first 2 re-decoded tokens get wrong PLE rows. Do NOT implement a fix here: lane
  `lane/ple-hist-rewind` (D:/AI/worktrees/ik-ple-hist) adds `llama_ple_history_set` behind
  `LONGSPEAR_PLE_HIST_REWIND=1`. When the coordinator says it is ready: merge it and call the setter after a
  tail restore (history = cached tokens before the resume position). GPU-window assumption: the xcheck
  comparison is only meaningful when BOTH paths set the history.

## Log
- 09-23: read order, draft C1, code; coordinator renamed stale BOX-LOCK and granted a compile slot
  (SL-1 tasklist check still binding).
- 09-23: commits 1-3 built and tested. Session hit the API limit mid commit 4; resumed. Commit 4 in progress:
  stateos-v2.h gained eligibility/order/search/SHA-256; server-context.cpp has create_tail_snapshot();
  still to wire: call at release before create_checkpoint, sha check in apply_checkpoint's search.
