# SV2-E1 lane progress (lane/stateos-tail, base fix/dsa-inv-sum-overrun @ d583c220)

Order: `D:/Projects/longspear/docs/drafts/stateos-v2-merged-plan-20260924.md` §7 ORDER SV2-E1
(amendments §1 M1/M3/M4/M8/Q1, §2, V1, V2). Draft C1: `docs/drafts/stateos-v2-design-20260924.md`.

## Status
- [x] worktree created (`git worktree add -b lane/stateos-tail ... d583c220`)
- [x] commit 0 code-reading note (this file)
- [ ] commit 1 telemetry
- [ ] commit 2 helper refactor + test
- [ ] commit 3 capped/shadow writer + tests
- [ ] commit 4 tail snapshot
- [ ] commit 5 xcheck
- [ ] commit 6 tests + tools + build script + launchers
- [ ] build (compile slot granted by coordinator 09-23; stale BOX-LOCK renamed by coordinator;
      check `tasklist` for SL-1 nvcc/cl/cmake/ninja before every compile)

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

## Log
- 09-23: read order, draft C1, code; coordinator renamed stale BOX-LOCK and granted a compile slot
  (SL-1 tasklist check still binding).
