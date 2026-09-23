# SL-1 progress — tail-aware PER_STEP speculative checkpoints (qwen4exp)

Worktree `D:/AI/worktrees/sl1-spec-ckpt`, branch `lane/sl1-spec-ckpt` from `fix/dsa-inv-sum-overrun` @ d583c220.
Order: `D:/Projects/longspear/docs/drafts/decode-throughput-merged-plan-20260924.md` §5 (SL-1).

## Checklist
- [ ] commit 0 — `fix(iqk)`: empty row-group guard at the 7 MoE/dense partition sites
- [ ] commit 1 — `[spec-host]` per-round host timing (`LONGSPEAR_SPEC_HOST_TIMING=1`)
- [ ] commit 2 — hybrid `cells[seq].pos` guard (code-reading note below first)
- [ ] commit 3 — PLE-tail per-step save/restore (`LONGSPEAR_PER_STEP_PLE_TAIL=1`)
- [ ] commit 4 — explicit capacity (`LONGSPEAR_SPEC_CKPT_MAX_TOKENS`) + clamp (`LONGSPEAR_SPEC_CLAMP_TO_CKPT`)
- [ ] commit 5 — persistent checkpoint sampler (`LONGSPEAR_SPEC_CKPT_LEAN=1`)
- [ ] commit 6 — per-step vs replay crosscheck (`LONGSPEAR_SPEC_CKPT_CROSSCHECK=1`)
- [ ] commit 7 — tests (a)-(d), build script, launchers
- [ ] build `build-sl1` (only when no other compile is running), tests run with exit codes
- [ ] `.lane/GPU-WINDOW.md` (coordinator recipe), report

## Commit 2 code-reading note — `kv.cells[seq_id].pos = accepted_pos` (src/llama.cpp PER_STEP restore)
What the write means depends on the cache kind:
- **Pure recurrent (Mamba, `kv.recurrent`, `llm_arch_is_recurrent`)**: `cells[]` is indexed by sequence; cell
  `seq_id` holds that sequence's last position. The write moves it to `accepted_pos`, which is what the following
  `llama_kv_cache_seq_rm(accepted_pos+1, -1)` needs: its recurrent branch refuses a partial removal when
  `0 < p0 <= cells[seq_id].pos`. Correct and needed there.
- **Hybrid (qwen4exp / qwen3next: `recurrent=false`, `hybrid=true`)**: `cells[]` holds one attention cell per KV
  position; the recurrent rows live in `s_l`, not in `cells`. The write therefore rewrites the position of
  *attention cell 0* (seq_id 0) from 0 to `accepted_pos`. Effects, by reader:
  - KQ mask (`llama_set_inputs`, `cells[i].pos > pos → -inf`): the next queries sit at `> accepted_pos`, so cell 0
    stays visible. No change on the next pass.
  - `seq_pos_max`: unchanged (`accepted_pos` ≤ the true max).
  - The `seq_rm(accepted_pos+1, -1)` right after: does not touch cell 0.
  - QSA inputs (`inp_qsa` fill): `blk_pos[0]` = min position over block 0, becomes 1 instead of 0, so block 0's
    pooled key is re-roped at position 1 whenever block 0 is re-pooled (a stale rebuild). The per-query bias gives
    `+1e9` to cells with `tail_start <= cell_pos <= q`: cell 0 now passes that test whenever `accepted_pos` falls in
    the query's current block tail, so cell 0 is force-selected into top-k and displaces one real cell. Numeric
    divergence, not a crash.
  - Later truncation (`seq_rm(p0, -1)` for prompt-cache reuse, `p0 <= accepted_pos`): cell 0 matches `pos >= p0`
    and is **freed**. The prefix loses token 0's K/V, `head` drops to 0, and the next token is written into cell 0
    out of position order (QSA keys blocks by cell index). Wrong outputs for the rest of that slot's life.
  - State save/load (`llama_state_seq_*`): serialises the corrupted position.
  - `slot.n_past` after DIRECT: the server recomputes it from `cache_tokens` (`n_past = cache_tokens.n_tokens()`
    after pushing the accepted ids), independent of the restore mode, so it is correct in both modes;
    `spec_pos_base` = `ckpt.n_past + 1`, and `accepted_pos = ckpt.n_past + step` keeps KV positions up to the last
    accepted draft.
  Production never reaches this line today (gpu-fallback). Fix: write only when `kv.recurrent`, behind
  `LONGSPEAR_PER_STEP_PLE_TAIL=1` (the per-step-on-qwen4exp flag), so flag-off keeps today's code path. Upstream
  candidate (ik code, 2026-04-24).
- **QSA pooled-indexer rows written by rejected drafts (both modes the same)**: the verify pass writes raw indexer
  keys (`kr_l`) at the cells of every verify token and scatters pooled block keys (`kp_l`) for the blocks it
  touches, rejected drafts included. After rejection both modes free the rejected cells (`seq_rm`); their `kr`
  rows are stale but empty cells are never pooled (`blk_of = -1`) and are overwritten on reuse. A block's pooled
  key is ranked only when the block is complete (`filled == r`, not mixed); an incomplete block gets `-inf`, or the
  `+1e9` tail boost that ignores the score. A block becomes complete only in a pass that writes its last member,
  and that pass touches the block (`touched` spans `head .. head+n_tokens-1`) and re-pools it from its current
  members. gpu-fallback's replay re-pools earlier, but only blocks that are still incomplete (the one holding the
  accepted tail), whose pooled key is not ranked anyway. So a pooled key computed with a rejected draft's `kr` row
  can never be ranked in either mode. Verified by reading `llama_set_inputs` (QSA block, `src/llama.cpp` ~5351-5493)
  and `qwen4exp_qsa_mask` (`src/graphs/build_qwen4exp.cpp`).

## Log
- 18:25 read plan + draft + code (iqk partition sites, delta-net per-step, kv per_step_alloc/restore, spec ckpt
  init/save/restore, server clamp site, sampler clone). `D:/AI/ik_llama-qwen4exp/BOX-LOCK.json` exists but is stale
  (since 09-17, eta 09-18). A GPU campaign (singles attempt 3, `bench/gpu-justify/20260923-singles-tc-lar.md`) is
  pending/active and was voided once by another session's CPU load: check it before compiling.
