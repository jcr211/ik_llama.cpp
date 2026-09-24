# SL-1 progress — tail-aware PER_STEP speculative checkpoints (qwen4exp)

Worktree `D:/AI/worktrees/sl1-spec-ckpt`, branch `lane/sl1-spec-ckpt` from `fix/dsa-inv-sum-overrun` @ d583c220.
Order: `D:/Projects/longspear/docs/drafts/decode-throughput-merged-plan-20260924.md` §5 (SL-1).

## Checklist
(hashes after the autosquash rebase; every code commit compiles on its own, see the log)
- [x] commit 0 — `fix(iqk)`: empty row-group guard at the 7 MoE/dense partition sites (03450ffc)
- [x] commit 1 — `[spec-host]` per-round host timing (`LONGSPEAR_SPEC_HOST_TIMING=1`) (9b72c3fa)
- [x] commit 2 — hybrid `cells[seq].pos` guard (code-reading note below first) (1144be68)
- [x] commit 3 — PLE-tail per-step save/restore (`LONGSPEAR_PER_STEP_PLE_TAIL=1`) (100d62cd)
- [x] commit 4 — explicit capacity (`LONGSPEAR_SPEC_CKPT_MAX_TOKENS`) + clamp (`LONGSPEAR_SPEC_CLAMP_TO_CKPT`) (277cae8f)
- [x] commit 5 — persistent checkpoint sampler (`LONGSPEAR_SPEC_CKPT_LEAN=1`) (c6cbebae)
- [x] commit 6 — per-step vs replay crosscheck (`LONGSPEAR_SPEC_CKPT_CROSSCHECK=1`) (03b685f2)
- [x] commit 7 — tests (a)-(d), build script, launchers (4e5a606f, 8cf423dc)
- [x] build `build-sl1`, every commit compiled in order, tests (a)-(d) exit 0
- [x] `.lane/GPU-WINDOW.md` (coordinator recipe); report delivered in the final message (the REPORT.md write was
  refused by a hook)

## Fix round 1 (cross-family review of c4c50b2b: sessions/.council-tmp/sf-merge/opus-sl1-review.md et al.)
- [x] B1 host PLE n-gram history rewind — merged `lane/ple-hist-rewind` (e3cf8bf5); the resume keys on `path_result`
  so the crosscheck replay resumes at the checkpoint and the direct restore after the accepted drafts (8d4489a9);
  every launcher arm sets `LONGSPEAR_PLE_HIST_REWIND=1` + `LONGSPEAR_PLE_HIST_LOG=1`; gate `[ple-hist] reset` = 0 with
  the log live (ab8ce58e). Rebuilt (no other compile running), five tests exit 0 with CUDA_VISIBLE_DEVICES=-1.
- [x] B2 clamp reaches ngram-mod/suffix bookkeeping (`common_speculative_truncate_draft`) + test-spec-ckpt-clamp (4d127fd6)
- [x] S1 `k_prop` in [spec-host]; gate compares pre-clamp drafts and fit-round acceptance (4d127fd6, c5491063)
- [x] S2 crosscheck commits the MTP companion the per-step way; row A2 gates mtp_skip (4d127fd6, c5491063)
- [x] S3 auto-stop chain `.lane/w-sl1-chain.sh` + fixture selftest 9/9 (1f0b1d8f)
- [x] AMEND 1 / AMEND 2 adopted into GPU-WINDOW.md §2, §3, §8 (INCONCLUSIVE rule at the 600-round cap)
- [x] N1 PCIe delta from the row's first numeric sample, NA STOPs; N2 failed restore STOPs, per-arm mode check (c5491063)
- [x] DeepSeek #6 proposal_dists only shortened (4d127fd6)
- [x] rebuild in a checked slot, tests (a)-(d) + clamp test exit 0

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
- 18:48 commits 0-7 written. Box check: no nvcc/cl/cmake/ninja/MSBuild/cargo/rustc, no llama-server, :8099
  not listening, singles attempt 3 not launched (the other project's Rust CI still pending). Building with
  `.lane/run-build.sh`, which kills this lane's build the moment any llama-server process appears.

- 18:55 build-sl1 OK after three fixes (forward declaration of common_speculative_checkpoint_save missing the new
  parameters; ggml_quantize_chunk takes a user_data argument; the sampler test must not call
  common_reasoning_budget_get_state because common.lib defines it twice). The first configure failed because the
  build script used `RC` for the exit code, which CMake reads as the resource compiler: renamed. The -1 exit of a
  link failure was caught by the string check (`build rc=-1`).
- 18:57 CPU tests: test-ple-perstep exit 0 (a, b), test-iqk-moe-chunks exit 0 (c, 16 cases), test-spec-ckpt-sampler
  exit 0 (d, 3 rounds). Mutation check: with the PLE window at j instead of j+1 the test fails 127 checks (a and b).
- 19:02 fixups autosquashed into commits 1, 5, 7 so each commit compiles on its own; tree identical to before.
- GPU-WINDOW.md written; sl1-gate.sh self-test 7/7 on synthetic telemetry.
- 19:03 per-commit compile loop (`.lane/per-commit-build.sh`): llama-server rc=0 at every code commit, the three
  tests rc=0 at commit 7. Warning scan (`.lane/own-warnings.sh`, blame after d583c220): one nodiscard in the sampler
  test, fixed; none in the library/server code. Launchers parse clean; refusal paths exit 2 without launching.
- Trial merge with lane/stateos-lane1 (ec0b962a) via `git merge-tree`: code auto-merges
  (speculative.cpp/.h, server-context.cpp, llama.h, llama.cpp, tests/CMakeLists.txt); only .lane/PROGRESS.md
  conflicts.

## VRAM arithmetic at M=5 (for the preflight; real sizes print at startup)
- Per GDN row: SSM state 3,145,728 B, conv state 122,880 B (row 3,268,608 B; 36 GDN rows). PLE tail
  hist x hc_dim = 9 x 10,240 f32 = 368,640 B in layer 1's row. Full shadow 36 x 3,268,608 + 368,640 = 112.57 MiB.
- per_step_alloc(M=5), one buffer on CUDA0: per_step_ssm (M-1) x 3,145,728 x 36 = 453.0 MB; per_step_qkv (conv)
  M x 122,880 x 36 = 22.1 MB; per_step_ple (M-1) x 368,640 x 1 = 1.47 MB. Sum 476.6 MB (454.5 MiB).
- The conv-only shadow stays allocated in per-step mode (4.4 MB, unused by restore); the full 112.57 MiB shadow is
  NOT allocated, so vs production gpu-fallback: +476.6 + 4.4 - 118.0 = **+363 MB net**.
- Crosscheck (probe only) allocates the full shadow instead of the conv-only one: +476.6 + 118.0 = +594.6 MB vs
  flag-off gpu-fallback's 118.0, i.e. **+476.6 MB net** over production. The preflight should run the probe config
  (the larger) or both.
- Startup lines to read: `per_step_alloc: CUDA0 per-step buffer = ... MiB (max_tokens=5)`,
  `per_step_alloc: CUDA0 per-step PLE history = 1.406 MiB of it (4 slots)`, `checkpoint_alloc_shadows: CUDA0 shadow
  buffer = ... MiB (conv-state only)`.
