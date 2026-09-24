# W-SL1 — GPU window recipe for the coordinator (the lane does not run it)

Lever: SL-1 tail-aware PER_STEP speculative checkpoints (`lane/sl1-spec-ckpt`, worktree
`D:/AI/worktrees/sl1-spec-ckpt`). Plan: `docs/drafts/decode-throughput-merged-plan-20260924.md` §4. This file turns it
into commands. AMEND 1 (probe length) and AMEND 2 (preflight config) are **ADOPTED** by the coordinator (2026-09-23);
the fix round after the cross-family review (B2, S1, S2, S3, N1, N2) is folded in. **B1 is open**: the window must
not start until `lane/ple-hist-rewind` is merged and its setter is called at the per-step direct restore and the
crosscheck's gpu-fallback replay (§9).

## 0. Preconditions (no GPU yet)
1. Cross-family review of the SL-1 diff (DeepSeek inline brief + fresh Opus repo-rooted seat, `cross-family-review`
   skill) passed on the HEAD that includes the ple-hist merge; HEAD frozen. W-SL1 is GPU-gating.
2. One binary: `build-perstep.cmd` at that HEAD → `D:/AI/worktrees/sl1-spec-ckpt/build-sl1/bin/llama-server.exe`
   (build-avx512-prod flags, CUDA 13.0, sm_120). CPU tests green (`bash .lane/run-tests.sh`). Record
   `git rev-parse HEAD` and the exe hash.
3. `bench/gpu-justify/<YYYYMMDD>-spec-perstep.md` committed (§8 is the text) and posted to James at launch.
4. Box: take the slot (`gpu-slot` skill, BOX-LOCK), `nvidia-smi` quiet, no other GPU consumer, no compile and no
   other project's CI running. `D:/AI/ik_llama-qwen4exp/BOX-LOCK.json` holds a stale 09-17 lock; replace it.
5. `native-replay.sh` and `bench-decode.sh` take a launcher *basename* in `D:/AI/ik_llama-qwen4exp` and pass no
   arguments. Copy the four argument-free launchers there (they carry no key; each reads it from
   `D:/AI/llama-swap/config.yaml` at launch and logs to `ik-serve-8099.{out,err}.log`):
   - `launch-perstep-8099.ps1` → **A2** (per-step + `LONGSPEAR_PER_STEP_PLE_TAIL=1` + `LONGSPEAR_SPEC_CKPT_MAX_TOKENS=5`
     + `LONGSPEAR_SPEC_CLAMP_TO_CKPT=1` + `LONGSPEAR_SPEC_CKPT_LEAN=1` + `LONGSPEAR_SPEC_HOST_TIMING=1`)
   - `launch-perstep-p0-8099.ps1` → **P0** (gpu-fallback, only `LONGSPEAR_SPEC_HOST_TIMING=1`)
   - `launch-perstep-a0-8099.ps1` → **A0** (no `--spec-type`)
   - `launch-perstep-xcheck-8099.ps1` → A2 + `LONGSPEAR_SPEC_CKPT_CROSSCHECK=1` (preflight and probe only)
   All four run the same exe with the standing args (`-ncmoe 37 -c 196608 -ub 512 -ctk/-ctv q8_0 -np 1 -t 24 -tb 32
   --temp 1.0 --top-p 0.95 --top-k 20 -rtr -muge`, `LONGSPEAR_VERIFY_TIMING=1`, `LONGSPEAR_CG_REVIVE=1`). The wrappers
   call `D:\AI\worktrees\sl1-spec-ckpt\launch-perstep-8099.ps1` by absolute path; they refuse a busy port.
   B1's rewind flag (`LONGSPEAR_PLE_HIST_REWIND=1`, from the ple-hist lane) must be added to the A2 and crosscheck
   arms when that branch merges; P0 keeps it off (production behaviour) unless the coordinator decides otherwise.
6. The whole window is one command: `bash D:/AI/worktrees/sl1-spec-ckpt/.lane/w-sl1-chain.sh <stamp>` (§7). Its gates
   are `.lane/sl1-gate.sh` (self-test `.lane/gate-selftest.sh`, 24/24) and its decision logic is tested on fixtures
   by `.lane/w-sl1-chain-selftest.sh` (9/9). Gate exits: 0 PASS, 1 STOP, 2 no data, 3 blocks promotion (stepcost),
   4 INCONCLUSIVE (probe).

## 1. Telemetry the gates read (all written by the binary itself)
- `[spec-host] slot mode K k_prop accepted restore_result redecode_n ckpt_init_us ckpt_save_us cells_copy_us
  shadow_copy_us sync_us sampler_init_us sampler_clone_us restore_us redecode_us draft_host_us sample_us mtp_skip
  clamp xcheck` — one line per verify round.
  - `K` is the verified batch; `k_prop = K + clamp` the batch the drafter proposed before the capacity clamp.
  - `restore_result` is the code's own return value: `none` = every draft accepted, `direct` = per-step restore,
    `replay` = base replay required, `failed`.
  - `redecode_n` counts tokens re-decoded because the restore required it. A crosscheck round's diagnostic replay is
    **not** in it; the line says `xcheck=1`.
  - `mtp_skip` = MTP draft calls since the previous verify round that returned nothing for lack of a target hidden
    state (the M7 failure). A skipped round builds no verify batch, so it shows up on the next line.
- `[ckpt-xcheck] j comp={gdn_s,gdn_conv,ple_tail} n_bitequal n relL2 max_layer_relL2` — per-step state (probe) vs the
  gpu-fallback restore + replay state (oracle, kept), summed over layers, per rejected round.
- `[vt] K n_kv mtp_op us build compute ...` (standing `LONGSPEAR_VERIFY_TIMING=1`) — `mtp_op=0` = main-model passes.
- Server `eval time = X ms / N tokens` lines — aggregate decode tok/s = ΣN / ΣX.
- `pcie-telemetry.sh` column 2 = "Replays Since Reset". The gate's delta runs from the first numeric sample of the
  row's own log to the last; a log with no numeric sample (header only, or all `NA`) is a STOP (N1).
- Startup: `per_step_alloc: CUDA0 per-step buffer = … MiB (max_tokens=5)`, `per_step_alloc: CUDA0 per-step PLE history
  = 1.406 MiB of it (4 slots)`, `speculative checkpoint capacity = 5 tokens (LONGSPEAR_SPEC_CKPT_MAX_TOKENS; stage
  chain needs 17)`, `fixed recurrent checkpoint mode = per-step (max_tokens=5)`. A2 prints all four; P0 prints
  `fixed recurrent checkpoint mode = gpu-fallback`.

## 2. Step 0 — VRAM preflight (AMEND 2, adopted: crosscheck config)
Expected: per-step buffers 476.6 MB (SSM 453.0 + conv 22.1 + PLE 1.47) + conv-only shadow 4.4 MB − the 112.57 MiB full
shadow production allocates = **+363 MB net** for A2. The crosscheck config keeps the full shadow instead of the
conv-only one (its other extras are host memory): **+477 MB net**, a strict superset of A2's static allocations. The
preflight therefore runs on `launch-perstep-xcheck-8099.ps1`; if it fits, A2 fits.
1. `mem-trace.sh` and `pcie-telemetry.sh` at 1 Hz.
2. One request from `.lane/sl1-long-request.mjs`: the fork's sources, measured with `/tokenize` to 131,072..180,000
   tokens, 256 decode tokens at temp 1.0 (streamed). The prefill takes the KV/FA temporaries to depth; the decode then
   runs verify rounds with per-step saves at that depth.
3. **Kill:** high-water > 32,351 MiB, no sample, a CUDA error or a PCIe replay increment → stop; no arm runs.
   The chain logs the startup buffer lines and the high-water.

## 3. Step 1 — mechanism probe (AMEND 1, adopted)
1. A2 + crosscheck native replay of the two tasks. A watcher (in the chain) polls `sl1-gate.sh jcount` on the live
   server log every 10 s and stops the probe server once **every j ∈ 0..3 has ≥ 20 crosscheck rows, or at 600 verify
   rounds**, whichever comes first (150 rounds give ≈ 14 rows at j=3 at 64 % acceptance, below the plan's own
   criterion).
2. `sl1-gate.sh probe replay-sl1-<stamp>-probe.err.log 5` — each a STOP on miss:
   - `mode=per-step` on every `[spec-host]` round; no `restore_result=failed` (N2);
   - restore-required replays (`redecode_n > 0`) on ≤ 1 % of rejected rounds (expected 0);
   - PLE tail bit-equal on ≥ 99 % of crosscheck rows, else median relL2 ≤ 10 × the GDN-S median and max ≤ 0.05.
     `gdn_s`/`gdn_conv` are reported, not gated (batch-shape noise, M5). **Without B1's fix this check STOPs**: the
     replay resets the host PLE n-gram history, so the oracle's tail is wrong in 1-2 of 9 columns (relL2 ≈ 0.3);
   - `mtp_skip` ≤ 10 % of rejected rounds (M7). Crosscheck rounds keep the replay-derived target state but commit the
     MTP companion from the verify pass's hidden rows, exactly as a direct restore does (S2), so this measures the
     per-step path;
   - no CUDA error, no PCIe replay increment.
   **INCONCLUSIVE rule:** if any j is still under 20 rows when the probe ends (600-round cap or task budget) and no
   other check missed, the gate exits 4: the probe is INCONCLUSIVE, **not a pass, and no arm runs**. A mechanism miss
   outranks a short histogram (exit 1).

## 4. Step 2 — arms (interleaved, fixed order, fresh process per row)
Tasks `caller-omitted-refactor-01,duplicate-finalization-01`, budget 20m/task, production sampler (temp 1.0).
Order (fixed now): **P0-1, A2-1, A0-1, A2-2, P0-2, A0-2**; each row under its own `pcie-telemetry.sh` log.
- After **every** row: `sl1-gate.sh row <P0|A2|A0> replay-….err.log pcie-telemetry-….log` — STOP on:
  - the row decoded no tokens; native-replay verdict not CLEAN (CRASH/OOM → KILLED, fewer than 2 tasks → VOID);
  - **A2**: any round not `mode=per-step` (an A2 row that ran without per-step fails, it is not skipped — N2); a
    failed restore; restore-required replay calls > 1 % of main passes; `mtp_skip` > 10 % of rejected rounds (S2);
  - **P0**: any round not `mode=gpu-fallback`; a failed restore;
  - **A0**: any verify round (speculation must be off);
  - any CUDA error; a PCIe replay increment over the row, or no numeric PCIe sample (N1).
- Pair gate after A2-1 (P0-1 vs A2-1) and after P0-2 (P0-2 vs A2-2), STOP on either miss (S1, the clamp confound):
  - **drafts per verify** compared on the **pre-clamp proposal**, Σ(`k_prop` − 1)/rounds in both arms; A2 must be
    ≥ 90 % of P0. (A post-clamp count would put A2 at ≈ 0.73-0.80 of P0 from the clamp alone and false-kill at r2.)
  - **acceptance** compared only on rounds whose proposal fits the capacity, `k_prop` ≤ 5, in both arms; A2 ≥ P0 − 3
    pts. Rounds with `k_prop` > 5 are reported separately: A2's `clamp_acceptance` (accepted of the 4 verified) beside
    P0's `first_m1_acceptance` (min(accepted, 4)/4 on the same class of rounds), plus both arms' `mtp_skip` per
    rejected round.
  - prints `GAIN_PCT` = A2 vs P0 aggregate decode tok/s.
- Decision after P0-2 (both pairs in hand):
  - both gains < +8 % → **KILLED**, stop at once (A0-2 is not run);
  - gains of opposite sign → after A0-2, the preregistered third rep **P0-3, A2-3** with the same gates; then
    **candidate** if at least 2 of the 3 pairs reach +8 %, otherwise **KILLED**;
  - both ≥ +8 % → **candidate**;
  - anything else (same sign, one above and one below +8 %) → **HOLD**: neither the kill nor the promotion rule is
    met; step 3 cannot change a decision and is skipped; the coordinator decides.
Metrics per row (the gate prints them): decode tok/s, rounds, rejected, direct/replay/failed, replay calls vs main
passes, drafts per verify (post- and pre-clamp), acceptance (all and fit rounds), clamp rounds, `mtp_skip`, per-verify
host µs by bucket (P0 carries the same telemetry, so L2's host attribution comes from P0's `ckpt_save_us`/
`cells_copy_us`/`shadow_copy_us`/`sync_us`/`sampler_init_us`), CUDA errors, PCIe replays.
Caveat: `native-replay.sh` sends 3 pre-warm requests before the tasks; they land in every arm's log alike.

## 5. Step 3 — fixed-context legs (candidates only)
`bench-decode.sh` P0/A2 × 2 (3K prompt, 5 measured requests each), then `sl1-gate.sh stepcost` over the four
`bench-….err.log` files: mean main-pass compute per K for K = 2..5, A2 vs P0. A regression > 3 % exits 3: reported,
**blocks promotion** (verdict CANDIDATE-BLOCKED), not a kill. Spare legs L4 (`-tb 24`, launcher-only) and L6 (only if
built) are independent, flag-off vs flag-on on the same binary, and not in the chain.

## 6. Step 4 — fidelity (manual, same window, offline-scored)
v2 statistical gate (paired greedy first-divergence + per-step KL, ≥ 12 prompts × 64 tokens), A2 vs P0 on this binary
(stock control = P0). The crosscheck in step 1 is the mechanism gate; this is the output gate. Greedy changes the
workload, so it is not a throughput arm (M13). Note (B1): P0 itself replays with the host PLE history reset, so it is
an imperfect oracle for this step until the ple-hist fix is on in P0 as well — the coordinator's call.

## 7. Auto-stop — `.lane/w-sl1-chain.sh <stamp>`
Runs steps 0-3 in order with the gates above and stops at the first miss (exit 1, verdict KILLED / VOID /
INCONCLUSIVE); HOLD, CANDIDATE and CANDIDATE-BLOCKED exit 0. Preconditions it checks: the justify file exists, the
four launchers and `native-replay.sh`/`bench-decode.sh` are in the fork dir, and no llama-server survives its stop.
On any exit it stops telemetry and every llama-server and relaunches `launch-standing-8099.ps1` (`RESTORE=0` skips
that). All logs go to `D:/AI/ik_llama-qwen4exp` with the `sl1-<stamp>` prefix; the decisions to
`w-sl1-<stamp>.chain.log`. A lever that fails its mechanism criterion is stopped at once; the prereg N is never run
out after a miss.

## 8. Preregistration text for `bench/gpu-justify/<YYYYMMDD>-spec-perstep.md`
- **Decision it changes:** whether qwen4exp serving moves from gpu-fallback to tail-aware PER_STEP (A2); whether
  speculation pays at all (A0 matched on the same tasks); whether `-tb 24` / libomp get their own lanes.
- **Why offline cannot answer:** the lever's value is GPU wall time under agentic traffic. The CPU tests prove state
  equivalence (PLE per-step slots bit-equal to sequential single-token histories for K ∈ {2..5, 17}; slot j = after
  token j for delta-net, conv and PLE buffers; commit-0 guard bit-identical at 20/24/32 threads; lean sampler
  equivalence; clamped drafts do not trip ngram-mod's low-acceptance reset) — not throughput, and not the real
  model's per-step path on CUDA. Reviewers' "not yet" (B1) is honoured: no launch before the ple-hist merge.
- **Mechanism kill criteria (the lever's own lines):**
  - preflight: VRAM high-water ≤ 32,351 MiB on the crosscheck config (AMEND 2);
  - probe: per-step on every round, no failed restore, restore-required replays ≤ 1 % of rejected rounds, PLE tail
    bit-equal ≥ 99 % (or relL2 within 10× GDN-S and ≤ 0.05), `mtp_skip` ≤ 10 % of rejected rounds, every j ∈ 0..3
    ≥ 20 rows within 600 rounds — else INCONCLUSIVE, no arms (AMEND 1);
  - every row: the per-arm mode check, no failed restore, A2 replay calls ≤ 1 % of main passes, A2 `mtp_skip` ≤ 10 %
    of rejected rounds, no CUDA error, no PCIe replay increment;
  - pairs: acceptance on `k_prop` ≤ 5 rounds ≥ P0 − 3 pts, pre-clamp drafts per verify ≥ 90 % of P0.
- **Outcome rules:** both A2 pairs < +8 % → killed; opposite signs → one third rep, candidate only with ≥ 2 of 3
  pairs at +8 %; both ≥ +8 % → candidate (then step 3: a K=2..5 step-cost regression > 3 % blocks promotion);
  otherwise HOLD.
- **Auto-stop:** `.lane/w-sl1-chain.sh` (§7).
- **Binary:** `lane/sl1-spec-ckpt` @ `<sha>`, `build-sl1`, flags select the arms; production args otherwise identical;
  API key from config at launch.
- **N:** 2 per arm (3 only on direction disagreement), all rows reported, failures ledgered as prominently as wins.
- **Promotion to default:** S5/hard-band battery after W-SL1 per the battery-first rule (or James's waiver, ledgered
  with revert path = launcher `--spec-ckpt-mode gpu-fallback` and the SL-1 env flags unset).

## 9. Known limits and what to watch
- **B1 (open, owned by `lane/ple-hist-rewind`):** the host-side PLE n-gram history (`lctx.ple_hist`) is not rewound by
  any restore. After a direct restore the next batch starts below `next_pos`, the history resets to EOS, and the
  bonus token and the one after it get wrong PLE rows; the crosscheck's replay has the same reset, so its oracle is
  contaminated. This lane calls `llama_ple_history_set` at its two sites once that branch is merged.
- Per-step on qwen4exp without `LONGSPEAR_PER_STEP_PLE_TAIL=1` prints a one-time WARNING and keeps the old
  (contaminating) behaviour; flag-unset AUTO still resolves to per-step as before (M8b). The launchers always pass
  the mode explicitly.
- Capacity 5 + clamp: ngram-mod 16-token drafts are cut to 4 (K = 5); without the clamp they would run root-only. The
  drafting stage's acceptance bookkeeping sees the verified length (B2), so a fully accepted clamped round is not
  low acceptance. The base's pre-existing `n_draft_max` truncation has the same bookkeeping flaw (flag-off, untouched).
- The pre-existing delta-net per-step restore copies `s_l->ne[1]` rows with a stride that ignores the PLE tail, so it
  is only correct with one state slot (production is `-np 1`). The PLE tail restore writes only the restored
  sequence's row.
- The PLE per-step path does not support a split (`-sm graph`) state row; `per_step_alloc` refuses it with an error.
- Crosscheck rounds are slower (a full-row D2H read + a replay per rejected round); never measure throughput with it.
