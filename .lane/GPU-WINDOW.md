# W-SL1 — GPU window recipe for the coordinator (the lane does not run it)

Lever: SL-1 tail-aware PER_STEP speculative checkpoints (`lane/sl1-spec-ckpt`, worktree
`D:/AI/worktrees/sl1-spec-ckpt`). Plan: `docs/drafts/decode-throughput-merged-plan-20260924.md` §4 (this file turns it
into commands; where it deviates, the deviation is marked **AMEND** and is the coordinator's call).

## 0. Preconditions (no GPU yet)
1. Cross-family review of the SL-1 diff (DeepSeek inline brief + fresh Opus repo-rooted seat, `cross-family-review`
   skill) done, fix rounds merged, HEAD frozen. W-SL1 is GPU-gating.
2. One binary: `build-perstep.cmd` at that HEAD → `D:/AI/worktrees/sl1-spec-ckpt/build-sl1/bin/llama-server.exe`
   (build-avx512-prod flags, CUDA 13.0, sm_120). CPU tests green (report). Record `git rev-parse HEAD` and the exe hash.
3. `bench/gpu-justify/<YYYYMMDD>-spec-perstep.md` committed (§8 below is the text) and posted to James at launch.
4. Box: take the slot (`gpu-slot` skill, BOX-LOCK), stop the standing :8099, `nvidia-smi` quiet, no other GPU
   consumer, no compile and no other project's CI running, PCIe "Replays Since Reset" read before.
   Note: `D:/AI/ik_llama-qwen4exp/BOX-LOCK.json` currently holds a stale 09-17 lock; replace it with this window's.
5. `native-replay.sh` and `bench-decode.sh` take a launcher *basename* in `D:/AI/ik_llama-qwen4exp` and pass no
   arguments. Copy the four argument-free launchers there (they carry no key; each reads it from
   `D:/AI/llama-swap/config.yaml` at launch and logs to `ik-serve-8099.{out,err}.log`):
   - `launch-perstep-8099.ps1` → **A2** (per-step + `LONGSPEAR_PER_STEP_PLE_TAIL=1` + `LONGSPEAR_SPEC_CKPT_MAX_TOKENS=5`
     + `LONGSPEAR_SPEC_CLAMP_TO_CKPT=1` + `LONGSPEAR_SPEC_CKPT_LEAN=1` + `LONGSPEAR_SPEC_HOST_TIMING=1`)
   - `launch-perstep-p0-8099.ps1` → **P0** (gpu-fallback, only `LONGSPEAR_SPEC_HOST_TIMING=1`)
   - `launch-perstep-a0-8099.ps1` → **A0** (no `--spec-type`)
   - `launch-perstep-xcheck-8099.ps1` → A2 + `LONGSPEAR_SPEC_CKPT_CROSSCHECK=1` (probe only)
   All four run the same exe with the standing args (`-ncmoe 37 -c 196608 -ub 512 -ctk/-ctv q8_0 -np 1 -t 24 -tb 32
   --temp 1.0 --top-p 0.95 --top-k 20 -rtr -muge`, `LONGSPEAR_VERIFY_TIMING=1`, `LONGSPEAR_CG_REVIVE=1`). The wrappers
   call `D:\AI\worktrees\sl1-spec-ckpt\launch-perstep-8099.ps1` by absolute path; they refuse a busy port.
6. Gate script: `bash D:/AI/worktrees/sl1-spec-ckpt/.lane/sl1-gate.sh` (self-test: `.lane/gate-selftest.sh`, 7/7 on
   synthetic telemetry). Exit 0 = PASS, 1 = STOP.

## 1. Telemetry the gates read (all written by the binary itself)
- `[spec-host] slot mode K accepted restore_result redecode_n ckpt_init_us ckpt_save_us cells_copy_us shadow_copy_us
  sync_us sampler_init_us sampler_clone_us restore_us redecode_us draft_host_us sample_us mtp_skip clamp xcheck` —
  one line per verify round. `restore_result` is the code's own return value (`none` = every draft accepted,
  `direct` = per-step restore, `replay` = base replay required). `redecode_n` counts tokens re-decoded because the
  restore required it. A crosscheck round's diagnostic replay is **not** in `redecode_n`; the line says `xcheck=1`.
  `mtp_skip` = MTP draft calls since the previous verify round that returned nothing for lack of a target hidden
  state (the "skips one round after invalidate" failure). `clamp` = draft tokens cut by the capacity clamp.
- `[ckpt-xcheck] j comp={gdn_s,gdn_conv,ple_tail} n_bitequal n relL2 max_layer_relL2` — per-step state (probe) vs the
  gpu-fallback restore + replay state (oracle, kept), summed over layers, per rejected round.
- `[vt] K n_kv mtp_op us ...` (standing `LONGSPEAR_VERIFY_TIMING=1`) — `mtp_op=0` lines are main-model passes.
- Server `eval time = X ms / N tokens` lines — aggregate decode tok/s = ΣN / ΣX.
- Startup: `per_step_alloc: CUDA0 per-step buffer = … MiB (max_tokens=5)`, `per_step_alloc: CUDA0 per-step PLE history
  = 1.406 MiB of it (4 slots)`, `speculative checkpoint capacity = 5 tokens (LONGSPEAR_SPEC_CKPT_MAX_TOKENS; stage
  chain needs 17)`, `fixed recurrent checkpoint mode = per-step (max_tokens=5)`. A2 must print all four; P0 prints
  `fixed recurrent checkpoint mode = gpu-fallback`.

## 2. Step 0 — VRAM preflight (≥128K-token prefix)
Expected (`.lane/PROGRESS.md`): per-step buffers 476.6 MB (SSM 453.0 + conv 22.1 + PLE 1.47) + conv-only shadow 4.4 MB
− the 112.57 MiB full shadow production allocates = **+363 MB net** for A2; the crosscheck config keeps the full shadow
instead of the conv-only one: **+477 MB net**. **AMEND:** run the preflight on the crosscheck launcher (the larger of
the two configs the window runs); if it fits, A2 fits.
1. `bash mem-trace.sh sl1-preflight 1800 &` and `bash pcie-telemetry.sh sl1-preflight 1800 &` (1 Hz).
2. Launch `launch-perstep-xcheck-8099.ps1`, wait for `/health`.
3. One request whose prompt tokenizes to ≥ 131,072 tokens (check with `/tokenize`; e.g. concatenated repo sources),
   `max_tokens` 256, temp 1.0 — the prefill takes the KV/FA temporaries to depth and the decode then runs verify
   rounds with per-step saves at that depth.
4. **Kill:** high-water > 32,351 MiB → stop; no arm runs. Also stop on any CUDA error or PCIe replay increment.
   Record the startup buffer lines and the high-water in the ledger.

## 3. Step 1 — mechanism probe (crosscheck on)
1. `bash native-replay.sh launch-perstep-xcheck-8099.ps1 sl1-probe caller-omitted-refactor-01,duplicate-finalization-01 20m`
   under `pcie-telemetry.sh`. Stop the probe once `grep -c "\[ckpt-xcheck\].*comp=gdn_s" ik-serve-8099.err.log` reaches
   the target below (or at the budget).
   **AMEND (round count):** the plan says "truncated at 150 verify rounds", but its own j-histogram criterion needs
   every j ∈ 0..3 ≥ 20×. At ~64 % per-draft acceptance, P(j=3) ≈ 0.64³·0.36 ≈ 9 % of 4-draft rounds, so 150 rounds give
   ≈ 14 j=3 rows and the probe would STOP on sample size, not on mechanism. Recommend: run until each j has ≥ 20 rows
   or 600 verify rounds, whichever first.
2. `bash .lane/sl1-gate.sh probe replay-sl1-probe.err.log 5` — checks, each a STOP on miss:
   - `mode=per-step` on every `[spec-host]` round;
   - required replays (`redecode_n > 0`) on ≤ 1 % of rejected rounds (expected 0: every rejected round `direct`);
   - PLE tail bit-equal on ≥ 99 % of crosscheck rows, else median relL2 ≤ 10 × the GDN-S median and max ≤ 0.05
     (the unfixed path shows O(1)); `gdn_s`/`gdn_conv` are reported, not gated (batch-shape noise, M5);
   - j histogram: every j ∈ 0..3 at least 20 rows;
   - `mtp_skip` ≤ 10 % of rejected rounds (M7);
   - no CUDA error.
   Also read by eye: `xcheck=1` on the rejected rounds and `restore_result=direct` there. The conv states and the PLE
   tail are shifted copies of their inputs, so they are bit-equal wherever those inputs (the qkv projection, the PLE
   key/value path) do not depend on the batch shape; where the GPU kernel choice changes with n_tokens (MMVQ/MMQ),
   they sit in the same small envelope as `gdn_s`. An unfixed tail shows relL2 of order 1.
3. Any STOP → the window ends here; no arm runs. Restore the standing server.

## 4. Step 2 — arms (interleaved, fixed order, fresh process per row)
Tasks `caller-omitted-refactor-01,duplicate-finalization-01`, budget 20m/task, production sampler (temp 1.0).
Order (seeded, fixed now): **P0, A2, A0, A2, P0, A0** — rows r1..r6:
```
bash native-replay.sh launch-perstep-p0-8099.ps1 sl1-P0-1 caller-omitted-refactor-01,duplicate-finalization-01 20m
bash native-replay.sh launch-perstep-8099.ps1    sl1-A2-1 caller-omitted-refactor-01,duplicate-finalization-01 20m
bash native-replay.sh launch-perstep-a0-8099.ps1 sl1-A0-1 caller-omitted-refactor-01,duplicate-finalization-01 20m
bash native-replay.sh launch-perstep-8099.ps1    sl1-A2-2 caller-omitted-refactor-01,duplicate-finalization-01 20m
bash native-replay.sh launch-perstep-p0-8099.ps1 sl1-P0-2 caller-omitted-refactor-01,duplicate-finalization-01 20m
bash native-replay.sh launch-perstep-a0-8099.ps1 sl1-A0-2 caller-omitted-refactor-01,duplicate-finalization-01 20m
```
Each row runs under `pcie-telemetry.sh sl1-<row>`. After **every** row (auto-stop, §7):
- `bash .lane/sl1-gate.sh row replay-sl1-<row>.err.log pcie-telemetry-sl1-<row>.log` — STOP on: A2 replay passes
  (restore-required replay calls) > 1 % of main passes (`[vt] mtp_op=0`); any CUDA error; any PCIe replay increment.
- After A2-1 (r2): `bash .lane/sl1-gate.sh pair replay-sl1-P0-1.err.log replay-sl1-A2-1.err.log`; after P0-2 (r5):
  `… pair replay-sl1-P0-2.err.log replay-sl1-A2-2.err.log` — STOP on A2 acceptance < P0 − 3 pts or A2 drafts per
  verify < 90 % of P0.
- After both A2 reps: aggregate decode tok/s gain < +8 % in **both** pairs → STOP, lever killed. Reps disagree in
  direction → one preregistered third rep of P0 and A2 (P0-3, A2-3), then decide on all three.
Metrics recorded per row (the gate prints them): decode tok/s (ΣN/Σms of the "eval time" lines; primary), rounds,
rejected, direct/replay counts, replay calls vs main passes, K-mix (from `[vt]`), drafts per verify, acceptance,
`mtp_skip`, clamp rounds and their acceptance (reported separately, M8a), per-verify host µs by bucket (P0 carries the
same telemetry, so the host-overhead attribution L2 needs comes from P0's `ckpt_save_us`/`cells_copy_us`/
`shadow_copy_us`/`sync_us`/`sampler_init_us`), CUDA errors, PCIe replays, WHEA.
Caveat: `native-replay.sh` sends 3 pre-warm requests before the tasks; they land in the same log for every arm. The
battery JSON's per-task timings are the cross-check if the pre-warm share matters.

## 5. Step 3 — fixed-context legs (`bench-decode.sh`, 3K prompt, N=2 × 5 requests)
```
bash bench-decode.sh launch-perstep-p0-8099.ps1 sl1-P0-b1 5 ; bash bench-decode.sh launch-perstep-8099.ps1 sl1-A2-b1 5
bash bench-decode.sh launch-perstep-p0-8099.ps1 sl1-P0-b2 5 ; bash bench-decode.sh launch-perstep-8099.ps1 sl1-A2-b2 5
```
Per-K main-pass step cost from `[vt]` (the script's per-K table). K = 2..5 regression > 3 % A2 vs P0 → report; it
blocks promotion (per-step save writes: ≤ 0.4 ms at K=5 by bandwidth). Spare legs, independent, never stacked on A2:
L4 `-tb 24` (launcher-only: a copy of `launch-perstep-p0-8099.ps1` passing `-tb 24`; flag-off vs flag-on on the
same binary) and L6 only if its build landed in a compile slot.

## 6. Step 4 — fidelity (same window, offline-scored)
v2 statistical gate (paired greedy first-divergence + per-step KL, ≥ 12 prompts × 64 tokens), A2 vs P0 on this
binary (stock control = P0, the gpu-fallback oracle). The crosscheck in Step 1 is the mechanism gate; this is the
output gate. Note for the KL arm: greedy changes the workload, so it is not a throughput arm (M13).

## 7. Auto-stop
The chain script runs the Step-0 check, the probe gate, then each row followed by its `sl1-gate.sh` calls, and exits
on the first non-zero gate exit (restoring the standing server through `launch-standing-8099.ps1` either way). A
lever that fails its mechanism criterion is stopped at once; the prereg N is never run out after a miss.

## 8. Preregistration text for `bench/gpu-justify/<YYYYMMDD>-spec-perstep.md`
- **Decision it changes:** whether qwen4exp serving moves from gpu-fallback to tail-aware PER_STEP (A2); whether
  speculation pays at all (A0 matched on the same tasks); whether `-tb 24` / libomp get their own lanes.
- **Why offline cannot answer:** the lever's value is GPU wall time under agentic traffic. The CPU tests prove state
  equivalence (PLE per-step slots bit-equal to sequential single-token histories for K ∈ {2..5, 17}; slot j = after
  token j for delta-net, conv and PLE buffers; commit-0 guard bit-identical at 20/24/32 threads; lean sampler
  equivalence) — not throughput, and not the real model's per-step path on CUDA.
- **Mechanism kill criteria (read from the lever's own lines, after the probe and after every row):** probe §3.2;
  rows §4 (A2 replay calls > 1 % of main passes; acceptance < P0 − 3 pts; drafts/verify < 90 % of P0; CUDA error;
  PCIe replay increment; both A2 reps < +8 % → killed; disagreement → one third rep).
- **Auto-stop:** §7.
- **Binary:** `lane/sl1-spec-ckpt` @ `<sha>`, `build-sl1`, flags select the arms; production args otherwise identical;
  API key from config at launch.
- **N:** 2 per arm (3 only on direction disagreement), all rows reported, failures ledgered as prominently as wins.
- **Promotion to default:** S5/hard-band battery after W-SL1 per the battery-first rule (or James's waiver, ledgered
  with revert path = launcher `--spec-ckpt-mode gpu-fallback` and the SL-1 env flags unset).

## 9. Known limits and what to watch
- Per-step on qwen4exp without `LONGSPEAR_PER_STEP_PLE_TAIL=1` prints a one-time WARNING and keeps the old
  (contaminating) behaviour; flag-unset AUTO still resolves to per-step as before (M8b). The launchers always pass
  the mode explicitly.
- Capacity 5 + clamp: ngram-mod 16-token drafts are cut to 4 (K = 5); without the clamp they would run root-only.
  Watch `clamp_rounds` and their acceptance; stage 2 (dual mode vs input recompute) follows only if the clamp loses
  on those rounds.
- The pre-existing delta-net per-step restore copies `s_l->ne[1]` rows with a stride that ignores the PLE tail, so it
  is only correct with one state slot (production is `-np 1`). The PLE tail restore writes only the restored
  sequence's row.
- The PLE per-step path does not support a split (`-sm graph`) state row; `per_step_alloc` refuses it with an error.
- Crosscheck rounds are slower (a full-row D2H read + a replay per rejected round); never measure throughput with it.
