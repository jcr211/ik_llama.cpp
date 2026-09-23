# GPU verification recipe — State-OS v1 engine lane 1 (keyed header, refusals, one-op restore, companion)

Coordinator-run only, in an approved GPU window, on a quiet box. The lane did NOT run any of this. Lane 0's recipe
(MTP invalidate on `prompt_load` / `SLOT_RESTORE`) is the previous version of this file at commit `d583c220`.

Everything is scripted in `.lane/gpu-verify-l1.ps1` (this worktree). It stops every `llama-server` (standing `:8099`
included), runs the lane-1 build on **port 8101**, loopback only, **without `--api-key`**, and in a `finally` block
always relaunches the standing server with `D:\AI\ik_llama-qwen4exp\launch-standing-8099.ps1` and polls
`http://127.0.0.1:8099/health` (prints `production-restored:200` or `production-restored:FAILED`).

## Preconditions

1. Build exists: `D:\AI\worktrees\stateos-lane1\build-stateos-l1\bin\llama-server.exe` (see `.lane/REPORT.md` for the
   exact build commands and exit codes).
2. `bench/gpu-justify/<YYYYMMDD>-stateos-lane1.md` committed in the Longspear repo (draft text at the end of this file),
   and posted to James at launch.
3. Quiet box: no battery/campaign on the GPU; `nvidia-smi` shows nothing else resident; nobody else on `:8099`
   (a remote Tailscale client thrashes the single slot).
4. At least 25 GB free on `D:` (the 190K-token state is ~4–5 GB; the script deletes it at the end).
5. Diff the script's `$CommonArgs` + `$SpecArgs` against `launch-standing-8099.ps1` (single source of truth) and update
   the script if the standing flags changed. Never copy the launcher's `--api-key` into the script.

## Run

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\AI\worktrees\stateos-lane1\.lane\gpu-verify-l1.ps1
# acceptance leg only (spec off):             ... -File ...\gpu-verify-l1.ps1 -SkipSpecOn
# spec-on leg without the 190K measurement:   ... -File ...\gpu-verify-l1.ps1 -Skip192K
```

Expected wall time: 3 model loads (~1–2 min each: spec-off, spec-on, standing restore) + prefills (4K and 32K twice each
with the cold control, 32K once more spec-on, 190K once ≈ 4–6 min) + restores. Roughly 20–30 min total; with
`-SkipSpecOn` about 8–10 min.

Receipts (all under `D:\AI\worktrees\stateos-lane1\build-stateos-l1\gpu-verify\`): `verdict.txt` (human log),
`results.json` (every number), `launch-args.txt` (exact argv), `specoff.log/.err.log`, `specon.log/.err.log`,
`slots\` (the 4K/32K state files and the tampered copies).

## What each leg does

**Leg A — speculation OFF (the acceptance leg).** Server = production flags minus `--spec-type …` and
`--spec-ckpt-mode`. Synthetic prompt = `/tokenize` of deterministic "Record NNNNNN: sensor=… value=…" lines; `P` =
first 4096 or 32768 ids, `Z` = a short fixed question, `Q` = an unrelated 80-line prompt. All completions go to
`/completion` with `temperature 0, top_k 1, id_slot 0, cache_prompt true`. For each length:

1. `erase`; `/completion P n_predict=1` (the slot now holds exactly `P`: state S0).
2. `save` → `id4k.state` / `id32k.state` (records `n_written`, `stateos.bytes`, `save_ms`).
3. **warm**: `/completion P+Z n_predict=64` from the in-memory S0.
4. twice: `erase`; `/completion Q` (another conversation owns the slot); `restore`; `/completion P+Z n_predict=64`.
5. **cold** (report-only): `erase`; `/completion P+Z` (full prefill).

Then, holding the restored 4K S0 in the slot: a soft-field tamper (`build`) must restore with a warning; each hard field
tampered in a copy of `id4k.state` (`model_fingerprint, n_ctx, cache_type_k, cache_type_v, rope, kv_layout_version,
system_prompt_sha256, kv_geometry, n_tokens, token_sha256`) plus an unknown hard field, a fake and a real (lane-0 file
head) legacy/unkeyed file, a truncated file, a junk file and a missing file must each answer **409** (missing =
`state_missing`, legacy = `state_legacy_unkeyed`, truncated = `state_corrupt`, header = `state_refused` with
`refused_field` = the tampered key) with `slot_untouched: true`. Finally `/completion P+Z` must reproduce the warm output
with the same `prompt_n` — the slot still held S0 through every refusal.

**Leg B — speculation ON = production flags (report-only).** Same round at 32K with `n_predict=128`: the save must say
`companion: saved`, restores `companion: loaded`; the spec-off `id32k.state` (no COMP section) is restored too to read
acceptance without the companion (or its 409 if the spec-on target geometry differs — also informative). Then the 190K
round (`P` = 190000 ids, `n_predict=16`, no cold control) measures the state bytes at the production context.

## Pass criteria (Leg A) and kill criteria

- `GET /props` carries `"stateos": {"version": 1, "keyed_header": true, "companion": false}` on the spec-off server
  (`legA.props_stateos.pass`) and `companion: true` on the spec-on server (`legB.props_stateos.pass`). The appliance
  enables "Model state rewind" only when this object is present with version ≥ 1. Manual check:
  `(Invoke-RestMethod http://127.0.0.1:8101/props).stateos`.
- `identity_4k.verdict` and `identity_32k.verdict` = `PASS`: restored output == in-memory (warm) output, byte-exact
  text, for both restores, and `prompt_n` equal to warm's and ≤ |Z|+1 (no re-prefill). `PASS-IDENTITY /
  REUSE-INCONCLUSIVE` means identity held but the warm run itself re-prefilled (read `prompt_n` in results.json).
- `refusals`: every entry `pass: true`; `soft_build.pass: true`; `slot_untouched_after_refusals.pass: true`.
- Report-only: `identity_cold_vs_warm_report_only` (a cold/warm difference is the known batch-shape arithmetic effect,
  not a state defect), everything in Leg B, `restored_runs_agree` (false = engine run-to-run nondeterminism: mark
  INCONCLUSIVE, not FAIL).
- **Kill (auto-stop in the script):** 4K identity FAIL stops the run before 32K (merged plan: "greedy identity fails at
  4K/32K → do not ship v1 beyond lane 0"); the first hard-field refusal that is not a 409 naming its field stops the run.

## Measured state bytes (paste into the report/ledger)

From `results.json`: `legA.identity_32k.save.bytes` (spec-off 32K: `main`, `checkpoints`, `tokens`, `file`),
`legB.companion_32k.save.bytes` (spec-on 32K incl. `companion`), `legB.bytes_190k.save.bytes` (spec-on 190K), plus
`save_ms` and each `restored[].restore.restore_ms`. p-lite's estimate to compare against: ≈22.7 KiB/token (32K ≈ 726 MiB,
192K ≈ 4.3 GB) + companion ≈1 KiB/token.

## Manual steps (debugging a single call; server on 8101 without an API key)

```powershell
$b = 'http://127.0.0.1:8101'
Invoke-RestMethod -Method Post -Uri "$b/slots/0?action=save"    -ContentType 'application/json' -Body '{"filename":"x.state"}'
Invoke-RestMethod -Method Post -Uri "$b/slots/0?action=restore" -ContentType 'application/json' -Body '{"filename":"x.state"}'
# refusals throw in Invoke-RestMethod; read the body with:
try { Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$b/slots/0?action=restore" -ContentType 'application/json' -Body '{"filename":"x.state"}' } catch { $_.ErrorDetails.Message }
Invoke-RestMethod -Uri "$b/list"   # saved states, now including State-OS files ("format": "stateos-v1")
```

If the script dies without restoring production (it should not — the restore is in `finally`):

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\AI\ik_llama-qwen4exp\launch-standing-8099.ps1
Invoke-RestMethod http://127.0.0.1:8099/health   # expect status ok within ~2 min
```

## Draft `bench/gpu-justify/<YYYYMMDD>-stateos-lane1.md`

1. **Decision it changes:** whether State-OS v1 (keyed save/restore) ships beyond lane 0 and the harness lane's
   `--state-fork` may rely on it (merged-plan kill criterion: greedy identity at 4K/32K), and the per-state byte numbers
   that set the harness byte budget.
2. **Why offline cannot answer:** the header codec, refusal logic and container parser are covered by
   `test-stateos-header` (CPU, passed in the lane); whether a restored qwen4exp slot (recomputed pooled indexer `kp_l`,
   recurrent state, companion KV) continues byte-identically, and what the files weigh, needs the real model on the GPU.
   No reviewer advised waiting.
3. **Mechanism kill criteria:** (a) the first 4K identity round must show restored == in-memory output; a FAIL stops the
   run before 32K; (b) the first tampered-header restore must answer 409 naming its field; otherwise stop. Read after
   the probe and after every round from `verdict.txt`.
4. **Auto-stop:** both criteria `throw` inside the script; the `finally` block restores `:8099` and polls its health.
