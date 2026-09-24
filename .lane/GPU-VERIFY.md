# GPU verification recipe — State-OS v1 engine lane 1 F11 (keyed header, refusals, one-op restore, companion, F11)

Coordinator-run only, in an approved GPU window, on a quiet box. The lane did NOT run any of this. Lane 0's recipe
(MTP invalidate on `prompt_load` / `SLOT_RESTORE`) is the previous version of this file at commit `d583c220`.

Everything is scripted in `.lane/gpu-verify-l1.ps1` (this worktree). It stops every `llama-server` (standing `:8099`
included), runs the **F11 build** (`build-stateos-f11`) on **port 8101**, loopback only, **without `--api-key`**, and in
a `finally` block always relaunches the standing server with `D:\AI\ik_llama-qwen4exp\launch-standing-8099.ps1` and
polls `http://127.0.0.1:8099/health` (prints `production-restored:200` or `production-restored:FAILED`). Its receipts go
to `build-stateos-f11\gpu-verify\`; lane 1's receipts (`D:\AI\worktrees\stateos-lane1\build-stateos-l1\gpu-verify\`,
binary 7c77724b) are only read (dry run, and the real 7c77724b `id4k.state` for the effective_model refusal).

## Preconditions

1. Build exists: `D:\AI\worktrees\stateos-lane1-f11\build-stateos-f11\bin\llama-server.exe`, newer than every tracked
   source file (see `.lane/REPORT.md` and `.lane/REPORT-acceptance-harness.md` for the build and its exit codes).
2. `bench/gpu-justify/<YYYYMMDD>-stateos-lane1-f11.md` committed in the Longspear repo in the TEMPLATE headings (draft
   text at the end of this file), and posted to James at launch.
3. Quiet box: no battery/campaign on the GPU; `nvidia-smi` shows nothing else resident; nobody else on `:8099`
   (a remote Tailscale client thrashes the single slot).
4. At least 25 GB free on `D:` (the 190K-token state is ~4–5 GB; the script deletes it at the end).
5. Diff the script's `$CommonArgs` + `$SpecArgs` against `launch-standing-8099.ps1` (single source of truth) and update
   the script if the standing flags changed. Never copy the launcher's `--api-key` into the script.
6. The server computes `model_fingerprint_v2` once at startup when `--slot-save-path` is set (16 × 64 KiB samples per
   shard plus every shard's header; the log line `State-OS model_fingerprint_v2 <hex> (<ms>)` gives its cost). If it
   fails, the log says `State-OS disabled`, `/props` omits `stateos` and save/restore answer 500.
7. `--verbose` echoes request data; the 190K round writes MB-sized log lines. Harmless, but budget the disk.
8. The script sets `LONGSPEAR_PLE_HIST_REWIND=1` and `LONGSPEAR_PLE_HIST_LOG=1` (plus `LONGSPEAR_VERIFY_TIMING=1`,
   `LONGSPEAR_CG_REVIVE=1`) in the environment every 8101 server inherits, in both legs. It records them per server as
   `<name> env: ...` in `launch-args.txt`, and removes the two PLE switches before relaunching the standing server.

## PLE n-gram history (merged `lane/ple-hist-rewind`)

qwen4exp keeps a host-side PLE n-gram history (`lctx.ple_hist`) outside the KV cache and outside the sequence state.
Before the merge, a restore left it pointing at the previous conversation (Q). The first 2 decoded tokens after every
restore round then used wrong PLE rows, and the identity legs would have failed on that alone.

- **The fix:** with `LONGSPEAR_PLE_HIST_REWIND=1`, the server rebuilds the history at one choke point,
  `batch_pending_prompt`, from `system_tokens` + `cache_tokens[0..n_past)`. It runs on the first prompt batch of
  every request with `p0 > 0`, before the next decode.
- **The State-OS restore path goes through it.** The restore installs `cache_tokens` from TOKS. The next request's
  common-prefix `n_past`, and so `p0`, then comes from those tokens, so no extra `llama_ple_history_set` call is needed
  in the restore itself.
- **An empty-state or failed restore** leaves the slot empty, so `p0 = 0`, and position 0 uses the EOS-padding
  convention.
- **What the script counts:** for each restore round, from the restore through its continuation, the
  `[ple-hist] reset seq=… pos=N` lines with N > 0 (stderr goes to `<name>.err.log`; both logs are read). It also counts
  the `[ple-hist] set … site=server-resume` lines.
- **Where the counts go:** `identity_*.restored[].ple`, `empty_roundtrip.ple`, `main_tamper.ple`,
  `no_companion_32k.ple`, and the summary `results.json → ple_hist`.
- **Pass:** 0 resets at pos > 0 in every round (text-only prompts), and at least one `server-resume` set per identity
  restore round (`identity_*.ple_ok`, `ple_hist.all_ok`). The empty-slot and MAIN-tamper steps also require 0 resets at
  pos > 0 as part of their mechanism condition.

## Run

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\AI\worktrees\stateos-lane1-f11\.lane\gpu-verify-l1.ps1
# acceptance leg only (spec off):             ... -File ...\gpu-verify-l1.ps1 -SkipSpecOn
# spec-on leg without the 190K measurement:   ... -File ...\gpu-verify-l1.ps1 -Skip192K
# parsers only, NO server / GPU / process stop:  ... -File ...\gpu-verify-l1.ps1 -DryRun [-DryRunDir <receipts dir>] [-DryRunLog <server .log>]
# e.g. against lane 1's receipts:  ... -DryRun -DryRunDir D:\AI\worktrees\stateos-lane1\build-stateos-l1\gpu-verify
```

**Dry run first.** `-DryRun` exits before the preflight: it stops nothing, starts nothing, creates no directory and
does not write the receipts. It reads the receipts in `-DryRunDir` (default this build's `gpu-verify\`): every
request/response pair recorded in `specoff.log` and `specon.log` (or one `-DryRunLog`) goes through the same parsers the
live run uses (tokenize, completion incl. draft acceptance, save, restore, erase, /props). It also checks:
- the `[ple-hist]` counter on each recorded log pair and on known synthetic lines;
- draft acceptance summed over the recorded completions, and on synthetic drafted/undrafted responses;
- that a missing response field throws an error naming the field, and the F11 restore shape (`stateos.checkpoints`);
- the identity verdict rule on all eight branches, then on every identity leg in the recorded `results.json`
  (`recorded=… now=… [reason]`), with each leg's warm/round-1/round-2 draft acceptance read back from the recorded
  responses (matched in order by text, `prompt_n` and `predicted_n`);
- the header readers on a recorded `slots\id4k.state` (and whether it carries `effective_model`).

A restore recorded by a binary before `b29a940c` (lane 1's 7c77724b) has no `stateos.checkpoints` status, which the F11
parser requires; the dry run prints those as `skip … pre-F11 recording` and counts them, and uses the checkpoint counts
only (`ckpt: counts only`) for the recorded legs. It exits 0 only when everything else parses. It works in Windows
PowerShell 5.1 and PowerShell 7.

Note for the coordinator: `.claude/hooks/gpu-spend-guard.mjs` matches any command naming `gpu-verify` and has no
`-DryRun` exemption for it (its entry has `dry: []`), so even a dry run is refused unless the command goes through a
wrapper that hardcodes `-DryRun` (what this lane did) or the hook gains `dry: ["-DryRun"]`.

This branch adds two things:
- The dry run can read a log that a live server still holds open.
- The restore parser requires `stateos.checkpoints` and `stateos.checkpoints_restored`, so the `ckpt_ok` hard check
  cannot read a missing field as "0 restored". An empty-slot restore counts as `absent`.

**2026-09-24 abort ("Cannot index into a null array", 10 s into `Test-Identity 'id4k'`).** Root cause is a script bug,
not the server:
- The abort came after the warm continuation, when `Ple-Counts` counted the first `[ple-hist] set ...
  site=server-resume` line in `specoff.err.log`.
- `Ple-Counts` wrapped `Read-LogSince`'s single `string[]` in `@(...)`, so each log became one array element.
- `-match` on an array filters and does not set `$Matches`, so `$Matches[1]` indexed null.
- Every response the server returned matched what the script expects; the recorded `specoff.log` shows it.

The fix:
- The counters read with `[regex]::Match` and assert that every line is a string.
- Every response field the run depends on goes through `Need`, which throws naming the missing field.
- The refusal and negative checks use `Field`, which records `<missing: path>` and fails that check.

Expected wall time: 3 model loads (~1–2 min each: spec-off, spec-on, standing restore) + prefills (4K and 32K twice each
with the cold control, 32K twice more spec-on for the warm run and the warm-vs-warm control, 190K once, or twice when
its restored output differs ≈ 5–8 min) + restores + the F11 steps (seconds, plus one 538 MB file copy). Roughly
20–30 min total (lane 1's run: 17 min); with `-SkipSpecOn` about 8–10 min.

Receipts (all under `D:\AI\worktrees\stateos-lane1-f11\build-stateos-f11\gpu-verify\`): `verdict.txt` (human log),
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
tampered in a copy of `id4k.state` (`model_fingerprint_v2, effective_model, n_ctx, cache_type_k, cache_type_v, rope, kv_layout_version,
system_prompt_sha256, kv_geometry, n_tokens, token_sha256`) plus an unknown hard field, a fake and a real (lane-0 file
head) legacy/unkeyed file, a truncated file, a junk file and a missing file must each answer **409** (missing =
`state_missing`, legacy = `state_legacy_unkeyed`, truncated = `state_corrupt`, header = `state_refused` with
`refused_field` = the tampered key) with `slot_untouched: true`. Then `/completion P+Z` must reproduce the warm output
with the same `prompt_n` — the slot still held S0 through every refusal.

Destructive paths (review MUST-3), oracle = the 4K cold output (a full prefill of `P+Z`):
- **Empty-slot round trip:** `erase` → `save empty.state` (200, `n_saved` 0) → `/completion Q` → `restore empty.state` →
  200 with `stateos.empty: true` (restore of an empty state is an erase; the loader is not called) → `/health` ok →
  `/completion P+Z` re-prefills (`prompt_n` = |P+Z|) and equals the cold output. Before the fix this aborted the server
  (`read_kv_cache_meta` indexed `cells[head - 1]` for 0 cells).
- **MAIN tamper:** a copy of `id4k.state` whose MAIN `cell_count` is +1 (container still well-formed, so verification
  passes) → `restore` → **500** with `slot_untouched: false` → `/health` ok → `/completion P+Z` re-prefills and equals
  the cold output.
- **Leg B, companion sub-header tamper:** a copy of `on32k.state` with one hex digit of `companion_kv_geometry` changed
  (same length) → **200** with `stateos.companion` = `skipped: companion field 'companion_kv_geometry' differs …`.

Save and restore responses also carry `stateos.kv_pos_max` (report-only): on legitimate flows it should equal
`n_tokens - 1` (or -1 for an empty state). Read it from `results.json`; a systematic mismatch is worth knowing before the
harness relies on token-exact reuse.

**Leg B — speculation ON = production flags (report-only).** Same round at 32K with `n_predict=128`: the save must say
`companion: saved`, restores `companion: loaded`; the spec-off `id32k.state` (no COMP section) is restored too to read
acceptance without the companion (or its 409 if the spec-on target geometry differs — also informative; lane 1 got 409
`kv_geometry`). Then the 190K round (`P` = 190000 ids, `n_predict=16`, no cold control) measures the state bytes at the
production context.

- **Warm-vs-warm control (`companion_32k.warm_control`, always at 32K; at 190K only when a restored output differs
  from warm).** After the two restore rounds: `erase` → `/completion P n_predict=1` → `/completion P+Z` — the same
  prefill-built S0 and the same continuation as the warm run, with no restore anywhere. Why this and not two
  completions from one cached prefix: a second `P+Z` request on a slot that already holds `P+Z+generated` would truncate
  the generated tail, which on this hybrid model goes through the checkpoint/rollback machinery, i.e. state again, and
  its `Z` would not be re-prefilled, so the batch shapes would differ from the warm run. Rebuilding S0 by prefill keeps
  every computation of the warm run and removes only the restore. What still differs between the two warm runs is
  decode-side: spec-decode draft/verify batch shapes and the server-wide ngram-mod table, which grew since the first warm
  run (neither is slot state). `agrees_with_warm` records the outcome.
- **Draft acceptance** comes from each `/completion` response: the server adds `timings.draft_n` and
  `timings.draft_n_accepted` (plus `draft_by_depth`) whenever the request drafted (`n_draft_total > 0`); absent means 0
  drafted. Recorded as `warm_acceptance`, `restored[].acceptance`, `warm_control.acceptance`,
  `no_companion_32k.acceptance` (`accepted`, `generated`, `rate`). Lane 1's run recorded 0/0/null everywhere because
  the old counter matched the `draft acceptance rate = …` line in `specon.log` (stdout), but the server prints that line
  with `SLT_CNT` to stderr (`specon.err.log`); the responses had the numbers all along (e.g. on32k warm 84/107, restored
  107/116 and 113/123, on190k 8/24 each — read back by the dry run).

## Pass criteria (Leg A) and kill criteria

- `GET /props` carries `"stateos": {"version": 1, "keyed_header": true, "companion": false}` on the spec-off server
  (`legA.props_stateos.pass`) and `companion: true` on the spec-on server (`legB.props_stateos.pass`). The appliance
  enables "Model state rewind" only when this object is present with version ≥ 1. Manual check:
  `(Invoke-RestMethod http://127.0.0.1:8101/props).stateos`.
- `identity_4k.verdict` and `identity_32k.verdict` = `PASS`: restored output == in-memory (warm) output, byte-exact
  text, for both restores, and `prompt_n` equal to warm's and ≤ |Z|+1 (no re-prefill). `PASS-IDENTITY /
  REUSE-INCONCLUSIVE` means identity held but the warm run itself re-prefilled (read `prompt_n` in results.json).
- **Identity verdict rule, every identity leg** (`Get-IdentityVerdict`, shared by the live run and the dry run; the
  reason is in `verdict_reason`, and in `fail_reason` for a FAIL):
  1. checkpoints not all restored (`ckpt_ok` false) → `FAIL`, whatever the outputs;
  2. both restored outputs == warm → `PASS` (or `PASS-IDENTITY / REUSE-INCONCLUSIVE`, above);
  3. restored ≠ warm and the warm-vs-warm control ran (Leg B): the two warm runs **also disagree** → `INCONCLUSIVE`
     (attributed to decode nondeterminism); the two warm runs **agree** while restored differs → `FAIL` (a state-defect
     signal: Leg B fails);
  4. restored ≠ warm, no control (Leg A, and 190K when it was not needed): `restored_runs_agree` false →
     `INCONCLUSIVE` (engine run-to-run nondeterminism); the two restored runs agree with each other → `FAIL`.

  Only a 4K `FAIL` stops the run. Lane 1's `on32k` (restored ≠ warm, the two restored runs different from each other,
  no control) was labelled `FAIL` by the old script; under this rule it is `INCONCLUSIVE` (the dry run shows
  `recorded=FAIL now=INCONCLUSIVE`), and the F11 run's control decides between rule 3's two outcomes.
- `identity_*.ckpt_ok` (hard: `false` makes the leg's `verdict` FAIL, and at 4K the run stops): every restore round reports `stateos.checkpoints == "restored"` and
  `checkpoints_restored == save.checkpoints_saved`. The rounds restore right after a short conversation (Q), which is
  where a bound measured on the slot's current length wrongly refused or dropped checkpoints.
- `refusals`: every entry `pass: true`; `soft_build.pass: true`; `slot_untouched_after_refusals.pass: true`.
- `empty_roundtrip.verdict` and `main_tamper.verdict` (Leg A) = `PASS`. Their mechanism conditions (status codes,
  `slot_untouched`, server alive, full re-prefill) must hold in every case. If only the output differs from the cold
  run while `identity_4k.restored_runs_agree` is false, the verdict is `INCONCLUSIVE` (engine nondeterminism, the same
  rule as the identity legs), not `FAIL`.
- `comp_tamper.verdict` (Leg B, report-only leg but a hard expectation) = `PASS`. If `on32k.state` carries no COMP
  section, the step records `FAIL` with the reason and the run continues to the 190K measurement.
- Report-only: `identity_cold_vs_warm_report_only` (a cold/warm difference is the known batch-shape arithmetic effect,
  not a state defect), everything in Leg B, `restored_runs_agree` (false = engine run-to-run nondeterminism: mark
  INCONCLUSIVE, not FAIL — rule 4 above), `warm_control` (rule 3), the draft acceptance numbers.
- F11 (hard expectations, no kill; see the next section): `tmp_cleanup.pass`, the F11 entries of `refusals`
  (`<effective_model absent>`, `<7c77724b file>`, `<unreadable: directory>`, `<reserved name: save|restore|rename>`),
  `list_redacted.pass`, `adapter_generation.pass`; `effective_model_header.value` is report-only (expect `none`).

## F11 behaviours: what the script exercises on the real model, and what it leaves to the CPU tests

| F11 change | On the GPU (Leg A unless noted) | Why not / where else |
|---|---|---|
| `effective_model` hard field (`0c1bebea`, `1e94d160`, `c5d66f76`) | tampered value → 409 `refused_field: effective_model` (in the hard-field loop); the field removed → 409 naming it; lane 1's real 7c77724b `id4k.state` → 409 naming it; the value in `id4k.state` recorded (`effective_model_header`, expect `none`) | — |
| runtime adapter generation (`e2b76a3a`, `c5d66f76`) | `adapter_generation`: bad control-vector id → 400 and the slot still saves; empty `/control-vectors/apply` (200) → save 409 `state_adapters_changed` with `slot_untouched`; erase + re-prefill → save 200 with the same `effective_model`, and `id4k.state` still restores | no LoRA or control-vector file exists for this model, so an applied adapter/vector and a stamp change cannot be shown; `test-stateos-header` covers the stamp parts and `stateos_apply_scales` |
| failed apply → `unknown` sentinel, `state_adapters_unknown` on save and restore (`c5d66f76`, `5eab5279`, `e3e4b7ab`) | not exercised | needs `apply_control_vectors_internal` to fail, i.e. a loaded vector file; CPU tests cover the predicates |
| `--lora-init-without-apply` (`0acc6624`) | not exercised | needs a LoRA adapter for this model; `stateos_lora_parts` unit-tested |
| stale legacy system prompt stamp (`9729d763`) | not exercised | the legacy `system_prompt` field releases and clears every slot and changes `system_prompt_sha256`, which would disturb the other steps; `stateos_slot_start_gen` unit-tested |
| control-vector buffer sizing, upstream fix (`8347edf7`) | not exercised | needs vector files of different lengths; `stateos_cvec_accumulate` unit-tested |
| CKPT section bound, independent of the target slot (`b29a940c`, `4d44c638`) | every identity round: `ckpt_ok` (hard) — 3 / 17 / 32 checkpoints restored after a short Q conversation, 32 = the full list at 190K | the over-budget skip and oversized-record paths need crafted files: `test-stateos-header` |
| `.stateos.tmp` reserved names (`fea27876`, `91579f16`) | save `reserved.stateos.tmp` and restore `RESERVED.STATEOS.TMP` → 409 `state_name_reserved`; `/rename_prompt` onto `renamed.stateos.tmp` → 409 `state_name_reserved`, source kept | trailing-dot/space spellings are refused earlier by `fs_validate_filename` (400); normalisation unit-tested |
| stale `*.stateos.tmp` cleanup at startup (`842e16f6`) | `tmp_cleanup`: a 2-hour-old temp planted before the spec-off start is gone, a fresh one is kept | — |
| flush before the commit rename; temp exact-size check (`842e16f6`, `60f9c9a4`) | implicitly, every save | not observable without fault injection (power loss, a temp replaced mid-save); code review + unit test of `stateos_container_size` |
| `state_unreadable` (`a3ff750d`) | a directory named `dir-not-file.state` → 409 `state_unreadable`, slot untouched | permission-denied files need an ACL change on the box; not done |
| `/list` redaction and legacy range check (`4e1f27ca`) | `list_redacted`: the `id4k.state` entry is `stateos-v1`, `prompt: null`, `prompt_redacted`, `token_count` 4096, `token_sha256` = the save's | out-of-range legacy ids need a crafted legacy file; unit-tested |
| layout-descriptor renderer (`14c84ffe`) | implicitly: `kv_geometry` matches on every same-binary restore | byte-identity with the 7c77724b text cannot show on the GPU (the 7c77724b file is refused earlier, on `effective_model`); goldens in `test-stateos-layout` |
| N5/N6 nits (`9436d471`) | the empty-restore reply is recorded (`empty_roundtrip.restore`) | tokens > n_ctx and the post-commit reply need crafted files / faults: CPU tests |
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

## Draft `bench/gpu-justify/<YYYYMMDD>-stateos-lane1-f11.md`

The committed file must use the TEMPLATE headings (`## Decision this run can change`, `## Why existing or offline data
cannot answer`, `## Mechanism kill criteria`, `## Auto-stop`) or the guard refuses it. Content:

1. **Decision it changes:** whether the F11 build replaces 7c77724b as the State-OS v1 engine (its restore path changed:
   CKPT bound, `effective_model`, adapter stamp), and whether Leg B's spec-on restored ≠ warm is a state defect (warm-vs-
   warm control) or decode nondeterminism — which decides if spec-on State-OS may ship.
2. **Why offline cannot answer:** the header codec, refusal logic, container parser and the F11 predicates are covered
   by `test-stateos-header` (291 checks) and `test-stateos-layout` (CPU); lane 1's receipts (7c77724b) cannot show the
   F11 restore path, and they carry no warm-vs-warm control. Whether an F11-restored qwen4exp slot continues
   byte-identically needs the real model on the GPU. No reviewer advised waiting.
3. **Mechanism kill criteria:** (a) the first 4K identity round must show restored == in-memory output; a FAIL stops the
   run before 32K; (b) the first tampered-header restore must answer 409 naming its field; otherwise stop. Read after
   the probe and after every round from `verdict.txt`.
4. **Auto-stop:** both criteria `throw` inside the script; the `finally` block restores `:8099` and polls its health.
