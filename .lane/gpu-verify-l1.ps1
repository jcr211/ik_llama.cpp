# State-OS lane 1 F11 GPU acceptance (coordinator-run; see .lane/GPU-VERIFY.md). NOT run by the lane.
#
# Takes the GPU slot (stops every llama-server, including the standing :8099), runs the F11 build on port 8101
# WITHOUT --api-key (loopback only), then ALWAYS relaunches the standing server via launch-standing-8099.ps1 and polls
# :8099/health. Receipts: build-stateos-f11\gpu-verify\{verdict.txt, results.json, launch-args.txt, *.log} (lane 1's
# receipts under build-stateos-l1 are never written by this script).
#
# Leg A (spec OFF, the acceptance leg): greedy identity (temperature 0) across save -> pollute -> restore at 4K and 32K,
#   one 409 refusal per hard header field + unknown-hard-field + legacy/unkeyed + corrupt + missing, a soft-field warning,
#   a continuation proving the slot was untouched by every refusal, then the destructive paths: an empty-slot round trip
#   and a MAIN-payload tamper (500, slot cleared, server alive, correct re-prefill). Auto-stops when the 4K or the 32K
#   identity leg is anything but PASS* (spec off: restored != warm is FAIL, no nondeterminism allowance) or on a refusal
#   that is not a 409 naming its field (mechanism kill criteria).
# Before the preflight touches anything, and again before each launch, the build is attested (Assert-Binary): the
#   commit embedded at build time is in the exe, is an ancestor of HEAD with identical build inputs, no build input is
#   modified, and the exe + llama/ggml/mtmd DLLs are hashed and newer than their own inputs. With production down,
#   `llama-server --version` must print the same commit. The production stop is inside the protected block; finally
#   relaunches production before saving receipts. Exit code: 0 only when the run completed, every hard expectation
#   held and production is back (hard_failures in results.json).
#   F11 behaviours observable on the real model: the startup cleanup of stale *.stateos.tmp files, effective_model in
#   the save header and its refusal (tampered, missing, and a real 7c77724b file), state_unreadable, the reserved
#   .stateos.tmp names (save, restore, /rename_prompt), /list redaction, and the runtime adapter generation (an empty
#   /control-vectors/apply makes the slot unsaveable until it re-prefills; a bad id changes nothing).
# Leg B (spec ON = production flags, report-only): companion section saved/loaded at 32K with a warm-vs-warm control
#   (two warm runs, no restore between them; valid only with equal prompt_n and no prompt-cache load; restored != warm
#   is UNPROVEN without a valid control, FAIL when the valid control agrees, INCONCLUSIVE when it also disagrees),
#   a companion sub-header tamper (200, companion skipped), draft acceptance
#   (from each /completion response's timings) with the companion vs a companion-less file, and the state bytes +
#   restore time at 190K tokens (production context 196608).
#
# -DryRun: no server, no GPU, no process is stopped, nothing is written. Runs the response parsers below against the
#   requests/responses a previous run recorded in its --verbose server logs (specoff.log and specon.log in -DryRunDir,
#   default this build's gpu-verify dir; or one -DryRunLog), the [ple-hist] counter and draft acceptance on those and on
#   synthetic inputs, the identity verdict rule on the recorded results.json, and the header readers against a recorded
#   slots\id4k.state; exits 0 when everything parsed, 1 otherwise.
param(
    [switch] $SkipSpecOn,
    [switch] $Skip192K,
    [switch] $DryRun,
    [string] $DryRunLog,
    [string] $DryRunDir
)
$ErrorActionPreference = 'Stop'

$Worktree   = 'D:\AI\worktrees\stateos-lane1-f11'
$PatchedExe = Join-Path $Worktree 'build-stateos-f11\bin\llama-server.exe'
$Model      = 'D:\AI\LLM Models\custom\Qwen3.8-Flash-Next-MXFP4moe-ngramQ8-MTP.gguf'
$Standing   = 'D:\AI\ik_llama-qwen4exp\launch-standing-8099.ps1'
$Lane0Slot  = 'D:\AI\worktrees\stateos-lane0\build\gpu-verify\slots\mtp-invalidate-slot.bin' # a real legacy file, if still present
$Lane1Slot  = 'D:\AI\worktrees\stateos-lane1\build-stateos-l1\gpu-verify\slots\id4k.state'    # saved by 7c77724b (no effective_model), if present
$Port       = 8101
$Base       = "http://127.0.0.1:$Port"
$Root       = Join-Path $Worktree 'build-stateos-f11\gpu-verify'
$SlotDir    = Join-Path $Root 'slots'
$VerdictTxt = Join-Path $Root 'verdict.txt'
$ResultsJs  = Join-Path $Root 'results.json'

# Production flags from launch-standing-8099.ps1 minus -m/--api-key/--host/--port and minus the speculation flags
# (diff against that launcher before running; it is the single source of truth).
$CommonArgs = @('-ngl','999','-ncmoe','37','-fa','1','-c','196608','-ub','512','-ctk','q8_0','-ctv','q8_0','-np','1','-t','24','-tb','32',
                '--jinja','--temp','1.0','--top-p','0.95','--top-k','20','--min-p','0.0','--reasoning-budget','1024','-rtr','-muge')
$SpecArgs   = @('--spec-type','ngram-mod:n_min=4','--spec-type','mtp:n_max=4','--spec-ckpt-mode','gpu-fallback')
# Inherited by every 8101 server (both legs). PLE_HIST_REWIND=1 rebuilds the qwen4exp PLE n-gram history at the one
# server choke point before the next decode (restores included); PLE_HIST_LOG=1 prints the [ple-hist] telemetry the
# per-round check below counts. Both are removed again before the standing server is relaunched.
$ServerEnv = [ordered]@{
    LONGSPEAR_VERIFY_TIMING   = '1'
    LONGSPEAR_CG_REVIVE       = '1'
    LONGSPEAR_PLE_HIST_REWIND = '1'
    LONGSPEAR_PLE_HIST_LOG    = '1'
}
# (set inside the protected block, after the preflight: a refused preflight leaves this shell's environment alone)

if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $Root, $SlotDir | Out-Null }
Add-Type -AssemblyName System.Net.Http
$Http = New-Object System.Net.Http.HttpClient
$Http.Timeout = [TimeSpan]::FromMinutes(60)   # upper bound; each call carries its own, shorter timeout (Api -TimeoutSec)
$Results = [ordered]@{ started = (Get-Date).ToUniversalTime().ToString('o'); binary = [ordered]@{}; legA = [ordered]@{}; legB = [ordered]@{} }

function Log([string] $m) {
    $line = "[$((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))] $m"
    if (-not $DryRun) { Add-Content -LiteralPath $VerdictTxt -Value $line -Encoding utf8 } # a dry run leaves the receipts alone
    Write-Host $line
}
function Short([string] $s) { if ($null -eq $s) { '' } elseif ($s.Length -gt 400) { $s.Substring(0, 400) + '...' } else { $s } }

# A required response field by dotted path ('timings.prompt_n'). Throws naming the field when the body is not JSON or any
# step is missing or null, so a changed response shape fails loudly instead of as "Cannot index into a null array".
function Need($Resp, [string] $Path, [string] $What) {
    if ($null -eq $Resp) { throw "$What`: no response (wanted field '$Path')" }
    $v = $Resp.Body
    if ($null -eq $v) { throw "$What`: HTTP $($Resp.Status) body is not JSON (wanted field '$Path'): $(Short $Resp.Raw)" }
    foreach ($seg in $Path.Split('.')) {
        if (($null -eq $v) -or -not (@($v.PSObject.Properties | ForEach-Object { $_.Name }) -contains $seg)) {
            throw "$What`: response field '$Path' is missing (no '$seg'; HTTP $($Resp.Status)): $(Short $Resp.Raw)"
        }
        $v = $v.$seg
    }
    if ($null -eq $v) { throw "$What`: response field '$Path' is null (HTTP $($Resp.Status)): $(Short $Resp.Raw)" }
    # not Write-Output -NoEnumerate: PowerShell 7 wraps a scalar in a List[object] that way
    if ($v -is [System.Array]) { return ,$v } else { return $v }
}
# The same lookup for the refusal / negative paths, where a missing field must fail that check, not abort the run:
# returns the value, or the marker '<missing: path>' (never equal to an expected value; recorded in results.json).
function Field($Resp, [string] $Path) {
    $v = if ($null -eq $Resp) { $null } else { $Resp.Body }
    foreach ($seg in $Path.Split('.')) {
        if (($null -eq $v) -or -not (@($v.PSObject.Properties | ForEach-Object { $_.Name }) -contains $seg)) { return "<missing: $Path>" }
        $v = $v.$seg
    }
    if ($null -eq $v) { return "<missing: $Path>" }
    if ($v -is [System.Array]) { return ,$v } else { return $v }
}
function Save-Results { $Results | ConvertTo-Json -Depth 10 | Out-File -LiteralPath $ResultsJs -Encoding utf8 }

# Timeouts: 60 s for calls that do no prefill (props, tokenize, list, erase, adapters, rename), 600 s for slot save/
# restore (lane 1: 190K save 23 s, restore 9 s), 3600 s for completions (190K prefill). A hung small call throws, the
# run aborts, and the finally block relaunches production instead of waiting out the client's 60-minute bound.
function Api([string] $Method, [string] $Path, [string] $Json, [int] $TimeoutSec = 60) {
    $req = New-Object System.Net.Http.HttpRequestMessage((New-Object System.Net.Http.HttpMethod($Method)), "$Base$Path")
    if ($Json) { $req.Content = New-Object System.Net.Http.StringContent($Json, [System.Text.Encoding]::UTF8, 'application/json') }
    $cts = New-Object System.Threading.CancellationTokenSource ([TimeSpan]::FromSeconds($TimeoutSec))
    try {
        $resp = $Http.SendAsync($req, $cts.Token).GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    } catch {
        # PowerShell wraps the TaskCanceledException in a MethodInvocationException: test the token, not the type
        if ($cts.IsCancellationRequested) { throw "$Method $Path timed out after $TimeoutSec s" } else { throw }
    } finally { $cts.Dispose() }
    $obj = $null
    try { $obj = $text | ConvertFrom-Json } catch {}
    [pscustomobject]@{ Status = [int] $resp.StatusCode; Body = $obj; Raw = $text }
}

# ---- response parsers (shared by the live run and -DryRun) ----
function Parse-Tokens($r) {
    if ($r.Status -ne 200) { throw "tokenize failed: $($r.Status) $(Short $r.Raw)" }
    $t = Need $r 'tokens' 'tokenize'
    return ,([int[]] @($t))
}
# Draft acceptance is per request in the response: server_slot::get_formated_timings adds timings.draft_n and
# timings.draft_n_accepted only when the request drafted (n_draft_total > 0), so an absent pair means 0 drafted.
# (The "draft acceptance rate = ..." line is SLT_CNT output on stderr, <name>.err.log, which the old counter never read.)
function Parse-Completion($r) {
    if ($r.Status -ne 200) { throw "completion failed: $($r.Status) $(Short $r.Raw)" }
    $dn = Field $r 'timings.draft_n'; $da = Field $r 'timings.draft_n_accepted'
    [pscustomobject]@{
        content          = [string] (Need $r 'content' 'completion')
        prompt_n         = [int]    (Need $r 'timings.prompt_n' 'completion')
        prompt_ms        = [double] (Need $r 'timings.prompt_ms' 'completion')
        predicted_n      = [int]    (Need $r 'timings.predicted_n' 'completion')
        draft_n          = $(if ($dn -is [ValueType]) { [int] $dn } else { 0 })
        draft_n_accepted = $(if ($da -is [ValueType]) { [int] $da } else { 0 })
    }
}
function Parse-Save($r, [string] $What) {
    if ($r.Status -ne 200) { throw "$What save: $($r.Status) $(Short $r.Raw)" }
    [ordered]@{
        n_saved           = Need $r 'n_saved' "$What save"
        n_written         = Need $r 'n_written' "$What save"
        save_ms           = Need $r 'timings.save_ms' "$What save"
        bytes             = Need $r 'stateos.bytes' "$What save"
        companion         = Need $r 'stateos.companion' "$What save"
        checkpoints_saved = Need $r 'stateos.checkpoints_saved' "$What save"
        token_sha256      = Need $r 'stateos.token_sha256' "$What save"
    }
}
function Parse-Restore($r, [string] $What) {
    if ($r.Status -ne 200) { throw "$What restore: $($r.Status) $(Short $r.Raw)" }
    [ordered]@{
        n_restored = Need $r 'n_restored' "$What restore"
        n_read     = Need $r 'n_read' "$What restore"
        restore_ms = Need $r 'timings.restore_ms' "$What restore"
        stateos    = Need $r 'stateos' "$What restore"
        # the ckpt_ok hard check reads these: a missing field must not pass as "0 restored" (an empty-slot restore has no
        # checkpoint status: it restores as an erase)
        checkpoints          = if ((Field $r 'stateos.empty') -eq $true) { 'absent' } else { Need $r 'stateos.checkpoints' "$What restore" }
        checkpoints_restored = Need $r 'stateos.checkpoints_restored' "$What restore"
        companion  = [string] (Field $r 'stateos.companion')   # Leg B's mechanism requires "loaded"
    }
}

# GET /list answers a JSON array of {filename, filesize, mtime, token_count, format, prompt, stateos?} over every regular
# file in --slot-save-path (server.cpp list_saved_prompts). Find one entry by exact file name.
# Never take .Count of an `if`-expression's result: the `if` unrolls a one-element @(...) into the bare PSCustomObject,
# and in Windows PowerShell 5.1 a PSCustomObject has no .Count (it is $null), which made the 2026-09-24 F11 run record
# `entry: null` although the server listed id4k.state (the recorded body in specoff.log has it).
function Find-ListEntry($Resp, [string] $Name) {
    if (($null -eq $Resp) -or ($Resp.Status -ne 200) -or ($null -eq $Resp.Body)) { return $null }
    $hits = New-Object System.Collections.Generic.List[object]
    foreach ($e in @($Resp.Body)) { if (($null -ne $e) -and ([string] $e.filename -ceq $Name)) { $hits.Add($e) } }
    if ($hits.Count -eq 1) { return $hits[0] }
    return $null
}
# a State-OS /list entry names the state (format, count, digest) and redacts its text (F11 4e1f27ca)
function Test-ListEntry($E, [string] $TokenSha, [int] $NTokens) {
    if ($null -eq $E) { return $false }
    $r = [pscustomobject]@{ Status = 200; Body = $E }
    return ((Field $r 'format') -eq 'stateos-v1') -and (@($E.PSObject.Properties.Name) -contains 'prompt') -and ($null -eq $E.prompt) -and
           ((Field $r 'stateos.prompt_redacted') -eq $true) -and ((Field $r 'token_count') -eq $NTokens) -and
           ((Field $r 'stateos.token_sha256') -eq $TokenSha)
}

function Get-Tokens([string] $Text) {
    $r = Api 'POST' '/tokenize' (@{ content = $Text } | ConvertTo-Json -Compress)
    return ,(Parse-Tokens $r)
}

function New-SyntheticText([int] $Lines, [int] $Seed) {
    $words = @('alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel','india','juliet','kilo','lima')
    $status = @('ok','warn','fail')
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $Lines; $i++) {
        $v = (($i + $Seed) * 7919) % 100003
        [void] $sb.Append("Record $($i.ToString('D6')): sensor=$((($i * 31) + $Seed) % 97) value=$v status=$($status[$i % 3]) tag=$($words[($i + $Seed) % 12]) $($words[$v % 12]).`n")
    }
    $sb.ToString()
}

# Completion timeout sized by the expected work: 120 s + 5 ms per token expected to be prefilled (200 tokens/s, 2-3x
# below lane 1's measured prefill rates) + 0.25 s per token to generate, capped at 3600 s. A continuation expected to
# reuse the cached prefix passes only its suffix (e.g. 4K/32K P+Z: ~150 s); only a real 32K/190K prefill gets a long
# bound (32K: ~285 s, 190K: ~1070 s). If a "reusing" call re-prefills instead, it times out and the run aborts into the
# finally block (production first), which is the right outcome for that defect.
function Get-CompletionTimeout([int] $ExpectPrefill, [int] $NPredict) { [int] [math]::Min(3600, 120 + 0.005 * $ExpectPrefill + 0.25 * $NPredict) }
function Complete([int[]] $Ids, [int] $NPredict, [int] $ExpectPrefill = -1) {
    if ($ExpectPrefill -lt 0) { $ExpectPrefill = $Ids.Length }
    $json = '{"prompt":[' + ($Ids -join ',') + '],"n_predict":' + $NPredict +
            ',"temperature":0,"top_k":1,"top_p":1,"min_p":0,"seed":1234,"cache_prompt":true,"id_slot":0,"stream":false}'
    Parse-Completion (Api 'POST' '/completion' $json (Get-CompletionTimeout $ExpectPrefill $NPredict))
}

function Slot([string] $Action, [string] $File) {
    $json = if ($File) { '{"filename":"' + $File + '"}' } else { '{}' }
    Api 'POST' "/slots/0?action=$Action" $json $(if ($Action -eq 'erase') { 60 } else { 600 })
}

function Concat([int[]] $A, [int[]] $B) { $r = New-Object int[] ($A.Length + $B.Length); [Array]::Copy($A, $r, $A.Length); [Array]::Copy($B, 0, $r, $A.Length, $B.Length); return ,$r }
function Head([int[]] $A, [int] $N) { $r = New-Object int[] $N; [Array]::Copy($A, $r, $N); return ,$r }

# log lines appended since a byte offset, read while the server still holds the file open
# Returns ONE string[] object. Assign it to a [string[]] variable or pass it directly: wrapping the call in @(...)
# nests the whole array as a single element (the 2026-09-24 abort: Ple-Counts did @(Read-LogSince ..) + @(..)).
function Read-LogSince([string] $Path, [long] $Offset) {
    if (-not (Test-Path -LiteralPath $Path)) { return ,([string[]] @()) }
    $fs = New-Object System.IO.FileStream($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        [void] $fs.Seek($Offset, [System.IO.SeekOrigin]::Begin)
        $sr = New-Object System.IO.StreamReader($fs)
        return ,([string[]] ($sr.ReadToEnd() -split "`n"))
    } finally { $fs.Dispose() }
}
function Assert-Line($l, [string] $What) {
    if (($null -ne $l) -and ($l -isnot [string])) { throw "internal: a $What log line is a $($l.GetType().FullName), not a string (an array nested by @())" }
}
function Log-Size([string] $Path) { if (Test-Path -LiteralPath $Path) { (Get-Item -LiteralPath $Path).Length } else { 0 } }

# [ple-hist] telemetry: fprintf(stderr), so it lands in <name>.err.log; read both files to be safe.
# A "reset" at pos > 0 means a decode found no history for a position that has predecessors (the defect the
# ple-hist merge fixes); expected 0 for text-only prompts. "set ... site=server-resume" is the choke point firing.
function Ple-Offsets([string] $LogPath) {
    $err = $LogPath -replace '\.log$', '.err.log'
    [pscustomobject]@{ Out = (Log-Size $LogPath); Err = (Log-Size $err); OutPath = $LogPath; ErrPath = $err }
}
# [regex]::Match, never -match/$Matches: -match on an array filters and leaves $Matches unset (null).
function Ple-Counts($Offsets) {
    [string[]] $out = Read-LogSince $Offsets.OutPath $Offsets.Out
    [string[]] $err = Read-LogSince $Offsets.ErrPath $Offsets.Err
    $resets = 0; $resume = 0; $sets = 0
    foreach ($l in (@($out) + @($err))) {
        Assert-Line $l '[ple-hist]'
        if ($null -eq $l) { continue }
        $m = [regex]::Match($l, '\[ple-hist\] reset seq=\d+ pos=(\d+)')
        if ($m.Success) { if ([long] $m.Groups[1].Value -gt 0) { $resets++ }; continue }
        $m = [regex]::Match($l, '\[ple-hist\] set .*site=(\S+)')
        if ($m.Success) { $sets++; if ($m.Groups[1].Value -eq 'server-resume') { $resume++ } }
    }
    [pscustomobject]@{ resets_pos_gt0 = $resets; sets = $sets; sets_server_resume = $resume }
}
# draft acceptance of one parsed completion (Parse-Completion), from its response timings
function Acceptance($C) {
    $a = [int] $C.draft_n_accepted; $g = [int] $C.draft_n
    [pscustomobject]@{ accepted = $a; generated = $g; rate = $(if ($g) { [math]::Round($a / $g, 4) } else { $null }) }
}

# ---- binary identity: attest which build runs --------------------------------------------------------------------
# Build inputs (what the server and its DLLs are compiled from; docs, scripts, tests, .lane/ are not). The attestation:
#  1. the commit embedded at build time (LLAMA_COMMIT, from common\build-info.cpp, which .lane\build-f11.cmd regenerates
#     on every build) is present in the exe's bytes, resolves to an ancestor of HEAD, and has the SAME build inputs as
#     HEAD (git diff over $BuildInputs is empty): the exe was built from the inputs HEAD has;
#  2. no tracked build input is modified in the worktree;
#  3. each artifact (exe, llama.dll, ggml.dll, mtmd.dll) has a SHA-256 (a failed hash refuses) and is newer than every
#     tracked file among its own inputs (ninja relinks a DLL only when its inputs change, so each DLL is gated on its own
#     set, not on the server's).
# In the live run, with production already down, `llama-server --version` must print the same commit (Assert-VersionOutput).
$BuildInputs = @('CMakeLists.txt', 'cmake', 'common', 'ggml', 'include', 'src', 'vendor', 'examples/CMakeLists.txt', 'examples/server', 'examples/mtmd')
$BinDir = Split-Path -Parent $PatchedExe
$Artifacts = [ordered]@{
    'llama-server.exe' = $BuildInputs
    'llama.dll'        = @('CMakeLists.txt', 'cmake', 'src', 'include', 'ggml/include')
    'ggml.dll'         = @('CMakeLists.txt', 'cmake', 'ggml')
    'mtmd.dll'         = @('CMakeLists.txt', 'cmake', 'examples/mtmd', 'include', 'ggml/include')
}
function Get-EmbeddedCommit([string] $BuildInfoText) {
    $m = [regex]::Match($BuildInfoText, 'LLAMA_COMMIT\s*=\s*"([0-9a-fA-F]{4,40})"')
    if ($m.Success) { $m.Groups[1].Value } else { $null }
}
# the "version: N (commit)" line of `llama-server --version` (common.cpp prints it to stderr)
function Get-VersionCommit([string] $Text) {
    $m = [regex]::Match([string] $Text, 'version:\s*\d+\s*\(([0-9a-fA-F]{4,40})\)')
    if ($m.Success) { $m.Groups[1].Value } else { $null }
}
function Test-BytesContain([string] $Path, [string] $Needle) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    # Latin1 maps every byte to one char, so an index into the string is an index into the file. The literal is
    # NUL-terminated but not necessarily NUL-preceded (MSVC packs .rdata), so match "<commit>\0".
    $text = [System.Text.Encoding]::GetEncoding(28591).GetString($bytes)
    return ($text.IndexOf("$Needle`0", [System.StringComparison]::Ordinal) -ge 0)
}
function Get-BinaryIdentity {
    $ErrorActionPreference = 'Continue'   # local: in PS 5.1 a git warning on stderr must not become a terminating error
    $b = [ordered]@{ exe = $PatchedExe; sha256 = $null; head = $null; embedded_commit = $null; embedded_full = $null; artifacts = [ordered]@{}; dirty = @(); ok = $false; reason = '' }
    $why = @()
    $b.head = [string] (& git -C $Worktree rev-parse HEAD 2>$null)
    if (($LASTEXITCODE -ne 0) -or -not $b.head) { $b.reason = 'git rev-parse HEAD failed'; return $b }
    $b.dirty = @(& git -C $Worktree status --porcelain --untracked-files=no -- @BuildInputs 2>$null)
    if ($LASTEXITCODE -ne 0) { $b.reason = 'git status failed'; return $b }
    if ($b.dirty.Count) { $why += "build inputs modified in the worktree: $($b.dirty -join ', ')" }
    foreach ($name in $Artifacts.Keys) {
        $p = Join-Path $BinDir $name
        $a = [ordered]@{ path = $p; sha256 = $null; mtime = $null; newest_input = $null; newest_input_mtime = $null }
        $b.artifacts[$name] = $a
        if (-not (Test-Path -LiteralPath $p)) { $why += "missing $p"; continue }
        $it = Get-Item -LiteralPath $p
        $a.mtime = $it.LastWriteTime.ToString('o')
        try { $a.sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $p -ErrorAction Stop).Hash.ToLowerInvariant() } catch {}
        if (-not $a.sha256) { $why += "no SHA-256 for $name" }
        $files = @(& git -C $Worktree ls-files -- @($Artifacts[$name]) 2>$null)
        if (($LASTEXITCODE -ne 0) -or ($files.Count -eq 0)) { $why += "git ls-files found no inputs for $name"; continue }
        $newest = $null
        foreach ($f in $files) {
            $fi = Get-Item -LiteralPath (Join-Path $Worktree $f) -ErrorAction SilentlyContinue
            if ($fi -and (($null -eq $newest) -or ($fi.LastWriteTime -gt $newest.LastWriteTime))) { $newest = $fi }
        }
        $a.newest_input = $newest.FullName; $a.newest_input_mtime = $newest.LastWriteTime.ToString('o')
        if ($newest.LastWriteTime -gt $it.LastWriteTime) { $why += "$name is older than its input $($newest.FullName) ($($a.newest_input_mtime)): rebuild" }
    }
    $b.sha256 = $b.artifacts['llama-server.exe'].sha256
    # the embedded commit
    $bi = Join-Path $Worktree 'common\build-info.cpp'
    $emb = if (Test-Path -LiteralPath $bi) { Get-EmbeddedCommit (Get-Content -LiteralPath $bi -Raw) } else { $null }
    $b.embedded_commit = $emb
    if (-not $emb) { $why += 'no LLAMA_COMMIT in common\build-info.cpp (build with .lane\build-f11.cmd)' }
    else {
        if ((Test-Path -LiteralPath $PatchedExe) -and -not (Test-BytesContain $PatchedExe $emb)) { $why += "the exe does not embed build-info's commit $emb (build-info.cpp regenerated after the link?)" }
        $full = [string] (& git -C $Worktree rev-parse --verify --quiet "$emb^{commit}" 2>$null)
        if (($LASTEXITCODE -ne 0) -or -not $full) { $why += "embedded commit $emb does not resolve" }
        else {
            $b.embedded_full = $full
            & git -C $Worktree merge-base --is-ancestor $full $b.head 2>$null
            if ($LASTEXITCODE -ne 0) { $why += "embedded commit $emb is not an ancestor of HEAD" }
            & git -C $Worktree diff --quiet $full $b.head -- @BuildInputs 2>$null
            if ($LASTEXITCODE -ne 0) { $why += "build inputs differ between the embedded commit $emb and HEAD: rebuild" }
        }
    }
    $b.reason = $why -join '; '
    $b.ok = ($why.Count -eq 0)
    return $b
}
function Get-ArtifactDigest($B) { (@($B.artifacts.Keys | ForEach-Object { "$($_)=$($B.artifacts[$_].sha256)" }) -join ';') }
function Assert-Binary([string] $When) {
    $b = Get-BinaryIdentity
    $Results.binary[$When] = $b
    Log "binary ($When): exe sha256=$($b.sha256) embedded=$($b.embedded_commit) HEAD=$($b.head) ok=$($b.ok)$(if ($b.reason) { " [$($b.reason)]" })"
    if (-not $b.ok) { throw "REFUSED: binary identity ($When): $($b.reason)" }
    if ($Results.binary.Contains('preflight') -and ((Get-ArtifactDigest $b) -ne (Get-ArtifactDigest $Results.binary.preflight))) { throw "REFUSED: an artifact changed since the preflight ($When)" }
}
# `llama-server --version` exits in the argument parser, before any model or backend work; run with production already
# down, 30 s bound, output to files
function Assert-VersionOutput {
    $o = Join-Path $Root 'version.out.txt'; $e = Join-Path $Root 'version.err.txt'
    $p = Start-Process -FilePath $PatchedExe -ArgumentList '--version' -PassThru -WindowStyle Hidden -RedirectStandardOutput $o -RedirectStandardError $e
    if (-not $p.WaitForExit(30000)) { try { $p.Kill() } catch {}; throw 'REFUSED: llama-server --version did not exit within 30 s' }
    $txt = ((Get-Content -LiteralPath $e -Raw -ErrorAction SilentlyContinue) + "`n" + (Get-Content -LiteralPath $o -Raw -ErrorAction SilentlyContinue))
    $vc = Get-VersionCommit $txt
    $Results.binary.version_output = [ordered]@{ commit = $vc; text = (Short $txt) }
    Log "llama-server --version: commit=$vc (embedded per build-info: $($Results.binary.preflight.embedded_commit))"
    if (-not $vc -or ($vc -ne $Results.binary.preflight.embedded_commit)) { throw "REFUSED: --version reports '$vc', expected the attested $($Results.binary.preflight.embedded_commit)" }
}

function Start-TestServer([string] $Name, [string[]] $Extra) {
    Assert-Binary $Name
    $log = Join-Path $Root "$Name.log"; $err = Join-Path $Root "$Name.err.log"
    Remove-Item -LiteralPath $log, $err -ErrorAction SilentlyContinue
    # Start-Process joins ArgumentList with spaces and does not quote: paths with spaces carry their own quotes
    $argv = @('-m', ('"' + $Model + '"'), '--host', '127.0.0.1', '--port', "$Port", '--slot-save-path', ('"' + $SlotDir + '"'), '--verbose') + $CommonArgs + $Extra
    Add-Content -LiteralPath (Join-Path $Root 'launch-args.txt') -Value "$Name`: $PatchedExe $($argv -join ' ')" -Encoding utf8
    Add-Content -LiteralPath (Join-Path $Root 'launch-args.txt') -Value "$Name binary: sha256=$($Results.binary[$Name].sha256) HEAD=$($Results.binary[$Name].head)" -Encoding utf8
    $envLine = ($ServerEnv.Keys | ForEach-Object { "$_=" + [Environment]::GetEnvironmentVariable($_) }) -join ' '
    Add-Content -LiteralPath (Join-Path $Root 'launch-args.txt') -Value "$Name env: $envLine" -Encoding utf8
    $p = Start-Process -FilePath $PatchedExe -ArgumentList $argv -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err
    $healthy = $false
    for ($i = 0; $i -lt 180 -and -not $healthy; $i++) {
        try { $h = Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 2; $healthy = ($h.status -eq 'ok') } catch {}
        if (-not $healthy) { Start-Sleep -Seconds 5 }
        if ($p.HasExited) { throw "$Name server exited during load (see $err)" }
    }
    if (-not $healthy) { throw "$Name server did not become healthy in 15 min" }
    Log "$Name healthy (pid $($p.Id))"
    return $p
}
function Stop-TestServer($p) {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    try { $p.WaitForExit(30000) | Out-Null } catch {}
    Start-Sleep -Seconds 3
}

# ---- header tampering (container: u32 magic | u32 version | u32 header_len | header | sections...) ----
function Read-Exact([System.IO.Stream] $s, [byte[]] $buf, [int] $n) {
    $off = 0
    while ($off -lt $n) { $k = $s.Read($buf, $off, $n - $off); if ($k -le 0) { throw 'short read' }; $off += $k }
}
function New-TamperedCopy([string] $Src, [string] $Dst, [scriptblock] $EditHeader) {
    $in = [System.IO.File]::OpenRead($Src)
    try {
        $pre = New-Object byte[] 12; Read-Exact $in $pre 12
        $len = [BitConverter]::ToUInt32($pre, 8)
        $hb = New-Object byte[] $len; Read-Exact $in $hb ([int] $len)
        $text = & $EditHeader ([System.Text.Encoding]::UTF8.GetString($hb))
        $nb = [System.Text.Encoding]::UTF8.GetBytes($text)
        $out = [System.IO.File]::Create($Dst)
        try {
            $out.Write($pre, 0, 8)
            $out.Write([BitConverter]::GetBytes([uint32] $nb.Length), 0, 4)
            $out.Write($nb, 0, $nb.Length)
            $in.CopyTo($out)
        } finally { $out.Dispose() }
    } finally { $in.Dispose() }
}
function Set-HeaderValue([string] $Text, [string] $Key, [string] $Value) {
    $found = $false
    $lines = foreach ($l in ($Text -split "`n")) {
        if ($l -match ('^[HSI] ' + [regex]::Escape($Key) + '=')) { $found = $true; $l.Substring(0, 2) + $Key + '=' + $Value } else { $l }
    }
    if (-not $found) { throw "field '$Key' is not in the header" }
    return ($lines -join "`n")
}
# payload [offset, size) of a container section ('MAIN', 'COMP', ...), walking the section table
function Find-Section([string] $Path, [string] $Tag) {
    $in = [System.IO.File]::OpenRead($Path)
    try {
        $pre = New-Object byte[] 12; Read-Exact $in $pre 12
        $pos = [long] 12 + [BitConverter]::ToUInt32($pre, 8)
        $sh = New-Object byte[] 16
        while ($pos + 16 -le $in.Length) {
            [void] $in.Seek($pos, [System.IO.SeekOrigin]::Begin); Read-Exact $in $sh 16
            $t = [System.Text.Encoding]::ASCII.GetString($sh, 0, 4).TrimEnd([char] 0)
            $size = [long] [BitConverter]::ToUInt64($sh, 8)
            if ($t -eq $Tag) { return [pscustomobject]@{ Offset = $pos + 16; Size = $size } }
            if ($t -eq 'END') { break }
            $pos += 16 + $size
        }
    } finally { $in.Dispose() }
    throw "section '$Tag' not found in $Path"
}
function Write-BytesAt([string] $Path, [long] $Offset, [byte[]] $Bytes) {
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite)
    try { [void] $fs.Seek($Offset, [System.IO.SeekOrigin]::Begin); $fs.Write($Bytes, 0, $Bytes.Length) } finally { $fs.Dispose() }
}
function Read-BytesAt([string] $Path, [long] $Offset, [int] $N) {
    $fs = [System.IO.File]::OpenRead($Path)
    try { [void] $fs.Seek($Offset, [System.IO.SeekOrigin]::Begin); $b = New-Object byte[] $N; Read-Exact $fs $b $N; return ,$b } finally { $fs.Dispose() }
}

function Get-HeaderValue([string] $Path, [string] $Key) {
    $in = [System.IO.File]::OpenRead($Path)
    try {
        $pre = New-Object byte[] 12; Read-Exact $in $pre 12
        $len = [BitConverter]::ToUInt32($pre, 8)
        $hb = New-Object byte[] $len; Read-Exact $in $hb ([int] $len)
    } finally { $in.Dispose() }
    foreach ($l in ([System.Text.Encoding]::UTF8.GetString($hb) -split "`n")) {
        $m = [regex]::Match($l, '^[HSI] ' + [regex]::Escape($Key) + '=(.*)$')
        if ($m.Success) { return $m.Groups[1].Value }
    }
    return $null
}
function Need-HeaderValue([string] $Path, [string] $Key) {
    $v = Get-HeaderValue $Path $Key
    if ($null -eq $v) { throw "header field '$Key' is missing in $Path" }
    return $v
}

# ---- identity verdict (shared by the live run and -DryRun; GPU-VERIFY.md "Pass criteria") ----------------------
#   mechanism not met (Test-IdentityMechanism)            -> FAIL, whatever the outputs say
#   both restored outputs == warm                         -> PASS, or PASS-IDENTITY / REUSE-INCONCLUSIVE if warm re-prefilled
#   restored != warm, speculation OFF (Leg A)             -> FAIL: spec-off decode is deterministic (lane 1: both 4K and
#                                                            32K restored runs agreed with warm), so no allowance
#   restored != warm, speculation ON (Leg B):
#       no valid warm-vs-warm control                     -> UNPROVEN (neither cleared nor failed)
#       valid control, the two warm runs agree            -> FAIL (state-defect signal)
#       valid control, the two warm runs also disagree    -> INCONCLUSIVE (decode nondeterminism is present; a state
#                                                            defect is not excluded)
# $WarmVsWarm: $null when no VALID control ran, else whether the two warm runs produced the same text.
# Reuse (prompt_n = tokens the request had to evaluate): a continuation that used the cached/restored prefix evaluates
# at most |Z|+1 tokens. A restored run that evaluated more (or, when warm reused, a different count than warm) re-prefilled:
# the server dropped the restored state (e.g. verify_restored_checkpoint failed -> do_reset), so identity would say
# nothing about the restore -> FAIL. REUSE-INCONCLUSIVE is only the case where warm alone re-prefilled.
function Get-IdentityVerdict([bool] $MechOk, [string] $MechReason, [string] $Warm, [string] $Restored1, [string] $Restored2,
                             [int] $WarmPn, [int] $R1Pn, [int] $R2Pn, [int] $NSuffix, $WarmVsWarm, [bool] $SpecOn) {
    $same = ($Restored1 -ceq $Warm) -and ($Restored2 -ceq $Warm)
    $warmReuse = ($WarmPn -le ($NSuffix + 1))
    $restoredReuse = ($R1Pn -le ($NSuffix + 1)) -and ($R2Pn -le ($NSuffix + 1)) -and ((-not $warmReuse) -or (($R1Pn -eq $WarmPn) -and ($R2Pn -eq $WarmPn)))
    $v = if (-not $MechOk) { 'FAIL', $MechReason }
         elseif (-not $restoredReuse) { 'FAIL', "restored run re-prefilled: the restored state was not used (prompt_n warm=$WarmPn restored=$R1Pn/$R2Pn, |Z|+1=$($NSuffix + 1))" }
         elseif ($same -and $warmReuse) { 'PASS', '' }
         elseif ($same) { 'PASS-IDENTITY / REUSE-INCONCLUSIVE', "identity held; the warm run alone re-prefilled (prompt_n=$WarmPn)" }
         elseif (-not $SpecOn) { 'FAIL', 'restored != warm with speculation off (no nondeterminism allowance in Leg A)' }
         elseif ($null -eq $WarmVsWarm) { 'UNPROVEN', 'restored != warm with speculation on, and no valid warm-vs-warm control' }
         elseif ([bool] $WarmVsWarm) { 'FAIL', 'restored != warm while two warm runs (no restore) agree: state-defect signal' }
         else { 'INCONCLUSIVE', 'restored != warm, and two warm runs (no restore) also disagree: decode nondeterminism present, a state defect not excluded' }
    [pscustomobject]@{ verdict = $v[0]; reason = $v[1] }
}
# The identity leg's mechanism, required before any PASS: the save and both restores cover exactly the prompt, at least
# one checkpoint is saved and every one comes back with status "restored" (review F11 P1: the restores follow a short Q
# conversation, where a slot-length-dependent bound got it wrong), and every compared run produced output.
# $Restores: objects with n_restored, checkpoints (status; $null when -StatusUnknown), checkpoints_restored, companion.
# $Outs: parsed completions (content, predicted_n) of warm, round 1, round 2.
# -RequireCompanion (Leg B, spec on): the save says companion "saved" and every restore "loaded" (recipe: Leg B).
function Test-IdentityMechanism([int] $NPrompt, $Save, $Restores, $Outs, [switch] $StatusUnknown, [switch] $RequireCompanion) {
    $why = @()
    $saved = [int] $Save.checkpoints_saved
    if ([int] $Save.n_saved -ne $NPrompt) { $why += "save n_saved=$($Save.n_saved), expected $NPrompt" }
    if ($saved -lt 1) { $why += 'no checkpoint saved (the checkpoint restore would go unexercised)' }
    if ($RequireCompanion -and ([string] $Save.companion -ne 'saved')) { $why += "save companion '$($Save.companion)', expected 'saved'" }
    $k = 0
    foreach ($rs in $Restores) {
        $k++
        if ($RequireCompanion -and ([string] $rs.companion -ne 'loaded')) { $why += "round $k companion '$($rs.companion)', expected 'loaded'" }
        if ([int] $rs.n_restored -ne $NPrompt) { $why += "round $k n_restored=$($rs.n_restored), expected $NPrompt" }
        if (([int] $rs.checkpoints_restored -lt 1) -or ([int] $rs.checkpoints_restored -ne $saved)) { $why += "round $k checkpoints_restored=$($rs.checkpoints_restored) of $saved saved" }
        if (-not $StatusUnknown -and ([string] $rs.checkpoints -ne 'restored')) { $why += "round $k checkpoints status '$($rs.checkpoints)'" }
    }
    $names = @('warm', 'round 1', 'round 2')
    for ($i = 0; $i -lt @($Outs).Count; $i++) {
        $o = @($Outs)[$i]
        if (([int] $o.predicted_n -le 0) -or [string]::IsNullOrEmpty([string] $o.content)) { $why += "$($names[$i]) produced no output (predicted_n=$($o.predicted_n))" }
    }
    [pscustomobject]@{ ok = ($why.Count -eq 0); reason = ($why -join '; ') }
}
# index of the first difference between two sequences (strings or int arrays), -1 when equal
function First-Divergence($A, $B) {
    $a = @($(if ($A -is [string]) { $A.ToCharArray() } else { $A })); $b = @($(if ($B -is [string]) { $B.ToCharArray() } else { $B }))
    $n = [math]::Min($a.Count, $b.Count)
    for ($i = 0; $i -lt $n; $i++) { if ($a[$i] -cne $b[$i]) { return $i } }
    if ($a.Count -eq $b.Count) { return -1 } else { return $n }
}

# ---- one identity round: in-memory continuation vs restored continuation (x2) vs cold (report-only) ----
# -WarmControl 'always' | 'if-differs' (Leg B only): after the restore rounds, rebuild S0 by prefill (erase, P, no
# restore) and run the same warm continuation again; 'if-differs' only when a restored output differs from warm (spend
# GPU time only where the control can change the verdict).
# RAM prompt-cache activity in a window (a cached state loaded instead of a prefill): both logs, since LLAMA_LOG_INFO
# and LOG_VERBOSE may land in different streams
function Cache-Lines($Offsets) {
    [string[]] $out = Read-LogSince $Offsets.OutPath $Offsets.Out
    [string[]] $err = Read-LogSince $Offsets.ErrPath $Offsets.Err
    @(@($out) + @($err) | Where-Object { ($null -ne $_) -and (($_ -like '*prompt cache load*') -or ($_ -like '*MTP invalidate: prompt_load*')) }).Count
}
function Test-Identity([string] $Tag, [int[]] $P, [int[]] $Z, [int[]] $Q, [int] $NGen, [switch] $NoCold, [string] $LogPath, [string] $WarmControl, [switch] $SpecOn) {
    $PZ = Concat $P $Z
    $res = [ordered]@{ n_prompt = $P.Length; n_suffix = $Z.Length; spec_on = [bool] $SpecOn }
    $pw = Ple-Offsets $LogPath                                   # warm window: erase through the warm continuation
    [void] (Slot 'erase' $null)
    [void] (Complete $P 1)                                       # in-memory state S0: the slot holds P
    $res.save = Parse-Save (Slot 'save' "$Tag.state") $Tag
    $po = Ple-Offsets $LogPath
    $w = Complete $PZ $NGen ($Z.Length + 1)                      # continuation from the in-memory S0 (reuses P)
    $res.warm = $w; $res.warm_acceptance = Acceptance $w; $res.warm_ple = Ple-Counts $po
    $warmCache = Cache-Lines $pw

    $runs = @()
    foreach ($k in 1, 2) {
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)      # another conversation occupies the slot
        $po = Ple-Offsets $LogPath                               # from the restore through the continuation
        $rsp = Parse-Restore (Slot 'restore' "$Tag.state") "$Tag round $k"
        $r = Complete $PZ $NGen ($Z.Length + 1)
        $ple = Ple-Counts $po
        $runs += ,([ordered]@{ restore = $rsp; out = $r; acceptance = (Acceptance $r); ple = $ple })
        Log "$Tag restore round $k`: [ple-hist] resets at pos>0 = $($ple.resets_pos_gt0) (expect 0), server-resume sets = $($ple.sets_server_resume)"
    }
    if ($runs.Count -ne 2) { throw "internal: $Tag has $($runs.Count) restore rounds recorded, expected 2" }
    $res.restored = $runs
    $res.ple_resets_pos_gt0 = [int] $runs[0].ple.resets_pos_gt0 + [int] $runs[1].ple.resets_pos_gt0
    $res.ple_ok = ($res.ple_resets_pos_gt0 -eq 0) -and ($runs[0].ple.sets_server_resume -ge 1) -and ($runs[1].ple.sets_server_resume -ge 1)
    $saved = [int] $res.save.checkpoints_saved
    $mech = Test-IdentityMechanism $P.Length $res.save @($runs[0].restore, $runs[1].restore) @($w, $runs[0].out, $runs[1].out) -RequireCompanion:([bool] $SpecOn)
    $res.ckpt_ok = $mech.ok
    $res.mechanism = [ordered]@{ ok = $mech.ok; reason = $mech.reason }
    Log "$Tag mechanism: saved n=$($res.save.n_saved) ckpt=$saved; restored n=$($runs[0].restore.n_restored)/$($runs[1].restore.n_restored) ckpt=$($runs[0].restore.checkpoints_restored)/$($runs[1].restore.checkpoints_restored) status='$($runs[0].restore.checkpoints)'; ok=$($mech.ok)$(if ($mech.reason) { " [$($mech.reason)]" })"
    $same1 = $runs[0].out.content -ceq $w.content
    $same2 = $runs[1].out.content -ceq $w.content

    # warm-vs-warm control (Leg B): S0 rebuilt by the same erase + P prefill as the warm run, no restore anywhere.
    # Valid only when the control re-used exactly as much as the warm run (same prompt_n) and neither window shows a RAM
    # prompt-cache load; otherwise $wvw stays $null and a restored != warm stays UNPROVEN. Uncontrolled even when valid:
    # the server-wide ngram-mod table and the MTP draft history grew between the first warm run and the control.
    $wvw = $null
    if (($WarmControl -eq 'always') -or (($WarmControl -eq 'if-differs') -and -not ($same1 -and $same2))) {
        $pc = Ple-Offsets $LogPath
        [void] (Slot 'erase' $null)
        [void] (Complete $P 1)
        $w2 = Complete $PZ $NGen ($Z.Length + 1)
        $ctlCache = Cache-Lines $pc
        $why = @()
        if ($w2.prompt_n -ne $w.prompt_n) { $why += "control prompt_n=$($w2.prompt_n) != warm prompt_n=$($w.prompt_n)" }
        if ($ctlCache -gt 0) { $why += "$ctlCache prompt-cache line(s) in the control window" }
        if ($warmCache -gt 0) { $why += "$warmCache prompt-cache line(s) in the warm window" }
        if (($w2.predicted_n -le 0) -or [string]::IsNullOrEmpty($w2.content)) { $why += 'control produced no output' }
        $agrees = ($w2.content -ceq $w.content)
        $valid = ($why.Count -eq 0)
        if ($valid) { $wvw = $agrees }
        $res.warm_control = [ordered]@{ out = $w2; acceptance = (Acceptance $w2); agrees_with_warm = $agrees; valid = $valid; invalid_reason = ($why -join '; ')
                                        uncontrolled = 'server-wide ngram-mod table and MTP draft history differ between the first warm run and the control' }
        Log "$Tag warm-vs-warm control (no restore): agrees=$agrees valid=$valid$(if ($why.Count) { " [$($why -join '; ')]" }) (warm2 prompt_n=$($w2.prompt_n), acc=$($res.warm_control.acceptance.accepted)/$($res.warm_control.acceptance.generated))"
    }
    # where the outputs part: character index, and token index after re-tokenizing the texts (report-only)
    $div = [ordered]@{}
    $pairs = [ordered]@{ restored1_vs_warm = $runs[0].out.content; restored2_vs_warm = $runs[1].out.content }
    if ($res.Contains('warm_control')) { $pairs.warm2_vs_warm = $res.warm_control.out.content }
    $wt = $null; try { $wt = Get-Tokens $w.content } catch {}
    foreach ($pk in $pairs.Keys) {
        $tok = $null
        if ($null -ne $wt) { try { $tok = First-Divergence $wt (Get-Tokens $pairs[$pk]) } catch {} }
        $div[$pk] = [ordered]@{ char = (First-Divergence $w.content $pairs[$pk]); token_retokenized = $tok }
    }
    $res.first_divergence = $div
    if (-not $NoCold) {
        [void] (Slot 'erase' $null)
        $res.cold = Complete $PZ $NGen                           # full prefill of P+Z, report-only
    }
    $reuse = ($runs[0].out.prompt_n -eq $w.prompt_n) -and ($runs[1].out.prompt_n -eq $w.prompt_n) -and ($w.prompt_n -le ($Z.Length + 1))
    $res.identity_restored_vs_warm = ($same1 -and $same2)
    $res.no_reprefill = $reuse
    $res.restored_runs_agree = ($runs[0].out.content -ceq $runs[1].out.content)
    if (-not $NoCold) { $res.identity_cold_vs_warm_report_only = ($res.cold.content -ceq $w.content) }
    $vd = Get-IdentityVerdict $mech.ok $mech.reason $w.content $runs[0].out.content $runs[1].out.content $w.prompt_n $runs[0].out.prompt_n $runs[1].out.prompt_n $Z.Length $wvw ([bool] $SpecOn)
    $res.verdict = $vd.verdict
    if ($vd.reason) { $res.verdict_reason = $vd.reason }
    if ($vd.verdict -eq 'FAIL') { $res.fail_reason = $vd.reason }
    Log ("$Tag identity: {0}{6} (warm prompt_n={1}, restored prompt_n={2}/{3}, restore_ms={4:N1}, file={5} B; acc warm={7} restored={8}/{9})" -f $res.verdict, $w.prompt_n, $runs[0].out.prompt_n, $runs[1].out.prompt_n, $runs[0].restore.restore_ms, $res.save.n_written,
        $(if ($vd.reason) { " [$($vd.reason)]" } else { '' }), $res.warm_acceptance.rate, $runs[0].acceptance.rate, $runs[1].acceptance.rate)
    return $res
}

# ---- -DryRun: the parsers above against a recorded --verbose server log (no server, no GPU, nothing stopped) ------
# The server logs each request as an INFO line (status, method, path, params) followed by a VERB line carrying
# request="..." response="..." (JSON-escaped strings).
function Get-RecordedExchanges([string] $LogPath) {
    $list = New-Object System.Collections.Generic.List[object]
    $cur = $null
    # FileShare.ReadWrite: a log a live server still holds open for writing can be read (File.ReadLines refuses it)
    $fs = New-Object System.IO.FileStream($LogPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    $sr = New-Object System.IO.StreamReader($fs)
    try { $all = [string[]] ($sr.ReadToEnd() -split "`n") } finally { $sr.Dispose(); $fs.Dispose() }
    foreach ($l in $all) {
        $l = $l.TrimEnd("`r")
        $m = [regex]::Match($l, 'log_server_request\] request \|.* status=(\d+) method="(\w+)" path="([^"]*)" params=(.*)$')
        if ($m.Success) { $cur = [pscustomobject]@{ Status = [int] $m.Groups[1].Value; Method = $m.Groups[2].Value; Path = $m.Groups[3].Value; Params = $m.Groups[4].Value.Trim() }; continue }
        $m = [regex]::Match($l, ' response=("(?:[^"\\]|\\.)*")\s*$')
        if ($m.Success -and $cur) {
            $raw = [string] ($m.Groups[1].Value | ConvertFrom-Json)
            $obj = $null
            try { $obj = $raw | ConvertFrom-Json } catch {}
            $list.Add([pscustomobject]@{ Path = $cur.Path; Params = $cur.Params; Method = $cur.Method; Status = $cur.Status; Body = $obj; Raw = $raw })
            $cur = $null
        }
    }
    return ,$list
}
function Invoke-DryRun {
    $dir = if ($DryRunDir) { $DryRunDir } else { $Root }
    $logs = @(if ($DryRunLog) { $DryRunLog } else { @('specoff.log', 'specon.log') | ForEach-Object { Join-Path $dir $_ } | Where-Object { Test-Path -LiteralPath $_ } })
    if ($logs.Count -eq 0) { throw "dry run: no recorded specoff.log / specon.log in $dir" }
    foreach ($lp in $logs) { if (-not (Test-Path -LiteralPath $lp)) { throw "dry run: no recorded log at $lp" } }
    $check = {
        param([string] $Name, [scriptblock] $Body)
        try { $out = & $Body; Write-Host "  ok    $Name$(if ($out) { ": $out" })" }
        catch { Write-Host "  FAIL  $Name`: $($_.Exception.Message)"; $script:dryFails++ }
    }
    $script:dryFails = 0; $script:dryPreF11 = 0
    Write-Host "dry run: receipts in $dir"
    $completionsByLog = @{}

    foreach ($logPath in $logs) {
        Write-Host "dry run against $logPath"
        $ex = Get-RecordedExchanges $logPath
        Write-Host "  recorded exchanges with a response: $($ex.Count)"
        $seen = @{}
        $saveSha = @{}
        $comps = New-Object System.Collections.Generic.List[object]
        foreach ($e in $ex) {
            $kind = if ($e.Path -like '/slots/*') { 'slots ' + ([regex]::Match($e.Params, '"action":"(\w+)"').Groups[1].Value) } else { $e.Path }
            $seen[$kind] = 1 + [int] $seen[$kind]
            switch ($kind) {
                '/tokenize'     { & $check "tokenize #$($seen[$kind])" { "$((Parse-Tokens $e).Length) tokens" } }
                '/completion'   {
                    & $check "completion #$($seen[$kind])" {
                        $c = Parse-Completion $e; $comps.Add($c); $a = Acceptance $c
                        "prompt_n=$($c.prompt_n) predicted_n=$($c.predicted_n) content=$($c.content.Length) chars draft=$($a.accepted)/$($a.generated)"
                    }
                }
                '/props'        { & $check '/props stateos' { $v = Need $e 'stateos.version' '/props'; [void] (Need $e 'stateos.keyed_header' '/props'); [void] (Need $e 'stateos.companion' '/props'); "version=$v" } }
                'slots save'    {
                    # the F11 run makes deliberate save refusals (reserved name, adapters changed): parse them as refusals
                    if ($e.Status -eq 200) { & $check "slots save #$($seen[$kind])" { $s = Parse-Save $e 'dry'; $saveSha[[string] (Field $e 'filename')] = [string] $s.token_sha256; "n_saved=$($s.n_saved) n_written=$($s.n_written) checkpoints_saved=$($s.checkpoints_saved)" } }
                    else { & $check "slots save refusal #$($seen[$kind])" { "status=$($e.Status) type=$(Field $e 'error.type') slot_untouched=$(Field $e 'error.slot_untouched')" } }
                }
                '/list'         {
                    # the live lookup, on the recorded body: id4k.state must be found and redacted, with the save's digest
                    & $check "/list #$($seen[$kind])" {
                        $n = @($e.Body).Count
                        $e0 = Find-ListEntry $e 'id4k.state'
                        if ($null -eq $e0) { throw "id4k.state not found among $n entries" }
                        $sha = $saveSha['id4k.state']
                        if (-not $sha) { throw 'no recorded id4k.state save to compare the digest with' }
                        if (-not (Test-ListEntry $e0 $sha 4096)) { throw "id4k.state entry fails the redaction check: $($e0 | ConvertTo-Json -Compress -Depth 5)" }
                        "$n entries; id4k.state: format=$($e0.format) token_count=$($e0.token_count) prompt=null prompt_redacted=$($e0.stateos.prompt_redacted) digest matches the save"
                    }
                }
                'slots erase'   { & $check "slots erase #$($seen[$kind])" { "n_erased=$(Need $e 'n_erased' 'erase')" } }
                'slots restore' {
                    # binaries before b29a940c (lane 1's 7c77724b) did not report the checkpoints status the F11 parser
                    # requires: such a recording is a known older shape (skip, counted), not a parser failure
                    $preF11 = ($e.Status -eq 200) -and ((Field $e 'stateos.empty') -ne $true) -and ((Field $e 'stateos.checkpoints') -like '<missing:*') -and ((Field $e 'stateos.checkpoints_restored') -is [ValueType])
                    if ($preF11) { $script:dryPreF11++; Write-Host "  skip  slots restore #$($seen[$kind]): pre-F11 recording (no stateos.checkpoints status; checkpoints_restored=$(Field $e 'stateos.checkpoints_restored'))" }
                    elseif ($e.Status -eq 200) { & $check "slots restore #$($seen[$kind])" { $r = Parse-Restore $e 'dry'; "n_restored=$($r.n_restored) checkpoints=$($r.checkpoints) restored=$($r.checkpoints_restored)" } }
                    else { & $check "slots restore refusal #$($seen[$kind])" { "status=$($e.Status) type=$(Field $e 'error.type') refused_field=$(Field $e 'error.refused_field') slot_untouched=$(Field $e 'error.slot_untouched')" } }
                }
            }
        }
        # the run tokenizes on the spec-off server only
        $needed = if ([System.IO.Path]::GetFileName($logPath) -eq 'specon.log') { @('/completion', 'slots save') } else { @('/tokenize', '/completion', 'slots save') }
        foreach ($need in $needed) {
            if (-not $seen[$need]) { Write-Host "  FAIL  no recorded '$need' exchange in the log"; $script:dryFails++ }
        }
        $completionsByLog[[System.IO.Path]::GetFileName($logPath)] = $comps
        # the counters over the recorded logs (the abort was here: Ple-Counts on the recorded [ple-hist] set line)
        & $check 'Ple-Counts over the recorded log pair' { $p = Ple-Counts ([pscustomobject]@{ Out = 0; Err = 0; OutPath = $logPath; ErrPath = ($logPath -replace '\.log$', '.err.log') }); "resets_pos_gt0=$($p.resets_pos_gt0) sets=$($p.sets) server_resume=$($p.sets_server_resume)" }
        & $check 'draft acceptance from the recorded /completion responses' {
            $d = @($comps | Where-Object { $_.draft_n -gt 0 })
            $sa = 0; $sg = 0; foreach ($c in $d) { $sa += $c.draft_n_accepted; $sg += $c.draft_n }
            "$($d.Count) of $($comps.Count) completions drafted; accepted/generated over them = $sa/$sg"
        }
    }

    # a changed response shape must throw naming the field, and Field must mark it
    & $check 'missing field throws naming it' {
        $fake = [pscustomobject]@{ Status = 200; Body = ('{"content":"x","timings":{"prompt_ms":1}}' | ConvertFrom-Json); Raw = '{"content":"x","timings":{"prompt_ms":1}}' }
        $msg = $null
        try { [void] (Parse-Completion $fake) } catch { $msg = $_.Exception.Message }
        if (($null -eq $msg) -or ($msg -notlike "*'timings.prompt_n'*")) { throw "expected a throw naming 'timings.prompt_n', got: $msg" }
        $mk = Field $fake 'error.refused_field'
        if ($mk -ne '<missing: error.refused_field>') { throw "Field marker was '$mk'" }
        $msg
    }

    # the [ple-hist] counter against known lines (expected: 1 reset at pos>0, 2 sets, 1 at server-resume)
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('gpu-verify-dry-' + [guid]::NewGuid().ToString('N') + '.log')
    try {
        Set-Content -LiteralPath $tmp -Encoding utf8 -Value @(
            '[ple-hist] reset seq=0 pos=0', '[ple-hist] reset seq=0 pos=17', '[ple-hist] set seq=0 next_pos=4096 n_prev=2 site=server-resume',
            '[ple-hist] set seq=0 next_pos=12 n_prev=2 site=decode')
        & $check 'Ple-Counts on known lines' {
            $p = Ple-Counts ([pscustomobject]@{ Out = 0; Err = 0; OutPath = $tmp; ErrPath = ($tmp + '.none') })
            if (($p.resets_pos_gt0 -ne 1) -or ($p.sets -ne 2) -or ($p.sets_server_resume -ne 1)) { throw "got resets=$($p.resets_pos_gt0) sets=$($p.sets) resume=$($p.sets_server_resume), expected 1/2/1" }
            'resets=1 sets=2 resume=1'
        }
    } finally { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
    # acceptance from known responses: drafted (3/6) and not drafted (no draft fields: 0/0, rate null)
    & $check 'Acceptance on known responses' {
        $j1 = '{"content":"x","timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":8,"draft_n":6,"draft_n_accepted":3}}'
        $j0 = '{"content":"x","timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":8}}'
        $a1 = Acceptance (Parse-Completion ([pscustomobject]@{ Status = 200; Body = ($j1 | ConvertFrom-Json); Raw = $j1 }))
        $a0 = Acceptance (Parse-Completion ([pscustomobject]@{ Status = 200; Body = ($j0 | ConvertFrom-Json); Raw = $j0 }))
        if (($a1.accepted -ne 3) -or ($a1.generated -ne 6) -or ($a1.rate -ne 0.5)) { throw "drafted: got $($a1.accepted)/$($a1.generated) rate $($a1.rate), expected 3/6 0.5" }
        if (($a0.accepted -ne 0) -or ($a0.generated -ne 0) -or ($null -ne $a0.rate)) { throw "not drafted: got $($a0.accepted)/$($a0.generated) rate $($a0.rate), expected 0/0 null" }
        '3/6 rate 0.5; undrafted 0/0 rate null'
    }
    # the F11 restore shape (what b29a940c+ answers) parses, and the same body without the checkpoints status throws
    & $check 'Parse-Restore on the F11 shape' {
        $j = '{"n_restored":4096,"n_read":1,"timings":{"restore_ms":1},"stateos":{"checkpoints":"restored","checkpoints_restored":3}}'
        $r = Parse-Restore ([pscustomobject]@{ Status = 200; Body = ($j | ConvertFrom-Json); Raw = $j }) 'dry'
        if (($r.checkpoints -ne 'restored') -or ($r.checkpoints_restored -ne 3)) { throw "got '$($r.checkpoints)' / $($r.checkpoints_restored)" }
        $j2 = '{"n_restored":4096,"n_read":1,"timings":{"restore_ms":1},"stateos":{"checkpoints_restored":3}}'
        $msg = $null
        try { [void] (Parse-Restore ([pscustomobject]@{ Status = 200; Body = ($j2 | ConvertFrom-Json); Raw = $j2 }) 'dry') } catch { $msg = $_.Exception.Message }
        if (($null -eq $msg) -or ($msg -notlike "*'stateos.checkpoints'*")) { throw "expected a throw naming 'stateos.checkpoints', got: $msg" }
        "checkpoints=restored/3; without the status: throws"
    }
    # the identity verdict rule on every branch (GPU-VERIFY.md "Pass criteria")
    & $check 'identity verdict rule' {
        # mechanism ok, warm, round 1, round 2, prompt_n warm/r1/r2 (|Z| = 18), warm-vs-warm (valid control; $null =
        # none), spec on -> expected
        $cases = @(
            @($false, 'w', 'w', 'w', 18, 18, 18, $null, $false, 'FAIL'),
            @($true, 'w', 'w', 'w', 18, 18, 18, $null, $false, 'PASS'),
            @($true, 'w', 'w', 'w', 4114, 18, 18, $null, $false, 'PASS-IDENTITY / REUSE-INCONCLUSIVE'),   # warm alone re-prefilled
            @($true, 'w', 'w', 'w', 18, 4114, 18, $null, $false, 'FAIL'),     # a restored run re-prefilled, warm reused
            @($true, 'w', 'w', 'w', 18, 18, 12, $null, $false, 'FAIL'),       # warm reused, a restored prompt_n differs
            @($true, 'w', 'w', 'w', 4114, 4114, 4114, $null, $false, 'FAIL'), # everything re-prefilled: the restore unused
            @($true, 'w', 'w', 'w', 4114, 18, 4114, $null, $true, 'FAIL'),    # spec on: same rule
            @($true, 'w', 'a', 'b', 18, 18, 18, $null, $false, 'FAIL'),
            @($true, 'w', 'a', 'a', 18, 18, 18, $null, $false, 'FAIL'),
            @($true, 'w', 'a', 'b', 18, 18, 18, $null, $true, 'UNPROVEN'),
            @($true, 'w', 'a', 'a', 18, 18, 18, $true, $true, 'FAIL'),
            @($true, 'w', 'a', 'b', 18, 18, 18, $false, $true, 'INCONCLUSIVE'),
            @($true, 'w', 'w', 'w', 18, 18, 18, $false, $true, 'PASS'),
            @($false, 'w', 'w', 'w', 18, 18, 18, $false, $true, 'FAIL'))
        foreach ($c in $cases) {
            $got = (Get-IdentityVerdict $c[0] 'mechanism' $c[1] $c[2] $c[3] $c[4] $c[5] $c[6] 18 $c[7] $c[8]).verdict
            if ($got -ne $c[9]) { throw "mech=$($c[0]) restored=$($c[2])/$($c[3]) prompt_n=$($c[4])/$($c[5])/$($c[6]) warm-vs-warm=$($c[7]) spec_on=$($c[8]): got '$got', expected '$($c[9])'" }
        }
        "$($cases.Count) cases"
    }
    & $check 'identity mechanism rule' {
        $save = [pscustomobject]@{ n_saved = 4096; checkpoints_saved = 3 }
        $rsOk = [pscustomobject]@{ n_restored = 4096; checkpoints = 'restored'; checkpoints_restored = 3 }
        $out = [pscustomobject]@{ content = 'x'; predicted_n = 64 }
        $bad = @(
            @{ n = 'zero checkpoints'; s = [pscustomobject]@{ n_saved = 4096; checkpoints_saved = 0 }; r = [pscustomobject]@{ n_restored = 4096; checkpoints = 'absent'; checkpoints_restored = 0 }; o = $out },
            @{ n = 'n_saved short'; s = [pscustomobject]@{ n_saved = 4000; checkpoints_saved = 3 }; r = $rsOk; o = $out },
            @{ n = 'n_restored short'; s = $save; r = [pscustomobject]@{ n_restored = 10; checkpoints = 'restored'; checkpoints_restored = 3 }; o = $out },
            @{ n = 'status skipped'; s = $save; r = [pscustomobject]@{ n_restored = 4096; checkpoints = 'skipped: over budget'; checkpoints_restored = 3 }; o = $out },
            @{ n = 'empty output'; s = $save; r = $rsOk; o = [pscustomobject]@{ content = ''; predicted_n = 0 } })
        if (-not (Test-IdentityMechanism 4096 $save @($rsOk, $rsOk) @($out, $out, $out)).ok) { throw 'a good leg was refused' }
        foreach ($b in $bad) { if ((Test-IdentityMechanism 4096 $b.s @($b.r, $b.r) @($b.o, $b.o, $b.o)).ok) { throw "'$($b.n)' passed" } }
        # Leg B: companion saved + loaded required
        $saveC = [pscustomobject]@{ n_saved = 4096; checkpoints_saved = 3; companion = 'saved' }
        $rsC = [pscustomobject]@{ n_restored = 4096; checkpoints = 'restored'; checkpoints_restored = 3; companion = 'loaded' }
        $rsSkip = [pscustomobject]@{ n_restored = 4096; checkpoints = 'restored'; checkpoints_restored = 3; companion = "skipped: companion field 'companion_kv_geometry' differs" }
        if (-not (Test-IdentityMechanism 4096 $saveC @($rsC, $rsC) @($out, $out, $out) -RequireCompanion).ok) { throw 'a good spec-on leg was refused' }
        if ((Test-IdentityMechanism 4096 $saveC @($rsC, $rsSkip) @($out, $out, $out) -RequireCompanion).ok) { throw 'a skipped companion passed' }
        if ((Test-IdentityMechanism 4096 $save @($rsC, $rsC) @($out, $out, $out) -RequireCompanion).ok) { throw 'a save without companion passed' }
        if (-not (Test-IdentityMechanism 4096 $save @($rsOk, $rsOk) @($out, $out, $out)).ok) { throw 'spec off must not require a companion' }
        "good leg ok; $($bad.Count) bad legs refused; spec on: companion loaded ok, skipped / not saved refused"
    }
    & $check 'attestation parsers + completion timeouts' {
        if ((Get-EmbeddedCommit "int LLAMA_BUILD_NUMBER = 4975;`nchar const *LLAMA_COMMIT = `"bb0b30ea`";`n") -ne 'bb0b30ea') { throw 'build-info parse' }
        if ($null -ne (Get-EmbeddedCommit 'char const *LLAMA_COMMIT = "unknown";')) { throw "build-info 'unknown' must not attest" }
        if ((Get-VersionCommit "version: 4975 (bb0b30ea)`nbuilt with MSVC 19.44.35228.0 for `n") -ne 'bb0b30ea') { throw '--version parse' }
        if ($null -ne (Get-VersionCommit 'garbage')) { throw '--version garbage parsed' }
        $tb = Join-Path ([System.IO.Path]::GetTempPath()) ('gpu-verify-dry-' + [guid]::NewGuid().ToString('N') + '.bin')
        try {
            [System.IO.File]::WriteAllBytes($tb, [byte[]] (0x5c, 0x27, 0x61, 0x62, 0x63, 0x64, 0x31, 0x32, 0x33, 0x34, 0, 0, 0x4d))   # \'abcd1234\0\0M
            if (-not (Test-BytesContain $tb 'abcd1234') -or (Test-BytesContain $tb 'deadbeef') -or (Test-BytesContain $tb 'abcd123')) { throw 'byte search' }
        } finally { Remove-Item -LiteralPath $tb -Force -ErrorAction SilentlyContinue }
        $t4 = Get-CompletionTimeout 19 64; $t32 = Get-CompletionTimeout 32768 1; $t190 = Get-CompletionTimeout 190000 1; $tq = Get-CompletionTimeout 2365 8
        if (($t4 -gt 300) -or ($tq -gt 300) -or ($t32 -gt 600) -or ($t190 -lt 900) -or ($t190 -gt 3600)) { throw "timeouts 4K-cont=$t4 Q=$tq 32K=$t32 190K=$t190" }
        "build-info bb0b30ea, 'unknown' refused; --version bb0b30ea; timeouts: continuation $t4 s, Q $tq s, 32K prefill $t32 s, 190K prefill $t190 s"
    }
    # /list lookup on the real response shape (entries as the F11 server answered 2026-09-24, specoff.log): a match
    # among several entries (the 5.1 PSCustomObject .Count case that recorded entry:null), a one-entry body, a
    # missing name, a wrong digest, an unredacted prompt
    & $check '/list lookup on the recorded response shape' {
        $sha = '40c37ffae3bce716e9e05e03dc0b898543e98f68b0b970755cf37ed6fcc7ec4a'
        $st = { param($n, $c, $p) '{"filename":"' + $n + '","filesize":538354121,"mtime":"2026-09-24 05:53:40","token_count":' + $c + ',"format":"stateos-v1","prompt":' + $p + ',"stateos":{"token_sha256":"' + $sha + '","prompt_redacted":true}}' }
        $legacy = '{"filename":"legacy-fake.state","filesize":16,"mtime":"2026-09-24 05:55:12","token_count":0,"format":"llama-seq","prompt":""}'
        $mk = { param($j) $o = $null; try { $o = $j | ConvertFrom-Json } catch {}; [pscustomobject]@{ Status = 200; Body = $o; Raw = $j } }
        $many = & $mk ('[' + (& $st 'hard-rope.state' 4096 'null') + ',' + (& $st 'id32k.state' 32768 'null') + ',' + (& $st 'id4k.state' 4096 'null') + ',' + $legacy + ']')
        $one = & $mk ('[' + (& $st 'id4k.state' 4096 'null') + ']')
        $e1 = Find-ListEntry $many 'id4k.state'; $e2 = Find-ListEntry $one 'id4k.state'
        if (($null -eq $e1) -or -not (Test-ListEntry $e1 $sha 4096)) { throw 'not found / not passed among several entries' }
        if (($null -eq $e2) -or -not (Test-ListEntry $e2 $sha 4096)) { throw 'not found / not passed in a one-entry body' }
        if ($null -ne (Find-ListEntry $many 'missing.state')) { throw 'a missing name was found' }
        if ($null -ne (Find-ListEntry $many 'ID4K.STATE')) { throw 'the lookup must be exact' }
        if (Test-ListEntry $e1 ('0' * 64) 4096) { throw 'a wrong digest passed' }
        $leak = Find-ListEntry (& $mk ('[' + (& $st 'id4k.state' 4096 '"Record 000000: ..."') + ']')) 'id4k.state'
        if (Test-ListEntry $leak $sha 4096) { throw 'an unredacted prompt passed' }
        if ($null -ne (Find-ListEntry ([pscustomobject]@{ Status = 500; Body = $null; Raw = '' }) 'id4k.state')) { throw 'a 500 was searched' }
        'found among 4 and in a 1-entry body; missing / case-changed name, wrong digest, unredacted prompt, 500 all refused'
    }
    & $check 'First-Divergence' {
        if ((First-Divergence 'abc' 'abd') -ne 2 -or (First-Divergence 'abc' 'abc') -ne -1 -or (First-Divergence 'ab' 'abc') -ne 2 -or (First-Divergence @(1, 2, 3) @(1, 5)) -ne 1) { throw 'wrong index' }
        'abc/abd=2 equal=-1 prefix=2 tokens=1'
    }
    # binary identity of this worktree's build (read-only: hash, git, mtimes; nothing refused in a dry run)
    & $check 'binary identity (this worktree)' {
        $b = Get-BinaryIdentity
        $arts = @($b.artifacts.Keys | ForEach-Object { "$_ $(if ($b.artifacts[$_].sha256) { $b.artifacts[$_].sha256.Substring(0, 12) } else { 'NO-HASH' }) (mtime $($b.artifacts[$_].mtime), newest input $($b.artifacts[$_].newest_input_mtime))" }) -join '; '
        "embedded=$($b.embedded_commit) HEAD=$($b.head) ok=$($b.ok)$(if ($b.reason) { " [$($b.reason)]" }); $arts"
    }

    # the verdict rule on the recorded identity legs, and each leg's draft acceptance read back from the recorded
    # /completion responses (the legs ran in results.json order: warm, round 1, round 2, [control] in each)
    $rj = Join-Path $dir 'results.json'
    if (-not $DryRunLog -and (Test-Path -LiteralPath $rj)) {
        $j = Get-Content -LiteralPath $rj -Raw -Encoding utf8 | ConvertFrom-Json
        foreach ($legName in 'legA', 'legB') {
            $leg = $j.$legName
            if ($null -eq $leg) { continue }
            $comps = $completionsByLog[$(if ($legName -eq 'legA') { 'specoff.log' } else { 'specon.log' })]
            $pos = @{ cursor = 0 }                             # a reference: the check scriptblocks run in a child scope
            foreach ($p in $leg.PSObject.Properties) {
                $v = $p.Value
                if (($null -eq $v) -or -not (@($v.PSObject.Properties.Name) -contains 'restored') -or -not (@($v.PSObject.Properties.Name) -contains 'warm')) { continue }
                $name = "$legName.$($p.Name)"
                & $check "verdict $name" {
                    $rs = @($v.restored)
                    if ($rs.Count -ne 2) { throw "$($rs.Count) restore rounds recorded, expected 2" }
                    # restores normalised: an F11 recording has checkpoints/checkpoints_restored at the top; a pre-F11 one
                    # (7c77724b) only stateos.checkpoints_restored and no status (mechanism on the counts only)
                    $ckNote = ''
                    $norm = foreach ($x in $rs) {
                        $top = @($x.restore.PSObject.Properties.Name) -contains 'checkpoints'
                        if (-not $top) { $ckNote = ' (ckpt status: pre-F11 recording, counts only)' }
                        [pscustomobject]@{ n_restored = $x.restore.n_restored; checkpoints = $(if ($top) { $x.restore.checkpoints } else { $null })
                                           checkpoints_restored = $(if (@($x.restore.PSObject.Properties.Name) -contains 'checkpoints_restored') { $x.restore.checkpoints_restored } else { $x.restore.stateos.checkpoints_restored })
                                           companion = $(if (@($x.restore.PSObject.Properties.Name) -contains 'companion') { $x.restore.companion } else { $x.restore.stateos.companion }) }
                    }
                    $w = $v.warm
                    $specOn = if (@($v.PSObject.Properties.Name) -contains 'spec_on') { [bool] $v.spec_on } else { $legName -eq 'legB' }
                    $mech = Test-IdentityMechanism ([int] $v.n_prompt) $v.save @($norm) @($w, $rs[0].out, $rs[1].out) -StatusUnknown:([bool] $ckNote) -RequireCompanion:$specOn
                    $hasCtl = @($v.PSObject.Properties.Name) -contains 'warm_control'
                    $wvw = if ($hasCtl -and (@($v.warm_control.PSObject.Properties.Name) -contains 'valid') -and [bool] $v.warm_control.valid) { [bool] $v.warm_control.agrees_with_warm } else { $null }
                    $vd = Get-IdentityVerdict $mech.ok $mech.reason ([string] $w.content) ([string] $rs[0].out.content) ([string] $rs[1].out.content) ([int] $w.prompt_n) ([int] $rs[0].out.prompt_n) ([int] $rs[1].out.prompt_n) ([int] $v.n_suffix) $wvw $specOn
                    $dv = "first divergence (chars) r1=$(First-Divergence ([string] $w.content) ([string] $rs[0].out.content)) r2=$(First-Divergence ([string] $w.content) ([string] $rs[1].out.content))"
                    # acceptance: the next recorded completions whose text and length match warm, round 1, round 2
                    $acc = @()
                    if ($comps) {
                        $targets = @($w, $rs[0].out, $rs[1].out)
                        if ($hasCtl) { $targets += $v.warm_control.out }
                        foreach ($t in $targets) {
                            $hit = $null
                            for ($i = $pos.cursor; $i -lt $comps.Count; $i++) {
                                if (($comps[$i].content -ceq [string] $t.content) -and ($comps[$i].predicted_n -eq [int] $t.predicted_n) -and ($comps[$i].prompt_n -eq [int] $t.prompt_n)) { $hit = $comps[$i]; $pos.cursor = $i + 1; break }
                            }
                            $acc += $(if ($hit) { $a = Acceptance $hit; "$($a.accepted)/$($a.generated)" } else { 'not found' })
                        }
                    }
                    "recorded=$($v.verdict) now=$($vd.verdict)$(if ($vd.reason) { " [$($vd.reason)]" })$ckNote; $dv; draft acceptance warm/r1/r2$(if ($hasCtl) { '/control' }) = $($acc -join ' ')"
                }
            }
        }
    } elseif (-not $DryRunLog) { Write-Host "  skip  no recorded $rj" }

    # the header readers on a recorded state file (read-only)
    $st = Join-Path (Join-Path $dir 'slots') 'id4k.state'
    if (Test-Path -LiteralPath $st) {
        & $check 'id4k.state header + MAIN section' {
            $fp = Need-HeaderValue $st 'model_fingerprint_v2'; [void] (Need-HeaderValue $st 'rope'); $mm = Find-Section $st 'MAIN'
            $em = Get-HeaderValue $st 'effective_model'
            "fingerprint=$($fp.Substring(0, 12))... MAIN at $($mm.Offset), $($mm.Size) B; effective_model=$(if ($null -eq $em) { '<absent: a pre-F11 file, which F11 refuses>' } else { $em })"
        }
    } else { Write-Host "  skip  no recorded $st" }

    Write-Host "dry run: $($script:dryFails) failure(s); $($script:dryPreF11) restore(s) recorded in the pre-F11 shape (skipped)"
    return [int] ($script:dryFails -ne 0)
}
if ($DryRun) {
    $rc = 1
    try { $rc = Invoke-DryRun } catch { Write-Host "dry run ERROR: $($_.Exception.Message)" }
    exit $rc
}

# ---- preflight ------------------------------------------------------------------------------------
Add-Content -LiteralPath $VerdictTxt -Value "==== $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) lane-1 GPU verify ====" -Encoding utf8
foreach ($f in @($PatchedExe, $Model, $Standing)) { if (-not (Test-Path -LiteralPath $f)) { throw "missing: $f" } }
$free = (Get-PSDrive -Name ($Root.Substring(0, 1))).Free
Log ("free space on {0}: {1:N1} GB" -f $Root.Substring(0, 1), ($free / 1GB))
if ($free -lt 25GB) { throw 'need >= 25 GB free for the 190K state files' }
# refuse before production is touched: the exe must be the build of this worktree's clean HEAD
Assert-Binary 'preflight'
Add-Content -LiteralPath (Join-Path $Root 'launch-args.txt') -Encoding utf8 -Value "==== $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) attempt: exe sha256 $($Results.binary.preflight.sha256), HEAD $($Results.binary.preflight.head) ===="

$proc = $null
$prodOk = $false
try {
    # ---- take the slot (inside the protected block: whatever fails from here on, finally relaunches production) ----
    foreach ($k in $ServerEnv.Keys) { Set-Item -Path "Env:$k" -Value $ServerEnv[$k] }
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 4
    Log "gpu after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    # second attestation, with production already down: the exe itself prints the commit it was built from
    Assert-VersionOutput

    # ================= Leg A: speculation OFF (acceptance) =================
    # F11 (842e16f6): at startup with --slot-save-path the server removes *.stateos.tmp files older than 1 h (only that
    # suffix, regular files). Plant a 2-hour-old one and a fresh one before the start.
    $staleTmp = Join-Path $SlotDir 'stale-cleanup.stateos.tmp'; $freshTmp = Join-Path $SlotDir 'fresh-keep.stateos.tmp'
    foreach ($f in $staleTmp, $freshTmp) { [System.IO.File]::WriteAllBytes($f, [byte[]] (1, 2, 3, 4)) }
    (Get-Item -LiteralPath $staleTmp).LastWriteTime = (Get-Date).AddHours(-2)
    $proc = Start-TestServer 'specoff' @()
    $logA = Join-Path $Root 'specoff.log'
    $tmpOk = (-not (Test-Path -LiteralPath $staleTmp)) -and (Test-Path -LiteralPath $freshTmp)
    $Results.legA.tmp_cleanup = [ordered]@{ stale_removed = (-not (Test-Path -LiteralPath $staleTmp)); fresh_kept = (Test-Path -LiteralPath $freshTmp); pass = $tmpOk }
    Log "startup *.stateos.tmp cleanup: stale (2 h) removed=$($Results.legA.tmp_cleanup.stale_removed) fresh kept=$($Results.legA.tmp_cleanup.fresh_kept) pass=$tmpOk"
    Remove-Item -LiteralPath $staleTmp, $freshTmp -Force -ErrorAction SilentlyContinue
    $props = Api 'GET' '/props' $null
    $propsOk = ($props.Status -eq 200) -and ((Field $props 'stateos.version') -is [ValueType]) -and ((Field $props 'stateos.version') -ge 1) -and
               ((Field $props 'stateos.keyed_header') -eq $true) -and ((Field $props 'stateos.companion') -eq $false)
    $Results.legA.props_stateos = [ordered]@{ value = (Field $props 'stateos'); pass = $propsOk }
    Log "GET /props stateos (spec off, expect companion=false): $((Field $props 'stateos') | ConvertTo-Json -Compress) pass=$propsOk"
    $text = New-SyntheticText 9000 7
    $all = Get-Tokens $text
    if ($all.Length -lt 200000) { $all = Concat $all (Get-Tokens (New-SyntheticText 9000 11)) }
    $Z = Get-Tokens "`n`nIn one sentence, which record has the largest value, and what is its tag?"
    $Q = Get-Tokens (New-SyntheticText 80 99)
    Log "synthetic tokens: $($all.Length); suffix Z=$($Z.Length); pollution Q=$($Q.Length)"

    $Results.legA.identity_4k = Test-Identity 'id4k' (Head $all 4096) $Z $Q 64 -LogPath $logA
    Save-Results
    # Leg A kill: anything but PASS* stops the run (spec off has no nondeterminism allowance)
    if ([string] $Results.legA.identity_4k.verdict -notlike 'PASS*') { throw "KILL: 4K identity leg $($Results.legA.identity_4k.verdict) ($($Results.legA.identity_4k.verdict_reason)); stopping before 32K" }
    $Results.legA.identity_32k = Test-Identity 'id32k' (Head $all 32768) $Z $Q 64 -LogPath $logA
    Save-Results
    if ([string] $Results.legA.identity_32k.verdict -notlike 'PASS*') { throw "KILL: 32K identity leg $($Results.legA.identity_32k.verdict) ($($Results.legA.identity_32k.verdict_reason)); stopping before the refusals" }

    # --- refusals: establish a known slot state (the 4K S0), then every refusal must leave it in place
    $good = Join-Path $SlotDir 'id4k.state'
    $rs = Slot 'restore' 'id4k.state'
    if ($rs.Status -ne 200) { throw "baseline restore failed: $($rs.Status)" }
    # F11 effective_model (report-only value): production has no LoRA, control vector or --override-kv, so "none"
    $effModel = Get-HeaderValue $good 'effective_model'
    $Results.legA.effective_model_header = [ordered]@{ value = $effModel; is_none = ($effModel -eq 'none') }
    Log "id4k.state effective_model = '$effModel' (expect 'none' under the production flags)"

    # soft field: warn and proceed
    New-TamperedCopy $good (Join-Path $SlotDir 'soft-build.state') { param($t) Set-HeaderValue $t 'build' '1-softfieldtest' }
    $soft = Slot 'restore' 'soft-build.state'
    $softWarn = Field $soft 'stateos.warnings'
    $softOk = ($soft.Status -eq 200) -and ($softWarn -isnot [string]) -and (@($softWarn | Where-Object { $_.field -eq 'build' }).Count -eq 1)
    $Results.legA.soft_build = [ordered]@{ status = $soft.Status; warnings = $softWarn; pass = $softOk }
    Log "soft field 'build': status=$($soft.Status) pass=$softOk"

    $hard = [ordered]@{
        model_fingerprint_v2 = 'ffff' + (Need-HeaderValue $good 'model_fingerprint_v2').Substring(4)
        effective_model      = '3' * 64
        n_ctx                = '65536'
        cache_type_k         = 'f16'
        cache_type_v         = 'f16'
        rope                 = (Need-HeaderValue $good 'rope') + ' tampered=1'
        kv_layout_version    = 'stateos-kv/999 llama-seq/4'
        system_prompt_sha256 = '0' * 64
        kv_geometry          = '1' * 64
        n_tokens             = '4097'
        token_sha256         = '2' * 64
    }
    $ref = [ordered]@{}
    $first = $true
    foreach ($k in $hard.Keys) {
        $v = $hard[$k]
        New-TamperedCopy $good (Join-Path $SlotDir "hard-$k.state") ([scriptblock]::Create("param(`$t) Set-HeaderValue `$t '$k' '$v'"))
        $r = Slot 'restore' "hard-$k.state"
        $ok = ($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq $k) -and ((Field $r 'error.slot_untouched') -eq $true)
        $ref[$k] = [ordered]@{ status = $r.Status; refused_field = (Field $r 'error.refused_field'); message = (Field $r 'error.message'); pass = $ok }
        Log "hard field '$k': status=$($r.Status) refused_field=$((Field $r 'error.refused_field')) pass=$ok"
        if ($first -and -not $ok) { throw "KILL: the first hard-field refusal did not answer 409 naming '$k': $(Short $r.Raw)" }
        $first = $false
    }
    New-TamperedCopy $good (Join-Path $SlotDir 'hard-unknown.state') { param($t) $t + "H future_field=x`n" }
    $r = Slot 'restore' 'hard-unknown.state'
    $ref['<unknown hard field>'] = [ordered]@{ status = $r.Status; refused_field = (Field $r 'error.refused_field'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'future_field') -and ((Field $r 'error.slot_untouched') -eq $true)) }

    [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'legacy-fake.state'), [byte[]] (0x71,0x73,0x67,0x67, 4,0,0,0, 0,0,0,0, 0,0,0,0))
    $r = Slot 'restore' 'legacy-fake.state'
    $ref['<legacy fake>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_legacy_unkeyed') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    if (Test-Path -LiteralPath $Lane0Slot) {
        $src = [System.IO.File]::OpenRead($Lane0Slot)
        try { $buf = New-Object byte[] 65536; $n = $src.Read($buf, 0, 65536) } finally { $src.Dispose() }
        if ($n -le 0) { throw "lane-0 legacy file is empty: $Lane0Slot" }
        [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'legacy-real.state'), [byte[]] $buf[0..($n - 1)])
        $r = Slot 'restore' 'legacy-real.state'
        $ref['<legacy real, lane-0 file head>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_legacy_unkeyed') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    } else {
        # recorded, not dropped: a missing input is a check that did not run, which fails the expectation
        $ref['<legacy real, lane-0 file head>'] = [ordered]@{ skipped = "missing input $Lane0Slot"; pass = $false }
    }
    $trunc = Join-Path $SlotDir 'truncated.state'
    $in = [System.IO.File]::OpenRead($good)
    try {
        $out = [System.IO.File]::Create($trunc)
        try {
            $buf = New-Object byte[] (1MB); $left = $in.Length - 100
            while ($left -gt 0) { $k = $in.Read($buf, 0, [int] [math]::Min($buf.Length, $left)); if ($k -le 0) { break }; $out.Write($buf, 0, $k); $left -= $k }
        } finally { $out.Dispose() }
    } finally { $in.Dispose() }
    $r = Slot 'restore' 'truncated.state'
    $ref['<truncated>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_corrupt') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'junk.state'), [byte[]] (0x4A,0x55,0x4E,0x4B, 1,2,3,4))
    $r = Slot 'restore' 'junk.state'
    $ref['<unrecognized>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'format') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    $r = Slot 'restore' 'does-not-exist.state'
    $ref['<missing>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_missing') -and ((Field $r 'error.slot_untouched') -eq $true)) }

    # F11 refusals. effective_model absent (what every 7c77724b file looks like): fail closed, naming the field.
    New-TamperedCopy $good (Join-Path $SlotDir 'no-effective-model.state') { param($t) (($t -split "`n") | Where-Object { $_ -notmatch '^[HSI] effective_model=' }) -join "`n" }
    $r = Slot 'restore' 'no-effective-model.state'
    $ref['<effective_model absent>'] = [ordered]@{ status = $r.Status; refused_field = (Field $r 'error.refused_field'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'effective_model') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    if (Test-Path -LiteralPath $Lane1Slot) {                   # the real thing: tonight's acceptance-binary 4K file
        Copy-Item -LiteralPath $Lane1Slot -Destination (Join-Path $SlotDir 'lane1-7c77724b.state') -Force
        $r = Slot 'restore' 'lane1-7c77724b.state'
        $ref['<7c77724b file>'] = [ordered]@{ status = $r.Status; refused_field = (Field $r 'error.refused_field'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'effective_model') -and ((Field $r 'error.slot_untouched') -eq $true)) }
        Remove-Item -LiteralPath (Join-Path $SlotDir 'lane1-7c77724b.state') -Force -ErrorAction SilentlyContinue
    } else {
        $ref['<7c77724b file>'] = [ordered]@{ skipped = "missing input $Lane1Slot"; pass = $false }
    }
    # a path that exists but is not a regular file: state_unreadable, not state_missing (a3ff750d)
    $dirState = Join-Path $SlotDir 'dir-not-file.state'
    New-Item -ItemType Directory -Force -Path $dirState | Out-Null
    try {
        $r = Slot 'restore' 'dir-not-file.state'
        $ref['<unreadable: directory>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_unreadable') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    } finally { Remove-Item -LiteralPath $dirState -Force -ErrorAction SilentlyContinue }
    # the in-progress-save suffix is reserved (fea27876, 91579f16), case-insensitively, for save, restore and rename
    $r = Slot 'save' 'reserved.stateos.tmp'
    $ref['<reserved name: save>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_name_reserved')) }
    $r = Slot 'restore' 'RESERVED.STATEOS.TMP'
    $ref['<reserved name: restore>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_name_reserved')) }
    $r = Api 'POST' '/rename_prompt' '{"old_filename":"junk.state","new_filename":"renamed.stateos.tmp"}'
    $ref['<reserved name: rename>'] = [ordered]@{ status = $r.Status; type = (Field $r 'type'); pass = (($r.Status -eq 409) -and ((Field $r 'type') -eq 'state_name_reserved') -and (Test-Path -LiteralPath (Join-Path $SlotDir 'junk.state'))) }
    $Results.legA.refusals = $ref

    # every refusal above must have left the restored 4K S0 in the slot: the continuation is the in-memory one
    $after = Complete (Concat (Head $all 4096) $Z) 64 ($Z.Length + 1)
    $untouched = ($after.content -ceq $Results.legA.identity_4k.warm.content) -and ($after.prompt_n -eq $Results.legA.identity_4k.warm.prompt_n)
    $Results.legA.slot_untouched_after_refusals = [ordered]@{ prompt_n = $after.prompt_n; same_output = ($after.content -ceq $Results.legA.identity_4k.warm.content); pass = $untouched }
    $nRef = @($ref.Values | Where-Object { $_.pass }).Count
    Log "refusals: $nRef/$($ref.Count) pass; slot untouched after refusals: $untouched"

    # F11 /list (4e1f27ca): a State-OS entry names the state (format, token count, token digest), never its text
    $ls = Api 'GET' '/list' $null
    $e0 = Find-ListEntry $ls 'id4k.state'
    $e0r = if ($e0) { [pscustomobject]@{ Status = 200; Body = $e0 } } else { $null }
    $listOk = Test-ListEntry $e0 $Results.legA.identity_4k.save.token_sha256 4096
    $Results.legA.list_redacted = [ordered]@{ status = $ls.Status; raw = $ls.Raw; entry = $e0; pass = $listOk }
    Log "GET /list id4k.state: status=$($ls.Status) format=$(Field $e0r 'format') prompt_redacted=$(Field $e0r 'stateos.prompt_redacted') token_count=$(Field $e0r 'token_count') pass=$listOk"
    Save-Results

    # --- destructive paths (review M3). Oracle for "correct re-prefill": the 4K cold output (full prefill of P+Z, the
    # same computation). Speculation is off here, so there is no nondeterminism allowance: a mechanism miss or an output
    # difference is FAIL.
    $PZ4 = Concat (Head $all 4096) $Z
    $cold4 = $Results.legA.identity_4k.cold
    function Destructive-Verdict([bool] $Mechanism, [bool] $SameOutput) {
        if ($Mechanism -and $SameOutput) { 'PASS' } else { 'FAIL' }
    }

    # (a) empty-slot round trip: a state saved right after erase restores as an erase (no loader call), server stays up
    [void] (Slot 'erase' $null)
    $se = Slot 'save' 'empty.state'
    [void] (Complete $Q 8)                                   # something to erase
    $poA = Ple-Offsets $logA
    $re = Slot 'restore' 'empty.state'
    $alive = $false; try { $alive = ((Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 5).status -eq 'ok') } catch {}
    $ae = Complete $PZ4 64
    $pleA = Ple-Counts $poA
    Log "empty-slot round trip: [ple-hist] resets at pos>0 = $($pleA.resets_pos_gt0) (expect 0)"
    $emptyMech = ($se.Status -eq 200) -and ((Field $se 'n_saved') -is [ValueType]) -and ((Field $se 'n_saved') -eq 0) -and ($re.Status -eq 200) -and ((Field $re 'stateos.empty') -eq $true) -and $alive -and
                 ($ae.prompt_n -eq $PZ4.Length) -and ($pleA.resets_pos_gt0 -eq 0)
    $emptySame = ($ae.content -ceq $cold4.content)
    $emptyVerdict = Destructive-Verdict $emptyMech $emptySame
    $Results.legA.empty_roundtrip = [ordered]@{ save_status = $se.Status; restore_status = $re.Status; restore = (Field $re 'stateos'); alive = $alive; next = $ae; ple = $pleA; mechanism = $emptyMech; same_as_cold = $emptySame; verdict = $emptyVerdict; pass = ($emptyVerdict -eq 'PASS') }
    Log "empty-slot round trip: save=$($se.Status) restore=$($re.Status) empty=$((Field $re 'stateos.empty')) alive=$alive next prompt_n=$($ae.prompt_n)/$($PZ4.Length) same-as-cold=$emptySame verdict=$emptyVerdict"

    # (b) MAIN tamper: cell_count + 1 inside a well-formed container -> the loader fails after its seq_rm -> 500,
    #     slot_untouched:false, server alive, the next request re-prefills and is correct
    $bad = Join-Path $SlotDir 'main-tamper.state'
    Copy-Item -LiteralPath $good -Destination $bad -Force
    $m = Find-Section $bad 'MAIN'
    $cc = [BitConverter]::ToUInt32((Read-BytesAt $bad $m.Offset 4), 0)
    Write-BytesAt $bad $m.Offset ([BitConverter]::GetBytes([uint32] ($cc + 1)))
    $rs = Slot 'restore' 'id4k.state'                         # the slot holds S0 before the tamper
    $poB = Ple-Offsets $logA
    $rt = Slot 'restore' 'main-tamper.state'
    $alive = $false; try { $alive = ((Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 5).status -eq 'ok') } catch {}
    $at = Complete $PZ4 64
    $pleB = Ple-Counts $poB
    Log "MAIN tamper: [ple-hist] resets at pos>0 = $($pleB.resets_pos_gt0) (expect 0)"
    $tamperMech = ($rs.Status -eq 200) -and ($rt.Status -eq 500) -and ((Field $rt 'error.slot_untouched') -eq $false) -and $alive -and
                  ($at.prompt_n -eq $PZ4.Length) -and ($pleB.resets_pos_gt0 -eq 0)
    $tamperSame = ($at.content -ceq $cold4.content)
    $tamperVerdict = Destructive-Verdict $tamperMech $tamperSame
    $Results.legA.main_tamper = [ordered]@{ cell_count = $cc; status = $rt.Status; error = (Field $rt 'error'); alive = $alive; next = $at; ple = $pleB; mechanism = $tamperMech; same_as_cold = $tamperSame; verdict = $tamperVerdict; pass = ($tamperVerdict -eq 'PASS') }
    Log "MAIN tamper (cell_count $cc -> $($cc + 1)): status=$($rt.Status) slot_untouched=$((Field $rt 'error.slot_untouched')) alive=$alive next prompt_n=$($at.prompt_n)/$($PZ4.Length) same-as-cold=$tamperSame verdict=$tamperVerdict"
    Save-Results

    # (d) F11 runtime adapter generation (e2b76a3a, c5d66f76), last in Leg A so it cannot touch the legs above. No
    #     adapter file is needed: a bad control-vector id answers 400 and changes nothing (the slot stays saveable); an
    #     empty /control-vectors/apply is a successful apply that disables steering and bumps the generation, so a slot
    #     built before it is refused (409 state_adapters_changed, slot untouched) until it re-prefills from empty; the
    #     stamp stays what it was (no vector applied), so id4k.state still restores. Guarded: a failure is recorded and
    #     the run continues to Leg B.
    try {
        $gen = [ordered]@{}
        $gen.lora_list = (Api 'GET' '/lora-adapters' $null).Raw
        $gen.cvec_list = (Api 'GET' '/control-vectors' $null).Raw
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)
        $bid = Api 'POST' '/control-vectors/apply' '[{"id":99,"scale":1.0}]'
        $s1 = Slot 'save' 'cv-badid.state'
        $gen.bad_id = [ordered]@{ apply_status = $bid.Status; save_status = $s1.Status; pass = (($bid.Status -eq 400) -and ($s1.Status -eq 200)) }
        $emp = Api 'POST' '/control-vectors/apply' '[]'
        $s2 = Slot 'save' 'cv-stale.state'
        $gen.empty_apply = [ordered]@{ apply_status = $emp.Status; save_status = $s2.Status; type = (Field $s2 'error.type'); slot_untouched = (Field $s2 'error.slot_untouched')
                                       pass = (($emp.Status -eq 200) -and ($s2.Status -eq 409) -and ((Field $s2 'error.type') -eq 'state_adapters_changed') -and ((Field $s2 'error.slot_untouched') -eq $true)) }
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)
        $s3 = Slot 'save' 'cv-fresh.state'
        $freshEff = if ($s3.Status -eq 200) { Get-HeaderValue (Join-Path $SlotDir 'cv-fresh.state') 'effective_model' } else { $null }
        $r4 = Slot 'restore' 'id4k.state'
        $gen.after_reprefill = [ordered]@{ save_status = $s3.Status; effective_model = $freshEff; restore_id4k_status = $r4.Status
                                           pass = (($s3.Status -eq 200) -and ($freshEff -eq $effModel) -and ($r4.Status -eq 200)) }
        $gen.pass = $gen.bad_id.pass -and $gen.empty_apply.pass -and $gen.after_reprefill.pass
        $Results.legA.adapter_generation = $gen
        Log "adapter generation: bad id apply=$($bid.Status) save=$($s1.Status); empty apply=$($emp.Status) save=$($s2.Status) $(Field $s2 'error.type'); after re-prefill save=$($s3.Status) effective_model='$freshEff' restore id4k=$($r4.Status); pass=$($gen.pass)"
    } catch {
        $Results.legA.adapter_generation = [ordered]@{ pass = $false; reason = $_.Exception.Message }
        Log "adapter generation: FAIL ($($_.Exception.Message)); continuing"
    }
    Save-Results
    Stop-TestServer $proc; $proc = $null

    # ================= Leg B: speculation ON = production flags (report-only) =================
    if (-not $SkipSpecOn) {
        $proc = Start-TestServer 'specon' $SpecArgs
        $logB = Join-Path $Root 'specon.log'
        $props = Api 'GET' '/props' $null
        $propsOk = ($props.Status -eq 200) -and ((Field $props 'stateos.version') -is [ValueType]) -and ((Field $props 'stateos.version') -ge 1) -and
                   ((Field $props 'stateos.keyed_header') -eq $true) -and ((Field $props 'stateos.companion') -eq $true)
        $Results.legB.props_stateos = [ordered]@{ value = (Field $props 'stateos'); pass = $propsOk }
        Log "GET /props stateos (spec on, expect companion=true): $((Field $props 'stateos') | ConvertTo-Json -Compress) pass=$propsOk"
        # the warm-vs-warm control always runs here: it decides whether a spec-on restored != warm is a state defect
        $Results.legB.companion_32k = Test-Identity 'on32k' (Head $all 32768) $Z $Q 128 -NoCold -LogPath $logB -WarmControl 'always' -SpecOn
        # the spec-off 32K file has no COMP section: restore it here to see acceptance without the companion
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)
        $poNc = Ple-Offsets $logB
        $rs = Slot 'restore' 'id32k.state'
        $nc = [ordered]@{ status = $rs.Status; stateos = (Field $rs 'stateos'); error = (Field $rs 'error') }
        if ($rs.Status -eq 200) {
            $nc.out = Complete (Concat (Head $all 32768) $Z) 128 ($Z.Length + 1)
            $nc.acceptance = Acceptance $nc.out
            $nc.ple = Ple-Counts $poNc
            Log "no-companion restore: [ple-hist] resets at pos>0 = $($nc.ple.resets_pos_gt0) (expect 0)"
        }
        $Results.legB.no_companion_32k = $nc
        Log "spec-on 32K: companion=$($Results.legB.companion_32k.restored[0].restore.stateos.companion) acc(warm)=$($Results.legB.companion_32k.warm_acceptance.rate) acc(restored)=$($Results.legB.companion_32k.restored[0].acceptance.rate) acc(no companion)=$($nc.acceptance.rate) status(no companion file)=$($rs.Status)"
        Save-Results

        # (c) companion sub-header tamper (review M3): a length-preserving edit of companion_kv_geometry -> 200 with
        #     the companion skipped, never a refusal
        #     Guarded: a missing COMP section records a FAIL here and never aborts the 190K measurement below.
        $on32 = Join-Path $SlotDir 'on32k.state'
        $ct = Join-Path $SlotDir 'comp-tamper.state'
        $savedComp = [string] $Results.legB.companion_32k.save.companion
        if ($savedComp -ne 'saved') {
            $Results.legB.comp_tamper = [ordered]@{ verdict = 'FAIL'; pass = $false; reason = "on32k.state has no COMP section (save said: '$savedComp')" }
            Log "COMP sub-header tamper: FAIL (no COMP section to tamper; save said '$savedComp')"
        } else {
            try {
                Copy-Item -LiteralPath $on32 -Destination $ct -Force
                $c = Find-Section $ct 'COMP'
                $sublen = [BitConverter]::ToUInt32((Read-BytesAt $ct $c.Offset 4), 0)
                if (($sublen -eq 0) -or ($sublen + 4 -gt $c.Size)) { throw "COMP sub-header length $sublen does not fit the $($c.Size)-byte section" }
                $sub = [System.Text.Encoding]::ASCII.GetString((Read-BytesAt $ct ($c.Offset + 4) ([int] $sublen)))
                $gpos = $sub.IndexOf('companion_kv_geometry=')
                if ($gpos -lt 0) { throw 'companion_kv_geometry not found in the COMP sub-header' }
                $vpos = $gpos + 'companion_kv_geometry='.Length
                if ($vpos -ge $sub.Length) { throw 'companion_kv_geometry has no value in the COMP sub-header' }
                $newc = if ($sub[$vpos] -eq '0') { [byte][char] '1' } else { [byte][char] '0' }
                Write-BytesAt $ct ($c.Offset + 4 + $vpos) ([byte[]] @($newc))
                [void] (Slot 'erase' $null); [void] (Complete $Q 8)
                $rc = Slot 'restore' 'comp-tamper.state'
                $rcComp = [string] (Field $rc 'stateos.companion')
                $compOk = ($rc.Status -eq 200) -and $rcComp.StartsWith("skipped: companion field 'companion_kv_geometry'")
                $Results.legB.comp_tamper = [ordered]@{ status = $rc.Status; companion = $rcComp; verdict = $(if ($compOk) { 'PASS' } else { 'FAIL' }); pass = $compOk }
                Log "COMP sub-header tamper: status=$($rc.Status) companion='$rcComp' pass=$compOk"
            } catch {
                $Results.legB.comp_tamper = [ordered]@{ verdict = 'FAIL'; pass = $false; reason = $_.Exception.Message }
                Log "COMP sub-header tamper: FAIL ($($_.Exception.Message)); continuing to the 190K measurement"
            } finally {
                Remove-Item -LiteralPath $ct -Force -ErrorAction SilentlyContinue
            }
        }
        Save-Results

        if (-not $Skip192K) {
            $Results.legB.bytes_190k = Test-Identity 'on190k' (Head $all 190000) $Z $Q 16 -NoCold -LogPath $logB -WarmControl 'if-differs' -SpecOn
            Save-Results
        }
        Stop-TestServer $proc; $proc = $null
    }

    $invalidates = @()
    foreach ($f in 'specoff.log', 'specoff.err.log', 'specon.log', 'specon.err.log') {
        $p = Join-Path $Root $f
        if (Test-Path -LiteralPath $p) { $invalidates += @(Select-String -LiteralPath $p -Pattern 'MTP invalidate: SLOT_RESTORE' | ForEach-Object { $_.Line }) }
    }
    $Results.restore_invalidate_lines = $invalidates.Count
    Log "MTP invalidate: SLOT_RESTORE lines: $($invalidates.Count)"

    # PLE n-gram history across every identity round (expected: 0 resets at pos > 0, the choke point fired each time)
    $pleRounds = [ordered]@{}
    foreach ($leg in 'legA', 'legB') {
        foreach ($key in @($Results[$leg].Keys)) {
            $v = $Results[$leg][$key]
            if ($v -is [System.Collections.IDictionary] -and $v.Contains('ple_ok')) {
                $pleRounds["$leg.$key"] = [ordered]@{ resets_pos_gt0 = $v.ple_resets_pos_gt0; ok = $v.ple_ok }
            }
        }
    }
    $Results.ple_hist = [ordered]@{ rounds = $pleRounds; all_ok = (@($pleRounds.Values | Where-Object { -not $_.ok }).Count -eq 0) }
    Log "[ple-hist] identity rounds: $(($pleRounds.Keys | ForEach-Object { "$_=$($pleRounds[$_].resets_pos_gt0)" }) -join ' ') all_ok=$($Results.ple_hist.all_ok)"
    $Results.completed = $true
} catch {
    $Results.aborted = $_.Exception.Message
    try { Log "ABORTED: $($_.Exception.Message)" } catch {}
} finally {
    # Production first: every step before the relaunch is wrapped so none can skip it. The test-only PLE switches must
    # not leak into the standing server's inherited environment. Receipts are saved after the relaunch.
    try { if ($proc) { Stop-TestServer $proc } } catch {}
    try { Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue } catch {}
    try { Remove-Item Env:LONGSPEAR_PLE_HIST_REWIND, Env:LONGSPEAR_PLE_HIST_LOG -ErrorAction SilentlyContinue } catch {}
    try { Start-Sleep -Seconds 3 } catch {}
    $launchErr = $null
    try { Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$Standing -WindowStyle Hidden | Out-Null }
    catch { $launchErr = $_.Exception.Message }
    try { Log $(if ($launchErr) { "restore launcher FAILED to start: $launchErr" } else { 'restore launcher started' }) } catch {}
    if (-not $launchErr) {
        for ($i = 0; $i -lt 90 -and -not $prodOk; $i++) { Start-Sleep -Seconds 10; try { $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8099/health' -TimeoutSec 5; $prodOk = ($h.status -eq 'ok') } catch {} }
    }
    try { Log ('production-restored:' + $(if ($prodOk) { '200' } else { 'FAILED' })) } catch {}
    # neither completed nor aborted = the try was left another way (Ctrl+C / host stop): say so in the receipts
    try { if (-not $Results.Contains('completed') -and -not $Results.Contains('aborted')) { $Results.interrupted = $true; Log 'INTERRUPTED (neither completed nor aborted)' } } catch {}
    try { $Results.production_restored = $prodOk; Save-Results } catch { try { Write-Host "Save-Results failed: $($_.Exception.Message)" } catch {} }
    # the 190K files are ~4-5 GB each; keep the small ones as receipts
    try { Get-ChildItem -LiteralPath $SlotDir -Filter 'on190k*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue } catch {}
}

# ---- exit code: 0 only when the run completed, every hard expectation held and production is back ----------------
$hardFail = @()
if ($Results.aborted) { $hardFail += "aborted: $($Results.aborted)" }
elseif (-not $Results.Contains('completed')) { $hardFail += 'not completed' }
if (-not $prodOk) { $hardFail += 'production not restored' }
foreach ($leg in 'legA', 'legB') {
    foreach ($key in @($Results[$leg].Keys)) {
        $v = $Results[$leg][$key]
        if (-not ($v -is [System.Collections.IDictionary])) { continue }
        # Leg A identity legs must PASS*; a Leg B identity leg fails the run only on FAIL (UNPROVEN/INCONCLUSIVE are reported)
        if ($v.Contains('restored') -and $v.Contains('verdict')) {
            if ((($leg -eq 'legA') -and ([string] $v.verdict -notlike 'PASS*')) -or ([string] $v.verdict -eq 'FAIL')) { $hardFail += "$leg.$key verdict $($v.verdict)" }
            continue
        }
        if ($key -eq 'refusals') { foreach ($rk in @($v.Keys)) { if (-not $v[$rk].pass) { $hardFail += "$leg.refusals[$rk]" } }; continue }
        if ($v.Contains('pass') -and -not $v.pass) { $hardFail += "$leg.$key" }
    }
}
if ($Results.Contains('ple_hist') -and -not $Results.ple_hist.all_ok) { $hardFail += 'ple_hist' }
try { $Results.hard_failures = $hardFail; Save-Results } catch {}
Write-Host ("exit: " + $(if ($hardFail.Count) { "1 (" + ($hardFail -join '; ') + ")" } else { '0' }))
exit $(if ($hardFail.Count) { 1 } else { 0 })
