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
#   and a MAIN-payload tamper (500, slot cleared, server alive, correct re-prefill). Auto-stops on the first identity
#   FAIL or on a refusal that is not a 409 naming its field (mechanism kill criteria).
#   F11 behaviours observable on the real model: the startup cleanup of stale *.stateos.tmp files, effective_model in
#   the save header and its refusal (tampered, missing, and a real 7c77724b file), state_unreadable, the reserved
#   .stateos.tmp names (save, restore, /rename_prompt), /list redaction, and the runtime adapter generation (an empty
#   /control-vectors/apply makes the slot unsaveable until it re-prefills; a bad id changes nothing).
# Leg B (spec ON = production flags, report-only): companion section saved/loaded at 32K with a warm-vs-warm control
#   (two warm runs, no restore between them), a companion sub-header tamper (200, companion skipped), draft acceptance
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
foreach ($k in $ServerEnv.Keys) { Set-Item -Path "Env:$k" -Value $ServerEnv[$k] }

if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $Root, $SlotDir | Out-Null }
Add-Type -AssemblyName System.Net.Http
$Http = New-Object System.Net.Http.HttpClient
$Http.Timeout = [TimeSpan]::FromMinutes(60)
$Results = [ordered]@{ started = (Get-Date).ToUniversalTime().ToString('o'); legA = [ordered]@{}; legB = [ordered]@{} }

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

function Api([string] $Method, [string] $Path, [string] $Json) {
    $req = New-Object System.Net.Http.HttpRequestMessage((New-Object System.Net.Http.HttpMethod($Method)), "$Base$Path")
    if ($Json) { $req.Content = New-Object System.Net.Http.StringContent($Json, [System.Text.Encoding]::UTF8, 'application/json') }
    $resp = $Http.SendAsync($req).GetAwaiter().GetResult()
    $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
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
    }
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

function Complete([int[]] $Ids, [int] $NPredict) {
    $json = '{"prompt":[' + ($Ids -join ',') + '],"n_predict":' + $NPredict +
            ',"temperature":0,"top_k":1,"top_p":1,"min_p":0,"seed":1234,"cache_prompt":true,"id_slot":0,"stream":false}'
    Parse-Completion (Api 'POST' '/completion' $json)
}

function Slot([string] $Action, [string] $File) {
    $json = if ($File) { '{"filename":"' + $File + '"}' } else { '{}' }
    Api 'POST' "/slots/0?action=$Action" $json
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

function Start-TestServer([string] $Name, [string[]] $Extra) {
    $log = Join-Path $Root "$Name.log"; $err = Join-Path $Root "$Name.err.log"
    Remove-Item -LiteralPath $log, $err -ErrorAction SilentlyContinue
    # Start-Process joins ArgumentList with spaces and does not quote: paths with spaces carry their own quotes
    $argv = @('-m', ('"' + $Model + '"'), '--host', '127.0.0.1', '--port', "$Port", '--slot-save-path', ('"' + $SlotDir + '"'), '--verbose') + $CommonArgs + $Extra
    Add-Content -LiteralPath (Join-Path $Root 'launch-args.txt') -Value "$Name`: $PatchedExe $($argv -join ' ')" -Encoding utf8
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
#   checkpoints not all restored                         -> FAIL (hard, review F11-2 P3-2), whatever the outputs say
#   both restored outputs == warm                        -> PASS, or PASS-IDENTITY / REUSE-INCONCLUSIVE if warm re-prefilled
#   restored != warm, warm-vs-warm control run (spec on): the two warm runs disagree too -> INCONCLUSIVE (decode
#                                                          nondeterminism); they agree -> FAIL (state-defect signal)
#   restored != warm, no control: the two restored runs disagree with each other -> INCONCLUSIVE (engine run-to-run
#                                                          nondeterminism); they agree -> FAIL
# $WarmVsWarm: $null when the control did not run, else whether the two warm runs produced the same text.
function Get-IdentityVerdict([bool] $CkptOk, [string] $Warm, [string] $Restored1, [string] $Restored2, [bool] $Reuse, $WarmVsWarm) {
    $same = ($Restored1 -ceq $Warm) -and ($Restored2 -ceq $Warm)
    $v = if (-not $CkptOk) { 'FAIL', 'checkpoints not all restored' }
         elseif ($same -and $Reuse) { 'PASS', '' }
         elseif ($same) { 'PASS-IDENTITY / REUSE-INCONCLUSIVE', 'identity held but the warm run re-prefilled' }
         elseif ($null -ne $WarmVsWarm) {
             if ([bool] $WarmVsWarm) { 'FAIL', 'restored != warm while two warm runs (no restore) agree: state-defect signal' }
             else { 'INCONCLUSIVE', 'restored != warm, and two warm runs (no restore) also disagree: decode nondeterminism' }
         }
         elseif (-not ($Restored1 -ceq $Restored2)) { 'INCONCLUSIVE', 'restored != warm, and the two restored runs disagree with each other: engine run-to-run nondeterminism' }
         else { 'FAIL', 'restored != warm, and the two restored runs agree with each other' }
    [pscustomobject]@{ verdict = $v[0]; reason = $v[1] }
}
# every saved checkpoint comes back, whatever the slot held before the restore (review F11 P1: the restores here follow
# a short Q conversation, which a slot-length-dependent bound got wrong)
function Test-CkptOk([int] $Saved, $Restores) {
    foreach ($rs in $Restores) {
        $st = [string] $rs.checkpoints
        $okStatus = if ($Saved -gt 0) { $st -eq 'restored' } else { ($st -eq 'restored') -or ($st -eq 'absent') }
        if (-not $okStatus -or ([int] $rs.checkpoints_restored -ne $Saved)) { return $false }
    }
    return $true
}

# ---- one identity round: in-memory continuation vs restored continuation (x2) vs cold (report-only) ----
# -WarmControl 'always' | 'if-differs': after the restore rounds, rebuild S0 by prefill (erase, P, no restore) and run
# the same warm continuation again; 'if-differs' only when a restored output differs from warm (spend GPU time only
# where the control can change the verdict).
function Test-Identity([string] $Tag, [int[]] $P, [int[]] $Z, [int[]] $Q, [int] $NGen, [switch] $NoCold, [string] $LogPath, [string] $WarmControl) {
    $PZ = Concat $P $Z
    $res = [ordered]@{ n_prompt = $P.Length; n_suffix = $Z.Length }
    [void] (Slot 'erase' $null)
    [void] (Complete $P 1)                                       # in-memory state S0: the slot holds P
    $res.save = Parse-Save (Slot 'save' "$Tag.state") $Tag
    $po = Ple-Offsets $LogPath
    $w = Complete $PZ $NGen                                      # continuation from the in-memory S0
    $res.warm = $w; $res.warm_acceptance = Acceptance $w; $res.warm_ple = Ple-Counts $po

    $runs = @()
    foreach ($k in 1, 2) {
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)      # another conversation occupies the slot
        $po = Ple-Offsets $LogPath                               # from the restore through the continuation
        $rsp = Parse-Restore (Slot 'restore' "$Tag.state") "$Tag round $k"
        $r = Complete $PZ $NGen
        $ple = Ple-Counts $po
        $runs += ,([ordered]@{ restore = $rsp; out = $r; acceptance = (Acceptance $r); ple = $ple })
        Log "$Tag restore round $k`: [ple-hist] resets at pos>0 = $($ple.resets_pos_gt0) (expect 0), server-resume sets = $($ple.sets_server_resume)"
    }
    if ($runs.Count -ne 2) { throw "internal: $Tag has $($runs.Count) restore rounds recorded, expected 2" }
    $res.restored = $runs
    $res.ple_resets_pos_gt0 = [int] $runs[0].ple.resets_pos_gt0 + [int] $runs[1].ple.resets_pos_gt0
    $res.ple_ok = ($res.ple_resets_pos_gt0 -eq 0) -and ($runs[0].ple.sets_server_resume -ge 1) -and ($runs[1].ple.sets_server_resume -ge 1)
    $saved = [int] $res.save.checkpoints_saved
    $res.ckpt_ok = Test-CkptOk $saved @($runs[0].restore, $runs[1].restore)
    Log "$Tag checkpoints: saved=$saved restored=$($runs[0].restore.checkpoints_restored)/$($runs[1].restore.checkpoints_restored) status='$($runs[0].restore.checkpoints)' ok=$($res.ckpt_ok)"
    $same1 = $runs[0].out.content -ceq $w.content
    $same2 = $runs[1].out.content -ceq $w.content

    # warm-vs-warm control (report-only): S0 rebuilt by the same erase + P prefill as the warm run, no restore anywhere,
    # so a difference between the two warm runs is decode nondeterminism (spec-decode draft/verify batch shapes, the
    # server-wide ngram-mod table that grew since the first warm run), not the restored state.
    $wvw = $null
    if (($WarmControl -eq 'always') -or (($WarmControl -eq 'if-differs') -and -not ($same1 -and $same2))) {
        [void] (Slot 'erase' $null)
        [void] (Complete $P 1)
        $w2 = Complete $PZ $NGen
        $wvw = ($w2.content -ceq $w.content)
        $res.warm_control = [ordered]@{ out = $w2; acceptance = (Acceptance $w2); agrees_with_warm = $wvw }
        Log "$Tag warm-vs-warm control (no restore): agrees=$wvw (warm2 prompt_n=$($w2.prompt_n), acc=$($res.warm_control.acceptance.accepted)/$($res.warm_control.acceptance.generated))"
    }
    if (-not $NoCold) {
        [void] (Slot 'erase' $null)
        $res.cold = Complete $PZ $NGen                           # full prefill of P+Z, report-only
    }
    $reuse = ($runs[0].out.prompt_n -eq $w.prompt_n) -and ($runs[1].out.prompt_n -eq $w.prompt_n) -and ($w.prompt_n -le ($Z.Length + 1))
    $res.identity_restored_vs_warm = ($same1 -and $same2)
    $res.no_reprefill = $reuse
    $res.restored_runs_agree = ($runs[0].out.content -ceq $runs[1].out.content)
    if (-not $NoCold) { $res.identity_cold_vs_warm_report_only = ($res.cold.content -ceq $w.content) }
    $vd = Get-IdentityVerdict $res.ckpt_ok $w.content $runs[0].out.content $runs[1].out.content $reuse $wvw
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
    $logs = if ($DryRunLog) { @($DryRunLog) } else { @(@('specoff.log', 'specon.log') | ForEach-Object { Join-Path $dir $_ } | Where-Object { Test-Path -LiteralPath $_ }) }
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
                'slots save'    { & $check "slots save #$($seen[$kind])" { $s = Parse-Save $e 'dry'; "n_saved=$($s.n_saved) n_written=$($s.n_written) checkpoints_saved=$($s.checkpoints_saved)" } }
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
        $cases = @(
            @($false, 'w', 'w', 'w', $true, $null, 'FAIL'),
            @($true, 'w', 'w', 'w', $true, $null, 'PASS'),
            @($true, 'w', 'w', 'w', $false, $null, 'PASS-IDENTITY / REUSE-INCONCLUSIVE'),
            @($true, 'w', 'a', 'b', $true, $null, 'INCONCLUSIVE'),
            @($true, 'w', 'a', 'a', $true, $null, 'FAIL'),
            @($true, 'w', 'a', 'b', $true, $true, 'FAIL'),
            @($true, 'w', 'a', 'a', $true, $false, 'INCONCLUSIVE'),
            @($true, 'w', 'w', 'w', $true, $false, 'PASS'))
        foreach ($c in $cases) {
            $got = (Get-IdentityVerdict $c[0] $c[1] $c[2] $c[3] $c[4] $c[5]).verdict
            if ($got -ne $c[6]) { throw "ckpt=$($c[0]) restored=$($c[2])/$($c[3]) reuse=$($c[4]) warm-vs-warm=$($c[5]): got '$got', expected '$($c[6])'" }
        }
        "$($cases.Count) cases"
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
                    $saved = [int] $v.save.checkpoints_saved
                    $ckNote = ''
                    $ck = if (@($v.PSObject.Properties.Name) -contains 'ckpt_ok') { [bool] $v.ckpt_ok } else {
                        # recorded by a binary before the checkpoints status field: compare the counts only
                        $ckNote = ' (ckpt: counts only)'
                        $ok = $true
                        foreach ($x in $rs) { $n = if (@($x.restore.PSObject.Properties.Name) -contains 'checkpoints_restored') { $x.restore.checkpoints_restored } else { $x.restore.stateos.checkpoints_restored }; if ([int] $n -ne $saved) { $ok = $false } }
                        $ok
                    }
                    $w = $v.warm
                    $reuse = ($rs[0].out.prompt_n -eq $w.prompt_n) -and ($rs[1].out.prompt_n -eq $w.prompt_n) -and ($w.prompt_n -le ([int] $v.n_suffix + 1))
                    $wvw = if (@($v.PSObject.Properties.Name) -contains 'warm_control') { [bool] $v.warm_control.agrees_with_warm } else { $null }
                    $vd = Get-IdentityVerdict $ck ([string] $w.content) ([string] $rs[0].out.content) ([string] $rs[1].out.content) $reuse $wvw
                    # acceptance: the next recorded completions whose text and length match warm, round 1, round 2
                    $acc = @()
                    if ($comps) {
                        $targets = @($w, $rs[0].out, $rs[1].out)
                        if ($null -ne $wvw) { $targets += $v.warm_control.out }
                        foreach ($t in $targets) {
                            $hit = $null
                            for ($i = $pos.cursor; $i -lt $comps.Count; $i++) {
                                if (($comps[$i].content -ceq [string] $t.content) -and ($comps[$i].predicted_n -eq [int] $t.predicted_n) -and ($comps[$i].prompt_n -eq [int] $t.prompt_n)) { $hit = $comps[$i]; $pos.cursor = $i + 1; break }
                            }
                            $acc += $(if ($hit) { $a = Acceptance $hit; "$($a.accepted)/$($a.generated)" } else { 'not found' })
                        }
                    }
                    "recorded=$($v.verdict) now=$($vd.verdict)$(if ($vd.reason) { " [$($vd.reason)]" })$ckNote; draft acceptance warm/r1/r2$(if ($null -ne $wvw) { '/control' }) = $($acc -join ' ')"
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

# ---- take the slot --------------------------------------------------------------------------------
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 4
Log "gpu after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
$proc = $null
try {
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
    if ($Results.legA.identity_4k.verdict -eq 'FAIL') { throw "KILL: 4K identity leg failed ($(if ($Results.legA.identity_4k.fail_reason) { $Results.legA.identity_4k.fail_reason } else { 'restored != in-memory continuation' })); stopping before 32K" }
    $Results.legA.identity_32k = Test-Identity 'id32k' (Head $all 32768) $Z $Q 64 -LogPath $logA
    Save-Results

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
    $ref['<unknown hard field>'] = [ordered]@{ status = $r.Status; refused_field = (Field $r 'error.refused_field'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'future_field')) }

    [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'legacy-fake.state'), [byte[]] (0x71,0x73,0x67,0x67, 4,0,0,0, 0,0,0,0, 0,0,0,0))
    $r = Slot 'restore' 'legacy-fake.state'
    $ref['<legacy fake>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_legacy_unkeyed')) }
    if (Test-Path -LiteralPath $Lane0Slot) {
        $src = [System.IO.File]::OpenRead($Lane0Slot)
        try { $buf = New-Object byte[] 65536; $n = $src.Read($buf, 0, 65536) } finally { $src.Dispose() }
        if ($n -le 0) { throw "lane-0 legacy file is empty: $Lane0Slot" }
        [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'legacy-real.state'), [byte[]] $buf[0..($n - 1)])
        $r = Slot 'restore' 'legacy-real.state'
        $ref['<legacy real, lane-0 file head>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_legacy_unkeyed')) }
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
    $ref['<truncated>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_corrupt')) }
    [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'junk.state'), [byte[]] (0x4A,0x55,0x4E,0x4B, 1,2,3,4))
    $r = Slot 'restore' 'junk.state'
    $ref['<unrecognized>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'format')) }
    $r = Slot 'restore' 'does-not-exist.state'
    $ref['<missing>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_missing')) }

    # F11 refusals. effective_model absent (what every 7c77724b file looks like): fail closed, naming the field.
    New-TamperedCopy $good (Join-Path $SlotDir 'no-effective-model.state') { param($t) (($t -split "`n") | Where-Object { $_ -notmatch '^[HSI] effective_model=' }) -join "`n" }
    $r = Slot 'restore' 'no-effective-model.state'
    $ref['<effective_model absent>'] = [ordered]@{ status = $r.Status; refused_field = (Field $r 'error.refused_field'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'effective_model') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    if (Test-Path -LiteralPath $Lane1Slot) {                   # the real thing: tonight's acceptance-binary 4K file
        Copy-Item -LiteralPath $Lane1Slot -Destination (Join-Path $SlotDir 'lane1-7c77724b.state') -Force
        $r = Slot 'restore' 'lane1-7c77724b.state'
        $ref['<7c77724b file>'] = [ordered]@{ status = $r.Status; refused_field = (Field $r 'error.refused_field'); message = (Field $r 'error.message'); pass = (($r.Status -eq 409) -and ((Field $r 'error.refused_field') -eq 'effective_model') -and ((Field $r 'error.slot_untouched') -eq $true)) }
        Remove-Item -LiteralPath (Join-Path $SlotDir 'lane1-7c77724b.state') -Force -ErrorAction SilentlyContinue
    }
    # a path that exists but is not a regular file: state_unreadable, not state_missing (a3ff750d)
    New-Item -ItemType Directory -Force -Path (Join-Path $SlotDir 'dir-not-file.state') | Out-Null
    $r = Slot 'restore' 'dir-not-file.state'
    $ref['<unreadable: directory>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_unreadable') -and ((Field $r 'error.slot_untouched') -eq $true)) }
    # the in-progress-save suffix is reserved (fea27876, 91579f16), case-insensitively, for save, restore and rename
    $r = Slot 'save' 'reserved.stateos.tmp'
    $ref['<reserved name: save>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_name_reserved')) }
    $r = Slot 'restore' 'RESERVED.STATEOS.TMP'
    $ref['<reserved name: restore>'] = [ordered]@{ status = $r.Status; type = (Field $r 'error.type'); pass = (($r.Status -eq 409) -and ((Field $r 'error.type') -eq 'state_name_reserved')) }
    $r = Api 'POST' '/rename_prompt' '{"old_filename":"junk.state","new_filename":"renamed.stateos.tmp"}'
    $ref['<reserved name: rename>'] = [ordered]@{ status = $r.Status; type = (Field $r 'type'); pass = (($r.Status -eq 409) -and ((Field $r 'type') -eq 'state_name_reserved') -and (Test-Path -LiteralPath (Join-Path $SlotDir 'junk.state'))) }
    $Results.legA.refusals = $ref

    # every refusal above must have left the restored 4K S0 in the slot: the continuation is the in-memory one
    $after = Complete (Concat (Head $all 4096) $Z) 64
    $untouched = ($after.content -ceq $Results.legA.identity_4k.warm.content) -and ($after.prompt_n -eq $Results.legA.identity_4k.warm.prompt_n)
    $Results.legA.slot_untouched_after_refusals = [ordered]@{ prompt_n = $after.prompt_n; same_output = ($after.content -ceq $Results.legA.identity_4k.warm.content); pass = $untouched }
    $nRef = @($ref.Values | Where-Object { $_.pass }).Count
    Log "refusals: $nRef/$($ref.Count) pass; slot untouched after refusals: $untouched"

    # F11 /list (4e1f27ca): a State-OS entry names the state (format, token count, token digest), never its text
    $ls = Api 'GET' '/list' $null
    $ent = if (($ls.Status -eq 200) -and ($null -ne $ls.Body)) { @($ls.Body | Where-Object { $_.filename -eq 'id4k.state' }) } else { @() }
    $e0 = if ($ent.Count -eq 1) { $ent[0] } else { $null }
    $e0r = if ($e0) { [pscustomobject]@{ Status = 200; Body = $e0 } } else { $null }
    $listOk = ($null -ne $e0) -and ((Field $e0r 'format') -eq 'stateos-v1') -and (@($e0.PSObject.Properties.Name) -contains 'prompt') -and ($null -eq $e0.prompt) -and
              ((Field $e0r 'stateos.prompt_redacted') -eq $true) -and ((Field $e0r 'token_count') -eq 4096) -and
              ((Field $e0r 'stateos.token_sha256') -eq $Results.legA.identity_4k.save.token_sha256)
    $Results.legA.list_redacted = [ordered]@{ status = $ls.Status; entry = $e0; pass = $listOk }
    Log "GET /list id4k.state: status=$($ls.Status) format=$(Field $e0r 'format') prompt_redacted=$(Field $e0r 'stateos.prompt_redacted') token_count=$(Field $e0r 'token_count') pass=$listOk"
    Save-Results

    # --- destructive paths (review M3). Oracle for "correct re-prefill": the 4K cold output (full prefill of P+Z).
    # Same rule as the identity legs: an output difference while the 4K restored runs disagreed with each other is
    # engine run-to-run nondeterminism -> INCONCLUSIVE, not FAIL. Every mechanism condition must hold regardless.
    $PZ4 = Concat (Head $all 4096) $Z
    $cold4 = $Results.legA.identity_4k.cold
    $engineDeterministic = [bool] $Results.legA.identity_4k.restored_runs_agree
    function Destructive-Verdict([bool] $Mechanism, [bool] $SameOutput) {
        if (-not $Mechanism) { 'FAIL' } elseif ($SameOutput) { 'PASS' } elseif (-not $engineDeterministic) { 'INCONCLUSIVE' } else { 'FAIL' }
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
        $Results.legB.companion_32k = Test-Identity 'on32k' (Head $all 32768) $Z $Q 128 -NoCold -LogPath $logB -WarmControl 'always'
        # the spec-off 32K file has no COMP section: restore it here to see acceptance without the companion
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)
        $poNc = Ple-Offsets $logB
        $rs = Slot 'restore' 'id32k.state'
        $nc = [ordered]@{ status = $rs.Status; stateos = (Field $rs 'stateos'); error = (Field $rs 'error') }
        if ($rs.Status -eq 200) {
            $nc.out = Complete (Concat (Head $all 32768) $Z) 128
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
            $Results.legB.bytes_190k = Test-Identity 'on190k' (Head $all 190000) $Z $Q 16 -NoCold -LogPath $logB -WarmControl 'if-differs'
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
} catch {
    Log "ABORTED: $($_.Exception.Message)"
    $Results.aborted = $_.Exception.Message
} finally {
    Save-Results
    if ($proc) { Stop-TestServer $proc }
    # restore the standing server no matter what (detached; never block on it). The test-only PLE switches must not
    # leak into the standing server's inherited environment.
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Remove-Item Env:LONGSPEAR_PLE_HIST_REWIND, Env:LONGSPEAR_PLE_HIST_LOG -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$Standing -WindowStyle Hidden | Out-Null
    Log 'restore launcher started'
    $ok = $false
    for ($i = 0; $i -lt 90 -and -not $ok; $i++) { Start-Sleep -Seconds 10; try { $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8099/health' -TimeoutSec 5; $ok = ($h.status -eq 'ok') } catch {} }
    Log ('production-restored:' + $(if ($ok) { '200' } else { 'FAILED' }))
    # the 190K files are ~4-5 GB each; keep the small ones as receipts
    Get-ChildItem -LiteralPath $SlotDir -Filter 'on190k*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}
