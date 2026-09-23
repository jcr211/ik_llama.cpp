# State-OS lane 1 GPU acceptance (coordinator-run; see .lane/GPU-VERIFY.md). NOT run by the lane.
#
# Takes the GPU slot (stops every llama-server, including the standing :8099), runs the lane-1 build on port 8101
# WITHOUT --api-key (loopback only), then ALWAYS relaunches the standing server via launch-standing-8099.ps1 and polls
# :8099/health. Receipts: build-stateos-l1\gpu-verify\{verdict.txt, results.json, launch-args.txt, *.log}.
#
# Leg A (spec OFF, the acceptance leg): greedy identity (temperature 0) across save -> pollute -> restore at 4K and 32K,
#   one 409 refusal per hard header field + unknown-hard-field + legacy/unkeyed + corrupt + missing, a soft-field warning,
#   a continuation proving the slot was untouched by every refusal, then the destructive paths: an empty-slot round trip
#   and a MAIN-payload tamper (500, slot cleared, server alive, correct re-prefill). Auto-stops on the first identity
#   FAIL or on a refusal that is not a 409 naming its field (mechanism kill criteria).
# Leg B (spec ON = production flags, report-only): companion section saved/loaded at 32K, a companion sub-header tamper
#   (200, companion skipped), draft acceptance with the companion vs a companion-less file, and the state bytes +
#   restore time at 190K tokens (production context 196608).
param(
    [switch] $SkipSpecOn,
    [switch] $Skip192K
)
$ErrorActionPreference = 'Stop'

$Worktree   = 'D:\AI\worktrees\stateos-lane1'
$PatchedExe = Join-Path $Worktree 'build-stateos-l1\bin\llama-server.exe'
$Model      = 'D:\AI\LLM Models\custom\Qwen3.8-Flash-Next-MXFP4moe-ngramQ8-MTP.gguf'
$Standing   = 'D:\AI\ik_llama-qwen4exp\launch-standing-8099.ps1'
$Lane0Slot  = 'D:\AI\worktrees\stateos-lane0\build\gpu-verify\slots\mtp-invalidate-slot.bin' # a real legacy file, if still present
$Port       = 8101
$Base       = "http://127.0.0.1:$Port"
$Root       = Join-Path $Worktree 'build-stateos-l1\gpu-verify'
$SlotDir    = Join-Path $Root 'slots'
$VerdictTxt = Join-Path $Root 'verdict.txt'
$ResultsJs  = Join-Path $Root 'results.json'

# Production flags from launch-standing-8099.ps1 minus -m/--api-key/--host/--port and minus the speculation flags
# (diff against that launcher before running; it is the single source of truth).
$CommonArgs = @('-ngl','999','-ncmoe','37','-fa','1','-c','196608','-ub','512','-ctk','q8_0','-ctv','q8_0','-np','1','-t','24','-tb','32',
                '--jinja','--temp','1.0','--top-p','0.95','--top-k','20','--min-p','0.0','--reasoning-budget','1024','-rtr','-muge')
$SpecArgs   = @('--spec-type','ngram-mod:n_min=4','--spec-type','mtp:n_max=4','--spec-ckpt-mode','gpu-fallback')
$env:LONGSPEAR_VERIFY_TIMING = '1'
$env:LONGSPEAR_CG_REVIVE = '1'

New-Item -ItemType Directory -Force -Path $Root, $SlotDir | Out-Null
Add-Type -AssemblyName System.Net.Http
$Http = New-Object System.Net.Http.HttpClient
$Http.Timeout = [TimeSpan]::FromMinutes(60)
$Results = [ordered]@{ started = (Get-Date).ToUniversalTime().ToString('o'); legA = [ordered]@{}; legB = [ordered]@{} }

function Log([string] $m) {
    $line = "[$((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))] $m"
    Add-Content -LiteralPath $VerdictTxt -Value $line -Encoding utf8
    Write-Host $line
}
function Short([string] $s) { if ($null -eq $s) { '' } elseif ($s.Length -gt 400) { $s.Substring(0, 400) + '...' } else { $s } }
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

function Get-Tokens([string] $Text) {
    $r = Api 'POST' '/tokenize' (@{ content = $Text } | ConvertTo-Json -Compress)
    if ($r.Status -ne 200) { throw "tokenize failed: $($r.Status) $(Short $r.Raw)" }
    return ,([int[]] $r.Body.tokens)
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
    $r = Api 'POST' '/completion' $json
    if ($r.Status -ne 200) { throw "completion failed: $($r.Status) $(Short $r.Raw)" }
    [pscustomobject]@{ content = [string] $r.Body.content; prompt_n = [int] $r.Body.timings.prompt_n; prompt_ms = [double] $r.Body.timings.prompt_ms; predicted_n = [int] $r.Body.timings.predicted_n }
}

function Slot([string] $Action, [string] $File) {
    $json = if ($File) { '{"filename":"' + $File + '"}' } else { '{}' }
    Api 'POST' "/slots/0?action=$Action" $json
}

function Concat([int[]] $A, [int[]] $B) { $r = New-Object int[] ($A.Length + $B.Length); [Array]::Copy($A, $r, $A.Length); [Array]::Copy($B, 0, $r, $A.Length, $B.Length); return ,$r }
function Head([int[]] $A, [int] $N) { $r = New-Object int[] $N; [Array]::Copy($A, $r, $N); return ,$r }

# log lines appended since a byte offset, read while the server still holds the file open
function Read-LogSince([string] $Path, [long] $Offset) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    $fs = New-Object System.IO.FileStream($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        [void] $fs.Seek($Offset, [System.IO.SeekOrigin]::Begin)
        $sr = New-Object System.IO.StreamReader($fs)
        return ,($sr.ReadToEnd() -split "`n")
    } finally { $fs.Dispose() }
}
function Log-Size([string] $Path) { if (Test-Path -LiteralPath $Path) { (Get-Item -LiteralPath $Path).Length } else { 0 } }
function Acceptance([string[]] $Lines) {
    $a = 0; $g = 0
    foreach ($l in $Lines) { if ($l -match 'draft acceptance rate = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)') { $a += [int] $Matches[1]; $g += [int] $Matches[2] } }
    [pscustomobject]@{ accepted = $a; generated = $g; rate = $(if ($g) { [math]::Round($a / $g, 4) } else { $null }) }
}

function Start-TestServer([string] $Name, [string[]] $Extra) {
    $log = Join-Path $Root "$Name.log"; $err = Join-Path $Root "$Name.err.log"
    Remove-Item -LiteralPath $log, $err -ErrorAction SilentlyContinue
    # Start-Process joins ArgumentList with spaces and does not quote: paths with spaces carry their own quotes
    $argv = @('-m', ('"' + $Model + '"'), '--host', '127.0.0.1', '--port', "$Port", '--slot-save-path', ('"' + $SlotDir + '"'), '--verbose') + $CommonArgs + $Extra
    Add-Content -LiteralPath (Join-Path $Root 'launch-args.txt') -Value "$Name`: $PatchedExe $($argv -join ' ')" -Encoding utf8
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
    foreach ($l in ([System.Text.Encoding]::UTF8.GetString($hb) -split "`n")) { if ($l -match ('^[HSI] ' + [regex]::Escape($Key) + '=(.*)$')) { return $Matches[1] } }
    return $null
}

# ---- one identity round: in-memory continuation vs restored continuation (x2) vs cold (report-only) ----
function Test-Identity([string] $Tag, [int[]] $P, [int[]] $Z, [int[]] $Q, [int] $NGen, [switch] $NoCold, [string] $LogPath) {
    $PZ = Concat $P $Z
    $res = [ordered]@{ n_prompt = $P.Length; n_suffix = $Z.Length }
    [void] (Slot 'erase' $null)
    [void] (Complete $P 1)                                       # in-memory state S0: the slot holds P
    $s = Slot 'save' "$Tag.state"
    if ($s.Status -ne 200) { throw "$Tag save: $($s.Status) $(Short $s.Raw)" }
    $res.save = [ordered]@{ n_saved = $s.Body.n_saved; n_written = $s.Body.n_written; save_ms = $s.Body.timings.save_ms; bytes = $s.Body.stateos.bytes; companion = $s.Body.stateos.companion; checkpoints_saved = $s.Body.stateos.checkpoints_saved; token_sha256 = $s.Body.stateos.token_sha256 }
    $o = Log-Size $LogPath
    $w = Complete $PZ $NGen                                      # continuation from the in-memory S0
    $res.warm = $w; $res.warm_acceptance = Acceptance (Read-LogSince $LogPath $o)

    $runs = @()
    foreach ($k in 1, 2) {
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)      # another conversation occupies the slot
        $rs = Slot 'restore' "$Tag.state"
        if ($rs.Status -ne 200) { throw "$Tag restore $k`: $($rs.Status) $(Short $rs.Raw)" }
        $o = Log-Size $LogPath
        $r = Complete $PZ $NGen
        $runs += [ordered]@{ restore = [ordered]@{ n_restored = $rs.Body.n_restored; n_read = $rs.Body.n_read; restore_ms = $rs.Body.timings.restore_ms; stateos = $rs.Body.stateos }; out = $r; acceptance = (Acceptance (Read-LogSince $LogPath $o)) }
    }
    $res.restored = $runs
    if (-not $NoCold) {
        [void] (Slot 'erase' $null)
        $res.cold = Complete $PZ $NGen                           # full prefill of P+Z, report-only
    }
    $same1 = $runs[0].out.content -ceq $w.content
    $same2 = $runs[1].out.content -ceq $w.content
    $reuse = ($runs[0].out.prompt_n -eq $w.prompt_n) -and ($runs[1].out.prompt_n -eq $w.prompt_n) -and ($w.prompt_n -le ($Z.Length + 1))
    $res.identity_restored_vs_warm = ($same1 -and $same2)
    $res.no_reprefill = $reuse
    $res.restored_runs_agree = ($runs[0].out.content -ceq $runs[1].out.content)
    if (-not $NoCold) { $res.identity_cold_vs_warm_report_only = ($res.cold.content -ceq $w.content) }
    $res.verdict = if ($same1 -and $same2 -and $reuse) { 'PASS' } elseif ($same1 -and $same2) { 'PASS-IDENTITY / REUSE-INCONCLUSIVE' } else { 'FAIL' }
    Log ("$Tag identity: {0} (warm prompt_n={1}, restored prompt_n={2}/{3}, restore_ms={4:N1}, file={5} B)" -f $res.verdict, $w.prompt_n, $runs[0].out.prompt_n, $runs[1].out.prompt_n, $runs[0].restore.restore_ms, $res.save.n_written)
    return $res
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
    $proc = Start-TestServer 'specoff' @()
    $logA = Join-Path $Root 'specoff.log'
    $props = Api 'GET' '/props' $null
    $propsOk = ($props.Status -eq 200) -and ($props.Body.stateos.version -ge 1) -and ($props.Body.stateos.keyed_header -eq $true) -and ($props.Body.stateos.companion -eq $false)
    $Results.legA.props_stateos = [ordered]@{ value = $props.Body.stateos; pass = $propsOk }
    Log "GET /props stateos (spec off, expect companion=false): $($props.Body.stateos | ConvertTo-Json -Compress) pass=$propsOk"
    $text = New-SyntheticText 9000 7
    $all = Get-Tokens $text
    if ($all.Length -lt 200000) { $all = Concat $all (Get-Tokens (New-SyntheticText 9000 11)) }
    $Z = Get-Tokens "`n`nIn one sentence, which record has the largest value, and what is its tag?"
    $Q = Get-Tokens (New-SyntheticText 80 99)
    Log "synthetic tokens: $($all.Length); suffix Z=$($Z.Length); pollution Q=$($Q.Length)"

    $Results.legA.identity_4k = Test-Identity 'id4k' (Head $all 4096) $Z $Q 64 -LogPath $logA
    Save-Results
    if ($Results.legA.identity_4k.verdict -eq 'FAIL') { throw 'KILL: greedy identity failed at 4K (restored != in-memory continuation); stopping before 32K' }
    $Results.legA.identity_32k = Test-Identity 'id32k' (Head $all 32768) $Z $Q 64 -LogPath $logA
    Save-Results

    # --- refusals: establish a known slot state (the 4K S0), then every refusal must leave it in place
    $good = Join-Path $SlotDir 'id4k.state'
    $rs = Slot 'restore' 'id4k.state'
    if ($rs.Status -ne 200) { throw "baseline restore failed: $($rs.Status)" }

    # soft field: warn and proceed
    New-TamperedCopy $good (Join-Path $SlotDir 'soft-build.state') { param($t) Set-HeaderValue $t 'build' '1-softfieldtest' }
    $soft = Slot 'restore' 'soft-build.state'
    $softOk = ($soft.Status -eq 200) -and (@($soft.Body.stateos.warnings | Where-Object { $_.field -eq 'build' }).Count -eq 1)
    $Results.legA.soft_build = [ordered]@{ status = $soft.Status; warnings = $soft.Body.stateos.warnings; pass = $softOk }
    Log "soft field 'build': status=$($soft.Status) pass=$softOk"

    $hard = [ordered]@{
        model_fingerprint_v2 = 'ffff' + (Get-HeaderValue $good 'model_fingerprint_v2').Substring(4)
        n_ctx                = '65536'
        cache_type_k         = 'f16'
        cache_type_v         = 'f16'
        rope                 = (Get-HeaderValue $good 'rope') + ' tampered=1'
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
        $ok = ($r.Status -eq 409) -and ($r.Body.error.refused_field -eq $k) -and ($r.Body.error.slot_untouched -eq $true)
        $ref[$k] = [ordered]@{ status = $r.Status; refused_field = $r.Body.error.refused_field; message = $r.Body.error.message; pass = $ok }
        Log "hard field '$k': status=$($r.Status) refused_field=$($r.Body.error.refused_field) pass=$ok"
        if ($first -and -not $ok) { throw "KILL: the first hard-field refusal did not answer 409 naming '$k': $(Short $r.Raw)" }
        $first = $false
    }
    New-TamperedCopy $good (Join-Path $SlotDir 'hard-unknown.state') { param($t) $t + "H future_field=x`n" }
    $r = Slot 'restore' 'hard-unknown.state'
    $ref['<unknown hard field>'] = [ordered]@{ status = $r.Status; refused_field = $r.Body.error.refused_field; pass = (($r.Status -eq 409) -and ($r.Body.error.refused_field -eq 'future_field')) }

    [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'legacy-fake.state'), [byte[]] (0x71,0x73,0x67,0x67, 4,0,0,0, 0,0,0,0, 0,0,0,0))
    $r = Slot 'restore' 'legacy-fake.state'
    $ref['<legacy fake>'] = [ordered]@{ status = $r.Status; type = $r.Body.error.type; message = $r.Body.error.message; pass = (($r.Status -eq 409) -and ($r.Body.error.type -eq 'state_legacy_unkeyed')) }
    if (Test-Path -LiteralPath $Lane0Slot) {
        $src = [System.IO.File]::OpenRead($Lane0Slot)
        try { $buf = New-Object byte[] 65536; $n = $src.Read($buf, 0, 65536) } finally { $src.Dispose() }
        [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'legacy-real.state'), [byte[]] $buf[0..($n - 1)])
        $r = Slot 'restore' 'legacy-real.state'
        $ref['<legacy real, lane-0 file head>'] = [ordered]@{ status = $r.Status; type = $r.Body.error.type; pass = (($r.Status -eq 409) -and ($r.Body.error.type -eq 'state_legacy_unkeyed')) }
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
    $ref['<truncated>'] = [ordered]@{ status = $r.Status; type = $r.Body.error.type; message = $r.Body.error.message; pass = (($r.Status -eq 409) -and ($r.Body.error.type -eq 'state_corrupt')) }
    [System.IO.File]::WriteAllBytes((Join-Path $SlotDir 'junk.state'), [byte[]] (0x4A,0x55,0x4E,0x4B, 1,2,3,4))
    $r = Slot 'restore' 'junk.state'
    $ref['<unrecognized>'] = [ordered]@{ status = $r.Status; type = $r.Body.error.type; pass = (($r.Status -eq 409) -and ($r.Body.error.refused_field -eq 'format')) }
    $r = Slot 'restore' 'does-not-exist.state'
    $ref['<missing>'] = [ordered]@{ status = $r.Status; type = $r.Body.error.type; pass = (($r.Status -eq 409) -and ($r.Body.error.type -eq 'state_missing')) }
    $Results.legA.refusals = $ref

    # every refusal above must have left the restored 4K S0 in the slot: the continuation is the in-memory one
    $after = Complete (Concat (Head $all 4096) $Z) 64
    $untouched = ($after.content -ceq $Results.legA.identity_4k.warm.content) -and ($after.prompt_n -eq $Results.legA.identity_4k.warm.prompt_n)
    $Results.legA.slot_untouched_after_refusals = [ordered]@{ prompt_n = $after.prompt_n; same_output = ($after.content -ceq $Results.legA.identity_4k.warm.content); pass = $untouched }
    $nRef = @($ref.Values | Where-Object { $_.pass }).Count
    Log "refusals: $nRef/$($ref.Count) pass; slot untouched after refusals: $untouched"
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
    $re = Slot 'restore' 'empty.state'
    $alive = $false; try { $alive = ((Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 5).status -eq 'ok') } catch {}
    $ae = Complete $PZ4 64
    $emptyMech = ($se.Status -eq 200) -and ($se.Body.n_saved -eq 0) -and ($re.Status -eq 200) -and ($re.Body.stateos.empty -eq $true) -and $alive -and
                 ($ae.prompt_n -eq $PZ4.Length)
    $emptySame = ($ae.content -ceq $cold4.content)
    $emptyVerdict = Destructive-Verdict $emptyMech $emptySame
    $Results.legA.empty_roundtrip = [ordered]@{ save_status = $se.Status; restore_status = $re.Status; restore = $re.Body.stateos; alive = $alive; next = $ae; mechanism = $emptyMech; same_as_cold = $emptySame; verdict = $emptyVerdict; pass = ($emptyVerdict -eq 'PASS') }
    Log "empty-slot round trip: save=$($se.Status) restore=$($re.Status) empty=$($re.Body.stateos.empty) alive=$alive next prompt_n=$($ae.prompt_n)/$($PZ4.Length) same-as-cold=$emptySame verdict=$emptyVerdict"

    # (b) MAIN tamper: cell_count + 1 inside a well-formed container -> the loader fails after its seq_rm -> 500,
    #     slot_untouched:false, server alive, the next request re-prefills and is correct
    $bad = Join-Path $SlotDir 'main-tamper.state'
    Copy-Item -LiteralPath $good -Destination $bad -Force
    $m = Find-Section $bad 'MAIN'
    $cc = [BitConverter]::ToUInt32((Read-BytesAt $bad $m.Offset 4), 0)
    Write-BytesAt $bad $m.Offset ([BitConverter]::GetBytes([uint32] ($cc + 1)))
    $rs = Slot 'restore' 'id4k.state'                         # the slot holds S0 before the tamper
    $rt = Slot 'restore' 'main-tamper.state'
    $alive = $false; try { $alive = ((Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 5).status -eq 'ok') } catch {}
    $at = Complete $PZ4 64
    $tamperMech = ($rs.Status -eq 200) -and ($rt.Status -eq 500) -and ($rt.Body.error.slot_untouched -eq $false) -and $alive -and
                  ($at.prompt_n -eq $PZ4.Length)
    $tamperSame = ($at.content -ceq $cold4.content)
    $tamperVerdict = Destructive-Verdict $tamperMech $tamperSame
    $Results.legA.main_tamper = [ordered]@{ cell_count = $cc; status = $rt.Status; error = $rt.Body.error; alive = $alive; next = $at; mechanism = $tamperMech; same_as_cold = $tamperSame; verdict = $tamperVerdict; pass = ($tamperVerdict -eq 'PASS') }
    Log "MAIN tamper (cell_count $cc -> $($cc + 1)): status=$($rt.Status) slot_untouched=$($rt.Body.error.slot_untouched) alive=$alive next prompt_n=$($at.prompt_n)/$($PZ4.Length) same-as-cold=$tamperSame verdict=$tamperVerdict"
    Save-Results
    Stop-TestServer $proc; $proc = $null

    # ================= Leg B: speculation ON = production flags (report-only) =================
    if (-not $SkipSpecOn) {
        $proc = Start-TestServer 'specon' $SpecArgs
        $logB = Join-Path $Root 'specon.log'
        $props = Api 'GET' '/props' $null
        $propsOk = ($props.Status -eq 200) -and ($props.Body.stateos.version -ge 1) -and ($props.Body.stateos.keyed_header -eq $true) -and ($props.Body.stateos.companion -eq $true)
        $Results.legB.props_stateos = [ordered]@{ value = $props.Body.stateos; pass = $propsOk }
        Log "GET /props stateos (spec on, expect companion=true): $($props.Body.stateos | ConvertTo-Json -Compress) pass=$propsOk"
        $Results.legB.companion_32k = Test-Identity 'on32k' (Head $all 32768) $Z $Q 128 -NoCold -LogPath $logB
        # the spec-off 32K file has no COMP section: restore it here to see acceptance without the companion
        [void] (Slot 'erase' $null); [void] (Complete $Q 8)
        $rs = Slot 'restore' 'id32k.state'
        $nc = [ordered]@{ status = $rs.Status; stateos = $rs.Body.stateos; error = $rs.Body.error }
        if ($rs.Status -eq 200) {
            $o = Log-Size $logB
            $nc.out = Complete (Concat (Head $all 32768) $Z) 128
            $nc.acceptance = Acceptance (Read-LogSince $logB $o)
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
                $sub = [System.Text.Encoding]::ASCII.GetString((Read-BytesAt $ct ($c.Offset + 4) ([int] $sublen)))
                $gpos = $sub.IndexOf('companion_kv_geometry=')
                if ($gpos -lt 0) { throw 'companion_kv_geometry not found in the COMP sub-header' }
                $vpos = $gpos + 'companion_kv_geometry='.Length
                $newc = if ($sub[$vpos] -eq '0') { [byte][char] '1' } else { [byte][char] '0' }
                Write-BytesAt $ct ($c.Offset + 4 + $vpos) ([byte[]] @($newc))
                [void] (Slot 'erase' $null); [void] (Complete $Q 8)
                $rc = Slot 'restore' 'comp-tamper.state'
                $compOk = ($rc.Status -eq 200) -and ([string] $rc.Body.stateos.companion).StartsWith("skipped: companion field 'companion_kv_geometry'")
                $Results.legB.comp_tamper = [ordered]@{ status = $rc.Status; companion = $rc.Body.stateos.companion; verdict = $(if ($compOk) { 'PASS' } else { 'FAIL' }); pass = $compOk }
                Log "COMP sub-header tamper: status=$($rc.Status) companion='$($rc.Body.stateos.companion)' pass=$compOk"
            } catch {
                $Results.legB.comp_tamper = [ordered]@{ verdict = 'FAIL'; pass = $false; reason = $_.Exception.Message }
                Log "COMP sub-header tamper: FAIL ($($_.Exception.Message)); continuing to the 190K measurement"
            } finally {
                Remove-Item -LiteralPath $ct -Force -ErrorAction SilentlyContinue
            }
        }
        Save-Results

        if (-not $Skip192K) {
            $Results.legB.bytes_190k = Test-Identity 'on190k' (Head $all 190000) $Z $Q 16 -NoCold -LogPath $logB
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
} catch {
    Log "ABORTED: $($_.Exception.Message)"
    $Results.aborted = $_.Exception.Message
} finally {
    Save-Results
    if ($proc) { Stop-TestServer $proc }
    # restore the standing server no matter what (detached; never block on it)
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$Standing -WindowStyle Hidden | Out-Null
    Log 'restore launcher started'
    $ok = $false
    for ($i = 0; $i -lt 90 -and -not $ok; $i++) { Start-Sleep -Seconds 10; try { $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8099/health' -TimeoutSec 5; $ok = ($h.status -eq 'ok') } catch {} }
    Log ('production-restored:' + $(if ($ok) { '200' } else { 'FAILED' }))
    # the 190K files are ~4-5 GB each; keep the small ones as receipts
    Get-ChildItem -LiteralPath $SlotDir -Filter 'on190k*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}
