# GPU verification recipe

Run only in an approved GPU window, with the production service stopped. This recipe uses port 8101, not 8099.

The C++ speculative object is opaque and `common_speculative_init()` requires a loaded target model/context, so the
invalidation contract cannot be exercised by the existing model-free CTest fixtures. The two new call sites therefore
emit verbose evidence in the form `MTP invalidate: <reason> slot=<id> pos=<first-invalid-position>`.

## Inputs

Set these to the coordinator-owned baseline build, this lane's patched build, production model, and the existing
four-task proxy recording. Keep the production MTP/configuration flags in `$ProductionArgs`; do not include `--model`,
`--port`, `--cache-ram`, `--slot-save-path`, or `--verbose` there.

```powershell
$BaselineExe = 'D:\path\to\62cc77b6\llama-server.exe'
$PatchedExe  = 'D:\AI\worktrees\stateos-lane0\build\bin\llama-server.exe'
$Model       = 'D:\path\to\production-model.gguf'
$ReplayJsonl = 'D:\Projects\longspear\sessions\repro\FOUR-TASK-RECORDING.jsonl'
$ApiKey      = $env:LONGSPEAR_API_KEY
$ProductionArgs = @(
    '--ctx-size', '65536'
    '--spec-type', 'mtp:n_max=4,p_min=0.0'
    # Copy every other production performance/KV flag here unchanged.
)
```

## A/B replay

This sends every recorded request body twice to the same server process. The first pass populates the 8192 MiB RAM
cache and the second pass exercises cache loads. It also saves/restores slot 0 once to cover the file-restore call site.

```powershell
$Port = 8101
$Root = 'D:\AI\worktrees\stateos-lane0\build\gpu-verify'
$SlotDir = Join-Path $Root 'slots'
New-Item -ItemType Directory -Force -Path $Root, $SlotDir | Out-Null
$Headers = @{ Authorization = "Bearer $ApiKey"; 'Content-Type' = 'application/json' }

function Invoke-RecordedPass([string] $Recording, [string] $Endpoint) {
    $requests = Get-Content -LiteralPath $Recording | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $null -ne $_.body }
    if (($requests | Measure-Object).Count -eq 0) { throw 'recording has no request bodies' }
    foreach ($request in $requests) {
        $body = if ($request.body -is [string]) { $request.body } else {
            $request.body | ConvertTo-Json -Depth 100 -Compress
        }
        Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$Endpoint/v1/chat/completions" `
            -Headers $Headers -Body $body -TimeoutSec 900 | Out-Null
    }
}

foreach ($variant in [ordered]@{ before = $BaselineExe; after = $PatchedExe }.GetEnumerator()) {
    $log = Join-Path $Root "$($variant.Key).log"
    $err = Join-Path $Root "$($variant.Key).err.log"
    $args = @('--model', $Model, '--host', '127.0.0.1', '--port', "$Port", '--cache-ram', '8192',
        '--slot-save-path', "$SlotDir\", '--verbose') + $ProductionArgs
    $proc = Start-Process -FilePath $variant.Value -ArgumentList $args -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError $err
    try {
        $healthy = $false
        1..120 | ForEach-Object {
            if (-not $healthy) {
                try {
                    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
                    $healthy = $health.status -eq 'ok'
                } catch {}
                if (-not $healthy) { Start-Sleep -Seconds 5 }
            }
        }
        if (-not $healthy) { throw "$($variant.Key) server did not become healthy" }

        Invoke-RecordedPass $ReplayJsonl "http://127.0.0.1:$Port"
        Invoke-RecordedPass $ReplayJsonl "http://127.0.0.1:$Port"

        if ($variant.Key -eq 'after') {
            $slotBody = '{"filename":"mtp-invalidate-slot.bin"}'
            Invoke-WebRequest -UseBasicParsing -Method Post -Uri "http://127.0.0.1:$Port/slots/0?action=save" `
                -Headers $Headers -Body $slotBody -TimeoutSec 900 | Out-Null
            Invoke-WebRequest -UseBasicParsing -Method Post -Uri "http://127.0.0.1:$Port/slots/0?action=restore" `
                -Headers $Headers -Body $slotBody -TimeoutSec 900 | Out-Null
        }
    } finally {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id }
        $proc.WaitForExit()
    }
}
```

## Evidence and acceptance-rate comparison

```powershell
$after = Get-Content -LiteralPath (Join-Path $Root 'after.log')
$cacheLoads = @($after | Select-String 'found better prompt with').Count
$cacheInvalidates = @($after | Select-String 'MTP invalidate: prompt_load').Count
$restoreInvalidates = @($after | Select-String 'MTP invalidate: SLOT_RESTORE').Count
"cache loads=$cacheLoads prompt invalidates=$cacheInvalidates restore invalidates=$restoreInvalidates"
if ($cacheLoads -eq 0 -or $cacheLoads -ne $cacheInvalidates) { throw 'missing invalidate on a cache load' }
if ($restoreInvalidates -ne 1) { throw 'missing SLOT_RESTORE invalidate' }

foreach ($name in 'before', 'after') {
    $accepted = 0; $generated = 0
    Get-Content -LiteralPath (Join-Path $Root "$name.log") | ForEach-Object {
        if ($_ -match 'draft acceptance rate = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)') {
            $accepted += [int]$Matches[1]; $generated += [int]$Matches[2]
        }
    }
    $rate = if ($generated) { $accepted / $generated } else { 0 }
    "{0}: accepted={1} generated={2} weighted_rate={3:P3}" -f $name, $accepted, $generated, $rate
}
```

Pass criteria: every observed RAM-cache load has one `prompt_load` invalidate line, the explicit restore has one
`SLOT_RESTORE` line, all positions are the first position after the restored tokens, and the patched weighted draft
acceptance rate does not regress versus the baseline. Preserve both logs and the exact resolved launch arguments.
