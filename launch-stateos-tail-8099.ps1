# SV2-E1 (State-OS v2 tail snapshot) :8099 launcher for the coordinator's W-SV2 window.
# Same serving args as D:\AI\ik_llama-qwen4exp\launch-standing-8099.ps1 (gpu-fallback spec checkpoints,
# 192K, -ncmoe 37), binary from this lane's build-stateos-tail. The API key is read from
# D:\AI\llama-swap\config.yaml at launch, never stored here.
#
# Always set: LONGSPEAR_PLE_HIST_REWIND=1 and LONGSPEAR_PLE_HIST_LOG=1 (every arm, see below).
# State-OS v2 flags (all unset unless a switch is given, so inherited values cannot leak in):
#   -DivLog  LONGSPEAR_STATEOS_DIV_LOG=1        [stateos-div] telemetry
#   -Tail    LONGSPEAR_STATEOS_TAIL_SNAPSHOT=1  tail snapshot at release (C1)
#   -Xcheck  LONGSPEAR_STATEOS_TAIL_XCHECK=1    diagnostic cross-check (use launch-stateos-tail-xcheck-8099.ps1)
# -ExtraArgs is appended last (later flags win), e.g. '-no-fmoe -no-fug' or '-fa 0' for the benign gate arms.
# Logs: D:\AI\ik_llama-qwen4exp\<LogStem>.out.log / .err.log (read them with tools/stateos-div-census.mjs).
param(
    [switch]$DivLog,
    [switch]$Tail,
    [switch]$Xcheck,
    [string]$ExtraArgs = '',
    [string]$LogStem = 'ik-serve-8099-stateos-tail'
)
$ErrorActionPreference = 'Stop'

if (Test-Path 'D:\AI\ik_llama-qwen4exp\BOX-LOCK.json') {
    throw 'BOX-LOCK.json exists: a measurement owns the box; not launching'
}
$exe = 'D:\AI\worktrees\ik-stateos-tail\build-stateos-tail\bin\llama-server.exe'
if (-not (Test-Path $exe)) { throw "missing $exe (run build-stateos-tail.cmd)" }

$cfg = Get-Content -Raw 'D:\AI\llama-swap\config.yaml'
$m = [regex]::Match($cfg, '--api-key +"?([^"\s]+)')
if (-not $m.Success) { throw 'no --api-key line in D:\AI\llama-swap\config.yaml' }
$key = $m.Groups[1].Value

# the standing launcher's env, then the State-OS v2 flags exactly as requested
$env:LONGSPEAR_VERIFY_TIMING = '1'
$env:LONGSPEAR_CG_REVIVE = '1'
# PLE n-gram history repair after every rewind (lane/ple-hist-rewind), in EVERY W-SV2 arm, flag-off
# arms included, so the arms differ only in the tail lever; its log feeds the chain's mechanism check
# ([ple-hist] reset at pos > 0 must stay 0: tools/stateos-div-census.mjs --check)
$env:LONGSPEAR_PLE_HIST_REWIND = '1'
$env:LONGSPEAR_PLE_HIST_LOG = '1'
Remove-Item Env:LONGSPEAR_OP_CENSUS -ErrorAction SilentlyContinue
Remove-Item Env:LONGSPEAR_CG_DEBUG -ErrorAction SilentlyContinue
Remove-Item Env:LONGSPEAR_CG_DEBUG2 -ErrorAction SilentlyContinue
foreach ($pair in @(
        @('LONGSPEAR_STATEOS_DIV_LOG', $DivLog),
        @('LONGSPEAR_STATEOS_TAIL_SNAPSHOT', $Tail),
        @('LONGSPEAR_STATEOS_TAIL_XCHECK', $Xcheck))) {
    if ($pair[1]) {
        Set-Item -Path ("Env:" + $pair[0]) -Value '1'
    } else {
        Remove-Item -Path ("Env:" + $pair[0]) -ErrorAction SilentlyContinue
    }
}

$argsx = '-m "D:\AI\LLM Models\custom\Qwen3.8-Flash-Next-MXFP4moe-ngramQ8-MTP.gguf" --api-key "' + $key + '" -ngl 999 -ncmoe 37 -fa 1 -c 196608 -ub 512 -ctk q8_0 -ctv q8_0 -np 1 -t 24 -tb 32 --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --host 0.0.0.0 --port 8099 --spec-type ngram-mod:n_min=4 --spec-type mtp:n_max=4 --reasoning-budget 1024 --spec-ckpt-mode gpu-fallback -rtr -muge'
if ($ExtraArgs -ne '') { $argsx = $argsx + ' ' + $ExtraArgs }

Start-Process -FilePath $exe -ArgumentList $argsx -WindowStyle Hidden `
    -RedirectStandardOutput ("D:\AI\ik_llama-qwen4exp\" + $LogStem + ".out.log") `
    -RedirectStandardError ("D:\AI\ik_llama-qwen4exp\" + $LogStem + ".err.log")
Write-Output ("stateos-tail-8099-launched divlog=" + [int][bool]$DivLog + " tail=" + [int][bool]$Tail + " xcheck=" + [int][bool]$Xcheck + " ple_hist_rewind=1 ple_hist_log=1 extra='" + $ExtraArgs + "' log=" + $LogStem)
