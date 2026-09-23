# SL-1 launcher for the W-SL1 window: the build-sl1 binary (build-perstep.cmd) on :8099 with the
# standing production arguments (launch-standing-8099.ps1) except the arm's checkpoint mode and flags.
#   -Arm A2  per-step + PLE tail + capacity 5 + clamp + lean sampler + [spec-host] (default)
#   -Arm P0  production gpu-fallback, flags unset except [spec-host] (same telemetry cost as A2)
#   -Arm A0  speculation off (no --spec-type), control arm
#   -Crosscheck (A2 only) adds LONGSPEAR_SPEC_CKPT_CROSSCHECK=1 for the step-1 probe
# Logs go to D:\AI\ik_llama-qwen4exp\ik-serve-8099.{out,err}.log, where native-replay.sh and
# bench-decode.sh read them (they take a launcher basename in that directory and pass no arguments:
# use the per-arm wrappers launch-perstep-{p0,a0,xcheck}-8099.ps1, or this file for A2).
# The API key is read from D:\AI\llama-swap\config.yaml at launch and never printed or written.
# Refuses to start while anything listens on the port.
param(
    [ValidateSet('A2', 'P0', 'A0')]
    [string]$Arm = 'A2',
    [switch]$Crosscheck,
    [string]$Exe = 'D:\AI\worktrees\sl1-spec-ckpt\build-sl1\bin\llama-server.exe',
    [string]$LogDir = 'D:\AI\ik_llama-qwen4exp',
    [string]$LogName = 'ik-serve-8099',
    [int]$Port = 8099
)

function Fail([int]$Code, [string]$Message) {
    [Console]::Error.WriteLine("launch-perstep: $Message")
    exit $Code
}

$Model = 'D:\AI\LLM Models\custom\Qwen3.8-Flash-Next-MXFP4moe-ngramQ8-MTP.gguf'
if (-not (Test-Path -LiteralPath $Exe))   { Fail 2 "missing $Exe (run build-perstep.cmd)" }
if (-not (Test-Path -LiteralPath $Model)) { Fail 2 "missing $Model" }
if ($Crosscheck -and $Arm -ne 'A2')       { Fail 2 '-Crosscheck applies to the A2 arm only' }

$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listening) { Fail 3 "port $Port is in use (pid $($listening[0].OwningProcess)); stop it first" }

$ApiKey = ([regex]::Match((Get-Content 'D:\AI\llama-swap\config.yaml' -Raw), 'sk-lm-[A-Za-z0-9]+')).Value
if (-not $ApiKey) { Fail 2 'no API key found in D:\AI\llama-swap\config.yaml' }

# standing environment, then only this arm's SL-1 flags
$env:LONGSPEAR_VERIFY_TIMING = '1'
$env:LONGSPEAR_CG_REVIVE = '1'
foreach ($name in @('LONGSPEAR_OP_CENSUS', 'LONGSPEAR_CG_DEBUG', 'LONGSPEAR_CG_DEBUG2',
                    'LONGSPEAR_PER_STEP_PLE_TAIL', 'LONGSPEAR_SPEC_CKPT_MAX_TOKENS', 'LONGSPEAR_SPEC_CLAMP_TO_CKPT',
                    'LONGSPEAR_SPEC_CKPT_LEAN', 'LONGSPEAR_SPEC_HOST_TIMING', 'LONGSPEAR_SPEC_CKPT_CROSSCHECK')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$spec = '--spec-type ngram-mod:n_min=4 --spec-type mtp:n_max=4'
switch ($Arm) {
    'A2' {
        $ckpt = '--spec-ckpt-mode per-step'
        $env:LONGSPEAR_PER_STEP_PLE_TAIL    = '1'
        $env:LONGSPEAR_SPEC_CKPT_MAX_TOKENS = '5'
        $env:LONGSPEAR_SPEC_CLAMP_TO_CKPT   = '1'
        $env:LONGSPEAR_SPEC_CKPT_LEAN       = '1'
        $env:LONGSPEAR_SPEC_HOST_TIMING     = '1'
        if ($Crosscheck) { $env:LONGSPEAR_SPEC_CKPT_CROSSCHECK = '1' }
    }
    'P0' {
        $ckpt = '--spec-ckpt-mode gpu-fallback'
        $env:LONGSPEAR_SPEC_HOST_TIMING = '1'
    }
    'A0' {
        $spec = ''
        $ckpt = ''
    }
}

$argsx = "-m `"$Model`" --api-key $ApiKey -ngl 999 -ncmoe 37 -fa 1 -c 196608 -ub 512 -ctk q8_0 -ctv q8_0 -np 1 -t 24 -tb 32 --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --host 0.0.0.0 --port $Port $spec --reasoning-budget 1024 $ckpt -rtr -muge"

$out = Join-Path $LogDir "$LogName.out.log"
$err = Join-Path $LogDir "$LogName.err.log"

$p = Start-Process -FilePath $Exe -ArgumentList $argsx -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
Write-Output "sl1-launched arm=$Arm crosscheck=$([bool]$Crosscheck) pid=$($p.Id) port=$Port err=$err"
exit 0
