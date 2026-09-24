# SV2-E1 W-SV2 chain: port-ownership check, run after each launch-stateos-tail-8099.ps1 once the server
# answers, and BEFORE any traffic. httplib binds with SO_REUSEADDR, so on Windows a second server can bind
# :8099 while an old one still listens (the log shows no bind failure); requests may then reach either.
# Writes D:\AI\ik_llama-qwen4exp\<LogStem>.port with one line:
#   [stateos-port] ok pid=<launched> listeners=<pid list>          exactly one listener, the launched PID
#   [stateos-port] shared pid=<launched> listeners=<pid list>      another process listens too
#   [stateos-port] not-owned pid=<launched> listeners=<pid list>   the launched PID does not listen
# The census and the gate read it: anything but "ok" (or no .port file) = VOID "mislaunched".
# Exit 0 when ok, 3 otherwise.
param(
    [Parameter(Mandatory = $true)][string]$LogStem,
    [int]$Port = 8099
)
$ErrorActionPreference = 'Stop'
$base = 'D:\AI\ik_llama-qwen4exp\' + $LogStem
if (-not (Test-Path ($base + '.pid'))) { throw "missing $base.pid (not launched by launch-stateos-tail-8099.ps1)" }
$launched = [int](Get-Content -Raw ($base + '.pid')).Trim()
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
$list = ($listeners -join ',')
if ($listeners.Count -eq 1 -and $listeners[0] -eq $launched) {
    $state = 'ok'
} elseif ($listeners -contains $launched) {
    $state = 'shared'
} else {
    $state = 'not-owned'
}
$line = '[stateos-port] ' + $state + ' pid=' + $launched + ' listeners=' + $list + ' port=' + $Port
Set-Content -Path ($base + '.port') -Value $line -Encoding ascii
Write-Output $line
if ($state -ne 'ok') { exit 3 }
exit 0
