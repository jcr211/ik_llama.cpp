# SV2-E1 W-SV2 step 2 (mechanism probe) launcher: tail snapshot + cross-check + telemetry.
# Thin wrapper over launch-stateos-tail-8099.ps1; same serving args. -LogStem is mandatory and new.
param(
    [Parameter(Mandatory = $true)][string]$LogStem,
    [string]$ExtraArgs = '',
    [string]$LockOwner = 'W-SV2 chain'
)
& (Join-Path $PSScriptRoot 'launch-stateos-tail-8099.ps1') -DivLog -Tail -Xcheck -ExtraArgs $ExtraArgs -LogStem $LogStem -LockOwner $LockOwner
