# SV2-E1 W-SV2 step 2 (mechanism probe) launcher: tail snapshot + cross-check + telemetry.
# Thin wrapper over launch-stateos-tail-8099.ps1; same serving args, separate log stem.
param([string]$ExtraArgs = '')
& (Join-Path $PSScriptRoot 'launch-stateos-tail-8099.ps1') -DivLog -Tail -Xcheck -ExtraArgs $ExtraArgs -LogStem 'ik-serve-8099-stateos-tail-xcheck'
