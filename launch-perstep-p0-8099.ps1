# SL-1 W-SL1 arm P0: production gpu-fallback on the build-sl1 binary, flags unset except
# LONGSPEAR_SPEC_HOST_TIMING=1 (the same telemetry cost as A2). Argument-free wrapper for
# native-replay.sh / bench-decode.sh.
& 'D:\AI\worktrees\sl1-spec-ckpt\launch-perstep-8099.ps1' -Arm P0
exit $LASTEXITCODE
