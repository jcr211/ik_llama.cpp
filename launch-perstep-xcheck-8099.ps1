# SL-1 W-SL1 step 1 probe: the A2 arm plus LONGSPEAR_SPEC_CKPT_CROSSCHECK=1 (diagnostic only; also
# allocates the full 112.57 MiB gpu-fallback shadow and redoes every rejected round by replay).
# Argument-free wrapper, so native-replay.sh can launch it by basename from D:\AI\ik_llama-qwen4exp.
& 'D:\AI\worktrees\sl1-spec-ckpt\launch-perstep-8099.ps1' -Arm A2 -Crosscheck
exit $LASTEXITCODE
