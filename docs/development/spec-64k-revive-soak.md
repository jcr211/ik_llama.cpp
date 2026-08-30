# 64K speculative CUDA-graph revival soak

This coordinator-only soak reproduces the production shape without making model execution part of the test suite. Run it only in an exclusive GPU window from `fix/spec-64k-illegal-access`.

## Build

```powershell
cmake --build .\build-census --config Release --target llama-server -j 24
```

## Server

The arguments below match the standing `:8099` census service. Keep `LONGSPEAR_CG_DEBUG2` unset: DEBUG2 already bypasses the UID shortcut and would mask this regression.

```powershell
$env:LONGSPEAR_VERIFY_TIMING = '1'
$env:LONGSPEAR_CG_REVIVE = '1'
Remove-Item Env:LONGSPEAR_CG_DEBUG -ErrorAction SilentlyContinue
Remove-Item Env:LONGSPEAR_CG_DEBUG2 -ErrorAction SilentlyContinue
Remove-Item Env:LONGSPEAR_OP_CENSUS -ErrorAction SilentlyContinue

.\build-census\bin\llama-server.exe `
  -m "D:\AI\LLM Models\custom\Qwen3.8-Flash-Next-MXFP4moe-ngramQ8-MTP.gguf" `
  -ngl 999 -ncmoe 38 -fa 1 -c 196608 -ub 512 -ctk q8_0 -ctv q8_0 `
  -np 1 -t 24 -tb 32 --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 `
  --host 127.0.0.1 --port 8099 `
  --spec-type ngram-mod:n_min=4 --spec-type mtp:n_max=4 `
  --reasoning-budget 1024 --spec-ckpt-mode gpu-fallback -rtr -muge `
  1> .\spec-64k-soak.out.log 2> .\spec-64k-soak.err.log
```

## Driver

In a second PowerShell window:

```powershell
python .\scripts\spec-64k-soak.py `
  --base-url http://127.0.0.1:8099 `
  --seed-tokens 66000 `
  --target-tokens 105000 `
  --chunk 4096 `
  2>&1 | Tee-Object .\spec-64k-soak.driver.log
```

The driver creates a repetitive prompt just above 66K tokens, then repeatedly extends the same single-slot cached prompt with deterministic 4K-token completions. This gives `ngram-mod` enough repeated history to propose its full 16-token draft and keeps speculative verification active while the KV depth crosses 100K.

Pass criteria:

- the driver emits `{"event": "complete", ...}` with `prompt_tokens >= 105000`;
- the verify trace contains repeated `K=17` rows both below and above 65,536, and rows above 100,000;
- there is no CUDA error, server exit, checkpoint failure, or context shift;
- run the same recipe twice from a fresh server process because the failure required sustained graph-build churn.

For a negative-control confirmation, build the parent commit, run the same recipe with the same environment, and retain the failing error-log tail if it reproduces. Do not enable DEBUG2 for either side.
