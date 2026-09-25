# Production binary build attestation — 2026-09-25

- Source merge: `6f6b75c8a199c4f79dd2b4a992fb9730464cc348` on `lane/prod-f11-ple`.
- Parents: `fe5b967fdaca6705c05721812204b9f03169c3a7` and
  `fc3b0fbeb9bec1887f7e26c2d8f28cedc55630cc`. Merge was clean.
- Build directory: `build-avx512-prod-6f6b75c8`.
- Build command: `.lane/build-prod.cmd`, `-j 12`, exit 0. The script deleted `common/build-info.cpp`
  before configuring and regenerated it with `LLAMA_COMMIT="6f6b75c8"`.
- Cache source: `.lane/served-CMakeCache.txt`. The build script copied every selected `GGML_*`,
  `LLAMA_*`, build type, CUDA architecture/toolkit, compiler, and C/C++/CUDA flag entry through
  a CMake initial-cache file. `GGML_AVX512_BF16`, `GGML_AVX512_VBMI`, and
  `GGML_AVX512_VNNI` are all `ON`.
- Selected served-cache value diff: **empty**. `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`
  normalize from served `cl` to the full `cl.exe` path in the generated cache. CMake also adds
  its own `-ADVANCED`/`-STRINGS` metadata. `.lane/cache-diff.out` records the comparison.
- `llama-server.exe --version` (no model or port): `version: 4981 (6f6b75c8)`;
  `built with MSVC 19.44.35228.0 for`.

## SHA-256 (`build-avx512-prod-6f6b75c8/bin`)

| File | SHA-256 |
| --- | --- |
| `llama-server.exe` | `E56F4A7620006D81954CBDEF6E754974EE8D72713CCAFAD338609A8FC9A9A35E` |
| `ggml.dll` | `1F65AC7B96EB9E6FD18F7DB5275FFFBAA63388416B5F899B92C8D520BE697212` |
| `llama.dll` | `252CF67615FB52DF1D8A2CFD6BFB34818DEF9DC23B31F180B58110254A992D98` |
| `mtmd.dll` | `CA1BA06D7341AF63C9CA37DAACD24AFF85EB9BA3F206C57D34367E8AEECE265C` |

## Model-free verification

- `.lane/ctest.out`: 4/4 passed: `test-speculative-params`, `test-stateos-header`,
  `test-ple-hist`, and `test-stateos-layout`.
- Direct `test-stateos-header` with the repository's two vocab-only GGUF fixtures:
  291 checks, 0 failures. Output: `.lane/tmp/test-stateos-header.out`.
- `CUDA_VISIBLE_DEVICES=-1` and `.lane/tmp` as `TEMP`/`TMP` for verification.
- `LONGSPEAR_PLE_HIST_REWIND` remains enabled only when the environment value is exactly `1`.
