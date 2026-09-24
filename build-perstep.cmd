@echo off
rem SL-1 build: tail-aware PER_STEP speculative checkpoints (lane/sl1-spec-ckpt).
rem Same configure flags as build-avx512-prod.cmd (AVX-512 + VNNI/VBMI/BF16, CUDA sm_120, Release,
rem OpenMP, cl), in its own build dir build-sl1 next to this script. Every new behaviour is behind a
rem LONGSPEAR_* environment flag, so this one binary serves every W-SL1 arm.
rem Exit codes are compared as strings: "if errorlevel 1" misses a -1 from a crashed tool.
rem Usage: build-perstep.cmd            (server + the SL-1 CPU tests)
rem        build-perstep.cmd server     (llama-server only)
setlocal EnableExtensions

set "SOURCE_DIR=%~dp0"
set "SOURCE_DIR=%SOURCE_DIR:\=/%"
set "SOURCE_DIR=%SOURCE_DIR:~0,-1%"
set "BUILD_DIR=%SOURCE_DIR%/build-sl1"
set "CUDA13_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

set "TARGETS=llama-server test-ple-perstep test-iqk-moe-chunks test-spec-ckpt-sampler test-spec-ckpt-clamp"
if /i "%~1"=="server" set "TARGETS=llama-server"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "STEP_RC=%ERRORLEVEL%"
if not "%STEP_RC%"=="0" (
    echo BUILD_PERSTEP_FAIL vcvars rc=%STEP_RC%
    exit /b 11
)

git -C "%SOURCE_DIR%" --no-pager rev-parse HEAD
git -C "%SOURCE_DIR%" --no-pager status --short --untracked-files=no

cmake -S "%SOURCE_DIR%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl "-DCMAKE_CUDA_COMPILER=%CUDA13_ROOT%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA13_ROOT%"
set "STEP_RC=%ERRORLEVEL%"
if not "%STEP_RC%"=="0" (
    echo BUILD_PERSTEP_FAIL configure rc=%STEP_RC%
    exit /b 12
)

cmake --build "%BUILD_DIR%" --config Release -j 24 --target %TARGETS%
set "STEP_RC=%ERRORLEVEL%"
if not "%STEP_RC%"=="0" (
    echo BUILD_PERSTEP_FAIL build rc=%STEP_RC%
    exit /b 13
)

set "BIN_DIR=%BUILD_DIR:/=\%\bin"
if not exist "%BIN_DIR%\llama-server.exe" (
    echo BUILD_PERSTEP_FAIL llama-server.exe missing
    exit /b 14
)

echo BUILD_PERSTEP_OK %BUILD_DIR%/bin
exit /b 0
