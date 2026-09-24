@echo off
rem SV2-E1 (State-OS v2 tail snapshot) Release build into build-stateos-tail.
rem Same configure flags as D:\AI\ik_llama-qwen4exp\build-avx512-prod.cmd (Ninja, CUDA arch 120,
rem AVX-512 + VNNI/VBMI/BF16, cl) plus LLAMA_BUILD_TESTS=ON for the lane's unit tests.
rem The State-OS v2 flags are runtime env vars, so one build covers both flag states.
rem
rem Usage (from any directory): cmd /c D:\AI\worktrees\ik-stateos-tail\build-stateos-tail.cmd [targets...]
rem Default targets: llama-server test-partial-state test-stateos-tail.
rem Log: .lane\build-stateos-tail.log. Exit: 0 ok, 11 vcvars, 12 configure, 13 build, 90 artifact missing.
rem Each step's status is compared as a string, so a cmd errorlevel of -1 (which `if errorlevel 1`
rem does not catch) still fails the script, and every requested artifact must exist afterwards.
setlocal EnableExtensions
set "SOURCE_DIR=D:/AI/worktrees/ik-stateos-tail"
set "BUILD_DIR=D:/AI/worktrees/ik-stateos-tail/build-stateos-tail"
set "LOG=D:\AI\worktrees\ik-stateos-tail\.lane\build-stateos-tail.log"
set "TARGETS=%*"
if "%TARGETS%"=="" set "TARGETS=llama-server test-partial-state test-stateos-tail"

call :run > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo build-stateos-tail exit=%RC%
exit /b %RC%

:run
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not "%ERRORLEVEL%"=="0" exit /b 11

git -C "%SOURCE_DIR%" --no-pager rev-parse HEAD
git -C "%SOURCE_DIR%" --no-pager status --short

echo ==== configure
cmake -S "%SOURCE_DIR%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DLLAMA_BUILD_TESTS=ON
set "CFG_RC=%ERRORLEVEL%"
echo configure exit=%CFG_RC%
if not "%CFG_RC%"=="0" exit /b 12

echo ==== build %TARGETS%
cmake --build "%BUILD_DIR%" --config Release -j 24 --target %TARGETS%
set "BUILD_RC=%ERRORLEVEL%"
echo build exit=%BUILD_RC%
if not "%BUILD_RC%"=="0" exit /b 13

for %%T in (%TARGETS%) do (
    if not exist "%BUILD_DIR%/bin/%%T.exe" (
        echo missing artifact %BUILD_DIR%/bin/%%T.exe
        exit /b 90
    )
    for %%F in ("%BUILD_DIR%/bin/%%T.exe") do echo artifact %%~fF %%~zF bytes %%~tF
)
echo BUILD_STATEOS_TAIL_OK
exit /b 0
