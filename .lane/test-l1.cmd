@echo off
rem State-OS lane 1 non-GPU tests. CUDA devices are hidden so nothing can touch the GPU.
rem Run: cmd /c D:\AI\worktrees\stateos-lane1\.lane\test-l1.cmd   (writes .lane\test-l1.log)
setlocal EnableExtensions
set "SRC=%~dp0.."
set "BLD=%~dp0..\build-stateos-l1"
set "LOG=%~dp0test-l1.log"
set "CUDA_VISIBLE_DEVICES=-1"

call :run > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo test-l1 exit=%RC%
exit /b %RC%

:run
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
if not "%ERRORLEVEL%"=="0" exit /b 11

echo CUDA_VISIBLE_DEVICES=[%CUDA_VISIBLE_DEVICES%]
echo ==== test-stateos-header (direct)
"%BLD%\bin\test-stateos-header.exe" "%SRC%\models\ggml-vocab-qwen2.gguf" "%SRC%\models\ggml-vocab-llama-bpe.gguf"
set "T1=%ERRORLEVEL%"
echo test-stateos-header exit=%T1%

echo ==== ctest
ctest --test-dir "%BLD%" -R "test-stateos-header|test-speculative-params|test-ple-hist" --output-on-failure
set "T2=%ERRORLEVEL%"
echo ctest exit=%T2%

if not "%T1%"=="0" exit /b 21
if not "%T2%"=="0" exit /b 22
exit /b 0
