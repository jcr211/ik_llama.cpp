@echo off
rem State-OS lane 1 follow-ups (F11) build into build-stateos-f11 (same flags as build-l1.cmd).
rem Run: cmd /c D:\AI\worktrees\stateos-lane1-f11\.lane\build-f11.cmd   (writes .lane\build-f11.log)
setlocal EnableExtensions
set "SRC=%~dp0.."
set "BLD=%~dp0..\build-stateos-f11"
set "LOG=%~dp0build-f11.log"

call :run > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo build-f11 exit=%RC%
exit /b %RC%

:run
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not "%ERRORLEVEL%"=="0" exit /b 11

echo ==== configure
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_AVX512=ON "-DCMAKE_CUDA_ARCHITECTURES=120" -DLLAMA_BUILD_TESTS=ON
set "CFG_RC=%ERRORLEVEL%"
echo configure exit=%CFG_RC%
if not "%CFG_RC%"=="0" exit /b 12

echo ==== build
cmake --build "%BLD%" --target llama-server test-stateos-header test-speculative-params test-ple-hist test-stateos-layout -j 12
set "BUILD_RC=%ERRORLEVEL%"
echo build exit=%BUILD_RC%
if not "%BUILD_RC%"=="0" exit /b 13

if not exist "%BLD%\bin\llama-server.exe" exit /b 90
if not exist "%BLD%\bin\test-stateos-header.exe" exit /b 91
exit /b 0
