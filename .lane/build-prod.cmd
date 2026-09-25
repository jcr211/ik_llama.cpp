@echo off
rem Run only from this worktree. A local cache snapshot enables exact served-build flags.
setlocal EnableExtensions
set "SRC=%~dp0.."
set "CUDA_VISIBLE_DEVICES=-1"
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set "TEMP=%~dp0tmp"
set "TMP=%TEMP%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
if errorlevel 1 exit /b 11
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-prod.ps1"
exit /b %ERRORLEVEL%
