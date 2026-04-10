@echo off
setlocal

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_exe.ps1"
set "exit_code=%errorlevel%"

if not "%exit_code%"=="0" (
    echo.
    echo The EXE build failed with code %exit_code%.
    pause
)

exit /b %exit_code%
