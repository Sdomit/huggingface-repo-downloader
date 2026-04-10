@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    python -m hf_downloader
    set "exit_code=%errorlevel%"
    goto :done
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.13 -m hf_downloader
    set "exit_code=%errorlevel%"
    if "%exit_code%"=="0" goto :done

    py -3 -m hf_downloader
    set "exit_code=%errorlevel%"
    goto :done
)

echo Python was not found. Install Python or use the py launcher.
set "exit_code=1"

:done
if not "%exit_code%"=="0" (
    echo.
    echo The launcher exited with code %exit_code%.
    pause
)

exit /b %exit_code%
