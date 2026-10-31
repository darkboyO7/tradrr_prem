@echo off
setlocal
title Trading Intelligence Platform Desktop
cd /d "%~dp0"

echo ==============================================================================
echo Launching Trading Intelligence Desktop Application...
echo Target Scope: NSE, BSE, Quotex Research Only
echo Platform: Native Windows 10/11
echo ==============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please run setup_environment.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m desktop.app %*
endlocal
