@echo off
setlocal
title Daily Health Check - Trading Intelligence Platform
cd /d "%~dp0"

echo ==============================================================================
echo Generating Prospective Paper-Trading and Platform Health Report...
echo ==============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python not found in .venv\Scripts\python.exe
    pause
    exit /b 1
)

.venv\Scripts\python.exe scripts/paper_diagnostics.py --health-report
endlocal
