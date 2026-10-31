@echo off
setlocal
title Diagnostics - Trading Intelligence Platform
cd /d "%~dp0"

echo ==============================================================================
echo Running Platform Diagnostics...
echo ==============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python not found in .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo [1/2] Operational Status:
.venv\Scripts\python.exe scripts/paper_diagnostics.py --status
echo.
echo [2/2] Acceptance Verification:
.venv\Scripts\python.exe scripts/acceptance_verifier.py
endlocal
