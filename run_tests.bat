@echo off
setlocal
title Run Test Suite - Trading Intelligence Platform
cd /d "%~dp0"

echo ==============================================================================
echo Executing Platform Test Suite (Pytest)...
echo ==============================================================================

if not exist ".venv\Scripts\pytest.exe" (
    echo [ERROR] Pytest not found in .venv\Scripts\pytest.exe
    pause
    exit /b 1
)

.venv\Scripts\pytest.exe %*
endlocal
