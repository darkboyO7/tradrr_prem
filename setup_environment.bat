@echo off
setlocal
title Environment Setup - Trading Intelligence Platform
cd /d "%~dp0"

echo ==============================================================================
echo Setting Up Trading Intelligence Platform Environment...
echo ==============================================================================

if not exist ".venv" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.12+ is installed.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists at .venv
)

echo [2/3] Installing dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [3/3] Installing editable project package...
.venv\Scripts\python.exe -m pip install -e . --no-deps
if errorlevel 1 (
    echo [ERROR] Failed to install package in editable mode.
    pause
    exit /b 1
)

echo ==============================================================================
echo Setup Complete! You can now run launch_app.bat or run_tests.bat
echo ==============================================================================
endlocal
