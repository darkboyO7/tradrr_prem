# Setup Virtual Environment and Dependencies
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Setting Up Trading Intelligence Platform Environment..." -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan

$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[1/3] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/3] Virtual environment already exists at $VenvDir" -ForegroundColor Green
}

Write-Host "[2/3] Installing pinned dependencies from requirements.txt..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host "[3/3] Installing package in editable mode..." -ForegroundColor Yellow
& $VenvPython -m pip install -e . --no-deps

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Setup Complete! You can now run .\launch_app.ps1 or .\run_tests.ps1" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
