# Run Full Test Suite
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Executing Platform Test Suite (Pytest)..." -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan

$Pytest = Join-Path $ScriptDir ".venv\Scripts\pytest.exe"
if (-not (Test-Path $Pytest)) {
    Write-Error "Pytest not found at $Pytest."
    exit 1
}

& $Pytest @args
