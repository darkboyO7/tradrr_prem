# Launch Trading Intelligence Desktop Application
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Launching Trading Intelligence Desktop Application..." -ForegroundColor Green
Write-Host "Target Scope: NSE, BSE, Quotex Research Only" -ForegroundColor Yellow
Write-Host "Platform: Native Windows 10/11" -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Cyan

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at $VenvPython. Run setup_environment.ps1 first."
    exit 1
}

& $VenvPython -m desktop.app @args
