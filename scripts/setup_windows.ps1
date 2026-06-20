$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (Test-Path (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe")) {
    $pythonExe = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    $pythonArgs = @()
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
    $pythonArgs = @()
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
    $pythonArgs = @("-3.11")
} else {
    Write-Host "Python is not available in PATH."
    Write-Host "Install Python 3.11.x from https://www.python.org/downloads/windows/ and enable 'Add python.exe to PATH'."
    exit 1
}

Write-Host "Creating virtual environment..."
& $pythonExe @pythonArgs -m venv .venv

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing dependencies..."
& $venvPython -m pip install -e ".[dev]"

Write-Host ""
Write-Host "Setup complete. Run the app with:"
Write-Host "  .\scripts\run_app.ps1"
