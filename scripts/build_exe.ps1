param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pyinstaller = Join-Path $repoRoot ".venv\Scripts\pyinstaller.exe"
$spec = Join-Path $repoRoot "tabletennis_analyzer.spec"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
    exit 1
}

if (-not (Test-Path $pyinstaller)) {
    & $venvPython -m pip install pyinstaller
}

Set-Location $repoRoot

$argsList = @("--noconfirm", $spec)
if ($Clean) {
    $argsList = @("--clean") + $argsList
}

& $pyinstaller @argsList
exit $LASTEXITCODE

