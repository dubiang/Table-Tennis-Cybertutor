param(
    [Parameter(Mandatory = $true)]
    [string]$UserVideo,

    [Parameter(Mandatory = $true)]
    [string]$CoachVideo,

    [string]$OutputDir = "outputs\user_coach_motion",

    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$localFfmpeg = Get-ChildItem -Path (Join-Path $repoRoot ".tools\ffmpeg") -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1

if ($null -ne $localFfmpeg) {
    $env:Path = "$($localFfmpeg.DirectoryName);$env:Path"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
    exit 1
}

$argsList = @(
    "-m",
    "tabletennis_analyzer.cli",
    "pair",
    "--user-video",
    $UserVideo,
    "--coach-video",
    $CoachVideo,
    "--output-dir",
    $OutputDir
)

if ($MaxFrames -gt 0) {
    $argsList += @("--max-frames", $MaxFrames)
}

& $venvPython @argsList
exit $LASTEXITCODE
