param(
    [Parameter(Mandatory = $true)]
    [string]$Video,

    [string]$OutputDir = "outputs\single_video",

    [string]$Label = "video",

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

$argsList = @("-m", "tabletennis_analyzer.cli", "single", $Video, "--output-dir", $OutputDir, "--label", $Label)
if ($MaxFrames -gt 0) {
    $argsList += @("--max-frames", $MaxFrames)
}

& $venvPython @argsList
exit $LASTEXITCODE
