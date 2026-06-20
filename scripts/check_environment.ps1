$ErrorActionPreference = "Stop"

Write-Host "Checking Table Tennis Analyzer environment..."

function Test-Command($Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Host "[missing] $Name"
        return $false
    }
    Write-Host "[ok] $Name -> $($cmd.Source)"
    return $true
}

function Test-PythonExecutable($Command, $Arguments) {
    try {
        $version = & $Command @Arguments --version 2>$null
        if ($LASTEXITCODE -eq 0 -and ($version -match "Python 3\.(11|12)\.")) {
            Write-Host "[ok] python -> $version"
            return $true
        } elseif ($LASTEXITCODE -eq 0) {
            Write-Host "[unsupported] python -> $version; expected Python 3.11 or 3.12"
        }
    } catch {
        return $false
    }
    return $false
}

$hasPython = $false
$localPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
if ((Test-Path $localPython) -and (Test-PythonExecutable $localPython @())) {
    $hasPython = $true
} elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-PythonExecutable "python" @())) {
    $hasPython = $true
} elseif ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-PythonExecutable "py" @("-3.11"))) {
    $hasPython = $true
} else {
    Write-Host "[missing] compatible Python"
}

$hasFfmpeg = Test-Command "ffmpeg"
if (-not $hasFfmpeg) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $localFfmpeg = Get-ChildItem -Path (Join-Path $repoRoot ".tools\ffmpeg") -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $localFfmpeg) {
        Write-Host "[ok] ffmpeg -> $($localFfmpeg.FullName)"
    }
}

if (-not $hasPython) {
    Write-Host ""
    Write-Host "Python is not available in PATH. Install Python 3.11.x, then rerun scripts\setup_windows.ps1."
    exit 1
}

Write-Host ""
Write-Host "Environment check finished."
