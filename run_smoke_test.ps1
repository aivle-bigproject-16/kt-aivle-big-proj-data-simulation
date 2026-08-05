param(
    [string]$Python = "C:\Users\User\Documents\Codex\rgb-augmentation-venv\Scripts\python.exe",
    [string]$RawRoot = "E:\103.배터리 불량 이미지 데이터",
    [string]$Cache = ".\cache\raw_scan_v1.sqlite",
    [string]$PlanDir = "E:\server-simulation-v1.2-plan",
    [string]$Output = "E:\server-simulation-v1.2-smoke",
    [string]$EngineRoot = "C:\Users\User\Documents\Codex\2026-07-22\aivle-bigproject-16-kt-aivle-big\work\kt-aivle-big-proj-data-augmentation-v1.9-severe",
    [int]$PerGroup = 2,
    [switch]$RefreshCache,
    [switch]$FullScan
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repo
$sourceRoot = Join-Path -Path $repo -ChildPath "src"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $RawRoot -PathType Container)) {
    throw "Raw data root not found: $RawRoot"
}
if (-not (Test-Path -LiteralPath $EngineRoot -PathType Container)) {
    throw "quality-fail-augment v2.0 engine root not found: $EngineRoot"
}
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Python source directory not found: $sourceRoot"
}

# src-layout 패키지를 pip install하지 않았어도 현재 저장소 코드를 직접 찾게 한다.
$previousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
    $env:PYTHONPATH = $sourceRoot
} else {
    $env:PYTHONPATH = "$sourceRoot$([IO.Path]::PathSeparator)$previousPythonPath"
}

& $Python -c "import server_sim_dataset, server_sim_dataset.cli; print('server_sim_dataset import: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Cannot import server_sim_dataset from: $sourceRoot"
}

$arguments = @(
    "-m", "server_sim_dataset.cli", "smoke",
    "--raw-root", $RawRoot,
    "--cache", $Cache,
    "--plan-dir", $PlanDir,
    "--output", $Output,
    "--engine-root", $EngineRoot,
    "--per-group", [string]$PerGroup
)
if ($RefreshCache) {
    $arguments += "--refresh-cache"
}
if ($FullScan) {
    $arguments += "--full-scan"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Smoke test process failed with exit code $LASTEXITCODE"
}

$summaryPath = Join-Path -Path $Output -ChildPath "smoke_test_summary.json"
if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
    throw "Smoke summary was not created: $summaryPath"
}
$summary = Get-Content -LiteralPath $summaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
if ($summary.status -ne "passed") {
    throw "Smoke test did not pass. See: $summaryPath"
}

Write-Host "Smoke test PASSED" -ForegroundColor Green
Write-Host "Summary: $summaryPath"
Write-Host "Generated samples: $($summary.generation.succeeded)"
