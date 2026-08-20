param(
    [Parameter(Mandatory = $true)]
    [string]$RawRoot,
    [string]$EngineRoot = "..\kt-aivle-big-proj-data-augmentation",
    [string]$Output = ".\work\smoke-quick-v16",
    [int]$PerGroup = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RawRoot = [IO.Path]::GetFullPath($RawRoot)
if (-not [IO.Path]::IsPathRooted($EngineRoot)) {
    $EngineRoot = Join-Path $RepoRoot $EngineRoot
}
if (-not [IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $RepoRoot $Output
}
$EngineRoot = [IO.Path]::GetFullPath($EngineRoot)
$Output = [IO.Path]::GetFullPath($Output)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $RawRoot -PathType Container)) {
    throw "RAW root not found: $RawRoot"
}
if (-not (Test-Path -LiteralPath $EngineRoot -PathType Container)) {
    throw "Failure engine root not found: $EngineRoot"
}
if (Test-Path -LiteralPath $Output) {
    $existing = @(Get-ChildItem -LiteralPath $Output -Force)
    if ($existing.Count -gt 0) {
        throw "Smoke output must be empty or absent: $Output"
    }
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
Set-Location -LiteralPath $RepoRoot

Write-Host "Quick smoke test: 12 CT/RGB x normal/defective x capture routes"
Write-Host "RAW: $RawRoot"
Write-Host "Output: $Output"

& $Python -X utf8 -m server_sim_dataset.cli --log-level INFO smoke `
    --raw-root $RawRoot `
    --cache ".\cache\raw_scan_v2.sqlite" `
    --plan-dir ".\work\plan-40ids" `
    --output $Output `
    --engine-root $EngineRoot `
    --per-group $PerGroup

if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed with exit code $LASTEXITCODE. See $Output\smoke_test_summary.json"
}

Write-Host "Smoke test passed. Summary: $Output\smoke_test_summary.json"
