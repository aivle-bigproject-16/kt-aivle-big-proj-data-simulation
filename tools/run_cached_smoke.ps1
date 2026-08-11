param(
    [string]$CachePath = "$env:USERPROFILE\Downloads\raw_scan_v2.sqlite",
    [string]$RawRoot = "",
    [string]$EngineRoot = "",
    [string]$WorkRoot = ".\work\cached-smoke",
    [int]$PerGroup = 2,
    [switch]$GenerateSamples
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$cache = [System.IO.Path]::GetFullPath($CachePath)
$engine = if ($EngineRoot) { [System.IO.Path]::GetFullPath($EngineRoot) } else { "" }
$work = if ([System.IO.Path]::IsPathRooted($WorkRoot)) {
    [System.IO.Path]::GetFullPath($WorkRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $WorkRoot))
}
$planDir = Join-Path $work "plan"
$smokeOutput = Join-Path $work "smoke"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python venv not found: $python`nCreate it with: py -3.13 -m venv .venv"
}
if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) {
    throw "Scan cache not found: $cache"
}
if ($PerGroup -lt 1) {
    throw "PerGroup must be at least 1: $PerGroup"
}
if (Test-Path -LiteralPath $work) {
    $existing = @(Get-ChildItem -LiteralPath $work -Force)
    if ($existing.Count -gt 0) {
        throw "Smoke work directory is not empty: $work`nUse a new -WorkRoot value."
    }
}

$cacheMetadataJson = & $python -X utf8 -c `
    "import sqlite3,sys,json; db=sqlite3.connect(sys.argv[1]); print(json.dumps(dict(db.execute('select key,value from metadata'))))" `
    $cache
$cacheMetadata = $cacheMetadataJson | ConvertFrom-Json
if ($cacheMetadata.cache_version -ne "2") {
    throw "Unsupported scan cache version '$($cacheMetadata.cache_version)'; expected version 2."
}

if ($GenerateSamples) {
    if (-not $EngineRoot -or -not (Test-Path -LiteralPath $engine -PathType Container)) {
        throw "GenerateSamples requires a valid -EngineRoot: $engine"
    }
    if (-not $RawRoot) {
        $RawRoot = $cacheMetadata.raw_root
    }
    $raw = [System.IO.Path]::GetFullPath($RawRoot.Trim())
    if (-not (Test-Path -LiteralPath $raw -PathType Container)) {
        throw "Raw root does not exist: $raw`nPlan-only preflight can use the downloaded cache, but generation smoke requires local raw images."
    }
    $cachedRaw = & $python -X utf8 -c `
        "import sqlite3,sys,pathlib; db=sqlite3.connect(sys.argv[1]); print(pathlib.Path(dict(db.execute('select key,value from metadata'))['raw_root']).resolve())" `
        $cache
    $resolvedRaw = & $python -X utf8 -c "import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())" $raw
    if ($cachedRaw.Trim() -ne $resolvedRaw.Trim()) {
        throw "Cache/raw-root mismatch. Cache records '$cachedRaw', but -RawRoot resolves to '$resolvedRaw'. Rebuild a local cache before generation smoke."
    }
}

Push-Location $repoRoot
try {
    Write-Host "[1/2] Building the full plan directly from the existing cache..."
    & $python -X utf8 -c `
        "import sys; from pathlib import Path; from server_sim_dataset.planner import build_plan; from server_sim_dataset.reports import feasibility_audit; cache=Path(sys.argv[1]); out=Path(sys.argv[2]); feasibility_audit(cache,out.parent/'raw_extraction_feasibility.json'); build_plan(cache,out)" `
        $cache $planDir
    if ($LASTEXITCODE -ne 0) { throw "Plan preflight failed with exit code $LASTEXITCODE" }

    if (-not $GenerateSamples) {
        Write-Host "Plan preflight PASSED: $planDir" -ForegroundColor Green
        Write-Host "Add -GenerateSamples -RawRoot <path> -EngineRoot <path> for image generation smoke."
        return
    }

    Write-Host "[2/2] Generating CT/RGB PASS, FAIL, and recapture smoke samples..."
    & $python -X utf8 -m server_sim_dataset.cli smoke `
        --raw-root $raw `
        --cache $cache `
        --plan-dir $planDir `
        --output $smokeOutput `
        --engine-root $engine `
        --per-group $PerGroup `
        --full-scan
    if ($LASTEXITCODE -ne 0) { throw "Smoke generation failed with exit code $LASTEXITCODE" }

    $summary = Join-Path $smokeOutput "smoke_test_summary.json"
    if (-not (Test-Path -LiteralPath $summary -PathType Leaf)) {
        throw "Smoke summary was not created: $summary"
    }
    $result = Get-Content -LiteralPath $summary -Raw -Encoding utf8 | ConvertFrom-Json
    if ($result.status -ne "passed") {
        throw "Smoke summary status is '$($result.status)': $summary"
    }
    $expectedRoutes = @(
        "CT_initial_capture_PASS", "CT_initial_capture_FAIL", "CT_recapture_PASS",
        "RGB_initial_capture_PASS", "RGB_initial_capture_FAIL", "RGB_recapture_PASS"
    )
    foreach ($route in $expectedRoutes) {
        $count = $result.route_counts.$route
        if ($count -ne $PerGroup) {
            throw "Smoke route '$route' has $count samples; expected $PerGroup."
        }
    }
    if ($result.route_counts.total -ne ($PerGroup * $expectedRoutes.Count)) {
        throw "Smoke total is $($result.route_counts.total); expected $($PerGroup * $expectedRoutes.Count)."
    }
    Write-Host "Smoke PASSED: $summary" -ForegroundColor Green
}
finally {
    Pop-Location
}
