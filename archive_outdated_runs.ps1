# Archive outdated run artifacts (safe cleanup)
# Moves non-essential files to runs\archive\outdated_<timestamp> and zips them.
# Keeps all ciwGAN checkpoints, TensorBoard logs, metrics, plots, and generated samples.

param(
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

Write-Host "=== Archiving outdated files (safe restore point) ===" -ForegroundColor Cyan

# Timestamp for unique archive directory and zip
$ts = (Get-Date).ToString('yyyyMMdd_HHmmss')
$archiveBase = Join-Path -Path "runs" -ChildPath "archive"
$archiveRoot = Join-Path -Path $archiveBase -ChildPath ("outdated_" + $ts)
$zipPath = Join-Path -Path $archiveBase -ChildPath ("outdated_" + $ts + ".zip")

# Ensure archive directories exist
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null

# Collect targets to archive
$targets = @()

# 1) Old TensorBoard logs (timecnn_*)
$tbRoot = Join-Path -Path "runs" -ChildPath "tb"
if (Test-Path $tbRoot) {
    $timecnnTB = Get-ChildItem -Path $tbRoot -Directory | Where-Object { $_.Name -match '^timecnn_' }
    foreach ($d in $timecnnTB) {
        $targets += @{ Type = 'dir'; Source = $d.FullName; Dest = (Join-Path -Path (Join-Path $archiveRoot 'tb') -ChildPath $d.Name) }
    }
}

# 2) Old experiment directories
$oldDirs = @('specgan','specgan_wsl','generated_samples','vot_debug','variants')
foreach ($name in $oldDirs) {
    $full = Join-Path -Path "runs" -ChildPath $name
    if (Test-Path $full) {
        $targets += @{ Type = 'dir'; Source = $full; Dest = (Join-Path -Path $archiveRoot -ChildPath $name) }
    }
}

# 3) Classifier artifacts (logreg_*) in runs root
$logregFiles = Get-ChildItem -Path "runs" -File | Where-Object { $_.Name -match '^logreg_' }
foreach ($f in $logregFiles) {
    $targets += @{ Type = 'file'; Source = $f.FullName; Dest = (Join-Path -Path (Join-Path $archiveRoot 'logreg') -ChildPath $f.Name) }
}

# 4) Redundant CSV files in runs root
$redundantCSVs = @(
    'vot.csv',
    'vot_variants.csv',
    'vot_real_vietnamese.csv',
    'vot_gen_long.csv',
    'vot_gen_short.csv',
    'intensity_gen_long.csv',
    'intensity_gen_short.csv'
)
foreach ($name in $redundantCSVs) {
    $full = Join-Path -Path "runs" -ChildPath $name
    if (Test-Path $full) {
        $targets += @{ Type = 'file'; Source = $full; Dest = (Join-Path -Path (Join-Path $archiveRoot 'redundant_csv') -ChildPath $name) }
    }
}

# Summary
$dirCount = ($targets | Where-Object { $_.Type -eq 'dir' }).Count
$fileCount = ($targets | Where-Object { $_.Type -eq 'file' }).Count
Write-Host ("Will archive: {0} directories and {1} files" -f $dirCount, $fileCount) -ForegroundColor Yellow
Write-Host ("Archive folder: {0}" -f $archiveRoot)
Write-Host ("Zip output   : {0}" -f $zipPath)

if ($WhatIfOnly) {
    Write-Host "WhatIfOnly set. Listing planned moves:" -ForegroundColor Yellow
    $targets | ForEach-Object {
        Write-Host ("  {0}: {1} -> {2}" -f $_.Type, $_.Source, $_.Dest)
    }
    exit 0
}

# Execute moves
foreach ($t in $targets) {
    try {
        $destDir = if ($t.Type -eq 'dir') { Split-Path -Path $t.Dest -Parent } else { Split-Path -Path $t.Dest -Parent }
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
        if ($t.Type -eq 'dir') {
            # Move whole directory
            Move-Item -Path $t.Source -Destination $t.Dest -Force
        } else {
            Move-Item -Path $t.Source -Destination $t.Dest -Force
        }
        Write-Host ("  ✓ Moved {0}" -f $t.Source) -ForegroundColor Green
    } catch {
        Write-Warning ("  ! Failed to move {0}: {1}" -f $t.Source, $_.Exception.Message)
    }
}

# Compute total size archived
$archSize = 0
if (Test-Path $archiveRoot) {
    $archSize = (Get-ChildItem -Path $archiveRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
}
$sizeMB = [math]::Round($archSize / 1MB, 2)
Write-Host ("Archived size: {0} MB" -f $sizeMB) -ForegroundColor Green

# Create zip
try {
    Write-Host "Creating zip..." -ForegroundColor Cyan
    if (Test-Path $zipPath) { Remove-Item -Path $zipPath -Force }
    Compress-Archive -Path (Join-Path $archiveRoot '*') -DestinationPath $zipPath -Force
    Write-Host ("  ✓ Zip created: {0}" -f $zipPath) -ForegroundColor Green
} catch {
    Write-Warning ("  ! Failed to create zip: {0}" -f $_.Exception.Message)
}

Write-Host "=== Archive complete ===" -ForegroundColor Cyan

# Post-check: show remaining TB runs (should be ciwgan_*)
if (Test-Path $tbRoot) {
    $remainingTB = Get-ChildItem -Path $tbRoot -Directory | Select-Object -ExpandProperty Name
    Write-Host "Remaining TensorBoard runs:" -ForegroundColor Yellow
    $remainingTB | ForEach-Object { Write-Host ("  - {0}" -f $_) }
}
