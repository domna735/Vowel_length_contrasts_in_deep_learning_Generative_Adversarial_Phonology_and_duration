# Cleanup script for GAN_2025 project
# This removes old classifier experiments and outdated files
# Keeps only ciwGAN-related results for PhD application

Write-Host "=== GAN_2025 Cleanup Script ===" -ForegroundColor Cyan
Write-Host "This will DELETE old files not needed for PhD work" -ForegroundColor Yellow
Write-Host ""

# Calculate sizes before cleanup
$oldClassifierFiles = Get-ChildItem -Path "runs" -Recurse -File | Where-Object { $_.Name -match "^logreg_" }
$oldTBDirs = Get-ChildItem -Path "runs\tb" -Directory | Where-Object { $_.Name -match "^timecnn_" }
$oldDirs = @("runs\specgan", "runs\specgan_wsl", "runs\generated_samples", "runs\vot_debug", "runs\variants")

$totalSize = 0
$totalSize += ($oldClassifierFiles | Measure-Object -Property Length -Sum).Sum
$totalSize += ($oldTBDirs | Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum).Sum
foreach ($dir in $oldDirs) {
    if (Test-Path $dir) {
        $totalSize += (Get-ChildItem -Path $dir -Recurse -File | Measure-Object -Property Length -Sum).Sum
    }
}

$sizeMB = [math]::Round($totalSize / 1MB, 2)
Write-Host "Total space to be freed: $sizeMB MB" -ForegroundColor Green
Write-Host ""

# List what will be deleted
Write-Host "Files to be deleted:" -ForegroundColor Yellow
Write-Host "  - 60+ logistic regression classifier files (logreg_*.json, *.pkl, *.csv, *.h5)"
Write-Host "  - 17 TensorBoard folders from old timecnn experiments"
Write-Host "  - runs\specgan\ (40 old SpecGAN files)"
Write-Host "  - runs\specgan_wsl\ (WSL experiments)"
Write-Host "  - runs\generated_samples\ (old samples)"
Write-Host "  - runs\vot_debug\ (debug files)"
Write-Host "  - runs\variants\ (experimental variants)"
Write-Host "  - 5 redundant CSV files (vot.csv, vot_variants.csv, vot_gen_*.csv, intensity_gen_*.csv)"
Write-Host ""

Write-Host "Files to be KEPT:" -ForegroundColor Green
Write-Host "  - All 4 ciwGAN checkpoints (pilot, 30ep, 100ep)"
Write-Host "  - All 4 ciwGAN TensorBoard folders (for PhD graphs)"
Write-Host "  - All generated samples in runs\gen\"
Write-Host "  - All comparison CSVs in runs\compare\"
Write-Host "  - All plots in runs\plots\"
Write-Host "  - All current VOT/intensity CSVs (real, 30ep, 100ep, normalized)"
Write-Host ""

# Ask for confirmation
$confirm = Read-Host "Do you want to proceed with cleanup? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Cleanup cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Starting cleanup..." -ForegroundColor Cyan

# Delete old classifier files
Write-Host "Deleting logistic regression classifier files..." -ForegroundColor Yellow
Get-ChildItem -Path "runs" -File | Where-Object { $_.Name -match "^logreg_" } | Remove-Item -Force
Write-Host "  ✓ Deleted classifier files" -ForegroundColor Green

# Delete old TensorBoard timecnn folders
Write-Host "Deleting old TensorBoard timecnn folders..." -ForegroundColor Yellow
Get-ChildItem -Path "runs\tb" -Directory | Where-Object { $_.Name -match "^timecnn_" } | Remove-Item -Recurse -Force
Write-Host "  ✓ Deleted timecnn TensorBoard folders" -ForegroundColor Green

# Delete old directories
Write-Host "Deleting old experiment directories..." -ForegroundColor Yellow
$oldDirs = @("runs\specgan", "runs\specgan_wsl", "runs\generated_samples", "runs\vot_debug", "runs\variants")
foreach ($dir in $oldDirs) {
    if (Test-Path $dir) {
        Remove-Item -Path $dir -Recurse -Force
        Write-Host "  ✓ Deleted $dir" -ForegroundColor Green
    }
}

# Delete redundant CSV files
Write-Host "Deleting redundant CSV files..." -ForegroundColor Yellow
$redundantCSVs = @(
    "runs\vot.csv",
    "runs\vot_variants.csv", 
    "runs\vot_real_vietnamese.csv",
    "runs\vot_gen_long.csv",
    "runs\vot_gen_short.csv",
    "runs\intensity_gen_long.csv",
    "runs\intensity_gen_short.csv"
)
foreach ($csv in $redundantCSVs) {
    if (Test-Path $csv) {
        Remove-Item -Path $csv -Force
        Write-Host "  ✓ Deleted $(Split-Path -Leaf $csv)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== Cleanup Complete ===" -ForegroundColor Green
Write-Host "Freed up approximately $sizeMB MB of disk space" -ForegroundColor Green
Write-Host ""
Write-Host "Remaining files are all relevant to ciwGAN PhD work:" -ForegroundColor Cyan
Write-Host "  ✓ 4 ciwGAN checkpoints (pilot, 30ep, 100ep)"
Write-Host "  ✓ 4 ciwGAN TensorBoard folders"
Write-Host "  ✓ All generated samples (runs\gen\)"
Write-Host "  ✓ All comparison CSVs (runs\compare\)"
Write-Host "  ✓ All plots (runs\plots\)"
Write-Host "  ✓ Current VOT/intensity metrics"
Write-Host ""
Write-Host "TensorBoard will now only show 4 clean runs:" -ForegroundColor Yellow
Write-Host "  - ciwgan_20251109T044338Z (1-epoch pilot)"
Write-Host "  - ciwgan_20251109T060925Z (30-epoch baseline)"
Write-Host "  - ciwgan_20251109T065758Z (failed 100-epoch)"
Write-Host "  - ciwgan_20251109T071313Z (final 100-epoch) ← Use this for PhD"
Write-Host ""
Write-Host "You can restart TensorBoard with:" -ForegroundColor Cyan
Write-Host "  tensorboard --logdir=runs\tb\ciwgan_20251109T071313Z --port=6006" -ForegroundColor White
Write-Host "  OR to see all 4 runs:" -ForegroundColor Cyan
Write-Host "  tensorboard --logdir=runs\tb --port=6006" -ForegroundColor White
