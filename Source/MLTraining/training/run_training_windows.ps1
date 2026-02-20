# Windows PowerShell training script for RTX 4080S
# Usage: .\run_training_windows.ps1 [-Live]

param(
    [switch]$Live
)

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  SuperPitchMonitor Training (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check conda
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    Write-Host "ERROR: Conda not found in PATH" -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda first"
    exit 1
}

# Activate environment
& conda activate spm_ml
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate conda environment 'spm_ml'" -ForegroundColor Red
    exit 1
}

# Check CUDA
Write-Host "Checking GPU..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Run training
if ($Live) {
    python train_live.py
} else {
    python train.py
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Training COMPLETED" -ForegroundColor Green
} else {
    Write-Host "  Training FAILED" -ForegroundColor Red
}
Write-Host "==========================================" -ForegroundColor Cyan
