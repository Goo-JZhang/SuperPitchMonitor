@echo off
REM Windows training script for RTX 4080S
REM Usage: run_training_windows.bat [train|train_live]

cd /d "%~dp0"

echo ==========================================
echo   SuperPitchMonitor Training (Windows)
echo ==========================================
echo.

REM Check conda installation
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: Conda not found in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Activate conda environment
call conda activate spm_ml
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'spm_ml'
    echo Please create the environment first:
    echo   conda create -n spm_ml python=3.10
    echo   conda activate spm_ml
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo   pip install h5py numpy matplotlib tqdm
    pause
    exit /b 1
)

REM Check CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
if errorlevel 1 (
    echo ERROR: PyTorch not properly installed
    pause
    exit /b 1
)

echo.
echo Starting training...
echo.

REM Choose script
if "%~1"=="live" (
    python train_live.py
) else (
    python train.py
)

echo.
echo ==========================================
if errorlevel 1 (
    echo   Training FAILED
) else (
    echo   Training COMPLETED
)
echo ==========================================
pause
