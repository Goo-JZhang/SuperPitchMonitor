# SuperPitchMonitor ML Training Environment

## Environment Info

- **Environment Name**: `spm_ml`
- **Python Version**: 3.10.19
- **Platform**: macOS Apple Silicon (ARM64)

## Quick Setup

### Option 1: Using environment.yml (Recommended)

```bash
# Clone the repository first
git clone <repository-url>
cd SuperPitchMonitor

# Create environment from file
conda env create -f Source/MLTraining/environment.yml

# Activate environment
conda activate spm_ml
```

### Option 2: Manual Setup

```bash
# Create new environment
conda create -n spm_ml python=3.10.19

# Activate environment
conda activate spm_ml

# Install PyTorch (CPU version for macOS)
conda install pytorch=2.5.1 -c pytorch

# Install other conda packages
conda install numpy=2.2.5 h5py pyyaml tqdm

# Install pip packages
pip install onnx==1.20.1 onnxruntime==1.23.2 matplotlib==3.10.8
```

## Package Versions

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.10.19 | Python runtime |
| pytorch | 2.5.1 | Deep learning framework |
| numpy | 2.2.5 | Numerical computing |
| onnx | 1.20.1 | Model export format |
| onnxruntime | 1.23.2 | Inference engine |

### Visualization & Data

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | 3.10.8 | Training visualization |
| h5py | 3.15.1 | Dataset storage (legacy) |
| tqdm | 4.67.3 | Progress bars |

### System Dependencies

- **BLAS**: OpenBLAS (for numpy matrix operations)
- **libgfortran**: 15.2.0 (Fortran runtime for BLAS)

## Verify Installation

```bash
# Activate environment
conda activate spm_ml

# Test imports
python3 << 'EOF'
import torch
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
EOF
```

## Export Current Environment

If you need to export your current environment:

```bash
conda activate spm_ml
conda env export --no-builds > environment.yml
```

## Notes

- **macOS Apple Silicon**: PyTorch uses MPS (Metal Performance Shaders) for GPU acceleration
- **Windows/Linux**: Install CUDA-enabled PyTorch for GPU support
- **Data Generation**: Uses NumPy only, no PyTorch required
- **Training**: Requires PyTorch with MPS/CUDA support
