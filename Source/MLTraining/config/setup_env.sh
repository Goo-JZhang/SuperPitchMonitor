#!/bin/bash
# 手动创建spm_ml环境的脚本

set -e

echo "================================"
echo "SuperPitchMonitor ML Environment Setup"
echo "================================"

# 创建conda环境
echo "Creating conda environment 'spm_ml'..."
conda create -n spm_ml python=3.10 -y

# 激活环境
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spm_ml

# 安装PyTorch (Apple Silicon版本)
echo "Installing PyTorch..."
conda install pytorch::pytorch numpy scipy -y

# 使用pip安装其他包
echo "Installing other packages via pip..."
pip install --upgrade pip

pip install \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    onnx>=1.14.0 \
    onnxruntime>=1.15.0 \
    tensorboard>=2.13.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0 \
    scikit-learn>=1.3.0

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "To activate the environment:"
echo "  conda activate spm_ml"
echo ""
echo "To export the stub model:"
echo "  cd Source/MLTraining/model_export/"
echo "  python export_stub_model.py"
