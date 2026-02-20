#!/bin/bash
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spm_ml

echo "=========================================="
echo "  Generating Phase-Augmented Dataset"
echo "=========================================="
echo "Features:"
echo "  - Random initial phase per sample"
echo "  - 20480 samples (2048 bins x 10)"
echo "  - Output: sanity_check_20k_phase.hdf5"
echo ""

python3 hdf5_writer.py     --output ../../../TrainingData/sanity_check_20k_phase.hdf5     --sanity

echo ""
echo "=========================================="
if [ $? -eq 0 ]; then
    echo "  Generation completed!"
else
    echo "  Generation failed!"
fi
echo "=========================================="
read -p "Press Enter to close..."
