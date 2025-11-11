#!/bin/bash

# Google Colab Setup Script for A100 Training
# Run this first in Colab to set up the environment

echo "=========================================="
echo "Setting up Google Colab for A100 Training"
echo "=========================================="
echo ""

# Check GPU
echo "Checking GPU..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"None\"}')"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q miditok symusic tqdm matplotlib

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Upload your data or run preprocessing:"
echo "     python utils/augment_miditok.py"
echo ""
echo "  2. Start training:"
echo "     bash scripts/train_a100.sh"
echo ""
echo "  3. Or run individual models (see docs/A100_TRAINING.md)"
echo ""
