# Environment Specification

**Project**: MIDI Music Generation  
**Last Updated**: October 2025

---

## System Information

### Hardware
- **Device**: MacBook Air (M1, 2020)
- **Chip**: Apple M1 (8-core CPU, 7-core GPU)
- **Memory**: 8GB unified memory
- **Storage**: SSD

### Operating System
- **OS**: macOS Sequoia
- **Architecture**: ARM64 (Apple Silicon)

### Acceleration
- **Backend**: MPS (Metal Performance Shaders)
- **PyTorch MPS Support**: Enabled
- **GPU Acceleration**: Yes (via Metal)

---

## Python Environment

### Python Version
```bash
python --version
# Python 3.10.6 (or 3.10+)
```

### Virtual Environment
```bash
# Created with:
python3 -m venv .venv

# Activated with:
source .venv/bin/activate  # macOS/Linux
```

---

## Dependencies

### Core ML Libraries
```
torch==2.9.0              # PyTorch with MPS support
numpy==1.26.4             # Numerical computing (downgraded for sklearn compatibility)
scikit-learn==1.7.2       # Machine learning utilities
```

### Music Processing
```
miditok==3.0.0            # MIDI tokenization
music21==9.1.0            # Music analysis
pretty-midi==0.2.10       # MIDI manipulation
```

### Data & Utilities
```
pandas==2.2.3             # Data manipulation
matplotlib==3.9.2         # Plotting
seaborn==0.13.2           # Statistical visualization
tqdm==4.66.5              # Progress bars
```

### Complete List
See `requirements.txt` for full dependency list with exact versions.

---

## Installation Verification

### Check PyTorch MPS Support
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Should output:
# PyTorch version: 2.9.0
# MPS available: True
# MPS built: True
```

### Check MIDI Libraries
```python
import pretty_midi
import music21
import miditok

print("✅ All MIDI libraries imported successfully")
```

### Quick System Test
```python
import torch
import numpy as np

# Test MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(100, 100).to(device)
y = torch.mm(x, x.t())
print(f"✅ MPS computation works! Device: {device}")
```

---

## Performance Characteristics

### Training Speed (LSTM Generator, 971K params)
- **Per Epoch**: ~3 minutes (batch_size=128)
- **5 Epochs**: ~15 minutes
- **20 Epochs**: ~60 minutes

### Memory Usage
- **Training**: ~2-3 GB RAM
- **Peak**: ~4 GB RAM (with data loading)
- **Available**: 8 GB total (sufficient for local training)

### MPS vs CPU Speedup
- **MPS**: ~3x faster than CPU
- **Training**: ~3 min/epoch (MPS) vs ~9 min/epoch (CPU)

---

## Reproducibility Configuration

### Random Seeds
All experiments use fixed seeds for reproducibility:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set at start of all training scripts
set_seed(42)
```

### Deterministic Operations
- Fixed random seed (42)
- Deterministic data splitting
- Consistent preprocessing order
- No random augmentation (for reproducibility)

---

## Known Compatibility Issues

### NumPy Version
- **Issue**: sklearn incompatible with NumPy 2.x
- **Solution**: Pin to numpy==1.26.4
- **Status**: ✅ Resolved

### MPS Backend Limitations
- **Issue**: `num_workers > 0` causes multiprocessing errors
- **Solution**: Set `num_workers=0` in DataLoader
- **Status**: ✅ Resolved

### PyTorch DataLoader
- **Issue**: `pin_memory=True` not supported on MPS
- **Solution**: Set `pin_memory=False`
- **Status**: ✅ Resolved

---

## Installation Instructions

### Fresh Setup (Recommended)
```bash
# Clone repository
git clone https://github.com/csce585-mlsystems/csce585-midi.git
cd csce585-midi

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
```

### Troubleshooting

**If NumPy version error**:
```bash
pip install "numpy<2.0"
pip install scikit-learn --force-reinstall
```

**If MPS not available**:
- Ensure macOS 12.3+ (Monterey or later)
- Ensure PyTorch 1.12+ installed
- Check: `torch.backends.mps.is_available()`

**If MIDI library errors**:
```bash
pip install pretty-midi music21 miditok --force-reinstall
```

---

## Environment Variables

### Recommended Settings
```bash
# Disable telemetry (optional)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set number of threads (optional, for consistency)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### For Training
```bash
# Activate environment
source .venv/bin/activate

# Run training
python training/train_generator.py --model_type lstm --epochs 20
```

---

## Testing Environment Setup

### Quick Test Script
```bash
# Save as test_environment.sh
#!/bin/bash

echo "Testing environment setup..."

# Test Python
python --version

# Test imports
python << EOF
import torch
import numpy as np
import sklearn
import pretty_midi

print("✅ PyTorch:", torch.__version__)
print("✅ NumPy:", np.__version__)
print("✅ scikit-learn:", sklearn.__version__)
print("✅ MPS available:", torch.backends.mps.is_available())
EOF

echo "✅ Environment setup complete!"
```

---

## Version History

### October 2025 (Current)
- Python 3.10.6
- PyTorch 2.9.0
- NumPy 1.26.4
- MPS backend enabled

### Known Working Combinations
- ✅ macOS Sequoia + Python 3.10 + PyTorch 2.9.0
- ✅ MPS acceleration functional
- ✅ All MIDI libraries compatible

---

## Contact & Support

### Issues
If environment setup fails:
1. Check Python version (3.10+)
2. Verify macOS version (12.3+)
3. Ensure latest pip: `pip install --upgrade pip`
4. Try fresh virtual environment
5. Check `requirements.txt` for version conflicts

### Hardware Requirements
**Minimum**:
- 8GB RAM (for small models)
- Apple Silicon M1/M2 (for MPS) OR
- CUDA-capable GPU (for CUDA) OR
- CPU-only (slower)

**Recommended**:
- 16GB RAM (for larger models)
- Apple Silicon M1 Pro/Max or M2 Pro/Max
- SSD storage

---

## Appendix: Full Dependency List

Generated from `requirements.txt`:
```
torch==2.9.0
numpy==1.26.4
scikit-learn==1.7.2
miditok==3.0.0
music21==9.1.0
pretty-midi==0.2.10
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2
tqdm==4.66.5
```

All dependencies tested and confirmed working on Apple Silicon M1 with macOS Sequoia.
