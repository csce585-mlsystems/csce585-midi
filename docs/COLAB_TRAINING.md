# Google Colab A100 Training Guide

This guide explains how to train large music generation models on Google Colab's A100 GPU with automatic checkpoint saving to Google Drive.

## Prerequisites

1. **Google Colab Account** with A100 GPU access (Colab Pro or Colab Pro+)
2. **Google Drive** with at least 5GB free space for model checkpoints
3. **GitHub** account to clone the repository

## Quick Start (5 minutes)

### 1. Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **File → New Notebook**
3. Change runtime to **GPU (A100)**: 
   - Click **Runtime → Change runtime type**
   - Select **A100 GPU** under Hardware accelerator
   - Click **Save**

### 2. Mount Google Drive

Run this in a code cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the link and authorize access.

### 3. Clone Repository and Run Training

Run this in a new code cell:

```bash
%%bash

# Clone repository
cd /content
git clone https://github.com/csce585-mlsystems/csce585-midi.git
cd csce585-midi

# Make script executable
chmod +x scripts/train_colab_a100.sh

# Run training
./scripts/train_colab_a100.sh
```

That's it! The script will:
- ✅ Mount Google Drive
- ✅ Download the Nottingham dataset
- ✅ Preprocess data
- ✅ Train 3 large models (LSTM, GRU, Transformer)
- ✅ Save checkpoints to Google Drive every epoch

## What Gets Trained

### Model 1: Large LSTM (3-4 hours)
- **Architecture**: 1024 hidden units, 4 layers
- **Parameters**: ~85M
- **Memory**: ~8GB VRAM
- **Checkpoint**: `/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/lstm_best.pt`

### Model 2: Large GRU (2-3 hours)
- **Architecture**: 1024 hidden units, 4 layers  
- **Parameters**: ~64M
- **Memory**: ~6GB VRAM
- **Checkpoint**: `/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/gru_best.pt`

### Model 3: Large Transformer (5-6 hours)
- **Architecture**: 1024 d_model, 8 layers, 16 heads
- **Parameters**: ~120M
- **Memory**: ~12GB VRAM
- **Checkpoint**: `/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/transformer_best.pt`

**Total Training Time**: ~10-13 hours

## Monitoring Training

### Watch Progress in Real-Time

The training script prints updates every 500 batches:

```
Epoch 1/20, Batch 500/1149, Loss: 2.3456, Avg Loss: 2.4123, Speed: 123.45 batches/sec, ETA: 5.2s
Epoch 1/20, Batch 1000/1149, Loss: 2.1234, Avg Loss: 2.2567, Speed: 124.67 batches/sec, ETA: 1.2s
```

### Check Saved Models

Run this in a code cell to see your saved models:

```bash
!ls -lh /content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/
```

You should see:
```
lstm_best.pt          85M
gru_best.pt           64M
transformer_best.pt   120M
```

## Checkpoint System

### Automatic Saving

Everything is automatically saved to Google Drive:
- ✅ **Model checkpoints** (.pth files) - Every epoch that improves validation loss
- ✅ **Training plots** (loss curves) - At end of training
- ✅ **Training logs** (summaries & loss arrays) - At end of training
- ✅ **Persistent** across Colab sessions
- ✅ **Accessible** from any device via Google Drive

### Checkpoint Location

```
/content/drive/MyDrive/csce585_model_checkpoints/
└── miditok_augmented/
    ├── models/
    │   ├── lstm_20251111_143022.pth
    │   ├── gru_20251111_163045.pth
    │   └── transformer_20251111_193012.pth
    ├── outputs/
    │   └── plots/
    │       ├── loss_curve_lstm_20251111_143022.png
    │       ├── loss_curve_gru_20251111_163045.png
    │       └── loss_curve_transformer_20251111_193012.png
    └── logs/
        ├── training_summary_lstm_20251111_143022.txt
        ├── train_losses_lstm_20251111_143022.npy
        └── val_losses_lstm_20251111_143022.npy
```

### View Training Results

**Check what was saved:**

```python
from pathlib import Path

checkpoint_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented')

# Check models
models = list((checkpoint_dir / 'models').glob('*.pth'))
print(f"✅ Models: {len(models)}")
for m in models:
    print(f"  {m.name} ({m.stat().st_size / 1e6:.1f} MB)")

# Check plots
plots = list((checkpoint_dir / 'outputs' / 'plots').glob('*.png'))
print(f"\n✅ Plots: {len(plots)}")
for p in plots:
    print(f"  {p.name}")

# Check logs
logs = list((checkpoint_dir / 'logs').glob('training_summary_*.txt'))
print(f"\n✅ Training Summaries: {len(logs)}")
```

**View training summary:**

```python
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')
summaries = sorted(log_dir.glob('training_summary_lstm_*.txt'))

if summaries:
    print("Latest LSTM training summary:\n")
    with open(summaries[-1], 'r') as f:
        print(f.read())
```

**Display loss curve:**

```python
from IPython.display import Image, display
from pathlib import Path

plots_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/outputs/plots')
lstm_plot = sorted(plots_dir.glob('loss_curve_lstm_*.png'))[-1]

display(Image(filename=str(lstm_plot)))
```

## Troubleshooting

### Problem: "Drive mount failed"

**Solution**: Re-authorize Google Drive access:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Problem: "CUDA out of memory"

**Solution**: Your Colab session might not have an A100 GPU. Check:

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

Should show:
```
GPU: NVIDIA A100-SXM4-40GB
VRAM: 40.0GB
```

If not, change runtime to A100 (Runtime → Change runtime type).

### Problem: "Session timed out before training finished"

**Solution**: Your checkpoints are safe in Google Drive! Resume training:

1. Start a new Colab session with A100 GPU
2. Mount Google Drive
3. Clone repository
4. Run training script again

The script will load the best checkpoint automatically and continue training.

### Problem: "Checkpoints not appearing in Google Drive"

**Solution**: Check if Drive is mounted:

```bash
!ls /content/drive/MyDrive/
```

If empty, remount:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Problem: "Training is too slow"

**Solution**: Check you're using the augmented dataset:

```bash
!ls -lh data/miditok/
```

Should show `augmented_sequences.npy` (~150MB). If missing, run:

```bash
!python utils/augment_miditok.py
```

## Advanced Usage

### Train Single Model

To train just one model instead of all three:

```bash
%%bash
cd /content/csce585-midi

python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 256 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints
```

### Custom Checkpoint Directory

Use a different Google Drive folder:

```bash
%%bash
cd /content/csce585-midi

# Create custom directory
CUSTOM_DIR="/content/drive/MyDrive/my_models"
mkdir -p "$CUSTOM_DIR"

# Train with custom directory
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type transformer \
    --checkpoint_dir "$CUSTOM_DIR" \
    # ... other args
```

### Download Checkpoints Locally

After training, download models to your computer:

```python
from google.colab import files

# Download LSTM model
files.download('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/lstm_best.pt')

# Download GRU model
files.download('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/gru_best.pt')

# Download Transformer model
files.download('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/transformer_best.pt')
```

## Resource Management

### A100 GPU Specs

- **VRAM**: 40GB
- **CUDA Cores**: 6,912
- **Tensor Cores**: 432
- **Memory Bandwidth**: 1,555 GB/s

### Recommended Batch Sizes

| Model Type | Batch Size | VRAM Usage | Training Speed |
|------------|------------|------------|----------------|
| LSTM 1024h/4l | 256 | ~8GB | Fast |
| GRU 1024h/4l | 256 | ~6GB | Faster |
| Transformer 1024d/8l | 128 | ~12GB | Slower |

### Memory-Efficient Settings

If running out of memory, reduce batch size:

```bash
python training/train_generator.py \
    --batch_size 64 \    # Reduced from 256
    # ... other args
```

## Cost Estimation

### Google Colab Pricing (as of 2024)

- **Colab Pro**: $9.99/month (limited A100 access)
- **Colab Pro+**: $49.99/month (priority A100 access)

### Training Cost

For full 3-model training (~12 hours):
- **Colab Pro**: May require 2-3 sessions (~$10-15)
- **Colab Pro+**: Single session (~$50/month subscription)

### Cost Optimization

1. Train one model at a time
2. Use early stopping (patience=5)
3. Monitor validation loss to stop when converged

## Next Steps

After training completes:

1. **Generate Music**: Use `generate.py` with your trained models
2. **Evaluate Models**: Use `evaluate.py` to compare performance
3. **Experiment**: Try different hyperparameters
4. **Share**: Upload your best models to GitHub

## Support

If you encounter issues:

1. Check [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review [docs/AUGMENTATION.md](AUGMENTATION.md) for dataset details
3. See [scripts/README.md](../scripts/README.md) for training scripts

## References

- [Google Colab Documentation](https://colab.research.google.com/notebooks/welcome.ipynb)
- [MIDITok Library](https://github.com/Natooz/MIDITok)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
