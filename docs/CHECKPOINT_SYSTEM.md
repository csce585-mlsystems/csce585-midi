# Google Colab Checkpoint System - Summary

## What Was Added

The checkpoint system allows you to save model checkpoints to Google Drive, ensuring your trained models persist even if your Colab session disconnects or times out.

## Changes Made

### 1. Updated `training/train_generator.py`

**Added checkpoint_dir parameter:**
```python
def train(..., checkpoint_dir=None):
    """
    checkpoint_dir: Optional path to save checkpoints (e.g., Google Drive path)
                   If None, saves to local models/generators/checkpoints/
    """
```

**Modified directory logic:**
```python
if checkpoint_dir:
    MODEL_DIR = Path(checkpoint_dir) / dataset
    print(f"Using checkpoint directory: {MODEL_DIR}")
else:
    MODEL_DIR = Path(f"models/generators/checkpoints/{dataset}")
```

**Added command-line argument:**
```python
parser.add_argument('--checkpoint_dir', type=str, default=None,
                   help='Directory to save checkpoints (e.g., Google Drive path)')
```

### 2. Created `scripts/train_colab_a100.sh`

Full-featured training script that:
- ✅ Auto-detects Colab environment
- ✅ Mounts Google Drive
- ✅ Creates checkpoint directory
- ✅ Downloads and preprocesses data
- ✅ Trains 3 large models with checkpoint saving
- ✅ Provides progress updates

### 3. Created `docs/COLAB_TRAINING.md`

Comprehensive guide covering:
- Quick start (5 minutes)
- Model specifications
- Monitoring training
- Checkpoint system details
- Troubleshooting common issues
- Advanced usage
- Cost estimation

### 4. Updated `docs/COLAB_COMMANDS.md`

Added checkpoint_dir argument to all training commands:
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints \
    # ... other args
```

## How to Use

### Option 1: Automated Training (Recommended)

```bash
%%bash
cd /content
git clone https://github.com/csce585-mlsystems/csce585-midi.git
cd csce585-midi
chmod +x scripts/train_colab_a100.sh
./scripts/train_colab_a100.sh
```

This runs everything automatically and saves to:
```
/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/
```

### Option 2: Manual Training

```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --checkpoint_dir /content/drive/MyDrive/my_checkpoints \
    # ... other arguments
```

### Option 3: Local Training (without checkpoints)

```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    # ... other arguments (no checkpoint_dir)
```

Saves to `models/generators/checkpoints/miditok_augmented/` locally.

## Checkpoint Structure

```
/content/drive/MyDrive/csce585_model_checkpoints/
└── miditok_augmented/
    ├── models/
    │   ├── lstm_20251111_143022.pth      # Best LSTM model (~85MB)
    │   ├── gru_20251111_163045.pth       # Best GRU model (~64MB)
    │   └── transformer_20251111_193012.pth # Best Transformer (~120MB)
    ├── outputs/
    │   └── plots/
    │       ├── loss_curve_lstm_20251111_143022.png
    │       ├── loss_curve_gru_20251111_163045.png
    │       └── loss_curve_transformer_20251111_193012.png
    └── logs/
        ├── models.csv                     # Experiment tracking CSV
        ├── training_summary_lstm_20251111_143022.txt
        ├── training_summary_gru_20251111_163045.txt
        ├── training_summary_transformer_20251111_193012.txt
        ├── train_losses_lstm_20251111_143022.npy
        ├── val_losses_lstm_20251111_143022.npy
        └── ... (similar for other models)
```

### What Gets Saved

**Models (`models/`):**
- Trained model weights (.pth files)
- Contains full state_dict for model inference

**Plots (`outputs/plots/`):**
- Training and validation loss curves (PNG)
- High-resolution (150 DPI) for presentations
- Shows both train and val loss on same plot

**Logs (`logs/`):**
- `models.csv`: Experiment tracking with hyperparameters and results
- `training_summary_*.txt`: Human-readable training report with:
  - Model configuration (vocab size, layers, params, etc.)
  - Training configuration (epochs, batch size, learning rate)
  - Results (final losses, best losses, training time)
  - File paths to saved artifacts
- `train_losses_*.npy`: Training loss array for analysis
- `val_losses_*.npy`: Validation loss array for analysis


## Verifying Checkpoints

### Check if checkpoints exist:

```python
from pathlib import Path

checkpoint_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented')

# Check models
model_dir = checkpoint_dir / 'models'
if model_dir.exists():
    models = list(model_dir.glob('*.pth'))
    print(f"✅ Found {len(models)} model checkpoints:")
    for model in sorted(models):
        size_mb = model.stat().st_size / 1e6
        print(f"  {model.name}: {size_mb:.1f} MB")

# Check plots
plots_dir = checkpoint_dir / 'outputs' / 'plots'
if plots_dir.exists():
    plots = list(plots_dir.glob('*.png'))
    print(f"\n✅ Found {len(plots)} training plots")

# Check logs
logs_dir = checkpoint_dir / 'logs'
if logs_dir.exists():
    summaries = list(logs_dir.glob('training_summary_*.txt'))
    print(f"\n✅ Found {len(summaries)} training summaries")
```

### View Training Summary:

```python
from pathlib import Path

# Read the most recent LSTM training summary
log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')
summaries = sorted(log_dir.glob('training_summary_lstm_*.txt'))

if summaries:
    with open(summaries[-1], 'r') as f:
        print(f.read())
```

### Load Loss History:

```python
import numpy as np
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')

# Load loss arrays
train_losses = np.load(log_dir / 'train_losses_lstm_20251111_143022.npy')
val_losses = np.load(log_dir / 'val_losses_lstm_20251111_143022.npy')

print(f"Training epochs: {len(train_losses)}")
print(f"Best training loss: {train_losses.min():.4f}")
print(f"Best validation loss: {val_losses.min():.4f}")
```

### Load a Model:

```python
import torch

model_path = '/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/models/lstm_20251111_143022.pth'
model_state = torch.load(model_path, map_location='cuda')

print(f"✅ Model loaded successfully!")
print(f"   Parameters: {sum(p.numel() for p in model_state.values()) / 1e6:.1f}M")
```

## Advantages

### Without checkpoint_dir (local saving):
- ❌ Lost if Colab session disconnects
- ❌ Lost after 12 hours (Colab limit)
- ❌ Can't access from other sessions
- ✅ Slightly faster (no Drive I/O)

### With checkpoint_dir (Drive saving):
- ✅ Persistent across sessions
- ✅ Accessible from any device
- ✅ Backed up to Google Drive
- ✅ Can resume training easily
- ❌ Slightly slower (Drive I/O overhead ~1-2%)

## Training Timeline

| Time | Event |
|------|-------|
| 0:00 | Start training LSTM |
| 0:15 | First checkpoint saved to Drive |
| 3:00 | LSTM complete, start GRU |
| 3:15 | First GRU checkpoint saved |
| 5:00 | GRU complete, start Transformer |
| 5:30 | First Transformer checkpoint saved |
| 12:00 | All training complete |

## Troubleshooting

### Drive not mounted?
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Checkpoints not appearing?
1. Check Drive is mounted: `!ls /content/drive/MyDrive/`
2. Verify directory exists: `!ls /content/drive/MyDrive/csce585_model_checkpoints/`
3. Check training logs for "Saved best model" messages

### Session disconnected mid-training?
Your checkpoints are safe! Start new session and:
1. Mount Drive
2. List checkpoints: `!ls /content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/`
3. Your best model is saved, you can resume or use it for generation

## Cost Analysis

### Storage:

**Per model training:**
- Model checkpoint (.pth): 60-120 MB
- Loss curve plot (.png): ~0.5 MB
- Training summary (.txt): ~2 KB
- Loss arrays (.npy): ~1 KB each

**3 models total:**
- Models: ~270 MB
- Plots: ~1.5 MB
- Logs: ~500 KB
- **Total: ~275 MB**

Google Drive free tier: 15 GB
**Storage cost**: Free ✅

### Compute:
- A100 training: ~12 hours
- Colab Pro ($9.99/mo): Limited A100 access
- Colab Pro+ ($49.99/mo): Priority A100 access
- **Estimated cost**: $10-50 depending on tier

## Next Steps

After training with checkpoints:

1. **Generate music**:
```python
!python generate.py \
    --model_path /content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/lstm_best.pt
```

2. **Download models locally**:
```python
from google.colab import files
files.download('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/lstm_best.pt')
```

3. **Share with others**: Share your Google Drive folder

4. **Resume training**: Use the same checkpoint_dir to continue from where you left off

## References

- Main guide: [docs/COLAB_TRAINING.md](COLAB_TRAINING.md)
- Quick commands: [docs/COLAB_COMMANDS.md](COLAB_COMMANDS.md)
- Training script: [scripts/train_colab_a100.sh](../scripts/train_colab_a100.sh)
- Python trainer: [training/train_generator.py](../training/train_generator.py)
