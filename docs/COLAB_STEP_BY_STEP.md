# Google Colab - Step by Step Guide

This guide walks you through setting up and training models on Google Colab **without** using complex shell scripts.

## Prerequisites

- Google Colab account with A100 GPU access
- Google Drive with ~5GB free space

## Step 1: Open Colab and Set GPU

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **File → New Notebook**
3. Click **Runtime → Change runtime type**
4. Select **A100 GPU** under Hardware accelerator
5. Click **Save**

## Step 2: Mount Google Drive

**Cell 1:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the authorization link and grant access.

Expected output:
```
Mounted at /content/drive
```

## Step 3: Verify GPU

**Cell 2:**
```python
import torch

print("GPU Information:")
print(f"  Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ❌ No GPU! Change runtime to A100")
```

Expected output:
```
GPU Information:
  Available: True
  Device: NVIDIA A100-SXM4-40GB
  VRAM: 40.0 GB
```

## Step 4: Clone Repository

**Cell 3:**
```bash
%%bash
cd /content
git clone https://github.com/csce585-mlsystems/csce585-midi.git
```

## Step 5: Install Dependencies

**Cell 4:**
```bash
%%bash
cd /content/csce585-midi
pip install -q miditok symusic tqdm matplotlib
```

This takes ~2 minutes.

## Step 6: Create Checkpoint Directory

**Cell 5:**
```python
import os

checkpoint_dir = '/content/drive/MyDrive/csce585_model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"✅ Checkpoint directory ready: {checkpoint_dir}")
```

## Step 7: Download Dataset

**Cell 6:**
```bash
%%bash
cd /content/csce585-midi/data
git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
```

## Step 8: Preprocess Data

**Cell 7 - Naive Preprocessing:**
```bash
%%bash
cd /content/csce585-midi
python utils/preprocess_naive.py
```

Takes ~5 minutes. You should see:
```
Processing: 100%|██████████| 1034/1034 [04:32<00:00]
✓ Saved 2055 sequences to data/naive/sequences.npy
```

**Cell 8 - MIDITok Preprocessing:**
```bash
%%bash
cd /content/csce585-midi
python utils/preprocess_miditok.py
```

Takes ~3 minutes. You should see:
```
Processing MIDI files: 100%|██████████| 1034/1034
✓ Saved 2055 sequences to data/miditok/sequences.npy
```

**Cell 9 - Measure Dataset:**
```bash
%%bash
cd /content/csce585-midi
python utils/measure_dataset.py
```

Takes ~2 minutes.

**Cell 10 - Augmented Dataset (IMPORTANT):**
```bash
%%bash
cd /content/csce585-midi
python utils/augment_miditok.py
```

Takes ~15-20 minutes. You should see:
```
Processing scores: 100%|██████████| 1034/1034
Transposition [-5]: 100%|██████████| 1034/1034
Transposition [-3]: 100%|██████████| 1034/1034
...
✓ Saved 14385 augmented sequences
```

**This is critical** - creates the augmented training data!

## Step 9: Train Models

Now you can train models! Each of these goes in a separate cell.

### Train Large LSTM (3-4 hours)

**Cell 11:**
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

You'll see progress updates:
```
Vocab size: 284, Sequence length: 100, Device: cuda
Train sequences: 12946, Val sequences: 1439
Using checkpoint directory: /content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented

Epoch 1/20, Batch 500/1149, Loss: 2.3456, Avg Loss: 2.4123, Speed: 123.45 batches/sec
...
```

### Train Large GRU (2-3 hours)

**Cell 12:**
```bash
%%bash
cd /content/csce585-midi

python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
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

### Train Large Transformer (5-6 hours)

**Cell 13:**
```bash
%%bash
cd /content/csce585-midi

python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type transformer \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.0001 \
    --d_model 1024 \
    --nhead 16 \
    --transformer_layers 8 \
    --dim_feedforward 4096 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints
```

## Step 10: Check Results

**Cell 14 - List Saved Files:**
```python
from pathlib import Path

checkpoint_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented')

# Check models
print("="*60)
print("MODELS")
print("="*60)
model_dir = checkpoint_dir / 'models'
if model_dir.exists():
    for model in sorted(model_dir.glob('*.pth')):
        size_mb = model.stat().st_size / 1e6
        print(f"  {model.name:40s} {size_mb:6.1f} MB")

# Check plots
print("\n" + "="*60)
print("TRAINING PLOTS")
print("="*60)
plots_dir = checkpoint_dir / 'outputs' / 'plots'
if plots_dir.exists():
    for plot in sorted(plots_dir.glob('*.png')):
        print(f"  {plot.name}")

# Check logs
print("\n" + "="*60)
print("TRAINING LOGS")
print("="*60)
logs_dir = checkpoint_dir / 'logs'
if logs_dir.exists():
    summaries = list(logs_dir.glob('training_summary_*.txt'))
    print(f"  Training summaries: {len(summaries)}")
```

**Cell 15 - View Training Summary:**
```python
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')
summaries = sorted(log_dir.glob('training_summary_*.txt'))

if summaries:
    print("Latest Training Summary:")
    print("="*80)
    with open(summaries[-1], 'r') as f:
        print(f.read())
```

**Cell 16 - Display Loss Curve:**
```python
from IPython.display import Image, display
from pathlib import Path

plots_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/outputs/plots')

if plots_dir.exists():
    for plot in sorted(plots_dir.glob('*.png')):
        print(f"📊 {plot.name}")
        display(Image(filename=str(plot), width=800))
        print("\n")
```

## Common Issues

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size:
```bash
--batch_size 128  # instead of 256
```

### Issue: "Sequences not found"

**Solution:** Make sure you ran all preprocessing steps (cells 7-10).

Check:
```bash
%%bash
cd /content/csce585-midi
ls -lh data/miditok/
```

You should see `augmented_sequences.npy` (~150MB).

### Issue: "Session disconnected"

**Don't panic!** Your checkpoints are safe in Google Drive.

1. Start a new Colab session with A100 GPU
2. Mount Drive (Cell 1)
3. Check your saved models (Cell 14)
4. Continue from where you left off

### Issue: "Training is slow"

Check GPU utilization:
```bash
%%bash
nvidia-smi
```

Should show:
- GPU-Util: >80%
- Memory-Usage: 8-12 GB for LSTM/GRU, 15-20 GB for Transformer

## Tips

1. **Save notebook frequently**: File → Save a copy in Drive

2. **Keep notebook tab open**: Don't close the browser during training

3. **Monitor in Google Drive**: Open Drive in another tab to watch checkpoint files appear

4. **Train one at a time**: Don't try to train multiple models simultaneously

5. **Use background execution**: Runtime → Manage sessions to see all active notebooks

## Timeline

| Time | Activity |
|------|----------|
| 0:00 | Start setup (Cells 1-6) |
| 0:05 | Download dataset (Cell 7) |
| 0:10 | Naive preprocessing (Cell 8) |
| 0:15 | MIDITok preprocessing (Cell 9) |
| 0:17 | Measure dataset (Cell 10) |
| 0:20 | Start augmentation (Cell 11) |
| 0:40 | Augmentation complete, start LSTM training |
| 4:00 | LSTM complete, start GRU training |
| 6:30 | GRU complete, start Transformer training |
| 12:30 | All training complete! |

## Next Steps

After training:

1. **Generate music**: Use `generate.py` with your trained models
2. **Evaluate models**: Compare validation losses
3. **Download results**: Use Cell 16 download code
4. **Share models**: Upload to GitHub or share Drive folder

## Support

If something goes wrong:
- Check [docs/COLAB_TRAINING.md](COLAB_TRAINING.md) for detailed guide
- See [docs/LOGS_AND_PLOTS.md](LOGS_AND_PLOTS.md) for result documentation
- Review [docs/CHECKPOINT_SYSTEM.md](CHECKPOINT_SYSTEM.md) for checkpoint details
