# Google Colab Quick Start Commands

Copy and paste these commands into Google Colab code cells to quickly set up and train your music generation models.

## Cell 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the authorization link and grant access.

## Cell 2: Verify A100 GPU

```python
import torch

print("GPU Information:")
print(f"  Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("  ⚠️ No GPU detected! Change runtime to A100.")
```

Expected output:
```
GPU Information:
  Available: True
  Device: NVIDIA A100-SXM4-40GB
  VRAM: 40.0 GB
  CUDA Version: 12.2
```

## Cell 3: Setup and Train (All-in-One)

**First, mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Then, clone repository and run training:**
```bash
%%bash

# Clone repository
cd /content
if [ ! -d "csce585-midi" ]; then
    git clone https://github.com/csce585-mlsystems/csce585-midi.git
fi
cd csce585-midi

# Install dependencies
pip install -q miditok symusic tqdm matplotlib

# Run full training pipeline
bash scripts/train_colab_a100.sh
```

This will take ~10-13 hours to complete all three models.

## Alternative: Individual Model Training

### Cell 3a: Setup Only

**First, mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Then, setup the project:**
```bash
%%bash

# Clone and install
cd /content
if [ ! -d "csce585-midi" ]; then
    git clone https://github.com/csce585-mlsystems/csce585-midi.git
fi
cd csce585-midi

# Install dependencies
pip install -q miditok symusic tqdm matplotlib

# Download data
if [ ! -d "data/nottingham-dataset-master" ]; then
    cd data
    git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
    cd ..
fi

# Preprocess
python utils/preprocess_naive.py
python utils/preprocess_miditok.py
python utils/measure_dataset.py
python utils/augment_miditok.py

echo "✅ Setup complete!"
```

### Cell 3b: Train LSTM (3-4 hours)

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

### Large GRU (Fastest - 2-3 hours)
```python
!python training/train_generator.py \
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

### Large Transformer (Best Quality - 5-6 hours)
```python
!python training/train_generator.py \
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

### Mega Model (Push Limits - 10-12 hours)
```python
!python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type transformer \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.00005 \
    --d_model 2048 \
    --nhead 32 \
    --transformer_layers 12 \
    --dim_feedforward 8192 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda
```

## 📊 Monitor Training

### Watch GPU Usage
```python
!watch -n 1 nvidia-smi
```

### Check Training Logs
```python
!tail -f logs/generators/miditok_augmented/models.csv
```

### View Loss Plots (after training)
```python
import pandas as pd
import matplotlib.pyplot as plt

logs = pd.read_csv('logs/generators/miditok_augmented/models.csv')
print(logs.tail())
```

## 💾 Download Results

### Download Trained Model
```python
from google.colab import files

# Find latest model
import glob
latest_model = sorted(glob.glob('models/generators/checkpoints/miditok_augmented/*.pth'))[-1]
print(f"Downloading: {latest_model}")
files.download(latest_model)
```

### Download All Models
```python
!zip -r models.zip models/generators/checkpoints/miditok_augmented/
files.download('models.zip')
```

### Download Logs
```python
!zip -r logs.zip logs/generators/miditok_augmented/
files.download('logs.zip')
```

## 🎵 Generate Music (after training)

```python
!python generate.py \
    --model_path models/generators/checkpoints/miditok_augmented/lstm_YYYYMMDD_HHMMSS.pth \
    --dataset miditok_augmented \
    --output_path outputs/generated_music.mid \
    --num_notes 500
```

## ⚡ Pro Tips for Colab

### Keep Session Alive
```python
# Run this in a cell to prevent disconnection
import time
from IPython.display import display, Javascript

while True:
    display(Javascript('document.querySelector("colab-toolbar-button#connect").click()'))
    time.sleep(60)
```

### Monitor Training in Background
```python
# Start training in background
!nohup python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 256 \
    --hidden_size 1024 \
    --num_layers 4 \
    --device cuda > training.log 2>&1 &

# Check progress
!tail -f training.log
```

### Save to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# After training, copy models
!cp -r models /content/drive/MyDrive/csce585-models/
!cp -r logs /content/drive/MyDrive/csce585-logs/
```

## 🔥 Full Training Pipeline (One Command)

```python
# Train all large models (12-16 hours total)
!bash scripts/train_a100.sh
```

This trains:
1. Large LSTM (512h, 3l) - ~3 hours
2. XL LSTM (1024h, 4l) - ~4 hours
3. Large GRU (512h, 3l) - ~2 hours
4. XL GRU (1024h, 4l) - ~3 hours
5. Large Transformer (512d, 6l) - ~4 hours
6. XL Transformer (1024d, 8l) - ~6 hours

## 📈 Compare Results

```python
!python scripts/compare_augmentation.py
```

## 🎓 Troubleshooting

### Out of Memory
```python
# Reduce batch size
--batch_size 64  # instead of 256
```

### Slow Training
```python
# Check GPU utilization
!nvidia-smi

# Should show high GPU usage (>80%)
```

### Session Timeout
Save checkpoints frequently by reducing patience:
```python
--patience 3  # Stop earlier, saves more often
```

## 📊 View Results After Training

### Check Saved Files

```python
from pathlib import Path

checkpoint_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented')

# Check models
print("="*60)
print("MODELS")
print("="*60)
model_dir = checkpoint_dir / 'models'
if model_dir.exists():
    models = list(model_dir.glob('*.pth'))
    total_size = 0
    for model in sorted(models):
        size_mb = model.stat().st_size / 1e6
        total_size += size_mb
        print(f"  {model.name:40s} {size_mb:6.1f} MB")
    print(f"\n  Total: {total_size:.1f} MB")

# Check plots
print("\n" + "="*60)
print("TRAINING PLOTS")
print("="*60)
plots_dir = checkpoint_dir / 'outputs' / 'plots'
if plots_dir.exists():
    plots = list(plots_dir.glob('*.png'))
    for plot in sorted(plots):
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

### View Training Summary

```python
from pathlib import Path

log_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented/logs')
summaries = sorted(log_dir.glob('training_summary_lstm_*.txt'))

if summaries:
    print("Latest LSTM Training Summary:")
    print("="*80)
    with open(summaries[-1], 'r') as f:
        print(f.read())
```

### Display Loss Curves

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

### Download All Results

```python
from google.colab import files
from pathlib import Path
import shutil

checkpoint_dir = Path('/content/drive/MyDrive/csce585_model_checkpoints/miditok_augmented')

# Create zip of all results
shutil.make_archive('/content/training_results', 'zip', checkpoint_dir)

print("📦 Downloading all training results...")
files.download('/content/training_results.zip')
```

