# Google Colab Quick Start Commands

## 🚀 Copy-Paste Commands for Colab

### 1. Clone Repository
```python
!git clone https://github.com/csce585-mlsystems/csce585-midi.git
%cd csce585-midi
```

### 2. Setup Environment
```python
!bash scripts/colab_setup.sh
```

### 3. Check GPU
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Available: {torch.cuda.is_available()}")
!nvidia-smi
```

### 4. Create Augmented Dataset (if needed)
```python
!python utils/augment_miditok.py
```

### 5. Quick Test (2 minutes)
```python
!python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 1 \
    --batch_size 256 \
    --hidden_size 512 \
    --num_layers 3 \
    --max_batches 100 \
    --device cuda
```

## 🎯 Recommended Training Commands

### Large LSTM (Recommended - 3-4 hours)
```python
!python training/train_generator.py \
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
    --device cuda
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
    --device cuda
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
    --device cuda
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
