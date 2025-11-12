# A100 GPU Performance Optimization Guide

## Problem: Training is Too Slow (6+ hours per model)

If you're experiencing slow training on an A100 GPU, here are the optimizations that will speed things up dramatically.

## Critical Fix #1: DataLoader Configuration ✅ FIXED

**Problem:** The code was configured for macOS MPS with:
- `num_workers=0` (single-threaded data loading)
- `pin_memory=False` (slow CPU→GPU transfers)

**Solution:** Now automatically detects CUDA and uses:
- `num_workers=4` (parallel data loading)
- `pin_memory=True` (fast memory transfers)
- `persistent_workers=True` (reuses worker processes)

**Expected speedup:** 2-4x faster

## Expected Training Times on A100

With the DataLoader fix, you should see:

| Model | Parameters | Expected Time | Batches/sec |
|-------|-----------|---------------|-------------|
| LSTM 1024h/4l | ~85M | **2-3 hours** | 120-150 |
| GRU 1024h/4l | ~64M | **1.5-2 hours** | 150-180 |
| Transformer 1024d/8l | ~120M | **4-5 hours** | 80-100 |

If you're still seeing 6+ hours, check the issues below.

## Additional Optimizations

### 1. Check Batch Size

**Verify you're using large batches:**
```bash
# In your training command, ensure:
--batch_size 256  # for LSTM/GRU
--batch_size 128  # for Transformer
```

Smaller batches = slower training. A100 has 40GB VRAM, use it!

### 2. Verify GPU Utilization

**Monitor during training:**
```bash
%%bash
watch -n 1 nvidia-smi
```

You should see:
- **GPU-Util: 85-95%** (if lower, data loading is bottleneck)
- **Memory-Usage: 8-15GB** (if lower, increase batch size)
- **Power: 200-300W** (if lower, GPU is idle)

### 3. Check You're Actually Using CUDA

**In training output, verify:**
```
Device: cuda
CUDA: True
GPU: NVIDIA A100-SXM4-40GB
```

If it says `Device: cpu`, you're not using the GPU at all!

### 4. Increase Batch Size for A100

The A100 has **40GB VRAM**. You can use much larger batches:

**For LSTM/GRU:**
```bash
--batch_size 512  # Instead of 256
```

**For Transformer:**
```bash
--batch_size 256  # Instead of 128
```

**Expected speedup:** 1.5-2x faster

### 5. Use Mixed Precision Training

Add this to your training script for another 2x speedup:

```python
# At the top of train_generator.py, add:
from torch.cuda.amp import autocast, GradScaler

# In training loop, replace:
# optimizer.zero_grad()
# output, _ = model(x)
# loss = loss_function(output.view(-1, vocab_size), y.view(-1))
# loss.backward()
# optimizer.step()

# With:
scaler = GradScaler()
optimizer.zero_grad()
with autocast():
    output, _ = model(x)
    loss = loss_function(output.view(-1, vocab_size), y.view(-1))
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Expected speedup:** 2x faster, uses half the memory

## Quick Performance Test

Run this in a Colab cell to test your setup:

```python
import torch
import time

device = torch.device('cuda')
batch_size = 256
seq_len = 100
vocab_size = 284
hidden_size = 1024

# Create dummy model
model = torch.nn.LSTM(
    input_size=256,
    hidden_size=hidden_size,
    num_layers=4,
    batch_first=True
).to(device)

embedding = torch.nn.Embedding(vocab_size, 256).to(device)
fc = torch.nn.Linear(hidden_size, vocab_size).to(device)

# Warm up
x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
for _ in range(10):
    embedded = embedding(x)
    output, _ = model(embedded)
    logits = fc(output)

# Benchmark
torch.cuda.synchronize()
start = time.time()
iterations = 100

for _ in range(iterations):
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    embedded = embedding(x)
    output, _ = model(embedded)
    logits = fc(output)

torch.cuda.synchronize()
elapsed = time.time() - start

batches_per_sec = iterations / elapsed
print(f"Performance: {batches_per_sec:.1f} batches/sec")
print(f"Expected for LSTM training: ~120-150 batches/sec")

if batches_per_sec < 80:
    print("⚠️  WARNING: Performance is too low!")
    print("   Check: GPU utilization, DataLoader workers, batch size")
elif batches_per_sec < 120:
    print("⚠️  Performance is acceptable but could be better")
    print("   Consider: Increasing batch size, enabling mixed precision")
else:
    print("✅ Performance looks good!")
```

## Troubleshooting Slow Training

### Symptom: Low GPU Utilization (<50%)

**Cause:** Data loading bottleneck

**Fix:**
1. Increase `num_workers` in DataLoader (try 4, 8, or 16)
2. Use `persistent_workers=True`
3. Use `pin_memory=True`

### Symptom: High GPU Utilization but Slow

**Cause:** Batch size too small

**Fix:** Increase batch size until you hit memory limit

### Symptom: Training pauses between epochs

**Cause:** Validation taking too long or Drive I/O slow

**Fix:**
1. Use smaller validation split: `--val_split 0.05`
2. Save checkpoints less frequently
3. Use local SSD for intermediate saves, copy to Drive after

### Symptom: "CUDA out of memory"

**Cause:** Batch size too large

**Fix:** Reduce batch size by 50% and retry

## Recommended Colab Training Commands

### Fast LSTM (1.5-2 hours):
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 512 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.05 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints
```

### Fast GRU (1-1.5 hours):
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 20 \
    --batch_size 512 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.05 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints
```

### Fast Transformer (3-4 hours):
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type transformer \
    --epochs 20 \
    --batch_size 256 \
    --lr 0.0001 \
    --d_model 1024 \
    --nhead 16 \
    --transformer_layers 8 \
    --dim_feedforward 4096 \
    --dropout 0.3 \
    --val_split 0.05 \
    --patience 5 \
    --device cuda \
    --checkpoint_dir /content/drive/MyDrive/csce585_model_checkpoints
```

Note the changes:
- ✅ Increased batch sizes (512 for RNN, 256 for Transformer)
- ✅ Reduced validation split (0.05 instead of 0.1) for faster validation
- ✅ Using fixed DataLoader configuration

## Summary

**Main issue:** DataLoader was configured for macOS, not CUDA
**Fix applied:** Auto-detect device and optimize DataLoader settings
**Expected result:** 2-4x faster training (2-3 hours instead of 6+ hours)

**Additional speedups available:**
- Larger batch sizes: +50-100% faster
- Mixed precision: +100% faster  
- Smaller validation split: +10-20% faster

Combined, these can reduce LSTM training from 6 hours to **under 1 hour**!
