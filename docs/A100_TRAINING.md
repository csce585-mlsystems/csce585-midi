# A100 GPU Training Guide

Optimized training commands for Google Colab with A100 GPU.

## 🚀 Quick Start

### Full Pipeline (All Large Models)
```bash
bash scripts/train_a100.sh
```

This trains 6 large models:
- Large LSTM (512 hidden, 3 layers)
- XL LSTM (1024 hidden, 4 layers)
- Large GRU (512 hidden, 3 layers)
- XL GRU (1024 hidden, 4 layers)
- Large Transformer (512 d_model, 6 layers)
- XL Transformer (1024 d_model, 8 layers)

## 📊 Individual Model Commands

### Large LSTM Models

#### Large LSTM (512 hidden, 3 layers)
**~6M parameters, Fast training**
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_size 512 \
    --num_layers 3 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda
```

#### Extra Large LSTM (1024 hidden, 4 layers)
**~30M parameters, Best quality**
```bash
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
    --device cuda
```

### Large GRU Models

#### Large GRU (512 hidden, 3 layers)
**~4.5M parameters, Fast & efficient**
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_size 512 \
    --num_layers 3 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda
```

#### Extra Large GRU (1024 hidden, 4 layers)
**~22M parameters, Great balance**
```bash
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
    --device cuda
```

### Large Transformer Models

#### Large Transformer (512 d_model, 8 heads, 6 layers)
**~10M parameters, Modern architecture**
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type transformer \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.0001 \
    --d_model 512 \
    --nhead 8 \
    --transformer_layers 6 \
    --dim_feedforward 2048 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda
```

#### Extra Large Transformer (1024 d_model, 16 heads, 8 layers)
**~50M parameters, State-of-the-art**
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
    --val_split 0.1 \
    --patience 5 \
    --device cuda
```

## 🎯 Quick Test Models

### Small Test (2 epochs, verify everything works)
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 2 \
    --batch_size 256 \
    --lr 0.001 \
    --hidden_size 512 \
    --num_layers 3 \
    --max_batches 500 \
    --device cuda
```

### Medium Test (5 epochs, check overfitting)
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 5 \
    --batch_size 256 \
    --lr 0.001 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.1 \
    --device cuda
```

## ⚡ A100 Optimizations

### Batch Size Guidelines
- **Small models (256-512 hidden)**: batch_size=128-256
- **Large models (1024+ hidden)**: batch_size=256-512
- **XL Transformers**: batch_size=128-256 (memory intensive)

### Learning Rate Guidelines
- **LSTM/GRU**: lr=0.001 (standard)
- **Transformer**: lr=0.0001 (lower for stability)
- **XL models**: lr=0.0005 (conservative start)

### Memory Usage Estimates
```
Small LSTM (256h, 2l):    ~2GB VRAM   | Batch: 256
Large LSTM (512h, 3l):    ~4GB VRAM   | Batch: 256
XL LSTM (1024h, 4l):      ~8GB VRAM   | Batch: 256

Small GRU (256h, 2l):     ~1.5GB VRAM | Batch: 256
Large GRU (512h, 3l):     ~3GB VRAM   | Batch: 256
XL GRU (1024h, 4l):       ~6GB VRAM   | Batch: 256

Large Transformer (512):   ~6GB VRAM   | Batch: 128
XL Transformer (1024):     ~15GB VRAM  | Batch: 128

A100 has 40GB VRAM - plenty of headroom! 🎉
```

## 🔥 Extreme Models (Push A100 to Limit)

### Mega LSTM (2048 hidden, 5 layers)
**~150M parameters**
```bash
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.0005 \
    --hidden_size 2048 \
    --num_layers 5 \
    --dropout 0.4 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda
```

### Mega Transformer (2048 d_model, 32 heads, 12 layers)
**~200M parameters - GPT-scale!**
```bash
python training/train_generator.py \
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

## 📈 Expected Training Times (A100)

With augmented dataset (~8.3M samples):

**Per Epoch:**
- Small models (256h): ~5-10 minutes
- Large models (512h): ~10-15 minutes
- XL models (1024h): ~15-25 minutes
- Mega models (2048h): ~30-45 minutes

**Full Training (20 epochs with early stopping):**
- Small models: ~1-2 hours
- Large models: ~3-4 hours
- XL models: ~5-7 hours
- Mega models: ~10-12 hours

With batch_size=256, you can train **much faster** than on CPU/MPS!

## 🎓 Recommended Training Strategy

### Phase 1: Quick Validation (30 minutes)
Train small models with 2 epochs to verify:
```bash
# Quick LSTM test
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 2 \
    --batch_size 256 \
    --hidden_size 512 \
    --num_layers 3 \
    --device cuda

# Quick GRU test
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 2 \
    --batch_size 256 \
    --hidden_size 512 \
    --num_layers 3 \
    --device cuda
```

### Phase 2: Large Models (3-4 hours each)
```bash
# Best LSTM
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 20 \
    --batch_size 256 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda

# Best GRU
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 20 \
    --batch_size 256 \
    --hidden_size 1024 \
    --num_layers 4 \
    --dropout 0.4 \
    --val_split 0.1 \
    --patience 5 \
    --device cuda
```

### Phase 3: Transformers (5-7 hours)
```bash
# Large Transformer
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
    --device cuda
```

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size or model size:
```bash
--batch_size 64  # Instead of 256
--hidden_size 512  # Instead of 1024
```

### Training Too Slow
Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Should see ~80-100% GPU utilization. If low:
- Increase batch size
- Check data loading bottleneck
- Ensure using `--device cuda`

### Model Not Improving
- Increase model size
- Lower learning rate
- Increase dropout
- Train longer (more epochs)

## 📊 Monitoring Training

### In Colab
```python
# Check GPU usage
!nvidia-smi

# Monitor training logs
!tail -f logs/generators/miditok_augmented/models.csv
```

### Compare Results
```bash
python scripts/compare_augmentation.py
```

## 🎯 Best Models for A100

Based on parameter count and quality:

**Best Overall:** XL LSTM (1024h, 4l) - Great balance of speed & quality  
**Fastest:** Large GRU (512h, 3l) - Quick training, good results  
**Most Advanced:** XL Transformer (1024d, 8l) - State-of-the-art architecture  
**Extreme:** Mega Transformer (2048d, 12l) - Push the limits!

Happy training on that A100! 🚀
