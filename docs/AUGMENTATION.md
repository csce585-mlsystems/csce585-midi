# Data Augmentation for MIDITok

This directory contains utilities for augmenting the MIDITok dataset through musical transposition to reduce overfitting.

## Problem

The original MIDITok preprocessing was experiencing severe overfitting:
- Training loss: 0.22
- Validation loss: 0.67 (3x higher!)
- Only 2,055 unique songs in the dataset
- Model memorizing patterns rather than generalizing

## Solution: Data Augmentation via Transposition

Musical transposition shifts all notes by a fixed number of semitones while preserving the musical structure. This creates new training examples that:
- **Increase dataset diversity** (5-7x more training data)
- **Teach pitch-invariant patterns** (model learns musical structure, not absolute pitches)
- **Reduce overfitting** by exposing the model to more note combinations

## Files

- **`augment_miditok.py`** - Main augmentation script
- **`train_augmented.sh`** - Complete pipeline to preprocess and train
- **`compare_augmentation.py`** - Analysis tool to compare original vs augmented datasets

## Usage

### Quick Start (Recommended)

Run the complete pipeline with one command:

```bash
bash scripts/train_augmented.sh
```

This will:
1. Create augmented dataset with 7 transpositions (-5, -3, -1, 0, +1, +3, +5 semitones)
2. Train LSTM model on augmented data
3. Train GRU model on augmented data

### Step-by-Step

#### 1. Create Augmented Dataset

```bash
python utils/augment_miditok.py --transpositions="-5,-3,-1,0,1,3,5" --output_dir="data/miditok_augmented"
```

**Options:**
- `--transpositions`: Comma-separated semitone values (default: `-5,-3,-1,0,1,3,5`)
- `--output_dir`: Output directory (default: `data/miditok_augmented`)

**Common transposition schemes:**
- **Conservative** (3x): `-1,0,1` (minimal transposition)
- **Moderate** (5x): `-3,-1,0,1,3` (typical)
- **Aggressive** (7x): `-5,-3,-1,0,1,3,5` (maximum diversity)
- **Extreme** (9x): `-6,-4,-2,0,2,4,6` (even more)

#### 2. Analyze the Results

```bash
python scripts/compare_augmentation.py
```

This shows:
- Dataset size comparison
- Training sample counts
- Data density improvements
- Expected benefits

#### 3. Train Models

```bash
# LSTM
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type lstm \
    --epochs 15 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 4

# GRU
python training/train_generator.py \
    --dataset miditok_augmented \
    --model_type gru \
    --epochs 15 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --val_split 0.1 \
    --patience 4
```

## Expected Results

### Dataset Growth
- **Original**: ~2,055 sequences → ~1,178,694 training samples
- **Augmented (7x)**: ~14,385 sequences → ~8,250,858 training samples
- **Improvement**: 7x more diverse training data

### Overfitting Reduction
- More samples per vocab entry (4,873 → 34,111)
- Better generalization from pitch-invariant learning
- Lower validation loss gap

### Training Impact
- **Original**: Validation loss increases while training loss decreases
- **Augmented**: Validation loss should track training loss more closely
- **Goal**: Reduce the train/val loss gap from 0.45 to < 0.2

## How Transposition Works

Transposition shifts all MIDI note pitches by a constant:

```python
# Original: C-E-G (C major chord)
original_pitches = [60, 64, 67]  # C4, E4, G4

# Transpose up 3 semitones (to E♭)
transposed_pitches = [63, 67, 70]  # E♭4, G4, B♭4
```

The musical relationships are preserved:
- Intervals between notes stay the same
- Rhythm and timing unchanged
- Only the absolute pitch changes

## Technical Details

### Transposition Implementation
- Uses `symusic.Score` for MIDI manipulation
- Transposes note pitches while preserving timing
- Skips drum tracks (drums are pitch-independent)
- Keeps notes within valid MIDI range (0-127)

### Token Vocabulary
- Vocabulary remains the same (284 tokens)
- Only note pitch tokens change
- Timing, velocity, and structure tokens unchanged

### Memory Efficiency
- Uses lazy dataset loading
- Sequences generated on-the-fly during training
- No memory explosion despite 7x more data

## Troubleshooting

### Out of Memory
If you run out of memory:
1. Reduce transpositions: `--transpositions="-2,0,2"` (3x instead of 7x)
2. Reduce batch size: `--batch_size 32`
3. Use subsampling in training: `--subsample_ratio 0.5`

### Still Overfitting
If overfitting persists:
1. Increase dropout: `--dropout 0.4` or `0.5`
2. Add more transpositions: `-6,-4,-2,0,2,4,6`
3. Reduce model size: `--hidden_size 128`
4. Increase weight decay in the training code

### Training Too Slow
If training is too slow:
1. Increase batch size: `--batch_size 128`
2. Reduce transpositions (fewer but still helps)
3. Use subsampling: `--subsample_ratio 0.5`

## References

- **Data Augmentation in Music**: Common technique in MIR (Music Information Retrieval)
- **Pitch Transposition**: Preserves musical structure while creating new examples
- **Overfitting Prevention**: Standard deep learning regularization technique

## Next Steps

After creating augmented data:
1. ✅ Train models (LSTM, GRU, Transformer)
2. 📊 Compare training/validation loss curves
3. 🎵 Generate music and evaluate quality
4. 📈 Measure improvement in metrics

For questions or issues, check the main project README or training logs.
