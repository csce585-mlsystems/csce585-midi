# csce585-midi

## Group Info  
Cade Stocker 
- Email: cstocker@email.sc.edu  

## Project Summary/Abstract  

## Problem Description  
This project explores AI music generation via MIDI files. MIDI files contain instructions, rather than audio data, which tell software how to play a song. These instructions are contained in chunks, containing event data such as notes and control changes. Despite not being human readable, MIDI data is easily translatable into a variety of formats, and is used as the core for Digital Audio Workstation editors. Although AI models such as MusicLM exist to generate music, these create raw audio in the form of waveforms. As such, it is very hard for a user to iterate upon its creations, as changes would require the entire waveform to be regenerated. The use of MIDI allows for small, incremental tweaks, while still keeping the end user as part of the process through their DAW.

I am following the architecture described in "Musenet: Music Generation using Abstractive and Generative Methods". The models are currently trained on the Nottingham dataset, a collection of around 1,200 British and American folk tunes. This relatively small dataset allows different types of models to be quickly trained, allowing easy comparison.

As described in the paper discussing the architecture of Musenet, I will be using a discriminator (which selects the chord for the next measure based on previous measures) and a generator (which generates notes based on previous measures, and the output on the discriminator). The use of a factory design pattern for both the generator and discriminator enables multiple types of each to be trained (transformer, lstm, mlp, ...)

Output from the models (MIDI File) is analyzed by PrettyMIDI library, which finds metrics such as:
	- Polyphony
	- Number of notes
	- Pitch range
	- Note density
	- Duration

- Motivation  
	- Provide quantitative comparisons between different variations of both training and genreation methods.
	- Recreating Musenet architecture but with more customizable features.

- Challenges  
	- Small size of Nottingham dataset
	- Small models trained locally (Macbook Air M1)
	- Original project idea was taking a user text description and turning it into a MIDI file
		- Might be too large of a project to tackle alone for this course
		- May need larger dataset
		- Models could be too small do realistically do

## Contribution  

## References   

Zhu, Y., Baca, J., Rekabdar, B., Rawassizadeh, R. (2023). A Survey of AI Music Generation Tools and Models. [arXiv:2308.12982](https://arxiv.org/abs/2308.12982)

Briot, J., Hadjeres, G., Pachet, F. (2017). Deep Learning Techniques for Music Generation -- A Survey.[arXiv:1709.01620](https://arxiv.org/abs/1709.01620)

Bhandari, K., Roy, A., Wang, K., Puri, G., Colton, S., Herremans, D. (2024). Text2midi: Generating Symbolic Music from Captions. [arXiv:2412.16526](https://arxiv.org/abs/2412.16526)

Yang, L., Chou, S., Yang, Y. (2017). MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation. [arXiv:1703.10847](https://arxiv.org/abs/1703.10847)

Tian, S., Zhang, C., Yuan, W., Tan, W., Zhu, W. (2025). XMusic: Towards a Generalized and Controllable Symbolic Music Generation Framework. [arXiv:2501.08809](https://arxiv.org/abs/2501.08809)

Colin Raffel. **"Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching"**. _PhD Thesis_, 2016. https://colinraffel.com/publications/thesis.pdf

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/csce585-mlsystems/csce585-midi.git
cd csce585-midi
```

### 2. Download the Dataset

The Nottingham MIDI dataset is included in the repository. If you need to download it separately:

```bash
# Option 1: Clone the dataset repository
cd data/
git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
cd ..

# Option 2: Download and extract manually
# Visit: https://github.com/jukedeck/nottingham-dataset
# Download ZIP and extract to data/nottingham-dataset-master/
```

The dataset should be at: `data/nottingham-dataset-master/MIDI/*.mid` (~1000 folk tunes)

### 3. Install Dependencies

**Option A - Using uv (Recommended)**:
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies from uv.lock
uv sync

# Activate the environment
source .venv/bin/activate
```

**Option B - Using pip**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Install Dependencies

**Option A - Using uv (Recommended)**:
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies from uv.lock
uv sync

# Activate the environment
source .venv/bin/activate
```

**Option B - Using pip**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Verify Setup
```bash
# Quick test
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import miditok; import pretty_midi; print('Dependencies OK')"

# Check dataset
ls data/nottingham-dataset-master/MIDI/*.mid | wc -l  # Should show ~1000
```

### 5. Preprocess the Dataset (Optional)
```bash
# Activate your environment first
source .venv/bin/activate  # or: .venv/bin/python for individual commands

# Preprocess for naive tokenization (creates data/naive/)
python utils/preprocess_naive.py

# Preprocess for MidiTok tokenization (creates data/miditok/)
python utils/preprocess_miditok.py

# Preprocess for measure-based discriminator (creates data/measures/)
python utils/measure_dataset.py
```

This creates:
- `data/naive/sequences.npy` (983KB) - Naive tokenized sequences
- `data/naive/note_to_int.pkl` (1.7KB) - Vocabulary mapping
- `data/miditok/sequences.npy` (2.6MB) - MidiTok tokenized sequences  
- `data/miditok/vocab.json` (6KB) - MidiTok vocabulary
- `data/measures/measure_sequences.npy` (3.5MB) - Measure-based sequences
- `data/measures/pitch_vocab.pkl` (354B) - Pitch vocabulary

### 4. Verify Setup
```bash
# Quick test
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import miditok; import pretty_midi; print('Dependencies OK')"

# Verify preprocessed data exists
ls data/naive/sequences.npy data/miditok/sequences.npy data/measures/measure_sequences.npy
```

**Note**: The Nottingham MIDI dataset is included in this repository at `data/nottingham-dataset-master/MIDI/` (~1200 folk tunes).

## Hardware and Environment

### Development Platform
- **Hardware**: MacBook Air M1 (8GB RAM)
- **OS**: macOS Sequoia
- **Acceleration**: MPS (Metal Performance Shaders) / CUDA / CPU
- **Python**: 3.10+
- **PyTorch**: 2.9.0+

### Training Time Estimates
On M1 MacBook Air:
- **LSTM Generator** (5 epochs): ~15 minutes
- **GRU Generator** (20 epochs): ~50 minutes
- **Transformer models**: ~1-2 hours

---

## Project Structure

```
csce585-midi/
├── training/                    # Training scripts
│   ├── train_generator.py       # Train generator models
│   └── train_discriminator.py   # Train discriminator models
├── models/                      # Model architectures
│   ├── generators/              # LSTM, GRU, Transformer generators
│   └── discriminators/          # MLP, LSTM, Transformer discriminators
├── utils/                       # Helper functions
│   ├── midi_utils.py           # MIDI file handling
│   ├── sampling.py             # Generation sampling strategies
│   └── preprocess_*.py         # Data preprocessing
├── data/                        # Datasets and preprocessed data
│   ├── nottingham-dataset-master/  # Raw MIDI files
│   ├── naive/                   # Naive tokenization
│   └── miditok/                # MidiTok tokenization
├── evaluate.py                  # Evaluate generated MIDI
├── generate.py                  # Generate music from trained models
├── pyproject.toml              # Project dependencies
└── uv.lock                     # Locked dependency versions
```

---

## Quick Start

### 1. Train a Generator Model
```bash
# Activate environment
source .venv/bin/activate

# Train LSTM generator (15 min)
python training/train_generator.py --model_type lstm --epochs 5

# Train with custom settings
python training/train_generator.py \
    --model_type lstm \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_size 512
```

**Available generator types**: `lstm`, `gru`, `transformer`

### 2. Generate Music
```bash
# Generate with trained model
python generate.py \
    --model_path models/naive/lstm_20251020_210637.pth \
    --output outputs/my_song.mid \
    --strategy greedy \
    --length 200

# Try different sampling strategies
python generate.py --model_path models/naive/lstm_*.pth --strategy top_k --k 5
python generate.py --model_path models/naive/lstm_*.pth --strategy top_p --p 0.9
```

**Available strategies**: `greedy`, `top_k`, `top_p`, `temperature`

### 3. Evaluate Generated MIDI
```bash
# Evaluate single file
python evaluate.py outputs/my_song.mid

# Evaluate multiple files
python evaluate.py outputs/midi/*.mid
```

**Metrics computed**:
- Note density
- Pitch range
- Polyphony
- Duration
- Number of notes

---

## Training Options

### Generator Training
```bash
python training/train_generator.py [OPTIONS]

Options:
  --model_type {lstm,gru,transformer}   Model architecture (default: lstm)
  --epochs INT                          Number of epochs (default: 10)
  --batch_size INT                      Batch size (default: 128)
  --lr FLOAT                            Learning rate (default: 0.001)
  --hidden_size INT                     Hidden layer size (default: 512)
  --num_layers INT                      Number of layers (default: 2)
  --dropout FLOAT                       Dropout rate (default: 0.3)
  --seed INT                            Random seed for reproducibility
```

### Discriminator Training
```bash
python training/train_discriminator.py [OPTIONS]

Options:
  --model_type {mlp,lstm,transformer}   Model architecture
  --epochs INT                          Number of epochs
  --batch_size INT                      Batch size
  --lr FLOAT                            Learning rate
```

---

## What's Included

**Source Code** (All included in repo)
- Generator models: LSTM, GRU, Transformer
- Discriminator models: MLP, LSTM, Transformer  
- Training scripts with factory pattern
- Evaluation and generation utilities
- Data preprocessing pipelines

**Data** (Included in repo)
- Nottingham MIDI dataset (~1200 folk tunes) at `data/nottingham-dataset-master/MIDI/`
- Small tokenizer configs: `vocab.json`, `note_to_int.pkl` (~16KB total)

**Data** (Generated by preprocessing - NOT in repo)
- `data/naive/sequences.npy` (983KB) - Run `utils/preprocess_naive.py`
- `data/miditok/sequences.npy` (2.6MB) - Run `utils/preprocess_miditok.py`
- `data/measures/measure_sequences.npy` (3.5MB) - Run `utils/measure_dataset.py`

**Not Included** (`.gitignore`'d - too large or regeneratable)
- Trained model checkpoints (`*.pth`, `*.pt`) - Train locally
- Generated MIDI outputs - Regenerate with `generate.py`
- Training logs and plots - Created during training

---

## Dependencies

All dependencies are managed via `pyproject.toml` and locked in `uv.lock`:

**Core**:
- `torch>=2.9.0` - Deep learning framework
- `numpy>=1.26.4` - Numerical computing

**Music Processing**:
- `miditok>=3.0.0` - MIDI tokenization
- `music21>=9.1.0` - Music analysis
- `pretty-midi>=0.2.10` - MIDI file handling

**ML/Visualization**:
- `scikit-learn>=1.7.2` - Metrics
- `matplotlib>=3.9.2`, `seaborn>=0.13.2` - Plotting
- `pandas>=2.2.3` - Data management
- `tqdm>=4.66.5` - Progress bars


## Troubleshooting

**Installation Issues**:
```bash
# If uv sync fails, try pip
pip install -e .

# Check Python version
python --version  # Should be 3.10+
```
