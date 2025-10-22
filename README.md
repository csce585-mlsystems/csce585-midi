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

### 2. Install Dependencies (using uv - Required)

**Important**: This project uses `uv` for dependency management as required by the course.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies from uv.lock
uv sync

# Activate the environment
source .venv/bin/activate
```

**Alternative (if uv.lock not yet generated)**:
```bash
# Using pip with requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the MIDI Dataset (if you have issues with the dataset in this repo)

**Important**: Due to GitHub file limitations, the MIDI dataset (3,089 files) must be downloaded separately.

**Option A - Direct Download (Recommended)**:
```bash
# Download the original Nottingham dataset
wget https://github.com/jukedeck/nottingham-dataset/archive/master.zip
unzip master.zip
mv nottingham-dataset-master/MIDI data/nottingham-dataset-master/
rm -rf nottingham-dataset-master master.zip
```

**Option B - Git Clone**:
```bash
cd data/
git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
cd ..
```

**Option C - Manual Download**:
1. Visit: https://github.com/jukedeck/nottingham-dataset
2. Download as ZIP
3. Extract and place the `MIDI/` folder in `data/nottingham-dataset-master/MIDI/`

### 4. Verify Setup
```bash
# Check that data files exist
ls data/measures/measure_sequences.npy  # Should exist (included in repo)
ls data/nottingham-dataset-master/MIDI/*.mid | wc -l  # Should show ~1000 files

# Quick test
python -c "import torch; print('PyTorch:', torch.__version__)"
```

## Hardware and Environment

### Development Platform
- **Hardware**: MacBook Air M1 (8GB RAM)
- **OS**: macOS Sequoia
- **Acceleration**: MPS (Metal Performance Shaders)
- **Python**: 3.10+
- **PyTorch**: 2.9.0 with MPS support

### Training Time Estimates
On M1 MacBook Air:
- **LSTM Generator** (971K params, 5 epochs): ~15 minutes
- **LSTM Generator** (20 epochs): ~1 hour
- **GRU Generator** (789K params, 20 epochs): ~50 minutes
- **Full training suite** (6 models): 4-6 hours

See `docs/ENVIRONMENT.md` for complete specifications and reproducibility details.

---

## Reproducing Milestone P1 Results

The following instructions reproduce the preliminary experiment results presented in Milestone P1.

### Prerequisites
1. Complete setup steps 1-4 above (clone, install dependencies, download dataset, verify)
2. Ensure you're in the project root directory with activated virtual environment

### Quick Test (5 minutes)
Run a minimal training experiment to verify setup:

```bash
# Train LSTM generator for 2 epochs (quick test)
python training/train_generator.py --model_type lstm --epochs 2 --batch_size 128

# Expected output:
# - Training should complete in ~6 minutes
# - Final loss should be around 3.2-3.4
# - Model saved to models/naive/lstm_*.pth
```

### Full P1 Experiment (15 minutes)
Reproduce the exact experiment from the P1 report:

```bash
# Train LSTM generator for 5 epochs (as reported in P1)
python training/train_generator.py \
    --model_type lstm \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-3 \
    --seed 42

# Expected results (from docs/P1_PRELIMINARY_RESULTS.md):
# - Epoch 1 loss: ~3.79
# - Epoch 5 loss: ~2.79
# - Training time: ~15 minutes
# - Model parameters: 971K
```

### Verify Results
```bash
# Check training logs
cat logs/generators/naive/models.csv

# View training loss plot
ls outputs/naive/training_loss/

# Generate sample music
python generate.py --model_path models/naive/lstm_*.pth --output test_output.mid
```

### Expected Artifacts
After running the P1 experiment, you should have:
- Trained model: `models/naive/lstm_YYYYMMDD_HHMMSS.pth`
- Training log: `logs/generators/naive/models.csv`
- Loss plot: `outputs/naive/training_loss/training_loss_YYYYMMDD_HHMMSS.png`
- Generated MIDI: `test_output.mid`

### Troubleshooting
- **Out of memory**: Reduce `--batch_size` to 64 or 32
- **MPS not available**: Training will fall back to CPU (slower)
- **Import errors**: Run `uv sync` to ensure all dependencies installed

For detailed results and analysis, see `docs/P1_PRELIMINARY_RESULTS.md`.

---

## What's Included in This Repository

**All Source Code**
- Generator models (LSTM, GRU, Transformer)
- Discriminator models (MLP, LSTM, Transformer)
- Training scripts with factory pattern
- Evaluation and generation scripts

**Preprocessed Data**
- `data/measures/measure_sequences.npy` (3.5MB) - Measure-based sequences
- `data/miditok/sequences.npy` (2.7MB) - MidiTok tokenization
- `data/naive/sequences.npy` (984KB) - Naive tokenization
- Vocabulary and tokenizer files

**Documentation**
- Project proposal and implementation details
- Architecture documentation
- Training visualizations

**Not Included** (Too large or regeneratable)
- MIDI dataset (3,089 files) - **Download separately** (see above)
- Trained model checkpoints (`.pth` files) - Regenerate by training
- Generated MIDI outputs - Regenerate with `generate.py`

## How to Run

### Quick Start - Train a Model (15 minutes)
```bash
source .venv/bin/activate
python training/train_generator.py --model_type lstm --epochs 5
```

### Full Training Suite (4-6 hours)
```bash
source .venv/bin/activate
python scripts/train_all.py
```

This trains:
- 4 discriminator models (MLP, LSTM, Transformer variants)
- 2 generator models (LSTM, GRU)

### Generate Music
```bash
python generate.py --model_path models/naive/lstm_*.pth --output outputs/my_song.mid
```

### Evaluate Models
```bash
python evaluate.py --model_path models/naive/lstm_*.pth
```

---

## Dependencies  

## Directory Structure  
- /training
	The training directory includes files for training both the generator and discriminator models. Both use a factory design pattern, allowing the user to pick between several architectures for each model.

- /outputs
	Currently the outputs directory includes a generator folder and a graphics folder. Inside of /generator, the outputs are separated into /miditok and /naive (these are the two ways of tokenizing that I'm comparing). Generated midi files and automatically generated training loss plots are stored inside of their respective tokenization directory.

- /logs
	Split into discriminators and generators. Generators (each tokenization type) has several log files tracking training data, output data, and evaluation data for each created midi file.

## How to Run  
Using python 3.10.6
## Demo  
---
