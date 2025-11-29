# csce585-midi

## Group Info

Cade Stocker

- Email: cstocker@email.sc.edu

  

## Requirements
Use uv sync as detailed below in Setup Instructions section of README

Additional info on this [wiki page](https://github.com/csce585-mlsystems/csce585-midi/wiki/Dependencies).

## Python Version

This project uses **Python 3.11.10**. Python 3.13+ is not supported due to NumPy compatibility issues.

The project includes a `.python-version` file that automatically selects Python 3.11 for tools like `uv` and `pyenv`.

## Project Summary/Abstract
This project is a customizable MIDI training/generation framework. Any dataset of MIDI files can be placed in the project. Datasets can be preprocessed using 3 different tokenization types, described on this [wiki page](https://github.com/csce585-mlsystems/csce585-midi/wiki/Preprocessing). Users can train their own custom [generator and discriminator models](https://github.com/csce585-mlsystems/csce585-midi/wiki/Models) with many different customizable options. Training and generation data are logged, allowing easy experimentation. Users may additionally use any MIDI file they wish as a seed during generation, allowing the model to build off of any existing song. The experimentation portion of this project focuses on the quantitative differences in different measures of generated MIDI files based on differently trained models as well as with different generation methods.
  

## Problem Description

This project explores AI music generation via MIDI files. MIDI files contain instructions, rather than audio data, which tell software how to play a song. These instructions are contained in chunks, containing event data such as notes and control changes. Despite not being human readable, MIDI data is easily translatable into a variety of formats, and is used as the core for Digital Audio Workstation editors. Although AI models such as MusicLM exist to generate music, these create raw audio in the form of waveforms. As such, it is very hard for a user to iterate upon its creations, as changes would require the entire waveform to be regenerated. The use of MIDI allows for small, incremental tweaks, while still keeping the end user as part of the process through their DAW.

I am following the architecture described in "Musenet: Music Generation using Abstractive and Generative Methods". As described in the paper, I will be using a discriminator (which selects the chord for the next measure based on previous measures) and a generator (which generates notes based on previous measures, and the output on the discriminator). The use of a factory design pattern for both the generator and discriminator enables multiple types of each to be trained (transformer, lstm, mlp, ...)

Output from the models (MIDI File) is analyzed by PrettyMIDI library, which finds metrics such as:

- Polyphony

- Number of notes

- Pitch range

- Note density

- Duration

The customizable features of the project will allow easy experimentation with variations in training and generation.

## Motivations

- Provide quantitative comparisons between different variations of both training and generation methods.

- Recreating Musenet architecture but with more customizable features.

- Framework that can allow users to train their own MIDI generating models and generate MIDI files easily.

- Allow users to work with any MIDI dataset that they want.

## Challenges

- Small size of Nottingham dataset was causing overfitting within the models trained on miditok tokens.

- Small models trained locally (Macbook Air M1) made it difficult to train a working transformer that could outperform the baseline naive lstm models.

- The original project idea was taking a user text description and turning it into a MIDI file. This was a large project scope to do by myself,
  so the project shifted more towards making a playground for users to build their own models in an easy and modular way.

- I ran into many issues while learning how to run my project on the the cloud. Eventually, after many problems with Colab and AWS EC2 instances, I eventually used Lambda Cloud to train models with a GH200 GPU.

  

## Contribution
- **[Extension of Existing Work]**
   - [Abhilash Pal, Saurav Saha, R. Anita (2020) Musenet : Music Generation using Abstractive and Generative Methods](https://www.researchgate.net/publication/363856706_Musenet_Music_Generation_using_Abstractive_and_Generative_Methods)
- **[Novel Contribution]**
   - Gives the user the ability to experiment with the Musenet architecture
   - Helpful utilities are contained for working with MIDI files
   - Compares different ways of tokenizing MIDI files 
  
  

## References
[Abhilash Pal, Saurav Saha, R. Anita (2020) Musenet : Music Generation using Abstractive and Generative Methods](https://www.researchgate.net/publication/363856706_Musenet_Music_Generation_using_Abstractive_and_Generative_Methods)

[Zheng Jiang (2019) Automatic Analysis of Music in Standard MIDI Files](https://www.cs.cmu.edu/~music/cmp/theses/zheng_jiang_thesis.pdf)

  

[Zhu, Y., Baca, J., Rekabdar, B., Rawassizadeh, R. (2023). A Survey of AI Music Generation Tools and Models](https://arxiv.org/abs/2308.12982)

  

[Briot, J., Hadjeres, G., Pachet, F. (2017). Deep Learning Techniques for Music Generation -- A Survey](https://arxiv.org/abs/1709.01620)

  

[Bhandari, K., Roy, A., Wang, K., Puri, G., Colton, S., Herremans, D. (2024). Text2midi: Generating Symbolic Music from Captions](https://arxiv.org/abs/2412.16526)

  

[Yang, L., Chou, S., Yang, Y. (2017). MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation](https://arxiv.org/abs/1703.10847)

  

[Tian, S., Zhang, C., Yuan, W., Tan, W., Zhu, W. (2025). XMusic: Towards a Generalized and Controllable Symbolic Music Generation Framework](https://arxiv.org/abs/2501.08809)

  

[Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". _PhD Thesis_, 2016](https://colinraffel.com/publications/thesis.pdf)

---  

### Hardware and Environment

-  **Hardware**: MacBook Air M1 (8GB RAM) and A100 GPU via Google Colab

-  **OS**: macOS Sequoia

-  **Acceleration**: MPS (Metal Performance Shaders) / CUDA / CPU

-  **Python**: 3.11.10

-  **PyTorch**: 2.9.0+ on my Macbook, but please view this [wiki page](https://github.com/csce585-mlsystems/csce585-midi/wiki/Dependencies) if you'd liike to use Cuda.

## Project Structure

**Please refer to the wiki if you have any questions about the project**
```
- data/
  - holds your dataset(s)
  - holds output from preprocessing scripts ex. ```data/nottingham_naive```
- docs/
  - holds P0, P1, P2 for class presentations
- models/
  - models/generators/
    - generator_factory.py     # allows you to create a generator of specified type
    - generator_gru.py         # definition of GRUGenerator class
    - generator_lstm.py        # definition of LSTMGenerator class
    - generator_transformer.py # definition of TransformerGenerator class
  - models/discriminators/     # discriminator dir follows same pattern as above
    - discriminator_factory.py
    - discriminator_lstm.py
    - discriminator_mlp.py
    - discriminator_transformer.py
- tests/ # tests for pytest
- training/
  - train_discriminator.py  # script for training discriminator models
  - train_generator.py      # script for training generator models
- utils/
  - augment_dataset.py      # script for preprocessing a dataset with augmentation (transposition)
  - download_small_aria.sh  # script for downloading a dataset with around 30,000 MIDI files
  - measure_dataset.py      # script for preprocessing data for discriminators
  - preprocess_all.sh       # script to run naive, miditok, and measure preprocessing
  - preprocess_miditok.py   # script to preprocess a dataset with miditok tokens
  - preprocess_naive.py     # script to preprocess a dataset with naive tokens
  - sampling.py             # defines different sampling methods (top-p, top-k, random, greedy)
  - seed_control.py         # seed control for reliability across experiments
  - seed_selection.py       # finds a seed from a dataset by looking for sequences matching criteria given by the user
  - midi_to_seed.py         # takes any midi file and turns into tokens from specified preprocessed data
  - analyze_logs.py         # analyzes all logs and creates a report
  - diagnose_generation.py  # takes a model and generates several MIDI files to tests the generator's output
  - find_best_midis.py      # looks through all generated MIDI files and ranks them
  - 
- .coverage                 # used for coverage in pytest
- .coveragec                # same as above
- .gitignore                # files to ignore from version control
- .python-version           # specifies that this project uses python 3.11.10
- evaluate.py               # evaluates a midi file by metrics like pitch range, polyphony, etc.
- generate.py               # script for using a model to generate a MIDI file
- pyproject.toml            # used for specifying dependencies for uv lock
- uv.lock                   # used for installing dependencies
- SetupNotebook.ipynb       # colab notebook that follows the steps listed in this README
```
---

## Setup Instructions
### 1. Clone the Repository

```bash
git clone https://github.com/csce585-mlsystems/csce585-midi.git
cd csce585-midi
```

### 2. Download the Dataset
**Download the nottingham dataset (midi files)**
This is the default for the project. 

Please refer to both [Preprocessing Wiki Page](https://github.com/csce585-mlsystems/csce585-midi/wiki/Preprocessing) and [Using Other Datasets Wiki Page](https://github.com/csce585-mlsystems/csce585-midi/wiki/Using-Other-Datasets) if you'd like to use a different dataset.
```bash

# Option 1: Clone the dataset repository
cd data/
git clone https://github.com/jukedeck/nottingham-dataset.git nottingham-dataset-master
cd ..
```
```
# Option 2: Download and extract manually
Visit: https://github.com/jukedeck/nottingham-dataset
# Download ZIP and extract to data/nottingham-dataset-master/
```
The dataset should be at: `data/nottingham-dataset-master/MIDI/*.mid` (~1200 folk tunes)

  

### 3. Install Dependencies and Activate

```bash
uv sync
source  .venv/bin/activate
```
  
### 4. Verify Setup

```bash

# Quick test
python  -c  "import torch; print('PyTorch:', torch.__version__)"
python  -c  "import miditok; import pretty_midi; print('Dependencies OK')"

# Check dataset
ls  data/nottingham-dataset-master/MIDI/*.mid  |  wc  -l  # Should show ~1000
```

  

### 5. Preprocess the Dataset
**Suggested:**
```bash
# Preprocess all
./utils/preprocess_all.sh data/nottingham-dataset-master/MIDI
```

**If you'd like to preprocess a dataset with a specified tokenizer:**
```bash
# Preprocess for naive tokenization
python utils/preprocess_naive.py --dataset data/nottingham-dataset-master/MIDI --output_dir data/nottingham_naive
```

```bash
# Preprocess for MidiTok tokenization
python utils/preprocess_miditok.py --dataset data/nottingham-dataset-master/MIDI --output_dir data/nottingham_miditok
```

```bash
# Preprocess for measure-based tokens for discriminator
python utils/measure_dataset.py --dataset data/nottingham-dataset-master/MIDI --output_dir data/nottingham_measure
```

```bash
# Augmenting transposes each tokenized song to several different keys in your desired token type, thereby creating more data
python utils/augment_dataset.py --input_dir data/nottingham-dataset-master/MIDI --token_type naive --output_dir data/nottingham_naive_augmented --transpositions=-5,3,-1,0,1,2,5
```

This creates:

**Naive**
-  `data/nottingham-dataset-master_naive/sequences.npy` - Naive tokenized sequences (sequences of IDs representing tokens)

-  `data/nottingham-dataset-master_naive/note_to_int.pkl`  - Vocabulary mapping (dictionary of notes mapped to their ID)

**Miditok**

-  `data/nottingham-dataset-master_miditok/sequences.npy`  - MidiTok tokenized sequences (sequence of the IDs of the notes)

-  `data/nottingham-dataset-master_miditok/vocab.json`  - MidiTok vocabulary (dictionary connecting tokens to their IDs)

-  `data/nottingham-dataset-master_miditok/config.json` - Config details about Miditok (vocab size, number of sequences, etc.)

- `data/nottingham-dataset-master_miditok/tokenizer.json` - Info about the tokenizer (chord maps, pitch range, etc.)

**Miditok Augmented (Miditok tokenized songs transposed to different keys)**
-  `data/miditok_augmented/sequences.npy`  - MidiTok tokenized sequences (sequence of the IDs of the notes)

-  `data/miditok_augmented/vocab.json`  - MidiTok vocabulary (dictionary connecting tokens to their IDs)

-  `data/miditok_augmented/config.json` - Config details about Miditok (vocab size, number of sequences, etc.)

- `data/miditok_augmented/tokenizer.json` - Info about the tokenizer (chord maps, pitch range, etc.)

**Measures (Discriminator)**

-  `data/nottingham_naive_augmented/measure_sequences.npy`  - Measure-based sequences (all the notes in a measure for each measure)

-  `data/nottingham_naive_augmented/pitch_vocab.pkl` - Pitch vocabulary

  

### 4. Verify Setup
```bash
# Quick test
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import miditok; import pretty_midi; print('Dependencies OK')"

# Verify preprocessed data exists
ls data/naive/sequences.npy data/miditok/sequences.npy data/measures/measure_sequences.npy
```

## Quick Start

### 1. Train a Generator Model
Command-line arguments for generator training are detailed in the on this [wiki page](https://github.com/csce585-mlsystems/csce585-midi/wiki/Training).

```bash
# Make sure venv is activated
source  .venv/bin/activate

# Train a small naive LSTM generator to make sure everything is working
python training/train_generator.py  --dataset data/nottingham-dataset-master_naive --model_type  lstm  --epochs  4 --max_batches 5

# Train with custom settings (optional)
python  training/train_generator.py  \
--dataset data/nottingham_naive \
--model_type lstm \
--epochs  20  \
--batch_size 128 \
--lr  0.001  \
--hidden_size 512
```

**Available generator types**: `lstm`, `gru`, `transformer`
**Available discriminator types**: `lstm`, `mlp`, `transformer`


### 2. Generate Music

```bash
# Generate with trained model (substitute with real model path of model you trained and the model's type
python generate.py --data_dir data/nottingham-dataset-master_naive \
--model_path models/generators/checkpoints/nottingham-dataset-master_naive/transformer_20251127_104724.pth \
--model_type transformer 

# Try different sampling strategies
python  generate.py  --model_path  models/<path>.pth  ---model_type <type> --data_dir <directory to the preprocessed data> --strategy  top_k  --k  5
python  generate.py  --model_path  models/<path>.pth  ---model_type <type> --data_dir <directory to the preprocessed data>  --strategy  top_p  --p  0.9
```

**Available strategies**: `greedy`, `top_k`, `top_p`, `temperature`

  

### 3. Evaluate Generated MIDI (THIS IS DONE AUTOMATICALLY WHEM MIDI FILES ARE GENRERATED SO YOU DON'T HAVE TO WORRY ABOUT IT.
You could also do it manually if you really felt like it:

```bash
# Evaluate single file
python  evaluate.py  outputs/my_song.mid
  
# Evaluate multiple files
python  evaluate.py  outputs/midi/*.mid

```
 
**Metrics computed**:

- Note density

- Pitch range

- Polyphony

- Duration

- Number of notes

---

  
## Additional Info
Please visit [The Wiki](https://github.com/csce585-mlsystems/csce585-midi/wiki) if you have any questions. I tried to make it a good resource for using this project.
  

## What's Included
**Source Code** (All included in repo)

- Generator models: LSTM, GRU, Transformer

- Discriminator models: MLP, LSTM, Transformer

- Training scripts with factory pattern

- Evaluation and generation utilities

- Data preprocessing pipelines

- Util scripts
  

**Not Included** (`.gitignore`'d - too large or regeneratable)

- Trained model checkpoints (`*.pth`, `*.pt`) - Train locally

- Generated MIDI outputs - Regenerate with `generate.py`

- Training logs and plots - Created during training

---

## Dependencies

All dependencies are managed via `pyproject.toml` and locked in `uv.lock`:

**Here is the [wiki page](https://github.com/csce585-mlsystems/csce585-midi/wiki/Dependencies) for dependencies.**

## Troubleshooting
Refer to the dependencies wiki if you start having issues with cuda. There is also a wiki page describing how to use other datasets.

Make sure to use ```screen``` in your terminal if you're training models or preprocessing large datasets on the cloud.
