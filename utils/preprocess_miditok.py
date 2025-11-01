import miditok
from symusic import Score
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

"""
Looks over every midi file in the input directory, as a score,
then tokenize each track (with REMI). This converts musical events (notes, timing, etc.)
into integers. Tokenized data is then saved with numpy.
    Sequences go to sequences.npy
    config is saved in json file (holds data needed to decode tokens live vocab, special tokens, ...)
    tokenizer params - config settings that define how the midi is converted to tokens

    here's an example from claude sonnet 4.5 of what would be in the tokenizer.json file:
    {
    "tokenizer_name": "REMI",
    "pitch_range": [21, 109],           // MIDI notes (A0 to C#8)
    "beat_res": {                       // Time resolution per beat
        "0": 8,                           // 8 positions per quarter note
        "1": 4
    },
    "num_velocities": 32,               // Velocity levels (0-127 → 32 bins)
    "special_tokens": [
        "PAD",                            // Padding token
        "BOS",                            // Beginning of sequence
        "EOS",                            // End of sequence
        "MASK"                            // For masked language modeling
    ],
    "use_chords": false,
    "use_rests": false,
    "use_tempos": true,
    "use_time_signatures": true,
    "use_programs": true,               // Track instruments
    "one_token_stream_for_programs": true,
    "program_changes": false,
    "sustain_pedal_duration": false,
    "pitch_bend": false,
    "delete_equal_successive_time_sig_changes": true,
    "delete_equal_successive_tempo_changes": true
    }

"""

INPUT_DIR = Path("data/nottingham-dataset-master/MIDI")
OUTPUT_DIR = Path("data/miditok")
OUTPUT_DIR.mkdir(exist_ok=True)

# Default training seq length (used in train.py slicing, not here)
SEQ_LENGTH = 100

def preprocess_miditok():
    """
    Preprocess MIDI files from nottingham using miditok

    1. load all MIDI files from INPUT_DIR
    2. tokenize each file using REMI tokenization
    3. save tokenized sequences as .npy file
    4. save tokenizer config and vocab
    """

    print("=" * 60)
    print("MIDITOK PREPROCESSING")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sequence length (for training): {SEQ_LENGTH}")
    print()

    # init tokenizer
    tokenizer = miditok.REMI()
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")
    print()

    # collect all midi files
    midi_files = list(INPUT_DIR.glob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files")
    print()

    sequences = []
    skipped_files = []

    # process each MIDI
    print("Processing MIDI files...")
    # tqdm progress bar for terminal
    for midi_file in tqdm(midi_files, desc="Tokenizing"):
        try:
            # load MIDI as a score
            score = Score(midi_file)
            # tokenize the score
            token_seqs = tokenizer(score)

            # Handle multiple tracks (list of TokSequence)
            if isinstance(token_seqs, list):
                for ts in token_seqs:
                    # skip any empty tracks
                    if len(ts.ids) > 0:
                        sequences.append(ts.ids)  # save each full track as ints
            # else if it's just one track
            else:
                sequences.append(token_seqs.ids)

        except RuntimeError as e:
            if "MiniMidi" in str(e) or "MThd" in str(e):
                skipped_files.append((midi_file.name, "Invalid MIDI format"))
            else:
                skipped_files.append((midi_file.name, str(e)))
        except Exception as e:
            skipped_files.append((midi_file.name, str(e)))
    
    print()
    print(f"Successfully tokenized: {len(sequences)} sequences")
    print(f"Skipped {len(skipped_files)} files")

    if skipped_files:
        print("\nSkipped files:")
        print("first 10 skipped files:\n")
        for filename, reason in skipped_files[:10]: # first 10
            print(f" - {filename}: {reason}")
        if len(skipped_files) > 10:
            print(f" ... and {len(skipped_files) - 10} more")

    print()

    # stats
    seq_lengths = [len(seq) for seq in sequences]
    print("Sequence stats:")
    print(f"    Total sequences: {len(sequences)}")
    print(f"    Min length: {min(seq_lengths) if seq_lengths else 0}")
    print(f"    Max length: {max(seq_lengths) if seq_lengths else 0}")
    print(f"    Mean length: {np.mean(seq_lengths):.1f}" if seq_lengths else 0)
    print(f"    Median length: {np.median(seq_lengths):.1f}" if seq_lengths else 0)
    print()

    # save sequences
    sequences_path = OUTPUT_DIR / "sequences.npy"
    np.save(sequences_path, np.array(sequences, dtype=object), allow_pickle=True)
    print(f"Saved sequences to: {sequences_path}")

    # save tokenizer params
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    tokenizer.save_params(tokenizer_path)
    print(f"Saved tokenizer params to: {tokenizer_path}")

    # save vocab
    vocab = tokenizer.vocab
    vocab_path = OUTPUT_DIR / "vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab to: {vocab_path}")

    # save metadata
    config = {
        "seq_length": SEQ_LENGTH,
        "tokenizer": tokenizer.__class__.__name__,
        "num_sequences": len(sequences),
        "num_files_processed": len(midi_files) - len(skipped_files),
        "num_files_skipped": len(skipped_files),
        "vocab_size": len(tokenizer),
        "min_seq_length": int(min(seq_lengths)) if seq_lengths else 0,
        "max_seq_length": int(max(seq_lengths)) if seq_lengths else 0,
        "mean_seq_length": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
    }

    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print(f"  - sequences.npy      ({len(sequences)} sequences)")
    print(f"  - tokenizer.json     (tokenizer configuration)")
    print(f"  - vocab.json         ({len(tokenizer)} tokens)")
    print(f"  - config.json        (dataset metadata)")
    print()
    print("Next steps:")
    print("  1. Train a model: python training/train_generator.py --dataset miditok")
    print("  2. Generate music: python generate.py --model_path models/miditok/model.pth")
    print()

if __name__ == "__main__":
    preprocess_miditok()