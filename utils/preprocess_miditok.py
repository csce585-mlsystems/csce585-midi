import argparse
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
        sequences of ints representing the notes (sequence for each song)
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

OUTPUT_DIR = Path("data/miditok")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# default training seq length
SEQ_LENGTH = 100

def preprocess_miditok(input_dir, output_dir=None):
    """
    preprocess midi files using miditok remi without any augmentation
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # initialize tokenizer
    tokenizer = miditok.REMI()

    # collect all midi files (recursive search)
    midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))

    sequences = []
    skipped_files = []
    num_files_processed = 0

    print(f"processing {len(midi_files)} MIDI files from {input_dir}")

    for midi_file in tqdm(midi_files, desc="processing files"):
        try:
            # load score
            score = Score(midi_file)

            # tokenize
            token_seqs = tokenizer(score)

            # handle potential list of seqs (tracks)
            song_tracks = []
            if isinstance(token_seqs, list):
                for ts in token_seqs:
                    if len(ts.ids) > 0:
                        song_tracks.append(ts.ids)
            else:
                if len(token_seqs.ids) > 0:
                    song_tracks.append(token_seqs.ids)
            
            if song_tracks:
                sequences.append(song_tracks)
                num_files_processed += 1

        except Exception as e:
            skipped_files.append((midi_file.name, str(e)))

    if skipped_files:
        print("First 5 skipped files errors:")
        for name, err in skipped_files[:5]:
            print(f"    {name}: {err}")

    # save sequences
    sequences_path = output_dir/ "sequences.npy"
    np.save(sequences_path, np.array(sequences, dtype=object), allow_pickle=True)
    print(f"Saved sequences to: {sequences_path}")

    # save tokenizer params
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save_params(tokenizer_path)

    # save vocab
    vocab = tokenizer.vocab
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    # calculate statistics
    all_tracks = [track for song in sequences for track in song]
    seq_lengths = [len(track) for track in all_tracks] if all_tracks else [0]

    # save metadata
    config = {
        "seq_length": SEQ_LENGTH,
        "tokenizer": tokenizer.__class__.__name__,
        "num_sequences": len(sequences),
        "num_files_processed": num_files_processed,
        "num_files_skipped": len(skipped_files),
        "vocab_size": len(tokenizer),
        "min_seq_length": min(seq_lengths) if seq_lengths else 0,
        "max_seq_length": max(seq_lengths) if seq_lengths else 0,
        "mean_seq_length": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
        "augmentation": {
            "enabled": False
        }
    }

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIDI with MIDITok (no augmentation)")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Path to the directory containing raw MIDI files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/miditok",
        help="Output directory for processed data"
    )
    
    args = parser.parse_args()
    
    preprocess_miditok(
        input_dir=args.dataset, 
        output_dir=args.output_dir
    )