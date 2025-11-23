import miditok
from symusic import Score
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import argparse

"""
Data Augmentation for MIDITok preprocessing.

This script augments the MIDI dataset by transposing songs to different keys.
Musical transposition shifts all notes by a fixed number of semitones, creating
new training examples while preserving the musical structure.

Common transpositions:
- Up/down by 1-6 semitones (half steps)
- Avoids extreme transpositions that might go out of MIDI pitch range

This helps reduce overfitting by:
1. Increasing dataset diversity (5x-7x more training examples)
2. Teaching the model pitch-invariant patterns
3. Exposing the model to more note combinations
"""

# INPUT_DIR = Path("data/nottingham-dataset-master/MIDI")
OUTPUT_DIR = Path("data/miditok_augmented")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default training seq length
SEQ_LENGTH = 100

# Transposition intervals (in semitones)
# Positive = transpose up, negative = transpose down
DEFAULT_TRANSPOSITIONS = [-5, -3, -1, 0, 1, 3, 5]  # 7 versions of each song


def transpose_score(score, semitones):
    """
    Transpose a symusic Score by a given number of semitones.
    
    Args:
        score: symusic.Score object
        semitones: int, number of semitones to transpose (positive = up, negative = down)
    
    Returns:
        transposed_score: new symusic.Score object with transposed notes
    """
    # Create a deep copy of the score
    transposed = score.copy()
    
    # Transpose each track (skip drum tracks)
    for track in transposed.tracks:
        if not track.is_drum:
            # Use symusic's built-in shift_pitch method
            # This method handles the transposition and boundary checking
            track.shift_pitch(semitones)
            
            # Optional: filter out notes that went out of range (if any)
            # symusic should handle this automatically, but we can be extra safe
            notes_to_remove = []
            for i, note in enumerate(track.notes):
                if note.pitch < 0 or note.pitch > 127:
                    notes_to_remove.append(i)
            
            # Remove invalid notes in reverse order to maintain indices
            for i in reversed(notes_to_remove):
                track.notes.pop(i)
    
    return transposed


def preprocess_miditok_augmented(input_dir, transpositions=None, output_dir=None):
    """
    Preprocess MIDI files with data augmentation via transposition.
    
    Args:
        input_dir: path to the dir holding the midi files
        transpositions: list of int, semitone intervals to transpose to
        output_dir: Path, output directory for augmented data
    """

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileExistsError(f"Input directory not found")

    if transpositions is None:
        transpositions = DEFAULT_TRANSPOSITIONS
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    tokenizer = miditok.REMI()

    # Collect all MIDI files (recursive search)
    midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))

    sequences = []
    skipped_files = []
    augmentation_stats = {t: 0 for t in transpositions}

    # Process each MIDI file with augmentation
    print("Processing MIDI files with transposition augmentation...")
    
    for midi_file in tqdm(midi_files, desc="Processing files"):
        try:
            # Load original MIDI as a score
            original_score = Score(midi_file)
            
            # Apply each transposition
            for semitones in transpositions:
                try:
                    # Transpose the score
                    if semitones == 0:
                        # No transposition needed for original
                        transposed_score = original_score
                    else:
                        transposed_score = transpose_score(original_score, semitones)
                    
                    # Tokenize the transposed score
                    token_seqs = tokenizer(transposed_score)
                    
                    # Handle multiple tracks (list of TokSequence)
                    if isinstance(token_seqs, list):
                        for ts in token_seqs:
                            # Skip any empty tracks
                            if len(ts.ids) > 0:
                                sequences.append(ts.ids)
                                augmentation_stats[semitones] += 1
                    else:
                        # Single track
                        if len(token_seqs.ids) > 0:
                            sequences.append(token_seqs.ids)
                            augmentation_stats[semitones] += 1
                
                except Exception as e:
                    # If transposition fails, skip this transposition
                    skipped_files.append((f"{midi_file.name} (transpose {semitones:+d})", str(e)))
        
        except RuntimeError as e:
            if "MiniMidi" in str(e) or "MThd" in str(e):
                skipped_files.append((midi_file.name, "Invalid MIDI format"))
            else:
                skipped_files.append((midi_file.name, str(e)))
        except Exception as e:
            skipped_files.append((midi_file.name, str(e)))

    if skipped_files:
        print("\nSkipped files/transpositions:")
        print("First 10:\n")
        for filename, reason in skipped_files[:10]:
            print(f"  - {filename}: {reason}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")

    print()

    # Calculate statistics
    seq_lengths = [len(seq) for seq in sequences]
    print("Sequence statistics:")
    print(f"    Total sequences: {len(sequences):,}")
    print(f"    Min length: {min(seq_lengths) if seq_lengths else 0}")
    print(f"    Max length: {max(seq_lengths) if seq_lengths else 0}")
    print(f"    Mean length: {np.mean(seq_lengths):.1f}" if seq_lengths else 0)
    print(f"    Median length: {np.median(seq_lengths):.1f}" if seq_lengths else 0)
    print()

    # Save sequences
    sequences_path = output_dir / "sequences.npy"
    np.save(sequences_path, np.array(sequences, dtype=object), allow_pickle=True)
    print(f"Saved sequences to: {sequences_path}")

    # Save tokenizer params
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save_params(tokenizer_path)
    print(f"Saved tokenizer params to: {tokenizer_path}")

    # Save vocab
    vocab = tokenizer.vocab
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab to: {vocab_path}")

    # Save metadata
    config = {
        "seq_length": SEQ_LENGTH,
        "tokenizer": tokenizer.__class__.__name__,
        "num_sequences": len(sequences),
        "num_files_processed": len(midi_files) - len([s for s in skipped_files if "transpose" not in s[0]]),
        "num_files_skipped": len([s for s in skipped_files if "transpose" not in s[0]]),
        "vocab_size": len(tokenizer),
        "min_seq_length": int(min(seq_lengths)) if seq_lengths else 0,
        "max_seq_length": int(max(seq_lengths)) if seq_lengths else 0,
        "mean_seq_length": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
        "augmentation": {
            "enabled": True,
            "transpositions": transpositions,
            "augmentation_factor": len(transpositions),
            "breakdown": augmentation_stats
        }
    }

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Calculate expected training samples
    valid_seqs = [seq for seq in sequences if len(seq) >= SEQ_LENGTH]
    total_samples = sum(len(seq) - SEQ_LENGTH for seq in valid_seqs)
    print(f"Expected training samples (seq_length={SEQ_LENGTH}):")
    print(f"  - Valid sequences: {len(valid_seqs):,}")
    print(f"  - Total samples: {total_samples:,}")
    print(f"  - Samples per vocab entry: {total_samples / len(tokenizer):.1f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIDI with data augmentation")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Path to the directory containing raw MIDI files (e.g., 'data/nottingham/MIDI')"
    )
    parser.add_argument(
        "--transpositions",
        type=str,
        default="-5,-3,-1,0,1,3,5",
        help="Comma-separated list of semitone transpositions (e.g., '-5,-3,-1,0,1,3,5')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/miditok_augmented",
        help="Output directory for augmented data"
    )
    
    args = parser.parse_args()
    
    # Parse transpositions
    transpositions = [int(t.strip()) for t in args.transpositions.split(",")]
    
    preprocess_miditok_augmented(input_dir=args.input_dir, transpositions=transpositions, output_dir=args.output_dir)