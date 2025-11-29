"""
Utility to convert a MIDI file into a seed sequence that can be used for generation.

This module takes an arbitrary MIDI file (not necessarily from the training dataset)
and converts it using the same preprocessing logic as the dataset, producing a seed
sequence compatible with trained models.
"""

import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from symusic import Score
import miditok
import sys

# Add utils directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_naive import midi_to_notes, pitch_to_string


def midi_to_seed_naive(midi_filepath, dataset_dir="data/naive", seq_length=None):
    """
    Convert a MIDI file to a seed sequence using naive preprocessing.
    
    Args:
        midi_filepath: Path to the MIDI file to convert
        dataset_dir: Path to the preprocessed dataset directory (contains note_to_int.pkl)
        seq_length: Desired length of seed sequence. If None, uses entire file.
                   If longer than file, pads. If shorter, truncates from start.
    
    Returns:
        List of integers representing the seed sequence, or None on error
    """
    midi_filepath = Path(midi_filepath)
    dataset_dir = Path(dataset_dir)
    
    # Validate inputs
    if not midi_filepath.exists():
        print(f"Error: MIDI file not found: {midi_filepath}")
        return None
    
    vocab_file = dataset_dir / "note_to_int.pkl"
    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found: {vocab_file}")
        print("Please run preprocessing first to generate the vocabulary.")
        return None
    
    # Load the vocabulary mapping
    with open(vocab_file, "rb") as f:
        vocab_data = pickle.load(f)
    note_to_int = vocab_data["note_to_int"]
    
    print(f"Loading MIDI file: {midi_filepath}")
    
    # Extract notes from MIDI file using same logic as preprocessing
    notes = midi_to_notes(str(midi_filepath))
    
    if not notes:
        print(f"Error: Could not extract notes from {midi_filepath}")
        return None
    
    print(f"Extracted {len(notes)} notes from MIDI file")
    
    # Convert notes to integers
    seed_sequence = []
    unknown_notes = set()
    
    for note in notes:
        if note in note_to_int:
            seed_sequence.append(note_to_int[note])
        else:
            # Track unknown notes for reporting
            unknown_notes.add(note)
    
    if unknown_notes:
        print(f"Warning: {len(unknown_notes)} unique notes not in vocabulary were skipped:")
        for note in sorted(list(unknown_notes))[:10]:  # Show first 10
            print(f"  - {note}")
        if len(unknown_notes) > 10:
            print(f"  ... and {len(unknown_notes) - 10} more")
    
    if not seed_sequence:
        print("Error: No valid notes found in vocabulary")
        return None
    
    print(f"Converted to {len(seed_sequence)} integer tokens")
    
    # Handle sequence length
    if seq_length is not None:
        if len(seed_sequence) > seq_length:
            # Take the last seq_length tokens (most recent music)
            print(f"Truncating sequence to last {seq_length} tokens")
            seed_sequence = seed_sequence[-seq_length:]
        elif len(seed_sequence) < seq_length:
            # Pad by repeating the sequence
            print(f"Padding sequence from {len(seed_sequence)} to {seq_length} tokens")
            original_len = len(seed_sequence)
            while len(seed_sequence) < seq_length:
                # Repeat sequence as many times as needed
                remaining = seq_length - len(seed_sequence)
                seed_sequence.extend(seed_sequence[:min(original_len, remaining)])
            seed_sequence = seed_sequence[:seq_length]
    
    print(f"Final seed sequence length: {len(seed_sequence)}")
    return seed_sequence


def midi_to_seed_miditok(midi_filepath, dataset_dir="data/miditok", seq_length=None):
    """
    Convert a MIDI file to a seed sequence using MidiTok preprocessing.
    
    Args:
        midi_filepath: Path to the MIDI file to convert
        dataset_dir: Path to the preprocessed dataset directory (contains vocab.json and tokenizer config)
        seq_length: Desired length of seed sequence. If None, uses entire file.
                   If longer than file, pads. If shorter, truncates from start.
    
    Returns:
        List of integers representing the seed sequence, or None on error
    """
    midi_filepath = Path(midi_filepath)
    dataset_dir = Path(dataset_dir)
    
    # Validate inputs
    if not midi_filepath.exists():
        print(f"Error: MIDI file not found: {midi_filepath}")
        return None
    
    vocab_file = dataset_dir / "vocab.json"
    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found: {vocab_file}")
        print("Please run preprocessing first to generate the vocabulary.")
        return None
    
    # Load vocabulary
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    
    print(f"Loading MIDI file: {midi_filepath}")
    
    # Tokenize the MIDI file using miditok
    try:
        # Use REMI tokenizer (same as preprocessing)
        tokenizer = miditok.REMI()
        
        # Load the MIDI file as a Score
        score = Score(str(midi_filepath))
        
        # Tokenize the score
        tokens = tokenizer.encode(score)
        
        # tokens is a list of TokSequence objects (one per track)
        # Flatten all tracks into a single sequence
        seed_sequence = []
        for tok_seq in tokens:
            seed_sequence.extend(tok_seq.ids)
        
        if not seed_sequence:
            print(f"Error: No tokens extracted from {midi_filepath}")
            return None
        
        print(f"Extracted {len(seed_sequence)} tokens from MIDI file")
        
        # Filter out any tokens not in vocabulary
        vocab_size = len(vocab)
        filtered_sequence = [t for t in seed_sequence if t < vocab_size]
        
        if len(filtered_sequence) < len(seed_sequence):
            print(f"Warning: Filtered out {len(seed_sequence) - len(filtered_sequence)} out-of-vocabulary tokens")
        
        seed_sequence = filtered_sequence
        
        if not seed_sequence:
            print("Error: No valid tokens found in vocabulary")
            return None
        
        print(f"Converted to {len(seed_sequence)} valid tokens")
        
    except Exception as e:
        print(f"Error tokenizing MIDI file: {e}")
        return None
    
    # Handle sequence length
    if seq_length is not None:
        if len(seed_sequence) > seq_length:
            # Take the last seq_length tokens (most recent music)
            print(f"Truncating sequence to last {seq_length} tokens")
            seed_sequence = seed_sequence[-seq_length:]
        elif len(seed_sequence) < seq_length:
            # Pad by repeating the sequence
            print(f"Padding sequence from {len(seed_sequence)} to {seq_length} tokens")
            original_len = len(seed_sequence)
            while len(seed_sequence) < seq_length:
                # Repeat sequence as many times as needed
                remaining = seq_length - len(seed_sequence)
                seed_sequence.extend(seed_sequence[:min(original_len, remaining)])
            seed_sequence = seed_sequence[:seq_length]
    
    print(f"Final seed sequence length: {len(seed_sequence)}")
    return seed_sequence


def midi_to_seed(midi_filepath, dataset_dir, seq_length=None):
    """
    Convert a MIDI file to a seed sequence compatible with a trained model.
    
    Automatically detects whether to use naive or miditok preprocessing based on
    the dataset directory structure.
    
    Args:
        midi_filepath: Path to the MIDI file to convert
        dataset_dir: Path to the preprocessed dataset directory
        seq_length: Desired length of seed sequence. If None, uses entire file.
    
    Returns:
        List of integers representing the seed sequence, or None on error
    
    Examples:
        # For a model trained on naive preprocessed data
        seed = midi_to_seed("my_song.mid", "data/naive", seq_length=50)
        
        # For a model trained on miditok preprocessed data
        seed = midi_to_seed("my_song.mid", "data/miditok", seq_length=100)
    """
    dataset_dir = Path(dataset_dir)
    
    # Detect dataset type by checking which vocab file exists
    if (dataset_dir / "note_to_int.pkl").exists():
        print("Detected naive preprocessing")
        return midi_to_seed_naive(midi_filepath, dataset_dir, seq_length)
    elif (dataset_dir / "vocab.json").exists():
        print("Detected miditok preprocessing")
        return midi_to_seed_miditok(midi_filepath, dataset_dir, seq_length)
    else:
        print(f"Error: Could not detect dataset type in {dataset_dir}")
        print("Expected to find either 'note_to_int.pkl' (naive) or 'vocab.json' (miditok)")
        return None


def save_seed(seed_sequence, output_file):
    """
    Save a seed sequence to a file for later use.
    
    Args:
        seed_sequence: List of integers representing the seed
        output_file: Path to save the seed (will be saved as .npy file)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_file, np.array(seed_sequence, dtype=np.int32))
    print(f"Saved seed sequence to {output_file}")


def load_seed(seed_file):
    """
    Load a seed sequence from a file.
    
    Args:
        seed_file: Path to the seed file (.npy)
    
    Returns:
        List of integers representing the seed
    """
    seed_file = Path(seed_file)
    if not seed_file.exists():
        print(f"Error: Seed file not found: {seed_file}")
        return None
    
    seed = np.load(seed_file, allow_pickle=True)
    print(f"Loaded seed sequence from {seed_file} (length: {len(seed)})")
    return seed.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a MIDI file to a seed sequence for generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a MIDI file for use with a naive-trained model
  python utils/midi_to_seed.py --midi my_song.mid --dataset data/naive --seq_length 50
  
  # Convert a MIDI file for use with a miditok-trained model
  python utils/midi_to_seed.py --midi my_song.mid --dataset data/miditok --seq_length 100
  
  # Save the seed to a file for later use
  python utils/midi_to_seed.py --midi my_song.mid --dataset data/naive --output seed.npy
        """
    )
    parser.add_argument("--midi", type=str, required=True,
                        help="Path to the MIDI file to convert")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the preprocessed dataset directory (e.g., data/naive or data/miditok)")
    parser.add_argument("--seq_length", type=int, default=None,
                        help="Desired seed sequence length. If not specified, uses entire file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the seed sequence (optional). Will be saved as .npy file.")
    
    args = parser.parse_args()
    
    # Convert MIDI to seed
    seed = midi_to_seed(args.midi, args.dataset, args.seq_length)
    
    if seed is None:
        print("\nFailed to convert MIDI to seed")
        exit(1)
    
    # Save if requested
    if args.output:
        save_seed(seed, args.output)
    
    print("\nSuccess! You can now use this seed for generation.")
