
"""
Making this to get rid of current augment_miditok. I want this file to be used for tokenizing a given dataset by the given 
token type, as well as transposing the songs into multtiple keys to make the dataset larger.
output should go to specified directory given from command-line

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

import os
import miditok
from symusic import Score
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import argparse
import tempfile
import pickle
import sys
import signal
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count

# Default training seq length
SEQ_LENGTH = 100
TIMEOUT_SECONDS = 30  # Timeout for processing a single file

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Transposition intervals (in semitones)
# Positive = transpose up, negative = transpose down
DEFAULT_TRANSPOSITIONS = [-5, -3, -1, 0, 1, 3, 5]  # 7 versions of each song

# Helper function for fast normal order calculation
def fast_normal_order(pcs):
    """
    Calculate normal order of a list of pitch classes (integers 0-11).
    Matches music21.chord.Chord(pcs).normalOrder but much faster.
    """
    # Remove duplicates and sort
    pcs = sorted(list(set(pcs)))
    if not pcs:
        return []
    
    n = len(pcs)
    if n == 1:
        return pcs
    
    # Create doubled list to handle wrap-around
    doubled = pcs + [pc + 12 for pc in pcs]
    
    best_rotation = pcs
    min_span = 100
    
    # Check all rotations of length n
    for i in range(n):
        # Window from i to i+n-1
        span = doubled[i + n - 1] - doubled[i]
        
        if span < min_span:
            min_span = span
            best_rotation = [x % 12 for x in doubled[i : i + n]]
        elif span == min_span:
            # Tie-breaking: music21 prefers the one with smaller intervals at the start
            # This is equivalent to lexicographical comparison of the rotation
            curr_rotation = [x % 12 for x in doubled[i : i + n]]
            if curr_rotation < best_rotation:
                best_rotation = curr_rotation
                
    return best_rotation

# Helper function for multiprocessing
def process_file_naive(args):
    """
    Process a single MIDI file for naive tokenization with augmentation.
    Args:
        args: tuple of (midi_file_path, transpositions)
    Returns:
        list of (semitones, notes_list) tuples
    """
    midi_file, transpositions = args
    results = []
    
    try:
        with time_limit(TIMEOUT_SECONDS):
            # Import music21 here to avoid pickling issues
            from music21 import converter, instrument, note, chord, pitch
            
            # Pre-compute pitch names map if not cached (local to process)
            midi_to_name = {i: pitch.Pitch(i).nameWithOctave for i in range(128)}
                    
            try:
                # Parse ONCE with music21
                try:
                    midi = converter.parse(midi_file)
                except Exception as e:
                    return [("file_error", f"Parse error: {str(e)}")]

                # Get the part to process (logic from midi_to_notes)
                parts = instrument.partitionByInstrument(midi)
                if parts:
                    original_part = parts.parts[0]
                else:
                    original_part = midi.flat

                # Extract base events (int for Note, list[int] for Chord)
                base_events = []
                iterator = original_part.recurse()
                
                for element in iterator:
                    if isinstance(element, note.Note):
                        base_events.append(('note', element.pitch.midi))
                    elif isinstance(element, chord.Chord):
                        base_events.append(('chord', element.normalOrder))

                # Apply transpositions to the extracted events
                for semitones in transpositions:
                    try:
                        notes = []
                        for event_type, data in base_events:
                            if event_type == 'note':
                                # Transpose pitch
                                new_midi = data + semitones
                                if 0 <= new_midi <= 127:
                                    notes.append(midi_to_name[new_midi])
                                
                            elif event_type == 'chord':
                                # Transpose PCs
                                new_pcs = [(pc + semitones) % 12 for pc in data]
                                # Calculate normal order
                                no = fast_normal_order(new_pcs)
                                notes.append('.'.join(str(n) for n in no))
                        
                        if notes:
                            results.append((semitones, notes))
                            
                    except Exception as e:
                        results.append((semitones, f"Error: {str(e)}"))
                        
            except Exception as e:
                return [("file_error", str(e))]
                
    except TimeoutException:
        return [("file_error", "Timeout")]
    except Exception as e:
        return [("file_error", f"Unexpected error: {str(e)}")]
        
    return results

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


def augment_dataset(input_dir, transpositions=None, output_dir=None, token_type="naive"):
    """
    Preprocess MIDI files with data augmentation via transposition.
    
    Args:
        input_dir: path to the dir holding the midi files
        transpositions: list of int, semitone intervals to transpose to
        output_dir: Path, output directory for augmented data
    """

    # make sure token type is valid
    token_type = token_type.lower()

    if token_type != "naive" and token_type != "miditok":
        raise ValueError(f"You must pick one of the two token_types (naive and miditok). '{token_type}' is invalid")

    # DIRECTORIES (MAKE OUTPUT_DIR IF IT DOESN'T EXIST ALREADY)
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileExistsError(f"Input directory not found")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if transpositions is None:
        transpositions = DEFAULT_TRANSPOSITIONS

    # IF YOU'RE USING MIDITOK AS TOKENIZER
    if token_type == "miditok":
        # Initialize tokenizer
        tokenizer = miditok.REMI()

        # Collect all MIDI files (recursive search)
        midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))

        sequences = []
        skipped_files = []
        augmentation_stats = {t: 0 for t in transpositions}

        # Process each MIDI file with augmentation
        print("Processing MIDI files with transposition augmentation...")
        
        """
        Go over each file. Transpose it, then tokenize it (make it a sequence)
        """
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
            
            # make sure it's the right midi format
            except RuntimeError as e:
                if "MiniMidi" in str(e) or "MThd" in str(e):
                    skipped_files.append((midi_file.name, "Invalid MIDI format"))
                else:
                    skipped_files.append((midi_file.name, str(e)))
            except Exception as e:
                skipped_files.append((midi_file.name, str(e)))

        # print the files that got skipped over
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

    # IF USING NAIVE TOKENS
    elif token_type == "naive":
        sequences_str = []
        skipped_files = []
        augmentation_stats = {t: 0 for t in transpositions}

        # Collect all MIDI files (recursive search)
        midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))

        # Process each MIDI file with augmentation using multiprocessing
        print(f"Processing {len(midi_files)} MIDI files with transposition augmentation (Naive) using {cpu_count()} cores...")
        
        # Prepare arguments for worker function
        tasks = [(str(f), transpositions) for f in midi_files]
        
        with Pool(processes=cpu_count()) as pool:
            # Use imap_unordered for better performance as order doesn't matter for collection
            # But we need to collect results
            results = list(tqdm(pool.imap(process_file_naive, tasks), total=len(tasks)))
            
        # Process results
        for i, file_results in enumerate(results):
            midi_filename = midi_files[i].name
            
            # Check for file-level error
            if len(file_results) == 1 and file_results[0][0] == "file_error":
                skipped_files.append((midi_filename, file_results[0][1]))
                continue
                
            for semitones, output in file_results:
                if isinstance(output, str) and output.startswith("Error:"):
                    skipped_files.append((f"{midi_filename} (transpose {semitones:+d})", output))
                elif isinstance(output, list):
                    # It's a list of notes
                    if output:
                        sequences_str.append(output)
                        augmentation_stats[semitones] += 1

        # print the files that got skipped over
        if skipped_files:
            print("\nSkipped files/transpositions:")
            print("First 10:\n")
            for filename, reason in skipped_files[:10]:
                print(f"  - {filename}: {reason}")
            if len(skipped_files) > 10:
                print(f"  ... and {len(skipped_files) - 10} more")

        print()

        # Build vocabulary
        print("Building vocabulary...")
        all_notes = set()
        for seq in sequences_str:
            all_notes.update(seq)
        
        pitchnames = sorted(all_notes)
        note_to_int = {note: number for number, note in enumerate(pitchnames)}
        int_to_note = {number: note for number, note in enumerate(pitchnames)}
        
        print(f"Vocab size: {len(note_to_int)}")
        
        # Convert to ints
        print("Converting to integers...")
        sequences = []
        for seq in sequences_str:
            sequences.append([note_to_int[n] for n in seq])

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

        # Save vocab (pickle for naive compatibility)
        vocab_path = output_dir / "note_to_int.pkl"
        with open(vocab_path, "wb") as f:
            pickle.dump({"note_to_int": note_to_int, "int_to_note": int_to_note}, f)
        print(f"Saved vocab to: {vocab_path}")
        
        # Also save as json for inspection
        vocab_json_path = output_dir / "vocab.json"
        with open(vocab_json_path, "w") as f:
            json.dump(note_to_int, f, indent=2)

        # Save metadata
        config = {
            "seq_length": SEQ_LENGTH,
            "tokenizer": "naive",
            "num_sequences": len(sequences),
            "num_files_processed": len(midi_files) - len([s for s in skipped_files if "transpose" not in s[0]]),
            "num_files_skipped": len([s for s in skipped_files if "transpose" not in s[0]]),
            "vocab_size": len(note_to_int),
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
        if len(note_to_int) > 0:
            print(f"  - Samples per vocab entry: {total_samples / len(note_to_int):.1f}")
        else:
            print(f"  - Samples per vocab entry: 0.0")
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
    parser.add_argument(
        "--token_type",
        type=str,
        default="miditok",
        choices=["miditok", "naive"],
        help="Tokenization method: 'miditok' or 'naive'"
    )
    
    args = parser.parse_args()
    
    # Parse transpositions
    transpositions = [int(t.strip()) for t in args.transpositions.split(",")]
    
    augment_dataset(
        input_dir=args.input_dir, 
        transpositions=transpositions, 
        output_dir=args.output_dir,
        token_type=args.token_type
    )