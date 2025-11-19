import os
from pathlib import Path
import numpy as np
from music21 import converter, instrument, note, chord
import pickle
import signal
from contextlib import contextmanager
from tqdm import tqdm


"""Preprocess all files in the data/nottingham-dataset-master/MIDI/ directory and save
the processed sequences as a numpy array in data/sequences.npy. Each sequence is a list of integers
representing notes/chords. The mapping from notes/chords to integers is also built and can be saved if needed."""


"""This preprocessing uses the music21 library to parse MIDI files and extract notes and chords.
calling it naive because it does not do any advanced filtering or cleaning of the data. other preprocessing
file uses miditok which is more robust. having these separate allows experimentation with different preprocessing methods."""

# create output directory if it doesn't exist
OUTPUT_DIR = Path("data/naive")
OUTPUT_DIR.mkdir(exist_ok=True)

# output files
OUTPUT_FILE = OUTPUT_DIR / "sequences.npy"
VOCAB_FILE = OUTPUT_DIR / "note_to_int.pkl"

# path to the MIDI dataset
DATA_DIR = "data/nottingham-dataset-master/MIDI/"

# make sure it times out if taking too long
TIMEOUT_SECONDS = 40

# type of exception
class TimeoutException(Exception):
    pass

# exception if longer than timelimit
@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

"""
    Convert a MIDI file to a sequence of note/chord strings.
    Output format: ['C4', 'E4', 'G4', 'C5', 'E5', 'G5', ...]
    Chords are represented as dot-separated pitch classes, e.g., '0.4.
"""
def midi_to_notes(file_path):
    try:
        # Load the MIDI file
        midi = converter.parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []
    
    # Extract notes and chords
    notes = []
    # get the instrument parts
    parts = instrument.partitionByInstrument(midi)

    try:
        # Get all elements from the first part
        elements = parts.parts[0].recurse() if parts else midi.flat.notes
    except Exception as e:
        print(f"error getting elements")
        return []

    # Convert notes and chords to string representation
    for elem in elements:
        # If it's a note, get its pitch
        if isinstance(elem, note.Note):
            notes.append(str(elem.pitch))
        # If it's a chord, get the normal order of pitches
        elif isinstance(elem, chord.Chord):
            notes.append('.'.join(str(n) for n in elem.normalOrder))

    return notes

"""Build the dataset by processing all MIDI files in the DATA_DIR."""
def build_dataset(data_dir=DATA_DIR, output_file=OUTPUT_FILE):
    sequences = []
    skipped_files = []
    
    # collect all MIDI files
    midi_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                midi_files.append(os.path.join(root, file))

    print(f"Found {len(midi_files)} MIDI files")
    print(f"Processing with {TIMEOUT_SECONDS}s timeout per file...\n")

    # process with progress bar and timeout
    for file_path in tqdm(midi_files, desc="Processing"):
        try:
            with time_limit(TIMEOUT_SECONDS):
                notes = midi_to_notes(file_path)
                if notes:
                    sequences.append(notes)
                else:
                    skipped_files.append((Path(file_path).name, "No notes found"))
        except TimeoutException:
            skipped_files.append((Path(file_path).name, f"Timeout (>{TIMEOUT_SECONDS}s)"))
        except Exception as e:
            skipped_files.append((Path(file_path).name, str(e)))
        
    print(f"\nSuccessfully processed: {len(sequences)} files")
    print(f"Skipped: {len(skipped_files)} files")

    # print the skipped files
    if skipped_files and len(skipped_files) <= 10:
        print("\nSkipped files:")
        for filename, reason in skipped_files:
            print(f" - {filename}: {reason}")
    elif skipped_files:
        print(f"\nFirst 10 skipped files:")
        for filename, reason in skipped_files[:10]:
            print(f" - {filename}: {reason}")
        print(f" ... and {len(skipped_files) - 10} more")

    # flatten and build vocab
    all_notes = [note for seq in sequences for note in seq]
    # sort to ensure consistent ordering
    vocab = sorted(set(all_notes))
    # map notes -> ints
    note_to_int = {note: i for i, note in enumerate(vocab)}

    # convert sequences to int format
    int_sequences = [[note_to_int[n] for n in seq] for seq in sequences]

    np.save(output_file, np.array(int_sequences, dtype=object), allow_pickle=True)
    print(f"saved {len(int_sequences)} sequences to {output_file}")
    print(f"Vocabulary size: {len(vocab)}")

    # save int -> note mapping
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump({"note_to_int": note_to_int,
                     "int_to_note": {i: n for n, i in note_to_int.items()}}, f)
        
    print(f"Saved note-to-int mapping to {VOCAB_FILE}")

if __name__ == "__main__":
    build_dataset()