import argparse
import os
from pathlib import Path
import numpy as np
from music21 import converter, instrument, note, chord
import pickle
import signal
from contextlib import contextmanager
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from numpy.lib.stride_tricks import sliding_window_view

# Constants
DATA_DIR = Path("data/nottingham-dataset-master/MIDI")
OUTPUT_DIR = Path("data/naive")
OUTPUT_FILE = OUTPUT_DIR / "sequences.npy"
VOCAB_FILE = OUTPUT_DIR / "note_to_int.pkl"
TIMEOUT_SECONDS = 10

class TimeoutException(Exception):
    pass

# timer (if the code running with this timer doesn't finish within the time
# limit, gives timeout exception)
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield  # allows you to use 'with' and call this function
    finally:
        signal.alarm(0)

"""Preprocess all files in the data/nottingham-dataset-master/MIDI/ directory and save
the processed sequences as a numpy array in data/sequences.npy. Each sequence is a list of integers
representing notes/chords. The mapping from notes/chords to integers is also built and can be saved if needed."""


"""This preprocessing uses the music21 library to parse MIDI files and extract notes and chords.
calling it naive because it does not do any advanced filtering or cleaning of the data. other preprocessing
file uses miditok which is more robust. having these separate allows experimentation with different preprocessing methods."""

def midi_to_notes(midi_file):
    """
    Parses a MIDI file and returns a list of notes/chords as strings.
    """
    notes = []
    try:
        midi = converter.parse(midi_file)
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        # represent chords and notes differently
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    except Exception as e:
        print(f"Failed to parse {midi_file}: {e}")
        return []
    return notes

def build_dataset(data_dir, output_file, vocab_file=None, seq_length=100):
    input_dir = Path(data_dir)
    output_file = Path(output_file)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    notes = []
    # recursive search for .mid and .midi files
    midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))

    print(f"Preprocessing {len(midi_files)} files from {input_dir}...")

    # parallel processing
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, midi_files), total=len(midi_files)))
    
    for res in results:
        notes.extend(res)

    # create mapping
    pitchnames = sorted(set(item for item in notes))  # put all of the notes you found in here
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))  # map the notes to ints
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))  # map ints to notes

    print(f"Vocab size: {len(note_to_int)}")

    # save dictionary
    if vocab_file:
        vocab_file = Path(vocab_file)
        with open(vocab_file, "wb") as f:
            pickle.dump({"note_to_int": note_to_int, "int_to_note": int_to_note}, f)

    # convert notes to ints
    print("Converting notes to integers...")
    all_ints = np.array([note_to_int[n] for n in notes], dtype=np.int32)

    print("Creating sequences...")
    
    if len(all_ints) < seq_length:
        print(f"Warning: Total notes ({len(all_ints)}) is less than sequence length ({seq_length}). No sequences created.")
        sequences = np.empty((0, seq_length), dtype=np.int32)
    else:
        # Use sliding_window_view for fast sequence generation
        try:
            sequences = sliding_window_view(all_ints, window_shape=seq_length)
        except AttributeError:
            # Fallback for older numpy versions
            shape = (all_ints.shape[0] - seq_length + 1, seq_length)
            strides = (all_ints.strides[0], all_ints.strides[0])
            sequences = np.lib.stride_tricks.as_strided(all_ints, shape=shape, strides=strides)

    # save sequences
    np.save(output_file, sequences)
    print(f"Saved {len(sequences)} sequences to {output_file}")

def preprocess_naive(input_dir, output_dir="data/naive", seq_length=100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    build_dataset(
        data_dir=input_dir,
        output_file=output_dir / "sequences.npy",
        vocab_file=output_dir / "note_to_int.pkl",
        seq_length=seq_length
    )

def process_file(file_path):
    """
    Wrapper for midi_to_notes to handle timeout and exceptions in multiprocessing.
    """
    try:
        with time_limit(TIMEOUT_SECONDS):
            return midi_to_notes(file_path)
    except TimeoutException:
        return []
    except Exception:
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to raw MIDI files")
    parser.add_argument("--output_dir", type=str, default="data/naive", help="Output directory")
    args = parser.parse_args()

    preprocess_naive(args.dataset, args.output_dir)
