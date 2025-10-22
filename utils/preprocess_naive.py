import os
from pathlib import Path
import numpy as np
from music21 import converter, instrument, note, chord
import pickle

"""Preprocess all files in the data/nottingham-dataset-master/MIDI/ directory and save
the processed sequences as a numpy array in data/sequences.npy. Each sequence is a list of integers
representing notes/chords. The mapping from notes/chords to integers is also built and can be saved if needed."""


"""This preprocessing uses the music21 library to parse MIDI files and extract notes and chords.
calling it naive because it does not do any advanced filtering or cleaning of the data. other preprocessing
file uses miditok which is more robust. having these separate allows experimentation with different preprocessing methods."""

# create output directory if it doesn't exist
OUTPUT_DIR = Path("data/naive")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "sequences.npy"
VOCAB_FILE = OUTPUT_DIR / "note_to_int.pkl"

# path to the MIDI dataset
DATA_DIR = "data/nottingham-dataset-master/MIDI/"
# path to save the processed sequences
#OUTPUT_FILE = "data/naive/sequences.npy"

"""Convert a MIDI file to a sequence of note/chord strings.
    Output format: ['C4', 'E4', 'G4', 'C5', 'E5', 'G5', ...]
    Chords are represented as dot-separated pitch classes, e.g., '0.4."""
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
    # Get all elements from the first part
    elements = parts.parts[0].recurse() if parts else midi.flat.notes

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
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                file_path = os.path.join(root, file)
                notes = midi_to_notes(file_path)
                if notes:
                    sequences.append(notes)

    # flatten and build vocabulary (possible tokens)
    all_notes = [note for seq in sequences for note in seq]
    # sort them to ensure consistent ordering
    vocab = sorted(set(all_notes))
    # map notes to integers
    note_to_int = {note: i for i, note in enumerate(vocab)}

    # convert sequences to int format
    int_sequences = [[note_to_int[n] for n in seq] for seq in sequences]

    np.save(output_file, np.array(int_sequences, dtype=object), allow_pickle=True)
    print(f"Saved {len(int_sequences)} sequences to {output_file}")
    print(f"Vocabulary size: {len(vocab)}")

    # save the note-to-int mapping
    with open("data/naive/note_to_int.pkl", "wb") as f:
        pickle.dump({"note_to_int": note_to_int,
                     "int_to_note": {i: n for n, i in note_to_int.items()}}, f)

    """Saved to the pickle file like this: {"C4": 0, "E4": 1, ...}"""
    print("Saved note-to-int mapping to data/naive/note_to_int.pkl")

if __name__ == "__main__":
    build_dataset()