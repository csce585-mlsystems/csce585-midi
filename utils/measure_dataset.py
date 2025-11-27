import argparse
import numpy as np
from pathlib import Path
import pretty_midi
from collections import defaultdict # allows you to set default values for each key
import pickle

from tqdm import tqdm # for saving/loading data

""" I keep getting runtime warnings. I asked Claude if they matter:
    responded with: 
    
        The warning is about MIDI file format standards (Type 0 vs Type 1)

        Some files in your Nottingham dataset have tempo/key/time signature events on instrument tracks instead of the dedicated control track
        
        This is a formatting issue, not a data corruption issue
"""

""" don't need to use miditok tokens. the discriminator will basically predict a list of notes likely to appear
in the next measure. So we can just use the raw MIDI pitches (integers 0-127) as our "vocab" 

Think of it as using all the pitches from a previous measure to predict the pitches in the next measure"""

# path to midi folder (data)
MIDI_FOLDER = Path("data/nottingham-dataset-master/MIDI")

def midi_to_measure_pitches(midi_path, beats_per_bar=4, tempo_bpm_default=120.0, quantize=None):
    """ Convert a MIDI file to a list of measures, where each measure is a 
    set of unique pitches (integers)."""
    
    try:
        # load the MIDI file
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None
    
    # pick median tempo if available
    try:
        # returns array of tempos (bpm) and when each of those tempos start
        tempos, tempo_times = pm.get_tempo_changes()
        tempo_bpm = float(np.median(tempos)) if len(tempos) > 0 else tempo_bpm_default
    except Exception as e:
        # if error, just use default tempo
        tempo_bpm = tempo_bpm_default

    # if no tempo changes, use default tempo
    try:
        # how long each beat is in seconds
        seconds_per_beat = 60.0 / tempo_bpm
    except Exception as e:
        # if error, just use default value
        seconds_per_beat = 60.0 / tempo_bpm_default

    # how long each measure is in seconds
    measure_seconds = seconds_per_beat * beats_per_bar
    # total length of the MIDI in seconds
    end_time = pm.get_end_time()
    # how long the midi is in secs / how long each measure is in secs = number of measures
    num_measures = int(np.ceil(end_time / measure_seconds))

    # create a list of sets, one set per measure to hold unique pitches
    measures = [set() for _ in range(num_measures)]

    # iterate over each note from each instrument
    for inst in pm.instruments:
        for n in inst.notes:
            # get the start time of the note
            start = n.start
            # floor of the start time divided by the measure length gives the measure index
            measure_idx = int(start // measure_seconds) # the measure you're in

            # only add if within range (some notes might start after the end time due to rounding)
            if 0 <= measure_idx < num_measures:
                # add each pitch to the appropriate measure set (ie. measure 1 contains an A, C, E etc.)
                measures[measure_idx].add(n.pitch)

    # return the list of sets
    return measures

def build_pitch_vocab_from_midi_folder(midi_folder=MIDI_FOLDER):
    """ Build a pitch vocabulary from all MIDI files in the specified folder.
    Returns a sorted list of unique pitches and a mapping from pitch to integer index. """

    # folder holding the midi files
    midi_folder = Path(midi_folder)
    # set to hold unique pitches across all files (all possible notes)
    pitches = set()

    # get anything ending in .mid or .midi
    files = list(midi_folder.rglob("*.mid")) + list(midi_folder.rglob("*.midi"))

    # for each midi file
    for f in files:
        measures = midi_to_measure_pitches(f)
        for m in measures:
            pitches.update(m)

    # create a sorted list of pitches and a mapping from pitch to int
    pitch_vocab = sorted(pitches) # sort the pitches by pitch number
    mapping = {p:i for i, p in enumerate(pitch_vocab)} # associate each pitch with an int
    
    return pitch_vocab, mapping

def build_measure_dataset(midi_folder, out_dir="data/measures", beats_per_bar=4, tempo_bpm=120.0):
    """ Measure dataset is a sequence of measures, where each measure is represented as a binary vector
    indicating which pitches (from the pitch vocab) are present in that measure."""
    
    # where the output data will be saved
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True) # ensure output dir exists
    
    # where to find the midi files
    midi_folder = Path(midi_folder)

    # build pitch mapping
    # reminder: pitch_vocab = all possible pitches (sorted list of ints)
    # mapping = dict mapping pitch to int index
    pitch_vocab, mapping = build_pitch_vocab_from_midi_folder(midi_folder)

    with open(out_dir / "pitch_vocab.pkl", "wb") as f:
        pickle.dump({"vocab": pitch_vocab, "mapping": mapping}, f)

    all_examples = [] # to hold all the measure examples

    # get all of the files recursively
    files = list(midi_folder.rglob("*.mid")) + list(midi_folder.rglob("*.midi"))

    # for each midi file
    for f in tqdm(files):
        # get the measures (list of sets of pitches)
        measures = midi_to_measure_pitches(f, beats_per_bar=beats_per_bar, tempo_bpm_default=tempo_bpm)
        if len(measures) < 2 or measures is None:
            continue # skip files with less than 2 measures

        # convert each measure set to a binary vector
        # each vector is of length len(pitch_vocab), with 1 if the pitch is present, 0 if not
        measure_vecs = []
        for m in measures:
            vec = np.zeros(len(pitch_vocab), dtype=np.uint8) # binary vector of vocab size
            for pitch in m:
                # if pitch is there, fill it in as 1 (true)
                if pitch in mapping:
                    vec[mapping[pitch]] = 1

            # append the vector for this measure
            measure_vecs.append(vec)

        # append the measure vectors for this file to the overall examples list
        all_examples.append(np.array(measure_vecs, dtype=object))

    # save the measure dataset as a numpy file (array of arrays)
    examples_array = np.array(all_examples, dtype=object)
    np.save(out_dir / "measure_sequences.npy", examples_array, allow_pickle=True)
    print(f"Saved measurements: {len(all_examples)} sequences to {out_dir / 'measure_sequences.npy'}")
    print(f"Pitch vocab size: {len(pitch_vocab)} saved to {out_dir / 'pitch_vocab.pkl'}")
    return all_examples, pitch_vocab, mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIDI dataset into measure sequences.")
    parser.add_argument("--dataset", type=str, required=True, help="path to the dataset you want to preprocess")
    parser.add_argument("--output_dir", type=str, default="data/measures", help="where you want the output to be stored")

    args = parser.parse_args()

    # build the measure dataset from the MIDI folder
    build_measure_dataset(midi_folder=args.dataset, out_dir=args.output_dir)