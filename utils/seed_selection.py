"""smart seed selection without use of any models"""
import numpy as np
from pathlib import Path
import pickle

def analyze_sequence_properties(sequence, int_to_note):
    """
    analyze musical properties of a sequence

    returns:
        average pitch
        pitch range
        number of unique pitches
        length of sequence
    """
    notes = [int_to_note[token] for token in sequence]  # get notes from sequence

    # extract pitch numbers (simple heuristic)
    pitches = []
    for note_str in notes:
        if '.' in note_str:  # chord
            pitches.extend([int(p) for p in note_str.split('.')])
        else:  # single note
            try:
                pitch = int(''.join(filter(str.isdigit, note_str)))
                pitches.append(pitch)
            except:
                pass
    
    if len(pitches) == 0:
        return None
    
    return {
        'avg_pitch': np.mean(pitches),
        'pitch_range': max(pitches) - min(pitches),
        'num_unique_pitches': len(set(pitches)),
        'length': len(sequence)
    }

def find_seed_by_characteristics(
    sequences,
    int_to_note,
    pitch_preference="medium",  # "low", "medium", "high"
    complexity="medium",        # "simple", "medium", "complex"
    length="medium"             # "short", "medium", "long"
):
    """
    Find a seed sequence matching desired characteristics specified by user
    
    args:
        pitch_preference: desired pitch range
            - "low": 40-60 (bass/tenor)
            - "medium": 60-72 (comfortable singing range)
            - "high": 72-84 (soprano)
        complexity: number of unique pitches
            - "simple": 3-5 pitches (easy songs)
            - "medium": 6-8 pitches
            - "complex": 9+ pitches
        length: sequence length
            - "short": < 50 notes
            - "medium": 50-100 notes
            - "long": 100+ notes
    """

    # define target ranges
    pitch_ranges = {
        "low": (40, 60),
        "medium": (60, 72),
        "high": (72, 84)
    }

    complexity_ranges = {
        "simple": (3, 5),
        "medium": (6, 8),
        "complex": (9, 20)
    }

    length_ranges = {
        "short": (20, 50),
        "medium": (50, 100),
        "long": (100, 200)
    }

    # analyze all sequences and only keep the ones that match desired properites
    candidates = []
    for seq in sequences:
        props = analyze_sequence_properties(seq, int_to_note)
        if props is None:
            continue

        # check if matches criteria
        pitch_min, pitch_max = pitch_ranges[pitch_preference]
        if not (pitch_min <= props['avg_pitch'] <= pitch_max):
            continue

        comp_min, comp_max = complexity_ranges[complexity]
        if not (comp_min <= props['num_unique_pitches'] <= comp_max):
            continue

        len_min, len_max = length_ranges[length]
        if not (len_min <= props['length'] <= len_max):
            continue

        candidates.append((seq, props))

    if not candidates:
        print(f"no sequences match criteria, using random seed")
        return list(sequences[np.random.randint(len(sequences))])
    
    # pick randomly from candidates
    selected_seq, props = candidates[np.random.randint(len(candidates))]

    print(f"Selected seed:")
    print(f"    avg pitch: {props['avg_pitch']:.1f}")
    print(f"    unique pitches: {props['num_unique_pitches']}")
    print(f"    length: {props['length']}")

    return list(selected_seq)