"""smart seed selection without use of any models"""
import numpy as np
from pathlib import Path
import pickle

def _compute_data_ranges(sequences, int_to_note, dataset, sample_size=2000):
    """
    This takes the given sequences and figures out the values that should be used for ranges. These ranges allow the user to
    easily select the type of seed they want. Made to fix the issue where no seeds were found by the hardcoded range values.
    """
    sample = sequences if len(sequences) <= sample_size else sequences[np.random.choice(len(sequences), sample_size, replace=False)]
    avg_pitches = []
    unique_counts = []
    lengths = []
    for seq in sample:
        props = analyze_sequence_properties(seq, int_to_note, dataset)
        if props is None:
            continue
        avg_pitches.append(props['avg_pitch'])
        unique_counts.append(props['num_unique_pitches'])
        lengths.append(props['length'])
    if len(avg_pitches) == 0:
        # go back to defaults
        return {
            'pitch_ranges': {"low": (0, 40), "medium": (40, 72), "high": (72, 127)},
            'complexity_ranges': {"simple": (1, 5), "medium": (6, 9), "complex": (10, 100)},
            'length_ranges': {"short": (1, 50), "medium": (51, 150), "long": (151, 1000)}
        }
    
    p33, p66 = np.percentile(avg_pitches, [33, 66])  # get value where 1/3 of dataset is less than that value (same with 66)
    u33, u66 = np.percentile(unique_counts, [33, 66])
    l33, l66 = np.percentile(lengths, [33, 66])
    pitch_min, pitch_max = int(min(avg_pitches)), int(max(avg_pitches))
    uniq_min, uniq_max = int(min(unique_counts)), int(max(unique_counts))
    len_min, len_max = int(min(lengths)), int(max(lengths))

    pitch_ranges = {
        "low": (pitch_min, max(pitch_min, int(np.floor(p33)))),
        "medium": (int(np.floor(p33)), int(np.ceil(p66))),
        "high": (int(np.ceil(p66)), pitch_max)
    }
    complexity_ranges = {
        "simple": (uniq_min, max(uniq_min, int(np.floor(u33)))),
        "medium": (int(np.floor(u33)), int(np.ceil(u66))),
        "complex": (int(np.ceil(u66)), uniq_max)
    }
    length_ranges = {
        "short": (len_min, max(len_min, int(np.floor(l33)))),
        "medium": (int(np.floor(l33)), int(np.ceil(l66))),
        "long": (int(np.ceil(l66)), len_max)
    }
    return {
        'pitch_ranges': pitch_ranges,
        'complexity_ranges': complexity_ranges,
        'length_ranges': length_ranges
    }

def analyze_sequence_properties(sequence, int_to_note, dataset="naive"):
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

    if dataset == "naive":    
        for note_str in notes:
            if '.' in note_str:  # chord
                
                pitches.extend([int(p) for p in note_str.split('.')])
            else:  # single note
                try:
                    pitch = int(''.join(filter(str.isdigit, note_str)))
                    pitches.append(pitch)
                except:
                    pass
    elif dataset == "miditok" or dataset == "miditok_augmented":
        for note_str in notes:
            try:
                pitch = int(''.join(filter(str.isdigit, note_str)))
                pitches.append(pitch)
            except:
                pass
    else:
        print(f"Invalid dataset {dataset} in seed_selection.py")
        return
    
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
    dataset="naive",
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

    # call the helper function to figure out the ranges that should be used for these seqs
    ranges = _compute_data_ranges(sequences, int_to_note, dataset)
    pitch_ranges = ranges['pitch_ranges']
    complexity_ranges = ranges['complexity_ranges']
    length_ranges = ranges['length_ranges']

    # analyze all sequences and only keep the ones that match desired properites
    candidates = []
    for seq in sequences:
        props = analyze_sequence_properties(seq, int_to_note, dataset=dataset)
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