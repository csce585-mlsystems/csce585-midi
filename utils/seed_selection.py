"""smart seed selection without use of any models"""
import numpy as np
from pathlib import Path
import pickle
from utils.preprocess_naive import midi_to_notes

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

def get_seed_by_filename(filename, dataset="naive", length=None):
    """
    Get a seed sequence from a specific MIDI file by its filename.
    This lets the user pick a song they want the model to continue/recreate.
    
    Args:
        filename: The name of the MIDI file (e.g., "ashover.mid")
        dataset: Which dataset to use ("naive", "miditok", or "miditok_augmented")
        length: If specified, truncate/pad the seed to this length. If None, use full sequence.
    
    Returns:
        The seed sequence as a list of integers, or None if file not found
    """
    import json
    
    # Determine paths based on dataset
    if dataset == "naive":
        data_dir = Path("data/naive")
        mapping_file = data_dir / "filename_to_index.json"
        sequences_file = data_dir / "sequences.npy"
    elif dataset == "miditok":
        data_dir = Path("data/miditok")
        mapping_file = data_dir / "filename_to_index.json"
        sequences_file = data_dir / "sequences.npy"
    elif dataset == "miditok_augmented":
        data_dir = Path("data/miditok_augmented")
        mapping_file = data_dir / "filename_to_index.json"
        sequences_file = data_dir / "sequences.npy"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Check if files exist
    if not mapping_file.exists():
        print(f"Error: Mapping file not found at {mapping_file}")
        print("Please run preprocessing to generate the filename mapping.")
        return None
    
    if not sequences_file.exists():
        print(f"Error: Sequences file not found at {sequences_file}")
        return None
    
    # Load the filename-to-index mapping
    with open(mapping_file, 'r') as f:
        filename_to_index = json.load(f)
    
    # Check if the filename exists
    if filename not in filename_to_index:
        print(f"Error: File '{filename}' not found in dataset.")
        print(f"Available files ({len(filename_to_index)} total):")
        # Show first 10 filenames as examples
        for i, name in enumerate(sorted(filename_to_index.keys())[:10]):
            print(f"  - {name}")
        if len(filename_to_index) > 10:
            print(f"  ... and {len(filename_to_index) - 10} more")
        return None
    
    # Get the sequence index
    seq_index = filename_to_index[filename]
    
    # Load the sequences
    sequences = np.load(sequences_file, allow_pickle=True)
    
    # Get the sequence
    seed_seq = sequences[seq_index]
    
    # For miditok datasets, sequences are nested (list of tracks)
    # Flatten them for seed selection
    if dataset in ["miditok", "miditok_augmented"]:
        if isinstance(seed_seq, list) and len(seed_seq) > 0:
            # Flatten all tracks into one sequence
            flattened = []
            for track in seed_seq:
                if isinstance(track, (list, np.ndarray)):
                    flattened.extend(track)
                else:
                    flattened.append(track)
            seed_seq = flattened
    
    # Convert to list if it's a numpy array
    seed_seq = list(seed_seq)
    
    # Handle length parameter
    if length is not None:
        if len(seed_seq) > length:
            # Truncate to desired length
            seed_seq = seed_seq[:length]
        elif len(seed_seq) < length:
            # Pad with the sequence repeated
            while len(seed_seq) < length:
                seed_seq.extend(seed_seq[:min(len(seed_seq), length - len(seed_seq))])
            seed_seq = seed_seq[:length]
    
    print(f"Selected seed from file: {filename}")
    print(f"  Sequence index: {seq_index}")
    print(f"  Original length: {len(sequences[seq_index])}")
    print(f"  Seed length: {len(seed_seq)}")
    
    return seed_seq


def list_available_files(dataset="naive", search_term=None):
    """
    List all available MIDI filenames in the dataset.
    
    Args:
        dataset: Which dataset to use ("naive", "miditok", or "miditok_augmented")
        search_term: Optional search term to filter filenames
    
    Returns:
        List of filenames, or None if mapping file not found
    """
    import json
    
    # Determine paths based on dataset
    if dataset == "naive":
        mapping_file = Path("data/naive/filename_to_index.json")
    elif dataset == "miditok":
        mapping_file = Path("data/miditok/filename_to_index.json")
    elif dataset == "miditok_augmented":
        mapping_file = Path("data/miditok_augmented/filename_to_index.json")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Check if file exists
    if not mapping_file.exists():
        print(f"Error: Mapping file not found at {mapping_file}")
        print("Please run preprocessing to generate the filename mapping.")
        return None
    
    # Load the mapping
    with open(mapping_file, 'r') as f:
        filename_to_index = json.load(f)
    
    filenames = sorted(filename_to_index.keys())
    
    # Filter by search term if provided
    if search_term:
        filenames = [f for f in filenames if search_term.lower() in f.lower()]
    
    print(f"Found {len(filenames)} files in {dataset} dataset" + 
          (f" matching '{search_term}'" if search_term else ""))
    
    return filenames


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