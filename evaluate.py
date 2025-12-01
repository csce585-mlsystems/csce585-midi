import argparse
import csv
from pathlib import Path
import pretty_midi
import math
from collections import Counter
import numpy as np

"""
Get all of the notes from a MIDI file, and then calculate:
    note density,
    pitches,
    pitch range,
    average polyphony,
    number of notes,
    scale consistency,
    pitch entropy
"""
def evaluate_midi(midi_path, logfile=None):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return None
    
    # collect notes
    notes = [n for instr in pm.instruments for n in instr.notes]
    num_notes = len(notes)
    duration = pm.get_end_time() if num_notes > 0 else 0.0 # total duration in seconds

    # metrics
    note_density = num_notes / duration if duration > 0 else 0
    pitches = [n.pitch for n in notes]
    pitch_range = max(pitches) - min(pitches) if pitches else 0
    
    # Calculate true polyphony (average number of notes playing simultaneously)
    if duration > 0:
        # Create a timeline of events (note_on, note_off)
        events = []
        for note in notes:
            events.append((note.start, 1))
            events.append((note.end, -1))
        events.sort()
        
        current_polyphony = 0
        polyphony_area = 0
        last_time = 0
        
        for time, change in events:
            polyphony_area += current_polyphony * (time - last_time)
            current_polyphony += change
            last_time = time
            
        avg_polyphony = polyphony_area / duration
    else:
        avg_polyphony = 0

    # Scale Consistency
    scale_consistency = calculate_scale_consistency(pitches)
    
    # Pitch Entropy
    pitch_entropy = calculate_pitch_entropy(pitches)
    
    # Pitch Class Entropy
    pitch_class_entropy = calculate_pitch_class_entropy(pitches)

    return {
        "output_file": str(midi_path),
        "num_notes": num_notes,
        "duration": duration,
        "note_density": note_density,
        "pitch_range": pitch_range,
        "avg_polyphony": avg_polyphony,
        "scale_consistency": scale_consistency,
        "pitch_entropy": pitch_entropy,
        "pitch_class_entropy": pitch_class_entropy
    }

"""
See what how many of the notes stay within 
"""
def calculate_scale_consistency(pitches):
    if not pitches:
        return 0.0
    
    pitch_classes = [p % 12 for p in pitches]
    pc_counts = Counter(pitch_classes)
    total_notes = len(pitches)
    
    # Major scale intervals: 0, 2, 4, 5, 7, 9, 11
    major_intervals = {0, 2, 4, 5, 7, 9, 11}
    # Minor scale intervals: 0, 2, 3, 5, 7, 8, 10 (Natural Minor)
    minor_intervals = {0, 2, 3, 5, 7, 8, 10}
    
    max_consistency = 0.0
    
    for root in range(12):
        # Check Major
        major_scale = {(root + i) % 12 for i in major_intervals}
        in_scale_count = sum(count for pc, count in pc_counts.items() if pc in major_scale)
        max_consistency = max(max_consistency, in_scale_count / total_notes)
        
        # Check Minor
        minor_scale = {(root + i) % 12 for i in minor_intervals}
        in_scale_count = sum(count for pc, count in pc_counts.items() if pc in minor_scale)
        max_consistency = max(max_consistency, in_scale_count / total_notes)
        
    return max_consistency

def calculate_pitch_entropy(pitches):
    if not pitches:
        return 0.0
    pitch_counts = Counter(pitches)
    total = len(pitches)
    probs = [count / total for count in pitch_counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def calculate_pitch_class_entropy(pitches):
    if not pitches:
        return 0.0
    pitch_classes = [p % 12 for p in pitches]
    pc_counts = Counter(pitch_classes)
    total = len(pitches)
    probs = [count / total for count in pc_counts.values()]
    return -sum(p * math.log2(p) for p in probs)


# log evaluation results to a CSV file
def log_evaluation(results, logfile):
    # make sure file exists
    file_exists = Path(logfile).exists()
    with open(logfile, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

# if calling this script directly, evaluate provided MIDI files
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated MIDI files.")
    parser.add_argument("midi_files", nargs="+", help="Paths to generated MIDI files to evaluate.")
    args = parser.parse_args()

    # evaluate each provided MIDI file and log results
    for midi_file in args.midi_files:
        results = evaluate_midi(midi_file)
        if results:
            log_evaluation(results)
            print(f"Evaluated {midi_file}: {results}")
        else:
            print(f"Failed to evaluate {midi_file}.")