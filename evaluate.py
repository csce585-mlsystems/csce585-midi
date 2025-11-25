import argparse
import csv
from pathlib import Path
import pretty_midi

"""
Get all of the notes from a MIDI file, and then calculate:
    note density,
    pitches,
    pitch range,
    average polyphony,
    number of notes
"""
def evaluate_midi(midi_path, logfile):
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
    avg_polyphony = (
        sum(len(instr.notes) for instr in pm.instruments) / len(pm.instruments)
        if pm.instruments else 0
    ) # average number of notes played simultaneously

    return {
        "output_file": str(midi_path),
        "num_notes": num_notes,
        "duration": duration,
        "note_density": note_density,
        "pitch_range": pitch_range,
        "avg_polyphony": avg_polyphony
    }

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