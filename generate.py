import pickle
import torch
import argparse
import numpy as np
from pathlib import Path
from music21 import stream, note, chord
import models
from models.lstm import LSTMGenerator
from utils.sampling import sample_next_note
from datetime import datetime

# pick device
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# log data about generated midis
LOG_FILE = "logs/output_midis.csv"

# default data paths
SEQUENCES_PATH = "data/sequences.npy"
VOCAB_PATH = "data/note_to_int.pkl"
""" VOCAB is the mapping from notes/chords to integers, saved as a pickle file. These are basically all the
'words'(tokens) the model knows."""

# find the latest model checkpoint in the models/ directory
# if none found, use the default path
latest_model = None
model_files = list(Path("models").glob("lstm_*.pth")) # get the path to model folder

if model_files:
    # gets the timestamp from the filename
    # assumes filenames are like lstm_MMDDHHMM.pth
    def extract_timestamp(filepath):
        """Extract timestamp from filename like lstm_MMDDHHMM.pth"""
        try:
            # Extract the timestamp part after 'lstm_' and before '.pth'
            timestamp_str = filepath.stem.split('_')[1]
            # Convert MMDDHHMM format to a comparable integer
            return int(timestamp_str)
        except (IndexError, ValueError):
            # If parsing fails, fall back to file modification time
            return int(filepath.stat().st_mtime)
    
    # get the most recently created model based on filename timestamp
    latest_model = max(model_files, key=extract_timestamp)
    MODEL_PATH = latest_model
    print(f"Found latest model checkpoint: {latest_model}")
else:
    # if none are found, use the default path
    print("No model checkpoints found in models/ directory. Using default path.")
    MODEL_PATH = "models/lstm_checkpoint.pth"

""" Debugging: the model being selected was trained on just nottingham dataset, newest model was trained on additional midis from
    Thomas (pokemon songs)
    
    UPDATE: USING ONLY NOTTINGHAM NOW AS OF 9/27/2025
    """
#print(f"Loading model from {MODEL_PATH}")

OUTPUT_FILE = "outputs/midi/generated.mid"

def generate(
    strategy="greedy",
    generate_length=100,
    seq_length=50,
    temperature=1.0,
    k=5, # for top-k sampling
    p=0.9, # for nucleus sampling
    model_path=MODEL_PATH,
    vocab_path=VOCAB_PATH,
    sequences_path=SEQUENCES_PATH,
    output_file=OUTPUT_FILE
):
    # load vocab and data
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # create reverse lookup for int to note and vice versa
    note_to_int = vocab["note_to_int"]
    int_to_note = vocab["int_to_note"]
    vocab_size = len(note_to_int)

    # load the sequences (the training data)
    sequences = np.load(sequences_path, allow_pickle=True)

    # pick random seed
    # takes a random sequence, then slices the first seq_length elements
    # think of a seq_length as a certain amount of beats or measures
    seed = list(sequences[np.random.randint(0, len(sequences))][:seq_length])
    generated = seed.copy() # to store the full generated sequence

    # load the model
    model = LSTMGenerator(vocab_size) # initialize the model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)) # load trained weights
    model.to(DEVICE) # move model to the appropriate device
    model.eval() # set to eval mode

    # prepare the input (seed) as a tensor
    input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # generate notes
    for _ in range(generate_length):
        # don't need gradients for inference (save time and space and just unecessary overall)
        with torch.no_grad():
            output, _ = model(input_seq) # get model's output
            logits = output[:, -1, :] # get the logits for the last time step (the next note prediction)

            # sample the next note using the specified strategy
            # .item() converts a single-value tensor to a standard Python number
            next_note_int = sample_next_note(
                logits, strategy=strategy, temperature=temperature, k=k, p=p).item()
            
            generated.append(next_note_int) # add to the generated sequence
            # prepare the input for the next time step
            # -seq_length index is same as first index in the current input_seq
            input_seq = torch.tensor(generated[-seq_length:], dtype=torch.long).unsqueeze(0).to(DEVICE)


    # convert generated integers back to midi
    midi_stream = stream.Stream()
    for idx in generated:
        symbol = int_to_note[idx]
        if "." in symbol: # chord
            # create note objects for each pitch in the chord
            notes_in_chord = [note.Note(int(n)) for n in symbol.split(".")]
            for n in notes_in_chord:
                n.quarterLength = 0.5 # set duration
            midi_stream.append(chord.Chord(notes_in_chord))
        else: # single note
            n = note.Note(symbol)
            n.quarterLength = 0.5 # set duration
            midi_stream.append(n)

    # wrap as a path object for easier handling
    output_file = Path(output_file)
    # create output directory if it doesn't exist
    output_file.parent.mkdir(exist_ok=True)

    # add timestamp so each midi will have a unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_file.with_name(f"{output_file.stem}_{timestamp}{output_file.suffix}")

    # save the generated midi
    midi_stream.write("midi", fp=output_file)
    print(f"Generated MIDI saved to {output_file}")

    # log details about the generated midi
    log_generated_midi(output_file, strategy, generate_length, temperature, k, p, model_path)

def log_generated_midi(output_file, strategy, generate_length, temperature, k, p, model_path):
    """Log details about the generated MIDI to a CSV file."""
    import csv
    import os

    # Prepare log entry
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_file": str(output_file),
        "strategy": strategy,
        "generate_length": generate_length,
        "temperature": temperature,
        "k": k,
        "p": p,
        "model_path": model_path
    }

    # Check if log file exists
    file_exists = os.path.isfile(LOG_FILE)

    # Write to CSV
    with open(LOG_FILE, mode='a', newline='') as csvfile:
        fieldnames = ["timestamp", "output_file", "strategy", "generate_length", "temperature", "k", "p", "model_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file didn't exist before
        if not file_exists:
            writer.writeheader()

        writer.writerow(log_entry)
    print(f"Logged generated MIDI details to {LOG_FILE}")

    # log the evaluation of the midi
    from evaluate import evaluate_midi, log_evaluation
    results = evaluate_midi(output_file)
    if results:
        log_evaluation(results)
        print(f"Evaluated and logged MIDI evaluation results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI with trained LSTM")
    parser.add_argument("--strategy", type=str, default="greedy",
                        choices=["greedy", "random", "top_k", "top_p"],
                        help="Sampling strategy")
    parser.add_argument("--generate_length", type=int, default=100,
                        help="How many notes to generate")
    parser.add_argument("--seq_length", type=int, default=50,
                        help="Input sequence length (must match training)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--k", type=int, default=5,
                        help="Top-k value (for top_k sampling)")
    parser.add_argument("--p", type=float, default=0.9,
                        help="Top-p (nucleus) value (for top_p sampling)")
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH),
                        help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, default=str(VOCAB_PATH),
                        help="Path to vocab file")
    parser.add_argument("--sequences_path", type=str, default=str(SEQUENCES_PATH),
                        help="Path to tokenized sequences")
    parser.add_argument("--output_file", type=str, default=str(OUTPUT_FILE),
                        help="Output MIDI filename")

    args = parser.parse_args()
    generate(**vars(args))