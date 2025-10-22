import json
import pickle
import sys
import miditok
from symusic import Score
import torch
import argparse
import numpy as np
from pathlib import Path
from music21 import stream, note, chord
from utils.sampling import sample_next_note
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from models.generators.generator_factory import get_generator

# pick device
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

""" GETTING RID OF THIS. USING A SPECIFIC MODEL PATH INSTEAD.
 ORIGINALLY THIS WAS MADE TO FIND THE MOST RECENT MODEL BY TIMESTAMP."""
# if model_files:
#     # gets the timestamp from the filename
#     # assumes filenames are like lstm_MMDDHHMM.pth
#     def extract_timestamp(filepath):
#         """Extract timestamp from filename like lstm_MMDDHHMM.pth"""
#         try:
#             # Extract the timestamp part after 'lstm_' and before '.pth'
#             timestamp_str = filepath.stem.split('_')[1]
#             # Convert MMDDHHMM format to a comparable integer
#             return int(timestamp_str)
#         except (IndexError, ValueError):
#             # If parsing fails, fall back to file modification time
#             return int(filepath.stat().st_mtime)
    
#     # get the most recently created model based on filename timestamp
#     latest_model = max(model_files, key=extract_timestamp)
#     MODEL_PATH = latest_model
#     print(f"Found latest model checkpoint: {latest_model}")
# else:
#     # if none are found, use the default path
#     print("No model checkpoints found in models/ directory. Using default path.")
#     MODEL_PATH = "models/lstm_checkpoint.pth"


"""  UPDATE: USING ONLY NOTTINGHAM NOW AS OF 9/27/2025 """

def generate(
    strategy="greedy",
    generate_length=100,
    seq_length=50,
    temperature=1.0,
    k=5, # for top-k sampling
    p=0.9, # for nucleus sampling
    model_path=None,
    model_type="lstm", # specify which architecture was used
):
    
    # infer dataset (miditok or naive) from the model file path
    model_path = Path(model_path)
    if "miditok" in model_path.parts:
        dataset = "miditok"
    elif "naive" in model_path.parts:
        dataset = "naive"
    else:
        raise ValueError("Cannot infer dataset from model path. Please ensure the path contains either 'miditok' or 'naive'.")
    
    # make sure model path provided
    if model_path is None:
        raise ValueError("Please provide a valid model checkpoint path using --model_path")
    
    # you know which directories to look in by the model's filepath
    DATA_DIR = Path(f"data/{dataset}")
    LOG_FILE = Path(f"logs/generators/{dataset}/output_midis.csv")
    OUTPUT_DIR = Path(f"outputs/generator/{dataset}/midi")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # ensure output dir exists
    
    # load the vocab
    if dataset == "miditok":
        with open(DATA_DIR / "vocab.json", "r") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        int_to_note = {i: token for i, token in enumerate(vocab)}
    elif dataset == "naive":
        with open(DATA_DIR / "note_to_int.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        int_to_note = {i: n for n, i in vocab_data["note_to_int"].items()}
        vocab_size = len(int_to_note)
    else:
        raise ValueError("Dataset must be either 'miditok' or 'naive'")

    # load data and seed
    sequences = np.load(DATA_DIR / "sequences.npy", allow_pickle=True)
    seed = list(sequences[np.random.randint(len(sequences))][:seq_length])
    generated = seed.copy() # start with the seed

    # load the model using factory pattern
    # Initialize with arbitrary default parameters - they'll be overwritten by checkpoint
    model = get_generator(model_type, vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)) # load trained weights
    model.to(DEVICE) # move model to the appropriate device
    model.eval() # set to eval mode
    print(f"Loaded {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # prepare the input (seed) as a tensor
    input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # generate tokens (notes/chords etc.)
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

    # output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"generated_{timestamp}.mid"

    # decode the generated sequence back to notes/chords
    if dataset == "miditok":
        # detokenize using miditok
        tokenizer = miditok.REMI()

        # wrap into TokSequence
        tokseq = miditok.TokSequence(ids=[int(t) for t in generated])

        # decode back to a symusic score
        score = tokenizer.decode([tokseq])

        # export to midi
        score.dump_midi(str(output_file))
    else:
        # naive decoding
        midi_stream = stream.Stream()
        for idx in generated:
            symbol = int_to_note[idx]
            if '.' in symbol or symbol.isdigit(): # chord
                notes_in_chord = [note.Note(int(n)) for n in symbol.split('.')]
                for n in notes_in_chord:
                    n.quarterLength = 0.5
                midi_stream.append(chord.Chord(notes_in_chord))
            else: # single note
                n = note.Note(symbol)
                n.quarterLength = 0.5
                midi_stream.append(n)

        # write the midi file
        midi_stream.write("midi", fp=output_file)

    print(f"Generated midi saved to {output_file}")

    # log details about the generated midi
    log_generated_midi(output_file, strategy, generate_length, temperature, k, p, model_path, LOG_FILE)

def log_generated_midi(output_file, strategy, generate_length, temperature, k, p, model_path, log_file):
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
    file_exists = log_file.exists()
    log_file.parent.mkdir(parents=True, exist_ok=True) # ensure log dir exists

    # evaluation file
    eval_file = log_file.parent / "evaluation_log.csv"
    eval_file.parent.mkdir(parents=True, exist_ok=True) # ensure eval dir exists

    # Write to CSV
    with open(log_file, mode='a', newline='') as csvfile:
        fieldnames = ["timestamp", "output_file", "strategy", "generate_length", "temperature", "k", "p", "model_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file didn't exist before
        if not file_exists:
            writer.writeheader()

        writer.writerow(log_entry)
    print(f"Logged generated MIDI details to {log_file}")

    # log the evaluation of the midi
    from evaluate import evaluate_midi, log_evaluation
    results = evaluate_midi(output_file, log_file) # had to pass log_file as well as the path to midi file
    if results:
        log_evaluation(results, eval_file)
        print(f"Evaluated and logged MIDI evaluation results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI with trained generator model")
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "gru", "transformer"],
                        help="Type of generator architecture")
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
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")

    args = parser.parse_args()
    generate(**vars(args))