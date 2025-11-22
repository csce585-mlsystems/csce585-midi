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
from utils.seed_selection import get_seed_by_filename
from datetime import datetime

from utils.seed_selection import find_seed_by_characteristics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from models.generators.generator_factory import get_generator
from models.discriminators.discriminator_factory import get_discriminator

# pick device
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def load_discriminator(discriminator_path, model_type, pitch_dim=52, context_measures=4):
    """Load a trained discriminator model from checkpoint."""

    # load the model using factory pattern
    discriminator = get_discriminator(model_type, pitch_dim=pitch_dim, context_measures=context_measures)
    # load the trained weights
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=DEVICE))
    # move to device
    discriminator.to(DEVICE)
    # set to eval mode (this means layers like dropout, batchnorm behave appropriately)
    discriminator.eval()
    print(f"Loaded {model_type} discriminator for guided generation")
    return discriminator

def apply_discriminator_guidance(logits, discriminator, context_measures, generated_so_far, int_to_note, guidance_strength=0.5, dataset="naive"):
    """
    Apply discriminator guidance to adjust generation probabilities.
    
    Args:
        logits: Tensor of shape (1, vocab_size) - model's output logits for next token
        discriminator: the loaded discriminator model
        context_measures: number of measures used as context for the discriminator
        generated_so_far: list of generated token integers so far
        int_to_note: mapping from integer to note/chord string
        guidance_strength: float, how strongly to apply the discriminator's feedback
    Returns:
        adjusted_logits: Tensor of shape (1, vocab_size) - adjusted logits
    """

    if discriminator is None: # make sure discriminator exists
        return logits
    
    # different required context lengths based on dataset being used
    if dataset == "miditok":
        # miditok uses multiple tokens per note, so need more tokens for context
        tokens_per_measure = 100 # conservative estimate
        min_tokens = tokens_per_measure

    else: # naive
        tokens_per_measure = 16
        min_tokens = 16

    # check if we have enough context
    if len(generated_so_far) < min_tokens:
        return logits
    
    # extract recent tokens to form context (last n measures)
    context_length = context_measures * tokens_per_measure
    recent_tokens = generated_so_far[-context_length:] if len(generated_so_far) >= context_length else generated_so_far

    # convert tokens to pitch representation for discriminator
    pitches = []

    MIDDLE_C = 60  # MIDI number for Middle C (defalut)

    if dataset == "miditok":
        # for miditok, extract only pitch tokens
        for token_idx in recent_tokens:
            # get the token string
            token_str = int_to_note.get(token_idx, "")
            if token_str.startswith("Pitch_"):
                try:
                    pitch = int(token_str.split("_")[1])
                    pitches.append(pitch)
                except (ValueError, IndexError):
                    pass # skip malformed tokens
    else: # naive
        # for naive, each token is a note or chord
        for note_idx in recent_tokens:
            note_symbol = int_to_note.get(note_idx, f"{MIDDLE_C}") # default to middle C

            if '.' in note_symbol: # chord
                # take first note of chord
                pitches.append(int(note_symbol.split('.')[0]))
            elif note_symbol.isdigit(): # single note as integer
                pitches.append(int(note_symbol))
            else:
                # convert note name to midi pitch
                try:
                    # use music21 to convert note name to midi number
                    n = note.Note(note_symbol)
                    pitches.append(n.pitch.midi)
                except:
                    pitches.append(MIDDLE_C) # default to middle C on error

    # need at least one measure worth of pitches
    notes_per_measure = 16
    min_pitches = notes_per_measure

    if len(pitches) < min_pitches:
        return logits # not enough pitches for context
    
    # pad/truncate to expected length (context_measures * notes_per_measure)
    expected_pitch_length = context_measures * notes_per_measure
    if len(pitches) < expected_pitch_length:
        # pad with middle C
        # creates a new list with padding at the start and existing pitches at the end
        pitches = [MIDDLE_C] * (expected_pitch_length - len(pitches)) + pitches
    else:
        # take most recent
        pitches = pitches[-expected_pitch_length:]

    # create one-hot encoding for pitches (52 pitch classes for multiple octaves)
    pitch_dim = 52
    # makes tensor of shape (1, context_measures, pitch_dim) filled with zeros
    context_tensor = torch.zeros(1, context_measures, pitch_dim).to(DEVICE)

    # fill context tensor measure by measure
    for m in range(context_measures):
        # get index of note for beginning and end
        measure_start = m * notes_per_measure
        measure_end = (m+1) * notes_per_measure
        # grab those notes from pitches
        measure_pitches = pitches[measure_start:measure_end]

        for p in measure_pitches:
            # map midi pitch to index (C3=48 and D3=50)
            pitch_idx = p - 48 # start from C3
            if 0 <= pitch_idx < pitch_dim:
                context_tensor[0, m, pitch_idx] = 1.0 # set one-hot

    # get discrim prediction
    with torch.no_grad():
        chord_probs = discriminator(context_tensor)  # logits for each pitch class

    # apply guidance by boosting probs of notes in predicted chord (chord tone soloing)
    """
    I'm imagining this as a musician training part of their brain on chord tones and arpeggios
    This basically gives the model info about which notes are chord tones and fit harmonically.
    I still want the generator to have freedom to pick non-chord tones, so I won't force it too much.
    """
    adjusted_logits = logits.clone()

    # get top predicted chords
    top_chords = torch.topk(chord_probs, k=3, dim=-1).indices[0]  # get top 3 chord tone indices

    # boost tokens that correspond to these chord tones
    for chord_idx in top_chords:
        # map chord index to pitch classes
        # assuming chord_idx represents root note
        # plus chord type (major, minor, etc.) encoded in higher bits
        root = chord_idx % 12  # get root pitch class (0-11)

        # for simplicity, major triad
        chord_pitches = [
            root,
            (root + 4) % 12, # major 3rd
            (root + 7) % 12  # perfect 5th
        ]

        # boost logit tokens for representing these pitch classes
        for token_idx, token_str in int_to_note.items():
            if dataset == "miditok":
                # only boost pitch tokens
                if token_str.startswith("Pitch_"):
                    try:
                        pitch = int(token_str.split("_")[1])
                        pitch_class = pitch % 12
                        if pitch_class in chord_pitches:
                            adjusted_logits[0, token_idx] += guidance_strength
                    except (ValueError, IndexError):
                        pass # skip malformed tokens
            else: # naive
                # check if note/chord contains any of the predicted pitches
                if '.' in token_str: # chord
                    # get all pitch classes in chord
                    note_pitches = [int(n) % 12 for n in token_str.split('.')]
                elif token_str.isdigit(): # if it's a digit, it's a single note
                    # get the pitch class
                    note_pitches = [int(token_str) % 12]
                else: # single note
                    try:
                        # convert note name to pitch class
                        n = note.Note(token_str)
                        note_pitches = [n.pitch.midi % 12]
                    except:
                        continue

                # if note contains any pitch from predicted chord, boost it
                if any(p in chord_pitches for p in note_pitches):
                    adjusted_logits[0, token_idx] += guidance_strength

    return adjusted_logits


def generate(
    strategy="greedy",
    generate_length=100,
    seq_length=50,
    temperature=1.0,
    k=5, # for top-k sampling
    p=0.9, # for nucleus sampling
    model_path=None,
    model_type="lstm", # specify which architecture was used
    discriminator_path=None,
    discriminator_type=None,
    guidance_strength=0.5,
    context_measures=4,
    guidance_frequency=1,
    seed_style="random",
    pitch_preference="medium",
    complexity="medium",
    seed_length="medium",
    # adding these too allow different model sizes
    hidden_size=None,
    num_layers=None,
    embed_size=None,
    d_model=None,
    nhead=None,
    dim_feedforward=None,
    seed_file=None,
):
    
    # infer dataset (miditok or naive) from the model file path
    model_path = Path(model_path)
    if "miditok_augmented" in model_path.parts:
        dataset = "miditok_augmented"
    elif "miditok" in model_path.parts:
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
    LOG_FILE = Path(f"logs/generators/{dataset}/midi/output_midis.csv")
    OUTPUT_DIR = Path(f"outputs/generators/{dataset}/midi")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # ensure output dir exists
    
    # load the vocab
    if dataset == "miditok" or dataset == "miditok_augmented":
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

    # if user wants a specific file to be used as the seed
    if seed_file is not None:
        # Extract just the filename if a full path was provided
        seed_filename = Path(seed_file).name
        print(f"Using seed file: {seed_filename}")
        seed = get_seed_by_filename(filename=seed_filename, dataset=dataset, length=seq_length)
        if seed is None:
            print("\nError: Could not load seed from file.")
            print(f"Note: Use just the filename (e.g., 'reelsa-c33.mid'), not the full path.")
            sys.exit(1)
    # get a random seed
    elif seed_style == "random":
        idx = np.random.randint(0, len(sequences))
        seed = sequences[idx]

    # smart seed selection
    else:
        seed = find_seed_by_characteristics(
            sequences=sequences, int_to_note=int_to_note,
            pitch_preference=pitch_preference,
            complexity=complexity,
            length=seed_length,
            dataset=dataset
        )[:seq_length]

    # Handle nested list structure (tracks) from miditok for ANY seed selection method
    if isinstance(seed, (list, np.ndarray)) and len(seed) > 0:
        # Check if first element is a list/array (indicating tracks)
        if isinstance(seed[0], (list, np.ndarray)):
            # flatten tracks into single sequence
            flat_seed = []
            for track in seed:
                if isinstance(track, (list, np.ndarray)):
                    flat_seed.extend(track)
                else:
                    flat_seed.append(track)
            seed = flat_seed

    # ensure seed is a list and slice to correct length
    if isinstance(seed, np.ndarray):
        seed = seed.tolist()

    if len(seed) > seq_length:
        seed = seed[:seq_length]

    generated = seed.copy() # start with the seed

    # load the model using factory pattern
    # Load checkpoint first to inspect architecture
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # infer architecture from checkpoint key
    inferred_num_layers = 2  # default that was used in training
    inferred_hidden_size = 256 # default

    if model_type == "transformer":
        # Count transformer decoder layers in checkpoint
        layer_keys = [k for k in checkpoint.keys() if k.startswith("transformer_decoder.layers.")]
        if layer_keys:
            layer_nums = [int(k.split('.')[2]) for k in layer_keys if k.split('.')[2].isdigit()]
            if layer_nums:
                num_layers = max(layer_nums) + 1  # +1 because layers are 0-indexed

        # infer d_model from embedding weight shape
        if "embedding.weight" in checkpoint:
            inferred_hidden_size = checkpoint["embedding.weight"].shape[1]

    else:  # LSTM or GRU
        # count rnn layers by looking for weight keys
        layer_keys = []
        if model_type == "lstm":
            layer_keys = [k for k in checkpoint.keys() if k.startswith("lstm.weight_ih_l")]
        elif model_type == "gru":
            layer_keys = [k for k in checkpoint.keys() if k.startswith("gru.weight_ih_l")]

        if layer_keys:
            layer_nums = [int(k.split('_l')[1]) for k in layer_keys]
            inferred_num_layers = max(layer_nums) + 1 # +1 bc zero indexed

        # infer hidden_size from first layer weight shape
        weight_key = f"{model_type}.weight_hh_l0"
        if weight_key in checkpoint:
            # weight_hh shape is (3*hidden_size, hidden_size) for gru
            # or (4*hidden_size, hidden_size) for lstm
            weight_shape = checkpoint[weight_key].shape
            multiplier = 3 if model_type == "gru" else 4
            inferred_hidden_size = weight_shape[1] # second dim is hidden_size

    # use command-line args if provided otherwise use inferred
    final_num_layers = num_layers if num_layers is not None else inferred_num_layers
    final_hidden_size = hidden_size if hidden_size is not None else inferred_hidden_size
    final_embed_size = embed_size if embed_size is not None else 128

    print(f"Inferred architecture: {final_num_layers} layers, hidden_size={final_hidden_size}")
    
    # Create model with correct architecture
    if model_type == "transformer":
        final_d_model = d_model if d_model is not None else final_hidden_size
        final_nhead = nhead if nhead is not None else 8
        final_dim_feedforward = dim_feedforward if dim_feedforward is not None else 1024
        
        model = get_generator(
            model_type, 
            vocab_size, 
            num_layers=final_num_layers,
            d_model=final_d_model,
            nhead=final_nhead,
            dim_feedforward=final_dim_feedforward,
            dropout=0.1
        )
    else:  # LSTM/GRU
        model = get_generator(
            model_type,
            vocab_size,
            embed_size=final_embed_size,
            hidden_size=final_hidden_size,
            num_layers=final_num_layers,
            dropout=0.2
        )
    
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # load discriminator if provided
    discriminator = None
    if discriminator_path and discriminator_type:
        discriminator = load_discriminator(discriminator_path, discriminator_type, context_measures=context_measures)
        print(f"Using discriminator guidance with strength {guidance_strength} and context of {context_measures} measures")

    # prepare the input (seed) as a tensor
    input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # generate tokens (notes/chords etc.)
    for i in range(generate_length):
        # don't need gradients for inference (save time and space and just unecessary overall)
        with torch.no_grad():
            output, _ = model(input_seq) # get model's output
            logits = output[:, -1, :] # get the logits for the last time step (the next note prediction)

            # apply discriminator guidance if available
            if discriminator is not None and i % guidance_frequency == 0:
                logits = apply_discriminator_guidance(
                    logits, discriminator, context_measures,
                    generated, int_to_note, guidance_strength,
                    dataset=dataset # dataset type
                )

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
    if dataset == "miditok" or dataset == "miditok_augmented":
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
    parser.add_argument("--discriminator_path", type=str, default=None,
                        help="Path to the trained discriminator checkpoint (for guidance)")
    parser.add_argument("--discriminator_type", type=str, default=None,
                        choices=["mlp", "lstm", "transformer"])
    parser.add_argument("--guidance_strength", type=float, default=0.5,
                        help="Strength of discriminator guidance during generation")
    parser.add_argument("--context_measures", type=int, default=4,
                        help="Number of measures used as context for the discriminator")
    parser.add_argument("--seed_style", type=str, default="random",
                        choices=["random", "smart"], help="Pick whether to use random seed or select one" \
                        " based on given constraints")
    parser.add_argument("--pitch_preference", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Pick your desired pitch range")
    parser.add_argument("--complexity", type=str, default="medium",
                        choices=["simple", "medium", "complex"],
                        help="Pick number of unique pitches\nsimple=3-5 pitches\nmedium=6-8 pitches\ncomplex=9+ pitches)")
    parser.add_argument("--seed_length", type=str, default="medium",
                        choices=["short", "medium", "long"],
                        help="Length of seed\nshort: < 50 notes\n medium: 50-100 notes\nlong: 100+ notes")
    parser.add_argument("--seed_file", type=str, default=None,
                        help="Filename of the MIDI file to use as seed (e.g., 'reelsa-c33.mid'). File must exist in the preprocessed dataset.")
    args = parser.parse_args()
    generate(**vars(args))