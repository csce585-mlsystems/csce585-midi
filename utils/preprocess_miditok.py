import miditok
from symusic import Score
from pathlib import Path
import numpy as np
import json

INPUT_DIR = Path("data/nottingham-dataset-master/MIDI")
OUTPUT_DIR = Path("data/miditok")
OUTPUT_DIR.mkdir(exist_ok=True)

# Default training seq length (used in train.py slicing, not here)
SEQ_LENGTH = 50  

def preprocess_miditok():
    # Create tokenizer (REMI for now, but could use TSD or CPWord)
    tokenizer = miditok.REMI()

    sequences = []
    for midi_file in INPUT_DIR.glob("*.mid"):
        try:
            score = Score(midi_file)  # load MIDI
            token_seqs = tokenizer(score)

            # Handle multiple tracks (list of TokSequence)
            if isinstance(token_seqs, list):
                for ts in token_seqs:
                    sequences.append(ts.ids)  # save full track as ints
            else:
                sequences.append(token_seqs.ids)

        except Exception as e:
            print(f" Skipping {midi_file}: {e}")

    # Save sequences (variable length, not fixed)
    np.save(
        OUTPUT_DIR / "sequences.npy",
        np.array(sequences, dtype=object),
        allow_pickle=True,
    )
    print(f" Saved {len(sequences)} full token sequences to {OUTPUT_DIR}/sequences.npy")

    # Save tokenizer params (so decode uses the same config)
    tokenizer.save_params(OUTPUT_DIR / "tokenizer.json")
    print(f" Saved tokenizer params to {OUTPUT_DIR}/tokenizer.json")

    # Save config.json (meta info for train.py)
    config = {
        "seq_length": SEQ_LENGTH,
        "tokenizer": "REMI",
        "num_sequences": len(sequences),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved dataset config to {OUTPUT_DIR}/config.json")

if __name__ == "__main__":
    preprocess_miditok()
