import torch
import symusic

from miditok import REMI, TokenizerConfig
from pathlib import Path
from itertools import chain

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

MIDI_PATH = Path("./data/midi")
OUT_PATH = Path("./data/tokens")
MAX_LEN = 1024
SHARD_SIZE = 2000
N_WORKERS = max(1, cpu_count() - 1)

UNKNOWN_LABEL = "unknown"
EMOTIONS = ["exciting", "warm", "happy", "romantic", "funny", "sad", "angry", "lazy", "quiet", "fear", "magnificent", UNKNOWN_LABEL]
GENRES = ["rock", "pop", "country", "jazz", "classical", "folk", UNKNOWN_LABEL]

# General MIDI has a bank of 128 instruments with set IDs, get all for tokenization, store drums as 128
INSTRUMENTS = [f"prog_{p}" for p in range(129)]

SPECIAL_TOKENS = {
    "Genre": GENRES,
    "Emotion": EMOTIONS,
    "Instrument": INSTRUMENTS,
}

####
def instruments_to_tokens(programs):
    return [f"prog_{p}" for p in programs]

def add_conditioning(tokens, genre, emotion, programs, tokenizer):
    cond = [genre, emotion]
    cond += instruments_to_tokens(programs)

    cond_ids = [tokenizer.vocab[e] for e in cond if e in tokenizer.vocab]
    return cond_ids + tokens
    
def midi_to_tokens(file, tokenizer, genre, emotion):

    midi = symusic.Score(str(file))
    token_sequences = tokenizer(midi)
    #flatten each track into one sequence
    tokens = list(chain.from_iterable(seq.ids for seq in token_sequences))

    programs = [inst.program if not inst.is_drum else 128 for inst in midi.tracks]    
    tokens = add_conditioning(tokens, genre, emotion, programs, tokenizer)    
    return tokens, programs


_worker_tokenizer = None

def init_worker(tok_cfg):
    global _worker_tokenizer
    _worker_tokenizer = REMI(tok_cfg)

def process_file(file):
    global _worker_tokenizer
    try:
        genre, emotion = parse_filename(file)
        tokens, programs = midi_to_tokens(file, _worker_tokenizer, genre, emotion)
        return (tokens, genre, emotion, programs, None)
    except Exception as e:
        return None, None, None, None, f"{file}: {e}"

def preprocess_data(tokenizer, data_path=MIDI_PATH, out_path=OUT_PATH, shard_size=SHARD_SIZE, n_workers = N_WORKERS):
    out_path.mkdir(parents=True, exist_ok=True)
    
    files = list(data_path.rglob("*.mid")) + list(data_path.rglob("*.midi"))
    files = sorted(files)
    shard_items = []
    shard_index = 0
    errors = []

    print(f"Found {len(files)} MIDIS, using {n_workers} workers to parse")
    with Pool(processes=n_workers, initializer=init_worker, initargs=(tokenizer,)) as pool:
        for i, result in enumerate(
                tqdm(pool.imap(process_file, files), total=len(files)), start=1
        ):
            tokens, genre, emotion, programs, error = result
            if error:
                errors.append(error)
                continue
            if tokens is None:
                continue

            shard_items.append((tokens, genre, emotion, programs))

            shard_items.append(result)
            if (i + 1) % shard_size == 0 or (i + 1) == len(files):
                shard_file = out_path / f"shard_{shard_index:03d}.pt"
                torch.save(shard_items, shard_file)
                print(f"saved {len(shard_items)} items to {shard_file}")
                shard_items = []
                shard_index += 1

def preprocess_data_into_individuals(tokenizer, data_path=MIDI_PATH, out_path=OUT_PATH):
    out_path.mkdir(parents=True, exist_ok=True)
    
    for f in data_path.glob("*.midi"):
        genre, emotion = parse_filename(f)
        
        tokens, programs = midi_to_tokens(f, tokenizer, genre, emotion)
        save_data = (tokens, genre, emotion, programs)
        torch.save(save_data, out_path / (f.stem + ".pt"))
        
            
def parse_filename(f: Path):
    name = f.stem
    if name.startswith("XMIDI"):
        s = name.split("_")

        emotion = s[1]
        genre = s[2]

        if genre not in GENRES:
            genre = UNKNOWN_LABEL
        if emotion not in EMOTIONS:
            emotion = UNKNOWN_LABEL
        return genre, emotion
    return UNKNOWN_LABEL, UNKNOWN_LABEL

if __name__ == "__main__":
    token_config = TokenizerConfig()
    token_config.additional_params = SPECIAL_TOKENS

    # tokenizer = REMI(token_config)

    print("Beginning Preprocessing")
    preprocess_data(token_config)
    print("Done with Preprocessing")