import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from src.config import TokenLabels, MidiTokenization
from src.tokenizer_setup import build_tokenizer, save_tokenizer
from transformers import PreTrainedTokenizerFast
from miditok import TokSequence

MIDITOK = build_tokenizer()
TOKENIZER = save_tokenizer(MIDITOK, out_path=MidiTokenization.OUT_PATH)

def parse_conditioning_from_filename(file_path: Path):
    name = file_path.stem
    parts = name.split("_")
    if len(parts) >= 3:
        emotion, genre = parts[1], parts[2]
        emotion_token = f"emotion_{emotion}" if f"emotion_{emotion}" in TokenLabels.EMOTIONS else TokenLabels.UNKNOWN_EMOTION
        genre_token = f"genre_{genre}" if f"genre_{genre}" in TokenLabels.GENRES else TokenLabels.UNKNOWN_GENRE
        return genre_token, emotion_token
    return TokenLabels.UNKNOWN_GENRE, TokenLabels.UNKNOWN_EMOTION

def process_file(file_path: Path):
    try:
        sequence = MIDITOK.encode(file_path)
        g, e = parse_conditioning_from_filename(file_path)
        conditioned_tokens = [g, e] + sequence.tokens
        ids = TOKENIZER.convert_tokens_to_ids(conditioned_tokens)
        return ids
    except Exception as e:
        return None
    
def preprocess_dataset():
    data_path=MidiTokenization.MIDI_PATH
    midi_files = list(data_path.rglob("*.mid")) + list(data_path.rglob("*.midi"))

    out_path = Path(MidiTokenization.OUT_PATH)
    out_path.mkdir(parents=True, exist_ok=True)

    shard_index, buffer = 0, []

    with Pool(processes=MidiTokenization.N_WORKERS) as pool:
        process_function = partial(process_file)
        for tokens in tqdm(pool.imap_unordered(process_function, midi_files), total=len(midi_files)):
            if tokens is None:
                continue
            buffer.append(tokens)

            if len(buffer) >= MidiTokenization.SHARD_SIZE:
                shard_file = out_path / f"shard_{shard_index:03d}.pt"
                torch.save(buffer, shard_file)
                print(f"saved {len(buffer)} items to {shard_file}")
                buffer.clear()
                shard_index += 1
    if buffer:
        shard_file = out_path / f"shard_{shard_index:03d}.pt"
        torch.save(buffer, shard_file)
        print(f"saved {len(buffer)} items to {shard_file}")