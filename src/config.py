from pathlib import Path
from multiprocessing import cpu_count
    
class MidiTokenization:
    MIDI_PATH = Path("./data/midi")
    OUT_PATH = Path("./data/tokens")
    SHARD_SIZE = 2000
    N_WORKERS = max(1, cpu_count() - 1)

class TokenLabels:
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