from pathlib import Path
from multiprocessing import cpu_count
from miditok import REMI, TokenizerConfig
    
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

class Model:
    SEQ_LEN = 512
    D_MODEL = 384
    N_LAYERS = 6
    N_HEADS = 6
    D_FF = 1536

class Training:
    CHECKPOINT_PATH = Path("./data/checkpoints")
    EPOCHS = 5
    BATCH_SIZE = 1
    ACCUMULATION_STEPS = 16
    LR = 2e-4
    WEIGHT_DECAY = 0.01
    PRINT_EVERY = 100

class Generation:
    TEMP = 1.0
    TOP_K = 50

def get_tokenizer():
    token_config = TokenizerConfig()
    token_config.additional_params = TokenLabels.SPECIAL_TOKENS
    return REMI(token_config)

# Instantiate tokenizer globally (shared by train/eval scripts)
TOKENIZER = get_tokenizer()
VOCAB_SIZE = len(TOKENIZER.vocab)