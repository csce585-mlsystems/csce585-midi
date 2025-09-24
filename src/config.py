from pathlib import Path
from multiprocessing import cpu_count
from miditok import REMI, TokenizerConfig
    
class MidiTokenization:
    MIDI_PATH = Path("./data/midi")
    OUT_PATH = Path("./data/tokens")
    SHARD_SIZE = 2000
    N_WORKERS = max(1, cpu_count() - 1)

class TokenLabels:
    UNKNOWN_GENRE = "genre_unknown"
    UNKNOWN_EMOTION = "emotion_unknown"
    EMOTIONS = [
        "emotion_exciting", 
        "emotion_warm", 
        "emotion_happy", 
        "emotion_romantic", 
        "emotion_funny", 
        "emotion_sad", 
        "emotion_angry", 
        "emotion_lazy", 
        "emotion_quiet", 
        "emotion_fear", 
        "emotion_magnificent", 
        UNKNOWN_EMOTION
    ]

    GENRES = [
    "genre_rock", 
    "genre_pop", 
    "genre_country", 
    "genre_jazz", 
    "genre_classical", 
    "genre_folk", 
    UNKNOWN_GENRE
    ]

    # General MIDI has a bank of 128 instruments with set IDs, get all for tokenization, store drums as 128
    INSTRUMENTS = [f"prog_{p}" for p in range(129)]

    SPECIAL_TOKENS = {
        "Genre": GENRES,
        "Emotion": EMOTIONS,
        "Instrument": INSTRUMENTS,
    }

class Model:
    SEQ_LEN = 1024
    D_MODEL = 512
    N_LAYERS = 8
    N_HEADS = 8
    D_FF = 2048

class Training:
    CHECKPOINT_PATH = Path("./data/checkpoints")
    TOTAL_STEPS = 100000
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2
    LR = 1e-4
    WEIGHT_DECAY = 0.01
    PRINT_EVERY = 50
    CHECKPOINT_EVERY = 10000

class Generation:
    TEMP = 1.0
    TOP_K = 50

def get_tokenizer():
    token_config = TokenizerConfig()
    token_config.special_tokens += TokenLabels.GENRES + TokenLabels.EMOTIONS + TokenLabels.INSTRUMENTS
    return REMI(token_config)

# Instantiate tokenizer globally (shared by train/eval scripts)
TOKENIZER = get_tokenizer()
VOCAB_SIZE = len(TOKENIZER.vocab)