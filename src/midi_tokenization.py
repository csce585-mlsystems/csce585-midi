import pretty_midi
import torch
from miditok import REMI, TokenizerConfig
from pathlib import Path
from itertools import chain

MIDI_PATH = Path("./data/midi")
OUT_PATH = Path("./data/tokens")
MAX_LEN = 1024


UNKNOWN_LABEL = "unknown"
EMOTIONS = ["exciting", "warm", "happy", "romantic", "funny", "sad", "angry", "lazy", "quiet", "fear", "magnificent", UNKNOWN_LABEL]
GENRES = ["rock", "pop", "country", "jazz", "classical", "folk", UNKNOWN_LABEL]

# General MIDI has a bank of 128 instruments with set IDs, get all for tokenization
DRUM_TOKEN = "drums"
INSTRUMENTS = [
    pretty_midi.program_to_instrument_name(p).lower().replace(' ', '_')
    for p in range(128)
]
# percussion is a special case in MIDI
INSTRUMENTS.append(DRUM_TOKEN)

SPECIAL_TOKENS = {
    "Genre": GENRES,
    "Emotion": EMOTIONS,
    "Instrument": INSTRUMENTS,
}

####
def instruments_to_tokens(programs):
    instrument_tokens = []
    for i in programs:
        if i == 128:
            instrument_tokens.append(DRUM_TOKEN)
        else:
            token = pretty_midi.program_to_instrument_name(i)
            if token in SPECIAL_TOKENS["Instrument"]:
                instrument_tokens.append(token)
    return instrument_tokens

def add_conditioning(tokens, genre, emotion, programs, tokenizer):
    cond = [genre, emotion]
    cond += instruments_to_tokens(programs)

    cond_ids = [tokenizer.vocab[e] for e in cond if e in tokenizer.vocab]
    return cond_ids + tokens
    

def midi_to_tokens(file, tokenizer, genre, emotion):
    token_sequences = tokenizer(file)
    #flatten each track into one sequence
    tokens = list(chain.from_iterable(seq.ids for seq in token_sequences))

    pm = pretty_midi.PrettyMIDI(str(file))
    programs = [inst.program if not inst.is_drum else 128 for inst in pm.instruments]
    tokens = add_conditioning(tokens, genre, emotion, programs, tokenizer)    
    return tokens, programs

def preprocess_data(tokenizer, data_path=MIDI_PATH, out_path=OUT_PATH):
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

    tokenizer = REMI(token_config)

    print("Beginning Preprocessing")
    preprocess_data(tokenizer)
    print("Done with Preprocessing")