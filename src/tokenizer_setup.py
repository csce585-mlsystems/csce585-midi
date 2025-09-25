import json
from miditok import REMI, TokenizerConfig
from pathlib import Path
from src.config import TokenLabels, MidiTokenization
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

def build_tokenizer() -> REMI:
    token_config = TokenizerConfig()
    token_config.special_tokens += TokenLabels.GENRES + TokenLabels.EMOTIONS

    token_config.use_programs = True
    token_config.use_chords = True
    token_config.use_rests = True
    token_config.use_tempos = True
    token_config.use_time_signatures = True
    token_config.use_sustain_pedals = False
    token_config.one_token_stream_for_programs = True

    tokenizer = REMI(token_config)
    return tokenizer

def save_tokenizer(remi_tokenizer, out_path = MidiTokenization.OUT_PATH) -> PreTrainedTokenizerFast:
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    vocab = {tok: i for i, tok in enumerate(remi_tokenizer.vocab)}
    next_id = len(vocab)

    for token in TokenLabels.GENRES + TokenLabels.EMOTIONS:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="UNK_None"))
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer_json = out_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        bos_token="BOS_None",
        eos_token="EOS_None",
        pad_token="PAD_None",
        unk_token="UNK_None",
    )

    new_tokenizer.save_pretrained(out_path)
    return new_tokenizer