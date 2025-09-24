#debug:
import torch, random
from datetime import datetime
from pathlib import Path
from miditok import TokSequence
from src.config import TOKENIZER, MidiTokenization
from src import convert_checkpoint, generate
from src.config import TOKENIZER
in_checkpoint = "./data/checkpoints/model_step5000.pt"
out_checkpoint = "./data/checkpoints/inference_model_step5000.pt"

convert_checkpoint.convert_checkpoint(in_checkpoint, out_checkpoint, half_precision=False)

model = generate.load_inference_model("./data/checkpoints/inference_model_step5000.pt")

start_tokens = [
    TOKENIZER.vocab["genre_unknown"],
    TOKENIZER.vocab["genre_unknown"],
    TOKENIZER.vocab["prog_0"],   # distorted guitar
]

tokens = generate.generate(model,start_tokens)
sequence = TokSequence(ids=tokens)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
generate.sequence_to_midi(sequence, f"./data/out/{timestamp}.mid")