#debug:
import torch
from src import convert_checkpoint, generate
from src.config import TOKENIZER
in_checkpoint = "./data/checkpoints/model_step90000.pt"
out_checkpoint = "./data/checkpoints/inference_model_step90000.pt"

convert_checkpoint.convert_checkpoint(in_checkpoint, out_checkpoint, half_precision=False)

model = generate.load_inference_model("./data/checkpoints/inference_model_step90000.pt")

start_tokens = [
    TOKENIZER.vocab["genre_rock"],
    TOKENIZER.vocab["emotion_angry"],
    TOKENIZER.vocab["prog_30"],   # distorted guitar
    TOKENIZER.vocab["prog_128"],  # drums
]

tokens = generate.generate(model,start_tokens)

generate.tokens_to_midi(tokens, "test.mid")