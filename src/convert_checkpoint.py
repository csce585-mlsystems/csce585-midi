import torch
from pathlib import Path
from src.model import MidiModel
from src.config import Model, VOCAB_SIZE, Training

# converts a training checkpoint containing optimizer state into weights-only for inference
def convert_checkpoint(in_path: str, out_path: str, half_precision = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {in_path}")
    ckpt = torch.load(in_path, map_location=device)

    model = MidiModel(
        VOCAB_SIZE, Model.D_MODEL,
        Model.N_LAYERS, Model.N_HEADS,
        Model.D_FF, Model.SEQ_LEN,
        Training.LR, Training.WEIGHT_DECAY
    )
    model.load_state_dict(ckpt["model_state_dict"])

    state_dict = model.state_dict()
    if half_precision:
        print("converting weights.")
        state_dict = {k: v.half() for k, v in state_dict.items()}
    
    torch.save(state_dict, out_path)
    print(f"saved inference model to {out_path}")


