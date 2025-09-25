import torch
from src.config import TOKENIZER, VOCAB_SIZE, Model, Generation, Training
from src.model import MidiModel

def load_inference_model(checkpoint_path: str): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MidiModel(
        VOCAB_SIZE, Model.D_MODEL,
        Model.N_LAYERS, Model.N_HEADS,
        Model.D_FF, Model.SEQ_LEN,
        Training.LR, Training.WEIGHT_DECAY
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded model from {checkpoint_path}")
    return model

@torch.inference_mode
def generate(model, start_tokens, max_len=Model.SEQ_LEN,
    temperature=Generation.TEMP, top_k=Generation.TOP_K):
    device = next(model.parameters()).device
    x = torch.tensor(start_tokens, device=device).unsqueeze(0)  # [1, T]
    past_kvs = None

    for _ in range(max_len - len(start_tokens)):
        logits, past_kvs = model(x[:, -1:], past_kvs, use_cache=True)
        logits = logits[:, -1, :] / temperature

        values, indices = torch.topk(logits, k=top_k)
        probs = torch.softmax(values, dim=-1)

        next_token = indices[0, torch.multinomial(probs, 1)]
        next_token = next_token.view(1, 1)

        x = torch.cat([x, next_token], dim=1)

    return x.squeeze(0).tolist()

def sequence_to_midi(sequence, out_path):
    sequence = TOKENIZER.decode([sequence])
    sequence.dump_midi(out_path)
    print(f"saved MIDI to {out_path}")