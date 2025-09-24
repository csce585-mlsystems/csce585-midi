import torch
from config import TOKENIZER, VOCAB_SIZE, Model, Generation
from model import MidiModel

@torch.no_grad #more efficient memory usage for inference
def generate(model, start_tokens, max_len=Model.SEQ_LEN, temperature=Generation.TEMP, top_k=Generation.TOP_K):
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor(start_tokens, device=device).unsqueeze(0)  # [1, T]

    for _ in range(max_len - len(start_tokens)):
        logits = model(x)[:, -1, :]   # last step logits
        logits = logits / temperature

        # top-k filtering
        if top_k > 0:
            values, indices = torch.topk(logits, k=top_k)
            probs = torch.softmax(values, dim=-1)
            next_token = indices[0, torch.multinomial(probs, 1)]
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze()

        # append new token
        x = torch.cat([x, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        # optional EOS stop
        if next_token.item() == TOKENIZER.eos_token_id:
            break

    return x.squeeze().tolist()