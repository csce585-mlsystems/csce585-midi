import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from src.config import Model, Training, MidiTokenization, VOCAB_SIZE
from src.dataset import MidiDataset
from src.model import MidiModel

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shards = list(MidiTokenization.OUT_PATH.glob("*.pt"))
    shards.sort()

    dataset = MidiDataset(shards, seq_len=Model.SEQ_LEN, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=Training.BATCH_SIZE)

    model = MidiModel(VOCAB_SIZE, Model.D_MODEL, 
                      Model.N_LAYERS, Model.N_HEADS, 
                      Model.D_FF, Model.SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Training.LR, weight_decay=Training.WEIGHT_DECAY)

    Training.CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(Training.CHECKPOINT_PATH.glob("model_step*.pt"))
    if ckpts:
        latest_ckpt = ckpts[-1]
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"] + 1
        print(f"resuming from {latest_ckpt} at step {step}")
    else:
        step = 0
        print("starting new training")
    for batch in dataloader:
        x = batch.to(device)

        # forward with mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                x[:, 1:].reshape(-1),
                ignore_index=0
            ) / Training.ACCUMULATION_STEPS

        loss.backward()

        if (step + 1) % Training.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % Training.PRINT_EVERY == 0:
            print(f"step {step} loss {loss.item() * Training.ACCUMULATION_STEPS:.4f}")

        if step % Training.CHECKPOINT_EVERY == 0 and step > 0 :
            ckpt_path = Training.CHECKPOINT_PATH / f"model_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint {ckpt_path}")
        step += 1

if __name__ == "__main__":
    train()