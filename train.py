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
    
    dataset = MidiDataset(shards)
    dataloader = DataLoader(dataset, batch_size=Training.BATCH_SIZE)

    model = MidiModel(VOCAB_SIZE, Model.D_MODEL, 
                      Model.N_LAYERS, Model.N_HEADS, 
                      Model.D_FF, Model.SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Training.LR, weight_decay=Training.WEIGHT_DECAY)

    Training.CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(Training.CHECKPOINT_PATH.glob("model_epoch*.pt"))
    if ckpts:
        latest_ckpt = ckpts[-1]
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"resuming from {latest_ckpt} at epoch {start_epoch}")
    else:
        start_epoch = 0
        print("starting new training")
    step = 0
    for epoch in range(Training.EPOCHS):
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
                print(f"epoch {epoch} step {step} loss {loss.item() * Training.ACCUMULATION_STEPS:.4f}")

            step += 1

        # save checkpoint after each epoch
        ckpt_path = Training.CHECKPOINT_PATH / f"model_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint {ckpt_path}")

if __name__ == "__main__":
    train()