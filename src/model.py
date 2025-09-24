import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.config import TOKENIZER, TokenLabels, Training
def get_ignored_ids():
    ignored_ids = {0}  # pad
    for group in TokenLabels.SPECIAL_TOKENS.values():
        for tok in group:
            if tok in TOKENIZER.vocab:
                ignored_ids.add(TOKENIZER.vocab[tok])
    return ignored_ids
    
class MidiModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.ignored_ids = get_ignored_ids()
    
    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.transformer(h, h, tgt_mask=mask)
        return self.lm_head(h)
    
    def shared_step(self, batch, stage):
        logits = self(batch[:, :-1])
        targets = batch[:, 1:].reshape(-1)

        mask = torch.isin(targets, torch.tensor(list(self.ignored_ids), device=targets.device))
        targets = targets.clone()
        targets[mask] = -100

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets,
            ignore_index=-100,
            reduction="mean"
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch):
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch):
        return self.shared_step(batch, "val")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        def lr_lambda(step):
            warmup = Training.WARMUP_STEPS
            total_steps = Training.TOTAL_STEPS
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]