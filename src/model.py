import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.config import TOKENIZER, TokenLabels, Training
from torch.optim.lr_scheduler import LambdaLR


def get_ignored_ids():
    ignored_ids = {0}  # pad
    for group in TokenLabels.SPECIAL_TOKENS.values():
        for tok in group:
            if tok in TOKENIZER.vocab:
                ignored_ids.add(TOKENIZER.vocab[tok])
    return ignored_ids

def causal_mask(seq_len, device):
    # shape [T, T] with -inf above diagonal
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

class CachingDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, past_kv=None, attn_mask=None):
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, x], dim=1)
            v = torch.cat([past_v, x], dim=1)
        else:
            k, v = x, x

        attn_out, _ = self.self_attn(
            x, k, v,
            attn_mask=attn_mask,
            need_weights=False
        )
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x, (k, v)


class MidiModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.layers = nn.ModuleList([
            CachingDecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.ignored_ids = get_ignored_ids()

    def forward(self, x, past_kvs=None, use_cache=False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.hparams.d_model) + self.pos_emb(pos)

        new_kvs = []
        if use_cache:  # inference
            for i, layer in enumerate(self.layers):
                past_kv = None if past_kvs is None else past_kvs[i]
                h, kv = layer(h, past_kv, attn_mask=None)
                new_kvs.append(kv)
            h = self.ln(h)
            logits = self.lm_head(h)
            return logits, new_kvs
        else:  # training
            attn_mask = causal_mask(T, x.device)
            for layer in self.layers:
                h, _ = layer(h, attn_mask=attn_mask)
            h = self.ln(h)
            logits = self.lm_head(h)
            return logits

    def shared_step(self, batch, stage):
        logits = self(batch[:, :-1], use_cache=False)
        targets = batch[:, 1:].reshape(-1)

        mask = torch.isin(
            targets, torch.tensor(list(self.ignored_ids), device=targets.device)
        )
        targets = targets.clone()
        targets[mask] = -100

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets,
            ignore_index=-100,
            reduction="mean",
            label_smoothing=0.1,
        )

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        warmup = Training.WARMUP_STEPS
        total_steps = Training.TOTAL_STEPS

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            progress = min(progress, 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx: int = 0,  # default if not passed
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm"
    ):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )
