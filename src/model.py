import torch
import torch.nn as nn

class MidiModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim=d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.transformer(h, h, tgt_mask=mask)
        return self.lm_head(h)