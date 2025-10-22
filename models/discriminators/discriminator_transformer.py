import torch.nn as nn
import torch

class SimpleTransformerDiscriminator(nn.Module):
    def __init__(self, pitch_dim, context_measures=4, d_model=128, nhead=4, 
                 num_layers=2, dim_feedforward=256, dropout=0.1, pool="mean"):
        super().__init__()
        self.pitch_dim = pitch_dim
        self.context_measures = context_measures
        self.pool = pool
        self.input_proj = nn.Linear(pitch_dim, d_model)  # Project pitch one-hot to d_model size
        # encoder layer for the transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        # stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Determine output size based on pooling strategy
        if pool == "concat":
            fc_input_size = d_model * context_measures
        else:  # mean or max pooling
            fc_input_size = d_model
            
        # final output layer to predict pitch logits
        self.fc = nn.Linear(fc_input_size, pitch_dim)

    def forward(self, x):
        # x is (B, M, P) - Batch, Measures, Pitch_dim
        x_proj = self.input_proj(x.float())  # (B, M, d_model)
        # pass through transformer encoder
        out = self.encoder(x_proj)  # (B, M, d_model)

        # pooling over measures
        if self.pool == "mean":
            pooled = out.mean(dim=1)  # average over measures
        elif self.pool == "max":
            pooled, _ = out.max(dim=1)  # max over measures
        elif self.pool == "concat":
            B, M, D = out.shape  # Batch, Measures, d_model
            pooled = out.view(B, M * D)  # flatten measures and d_model
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pool}\n Use 'concat', 'mean', or 'max'.")
        
        # final linear layer to get logits
        logits = self.fc(pooled)
        return logits  # logits for each pitch class