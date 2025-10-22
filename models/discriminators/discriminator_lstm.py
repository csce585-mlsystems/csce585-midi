import torch.nn as nn
import torch

class DiscriminatorLSTM(nn.Module):
    def __init__(self, pitch_dim, context_measures=4, embed_size=128, hidden_size=256, num_layers=1, dropout=0.2):
        """An LSTM-based discriminator model.
        pitch_dim: Number of unique pitch tokens
        context_measures: Number of measures in the input sequence
        embed_size: Size of embedding for each pitch (to convert one-hot to dense vector)
            one-hot means each pitch is represented as a vector with one "1" and rest "0"s
        hidden_size: Number of hidden units in the LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        """

        super().__init__()
        self.embed = nn.Linear(pitch_dim, embed_size)  # Embed pitch one-hot to dense vector
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, pitch_dim)  # Output layer to predict pitch logits
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B, M, P) - Batch, Measures, Pitch_dim
        # Each "sequence" here is M measures of music
        # Each measure contains P possible pitches (which notes are active)
        
        # applies a linear transformation to the incoming data: y = xA^T + b
        x_proj = torch.relu(self.embed(x.float()))  # (B, M, embed_size)
        # pass through LSTM (returns output and (h_n, c_n) hidden states)
        out, _ = self.lstm(x_proj)  # (B, M, hidden_size)
        # take the output of the last time step
        last = out[:, -1, :]  # last time step (B, hidden_size)
        # applies dropout and then final linear layer
        logits = self.fc(self.dropout(last))  # (B, pitch_dim)
        return logits  # logits for each pitch class