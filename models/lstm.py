import torch
import torch.nn as nn

"""LSTM-based model for MIDI sequence generation. Input is a sequence of integers (notes/chords),
output is a sequence of probabilities over the vocabulary for the next note/chord at each time step"""

class LSTMGenerator(nn.Module):
    # vocab_size: size of the possible tokens (notes/chords)
    # embed_size: size of the embedding vectors
    # hidden_size: number of features in the hidden state of LSTM
    # num_layers: number of stacked LSTM layers
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    # forward pass
    def forward(self, x, hidden=None):
        x = self.embedding(x) # convert input indices to embeddings
        out, hidden = self.lstm(x, hidden) # pass through LSTM
        out = self.fc(out) # project to vocab size
        return out, hidden # return output and hidden state