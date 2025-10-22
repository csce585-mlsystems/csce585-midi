import torch
import torch.nn as nn

"""LSTM-based model for MIDI sequence generation. 
Input is a sequence of integers (notes/chords), output is a sequence of 
probabilities over the vocabulary for the next note/chord at each time step."""

class LSTMGenerator(nn.Module):
    """
    LSTM-based autoregressive music generator.
    
    Args:
        vocab_size: Size of the token vocabulary
        embed_size: Size of the embedding vectors (default: 128)
        hidden_size: Number of features in the hidden state (default: 256)
        num_layers: Number of stacked LSTM layers (default: 2)
        dropout: Dropout rate between LSTM layers (default: 0.2)
    """
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layers with dropout between layers if num_layers > 1
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection to vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            hidden: Optional hidden state from previous step (h, c) tuple
        
        Returns:
            output: Logits of shape (batch_size, seq_len, vocab_size)
            hidden: Hidden state tuple (h, c) for next step
        """
        # Embed input tokens
        x = self.embedding(x)
        
        # Pass through LSTM
        out, hidden = self.lstm(x, hidden)
        
        # Project to vocabulary
        out = self.fc(out)
        
        return out, hidden
