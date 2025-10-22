import torch
import torch.nn as nn
import math

"""Transformer-based model for MIDI sequence generation. 
Uses causal (autoregressive) masking to generate sequences token by token."""

class TransformerGenerator(nn.Module):
    """
    Transformer-based autoregressive music generator.
    
    Args:
        vocab_size: Size of the token vocabulary
        d_model: Dimension of embeddings and transformer (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer decoder layers (default: 6)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
        max_seq_length: Maximum sequence length for positional encoding (default: 5000)
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, max_seq_length=5000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer decoder layers (autoregressive)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x, memory=None):
        """
        Forward pass.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            memory: Optional memory tensor for cross-attention (not used in autoregressive mode)
        
        Returns:
            output: Logits of shape (batch_size, seq_len, vocab_size)
            None: For compatibility with LSTM interface (hidden state)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Create causal mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embedding
        x = self.pos_encoder(x)
        
        # If no memory provided, use self-attention only (standard autoregressive)
        if memory is None:
            memory = x
        
        # Pass through transformer decoder
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output, None  # Return None for hidden state compatibility


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
