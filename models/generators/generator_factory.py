"""
Factory function for creating different types of generator models.
Supports LSTM, GRU, and Transformer architectures.
"""

from models.generators.generator_lstm import LSTMGenerator
from models.generators.generator_gru import GRUGenerator
from models.generators.generator_transformer import TransformerGenerator


def get_generator(model_type, vocab_size, **kwargs):
    """
    Factory function to create a generator model based on type.
    
    Args:
        model_type: Type of generator - "lstm", "gru", or "transformer"
        vocab_size: Size of the vocabulary
        **kwargs: Additional architecture-specific parameters
        
    Common kwargs:
        embed_size: Embedding dimension (LSTM/GRU: 128, Transformer d_model: 256)
        hidden_size: Hidden dimension (LSTM/GRU: 256, Transformer dim_feedforward: 1024)
        num_layers: Number of layers (default: 2 for LSTM/GRU, 6 for Transformer)
        dropout: Dropout rate (default: 0.2 for LSTM/GRU, 0.1 for Transformer)
        
    LSTM/GRU specific:
        embed_size: Size of embedding vectors (default: 128)
        hidden_size: Size of hidden state (default: 256)
        num_layers: Number of stacked layers (default: 2)
        dropout: Dropout between layers (default: 0.2)
        
    Transformer specific:
        d_model: Model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of decoder layers (default: 6)
        dim_feedforward: Feedforward dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        max_seq_length: Maximum sequence length (default: 5000)
        
    Returns:
        A generator model instance
        
    Example:
        >>> # Create LSTM generator
        >>> model = get_generator('lstm', vocab_size=256, embed_size=128, hidden_size=256)
        
        >>> # Create Transformer generator
        >>> model = get_generator('transformer', vocab_size=256, d_model=256, nhead=8)
    """
    model_type = model_type.lower()
    
    if model_type == "lstm":
        # Filter kwargs for LSTM
        lstm_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['embed_size', 'hidden_size', 'num_layers', 'dropout']}
        return LSTMGenerator(vocab_size, **lstm_kwargs)
    
    elif model_type == "gru":
        # Filter kwargs for GRU (same interface as LSTM)
        gru_kwargs = {k: v for k, v in kwargs.items() 
                     if k in ['embed_size', 'hidden_size', 'num_layers', 'dropout']}
        return GRUGenerator(vocab_size, **gru_kwargs)
    
    elif model_type == "transformer":
        # Filter kwargs for Transformer
        transformer_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 
                                     'dropout', 'max_seq_length']}
        
        # Map common RNN arguments to Transformer equivalents if provided
        if 'embed_size' in kwargs and 'd_model' not in transformer_kwargs:
            transformer_kwargs['d_model'] = kwargs['embed_size']
        if 'hidden_size' in kwargs and 'dim_feedforward' not in transformer_kwargs:
            # Transformer feedforward is typically 4x the model dimension
            transformer_kwargs['dim_feedforward'] = kwargs['hidden_size'] * 4
            
        return TransformerGenerator(vocab_size, **transformer_kwargs)
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from 'lstm', 'gru', or 'transformer'."
        )


def get_default_config(model_type):
    """
    Get default hyperparameters for a given model type.
    
    Args:
        model_type: Type of generator - "lstm", "gru", or "transformer"
        
    Returns:
        Dictionary of default hyperparameters
    """
    model_type = model_type.lower()
    
    if model_type == "lstm":
        return {
            'embed_size': 128,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.2,
        }
    
    elif model_type == "gru":
        return {
            'embed_size': 128,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.2,
        }
    
    elif model_type == "transformer":
        return {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 5000,
        }
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from 'lstm', 'gru', or 'transformer'."
        )
