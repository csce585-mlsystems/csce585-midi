from models.discriminators.discriminator_mlp import DiscriminatorMLP
from models.discriminators.discriminator_lstm import DiscriminatorLSTM
from models.discriminators.discriminator_transformer import SimpleTransformerDiscriminator

def get_discriminator(model_type, pitch_dim, context_measures=4, **kwargs):
    """
    Factory function to create a discriminator model based on type.
        model_type: "mlp", "lstm", or "transformer"
        pitch_dim: size of input vector
        context_measures: how many measures used as context for predicting future measures
        
        hidden1: Size of the first hidden layer.
        hidden2: Size of the second hidden layer.
        pool: Pooling method (e.g., "concat", "max", "mean").
        dropout: Dropout probability.

        lstm and mlp args:
        embed_size: Dimension of the input embedding.
        hidden_size: Number of features in the LSTM hidden state.
        num_layers: Number of recurrent layers.
        dropout: Dropout probability.

        transformer args:
        embed_size: Maps to d_model (the number of expected features in the input).
        hidden_size: Maps to dim_feedforward (usually multiplied by 2 internally).
        num_heads: Maps to nhead (number of heads in the multiheadattention models).
        num_layers: Number of sub-encoder-layers in the encoder.
        pool: Pooling method.
        dropout: Dropout probability.
    """

    model_type = model_type.lower()

    if model_type == "mlp":
        # Filter kwargs for MLP - only pass relevant arguments
        mlp_kwargs = {k: v for k, v in kwargs.items() 
                     if k in ['hidden_sizes', 'dropout', 'pool', 'hidden1', 'hidden2']}
        # Convert hidden1, hidden2 to hidden_sizes list if provided
        if 'hidden1' in mlp_kwargs and 'hidden2' in mlp_kwargs:
            mlp_kwargs['hidden_sizes'] = [mlp_kwargs.pop('hidden1'), mlp_kwargs.pop('hidden2')]
        return DiscriminatorMLP(pitch_dim, context_measures=context_measures, **mlp_kwargs)
    
    elif model_type == "lstm":
        # Filter kwargs for LSTM - only pass relevant arguments
        lstm_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['embed_size', 'hidden_size', 'num_layers', 'dropout']}
        return DiscriminatorLSTM(pitch_dim, context_measures=context_measures, **lstm_kwargs)
    
    elif model_type == "transformer":
        # Filter kwargs for Transformer - only pass relevant arguments
        transformer_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout', 'pool']}
        # Map common arguments to transformer-specific names
        if 'embed_size' in kwargs:
            transformer_kwargs['d_model'] = kwargs['embed_size']
        if 'hidden_size' in kwargs:
            transformer_kwargs['dim_feedforward'] = kwargs['hidden_size'] * 2  # Common pattern
        if 'num_heads' in kwargs:
            transformer_kwargs['nhead'] = kwargs['num_heads']
        return SimpleTransformerDiscriminator(pitch_dim, context_measures=context_measures, **transformer_kwargs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'mlp', 'lstm', or 'transformer'.")