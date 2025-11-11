import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.discriminators.discriminator_factory import get_discriminator
from models.discriminators.discriminator_mlp import DiscriminatorMLP
from models.discriminators.discriminator_lstm import DiscriminatorLSTM
from models.discriminators.discriminator_transformer import SimpleTransformerDiscriminator


# Test fixtures
@pytest.fixture
def batch_size():
    return 8

@pytest.fixture
def context_measures():
    return 4

@pytest.fixture
def pitch_dim():
    return 128  # Common pitch dimension for MIDI

@pytest.fixture
def sample_input(batch_size, context_measures, pitch_dim):
    """Create a sample input tensor (B, M, P)"""
    return torch.randn(batch_size, context_measures, pitch_dim)


class TestDiscriminatorFactory:
    """Test the discriminator factory function"""
    
    def test_factory_creates_mlp(self, pitch_dim):
        model = get_discriminator("mlp", pitch_dim)
        assert isinstance(model, DiscriminatorMLP)
    
    def test_factory_creates_lstm(self, pitch_dim):
        model = get_discriminator("lstm", pitch_dim)
        assert isinstance(model, DiscriminatorLSTM)
    
    def test_factory_creates_transformer(self, pitch_dim):
        model = get_discriminator("transformer", pitch_dim)
        assert isinstance(model, SimpleTransformerDiscriminator)
    
    def test_factory_case_insensitive(self, pitch_dim):
        """Test that model_type is case-insensitive"""
        model1 = get_discriminator("MLP", pitch_dim)
        model2 = get_discriminator("Lstm", pitch_dim)
        model3 = get_discriminator("TRANSFORMER", pitch_dim)
        
        assert isinstance(model1, DiscriminatorMLP)
        assert isinstance(model2, DiscriminatorLSTM)
        assert isinstance(model3, SimpleTransformerDiscriminator)
    
    def test_factory_invalid_type(self, pitch_dim):
        """Test that invalid model type raises error"""
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_discriminator("invalid_model", pitch_dim)
    
    def test_factory_passes_context_measures(self, pitch_dim):
        """Test that context_measures is passed correctly"""
        context = 8
        model = get_discriminator("mlp", pitch_dim, context_measures=context)
        assert model.context_measures == context
    
    def test_factory_mlp_kwargs(self, pitch_dim):
        """Test MLP-specific kwargs are handled correctly"""
        model = get_discriminator(
            "mlp", 
            pitch_dim, 
            hidden_sizes=[256, 128],
            dropout=0.3,
            pool="mean"
        )
        assert isinstance(model, DiscriminatorMLP)
        assert model.pool == "mean"
    
    def test_factory_lstm_kwargs(self, pitch_dim):
        """Test LSTM-specific kwargs are handled correctly"""
        model = get_discriminator(
            "lstm",
            pitch_dim,
            embed_size=64,
            hidden_size=128,
            num_layers=3,
            dropout=0.3
        )
        assert isinstance(model, DiscriminatorLSTM)
        assert model.lstm.num_layers == 3
    
    def test_factory_transformer_kwargs(self, pitch_dim):
        """Test Transformer-specific kwargs are handled correctly"""
        model = get_discriminator(
            "transformer",
            pitch_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        )
        assert isinstance(model, SimpleTransformerDiscriminator)
        assert model.encoder.layers[0].self_attn.num_heads == 8
    
    def test_factory_transformer_embed_size_mapping(self, pitch_dim):
        """Test that embed_size maps to d_model for transformer"""
        model = get_discriminator(
            "transformer",
            pitch_dim,
            embed_size=256  # Should map to d_model
        )
        assert isinstance(model, SimpleTransformerDiscriminator)


class TestDiscriminatorMLP:
    """Test the MLP discriminator model"""
    
    def test_initialization(self, pitch_dim, context_measures):
        """Test basic initialization"""
        model = DiscriminatorMLP(pitch_dim, context_measures=context_measures)
        assert model.pitch_dim == pitch_dim
        assert model.context_measures == context_measures
        assert model.pool == "concat"  # default
    
    def test_custom_hidden_sizes(self, pitch_dim):
        """Test custom hidden layer sizes"""
        hidden_sizes = [256, 128, 64]
        model = DiscriminatorMLP(pitch_dim, hidden_sizes=hidden_sizes)
        # Count Linear layers (excluding final output layer)
        linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == len(hidden_sizes) + 1  # +1 for output layer
    
    def test_forward_concat_pooling(self, sample_input, pitch_dim):
        """Test forward pass with concat pooling"""
        model = DiscriminatorMLP(pitch_dim, pool="concat")
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], pitch_dim)
    
    def test_forward_mean_pooling(self, sample_input, pitch_dim):
        """Test forward pass with mean pooling"""
        model = DiscriminatorMLP(pitch_dim, pool="mean")
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], pitch_dim)
    
    def test_forward_max_pooling(self, sample_input, pitch_dim):
        """Test forward pass with max pooling"""
        model = DiscriminatorMLP(pitch_dim, pool="max")
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], pitch_dim)
    
    def test_invalid_pooling(self, pitch_dim):
        """Test that invalid pooling strategy raises error"""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            DiscriminatorMLP(pitch_dim, pool="invalid")
    
    def test_dropout_rate(self, pitch_dim):
        """Test custom dropout rate"""
        model = DiscriminatorMLP(pitch_dim, dropout=0.5)
        dropout_layers = [m for m in model.net if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) > 0
        assert dropout_layers[0].p == 0.5
    
    def test_output_dtype(self, sample_input, pitch_dim):
        """Test output is float tensor"""
        model = DiscriminatorMLP(pitch_dim)
        output = model(sample_input)
        assert output.dtype == torch.float32
    
    def test_batch_independence(self, pitch_dim, context_measures):
        """Test that different batch samples produce different outputs"""
        model = DiscriminatorMLP(pitch_dim, context_measures=context_measures)
        
        # Create two different inputs
        input1 = torch.randn(1, context_measures, pitch_dim)
        input2 = torch.randn(1, context_measures, pitch_dim)
        
        output1 = model(input1)
        output2 = model(input2)
        
        # Outputs should be different (very unlikely to be equal)
        assert not torch.allclose(output1, output2)


class TestDiscriminatorLSTM:
    """Test the LSTM discriminator model"""
    
    def test_initialization(self, pitch_dim, context_measures):
        """Test basic initialization"""
        model = DiscriminatorLSTM(pitch_dim, context_measures=context_measures)
        assert model.lstm.input_size == 128  # default embed_size
        assert model.lstm.hidden_size == 256  # default
        assert model.lstm.num_layers == 1  # default
    
    def test_custom_architecture(self, pitch_dim):
        """Test custom architecture parameters"""
        model = DiscriminatorLSTM(
            pitch_dim,
            embed_size=64,
            hidden_size=128,
            num_layers=3,
            dropout=0.3
        )
        assert model.lstm.input_size == 64
        assert model.lstm.hidden_size == 128
        assert model.lstm.num_layers == 3
    
    def test_forward_pass(self, sample_input, pitch_dim):
        """Test forward pass"""
        model = DiscriminatorLSTM(pitch_dim)
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], pitch_dim)
    
    def test_output_dtype(self, sample_input, pitch_dim):
        """Test output is float tensor"""
        model = DiscriminatorLSTM(pitch_dim)
        output = model(sample_input)
        assert output.dtype == torch.float32
    
    def test_embedding_layer(self, pitch_dim):
        """Test embedding layer dimensions"""
        embed_size = 256
        model = DiscriminatorLSTM(pitch_dim, embed_size=embed_size)
        
        # Check embedding layer
        assert model.embed.in_features == pitch_dim
        assert model.embed.out_features == embed_size
    
    def test_output_layer(self, pitch_dim):
        """Test output layer dimensions"""
        hidden_size = 512
        model = DiscriminatorLSTM(pitch_dim, hidden_size=hidden_size)
        
        # Check output layer
        assert model.fc.in_features == hidden_size
        assert model.fc.out_features == pitch_dim
    
    def test_sequence_processing(self, pitch_dim):
        """Test that LSTM processes sequences correctly"""
        model = DiscriminatorLSTM(pitch_dim)
        
        # Different sequence lengths should work
        short_seq = torch.randn(4, 2, pitch_dim)  # 2 measures
        long_seq = torch.randn(4, 8, pitch_dim)   # 8 measures
        
        output_short = model(short_seq)
        output_long = model(long_seq)
        
        # Both should produce same output shape
        assert output_short.shape == output_long.shape == (4, pitch_dim)
    
    def test_batch_first(self, batch_size, pitch_dim):
        """Test that batch_first=True works correctly"""
        model = DiscriminatorLSTM(pitch_dim)
        input_tensor = torch.randn(batch_size, 4, pitch_dim)
        
        # Should not raise an error
        output = model(input_tensor)
        assert output.shape[0] == batch_size


class TestSimpleTransformerDiscriminator:
    """Test the Transformer discriminator model"""
    
    def test_initialization(self, pitch_dim, context_measures):
        """Test basic initialization"""
        model = SimpleTransformerDiscriminator(pitch_dim, context_measures=context_measures)
        assert model.pitch_dim == pitch_dim
        assert model.context_measures == context_measures
        assert model.pool == "mean"  # default
    
    def test_custom_architecture(self, pitch_dim):
        """Test custom architecture parameters"""
        model = SimpleTransformerDiscriminator(
            pitch_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.2
        )
        assert model.encoder.layers[0].self_attn.num_heads == 8
        assert len(model.encoder.layers) == 4
    
    def test_forward_mean_pooling(self, sample_input, pitch_dim):
        """Test forward pass with mean pooling"""
        model = SimpleTransformerDiscriminator(pitch_dim, pool="mean")
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], pitch_dim)
    
    def test_forward_max_pooling(self, sample_input, pitch_dim):
        """Test forward pass with max pooling"""
        model = SimpleTransformerDiscriminator(pitch_dim, pool="max")
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], pitch_dim)
    
    def test_forward_concat_pooling(self, sample_input, pitch_dim):
        """Test forward pass with concat pooling"""
        model = SimpleTransformerDiscriminator(pitch_dim, pool="concat")
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], pitch_dim)
    
    def test_invalid_pooling(self, sample_input, pitch_dim):
        """Test that invalid pooling strategy raises error"""
        model = SimpleTransformerDiscriminator(pitch_dim, pool="invalid")
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            model(sample_input)
    
    def test_attention_heads(self, pitch_dim):
        """Test that number of attention heads is correct"""
        nhead = 4
        d_model = 128
        model = SimpleTransformerDiscriminator(
            pitch_dim,
            d_model=d_model,
            nhead=nhead
        )
        assert model.encoder.layers[0].self_attn.num_heads == nhead
        # d_model must be divisible by nhead
        assert d_model % nhead == 0
    
    def test_invalid_nhead(self, pitch_dim):
        """Test that invalid nhead raises error (d_model not divisible by nhead)"""
        with pytest.raises(AssertionError):
            model = SimpleTransformerDiscriminator(
                pitch_dim,
                d_model=128,
                nhead=7  # 128 not divisible by 7
            )
            # Try to use the model
            sample = torch.randn(2, 4, pitch_dim)
            model(sample)
    
    def test_input_projection(self, pitch_dim):
        """Test input projection layer"""
        d_model = 256
        model = SimpleTransformerDiscriminator(pitch_dim, d_model=d_model)
        
        assert model.input_proj.in_features == pitch_dim
        assert model.input_proj.out_features == d_model
    
    def test_output_layer_concat(self, pitch_dim, context_measures):
        """Test output layer with concat pooling"""
        d_model = 128
        model = SimpleTransformerDiscriminator(
            pitch_dim,
            context_measures=context_measures,
            d_model=d_model,
            pool="concat"
        )
        
        # With concat, fc input should be d_model * context_measures
        assert model.fc.in_features == d_model * context_measures
        assert model.fc.out_features == pitch_dim
    
    def test_output_layer_mean(self, pitch_dim):
        """Test output layer with mean pooling"""
        d_model = 128
        model = SimpleTransformerDiscriminator(
            pitch_dim,
            d_model=d_model,
            pool="mean"
        )
        
        # With mean/max, fc input should be d_model
        assert model.fc.in_features == d_model
        assert model.fc.out_features == pitch_dim
    
    def test_different_sequence_lengths(self, batch_size, pitch_dim):
        """Test processing sequences of different lengths"""
        model = SimpleTransformerDiscriminator(pitch_dim)
        
        # Different sequence lengths
        short_seq = torch.randn(batch_size, 2, pitch_dim)
        long_seq = torch.randn(batch_size, 8, pitch_dim)
        
        output_short = model(short_seq)
        output_long = model(long_seq)
        
        # Both should produce same output shape
        assert output_short.shape == output_long.shape == (batch_size, pitch_dim)


class TestModelComparison:
    """Compare all three discriminator models"""
    
    def test_all_models_same_output_shape(self, sample_input, pitch_dim):
        """Test that all models produce the same output shape"""
        mlp = DiscriminatorMLP(pitch_dim)
        lstm = DiscriminatorLSTM(pitch_dim)
        transformer = SimpleTransformerDiscriminator(pitch_dim)
        
        output_mlp = mlp(sample_input)
        output_lstm = lstm(sample_input)
        output_transformer = transformer(sample_input)
        
        expected_shape = (sample_input.shape[0], pitch_dim)
        assert output_mlp.shape == expected_shape
        assert output_lstm.shape == expected_shape
        assert output_transformer.shape == expected_shape
    
    def test_all_models_produce_logits(self, sample_input, pitch_dim):
        """Test that all models produce logits (no activation)"""
        mlp = DiscriminatorMLP(pitch_dim)
        lstm = DiscriminatorLSTM(pitch_dim)
        transformer = SimpleTransformerDiscriminator(pitch_dim)
        
        output_mlp = mlp(sample_input)
        output_lstm = lstm(sample_input)
        output_transformer = transformer(sample_input)
        
        # Logits should not be bounded to [0, 1] or [-1, 1]
        # Check that at least some values are outside typical activation ranges
        assert (output_mlp.abs() > 1).any() or (output_mlp < 0).any()
        assert (output_lstm.abs() > 1).any() or (output_lstm < 0).any()
        assert (output_transformer.abs() > 1).any() or (output_transformer < 0).any()
    
    def test_parameter_counts(self, pitch_dim):
        """Compare parameter counts across models"""
        mlp = DiscriminatorMLP(pitch_dim, hidden_sizes=[256, 128])
        lstm = DiscriminatorLSTM(pitch_dim, embed_size=128, hidden_size=256)
        transformer = SimpleTransformerDiscriminator(pitch_dim, d_model=128, nhead=4)
        
        mlp_params = sum(p.numel() for p in mlp.parameters())
        lstm_params = sum(p.numel() for p in lstm.parameters())
        transformer_params = sum(p.numel() for p in transformer.parameters())
        
        # All should have some parameters
        assert mlp_params > 0
        assert lstm_params > 0
        assert transformer_params > 0
        
        # LSTM typically has more parameters than MLP
        # Transformer typically has most parameters
        print(f"\nParameter counts:")
        print(f"  MLP: {mlp_params:,}")
        print(f"  LSTM: {lstm_params:,}")
        print(f"  Transformer: {transformer_params:,}")


class TestGradientFlow:
    """Test that gradients flow correctly through models"""
    
    def test_mlp_gradient_flow(self, sample_input, pitch_dim):
        """Test gradient flow through MLP"""
        model = DiscriminatorMLP(pitch_dim)
        output = model(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_lstm_gradient_flow(self, sample_input, pitch_dim):
        """Test gradient flow through LSTM"""
        model = DiscriminatorLSTM(pitch_dim)
        output = model(sample_input)
        loss = output.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_transformer_gradient_flow(self, sample_input, pitch_dim):
        """Test gradient flow through Transformer"""
        model = SimpleTransformerDiscriminator(pitch_dim)
        output = model(sample_input)
        loss = output.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_batch(self, pitch_dim, context_measures):
        """Test with batch size of 1"""
        input_tensor = torch.randn(1, context_measures, pitch_dim)
        
        mlp = DiscriminatorMLP(pitch_dim)
        lstm = DiscriminatorLSTM(pitch_dim)
        transformer = SimpleTransformerDiscriminator(pitch_dim)
        
        # Should not raise errors
        output_mlp = mlp(input_tensor)
        output_lstm = lstm(input_tensor)
        output_transformer = transformer(input_tensor)
        
        assert output_mlp.shape == (1, pitch_dim)
        assert output_lstm.shape == (1, pitch_dim)
        assert output_transformer.shape == (1, pitch_dim)
    
    def test_large_batch(self, pitch_dim, context_measures):
        """Test with large batch size"""
        batch_size = 256
        input_tensor = torch.randn(batch_size, context_measures, pitch_dim)
        
        mlp = DiscriminatorMLP(pitch_dim)
        output = mlp(input_tensor)
        
        assert output.shape == (batch_size, pitch_dim)
    
    def test_zero_input(self, batch_size, context_measures, pitch_dim):
        """Test with zero input"""
        input_tensor = torch.zeros(batch_size, context_measures, pitch_dim)
        
        mlp = DiscriminatorMLP(pitch_dim)
        output = mlp(input_tensor)
        
        # Should produce output (not crash)
        assert output.shape == (batch_size, pitch_dim)
        assert not torch.isnan(output).any()
    
    def test_very_small_pitch_dim(self):
        """Test with very small pitch dimension"""
        pitch_dim = 2
        context_measures = 4
        input_tensor = torch.randn(4, context_measures, pitch_dim)
        
        mlp = DiscriminatorMLP(pitch_dim, hidden_sizes=[8])
        lstm = DiscriminatorLSTM(pitch_dim, embed_size=4, hidden_size=8)
        transformer = SimpleTransformerDiscriminator(pitch_dim, d_model=4, nhead=2)
        
        # Should work with small dimensions
        output_mlp = mlp(input_tensor)
        output_lstm = lstm(input_tensor)
        output_transformer = transformer(input_tensor)
        
        assert output_mlp.shape == (4, pitch_dim)
        assert output_lstm.shape == (4, pitch_dim)
        assert output_transformer.shape == (4, pitch_dim)


class TestModelPersistence:
    """Test saving and loading models"""
    
    def test_save_load_mlp(self, pitch_dim, tmp_path):
        """Test saving and loading MLP model"""
        model = DiscriminatorMLP(pitch_dim)
        
        # Save model
        save_path = tmp_path / "mlp_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = DiscriminatorMLP(pitch_dim)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Set to eval mode to disable dropout
        model.eval()
        loaded_model.eval()
        
        # Test that outputs are identical
        input_tensor = torch.randn(4, 4, pitch_dim)
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = loaded_model(input_tensor)
        
        assert torch.allclose(output1, output2)
    
    def test_save_load_lstm(self, pitch_dim, tmp_path):
        """Test saving and loading LSTM model"""
        model = DiscriminatorLSTM(pitch_dim)
        
        # Save model
        save_path = tmp_path / "lstm_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = DiscriminatorLSTM(pitch_dim)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Set to eval mode to disable dropout
        model.eval()
        loaded_model.eval()
        
        # Test that outputs are identical
        input_tensor = torch.randn(4, 4, pitch_dim)
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = loaded_model(input_tensor)
        
        assert torch.allclose(output1, output2)
    
    def test_save_load_transformer(self, pitch_dim, tmp_path):
        """Test saving and loading Transformer model"""
        model = SimpleTransformerDiscriminator(pitch_dim)
        
        # Save model
        save_path = tmp_path / "transformer_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = SimpleTransformerDiscriminator(pitch_dim)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Set to eval mode to disable dropout
        model.eval()
        loaded_model.eval()
        
        # Test that outputs are identical
        input_tensor = torch.randn(4, 4, pitch_dim)
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = loaded_model(input_tensor)
        
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
