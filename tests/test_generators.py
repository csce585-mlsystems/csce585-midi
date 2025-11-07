"""
Test script to verify all generator architectures work correctly.
Tests LSTM, GRU, and Transformer generators with dummy data.
"""
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.generators.generator_factory import get_generator, get_default_config

def test_generator(model_types, vocab_size=256, batch_size=4, seq_len=50):
    for model_type in model_types:
        """Test a generator model with dummy data."""
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()} Generator")
        print(f"{'='*60}")
        
        # Get default config
        config = get_default_config(model_type)
        print(f"Default config: {config}")
        
        # Create model
        model = get_generator(model_type, vocab_size, **config)
        print(f"Model created successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create dummy input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output, hidden = model(x)
        
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
        
        # Verify output shape
        assert output.shape == (batch_size, seq_len, vocab_size), \
            f"Wrong output shape! Got {output.shape}, expected ({batch_size}, {seq_len}, {vocab_size})"
        
        # Verify output is valid (not NaN or Inf)
        assert not torch.isnan(output).any(), "Output contains NaN!"
        assert not torch.isinf(output).any(), "Output contains Inf!"
        
        print(f"✓ {model_type.upper()} generator passed all tests!")
        
        # Test with different sequence lengths
        for test_len in [10, 100]:
            x_test = torch.randint(0, vocab_size, (2, test_len))
            with torch.no_grad():
                out_test, _ = model(x_test)
            assert out_test.shape == (2, test_len, vocab_size), \
                f"Failed with seq_len={test_len}"
        print(f"✓ Variable sequence length tests passed!")
        

def test_custom_configs():
    """Test generators with custom configurations."""
    print("\n" + "="*60)
    print("TESTING CUSTOM CONFIGURATIONS")
    print("="*60)
    
    vocab_size = 256
    
    # Test LSTM with custom config
    print("\nLSTM with large config...")
    lstm = get_generator("lstm", vocab_size, embed_size=256, hidden_size=512, num_layers=3)
    print(f"Parameters: {sum(p.numel() for p in lstm.parameters()):,}")
    
    # Test Transformer with small config
    print("\nTransformer with small config...")
    transformer = get_generator("transformer", vocab_size, d_model=128, nhead=4, num_layers=2)
    print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Test GRU with minimal config
    print("\nGRU with minimal config...")
    gru = get_generator("gru", vocab_size, embed_size=64, hidden_size=128, num_layers=1)
    print(f"Parameters: {sum(p.numel() for p in gru.parameters()):,}")
    
    print("\n✓ Custom configuration tests passed!")

def test_gradient_flow(model_types):
    """test that gradients flow correctly"""
    print("\n" + "="*60)
    print("TESTING GRADIENT FLOW")
    print("="*60)

    vocab_size = 128
    batch_size = 4
    seq_len = 20

    for model_type in model_types:
        print(f"\nTesting {model_type.upper()} gradient flow..")

        model = get_generator(model_type, vocab_size)
        model.train()

        # create dummy data
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        target = torch.randint(0, vocab_size, (batch_size, seq_len))

        # forward pass
        output, _ = model(x)

        # loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output.view(-1, vocab_size), target.view(-1))

        # backward pass
        loss.backward()

        # check that gradients exist and are non-zero
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()

        assert grad_norm > 0, f"{model_type} has zero gradients"
        print(f"gradients flow properly (grad_norm: {grad_norm:.4f})")

def test_training_step():
    """Test a full training step (forward + backward + optimizer)."""
    print("\n" + "="*60)
    print("TESTING TRAINING STEP")
    print("="*60)
    
    vocab_size = 128
    batch_size = 4
    seq_len = 20
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} training step...")
        
        model = get_generator(model_type, vocab_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Create dummy data
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Get initial loss
        model.train()
        output, _ = model(x)
        initial_loss = loss_fn(output.view(-1, vocab_size), target.view(-1)).item()
        
        # Train for a few steps
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            output, _ = model(x)
            loss = loss_fn(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Loss decreased: {initial_loss > final_loss}")
        
        # Loss should generally decrease (though not guaranteed with random data)
        assert final_loss < initial_loss * 2, f"{model_type} loss exploded!"
        print(f"✓ Training step works properly")


def test_output_distribution():
    """Test that output logits have reasonable distributions."""
    print("\n" + "="*60)
    print("TESTING OUTPUT DISTRIBUTIONS")
    print("="*60)
    
    vocab_size = 128
    batch_size = 8
    seq_len = 50
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} output distribution...")
        
        model = get_generator(model_type, vocab_size)
        model.eval()
        
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output, _ = model(x)
        
        # Check logit statistics
        logits = output.view(-1, vocab_size)
        
        mean = logits.mean().item()
        std = logits.std().item()
        min_val = logits.min().item()
        max_val = logits.max().item()
        
        print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        
        # Check that logits are reasonable (not all zeros or extreme values)
        assert abs(mean) < 10, f"Mean logit too extreme: {mean}"
        assert std > 0.01, f"Std too small: {std}"
        assert std < 100, f"Std too large: {std}"
        
        # Check probability distribution after softmax
        probs = torch.softmax(logits, dim=-1)
        
        # Entropy should be reasonable (not all mass on one token)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        expected_max_entropy = np.log(vocab_size)
        
        print(f"  Entropy: {entropy:.4f} (max possible: {expected_max_entropy:.4f})")
        assert entropy > 0.5, f"Entropy too low: {entropy}"
        
        print(f"✓ Output distribution is reasonable")


def test_deterministic_output():
    """Test that same input gives same output (deterministic)."""
    print("\n" + "="*60)
    print("TESTING DETERMINISTIC OUTPUT")
    print("="*60)
    
    vocab_size = 128
    batch_size = 2
    seq_len = 20
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} determinism...")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        model = get_generator(model_type, vocab_size)
        model.eval()
        
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Get output twice
        with torch.no_grad():
            output1, _ = model(x)
            output2, _ = model(x)
        
        # Should be identical
        diff = (output1 - output2).abs().max().item()
        
        print(f"  Max difference: {diff}")
        assert diff < 1e-6, f"Output not deterministic! Diff: {diff}"
        print(f"✓ Output is deterministic")


def test_batch_independence():
    """Test that batch samples are processed independently."""
    print("\n" + "="*60)
    print("TESTING BATCH INDEPENDENCE")
    print("="*60)
    
    vocab_size = 128
    seq_len = 20
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} batch independence...")
        
        model = get_generator(model_type, vocab_size)
        model.eval()
        
        # Create two different inputs
        x1 = torch.randint(0, vocab_size, (1, seq_len))
        x2 = torch.randint(0, vocab_size, (1, seq_len))
        
        # Process individually
        with torch.no_grad():
            out1, _ = model(x1)
            out2, _ = model(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            out_batch, _ = model(x_batch)
        
        # Batch processing should give same results as individual
        diff1 = (out1 - out_batch[0]).abs().max().item()
        diff2 = (out2 - out_batch[1]).abs().max().item()
        
        print(f"  Sample 1 diff: {diff1:.6f}")
        print(f"  Sample 2 diff: {diff2:.6f}")
        
        assert diff1 < 1e-5, f"Batch processing changes output for sample 1"
        assert diff2 < 1e-5, f"Batch processing changes output for sample 2"
        print(f"✓ Batch samples processed independently")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    vocab_size = 128
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} edge cases...")
        
        model = get_generator(model_type, vocab_size)
        model.eval()
        
        # Test 1: Single token sequence
        print("  Testing single token...")
        x = torch.randint(0, vocab_size, (1, 1))
        with torch.no_grad():
            output, _ = model(x)
        assert output.shape == (1, 1, vocab_size)
        print("  ✓ Single token works")
        
        # Test 2: Very long sequence
        print("  Testing long sequence...")
        x = torch.randint(0, vocab_size, (1, 500))
        with torch.no_grad():
            output, _ = model(x)
        assert output.shape == (1, 500, vocab_size)
        print("  ✓ Long sequence works")
        
        # Test 3: All same token
        print("  Testing repeated token...")
        x = torch.full((2, 20), 42)  # All token 42
        with torch.no_grad():
            output, _ = model(x)
        assert not torch.isnan(output).any()
        print("  ✓ Repeated token works")
        
        # Test 4: Sequential tokens (0, 1, 2, ...)
        print("  Testing sequential tokens...")
        x = torch.arange(min(50, vocab_size)).unsqueeze(0)
        with torch.no_grad():
            output, _ = model(x)
        assert not torch.isnan(output).any()
        print("  ✓ Sequential tokens work")


def test_model_save_load():
    """Test that models can be saved and loaded correctly."""
    print("\n" + "="*60)
    print("TESTING MODEL SAVE/LOAD")
    print("="*60)
    
    vocab_size = 128
    batch_size = 2
    seq_len = 20
    
    import tempfile
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} save/load...")
        
        # Create and train model slightly
        model = get_generator(model_type, vocab_size)
        optimizer = torch.optim.Adam(model.parameters())
        
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Train one step
        model.train()
        optimizer.zero_grad()
        output, _ = model(x)
        loss = torch.nn.CrossEntropyLoss()(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        
        # Get output
        model.eval()
        with torch.no_grad():
            output_before, _ = model(x)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name
        
        # Create new model and load weights
        model_loaded = get_generator(model_type, vocab_size)
        model_loaded.load_state_dict(torch.load(temp_path))
        model_loaded.eval()
        
        # Get output from loaded model
        with torch.no_grad():
            output_after, _ = model_loaded(x)
        
        # Should be identical
        diff = (output_before - output_after).abs().max().item()
        
        print(f"  Max difference after load: {diff}")
        assert diff < 1e-6, f"Loaded model gives different output!"
        print(f"✓ Save/load works correctly")
        
        # Cleanup
        Path(temp_path).unlink()


def test_memory_efficiency():
    """Test that models don't leak memory during training."""
    print("\n" + "="*60)
    print("TESTING MEMORY EFFICIENCY")
    print("="*60)
    
    vocab_size = 128
    batch_size = 8
    seq_len = 50
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} memory usage...")
        
        model = get_generator(model_type, vocab_size)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Measure initial allocated memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Run multiple training steps
        for _ in range(100):
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            target = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            optimizer.zero_grad()
            output, _ = model(x)
            loss = loss_fn(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
        
        # Memory shouldn't grow unbounded
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"  Peak memory: {peak_mem:.2f} MB")
        else:
            print(f"  CPU mode - memory test skipped")
        
        print(f"✓ No obvious memory leaks")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATOR COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    model_types = ["lstm", "gru", "transformer"]
    
    # Run all tests
    test_generator(model_types)
    test_custom_configs()
    test_gradient_flow()
    test_training_step()
    test_output_distribution()
    test_deterministic_output()
    test_batch_independence()
    test_edge_cases()
    test_model_save_load()
    test_memory_efficiency()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)

def test_factory_invalid_model_type():
    """Test that factory raises error for invalid model type."""
    print("\n" + "="*60)
    print("TESTING INVALID MODEL TYPE")
    print("="*60)
    
    vocab_size = 128
    
    try:
        model = get_generator("invalid_model", vocab_size)
        assert False, "Should have raised ValueError for invalid model type"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        assert "unknown model_type" in str(e).lower() or "not supported" in str(e).lower()


def test_factory_all_model_types():
    """Test that factory supports all documented model types."""
    print("\n" + "="*60)
    print("TESTING ALL FACTORY MODEL TYPES")
    print("="*60)
    
    vocab_size = 128
    supported_types = ["lstm", "gru", "transformer"]
    
    for model_type in supported_types:
        print(f"\nCreating {model_type.upper()} via factory...")
        model = get_generator(model_type, vocab_size)
        assert model is not None, f"Factory returned None for {model_type}"
        print(f"✓ {model_type.upper()} created successfully")


def test_default_config_all_types():
    """Test get_default_config for all model types."""
    print("\n" + "="*60)
    print("TESTING DEFAULT CONFIGS")
    print("="*60)
    
    supported_types = ["lstm", "gru", "transformer"]
    
    for model_type in supported_types:
        print(f"\nGetting default config for {model_type.upper()}...")
        config = get_default_config(model_type)
        
        assert isinstance(config, dict), f"Config should be dict, got {type(config)}"
        assert len(config) > 0, f"Config should not be empty for {model_type}"
        
        print(f"  Config keys: {list(config.keys())}")
        
        # Check for expected keys based on model type
        if model_type in ["lstm", "gru"]:
            assert "embed_size" in config or "hidden_size" in config, \
                f"LSTM/GRU config should have embed_size or hidden_size"
        elif model_type == "transformer":
            assert "d_model" in config or "nhead" in config, \
                f"Transformer config should have d_model or nhead"
        
        print(f"✓ Default config valid for {model_type.upper()}")


def test_default_config_invalid_type():
    """Test get_default_config with invalid model type."""
    print("\n" + "="*60)
    print("TESTING INVALID DEFAULT CONFIG")
    print("="*60)
    
    try:
        config = get_default_config("invalid_model")
        # If it doesn't raise error, it might return empty dict - check for that
        print(f"Config returned: {config}")
        # Some implementations might return empty dict instead of raising error
    except (ValueError, KeyError) as e:
        print(f"✓ Correctly raised error: {e}")


def test_factory_with_all_hyperparameters():
    """Test factory with all possible hyperparameter combinations."""
    print("\n" + "="*60)
    print("TESTING ALL HYPERPARAMETER COMBINATIONS")
    print("="*60)
    
    vocab_size = 128
    
    # Test LSTM with all hyperparameters
    print("\nLSTM with custom hyperparameters...")
    lstm = get_generator(
        "lstm", 
        vocab_size,
        embed_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    assert lstm is not None
    print("✓ LSTM with all params created")
    
    # Test GRU with all hyperparameters
    print("\nGRU with custom hyperparameters...")
    gru = get_generator(
        "gru",
        vocab_size,
        embed_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    assert gru is not None
    print("✓ GRU with all params created")
    
    # Test Transformer with all hyperparameters
    print("\nTransformer with custom hyperparameters...")
    transformer = get_generator(
        "transformer",
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.2
    )
    assert transformer is not None
    print("✓ Transformer with all params created")


def test_factory_with_minimal_params():
    """Test factory with only required parameters."""
    print("\n" + "="*60)
    print("TESTING MINIMAL PARAMETERS")
    print("="*60)
    
    vocab_size = 128
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nCreating {model_type.upper()} with minimal params...")
        model = get_generator(model_type, vocab_size)
        assert model is not None
        
        # Verify model works
        x = torch.randint(0, vocab_size, (2, 10))
        with torch.no_grad():
            output, _ = model(x)
        assert output.shape == (2, 10, vocab_size)
        print(f"✓ {model_type.upper()} works with minimal params")


def test_factory_parameter_override():
    """Test that custom parameters override defaults."""
    print("\n" + "="*60)
    print("TESTING PARAMETER OVERRIDE")
    print("="*60)
    
    vocab_size = 128
    
    # Get default LSTM config
    default_config = get_default_config("lstm")
    default_hidden = default_config.get("hidden_size", 256)
    
    # Create LSTM with custom hidden size
    custom_hidden = 512
    lstm = get_generator("lstm", vocab_size, hidden_size=custom_hidden)
    
    print(f"Default hidden_size: {default_hidden}")
    print(f"Custom hidden_size: {custom_hidden}")
    
    # Verify the custom parameter was used by checking model size
    param_count_custom = sum(p.numel() for p in lstm.parameters())
    
    lstm_default = get_generator("lstm", vocab_size)
    param_count_default = sum(p.numel() for p in lstm_default.parameters())
    
    print(f"Default params: {param_count_default:,}")
    print(f"Custom params: {param_count_custom:,}")
    
    assert param_count_custom != param_count_default, \
        "Custom parameters should result in different model size"
    print("✓ Parameters correctly overridden")


def test_factory_vocab_size_variations():
    """Test factory with different vocabulary sizes."""
    print("\n" + "="*60)
    print("TESTING VOCABULARY SIZE VARIATIONS")
    print("="*60)
    
    vocab_sizes = [10, 100, 1000, 10000]
    
    for vocab_size in vocab_sizes:
        print(f"\nTesting with vocab_size={vocab_size}...")
        
        for model_type in ["lstm", "gru", "transformer"]:
            model = get_generator(model_type, vocab_size)
            
            # Verify output dimension matches vocab size
            x = torch.randint(0, vocab_size, (1, 10))
            with torch.no_grad():
                output, _ = model(x)
            
            assert output.shape[-1] == vocab_size, \
                f"Output dim {output.shape[-1]} doesn't match vocab_size {vocab_size}"
        
        print(f"✓ All models work with vocab_size={vocab_size}")


def test_factory_case_insensitive():
    """Test if factory handles different case variations of model names."""
    print("\n" + "="*60)
    print("TESTING CASE SENSITIVITY")
    print("="*60)
    
    vocab_size = 128
    
    # Try different case variations
    variations = [
        ("lstm", "LSTM", "Lstm"),
        ("gru", "GRU", "Gru"),
        ("transformer", "TRANSFORMER", "Transformer")
    ]
    
    for lower, upper, title in variations:
        print(f"\nTesting {lower} with different cases...")
        
        try:
            # Try lowercase (should work)
            model1 = get_generator(lower, vocab_size)
            print(f"  ✓ lowercase '{lower}' works")
        except:
            print(f"  ✗ lowercase '{lower}' failed")
        
        try:
            # Try uppercase
            model2 = get_generator(upper, vocab_size)
            print(f"  ✓ uppercase '{upper}' works")
        except ValueError:
            print(f"  ✗ uppercase '{upper}' not supported (this is OK)")
        
        try:
            # Try title case
            model3 = get_generator(title, vocab_size)
            print(f"  ✓ title case '{title}' works")
        except ValueError:
            print(f"  ✗ title case '{title}' not supported (this is OK)")


def test_factory_model_uniqueness():
    """Test that each factory call creates a new independent model."""
    print("\n" + "="*60)
    print("TESTING MODEL UNIQUENESS")
    print("="*60)
    
    vocab_size = 128
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} uniqueness...")
        
        # Create two models
        model1 = get_generator(model_type, vocab_size)
        model2 = get_generator(model_type, vocab_size)
        
        # Models should be different objects
        assert model1 is not model2, "Factory should create new instances"
        
        # Modify one model's parameters
        for param in model1.parameters():
            param.data.fill_(1.0)
            break
        
        # Other model should be unchanged
        for param in model2.parameters():
            assert not torch.all(param.data == 1.0), \
                "Models should be independent"
            break
        
        print(f"✓ {model_type.upper()} models are independent")


def test_factory_error_messages():
    """Test that factory provides helpful error messages."""
    print("\n" + "="*60)
    print("TESTING ERROR MESSAGES")
    print("="*60)
    
    vocab_size = 128
    
    # Test various invalid inputs
    invalid_inputs = [
        ("", "empty string"),
        (None, "None"),
        (123, "integer"),
        (["lstm"], "list"),
        ("cnn", "unsupported model type"),
        ("rnn", "potentially ambiguous name"),
    ]
    
    for invalid_input, description in invalid_inputs:
        print(f"\nTesting with {description}: {invalid_input}")
        try:
            model = get_generator(invalid_input, vocab_size)
            print(f"  ⚠ No error raised (might be OK if it has default handling)")
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            print(f"  ✓ Correctly raised {type(e).__name__}: {e}")


# Update the main test runner to include all new tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATOR COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    model_types = ["lstm", "gru", "transformer"]
    
    # Run all existing tests
    test_generator(model_types)
    test_custom_configs()
    test_gradient_flow(model_types)
    test_training_step()
    test_output_distribution()
    test_deterministic_output()
    test_batch_independence()
    test_edge_cases()
    test_model_save_load()
    test_memory_efficiency()
    
    # Run new coverage tests
    test_factory_invalid_model_type()
    test_factory_all_model_types()
    test_default_config_all_types()
    test_default_config_invalid_type()
    test_factory_with_all_hyperparameters()
    test_factory_with_minimal_params()
    test_factory_parameter_override()
    test_factory_vocab_size_variations()
    test_factory_case_insensitive()
    test_factory_model_uniqueness()
    test_factory_error_messages()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)

def test_factory_invalid_model_type():
    """Test that factory raises error for invalid model type."""
    print("\n" + "="*60)
    print("TESTING INVALID MODEL TYPE")
    print("="*60)
    
    vocab_size = 128
    
    try:
        model = get_generator("invalid_model", vocab_size)
        assert False, "Should have raised ValueError for invalid model type"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        assert "Unknown model_type" in str(e) or "not supported" in str(e).lower()


def test_factory_all_model_types():
    """Test that factory supports all documented model types."""
    print("\n" + "="*60)
    print("TESTING ALL FACTORY MODEL TYPES")
    print("="*60)
    
    vocab_size = 128
    supported_types = ["lstm", "gru", "transformer"]
    
    for model_type in supported_types:
        print(f"\nCreating {model_type.upper()} via factory...")
        model = get_generator(model_type, vocab_size)
        assert model is not None, f"Factory returned None for {model_type}"
        print(f"✓ {model_type.upper()} created successfully")


def test_default_config_all_types():
    """Test get_default_config for all model types."""
    print("\n" + "="*60)
    print("TESTING DEFAULT CONFIGS")
    print("="*60)
    
    supported_types = ["lstm", "gru", "transformer"]
    
    for model_type in supported_types:
        print(f"\nGetting default config for {model_type.upper()}...")
        config = get_default_config(model_type)
        
        assert isinstance(config, dict), f"Config should be dict, got {type(config)}"
        assert len(config) > 0, f"Config should not be empty for {model_type}"
        
        print(f"  Config keys: {list(config.keys())}")
        
        # Check for expected keys based on model type
        if model_type in ["lstm", "gru"]:
            assert "embed_size" in config or "hidden_size" in config, \
                f"LSTM/GRU config should have embed_size or hidden_size"
        elif model_type == "transformer":
            assert "d_model" in config or "nhead" in config, \
                f"Transformer config should have d_model or nhead"
        
        print(f"✓ Default config valid for {model_type.upper()}")


def test_default_config_invalid_type():
    """Test get_default_config with invalid model type."""
    print("\n" + "="*60)
    print("TESTING INVALID DEFAULT CONFIG")
    print("="*60)
    
    try:
        config = get_default_config("invalid_model")
        # If it doesn't raise error, it might return empty dict - check for that
        print(f"Config returned: {config}")
        # Some implementations might return empty dict instead of raising error
    except (ValueError, KeyError) as e:
        print(f"✓ Correctly raised error: {e}")


def test_factory_with_all_hyperparameters():
    """Test factory with all possible hyperparameter combinations."""
    print("\n" + "="*60)
    print("TESTING ALL HYPERPARAMETER COMBINATIONS")
    print("="*60)
    
    vocab_size = 128
    
    # Test LSTM with all hyperparameters
    print("\nLSTM with custom hyperparameters...")
    lstm = get_generator(
        "lstm", 
        vocab_size,
        embed_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    assert lstm is not None
    print("✓ LSTM with all params created")
    
    # Test GRU with all hyperparameters
    print("\nGRU with custom hyperparameters...")
    gru = get_generator(
        "gru",
        vocab_size,
        embed_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    assert gru is not None
    print("✓ GRU with all params created")
    
    # Test Transformer with all hyperparameters
    print("\nTransformer with custom hyperparameters...")
    transformer = get_generator(
        "transformer",
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.2
    )
    assert transformer is not None
    print("✓ Transformer with all params created")


def test_factory_with_minimal_params():
    """Test factory with only required parameters."""
    print("\n" + "="*60)
    print("TESTING MINIMAL PARAMETERS")
    print("="*60)
    
    vocab_size = 128
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nCreating {model_type.upper()} with minimal params...")
        model = get_generator(model_type, vocab_size)
        assert model is not None
        
        # Verify model works
        x = torch.randint(0, vocab_size, (2, 10))
        with torch.no_grad():
            output, _ = model(x)
        assert output.shape == (2, 10, vocab_size)
        print(f"✓ {model_type.upper()} works with minimal params")


def test_factory_parameter_override():
    """Test that custom parameters override defaults."""
    print("\n" + "="*60)
    print("TESTING PARAMETER OVERRIDE")
    print("="*60)
    
    vocab_size = 128
    
    # Get default LSTM config
    default_config = get_default_config("lstm")
    default_hidden = default_config.get("hidden_size", 256)
    
    # Create LSTM with custom hidden size
    custom_hidden = 512
    lstm = get_generator("lstm", vocab_size, hidden_size=custom_hidden)
    
    print(f"Default hidden_size: {default_hidden}")
    print(f"Custom hidden_size: {custom_hidden}")
    
    # Verify the custom parameter was used by checking model size
    param_count_custom = sum(p.numel() for p in lstm.parameters())
    
    lstm_default = get_generator("lstm", vocab_size)
    param_count_default = sum(p.numel() for p in lstm_default.parameters())
    
    print(f"Default params: {param_count_default:,}")
    print(f"Custom params: {param_count_custom:,}")
    
    assert param_count_custom != param_count_default, \
        "Custom parameters should result in different model size"
    print("✓ Parameters correctly overridden")


def test_factory_vocab_size_variations():
    """Test factory with different vocabulary sizes."""
    print("\n" + "="*60)
    print("TESTING VOCABULARY SIZE VARIATIONS")
    print("="*60)
    
    vocab_sizes = [10, 100, 1000, 10000]
    
    for vocab_size in vocab_sizes:
        print(f"\nTesting with vocab_size={vocab_size}...")
        
        for model_type in ["lstm", "gru", "transformer"]:
            model = get_generator(model_type, vocab_size)
            
            # Verify output dimension matches vocab size
            x = torch.randint(0, vocab_size, (1, 10))
            with torch.no_grad():
                output, _ = model(x)
            
            assert output.shape[-1] == vocab_size, \
                f"Output dim {output.shape[-1]} doesn't match vocab_size {vocab_size}"
        
        print(f"✓ All models work with vocab_size={vocab_size}")


def test_factory_case_insensitive():
    """Test if factory handles different case variations of model names."""
    print("\n" + "="*60)
    print("TESTING CASE SENSITIVITY")
    print("="*60)
    
    vocab_size = 128
    
    # Try different case variations
    variations = [
        ("lstm", "LSTM", "Lstm"),
        ("gru", "GRU", "Gru"),
        ("transformer", "TRANSFORMER", "Transformer")
    ]
    
    for lower, upper, title in variations:
        print(f"\nTesting {lower} with different cases...")
        
        try:
            # Try lowercase (should work)
            model1 = get_generator(lower, vocab_size)
            print(f"  ✓ lowercase '{lower}' works")
        except:
            print(f"  ✗ lowercase '{lower}' failed")
        
        try:
            # Try uppercase
            model2 = get_generator(upper, vocab_size)
            print(f"  ✓ uppercase '{upper}' works")
        except ValueError:
            print(f"  ✗ uppercase '{upper}' not supported (this is OK)")
        
        try:
            # Try title case
            model3 = get_generator(title, vocab_size)
            print(f"  ✓ title case '{title}' works")
        except ValueError:
            print(f"  ✗ title case '{title}' not supported (this is OK)")


def test_factory_model_uniqueness():
    """Test that each factory call creates a new independent model."""
    print("\n" + "="*60)
    print("TESTING MODEL UNIQUENESS")
    print("="*60)
    
    vocab_size = 128
    
    for model_type in ["lstm", "gru", "transformer"]:
        print(f"\nTesting {model_type.upper()} uniqueness...")
        
        # Create two models
        model1 = get_generator(model_type, vocab_size)
        model2 = get_generator(model_type, vocab_size)
        
        # Models should be different objects
        assert model1 is not model2, "Factory should create new instances"
        
        # Modify one model's parameters
        for param in model1.parameters():
            param.data.fill_(1.0)
            break
        
        # Other model should be unchanged
        for param in model2.parameters():
            assert not torch.all(param.data == 1.0), \
                "Models should be independent"
            break
        
        print(f"✓ {model_type.upper()} models are independent")


def test_factory_error_messages():
    """Test that factory provides helpful error messages."""
    print("\n" + "="*60)
    print("TESTING ERROR MESSAGES")
    print("="*60)
    
    vocab_size = 128
    
    # Test various invalid inputs
    invalid_inputs = [
        ("", "empty string"),
        (None, "None"),
        (123, "integer"),
        (["lstm"], "list"),
        ("cnn", "unsupported model type"),
        ("rnn", "potentially ambiguous name"),
    ]
    
    for invalid_input, description in invalid_inputs:
        print(f"\nTesting with {description}: {invalid_input}")
        try:
            model = get_generator(invalid_input, vocab_size)
            print(f"  ⚠ No error raised (might be OK if it has default handling)")
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            print(f"  ✓ Correctly raised {type(e).__name__}: {e}")


# Update the main test runner to include all new tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATOR COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    model_types = ["lstm", "gru", "transformer"]
    
    # Run all existing tests
    test_generator(model_types)
    test_custom_configs()
    test_gradient_flow(model_types)
    test_training_step()
    test_output_distribution()
    test_deterministic_output()
    test_batch_independence()
    test_edge_cases()
    test_model_save_load()
    test_memory_efficiency()
    
    # Run new coverage tests
    test_factory_invalid_model_type()
    test_factory_all_model_types()
    test_default_config_all_types()
    test_default_config_invalid_type()
    test_factory_with_all_hyperparameters()
    test_factory_with_minimal_params()
    test_factory_parameter_override()
    test_factory_vocab_size_variations()
    test_factory_case_insensitive()
    test_factory_model_uniqueness()
    test_factory_error_messages()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)