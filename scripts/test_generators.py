"""
Test script to verify all generator architectures work correctly.
Tests LSTM, GRU, and Transformer generators with dummy data.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.generators.generator_factory import get_generator, get_default_config


def test_generator(model_type, vocab_size=256, batch_size=4, seq_len=50):
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
    
    return model


def test_all_generators():
    """Test all generator architectures."""
    print("\n" + "="*60)
    print("GENERATOR ARCHITECTURE TESTS")
    print("="*60)
    
    vocab_size = 256
    results = {}
    
    for model_type in ["lstm", "gru", "transformer"]:
        try:
            model = test_generator(model_type, vocab_size=vocab_size)
            results[model_type] = "✓ PASSED"
        except Exception as e:
            results[model_type] = f"✗ FAILED: {str(e)}"
            print(f"✗ {model_type.upper()} failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for model_type, status in results.items():
        print(f"{model_type.upper():15} {status}")
    print(f"{'='*60}\n")
    
    # Check if all passed
    all_passed = all("PASSED" in status for status in results.values())
    if all_passed:
        print("✓ All generator architectures working correctly!")
    else:
        print("✗ Some tests failed. Check output above.")
    
    return all_passed


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


if __name__ == "__main__":
    # Test all architectures
    all_passed = test_all_generators()
    
    # Test custom configs
    test_custom_configs()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
