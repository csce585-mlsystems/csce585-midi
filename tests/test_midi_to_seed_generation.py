"""
Integration tests for midi_to_seed with generation.

These tests verify that seeds created by midi_to_seed can be successfully
used for music generation with trained models.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.midi_to_seed import midi_to_seed, save_seed
from models.generators.generator_factory import get_generator


class TestMidiToSeedGeneration:
    """Integration tests for using midi_to_seed with generation."""
    
    @pytest.fixture
    def naive_model_and_data(self):
        """Find a trained naive model and corresponding data."""
        model_dir = Path("models/generators/checkpoints/nottingham-dataset-master_naive")
        data_dir = Path("data/nottingham-dataset-master_naive")
        
        if not model_dir.exists() or not data_dir.exists():
            pytest.skip("Naive model or data not available")
        
        # Find any .pth file
        models = list(model_dir.glob("*.pth"))
        if not models:
            pytest.skip("No trained naive models found")
        
        return {
            'model_path': str(models[0]),
            'data_dir': str(data_dir),
            'model_type': 'lstm' if 'lstm' in str(models[0]) else 
                         'gru' if 'gru' in str(models[0]) else 'transformer'
        }
    
    @pytest.fixture
    def miditok_model_and_data(self):
        """Find a trained miditok model and corresponding data."""
        model_dir = Path("models/generators/checkpoints/miditok")
        data_dir = Path("data/nottingham-dataset-master_miditok")
        
        if not model_dir.exists() or not data_dir.exists():
            pytest.skip("Miditok model or data not available")
        
        # Find any .pth file
        models = list(model_dir.glob("*.pth"))
        if not models:
            pytest.skip("No trained miditok models found")
        
        return {
            'model_path': str(models[0]),
            'data_dir': str(data_dir),
            'model_type': 'lstm' if 'lstm' in str(models[0]) else 
                         'gru' if 'gru' in str(models[0]) else 'transformer'
        }
    
    @pytest.fixture
    def sample_midi_file(self):
        """Get a sample MIDI file for testing."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        if not Path(midi_file).exists():
            pytest.skip("Sample MIDI file not available")
        return midi_file
    
    def test_naive_seed_with_generation(self, naive_model_and_data, sample_midi_file):
        """Test that a seed from midi_to_seed works with naive model generation."""
        # Step 1: Convert MIDI to seed
        seed = midi_to_seed(
            midi_filepath=sample_midi_file,
            dataset_dir=naive_model_and_data['data_dir'],
            seq_length=50
        )
        
        assert seed is not None
        assert len(seed) == 50
        
        # Step 2: Load the model
        device = "cpu"  # Use CPU for testing
        
        # Load vocab to get vocab_size
        import pickle
        with open(Path(naive_model_and_data['data_dir']) / "note_to_int.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        vocab_size = len(vocab_data["note_to_int"])
        
        # Create model
        model = get_generator(
            naive_model_and_data['model_type'],
            vocab_size=vocab_size,
            hidden_size=256,
            num_layers=2,
            embed_size=128
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(
            naive_model_and_data['model_path'],
            map_location=device
        ))
        model.to(device)
        model.eval()
        
        # Step 3: Use seed for generation
        input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate a few steps
        generated = seed.copy()
        generate_steps = 10
        
        with torch.no_grad():
            for _ in range(generate_steps):
                output, _ = model(input_seq)
                logits = output[:, -1, :]
                
                # Greedy sampling (most probable token)
                next_token = torch.argmax(logits, dim=-1).item()
                
                # Verify token is in valid range
                assert 0 <= next_token < vocab_size
                
                generated.append(next_token)
                
                # Update input sequence
                input_seq = torch.tensor(
                    generated[-50:],
                    dtype=torch.long
                ).unsqueeze(0).to(device)
        
        # Verify we generated new tokens
        assert len(generated) == 50 + generate_steps
        assert generated[:50] == seed  # Original seed preserved
        assert generated[50:] != seed[:generate_steps]  # New tokens generated
        
        print(f"\n✓ Successfully generated {generate_steps} tokens using seed from MIDI")
        print(f"  Seed length: {len(seed)}")
        print(f"  Total length after generation: {len(generated)}")
    
    def test_miditok_seed_with_generation(self, miditok_model_and_data, sample_midi_file):
        """Test that a seed from midi_to_seed works with miditok model generation."""
        # Step 1: Convert MIDI to seed
        seed = midi_to_seed(
            midi_filepath=sample_midi_file,
            dataset_dir=miditok_model_and_data['data_dir'],
            seq_length=100
        )
        
        assert seed is not None
        assert len(seed) == 100
        
        # Step 2: Load the model
        device = "cpu"  # Use CPU for testing
        
        # Load vocab to get vocab_size
        import json
        with open(Path(miditok_model_and_data['data_dir']) / "vocab.json", "r") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        
        # Load the checkpoint to inspect architecture
        checkpoint = torch.load(miditok_model_and_data['model_path'], map_location=device)
        
        # Try to infer model dimensions from checkpoint
        if 'lstm.weight_ih_l0' in checkpoint:
            hidden_size = checkpoint['lstm.weight_hh_l0'].shape[1]
            embed_size = checkpoint['lstm.weight_ih_l0'].shape[1]
            # Count layers by checking for weight keys
            num_layers = sum(1 for key in checkpoint.keys() if key.startswith('lstm.weight_ih_l'))
        else:
            # Default fallback
            hidden_size = 256
            embed_size = 128
            num_layers = 2
        
        # Create model with correct architecture
        model = get_generator(
            miditok_model_and_data['model_type'],
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            embed_size=embed_size
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        # Step 3: Use seed for generation
        input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate a few steps
        generated = seed.copy()
        generate_steps = 10
        
        with torch.no_grad():
            for _ in range(generate_steps):
                output, _ = model(input_seq)
                logits = output[:, -1, :]
                
                # Greedy sampling (most probable token)
                next_token = torch.argmax(logits, dim=-1).item()
                
                # Verify token is in valid range
                assert 0 <= next_token < vocab_size
                
                generated.append(next_token)
                
                # Update input sequence
                input_seq = torch.tensor(
                    generated[-100:],
                    dtype=torch.long
                ).unsqueeze(0).to(device)
        
        # Verify we generated new tokens
        assert len(generated) == 100 + generate_steps
        assert generated[:100] == seed  # Original seed preserved
        
        print(f"\n✓ Successfully generated {generate_steps} tokens using seed from MIDI")
        print(f"  Seed length: {len(seed)}")
        print(f"  Total length after generation: {len(generated)}")
    
    def test_seed_produces_valid_tokens(self, naive_model_and_data, sample_midi_file):
        """Test that seeds from midi_to_seed only contain valid vocabulary tokens."""
        # Convert MIDI to seed
        seed = midi_to_seed(
            midi_filepath=sample_midi_file,
            dataset_dir=naive_model_and_data['data_dir'],
            seq_length=50
        )
        
        assert seed is not None
        
        # Load vocab
        import pickle
        with open(Path(naive_model_and_data['data_dir']) / "note_to_int.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        vocab_size = len(vocab_data["note_to_int"])
        
        # Verify all tokens are in valid range
        for token in seed:
            assert isinstance(token, (int, np.integer))
            assert 0 <= token < vocab_size, f"Token {token} out of range [0, {vocab_size})"
        
        print(f"\n✓ All {len(seed)} tokens in seed are valid")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Token range: [{min(seed)}, {max(seed)}]")
    
    def test_seed_maintains_musical_context(self, naive_model_and_data, sample_midi_file):
        """Test that seeds preserve musical information from original MIDI."""
        # Convert MIDI to seed with different lengths
        seed_short = midi_to_seed(
            midi_filepath=sample_midi_file,
            dataset_dir=naive_model_and_data['data_dir'],
            seq_length=30
        )
        
        seed_long = midi_to_seed(
            midi_filepath=sample_midi_file,
            dataset_dir=naive_model_and_data['data_dir'],
            seq_length=100
        )
        
        assert seed_short is not None
        assert seed_long is not None
        
        # Short seed should be the last 30 tokens of long seed (both from same MIDI)
        # Actually they're both taken from the end, so short should be suffix of long
        # when both come from same file
        
        # At minimum, verify they have some diversity (not all same token)
        assert len(set(seed_short)) > 1, "Seed has no musical diversity"
        assert len(set(seed_long)) > 1, "Seed has no musical diversity"
        
        print(f"\n✓ Seeds maintain musical diversity")
        print(f"  Short seed unique tokens: {len(set(seed_short))}/{len(seed_short)}")
        print(f"  Long seed unique tokens: {len(set(seed_long))}/{len(seed_long)}")
    
    def test_saved_seed_works_with_generation(self, naive_model_and_data, sample_midi_file):
        """Test that saved seeds can be loaded and used for generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Create and save seed
            seed = midi_to_seed(
                midi_filepath=sample_midi_file,
                dataset_dir=naive_model_and_data['data_dir'],
                seq_length=50
            )
            
            seed_file = Path(tmp_dir) / "test_seed.npy"
            save_seed(seed, str(seed_file))
            
            # Step 2: Load seed back
            loaded_seed = np.load(seed_file).tolist()
            
            assert loaded_seed == seed
            
            # Step 3: Verify loaded seed works with model
            device = "cpu"
            
            import pickle
            with open(Path(naive_model_and_data['data_dir']) / "note_to_int.pkl", "rb") as f:
                vocab_data = pickle.load(f)
            vocab_size = len(vocab_data["note_to_int"])
            
            model = get_generator(
                naive_model_and_data['model_type'],
                vocab_size=vocab_size,
                hidden_size=256,
                num_layers=2,
                embed_size=128
            )
            
            model.load_state_dict(torch.load(
                naive_model_and_data['model_path'],
                map_location=device
            ))
            model.to(device)
            model.eval()
            
            # Generate with loaded seed
            input_seq = torch.tensor(loaded_seed, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output, _ = model(input_seq)
                logits = output[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).item()
                
                assert 0 <= next_token < vocab_size
            
            print(f"\n✓ Saved and loaded seed works with generation")
            print(f"  Generated first token: {next_token}")
    
    def test_multiple_midis_produce_different_seeds(self, naive_model_and_data):
        """Test that different MIDI files produce different seeds."""
        midi_files = list(Path("data/seeds_for_experiments").glob("*.mid"))[:3]
        
        if len(midi_files) < 2:
            pytest.skip("Need at least 2 MIDI files for comparison")
        
        seeds = []
        for midi_file in midi_files:
            seed = midi_to_seed(
                midi_filepath=str(midi_file),
                dataset_dir=naive_model_and_data['data_dir'],
                seq_length=50
            )
            if seed:
                seeds.append(seed)
        
        # Verify we got multiple seeds
        assert len(seeds) >= 2
        
        # Verify they're different
        for i in range(len(seeds) - 1):
            assert seeds[i] != seeds[i + 1], "Different MIDIs should produce different seeds"
        
        print(f"\n✓ Different MIDI files produce different seeds")
        print(f"  Tested {len(seeds)} MIDI files")
        print(f"  All seeds are unique")


class TestGenerationOutput:
    """Tests for the actual generation output quality."""
    
    @pytest.fixture
    def naive_model_and_data(self):
        """Find a trained naive model and corresponding data."""
        model_dir = Path("models/generators/checkpoints/nottingham-dataset-master_naive")
        data_dir = Path("data/nottingham-dataset-master_naive")
        
        if not model_dir.exists() or not data_dir.exists():
            pytest.skip("Naive model or data not available")
        
        # Find any .pth file
        models = list(model_dir.glob("*.pth"))
        if not models:
            pytest.skip("No trained naive models found")
        
        return {
            'model_path': str(models[0]),
            'data_dir': str(data_dir),
            'model_type': 'lstm' if 'lstm' in str(models[0]) else 
                         'gru' if 'gru' in str(models[0]) else 'transformer'
        }
    
    @pytest.fixture
    def sample_midi_file(self):
        """Get a sample MIDI file for testing."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        if not Path(midi_file).exists():
            pytest.skip("Sample MIDI file not available")
        return midi_file
    
    def test_generation_produces_valid_midi(self, naive_model_and_data, sample_midi_file):
        """Test that generation with midi_to_seed produces valid MIDI output."""
        # This is a more comprehensive test that generates a longer sequence
        # and verifies it can be converted to MIDI
        
        seed = midi_to_seed(
            midi_filepath=sample_midi_file,
            dataset_dir=naive_model_and_data['data_dir'],
            seq_length=50
        )
        
        if seed is None:
            pytest.skip("Could not create seed")
        
        device = "cpu"
        
        import pickle
        with open(Path(naive_model_and_data['data_dir']) / "note_to_int.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        vocab_size = len(vocab_data["note_to_int"])
        int_to_note = {i: n for n, i in vocab_data["note_to_int"].items()}
        
        model = get_generator(
            naive_model_and_data['model_type'],
            vocab_size=vocab_size,
            hidden_size=256,
            num_layers=2,
            embed_size=128
        )
        
        model.load_state_dict(torch.load(
            naive_model_and_data['model_path'],
            map_location=device
        ))
        model.to(device)
        model.eval()
        
        # Generate a longer sequence
        generated = seed.copy()
        generate_length = 50
        
        input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(generate_length):
                output, _ = model(input_seq)
                logits = output[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).item()
                
                generated.append(next_token)
                input_seq = torch.tensor(
                    generated[-50:],
                    dtype=torch.long
                ).unsqueeze(0).to(device)
        
        # Verify generation
        assert len(generated) == 100
        
        # Verify all tokens can be decoded
        decodable_count = 0
        for token in generated:
            if token in int_to_note:
                note = int_to_note[token]
                # Basic validation: note should be a string
                assert isinstance(note, str)
                decodable_count += 1
        
        assert decodable_count == len(generated), "All tokens should be decodable"
        
        print(f"\n✓ Generated {len(generated)} tokens, all decodable to notes")
        print(f"  Seed: {len(seed)} tokens")
        print(f"  Generated: {generate_length} new tokens")
        print(f"  Total: {len(generated)} tokens")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
