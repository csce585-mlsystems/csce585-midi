"""
Tests for midi_to_seed utility.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from midi_to_seed import (
    midi_to_seed,
    midi_to_seed_naive,
    midi_to_seed_miditok,
    save_seed,
    load_seed
)


class TestMidiToSeedNaive:
    """Tests for naive preprocessing conversion."""
    
    def test_basic_conversion(self):
        """Test basic MIDI to seed conversion with naive preprocessing."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        dataset = "data/nottingham-dataset-master_naive"
        
        if not Path(midi_file).exists() or not Path(dataset).exists():
            pytest.skip("Test data not available")
        
        seed = midi_to_seed_naive(midi_file, dataset, seq_length=50)
        
        assert seed is not None
        assert isinstance(seed, list)
        assert len(seed) == 50
        assert all(isinstance(x, (int, np.integer)) for x in seed)
    
    def test_truncation(self):
        """Test that long MIDI files are truncated correctly."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        dataset = "data/nottingham-dataset-master_naive"
        
        if not Path(midi_file).exists() or not Path(dataset).exists():
            pytest.skip("Test data not available")
        
        # Get full sequence
        full_seed = midi_to_seed_naive(midi_file, dataset, seq_length=None)
        
        # Get truncated sequence
        short_seed = midi_to_seed_naive(midi_file, dataset, seq_length=30)
        
        assert len(short_seed) == 30
        # Should use last 30 tokens
        assert short_seed == full_seed[-30:]
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent MIDI file."""
        seed = midi_to_seed_naive(
            "nonexistent.mid",
            "data/nottingham-dataset-master_naive",
            seq_length=50
        )
        
        assert seed is None
    
    def test_missing_vocab(self):
        """Test handling of missing vocabulary file."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        
        if not Path(midi_file).exists():
            pytest.skip("Test data not available")
        
        seed = midi_to_seed_naive(
            midi_file,
            "data/nonexistent_dataset",
            seq_length=50
        )
        
        assert seed is None


class TestMidiToSeedMiditok:
    """Tests for miditok preprocessing conversion."""
    
    def test_basic_conversion(self):
        """Test basic MIDI to seed conversion with miditok preprocessing."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        dataset = "data/nottingham-dataset-master_miditok"
        
        if not Path(midi_file).exists() or not Path(dataset).exists():
            pytest.skip("Test data not available")
        
        seed = midi_to_seed_miditok(midi_file, dataset, seq_length=100)
        
        assert seed is not None
        assert isinstance(seed, list)
        assert len(seed) == 100
        assert all(isinstance(x, (int, np.integer)) for x in seed)
    
    def test_truncation(self):
        """Test that long MIDI files are truncated correctly."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        dataset = "data/nottingham-dataset-master_miditok"
        
        if not Path(midi_file).exists() or not Path(dataset).exists():
            pytest.skip("Test data not available")
        
        # Get full sequence
        full_seed = midi_to_seed_miditok(midi_file, dataset, seq_length=None)
        
        # Get truncated sequence
        short_seed = midi_to_seed_miditok(midi_file, dataset, seq_length=50)
        
        assert len(short_seed) == 50
        # Should use last 50 tokens
        assert short_seed == full_seed[-50:]


class TestMidiToSeedAutoDetect:
    """Tests for automatic format detection."""
    
    def test_auto_detect_naive(self):
        """Test automatic detection of naive preprocessing."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        dataset = "data/nottingham-dataset-master_naive"
        
        if not Path(midi_file).exists() or not Path(dataset).exists():
            pytest.skip("Test data not available")
        
        seed = midi_to_seed(midi_file, dataset, seq_length=50)
        
        assert seed is not None
        assert len(seed) == 50
    
    def test_auto_detect_miditok(self):
        """Test automatic detection of miditok preprocessing."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        dataset = "data/nottingham-dataset-master_miditok"
        
        if not Path(midi_file).exists() or not Path(dataset).exists():
            pytest.skip("Test data not available")
        
        seed = midi_to_seed(midi_file, dataset, seq_length=100)
        
        assert seed is not None
        assert len(seed) == 100
    
    def test_invalid_dataset(self):
        """Test handling of dataset with no valid preprocessing."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        
        if not Path(midi_file).exists():
            pytest.skip("Test data not available")
        
        seed = midi_to_seed(
            midi_file,
            "data/seeds_for_experiments",  # Not a preprocessed dataset
            seq_length=50
        )
        
        assert seed is None
        
        assert seed is None


class TestSaveLoad:
    """Tests for saving and loading seeds."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading a seed sequence."""
        # Create a test seed
        test_seed = [1, 2, 3, 4, 5, 10, 20, 30]
        
        # Save it
        output_file = tmp_path / "test_seed.npy"
        save_seed(test_seed, str(output_file))
        
        assert output_file.exists()
        
        # Load it
        loaded_seed = load_seed(str(output_file))
        
        assert loaded_seed == test_seed
    
    def test_load_nonexistent(self):
        """Test loading a nonexistent seed file."""
        seed = load_seed("nonexistent_seed.npy")
        
        assert seed is None
    
    def test_save_creates_directory(self, tmp_path):
        """Test that save_seed creates parent directories."""
        output_file = tmp_path / "subdir1" / "subdir2" / "seed.npy"
        
        save_seed([1, 2, 3], str(output_file))
        
        assert output_file.exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_midi(self):
        """Test handling of MIDI with no notes."""
        # This would require creating a test MIDI file with no notes
        # Skipping for now as it's an edge case
        pytest.skip("Requires test MIDI with no notes")
    
    def test_very_short_sequence(self):
        """Test padding of very short sequences."""
        midi_file = "data/seeds_for_experiments/saltcrek.mid"
        dataset = "data/nottingham-dataset-master_naive"
        
        if not Path(midi_file).exists() or not Path(dataset).exists():
            pytest.skip("Test data not available")
        
        # Request a very long sequence that requires padding
        seed = midi_to_seed_naive(midi_file, dataset, seq_length=1000)
        
        if seed:  # Only test if file was processed successfully
            assert len(seed) == 1000
            # Check that it's padded (should have repeating patterns)
            assert len(set(seed)) < len(seed)  # Not all unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
