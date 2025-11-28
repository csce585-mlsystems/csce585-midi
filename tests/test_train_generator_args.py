"""
Comprehensive tests for train_generator.py MIDIDataset class
Tests various configurations and edge cases similar to test_generate_args.py
"""
import unittest
from unittest.mock import MagicMock, patch, mock_open
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_generator import MIDIDataset, log_experiment


class TestMIDIDatasetConfigurations(unittest.TestCase):
    """Test MIDIDataset with various configurations"""
    
    def test_dataset_with_different_seq_lengths(self):
        """Test dataset initialization with various sequence lengths"""
        sequences = np.random.randint(0, 100, size=(50, 200))
        
        seq_lengths = [10, 50, 100, 150]
        for seq_len in seq_lengths:
            dataset = MIDIDataset(sequences, seq_length=seq_len, subsample_ratio=1.0)
            self.assertGreater(len(dataset), 0)
            
            # Test __getitem__
            input_seq, target = dataset[0]
            self.assertEqual(input_seq.shape[0], seq_len)
            self.assertEqual(target.shape[0], seq_len)
            
    def test_dataset_with_different_subsample_ratios(self):
        """Test dataset with various subsampling ratios"""
        sequences = np.random.randint(0, 100, size=(100, 200))
        
        subsample_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
        lengths = []
        
        for ratio in subsample_ratios:
            dataset = MIDIDataset(sequences, seq_length=50, subsample_ratio=ratio)
            lengths.append(len(dataset))
            
        # Lengths should generally increase with ratio
        # (though not strictly monotonic due to discretization)
        self.assertLess(lengths[0], lengths[-1])
        
    def test_dataset_with_small_batch_compatibility(self):
        """Test that dataset works with small sequences"""
        sequences = np.random.randint(0, 100, size=(10, 100))
        dataset = MIDIDataset(sequences, seq_length=50, subsample_ratio=1.0)
        
        # Should create valid dataset
        self.assertGreater(len(dataset), 0)
        
    def test_dataset_with_large_sequences(self):
        """Test dataset with very long sequences"""
        sequences = np.random.randint(0, 100, size=(20, 1000))
        dataset = MIDIDataset(sequences, seq_length=50, subsample_ratio=1.0)
        
        # Should handle long sequences
        self.assertGreater(len(dataset), 100)
        
    def test_dataset_deterministic_sampling(self):
        """Test that same parameters give same dataset"""
        sequences = np.random.randint(0, 100, size=(50, 200))
        
        dataset1 = MIDIDataset(sequences, seq_length=50, subsample_ratio=0.5)
        dataset2 = MIDIDataset(sequences, seq_length=50, subsample_ratio=0.5)
        
        self.assertEqual(len(dataset1), len(dataset2))
        
    def test_dataset_target_correctness(self):
        """Test that targets are correctly offset from inputs"""
        # Create simple sequences we can verify
        sequences = np.arange(0, 300).reshape(3, 100)
        dataset = MIDIDataset(sequences, seq_length=10, subsample_ratio=1.0)
        
        input_seq, target = dataset[0]
        
        # Target should be input shifted by 1
        for i in range(len(input_seq)):
            self.assertEqual(target[i].item(), input_seq[i].item() + 1)
            
    def test_dataset_fast_path_vs_slow_path(self):
        """Test that 2D array (fast) and ragged array (slow) give similar results"""
        # Fast path: 2D array
        sequences_2d = np.random.randint(0, 100, size=(20, 150))
        dataset_fast = MIDIDataset(sequences_2d, seq_length=50, subsample_ratio=1.0)
        
        # Slow path: List of arrays
        sequences_list = [np.random.randint(0, 100, size=150) for _ in range(20)]
        dataset_slow = MIDIDataset(sequences_list, seq_length=50, subsample_ratio=1.0)
        
        # Both should create valid datasets
        self.assertGreater(len(dataset_fast), 0)
        self.assertGreater(len(dataset_slow), 0)
        
        # Fast path should be identified correctly
        self.assertTrue(dataset_fast.is_2d_array)
        self.assertFalse(dataset_slow.is_2d_array)


class TestMIDIDatasetEdgeCases(unittest.TestCase):
    """Test edge cases for MIDIDataset"""
    
    def test_dataset_with_short_sequences(self):
        """Test dataset with sequences shorter than seq_length"""
        short_sequences = [np.array([1, 2, 3]) for _ in range(10)]
        dataset = MIDIDataset(short_sequences, seq_length=50, subsample_ratio=1.0)
        
        # Should have length 0 or handle gracefully
        self.assertEqual(len(dataset), 0)
        
    def test_dataset_with_2d_array_fast_path(self):
        """Test that 2D numpy arrays use fast path"""
        # Create 2D array with sufficient length
        sequences_2d = np.random.randint(0, 100, size=(50, 200))
        dataset = MIDIDataset(sequences_2d, seq_length=50, subsample_ratio=1.0)
        
        # Should use fast path
        self.assertTrue(dataset.is_2d_array)
        self.assertGreater(len(dataset), 0)
        
        # Test __getitem__
        input_seq, target = dataset[0]
        self.assertEqual(input_seq.shape[0], 50)
        self.assertEqual(target.shape[0], 50)
        
    def test_dataset_with_ragged_array_slow_path(self):
        """Test that ragged arrays use slow path"""
        # Create ragged array (different lengths)
        sequences_ragged = [
            np.random.randint(0, 100, size=100),
            np.random.randint(0, 100, size=150),
            np.random.randint(0, 100, size=200),
        ]
        dataset = MIDIDataset(sequences_ragged, seq_length=50, subsample_ratio=1.0)
        
        # Should use slow path
        self.assertFalse(dataset.is_2d_array)
        self.assertGreater(len(dataset), 0)
        
    def test_dataset_safety_check_large_dataset(self):
        """Test that very large datasets trigger safety check"""
        # Create a dataset that would exceed 50M samples
        # With 10000 sequences of length 10000, and seq_length=50
        # samples_per_row = 10000 - 50 = 9950
        # total = 10000 * 9950 = 99.5M samples (exceeds 50M)
        large_sequences = np.random.randint(0, 100, size=(10000, 10000))
        
        dataset = MIDIDataset(large_sequences, seq_length=50, subsample_ratio=1.0)
        
        # Should auto-adjust step size
        self.assertLessEqual(dataset.total_samples, 50_000_000)
        
    def test_dataset_invalid_subsample_ratio(self):
        """Test that invalid subsample ratio raises error"""
        sequences = np.random.randint(0, 100, size=(10, 100))
        
        with self.assertRaises(ValueError):
            MIDIDataset(sequences, seq_length=50, subsample_ratio=0.0)
            
        with self.assertRaises(ValueError):
            MIDIDataset(sequences, seq_length=50, subsample_ratio=-0.1)
            
    def test_dataset_with_varying_row_lengths(self):
        """Test dataset handles varying sequence lengths in slow path"""
        sequences = [
            np.random.randint(0, 100, size=60),
            np.random.randint(0, 100, size=80),
            np.random.randint(0, 100, size=120),
            np.random.randint(0, 100, size=200),
        ]
        
        dataset = MIDIDataset(sequences, seq_length=50, subsample_ratio=1.0)
        self.assertGreater(len(dataset), 0)
        
        # All items should have correct shape
        for i in range(min(len(dataset), 10)):  # Test first 10
            input_seq, target = dataset[i]
            self.assertEqual(input_seq.shape[0], 50)
            self.assertEqual(target.shape[0], 50)


class TestLogExperiment(unittest.TestCase):
    """Test experiment logging functionality"""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('training.train_generator.Path.mkdir')
    @patch('os.path.isfile')
    def test_log_experiment_new_file(self, mock_isfile, mock_mkdir, mock_file):
        """Test logging to a new file (creates header)"""
        mock_isfile.return_value = False
        
        hparams = {"model_type": "lstm", "epochs": 10}
        results = {"final_loss": 0.5, "train_time_sec": 100}
        logfile = "test_log.csv"
        
        log_experiment(hparams, results, logfile)
        
        # Check that file was opened in append mode
        mock_file.assert_called_with(logfile, "a")
        
    @patch('builtins.open', new_callable=mock_open)
    @patch('training.train_generator.Path.mkdir')
    @patch('os.path.isfile')
    def test_log_experiment_existing_file(self, mock_isfile, mock_mkdir, mock_file):
        """Test logging to an existing file (no header)"""
        mock_isfile.return_value = True
        
        hparams = {"model_type": "gru", "epochs": 5}
        results = {"final_loss": 0.3, "train_time_sec": 50}
        logfile = "test_log.csv"
        
        log_experiment(hparams, results, logfile)
        
        # Should still open file
        mock_file.assert_called()
        
    @patch('builtins.open', new_callable=mock_open)
    @patch('training.train_generator.Path.mkdir')
    @patch('os.path.isfile')
    def test_log_experiment_with_various_params(self, mock_isfile, mock_mkdir, mock_file):
        """Test logging with different parameter combinations"""
        mock_isfile.return_value = False
        
        test_cases = [
            {
                "hparams": {"model_type": "lstm", "hidden_size": 256, "num_layers": 2},
                "results": {"final_loss": 0.5, "train_time_sec": 100}
            },
            {
                "hparams": {"model_type": "transformer", "d_model": 512, "nhead": 8},
                "results": {"final_loss": 0.3, "best_val_loss": 0.35}
            },
            {
                "hparams": {"model_type": "gru", "dropout": 0.4, "learning_rate": 0.001},
                "results": {"min_loss": 0.2, "max_loss": 1.5, "loss_std": 0.3}
            },
        ]
        
        for case in test_cases:
            log_experiment(case["hparams"], case["results"], "test_log.csv")
        
        # Should be called once per test case
        self.assertEqual(mock_file.call_count, len(test_cases))


class TestMIDIDatasetBatchingCompatibility(unittest.TestCase):
    """Test that MIDIDataset works well with DataLoader"""
    
    def test_dataset_with_dataloader(self):
        """Test dataset can be used with PyTorch DataLoader"""
        from torch.utils.data import DataLoader
        
        sequences = np.random.randint(0, 100, size=(50, 200))
        dataset = MIDIDataset(sequences, seq_length=50, subsample_ratio=1.0)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Get one batch
        batch_input, batch_target = next(iter(dataloader))
        
        # Check batch dimensions
        self.assertEqual(batch_input.shape[0], 16)  # batch size
        self.assertEqual(batch_input.shape[1], 50)  # seq length
        self.assertEqual(batch_target.shape[0], 16)
        self.assertEqual(batch_target.shape[1], 50)
        
    def test_dataset_with_different_batch_sizes(self):
        """Test dataset with various batch sizes"""
        from torch.utils.data import DataLoader
        
        sequences = np.random.randint(0, 100, size=(100, 200))
        dataset = MIDIDataset(sequences, seq_length=50, subsample_ratio=1.0)
        
        batch_sizes = [8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            batch_input, batch_target = next(iter(dataloader))
            
            self.assertEqual(batch_input.shape[0], batch_size)
            self.assertEqual(batch_input.shape[1], 50)


if __name__ == "__main__":
    unittest.main()
