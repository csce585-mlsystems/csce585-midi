"""
Comprehensive tests for train_generator.py
Tests the MIDIDataset, training loop, logging, and various configurations.
"""
import pytest
import numpy as np
import torch
import json
import pickle
import tempfile
import shutil
from pathlib import Path
import sys
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train_generator import MIDIDataset, train, log_experiment


class TestMIDIDataset:
    """Test suite for MIDIDataset class"""
    
    def test_dataset_initialization(self, sample_sequences):
        """Test that dataset initializes correctly"""
        dataset = MIDIDataset(sample_sequences, seq_length=50, subsample_ratio=1.0)
        
        # Check that dataset has correct number of sequences
        assert len(dataset.sequences) > 0
        assert all(len(seq) >= 50 for seq in dataset.sequences)
        
    def test_dataset_length(self, sample_sequences):
        """Test dataset length calculation"""
        seq_length = 50
        dataset = MIDIDataset(sample_sequences, seq_length=seq_length, subsample_ratio=1.0)
        
        # Calculate expected length
        expected_samples = sum(len(seq) - seq_length for seq in sample_sequences if len(seq) >= seq_length)
        assert len(dataset) == expected_samples
        
    def test_dataset_getitem(self, sample_sequences):
        """Test that __getitem__ returns correct shapes"""
        seq_length = 50
        dataset = MIDIDataset(sample_sequences, seq_length=seq_length, subsample_ratio=1.0)
        
        # Get first item
        input_seq, target = dataset[0]
        
        # Check shapes
        assert input_seq.shape == (seq_length,)
        assert target.shape == (seq_length,)
        
        # Check types
        assert input_seq.dtype == torch.long
        assert target.dtype == torch.long
        
    def test_dataset_subsampling(self, sample_sequences):
        """Test that subsampling reduces dataset size"""
        seq_length = 50
        
        # Subsampled dataset
        # Use a ratio that guarantees at least one sample remains
        # With 250 samples, 0.1 ratio -> step=10 -> 25 samples
        # NOTE: For ragged lists, we only take samples if seq_idx % step == 0
        # Since we only have 1 sequence (index 0), and 0 % anything is 0, it should be included.
        # However, the loop logic might be skipping it if not careful.
        # Let's use a slightly larger ratio to be safe, or ensure we have multiple sequences.
        
        # Create multiple sequences to test subsampling properly
        # Ensure they are long enough (e.g. 200) so that samples_per_row > 0
        long_seq = np.random.randint(0, 100, size=200)
        multi_seqs = np.array([long_seq for _ in range(20)], dtype=object)
        
        # Convert to 2D array to trigger fast path if possible, or keep as object array
        # The test fixture provides object array, so let's stick to that or convert if needed
        # But wait, if we use object array, it goes to slow path.
        # If we want to test fast path, we need 2D int array.
        # The previous failure showed "dataset initialized (fast mode)" which means it WAS treated as 2D array
        # because all rows had same length (50) but that length was == seq_length (50)
        # so samples_per_row = 50 - 50 = 0.
        
        # Let's make sure we have a 2D array with sufficient length
        multi_seqs_2d = np.vstack([long_seq for _ in range(20)])
        
        full_dataset = MIDIDataset(multi_seqs_2d, seq_length=seq_length, subsample_ratio=1.0)
        full_len = len(full_dataset)

        subsampled_dataset = MIDIDataset(multi_seqs_2d, seq_length=seq_length, subsample_ratio=0.1)
        subsampled_len = len(subsampled_dataset)
        
        # Subsampled should be significantly smaller
        assert subsampled_len < full_len
        assert subsampled_len > 0  # But not empty
        
    def test_dataset_target_offset(self, sample_sequences):
        """Test that target is offset by one position from input"""
        seq_length = 10
        dataset = MIDIDataset(sample_sequences, seq_length=seq_length, subsample_ratio=1.0)
        
        # Get first item
        input_seq, target = dataset[0]
        
        # Get the actual sequence from dataset
        seq_idx, start_idx = dataset.sequence_indices[0]
        original_seq = dataset.sequences[seq_idx]
        
        # Check that target is shifted by one
        expected_input = original_seq[start_idx:start_idx + seq_length]
        expected_target = original_seq[start_idx + 1:start_idx + seq_length + 1]
        
        assert torch.equal(input_seq, torch.tensor(expected_input, dtype=torch.long))
        assert torch.equal(target, torch.tensor(expected_target, dtype=torch.long))
        
    def test_dataset_filters_short_sequences(self):
        """Test that sequences shorter than seq_length are filtered out"""
        # Create sequences with varying lengths
        sequences = [
            np.random.randint(0, 100, size=10),   # Too short
            np.random.randint(0, 100, size=30),   # Too short
            np.random.randint(0, 100, size=100),  # Long enough
            np.random.randint(0, 100, size=150),  # Long enough
        ]
        sequences = np.array(sequences, dtype=object)
        
        seq_length = 50
        dataset = MIDIDataset(sequences, seq_length=seq_length, subsample_ratio=1.0)
        
        # Should only have 2 sequences (the ones >= 50)
        # Note: dataset.sequences stores the original array, but sequence_indices stores valid samples
        # We check how many unique sequence indices are present in self.sequence_indices
        unique_seq_indices = set(idx for idx, _ in dataset.sequence_indices)
        assert len(unique_seq_indices) == 2
        
    def test_dataset_invalid_subsample_ratio(self):
        """Test that negative subsample ratio raises error"""
        sequences = np.array([np.random.randint(0, 100, size=100)])
        
        with pytest.raises(ValueError):
            MIDIDataset(sequences, seq_length=50, subsample_ratio=-0.1)
            
    def test_dataset_empty_sequences(self):
        """Test handling of empty sequence array"""
        sequences = np.array([], dtype=object)
        dataset = MIDIDataset(sequences, seq_length=50, subsample_ratio=1.0)
        
        assert len(dataset) == 0


class TestLogExperiment:
    """Test suite for log_experiment function"""
    
    def test_log_creates_file(self, tmp_path):
        """Test that log file is created"""
        logfile = tmp_path / "test_log.csv"
        
        hparams = {"model": "lstm", "lr": 0.001}
        results = {"loss": 1.5, "accuracy": 0.85}
        
        log_experiment(hparams, results, logfile)
        
        assert logfile.exists()
        
    def test_log_contains_data(self, tmp_path):
        """Test that logged data can be read back"""
        logfile = tmp_path / "test_log.csv"
        
        hparams = {"model": "lstm", "lr": 0.001, "epochs": 10}
        results = {"loss": 1.5, "accuracy": 0.85}
        
        log_experiment(hparams, results, logfile)
        
        # Read back the data
        with open(logfile, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 1
        assert rows[0]['model'] == 'lstm'
        assert rows[0]['lr'] == '0.001'
        assert rows[0]['loss'] == '1.5'
        
    def test_log_appends_experiments(self, tmp_path):
        """Test that multiple experiments are appended"""
        logfile = tmp_path / "test_log.csv"
        
        # Log first experiment
        hparams1 = {"model": "lstm", "lr": 0.001}
        results1 = {"loss": 1.5}
        log_experiment(hparams1, results1, logfile)
        
        # Log second experiment
        hparams2 = {"model": "gru", "lr": 0.002}
        results2 = {"loss": 1.3}
        log_experiment(hparams2, results2, logfile)
        
        # Read back the data
        with open(logfile, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 2
        assert rows[0]['model'] == 'lstm'
        assert rows[1]['model'] == 'gru'
        
    def test_log_creates_parent_directory(self, tmp_path):
        """Test that parent directories are created if needed"""
        logfile = tmp_path / "nested" / "dir" / "test_log.csv"
        
        hparams = {"model": "lstm"}
        results = {"loss": 1.5}
        
        log_experiment(hparams, results, logfile)
        
        assert logfile.exists()


class TestTrainFunction:
    """Test suite for train function"""
    
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Create temporary test data directory with necessary files"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Create sample sequences
        sequences = np.array([
            np.random.randint(0, 50, size=100),
            np.random.randint(0, 50, size=100),
            np.random.randint(0, 50, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        # Create vocab file
        vocab = {str(i): i for i in range(50)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
            
        # Create config file
        config = {"seq_length": 50, "vocab_size": 50}
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)
            
        return data_dir
    
    @pytest.fixture
    def setup_naive_data(self, tmp_path):
        """Create temporary naive format test data"""
        data_dir = tmp_path / "test_naive"
        data_dir.mkdir()
        
        # Create sample sequences
        sequences = np.array([
            np.random.randint(0, 30, size=100),
            np.random.randint(0, 30, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        # Create note_to_int pickle
        note_to_int = {f"note_{i}": i for i in range(30)}
        vocab_data = {"note_to_int": note_to_int}
        with open(data_dir / "note_to_int.pkl", 'wb') as f:
            pickle.dump(vocab_data, f)
            
        return data_dir
    
    def test_train_basic_lstm(self, tmp_path, setup_test_data):
        """Test basic LSTM training"""
        # Move test data to expected location
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        # Change to tmp directory
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="test_dataset",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            # Check that model was saved
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "test_dataset"
            assert model_dir.exists()
            model_files = list(model_dir.glob("lstm_*.pth"))
            assert len(model_files) > 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_gru(self, tmp_path, setup_test_data):
        """Test GRU training"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="gru",
                dataset="test_dataset",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "test_dataset"
            model_files = list(model_dir.glob("gru_*.pth"))
            assert len(model_files) > 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_transformer(self, tmp_path, setup_test_data):
        """Test Transformer training"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="transformer",
                dataset="test_dataset",
                d_model=64,
                nhead=4,
                num_layers=1,
                dim_feedforward=128,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "test_dataset"
            model_files = list(model_dir.glob("transformer_*.pth"))
            assert len(model_files) > 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_validation(self, tmp_path, setup_test_data):
        """Test training with validation split"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="test_dataset",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=2,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1,
                val_split=0.2,
                patience=10
            )
            
            # Check that validation loss was saved
            log_dir = tmp_path / "logs" / "generators" / "test_dataset" / "models"
            val_loss_files = list(log_dir.glob("val_losses_*.npy"))
            assert len(val_loss_files) > 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_early_stopping(self, tmp_path, setup_test_data):
        """Test early stopping mechanism"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="test_dataset",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=10,  # High number
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1,
                val_split=0.2,
                patience=1  # Low patience for quick stopping
            )
            
            # Check that model was saved
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "test_dataset"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_checkpoint_dir(self, tmp_path, setup_test_data):
        """Test training with custom checkpoint directory"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        checkpoint_dir = tmp_path / "custom_checkpoints"
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="test_dataset",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1,
                checkpoint_dir=str(checkpoint_dir)
            )
            
            # Check that checkpoint was saved in custom directory
            model_dir = checkpoint_dir / "test_dataset" / "models"
            assert model_dir.exists()
            model_files = list(model_dir.glob("lstm_*.pth"))
            assert len(model_files) > 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_naive_format(self, tmp_path, setup_naive_data):
        """Test training with naive data format"""
        data_dir = tmp_path / "data" / "test_naive"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_naive_data), str(data_dir))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="test_naive",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "test_naive"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_creates_logs(self, tmp_path, setup_test_data):
        """Test that training creates all expected log files"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="test_dataset",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            log_dir = tmp_path / "logs" / "generators" / "test_dataset" / "models"
            
            # Check for CSV log
            assert (log_dir / "models.csv").exists()
            
            # Check for training summary
            summary_files = list(log_dir.glob("training_summary_*.txt"))
            assert len(summary_files) > 0
            
            # Check for loss arrays
            train_loss_files = list(log_dir.glob("train_losses_*.npy"))
            assert len(train_loss_files) > 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_creates_plots(self, tmp_path, setup_test_data):
        """Test that training creates loss curve plots"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(setup_test_data), str(data_dir))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="test_dataset",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            plots_dir = tmp_path / "outputs" / "generators" / "test_dataset" / "plots"
            plot_files = list(plots_dir.glob("loss_curve_*.png"))
            assert len(plot_files) > 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_vocab_size_mismatch_raises_error(self, tmp_path):
        """Test that vocab size mismatch is detected"""
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        
        # Create sequences with IDs that exceed vocab size
        sequences = np.array([
            np.random.randint(0, 100, size=100),  # IDs go up to 99
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        # Create vocab file with smaller vocab size
        vocab = {str(i): i for i in range(50)}  # Only 50 tokens
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
            
        config = {"seq_length": 50}
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with pytest.raises(ValueError, match="token id in data.*vocab_size"):
                train(
                    model_type="lstm",
                    dataset="test_dataset",
                    embed_size=32,
                    hidden_size=64,
                    num_layers=1,
                    batch_size=2,
                    epochs=1,
                    learning_rate=0.01,
                    device="cpu",
                    max_batches=2,
                    subsample_ratio=0.1
                )
        finally:
            os.chdir(original_cwd)


class TestTrainIntegration:
    """Integration tests for complete training workflows"""
    
    def test_full_training_pipeline(self, tmp_path):
        """Test complete training pipeline from data to saved model"""
        # Setup data
        data_dir = tmp_path / "data" / "integration_test"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 40, size=150),
            np.random.randint(0, 40, size=150),
            np.random.randint(0, 40, size=150),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(40)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
            
        config = {"seq_length": 50}
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Train model
            train(
                model_type="lstm",
                dataset="integration_test",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=2,
                learning_rate=0.01,
                device="cpu",
                max_batches=5,
                subsample_ratio=0.2,
                val_split=0.2
            )
            
            # Verify all outputs exist
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "integration_test"
            output_dir = tmp_path / "outputs" / "generators" / "integration_test"
            log_dir = tmp_path / "logs" / "generators" / "integration_test" / "models"
            
            assert model_dir.exists()
            assert output_dir.exists()
            assert log_dir.exists()
            
            # Check model file
            model_files = list(model_dir.glob("*.pth"))
            assert len(model_files) > 0
            
            # Check plots
            plot_files = list((output_dir / "plots").glob("*.png"))
            assert len(plot_files) > 0
            
            # Check logs
            assert (log_dir / "models.csv").exists()
            
            # Load and verify model can be used
            model_file = model_files[0]
            state_dict = torch.load(model_file, map_location='cpu')
            assert isinstance(state_dict, dict)
            assert len(state_dict) > 0
            
        finally:
            os.chdir(original_cwd)


class TestTrainEdgeCases:
    """Test edge cases and alternative code paths"""
    
    def test_train_without_config_file(self, tmp_path):
        """Test training when config.json doesn't exist (uses default seq_length)"""
        data_dir = tmp_path / "data" / "no_config"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 30, size=100),
            np.random.randint(0, 30, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(30)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        # Note: No config.json file created
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="no_config",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            # Should still work with default seq_length
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "no_config"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_list_vocab_format(self, tmp_path):
        """Test training with vocab.json as a list instead of dict"""
        data_dir = tmp_path / "data" / "list_vocab"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 25, size=100),
            np.random.randint(0, 25, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        # Vocab as list
        vocab = [f"token_{i}" for i in range(25)]
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="list_vocab",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "list_vocab"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_inverted_vocab_dict(self, tmp_path):
        """Test training with vocab.json having id->token mapping (inverted)"""
        data_dir = tmp_path / "data" / "inverted_vocab"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 20, size=100),
            np.random.randint(0, 20, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        # Vocab with id->token mapping (values are strings, not ints)
        vocab = {str(i): f"token_{i}" for i in range(20)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="inverted_vocab",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "inverted_vocab"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_missing_vocab_file_raises_error(self, tmp_path):
        """Test that missing vocab file raises FileNotFoundError"""
        data_dir = tmp_path / "data" / "no_vocab"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 30, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        # No vocab file created
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with pytest.raises(FileNotFoundError, match="no vocab file found"):
                train(
                    model_type="lstm",
                    dataset="no_vocab",
                    embed_size=32,
                    hidden_size=64,
                    num_layers=1,
                    batch_size=2,
                    epochs=1,
                    learning_rate=0.01,
                    device="cpu",
                    max_batches=2,
                    subsample_ratio=0.1
                )
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_miditok_dataset(self, tmp_path):
        """Test training with miditok dataset (tests scheduler and gradient clipping)"""
        data_dir = tmp_path / "data" / "miditok"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 40, size=100),
            np.random.randint(0, 40, size=100),
            np.random.randint(0, 40, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(40)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        config = {"seq_length": 50}
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="miditok",  # This triggers scheduler and gradient clipping
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=2,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1,
                val_split=0.2
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "miditok"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_miditok_augmented_dataset(self, tmp_path):
        """Test training with miditok_augmented dataset"""
        data_dir = tmp_path / "data" / "miditok_augmented"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 40, size=100),
            np.random.randint(0, 40, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(40)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        config = {"seq_length": 50}
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="miditok_augmented",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=2,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=0.1,
                val_split=0.2
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "miditok_augmented"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_auto_device_detection(self, tmp_path):
        """Test that device=None triggers auto-detection"""
        data_dir = tmp_path / "data" / "auto_device"
        data_dir.mkdir(parents=True)
        
        sequences = np.array([
            np.random.randint(0, 30, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(30)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="auto_device",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device=None,  # Should auto-detect
                max_batches=2,
                subsample_ratio=0.1
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "auto_device"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_empty_sequence_in_data(self, tmp_path):
        """Test training handles empty sequences in data"""
        data_dir = tmp_path / "data" / "empty_seq"
        data_dir.mkdir(parents=True)
        
        # Include an empty sequence
        sequences = np.array([
            np.array([]),  # Empty sequence
            np.random.randint(0, 30, size=100),
            np.random.randint(0, 30, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(30)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="empty_seq",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=2,
                subsample_ratio=1.0 # Use 1.0 to ensure we get samples from the few valid sequences
            )
            
            # Should handle empty sequence gracefully
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "empty_seq"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_non_int_tokens_raises_error(self, tmp_path):
        """Test that non-integer tokens in sequences raise an error"""
        data_dir = tmp_path / "data" / "bad_tokens"
        data_dir.mkdir(parents=True)
        
        # Create sequences with non-int values
        sequences = np.array([
            np.array(['a', 'b', 'c'] * 20),  # String tokens
            np.random.randint(0, 30, size=100),
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(30)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with pytest.raises(Exception):  # Should raise error during vocab validation
                train(
                    model_type="lstm",
                    dataset="bad_tokens",
                    embed_size=32,
                    hidden_size=64,
                    num_layers=1,
                    batch_size=2,
                    epochs=1,
                    learning_rate=0.01,
                    device="cpu",
                    max_batches=2,
                    subsample_ratio=0.1
                )
        finally:
            os.chdir(original_cwd)
    
    def test_train_with_many_batches_triggers_progress_print(self, tmp_path):
        """Test that training with many batches triggers progress printing"""
        data_dir = tmp_path / "data" / "many_batches"
        data_dir.mkdir(parents=True)
        
        # Create more sequences for more batches
        sequences = np.array([
            np.random.randint(0, 30, size=200) for _ in range(20)
        ], dtype=object)
        np.save(data_dir / "sequences.npy", sequences)
        
        vocab = {str(i): i for i in range(30)}
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f)
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            train(
                model_type="lstm",
                dataset="many_batches",
                embed_size=32,
                hidden_size=64,
                num_layers=1,
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                device="cpu",
                max_batches=600,  # Enough to trigger progress printing
                subsample_ratio=1.0
            )
            
            model_dir = tmp_path / "models" / "generators" / "checkpoints" / "many_batches"
            assert model_dir.exists()
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
