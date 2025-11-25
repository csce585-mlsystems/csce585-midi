import pytest
import numpy as np
import json
import tempfile
import shutil
import os
import torch
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train_generator import train

class TestTrainGeneratorPaths:
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for data"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def create_dummy_dataset(self, data_dir, dataset_name="test_dataset"):
        """Create dummy sequences.npy and vocab.json in the specified directory"""
        dataset_dir = data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy sequences (list of lists of ints)
        # Ensure max id is within vocab size (vocab has 5 items, ids 1-5)
        sequences = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=object)
        np.save(dataset_dir / "sequences.npy", sequences)
        
        # Create dummy vocab
        vocab = {"note_1": 1, "note_2": 2, "note_3": 3, "note_4": 4, "note_5": 5}
        with open(dataset_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
            
        # Create config.json with small seq_length
        config = {"seq_length": 4}
        with open(dataset_dir / "config.json", "w") as f:
            json.dump(config, f)
            
        return dataset_dir

    @patch("training.train_generator.get_generator")
    @patch("training.train_generator.log_experiment")
    @patch("torch.save")
    def test_train_with_absolute_path(self, mock_save, mock_log, mock_get_generator, temp_data_dir):
        """Test training with an absolute path to the dataset"""
        dataset_path = self.create_dummy_dataset(temp_data_dir, "abs_path_dataset")
        
        # Mock generator to avoid model creation
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        # Create a dummy parameter so optimizer doesn't complain
        dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
        mock_model.parameters.return_value = [dummy_param]
        mock_model.state_dict.return_value = {} # Return empty dict for state_dict
        
        # Configure model forward pass return value
        def forward_side_effect(x, *args, **kwargs):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            vocab_size = 6 # from dummy vocab
            return torch.randn(batch_size, seq_len, vocab_size, requires_grad=True), None
            
        mock_model.side_effect = forward_side_effect
        
        mock_get_generator.return_value = mock_model
        
        # Run train with epochs=1 and max_batches=1 to run minimal training loop
        # We expect it to run without error and find the files
        try:
            train(
                dataset=str(dataset_path),
                epochs=1,
                batch_size=2,
                model_type="lstm",
                max_batches=1,
                device="cpu"
            )
        except FileNotFoundError as e:
            pytest.fail(f"Failed to find dataset at absolute path: {e}")
        except Exception as e:
            pytest.fail(f"Training failed with unexpected error: {e}")

    @patch("training.train_generator.get_generator")
    @patch("training.train_generator.log_experiment")
    @patch("torch.save")
    def test_train_with_local_folder_name(self, mock_save, mock_log, mock_get_generator, temp_data_dir):
        """Test training with a folder name that exists in the current directory"""
        
        # Create a dataset in the current directory (we'll use temp_data_dir as the 'current' dir by changing cwd)
        dataset_name = "local_dataset"
        self.create_dummy_dataset(temp_data_dir, dataset_name)
        
        # Change cwd to temp_data_dir
        original_cwd = os.getcwd()
        os.chdir(temp_data_dir)
        
        try:
            # Mock generator
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            # Create a dummy parameter so optimizer doesn't complain
            dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
            mock_model.parameters.return_value = [dummy_param]
            mock_model.state_dict.return_value = {} # Return empty dict for state_dict
            
            # Configure model forward pass return value
            def forward_side_effect(x, *args, **kwargs):
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                vocab_size = 6 # from dummy vocab
                return torch.randn(batch_size, seq_len, vocab_size, requires_grad=True), None
                
            mock_model.side_effect = forward_side_effect
            
            mock_get_generator.return_value = mock_model
    
            # Run train with just the name
            train(
                dataset=dataset_name,
                epochs=1,
                batch_size=2,
                model_type="lstm",
                max_batches=1,
                device="cpu"
            )
        except Exception as e:
            pytest.fail(f"Training failed with local folder name: {e}")
        finally:
            os.chdir(original_cwd)
