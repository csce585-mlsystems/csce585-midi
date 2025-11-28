import unittest
from unittest.mock import MagicMock, patch, mock_open
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate import generate

class TestGenerateArgs(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.return_value = (torch.randn(1, 1, 10), None) # output, hidden
        self.mock_model.parameters.return_value = [torch.randn(1)]

        self.mock_discriminator = MagicMock()
        self.mock_discriminator.return_value = torch.randn(1, 52)

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.np.load')
    @patch('generate.pickle.load')
    @patch('generate.json.load')
    @patch('generate.sample_next_note')
    @patch('generate.stream.Stream')
    @patch('generate.miditok.REMI')
    @patch('evaluate.evaluate_midi')
    @patch('evaluate.log_evaluation')
    @patch('builtins.open', new_callable=mock_open)
    @patch('generate.Path.mkdir')
    def test_generate_naive_greedy(self, mock_mkdir, mock_file, mock_log_eval, mock_eval_midi, 
                                  mock_remi, mock_stream, mock_sample, mock_json_load, 
                                  mock_pickle_load, mock_np_load, mock_torch_load, mock_get_generator):
        
        # Setup mocks
        mock_get_generator.return_value = self.mock_model
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.randn(1024, 256)} # Mock checkpoint
        
        # Mock sequences
        mock_np_load.return_value = np.array([[1, 2, 3] * 20]) # Mock sequences
        
        # Mock vocab for naive - need to cover all possible tokens
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0, "D4": 1, "E4": 2, "F4": 3}}
        
        # Mock sample_next_note
        mock_sample.return_value = torch.tensor(0)
        
        # Run generate
        generate(
            strategy="greedy",
            generate_length=10,
            seq_length=50,
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            data_dir="data/naive"
        )
        
        # Assertions
        mock_get_generator.assert_called()
        mock_sample.assert_called()
        mock_stream.assert_called() # Should use music21 stream for naive
        mock_remi.assert_not_called() # Should not use miditok

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.np.load')
    @patch('generate.pickle.load')
    @patch('generate.json.load')
    @patch('generate.sample_next_note')
    @patch('generate.miditok.REMI')
    @patch('generate.miditok.TokSequence')
    @patch('evaluate.evaluate_midi')
    @patch('evaluate.log_evaluation')
    @patch('builtins.open', new_callable=mock_open)
    @patch('generate.Path.mkdir')
    def test_generate_miditok_top_k(self, mock_mkdir, mock_file, mock_log_eval, mock_eval_midi, 
                                   mock_tok_seq, mock_remi, mock_sample, mock_json_load, 
                                   mock_pickle_load, mock_np_load, mock_torch_load, mock_get_generator):
        
        # Setup mocks
        mock_get_generator.return_value = self.mock_model
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.randn(1024, 256)}
        
        mock_np_load.return_value = np.array([[1, 2, 3] * 20])
        
        # Mock vocab for miditok
        mock_json_load.return_value = ["Pitch_60", "Pitch_62", "Pitch_64"]
        
        mock_sample.return_value = torch.tensor(0)
        
        # Run generate
        generate(
            strategy="top_k",
            k=10,
            generate_length=10,
            seq_length=50,
            model_path="models/generators/miditok/lstm.pt",
            model_type="lstm",
            data_dir="data/miditok"
        )
        
        # Assertions
        mock_remi.assert_called() # Should use miditok
        mock_tok_seq.assert_called()

    @patch('generate.get_generator')
    @patch('generate.get_discriminator')
    @patch('generate.torch.load')
    @patch('generate.np.load')
    @patch('generate.pickle.load')
    @patch('generate.json.load')
    @patch('generate.sample_next_note')
    @patch('generate.stream.Stream')
    @patch('evaluate.evaluate_midi')
    @patch('evaluate.log_evaluation')
    @patch('builtins.open', new_callable=mock_open)
    @patch('generate.Path.mkdir')
    def test_generate_with_discriminator(self, mock_mkdir, mock_file, mock_log_eval, mock_eval_midi, 
                                        mock_stream, mock_sample, mock_json_load, 
                                        mock_pickle_load, mock_np_load, mock_torch_load, 
                                        mock_get_discriminator, mock_get_generator):
        
        # Setup mocks
        mock_get_generator.return_value = self.mock_model
        mock_get_discriminator.return_value = self.mock_discriminator
        
        # Mock checkpoints
        mock_torch_load.side_effect = [
            {"lstm.weight_ih_l0": torch.randn(1024, 256)}, # Generator
            {"state_dict": "mock"} # Discriminator
        ]
        
        mock_np_load.return_value = np.array([[1, 2, 3] * 20])
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0, "D4": 1, "E4": 2, "F4": 3}}
        mock_sample.return_value = torch.tensor(0)
        
        # Run generate
        generate(
            strategy="greedy",
            generate_length=10,
            seq_length=50,
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            discriminator_path="models/discriminators/naive/disc.pt",
            discriminator_type="lstm",
            guidance_strength=0.8,
            data_dir="data/naive"
        )
        
        # Assertions
        mock_get_discriminator.assert_called()
        self.mock_discriminator.eval.assert_called()
        # Discriminator should be called during generation
        self.mock_discriminator.assert_called()

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.np.load')
    @patch('generate.pickle.load')
    @patch('generate.json.load')
    @patch('generate.sample_next_note')
    @patch('generate.stream.Stream')
    @patch('evaluate.evaluate_midi')
    @patch('evaluate.log_evaluation')
    @patch('generate.get_seed_by_filename')
    @patch('builtins.open', new_callable=mock_open)
    @patch('generate.Path.mkdir')
    def test_generate_with_seed_file(self, mock_mkdir, mock_file, mock_get_seed, mock_log_eval, 
                                    mock_eval_midi, mock_stream, mock_sample, mock_json_load, 
                                    mock_pickle_load, mock_np_load, mock_torch_load, mock_get_generator):
        
        # Setup mocks
        mock_get_generator.return_value = self.mock_model
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.randn(1024, 256)}
        mock_np_load.return_value = np.array([[1, 2, 3] * 20])
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0, "D4": 1, "E4": 2, "F4": 3}}
        mock_sample.return_value = torch.tensor(0)
        
        # Mock seed return
        mock_get_seed.return_value = [0, 1, 2] * 20
        
        # Run generate
        generate(
            strategy="greedy",
            generate_length=10,
            seq_length=50,
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            seed_file="test_song.mid",
            data_dir="data/naive"
        )
        
        # Assertions
        mock_get_seed.assert_called_with(filename="test_song.mid", dataset="naive", length=50)

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.np.load')
    @patch('generate.pickle.load')
    @patch('generate.json.load')
    @patch('generate.sample_next_note')
    @patch('generate.stream.Stream')
    @patch('evaluate.evaluate_midi')
    @patch('evaluate.log_evaluation')
    @patch('builtins.open', new_callable=mock_open)
    @patch('generate.Path.mkdir')
    def test_generate_transformer_model(self, mock_mkdir, mock_file, mock_log_eval, mock_eval_midi, 
                                       mock_stream, mock_sample, mock_json_load, 
                                       mock_pickle_load, mock_np_load, mock_torch_load, mock_get_generator):
        
        # Setup mocks
        mock_get_generator.return_value = self.mock_model
        
        # Mock transformer checkpoint to test architecture inference
        mock_torch_load.return_value = {
            "transformer_decoder.layers.0.self_attn.in_proj_weight": torch.randn(10, 10),
            "transformer_decoder.layers.5.self_attn.in_proj_weight": torch.randn(10, 10), # implies 6 layers
            "embedding.weight": torch.randn(100, 512) # implies hidden_size=512
        }
        
        mock_np_load.return_value = np.array([[1, 2, 3] * 20])
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0, "D4": 1, "E4": 2, "F4": 3}}
        mock_sample.return_value = torch.tensor(0)
        
        # Run generate
        generate(
            strategy="greedy",
            generate_length=10,
            seq_length=50,
            model_path="models/generators/naive/transformer.pt",
            model_type="transformer",
            data_dir="data/naive"
        )
        
        # Assertions
        # Check if get_generator was called with inferred params
        mock_get_generator.assert_called()
        call_kwargs = mock_get_generator.call_args[1]
        self.assertEqual(call_kwargs['num_layers'], 6)
        self.assertEqual(call_kwargs['d_model'], 512)

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.np.load')
    @patch('generate.pickle.load')
    @patch('generate.json.load')
    @patch('generate.sample_next_note')
    @patch('generate.stream.Stream')
    @patch('evaluate.evaluate_midi')
    @patch('evaluate.log_evaluation')
    @patch('generate.find_seed_by_characteristics')
    @patch('builtins.open', new_callable=mock_open)
    @patch('generate.Path.mkdir')
    def test_generate_smart_seed(self, mock_mkdir, mock_file, mock_find_seed, mock_log_eval, 
                                mock_eval_midi, mock_stream, mock_sample, mock_json_load, 
                                mock_pickle_load, mock_np_load, mock_torch_load, mock_get_generator):
        
        # Setup mocks
        mock_get_generator.return_value = self.mock_model
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.randn(1024, 256)}
        mock_np_load.return_value = np.array([[1, 2, 3] * 20])
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0, "D4": 1, "E4": 2, "F4": 3}}
        mock_sample.return_value = torch.tensor(0)
        
        # Mock smart seed return
        mock_find_seed.return_value = [0, 1, 2] * 20
        
        # Run generate
        generate(
            strategy="greedy",
            generate_length=10,
            seq_length=50,
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            seed_style="smart",
            pitch_preference="high",
            complexity="complex",
            seed_length="long",
            data_dir="data/naive"
        )
        
        # Assertions
        mock_find_seed.assert_called_with(
            sequences=mock_np_load.return_value,
            int_to_note={0: 'C4', 1: 'D4', 2: 'E4', 3: 'F4'},  # Reversed mapping
            pitch_preference="high",
            complexity="complex",
            length="long",
            dataset="naive"
        )
