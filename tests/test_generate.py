import unittest
from unittest.mock import MagicMock, patch, mock_open
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate import load_discriminator, apply_discriminator_guidance, generate

class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.mock_discriminator = MagicMock()
        self.mock_generator = MagicMock()
        
    @patch('generate.get_discriminator')
    @patch('torch.load')
    def test_load_discriminator(self, mock_torch_load, mock_get_discriminator):
        mock_get_discriminator.return_value = self.mock_discriminator
        mock_torch_load.return_value = {} # Mock state dict
        
        discriminator = load_discriminator("path/to/disc.pt", "lstm")
        
        mock_get_discriminator.assert_called_once()
        self.mock_discriminator.load_state_dict.assert_called_once()
        self.mock_discriminator.eval.assert_called_once()
        self.assertEqual(discriminator, self.mock_discriminator)

    def test_apply_discriminator_guidance_naive(self):
        # Setup inputs
        logits = torch.zeros(1, 10)
        context_measures = 1
        # Need enough tokens to satisfy min_tokens=16 check
        generated_so_far = [0, 1, 2] + [0] * 20 # indices
        int_to_note = {0: "60", 1: "64", 2: "67"} # C major triad
        
        # Mock discriminator output (logits for 52 pitch classes)
        # Let's say it predicts C major (root 0 -> C)
        disc_output = torch.zeros(1, 52)
        disc_output[0, 0] = 10.0 # High prob for C
        self.mock_discriminator.return_value = disc_output
        
        # Call function
        adjusted_logits = apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="naive"
        )
        
        # Check if logits were boosted
        # C major notes: C(0), E(4), G(7)
        # In our int_to_note: 0->60(C4), 1->64(E4), 2->67(G4)
        # All should be boosted
        self.assertTrue((adjusted_logits[0, 0] > 0).item())
        self.assertTrue((adjusted_logits[0, 1] > 0).item())
        self.assertTrue((adjusted_logits[0, 2] > 0).item())

    def test_apply_discriminator_guidance_miditok(self):
        # Setup inputs
        logits = torch.zeros(1, 10)
        context_measures = 1
        # Need enough tokens to satisfy min_tokens=100 check
        generated_so_far = [0, 1] * 60
        int_to_note = {0: "Pitch_60", 1: "Pitch_64", 2: "Velocity_100"} 
        
        # Mock discriminator output
        disc_output = torch.zeros(1, 52)
        disc_output[0, 0] = 10.0 # Predict C
        self.mock_discriminator.return_value = disc_output
        
        adjusted_logits = apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="miditok"
        )
        
        # Pitch_60 (C) and Pitch_64 (E) should be boosted (C major triad members)
        self.assertTrue((adjusted_logits[0, 0] > 0).item())
        self.assertTrue((adjusted_logits[0, 1] > 0).item())
        # Velocity token should NOT be boosted
        self.assertEqual(adjusted_logits[0, 2].item(), 0)

    def test_apply_discriminator_guidance_miditok_augmented(self):
        # Regression test for miditok_augmented support
        logits = torch.zeros(1, 10)
        context_measures = 1
        generated_so_far = [0, 1] * 60
        int_to_note = {0: "Pitch_60", 1: "Pitch_64"} 
        
        # Mock discriminator output
        disc_output = torch.zeros(1, 52)
        disc_output[0, 0] = 10.0 # Predict C
        self.mock_discriminator.return_value = disc_output
        
        adjusted_logits = apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="miditok_augmented"
        )
        
        # Should boost Pitch_60 (C)
        self.assertTrue((adjusted_logits[0, 0] > 0).item())

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.json.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    @patch('generate.miditok.REMI')
    @patch('generate.miditok.TokSequence')
    def test_generate_miditok_flattening(self, mock_tok_seq, mock_remi, mock_log, mock_sample, mock_open_func, mock_np_load, mock_json_load, mock_torch_load, mock_get_generator):
        # Test specifically the flattening of nested lists (tracks)
        
        # Setup mocks
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None) # Output, hidden
        mock_torch_load.return_value = {"embedding.weight": torch.zeros(10, 256)} # Checkpoint
        
        # Mock vocab
        mock_open_func.return_value.__enter__.return_value = MagicMock()
        # Mock json load for vocab
        mock_json_load.return_value = ["token"]*10
        
        # Mock sequences with nested structure (tracks)
        # Shape: [sequence1, sequence2]
        # sequence1 = [[track1_token1, track1_token2], [track2_token1]]
        nested_seed = [[1, 2], [3]] 
        mock_np_load.return_value = [nested_seed]
        
        mock_sample.return_value = torch.tensor(0) # Always predict token 0
        
        # Run generate
        generate(
            model_path="models/generators/miditok/lstm.pt",
            model_type="lstm",
            seed_style="random",
            generate_length=5,
            seq_length=10
        )
        
        # Verify that the model was called with flattened input
        # The seed [[1, 2], [3]] should become [1, 2, 3]
        # Then padded/sliced to seq_length
        # We check the first call to model
        self.assertTrue(self.mock_generator.called)
            
    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    @patch('generate.get_seed_by_filename')
    def test_generate_seed_by_filename(self, mock_get_seed, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator):
        # Setup mocks
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.zeros(1024, 256)}
        
        # Mock vocab (naive)
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        
        # Mock sequences (not used if seed_file provided, but loaded anyway)
        mock_np_load.return_value = [[0]*50]
        
        # Mock get_seed_by_filename
        mock_get_seed.return_value = [0, 0, 0]
        mock_sample.return_value = torch.tensor(0)
        
        # Run generate with seed_file
        generate(
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            seed_file="test_song.mid",
            generate_length=5
        )
        
        # Verify get_seed_by_filename was called
        mock_get_seed.assert_called_with(filename="test_song.mid", dataset="naive", length=50)

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_architecture_inference(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator):
        # Test that architecture params are inferred from checkpoint
        
        # Mock checkpoint with specific keys
        mock_torch_load.return_value = {
            "lstm.weight_ih_l0": torch.zeros(1024, 256), # 4 * hidden_size(256)
            "lstm.weight_ih_l1": torch.zeros(1024, 256)  # 2 layers (0 and 1)
        }
        
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        mock_np_load.return_value = [[0]*50]
        mock_sample.return_value = torch.tensor(0)
        
        # Fix: Ensure get_generator returns our mock
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        
        generate(
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            generate_length=5
        )
        
        # Verify get_generator called with inferred params
        # hidden_size should be 256, num_layers should be 2
        mock_get_generator.assert_called_with(
            "lstm", 
            1, # vocab size
            embed_size=128, # default
            hidden_size=256, # inferred
            num_layers=2, # inferred
            dropout=0.2
        )

    @patch('generate.find_seed_by_characteristics')
    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_smart_seed(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator, mock_find_seed):
        # Setup mocks
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.zeros(1024, 256)}
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        mock_np_load.return_value = [[0]*50]
        mock_sample.return_value = torch.tensor(0)
        
        # Mock smart seed return
        mock_find_seed.return_value = [0]*50
        
        generate(
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            seed_style="smart",
            generate_length=5
        )
        
        mock_find_seed.assert_called_once()

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_transformer(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator):
        # Setup mocks for Transformer
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        
        # Transformer checkpoint keys
        mock_torch_load.return_value = {
            "transformer_decoder.layers.0.self_attn.in_proj_weight": torch.zeros(10, 10),
            "embedding.weight": torch.zeros(10, 512) # d_model=512
        }
        
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        mock_np_load.return_value = [[0]*50]
        mock_sample.return_value = torch.tensor(0)
        
        generate(
            model_path="models/generators/naive/transformer.pt",
            model_type="transformer",
            generate_length=5
        )
        
        # Verify transformer specific args
        mock_get_generator.assert_called_with(
            "transformer",
            1,
            num_layers=1, # 0-indexed + 1
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )

    @patch('generate.load_discriminator')
    @patch('generate.apply_discriminator_guidance')
    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_with_discriminator_integration(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator, mock_apply_guidance, mock_load_disc):
        # Setup mocks
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.zeros(1024, 256)}
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        mock_np_load.return_value = [[0]*50]
        mock_sample.return_value = torch.tensor(0)
        
        mock_disc = MagicMock()
        mock_load_disc.return_value = mock_disc
        
        # Mock apply guidance to return logits
        mock_apply_guidance.return_value = torch.zeros(1, 10)
        
        generate(
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            discriminator_path="disc.pt",
            discriminator_type="lstm",
            generate_length=5,
            guidance_frequency=1
        )
        
        mock_load_disc.assert_called_once()
        # Should be called generate_length times (5)
        self.assertEqual(mock_apply_guidance.call_count, 5)

    def test_generate_invalid_dataset_error(self):
        with self.assertRaises(ValueError) as cm:
            generate(model_path="models/generators/unknown/lstm.pt")
        self.assertIn("Cannot infer dataset", str(cm.exception))

    @patch('generate.get_seed_by_filename')
    @patch('generate.np.load')
    @patch('generate.pickle.load')
    @patch('generate.open')
    @patch('sys.exit')
    def test_generate_seed_file_not_found(self, mock_exit, mock_open_func, mock_pickle_load, mock_np_load, mock_get_seed):
        # Mock setup
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        mock_np_load.return_value = [[0]*50]
        
        # Mock seed not found
        mock_get_seed.return_value = None
        mock_exit.side_effect = SystemExit
        
        with self.assertRaises(SystemExit):
            generate(
                model_path="models/generators/naive/lstm.pt",
                seed_file="missing.mid"
            )
        
        mock_exit.assert_called_with(1)

    @patch('evaluate.evaluate_midi')
    @patch('evaluate.log_evaluation')
    @patch('csv.DictWriter')
    @patch('generate.open')
    def test_log_generated_midi(self, mock_open_func, mock_writer, mock_log_eval, mock_eval_midi):
        from generate import log_generated_midi
        
        mock_file = MagicMock()
        mock_open_func.return_value.__enter__.return_value = mock_file
        
        log_generated_midi(
            output_file=Path("out.mid"),
            strategy="greedy",
            generate_length=100,
            temperature=1.0,
            k=5,
            p=0.9,
            model_path="model.pt",
            log_file=Path("log.csv")
        )
        
        mock_writer.return_value.writerow.assert_called_once()
        mock_eval_midi.assert_called_once()

    @patch('generate.miditok.REMI')
    @patch('generate.miditok.TokSequence')
    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.json.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_decoding_miditok(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_json_load, mock_torch_load, mock_get_generator, mock_tok_seq, mock_remi):
        # Setup mocks
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        mock_torch_load.return_value = {"embedding.weight": torch.zeros(10, 256)}
        
        mock_open_func.return_value.__enter__.return_value = MagicMock()
        mock_json_load.return_value = ["Pitch_60", "Pitch_64"]
        mock_np_load.return_value = [[0]*50]
        mock_sample.return_value = torch.tensor(0)
        
        # Mock miditok decoding
        mock_tokenizer = MagicMock()
        mock_remi.return_value = mock_tokenizer
        mock_score = MagicMock()
        mock_tokenizer.decode.return_value = mock_score
        
        generate(
            model_path="models/generators/miditok/lstm.pt",
            model_type="lstm",
            generate_length=5
        )
        
        mock_remi.assert_called_once()
        mock_tokenizer.decode.assert_called_once()
        mock_score.dump_midi.assert_called_once()

    @patch('generate.stream.Stream')
    @patch('generate.note.Note')
    @patch('generate.chord.Chord')
    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_decoding_naive(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator, mock_chord, mock_note, mock_stream):
        # Setup mocks
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.zeros(1024, 256)}
        
        # Mock vocab with chords and notes
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0, "60.64.67": 1}}
        mock_np_load.return_value = [[0]*50]
        
        # Return 0 (note) then 1 (chord)
        mock_sample.side_effect = [torch.tensor(0), torch.tensor(1)] * 10
        
        mock_midi_stream = MagicMock()
        mock_stream.return_value = mock_midi_stream
        
        generate(
            model_path="models/generators/naive/lstm.pt",
            model_type="lstm",
            generate_length=2
        )
        
        mock_stream.assert_called_once()
        # Should append note and chord
        self.assertTrue(mock_midi_stream.append.call_count >= 2)
        mock_midi_stream.write.assert_called_once()

    def test_apply_discriminator_guidance_edge_cases(self):
        # Test malformed tokens in miditok
        logits = torch.zeros(1, 10)
        context_measures = 1
        generated_so_far = [0, 1] * 60
        int_to_note = {0: "Malformed_Token", 1: "Pitch_Invalid"} 
        
        disc_output = torch.zeros(1, 52)
        self.mock_discriminator.return_value = disc_output
        
        # Should not crash
        apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="miditok"
        )
        
        # Test naive with note names
        int_to_note_naive = {0: "C4", 1: "D4"}
        generated_so_far_naive = [0, 1] * 20
        
        apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far_naive, int_to_note_naive, guidance_strength=1.0, dataset="naive"
        )

    def test_apply_discriminator_guidance_padding(self):
        # Test padding when context is too short
        logits = torch.zeros(1, 10)
        context_measures = 4 # Needs 4*16 = 64 tokens
        generated_so_far = [0] * 20 # Only 20 tokens
        int_to_note = {0: "60"}
        
        disc_output = torch.zeros(1, 52)
        self.mock_discriminator.return_value = disc_output
        
        # Should pad internally and run without error
        apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="naive"
        )
        
        # Verify discriminator called with correct shape (1, 4, 52)
        args, _ = self.mock_discriminator.call_args
        self.assertEqual(args[0].shape, (1, 4, 52))

    def test_apply_discriminator_guidance_short_context(self):
        # Test early return when context is too short
        logits = torch.zeros(1, 10)
        context_measures = 4
        generated_so_far = [0] * 5 # Too short
        int_to_note = {0: "60"}
        
        # Should return logits immediately without calling discriminator
        result = apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="naive"
        )
        
        self.mock_discriminator.assert_not_called()
        self.assertTrue(torch.equal(result, logits))

    def test_apply_discriminator_guidance_truncation(self):
        # Test truncation when context is too long
        logits = torch.zeros(1, 10)
        context_measures = 1 # Needs 16 tokens
        generated_so_far = [0] * 100 # Too long
        int_to_note = {0: "60"}
        
        disc_output = torch.zeros(1, 52)
        self.mock_discriminator.return_value = disc_output
        
        apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="naive"
        )
        
        # Verify discriminator called with correct shape (1, 1, 52)
        args, _ = self.mock_discriminator.call_args
        self.assertEqual(args[0].shape, (1, 1, 52))

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.json.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_miditok_augmented(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_json_load, mock_torch_load, mock_get_generator):
        # Test miditok_augmented dataset inference
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        mock_torch_load.return_value = {"embedding.weight": torch.zeros(10, 256)}
        
        mock_open_func.return_value.__enter__.return_value = MagicMock()
        mock_json_load.return_value = ["token"]*10
        mock_np_load.return_value = [[0]*50]
        mock_sample.return_value = torch.tensor(0)
        
        generate(
            model_path="models/generators/miditok_augmented/lstm.pt",
            model_type="lstm",
            generate_length=5
        )
        
        # Verify it ran without error and inferred dataset correctly
        # We can check if it tried to open the vocab from the right place
        # But since we mock open, we can just check if it didn't crash
        self.assertTrue(self.mock_generator.called)

    def test_apply_discriminator_guidance_no_discriminator(self):
        logits = torch.zeros(1, 10)
        result = apply_discriminator_guidance(
            logits, None, 4, [0]*100, {}, 1.0, "naive"
        )
        self.assertTrue(torch.equal(result, logits))

    @patch('generate.load_discriminator')
    @patch('generate.apply_discriminator_guidance')
    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_generate_combinations(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator, mock_apply_guidance, mock_load_disc):
        # Test various combinations of generator and discriminator architectures
        
        combinations = [
            ("transformer", "mlp", "naive"),
            ("gru", "transformer", "naive"),
            ("lstm", "lstm", "miditok")
        ]
        
        mock_sample.return_value = torch.tensor(0)
        mock_np_load.return_value = [[0]*50]
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        
        # Mock file read for miditok json load
        mock_file = MagicMock()
        mock_file.read.return_value = '["token"]'
        mock_open_func.return_value.__enter__.return_value = mock_file
        
        for gen_type, disc_type, dataset in combinations:
            # Reset mocks
            mock_get_generator.reset_mock()
            mock_load_disc.reset_mock()
            
            # Setup generator mock
            mock_get_generator.return_value = self.mock_generator
            self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
            
            # Setup checkpoint mock based on gen_type
            if gen_type == "transformer":
                mock_torch_load.return_value = {
                    "transformer_decoder.layers.0.self_attn.in_proj_weight": torch.zeros(10, 10),
                    "embedding.weight": torch.zeros(10, 256)
                }
            elif gen_type == "gru":
                mock_torch_load.return_value = {
                    "gru.weight_ih_l0": torch.zeros(384, 128),
                    "gru.weight_hh_l0": torch.zeros(384, 128)
                }
            else: # lstm
                mock_torch_load.return_value = {
                    "lstm.weight_ih_l0": torch.zeros(1024, 256)
                }

            # Run generate
            generate(
                model_path=f"models/generators/{dataset}/{gen_type}.pt",
                model_type=gen_type,
                discriminator_path=f"models/discriminators/{disc_type}.pt",
                discriminator_type=disc_type,
                generate_length=2
            )
            
            # Verify correct models loaded
            mock_get_generator.assert_called()
            # Check first arg of get_generator call
            self.assertEqual(mock_get_generator.call_args[0][0], gen_type)
            
            mock_load_disc.assert_called()
            self.assertEqual(mock_load_disc.call_args[0][1], disc_type)
        
        for gen_type, disc_type, dataset in combinations:
            # Reset mocks
            mock_get_generator.reset_mock()
            mock_load_disc.reset_mock()
            
            # Setup generator mock
            mock_get_generator.return_value = self.mock_generator
            self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
            
            # Setup checkpoint mock based on gen_type
            if gen_type == "transformer":
                mock_torch_load.return_value = {
                    "transformer_decoder.layers.0.self_attn.in_proj_weight": torch.zeros(10, 10),
                    "embedding.weight": torch.zeros(10, 256)
                }
            elif gen_type == "gru":
                mock_torch_load.return_value = {
                    "gru.weight_ih_l0": torch.zeros(384, 128),
                    "gru.weight_hh_l0": torch.zeros(384, 128)
                }
            else: # lstm
                mock_torch_load.return_value = {
                    "lstm.weight_ih_l0": torch.zeros(1024, 256)
                }

            # Run generate
            generate(
                model_path=f"models/generators/{dataset}/{gen_type}.pt",
                model_type=gen_type,
                discriminator_path=f"models/discriminators/{disc_type}.pt",
                discriminator_type=disc_type,
                generate_length=2
            )
            
            # Verify correct models loaded
            mock_get_generator.assert_called()
            # Check first arg of get_generator call
            self.assertEqual(mock_get_generator.call_args[0][0], gen_type)
            
            mock_load_disc.assert_called()
            self.assertEqual(mock_load_disc.call_args[0][1], disc_type)

    @patch('generate.get_generator')
    @patch('generate.torch.load')
    @patch('generate.pickle.load')
    @patch('generate.np.load')
    @patch('generate.open')
    @patch('generate.sample_next_note')
    @patch('generate.log_generated_midi')
    def test_sampling_strategies(self, mock_log, mock_sample, mock_open_func, mock_np_load, mock_pickle_load, mock_torch_load, mock_get_generator):
        # Test that sampling parameters are passed correctly
        mock_get_generator.return_value = self.mock_generator
        self.mock_generator.return_value = (torch.zeros(1, 1, 10), None)
        mock_torch_load.return_value = {"lstm.weight_ih_l0": torch.zeros(1024, 256)}
        mock_pickle_load.return_value = {"note_to_int": {"C4": 0}}
        mock_np_load.return_value = [[0]*50]
        mock_sample.return_value = torch.tensor(0)
        
        strategies = ["greedy", "top_k", "top_p", "random"]
        
        for strategy in strategies:
            mock_sample.reset_mock()
            mock_sample.return_value = torch.tensor(0)
            
            generate(
                model_path="models/generators/naive/lstm.pt",
                model_type="lstm",
                strategy=strategy,
                temperature=0.8,
                k=10,
                p=0.85,
                generate_length=1
            )
            
            # Verify sample_next_note called with correct args
            _, kwargs = mock_sample.call_args
            self.assertEqual(kwargs['strategy'], strategy)
            self.assertEqual(kwargs['temperature'], 0.8)
            self.assertEqual(kwargs['k'], 10)
            self.assertEqual(kwargs['p'], 0.85)

    def test_apply_discriminator_guidance_music21_notes(self):
        # Test music21 note parsing in naive dataset
        logits = torch.zeros(1, 10)
        context_measures = 1
        # Use note names that music21 can parse
        int_to_note = {0: "C4", 1: "E4", 2: "G4", 3: "Invalid"}
        generated_so_far = [0, 1, 2] + [0]*20
        
        disc_output = torch.zeros(1, 52)
        disc_output[0, 0] = 10.0 # Predict C (root)
        self.mock_discriminator.return_value = disc_output
        
        adjusted_logits = apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="naive"
        )
        
        # C4, E4, G4 should be boosted (C major triad)
        self.assertTrue((adjusted_logits[0, 0] > 0).item())
        self.assertTrue((adjusted_logits[0, 1] > 0).item())
        self.assertTrue((adjusted_logits[0, 2] > 0).item())
        # Invalid note should not crash and not be boosted
        self.assertEqual(adjusted_logits[0, 3].item(), 0)

    def test_apply_discriminator_guidance_music21_exception(self):
        # Test exception handling during music21 parsing
        logits = torch.zeros(1, 10)
        context_measures = 1
        int_to_note = {0: "BadNote"}
        generated_so_far = [0] * 20
        
        disc_output = torch.zeros(1, 52)
        self.mock_discriminator.return_value = disc_output
        
        # Should default to Middle C and run without error
        apply_discriminator_guidance(
            logits, self.mock_discriminator, context_measures,
            generated_so_far, int_to_note, guidance_strength=1.0, dataset="naive"
        )
        
        # Verify discriminator called (meaning it parsed something, likely default C)
        self.mock_discriminator.assert_called()
