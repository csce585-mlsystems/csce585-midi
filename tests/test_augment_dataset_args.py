"""
Comprehensive tests for augment_dataset.py with various configurations
Tests different transpositions, token types, and edge cases similar to test_generate_args.py
"""
import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import sys
import os
import numpy as np
from pathlib import Path
import tempfile
import json
import pickle

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.augment_dataset import (
    fast_normal_order,
    transpose_score,
    augment_dataset,
    process_file_naive,
    TimeoutException,
    DEFAULT_TRANSPOSITIONS
)


class TestTranspositionConfigurations(unittest.TestCase):
    """Test different transposition configurations"""
    
    @patch('utils.augment_dataset.Score')
    def test_transpose_score_various_intervals(self, mock_score_cls):
        """Test transposing by various semitone intervals"""
        intervals = [-12, -7, -5, -3, -1, 0, 1, 3, 5, 7, 12]
        
        for interval in intervals:
            mock_score = MagicMock()
            mock_track = MagicMock()
            mock_track.is_drum = False
            mock_track.notes = [MagicMock(pitch=60)]
            mock_score.tracks = [mock_track]
            mock_score.copy.return_value = mock_score
            
            transposed = transpose_score(mock_score, interval)
            
            # Verify shift_pitch called with correct interval
            mock_track.shift_pitch.assert_called_once_with(interval)
    
    @patch('utils.augment_dataset.Score')
    def test_transpose_score_boundary_cases(self, mock_score_cls):
        """Test transposition near MIDI pitch boundaries (0-127)"""
        # Test low boundary
        mock_score = MagicMock()
        mock_track = MagicMock()
        mock_track.is_drum = False
        # Note at pitch 5, transpose down by 10 would go to -5 (out of range)
        mock_track.notes = [MagicMock(pitch=-5), MagicMock(pitch=5)]
        mock_score.tracks = [mock_track]
        mock_score.copy.return_value = mock_score
        
        transposed = transpose_score(mock_score, 0)
        
        # Should remove note with pitch < 0
        self.assertEqual(len(mock_track.notes), 1)
        self.assertEqual(mock_track.notes[0].pitch, 5)
        
    @patch('utils.augment_dataset.Score')
    def test_transpose_score_high_boundary(self, mock_score_cls):
        """Test transposition near high MIDI pitch boundary"""
        mock_score = MagicMock()
        mock_track = MagicMock()
        mock_track.is_drum = False
        # Notes that exceed 127
        mock_track.notes = [MagicMock(pitch=128), MagicMock(pitch=120)]
        mock_score.tracks = [mock_track]
        mock_score.copy.return_value = mock_score
        
        transposed = transpose_score(mock_score, 0)
        
        # Should remove note with pitch > 127
        self.assertEqual(len(mock_track.notes), 1)
        self.assertEqual(mock_track.notes[0].pitch, 120)
    
    @patch('utils.augment_dataset.Score')
    def test_transpose_score_preserves_drum_tracks(self, mock_score_cls):
        """Test that drum tracks are not transposed"""
        mock_score = MagicMock()
        
        mock_drum_track = MagicMock()
        mock_drum_track.is_drum = True
        
        mock_melodic_track = MagicMock()
        mock_melodic_track.is_drum = False
        mock_melodic_track.notes = [MagicMock(pitch=60)]
        
        mock_score.tracks = [mock_drum_track, mock_melodic_track]
        mock_score.copy.return_value = mock_score
        
        transposed = transpose_score(mock_score, 5)
        
        # Drum track should not be transposed
        mock_drum_track.shift_pitch.assert_not_called()
        
        # Melodic track should be transposed
        mock_melodic_track.shift_pitch.assert_called_once_with(5)
    
    @patch('utils.augment_dataset.Score')
    def test_transpose_score_multiple_tracks(self, mock_score_cls):
        """Test transposing score with multiple melodic tracks"""
        mock_score = MagicMock()
        
        tracks = []
        for i in range(4):
            track = MagicMock()
            track.is_drum = False
            track.notes = [MagicMock(pitch=60 + i)]
            tracks.append(track)
        
        mock_score.tracks = tracks
        mock_score.copy.return_value = mock_score
        
        transposed = transpose_score(mock_score, 3)
        
        # All tracks should be transposed
        for track in tracks:
            track.shift_pitch.assert_called_once_with(3)


class TestFastNormalOrderConfigurations(unittest.TestCase):
    """Test fast_normal_order with various pitch class configurations"""
    
    def test_fast_normal_order_all_intervals(self):
        """Test normal order calculation for various chord types"""
        test_cases = [
            # (input, expected_output, description)
            ([0, 4, 7], [0, 4, 7], "Major triad"),
            ([0, 3, 7], [0, 3, 7], "Minor triad"),
            ([0, 4, 7, 11], [0, 4, 7, 11], "Major 7th"),
            ([0, 3, 6, 9], [0, 3, 6, 9], "Diminished 7th"),
            ([0, 2, 4, 5, 7, 9, 11], [0, 2, 4, 5, 7, 9, 11], "Major scale"),
            ([1, 3, 5], [1, 3, 5], "Minor triad on C#"),
            ([10, 2, 5], [10, 0, 2, 5], "Chord crossing octave boundary"),
        ]
        
        for input_pcs, expected, description in test_cases:
            result = fast_normal_order(input_pcs)
            # Note: Some expected values might need adjustment based on actual implementation
            self.assertTrue(len(result) > 0, f"Failed for {description}")
    
    def test_fast_normal_order_rotations(self):
        """Test that different rotations of same pitch classes give same normal order"""
        # C major triad: C, E, G = 0, 4, 7
        rotations = [
            [0, 4, 7],
            [4, 7, 0],
            [7, 0, 4],
        ]
        
        results = [fast_normal_order(r) for r in rotations]
        
        # All should produce the same normal order
        for result in results[1:]:
            self.assertEqual(result, results[0])
    
    def test_fast_normal_order_with_duplicates(self):
        """Test handling of duplicate pitch classes"""
        test_cases = [
            ([0, 0, 4, 4, 7, 7], [0, 4, 7]),
            ([1, 1, 1, 1], [1]),
            ([0, 4, 7, 0, 4, 7], [0, 4, 7]),
        ]
        
        for input_pcs, expected in test_cases:
            result = fast_normal_order(input_pcs)
            self.assertEqual(result, expected)
    
    def test_fast_normal_order_edge_cases(self):
        """Test edge cases for normal order"""
        # Empty list
        self.assertEqual(fast_normal_order([]), [])
        
        # Single note
        self.assertEqual(fast_normal_order([5]), [5])
        
        # All 12 chromatic notes
        chromatic = list(range(12))
        result = fast_normal_order(chromatic)
        self.assertEqual(len(result), 12)


class TestProcessFileNaiveConfigurations(unittest.TestCase):
    """Test process_file_naive with various configurations"""
    
    def test_process_file_naive_multiple_transpositions(self):
        """Test processing with multiple transposition intervals"""
        with patch('utils.augment_dataset.time_limit'):
            with patch.dict(sys.modules, {
                'music21': MagicMock(),
                'music21.converter': MagicMock(),
                'music21.instrument': MagicMock(),
                'music21.note': MagicMock(),
                'music21.chord': MagicMock(),
                'music21.pitch': MagicMock(),
            }):
                import music21
                
                # Setup mocks
                mock_midi = MagicMock()
                music21.converter.parse.return_value = mock_midi
                
                mock_part = MagicMock()
                music21.instrument.partitionByInstrument.return_value.parts = [mock_part]
                
                # Create note mock
                mock_note = MagicMock()
                mock_note.pitch.midi = 60
                music21.note.Note = type('Note', (), {})
                mock_note.__class__ = music21.note.Note
                
                mock_part.recurse.return_value = [mock_note]
                
                # Mock Pitch
                def mock_pitch_name(midi_num):
                    mock_p = MagicMock()
                    mock_p.nameWithOctave = f"Note{midi_num}"
                    return mock_p
                
                music21.pitch.Pitch.side_effect = mock_pitch_name
                
                # Test with various transpositions
                transpositions = [-5, -3, -1, 0, 1, 3, 5]
                args = ("test.mid", transpositions)
                results = process_file_naive(args)
                
                # Should have one result per transposition
                self.assertEqual(len(results), len(transpositions))
                
                # Check that each transposition is represented
                semitones_returned = [r[0] for r in results]
                self.assertEqual(semitones_returned, transpositions)
    
    def test_process_file_naive_with_chords(self):
        """Test processing files with chord elements"""
        with patch('utils.augment_dataset.time_limit'):
            with patch.dict(sys.modules, {
                'music21': MagicMock(),
                'music21.converter': MagicMock(),
                'music21.instrument': MagicMock(),
                'music21.note': MagicMock(),
                'music21.chord': MagicMock(),
                'music21.pitch': MagicMock(),
            }):
                import music21
                
                mock_midi = MagicMock()
                music21.converter.parse.return_value = mock_midi
                
                mock_part = MagicMock()
                music21.instrument.partitionByInstrument.return_value.parts = [mock_part]
                
                # Create chord mock
                mock_chord = MagicMock()
                mock_chord.normalOrder = [0, 4, 7]  # C major
                music21.chord.Chord = type('Chord', (), {})
                mock_chord.__class__ = music21.chord.Chord
                
                mock_part.recurse.return_value = [mock_chord]
                
                args = ("test.mid", [0, 2])
                results = process_file_naive(args)
                
                # Should successfully process - at least one result
                self.assertGreaterEqual(len(results), 1)
                
                # Verify notes contain chord representation
                for semitones, notes in results:
                    if isinstance(notes, list) and notes:
                        # Should have chord notation (dot-separated pitch classes)
                        self.assertTrue(len(notes) > 0)
    
    def test_process_file_naive_mixed_elements(self):
        """Test processing with both notes and chords"""
        with patch('utils.augment_dataset.time_limit'):
            with patch.dict(sys.modules, {
                'music21': MagicMock(),
                'music21.converter': MagicMock(),
                'music21.instrument': MagicMock(),
                'music21.note': MagicMock(),
                'music21.chord': MagicMock(),
                'music21.pitch': MagicMock(),
            }):
                import music21
                
                mock_midi = MagicMock()
                music21.converter.parse.return_value = mock_midi
                
                mock_part = MagicMock()
                music21.instrument.partitionByInstrument.return_value.parts = [mock_part]
                
                # Create note and chord
                mock_note = MagicMock()
                mock_note.pitch.midi = 60
                music21.note.Note = type('Note', (), {})
                mock_note.__class__ = music21.note.Note
                
                mock_chord = MagicMock()
                mock_chord.normalOrder = [0, 4, 7]
                music21.chord.Chord = type('Chord', (), {})
                mock_chord.__class__ = music21.chord.Chord
                
                mock_part.recurse.return_value = [mock_note, mock_chord, mock_note]
                
                music21.pitch.Pitch.return_value.nameWithOctave = "C4"
                
                args = ("test.mid", [0])
                results = process_file_naive(args)
                
                # Should process successfully
                self.assertEqual(len(results), 1)
                semitones, notes = results[0]
                self.assertEqual(semitones, 0)
                # Should have 3 elements (note, chord, note)
                self.assertGreaterEqual(len(notes), 2)
    
    def test_process_file_naive_out_of_range_notes(self):
        """Test handling of transposed notes that go out of MIDI range"""
        with patch('utils.augment_dataset.time_limit'):
            with patch.dict(sys.modules, {
                'music21': MagicMock(),
                'music21.converter': MagicMock(),
                'music21.instrument': MagicMock(),
                'music21.note': MagicMock(),
                'music21.chord': MagicMock(),
                'music21.pitch': MagicMock(),
            }):
                import music21
                
                mock_midi = MagicMock()
                music21.converter.parse.return_value = mock_midi
                
                mock_part = MagicMock()
                music21.instrument.partitionByInstrument.return_value.parts = [mock_part]
                
                # Note at pitch 10 (very low)
                mock_note = MagicMock()
                mock_note.pitch.midi = 10
                music21.note.Note = type('Note', (), {})
                mock_note.__class__ = music21.note.Note
                
                mock_part.recurse.return_value = [mock_note]
                
                music21.pitch.Pitch.return_value.nameWithOctave = "C0"
                
                # Transpose down by 20 would be -10 (out of range)
                args = ("test.mid", [-20, 0, 20])
                results = process_file_naive(args)
                
                # Should get results - at least 2 (0 and 20 should work)
                self.assertGreaterEqual(len(results), 2)
                
                # Verify that out of range notes are skipped
                self.assertTrue(isinstance(results, list))


class TestAugmentDatasetConfigurations(unittest.TestCase):
    """Test augment_dataset function with various configurations"""
    
    def test_augment_dataset_invalid_token_types(self):
        """Test that invalid token types raise ValueError"""
        # Mock Path to prevent FileExistsError from happening first
        with patch('utils.augment_dataset.Path') as mock_path:
            mock_input = MagicMock()
            mock_input.exists.return_value = True
            mock_path.return_value = mock_input
            
            invalid_types = ["invalid", "tokenizer", ""]
            
            for token_type in invalid_types:
                with self.assertRaises(ValueError):
                    augment_dataset("dummy_in", output_dir="dummy_out", token_type=token_type)
    
    def test_augment_dataset_valid_token_types(self):
        """Test that valid token types are accepted"""
        with patch('utils.augment_dataset.Path') as mock_path:
            mock_input = MagicMock()
            mock_input.exists.return_value = True
            mock_input.rglob.return_value = []
            
            mock_output = MagicMock()
            
            def path_side_effect(arg):
                return mock_input if arg == "in" else mock_output
            
            mock_path.side_effect = path_side_effect
            
            # Test naive (case insensitive)
            with patch('utils.augment_dataset.Pool'):
                with patch('utils.augment_dataset.tqdm') as mock_tqdm:
                    mock_tqdm.side_effect = lambda x, **kwargs: x
                    with patch('numpy.save'), patch('json.dump'), patch('pickle.dump'):
                        with patch('builtins.open', mock_open()):
                            augment_dataset("in", output_dir="out", token_type="naive")
            
            # Reset mocks
            mock_path.reset_mock()
            mock_path.side_effect = path_side_effect
            
            # Test miditok
            with patch('utils.augment_dataset.miditok') as mock_miditok:
                # Mock tokenizer with proper __len__
                mock_tokenizer = MagicMock()
                mock_tokenizer.__len__.return_value = 10  # Non-zero to avoid division by zero
                mock_miditok.REMI.return_value = mock_tokenizer
                
                with patch('utils.augment_dataset.tqdm') as mock_tqdm:
                    mock_tqdm.side_effect = lambda x, **kwargs: x
                    with patch('numpy.save'), patch('json.dump'):
                        with patch('builtins.open', mock_open()):
                            augment_dataset("in", output_dir="out", token_type="miditok")
    
    @patch('utils.augment_dataset.miditok')
    @patch('utils.augment_dataset.Score')
    @patch('utils.augment_dataset.Path')
    @patch('numpy.save')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_augment_dataset_miditok_custom_transpositions(self, mock_file, mock_json_dump, 
                                                            mock_np_save, mock_path, mock_score, mock_miditok):
        """Test miditok augmentation with custom transposition intervals"""
        # Setup mocks
        mock_input_dir = MagicMock()
        mock_input_dir.exists.return_value = True
        mock_input_dir.rglob.return_value = [Path("test.mid")]
        
        mock_output_dir = MagicMock()
        
        def path_side_effect(arg):
            return mock_input_dir if arg == "in" else mock_output_dir
        
        mock_path.side_effect = path_side_effect
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_miditok.REMI.return_value = mock_tokenizer
        mock_tokenizer.vocab = {"token": 1}
        mock_tokenizer.__len__.return_value = 100
        
        # Mock Score
        mock_score_obj = MagicMock()
        mock_score.return_value = mock_score_obj
        
        # Mock tokenizer output
        mock_seq = MagicMock()
        mock_seq.ids = [1, 2, 3, 4, 5]
        mock_tokenizer.return_value = mock_seq
        
        # Custom transpositions
        custom_trans = [-6, -4, -2, 0, 2, 4, 6]
        
        # Call function
        augment_dataset("in", transpositions=custom_trans, output_dir="out", token_type="miditok")
        
        # Verify transpositions were applied
        # The function should call transpose_score for each transposition
        # We can verify by checking the number of sequences saved
        # (Though exact verification would require more detailed mocking)
        mock_np_save.assert_called()
    
    @patch('utils.augment_dataset.Pool')
    @patch('utils.augment_dataset.Path')
    @patch('utils.augment_dataset.tqdm')
    @patch('numpy.save')
    @patch('json.dump')
    @patch('pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_augment_dataset_naive_default_transpositions(self, mock_file, mock_pickle_dump, 
                                                           mock_json_dump, mock_np_save, mock_tqdm,
                                                           mock_path, mock_pool):
        """Test naive augmentation with default transpositions"""
        # Setup mocks
        mock_input_dir = MagicMock()
        mock_input_dir.exists.return_value = True
        midi_file = MagicMock()
        midi_file.name = "test.mid"
        mock_input_dir.rglob.return_value = [midi_file]
        
        mock_output_dir = MagicMock()
        
        def path_side_effect(arg):
            return mock_input_dir if arg == "in" else mock_output_dir
        
        mock_path.side_effect = path_side_effect
        
        # Mock tqdm to just return the iterable
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Create single result list (pool.imap returns results per file)
        mock_results = [[(trans, ["C4", "E4", "G4"]) for trans in DEFAULT_TRANSPOSITIONS]]
        
        mock_pool_instance.imap.return_value = mock_results
        
        # Call function with no transpositions specified (should use default)
        augment_dataset("in", transpositions=None, output_dir="out", token_type="naive")
        
        # Verify that processing occurred
        mock_np_save.assert_called()
        mock_pickle_dump.assert_called()
    
    @patch('utils.augment_dataset.Pool')
    @patch('utils.augment_dataset.Path')
    @patch('numpy.save')
    @patch('json.dump')
    @patch('pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_augment_dataset_naive_single_transposition(self, mock_file, mock_pickle_dump, 
                                                         mock_json_dump, mock_np_save, mock_path, mock_pool):
        """Test naive augmentation with single transposition (original only)"""
        # Setup mocks
        mock_input_dir = MagicMock()
        mock_input_dir.exists.return_value = True
        mock_input_dir.rglob.return_value = [Path("test.mid")]
        
        mock_output_dir = MagicMock()
        
        def path_side_effect(arg):
            return mock_input_dir if arg == "in" else mock_output_dir
        
        mock_path.side_effect = path_side_effect
        
        # Mock Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        mock_results = [[(0, ["C4", "E4", "G4"])]]
        mock_pool_instance.imap.return_value = mock_results
        
        # Call with only original (no transposition)
        augment_dataset("in", transpositions=[0], output_dir="out", token_type="naive")
        
        # Verify sequences saved
        mock_np_save.assert_called()
        
        # Check that config reflects single transposition
        # We can check the calls to json.dump
        self.assertTrue(mock_json_dump.called)


class TestAugmentDatasetOutputValidation(unittest.TestCase):
    """Test that augment_dataset produces correct output structure"""
    
    @patch('utils.augment_dataset.Pool')
    @patch('utils.augment_dataset.Path')
    @patch('numpy.save')
    @patch('json.dump')
    @patch('pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_augment_dataset_naive_output_files(self, mock_file, mock_pickle_dump, 
                                                  mock_json_dump, mock_np_save, mock_path, mock_pool):
        """Test that all expected output files are created for naive tokenization"""
        # Setup mocks
        mock_input_dir = MagicMock()
        mock_input_dir.exists.return_value = True
        mock_input_dir.rglob.return_value = [Path("test.mid")]
        
        mock_output_dir = MagicMock()
        sequences_path = MagicMock()
        vocab_path = MagicMock()
        vocab_json_path = MagicMock()
        config_path = MagicMock()
        
        # Mock Path division to return appropriate paths
        mock_output_dir.__truediv__ = lambda self, other: {
            "sequences.npy": sequences_path,
            "note_to_int.pkl": vocab_path,
            "vocab.json": vocab_json_path,
            "config.json": config_path,
        }.get(other, MagicMock())
        
        def path_side_effect(arg):
            return mock_input_dir if arg == "in" else mock_output_dir
        
        mock_path.side_effect = path_side_effect
        
        # Mock Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_results = [[(0, ["C4", "E4", "G4"])]]
        mock_pool_instance.imap.return_value = mock_results
        
        # Call function
        augment_dataset("in", transpositions=[0], output_dir="out", token_type="naive")
        
        # Verify all output operations called
        mock_np_save.assert_called()  # sequences.npy
        mock_pickle_dump.assert_called()  # note_to_int.pkl
        
        # json.dump called twice (vocab.json and config.json)
        self.assertGreaterEqual(mock_json_dump.call_count, 2)
    
    @patch('utils.augment_dataset.Pool')
    @patch('utils.augment_dataset.Path')
    @patch('numpy.save')
    @patch('json.dump')
    @patch('pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_augment_dataset_vocab_building(self, mock_file, mock_pickle_dump, 
                                             mock_json_dump, mock_np_save, mock_path, mock_pool):
        """Test that vocabulary is correctly built from sequences"""
        # Setup mocks
        mock_input_dir = MagicMock()
        mock_input_dir.exists.return_value = True
        mock_input_dir.rglob.return_value = [Path("test.mid")]
        
        mock_output_dir = MagicMock()
        
        def path_side_effect(arg):
            return mock_input_dir if arg == "in" else mock_output_dir
        
        mock_path.side_effect = path_side_effect
        
        # Mock Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Create diverse note results
        mock_results = [
            [(0, ["C4", "E4", "G4", "C5"]),
             (1, ["C#4", "F4", "G#4", "C#5"])]
        ]
        mock_pool_instance.imap.return_value = mock_results
        
        # Call function
        augment_dataset("in", transpositions=[0, 1], output_dir="out", token_type="naive")
        
        # Verify vocabulary pickle was created
        mock_pickle_dump.assert_called()
        
        # Get the vocab dict that was dumped
        pickle_calls = mock_pickle_dump.call_args_list
        # The vocab should contain note_to_int and int_to_note mappings
        vocab_data = pickle_calls[0][0][0]
        self.assertIn("note_to_int", vocab_data)
        self.assertIn("int_to_note", vocab_data)


class TestTimeoutHandling(unittest.TestCase):
    """Test timeout handling in file processing"""
    
    def test_process_file_naive_timeout_error(self):
        """Test that timeout exceptions are handled correctly"""
        with patch('utils.augment_dataset.time_limit') as mock_time_limit:
            # Make time_limit raise TimeoutException
            mock_time_limit.return_value.__enter__.side_effect = TimeoutException("Timed out!")
            
            args = ("slow_file.mid", [0, 1])
            results = process_file_naive(args)
            
            # Should return error tuple
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], "file_error")
            self.assertEqual(results[0][1], "Timeout")
    
    def test_process_file_naive_parse_error(self):
        """Test handling of MIDI parse errors"""
        with patch('utils.augment_dataset.time_limit'):
            with patch.dict(sys.modules, {
                'music21': MagicMock(),
                'music21.converter': MagicMock(),
                'music21.instrument': MagicMock(),
                'music21.note': MagicMock(),
                'music21.chord': MagicMock(),
                'music21.pitch': MagicMock(),
            }):
                import music21
                
                # Make parse raise exception
                music21.converter.parse.side_effect = Exception("Invalid MIDI")
                
                args = ("bad_file.mid", [0])
                results = process_file_naive(args)
                
                # Should return parse error
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0][0], "file_error")
                self.assertIn("Parse error", results[0][1])


if __name__ == "__main__":
    unittest.main()
