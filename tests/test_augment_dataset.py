import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import sys
import os
import numpy as np
from pathlib import Path
import tempfile
import json
import pickle

# Add the project root to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.augment_dataset import (
    fast_normal_order,
    transpose_score,
    augment_dataset,
    process_file_naive,
    TimeoutException
)

class TestAugmentDataset(unittest.TestCase):

    def test_fast_normal_order(self):
        # Test empty
        self.assertEqual(fast_normal_order([]), [])
        
        # Test single
        self.assertEqual(fast_normal_order([0]), [0])
        
        # Test C Major (0, 4, 7)
        self.assertEqual(fast_normal_order([0, 4, 7]), [0, 4, 7])
        self.assertEqual(fast_normal_order([4, 7, 0]), [0, 4, 7])
        self.assertEqual(fast_normal_order([7, 0, 4]), [0, 4, 7])
        
        # Test duplicates
        self.assertEqual(fast_normal_order([0, 4, 7, 0]), [0, 4, 7])
        
        # Test specific case where rotation matters
        # C, D, E (0, 2, 4) -> span 4
        self.assertEqual(fast_normal_order([0, 2, 4]), [0, 2, 4])
        
        # Test wrap around
        # B, C (11, 0) -> should be [11, 0] or [0, 11]?
        # Normal order of B, C is [11, 0] because 0-11 is interval 1 (mod 12)
        # Wait, music21 normal order for [11, 0] is usually [11, 0] if we consider 11 as bottom?
        # Let's trace: sorted unique: [0, 11]. doubled: [0, 11, 12, 23].
        # n=2.
        # i=0: [0, 11]. span = 11 - 0 = 11.
        # i=1: [11, 12]. span = 12 - 11 = 1.
        # min_span = 1. best_rotation = [11, 12] % 12 = [11, 0].
        self.assertEqual(fast_normal_order([0, 11]), [11, 0])

    @patch('utils.augment_dataset.Score')
    def test_transpose_score(self, mock_score_cls):
        # Setup mock score
        mock_score = MagicMock()
        mock_track_normal = MagicMock()
        mock_track_normal.is_drum = False
        mock_track_normal.notes = [MagicMock(pitch=60), MagicMock(pitch=128)] # 128 is out of range
        
        mock_track_drum = MagicMock()
        mock_track_drum.is_drum = True
        
        mock_score.tracks = [mock_track_normal, mock_track_drum]
        mock_score.copy.return_value = mock_score
        
        # Call function
        transposed = transpose_score(mock_score, 2)
        
        # Verify copy
        mock_score.copy.assert_called_once()
        
        # Verify shift_pitch called on normal track
        mock_track_normal.shift_pitch.assert_called_once_with(2)
        
        # Verify NOT called on drum track
        mock_track_drum.shift_pitch.assert_not_called()
        
        # Verify out of range notes removal logic
        # The function iterates and removes notes with pitch < 0 or > 127
        # We mocked one note with pitch 128, it should be removed.
        # Since mock_track_normal.notes is a real list, we can check its content.
        self.assertEqual(len(mock_track_normal.notes), 1)
        self.assertEqual(mock_track_normal.notes[0].pitch, 60)

    @patch('utils.augment_dataset.time_limit')
    def test_process_file_naive_success(self, mock_time_limit):
        # Mock music21 imports
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
            
            # Mock partitionByInstrument
            mock_part = MagicMock()
            music21.instrument.partitionByInstrument.return_value.parts = [mock_part]
            
            # Mock recurse elements
            mock_note = MagicMock()
            mock_note.pitch.midi = 60
            # Make isinstance work for Note
            music21.note.Note = type('Note', (), {})
            mock_note.__class__ = music21.note.Note
            
            mock_chord = MagicMock()
            mock_chord.normalOrder = [0, 4, 7]
            # Make isinstance work for Chord
            music21.chord.Chord = type('Chord', (), {})
            mock_chord.__class__ = music21.chord.Chord
            
            mock_part.recurse.return_value = [mock_note, mock_chord]
            
            # Mock Pitch name
            music21.pitch.Pitch.return_value.nameWithOctave = "C4"
            
            # Call function
            args = ("dummy.mid", [0, 1])
            results = process_file_naive(args)
            
            # Verify results
            # We expect 2 results (one for each transposition)
            self.assertEqual(len(results), 2)
            
            # Transposition 0
            self.assertEqual(results[0][0], 0)
            # Note 60 -> C4 (mocked). Chord [0,4,7] -> [0,4,7]
            # The function uses fast_normal_order internally for chords
            
            # Transposition 1
            self.assertEqual(results[1][0], 1)
            # Note 61. Chord [1, 5, 8]

    def test_process_file_naive_timeout(self):
        # Mock time_limit to raise TimeoutException
        with patch('utils.augment_dataset.time_limit') as mock_ctx:
            mock_ctx.side_effect = TimeoutException("Timed out!")
            
            args = ("dummy.mid", [0])
            results = process_file_naive(args)
            
            self.assertEqual(results, [("file_error", "Timeout")])

    @patch('utils.augment_dataset.Path')
    def test_augment_dataset_validation(self, mock_path):
        # Test invalid token type
        with self.assertRaises(ValueError):
            augment_dataset("in", output_dir="out", token_type="invalid")
            
        # Test input dir not exists
        mock_path.return_value.exists.return_value = False
        with self.assertRaises(FileExistsError):
            augment_dataset("in", output_dir="out", token_type="naive")

    @patch('utils.augment_dataset.miditok')
    @patch('utils.augment_dataset.Score')
    @patch('utils.augment_dataset.Path')
    @patch('numpy.save')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_augment_dataset_miditok(self, mock_file, mock_json_dump, mock_np_save, mock_path, mock_score, mock_miditok):
        # Setup mocks
        mock_input_dir = MagicMock()
        mock_input_dir.exists.return_value = True
        mock_input_dir.rglob.return_value = [Path("test.mid")]
        
        mock_output_dir = MagicMock()
        
        def path_side_effect(arg):
            if arg == "in": return mock_input_dir
            if arg == "out": return mock_output_dir
            return MagicMock()
            
        mock_path.side_effect = path_side_effect
        
        # Mock Tokenizer
        mock_tokenizer = MagicMock()
        mock_miditok.REMI.return_value = mock_tokenizer
        mock_tokenizer.vocab = {"A": 1}
        mock_tokenizer.__len__.return_value = 10
        
        # Mock Score
        mock_score_obj = MagicMock()
        mock_score.return_value = mock_score_obj
        
        # Mock tokenizer call
        mock_seq = MagicMock()
        mock_seq.ids = [1, 2, 3]
        mock_tokenizer.return_value = mock_seq
        
        # Call function
        augment_dataset("in", transpositions=[0], output_dir="out", token_type="miditok")
        
        # Verify
        mock_miditok.REMI.assert_called_once()
        mock_score.assert_called()
        mock_tokenizer.assert_called()
        mock_np_save.assert_called()
        
        # Verify config saved
        self.assertTrue(mock_json_dump.called)

    @patch('utils.augment_dataset.Pool')
    @patch('utils.augment_dataset.Path')
    @patch('numpy.save')
    @patch('json.dump')
    @patch('pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_augment_dataset_naive(self, mock_file, mock_pickle_dump, mock_json_dump, mock_np_save, mock_path, mock_pool):
        # Setup mocks
        mock_input_dir = MagicMock()
        mock_input_dir.exists.return_value = True
        mock_input_dir.rglob.return_value = [Path("test.mid")]
        
        mock_output_dir = MagicMock()
        
        def path_side_effect(arg):
            if arg == "in": return mock_input_dir
            if arg == "out": return mock_output_dir
            return MagicMock()
            
        mock_path.side_effect = path_side_effect
        
        # Mock Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Mock pool results
        # List of (semitones, output)
        mock_results = [
            [(0, ["C4", "E4", "G4"]), (1, ["C#4", "F4", "G#4"])]
        ]
        mock_pool_instance.imap.return_value = mock_results
        
        # Call function
        augment_dataset("in", transpositions=[0, 1], output_dir="out", token_type="naive")
        
        # Verify
        mock_pool.assert_called()
        mock_np_save.assert_called()
        
        # Verify pickle dump (vocab)
        self.assertTrue(mock_pickle_dump.called)
        
        # Verify json dump (config)
        self.assertTrue(mock_json_dump.called)

if __name__ == '__main__':
    unittest.main()
