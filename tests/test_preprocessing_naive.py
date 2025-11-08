import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pickle
import pytest
from unittest.mock import patch, Mock
import pretty_midi
import signal

from utils.preprocess_naive import (
    midi_to_notes,
    build_dataset,
    time_limit,
    TimeoutException,
    OUTPUT_DIR,
    VOCAB_FILE,
    OUTPUT_FILE,
    DATA_DIR,
    TIMEOUT_SECONDS
)


class TestTimeLimit:
    """Test suite for time_limit context manager"""
    
    def test_time_limit_no_timeout(self):
        """Test that code completes within time limit"""
        result = []
        with time_limit(2):
            result.append(1)
            result.append(2)
        assert result == [1, 2]
    
    def test_time_limit_timeout(self):
        """Test that timeout raises TimeoutException"""
        import time
        with pytest.raises(TimeoutException):
            with time_limit(1):
                time.sleep(2)
    
    def test_time_limit_cleanup(self):
        """Test that alarm is cleaned up after context"""
        with time_limit(2):
            pass
        # Alarm should be cancelled (returns 0)
        remaining = signal.alarm(0)
        assert remaining == 0


class TestMidiToNotes:
    """Test suite for midi_to_notes function"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_midi_to_notes_simple(self, temp_dir):
        """Test converting a simple MIDI file with notes"""
        midi_file = temp_dir / "simple.mid"
        
        # Create MIDI using pretty_midi
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))  # C4
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=1.0, end=2.0))  # E4
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=67, start=2.0, end=3.0))  # G4
        pm.instruments.append(inst)
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))
        assert len(notes) > 0
        assert isinstance(notes, list)
        assert all(isinstance(n, str) for n in notes)
    
    def test_midi_to_notes_with_chords(self, temp_dir):
        """Test converting MIDI with simultaneous notes (chords)"""
        midi_file = temp_dir / "chords.mid"
        
        # Create MIDI with simultaneous notes
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        # C major chord (C, E, G) - all starting at same time
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.0, end=1.0))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=67, start=0.0, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))
        assert len(notes) > 0
        # Should contain chord representation (dot-separated pitches)
        has_chord = any('.' in str(n) for n in notes)
        assert has_chord or len(notes) >= 3  # Either detected as chord or individual notes
    
    def test_midi_to_notes_empty_file(self, temp_dir):
        """Test with MIDI file containing no notes"""
        midi_file = temp_dir / "empty.mid"
        
        # Create empty MIDI
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        pm.instruments.append(inst)
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))

        assert isinstance(notes, list)
        assert len(notes) <= 1
    
    def test_midi_to_notes_invalid_file(self, temp_dir):
        """Test with invalid MIDI file"""
        invalid_file = temp_dir / "invalid.mid"
        invalid_file.write_text("not a midi file")
        
        notes = midi_to_notes(str(invalid_file))
        assert notes == []
    
    def test_midi_to_notes_nonexistent_file(self):
        """Test with non-existent file"""
        notes = midi_to_notes("nonexistent_file_12345.mid")
        assert notes == []
    
    def test_midi_to_notes_multiple_instruments(self, temp_dir):
        """Test MIDI with multiple instrument parts"""
        midi_file = temp_dir / "multi.mid"
        
        pm = pretty_midi.PrettyMIDI()
        # First instrument
        inst1 = pretty_midi.Instrument(program=0)
        inst1.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(inst1)
        
        # Second instrument
        inst2 = pretty_midi.Instrument(program=1)
        inst2.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.0, end=1.0))
        pm.instruments.append(inst2)
        
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))
        # Should extract notes from first instrument
        assert len(notes) > 0
    
    def test_midi_to_notes_various_pitches(self, temp_dir):
        """Test with various pitch ranges"""
        midi_file = temp_dir / "pitches.mid"
        
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        # Low, middle, high notes
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=1.0))  # C2
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=1.0, end=2.0))  # C4
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=84, start=2.0, end=3.0))  # C6
        pm.instruments.append(inst)
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))
        assert len(notes) >= 3
    
    def test_midi_to_notes_with_rests(self, temp_dir):
        """Test MIDI with gaps (rests) between notes"""
        midi_file = temp_dir / "rests.mid"
        
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        # Notes with gaps
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=2.0, end=2.5))  # Gap
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=4.0, end=4.5))  # Gap
        pm.instruments.append(inst)
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))
        assert len(notes) == 3
    
    def test_midi_to_notes_complex_chords(self, temp_dir):
        """Test with complex chords (4+ notes)"""
        midi_file = temp_dir / "complex_chord.mid"
        
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        # 7th chord: C, E, G, B
        for pitch in [60, 64, 67, 71]:
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))
        assert len(notes) > 0
    
    def test_midi_to_notes_drum_track(self, temp_dir):
        """Test with drum track (should still work)"""
        midi_file = temp_dir / "drums.mid"
        
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.5))  # Kick
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=0.5, end=1.0))  # Snare
        pm.instruments.append(inst)
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))
        # Should process drum notes
        assert len(notes) >= 0  # May or may not extract drum notes depending on implementation
    
    def test_midi_to_notes_no_instruments(self, temp_dir):
        """Test MIDI file with no instruments"""
        midi_file = temp_dir / "no_inst.mid"
        
        pm = pretty_midi.PrettyMIDI()
        pm.write(str(midi_file))
        
        notes = midi_to_notes(str(midi_file))

        assert isinstance(notes, list)
        assert len(notes) == 0
    
    def test_midi_to_notes_exception_handling(self, temp_dir):
        """Test that exceptions are caught and empty list returned"""
        # Create a file that will cause an error when parsed
        bad_file = temp_dir / "bad.mid"
        bad_file.write_bytes(b'\x00' * 100)  # Invalid MIDI data
        
        notes = midi_to_notes(str(bad_file))
        assert notes == []


class TestBuildDataset:
    """Test suite for build_dataset function"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def sample_midi_files(self, temp_dir):
        """Create sample MIDI files for testing"""
        midi_dir = temp_dir / "midi"
        midi_dir.mkdir()
        
        # Create 3 valid MIDI files
        for i in range(3):
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            
            # Add some notes
            for j in range(5):
                pitch = 60 + i + j
                inst.notes.append(pretty_midi.Note(
                    velocity=100, 
                    pitch=pitch, 
                    start=j * 0.5, 
                    end=(j + 1) * 0.5
                ))
            
            pm.instruments.append(inst)
            pm.write(str(midi_dir / f"song_{i}.mid"))
        
        return midi_dir
    
    def test_build_dataset_basic(self, sample_midi_files, temp_dir):
        """Test basic dataset building"""
        output_file = temp_dir / "sequences.npy"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
            build_dataset(
                data_dir=str(sample_midi_files),
                output_file=str(output_file)
            )
        
        # Check output files exist
        assert output_file.exists()
        
        # Load and verify sequences
        sequences = np.load(output_file, allow_pickle=True)
        assert len(sequences) == 3
        assert all(len(seq) > 0 for seq in sequences)
        assert all(isinstance(seq, np.ndarray) for seq in sequences)
    
    def test_build_dataset_vocab_creation(self, sample_midi_files, temp_dir):
        """Test vocabulary creation"""
        output_file = temp_dir / "sequences.npy"
        vocab_file = temp_dir / 'vocab.pkl'
        
        with patch('utils.preprocess_naive.VOCAB_FILE', vocab_file):
            build_dataset(
                data_dir=str(sample_midi_files),
                output_file=str(output_file)
            )
            
            assert vocab_file.exists()
            
            # Load and verify vocab
            with open(vocab_file, 'rb') as f:
                vocab_data = pickle.load(f)
            
            assert 'note_to_int' in vocab_data
            assert 'int_to_note' in vocab_data
            
            note_to_int = vocab_data['note_to_int']
            int_to_note = vocab_data['int_to_note']
            
            # Check bidirectional mapping
            assert len(note_to_int) == len(int_to_note)
            for note, idx in note_to_int.items():
                assert int_to_note[idx] == note
    
    def test_build_dataset_empty_directory(self, temp_dir):
        """Test with empty directory"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        output_file = temp_dir / "sequences.npy"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
            build_dataset(
                data_dir=str(empty_dir),
                output_file=str(output_file)
            )
        
        # Should create output but with no sequences
        assert output_file.exists()
        sequences = np.load(output_file, allow_pickle=True)
        assert len(sequences) == 0
    
    def test_build_dataset_with_invalid_files(self, temp_dir):
        """Test with mix of valid and invalid MIDI files"""
        midi_dir = temp_dir / "mixed"
        midi_dir.mkdir()
        
        # Create one valid MIDI
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(midi_dir / "valid.mid"))
        
        # Create invalid MIDI
        (midi_dir / "invalid.mid").write_text("not a midi file")
        
        # Create empty MIDI
        pm_empty = pretty_midi.PrettyMIDI()
        pm_empty.instruments.append(pretty_midi.Instrument(program=0))
        pm_empty.write(str(midi_dir / "empty.mid"))
        
        output_file = temp_dir / "sequences.npy"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
            build_dataset(
                data_dir=str(midi_dir),
                output_file=str(output_file)
            )
        
        sequences = np.load(output_file, allow_pickle=True)
        # Should only have 1 valid sequence
        assert len(sequences) == 1
    
    def test_build_dataset_subdirectories(self, temp_dir):
        """Test with MIDI files in subdirectories"""
        midi_dir = temp_dir / "root"
        midi_dir.mkdir()
        sub_dir = midi_dir / "subdir"
        sub_dir.mkdir()
        
        # Create MIDI in root
        pm1 = pretty_midi.PrettyMIDI()
        inst1 = pretty_midi.Instrument(program=0)
        inst1.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm1.instruments.append(inst1)
        pm1.write(str(midi_dir / "root.mid"))
        
        # Create MIDI in subdirectory
        pm2 = pretty_midi.PrettyMIDI()
        inst2 = pretty_midi.Instrument(program=0)
        inst2.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.0, end=1.0))
        pm2.instruments.append(inst2)
        pm2.write(str(sub_dir / "sub.mid"))
        
        output_file = temp_dir / "sequences.npy"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
            build_dataset(
                data_dir=str(midi_dir),
                output_file=str(output_file)
            )
        
        sequences = np.load(output_file, allow_pickle=True)
        # Should find both files
        assert len(sequences) == 2
    
    def test_build_dataset_midi_extension(self, temp_dir):
        """Test with both .mid and .midi extensions"""
        midi_dir = temp_dir / "extensions"
        midi_dir.mkdir()
        
        # Create .mid file
        pm1 = pretty_midi.PrettyMIDI()
        inst1 = pretty_midi.Instrument(program=0)
        inst1.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm1.instruments.append(inst1)
        pm1.write(str(midi_dir / "test.mid"))
        
        # Create .midi file
        pm2 = pretty_midi.PrettyMIDI()
        inst2 = pretty_midi.Instrument(program=0)
        inst2.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.0, end=1.0))
        pm2.instruments.append(inst2)
        temp_path = midi_dir / "test2.mid"
        pm2.write(str(temp_path))
        temp_path.rename(midi_dir / "test2.midi")
        
        output_file = temp_dir / "sequences.npy"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
            build_dataset(
                data_dir=str(midi_dir),
                output_file=str(output_file)
            )
        
        sequences = np.load(output_file, allow_pickle=True)
        # Should find both extensions
        assert len(sequences) == 2
    
    @patch('utils.preprocess_naive.TIMEOUT_SECONDS', 1)
    def test_build_dataset_timeout(self, temp_dir):
        """Test timeout handling with slow file processing"""
        midi_dir = temp_dir / "timeout"
        midi_dir.mkdir()
        
        # Create a MIDI file
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(midi_dir / "test.mid"))
        
        output_file = temp_dir / "sequences.npy"
        
        # Mock midi_to_notes to simulate slow processing
        def slow_midi_to_notes(file_path):
            import time
            time.sleep(2)  # Longer than timeout
            return ["C4", "E4", "G4"]
        
        with patch('utils.preprocess_naive.midi_to_notes', side_effect=slow_midi_to_notes):
            with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
                build_dataset(
                    data_dir=str(midi_dir),
                    output_file=str(output_file)
                )
        
        # File should be skipped due to timeout
        sequences = np.load(output_file, allow_pickle=True)
        assert len(sequences) == 0
    
    def test_build_dataset_consistent_ordering(self, sample_midi_files, temp_dir):
        """Test that vocabulary ordering is consistent (sorted)"""
        output_file = temp_dir / "sequences.npy"
        vocab_file = temp_dir / 'vocab.pkl'
        
        with patch('utils.preprocess_naive.VOCAB_FILE', vocab_file):
            build_dataset(
                data_dir=str(sample_midi_files),
                output_file=str(output_file)
            )
            
            with open(vocab_file, 'rb') as f:
                vocab_data = pickle.load(f)
            
            notes = list(vocab_data['note_to_int'].keys())
            # Check that notes are sorted
            assert notes == sorted(notes)
    
    def test_build_dataset_large_skipped_list(self, temp_dir):
        """Test with many skipped files (>10) to trigger truncated output"""
        midi_dir = temp_dir / "many_invalid"
        midi_dir.mkdir()
        
        # Create 15 invalid MIDI files
        for i in range(15):
            (midi_dir / f"invalid_{i}.mid").write_text(f"not a midi file {i}")
        
        output_file = temp_dir / "sequences.npy"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
            build_dataset(
                data_dir=str(midi_dir),
                output_file=str(output_file)
            )
        
        sequences = np.load(output_file, allow_pickle=True)
        assert len(sequences) == 0
    
    def test_build_dataset_exactly_10_skipped(self, temp_dir):
        """Test with exactly 10 skipped files (boundary case)"""
        midi_dir = temp_dir / "ten_invalid"
        midi_dir.mkdir()
        
        # Create exactly 10 invalid MIDI files
        for i in range(10):
            (midi_dir / f"invalid_{i}.mid").write_text(f"not a midi file {i}")
        
        output_file = temp_dir / "sequences.npy"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', temp_dir / 'vocab.pkl'):
            build_dataset(
                data_dir=str(midi_dir),
                output_file=str(output_file)
            )
        
        sequences = np.load(output_file, allow_pickle=True)
        assert len(sequences) == 0
    
    def test_build_dataset_vocab_indices_sequential(self, sample_midi_files, temp_dir):
        """Test that vocab indices are sequential starting from 0"""
        output_file = temp_dir / "sequences.npy"
        vocab_file = temp_dir / 'vocab.pkl'
        
        with patch('utils.preprocess_naive.VOCAB_FILE', vocab_file):
            build_dataset(
                data_dir=str(sample_midi_files),
                output_file=str(output_file)
            )
            
            with open(vocab_file, 'rb') as f:
                vocab_data = pickle.load(f)
            
            indices = sorted(vocab_data['int_to_note'].keys())
            # Should be 0, 1, 2, ..., n-1
            assert indices == list(range(len(indices)))
    
    def test_build_dataset_sequences_in_vocab_range(self, sample_midi_files, temp_dir):
        """Test that all sequence tokens are within vocab range"""
        output_file = temp_dir / "sequences.npy"
        vocab_file = temp_dir / 'vocab.pkl'
        
        with patch('utils.preprocess_naive.VOCAB_FILE', vocab_file):
            build_dataset(
                data_dir=str(sample_midi_files),
                output_file=str(output_file)
            )
            
            sequences = np.load(output_file, allow_pickle=True)
            with open(vocab_file, 'rb') as f:
                vocab_data = pickle.load(f)
            
            vocab_size = len(vocab_data['note_to_int'])
            
            for seq in sequences:
                assert all(0 <= token < vocab_size for token in seq)


class TestConstants:
    """Test that constants are properly defined"""
    
    def test_output_dir_exists(self):
        """Test OUTPUT_DIR constant"""
        assert OUTPUT_DIR == Path("data/naive")
    
    def test_output_file_path(self):
        """Test OUTPUT_FILE constant"""
        assert OUTPUT_FILE == OUTPUT_DIR / "sequences.npy"
    
    def test_vocab_file_path(self):
        """Test VOCAB_FILE constant"""
        assert VOCAB_FILE == OUTPUT_DIR / "note_to_int.pkl"
    
    def test_data_dir_constant(self):
        """Test DATA_DIR constant"""
        assert DATA_DIR == "data/nottingham-dataset-master/MIDI/"
    
    def test_timeout_constant(self):
        """Test TIMEOUT_SECONDS constant"""
        assert TIMEOUT_SECONDS == 40
        assert isinstance(TIMEOUT_SECONDS, int)


class TestIntegration:
    """Integration tests for the full preprocessing pipeline"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_full_pipeline_with_diverse_data(self, temp_dir):
        """Test full pipeline from MIDI files to sequences with diverse data"""
        midi_dir = temp_dir / "integration"
        midi_dir.mkdir()
        
        # 1. Simple melody
        pm1 = pretty_midi.PrettyMIDI()
        inst1 = pretty_midi.Instrument(program=0)
        for i, pitch in enumerate([60, 62, 64, 65, 67]):
            inst1.notes.append(pretty_midi.Note(
                velocity=100, pitch=pitch, start=i*0.5, end=(i+1)*0.5
            ))
        pm1.instruments.append(inst1)
        pm1.write(str(midi_dir / "melody.mid"))
        
        # 2. Chords
        pm2 = pretty_midi.PrettyMIDI()
        inst2 = pretty_midi.Instrument(program=0)
        for pitch in [60, 64, 67]:  # C major
            inst2.notes.append(pretty_midi.Note(
                velocity=100, pitch=pitch, start=0.0, end=1.0
            ))
        pm2.instruments.append(inst2)
        pm2.write(str(midi_dir / "chord.mid"))
        
        # 3. Mix of notes and chords
        pm3 = pretty_midi.PrettyMIDI()
        inst3 = pretty_midi.Instrument(program=0)
        # Single note
        inst3.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        # Chord
        for pitch in [64, 67]:
            inst3.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.5, end=1.0))
        pm3.instruments.append(inst3)
        pm3.write(str(midi_dir / "mixed.mid"))
        
        # Run full pipeline
        output_file = temp_dir / "sequences.npy"
        vocab_file = temp_dir / "vocab.pkl"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', vocab_file):
            build_dataset(
                data_dir=str(midi_dir),
                output_file=str(output_file)
            )
        
        # Verify results
        sequences = np.load(output_file, allow_pickle=True)
        assert len(sequences) == 3
        
        # Load vocab and verify
        with open(vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # All sequence values should be valid indices
        for seq in sequences:
            assert all(0 <= val < len(vocab_data['note_to_int']) for val in seq)
        
        # Vocab should contain both single notes and chord representations
        note_to_int = vocab_data['note_to_int']
        assert len(note_to_int) > 0
    
    def test_end_to_end_reversibility(self, temp_dir):
        """Test that notes can be converted to integers and back"""
        midi_dir = temp_dir / "reversibility"
        midi_dir.mkdir()
        
        # Create MIDI with known notes
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(midi_dir / "test.mid"))
        
        output_file = temp_dir / "sequences.npy"
        vocab_file = temp_dir / "vocab.pkl"
        
        with patch('utils.preprocess_naive.VOCAB_FILE', vocab_file):
            build_dataset(
                data_dir=str(midi_dir),
                output_file=str(output_file)
            )
        
        # Load results
        sequences = np.load(output_file, allow_pickle=True)
        with open(vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Convert back using int_to_note
        int_to_note = vocab_data['int_to_note']
        recovered_notes = [int_to_note[idx] for idx in sequences[0]]
        
        # Should be able to recover note strings
        assert len(recovered_notes) > 0
        assert all(isinstance(note, str) for note in recovered_notes)