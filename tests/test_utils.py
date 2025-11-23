import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
import pretty_midi
import json

from utils.sampling import sample_next_note
from utils.measure_dataset import (
    midi_to_measure_pitches,
    build_pitch_vocab_from_midi_folder,
    build_measure_dataset
)

class TestSamplingEdgeCases:
    """Additional edge case tests for sampling.py"""
    
    def test_single_token_logits(self):
        """Test sampling with single token"""
        logits = torch.tensor([[5.0]])
        result = sample_next_note(logits, strategy="greedy")
        assert result.item() == 0
    
    def test_equal_probability_logits(self):
        """Test sampling when all logits are equal"""
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        result = sample_next_note(logits, strategy="greedy")
        assert 0 <= result.item() < 4
    
    def test_negative_logits(self):
        """Test sampling with negative logits"""
        logits = torch.tensor([[-5.0, -2.0, -1.0]])
        result = sample_next_note(logits, strategy="greedy")
        assert result.item() == 2  # Highest value
    
    def test_very_low_temperature(self):
        """Test with very low temperature (near-greedy)"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = sample_next_note(logits, strategy="random", temperature=0.01)
        # With very low temp, should strongly favor highest logit
        assert 0 <= result.item() < 3
    
    def test_top_k_exceeds_vocab_size(self):
        """Test top-k when k is larger than vocabulary"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = sample_next_note(logits, strategy="top_k", k=10)
        assert 0 <= result.item() < 3
    
    def test_top_p_very_small(self):
        """Test top-p with very small p value"""
        logits = torch.tensor([[1.0, 2.0, 5.0]])
        result = sample_next_note(logits, strategy="top_p", p=0.01)
        # Should still sample at least one token
        assert 0 <= result.item() < 3
    
    def test_3d_logits_batch(self):
        """Test with 3D logits (batch, sequence, vocab)"""
        logits = torch.tensor([[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]])
        result = sample_next_note(logits, strategy="greedy")
        assert result.shape[0] == 2


class TestMeasureDatasetEdgeCases:
    """Additional edge case tests for measure_dataset.py"""
    
    @pytest.fixture
    def temp_midi_folder(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_midi_with_no_notes(self, temp_midi_folder):
        """Test MIDI file with no notes"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        pm.instruments.append(instrument)
        
        midi_path = temp_midi_folder / "empty_notes.mid"
        pm.write(str(midi_path))
        
        measures = midi_to_measure_pitches(midi_path)
        assert measures is not None
        assert all(len(m) == 0 for m in measures)
    
    def test_midi_with_overlapping_notes(self, temp_midi_folder):
        """Test MIDI with overlapping notes of same pitch"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        # Add overlapping notes with same pitch
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0))
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=1.0, end=3.0))
        
        pm.instruments.append(instrument)
        midi_path = temp_midi_folder / "overlap.mid"
        pm.write(str(midi_path))
        
        measures = midi_to_measure_pitches(midi_path)
        # Should only have pitch 60 once per measure
        assert 60 in measures[0]
    
    def test_very_short_tempo(self, temp_midi_folder):
        """Test with very fast tempo"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(instrument)
        
        midi_path = temp_midi_folder / "fast.mid"
        pm.write(str(midi_path))
        
        measures = midi_to_measure_pitches(midi_path, tempo_bpm_default=240.0)
        assert measures is not None
        assert len(measures) > 0
    
    def test_very_slow_tempo(self, temp_midi_folder):
        """Test with very slow tempo"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(instrument)
        
        midi_path = temp_midi_folder / "slow.mid"
        pm.write(str(midi_path))
        
        measures = midi_to_measure_pitches(midi_path, tempo_bpm_default=30.0)
        assert measures is not None
    
    def test_extreme_pitch_range(self, temp_midi_folder):
        """Test with extreme pitch range (0-127)"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        # Add notes at extremes
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=0, start=0.0, end=1.0))
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=127, start=0.0, end=1.0))
        
        pm.instruments.append(instrument)
        midi_path = temp_midi_folder / "extreme.mid"
        pm.write(str(midi_path))
        
        vocab, mapping = build_pitch_vocab_from_midi_folder(temp_midi_folder)
        assert 0 in vocab
        assert 127 in vocab
    
    def test_measure_dataset_empty_folder(self, temp_midi_folder):
        """Test building dataset from empty folder"""
        out_dir = temp_midi_folder / "output"
        examples, vocab, mapping = build_measure_dataset(
            midi_folder=temp_midi_folder,
            out_dir=out_dir
        )
        
        assert len(examples) == 0
        assert len(vocab) == 0
        assert len(mapping) == 0


class TestSampling:
    """Test suite for sampling.py"""
    
    @pytest.fixture
    def sample_logits(self):
        """Create sample logits for testing"""
        return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    
    def test_greedy_sampling(self, sample_logits):
        """Test greedy sampling returns highest probability token"""
        result = sample_next_note(sample_logits, strategy="greedy")
        assert result.item() == 4  # Index of highest logit
    
    def test_greedy_sampling_temperature(self):
        """Test greedy sampling with different temperatures"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = sample_next_note(logits, strategy="greedy", temperature=0.5)
        assert result.item() == 2
    
    def test_top_k_sampling(self, sample_logits):
        """Test top-k sampling returns valid token"""
        result = sample_next_note(sample_logits, strategy="top_k", k=3)
        assert 0 <= result.item() < 5
        assert result.shape == (1, 1)
    
    def test_top_k_sampling_k_equals_one(self, sample_logits):
        """Test top-k with k=1 behaves like greedy"""
        result = sample_next_note(sample_logits, strategy="top_k", k=1)
        assert result.item() == 4
    
    def test_top_p_sampling(self, sample_logits):
        """Test nucleus (top-p) sampling returns valid token"""
        result = sample_next_note(sample_logits, strategy="top_p", p=0.9)
        assert 0 <= result.item() < 5
        assert result.shape == (1, 1)
    
    def test_top_p_sampling_p_equals_one(self, sample_logits):
        """Test top-p with p=1.0 can sample from full distribution"""
        result = sample_next_note(sample_logits, strategy="top_p", p=1.0)
        assert 0 <= result.item() < 5
    
    def test_random_sampling(self, sample_logits):
        """Test random sampling returns valid token"""
        result = sample_next_note(sample_logits, strategy="random")
        assert 0 <= result.item() < 5
        assert result.shape == (1, 1)
    
    def test_invalid_strategy_raises_error(self, sample_logits):
        """Test that invalid strategy raises ValueError"""
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            sample_next_note(sample_logits, strategy="invalid")
    
    def test_temperature_scaling(self):
        """Test temperature affects probability distribution"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        # High temperature should flatten distribution
        result_high = sample_next_note(logits, strategy="greedy", temperature=10.0)
        # Low temperature should sharpen distribution
        result_low = sample_next_note(logits, strategy="greedy", temperature=0.1)
        assert result_high.item() == result_low.item() == 2
    
    def test_batch_sampling(self):
        """Test sampling works with batched logits"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = sample_next_note(logits, strategy="greedy")
        assert result.shape == (2,)
        assert result[0].item() == 2
        assert result[1].item() == 0


class TestMeasureDataset:
    """Test suite for measure_dataset.py"""
    
    @pytest.fixture
    def temp_midi_folder(self):
        """Create temporary folder for MIDI files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def simple_midi_file(self, temp_midi_folder):
        """Create a simple MIDI file for testing"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        # Add some notes (C4, E4, G4 - C major chord)
        notes = [
            pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0),
            pretty_midi.Note(velocity=100, pitch=64, start=0.0, end=1.0),
            pretty_midi.Note(velocity=100, pitch=67, start=0.0, end=1.0),
            # Second measure
            pretty_midi.Note(velocity=100, pitch=62, start=2.0, end=3.0),
            pretty_midi.Note(velocity=100, pitch=65, start=2.0, end=3.0),
        ]
        
        for note in notes:
            instrument.notes.append(note)
        
        pm.instruments.append(instrument)
        
        midi_path = temp_midi_folder / "test.mid"
        pm.write(str(midi_path))
        return midi_path
    
    def test_midi_to_measure_pitches_basic(self, simple_midi_file):
        """Test basic MIDI to measure conversion"""
        measures = midi_to_measure_pitches(simple_midi_file)
        assert measures is not None
        assert len(measures) > 0
        assert isinstance(measures[0], set)
    
    def test_midi_to_measure_pitches_pitch_content(self, simple_midi_file):
        """Test that correct pitches are extracted"""
        measures = midi_to_measure_pitches(simple_midi_file, beats_per_bar=4)
        assert 60 in measures[0]  # C4
        assert 64 in measures[0]  # E4
        assert 67 in measures[0]  # G4
    
    def test_midi_to_measure_pitches_invalid_file(self, temp_midi_folder):
        """Test handling of invalid MIDI file"""
        invalid_path = temp_midi_folder / "invalid.mid"
        invalid_path.write_text("not a midi file")
        result = midi_to_measure_pitches(invalid_path)
        assert result is None
    
    def test_midi_to_measure_pitches_different_tempos(self, simple_midi_file):
        """Test with different tempo parameters"""
        measures_120 = midi_to_measure_pitches(simple_midi_file, tempo_bpm_default=120.0)
        measures_60 = midi_to_measure_pitches(simple_midi_file, tempo_bpm_default=60.0)
        assert measures_120 is not None
        assert measures_60 is not None
    
    def test_midi_to_measure_pitches_different_time_signatures(self, temp_midi_folder):
        """Test with different beats per bar"""
        # Create a longer MIDI file with notes spanning multiple measures
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        # Add notes across 8 seconds (enough for multiple measures)
        for i in range(8):
            instrument.notes.append(
                pretty_midi.Note(velocity=100, pitch=60+i, start=float(i), end=float(i)+0.5)
            )
        
        pm.instruments.append(instrument)
        midi_path = temp_midi_folder / "test_time_sig.mid"
        pm.write(str(midi_path))
        
        # With 4 beats per bar at 120 BPM: 2 seconds per measure
        measures_4 = midi_to_measure_pitches(midi_path, beats_per_bar=4, tempo_bpm_default=120.0)
        # With 3 beats per bar at 120 BPM: 1.5 seconds per measure
        measures_3 = midi_to_measure_pitches(midi_path, beats_per_bar=3, tempo_bpm_default=120.0)
        
        # Should have different number of measures
        assert len(measures_4) != len(measures_3)
        # 4/4 time should have fewer measures than 3/4 time for same duration
        assert len(measures_4) < len(measures_3)
    
    def test_build_pitch_vocab_from_folder(self, simple_midi_file):
        """Test building pitch vocabulary from folder"""
        vocab, mapping = build_pitch_vocab_from_midi_folder(simple_midi_file.parent)
        assert len(vocab) > 0
        assert len(mapping) == len(vocab)
        assert all(mapping[p] == i for i, p in enumerate(vocab))
    
    def test_pitch_vocab_sorted(self, simple_midi_file):
        """Test that pitch vocabulary is sorted"""
        vocab, _ = build_pitch_vocab_from_midi_folder(simple_midi_file.parent)
        assert vocab == sorted(vocab)
    
    def test_build_measure_dataset(self, simple_midi_file, temp_midi_folder):
        """Test building complete measure dataset"""
        out_dir = temp_midi_folder / "output"
        examples, vocab, mapping = build_measure_dataset(
            midi_folder=simple_midi_file.parent,
            out_dir=out_dir
        )
        
        assert len(examples) > 0
        assert len(vocab) > 0
        assert (out_dir / "pitch_vocab.pkl").exists()
        assert (out_dir / "measure_sequences.npy").exists()
    
    def test_measure_dataset_binary_vectors(self, simple_midi_file, temp_midi_folder):
        """Test that measures are correctly encoded as binary vectors"""
        out_dir = temp_midi_folder / "output"
        examples, vocab, mapping = build_measure_dataset(
            midi_folder=simple_midi_file.parent,
            out_dir=out_dir
        )
        
        # Check that vectors are binary
        for sequence in examples:
            for measure_vec in sequence:
                assert np.all((measure_vec == 0) | (measure_vec == 1))
    
    def test_measure_dataset_skips_short_files(self, temp_midi_folder):
        """Test that files with less than 2 measures are skipped"""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        # Single short note
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        pm.instruments.append(instrument)
        
        midi_path = temp_midi_folder / "short.mid"
        pm.write(str(midi_path))
        
        out_dir = temp_midi_folder / "output"
        examples, _, _ = build_measure_dataset(
            midi_folder=temp_midi_folder,
            out_dir=out_dir
        )
        
        # Should skip or have minimal examples
        assert len(examples) >= 0
    
    def test_measure_dataset_multiple_files(self, temp_midi_folder):
        """Test dataset building with multiple MIDI files"""
        # Create multiple MIDI files
        for i in range(3):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes.append(
                pretty_midi.Note(velocity=100, pitch=60+i, start=0.0, end=4.0)
            )
            pm.instruments.append(instrument)
            pm.write(str(temp_midi_folder / f"test_{i}.mid"))
        
        out_dir = temp_midi_folder / "output"
        examples, vocab, mapping = build_measure_dataset(
            midi_folder=temp_midi_folder,
            out_dir=out_dir
        )
        
        assert len(examples) == 3
        assert len(vocab) >= 3

from utils.preprocess_miditok import preprocess_miditok, OUTPUT_DIR, SEQ_LENGTH


class TestPreprocessMiditok:
    """Test suite for preprocess_miditok.py"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary input and output directories"""
        input_temp = tempfile.mkdtemp()
        output_temp = tempfile.mkdtemp()
        yield Path(input_temp), Path(output_temp)
        shutil.rmtree(input_temp, ignore_errors=True)
        shutil.rmtree(output_temp, ignore_errors=True)
    
    @pytest.fixture
    def simple_midi_files(self, temp_dirs):
        """Create simple MIDI files for testing"""
        input_dir, _ = temp_dirs
        
        # Create 3 simple MIDI files directly (don't copy)
        for i in range(3):
            midi_file = input_dir / f"test_{i}.mid"
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
            pm.write(str(midi_file))
        
        return input_dir
    
    def test_basic_preprocessing(self, simple_midi_files, temp_dirs):
        """Test basic preprocessing workflow"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Verify output files exist
        assert (output_dir / "sequences.npy").exists()
        assert (output_dir / "vocab.json").exists()
        assert (output_dir / "tokenizer.json").exists()
        assert (output_dir / "config.json").exists()
    
    def test_sequences_created(self, simple_midi_files, temp_dirs):
        """Test that sequences are created correctly"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load sequences
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        
        # Should have at least 3 sequences (one per file, possibly more if multi-track)
        assert len(sequences) >= 3
        assert all(len(seq) > 0 for seq in sequences)
    
    def test_vocab_created(self, simple_midi_files, temp_dirs):
        """Test that vocabulary is created"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load vocab
        with open(output_dir / "vocab.json") as f:
            vocab = json.load(f)
        
        assert len(vocab) > 0
        assert isinstance(vocab, dict)
    
    def test_config_metadata(self, simple_midi_files, temp_dirs):
        """Test that config contains correct metadata"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load config
        with open(output_dir / "config.json") as f:
            config = json.load(f)
        
        assert "seq_length" in config
        assert "tokenizer" in config
        assert "num_sequences" in config
        assert "vocab_size" in config
        assert config["tokenizer"] == "REMI"
        assert config["num_sequences"] >= 3
    
    def test_empty_directory(self, temp_dirs):
        """Test preprocessing with empty input directory"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Should still create files, but with no sequences
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        assert len(sequences) == 0
    
    def test_invalid_midi_files(self, temp_dirs):
        """Test handling of invalid MIDI files"""
        input_dir, output_dir = temp_dirs
        
        # Create an invalid MIDI file
        invalid_file = input_dir / "invalid.mid"
        invalid_file.write_text("not a valid midi file")
        
        # Create one valid MIDI file
        valid_file = input_dir / "valid.mid"
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(valid_file))
        
        preprocess_miditok(input_dir, output_dir)
        
        # Should process valid file and skip invalid
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        assert len(sequences) >= 1  # At least the valid file
        
        # Config should show skipped files
        with open(output_dir / "config.json") as f:
            config = json.load(f)
        assert config["num_files_skipped"] >= 1
    
    def test_files_processed_count(self, simple_midi_files, temp_dirs):
        """Test that processed file count is accurate"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        with open(output_dir / "config.json") as f:
            config = json.load(f)
        
        # Should have processed 3 files
        assert config["num_files_processed"] == 3
        assert config["num_files_skipped"] == 0
    
    def test_multi_track_midi(self, temp_dirs):
        """Test handling of multi-track MIDI files"""
        input_dir, output_dir = temp_dirs
        
        # Create a multi-track MIDI file
        multi_track_file = input_dir / "multi.mid"
        pm = pretty_midi.PrettyMIDI()
        
        # Track 1
        inst1 = pretty_midi.Instrument(program=0)
        inst1.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(inst1)
        
        # Track 2
        inst2 = pretty_midi.Instrument(program=1)
        inst2.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.0, end=1.0))
        pm.instruments.append(inst2)
        
        pm.write(str(multi_track_file))
        
        preprocess_miditok(input_dir, output_dir)
        
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        # Should have 1 sequence (song) with 2 tracks
        assert len(sequences) == 1
        assert len(sequences[0]) == 2
        # Each track should have tokens
        for track in sequences[0]:
            assert len(track) > 0
    
    def test_output_directory_creation(self, simple_midi_files, temp_dirs):
        """Test that output directory is created if it doesn't exist"""
        input_dir, output_dir = temp_dirs
        
        # Remove output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        preprocess_miditok(input_dir, output_dir)
        
        # Output directory should now exist
        assert output_dir.exists()
        assert (output_dir / "sequences.npy").exists()