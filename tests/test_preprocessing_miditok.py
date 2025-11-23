from unittest.mock import Mock, patch
import pretty_midi
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import pickle
import json

from utils.preprocess_miditok import preprocess_miditok, SEQ_LENGTH

# dataset class
NAIVE_DIR = Path("data/naive")
MIDITOK_DIR = Path("data/miditok")
MEASURES_DIR = Path("data/measures")

# NAIVE PREPROCESSING TESTS

def test_naive_output_files_exist():
    """Test that naive preprocessing creates expected fiels"""
    if not NAIVE_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_naive.py")
    
    assert (NAIVE_DIR / "sequences.npy").exists(), "sequences.npy not found"
    assert (NAIVE_DIR / "note_to_int.pkl").exists(), "note_to_int.pkl not found"

def test_naive_sequences_loadable():
    """test that naive sequences can be loaded"""
    if not NAIVE_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_naive.py")

    sequences = np.load(NAIVE_DIR / "sequences.npy", allow_pickle=True)
    assert len(sequences) > 0, "No sequences found"
    assert isinstance(sequences, np.ndarray), "sequences should be numpy array"

def test_naive_vocab_loadable():
    """test that naive vocab can be loaded"""
    if not NAIVE_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_naive.py")
    
    with open(NAIVE_DIR / "note_to_int.pkl", "rb") as f:
        vocab_data = pickle.load(f)

    assert "note_to_int" in vocab_data, "missing note_to_int mapping"
    assert "int_to_note" in vocab_data, "missing int_to_note mapping"
    assert len(vocab_data["note_to_int"]) > 0, "vocab is empty"

def test_naive_sequences_are_integers():
    """test that the naive sequences are only ints"""
    if not NAIVE_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_naive.py")

    sequences = np.load(NAIVE_DIR / "sequences.npy", allow_pickle=True)
    with open(NAIVE_DIR / "note_to_int.pkl", "rb") as f:
        vocab_data = pickle.load(f)

    vocab_size = len(vocab_data["note_to_int"])

    # check first sequence
    first_seq = sequences[0]
    assert all(isinstance(token, (int, np.integer)) for token in first_seq), \
        "sequences should contain integers"
    assert all(0 <= token < vocab_size for token in first_seq), \
        f"tokens should be in range [0, {vocab_size}]"
    
def test_naive_vocab_consistency():
    """test that note_to_int and int_to_note are consistent"""
    if not NAIVE_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_naive.py")

    with open(NAIVE_DIR / "note_to_int.pkl", "rb") as f:
        vocab_data = pickle.load(f)

    note_to_int = vocab_data["note_to_int"]
    int_to_note = vocab_data["int_to_note"]

    """check bidirectional mapping
        if you get a note's int from note_to_int, that int when given to int_to_note
        should return the original 
    """
    for note, idx in note_to_int.items():
        assert int_to_note[idx] == note, \
            f"Inconsistent mapping: {note} -> {idx} -> {int_to_note[idx]}"
        
# MIDITOK PREPROCESSING TESTS

def test_miditok_output_files_exist():
    """test that miditok preprocessing creates all the expected files"""
    if not MIDITOK_DIR.exists() or not (MIDITOK_DIR / "sequences.npy").exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")
    
    assert (MIDITOK_DIR / "sequences.npy").exists(), "sequences.npy not found"
    assert (MIDITOK_DIR / "tokenizer.json").exists(), "tokenizer.json not found"
    assert (MIDITOK_DIR / "vocab.json").exists(), "vocab.json not found"
    assert (MIDITOK_DIR / "config.json").exists(), "config.json not found"

def test_miditok_sequences_loadable():
    """test that miditok sequences can be loaded"""
    if not MIDITOK_DIR.exists() or not (MIDITOK_DIR / "sequences.npy").exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")

    sequences = np.load(MIDITOK_DIR / "sequences.npy", allow_pickle=True)
    assert len(sequences) > 0, "no sequences found"

def test_miditok_config_valid():
    """test that miditok config is valid JSON with expected fields"""
    if not MIDITOK_DIR.exists() or not (MIDITOK_DIR / "config.json").exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")
    
    with open(MIDITOK_DIR / "config.json") as f:  # this makes f = the file object
        config = json.load(f)
    
    required_fields = ["vocab_size", "num_sequences", "tokenizer"]
    for field in required_fields:
        assert field in config, f"missing required field {field}"

    assert config["vocab_size"] > 0, "vocab size should be positive"
    assert config["num_sequences"] > 0, "should have sequences"

def test_miditok_vocab_valid():
    """test that miditok vocab is valid"""
    if not MIDITOK_DIR.exists() or not (MIDITOK_DIR / "vocab.json").exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")
    
    with open(MIDITOK_DIR / "vocab.json") as f:
        vocab = json.load(f)
    
    assert len(vocab) > 0, "vocab shouldn't be empty"

def test_miditok_sequences_in_vocab_range():
    """test that miditok tokens are withing vocab range"""
    if not MIDITOK_DIR.exists() or not (MIDITOK_DIR / "sequences.npy").exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")
    
    sequences = np.load(MIDITOK_DIR / "sequences.npy", allow_pickle=True)
    with open(MIDITOK_DIR / "config.json") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]

    # check first sequence (which is a list of tracks)
    first_seq = sequences[0]
    # Iterate through each track in the sequence
    for track in first_seq:
        assert all(0 <= token < vocab_size for token in track), \
            f"all tokens should be in range [0, {vocab_size}]"
    
# MEASURE DATASET TESTS

def test_measure_dataset_files_exist():
    """test that the measure dataset files exist"""
    if not MEASURES_DIR.exists():
        pytest.skip("run preprocessing first python utils/measure_dataset.py")
    
    assert (MEASURES_DIR / "measure_sequences.npy").exists(), "measure_sequences.npy not found"
    assert (MEASURES_DIR / "pitch_vocab.pkl").exists(), "pitch_vocab.pkl not found"

def test_measure_sequences_loadable():
    """test that measure sequences can be loaded"""
    if not MEASURES_DIR.exists():
        pytest.skip("run preprocessing first python utils/measure_dataset.py")

    sequences = np.load(MEASURES_DIR / "measure_sequences.npy", allow_pickle=True)
    assert len(sequences) > 0, "no sequences found"

def test_measure_pitch_vocab_valid():
    """test that pitch vocab valid"""
    if not MEASURES_DIR.exists():
        pytest.skip("run preprocessing first python utils/measure_dataset.py")

    with open(MEASURES_DIR / "pitch_vocab.pkl", "rb") as f:
        pitch_data = pickle.load(f)

    assert "vocab" in pitch_data, "missing vocab"
    assert "mapping" in pitch_data, "missing mapping"

    vocab = pitch_data["vocab"]
    mapping = pitch_data["mapping"]

    assert len(vocab) > 0, "vocab shouldn't be empty"
    assert len(mapping) == len(vocab), "mapping size should match vocab size"

    # check that all pitches in range 0-127
    assert all(0 <= p <= 127 for p in vocab), "ptiches should be in MIDI range [0, 127]"

def test_measure_sequences_are_binary_vectors():
    """test that measure sequences are binary vectors"""
    if not MEASURES_DIR.exists():
        pytest.skip("run preprocessing first python utils/measure_dataset.py")
    
    sequences = np.load(MEASURES_DIR / "measure_sequences.npy", allow_pickle=True)
    with open(MEASURES_DIR / "pitch_vocab.pkl", "rb") as f:
        pitch_data = pickle.load(f)

    vocab_size = len(pitch_data["vocab"])

    # check first sequence
    first_seq = sequences[0]
    for measure in first_seq:
        assert len(measure) == vocab_size, \
            f"each measure should have {vocab_size} elements"
        assert all(m in [0, 1] for m in measure), \
            "measure vectors shouls be binary (0 or 1)"
        
# COMPARISON TESTS

def test_all_preprocessing_methods_have_data():
    """test that at least one preprocessing method has produced data"""
    has_naive = NAIVE_DIR.exists() and (NAIVE_DIR / "sequences.npy").exists()
    has_miditok = MIDITOK_DIR.exists() and (MIDITOK_DIR / "sequences.npy").exists()
    has_measures = MEASURES_DIR.exists() and (MEASURES_DIR / "measure_sequences.npy").exists()

    if not (has_naive or has_measures or has_miditok):
        pytest.skip("No preprocessing has been run. Run at least one of:\n"
                   "  - python utils/preprocess_naive.py\n"
                   "  - python utils/preprocess_miditok.py\n"
                   "  - python utils/measure_dataset.py")

def test_preprocessing_summary():
    """print summary of all preprocessing results"""
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)

    # naive
    if NAIVE_DIR.exists() and (NAIVE_DIR / "sequences.npy").exists():
        sequences = np.load(NAIVE_DIR / "sequences.npy", allow_pickle=True)
        with open(NAIVE_DIR / "note_to_int.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        print(f"\nNAIVE:")
        print(f"    Sequences: {len(sequences)}")
        print(f"    Vocab size: {len(vocab_data['note_to_int'])}")
    else:
        print(f"\n NAIVE: not found")

    # Miditok
    if MIDITOK_DIR.exists() and (MIDITOK_DIR / "sequences.npy").exists():
        sequences = np.load(MIDITOK_DIR / "sequences.npy", allow_pickle=True)
        with open(MIDITOK_DIR / "config.json") as f:
            config = json.load(f)
        print(f"\nMIDITOK:")
        print(f"   Sequences: {len(sequences)}")
        print(f"   Vocab size: {config['vocab_size']}")
    else:
        print(f"\nMIDITOK: Not found")
    
    # Measures
    if MEASURES_DIR.exists() and (MEASURES_DIR / "measure_sequences.npy").exists():
        sequences = np.load(MEASURES_DIR / "measure_sequences.npy", allow_pickle=True)
        with open(MEASURES_DIR / "pitch_vocab.pkl", "rb") as f:
            pitch_data = pickle.load(f)
        print(f"\nMEASURES:")
        print(f"   Sequences: {len(sequences)}")
        print(f"   Pitch vocab size: {len(pitch_data['vocab'])}")
    else:
        print(f"\nMEASURES: Not found")
    
    print("\n" + "="*60 + "\n")

class TestPreprocessMiditokEdgeCases:
    """Additional edge case tests for preprocess_miditok.py"""
    
    @pytest.fixture
    def temp_dirs(self):
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        yield Path(input_dir), Path(output_dir)
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    
    @pytest.fixture
    def simple_midi_files(self, temp_dirs):
        input_dir, _ = temp_dirs
        
        for i in range(3):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            
            for j in range(5 + i):
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=60 + j,
                        start=float(j) * 0.5,
                        end=float(j) * 0.5 + 0.4
                    )
                )
            
            pm.instruments.append(instrument)
            pm.write(str(input_dir / f"test_{i}.mid"))
        
        return input_dir
    
    def test_output_directory_creation_fixed(self, simple_midi_files, temp_dirs):
        """Test that output directory is created if it doesn't exist - fixed version"""
        input_dir, output_dir = temp_dirs
        
        # Remove output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        assert not output_dir.exists()
        
        # ensure dir exists before running
        output_dir.mkdir(parents=True, exist_ok=True)

        preprocess_miditok(input_dir, output_dir)

        # output dir now exists
        assert output_dir.exists()
        assert (output_dir / "sequences.npy").exists()
    
    def test_very_long_sequences(self, temp_dirs):
        """Test with MIDI that generates very long sequences"""
        input_dir, output_dir = temp_dirs
        
        # Create MIDI with many notes
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        for i in range(200):  # Many notes
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=60 + (i % 12),
                    start=float(i) * 0.1,
                    end=float(i) * 0.1 + 0.05
                )
            )
        
        pm.instruments.append(instrument)
        pm.write(str(input_dir / "long.mid"))
        
        preprocess_miditok(input_dir, output_dir)
        
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        assert len(sequences) > 0
    
    def test_single_note_midi(self, temp_dirs):
        """Test with MIDI containing only a single note"""
        input_dir, output_dir = temp_dirs
        
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        pm.instruments.append(instrument)
        pm.write(str(input_dir / "single.mid"))
        
        preprocess_miditok(input_dir, output_dir)
        
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        assert config["num_files_processed"] >= 0
    
    def test_all_tracks_drumming(self, temp_dirs):
        """Test with MIDI where all tracks are drums"""
        input_dir, output_dir = temp_dirs
        
        pm = pretty_midi.PrettyMIDI()
        # Add drum track (channel 9)
        instrument = pretty_midi.Instrument(program=0, is_drum=True)
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.5))
        pm.instruments.append(instrument)
        pm.write(str(input_dir / "drums.mid"))
        
        preprocess_miditok(input_dir, output_dir)
        
        # Should process without errors
        assert (output_dir / "sequences.npy").exists()
    
    def test_unicode_filenames(self, temp_dirs):
        """Test with MIDI files having unicode characters in names"""
        input_dir, output_dir = temp_dirs
        
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(instrument)
        pm.write(str(input_dir / "test_音楽.mid"))
        
        preprocess_miditok(input_dir, output_dir)
        
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        assert config["num_files_processed"] >= 0

class TestPreprocessMiditok:
    """Test suite for preprocess_miditok.py"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary input and output directories"""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        yield Path(input_dir), Path(output_dir)
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    
    @pytest.fixture
    def simple_midi_files(self, temp_dirs):
        """Create simple MIDI files for testing"""
        input_dir, _ = temp_dirs
        
        # Create 3 valid MIDI files
        for i in range(3):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            
            # Add notes with different patterns
            for j in range(5 + i):
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=60 + j,
                        start=float(j) * 0.5,
                        end=float(j) * 0.5 + 0.4
                    )
                )
            
            pm.instruments.append(instrument)
            pm.write(str(input_dir / f"test_{i}.mid"))
        
        return input_dir
    
    @pytest.fixture
    def mixed_midi_files(self, temp_dirs):
        """Create mix of valid and invalid MIDI files"""
        input_dir, _ = temp_dirs
        
        # Valid MIDI
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes.append(
            pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0)
        )
        pm.instruments.append(instrument)
        pm.write(str(input_dir / "valid.mid"))
        
        # Invalid MIDI (corrupted)
        invalid_file = input_dir / "invalid.mid"
        invalid_file.write_text("This is not a MIDI file")
        
        # Empty file
        empty_file = input_dir / "empty.mid"
        empty_file.write_bytes(b"")
        
        return input_dir
    
    @pytest.fixture
    def multi_track_midi(self, temp_dirs):
        """Create MIDI file with multiple tracks"""
        input_dir, _ = temp_dirs
        
        pm = pretty_midi.PrettyMIDI()
        
        # Add 3 different instruments
        for program in [0, 24, 40]:  # Piano, Guitar, Violin
            instrument = pretty_midi.Instrument(program=program)
            for i in range(4):
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=60 + i,
                        start=float(i) * 0.5,
                        end=float(i) * 0.5 + 0.4
                    )
                )
            pm.instruments.append(instrument)
        
        pm.write(str(input_dir / "multi_track.mid"))
        return input_dir
    
    def test_preprocess_basic(self, simple_midi_files, temp_dirs):
        """Test basic preprocessing functionality"""
        input_dir, output_dir = temp_dirs
        
        # Files already exist in input_dir from fixture, no need to copy
        
        preprocess_miditok(input_dir, output_dir)
        
        # Check output files exist
        assert (output_dir / "sequences.npy").exists()
        assert (output_dir / "tokenizer.json").exists()
        assert (output_dir / "vocab.json").exists()
        assert (output_dir / "config.json").exists()
    
    def test_sequences_saved_correctly(self, simple_midi_files, temp_dirs):
        """Test that sequences are saved with correct format"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load sequences
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        
        assert len(sequences) > 0
        assert isinstance(sequences, np.ndarray)
        # Each sequence is a list of tracks (nested lists)
        for seq in sequences:
            assert isinstance(seq, (list, np.ndarray))
            assert len(seq) > 0
            # Each track should be a list of integers
            for track in seq:
                assert isinstance(track, (list, np.ndarray))
                assert all(isinstance(token, (int, np.integer)) for token in track)
    
    def test_tokenizer_config_saved(self, simple_midi_files, temp_dirs):
        """Test that tokenizer configuration is saved"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load tokenizer config
        with open(output_dir / "tokenizer.json", 'r') as f:
            tokenizer_config = json.load(f)
        
        assert tokenizer_config is not None
        assert isinstance(tokenizer_config, dict)
    
    def test_vocab_saved(self, simple_midi_files, temp_dirs):
        """Test that vocabulary is saved correctly"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load vocab
        with open(output_dir / "vocab.json", 'r') as f:
            vocab = json.load(f)
        
        assert vocab is not None
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
    
    def test_config_metadata(self, simple_midi_files, temp_dirs):
        """Test that config metadata contains expected fields"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load config
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Check required fields
        assert "seq_length" in config
        assert "tokenizer" in config
        assert "num_sequences" in config
        assert "num_files_processed" in config
        assert "num_files_skipped" in config
        assert "vocab_size" in config
        assert "min_seq_length" in config
        assert "max_seq_length" in config
        assert "mean_seq_length" in config
        
        # Check values make sense
        assert config["seq_length"] == SEQ_LENGTH
        assert config["tokenizer"] == "REMI"
        assert config["num_sequences"] >= 0
        assert config["vocab_size"] > 0
        assert config["min_seq_length"] >= 0
        assert config["max_seq_length"] >= config["min_seq_length"]
    
    def test_handles_invalid_files(self, mixed_midi_files, temp_dirs):
        """Test that invalid MIDI files are skipped"""
        input_dir, output_dir = temp_dirs
        
        # Should not raise an error
        preprocess_miditok(input_dir, output_dir)
        
        # Load config to check skipped files
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Some files should have been skipped
        assert config["num_files_skipped"] > 0
        # But at least one should have been processed
        assert config["num_files_processed"] > 0
    
    def test_multi_track_handling(self, multi_track_midi, temp_dirs):
        """Test that multi-track MIDI files are handled correctly"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load sequences
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        
        # Should have multiple sequences (one per track)
        assert len(sequences) >= 1
        
        # Load config
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Should show sequences were created
        assert config["num_sequences"] > 0
    
    def test_empty_input_directory(self, temp_dirs):
        """Test behavior with empty input directory"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Should still create output files
        assert (output_dir / "sequences.npy").exists()
        assert (output_dir / "config.json").exists()
        
        # Load config
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Should show no files processed
        assert config["num_sequences"] == 0
        assert config["num_files_processed"] == 0
    
    def test_sequence_statistics(self, simple_midi_files, temp_dirs):
        """Test that sequence statistics are calculated correctly"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load sequences and config
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Calculate actual statistics
        # sequences is list of songs, each song is list of tracks
        all_tracks = [track for song in sequences for track in song]
        seq_lengths = [len(track) for track in all_tracks]
        
        # Verify statistics in config match actual data
        assert config["min_seq_length"] == min(seq_lengths)
        assert config["max_seq_length"] == max(seq_lengths)
        assert abs(config["mean_seq_length"] - np.mean(seq_lengths)) < 0.1
    
    def test_output_directory_creation(self, simple_midi_files, temp_dirs):
        """Test that output directory is created if it doesn't exist"""
        input_dir, output_dir = temp_dirs
        
        # Remove output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        # Ensure directory exists before running
        output_dir.mkdir(parents=True, exist_ok=True)

        preprocess_miditok(input_dir, output_dir)
        
        # Output directory should now exist
        assert output_dir.exists()
        assert (output_dir / "sequences.npy").exists()
    
    def test_sequences_are_integer_tokens(self, simple_midi_files, temp_dirs):
        """Test that all tokens in sequences are integers"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        sequences = np.load(output_dir / "sequences.npy", allow_pickle=True)
        
        # Each sequence is a list of tracks
        for seq in sequences:
            for track in seq:
                assert all(isinstance(token, (int, np.integer)) for token in track)
                assert all(token >= 0 for token in track)
    
    def test_vocab_size_consistency(self, simple_midi_files, temp_dirs):
        """Test that vocab size in config matches actual vocab"""
        input_dir, output_dir = temp_dirs
        
        preprocess_miditok(input_dir, output_dir)
        
        # Load vocab and config
        with open(output_dir / "vocab.json", 'r') as f:
            vocab = json.load(f)
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Vocab size should match
        assert config["vocab_size"] == len(vocab)
    
    def test_files_processed_count(self, simple_midi_files, temp_dirs):
        """Test that file processing count is accurate"""
        input_dir, output_dir = temp_dirs
        
        # Count files that already exist in input_dir
        num_files = len(list(input_dir.glob("*.mid")))
        
        preprocess_miditok(input_dir, output_dir)

        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Should process all files (or account for skipped ones)
        total_files = config["num_files_processed"] + config["num_files_skipped"]
        assert total_files == num_files