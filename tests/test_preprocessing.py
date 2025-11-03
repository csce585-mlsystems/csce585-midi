import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import pickle
import json

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
    if not MIDITOK_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")
    
    assert (MIDITOK_DIR / "sequences.npy").exists(), "sequences.npy not found"
    assert (MIDITOK_DIR / "tokenizer.json").exists(), "tokenizer.json not found"
    assert (MIDITOK_DIR / "vocab.json").exists(), "vocab.json not found"
    assert (MIDITOK_DIR / "config.json").exists(), "config.json not found"

def test_miditok_sequences_loadable():
    """test that miditok sequences can be loaded"""
    if not MIDITOK_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")

    sequences = np.load(MIDITOK_DIR / "sequences.npy", allow_pickle=True)
    assert len(sequences) > 0, "no sequences found"

def test_miditok_config_valid():
    """test that miditok config is valid JSON with expected fields"""
    if not MIDITOK_DIR.exists():
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
    if not MIDITOK_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")
    
    with open(MIDITOK_DIR / "vocab.json") as f:
        vocab = json.load(f)
    
    assert len(vocab) > 0, "vocab shouldn't be empty"

def test_miditok_sequences_in_vocab_range():
    """test that miditok tokens are withing vocab range"""
    if not MIDITOK_DIR.exists():
        pytest.skip("run preprocessing first: python utils/preprocess_miditok.py")
    
    sequences = np.load(MIDITOK_DIR / "sequences.npy", allow_pickle=True)
    with open(MIDITOK_DIR / "config.json") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]

    # check first sequence
    first_seq = sequences[0]
    assert all(0 <= token < vocab_size for token in first_seq), \
        f"all tokens should be in range [0, {vocab_size}]"
    
# MEASURE DATASET TESTS