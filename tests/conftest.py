"""fixtures for tests"""

import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture
def device():
    """get available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backend.mps.is_available():
        return torch.device("mps")

@pytest.fixture
def sample_sequences():
    """create sample sequences for testing"""
    # 5 sequences of different lengths
    sequences = [
        np.random.randint(0, 100, size=50),
        np.random.randint(0, 100, size=75),
        np.random.randint(0, 100, size=100),
        np.random.randint(0, 100, size=125),
        np.random.randint(0, 100, size=150),
    ]
    return np.array(sequences, dtype=object)

@pytest.fixture
def sample_measure_sequences():
    """create sample measure sequences (binary vectors)"""
    # 10 sequences, each with 8 measures, 16 pitches per measure
    sequences = []
    for _ in range(10):
        seq = [np.random.randint(0, 2, size=16) for _ in range(8)]
        sequences.append(seq)
    return np.array(sequences, dtype=object)

@pytest.fixture
def model_types():
    return ["lstm", "gru", "transformer"]