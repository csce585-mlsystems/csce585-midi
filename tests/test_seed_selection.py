"""
Tests for seed_selection.py
Tests smart seed selection without models
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from utils.seed_selection import (
    analyze_sequence_properties,
    find_seed_by_characteristics
)


class TestAnalyzeSequenceProperties:
    """Test suite for analyze_sequence_properties function"""
    
    def test_analyze_single_notes(self):
        """Test analysis of sequence with single notes"""
        sequence = [0, 1, 2, 3, 4]
        int_to_note = {
            0: '60',  # Middle C
            1: '62',  # D
            2: '64',  # E
            3: '65',  # F
            4: '67',  # G
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        assert props['avg_pitch'] == pytest.approx(63.6, abs=0.1)  # (60+62+64+65+67)/5 = 63.6
        assert props['pitch_range'] == 7  # 67-60
        assert props['num_unique_pitches'] == 5
        assert props['length'] == 5
    
    def test_analyze_chords(self):
        """Test analysis of sequence with chords"""
        sequence = [0, 1, 2]
        int_to_note = {
            0: '60.64.67',  # C major chord
            1: '62.65.69',  # D minor chord
            2: '64.67.71',  # E minor chord
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        # All pitches: 60,64,67,62,65,69,64,67,71
        assert props['avg_pitch'] == pytest.approx(65.44, abs=0.1)
        assert props['pitch_range'] == 11  # 71-60
        assert props['num_unique_pitches'] == 7  # 60,62,64,65,67,69,71
        assert props['length'] == 3
    
    def test_analyze_mixed_notes_and_chords(self):
        """Test analysis with mix of single notes and chords"""
        sequence = [0, 1, 2, 3]
        int_to_note = {
            0: '60',
            1: '62.65',  # chord
            2: '64',
            3: '67.71',  # chord
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        # Pitches: 60, 62, 65, 64, 67, 71
        assert props['avg_pitch'] == pytest.approx(64.83, abs=0.1)
        assert props['num_unique_pitches'] == 6
    
    def test_analyze_notes_with_text(self):
        """Test analysis with note names that include text"""
        sequence = [0, 1, 2]
        int_to_note = {
            0: 'note60',
            1: 'pitch62',
            2: 'C64',
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        assert props['avg_pitch'] == 62.0
        assert props['num_unique_pitches'] == 3
    
    def test_analyze_empty_pitches(self):
        """Test analysis when no pitches can be extracted"""
        sequence = [0, 1, 2]
        int_to_note = {
            0: 'rest',
            1: 'pause',
            2: 'wait',
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is None
    
    def test_analyze_single_pitch(self):
        """Test analysis with single pitch"""
        sequence = [0]
        int_to_note = {0: '60'}
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        assert props['avg_pitch'] == 60.0
        assert props['pitch_range'] == 0
        assert props['num_unique_pitches'] == 1
        assert props['length'] == 1
    
    def test_analyze_repeated_pitches(self):
        """Test analysis with repeated pitches"""
        sequence = [0, 0, 0, 1, 1]
        int_to_note = {
            0: '60',
            1: '62',
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        assert props['avg_pitch'] == pytest.approx(60.8, abs=0.1)
        assert props['num_unique_pitches'] == 2
        assert props['length'] == 5
    
    def test_analyze_wide_pitch_range(self):
        """Test analysis with wide pitch range"""
        sequence = [0, 1, 2]
        int_to_note = {
            0: '21',  # Very low (A0)
            1: '60',  # Middle C
            2: '108', # Very high (C8)
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        assert props['avg_pitch'] == 63.0
        assert props['pitch_range'] == 87
    
    def test_analyze_malformed_notes(self):
        """Test analysis with some malformed note entries"""
        sequence = [0, 1, 2, 3]
        int_to_note = {
            0: '60',
            1: 'invalid',
            2: '64',
            3: '',
        }
        
        props = analyze_sequence_properties(sequence, int_to_note)
        
        assert props is not None
        # Should only extract 60 and 64
        assert props['avg_pitch'] == 62.0
        assert props['num_unique_pitches'] == 2


class TestFindSeedByCharacteristics:
    """Test suite for find_seed_by_characteristics function"""
    
    def setup_method(self):
        """Setup test data"""
        # Create test sequences with known properties (as lists, with appropriate lengths)
        # Length ranges: short: 20-50, medium: 50-100, long: 100-200
        self.sequences = [
            list(range(0, 25)),      # seq 0: medium pitch (60-67), 5 unique, length 25 (short)
            list(range(25, 50)),     # seq 1: high pitch (72-79), 5 unique, length 25 (short)
            list(range(50, 75)),     # seq 2: low pitch (40-47), 5 unique, length 25 (short)
            list(range(75, 103)),    # seq 3: medium pitch (60-74), 8 unique, length 28 (short)
        ]
        
        self.int_to_note = {}
        
        # Seq 0: medium pitch (60-67), 5 unique pitches cycling, length 25
        pitches_0 = ['60', '62', '64', '65', '67']
        for i in range(25):
            self.int_to_note[i] = pitches_0[i % 5]
        
        # Seq 1: high pitch (72-79), 5 unique pitches cycling, length 25
        pitches_1 = ['72', '74', '76', '77', '79']
        for i in range(25, 50):
            self.int_to_note[i] = pitches_1[(i - 25) % 5]
        
        # Seq 2: low pitch (40-47), 5 unique pitches cycling, length 25
        pitches_2 = ['40', '42', '44', '45', '47']
        for i in range(50, 75):
            self.int_to_note[i] = pitches_2[(i - 50) % 5]
        
        # Seq 3: medium pitch (60-74), 8 unique pitches cycling, length 28
        pitches_3 = ['60', '62', '64', '65', '67', '69', '71', '74']
        for i in range(75, 103):
            self.int_to_note[i] = pitches_3[(i - 75) % 8]
    
    def test_find_medium_pitch_simple_complexity(self):
        """Test finding medium pitch, simple complexity seed"""
        seed = find_seed_by_characteristics(
            self.sequences,
            self.int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        assert seed is not None
        assert isinstance(seed, list)
        # Should find seq 0 (medium pitch, simple complexity, short)
        assert seed == list(self.sequences[0])
    
    def test_find_high_pitch_simple_complexity(self):
        """Test finding high pitch, simple complexity seed"""
        seed = find_seed_by_characteristics(
            self.sequences,
            self.int_to_note,
            pitch_preference="high",
            complexity="simple",
            length="short"
        )
        
        assert seed is not None
        assert isinstance(seed, list)
        # Should find seq 1 (high pitch, simple complexity, short)
        assert seed == list(self.sequences[1])
    
    def test_find_low_pitch_simple_complexity(self):
        """Test finding low pitch, simple complexity seed"""
        seed = find_seed_by_characteristics(
            self.sequences,
            self.int_to_note,
            pitch_preference="low",
            complexity="simple",
            length="short"
        )
        
        assert seed is not None
        assert isinstance(seed, list)
        # Should find seq 2 (low pitch, simple complexity, short)
        assert seed == list(self.sequences[2])
    
    def test_find_medium_complexity(self):
        """Test finding medium complexity seed"""
        seed = find_seed_by_characteristics(
            self.sequences,
            self.int_to_note,
            pitch_preference="medium",
            complexity="medium",
            length="short"
        )
        
        assert seed is not None
        assert isinstance(seed, list)
        # Should find seq 3 (medium pitch, medium complexity)
        assert seed == list(self.sequences[3])
    
    def test_no_matching_sequences(self, capsys):
        """Test when no sequences match the criteria"""
        # Request impossible criteria
        with patch('numpy.random.randint', return_value=0):
            seed = find_seed_by_characteristics(
                self.sequences,
                self.int_to_note,
                pitch_preference="high",
                complexity="complex",  # No sequences have 9+ pitches
                length="long"
            )
        
        captured = capsys.readouterr()
        assert "no sequences match criteria" in captured.out
        assert "using random seed" in captured.out
        assert seed is not None
    
    def test_multiple_candidates(self):
        """Test when multiple sequences match"""
        # Both seq 0 and seq 1 are simple complexity and short length
        # Let's make them both medium pitch
        modified_int_to_note = self.int_to_note.copy()
        # Modify seq 1 to be medium pitch instead of high
        for i in range(5, 10):
            modified_int_to_note[i] = str(60 + (i - 5) * 2)
        
        # Should randomly select from candidates
        with patch('numpy.random.randint') as mock_randint:
            # First call for selecting from candidates
            mock_randint.return_value = 0
            
            seed = find_seed_by_characteristics(
                self.sequences,
                modified_int_to_note,
                pitch_preference="medium",
                complexity="simple",
                length="short"
            )
            
            assert seed is not None
            assert mock_randint.called
    
    def test_length_short(self):
        """Test length preference: short (20-50 notes)"""
        sequences = [
            list(range(30)),   # short: 30 notes
            list(range(30, 110)),   # medium: 80 notes
            list(range(110, 260)),  # long: 150 notes
        ]
        
        # Create int_to_note with 4 unique pitches (simple complexity)
        pitches = ['60', '62', '64', '65']
        int_to_note = {i: pitches[i % 4] for i in range(260)}
        
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        assert len(seed) == 30
    
    def test_length_medium(self):
        """Test length preference: medium (50-100 notes)"""
        sequences = [
            list(range(30)),   # short: 30 notes
            list(range(30, 110)),   # medium: 80 notes
            list(range(110, 260)),  # long: 150 notes
        ]
        
        # Create int_to_note with 4 unique pitches (simple complexity)
        pitches = ['60', '62', '64', '65']
        int_to_note = {i: pitches[i % 4] for i in range(260)}
        
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="medium"
        )
        
        assert len(seed) == 80
    
    def test_length_long(self):
        """Test length preference: long (100-200 notes)"""
        sequences = [
            list(range(30)),   # short: 30 notes
            list(range(30, 110)),   # medium: 80 notes
            list(range(110, 260)),  # long: 150 notes
        ]
        
        # Create int_to_note with 4 unique pitches (simple complexity)
        pitches = ['60', '62', '64', '65']
        int_to_note = {i: pitches[i % 4] for i in range(260)}
        
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="long"
        )
        
        assert len(seed) == 150
    
    def test_prints_selected_properties(self, capsys):
        """Test that function prints selected seed properties"""
        seed = find_seed_by_characteristics(
            self.sequences,
            self.int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        captured = capsys.readouterr()
        assert "Selected seed:" in captured.out
        assert "avg pitch:" in captured.out
        assert "unique pitches:" in captured.out
        assert "length:" in captured.out
    
    def test_default_parameters(self):
        """Test with default parameter values"""
        seed = find_seed_by_characteristics(
            self.sequences,
            self.int_to_note
        )
        
        assert seed is not None
        assert isinstance(seed, list)
    
    def test_returns_list_not_numpy_array(self):
        """Test that function returns a list, not numpy array"""
        seed = find_seed_by_characteristics(
            self.sequences,
            self.int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        assert isinstance(seed, list)
        assert not isinstance(seed, np.ndarray)
    
    def test_complexity_simple(self):
        """Test complexity preference: simple (3-5 pitches)"""
        sequences = [
            list(range(25)),        # 4 unique pitches, length 25 (short)
            list(range(25, 55)),    # 7 unique pitches, length 30 (short)
        ]
        
        # First sequence: 4 unique pitches cycling
        pitches_simple = ['60', '62', '64', '65']
        # Second sequence: 7 unique pitches cycling
        pitches_medium = ['60', '62', '64', '65', '67', '69', '71']
        
        int_to_note = {}
        for i in range(25):
            int_to_note[i] = pitches_simple[i % 4]
        for i in range(25, 55):
            int_to_note[i] = pitches_medium[(i - 25) % 7]
        
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        assert len(seed) == 25  # Should select first sequence
    
    def test_complexity_complex(self):
        """Test complexity preference: complex (9+ pitches)"""
        sequences = [
            list(range(25)),        # 4 unique pitches, length 25 (short)
            list(range(25, 55)),    # 10 unique pitches, length 30 (short)
        ]
        
        # First sequence: 4 unique pitches
        pitches_simple = ['60', '62', '64', '65']
        # Second sequence: 10 unique pitches
        pitches_complex = [str(60 + i * 2) for i in range(10)]
        
        int_to_note = {}
        for i in range(25):
            int_to_note[i] = pitches_simple[i % 4]
        for i in range(25, 55):
            int_to_note[i] = pitches_complex[(i - 25) % 10]
        
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="complex",
            length="short"
        )
        
        assert len(seed) == 30  # Should select second sequence


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_sequences(self):
        """Test with empty sequences array"""
        sequences = []
        int_to_note = {}
        
        # Should raise an error when trying to access empty array
        with pytest.raises((IndexError, ValueError)):
            find_seed_by_characteristics(sequences, int_to_note)
    
    def test_single_sequence(self):
        """Test with single sequence"""
        sequences = np.array([[0, 1, 2]])
        int_to_note = {0: '60', 1: '62', 2: '64'}
        
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        assert seed == [0, 1, 2]
    
    def test_all_sequences_unparseable(self, capsys):
        """Test when all sequences have no extractable pitches"""
        sequences = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ])
        int_to_note = {i: 'rest' for i in range(6)}
        
        with patch('numpy.random.randint', return_value=0):
            seed = find_seed_by_characteristics(
                sequences,
                int_to_note,
                pitch_preference="medium",
                complexity="simple",
                length="short"
            )
        
        captured = capsys.readouterr()
        assert "no sequences match criteria" in captured.out
        assert seed is not None
    
    def test_boundary_pitch_values(self):
        """Test with boundary pitch values"""
        sequences = np.array([[0, 1, 2]])
        int_to_note = {
            0: '40',  # Exactly at low range min
            1: '60',  # Exactly at medium range min
            2: '72',  # Exactly at high range min
        }
        
        # Should match medium pitch preference (60 is at boundary)
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        assert seed == [0, 1, 2]
    
    def test_boundary_complexity_values(self):
        """Test with boundary complexity values"""
        sequences = np.array([[0, 1, 2]])
        int_to_note = {0: '60', 1: '62', 2: '64'}  # Exactly 3 pitches
        
        # 3 pitches is at simple complexity boundary
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="short"
        )
        
        assert seed == [0, 1, 2]
    
    def test_boundary_length_values(self):
        """Test with boundary length values"""
        sequences = np.array([list(range(50))])  # Exactly 50 notes
        int_to_note = {i: '60' for i in range(50)}
        
        # 50 notes is at medium length boundary
        seed = find_seed_by_characteristics(
            sequences,
            int_to_note,
            pitch_preference="medium",
            complexity="simple",
            length="medium"
        )
        
        assert len(seed) == 50


class TestIntegration:
    """Integration tests with realistic data"""
    
    def test_realistic_music_sequence(self):
        """Test with realistic music-like sequence"""
        # Create a C major scale sequence
        sequences = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        int_to_note = {
            0: '60',  # C
            1: '62',  # D
            2: '64',  # E
            3: '65',  # F
            4: '67',  # G
            5: '69',  # A
            6: '71',  # B
            7: '72',  # C (octave)
        }
        
        props = analyze_sequence_properties(sequences[0], int_to_note)
        
        assert props is not None
        assert 60 <= props['avg_pitch'] <= 72
        assert props['num_unique_pitches'] == 8
        assert props['length'] == 8
    
    def test_chord_progression(self):
        """Test with chord progression"""
        sequences = np.array([[0, 1, 2, 3]])
        int_to_note = {
            0: '60.64.67',  # C major
            1: '65.69.72',  # F major
            2: '67.71.74',  # G major
            3: '60.64.67',  # C major
        }
        
        props = analyze_sequence_properties(sequences[0], int_to_note)
        
        assert props is not None
        assert props['num_unique_pitches'] > 4  # Multiple unique pitches from chords
    
    def test_varied_pitch_ranges(self):
        """Test sequences with varied pitch characteristics"""
        # Create sequences with minimum length of 20 (short range starts at 20)
        sequences = [
            list(range(0, 25)),    # Bass range: 25 notes
            list(range(25, 50)),   # Tenor range: 25 notes
            list(range(50, 75)),   # Alto range: 25 notes
            list(range(75, 100)),  # Soprano range: 25 notes
        ]
        
        int_to_note = {}
        
        # Bass: avg = (36+40+43)/3 = 39.67 (below low range 40-60)
        bass_pitches = ['36', '40', '43']
        for i in range(0, 25):
            int_to_note[i] = bass_pitches[i % 3]
        
        # Tenor: avg = (48+52+55)/3 = 51.67 (in low range 40-60)
        tenor_pitches = ['48', '52', '55']
        for i in range(25, 50):
            int_to_note[i] = tenor_pitches[(i - 25) % 3]
        
        # Alto: avg = (60+64+67)/3 = 63.67 (in medium range 60-72)
        alto_pitches = ['60', '64', '67']
        for i in range(50, 75):
            int_to_note[i] = alto_pitches[(i - 50) % 3]
        
        # Soprano: avg = (72+76+79)/3 = 75.67 (in high range 72-84)
        soprano_pitches = ['72', '76', '79']
        for i in range(75, 100):
            int_to_note[i] = soprano_pitches[(i - 75) % 3]
        
        # Test low preference (40-60): should match seq 1 (tenor)
        seed_low = find_seed_by_characteristics(
            sequences, int_to_note,
            pitch_preference="low", complexity="simple", length="short"
        )
        assert seed_low == sequences[1]  # tenor range
        
        # Test medium preference (60-72): should match seq 2 (alto)
        seed_medium = find_seed_by_characteristics(
            sequences, int_to_note,
            pitch_preference="medium", complexity="simple", length="short"
        )
        assert seed_medium == sequences[2]  # alto range
        
        # Test high preference (72-84): should match seq 3 (soprano)
        seed_high = find_seed_by_characteristics(
            sequences, int_to_note,
            pitch_preference="high", complexity="simple", length="short"
        )
        assert seed_high == sequences[3]  # soprano range


class TestRandomization:
    """Test randomization behavior"""
    
    def test_random_selection_from_candidates(self):
        """Test that selection is random when multiple candidates exist"""
        sequences = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        
        # All sequences have same properties
        int_to_note = {i: '60' for i in range(9)}
        
        # Mock random to control selection
        with patch('numpy.random.randint') as mock_randint:
            mock_randint.return_value = 1
            
            seed = find_seed_by_characteristics(
                sequences, int_to_note,
                pitch_preference="medium", complexity="simple", length="short"
            )
            
            assert seed == [3, 4, 5]  # Second sequence
    
    def test_fallback_random_selection(self):
        """Test random fallback when no matches found"""
        sequences = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ])
        int_to_note = {i: '60' for i in range(6)}
        
        with patch('numpy.random.randint') as mock_randint:
            mock_randint.return_value = 1
            
            seed = find_seed_by_characteristics(
                sequences, int_to_note,
                pitch_preference="high",  # Won't match
                complexity="complex",  # Won't match
                length="long"  # Won't match
            )
            
            assert seed == [3, 4, 5]  # Second sequence (index 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
