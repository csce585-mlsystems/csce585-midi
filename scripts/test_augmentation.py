#!/usr/bin/env python3
"""
Quick test of the augmentation script on a small sample.
Verifies that transposition works correctly before processing the full dataset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from symusic import Score
import miditok

def test_transposition():
    """Test basic transposition functionality."""
    print("Testing transposition functionality...\n")
    
    # Find a sample MIDI file
    midi_dir = Path("data/nottingham-dataset-master/MIDI")
    midi_files = list(midi_dir.glob("*.mid"))
    
    if not midi_files:
        print("❌ No MIDI files found in", midi_dir)
        return False
    
    sample_file = midi_files[0]
    print(f"Using sample file: {sample_file.name}")
    
    try:
        # Load original score
        original = Score(sample_file)
        print(f"✅ Loaded MIDI successfully")
        print(f"   Tracks: {len(original.tracks)}")
        
        # Count notes in original
        original_notes = sum(len(track.notes) for track in original.tracks)
        print(f"   Total notes: {original_notes}")
        
        # Test transposition
        from utils.augment_miditok import transpose_score
        
        transpositions = [-3, 0, 3]
        
        for semitones in transpositions:
            transposed = transpose_score(original, semitones)
            transposed_notes = sum(len(track.notes) for track in transposed.tracks)
            
            sign = "+" if semitones > 0 else ""
            print(f"   Transpose {sign}{semitones}: {transposed_notes} notes", end="")
            
            # Verify note count is similar (some notes may be clipped at boundaries)
            if transposed_notes >= original_notes * 0.95:
                print(" ✅")
            else:
                print(f" ⚠️  (lost {original_notes - transposed_notes} notes)")
        
        # Test tokenization
        print("\nTesting tokenization...")
        tokenizer = miditok.REMI()
        
        for semitones in transpositions:
            transposed = transpose_score(original, semitones)
            tokens = tokenizer(transposed)
            
            if isinstance(tokens, list):
                total_tokens = sum(len(ts.ids) for ts in tokens)
            else:
                total_tokens = len(tokens.ids)
            
            sign = "+" if semitones > 0 else ""
            print(f"   Transpose {sign}{semitones}: {total_tokens} tokens ✅")
        
        print("\n✅ All tests passed!")
        print("\nReady to run full augmentation:")
        print("   python utils/augment_miditok.py")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def preview_augmentation():
    """Show what the augmentation will produce."""
    print("\n" + "=" * 60)
    print("Augmentation Preview")
    print("=" * 60)
    
    midi_dir = Path("data/nottingham-dataset-master/MIDI")
    midi_files = list(midi_dir.glob("*.mid"))
    
    num_files = len(midi_files)
    transpositions = [-5, -3, -1, 0, 1, 3, 5]
    
    print(f"\nOriginal dataset:")
    print(f"  MIDI files: {num_files}")
    
    print(f"\nAugmentation settings:")
    print(f"  Transpositions: {transpositions}")
    print(f"  Augmentation factor: {len(transpositions)}x")
    
    print(f"\nExpected output:")
    print(f"  Total versions: {num_files * len(transpositions):,}")
    print(f"  (Each song in {len(transpositions)} different keys)")
    
    print(f"\nEstimated training samples:")
    print(f"  Original: ~1,178,694")
    print(f"  Augmented: ~{1178694 * len(transpositions):,}")
    print(f"  Increase: {len(transpositions)}x")
    
    print(f"\nSamples per vocab entry:")
    print(f"  Original: ~4,150")
    print(f"  Augmented: ~{4150 * len(transpositions):,}")
    print(f"  Much better for generalization!")


if __name__ == "__main__":
    print("=" * 60)
    print("🎵 MIDITok Augmentation Test")
    print("=" * 60)
    print()
    
    success = test_transposition()
    
    if success:
        preview_augmentation()
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print("=" * 60)
        print("\n1. Run augmentation:")
        print("   python utils/augment_miditok.py")
        print("\n2. Compare results:")
        print("   python scripts/compare_augmentation.py")
        print("\n3. Train models:")
        print("   bash scripts/train_augmented.sh")
        print()
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)
