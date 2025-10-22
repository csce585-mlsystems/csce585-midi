#!/usr/bin/env python3
"""
Quick discriminator test - no sklearn dependencies
"""

import time
import torch
import numpy as np
from pathlib import Path
import pickle
from torch.utils.data import DataLoader, Dataset
from models.discriminators.discriminator_factory import get_discriminator

class MeasureDataset(Dataset):
    def __init__(self, sequences, context_measures=4):
        self.examples = []
        # iterate over all sequences
        for seq in sequences:
            # convert to numpy array
            seq = np.array(seq, dtype=object)
            # get sequence length
            L = len(seq)
            # if length is less than context_measures (not enough measures), skip (can't form a full context)
            if L <= context_measures:
                continue
            # iterate over the sequence to create (context, target) pairs
            for i in range(L - context_measures):
                context = np.stack(seq[i:i+context_measures]).astype(np.uint8)  # (M, P)
                target = np.array(seq[i+context_measures]).astype(np.uint8)  # (P,)
                self.examples.append((context, target))

    # method to get length of dataset (how many examples)
    def __len__(self): return len(self.examples)

    # method to get a specific example by index
    def __getitem__(self, idx):
        context, target = self.examples[idx]
        return torch.tensor(context, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def quick_test():
    """Quick test without sklearn metrics"""
    print("🧪 Quick discriminator test (no sklearn)")
    
    # Load data
    data_dir = Path("data/measures")
    sequences = np.load(data_dir/"measure_sequences.npy", allow_pickle=True)
    with open(data_dir/"pitch_vocab.pkl", "rb") as f:
        pv = pickle.load(f)
    pitch_dim = len(pv["vocab"])
    
    # Create small dataset
    train_ds = MeasureDataset(sequences[:10], context_measures=4)  # Just 10 sequences
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    # Test each model type
    models = ["mlp", "lstm", "transformer"]
    
    for model_type in models:
        print(f"\n🔧 Testing {model_type.upper()} model...")
        
        try:
            # Auto-detect device
            device_str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_str)
            print(f"Device: {device_str}")
            
            # Create model
            model = get_discriminator(model_type, pitch_dim, context_measures=4,
                                    hidden1=256, hidden2=128, embed_size=64, hidden_size=128).to(device)
            
            # Test forward pass
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                print(f"Input shape: {x.shape}, Output shape: {logits.shape}")
                break
            
            print(f"✅ {model_type.upper()} model working!")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} model failed: {e}")
    
    print(f"\n🎉 Test complete!")

if __name__ == "__main__":
    quick_test()