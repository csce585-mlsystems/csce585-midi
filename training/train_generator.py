import csv
import datetime
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm

# Add project root to path so we can import from models/
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.generators.generator_factory import get_generator, get_default_config

# make sure outputs directory exists
import os
os.makedirs("outputs", exist_ok=True)

# hyperparameters
#SEQ_LENGTH = 50 # notes per input sequence (getting rid of this and using config from preprocess_miditok.py)
# BATCH_SIZE = 32
# EPOCHS = 10
# LEARNING_RATE = 0.001
# DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
# model log file

DEFAULT_SEQ_LENGTH = 50

# LOG_FILE = f"logs/{dataset}/models.csv" #(need to go to correct folder based on preprocessing used)

# dataset
class MIDIDataset(Dataset):
    """Lazy-loading dataset that generates samples on-the-fly to avoid memory issues."""
    def __init__(self, sequences, seq_length=DEFAULT_SEQ_LENGTH, subsample_ratio=1.0):
        # Store references to sequences instead of creating all samples upfront
        self.sequences = [list(map(int, seq)) for seq in sequences if len(seq) >= seq_length]
        self.seq_length = seq_length
        
        # Calculate indices for each sequence
        self.sequence_indices = []
        for seq_idx, seq in enumerate(self.sequences):
            # num_samples is number of possible starting indexes that don't go out of bounds with given seq_length
            num_samples = len(seq) - seq_length
            if subsample_ratio < 1.0:
                # Subsample by taking every Nth sample
                step = int(1.0 / subsample_ratio)
                indices = list(range(0, num_samples, step))
            else:
                # enumerate each starting index
                indices = list(range(num_samples))
            
            for i in indices:
                self.sequence_indices.append((seq_idx, i)) # (which seq it's from, starting idx)
        
        print(f"Dataset initialized: {len(self.sequence_indices):,} samples from {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequence_indices) # amount of sequences
    
    def __getitem__(self, idx):
        # Generate sample on-the-fly
        seq_idx, start_idx = self.sequence_indices[idx]
        seq = self.sequences[seq_idx]
        
        input_seq = seq[start_idx:start_idx + self.seq_length]
        target = seq[start_idx + 1:start_idx + self.seq_length + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def log_experiment(hparams, results, logfile):
    # make sure logs directory exists
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)

    # fieldnames (names of columns) are the keys of hparams and results
    # hparams is a dictionary of hyperparameters
    # results is a dictionary of results (like final loss, accuracy, etc)
    fieldnames = list(hparams.keys()) + list(results.keys())
    # check if the log file exists
    file_exists = os.path.isfile(logfile)

    # open the file in append mode
    with open(logfile, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # write header if file is new
        if not file_exists:
            writer.writeheader()
        # write the experiment data
        writer.writerow({**hparams, **results})
    
def train(model_type="lstm", dataset="naive", embed_size=128, hidden_size=256, num_layers=2,
          batch_size=32, epochs=10, learning_rate=0.001, device=None, max_batches=None,
          d_model=256, nhead=8, dim_feedforward=1024, dropout=0.2, 
          subsample_ratio=1.0, patience=None, val_split=0.0):
    
    # directories
    DATA_DIR = Path(f"data/{dataset}")
    MODEL_DIR = Path(f"models/generators/checkpoints/{dataset}")
    OUTPUT_DIR = Path(f"outputs/generators/{dataset}")
    LOG_FILE = f"logs/generators/{dataset}/models.csv" #(need to go to correct folder based on preprocessing used)

    # make sure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = f"logs/{dataset}/models.csv" #(need to go to correct folder based on preprocessing used)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # load config file to get sequence length (if it exists)
    import json
    config_file = DATA_DIR / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
        seq_length = config.get("seq_length", DEFAULT_SEQ_LENGTH) # default to 50
    else:
        # No config file, use default
        seq_length = DEFAULT_SEQ_LENGTH

    # load the data
    sequences = np.load(DATA_DIR / "sequences.npy", allow_pickle=True)

    # size of possible tokens
    #vocab_size = max(max(seq) for seq in sequences) + 1

    # load vocab
    if (DATA_DIR / "note_to_int.pkl").exists(): # naive
        import pickle
        # load the note-to-int mapping
        with open(DATA_DIR / "note_to_int.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        vocab_size = len(vocab_data["note_to_int"])
    elif (DATA_DIR / "vocab.json").exists(): # miditok
        # load the vocab json file
        with open(DATA_DIR / "vocab.json", "r") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
    else:
        raise FileNotFoundError("No vocabulary file found in the data directory.")
    
    print(f"Vocab size: {vocab_size}, Sequence length: {seq_length}, Device: {device}")
    if subsample_ratio < 1.0:
        print(f"Subsample ratio: {subsample_ratio}")
    if val_split > 0:
        print(f"Val split: {val_split}, Early stopping patience: {patience if patience else 'disabled'}")

    # Split data into train and validation
    if val_split > 0:
        split_idx = int(len(sequences) * (1 - val_split))
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        print(f"Train sequences: {len(train_sequences)}, Val sequences: {len(val_sequences)}")
    else:
        train_sequences = sequences
        val_sequences = None

    # create dataset and dataloader with lazy loading
    train_dataset = MIDIDataset(train_sequences, seq_length=seq_length, subsample_ratio=subsample_ratio)
    # Note: num_workers=0 to avoid multiprocessing issues on macOS with MPS
    # pin_memory=False because MPS doesn't support it
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    # Create validation dataloader if needed
    if val_sequences is not None:
        val_dataset = MIDIDataset(val_sequences, seq_length=seq_length, subsample_ratio=1.0)  # Don't subsample val
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    else:
        val_dataloader = None

    # initialize the model using factory pattern
    model = get_generator(
        model_type, 
        vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    loss_function = nn.CrossEntropyLoss() # suitable for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model type: {model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # to track loss
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # training loop
    start_time = time.time() # start time tracking
    print(f"Starting training at: {start_time}")
    
    for epoch in range(epochs):
        
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0

        # iterate over batch indexes and data
        for batch_idx, (x, y) in enumerate(dataloader):
            # limit number of batches for quick testing
            if max_batches and batch_idx >= max_batches:
                break
            # move data to device (tensors)
            x, y = x.to(device), y.to(device)
            # forward pass, backward pass, optimize
            optimizer.zero_grad()
            # get the output from the model
            output, _ = model(x)
            # figure out the loss
            loss = loss_function(output.view(-1, vocab_size), y.view(-1))
            # backpropagation and optimization step
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()
            num_batches += 1

        # average loss for the epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device)
                    output, _ = model(x)
                    loss = loss_function(output.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check (only if patience is set)
            if patience is not None and patience > 0:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"  ✗ Val loss increased ({patience_counter}/{patience})")
                    
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        # Restore best model
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                            print("Restored best model weights")
                        break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # get current time stamp
    total_time = time.time() - start_time # total training time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # save the model
    model_filename = f"{model_type}_{timestamp}.pth"
    torch.save(model.state_dict(), MODEL_DIR / model_filename)
    print(f"Model saved to {MODEL_DIR / model_filename}")

    # plot the loss curve
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    # save the plot
    loss_plot_dir = Path(f"{OUTPUT_DIR}/training_loss")
    loss_plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(loss_plot_dir / f"training_loss_{timestamp}.png")
    #plt.show()

    # return hyperparameters and results for logging
    hparams = {
        "timestamp": timestamp,
        "model_type": model_type,
        "dataset": dataset,
        "vocab_size": vocab_size,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": device,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    
    # Add model-specific hyperparameters
    if model_type in ["lstm", "gru"]:
        hparams.update({
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        })
    elif model_type == "transformer":
        hparams.update({
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        })

    results = {
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "loss_std": np.std(losses),
        "train_time_sec": round(total_time, 2),
        "num_epochs_trained": len(losses),
        "early_stopped": len(losses) < epochs
    }
    
    # Add validation results if available
    if val_losses:
        results.update({
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "min_val_loss": min(val_losses),
        })

    # log the experiment
    log_experiment(hparams, results, log_file)

# allow command line arguments for hyperparameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a music generator model")
    
    # Model architecture
    parser.add_argument("--model_type", type=str, choices=["lstm", "gru", "transformer"],
                        default="lstm", help="Type of generator architecture")
    
    # Data and training
    parser.add_argument("--dataset", type=str, choices=["naive", "miditok"],
                        default="naive", help="Which dataset preprocessing to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_batches", type=int, default=None, 
                        help="Limit number of batches per epoch for quick testing")
    
    # LSTM/GRU architecture parameters
    parser.add_argument("--embed_size", type=int, default=128, 
                        help="Embedding size (for LSTM/GRU)")
    parser.add_argument("--hidden_size", type=int, default=256, 
                        help="Hidden size (for LSTM/GRU)")
    parser.add_argument("--num_layers", type=int, default=2, 
                        help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    
    # Transformer architecture parameters
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension (for Transformer)")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads (for Transformer)")
    parser.add_argument("--dim_feedforward", type=int, default=1024,
                        help="Feedforward dimension (for Transformer)")
    
    # Optimization parameters
    parser.add_argument("--subsample_ratio", type=float, default=1.0,
                        help="Ratio of dataset to use (0.0-1.0). Use <1.0 for faster training on large datasets")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Validation split ratio (0.0-1.0). Set to 0 to disable validation.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (epochs). Only used if val_split > 0. Set to None to disable.")
    
    args = parser.parse_args()

    train(
        model_type=args.model_type,
        dataset=args.dataset, 
        embed_size=args.embed_size,
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        learning_rate=args.lr,
        max_batches=args.max_batches,
        subsample_ratio=args.subsample_ratio,
        val_split=args.val_split,
        patience=args.patience
    )