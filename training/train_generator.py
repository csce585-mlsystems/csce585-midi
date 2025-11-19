import csv
import datetime
from pathlib import Path
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set matplotlib backend for non-interactive environments (Colab, servers)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import argparse
import time
from tqdm import tqdm

"""
    For my sanity:
        Naive Data looks like this:
            note_to_int.pkl - basically a dict holding {"note name" : associated id number (int)}
            sequences.npy - each song in the dataset turned into sequence of numbers (instead of [C, C, A, C] it would be like [43, 43, 41, 43])

        Miditok:
            config.json - holds stuff like vocab_size, tokenizer, number of sequences, etc.
            sequences.npy - stored the same way as miditok, but obviously different numbers because they have two different vocabs
            tokenizer.json - info about the tokenizer to ensure reproducibility
            vocab.json - dict of {"token name" : associated id number (int)}
"""

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
    def __init__(self, sequences, seq_length=DEFAULT_SEQ_LENGTH, subsample_ratio=0.05):
        # Store references to sequences instead of creating all samples upfront
        self.sequences = [list(map(int, seq)) for seq in sequences if len(seq) >= seq_length]
        self.seq_length = seq_length

        if subsample_ratio <= 0:
            print("Negative subsample ratio")
            raise ValueError
        
        # Calculate indices for each sequence
        self.sequence_indices = []
        for seq_idx, seq in enumerate(self.sequences):
            # num_samples is number of possible starting indexes that don't go out of bounds with given seq_length
            num_samples = len(seq) - seq_length
            if subsample_ratio < 1.0:
                # Subsample by taking every Nth sample
                step = max(1, int(1.0 / subsample_ratio))
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
          subsample_ratio=1.0, patience=None, val_split=0.0, checkpoint_dir=None):
    
    # directories
    DATA_DIR = Path(f"data/{dataset}")
    
    # Use checkpoint_dir if provided (for Google Drive), otherwise use local directories
    if checkpoint_dir:
        BASE_DIR = Path(checkpoint_dir)
        MODEL_DIR = BASE_DIR / dataset / "models"
        OUTPUT_DIR = BASE_DIR / dataset / "outputs"
        LOG_DIR = BASE_DIR / dataset / "logs"
        print(f"Using checkpoint directory: {BASE_DIR}")
        print(f"  Models: {MODEL_DIR}")
        print(f"  Outputs: {OUTPUT_DIR}")
        print(f"  Logs: {LOG_DIR}")
    else:
        MODEL_DIR = Path(f"models/generators/checkpoints/{dataset}")
        OUTPUT_DIR = Path(f"outputs/generators/{dataset}")
        LOG_DIR = Path(f"logs/generators/{dataset}/models")
    
    LOG_FILE = LOG_DIR / "models.csv"

    # make sure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # log_file = f"logs/{dataset}/models.csv" #(need to go to correct folder based on preprocessing used)

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
    max_id_in_data = -1
    sample_count = min(len(sequences), 100) # only check 100 sequences 

    # load vocab
    token_to_id = None
    vocab_size = None

    # if naive, determine vocab size by number of ids
    # ADD ONE BC FIRST ID = 0
    if (DATA_DIR / "note_to_int.pkl").exists(): # naive
        import pickle
        with open(DATA_DIR / "note_to_int.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        
        token_to_id = vocab_data["note_to_int"]
        # token_to_id is expected token(str)->id(int)
        # compute vocab_size as max_id+1 to be safe
        max_id = max(token_to_id.values()) if token_to_id else -1
        vocab_size = max_id + 1 if max_id >= 0 else len(token_to_id)

    elif (DATA_DIR / "vocab.json").exists(): # miditok
        with open(DATA_DIR / "vocab.json", "r") as f:
            vocab = json.load(f)

        if isinstance(vocab, dict):
            # detect whether values are ints (token->id) or strings (id->token)
            sample_val = next(iter(vocab.values()))
            if isinstance(sample_val, int):
                token_to_id = {str(k): int(v) for k, v in vocab.items()}
                max_id = max(token_to_id.values()) if token_to_id else -1
                vocab_size = max_id + 1
            else:
                # values are likely token strings with keys as ids
                token_to_id = {str(tok): idx for idx, tok in enumerate(vocab.items())}
                max_id = max(token_to_id.values()) if token_to_id else -1
                vocab_size = max_id + 1
        elif isinstance(vocab, list):
            token_to_id = {str(tok): idx for idx, tok in enumerate(vocab)}
            vocab_size = len(vocab)
        else:
            raise ValueError("unrecognized vocab.json format: expected dict or list.")
    else:
        raise FileNotFoundError(f"no vocab file found in the data directory ({DATA_DIR})")
    
    for i in range(sample_count):
        seq = sequences[i]
        if len(seq) == 0: # skip if it has 0 length
            continue
        # seq elements might already be ints. ensure you don't cast strings wrongly
        try:
            seq_max = max(int(x) for x in seq)
        except Exception as e:
            print("error: found non-int token(s) in sequences.npy - sample seq:", sequences[i][:50])
            raise
        
        max_id_in_data = max(max_id_in_data, seq_max)

    if max_id_in_data >= vocab_size:
        raise ValueError(
            f"token id in data ({max_id_in_data}) >= vocab_size ({vocab_size})."
            "this indicates vocab.json/vocab mapping and sequences are inconsistent"
        )
    
    print(f"Vocab size: {vocab_size}, Sequence length: {seq_length}, Device: {device}")
    if subsample_ratio < 1.0:
        print(f"Subsample ratio: {subsample_ratio}")
    if val_split > 0:
        print(f"Val split: {val_split}, Early stopping patience: {patience if patience else 'disabled'}")

    """
        val_split is the percentage of the dataset you want to use for validation
        (model won't be trained on these; they're only used to see how good it's
        learning to generalize)
    """
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
    
    # Configure DataLoader based on device
    # CUDA: Use multiple workers and pin memory for fast GPU transfer
    # MPS (macOS): Use single worker and no pinning due to compatibility issues
    if device == "cuda":
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create validation dataloader if needed
    if val_sequences is not None:
        val_dataset = MIDIDataset(val_sequences, seq_length=seq_length, subsample_ratio=1.0)  # Don't subsample val
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
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

    print(f"quick data check: sample_count={sample_count}, max_token_id_in_sampled_sequences={max_id_in_data}")
    print(f"interpreted vocab_size = {vocab_size} (from vocab.json)")

    # added this when trying to figure out why miditok was taking so long
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            assert module.num_embeddings >= vocab_size, (
                f"embedding num_embeddings ({module.num_embeddings}) < vocab size ({vocab_size})."
                "your model was likely built with th wrong vocab size. rebuild using correct vocab.json"
            )
            print(f"embedding check passed: {name}.num_embeddings = {module.num_embeddings}")
    
    # trying label smoothing to prevent overfitting (miditok)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1) # suitable for multi-class classification
    # added weight decay to try to fix overfitting (miditok)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # add a scheduler if miditok (changes learning rate)
    if dataset == 'miditok' or dataset == 'miditok_augmented':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
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
    
    # Calculate and display training info
    total_train_batches = len(dataloader) if not max_batches else min(max_batches, len(dataloader))
    batches_per_epoch = total_train_batches
    estimated_time_per_epoch = "calculating..."
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {batches_per_epoch:,}")
    print(f"Learning rate: {learning_rate}")
    if val_dataloader:
        print(f"Validation batches: {len(val_dataloader):,}")
        if patience:
            print(f"Early stopping patience: {patience}")
    print("="*60)
    print(f"\nLarge datasets may take several minutes per epoch")
    print(f" Watch for periodic progress updates during training\n")
    
    for epoch in range(epochs):
        
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        # Calculate how often to print updates (every 5% of batches or every 500 batches, whichever is smaller)
        total_batches = max_batches if max_batches else len(dataloader)
        print_interval = min(500, max(1, total_batches // 20))
        
        print(f"\nEpoch {epoch+1}/{epochs} - Training on {total_batches:,} batches...")
        
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

            if dataset == 'miditok' or dataset == 'miditok_augmented':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to help with overfitting (miditok)
            
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress periodically
            if batch_idx > 0 and batch_idx % print_interval == 0:
                elapsed = time.time() - epoch_start_time
                batches_per_sec = num_batches / elapsed
                eta_seconds = (total_batches - num_batches) / batches_per_sec if batches_per_sec > 0 else 0
                percent = 100 * num_batches / total_batches
                print(f"  Batch {num_batches:,}/{total_batches:,} ({percent:.1f}%) | "
                      f"Loss: {loss.item():.4f} | Avg Loss: {epoch_loss/num_batches:.4f} | "
                      f"Speed: {batches_per_sec:.1f} batch/s | ETA: {eta_seconds/60:.1f}m")

        # average loss for the epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            print(f"  Validating on {len(val_dataloader):,} batches...")
            
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device)
                    output, _ = model(x)
                    loss = loss_function(output.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item()
                    val_batches += 1
            
            # find average validation loss
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)

            if dataset == 'miditok' or dataset == 'miditok_augmented':
                scheduler.step(avg_val_loss) # call scheduler after validation (adjusts learning rate)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}")
            print(f"  Gap:        {avg_val_loss - avg_loss:+.4f}")
            
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
                    
                    # quit if model has been getting worse over the specified number of epochs (patience)
                    if patience_counter >= patience:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        print(f"{'='*60}")
                        # Restore best model
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                            print("Restored best model weights")
                        break
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            print(f"{'='*60}\n")

    # get current time stamp
    total_time = time.time() - start_time # total training time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # save the model
    model_filename = f"{model_type}_{timestamp}.pth"
    torch.save(model.state_dict(), MODEL_DIR / model_filename)
    print(f"Model saved to {MODEL_DIR / model_filename}")

    # Create plots directory
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # plot the training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Loss Curves - {model_type.upper()}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save the plot
    plt.savefig(plots_dir / f"loss_curve_{model_type}_{timestamp}.png", dpi=150)
    print(f"Loss plot saved to {plots_dir / f'loss_curve_{model_type}_{timestamp}.png'}")
    plt.close('all')  # close stuff to free memory

    # Save training summary as text file
    summary_file = LOG_DIR / f"training_summary_{model_type}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"TRAINING SUMMARY - {model_type.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Vocab Size: {vocab_size}\n")
        f.write(f"  Sequence Length: {seq_length}\n")
        if model_type in ["lstm", "gru"]:
            f.write(f"  Embedding Size: {embed_size}\n")
            f.write(f"  Hidden Size: {hidden_size}\n")
            f.write(f"  Num Layers: {num_layers}\n")
        elif model_type == "transformer":
            f.write(f"  d_model: {d_model}\n")
            f.write(f"  nhead: {nhead}\n")
            f.write(f"  Transformer Layers: {num_layers}\n")
            f.write(f"  Feedforward Dim: {dim_feedforward}\n")
        f.write(f"  Dropout: {dropout}\n")
        f.write(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Learning Rate: {learning_rate}\n")
        f.write(f"  Subsample Ratio: {subsample_ratio}\n")
        if val_split > 0:
            f.write(f"  Validation Split: {val_split}\n")
            f.write(f"  Early Stopping Patience: {patience if patience else 'disabled'}\n")
        f.write(f"\nTraining Results:\n")
        f.write(f"  Total Time: {total_time/60:.2f} minutes\n")
        f.write(f"  Epochs Completed: {len(losses)}/{epochs}\n")
        f.write(f"  Final Training Loss: {losses[-1]:.4f}\n")
        f.write(f"  Best Training Loss: {min(losses):.4f}\n")
        if val_losses:
            f.write(f"  Final Validation Loss: {val_losses[-1]:.4f}\n")
            f.write(f"  Best Validation Loss: {best_val_loss:.4f}\n")
            f.write(f"  Train/Val Gap: {val_losses[-1] - losses[-1]:+.4f}\n")
        f.write(f"\nModel saved to: {MODEL_DIR / model_filename}\n")
        f.write(f"Plot saved to: {plots_dir / f'loss_curve_{model_type}_{timestamp}.png'}\n")
        f.write("="*80 + "\n")
    
    print(f"Training summary saved to {summary_file}")

    # Save loss history as numpy arrays for later analysis
    np.save(LOG_DIR / f"train_losses_{model_type}_{timestamp}.npy", np.array(losses))
    if val_losses:
        np.save(LOG_DIR / f"val_losses_{model_type}_{timestamp}.npy", np.array(val_losses))
    print(f"Loss arrays saved to {LOG_DIR}")


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
    log_experiment(hparams, results, LOG_FILE)

# allow command line arguments for hyperparameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a music generator model")
    
    # Model architecture
    parser.add_argument("--model_type", type=str, choices=["lstm", "gru", "transformer"],
                        default="lstm", help="Type of generator architecture")
    
    # Data and training
    parser.add_argument("--dataset", type=str, choices=["naive", "miditok", "miditok_augmented"],
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
    
    # Checkpoint directory (for Google Drive)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints (e.g., '/content/drive/MyDrive/model_checkpoints'). If not provided, saves locally.")
    
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
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir
    )