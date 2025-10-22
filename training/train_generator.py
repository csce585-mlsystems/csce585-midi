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
    def __init__(self, sequences, seq_length=DEFAULT_SEQ_LENGTH):
        # sequences is a list of lists of integers
        self.data = []
        for seq in sequences:
            seq = list(map(int, seq))  # ensure integers

            # skip sequences shorter than seq_length
            if len(seq) < seq_length:
                continue
            # iterate over the sequence to create input-target pairs
            for i in range(len(seq) - seq_length):
                # create input and target sequences
                input_seq = seq[i:i+seq_length]
                # target sequence starting at index 1 to index[length+1] (to predict next note)
                target = seq[i+1:i+seq_length+1]
                self.data.append((input_seq, target))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get input and target sequences
        # input_seq, target = self.data[idx]
        # # convert to tensors
        # return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
        x, y = self.data[idx]
        x = [int(i) for i in x]  # ensure integers
        y = [int(i) for i in y]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

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
          d_model=256, nhead=8, dim_feedforward=1024, dropout=0.2):
    
    # directories
    DATA_DIR = Path(f"data/{dataset}")
    MODEL_DIR = Path(f"models/{dataset}")
    OUTPUT_DIR = Path(f"outputs/{dataset}")
    LOG_FILE = f"logs/{dataset}/models.csv" #(need to go to correct folder based on preprocessing used)

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

    # create dataset and dataloader
    dataset = MIDIDataset(sequences, seq_length=seq_length)
    # Note: num_workers=0 to avoid multiprocessing issues on macOS with MPS
    # pin_memory=False because MPS doesn't support it
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

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

    # training loop
    start_time = time.time() # start time tracking
    for epoch in range(epochs):
        # initialize loss
        epoch_loss = 0
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

        # average loss for the epoch
        avg_loss = epoch_loss / (batch_idx + 1)
        # track losses
        losses.append(avg_loss)
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
        "train_time_sec": round(total_time, 2)
    }

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
        max_batches=args.max_batches
    )