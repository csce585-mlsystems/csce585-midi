import csv
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from models.lstm import LSTMGenerator

# make sure outputs directory exists
import os
os.makedirs("outputs", exist_ok=True)

# hyperparameters
SEQ_LENGTH = 50 # notes per input sequence
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# model log file
LOG_FILE = "logs/models.csv"

# dataset
class MIDIDataset(Dataset):
    def __init__(self, sequences, seq_length=SEQ_LENGTH):
        # sequences is a list of lists of integers
        self.data = []
        for seq in sequences:
            # iterate over the sequence to create input-target pairs
            for i in range(len(seq) - seq_length):
                # create input and target sequences
                input_seq = seq[i:i+seq_length]
                # target is next note after the input sequence
                target = seq[i+1:i+seq_length+1]
                self.data.append((input_seq, target))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get input and target sequences
        input_seq, target = self.data[idx]
        # convert to tensors
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
def train(max_batches=None):
    # load the data
    sequences = np.load("data/sequences.npy", allow_pickle=True)
    # size of possible tokens
    vocab_size = max(max(seq) for seq in sequences) + 1

    # create dataset and dataloader
    dataset = MIDIDataset(sequences, seq_length=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialize the model, loss function, and optimizer
    model = LSTMGenerator(vocab_size).to(DEVICE)
    loss_function = nn.CrossEntropyLoss() # suitable for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # to track loss
    losses = []

    # training loop
    for epoch in range(EPOCHS):
        # initialize loss
        epoch_loss = 0
        # iterate over batch indexes and data
        for batch_idx, (x, y) in enumerate(dataloader):
            # limit number of batches for quick testing
            if max_batches and batch_idx >= max_batches:
                break
            # move data to device (tensors)
            x, y = x.to(DEVICE), y.to(DEVICE)
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
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # get current time stamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # save the model
    torch.save(model.state_dict(), f"models/lstm_{timestamp}.pth")

    # plot the loss curve
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    # save the plot
    plt.savefig(f"outputs/training_loss/training_loss_{timestamp}.png")
    plt.show()

    # return hyperparameters and results for logging
    hparams = {
        "timestamp": timestamp,
        "vocab_size": vocab_size,
        "seq_length": SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "device": DEVICE,
        "hidden_size": model.hidden_size if hasattr(model, "hidden_size") else None,
    }

    results = {
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "loss_std": np.std(losses),
    }

    # log the experiment
    log_experiment(hparams, results)

# log the info about the model trained
def log_experiment(hparams, results, logfile=LOG_FILE):
    # make sure logs directory exists
    Path("logs").mkdir(exist_ok=True)

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

if __name__ == "__main__":
    train()