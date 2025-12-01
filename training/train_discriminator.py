import argparse, time, json, pickle
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score

# Add project root to path so we can import from models/
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.discriminators.discriminator_factory import get_discriminator
import csv, os

# dataset
class MeasureDataset(Dataset):
    def __init__(self, sequences, context_measures=4):
        self.examples = []
        # iterate over all sequences
        for seq in sequences:
            # convert to numpy array for easier slicing
            seq = np.array(seq, dtype=object)

            # skip sequences shorter than context
            L = len(seq)
            if L <= context_measures:
                continue

            # create (context, target) pairs
            for i in range(L - context_measures):
                context = np.stack(seq[i:i+context_measures]).astype(np.float32)  # shape (context_measures, pitch_dim)
                target = np.array(seq[i+context_measures], dtype=np.float32)  # next measure (pitch_dim,)
                self.examples.append((context, target))

    def __len__(self):
        return len(self.examples)
    
    # return context, target, and index as tensors
    def __getitem__(self, idx):
        context, target = self.examples[idx]
        return torch.tensor(context, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), idx
    
# metrics utils
def topk_coverage(pred_probs, targets, k=5):
    """ pred_probs is (N, pitch_dim) - predicted probabilities for each pitch
        targets is (N, pitch_dim) - true binary vectors of pitches
        k is number of top predictions to consider"""

    topk = np.argsort(pred_probs, axis=1)[:, -k:]

    hits = 0
    total = 0

    # iterate over each example
    for i in range(pred_probs.shape[0]):
        # get true active pitch indices
        true_idxs = np.where(targets[i] > 0.5)[0]

        if len(true_idxs) == 0:
            continue  # skip if no true pitches

        # add 1 if any true pitch is in top-k predictions
        total += 1

        # check if any true pitch is in top-k predictions
        if any(t in topk[i] for t in true_idxs): hits += 1
    
    return hits / total if total > 0 else 0.0

# eval function that supports pitches (multi label) and chords (multi class)
def evaluate_model(model, loader, device, label_mode='pitches', threshold=0.5):
    # model to eval mode
    model.eval()

    all_pred = []
    all_true = []
    all_probs = []

    # no grad context
    with torch.no_grad():
        # for each batch in the dataloader
        # y is (B, pitch_dim) for chords or (B, pitch_dim) for pitches
        for x, y, _ in loader:
            x = x.to(device)
            out = model(x)
            
            # handle different output formats (dict, tuple, tensor)
            if isinstance(out, dict):
                # extract logits and intent
                logits = out.get('chord_logits') if 'chord_logits' in out else out.get('logits')
                intent = out.get('intent', None)
            elif isinstance(out, (list, tuple)) and len(out) >= 1:
                logits = out[0]
            else:
                logits = out

            # move logits/probs to CPU numpy
            if label_mode == 'pitches':
                # multi-label binary setup
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > threshold).astype(int)
                all_probs.append(probs)
                all_pred.append(preds)
                all_true.append(y.numpy().astype(int))

            else:
                # chords: logits shatpe (B, num_classes) (B is batch size)
                # if y is one-hot vector (P,), we need to convert to class index
                # assume y is piano-roll; convert to class by argmax (best chord label) if needed
                # supply chord labels as int indices if using 'chords' mode

                # find the true class indices (true_idx represents the true class index for each sample)
                # true class means the index of the class with the highest value in y
                if y.dim() == 2 and y.size(1) > 1:
                    # treat y as one-hot / multi-hot and convert to class index by argmax
                    true_idx = np.argmax(y.numpy(), axis=1)
                else:
                    true_idx = y.numpy().astype(int).squeeze()

                # get probs and preds
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_pred.append(preds)
                all_true.append(true_idx)

    # After collecting all batches, compute metrics
    # stack / flatten metrics for pitches
    if label_mode == 'pitches':
        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)
        all_probs = np.vstack(all_probs)
        micro_f1 = f1_score(all_true.flatten(), all_pred.flatten(), zero_division=0)
        micro_precision = precision_score(all_true.flatten(), all_pred.flatten(), zero_division=0)
        micro_recall = recall_score(all_true.flatten(), all_pred.flatten(), zero_division=0)
        topk = topk_coverage(all_probs, all_true, k=5)
        # return a dict of metrics
        return {"micro_f1": micro_f1, "micro_precision": micro_precision,
                 "micro_recall": micro_recall, "topk_coverage": topk}
    
    # stack / flatten metrics for chords
    else:
        all_pred = np.concatenate(all_pred, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        all_probs = np.vstack(all_probs)
        micro_f1 = f1_score(all_true, all_pred, average='micro', zero_division=0)
        micro_precision = precision_score(all_true, all_pred, average='micro', zero_division=0)
        micro_recall = recall_score(all_true, all_pred, average='micro', zero_division=0)
        
        # topk for multiclass
        topk = 0
        K = min(5, all_probs.shape[1])
        topk_idxs = np.argsort(all_probs, axis=1)[:, -K:]
        hits = np.sum([1 if all_true[i] in topk_idxs[i] else 0 for i in range(len(all_true))])
        topk = hits / len(all_true) if len(all_true) > 0 else 0.0
        return {"micro_f1": micro_f1, "micro_precision": micro_precision,
                 "micro_recall": micro_recall, "topk_coverage": topk}

# util to extract intent vectors for generator use
def extract_and_save_intents(model, dataset, device, out_path, label_mode='chords', batch_size=256):
    """
    Runs model over dataset (no shuffle) and saves intent vectors and chord probs (if available)
    as numpy arrays aligned by dataset index.

    an intent vector is a fixed-size representation of the user's intention,
    typically derived from the input data (e.g., text, audio) and used to guide the
    generation process in a controlled manner.
    """

    # dataloader (num_workers=0 to avoid multiprocessing issues)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # set model to eval mode
    model.eval()
    # number of examples
    N = len(dataset)

    intents = None
    chord_probs = None

    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            #model may return dict / tuple / logits
            out = model(x)
            # handle dict output
            if isinstance(out, dict):
                intent = out.get('intent', None)
                chord_logits = out.get('chord_logits', None)
            elif isinstance(out, (list, tuple)) and len(out) >= 2:
                chord_logits, intent = out[0].cpu(), out[1].cpu()
            # handle single output
            elif isinstance(out, (list, tuple)) and len(out) == 1:
                chord_logits = out[0].cpu()
                intent = None
            # handle tensor output
            else:
                # only logits returned
                chord_logits = out
                intent = None

            # move to cpu numpy
            if intent is not None:
                intent = intent.cpu().numpy()
                # allocate intents array if needed
                if intents is None:
                    intents = np.zeros((N, intent.shape[1]), dtype=np.float32)
                # fill in intents by index
                for i, idd in enumerate(idx.numpy()):
                    intents[idd] = intent[i]
            
            # move chord logits to cpu numpy and convert to probs
            if chord_logits is not None:
                # convert to probs
                probs = torch.softmax(chord_logits, dim=-1).cpu().numpy()
                # allocate chord_probs array if needed
                if chord_probs is None:
                    chord_probs = np.zeros((N, probs.shape[1]), dtype=np.float32)
                # fill in chord_probs by index
                for i, idd in enumerate(idx.numpy()):
                    chord_probs[idd] = probs[i]
    
    # save
    out_path = Path(out_path)
    # make parent dirs if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # save intents and chord_probs if available
    if intents is not None:
        np.save(out_path.with_suffix('.intents.npy'), intents)
        print(f"Saved intents to {out_path.with_suffix('.intents.npy')}")
    # save chord_probs if available
    if chord_probs is not None:
        np.save(out_path.with_suffix('.chord_probs.npy'), chord_probs)
        print(f"Saved chord probabilities to {out_path.with_suffix('.chord_probs.npy')}")
    return

# training function
def train(args):
    # load the data
    data_dir = Path(args.data_dir)
    sequences = np.load(data_dir/"measure_sequences.npy", allow_pickle=True)

    # load pitch vocab to get pitch_dim
    with open(data_dir/"pitch_vocab.pkl", "rb") as f:
        pitch_vocab = pickle.load(f)
    pitch_dim = len(pitch_vocab["vocab"])

    # shuffle and split
    np.random.shuffle(sequences)
    n = len(sequences)
    split = int(n * args.train_frac)
    train_seq = sequences[:split]
    val_seq = sequences[split:]

    train_dataset = MeasureDataset(train_seq, context_measures=args.context_measures)
    val_dataset = MeasureDataset(val_seq, context_measures=args.context_measures)
    # Note: num_workers=0 to avoid multiprocessing issues on macOS
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # device handling
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # build model from factory; factory should be adjusted to support label)mode if needed
    model = get_discriminator(args.model_type, pitch_dim, context_measures=args.context_measures, 
                              hidden_sizes=[args.hidden1, args.hidden2], pool=args.pool, 
                              embed_size=args.embed_size, hidden_size=args.hidden_size, 
                              label_mode=args.label_mode).to(device)

    # choose loss based on label mode
    if args.label_mode == 'pitches':
        criterion = nn.BCEWithLogitsLoss()
    else:
        # expect model to output chord logits of shape (B, num_classes)
        # and targets to be class index
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    results = {}

    for epoch in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            # adapt to model return types
            if isinstance(out, dict):
                # extract logits
                logits = out.get('chord_logits') if 'chord_logits' in out else out.get('logits')
            elif isinstance(out, (list, tuple)):
                logits = out[0]
            else:
                logits = out
            
            # compute loss based on label mode
            if args.label_mode == 'pitches':
                loss = criterion(logits, y)
            else:
                # convert y to class indices if needed
                if y.dim() == 2 and y.size(1) > 1:
                    y_idx = torch.argmax(y, dim=1).long().to(device)
                else:
                    y_idx = y.long().squeeze().to(device)
                loss = criterion(logits, y_idx)

            # backprop and step
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # compute average train loss
        train_loss = float(np.mean(losses))
        val_stats = evaluate_model(model, val_loader, device,label_mode=args.label_mode)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val F1: {val_stats['micro_f1']:.4f} - Time: {time.time()-t0:.2f}s")

        # save best model
        if val_stats["micro_f1"] > best_val:
            best_val = val_stats["micro_f1"]
            ckpt_dir = Path("models/discriminators/checkpoints") / args.model_type
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{args.model_type}_label{args.label_mode}_ctx{args.context_measures}_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

        results[epoch] = {"train_loss": train_loss, **val_stats}

    # Write CSV summary after all epochs complete
    outdir = Path("logs/discriminators")
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "train_summary.csv"
    
    # Check if file exists to determine if we need to write header
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    
    # Open in append mode to keep previous training runs
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if this is a new file
        if write_header:
            writer.writerow(["model_type", "context", "epoch", "train_loss", 
                           "micro_f1", "micro_precision", "micro_recall", "topk_coverage"])
        
        # Write each epoch's results
        for epoch, stats in results.items():
            writer.writerow([
                args.model_type,
                args.context_measures,
                epoch + 1,  # epochs are 1-indexed for display
                stats["train_loss"],
                stats["micro_f1"],
                stats["micro_precision"],
                stats["micro_recall"],
                stats["topk_coverage"]
            ])
    
    print(f"\nTraining complete! Summary saved to {summary_path}")
    print(f"Best validation F1: {best_val:.4f}")
    
    # extract and save intents for generator use if specified
    if args.save_intents:
        out_base = Path("data") / f"{args.model_type}_ctx{args.context_measures}"
        extract_and_save_intents(model, train_dataset, device, out_base / "train", label_mode=args.label_mode)
        extract_and_save_intents(model, val_dataset, device, out_base / "val", label_mode=args.label_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "lstm", "transformer"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--context_measures", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--pool", type=str, default="concat")
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--label_mode", type=str, default="pitches", choices=["pitches", "chords"],
                        help="pitches -> multi-label piano-roll prediction (BCE). chords -> multi-class chord prediction + intent vector (CrossEntropy).")
    parser.add_argument("--save_intents", action="store_true", help="Run trained model over datasets and save intent vectors and chord probs.")
    args = parser.parse_args()
    train(args)