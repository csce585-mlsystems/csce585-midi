import torch, random
from torch.utils.data import IterableDataset
from src.config import Model

class MidiDataset(IterableDataset):
    def __init__(self, shard_files, seq_len = Model.SEQ_LEN, shuffle=True):
        self.shard_files = shard_files
        self.seq_len = seq_len
        self.shuffle = shuffle
    
    def __iter__(self):
        while True:
            files = list(self.shard_files)
            if self.shuffle:
                random.shuffle(files)
            
            for file in files:
                data = torch.load(file)
                random.shuffle(data)

                for item in data:
                    tokens = item["tokens"][:self.seq_len].to(torch.long)
                    if len(tokens) < self.seq_len:
                        pad_len = self.seq_len - len(tokens)
                        tokens = torch.cat([tokens, torch.full((pad_len,), 0)])
                    yield tokens