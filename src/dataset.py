import torch, random
from torch.utils.data import IterableDataset
from config import Model

class MidiDataset(IterableDataset):
    def __init__(self, shard_files, seq_len = Model.SEQ_LEN, shuffle=True):
        self.shard_files = shard_files
        self.seq_len = seq_len
        self.shuffle = shuffle
    
    def __iter__(self):
        shard_files = list(self.shard_files)
        if self.shuffle:
            random.shuffle(shard_files)
        
        for file in self.shard_files:
            data = torch.load(file)
            random.shuffle(data)

            for tokens, genre, emotion, programs in data:
                tokens = tokens[:self.seq_len] #truncate if too long
                x = torch.tensor(tokens, dtype=torch.long)

                if len(x) < self.seq_len:
                    pad_len = self.seq_len - len(x)
                    x = torch.cat([x, torch.full((pad_len,), 0)])
                yield x