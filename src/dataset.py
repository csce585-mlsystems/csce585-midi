import torch, random
from torch.utils.data import IterableDataset, DataLoader
from src.config import Model, TOKENIZER, Training
import pytorch_lightning as pl
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
                print("Loading shard:", file)
                data = torch.load(file)
                random.shuffle(data)

                for item in data:
                    cond_tokens = []

                    # prepend conditioning
                    if "genre" in item and item["genre"] in TOKENIZER.vocab:
                        cond_tokens.append(TOKENIZER.vocab[item["genre"]])
                    if "emotion" in item and item["emotion"] in TOKENIZER.vocab:
                        cond_tokens.append(TOKENIZER.vocab[item["emotion"]])
                    if "programs" in item:
                        cond_tokens.extend([
                            TOKENIZER.vocab[p] for p in item["programs"] if p in TOKENIZER.vocab
                        ])
                                        
                    cond_ids = torch.tensor(cond_tokens, dtype=torch.long)
                    cond_len = len(cond_ids)

                    full_tokens = item["tokens"]

                    if len(full_tokens) > self.seq_len - cond_len:
                        # sample a random window of music tokens
                        start = random.randint(0, len(full_tokens) - (self.seq_len - cond_len))
                        music_tokens = full_tokens[start:start + (self.seq_len - cond_len)]
                    else:
                        music_tokens = full_tokens

                    tokens = torch.cat([cond_ids, music_tokens])

                    # pad if needed
                    if len(tokens) < self.seq_len:
                        pad_len = self.seq_len - len(tokens)
                        tokens = torch.cat([tokens, torch.full((pad_len,), 0, dtype=torch.long)])

                    yield tokens

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, shard_files, batch_size=Training.BATCH_SIZE, seq_len=Model.SEQ_LEN, num_workers=15):
        super().__init__()
        self.shard_files = shard_files
        self.batch_size = batch_size
        self.seq_len = seq_len

    def train_dataloader(self):
        dataset = MidiDataset(self.shard_files, self.seq_len, shuffle=True)
        return DataLoader(dataset, batch_size=self.batch_size)
    
    def validate_dataloader(self):
        dataset = MidiDataset(self.shard_files, self.seq_len, shuffle=False)
        return DataLoader(dataset, batch_size=self.batch_size)