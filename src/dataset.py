from datasets import IterableDataset
import torch
import random
from pathlib import Path
from src.config import MidiTokenization, Model

def window_chunks(ids, max_len=1024, stride=1024):
    for i in range(0, len(ids), stride):
        chunk = ids[i:i+max_len]
        if len(chunk) >= 32:
            yield {"input_ids": chunk}

from torch.utils.data import IterableDataset, DataLoader
class ShardedMidiDataset(IterableDataset):
    def __iter__(self):
        shards = list(Path(MidiTokenization.OUT_PATH).glob("shard_*.pt"))
        random.shuffle(shards)
        for shard in shards:
            sequences = torch.load(shard)
            random.shuffle(sequences)
            for ids in sequences:
                for sample in window_chunks(ids, Model.SEQ_LEN, stride=Model.SEQ_LEN):
                    yield sample