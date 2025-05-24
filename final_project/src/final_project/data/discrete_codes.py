import torch
from torch.utils.data import Dataset

class CodeSequenceDataset(Dataset):
    def __init__(self, sequence, seq_len):
        self.seq_len = seq_len
        self.sequence = torch.tensor(sequence, dtype=torch.long)

    def __len__(self):
        return len(self.sequence) - self.seq_len

    def __getitem__(self, idx):
        x = self.sequence[idx:idx + self.seq_len]
        y = self.sequence[idx + 1:idx + 1 + self.seq_len]
        return x, y
