import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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

def get_dataloaders(codes, seq_len, batch_size):
    train_codes, val_codes, test_codes = np.split(codes, [50000, 55000]) # 50k for training, 5k for validation, 5k for testing

    train_dataset = CodeSequenceDataset(train_codes, seq_len)
    val_dataset = CodeSequenceDataset(val_codes, seq_len)
    test_dataset = CodeSequenceDataset(test_codes, seq_len)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )