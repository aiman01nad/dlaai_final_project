import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ImageCodeSequenceDataset(Dataset):
    def __init__(self, code_maps):  # shape: (N, 49)
        self.sequences = torch.tensor(code_maps, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx][:-1]  # shape: (48,)
        y = self.sequences[idx][1:]   # shape: (48,)
        return x, y

def get_dataloaders(code_map, batch_size):
    train_codes, val_codes, test_codes = np.split(code_map, [40000, 45000]) # 40k train, 5k val, 5k test

    train_dataset = ImageCodeSequenceDataset(train_codes)
    val_dataset = ImageCodeSequenceDataset(val_codes)
    test_dataset = ImageCodeSequenceDataset(test_codes)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )