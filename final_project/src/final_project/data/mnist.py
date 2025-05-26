from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data_root = Path(__file__).resolve().parent

    full_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    train_set, val_set = random_split(full_train, [50000, 10000])  # 50k for training, 10k for validation

    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform) # 10k for testing

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )
