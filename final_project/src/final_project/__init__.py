import argparse
import torch
from final_project.data.mnist import get_dataloaders
from final_project.train.train_vae import train_vae

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_vae(epochs=args.epochs, device=args.device)

