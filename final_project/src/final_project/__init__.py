import argparse
import random
import numpy as np
import torch
from final_project.data.mnist import get_dataloaders
from final_project.models.vqvae import VQVAE
from final_project.train.train_vqvae import train_vqvae
from final_project.train.eval_vqvae import evaluate_model
from final_project.utils.visualize_latents import visualize_latents

# Hyperparameters
EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VQCVAE model parameters
hidden_dim = 128
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    model = VQVAE(hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_embeddings=num_embeddings, commitment_cost=commitment_cost)
    #train_vqvae(model, epochs=args.epochs, lr=args.learning_rate, batch_size=args.batch_size, device=args.device)
    #evaluate_model(model, checkpoint_path="src/final_project/checkpoints/vqvae.pth", output_dir="src/final_project/outputs/vqvae_eval", device=args.device)
    visualize_latents(model, model_path="src/final_project/checkpoints/vqvae.pth", output_dir="src/final_project/outputs/vqvae_eval", device=args.device, method='tsne')

