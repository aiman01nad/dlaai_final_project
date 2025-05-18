import argparse
import torch
from final_project.data.mnist import get_dataloaders
from final_project.train.train_vae import train_vae
from final_project.train.eval_vae import evaluate_model
from final_project.utils.visualize_latents import visualize_latents

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    #train_vae(epochs=args.epochs, device=args.device)
    #evaluate_model(checkpoint_path="src/final_project/checkpoints/vae.pth", output_dir="src/final_project/outputs/vae_eval", device=args.device)
    visualize_latents(model_path="src/final_project/checkpoints/vae.pth", output_dir="src/final_project/outputs/vae_eval", device=args.device, method='tsne')

