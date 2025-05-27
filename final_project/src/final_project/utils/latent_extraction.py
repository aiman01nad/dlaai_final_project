import torch
import numpy as np
import argparse
from final_project.models.vae_module import VAELightningModule
from final_project.models.vqvae_module import VQVAELightningModule
from final_project.utils.helpers import load_config, set_seed
from final_project.data.mnist import get_dataloaders

def extract_vae_latents(model, dataloader, device):
    model = model.to(device)
    model.eval()

    latents, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, _ = model.encoder(x) # Shape: [B, C=embedding_dim, H, W]
            latents.append(mu.cpu())
            labels.append(y)

    latents = torch.cat(latents, dim=0).numpy() # Shape: [N, C=embedding_dim, H, W]
    labels = torch.cat(labels, dim=0).numpy()
    return latents, labels

def extract_vqvae_codes(model, dataloader, device):
    model = model.to(device)
    model.eval()

    codes, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, _, indices = model(x)  # Shape: [B, H, W] where indices are the discrete codes
            codes.append(indices.cpu())
            labels.append(y)

    codes = torch.cat(codes, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return codes, labels

def flatten_latents(latents):
    # Flatten latent vectors if they have spatial dims [N, C, H, W]
    if latents.ndim == 4:
        N, C, H, W = latents.shape
        # Flatten spatial dims and channels into a single vector per example
        latents = latents.reshape(N, -1)
        print(f"Flattened latents shape: {latents.shape}")
    
def main():
    parser = argparse.ArgumentParser(description="Extract latents or discrete codes from VAE or VQVAE")
    parser.add_argument("--model_type", type=str, choices=["vae", "vqvae"], required=True, help="Model type to extract from")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model_type == "vae":
        module = VAELightningModule.load_from_checkpoint("src/final_project\checkpoints/vae/vae-epoch=18-val_loss=14163.7070.ckpt")
        model = module.model
        vae_config = load_config("src/final_project/configs/vae_config.yaml")
        batch_size = vae_config["training"]["batch_size"]
        train_loader, _, _ = get_dataloaders(batch_size=batch_size)

        latents, labels = extract_vae_latents(model, train_loader, device)
        print(f"Extracted latents shape: {latents.shape}")

        np.save('src/final_project/outputs/vae/vae_latents.npy', latents)
        np.save('src/final_project/outputs/vae/vae_labels.npy', labels)
        print("Saved VAE latents and labels.")

    elif args.model_type == "vqvae":
        module = VQVAELightningModule.load_from_checkpoint("src/final_project\checkpoints/vqvae/vqvae-epoch=29-val_loss=0.0132.ckpt")
        model = module.model
        vqvae_config = load_config("src/final_project/configs/vqvae_config.yaml")
        batch_size = vqvae_config["training"]["batch_size"]
        train_loader, _, _ = get_dataloaders(batch_size=batch_size)

        codes, labels = extract_vqvae_codes(model, train_loader, device)
        print(f"Extracted codes shape: {codes.shape}")

        np.save('src/final_project/outputs/vqvae/vqvae_codes.npy', codes)
        np.save('src/final_project/outputs/vqvae/vqvae_labels.npy', labels)
        print("Saved VQ-VAE codes and labels.")

if __name__ == "__main__":
    main()
