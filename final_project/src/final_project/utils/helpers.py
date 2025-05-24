from pathlib import Path
import random
import numpy as np
import torch
import yaml

from final_project.models.vae import VAE
from final_project.models.vqvae import VQVAE

def save_model(model, name):
    root = Path(__file__).resolve().parents[1]  # Go up to project root
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / name)
    print(f"Model saved to {ckpt_dir / name}")

def load_model(model_type, model_path, device):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    if model_type == 'vae':
        vae_config = load_config("src/final_project/configs/vae_config.yaml")
        hidden_dim = vae_config["model"]["hidden_dim"]
        embedding_dim = vae_config["model"]["embedding_dim"]
        model = VAE(hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    print(f"Model loaded from {model_path}")
    return model

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def extract_latents(model, dataloader, device):
    model = model.to(device)
    model.eval()
    latents, labels = [], []

    if isinstance(model, VAE):
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                mu, _ = model.encoder(x) # (batch, embedding_dim, 7, 7)
                latents.append(mu.cpu().view(x.size(0), -1))  # flatten to (batch, embedding_dim*7*7)
                labels.append(y)
    
    elif isinstance(model, VQVAE):
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                z = model.encoder(x) # encoder output (continous latent)
                latents.append(z.cpu().view(x.size(0), -1))  # flatten spatial dims
                labels.append(y)
    
    latents = torch.cat(latents).numpy()
    labels = torch.cat(labels).numpy()
    
    if isinstance(model, VAE):
        np.save('src/final_project/outputs/vae/vae_latents.npy', latents)
        np.save('src/final_project/outputs/vae/vae_labels.npy', labels)
    
    elif isinstance(model, VQVAE):
        np.save('src/final_project/outputs/vqvae/vqvae_latents.npy', latents)
        np.save('src/final_project/outputs/vqvae/vqvae_labels.npy', labels)

    print(f"Latents and labels saved to {Path('src/final_project/outputs')}")
    print(f"Latents shape: {latents.shape}, Labels shape: {labels.shape}")

    return latents, labels

def load_latents_and_labels(model_type):
    if model_type == 'vae':
        latents = np.load('src/final_project/outputs/vae_latents.npy')
        labels = np.load('src/final_project/outputs/vae_labels.npy')
    elif model_type == 'vqvae':
        latents = np.load('src/final_project/outputs/vqvae_latents.npy')
        labels = np.load('src/final_project/outputs/vqvae_labels.npy')
    else:
        raise ValueError("Invalid model type. Choose 'vae' or 'vqvae'.")
    
    return latents, labels