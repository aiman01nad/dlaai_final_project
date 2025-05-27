from pathlib import Path
import random
import numpy as np
import torch
import yaml

from final_project.models import VAE, VQVAE, Transformer

def save_model(model, name):
    root = Path(__file__).resolve().parents[1]  # Go up to project root
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / name)
    print(f"Model saved to {ckpt_dir / name}")

def load_model(model_type, model_path, device):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    
    if model_type not in ['vae', 'vqvae', 'transformer']:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from 'vae', 'vqvae', or 'transformer'.")

    if model_type == 'vae':
        vae_config = load_config("src/final_project/configs/vae_config.yaml")
        hidden_dim = vae_config["model"]["hidden_dim"]
        embedding_dim = vae_config["model"]["embedding_dim"]
        model = VAE(hidden_dim=hidden_dim, embedding_dim=embedding_dim)

    elif model_type == 'vqvae':
        vqvae_config = load_config("src/final_project/configs/vqvae_config.yaml")
        hidden_dim = vqvae_config["model"]["hidden_dim"]
        embedding_dim = vqvae_config["model"]["embedding_dim"]
        num_embeddings = vqvae_config["model"]["num_embeddings"]
        commitment_cost = vqvae_config["model"]["commitment_cost"]
        model = VQVAE(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost
        )

    elif model_type == 'transformer':
        transformer_config = load_config("src/final_project/configs/transformer_config.yaml")
        num_embeddings = transformer_config["model"]["num_embeddings"]
        seq_len = transformer_config["model"]["seq_len"]
        embedding_dim = transformer_config["model"]["embedding_dim"]
        nheads = transformer_config["model"]["nheads"]
        num_layers = transformer_config["model"]["num_layers"]
        feedforward_dim = transformer_config["model"]["feedforward_dim"]
        dropout = transformer_config["model"]["dropout"]

        model = Transformer(
            num_embeddings=num_embeddings,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            nheads=nheads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout
        )
        
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

    if isinstance(model, VAE):
        latents, labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                mu, _ = model.encoder(x)  # shape: [B, C, H, W]
                latents.append(mu.cpu())  # keep full shape
                labels.append(y)

        latents = torch.cat(latents, dim=0)  # shape: [N, C, H, W]
        labels = torch.cat(labels, dim=0)

        np.save('src/final_project/outputs/vae/vae_latents.npy', latents.numpy())
        np.save('src/final_project/outputs/vae/vae_labels.npy', labels.numpy())
        print("Latents and labels saved for VAE.")
        return latents.numpy(), labels.numpy()

    elif isinstance(model, VQVAE):
        codes = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                _, _, indices = model(x)  # shape: [B, H, W]
                codes.append(indices.cpu())

        codes = torch.cat(codes, dim=0)
        np.save('src/final_project/outputs/vqvae/vqvae_codes.npy', codes.numpy())
        print("Codes saved for VQVAE.")
        return codes.numpy()