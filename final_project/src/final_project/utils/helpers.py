from pathlib import Path
import random
import numpy as np
import torch
import yaml

from final_project.models.vae import VAE

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
