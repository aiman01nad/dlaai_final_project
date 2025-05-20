from pathlib import Path
import torch

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
        model = VAE(hidden_dim=64, embedding_dim=8)
        model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    return model