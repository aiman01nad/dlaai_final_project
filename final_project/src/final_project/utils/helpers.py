from pathlib import Path
import torch

def save_model(model, name):
    root = Path(__file__).resolve().parents[1]  # Go up to project root
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / name)
    print(f"Model saved to {ckpt_dir / name}")