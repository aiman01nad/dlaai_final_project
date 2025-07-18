from pathlib import Path
import random
import numpy as np
import torch
import yaml

def save_model(model, name):
    root = Path(__file__).resolve().parents[1]  # Go up to project root
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / name)
    print(f"Model saved to {ckpt_dir / name}")

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