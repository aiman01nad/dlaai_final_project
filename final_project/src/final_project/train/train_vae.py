from pathlib import Path
import torch
import torch.nn.functional as F

from final_project.models.vae import VQVAE
from final_project.data.mnist import get_dataloaders

def train_vae(epochs, lr=1e-3, batch_size=128, device='cpu'):
    train_loader, _ = get_dataloaders(batch_size)
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, vq_loss = model(data)
            recon_loss = F.mse_loss(recon_batch, data)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

        print(f'====> Epoch: {epoch} Average loss: {total_loss / len(train_loader):.4f}')

    save_model(model)
    return model

def save_model(model, name="vae.pth"):
    root = Path(__file__).resolve().parents[1]  # Go up to project root
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / name)
    print(f"Model saved to {ckpt_dir / name}")
