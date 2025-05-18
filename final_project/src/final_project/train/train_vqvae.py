import torch
import torch.nn.functional as F
from final_project.models.vqvae import VQVAE
from final_project.data.mnist import get_dataloaders
from final_project.utils.helpers import save_model

def train_vqvae(model, epochs, lr, batch_size, device):
    train_loader, _ = get_dataloaders(batch_size)
    model = model.to(device)
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

    save_model(model, "vqvae.pth")
    return model
