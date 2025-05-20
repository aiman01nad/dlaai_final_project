import torch
import torch.nn.functional as F
from final_project.models.vae import VAE
from final_project.data.mnist import get_dataloaders
from final_project.utils.helpers import save_model

def train_vae(model: VAE, epochs, lr, batch_size, device, beta=1.0):
    train_loader, _ = get_dataloaders(batch_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            recon_loss = F.binary_cross_entropy(recon_batch.view(-1, 28*28), data.view(-1, 28*28), reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta*kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
        
        print(f'====> Epoch: {epoch} Average loss: {total_loss / len(train_loader.dataset):.4f}')

    save_model(model, f"vae.pth")
    return model