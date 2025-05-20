import random
import numpy as np
import torch
import torch.nn.functional as F
from final_project.models import VAE
from final_project.data import get_dataloaders
from final_project.utils import save_model

def train_vae(hidden_dim, embedding_dim, epochs, lr, weight_decay, batch_size, device, beta):
    train_loader, _ = get_dataloaders(batch_size)
    model = VAE(hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

if __name__ == "__main__":
    import argparse
    from final_project.models.vae import VAE

    # Model parameters
    hidden_dim = 64
    embedding_dim = 2

    # Training parameters
    epochs = 40
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 128
    beta = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "vae.pth"

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(0)

    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--beta", type=float, default=beta)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--save_path", type=str, default=save_path)

    args = parser.parse_args()

    model = VAE(hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    train_vae(hidden_dim, embedding_dim, args.epochs, args.lr, args.batch_size, args.device, args.beta)
    print(f"Model saved to {args.save_path}")