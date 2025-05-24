import torch
import torch.nn.functional as F
from final_project.models import VAE
from final_project.data.mnist import get_dataloaders
from final_project.utils import save_model, load_config, set_seed

def train_vae(model: VAE, epochs, lr, weight_decay, batch_size, device, beta, save_name):
    train_loader, val_loader = get_dataloaders(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon, kl = elbo_loss(recon_batch, data, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
        
        print(f'====> Epoch: {epoch} Average loss: {total_loss / len(train_loader.dataset):.4f}')

        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss, recon, kl = elbo_loss(recon_batch, data, mu, logvar, beta)
                val_loss += loss.item()
                val_recon_loss += recon.item()
                val_kl_loss += kl.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_recon = val_recon_loss / len(val_loader.dataset)
        avg_val_kl = val_kl_loss / len(val_loader.dataset)

        print(f'====> Validation Loss: {avg_val_loss:.4f} | Recon: {avg_val_recon:.4f} | KL: {avg_val_kl:.4f}')

    save_model(model, save_name)
    return model

def elbo_loss(recon_x, x, mu, logvar, beta):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def main():
    set_seed()
    vae_config = load_config("src/final_project/configs/vae_config.yaml")

    hidden_dim = vae_config["model"]["hidden_dim"]
    embedding_dim = vae_config["model"]["embedding_dim"]

    epochs = vae_config["training"]["epochs"]
    lr = float(vae_config["training"]["learning_rate"])
    weight_decay = float(vae_config["training"]["weight_decay"])
    batch_size = vae_config["training"]["batch_size"]
    beta = vae_config["training"]["beta"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_name = "vae.pth"

    model = VAE(hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    model = model.to(device)

    model = train_vae(model, epochs, lr, weight_decay, batch_size, device, beta, save_name)

if __name__ == "__main__":
    main()