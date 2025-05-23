import torch
import torch.nn.functional as F
from final_project.models import VQVAE
from final_project.data import get_dataloaders
from final_project.utils import save_model
from final_project.utils.helpers import load_config, set_seed

def train_vqvae(model: VQVAE, epochs, lr, batch_size, device, save_name):
    train_loader, _ = get_dataloaders(batch_size)
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

    save_model(model, save_name)
    return model

def main():
    set_seed()
    vqvae_config = load_config("src/final_project/configs/vqvae_config.yaml")

    hidden_dim = vqvae_config["model"]["hidden_dim"]
    embedding_dim = vqvae_config["model"]["embedding_dim"]
    num_embeddings = vqvae_config["model"]["num_embeddings"]
    commitment_cost = vqvae_config["model"]["commitment_cost"]

    epochs = vqvae_config["training"]["epochs"]
    lr = vqvae_config["training"]["learning_rate"]
    batch_size = vqvae_config["training"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_name = "vqvae.pth"

    model = VQVAE(hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_embeddings=num_embeddings, commitment_cost=commitment_cost)
    model = model.to(device)

    model = train_vqvae(model, epochs, lr, batch_size, device, save_name)

if __name__ == "__main__":
    main()