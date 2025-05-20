from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
from final_project.data.mnist import get_dataloaders
from final_project.models.vae import VAE
from final_project.utils.helpers import load_model

def extract_latents_vae(model: VAE, dataloader, device):
    model.eval()
    latents, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, _ = model.encoder(x) # Using the mean of the posterior
            latents.append(mu.cpu())
            labels.append(y)

    latents = torch.cat(latents).numpy() # Shape: (N, latent_dim)
    labels = torch.cat(labels).numpy()
    return latents, labels

def visualize_latents_vae(latents, labels, save_path='src/final_project/outputs/vae_latent_plot.png'):
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.title("VAE Latent Space (2D t-SNE)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"Latent space visualization saved to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "src/final_project/checkpoints/vae3.pth"  # Path to the trained VAE model
    model = load_model('vae', model_path, device)  # Load the trained VAE model
    train_loader, test_loader = get_dataloaders(batch_size=128)
    
    latents, labels = extract_latents_vae(model, train_loader, device)
    visualize_latents_vae(latents, labels)