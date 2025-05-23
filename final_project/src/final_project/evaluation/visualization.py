from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch

from final_project.data.mnist import get_dataloaders
from final_project.utils.helpers import extract_latents, load_model

def visualize_latents(latents, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.title("Latent Space (2D t-SNE)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True)
    plt.savefig(save_path)

    print(f"Latent space visualization saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "src/final_project/checkpoints/vae.pth"
    model = load_model('vae', model_path=model_path, device=device)  # Load the trained VAE model
    train_loader, _ = get_dataloaders(batch_size=128)  # Load the MNIST dataset

    save_path = "src/final_project/outputs/vae_latent_plot.png"
    latents, labels = extract_latents(model, train_loader, device)
    visualize_latents(latents, labels, save_path)

if __name__ == "__main__":
    main()