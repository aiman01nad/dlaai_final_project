from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch

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

def visualize_codebook(codebook_latents, model, device):
    z = torch.tensor(codebook_latents).float().to(device)
    recon = model.decoder(z).cpu().detach()

    plt.figure(figsize=(10, 2))
    for i, img in enumerate(recon):
        plt.subplot(1, 10, i+1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Code {i}")
        plt.axis("off")
    plt.suptitle("Decoded Medoids (Codebook Entries)")
    plt.show()

    print("Codebook visualized.")