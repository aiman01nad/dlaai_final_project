from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

def visualize_latents_tsne(latents, labels, save_path):
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
    plt.show()

    print(f"Latent space visualization saved to {save_path}")

def visualize_latents_pca(latents, labels, save_path):
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    plt.figure(figsize=(6, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', s=100)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (latents_2d[i, 0], latents_2d[i, 1]))
    plt.title("Discrete Latent Codes (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

    print(f"Latent space with labels visualization saved to {save_path}")

def visualize_decoded_medoids(codebook_latents, model, device, save_path):
    z = torch.tensor(codebook_latents).float().to(device)
    recon = model.decoder(z).cpu().detach()

    plt.figure(figsize=(10, 2))
    for i, img in enumerate(recon):
        plt.subplot(1, 10, i+1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Code {i}")
        plt.axis("off")
    plt.suptitle("Decoded Medoids (Codebook Entries)")
    plt.savefig(save_path)
    plt.show()

    print("Decoded medoids visualization saved to", save_path)
