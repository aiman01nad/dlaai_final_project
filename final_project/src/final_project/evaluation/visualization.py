import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import umap.umap_ as umap

from final_project.models.vae_module import VAELightningModule
from final_project.utils.latent_extraction import flatten_latents

def save_code_histogram(codes, filename, num_embeddings=128, title="Histogram of Generated Codes"):
    plt.figure(figsize=(10, 4))
    plt.hist(codes, bins=range(num_embeddings+1))
    plt.title(title)
    plt.xlabel("Code Index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_generated_images(images, filename, nrow=8, title="Transformer Generated Samples"):
    grid = make_grid(images[:nrow**2], nrow=nrow, pad_value=1.0)
    plt.figure(figsize=(nrow, nrow))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def save_reconstruction_grid(x, x_hat, filename):
    x_grid = make_grid(x[:8], nrow=8)
    xhat_grid = make_grid(x_hat[:8], nrow=8)
    grid = torch.cat([x_grid, xhat_grid], dim=1)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title("Original (Top) vs. Reconstruction (Bottom)")
    plt.savefig(filename)
    plt.close()

def visualize_latents(latents, labels, save_path, method='tsne', title=None, annotate=False, n_classes=10):
    """Visualize high-dimensional latents in 2D using PCA or t-SNE"""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        method_title = "t-SNE"
    elif method == 'pca':
        reducer = PCA(n_components=2)
        method_title = "PCA"
    else:
        raise ValueError("method must be 'tsne', 'pca', or 'umap'")

    latents_2d = reducer.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", s=10)
    if annotate:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (latents_2d[i, 0], latents_2d[i, 1]))
    if n_classes is not None:
        plt.colorbar(scatter, ticks=range(n_classes), label="Class/Cluster")
    plt.title(title or f"Latent Space ({method_title})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"Latent space visualization saved to {save_path}")

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

def plot_code_histogram(code_map, save_path=None):
    flat = code_map.flatten()
    counts = np.bincount(flat, minlength=np.max(flat)+1)
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(counts)), counts)
    plt.title('Code Usage')
    plt.xlabel('Code index')
    plt.ylabel('Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    output_dir = "src/final_project/outputs/visualization"

    vae_latents = np.load("src/final_project/outputs/vae/vae_latents.npy") # (N, D, H, W)
    vae_latents_pooled = vae_latents.mean(axis=(2, 3))  # → (N, D)
    vae_labels = np.load("src/final_project/outputs/vae/vae_labels.npy") # (N, )

    vqvae_codes = np.load("src/final_project/outputs/vqvae/vqvae_codes.npy") # (N, H, W)
    vqvae_codes_flat = flatten_latents(vqvae_codes) # (N, H*W)
    #vqvae_codes_pooled = vqvae_codes.mean(axis=(1, 2))  # shape: (N, D)
    vqvae_labels = np.load("src/final_project/outputs/vqvae/vqvae_labels.npy") # (N, )

    # Visualize latents
    #for method in ['pca', 'tsne']:
    #    visualize_latents(vae_latents_pooled, vae_labels, f"{output_dir}/vae_latents_{method}.png", method=method, title=f"VAE Latents ({method})")
    #    visualize_latents(vqvae_codes_flat, vqvae_labels, f"{output_dir}/vqvae_latents_{method}.png", method=method, title=f"VQ-VAE Latents ({method})")

    # Code usage histogram
    plot_code_histogram(vqvae_codes, f"{output_dir}/vqvae_code_usage.png")

    vae_geodesic_codes = np.load("src/final_project/outputs/geodesic/geodesic_codes.npy") # (N, H, W)
    #vae_geodesic_codes_flat = flatten_latents(vae_geodesic_codes) # (N, H*W)
    #vae_geodesic_labels = np.load("src/final_project/outputs/geodesic/kmedoids_labels.npy") # (N*H*W, )
    codebook_latents = np.load("src/final_project/outputs/geodesic/codebook_latents.npy") # (num_codes, embedding_dim)

    # Codebook structure (t-SNE)
    embedded = TSNE(n_components=2).fit_transform(codebook_latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedded[:,0], embedded[:,1], c=np.arange(len(codebook_latents)), cmap="tab20")
    plt.title("Geodesic Codebook Structure (t-SNE)")
    plt.colorbar(label="Code index")
    plt.savefig(f"{output_dir}/geodesic_codebook_tsne.png")
    plt.close()
    print(f"Saved codebook t-SNE visualization to {output_dir}/geodesic_codebook_tsne.png")


if __name__ == '__main__':
    main()