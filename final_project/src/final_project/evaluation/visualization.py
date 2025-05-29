from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import umap.umap_ as umap
import seaborn as sns

from final_project.models.vae_module import VAELightningModule
from final_project.utils.latent_extraction import flatten_latents

def visualize_latents(latents, labels, save_path, method='tsne', title=None, annotate=False, n_classes=10):
    """
    Visualize high-dimensional latents in 2D using PCA, t-SNE, or UMAP.
    Args:
        latents: [N, D] array
        labels: [N] array (class or cluster labels)
        save_path: path to save the plot
        method: 'tsne', 'pca', or 'umap'
        title: plot title (optional)
        annotate: if True, annotate each point with its label (for small N)
        n_classes: number of classes for colorbar
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        method_title = "t-SNE"
    elif method == 'pca':
        reducer = PCA(n_components=2)
        method_title = "PCA"
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
        method_title = "UMAP"
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

def visualize_codebook(model, codebook_latents, C=8, H=7, W=7, n_cols=10):
    device = next(model.parameters()).device
    # Reshape flattened latents to (N, C, H, W)
    z = torch.tensor(codebook_latents, dtype=torch.float32, device=device).view(-1, C, H, W)
    model.eval()
    with torch.no_grad():
        decoded_imgs = model.decoder(z).cpu().numpy()  # (N, 1, 28, 28) or similar

    n = decoded_imgs.shape[0]
    n_rows = (n + n_cols - 1) // n_cols
    plt.figure(figsize=(1.5 * n_cols, 1.5 * n_rows))
    for i in range(n):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(decoded_imgs[i, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle("Decoded Cluster Medoids")
    plt.tight_layout()
    plt.show()

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

def plot_per_image_code_maps(vqvae_codes, geo_codes, num_images=5, save_path=None):
    for i in range(num_images):
        plt.imshow(vqvae_codes[i], cmap="tab20")
        plt.title(f"VQ-VAE Codes (Image {i})")
        plt.colorbar()
        plt.show()

        plt.imshow(geo_codes[i], cmap="tab20")
        plt.title(f"Geodesic Codes (Image {i})")
        plt.colorbar()
        plt.show()

def main():
    vae_latents = np.load("src/final_project/outputs/vae/vae_latents.npy") # (N, D, H, W)
    vae_latents_pooled = vae_latents.mean(axis=(2, 3))  # → (N, D)
    vae_labels = np.load("src/final_project/outputs/vae/vae_labels.npy") # (N, )

    #visualize_latents(vae_latents_pooled, vae_labels, "src/final_project/outputs/vae/vae_latents_pca.png", 'pca', "VAE Latent Space (PCA)")
    #visualize_latents(vae_latents_pooled, vae_labels, "src/final_project/outputs/vae/vae_latents_tsne.png", 'tsne', "VAE Latent Space (t-SNE)")
    #visualize_latents(vae_latents_pooled, vae_labels, "src/final_project/outputs/vae/vae_latents_umap.png", 'umap', "VAE Latent Space (UMAP)")

    vqvae_codes = np.load("src/final_project/outputs/vqvae/vqvae_codes.npy") # (N, H, W)
    vqvae_codes_flat = flatten_latents(vqvae_codes) # (N, H*W)
    vqvae_codes_pooled = vqvae_codes.mean(axis=(1, 2))  # shape: (N, D)
    vqvae_labels = np.load("src/final_project/outputs/vqvae/vqvae_labels.npy") # (N, )

    #visualize_latents(vqvae_codes_flat, vqvae_labels, "src/final_project/outputs/vqvae/vqvae_latents_flat_pca.png", 'pca', "VQVAE Latent Space (PCA)")
    #visualize_latents(vqvae_codes_flat, vqvae_labels, "src/final_project/outputs/vqvae/vqvae_latents_flat_tsne.png", 'tsne', "VQVAE Latent Space (t-SNE)")
    #visualize_latents(vqvae_codes_flat, vqvae_labels, "src/final_project/outputs/vqvae/vqvae_latents_flat_umap.png", 'umap', "VQVAE Latent Space (UMAP)")

    #visualize_latents(vqvae_codes_pooled, vqvae_labels, "src/final_project/outputs/vqvae/vqvae_latents_pooled_pca.png", 'pca', "VQVAE Latent Space (PCA)")
    #visualize_latents(vqvae_codes_pooled, vqvae_labels, "src/final_project/outputs/vqvae/vqvae_latents_pooled_tsne.png", 'tsne', "VQVAE Latent Space (t-SNE)")
    #visualize_latents(vqvae_codes_pooled, vqvae_labels, "src/final_project/outputs/vqvae/vqvae_latents_pooled_umap.png", 'umap', "VQVAE Latent Space (UMAP)")

    vae_geodesic_codes = np.load("src/final_project/outputs/geodesic/geodesic_codes.npy") # (N, H, W)
    vae_geodesic_codes_flat = flatten_latents(vae_geodesic_codes) # (N, H*W)
    vae_geodesic_labels = np.load("src/final_project/outputs/geodesic/kmedoids_labels.npy") # (N*H*W, )
    codebook_latents = np.load("src/final_project/outputs/geodesic/codebook_latents.npy") # (num_codes, embedding_dim)
    
    vae_module = VAELightningModule.load_from_checkpoint("src/final_project/checkpoints/vae/vae-epoch=18-val_loss=14163.7070.ckpt")
    vae = vae_module.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Code usage histogram
    #plot_code_histogram(vqvae_codes, "src/final_project/outputs/vqvae/vqvae_code_usage.png")
    #plot_code_histogram(vae_geodesic_codes, "src/final_project/outputs/geodesic/geodesic_code_usage.png")

    # Per-image code maps
    #plot_per_image_code_maps(vqvae_codes, vae_geodesic_codes, save_path="src/final_project/outputs/per_image_code_maps.png")

    # Codebook visualization
    #tiled = np.tile(codebook_latents[:, :, None, None], (1, 1, 7, 7))
    #visualize_codebook(vae, tiled)

    # Codebook structure
    embedded = TSNE(n_components=2).fit_transform(codebook_latents)
    plt.scatter(embedded[:,0], embedded[:,1], c=np.arange(len(codebook_latents)), cmap="tab20")
    plt.title("Geodesic Codebook Structure (t-SNE)")
    plt.colorbar(label="Code index")
    plt.show()


if __name__ == '__main__':
    main()