from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import umap
import seaborn as sns

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

def plot_latent_clusters(latents, labels, assignments, save_path):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    latents_2d = reducer.fit_transform(latents)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=labels, s=5, palette="tab10", legend=False)
    plt.title("Ground Truth Labels")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=assignments, s=5, palette="tab10", legend=False)
    plt.title("Geodesic Clusters")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

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
