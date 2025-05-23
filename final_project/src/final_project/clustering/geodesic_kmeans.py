from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

from final_project.data.mnist import get_dataloaders
from final_project.utils.helpers import load_config, load_model, set_seed, extract_latents

def compute_geodesic_matrix(latents, k):
    """Efficient kNN graph + geodesic distance computation using sparse matrices."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(latents)
    knn_graph = nbrs.kneighbors_graph(mode='distance')  # Sparse matrix (N x N)

    # Ensure it's in CSR format
    adj = csr_matrix(knn_graph)

    # Compute geodesic (shortest-path) distances
    geodesic_matrix = shortest_path(adj, directed=False)
    print("Geodesic matrix computed.")

    return geodesic_matrix

def cluster_with_kmedoids(distance_matrix, n_clusters):
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric='precomputed',
        init='k-medoids++',
        random_state=42
    )
    kmedoids.fit(distance_matrix)

    np.save('src/final_project/outputs/geodesic/kmedoids_labels.npy', kmedoids.labels_)
    print("Clustering done.")

    return kmedoids.labels_, kmedoids.medoid_indices_

def build_codebook(latents, medoid_indices):
    """Builds a codebook from the latents based on cluster labels and medoid indices."""
    codebook_latents = latents[medoid_indices] # Shape: (n_clusters, latent_dim)

    np.save('src/final_project/outputs/geodesic/codebook_latents.npy', codebook_latents)
    print("Codebook latents built saved.")

    return codebook_latents

def main():
    set_seed()
    vae_config = load_config("src/final_project/configs/vae_config.yaml")
    batch_size = vae_config["training"]["batch_size"]
    clustering_config = load_config("src/final_project/configs/clustering_config.yaml")
    k_neighbors = clustering_config["k_neighbors"]
    n_clusters = clustering_config["n_clusters"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_path = "src/final_project/checkpoints/vae.pth"
    model = load_model('vae', model_path, device)

    train_loader, _ = get_dataloaders(batch_size=batch_size)

    latents, _ = extract_latents(model, train_loader, device)
    latents = latents[:60000]  # Test on a subset of latents
    geodesic_matrix = compute_geodesic_matrix(latents, k=k_neighbors)
    
    _, medoid_indices = cluster_with_kmedoids(geodesic_matrix, n_clusters=n_clusters)

    build_codebook(latents, medoid_indices)

if __name__ == "__main__":
    main()