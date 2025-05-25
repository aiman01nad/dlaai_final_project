import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

from final_project.data.mnist import get_dataloaders
from final_project.utils.helpers import load_config, load_model, set_seed, extract_latents

def build_knn_graph(latents, k):
    """Build sparse k-NN graph from latents."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='manhattan').fit(latents)

    knn_graph = nbrs.kneighbors_graph(mode='distance')  # Sparse matrix (N x N)
    adj = csr_matrix(knn_graph)
    print("k-NN graph built.")
    return adj

def choose_medoids(latents, n_clusters, sample_size=5000):
    """Use K-Medoids on a subset to select representative medoids."""
    subset_indices = np.random.choice(len(latents), size=sample_size, replace=False)
    subset = latents[subset_indices]

    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', init='k-medoids++', random_state=42)
    kmedoids.fit(subset)

    # Map subset medoid indices to full dataset indices
    medoid_indices = subset_indices[kmedoids.medoid_indices_]
    print("Subset-based medoids selected.")
    return medoid_indices

def assign_clusters_geodesically(adj, medoid_indices, batch_size=50):
    """Assign each point to the closest medoid using batched Dijkstra."""
    n_samples = adj.shape[0]
    n_medoids = len(medoid_indices)
    labels = np.zeros(n_samples, dtype=int)
    min_dist = np.full(n_samples, np.inf)

    print(f"Assigning clusters with {n_medoids} medoids in batches...")

    for i in range(0, n_medoids, batch_size):
        batch_indices = medoid_indices[i:i+batch_size]
        print(f"Processing medoid batch {i} to {i + batch_size}")
        dists = dijkstra(csgraph=adj, directed=False, indices=batch_indices)

        for j, _ in enumerate(batch_indices):
            mask = dists[j] < min_dist
            labels[mask] = i + j  # Assign cluster index (0, 1, ..., n_clusters-1)
            min_dist[mask] = dists[j][mask]

    print("Cluster assignment complete.")
    return labels

def build_codebook(latents, medoid_indices):
    """Builds codebook from medoid latent vectors."""
    codebook_latents = latents[medoid_indices] # shape: (n_clusters, latent_dim)
    np.save('src/final_project/outputs/geodesic/codebook_latents.npy', codebook_latents)
    print("Codebook latents saved.")
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
    
    latents, _ = extract_latents(model, train_loader, device)  # shape: (60000, latent_dim)

    # Build sparse kNN graph
    adj = build_knn_graph(latents, k=k_neighbors)

    # Get medoids via K-Medoids clustering (Euclidean)
    medoid_indices = choose_medoids(latents, n_clusters=n_clusters)

    # Run Dijkstra from medoids, assign each point to closest
    labels = assign_clusters_geodesically(adj, medoid_indices)
    labels_reshaped = labels.reshape(60000, 7, 7)
    np.save('outputs/geodesic/kmedoids_code_maps.npy', labels_reshaped)
    #np.save('src/final_project/outputs/geodesic/kmedoids_labels.npy', labels)

    # Save codebook
    build_codebook(latents, medoid_indices)

if __name__ == "__main__":
    main()
