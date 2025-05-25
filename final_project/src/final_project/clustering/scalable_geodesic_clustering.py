import time
import tracemalloc
import torch
import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from annoy import AnnoyIndex
from final_project.data.mnist import get_dataloaders
from final_project.utils.helpers import load_config, load_model, set_seed, extract_latents

def profile_step(name, func, *args, **kwargs):
    print(f"\n▶️ Starting: {name}")
    tracemalloc.start()
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    print(f"⏱️ {name} - Time: {end - start:.2f}s | Memory: {peak / 1e6:.2f} MB peak\n")
    tracemalloc.stop()
    return result

def build_annoy_knn_graph(latents, k, metric='euclidean'):
    """ Builds a sparse k-NN graph using Annoy for approximate nearest neighbors using Euclidian distances."""
    N, d = latents.shape
    print(f"Annoy setup: {N:,} vectors of dim {d}, k={k}, metric={metric}")

    if metric not in ['euclidean', 'angular']:
        raise ValueError("Annoy metric must be 'euclidean' or 'angular'")

    index = AnnoyIndex(d, metric)
    for i in range(N):
        index.add_item(i, latents[i])

    print("Building Annoy index...")
    index.build(50)  # 50 trees is a good default

    print("Querying Annoy index for nearest neighbors...")
    rows, cols, vals = [], [], []
    for i in range(N):
        neighbors, distances = index.get_nns_by_item(i, k + 1, include_distances=True)
        # skip the first neighbor because it's the point itself
        for j, dist in zip(neighbors[1:], distances[1:]):
            rows.append(i)
            cols.append(j)
            vals.append(dist)

    index.save('src/final_project/outputs/geodesic/annoy_index.ann')
    adj = csr_matrix((vals, (rows, cols)), shape=(N, N))
    print("✅ Sparse k-NN graph built with Annoy.")
    return adj

def choose_landmark_medoids(latents, n_clusters, n_landmarks):
    """Landmark-based geodesic clustering using K-Medoids."""
    indices = np.random.choice(len(latents), size=n_landmarks, replace=False)
    landmark_latents = latents[indices]

    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', init='k-medoids++', random_state=42)
    kmedoids.fit(landmark_latents)
    medoid_indices = indices[kmedoids.medoid_indices_]
    return medoid_indices

def compute_landmark_distances(adj, medoid_indices):
    """Compute pairwise distances from each landmark using Dijkstra's algorithm."""
    print(f"Computing Dijkstra distances from {len(medoid_indices)} landmarks...")
    dists = dijkstra(csgraph=adj, directed=False, indices=medoid_indices)
    print("✅ Landmark distance matrix computed.")
    return dists

def approximate_geodesic_distance(i, j, landmark_dists):
    """Approximate geodesic distance between two points using landmark distances."""
    d_i = landmark_dists[:, i]
    d_j = landmark_dists[:, j]
    return np.min(d_i + d_j)

def sparse_geodesic_sampling(landmark_dists, sample_size=10000):
    """Sample approximate geodesic distances between random pairs of points."""
    N = landmark_dists.shape[1]
    pairs = np.random.randint(0, N, size=(sample_size, 2))
    approx_dists = np.zeros(sample_size, dtype=np.float32)
    print(f"Sampling {sample_size} approximate geodesic distances...")
    for idx, (i, j) in enumerate(tqdm(pairs, desc="Sampling geodesic distances")):
        approx_dists[idx] = approximate_geodesic_distance(i, j, landmark_dists)
    print("✅ Sampling done.")
    return pairs, approx_dists

def build_codebook(latents, medoid_indices):
    """Builds a codebook from the medoid latent vectors."""
    codebook_latents = latents[medoid_indices]
    np.save('src/final_project/outputs/geodesic/codebook_latents.npy', codebook_latents)
    print(f"Saved codebook latents: shape {codebook_latents.shape}")
    return codebook_latents

def main():
    set_seed()
    clustering_config = load_config("src/final_project/configs/clustering_config.yaml")

    batch_size = clustering_config["batch_size"]
    k = clustering_config["k_neighbors"]
    n_clusters = clustering_config["n_clusters"]
    n_landmarks = clustering_config["n_landmarks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model('vae', "src/final_project/checkpoints/vae.pth", device)
    train_loader, _ = get_dataloaders(batch_size=batch_size)

    # Extract & save latents
    latents, _ = profile_step("Extract Latents", extract_latents, model, train_loader, device)
    latents = latents.astype(np.float32)
    # Flatten 4D latents [N, C, H, W] to [N, D]
    N, C, H, W = latents.shape
    latents = latents.reshape(N, -1)  # Shape: [N, C*H*W]

    # Normalize for cosine/Euclidean similarity (if needed)
    latents /= np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8
    np.save('src/final_project/outputs/vae/vae_latents.npy', latents)
    print(f"✅ Saved full latent vectors: shape {latents.shape}")

    # Build k-NN graph
    adj = profile_step("Build KNN graph", build_annoy_knn_graph, latents, k)
    save_npz('src/final_project/outputs/geodesic/annoy_knn_graph.npz', adj)

    # Landmark medoids
    medoid_indices = profile_step("Select Landmark Medoids", choose_landmark_medoids, latents, n_clusters, n_landmarks)
    np.save('src/final_project/outputs/geodesic/medoid_indices.npy', medoid_indices)
    print(f"✅ Saved medoid indices: shape {medoid_indices.shape}")

    # Dijkstra distances
    landmark_dists = profile_step("Compute Landmark Distances (Dijkstra)", compute_landmark_distances, adj, medoid_indices)
    np.save('src/final_project/outputs/geodesic/landmark_dists.npy', landmark_dists)
    print(f"✅ Saved landmark Dijkstra distances: shape {landmark_dists.shape}")

    # Sample geodesic distances
    pairs, approx_dists = profile_step("Sparse Geodesic Distance Sampling", sparse_geodesic_sampling, landmark_dists, sample_size=10000)
    np.save('src/final_project/outputs/geodesic/approx_geodesic_pairs.npy', pairs)
    np.save('src/final_project/outputs/geodesic/approx_geodesic_dists.npy', approx_dists)

    # Save codebook
    _ = profile_step("Build Codebook", build_codebook, latents, medoid_indices)

    # Save clustering config for reproducibility (optional)
    import yaml
    with open("src/final_project/outputs/geodesic/clustering_config_used.yaml", "w") as f:
        yaml.dump(clustering_config, f)
    print("✅ Saved config used for clustering.")

if __name__ == "__main__":
    main()