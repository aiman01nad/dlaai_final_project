import time
import tracemalloc
import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix, save_npz
from annoy import AnnoyIndex
from torch import cdist
from final_project.utils.helpers import load_config, set_seed
from final_project.utils.latent_extraction import flatten_latents, reshape_cluster_labels

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

def build_annoy_knn_graph(latents, k, n_trees):
    """ Builds a sparse k-NN graph using Annoy for approximate nearest neighbors using Euclidian distances."""
    N, d = latents.shape
    print(f"Annoy setup: {N:,} vectors of dim {d}, k={k}")

    index = AnnoyIndex(d, 'euclidian')
    for i in range(N):
        index.add_item(i, latents[i])

    print("Building Annoy index...")
    index.build(n_trees)

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
    # Initializing cluster centers (landmarks) using farthest-point sampling
    landmark_indices = [np.random.randint(0, len(latents))]
    for _ in range(n_landmarks - 1):
        dist_to_selected = np.min(cdist(latents[landmark_indices], latents), axis=0)
        next_idx = np.argmax(dist_to_selected)
        landmark_indices.append(next_idx)

    landmark_indices = np.array(landmark_indices)
    landmark_latents = latents[landmark_indices]

    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', init='k-medoids++', random_state=42)
    kmedoids.fit(landmark_latents)
    medoid_indices = landmark_indices[kmedoids.medoid_indices_]
    return medoid_indices

def compute_landmark_distances(adj, medoid_indices):
    """Compute pairwise distances from each landmark using Dijkstra's algorithm."""
    print(f"Computing Dijkstra distances from {len(medoid_indices)} landmarks...")
    dists = dijkstra(csgraph=adj, directed=False, indices=medoid_indices)
    print("✅ Landmark distance matrix computed.")
    return dists

def build_codebook(latents, medoid_indices):
    """Builds a codebook from the medoid latent vectors."""
    codebook_latents = latents[medoid_indices]
    np.save('src/final_project/outputs/geodesic/codebook_latents.npy', codebook_latents)
    print(f"Saved codebook latents: shape {codebook_latents.shape}")

def main():
    set_seed()
    clustering_config = load_config("src/final_project/configs/clustering_config.yaml")

    k = clustering_config["k_neighbors"]
    n_trees = clustering_config["n_trees"]
    n_clusters = clustering_config["n_clusters"]
    n_landmarks = clustering_config["n_landmarks"]

    # Load pre-extracted latents
    latents = np.load('src/final_project/outputs/vae/vae_latents.npy').astype(np.float32)
    print(f"Loaded latents shape: {latents.shape}")
    N, C, H, W = latents.shape
    latents = flatten_latents(latents)
    latents /= np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8 # Normalize for cosine/Euclidean similarity

    # Build k-NN graph
    adj = profile_step("Build KNN graph", build_annoy_knn_graph, latents, k, n_trees)
    save_npz('src/final_project/outputs/geodesic/annoy_knn_graph.npz', adj)

    # Select landmark medoids for scalable geodesic approximation
    medoid_indices = profile_step("Select Landmark Medoids", choose_landmark_medoids, latents, n_clusters, n_landmarks)
    np.save('src/final_project/outputs/geodesic/medoid_indices.npy', medoid_indices)
    print(f"✅ Saved medoid indices: shape {medoid_indices.shape}")

    # Compute landmark distances via Dijkstra
    landmark_dists = profile_step("Compute Landmark Distances (Dijkstra)", compute_landmark_distances, adj, medoid_indices)
    np.save('src/final_project/outputs/geodesic/landmark_dists.npy', landmark_dists)
    print(f"✅ Saved landmark Dijkstra distances: shape {landmark_dists.shape}")

    # Assign each latent to closest medoid (cluster label)
    labels = np.argmin(landmark_dists, axis=0)
    np.save("src/final_project/outputs/geodesic/kmedoids_labels.npy", labels)
    print(f"Saved cluster labels shape: {labels.shape}")

    # Save codes for transformer training
    reshape_cluster_labels(labels, N, H, W)

    # Save codebook latents (medoids)
    profile_step("Build Codebook", build_codebook, latents, medoid_indices)

if __name__ == "__main__":
    main()