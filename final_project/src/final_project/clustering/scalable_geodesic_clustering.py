import time
import tracemalloc
import torch
import numpy as np
import faiss
from sklearn_extra.cluster import KMedoids
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from tqdm import tqdm
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

from tqdm import tqdm
import faiss
import numpy as np
from scipy.sparse import csr_matrix

def build_faiss_knn_graph(latents, k):
    """
    Build a k-NN graph using FAISS.
    Automatically uses GPU if available and shows progress.
    """
    latents = latents.astype(np.float32)
    N, d = latents.shape

    # Set up FAISS index
    try:
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        index = gpu_index
        print("Using FAISS GPU index.")
    except Exception as e:
        print("GPU FAISS not available, falling back to CPU. Reason:", str(e))
        index = faiss.IndexFlatL2(d)

    # Build the index
    index.add(latents)

    print("Searching for nearest neighbors...")
    distances, indices = index.search(latents, k + 1)

    # Build sparse matrix with tqdm progress bar
    rows, cols, vals = [], [], []
    for i in tqdm(range(N), desc="Building sparse adjacency"):
        for j in range(1, k + 1):  # skip self at j=0
            rows.append(i)
            cols.append(indices[i, j])
            vals.append(distances[i, j])

    adj = csr_matrix((vals, (rows, cols)), shape=(N, N))
    print("✅ FAISS k-NN graph built successfully.")
    return adj

def choose_landmark_medoids(latents, n_clusters, n_landmarks):
    indices = np.random.choice(len(latents), size=n_landmarks, replace=False)
    landmark_latents = latents[indices]

    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', init='k-medoids++', random_state=42)
    kmedoids.fit(landmark_latents)
    medoid_indices = indices[kmedoids.medoid_indices_]
    return medoid_indices

def assign_clusters_landmark(adj, medoid_indices):
    dists = dijkstra(csgraph=adj, directed=False, indices=medoid_indices)
    labels = np.argmin(dists, axis=0)
    return labels

def build_codebook(latents, medoid_indices):
    codebook_latents = latents[medoid_indices]
    np.save('src/final_project/outputs/geodesic/codebook_latents.npy', codebook_latents)
    return codebook_latents

def main():
    set_seed()
    vae_config = load_config("src/final_project/configs/vae_config.yaml")
    batch_size = vae_config["training"]["batch_size"]
    clustering_config = load_config("src/final_project/configs/clustering_config.yaml")
    k_neighbors = clustering_config["k_neighbors"]
    n_clusters = clustering_config["n_clusters"]
    n_landmarks = clustering_config["n_landmarks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model('vae', "src/final_project/checkpoints/vae.pth", device)
    train_loader, _ = get_dataloaders(batch_size=batch_size)

    latents, _ = profile_step("Extract Latents", extract_latents, model, train_loader, device)
    latents = latents.astype(np.float32)

    adj = profile_step("Build FAISS k-NN Graph", build_faiss_knn_graph, latents, k_neighbors)
    medoid_indices = profile_step("Select Landmark Medoids", choose_landmark_medoids, latents, n_clusters, n_landmarks)
    labels = profile_step("Assign Clusters (Landmark Dijkstra)", assign_clusters_landmark, adj, medoid_indices)

    labels_reshaped = labels.reshape(60000, 7, 7)
    np.save('src/final_project/outputs/geodesic/kmedoids_code_maps.npy', labels_reshaped)

    _ = profile_step("Build Codebook", build_codebook, latents, medoid_indices)

if __name__ == "__main__":
    main()
