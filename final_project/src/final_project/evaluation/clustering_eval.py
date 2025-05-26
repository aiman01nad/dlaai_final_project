import numpy as np
from scipy.stats import entropy

def compute_cluster_stats(assignments, n_clusters):
    counts = np.bincount(assignments, minlength=n_clusters)
    probs = counts / np.sum(counts)
    perp = np.exp(entropy(probs))
    print(f"Cluster counts: {counts}")
    print(f"Effective number of clusters (perplexity): {perp:.2f}")
    return counts, perp

def assign_clusters(latents, landmark_dists):
    # Assign each latent vector to its nearest medoid using landmark distances
    N = latents.shape[0]
    dists = landmark_dists[:, np.arange(N)]  # shape: (num_landmarks, N)
    assignments = np.argmin(dists, axis=0)   # (N,)
    return assignments

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

def approximate_geodesic_distance(i, j, landmark_dists):
    """Approximate geodesic distance between two points using landmark distances."""
    d_i = landmark_dists[:, i]
    d_j = landmark_dists[:, j]
    return np.min(d_i + d_j)