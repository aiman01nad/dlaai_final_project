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
