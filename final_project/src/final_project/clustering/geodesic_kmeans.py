import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from final_project.data.mnist import get_dataloaders
from final_project.utils import visualize_latents_vae
from final_project.utils.helpers import load_model
from final_project.utils.visualize_latents_vae import extract_latents_vae
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import calculate_distance_matrix

# def compute_geodesic_matrix(latents, k=10):
#     """Building a kNN graph and computing the geodesic distance matrix."""
#     nbrs = NearestNeighbors(n_neighbors=k).fit(latents)
#     distances, indices = nbrs.kneighbors(latents)

#     # Build the adjancy matrix
#     N = latents.shape[0] # Shape of latents: (N, latent_dim)
#     adj = np.full((N, N), np.inf)
#     for i in range(N):
#         for j in range(k):
#             adj[i, indices[i][j]] = distances[i][j]
#             adj[indices[i][j], i] = distances[i][j] # matrix is symmetric

#     adj = csr_matrix(adj) # Ensure the matrix is sparse to save memory
#     geodesic_matrix = shortest_path(adj, directed=False)
#     return geodesic_matrix

def compute_geodesic_matrix(latents, k=10):
    """Efficient kNN graph + geodesic distance computation using sparse matrices."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(latents)
    knn_graph = nbrs.kneighbors_graph(mode='distance')  # Sparse matrix (N x N)

    # Ensure it's in CSR format
    adj = csr_matrix(knn_graph)

    # Compute geodesic (shortest-path) distances
    geodesic_matrix = shortest_path(adj, directed=False)
    return geodesic_matrix

def cluster_with_kmedoids(distance_matrix, n_clusters=10):
    initial_medoids = kmeans_plusplus_initializer(distance_matrix, n_clusters, random_state=42).initialize()

    # Initialize KMedoids
    kmedoids_instance = kmedoids(data=distance_matrix, 
                                 initial_index_medoids=initial_medoids, 
                                 data_type='distance_matrix')
    kmedoids_instance.process()

    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()

    # Convert clusters to a list of cluster labels
    n_samples = distance_matrix.shape[0]
    labels = np.empty(n_samples, dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels[point_idx] = cluster_idx

    return labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "src/final_project/checkpoints/vae.pth"
    model = load_model('vae', model_path, device)
    train_loader, _ = get_dataloaders(batch_size=128)
    
    latents, _ = extract_latents_vae(model, train_loader, device)
    geodesic_matrix = compute_geodesic_matrix(latents, k=10)
    cluster_labels = cluster_with_kmedoids(geodesic_matrix, n_clusters=10)
    visualize_latents_vae(latents, cluster_labels, save_path='src/final_project/outputs/geodesic_kmedoids_latent_plot.png')

    print("Cluster labels:", cluster_labels)

if __name__ == "__main__":
    main()