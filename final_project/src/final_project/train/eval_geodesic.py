from final_project.evaluation import reconstruction_error, compute_cluster_stats, assign_clusters, plot_latent_clusters, visualize_codebook
from final_project.utils.helpers import load_model, set_seed
import torch
import numpy as np

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('vae', 'src/final_project/checkpoints/vae.pth', device)
    
    latents = np.load("src/final_project/outputs/vae/vae_latents.npy")
    labels = np.load("src/final_project/outputs/vae/vae_labels.npy")
    landmark_dists = np.load("src/final_project/outputs/geodesic/landmark_dists.npy")
    medoid_indices = np.load("src/final_project/outputs/geodesic/medoid_indices.npy")
    codebook_latents = np.load("src/final_project/outputs/geodesic/codebook_latents.npy")
    
    # Assign cluster ids
    assignments = assign_clusters(latents, landmark_dists)

    # Reshape flat latents into 4D shape expected by decoder
    embedding_dim = model.encoder.conv_mu.out_channels  # e.g., 8
    spatial_dim = model.encoder.conv_mu.out_channels  # Likely 7x7; get from model or hardcode if known
    spatial_dim = 7  # or compute from model if you want it dynamic

    latents_reshaped = latents.reshape(-1, embedding_dim, spatial_dim, spatial_dim)

    # Evaluation
    compute_cluster_stats(assignments, n_clusters=len(medoid_indices))
    reconstruction_error(model, latents_reshaped, medoid_indices, landmark_dists, device)

    # Visualization
    plot_latent_clusters(latents, labels, assignments, save_path='src/final_project/outputs/geodesic/geodesic_clusters_plot.png')
    visualize_codebook(model, codebook_latents.reshape(-1, embedding_dim, spatial_dim, spatial_dim))

if __name__ == "__main__":
    main()
