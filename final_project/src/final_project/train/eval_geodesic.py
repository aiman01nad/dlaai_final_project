from final_project.evaluation.visualization import visualize_decoded_medoids, visualize_latents_tsne, visualize_latents_pca
from final_project.utils.helpers import load_model, set_seed
from final_project.evaluation import clustering_eval as clust_eval
from final_project.evaluation import reconstruction_eval as recon_eval
import torch
import numpy as np

def main():
    set_seed()
    full_latents = np.load('src/final_project/outputs/vae/vae_latents.npy')
    codebook_latents = np.load('src/final_project/outputs/geodesic/codebook_latents.npy')
    labels = np.load('src/final_project/outputs/geodesic/kmedoids_labels.npy')
    save_path = 'src/final_project/outputs/geodesic/geodesic_latent_plot.png'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('vae', 'src/final_project/checkpoints/vae.pth', device)

    #visualize_latents_tsne(full_latents, labels, save_path)
    #visualize_decoded_medoids(codebook_latents, model, device, save_path='src/final_project/outputs/geodesic/decoded_medoids.png')

if __name__ == "__main__":
    main()
