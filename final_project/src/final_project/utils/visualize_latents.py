import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from final_project.models import VQVAE
from final_project.data import get_dataloaders

def visualize_latents(model: VQVAE, model_path, output_dir, device, method='tsne', batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    train_loader, test_loader = get_dataloaders(batch_size)

    latents, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            z = model.encoder(x) # encoder output (continous latent)
            latents.append(z.cpu().view(x.size(0), -1))  # flatten spatial dims
            labels.append(y)

    latents = torch.cat(latents).numpy()
    labels = torch.cat(labels).numpy()

    reducer = TSNE(n_components=2) if method == 'tsne' else PCA(n_components=2)
    reduced = reducer.fit_transform(latents)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f"{method.upper()} Visualization of VQVAE Latent Space")
    plt.savefig(os.path.join(output_dir, f"{method}_latent_plot.png"))
    plt.show()
