import numpy as np
import torch

from final_project.data.mnist import get_dataloaders

def reconstruction_error(model, latents, medoid_indices, landmark_dists, device, C=8, H=7, W=7):
    N = latents.shape[0]

    # Find closest medoid for each latent point
    distances = landmark_dists[:, np.arange(N)]  # shape: (num_medoids, N)
    closest_medoid_idx = np.argmin(distances, axis=0)  # (N,)

    # Quantize latents by replacing each latent with its closest medoid latent
    quantized_latents = latents[medoid_indices[closest_medoid_idx]]  # (N, D)

    # Reshape quantized latents back to 4D tensor for decoding
    z = torch.tensor(quantized_latents, dtype=torch.float32, device=device).view(-1, C, H, W)

    model.eval()
    with torch.no_grad():
        reconstructions = model.decoder(z).cpu().numpy()  # shape: (N, 1, 28, 28)

    # Load original images (for reconstruction error)
    train_loader, _ = get_dataloaders(batch_size=128)
    images = []
    for x, _ in train_loader:
        images.append(x)
    original_images = torch.cat(images, dim=0).numpy()

    # Compute MSE between reconstructions and original images
    mse = np.mean((reconstructions - original_images) ** 2)
    print(f"Reconstruction MSE (using quantized latents): {mse:.4f}")
    return mse