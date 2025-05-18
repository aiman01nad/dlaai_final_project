from matplotlib import pyplot as plt
import torch
import os
from final_project.models.vqvae import VQVAE
from final_project.data.mnist import get_dataloaders

def evaluate_model(model, checkpoint_path, output_dir, device, num_images=16):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    train_loader, test_loader = get_dataloaders(batch_size=num_images)
    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    images = images.to(device)

    reconstructions, _ = model(images)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i in range(num_images):
        # original images
        axes[0, i].imshow(images[i][0].cpu().detach(), cmap='gray')
        axes[0, i].axis('off')
        # reconstructed images
        axes[1, i].imshow(reconstructions[i][0].cpu().detach(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original")
            axes[1, i].set_title("Reconstructed")

    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(output_dir, "vqvae_reconstruction.png"), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
