import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tabulate import tabulate
from final_project.models.transformer_module import TransformerLightningModule
from final_project.models.vae import VAE
from final_project.models.vae_module import VAELightningModule
from final_project.models.vqvae import VQVAE
from final_project.models.transformer import Transformer
from final_project.evaluation.metrics import compute_ssim_psnr_batch, compute_perplexity, compute_fid
from final_project.data.mnist import get_dataloaders
from final_project.models.vqvae_module import VQVAELightningModule
from final_project.utils.helpers import set_seed
from final_project.utils.latent_extraction import flatten_latents

def evaluate_reconstruction(model, dataloader, device):
    model.eval()
    mse_total = 0
    all_x, all_xhat = [], []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_hat = model(x)[0] if hasattr(model, '__call__') else model.reconstruct(x)
            mse_total += F.mse_loss(x_hat, x, reduction='sum').item()
            all_x.append(x)
            all_xhat.append(x_hat)
    N = len(dataloader.dataset)
    return mse_total / N, torch.cat(all_x), torch.cat(all_xhat)

def plot_code_histogram(code_map, title='Code usage', save_path=None):
    flat = code_map.flatten()
    counts = np.bincount(flat, minlength=np.max(flat)+1)
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(counts)), counts)
    plt.title(title)
    plt.xlabel('Code index')
    plt.ylabel('Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_reconstruction_grid(x, x_hat, filename):
    x_grid = make_grid(x[:8], nrow=8)
    xhat_grid = make_grid(x_hat[:8], nrow=8)
    grid = torch.cat([x_grid, xhat_grid], dim=1)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title("Original (Top) vs. Reconstruction (Bottom)")
    plt.savefig(filename)
    plt.close()

def print_summary_report(metrics):
    table = [[
        name,
        f"{vals['mse']:.4f}",
        f"{vals['ssim']:.4f}",
        f"{vals['psnr']:.2f}",
        f"{vals.get('perplexity', '—'):.2f}" if 'perplexity' in vals else '—',
        f"{vals.get('nll', '—'):.4f}" if 'nll' in vals else '—',
        f"{vals.get('fid', '—'):.2f}" if 'fid' in vals else '—'
    ] for name, vals in metrics.items()]
    headers = ["Model", "MSE", "SSIM", "PSNR", "Perplexity", "NLL", "FID"]
    print("\n📊 Evaluation Summary:")
    print(tabulate(table, headers=headers, tablefmt="pretty"))

def main():
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test_loader = get_dataloaders(batch_size=128)

    metrics = {}

    # VAE + Geodesic Quantization
    vae_module = VAELightningModule.load_from_checkpoint("src/final_project/checkpoints/vae/vae-epoch=18-val_loss=14163.7070.ckpt")
    vae = vae_module.model

    mse, x, x_hat = evaluate_reconstruction(vae, test_loader, device)
    ssim, psnr = compute_ssim_psnr_batch(x, x_hat)
    metrics['VAE'] = {'mse': mse, 'ssim': ssim, 'psnr': psnr}
    save_reconstruction_grid(x, x_hat, 'vae_recon.png')

    # Evaluate VQ-VAE
    vqvae_module = VQVAELightningModule.load_from_checkpoint("src/final_project/checkpoints/vqvae/vqvae-epoch=29-val_loss=0.0132.ckpt")
    vqvae = vqvae_module.model

    mse, x, x_hat = evaluate_reconstruction(vqvae, test_loader, device)
    ssim, psnr = compute_ssim_psnr_batch(x, x_hat)
    perplexity = compute_perplexity(np.load("src/final_project/outputs/vqvae/vqvae_codes.npy"), vqvae.vq.num_embeddings)
    metrics['VQ-VAE'] = {'mse': mse, 'ssim': ssim, 'psnr': psnr, 'perplexity': perplexity}
    save_reconstruction_grid(x, x_hat, 'vqvae_recon.png')
    plot_code_histogram(np.load("src/final_project/outputs/vqvae/vqvae_codes.npy"), 'VQ-VAE Code Usage')

    # Evaluate Transformer on VAE-Geodesic codes
    transformer_module = TransformerLightningModule.load_from_checkpoint("src/final_project/checkpoints/transformer_vae-geodesic/transformer-epoch=09-val_loss=2.2358.ckpt")
    transformer = transformer_module.model

    code_map = np.load('src/final_project/outputs/geodesic/geodesic_codes.npy')
    code_map_flattened = flatten_latents(code_map)

    #fid = compute_fid(x, x_hat, device=device)
    perplexity = compute_perplexity(code_map_flattened, 128)  # e.g. 128 medoids
    metrics['Transformer (GeoQuant)'] = {'mse': 0, 'ssim': 0, 'psnr': 0, 'perplexity': perplexity}
    plot_code_histogram(code_map_flattened, 'Geodesic Quantization Code Usage')

    # Evaluate Transformer on VQVAE codes
    transformer_module = TransformerLightningModule.load_from_checkpoint("src/final_project/checkpoints/transformer_vqvae/transformer-epoch=09-val_loss=1.3112.ckpt")
    transformer = transformer_module.model

    code_map = np.load('src/final_project/outputs/vqvae/vqvae_codes.npy')
    code_map_flattened = flatten_latents(code_map)
    
    #fid = compute_fid(x, x_hat, device=device)
    perplexity = compute_perplexity(code_map_flattened, 128)  # e.g. 128 medoids
    metrics['Transformer (GeoQuant)'] = {'mse': 0, 'ssim': 0, 'psnr': 0, 'perplexity': perplexity}
    plot_code_histogram(code_map_flattened, 'Geodesic Quantization Code Usage')

    print_summary_report(metrics)

if __name__ == '__main__':
    main()