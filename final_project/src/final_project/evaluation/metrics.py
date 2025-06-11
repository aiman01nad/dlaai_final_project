import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_perplexity(code_map, num_codes):
    counts = np.bincount(code_map.flatten(), minlength=num_codes)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return np.exp(entropy)

def compute_ssim_psnr_batch(x, x_hat):
    """Assumes input: tensors (N, 1, H, W)"""
    x_np = x.cpu().numpy()
    x_hat_np = x_hat.cpu().numpy()
    ssim_vals = []
    psnr_vals = []
    for i in range(x.shape[0]):
        ssim_vals.append(ssim(x_np[i, 0], x_hat_np[i, 0], data_range=1.0))
        psnr_vals.append(psnr(x_np[i, 0], x_hat_np[i, 0], data_range=1.0))
    return np.mean(ssim_vals), np.mean(psnr_vals)

def evaluate_reconstruction(model, dataloader, device):
    model.eval()
    mse_total = 0
    all_x, all_xhat = [], []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_hat = model(x)[0]
            mse_total += F.mse_loss(x_hat, x, reduction='sum').item()
            all_x.append(x)
            all_xhat.append(x_hat)
    N = len(dataloader.dataset)
    return mse_total / N, torch.cat(all_x), torch.cat(all_xhat)