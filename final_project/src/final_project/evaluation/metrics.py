import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.linalg import sqrtm

@torch.no_grad()
def compute_fid(real_images, generated_images, device='cpu'):
    """Compute Fréchet Inception Distance (FID)"""
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    def get_activations(images):
        if isinstance(images, np.ndarray):
            images = torch.tensor(images)
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = images.expand(-1, 3, -1, -1)  # grayscale → RGB
        activations = inception(images)[0].detach().cpu().numpy()
        return activations

    act_real = get_activations(real_images.to(device))
    act_gen = get_activations(generated_images.to(device))

    mu1, sigma1 = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu2, sigma2 = act_gen.mean(axis=0), np.cov(act_gen, rowvar=False)

    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real

    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

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