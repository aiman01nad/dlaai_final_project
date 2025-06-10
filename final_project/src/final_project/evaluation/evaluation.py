import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabulate import tabulate

from final_project.models import TransformerLightningModule, VAELightningModule, VQVAELightningModule
from final_project.evaluation import compute_ssim_psnr_batch, compute_perplexity, save_code_histogram, save_generated_images, save_reconstruction_grid, evaluate_reconstruction
from final_project.data.mnist import get_dataloaders as get_mnist_dataloaders
from final_project.data.discrete_codes import get_dataloaders as get_code_dataloaders
from final_project.utils import set_seed, flatten_latents

def eval_and_save_recon(model, dataloader, device, out_path, name, code_path=None, num_embeddings=128):
    mse, x, x_hat = evaluate_reconstruction(model, dataloader, device)
    ssim, psnr = compute_ssim_psnr_batch(x, x_hat)
    metrics = {'mse': mse, 'ssim': ssim, 'psnr': psnr}
    save_reconstruction_grid(x, x_hat, out_path)
    if code_path is not None:
        codes = np.load(code_path)
        perplexity = compute_perplexity(flatten_latents(codes), num_embeddings)
        metrics['perplexity'] = perplexity
        save_code_histogram(flatten_latents(codes), out_path.replace('.png', '_code_hist.png'), num_embeddings, f"Histogram of {name} Codes")
    return metrics

def eval_transformer(transformer, decoder, test_loader, device, out_prefix, num_embeddings=128, temperature=1.0, num_samples=16, code_hist_path=None):
    generated_codes, generated_imgs = generate_and_reconstruct(
        transformer, decoder, device=device, num_samples=num_samples, temperature=temperature, out_prefix=out_prefix
    )
    save_generated_images(generated_imgs.cpu(), f'{out_prefix}_samples.png', title=f"Transformer Generated Samples (T={temperature})")
    if code_hist_path is not None:
        save_code_histogram(generated_codes.cpu().numpy().flatten(), code_hist_path, num_embeddings, f"Histogram of Generated Codes (T={temperature})")
    try:
        x_test, _ = next(iter(test_loader))
        mse = F.mse_loss(generated_imgs, x_test.to(device)[:generated_imgs.size(0)], reduction='mean').item()
        ssim, psnr = compute_ssim_psnr_batch(generated_imgs.cpu(), x_test[:generated_imgs.size(0)])
    except Exception as e:
        print("Error during transformer evaluation:", e)
        mse, ssim, psnr = 0, 0, 0
    return mse, ssim, psnr

def generate_and_reconstruct(transformer_module, decoder_model, start_token=None, max_len=49, temperature=1.0, num_samples=1, device='cpu', out_prefix=None):
    transformer_module.eval()
    decoder_model.eval()
    reconstructed_imgs = []
    generated_seqs = []
    num_embeddings = 128

    with torch.no_grad():
        for _ in range(num_samples):
            # Generate a sequence from transformer
            generated_seq = transformer_module.generate_sequence(
                max_len=max_len, temperature=temperature, start_token=start_token
            ).to(device)  # [1, 49]
            generated_seqs.append(generated_seq)

            # Decode
            if hasattr(decoder_model, 'decode_indices'):
                img = decoder_model.decode_indices(generated_seq.view(1, 7, 7))

            # VAE+geodesic path
            elif hasattr(decoder_model, 'decoder'): 
                # Load codebook latents (same ones used during clustering)
                codebook = np.load("src/final_project/outputs/geodesic/codebook_latents.npy")
                codebook = torch.tensor(codebook, dtype=torch.float32, device=device)

                # Lookup codebook vectors
                quantized = codebook[generated_seq]  # [1, 49, D]

                # Reshape to image feature map shape expected by VAE decoder: [B, D, 7, 7]
                quantized = quantized.view(1, 7, 7, -1).permute(0, 3, 1, 2).contiguous()

                # Decode with VAE
                img = decoder_model.decoder(quantized)
            
            else:
                raise ValueError("Decoder model does not support decoding from indices or quantized latents.")

            reconstructed_imgs.append(img)

    all_codes = torch.cat(generated_seqs, dim=0).cpu().numpy().flatten()
    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        save_code_histogram(
            all_codes,
            f"{out_prefix}_code_hist.png",
            num_embeddings=num_embeddings,
            title=f"Histogram of Generated Codes (T={temperature})"
        )
    used_codes = set(np.unique(np.load("src/final_project/outputs/vqvae/vqvae_codes.npy")))
    gen_codes = set(all_codes)
    print(f"Generated codes not in training set (T={temperature}):", gen_codes - used_codes)

    return torch.cat(generated_seqs, dim=0), torch.cat(reconstructed_imgs, dim=0)

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
    _, _, mnist_test_loader = get_mnist_dataloaders(batch_size=128) # Use MNIST dataloader for VAE/VQ-VAE evaluation
    
    metrics = {}

    # VAE + Geodesic Quantization
    vae_module = VAELightningModule.load_from_checkpoint("src/final_project/checkpoints/vae/vae-epoch=18-val_loss=14163.7070.ckpt").to(device)
    vae = vae_module.model.to(device)
    metrics['VAE'] = eval_and_save_recon(
        vae, mnist_test_loader, device,
        'src/final_project/outputs/vae/vae_recon.png',
        name='VAE'
    )

    # VQ-VAE
    vqvae_module = VQVAELightningModule.load_from_checkpoint("src/final_project/checkpoints/vqvae/vqvae-epoch=29-val_loss=0.0132.ckpt").to(device)
    vqvae = vqvae_module.model.to(device)
    metrics['VQ-VAE'] = eval_and_save_recon(
        vqvae, mnist_test_loader, device,
        'src/final_project/outputs/vqvae/vqvae_recon.png',
        name='VQ-VAE',
        code_path="src/final_project/outputs/vqvae/vqvae_codes.npy"
    )

    # VAE-Geodesic code histogram
    test_geodesic_codes = np.load('src/final_project/outputs/geodesic/geodesic_codes.npy')
    save_code_histogram(
        flatten_latents(test_geodesic_codes),
        'src/final_project/outputs/geodesic/geodesic_code_hist.png',
        num_embeddings=128,
        title="Histogram of Geodesic Codes (Test Set)"
    )

    # Transformer on VAE-Geodesic codes (use MNIST test loader for image comparison)
    transformer_module_vaegeo = TransformerLightningModule.load_from_checkpoint(
        "src/final_project/checkpoints/transformer_vae-geodesic/transformer-epoch=09-val_loss=2.2358.ckpt"
    ).to(device)

    perplexity_geo = compute_perplexity(flatten_latents(test_geodesic_codes), 128)
    mse_gen, ssim_gen, psnr_gen = eval_transformer(
        transformer_module_vaegeo, vae, mnist_test_loader, device,
        out_prefix='src/final_project/outputs/transformer_vaegeo/generated',
        num_embeddings=128, temperature=1.0, num_samples=16
    )
    metrics['Transformer (GeoQuant)'] = {
        'mse': mse_gen,
        'ssim': ssim_gen,
        'psnr': psnr_gen,
        'perplexity': perplexity_geo
    }

    # Transformer on VQ-VAE codes with temperature sweep (use code test loader)
    transformer_module_vqvae = TransformerLightningModule.load_from_checkpoint(
        "src/final_project/checkpoints/transformer_vqvae/transformer-epoch=09-val_loss=1.3112.ckpt"
    ).to(device)

    # Use discrete code dataloader for transformer evaluation
    code_map_vqvae = np.load('src/final_project/outputs/vqvae/vqvae_codes.npy')
    _, _, code_test_loader = get_code_dataloaders(code_map_vqvae, batch_size=128)
    code_map_vqvae_flat = flatten_latents(code_map_vqvae)
    perplexity_vq = compute_perplexity(code_map_vqvae_flat, 128)

    sweep_metrics = []
    for temp in [0.7, 1.0, 1.2, 1.5]:
        print(f"Sampling with temperature={temp}")
        out_prefix = f'src/final_project/outputs/transformer_vqvae/temp{temp}'
        mse, ssim, psnr = eval_transformer(
            transformer_module_vqvae, vqvae, mnist_test_loader, device,
            out_prefix=out_prefix, num_embeddings=128, temperature=temp, num_samples=16
        )
        sweep_metrics.append({
            'temperature': temp,
            'mse': mse,
            'ssim': ssim,
            'psnr': psnr,
            'perplexity': perplexity_vq
        })

    pd.DataFrame(sweep_metrics).to_csv('src/final_project/outputs/transformer_vqvae/temperature_sweep_metrics.csv', index=False)
    print("\nTemperature Sweep Metrics:")
    print(pd.DataFrame(sweep_metrics))

    print_summary_report(metrics)

if __name__ == '__main__':
    main()