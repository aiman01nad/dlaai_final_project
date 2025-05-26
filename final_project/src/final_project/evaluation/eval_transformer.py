from matplotlib import pyplot as plt
import numpy as np
import torch

from final_project.utils.helpers import load_model, set_seed

@torch.no_grad()
def generate(model, start_token, seq_len, device):
    model.eval()
    generated = torch.tensor([[start_token]], dtype=torch.long).to(device)  # Shape: (1, 1)

    for _ in range(seq_len - 1):
        logits = model(generated)  # (1, current_len, vocab_size)
        next_token_logits = logits[:, -1, :]  # Get last token prediction
        probs = torch.softmax(next_token_logits, dim=-1)  # (1, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)  # Sample from distribution

        generated = torch.cat([generated, next_token], dim=1)  # Append

    return generated.squeeze().cpu().numpy()  # (seq_len,)

def main():
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transfomer = load_model('transformer', 'src/final_project/checkpoints/transformer.pth', device)
    vae = load_model('vae', 'src/final_project/checkpoints/vae.pth', device)

    # Generate a sequence of length 49 starting from token 0
    generated_seq = generate(transfomer, start_token=0, seq_len=49, device=device)
    code_grid = generated_seq.reshape(7, 7)

    codebook = np.load("src/final_project/outputs/geodesic/codebook_latents.npy")  # shape: (num_codes, latent_dim)
    # Convert code indices to embeddings
    latent_vectors = codebook[code_grid]  # shape: (7, 7, latent_dim)
    latent_tensor = torch.tensor(latent_vectors, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # shape: (1, C, H, W)
    latent_tensor = latent_tensor.to(device)

    # Pass the generated sequence to the decoder
    generated_image = vae.decoder(latent_tensor)
    generated_image = generated_image.squeeze().cpu().detach().numpy()  # shape: (28, 28)
    plt.imshow(generated_image, cmap='gray')
    plt.axis('off')
    plt.show()

