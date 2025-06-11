# DLAAI Final Project: VQ-VAE a posteriori with Geodesic Quantization

This repository contains the final project for the Deep Learning and Artificial Intelligence (DLAAI) course at Sapienza. The project explores geometry-aware post-hoc quantization of VAE latent spaces using geodesic distances, and compares it to standard VQ-VAE baselines on MNIST.

## Project Overview

This project investigates whether geometry-aware quantization—using geodesic distances on a latent k-NN graph—can improve discrete latent representations in generative models. We:
- Train a standard VAE and a VQ-VAE on MNIST.
- Construct a k-NN graph over VAE latents and perform geodesic K-Medoids clustering to build a discrete codebook.
- Assign discrete codes to latents and train a Transformer prior over these codes.
- Compare reconstruction quality, codebook utilization, and generative performance across methods.

## Directory Structure

```
.
├── src/
│   └── final_project/
│       ├── checkpoints         # Saved checkpoints from training    
│       ├── clustering/         # Geodesic clustering and codebook construction
│       ├── configs/            # YAML config files for models and clustering
│       ├── data/               # MNIST dataloader and discrete code dataloader
│       ├── evaluation/         # Evaluation scripts, visualizations and metrics
│       ├── logs/               # Tensorboard logging from training
│       ├── models/             # Model definitions (VAE, VQ-VAE, Transformer)
│       ├── outputs/            # Saved outputs: codes, latents, figures
│       ├── train/              # Training scripts for VAE and VQ-VAE
│       └── utils/              # Helper functions
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/final_project.git
    cd final_project
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Unix/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Or, with [uv](https://github.com/astral-sh/uv):
    ```bash
    uv pip install -r requirements.txt
    ```

## How to Run

### 1. **Extract Latents and Codes**
Extract VAE or VQ-VAE latents/codes for clustering and evaluation:
```bash
python -m src.final_project.utils.latent_extraction --model_type vae
python -m src.final_project.utils.latent_extraction --model_type vqvae
```

### 2. **Geodesic Clustering**
Build the geodesic codebook and assign codes:
```bash
python -m src.final_project.clustering.geodesic_clustering
```

### 3. **Train Models**
Train VAE or VQ-VAE using provided configs:
```bash
python -m src.final_project.train.train_vae --config src/final_project/configs/vae_config.yaml
python -m src.final_project.train.train_vqvae --config src/final_project/configs/vqvae_config.yaml
```

### 4. **Train Transformer Prior**
Train a Transformer on the discrete code sequences:
```bash
python -m src.final_project.train.train_transformer --config src/final_project/configs/transformer_config.yaml
```

### 5. **Evaluate and Visualize**
Run evaluation and generate visualizations:
```bash
python -m src.final_project.evaluation.evaluation
python -m src.final_project.evaluation.visualization
```

## Results

- Quantitative metrics: MSE, SSIM, PSNR, codebook perplexity.
- Visualizations: Latent space (t-SNE/UMAP), code usage histograms, generated samples.
- See the `outputs/` directory for figures and logs.

---

**Note:**  
- For more details on each step, see the docstrings and comments in the respective scripts.
- Configuration files for each model and clustering step are in `src/final_project/configs/`.