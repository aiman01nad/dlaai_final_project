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
    git clone https://github.com/aiman01nad/dlaai_final_project.git
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

### 1. **Train VAE and VQ-VAE, and extract latents**
Train the VAE and VQ-VAE models on the MNIST dataset and save the latent representations:

I did this using Google Colab and the ``colab_setup_and_run.ipynb`` notebook. Just replace the path in the last cell with either ``!PYTHONPATH=src python src/final_project/train/train_vae.py`` or ``!PYTHONPATH=src python src/final_project/train/train_vqvae.py``

### 2. **Geodesic Clustering**
Build the geodesic codebook and assign codes:
```bash
uv run geodesic
```

### 3. **Train Transformer Prior**
Train a Transformer on the discrete code sequences: 

Using Google Colab, replace the last cell with ``!PYTHONPATH=src python src/final_project/train/train_transformer.py --dataset_type vqvae`` or ``!PYTHONPATH=src python src/final_project/train/train_transformer.py --dataset_type vae-geodesic``

### 4. **Evaluate and Visualize**
Run evaluation and generate visualizations:
```bash
$env:PYTHONPATH="src" # If getting ModuleNotFoundError: No module named 'final_project'
python src/final_project/evaluation/evaluation.py
python src/final_project/evaluation/visualization.py
```

## Results
- Quantitative metrics:
  - **MSE (Mean Squared Error):** Measures the average squared difference between reconstructed and original images.
  - **SSIM (Structural Similarity Index):** Evaluates the perceptual similarity between images, focusing on structural information.
  - **PSNR (Peak Signal-to-Noise Ratio):** Quantifies the ratio between the maximum possible signal and the noise affecting the quality.
  - **Codebook perplexity:** Indicates the diversity of code usage in the discrete latent space.
- Visualizations: Latent space (t-SNE/UMAP), code usage histograms, generated samples.
- See the `outputs/` directory for figures and logs.
- Run ``tensorboard --logdir=src/final_project/logs/path_to_desired_log/`` to see Tensorboard logs

---

**Note:**  
- For more details on each step, see the docstrings and comments in the respective scripts.
- Configuration files for each model and clustering step are in `src/final_project/configs/`.