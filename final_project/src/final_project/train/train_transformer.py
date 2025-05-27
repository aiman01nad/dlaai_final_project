import sys
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from final_project.models.transformer_module import TransformerLightningModule
from final_project.data.discrete_codes import get_dataloaders
from final_project.utils import load_config, set_seed


def main():
    set_seed()
    cfg = load_config("src/final_project/configs/transformer_config.yaml")
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    dataset_type = " ".join(sys.argv[1:])
    print(dataset_type)
    # Load dataset
    if dataset_type == "vqvae":
        code_maps = np.load("src/final_project/outputs/vqvae/vqvae_codes.npy")
    elif dataset_type == "vae-geodesic":
        code_maps = np.load("src/final_project/outputs/geodesic/kmedoids_code_maps.npy")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    code_map_flat = code_maps.reshape(code_maps.shape[0], -1)  # shape: (N, 7*7)
    train_loader, val_loader, test_loader = get_dataloaders(code_map_flat, train_cfg["batch_size"])

    module = TransformerLightningModule(
        num_embeddings=model_cfg["num_embeddings"],
        seq_len=code_map_flat.shape[1] - 1,
        embedding_dim=model_cfg["embedding_dim"],
        nheads=model_cfg["nheads"],
        num_layers=model_cfg["num_layers"],
        feedforward_dim=model_cfg["feedforward_dim"],
        dropout=model_cfg["dropout"],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"src/final_project/checkpoints/transformer_{dataset_type}/",
        filename="transformer-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    logger = TensorBoardLogger("src/final_project/logs", name=f"transformer_{dataset_type}")

    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"],
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto"
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(module, dataloaders=test_loader)


if __name__ == "__main__":
    main()
