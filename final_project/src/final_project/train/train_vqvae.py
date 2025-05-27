import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from final_project.models.vqvae_module import VQVAELightningModule
from final_project.data.mnist import get_dataloaders
from final_project.utils import load_config, set_seed

def main():
    set_seed()
    config = load_config("src/final_project/configs/vqvae_config.yaml")

    model_cfg = config["model"]
    train_cfg = config["training"]

    module = VQVAELightningModule(
        hidden_dim=model_cfg["hidden_dim"],
        embedding_dim=model_cfg["embedding_dim"],
        num_embeddings=model_cfg["num_embeddings"],
        commitment_cost=model_cfg["commitment_cost"],
        lr=float(train_cfg["learning_rate"]),
    )

    train_loader, val_loader, test_loader = get_dataloaders(train_cfg["batch_size"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/vqvae",
        filename="vqvae-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    logger = TensorBoardLogger("logs", name="vqvae")

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