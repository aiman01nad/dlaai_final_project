import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from final_project.models.vae_module import VAELightningModule
from final_project.data.mnist import get_dataloaders
from final_project.utils import load_config, set_seed


def main():
    set_seed()
    config = load_config("src/final_project/configs/vae_config.yaml")

    model_cfg = config["model"]
    train_cfg = config["training"]

    module = VAELightningModule(
        hidden_dim=model_cfg["hidden_dim"],
        embedding_dim=model_cfg["embedding_dim"],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        beta=train_cfg["beta"],
    )

    train_loader, val_loader, test_loader = get_dataloaders(train_cfg["batch_size"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="src/final_project/checkpoints/vae",
        filename="vae-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    logger = TensorBoardLogger("src/final_project/logs", name="vae")

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