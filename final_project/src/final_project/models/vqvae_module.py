import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from final_project.models.vqvae import VQVAE
from torchvision.utils import make_grid

class VQVAELightningModule(pl.LightningModule):
    def __init__(self, hidden_dim, embedding_dim, num_embeddings, commitment_cost, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VQVAE(hidden_dim, embedding_dim, num_embeddings, commitment_cost)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, vq_loss, _ = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, vq_loss, _ = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # Log images to TensorBoard
        if batch_idx == 0:
            grid = make_grid(torch.cat([x[:8], recon[:8]]), nrow=8, normalize=True)
            self.logger.experiment.add_image("Reconstructions", grid, self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        recon, vq_loss, _ = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        self.log("test_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
