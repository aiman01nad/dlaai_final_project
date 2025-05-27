import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from final_project.models.vae import VAE
from torchvision.utils import make_grid


class VAELightningModule(pl.LightningModule):
    def __init__(self, hidden_dim, embedding_dim, lr, weight_decay, beta):
        super().__init__()
        self.save_hyperparameters()
        self.model = VAE(hidden_dim, embedding_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta

    def forward(self, x):
        return self.model(x)

    def elbo_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x.view(x.size(0), -1), x.view(x.size(0), -1), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, mu, logvar = self.model(x)
        loss, recon_loss, kl = self.elbo_loss(recon, x, mu, logvar)
        self.log_dict({"train_loss": loss, "train_recon": recon_loss, "train_kl": kl}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, mu, logvar = self.model(x)
        loss, recon_loss, kl = self.elbo_loss(recon, x, mu, logvar)
        self.log_dict({"val_loss": loss, "val_recon": recon_loss, "val_kl": kl}, on_epoch=True, prog_bar=True)

        # Log reconstructions
        if batch_idx == 0:
            grid = make_grid(torch.cat([x[:8], recon[:8]]), nrow=8, normalize=True)
            self.logger.experiment.add_image("Reconstructions", grid, self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        recon, mu, logvar = self.model(x)
        loss, recon_loss, kl = self.elbo_loss(recon, x, mu, logvar)
        self.log_dict({"test_loss": loss, "test_recon": recon_loss, "test_kl": kl}, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
