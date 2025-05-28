import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from final_project.models.transformer import Transformer

class TransformerLightningModule(pl.LightningModule):
    def __init__(self, num_embeddings, seq_len, embedding_dim, nheads, num_layers, feedforward_dim, dropout, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(num_embeddings, seq_len, embedding_dim, nheads, num_layers, feedforward_dim, dropout)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, self.hparams.num_embeddings), y.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, self.hparams.num_embeddings), y.view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, self.hparams.num_embeddings), y.view(-1))
        self.log("test_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def generate_sequence(self, max_len=None, temperature=1.0, start_token=None):
        self.eval()
        generated = []

        if max_len is None:
            max_len = self.model.seq_len

        if start_token is None:
            token = torch.randint(0, self.hparams.num_embeddings, (1, 1), device=self.device)
        else:
            token = torch.tensor([[start_token]], device=self.device)

        generated.append(token)

        for _ in range(max_len - 1):
            input_seq = torch.cat(generated, dim=1)  # shape: (1, T)
            logits = self(input_seq)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)

        return torch.cat(generated, dim=1)  # shape: (1, 49)
