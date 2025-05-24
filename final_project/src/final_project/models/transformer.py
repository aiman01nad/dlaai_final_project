import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, num_embeddings, seq_len, embedding_dim, nheads, num_layers, feedforward_dim, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nheads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, x):
        B, T = x.size()
        x = self.token_emb(x) + self.pos_emb[:, :T, :]

        # Causal mask to prevent seeing future tokens
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device) 

        x = self.transformer(x, mask=mask)
        return self.output(x)