import torch
from torch.utils.data import DataLoader
from final_project.models.transformer import Transformer
from final_project.data.discrete_codes import CodeSequenceDataset
import numpy as np

from final_project.utils.helpers import load_config, save_model, set_seed

def train_transformer(model: Transformer, dataset: CodeSequenceDataset, batch_size, num_embeddings, epochs, lr, device, save_name):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # (B, seq_len, num_embeddings)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, num_embeddings), yb.view(-1))
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    save_model(model, save_name)
    return model

def main():
    set_seed()
    transformer_config = load_config("src/final_project/configs/transformer_config.yaml")

    num_embeddings = transformer_config["model"]["num_embeddings"]
    seq_len = transformer_config["model"]["seq_len"]
    embedding_dim = transformer_config["model"]["embedding_dim"]
    nheads = transformer_config["model"]["nheads"]
    num_layers = transformer_config["model"]["num_layers"]
    feedforward_dim = transformer_config["model"]["feedforward_dim"]
    dropout = transformer_config["model"]["dropout"]

    batch_size = transformer_config["training"]["batch_size"]
    epochs = transformer_config["training"]["epochs"]
    lr = float(transformer_config["training"]["learning_rate"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_name = "transformer.pth"

    model = Transformer(num_embeddings=num_embeddings, seq_len=seq_len,
                        embedding_dim=embedding_dim, nheads=nheads,
                        num_layers=num_layers, feedforward_dim=feedforward_dim,
                        dropout=dropout)
    model = model.to(device)

    codes = np.load("src/final_project/outputs/geodesic/kmedoids_labels.npy")  # shape: (60000,)
    dataset = CodeSequenceDataset(codes, seq_len)

    model = train_transformer(model, dataset, batch_size, num_embeddings, epochs, lr, device, save_name)

if __name__ == "__main__":
    main()


