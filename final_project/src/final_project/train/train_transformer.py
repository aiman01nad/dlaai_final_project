import torch
from final_project.models.transformer import Transformer
from final_project.data.discrete_codes import get_dataloaders
import numpy as np

from final_project.utils.helpers import load_config, save_model, set_seed

def train_transformer(model: Transformer, code_map_flat, batch_size, num_embeddings, epochs, lr, weight_decay, device, save_name):
    train_loader, val_loader, test_loader = get_dataloaders(code_map_flat, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, num_embeddings), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_logits = model(x_val)
                val_loss += torch.nn.functional.cross_entropy(val_logits.view(-1, num_embeddings), y_val.view(-1)).item()

        print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}, Val Loss = {val_loss / len(val_loader):.4f}")

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
    weight_decay = float(transformer_config["training"]["weight_decay"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_name = "transformer.pth"

    model = Transformer(num_embeddings=num_embeddings, seq_len=seq_len,
                        embedding_dim=embedding_dim, nheads=nheads,
                        num_layers=num_layers, feedforward_dim=feedforward_dim,
                        dropout=dropout)
    model = model.to(device)

    code_maps = np.load("src/final_project/outputs/geodesic/kmedoids_code_maps.npy") # shape: (60000, 7, 7)
    code_map_flat = code_maps.reshape(60000, -1)  # shape: (60000, 49)

    model = train_transformer(model, code_map_flat, batch_size, num_embeddings, epochs, lr, weight_decay, device, save_name)

if __name__ == "__main__":
    main()


