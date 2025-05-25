import torch
from torch import nn
class Encoder(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, 4, stride=2, padding=1) # Input: (1, 28, 28) -> Output: (hidden_dim, 14, 14)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, 4, stride=2, padding=1) # Output: (hidden_dim*2, 7, 7)
        self.conv_mu = nn.Conv2d(hidden_dim*2, embedding_dim, 1)  # Output: (embedding_dim, 7, 7)
        self.conv_logvar = nn.Conv2d(hidden_dim*2, embedding_dim, 1) # Output: (embedding_dim, 7, 7)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.conv2 = nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, stride=2, padding=1) # Output: (hidden_dim, 14, 14)
        self.conv1 = nn.ConvTranspose2d(hidden_dim, 1, 4, stride=2, padding=1) # Output: (1, 28, 28)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x
    
class VAE(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = Encoder(hidden_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = (logvar * 0.5).exp() # Convert the logvar to std            
            return torch.distributions.Normal(loc=mu, scale=std).rsample() # reparameterization trick

        else:
            return mu