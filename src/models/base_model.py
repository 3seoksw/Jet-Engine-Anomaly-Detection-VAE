import torch
import torch.nn as nn


class VAE_Base(nn.Module):
    def __init__(self, name: str, d_model: int, latent_dim: int):
        super().__init__()
        self.name = name

        # Encoder
        self.enc_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(d_model, latent_dim)
        self.enc_logvar = nn.Linear(d_model, latent_dim)

        # Decoder
        self.dec_linear = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
        )

        # RUL Head
        self.rul = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Softplus(),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc_linear(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor):
        return self.dec_linear(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
