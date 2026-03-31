import torch
import torch.nn as nn
from models.base_model import VAE_Base


class VAE(VAE_Base):
    def __init__(
        self,
        input_dim: int = 24,
        d_model: int = 64,
        n_layers: int = 1,
        latent_dim: int = 8,
    ):
        super().__init__("GRU_VAE", d_model, latent_dim)
        self.n_layers = n_layers
        self.input_dim = input_dim

        # Pre-encoder
        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
        )
        # Post-decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x: torch.Tensor):
        x = x.squeeze(1)

        h = self.enc_fc(x)

        mu, logvar = self.encode(h)
        z = self.reparameterize(mu, logvar)
        z_dec = self.decode(z)

        out = self.dec_fc(z_dec)
        out = out.unsqueeze(1)
        return out, mu, logvar


if __name__ == "__main__":
    x = torch.randn((64, 1, 24), dtype=torch.float32)
    model = VAE(True)
    recon, mu, logvar, rul = model(x)
    print(recon.shape, rul.shape)
