import torch
import torch.nn as nn
from models.base_model import VAE_Base


class RNN_VAE(VAE_Base):
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
        self.enc_rnn = nn.GRU(input_dim, d_model, num_layers=n_layers, batch_first=True)
        # Post-decoder
        self.dec_rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.dec_out = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor):
        _, window, _ = x.shape
        _, h_n = self.enc_rnn(x)
        h = h_n[-1]

        # Base VAE Process
        mu, logvar = self.encode(h)
        z = self.reparameterize(mu, logvar)
        z_dec = self.decode(z)  # [B, d_model]

        # Decoder hidden state
        h0 = z_dec.unsqueeze(0).repeat(self.n_layers, 1, 1)  # [n_layers, B, d_model]

        # Decoder input
        dec_input = z_dec.unsqueeze(1).repeat(1, window, 1)  # [B, window, d_model]

        out, _ = self.dec_rnn(dec_input, h0)
        out = self.dec_out(out)
        return out, mu, logvar


if __name__ == "__main__":
    x = torch.randn((64, 5, 24), dtype=torch.float32)
    model = RNN_VAE(True)
    out, mu, logvar, rul = model(x)
    print(out.shape, rul.shape, mu.shape)
