import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from data_module.cmapss_dataloader import FullDataLoader
from models.vae import VAE
from models.rnn_vae import RNN_VAE
from trainer.trainer import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(100)
    np.random.seed(100)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f" {device} loaded")

    d_loader = FullDataLoader(
        data_dir=cfg.data.data_dir,
        window=cfg.data.window,
        batch_size=cfg.data.batch_size,
    )
    input_dim = d_loader.get_input_dim()

    if cfg.data.window == 1:
        model = VAE(
            input_dim=input_dim,
            d_model=cfg.model.d_model,
            n_layers=1,
            latent_dim=cfg.model.latent_dim,
        )
    else:
        model = RNN_VAE(
            input_dim=input_dim,
            d_model=cfg.model.d_model,
            n_layers=1,
            latent_dim=cfg.model.latent_dim,
        )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    log_dir = HydraConfig.get().runtime.output_dir
    trainer = Trainer(
        d_loader=d_loader,
        model=model,
        device=device,
        lr=cfg.train.lr,
        beta=cfg.train.beta,
        gamma=cfg.train.gamma,
        log_dir=log_dir,
        n_epochs=cfg.train.epochs,
    )
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    main()
