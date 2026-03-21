import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from data_module.cmapss_dataloader import FullDataLoader
from models.rnn_vae import RNN_VAE
from sklearn.metrics import accuracy_score, f1_score


class Trainer:
    def __init__(
        self,
        d_loader: FullDataLoader,
        model: RNN_VAE,
        device: str = "cuda",
        lr: float = 1e-3,
        beta: float = 1e-3,
        gamma: float = 1.0,
        log_every: int = 100,
        log_dir: str = "",
    ):
        self.train_loader = d_loader.get_train_loader()
        self.val_loader = d_loader.get_val_loader()
        self.test_loader = d_loader.get_test_loader()

        self.model = model
        self.model.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.beta = beta
        self.gamma = gamma
        self.device = device

        self.log_every = log_every
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        self.feature_map = {
            "unit": 0,
            "cycle": 1,
            "engine": 2,
            "rul": 3,
            "health_idx": 4,
            "health_level": 5,
        }

        self.global_step = 0

    def compute_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        rul_hat: torch.Tensor,
        rul: torch.Tensor,
    ):
        recon_mse = F.mse_loss(x_hat, x)
        # recon_mse = ((x - x_hat) ** 2).mean(dim=(1, 2)).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        rul_mse = F.mse_loss(rul_hat, rul)

        loss = recon_mse + self.beta * kl + self.gamma * rul_mse
        return loss, recon_mse, kl, rul_mse

    def train(self, verbose: bool = True):
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x, meta = self._device_to(batch)
            x_hat, mu, logvar, rul_hat = self.model(x)
            rul = self.get_meta_feature(meta, "rul", -1)

            losses = self.compute_loss(x_hat, x, mu, logvar, rul_hat, rul)
            loss, recon_mse, kl, rul_mse = losses
            loss.backward()
            self.optimizer.step()
            loss = loss.item()

            rul_rmse = torch.sqrt(rul_mse.detach()).item()
            elbo = recon_mse.item() + self.beta * kl.item()

            if i % self.log_every == 0:
                self.writer.add_scalar("train/loss", loss, self.global_step)
                self.writer.add_scalar("train/rul_rmse", rul_rmse, self.global_step)
                self.writer.add_scalar("train/elbo", elbo, self.global_step)
                if verbose:
                    print(f"    loss: {loss:.3f}")

            self.global_step += 1

    def fit(self, n_epochs: int = 10, verbose: bool = True):
        for i in range(1, n_epochs + 1):
            if verbose:
                print(f" [Epoch {i}]  Training")
            self.train(verbose)
            if verbose:
                print(f" [Epoch {i}]  Validation")
            self.validate(verbose)

        torch.save(self.model.state_dict(), f"{self.log_dir}/model.pth")

    def validate(self, verbose: bool = True):
        self.model.eval()
        total_losses = []
        rul_rmses = []
        elbos = []
        ruls = []
        pred_ruls = []
        for _, batch in enumerate(self.val_loader):
            x, meta = self._device_to(batch)
            with torch.no_grad():
                x_hat, mu, logvar, rul_hat = self.model(x)
                rul = self.get_meta_feature(meta, "rul", -1)
                # Max RUL from training set is 542
                ruls.append(rul * 542)
                pred_ruls.append(rul_hat * 542)

                losses = self.compute_loss(x_hat, x, mu, logvar, rul_hat, rul)
                loss, recon_mse, kl, rul_mse = losses

                rul_rmse = torch.sqrt(rul_mse.detach()).item()
                elbo = recon_mse.item() + self.beta * kl.item()

                total_losses.append(loss.item())
                rul_rmses.append(rul_rmse)
                elbos.append(elbo)

        loss = np.mean(total_losses)
        rul_rmse = np.mean(rul_rmses)
        elbo = np.mean(elbos)
        self.writer.add_scalar("val/loss", loss, self.global_step)
        self.writer.add_scalar("val/rul_rmse", rul_rmse, self.global_step)
        self.writer.add_scalar("val/elbo", elbo, self.global_step)
        if verbose:
            print(f"    loss: {loss:.3f} | rul_rmse: {rul_rmse:.3f} | elbo: {elbo:3f}")

        rul = torch.cat(ruls)
        rul_hat = torch.cat(pred_ruls)
        accuracy, f1 = self.evaluate_rul_predictions(rul_hat, rul)
        self.writer.add_scalar("val/acc", accuracy, self.global_step)
        self.writer.add_scalar("val/f1", f1, self.global_step)
        if verbose:
            print(f"    accuracy: {accuracy:.3f}")
            print(f"    f1: {f1:.3f}")

    def evaluate_rul_predictions(self, rul_hat: torch.Tensor, rul: torch.Tensor):
        rul_hat, rul = rul_hat.detach().cpu(), rul.detach().cpu()

        pred_health_level = get_health_level(rul_hat)
        health_level = get_health_level(rul)

        accuracy = accuracy_score(health_level, pred_health_level)
        f1 = f1_score(health_level, pred_health_level, average="macro")

        return accuracy, f1

    def get_meta_feature(self, meta: torch.Tensor, feature: str, window_idx: int = -1):
        idx = self.feature_map[feature]
        return meta[:, window_idx, idx]

    def _device_to(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, meta = batch
        return x.to(self.device), meta.to(self.device)


def get_health_level(rul: torch.Tensor) -> np.ndarray:
    y = np.ones_like(rul)
    y[rul > 120] = 0
    y[rul < 30] = 2
    return y
