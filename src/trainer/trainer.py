import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from data_module.cmapss_dataloader import FullDataLoader
from trainer.anomaly_detector import ParametricAnomalyDetector
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from trainer.utils import (
    plot_loss_results,
    plot_performance,
    plot_confusion_matrix,
    plot_latent_3d_all,
    plot_degradation,
)


class Trainer:
    def __init__(
        self,
        d_loader: FullDataLoader,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        beta: float = 1e-3,
        gamma: float = 1.0,
        log_every: int = 100,
        log_dir: str = "",
        n_epochs: int = 10,
    ):
        self.n_epochs = n_epochs
        self.threshold = 0
        self.gamma = gamma
        self.detector = ParametricAnomalyDetector()

        self.train_loader = d_loader.get_train_loader()
        self.val_loader = d_loader.get_val_loader()
        self.test_loader = d_loader.get_test_loader()

        self.model = model
        self.model.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.device = device

        self.best_perf = 0
        self.best_model = copy.deepcopy(self.model).to(device)
        self.best_model.load_state_dict(self.model.state_dict())

        self.beta_max = beta
        self.beta_schedule = np.linspace(0.1, self.beta_max, n_epochs)
        self.beta = beta
        self.best_beta = beta
        self.warmup = 20

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

    def compute_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        free_bits=0.0,
    ):
        recon_mse = F.mse_loss(x_hat, x)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.clamp(kl, min=free_bits).mean()
        loss = recon_mse + self.beta * kl
        return loss, recon_mse, kl

    def compute_loss_per_sample(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        recon = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
        elbo = (recon + self.beta * kl).cpu().numpy()
        return elbo, recon.cpu().numpy(), kl.cpu().numpy()

    def calibrate_threshold(self, percentile: float = 85):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in self.train_loader:
                x, _ = self._device_to(batch)
                x_hat, mu, logvar = self.model(x)
                loss, recon, kl = self.compute_loss_per_sample(x_hat, x, mu, logvar)
                scores.append(kl)
        scores = np.concatenate(scores)
        return np.percentile(scores, percentile)

    def train(self, verbose: bool = True):
        self.model.train()
        elbos, recons, kls = [], [], []
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            x, _ = self._device_to(batch)
            x_hat, mu, logvar = self.model(x)

            losses = self.compute_loss(x_hat, x, mu, logvar)
            loss, recon, kl = losses
            loss.backward()
            self.optimizer.step()

            elbos.append(loss.item())
            recons.append(recon.item())
            kls.append(kl.item())
        if verbose:
            elbo, recon, kl = np.mean(elbos), np.mean(recons), np.mean(kls)
            print(f"    elbo: {elbo:.3f} | recon: {recon:.3f} | kl: {kl:.4f}")
        return {"elbo": elbos, "recon": recons, "kl": kls}

    def fit(self, verbose: bool = True):
        for epoch in range(1, self.n_epochs + 1):
            self.beta = self._update_beta(epoch)
            if verbose:
                print(f" [Epoch {epoch}]  Training - beta={self.beta:.3f}")
            train_losses = self.train(verbose)

            for loss_type, values in train_losses.items():
                self.writer.add_scalar(
                    f"train/loss_{loss_type}_mean", np.mean(values), epoch
                )
                self.writer.add_scalar(
                    f"train/loss_{loss_type}_std", np.std(values), epoch
                )

            threshold = self.calibrate_threshold()
            self.threshold = threshold

            if verbose:
                print(f" [Epoch {epoch}]  Validation")
            self.validate(epoch, verbose)

        torch.save(self.model.state_dict(), f"{self.log_dir}/model.pth")
        torch.save(self.best_model.state_dict(), f"{self.log_dir}/best_model.pth")

    def validate(self, epoch: int, verbose: bool = True):
        self.model.eval()
        elbos, recons, kls = [], [], []
        health_levels, engines, units = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                x, meta = self._device_to(batch)
                x_hat, mu, logvar = self.model(x)

                # Loss
                elbo, recon, kl = self.compute_loss_per_sample(x_hat, x, mu, logvar)
                elbos.append(elbo)
                recons.append(recon)
                kls.append(kl)

                # Engine-unit pairs
                health_levels.append(
                    self.get_meta_feature(meta, "health_level").cpu().numpy()
                )
                engines.append(self.get_meta_feature(meta, "engine").cpu().numpy())
                units.append(self.get_meta_feature(meta, "unit").cpu().numpy())

        # Loss logging
        elbos = np.concatenate(elbos)
        recons = np.concatenate(recons)
        kls = np.concatenate(kls)
        self.writer.add_scalar("val/loss_elbo", elbos.mean(), epoch)
        self.writer.add_scalar("val/loss_recon", recons.mean(), epoch)
        self.writer.add_scalar("val/loss_kl", kls.mean(), epoch)

        # Performance logging warmup (engine-unit pairing)
        health_levels = np.concatenate(health_levels)
        engines = np.concatenate(engines)
        units = np.concatenate(units)
        anomalies = (health_levels == 2).astype(int)

        # Performance logging
        # ELBO
        roc_auc_elbo = roc_auc_score(anomalies, elbos)
        pred_levels = self._predict(elbos, engines, units)
        f1_elbo = f1_score(anomalies, pred_levels)
        self.writer.add_scalar("val/roc_auc_elbo", roc_auc_elbo, epoch)
        self.writer.add_scalar("val/f1_elbo", f1_elbo, epoch)
        # Recon
        roc_auc_recon = roc_auc_score(anomalies, recons)
        pred_levels = self._predict(recons, engines, units)
        f1_recon = f1_score(anomalies, pred_levels)
        self.writer.add_scalar("val/roc_auc_recon", roc_auc_recon, epoch)
        self.writer.add_scalar("val/f1_recon", f1_recon, epoch)
        # KL
        roc_auc_kl = roc_auc_score(anomalies, kls)
        pred_levels = self._predict(kls, engines, units)
        f1_kl = f1_score(anomalies, pred_levels)
        self.writer.add_scalar("val/roc_auc_kl", roc_auc_kl, epoch)
        self.writer.add_scalar("val/f1_kl", f1_kl, epoch)

        if self.best_perf < roc_auc_kl:
            self.best_perf = roc_auc_kl
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_beta = self.beta

        if verbose:
            elbo, recon, kl = elbos.mean(), recons.mean(), kls.mean()
            t = f" (t={self.threshold:.4f})"
            print(f"    elbo: {elbo:.3f} | recon: {recon:.3f} | kl: {kl:.4f}")
            print(f"    roc_auc: {roc_auc_kl:.3f} | f1: {f1_kl:.3f}{t}")

    def test(self):
        # self.best_model.eval()
        self.model.eval()
        self.beta = self.best_beta
        elbos, recons, kls = [], [], []
        health_levels, engines, units = [], [], []
        for _, batch in enumerate(self.test_loader):
            x, meta = self._device_to(batch)
            with torch.no_grad():
                # x_hat, mu, logvar = self.best_model(x)
                x_hat, mu, logvar = self.model(x)

                elbo, recon, kl = self.compute_loss_per_sample(x_hat, x, mu, logvar)
                elbos.append(elbo)
                recons.append(recon)
                kls.append(kl)

                # Engine-unit pairs
                health_levels.append(
                    self.get_meta_feature(meta, "health_level").cpu().numpy()
                )
                engines.append(self.get_meta_feature(meta, "engine").cpu().numpy())
                units.append(self.get_meta_feature(meta, "unit").cpu().numpy())

        elbos = np.concatenate(elbos)
        recons = np.concatenate(recons)
        kls = np.concatenate(kls)
        self.writer.add_scalar("test/loss_elbo", elbos.mean())
        self.writer.add_scalar("test/loss_recon", recons.mean())
        self.writer.add_scalar("test/loss_kl", kls.mean())

        # Performance logging warmup (engine-unit pairing)
        health_levels = np.concatenate(health_levels)
        engines = np.concatenate(engines)
        units = np.concatenate(units)
        anomalies = (health_levels == 2).astype(int)

        # Performance logging
        # ELBO
        roc_auc_elbo = roc_auc_score(anomalies, elbos)
        pred_levels = self._predict(elbos, engines, units)
        f1_elbo = f1_score(anomalies, pred_levels)
        self.writer.add_scalar("test/roc_auc_elbo", roc_auc_elbo)
        self.writer.add_scalar("test/f1_elbo", f1_elbo)
        # Recon
        roc_auc_recon = roc_auc_score(anomalies, recons)
        pred_levels = self._predict(recons, engines, units)
        f1_recon = f1_score(anomalies, pred_levels)
        self.writer.add_scalar("test/roc_auc_recon", roc_auc_recon)
        self.writer.add_scalar("test/f1_recon", f1_recon)
        # KL
        roc_auc_kl = roc_auc_score(anomalies, kls)
        pred_levels = self._predict(kls, engines, units)
        f1_kl = f1_score(anomalies, pred_levels)
        self.writer.add_scalar("test/roc_auc_kl", roc_auc_kl)
        self.writer.add_scalar("test/f1_kl", f1_kl)

        self.writer.close()

        # Plot
        plot_loss_results(self.log_dir)
        plot_performance(self.log_dir)
        cm = confusion_matrix(anomalies, pred_levels)
        plot_confusion_matrix(cm, False, self.log_dir)
        plot_confusion_matrix(cm, True, self.log_dir)
        plot_latent_3d_all(
            self.best_model,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.device,
            f"{self.log_dir}/latent_3d.png",
            self.best_beta,
        )
        plot_degradation(
            self.best_model,
            self.test_loader,
            self.device,
            save_path=f"{self.log_dir}/degradation.png",
            beta=self.best_beta,
        )

    def _predict(
        self, kls: np.ndarray, engines: np.ndarray, units: np.ndarray
    ) -> np.ndarray:
        pred_levels = np.zeros(len(kls), dtype=int)
        for (_, _), idx in self._group_by_engine_unit(engines, units):
            # threshold = kls.mean() + self.gamma * kls.std()
            # threshold = np.percentile(kls[idx], 70)
            pred_levels[idx] = (kls[idx] > self.threshold).astype(int)
        return pred_levels

    def get_meta_feature(self, meta: torch.Tensor, feature: str, window_idx: int = -1):
        idx = self.feature_map[feature]
        return meta[:, window_idx, idx]

    def _group_by_engine_unit(self, engines, units):
        engines = np.array(engines)
        units = np.array(units)
        pairs = np.stack([engines, units], axis=1)
        pairs = np.unique(pairs, axis=0)
        groups = []
        for engine, unit in pairs:
            idx = np.where((engines == engine) & (units == unit))[0]
            groups.append(((engine, unit), idx))
        return groups

    def _update_beta(self, epoch):
        return self.beta_max
        # return min(self.beta_max, self.beta_max * epoch / self.warmup)
        # return self.beta_schedule[epoch - 1]

    def _device_to(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, meta = batch
        return x.to(self.device), meta.to(self.device)


def get_health_level_from_hi(health_idx: torch.Tensor):
    y = np.ones_like(health_idx)
    y[health_idx > 0.6] = 0
    y[health_idx < 0.25] = 2
    return y


def get_health_level(rul: torch.Tensor) -> np.ndarray:
    y = np.ones_like(rul)
    y[rul > 120] = 0
    y[rul < 30] = 2
    return y
