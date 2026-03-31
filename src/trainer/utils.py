import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    ScalarEvent,
)

# meta column indices
META = {
    "unit": 0,
    "cycle": 1,
    "engine": 2,
    "rul": 3,
    "health_idx": 4,
    "health_level": 5,
}


def plot_confusion_matrix(cm: np.ndarray, normalize: bool, save_dir: str):
    plt.figure(figsize=(10, 8))
    label = ""
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
        label = "norm"
    labels = ["Healthy", "Anomaly"]
    display = ConfusionMatrixDisplay(cm, display_labels=labels)
    display.plot()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{label}", dpi=300)
    plt.close()


def plot_loss_results(path: str):
    ea = EventAccumulator(path)
    ea.Reload()
    plt.rcParams.update({"font.size": 14})
    _plot_loss_results(ea, "elbo", path)
    _plot_loss_results(ea, "recon", path)
    _plot_loss_results(ea, "kl", path)


def _plot_loss_results(ea: EventAccumulator, loss_type: str, path: str):
    plt.figure(figsize=(10, 6))

    # Train set
    loss_mean = ea.Scalars(f"train/loss_{loss_type}_mean")
    loss_std = ea.Scalars(f"train/loss_{loss_type}_std")
    means, steps = _event_to_val_step(loss_mean)
    stds, _ = _event_to_val_step(loss_std)
    means, stds, steps = np.array(means), np.array(stds), np.array(steps)

    plt.plot(steps, means, label="Train Loss Mean", color="steelblue")
    plt.fill_between(
        steps,
        means - stds,
        means + stds,
        label="Train Loss Std",
        alpha=0.2,
        color="steelblue",
    )

    # Validation set
    losses = ea.Scalars(f"val/loss_{loss_type}")
    values, steps = _event_to_val_step(losses)
    np.array(steps)
    plt.plot(steps, values, label="Validation Loss", color="orange")

    loss = ea.Scalars(f"test/loss_{loss_type}")
    values, _ = _event_to_val_step(loss)
    test_loss = values[0]
    plt.axhline(
        test_loss,
        color="red",
        linestyle="--",
        label=f"Test Loss = {test_loss:.4f}",
    )

    if loss_type == "elbo":
        loss_fn_name = "ELBO"
    elif loss_type == "recon":
        loss_fn_name = "MSE"
    else:
        loss_fn_name = "KL"

    plt.legend()
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel(f"{loss_fn_name}")
    plt.tight_layout()
    plt.savefig(f"{path}/loss_{loss_type}.png", dpi=300)
    plt.close()


def plot_performance(path: str):
    ea = EventAccumulator(path)
    ea.Reload()
    _plot_performance_metric(ea, path, "roc_auc")
    _plot_performance_metric(ea, path, "f1")


def _plot_performance_metric(ea: EventAccumulator, path: str, metric: str):
    plt.figure(figsize=(10, 6))
    loss_types = ["elbo", "recon", "kl"]
    colours = ["#3498db", "#2ecc71", "#f39c12"]
    for i, loss_type in enumerate(loss_types):
        # Validation set
        perf_metric = ea.Scalars(f"val/{metric}_{loss_type}")
        values, steps = _event_to_val_step(perf_metric)

        v_m = np.array(values).mean()
        v_std = np.array(values).std()
        plt.plot(
            steps,
            values,
            label=rf"Val {loss_type.upper()} = {v_m:.2f}$\pm${v_std:.2f}",
            color=colours[i],
        )

        # Test set
        perf_metric = ea.Scalars(f"test/{metric}_{loss_type}")
        values, steps = _event_to_val_step(perf_metric)
        test_perf = values[0]
        plt.axhline(
            test_perf,
            linestyle="--",
            label=f"Test {loss_type.upper()} = {test_perf:.2f}",
            color=colours[i],
        )

    plt.legend()
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel(f"{metric.upper().replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(f"{path}/{metric}.png", dpi=300)
    plt.close()


def _event_to_val_step(events: list[ScalarEvent]):
    values = [e.value for e in events]
    steps = [e.step for e in events]
    return values, steps


def _draw_latent_3d(
    pca,
    all_mus,
    all_logvars,
    all_health_levels,
    all_scores,
    all_engines,
    loaders,
    unique_engines,
    engine_marker,
    cmap,
    norm,
    colorbar_label,
    save_path,
):
    fig = plt.figure(figsize=(24, 8))
    axes = []
    for i, (name, _) in enumerate(loaders):
        pcs = pca.transform(all_mus[i])
        eng = np.round(all_engines[i]).astype(int)
        colours = cmap(norm(all_scores[i]))

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for e in unique_engines:
            mask = eng == e
            if not mask.any():
                continue
            ax.scatter(
                pcs[mask, 0],
                pcs[mask, 1],
                pcs[mask, 2],
                c=colours[mask],
                s=2,
                alpha=0.4,
                marker=engine_marker[e],
                label=f"DS{e:02d}",
            )
        ax.set_title(name, pad=12)
        ax.set_xlabel("PC1", labelpad=8)
        ax.set_ylabel("PC2", labelpad=8)
        ax.set_zlabel("PC3", labelpad=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(
            title="Engine",
            fontsize=7,
            title_fontsize=7,
            markerscale=3,
            loc="upper left",
        )

        mu_vals = all_mus[i]
        sigma_vals = np.exp(0.5 * all_logvars[i])
        hl = np.round(all_health_levels[i]).astype(int)
        healthy_mask = hl == 0
        anomaly_mask = hl == 2

        def _fmt(mask, label):
            if not mask.any():
                return f"[{label}] n/a"
            m, s = mu_vals[mask].mean(), sigma_vals[mask].mean()
            return f"[{label}] μ={m:.3f}  σ={s:.3f}"

        stats = "\n".join(
            [_fmt(healthy_mask, "Healthy"), _fmt(anomaly_mask, "Anomaly")]
        )
        ax.text2D(
            0.02,
            0.02,
            stats,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
        )

        axes.append(ax)

    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.05, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])  # fixed strip to the right of Test
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label=colorbar_label)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_latent_3d_all(
    model,
    train_loader,
    val_loader,
    test_loader,
    device: str,
    save_path: str,
    beta,
    rul_clip: int = 200,
):
    loaders = [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]

    (
        all_mus,
        all_logvars,
        all_health_levels,
        all_ruls,
        all_kls,
        all_elbos,
        all_engines,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    model.eval()
    with torch.no_grad():
        for _, loader in loaders:
            mus, logvars, health_levels, ruls, kls, elbos, engines = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for x, meta in loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                recon = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
                elbo = (recon + beta * kl).cpu().numpy()
                mus.append(mu.cpu().numpy())
                logvars.append(logvar.cpu().numpy())
                health_levels.append(meta[:, -1, META["health_level"]].numpy())
                ruls.append(meta[:, -1, META["rul"]].numpy())
                kls.append(kl.cpu().numpy())
                elbos.append(elbo)
                engines.append(meta[:, -1, META["engine"]].numpy())
            all_mus.append(np.concatenate(mus))
            all_logvars.append(np.concatenate(logvars))
            all_health_levels.append(np.concatenate(health_levels))
            all_ruls.append(np.concatenate(ruls))
            all_kls.append(np.concatenate(kls))
            all_elbos.append(np.concatenate(elbos))
            all_engines.append(np.concatenate(engines))

    pca = PCA(n_components=3).fit(np.concatenate(all_mus))

    unique_engines = np.unique(np.round(np.concatenate(all_engines)).astype(int))
    markers = ["o", "s", "^"]
    engine_marker = {
        e: markers[idx % len(markers)] for idx, e in enumerate(unique_engines)
    }

    # --- RUL plot ---
    _draw_latent_3d(
        pca,
        all_mus,
        all_logvars,
        all_health_levels,
        [np.clip(r, 0, rul_clip) for r in all_ruls],
        all_engines,
        loaders,
        unique_engines,
        engine_marker,
        cm.RdYlGn,
        plt.Normalize(0, rul_clip),
        f"RUL (clipped at {rul_clip})",
        save_path,
    )

    # --- KL plot ---
    all_kls_flat = np.concatenate(all_kls)
    kl_clip = float(np.percentile(all_kls_flat, 95))
    _draw_latent_3d(
        pca,
        all_mus,
        all_logvars,
        all_health_levels,
        [np.clip(k, 0, kl_clip) for k in all_kls],
        all_engines,
        loaders,
        unique_engines,
        engine_marker,
        cm.YlOrRd,
        plt.Normalize(0, kl_clip),
        "KL score (95th pct clip)",
        save_path.replace(".png", "_kl.png"),
    )


def plot_degradation(
    model,
    test_loader,
    device: str,
    save_path: str,
    beta: float = 1.0,
    n_units: int = 6,
    anomaly_rul: int = 30,
):
    """
    For each of n_units randomly sampled engine-units, plot KL score over
    cycles. A vertical dashed line marks where RUL drops below anomaly_rul,
    and the region beyond is shaded red.
    """
    cycles_all, ruls_all, kls_all, engines_all, units_all = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for x, meta in test_loader:
            x = x.to(device)
            _, mu, logvar = model(x)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
            cycles_all.append(meta[:, -1, META["cycle"]].numpy())
            ruls_all.append(meta[:, -1, META["rul"]].numpy())
            kls_all.append(kl.cpu().numpy())
            engines_all.append(meta[:, -1, META["engine"]].numpy())
            units_all.append(meta[:, -1, META["unit"]].numpy())

    cycles = np.concatenate(cycles_all)
    ruls = np.concatenate(ruls_all)
    kls = np.concatenate(kls_all)
    engines = np.round(np.concatenate(engines_all)).astype(int)
    units = np.round(np.concatenate(units_all)).astype(int)

    # collect unique (engine, unit) pairs and sample n_units of them
    pairs = np.unique(np.stack([engines, units], axis=1), axis=0)
    rng = np.random.default_rng(0)
    selected = pairs[
        rng.choice(len(pairs), size=min(n_units, len(pairs)), replace=False)
    ]

    ncols = 3
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )

    for ax_idx, (eng, unit) in enumerate(selected):
        mask = (engines == eng) & (units == unit)
        cyc = cycles[mask]
        kl = kls[mask]
        rul = ruls[mask]

        order = np.argsort(cyc)
        cyc, kl, rul = cyc[order], kl[order], rul[order]

        # find the cycle where RUL first drops below the threshold
        onset_mask = rul < anomaly_rul
        onset_cycle = cyc[onset_mask][0] if onset_mask.any() else None

        ax = axes[ax_idx // ncols][ax_idx % ncols]
        ax.plot(cyc, kl, lw=1, color="steelblue", label="KL score")

        if onset_cycle is not None:
            ax.axvline(
                onset_cycle,
                color="red",
                linestyle="--",
                lw=1.2,
                label=f"RUL={anomaly_rul} (onset)",
            )
            ax.axvspan(
                onset_cycle, cyc.max(), alpha=0.12, color="red", label="Anomaly zone"
            )

        ax.set_title(f"DS{eng:02d} · Unit {int(unit)}", fontsize=10)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("KL score")
        ax.legend(fontsize=7)

    # hide unused axes
    for ax_idx in range(len(selected), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    plt.suptitle("KL score over degradation trajectory", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_or_show(save_path):
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    path = "outputs/20260328_1844"
    plot_performance(path)
