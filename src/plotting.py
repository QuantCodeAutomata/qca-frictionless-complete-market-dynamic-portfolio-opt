"""
Visualization module for CVaR-constrained dynamic portfolio optimization experiments.

Generates all required figures:
  1. Mean exposure path with 10-90% bands
  2. Mean trading-rate path with 10-90% bands (exp3, exp4 only)
  3. Terminal wealth distribution
  4. Interim diagnostic CVaR_alpha(-W_t) over time
  5. Representative favorable/unfavorable sample paths

Uses matplotlib with a clean, publication-quality style.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, List

from src.metrics import ExperimentMetrics, select_path_indices


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.6,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


def plot_exposure_paths(
    t_grid: np.ndarray,
    phi_paths: np.ndarray,    # (n_paths, n_time) or (n_paths, n_time+1)
    c: float,
    exp_name: str,
    save_path: Optional[Path] = None,
    merton_phi: Optional[float] = None,
) -> plt.Figure:
    """
    Plot mean exposure with 10-90% percentile band.

    Parameters
    ----------
    t_grid    : time grid (n_time+1,) — uses t_grid[:-1] or t_grid as appropriate
    phi_paths : exposure paths
    c         : CVaR limit (for title)
    exp_name  : experiment name (for title)
    save_path : optional file path to save figure
    merton_phi: optional Merton benchmark line
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    n_t = phi_paths.shape[1]
    t = t_grid[:n_t]

    mean_phi = np.mean(phi_paths, axis=0)
    p10 = np.percentile(phi_paths, 10, axis=0)
    p90 = np.percentile(phi_paths, 90, axis=0)

    ax.fill_between(t, p10, p90, alpha=0.2, color="steelblue", label="10–90% band")
    ax.plot(t, mean_phi, color="steelblue", label="Mean exposure")

    if merton_phi is not None:
        ax.axhline(merton_phi, linestyle="--", color="gray", alpha=0.7,
                   label=f"Merton benchmark φ*={merton_phi:.2f}")

    ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Risky exposure φ_t  ($)")
    ax.set_title(f"{exp_name}  |  CVaR limit c = {c}  |  Exposure paths")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_trading_rate_paths(
    t_grid: np.ndarray,
    u_paths: np.ndarray,      # (n_paths, n_time)
    c: float,
    exp_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot mean trading-rate path with 10-90% bands (exp3, exp4)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    t = t_grid[:u_paths.shape[1]]
    mean_u = np.mean(u_paths, axis=0)
    p10 = np.percentile(u_paths, 10, axis=0)
    p90 = np.percentile(u_paths, 90, axis=0)

    ax.fill_between(t, p10, p90, alpha=0.2, color="darkorange", label="10–90% band")
    ax.plot(t, mean_u, color="darkorange", label="Mean trading rate")
    ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Trading rate u_t  ($/yr)")
    ax.set_title(f"{exp_name}  |  CVaR limit c = {c}  |  Trading-rate paths")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_terminal_wealth_dist(
    W_T: np.ndarray,
    c: float,
    exp_name: str,
    alpha: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot histogram of terminal wealth W_T."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(W_T, bins=80, color="steelblue", alpha=0.75, density=True,
            edgecolor="white", linewidth=0.3)

    # Mark VaR and CVaR
    from src.cvar_utils import var_sample, cvar_sample
    var = var_sample(-W_T, alpha)
    ax.axvline(-var, linestyle="--", color="red", label=f"VaR_{alpha:.0%}")
    cvar = cvar_sample(-W_T, alpha)
    ax.axvline(-cvar, linestyle=":", color="darkred", label=f"CVaR_{alpha:.0%}={cvar:.4f}")
    ax.axvline(1.0, linestyle="-", color="gray", linewidth=0.8, alpha=0.6, label="w₀=1")

    ax.set_xlabel("Terminal wealth W_T")
    ax.set_ylabel("Density")
    ax.set_title(f"{exp_name}  |  CVaR limit c = {c}  |  Terminal wealth distribution")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_interim_cvar(
    t_grid: np.ndarray,
    interim_cvar: np.ndarray,
    c: float,
    exp_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot CVaR_alpha(-W_t) over time (interim diagnostic)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(t_grid, interim_cvar, color="steelblue", label="CVaR_α(−W_t)")
    ax.axhline(c, linestyle="--", color="red", alpha=0.8, label=f"CVaR limit c = {c}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("CVaR_α(−W_t)")
    ax.set_title(f"{exp_name}  |  CVaR limit c = {c}  |  Interim CVaR diagnostic")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_sample_paths(
    t_grid: np.ndarray,
    W_paths: np.ndarray,
    phi_paths: np.ndarray,
    c: float,
    exp_name: str,
    save_path: Optional[Path] = None,
    n_favorable: int = 3,
    n_unfavorable: int = 3,
) -> plt.Figure:
    """
    Plot selected favorable and unfavorable wealth and exposure sample paths.
    Selection rule: top/bottom n paths by terminal wealth W_T.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    fav_idx, unf_idx = select_path_indices(W_paths, n_favorable, n_unfavorable)

    n_t_phi = phi_paths.shape[1]
    t_phi = t_grid[:n_t_phi]

    ax = axes[0]
    for idx in fav_idx:
        ax.plot(t_grid, W_paths[idx], color="green", alpha=0.8, linewidth=1.2)
    for idx in unf_idx:
        ax.plot(t_grid, W_paths[idx], color="red", alpha=0.8, linewidth=1.2)
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Wealth W_t")
    ax.set_title("Sample wealth paths\n(green=favorable, red=unfavorable)")

    ax = axes[1]
    for idx in fav_idx:
        ax.plot(t_phi, phi_paths[idx], color="green", alpha=0.8, linewidth=1.2)
    for idx in unf_idx:
        ax.plot(t_phi, phi_paths[idx], color="red", alpha=0.8, linewidth=1.2)
    ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Exposure φ_t")
    ax.set_title("Sample exposure paths")

    fig.suptitle(f"{exp_name}  |  CVaR limit c = {c}", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_all_for_metric(
    m: ExperimentMetrics,
    alpha: float,
    results_dir: Path,
    merton_phi: Optional[float] = None,
    is_2d: bool = False,
) -> None:
    """
    Generate and save all required figures for a single ExperimentMetrics object.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{m.exp_name}_c{abs(m.c):.2f}".replace(".", "p")

    phi_paths = m.phi_paths
    if phi_paths is None:
        return

    # --- Exposure paths ---
    plot_exposure_paths(
        m.t_grid, phi_paths, m.c, m.exp_name,
        save_path=results_dir / f"{tag}_exposure.png",
        merton_phi=merton_phi,
    )

    # --- Trading-rate paths (2D only) ---
    if is_2d and m.u_paths is not None:
        plot_trading_rate_paths(
            m.t_grid, m.u_paths, m.c, m.exp_name,
            save_path=results_dir / f"{tag}_trading_rate.png",
        )

    # --- Terminal wealth distribution ---
    if m.W_paths is not None:
        plot_terminal_wealth_dist(
            m.W_paths[:, -1], m.c, m.exp_name, alpha,
            save_path=results_dir / f"{tag}_terminal_dist.png",
        )

    # --- Interim CVaR diagnostic ---
    if m.interim_cvar is not None and m.t_grid is not None:
        plot_interim_cvar(
            m.t_grid, m.interim_cvar, m.c, m.exp_name,
            save_path=results_dir / f"{tag}_interim_cvar.png",
        )

    # --- Sample paths ---
    if m.W_paths is not None and phi_paths is not None:
        plot_sample_paths(
            m.t_grid, m.W_paths, phi_paths, m.c, m.exp_name,
            save_path=results_dir / f"{tag}_sample_paths.png",
        )
