"""
Experiment metrics computation and reporting.

Computes and stores the summary statistics required by the paper's
Tables 2-5: lambda_hat, terminal CVaR, E[W_T], average exposure,
interim diagnostic CVaR_alpha(-W_t), and path diagnostics.

# Custom — Context7 found no library equivalent for this exact reporting
#           framework tied to the paper's table structure.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

from src.cvar_utils import cvar_sample, var_sample


@dataclass
class ExperimentMetrics:
    """All metrics for a single (experiment, c) combination."""
    exp_name: str
    c: float
    # Dual-search outputs
    lambda_hat: float
    eta_hat: float
    binding: bool
    # Terminal distribution statistics
    cvar_terminal: float       # CVaR_alpha(-W_T) from sample tail-average
    cvar_threshold: float      # CVaR estimate from threshold form (cross-check)
    E_WT: float                # Expected terminal wealth
    std_WT: float              # Std of terminal wealth
    # Control statistics
    avg_phi: float             # Time-average of cross-sectional mean exposure
    # Interim CVaR diagnostic: CVaR_alpha(-W_t) at each t from marginal distribution
    t_grid: Optional[np.ndarray] = field(default=None, repr=False)
    interim_cvar: Optional[np.ndarray] = field(default=None, repr=False)
    # Paper reference values
    paper_lambda: Optional[float] = None
    paper_cvar: Optional[float] = None
    paper_E_WT: Optional[float] = None
    paper_avg_phi: Optional[float] = None
    # Raw paths for plotting (subset)
    W_paths: Optional[np.ndarray] = field(default=None, repr=False)
    phi_paths: Optional[np.ndarray] = field(default=None, repr=False)
    u_paths: Optional[np.ndarray] = field(default=None, repr=False)


def compute_metrics_1d(
    exp_name: str,
    c: float,
    lambda_hat: float,
    eta_hat: float,
    binding: bool,
    W_paths: np.ndarray,         # (n_paths, n_time+1)
    phi_paths: np.ndarray,       # (n_paths, n_time)
    t_grid: np.ndarray,          # (n_time+1,)
    alpha: float,
    paper_ref: Optional[dict] = None,
) -> ExperimentMetrics:
    """
    Compute all metrics for a 1D (complete or incomplete market) experiment result.

    Parameters
    ----------
    exp_name   : experiment identifier string
    c          : CVaR limit
    lambda_hat : Lagrange multiplier at convergence
    eta_hat    : optimal threshold at convergence
    binding    : True if CVaR constraint is active
    W_paths    : simulated wealth paths  (n_paths, n_time+1)
    phi_paths  : simulated exposure paths (n_paths, n_time)
    t_grid     : time grid
    alpha      : CVaR confidence level
    paper_ref  : dict with paper reference values (optional)

    Returns
    -------
    ExperimentMetrics
    """
    W_T = W_paths[:, -1]
    losses = -W_T

    # Terminal CVaR (sample tail-average)
    cvar_t = cvar_sample(losses, alpha)

    # CVaR threshold-form cross-check at eta_hat
    excess = np.maximum(losses - eta_hat, 0.0)
    cvar_thresh = float(eta_hat + np.mean(excess) / (1.0 - alpha))

    # Expected terminal wealth
    E_WT = float(np.mean(W_T))
    std_WT = float(np.std(W_T))

    # Average risky exposure: time-average of cross-sectional mean
    mean_phi_t = np.mean(phi_paths, axis=0)  # (n_time,)
    dt = t_grid[1] - t_grid[0]
    avg_phi = float(np.mean(mean_phi_t))      # uniform time average

    # Interim diagnostic CVaR_alpha(-W_t)
    interim_cvar = np.array([
        cvar_sample(-W_paths[:, t_idx], alpha) for t_idx in range(W_paths.shape[1])
    ])

    paper_ref = paper_ref or {}
    return ExperimentMetrics(
        exp_name=exp_name, c=c,
        lambda_hat=lambda_hat, eta_hat=eta_hat, binding=binding,
        cvar_terminal=cvar_t, cvar_threshold=cvar_thresh,
        E_WT=E_WT, std_WT=std_WT,
        avg_phi=avg_phi,
        t_grid=t_grid, interim_cvar=interim_cvar,
        paper_lambda=paper_ref.get("lambda"),
        paper_cvar=paper_ref.get("cvar"),
        paper_E_WT=paper_ref.get("E_WT"),
        paper_avg_phi=paper_ref.get("avg_phi"),
        W_paths=W_paths,
        phi_paths=phi_paths,
    )


def compute_metrics_2d(
    exp_name: str,
    c: float,
    lambda_hat: float,
    eta_hat: float,
    binding: bool,
    W_paths: np.ndarray,         # (n_paths, n_time+1)
    phi_paths: np.ndarray,       # (n_paths, n_time+1)
    u_paths: np.ndarray,         # (n_paths, n_time)
    t_grid: np.ndarray,
    alpha: float,
    paper_ref: Optional[dict] = None,
) -> ExperimentMetrics:
    """
    Compute all metrics for a 2D (frictional) experiment result.
    """
    W_T = W_paths[:, -1]
    losses = -W_T

    cvar_t = cvar_sample(losses, alpha)
    excess = np.maximum(losses - eta_hat, 0.0)
    cvar_thresh = float(eta_hat + np.mean(excess) / (1.0 - alpha))

    E_WT = float(np.mean(W_T))
    std_WT = float(np.std(W_T))

    # Average exposure (excluding final endpoint, use exposure during [0,T))
    mean_phi_t = np.mean(phi_paths[:, :-1], axis=0)  # (n_time,)
    avg_phi = float(np.mean(mean_phi_t))

    interim_cvar = np.array([
        cvar_sample(-W_paths[:, t_idx], alpha) for t_idx in range(W_paths.shape[1])
    ])

    paper_ref = paper_ref or {}
    return ExperimentMetrics(
        exp_name=exp_name, c=c,
        lambda_hat=lambda_hat, eta_hat=eta_hat, binding=binding,
        cvar_terminal=cvar_t, cvar_threshold=cvar_thresh,
        E_WT=E_WT, std_WT=std_WT,
        avg_phi=avg_phi,
        t_grid=t_grid, interim_cvar=interim_cvar,
        paper_lambda=paper_ref.get("lambda"),
        paper_cvar=paper_ref.get("cvar"),
        paper_E_WT=paper_ref.get("E_WT"),
        paper_avg_phi=paper_ref.get("avg_phi"),
        W_paths=W_paths,
        phi_paths=phi_paths,
        u_paths=u_paths,
    )


def metrics_to_dataframe(metrics_list: List[ExperimentMetrics]) -> pd.DataFrame:
    """
    Convert a list of ExperimentMetrics into a summary DataFrame.

    Returns a DataFrame with columns for paper reference values alongside
    numerical estimates, suitable for Table 2-5 reproduction.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Experiment": m.exp_name,
            "c": m.c,
            "Binding": m.binding,
            "lambda_hat": round(m.lambda_hat, 6),
            "eta_hat": round(m.eta_hat, 6),
            "CVaR_terminal": round(m.cvar_terminal, 4),
            "CVaR_threshold": round(m.cvar_threshold, 4),
            "E[W_T]": round(m.E_WT, 4),
            "std[W_T]": round(m.std_WT, 4),
            "avg_phi": round(m.avg_phi, 4),
            "Paper_lambda": m.paper_lambda,
            "Paper_CVaR": m.paper_cvar,
            "Paper_E[W_T]": m.paper_E_WT,
            "Paper_avg_phi": m.paper_avg_phi,
        })
    return pd.DataFrame(rows)


def select_path_indices(
    W_paths: np.ndarray,
    n_favorable: int = 3,
    n_unfavorable: int = 3,
    rule: str = "terminal_wealth",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select representative favorable and unfavorable path indices.

    Rule: "terminal_wealth" — rank paths by W_T and select the top and bottom n.
    This is a deterministic rule documented per the paper's requirement.

    Parameters
    ----------
    W_paths       : wealth paths (n_paths, n_time+1)
    n_favorable   : number of favorable paths to select
    n_unfavorable : number of unfavorable paths to select
    rule          : selection rule (currently only "terminal_wealth")

    Returns
    -------
    (favorable_indices, unfavorable_indices)
    """
    W_T = W_paths[:, -1]
    sorted_idx = np.argsort(W_T)
    unfavorable = sorted_idx[:n_unfavorable]
    favorable = sorted_idx[-n_favorable:]
    return favorable, unfavorable
