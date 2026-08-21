"""
CVaR (Conditional Value-at-Risk) estimation utilities.

Two implementations:
  1. Sample tail-average (direct Monte Carlo estimate)
  2. Rockafellar-Uryasev threshold form: CVaR_alpha(L) = inf_eta {eta + 1/(1-alpha)*E[(L-eta)^+]}

# Custom — Context7 found no library equivalent for the dual-form threshold CVaR
# consistent with the paper's Rockafellar-Uryasev reformulation used in Lagrangian.
"""
import numpy as np


def cvar_sample(losses: np.ndarray, alpha: float) -> float:
    """
    Compute CVaR_alpha(L) from a sample of losses using the tail-average estimator.

    CVaR_alpha(L) = E[L | L >= VaR_alpha(L)]

    Parameters
    ----------
    losses : array of realised loss values L = -W_T
    alpha  : confidence level (e.g. 0.95)

    Returns
    -------
    float : estimated CVaR
    """
    assert 0 < alpha < 1, "alpha must be in (0, 1)"
    n = len(losses)
    sorted_l = np.sort(losses)
    # VaR index: smallest index such that fraction of exceedances is <= 1-alpha
    k = max(int(np.ceil(alpha * n)), 1)
    return float(np.mean(sorted_l[k - 1:]))


def cvar_threshold_form(losses: np.ndarray, alpha: float, eta: float) -> float:
    """
    Compute the Rockafellar-Uryasev upper bound on CVaR at a fixed threshold eta.

    G(eta) = eta + 1/(1-alpha) * E[(L - eta)^+]

    The infimum over eta of G(eta) equals CVaR_alpha(L).

    Parameters
    ----------
    losses : array of realised loss values
    alpha  : confidence level
    eta    : auxiliary threshold (controls the tail cutoff)

    Returns
    -------
    float : G(eta) >= CVaR_alpha(L)
    """
    excess = np.maximum(losses - eta, 0.0)
    return float(eta + np.mean(excess) / (1.0 - alpha))


def cvar_optimal_eta(losses: np.ndarray, alpha: float,
                     n_grid: int = 500) -> tuple[float, float]:
    """
    Estimate CVaR_alpha(L) and its optimal threshold eta* by brute-force grid search.

    Returns
    -------
    (cvar, eta_star) : minimiser and minimised value of G(eta)
    """
    eta_grid = np.linspace(np.quantile(losses, alpha - 0.1),
                           np.quantile(losses, alpha + 0.05), n_grid)
    g_vals = np.array([cvar_threshold_form(losses, alpha, e) for e in eta_grid])
    idx = np.argmin(g_vals)
    return float(g_vals[idx]), float(eta_grid[idx])


def var_sample(losses: np.ndarray, alpha: float) -> float:
    """
    Estimate VaR_alpha(L) = alpha-quantile of L.

    Parameters
    ----------
    losses : array of realised losses
    alpha  : confidence level
    """
    return float(np.quantile(losses, alpha))


def cvar_constraint_value(losses: np.ndarray, alpha: float,
                          eta: float, c: float) -> float:
    """
    Compute the Lagrangian constraint function value:
        C(phi, eta) = G(eta) - c
                    = eta + 1/(1-alpha)*E[(L-eta)^+] - c

    Positive value means the CVaR constraint is violated.

    Parameters
    ----------
    losses : simulated terminal losses L = -W_T
    alpha  : confidence level
    eta    : auxiliary threshold
    c      : CVaR limit (negative number, e.g. -0.94)
    """
    return cvar_threshold_form(losses, alpha, eta) - c
