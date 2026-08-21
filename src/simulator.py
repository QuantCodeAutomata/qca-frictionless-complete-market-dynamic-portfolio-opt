"""
Forward Monte Carlo simulation of Markov feedback policies.

Supports:
  simulate_1d : 1D policy (direct risky exposure phi) — exp1, exp2
  simulate_2d : 2D policy (trading rate u) — exp3, exp4

For each experiment, the simulator:
  1. Generates Brownian paths on a fine time grid.
  2. Looks up the optimal control at each (t, state) using the stored policy table
     (with linear interpolation in the state variables).
  3. Evolves wealth (and exposure for 2D) forward.

# Custom — Context7 found no library equivalent for this specific feedback-policy
#           forward simulation used in the CVaR dual-search evaluation.
"""
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from typing import Tuple, Optional

from src.config import ModelParams, GridParams, MCParams


# ---------------------------------------------------------------------------
# 1D simulator (exp1, exp2)
# ---------------------------------------------------------------------------

def simulate_1d(
    model: ModelParams,
    grid: GridParams,
    policies: np.ndarray,          # shape (n_time, n_w)
    mc: MCParams,
    n_paths: int,
    beta_perp: float = 0.0,
    rng: Optional = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-simulate the 1D feedback policy.

    Parameters
    ----------
    model    : model parameters
    grid     : grid parameters (used for w_grid and time grid)
    policies : optimal phi(t, w) array from backward DP  (n_time, n_w)
    mc       : Monte Carlo parameters
    n_paths  : number of simulation paths
    beta_perp: nontraded volatility
    rng      : numpy Generator (created from mc.seed if None)

    Returns
    -------
    W_paths  : wealth paths (n_paths, n_time+1)
    phi_paths: control paths (n_paths, n_time)
    t_grid   : time grid (n_time+1,)
    """
    if rng is None:
        rng = np.random.default_rng(mc.seed)

    dt = model.T / grid.n_time
    sqrt_dt = np.sqrt(dt)
    n_time = grid.n_time

    w_grid = np.linspace(grid.w_min, grid.w_max, grid.n_w)
    t_grid = np.linspace(0.0, model.T, n_time + 1)

    # Policy interpolators (one per time step)
    phi_interps = [
        interp1d(w_grid, policies[t_idx],
                 kind="linear", bounds_error=False,
                 fill_value=(policies[t_idx, 0], policies[t_idx, -1]))
        for t_idx in range(n_time)
    ]

    # Brownian increments
    dB = rng.standard_normal((n_paths, n_time)) * sqrt_dt           # traded BM
    if beta_perp > 0:
        dBperp = rng.standard_normal((n_paths, n_time)) * sqrt_dt   # nontraded BM
    else:
        dBperp = np.zeros((n_paths, n_time))

    W = np.empty((n_paths, n_time + 1))
    phi_path = np.empty((n_paths, n_time))
    W[:, 0] = model.w0

    for t_idx in range(n_time):
        w_curr = W[:, t_idx]
        phi = phi_interps[t_idx](w_curr)  # (n_paths,)
        phi_path[:, t_idx] = phi

        # Wealth update
        dW = (
            model.mu * phi * dt
            + model.sigma * phi * dB[:, t_idx]
            + beta_perp * dBperp[:, t_idx]
        )
        W[:, t_idx + 1] = w_curr + dW

    return W, phi_path, t_grid


# ---------------------------------------------------------------------------
# 2D simulator (exp3, exp4)
# ---------------------------------------------------------------------------

def simulate_2d(
    model: ModelParams,
    grid: GridParams,
    policies: np.ndarray,           # shape (n_time, n_w, n_phi_state)
    mc: MCParams,
    n_paths: int,
    phi0: float = 0.0,
    mode: str = "quadratic",        # "quadratic" or "sqrt_impact"
    Lambda: float = 0.01,
    Lambda_32: float = 0.0045,
    rng: Optional = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-simulate the 2D feedback policy (W, phi state).

    Parameters
    ----------
    model    : model parameters
    grid     : grid parameters
    policies : optimal u(t, w, phi) array from backward DP  (n_time, n_w, n_phi_state)
    mc       : Monte Carlo parameters
    n_paths  : number of simulation paths
    phi0     : initial risky exposure
    mode     : "quadratic" or "sqrt_impact"
    Lambda   : quadratic regularisation coefficient (exp3)
    Lambda_32: square-root impact coefficient (exp4)
    rng      : numpy Generator

    Returns
    -------
    W_paths  : wealth paths (n_paths, n_time+1)
    phi_paths: exposure paths (n_paths, n_time+1)
    u_paths  : trading-rate paths (n_paths, n_time)
    t_grid   : time grid (n_time+1,)
    """
    if rng is None:
        rng = np.random.default_rng(mc.seed)

    dt = model.T / grid.n_time
    sqrt_dt = np.sqrt(dt)
    n_time = grid.n_time

    w_grid = np.linspace(grid.w_min, grid.w_max, grid.n_w)
    phi_s_grid = np.linspace(grid.phi_state_min, grid.phi_state_max, grid.n_phi_state)
    t_grid = np.linspace(0.0, model.T, n_time + 1)

    # Policy interpolators (one per time step, 2D in state)
    u_interps = [
        RegularGridInterpolator(
            (w_grid, phi_s_grid), policies[t_idx],
            method="linear", bounds_error=False, fill_value=None,
        )
        for t_idx in range(n_time)
    ]

    # Brownian increments
    dB = rng.standard_normal((n_paths, n_time)) * sqrt_dt

    W = np.empty((n_paths, n_time + 1))
    Phi = np.empty((n_paths, n_time + 1))
    U = np.empty((n_paths, n_time))
    W[:, 0] = model.w0
    Phi[:, 0] = phi0

    for t_idx in range(n_time):
        w_curr = W[:, t_idx]
        phi_curr = Phi[:, t_idx]

        # Clamp state to grid bounds for interpolation
        w_c = np.clip(w_curr, w_grid[0], w_grid[-1])
        phi_c = np.clip(phi_curr, phi_s_grid[0], phi_s_grid[-1])

        pts = np.stack([w_c, phi_c], axis=1)       # (n_paths, 2)
        u = u_interps[t_idx](pts)                  # (n_paths,)
        U[:, t_idx] = u

        db = dB[:, t_idx]

        # Exposure update: d phi = u*dt + phi*(mu*dt + sigma*dB)
        Phi[:, t_idx + 1] = (
            phi_curr * (1.0 + model.mu * dt + model.sigma * db)
            + u * dt
        )

        # Wealth update
        if mode == "quadratic":
            dW = model.mu * phi_curr * dt + model.sigma * phi_curr * db
        else:  # sqrt_impact — impact deducted from wealth
            impact = (Lambda_32 / 1.5) * np.abs(u) ** 1.5 * dt
            dW = model.mu * phi_curr * dt + model.sigma * phi_curr * db - impact

        W[:, t_idx + 1] = w_curr + dW

    return W, Phi, U, t_grid


# ---------------------------------------------------------------------------
# Residual oracle helpers
# ---------------------------------------------------------------------------

def residual_from_simulation_1d(
    model: ModelParams,
    grid: GridParams,
    policies: np.ndarray,
    eta: float,
    c: float,
    mc: MCParams,
    n_paths: int,
    beta_perp: float = 0.0,
    rng: Optional = None,
) -> float:
    """
    Compute the CVaR constraint residual R = CVaR_alpha(-W_T; policy) - c.

    Uses the sample tail-average estimator, which equals min_eta G(eta; phi)
    (Rockafellar-Uryasev theorem), and is always correct regardless of the
    value of `eta`.  This is what the paper requires at lambda=0
    ("Compute R(0) = CVaR_alpha(-W_T^phi_0) - c") and remains consistent for
    all lambda in the bisection.

    Parameters
    ----------
    eta : kept for API compatibility; not used in the CVaR computation.
    """
    from src.cvar_utils import cvar_sample
    W, _, _ = simulate_1d(model, grid, policies, mc, n_paths, beta_perp, rng)
    losses = -W[:, -1]   # L_T = -W_T
    return float(cvar_sample(losses, model.alpha) - c)


def residual_from_simulation_2d(
    model: ModelParams,
    grid: GridParams,
    policies: np.ndarray,
    eta: float,
    c: float,
    mc: MCParams,
    n_paths: int,
    phi0: float = 0.0,
    mode: str = "quadratic",
    Lambda: float = 0.01,
    Lambda_32: float = 0.0045,
    rng: Optional = None,
) -> float:
    """
    Compute the CVaR constraint residual for the 2D policy.
    Uses sample tail-average (equivalent to min_eta G(eta)) for correctness.
    """
    from src.cvar_utils import cvar_sample
    W, _, _, _ = simulate_2d(
        model, grid, policies, mc, n_paths, phi0, mode, Lambda, Lambda_32, rng
    )
    losses = -W[:, -1]
    return float(cvar_sample(losses, model.alpha) - c)



