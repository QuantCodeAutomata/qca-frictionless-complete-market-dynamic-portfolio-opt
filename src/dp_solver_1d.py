"""
One-dimensional backward dynamic-programming solver for the terminal
CVaR-constrained portfolio problem with direct risky-dollar-exposure control.

Covers exp1 (complete market, beta_perp=0) and exp2 (incomplete market,
beta_perp>0). The state is wealth W only; the nontraded shock (beta_perp dB^perp)
is an exogenous noise source that enters the wealth variance but not the state
dimension of the value function, because the investor cannot control or observe
it separately.

# Custom — Context7 found no library equivalent for this specific
#           CVaR-constrained continuous-time dynamic program (paper Section 5).
"""
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple

from src.config import ModelParams, GridParams


def make_gauss_hermite(n_quad: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return abscissas and weights for N(0,1) Gaussian quadrature
    derived from Gauss-Hermite with the standard probabilistic scaling.

    x_k = xi_k * sqrt(2),  w_k = wi_k / sqrt(pi)

    so that sum_k w_k * f(x_k) ≈ E[f(Z)], Z ~ N(0,1).
    """
    xi, wi = np.polynomial.hermite.hermgauss(n_quad)
    eps = xi * np.sqrt(2.0)
    weights = wi / np.sqrt(np.pi)
    return eps, weights


def terminal_value(w_grid: np.ndarray,
                   lambda_val: float,
                   eta: float,
                   alpha: float) -> np.ndarray:
    """
    Compute V(T, w) = Psi_{lambda, eta}(w) = lambda/(1-alpha) * max(-w - eta, 0).

    Parameters
    ----------
    w_grid     : wealth grid (n_w,)
    lambda_val : Lagrange multiplier (>= 0)
    eta        : auxiliary CVaR threshold
    alpha      : CVaR confidence level
    """
    return (lambda_val / (1.0 - alpha)) * np.maximum(-w_grid - eta, 0.0)


def backward_dp_1d(
    model: ModelParams,
    grid: GridParams,
    lambda_val: float,
    eta: float,
    beta_perp: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run the backward dynamic program for the 1D wealth state.

    The Bellman recursion (Euler-Maruyama discretisation with dt = T/n_time):
      V(t, w) = min_{phi in phi_grid} {
          [ gamma/2 * (sigma*phi)^2 + gamma/2 * beta_perp^2 - mu*phi ] * dt
          + E_eps[ V(t+dt, w + mu*phi*dt + sqrt((sigma*phi)^2 + beta_perp^2)*sqrt(dt)*eps) ]
      }
    with terminal condition V(T, w) = Psi_{lambda, eta}(w).

    Vectorised over the full (n_w, n_phi) grid at every time step.

    Parameters
    ----------
    model      : model parameters
    grid       : discretisation parameters
    lambda_val : Lagrange multiplier (outer bisection variable)
    eta        : auxiliary threshold (inner golden-section variable)
    beta_perp  : nontraded endowment volatility (0 for exp1, 0.02 for exp2)

    Returns
    -------
    values   : value function array (n_time+1, n_w)
    policies : optimal control array (n_time, n_w) — risky exposure phi
    v0       : V(0, w0) scalar
    """
    dt = model.T / grid.n_time
    sqrt_dt = np.sqrt(dt)
    mu, sigma, gamma = model.mu, model.sigma, model.gamma
    alpha = model.alpha

    # Grids
    w_grid = np.linspace(grid.w_min, grid.w_max, grid.n_w)
    phi_grid = np.linspace(grid.phi_min, grid.phi_max, grid.n_phi)
    n_w, n_phi = len(w_grid), len(phi_grid)

    # Gauss-Hermite quadrature
    eps, weights = make_gauss_hermite(grid.n_quad)  # shapes (n_quad,)

    # --- Pre-compute drift and std arrays for all phi ---
    # drift[phi_j] = mu * phi_j * dt
    drift = mu * phi_grid * dt  # (n_phi,)
    # std[phi_j] = sqrt((sigma*phi_j)^2 + beta_perp^2) * sqrt_dt
    var_phi = (sigma * phi_grid) ** 2 + beta_perp ** 2
    std = np.sqrt(var_phi) * sqrt_dt  # (n_phi,)

    # Running cost for each phi (constant in w and t for this model)
    # gamma/2 * (sigma*phi)^2 + gamma/2 * beta_perp^2 - mu*phi, times dt
    running_cost_phi = (
        gamma / 2.0 * (sigma * phi_grid) ** 2
        + gamma / 2.0 * beta_perp ** 2
        - mu * phi_grid
    ) * dt  # (n_phi,)

    # Value function storage: shape (n_time+1, n_w)
    values = np.empty((grid.n_time + 1, n_w))
    policies = np.empty((grid.n_time, n_w))

    # Terminal condition
    values[grid.n_time] = terminal_value(w_grid, lambda_val, eta, alpha)

    # --- Backward induction ---
    for t_idx in range(grid.n_time - 1, -1, -1):
        V_next = values[t_idx + 1]  # (n_w,)

        # Build interpolator for V_next (linear, clamp extrapolation)
        # fill_value=(V_next[0], V_next[-1]) gives constant extrapolation
        interp_fn = interp1d(
            w_grid, V_next, kind="linear",
            bounds_error=False,
            fill_value=(V_next[0], V_next[-1]),
        )

        # w_next[w, phi, q] = w + drift[phi] + std[phi]*eps[q]
        # shape: (n_w, n_phi, n_quad)
        w_next = (
            w_grid[:, None, None]
            + drift[None, :, None]
            + std[None, :, None] * eps[None, None, :]
        )  # (n_w, n_phi, n_quad)

        # Interpolate V_next at all w_next values
        V_interp = interp_fn(w_next.ravel()).reshape(n_w, n_phi, grid.n_quad)

        # Expected continuation value: E[V_next(w')] over quadrature
        EV = V_interp @ weights  # (n_w, n_phi)

        # Total cost = running_cost(phi) + EV(w, phi)
        total = running_cost_phi[None, :] + EV  # (n_w, n_phi)

        # Optimal control
        best_idx = np.argmin(total, axis=1)  # (n_w,)
        values[t_idx] = total[np.arange(n_w), best_idx]
        policies[t_idx] = phi_grid[best_idx]

    # Interpolate V(0, w0)
    v0_fn = interp1d(w_grid, values[0], kind="linear",
                     bounds_error=False,
                     fill_value=(values[0, 0], values[0, -1]))
    v0 = float(v0_fn(model.w0))

    return values, policies, v0


def lagrangian_value_1d(
    model: ModelParams,
    grid: GridParams,
    lambda_val: float,
    eta: float,
    beta_perp: float = 0.0,
) -> float:
    """
    Compute the full Lagrangian objective value at (lambda, eta):

        V_lambda(eta) = V_DP(0, w0; lambda, eta) + lambda * (eta - c)

    where V_DP is the DP initial value and the second term accounts for
    the constant shift in the Lagrangian from the linear eta term.

    Note: c is NOT needed here because it is a constant offset that does not
    affect the golden-section minimiser but IS needed to interpret residuals.
    We return V_DP(0, w0) + lambda*eta; the caller adds -lambda*c if needed.

    Parameters
    ----------
    model, grid, lambda_val, eta, beta_perp : see backward_dp_1d
    """
    _, _, v0 = backward_dp_1d(model, grid, lambda_val, eta, beta_perp)
    # Full Lagrangian value (excluding the constant -lambda*c)
    return v0 + lambda_val * eta


def get_wealth_grid(grid: GridParams) -> np.ndarray:
    """Return the wealth grid used by the 1D DP."""
    return np.linspace(grid.w_min, grid.w_max, grid.n_w)


def get_phi_grid(grid: GridParams) -> np.ndarray:
    """Return the control (phi) grid used by the 1D DP."""
    return np.linspace(grid.phi_min, grid.phi_max, grid.n_phi)
