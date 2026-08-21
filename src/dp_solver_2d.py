"""
Two-dimensional backward dynamic-programming solver for frictional portfolio problems.

State: X_t = (W_t, phi_t) — wealth and current risky-dollar exposure.
Control: u_t = dot(phi_t) — signed dollar trading rate.

Covers:
  exp3 – Quadratic trading-rate regularisation (running cost += Lambda/2 * u^2)
  exp4 – Square-root price impact      (running cost += Lambda_32/1.5 * |u|^{1.5},
                                         wealth SDE also contains impact term)

Both state variables share the SAME Brownian driver B_t (correlated transitions).

# Custom — Context7 found no library equivalent for this 2-D CVaR-constrained
#           stochastic control problem with nested dual search (paper Section 6).
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional

from src.config import ModelParams, GridParams
from src.dp_solver_1d import make_gauss_hermite, terminal_value


# ---------------------------------------------------------------------------
# Running-cost helpers
# ---------------------------------------------------------------------------

def running_cost_quadratic(
    phi_grid: np.ndarray,
    u_grid: np.ndarray,
    mu: float, sigma: float, gamma: float,
    Lambda: float, dt: float,
) -> np.ndarray:
    """
    Running cost for quadratic trading-rate regularisation (exp3):
        c(phi, u) = [ gamma/2 * sigma^2 * phi^2  + Lambda/2 * u^2  - mu*phi ] * dt

    Returns array of shape (n_phi_state, n_u).
    """
    phi_c = (gamma / 2.0 * sigma ** 2 * phi_grid ** 2 - mu * phi_grid)  # (n_phi,)
    u_c = Lambda / 2.0 * u_grid ** 2                                      # (n_u,)
    return (phi_c[:, None] + u_c[None, :]) * dt  # (n_phi, n_u)


def running_cost_sqrt_impact(
    phi_grid: np.ndarray,
    u_grid: np.ndarray,
    mu: float, sigma: float, gamma: float,
    Lambda_32: float, dt: float,
) -> np.ndarray:
    """
    Running cost for square-root price impact (exp4):
        c(phi, u) = [ gamma/2 * sigma^2 * phi^2  + Lambda_32/1.5 * |u|^{1.5}  - mu*phi ] * dt

    Returns array of shape (n_phi_state, n_u).
    """
    phi_c = (gamma / 2.0 * sigma ** 2 * phi_grid ** 2 - mu * phi_grid)
    u_c = (Lambda_32 / 1.5) * np.abs(u_grid) ** 1.5
    return (phi_c[:, None] + u_c[None, :]) * dt


def wealth_impact_cost(
    u_grid: np.ndarray,
    Lambda_32: float, dt: float,
) -> np.ndarray:
    """
    Impact deduction from wealth per time step (exp4 only):
        delta_W_impact(u) = - Lambda_32/1.5 * |u|^{1.5} * dt

    Returns array of shape (n_u,).
    """
    return -(Lambda_32 / 1.5) * np.abs(u_grid) ** 1.5 * dt


# ---------------------------------------------------------------------------
# Core 2D backward DP
# ---------------------------------------------------------------------------

def backward_dp_2d(
    model: ModelParams,
    grid: GridParams,
    lambda_val: float,
    eta: float,
    mode: str = "quadratic",          # "quadratic" or "sqrt_impact"
    Lambda: float = 0.01,             # quadratic regularisation coefficient
    Lambda_32: float = 0.0045,        # square-root impact coefficient
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run the 2D backward dynamic program over state (W, phi) with control u.

    State transitions (one Euler-Maruyama step, shared Brownian increment eps ~ N(0,1)):
        phi' = phi * (1 + mu*dt + sigma*sqrt_dt*eps) + u*dt
        W'   = W + mu*phi*dt + sigma*phi*sqrt_dt*eps            [exp3]
        W'   = W + mu*phi*dt + sigma*phi*sqrt_dt*eps
               - (Lambda_32/1.5)*|u|^{1.5}*dt                  [exp4]

    Terminal condition: V(T, W, phi) = Psi_{lambda,eta}(W) = lambda/(1-alpha)*max(-W-eta,0)
    (does NOT depend on phi at maturity).

    Parameters
    ----------
    model      : model parameters
    grid       : discretisation parameters (uses 2D grid fields)
    lambda_val : Lagrange multiplier
    eta        : auxiliary threshold
    mode       : "quadratic" (exp3) or "sqrt_impact" (exp4)
    Lambda     : quadratic regularisation coefficient (exp3 only)
    Lambda_32  : square-root impact coefficient (exp4 only)

    Returns
    -------
    values   : (n_time+1, n_w, n_phi_state) array — value function
    policies : (n_time, n_w, n_phi_state)   array — optimal u
    v0_interp: V(0, w0, phi0) scalar (for Lagrangian objective)
    """
    dt = model.T / grid.n_time
    sqrt_dt = np.sqrt(dt)
    mu, sigma, gamma = model.mu, model.sigma, model.gamma
    alpha = model.alpha

    # State grids
    w_grid = np.linspace(grid.w_min, grid.w_max, grid.n_w)
    phi_s_grid = np.linspace(grid.phi_state_min, grid.phi_state_max, grid.n_phi_state)
    u_grid = np.linspace(grid.u_min, grid.u_max, grid.n_u)
    n_w, n_phi_s, n_u = len(w_grid), len(phi_s_grid), len(u_grid)

    # Gauss-Hermite quadrature
    eps_q, wts = make_gauss_hermite(grid.n_quad)  # (n_quad,)
    n_quad = len(eps_q)

    # Pre-compute running costs: shape (n_phi_s, n_u)
    if mode == "quadratic":
        rc = running_cost_quadratic(phi_s_grid, u_grid, mu, sigma, gamma, Lambda, dt)
        impact_on_w = np.zeros(n_u)  # no direct impact on wealth in exp3
    else:  # sqrt_impact
        rc = running_cost_sqrt_impact(phi_s_grid, u_grid, mu, sigma, gamma, Lambda_32, dt)
        impact_on_w = wealth_impact_cost(u_grid, Lambda_32, dt)  # (n_u,)

    # Phi state transitions (independent of wealth):
    # phi'[phi_s, u, q] = phi_s*(1 + mu*dt + sigma*sqrt_dt*eps_q) + u*dt
    phi_next = (
        phi_s_grid[:, None, None] * (1.0 + mu * dt + sigma * sqrt_dt * eps_q[None, None, :])
        + u_grid[None, :, None] * dt
    )  # (n_phi_s, n_u, n_quad)

    # Clip phi_next to state grid
    phi_next_c = np.clip(phi_next, phi_s_grid[0], phi_s_grid[-1])

    # Wealth transitions:
    # delta_W[w, phi_s, u, q] = mu*phi_s*dt + sigma*phi_s*sqrt_dt*eps_q + impact_on_w[u]
    # = phi_s*(mu*dt + sigma*sqrt_dt*eps_q) + impact_on_w[u]
    # shape: (n_phi_s, n_u, n_quad)
    delta_w_phi = (
        phi_s_grid[:, None, None] * (mu * dt + sigma * sqrt_dt * eps_q[None, None, :])
        + impact_on_w[None, :, None]
    )
    # delta_w[w, phi_s, u, q] = delta_w_phi[phi_s, u, q]  (same for all w)

    # Storage
    values = np.empty((grid.n_time + 1, n_w, n_phi_s))
    policies = np.empty((grid.n_time, n_w, n_phi_s))

    # Terminal value (depends only on W)
    psi_w = terminal_value(w_grid, lambda_val, eta, alpha)  # (n_w,)
    values[grid.n_time] = psi_w[:, None]  # broadcast to (n_w, n_phi_s)

    # --- Backward induction ---
    for t_idx in range(grid.n_time - 1, -1, -1):
        V_next = values[t_idx + 1]  # (n_w, n_phi_s)

        # Build 2D interpolator for V_next
        interp_fn = RegularGridInterpolator(
            (w_grid, phi_s_grid), V_next,
            method="linear", bounds_error=False,
            fill_value=None,  # extrapolate with nearest boundary value
        )

        # Compute w_next for every (w, phi_s, u, q)
        # w_next[i_w, i_phi, i_u, i_q] = w_grid[i_w] + delta_w_phi[i_phi, i_u, i_q]
        # shape: (n_w, n_phi_s, n_u, n_quad)
        w_next = (
            w_grid[:, None, None, None]
            + delta_w_phi[None, :, :, :]
        )

        # Clip w_next to grid bounds for interpolation
        w_next_c = np.clip(w_next, w_grid[0], w_grid[-1])

        # phi_next is the same for all w: shape (n_phi_s, n_u, n_quad).
        # Broadcast and copy to shape (n_w, n_phi_s, n_u, n_quad) for stacking.
        phi_next_full = np.broadcast_to(
            phi_next_c[None, :, :, :], (n_w, n_phi_s, n_u, n_quad)
        ).copy()

        # Flatten for batch interpolation
        flat_pts = np.stack([w_next_c.ravel(), phi_next_full.ravel()], axis=1)
        V_flat = interp_fn(flat_pts)  # (n_w*n_phi_s*n_u*n_quad,)
        V_next_at = V_flat.reshape(n_w, n_phi_s, n_u, n_quad)

        # Apply quadrature weights: E[V_next] shape (n_w, n_phi_s, n_u)
        EV = V_next_at @ wts  # (n_w, n_phi_s, n_u)

        # Total cost: rc[phi_s, u] + EV[w, phi_s, u]
        # rc: (n_phi_s, n_u) → broadcast to (n_w, n_phi_s, n_u)
        total = rc[None, :, :] + EV  # (n_w, n_phi_s, n_u)

        # Optimal u
        best_u_idx = np.argmin(total, axis=2)  # (n_w, n_phi_s)
        iw_idx = np.arange(n_w)[:, None]
        ip_idx = np.arange(n_phi_s)[None, :]
        values[t_idx] = total[iw_idx, ip_idx, best_u_idx]
        policies[t_idx] = u_grid[best_u_idx]

    # Interpolate V(0, w0, phi0) — phi0 = 0 for exp3/exp4
    phi0 = 0.0
    interp0 = RegularGridInterpolator(
        (w_grid, phi_s_grid), values[0],
        method="linear", bounds_error=False, fill_value=None,
    )
    v0_interp = float(interp0([[model.w0, phi0]])[0])

    return values, policies, v0_interp


def lagrangian_value_2d(
    model: ModelParams,
    grid: GridParams,
    lambda_val: float,
    eta: float,
    mode: str = "quadratic",
    Lambda: float = 0.01,
    Lambda_32: float = 0.0045,
) -> float:
    """
    Return V_lambda(eta) = V_DP(0, w0, phi0; lambda, eta) + lambda * eta.

    Used as the objective in the inner golden-section search over eta.
    """
    _, _, v0 = backward_dp_2d(
        model, grid, lambda_val, eta, mode=mode, Lambda=Lambda, Lambda_32=Lambda_32
    )
    return v0 + lambda_val * eta


def get_state_grids(grid: GridParams) -> Tuple[np.ndarray, np.ndarray]:
    """Return (w_grid, phi_state_grid) used by the 2D DP."""
    w_grid = np.linspace(grid.w_min, grid.w_max, grid.n_w)
    phi_s_grid = np.linspace(grid.phi_state_min, grid.phi_state_max, grid.n_phi_state)
    return w_grid, phi_s_grid
