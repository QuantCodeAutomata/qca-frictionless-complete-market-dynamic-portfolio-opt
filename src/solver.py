"""
High-level solver orchestrator for CVaR-constrained dynamic portfolio optimization.

Provides two public functions:
  solve_1d : for exp1 (complete market) and exp2 (incomplete market)
  solve_2d : for exp3 (quadratic regularisation) and exp4 (sqrt impact)

Each function:
  1. Defines oracle_lagrangian(lambda, eta) → V_lambda(eta) via backward DP
  2. Defines oracle_residual(lambda, eta) → (residual, values, policies)
     via backward DP + Monte Carlo simulation
  3. Calls bisection_lambda() which internally calls golden_section_eta()
  4. Runs final large-sample evaluation simulation
  5. Returns ExperimentMetrics

# Custom — Context7 found no library equivalent for this specific
#           CVaR-constrained stochastic control orchestration.
"""
import logging
import numpy as np
from typing import Union

from src.config import (
    ModelParams, GridParams, DualSearchParams, MCParams,
    Exp1Config, Exp2Config, Exp3Config, Exp4Config,
)
from src.dp_solver_1d import backward_dp_1d, lagrangian_value_1d
from src.dp_solver_2d import backward_dp_2d, lagrangian_value_2d
from src.dual_search import bisection_lambda
from src.simulator import (
    simulate_1d, simulate_2d,
    residual_from_simulation_1d, residual_from_simulation_2d,
)
from src.metrics import ExperimentMetrics, compute_metrics_1d, compute_metrics_2d

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1D solver (exp1, exp2)
# ---------------------------------------------------------------------------

def solve_1d(
    config: Union[Exp1Config, Exp2Config],
    c: float,
) -> ExperimentMetrics:
    """
    Solve the terminal CVaR-constrained portfolio problem for the 1D case.

    Parameters
    ----------
    config : Exp1Config (complete market) or Exp2Config (incomplete market)
    c      : CVaR limit (negative number)

    Returns
    -------
    ExperimentMetrics with all results and stored paths
    """
    model = config.model
    grid = config.grid
    search = config.search
    mc = config.mc
    beta_perp = config.beta_perp

    log.info("=" * 60)
    log.info("Solving 1D: %s  |  c = %.4f  |  beta_perp = %.4f",
             config.name, c, beta_perp)
    log.info("=" * 60)

    # Shared random state for reproducibility
    rng_inner = np.random.default_rng(mc.seed)
    rng_eval = np.random.default_rng(mc.seed + 1)

    # --- Oracle definitions ---

    def oracle_lagrangian(lambda_val: float, eta: float) -> float:
        """Return V_lambda(eta) = V_DP(0, w0) + lambda*eta."""
        return lagrangian_value_1d(model, grid, lambda_val, eta, beta_perp)

    def oracle_residual(lambda_val: float, eta: float):
        """
        Return (residual, values, policies):
          residual = C(phi_lambda, eta) = G(eta; phi_lambda) - c
        Uses a small MC sample during bisection iterations.
        """
        values, policies, _ = backward_dp_1d(model, grid, lambda_val, eta, beta_perp)
        residual = residual_from_simulation_1d(
            model, grid, policies, eta, c, mc,
            n_paths=mc.n_mc_residual,
            beta_perp=beta_perp,
            rng=np.random.default_rng(mc.seed + int(abs(lambda_val * 1000))),
        )
        return residual, values, policies

    # --- Run nested dual search ---
    lambda_hat, eta_hat, residual_final, values_final, policies_final = bisection_lambda(
        oracle_lagrangian, oracle_residual, search, c
    )
    binding = lambda_hat > 1e-6

    # --- Final evaluation with large MC sample ---
    log.info("Running final evaluation simulation (n_mc=%d) ...", mc.n_mc_eval)
    W_paths, phi_paths, t_grid = simulate_1d(
        model, grid, policies_final, mc,
        n_paths=mc.n_mc_eval,
        beta_perp=beta_perp,
        rng=rng_eval,
    )

    # --- Metrics ---
    paper_ref = config.paper_table.get(c, {})
    metrics = compute_metrics_1d(
        exp_name=config.name,
        c=c,
        lambda_hat=lambda_hat,
        eta_hat=eta_hat,
        binding=binding,
        W_paths=W_paths,
        phi_paths=phi_paths,
        t_grid=t_grid,
        alpha=model.alpha,
        paper_ref=paper_ref,
    )

    log.info("Results: lambda=%.6f  eta=%.6f  CVaR=%.4f  E[W_T]=%.4f  avg_phi=%.4f",
             lambda_hat, eta_hat, metrics.cvar_terminal, metrics.E_WT, metrics.avg_phi)
    if paper_ref:
        log.info("Paper:   lambda=%.6f               CVaR=%.4f  E[W_T]=%.4f  avg_phi=%.4f",
                 paper_ref.get("lambda", float("nan")),
                 paper_ref.get("cvar", float("nan")),
                 paper_ref.get("E_WT", float("nan")),
                 paper_ref.get("avg_phi", float("nan")))

    return metrics


# ---------------------------------------------------------------------------
# 2D solver (exp3, exp4)
# ---------------------------------------------------------------------------

def solve_2d(
    config: Union[Exp3Config, Exp4Config],
    c: float,
) -> ExperimentMetrics:
    """
    Solve the terminal CVaR-constrained portfolio problem for the 2D case.

    Parameters
    ----------
    config : Exp3Config (quadratic) or Exp4Config (sqrt impact)
    c      : CVaR limit (negative number)
    """
    model = config.model
    grid = config.grid
    search = config.search
    mc = config.mc

    if isinstance(config, Exp3Config):
        mode = "quadratic"
        Lambda = config.Lambda
        Lambda_32 = 0.0
        phi0 = config.phi0
    else:
        mode = "sqrt_impact"
        Lambda = 0.0
        Lambda_32 = config.Lambda_32
        phi0 = config.phi0

    log.info("=" * 60)
    log.info("Solving 2D: %s  |  c = %.4f  |  mode = %s", config.name, c, mode)
    log.info("=" * 60)

    rng_eval = np.random.default_rng(mc.seed + 1)

    # --- Oracle definitions ---

    def oracle_lagrangian(lambda_val: float, eta: float) -> float:
        return lagrangian_value_2d(
            model, grid, lambda_val, eta, mode=mode, Lambda=Lambda, Lambda_32=Lambda_32
        )

    def oracle_residual(lambda_val: float, eta: float):
        values, policies, _ = backward_dp_2d(
            model, grid, lambda_val, eta, mode=mode, Lambda=Lambda, Lambda_32=Lambda_32
        )
        residual = residual_from_simulation_2d(
            model, grid, policies, eta, c, mc,
            n_paths=mc.n_mc_residual,
            phi0=phi0, mode=mode, Lambda=Lambda, Lambda_32=Lambda_32,
            rng=np.random.default_rng(mc.seed + int(abs(lambda_val * 1000))),
        )
        return residual, values, policies

    # --- Nested dual search ---
    lambda_hat, eta_hat, residual_final, values_final, policies_final = bisection_lambda(
        oracle_lagrangian, oracle_residual, search, c
    )
    binding = lambda_hat > 1e-6

    # --- Final evaluation ---
    log.info("Running final evaluation simulation (n_mc=%d) ...", mc.n_mc_eval)
    W_paths, Phi_paths, U_paths, t_grid = simulate_2d(
        model, grid, policies_final, mc,
        n_paths=mc.n_mc_eval,
        phi0=phi0, mode=mode, Lambda=Lambda, Lambda_32=Lambda_32,
        rng=rng_eval,
    )

    # --- Metrics ---
    paper_ref = config.paper_table.get(c, {})
    metrics = compute_metrics_2d(
        exp_name=config.name,
        c=c,
        lambda_hat=lambda_hat,
        eta_hat=eta_hat,
        binding=binding,
        W_paths=W_paths,
        phi_paths=Phi_paths,
        u_paths=U_paths,
        t_grid=t_grid,
        alpha=model.alpha,
        paper_ref=paper_ref,
    )

    log.info("Results: lambda=%.6f  eta=%.6f  CVaR=%.4f  E[W_T]=%.4f  avg_phi=%.4f",
             lambda_hat, eta_hat, metrics.cvar_terminal, metrics.E_WT, metrics.avg_phi)
    if paper_ref:
        log.info("Paper:   lambda=%.6f               CVaR=%.4f  E[W_T]=%.4f  avg_phi=%.4f",
                 paper_ref.get("lambda", float("nan")),
                 paper_ref.get("cvar", float("nan")),
                 paper_ref.get("E_WT", float("nan")),
                 paper_ref.get("avg_phi", float("nan")))

    return metrics
