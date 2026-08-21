"""
Nested dual-search algorithm for the terminal CVaR-constrained dynamic portfolio problem.

Architecture (following the paper's methodology exactly):
  Outer loop : bisection over Lagrange multiplier lambda in [0, lambda_max]
  Inner loop : golden-section search over auxiliary threshold eta in E = [eta_low, eta_high]

The golden ratio constant used is rho = (sqrt(5) - 1) / 2  (paper confirms this value).

# Custom — Context7 found no library equivalent for this nested lambda/eta dual search
#           applied to a backward-DP oracle (paper Section 4-5).
"""
import numpy as np
import logging
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

from src.config import ModelParams, GridParams, DualSearchParams, MCParams
from src.cvar_utils import cvar_constraint_value, cvar_sample

log = logging.getLogger(__name__)

GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0   # rho ≈ 0.6180339887


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class EtaSearchResult:
    """Result of the inner golden-section search over eta."""
    eta_star: float
    v_star: float          # V_lambda(eta*) = v0_DP + lambda * eta*
    n_iters: int


@dataclass
class DualSearchResult:
    """Result of the full nested dual search."""
    lambda_hat: float
    eta_hat: float
    binding: bool          # True if CVaR constraint is active
    # Final metrics from out-of-sample simulation
    cvar_terminal: float   # estimated CVaR_alpha(-W_T)
    E_WT: float            # expected terminal wealth
    avg_phi: float         # average risky exposure
    # Residual at convergence
    residual: float
    # Stored for diagnostics
    lambda_history: list
    residual_history: list


# ---------------------------------------------------------------------------
# Type alias for inner oracle
# ---------------------------------------------------------------------------
# The oracle takes (lambda_val, eta) and returns (v_lagrangian, values, policies)
OracleCallable = Callable[[float, float], Tuple[float, object, object]]


# ---------------------------------------------------------------------------
# Golden-section search over eta  (inner loop)
# ---------------------------------------------------------------------------

def golden_section_eta(
    oracle_v: Callable[[float], float],
    eta_low: float,
    eta_high: float,
    n_iters: int,
) -> EtaSearchResult:
    """
    Minimise V_lambda(eta) over eta in [eta_low, eta_high] using golden-section search.

    V_lambda(eta) is assumed unimodal in eta (follows from convexity of the
    Lagrangian in eta; see paper Section 4).

    Parameters
    ----------
    oracle_v  : callable (eta) -> V_lambda(eta) scalar
    eta_low   : left bracket endpoint
    eta_high  : right bracket endpoint
    n_iters   : number of golden-section contractions

    Returns
    -------
    EtaSearchResult with eta_star and v_star
    """
    a, b = eta_low, eta_high

    # Initial interior evaluation points
    eta1 = b - GOLDEN_RATIO * (b - a)
    eta2 = a + GOLDEN_RATIO * (b - a)
    v1 = oracle_v(eta1)
    v2 = oracle_v(eta2)

    log.debug("  eta golden-section start: [%.4f, %.4f]", a, b)

    for k in range(n_iters):
        if v1 < v2:
            # Minimum in [a, eta2]
            b = eta2
            eta2, v2 = eta1, v1
            eta1 = b - GOLDEN_RATIO * (b - a)
            v1 = oracle_v(eta1)
        else:
            # Minimum in [eta1, b]
            a = eta1
            eta1, v1 = eta2, v2
            eta2 = a + GOLDEN_RATIO * (b - a)
            v2 = oracle_v(eta2)
        log.debug("  eta iter %d: [%.5f, %.5f]  v1=%.6f  v2=%.6f", k, a, b, v1, v2)

    eta_star = (a + b) / 2.0
    v_star = oracle_v(eta_star)
    return EtaSearchResult(eta_star=eta_star, v_star=v_star, n_iters=n_iters + 2)


# ---------------------------------------------------------------------------
# Outer bisection over lambda
# ---------------------------------------------------------------------------

def _run_inner_and_residual(
    lambda_val: float,
    search_cfg: DualSearchParams,
    oracle_lagrangian: Callable[[float, float], float],
    oracle_residual: Callable[[float, float], Tuple[float, object, object]],
) -> Tuple[float, float, float, object, object]:
    """
    For a given lambda, run the inner golden-section search over eta, then
    compute the CVaR constraint residual using the optimal policy.

    Returns (residual, eta_star, v_star, values, policies).
    """
    def v_lambda_eta(eta: float) -> float:
        return oracle_lagrangian(lambda_val, eta)

    eta_result = golden_section_eta(
        v_lambda_eta,
        search_cfg.eta_low, search_cfg.eta_high,
        search_cfg.n_golden,
    )
    eta_star = eta_result.eta_star

    # Get residual and policy for the optimal eta
    residual, values, policies = oracle_residual(lambda_val, eta_star)
    return residual, eta_star, eta_result.v_star, values, policies


def bisection_lambda(
    oracle_lagrangian: Callable[[float, float], float],
    oracle_residual: Callable[[float, float], Tuple[float, object, object]],
    search_cfg: DualSearchParams,
    c: float,
) -> Tuple[float, float, float, object, object]:
    """
    Outer bisection search over lambda in [0, lambda_max].

    Algorithm:
    1. Evaluate R(0) = CVaR_alpha(-W_T) - c with unconstrained policy.
    2. If R(0) <= 0 → nonbinding; return lambda_hat = 0.
    3. Else bisect lambda on [0, lambda_max] until |lambda_right - lambda_left| < tol.

    Parameters
    ----------
    oracle_lagrangian : (lambda, eta) -> V_lambda(eta) float
    oracle_residual   : (lambda, eta) -> (residual, values, policies)
    search_cfg        : dual search parameters
    c                 : CVaR limit (e.g. -0.94)

    Returns
    -------
    (lambda_hat, eta_hat, residual, values, policies)
    """
    lambda_history = []
    residual_history = []

    # --- Step 1: Unconstrained problem (lambda = 0) ---
    log.info("Evaluating unconstrained problem (lambda = 0) ...")
    residual_0, eta_0, _, values_0, policies_0 = _run_inner_and_residual(
        0.0, search_cfg, oracle_lagrangian, oracle_residual
    )
    log.info("  R(0) = %.6f  (c = %.4f)", residual_0, c)
    lambda_history.append(0.0)
    residual_history.append(residual_0)

    # --- Step 2: Nonbinding check ---
    if residual_0 <= 0.0:
        log.info("CVaR constraint is NONBINDING. lambda_hat = 0.")
        return 0.0, eta_0, residual_0, values_0, policies_0

    # --- Step 3: Bisection ---
    log.info("CVaR constraint is BINDING. Starting bisection over lambda ...")
    lam_lo, lam_hi = 0.0, search_cfg.lambda_max

    # Verify sign at lambda_max
    res_hi, _, _, _, _ = _run_inner_and_residual(
        lam_hi, search_cfg, oracle_lagrangian, oracle_residual
    )
    if res_hi > 0.0:
        log.warning("R(lambda_max=%.2f) = %.4f > 0; lambda_max may be too small.", lam_hi, res_hi)
        # Double lambda_max and try again
        lam_hi *= 3.0
        res_hi, _, _, _, _ = _run_inner_and_residual(
            lam_hi, search_cfg, oracle_lagrangian, oracle_residual
        )
        log.info("Extended lambda_max to %.2f; R = %.4f", lam_hi, res_hi)

    best_values, best_policies = values_0, policies_0
    best_eta, best_lam = eta_0, 0.0

    for k in range(search_cfg.n_bisect):
        lam_mid = (lam_lo + lam_hi) / 2.0
        residual_mid, eta_mid, _, values_mid, policies_mid = _run_inner_and_residual(
            lam_mid, search_cfg, oracle_lagrangian, oracle_residual
        )
        lambda_history.append(lam_mid)
        residual_history.append(residual_mid)

        log.info(
            "  bisect iter %2d: lambda=%.6f  eta=%.6f  R=%.6f  bracket=[%.6f, %.6f]",
            k, lam_mid, eta_mid, residual_mid, lam_lo, lam_hi
        )

        best_lam, best_eta = lam_mid, eta_mid
        best_values, best_policies = values_mid, policies_mid

        if residual_mid > 0.0:
            lam_lo = lam_mid      # lambda too small → push up
        else:
            lam_hi = lam_mid      # lambda too large → push down

        if (lam_hi - lam_lo) < search_cfg.bisect_tol:
            log.info("  Bisection converged at iteration %d.", k)
            break

    lambda_hat = (lam_lo + lam_hi) / 2.0
    # Final solve at lambda_hat
    residual_f, eta_f, _, values_f, policies_f = _run_inner_and_residual(
        lambda_hat, search_cfg, oracle_lagrangian, oracle_residual
    )
    log.info("Bisection done. lambda_hat=%.6f  eta_hat=%.6f  R=%.6f",
             lambda_hat, eta_f, residual_f)

    return lambda_hat, eta_f, residual_f, values_f, policies_f
