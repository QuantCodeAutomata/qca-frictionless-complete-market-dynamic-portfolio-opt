"""
Experiment configurations for CVaR-constrained dynamic portfolio optimization.

Covers four experiments from the paper:
  exp1 – Frictionless complete market (1D state: wealth)
  exp2 – Frictionless incomplete market (1D state + nontraded endowment)
  exp3 – Quadratic trading-rate regularization (2D state: wealth, exposure)
  exp4 – Square-root price impact (2D state: wealth, exposure)
"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ModelParams:
    """Shared continuous-time model parameters."""
    T: float = 1.0          # horizon
    r: float = 0.00         # risk-free rate
    mu: float = 0.08        # drift of risky asset
    sigma: float = 0.20     # volatility of risky asset
    gamma: float = 5.0      # risk-aversion coefficient
    w0: float = 1.0         # initial wealth
    alpha: float = 0.95     # CVaR confidence level


@dataclass
class GridParams:
    """Numerical discretization parameters."""
    n_time: int = 40               # number of time steps
    # 1-D wealth grid
    w_min: float = 0.05
    w_max: float = 4.0
    n_w: int = 80
    # Control grid for 1-D (direct exposure phi)
    phi_min: float = -0.5
    phi_max: float = 2.0
    n_phi: int = 60
    # 2-D phi (exposure) grid used as state variable in exp3/exp4
    phi_state_min: float = -0.5
    phi_state_max: float = 2.0
    n_phi_state: int = 30
    # Control grid for 2-D (trading rate u)
    u_min: float = -8.0
    u_max: float = 15.0
    n_u: int = 50
    # Gauss-Hermite quadrature order
    n_quad: int = 7


@dataclass
class DualSearchParams:
    """Parameters for the nested dual-search algorithm."""
    # Threshold interval E for eta (auxiliary CVaR threshold)
    eta_low: float = -2.0
    eta_high: float = 0.0
    n_golden: int = 12          # golden-section iterations (inner)
    lambda_max: float = 3.0     # upper bracket for lambda bisection
    n_bisect: int = 20          # bisection iterations (outer)
    bisect_tol: float = 1e-4    # convergence tolerance for lambda bracket


@dataclass
class MCParams:
    """Monte Carlo simulation parameters."""
    n_mc_residual: int = 8_000   # paths used for residual during bisection
    n_mc_eval: int = 40_000      # paths used for final evaluation
    seed: int = 42


@dataclass
class Exp1Config:
    """Complete-market frictionless experiment (exp_1)."""
    name: str = "exp1_complete_market"
    # Model
    model: ModelParams = field(default_factory=ModelParams)
    # CVaR limits to test
    c_values: Tuple[float, ...] = (-0.86, -0.94)
    # Numerical grids (1-D DP)
    grid: GridParams = field(default_factory=GridParams)
    # Dual search
    search: DualSearchParams = field(default_factory=DualSearchParams)
    # MC
    mc: MCParams = field(default_factory=MCParams)
    # Nontraded risk (zero for complete market)
    beta_perp: float = 0.0
    # Paper Table 2 reference values keyed by c
    paper_table: dict = field(default_factory=lambda: {
        -0.86: {"lambda": 0.0000, "cvar": -0.8671, "E_WT": 1.0322, "avg_phi": 0.4000},
        -0.94: {"lambda": 0.0720, "cvar": -0.9406, "E_WT": 1.0242, "avg_phi": 0.2981},
    })


@dataclass
class Exp2Config:
    """Incomplete-market frictionless experiment (exp_2)."""
    name: str = "exp2_incomplete_market"
    model: ModelParams = field(default_factory=ModelParams)
    c_values: Tuple[float, ...] = (-0.86, -0.94)
    grid: GridParams = field(default_factory=GridParams)
    search: DualSearchParams = field(default_factory=DualSearchParams)
    mc: MCParams = field(default_factory=MCParams)
    # Nontraded endowment volatility
    beta_perp: float = 0.02
    paper_table: dict = field(default_factory=lambda: {
        -0.86: {"lambda": 0.0000, "cvar": -0.8620, "E_WT": 1.0322, "avg_phi": 0.4000},
        -0.94: {"lambda": 0.1400, "cvar": -0.9405, "E_WT": 1.0206, "avg_phi": 0.2549},
    })


@dataclass
class Exp3Config:
    """Quadratic trading-rate regularization experiment (exp_3)."""
    name: str = "exp3_quadratic_reg"
    model: ModelParams = field(default_factory=ModelParams)
    c_values: Tuple[float, ...] = (-0.86, -0.94)
    # Use smaller grids for 2D problem (feasibility)
    grid: GridParams = field(default_factory=lambda: GridParams(
        n_time=20,
        w_min=0.05, w_max=3.5, n_w=50,
        phi_state_min=-0.3, phi_state_max=1.5, n_phi_state=30,
        u_min=-8.0, u_max=15.0, n_u=40,
        n_quad=5,
    ))
    search: DualSearchParams = field(default_factory=lambda: DualSearchParams(
        eta_low=-2.0, eta_high=0.0,
        n_golden=10, lambda_max=2.0, n_bisect=15, bisect_tol=1e-4
    ))
    mc: MCParams = field(default_factory=lambda: MCParams(
        n_mc_residual=5_000, n_mc_eval=30_000, seed=42
    ))
    # Quadratic regularization coefficient
    Lambda: float = 0.01
    # Initial risky exposure
    phi0: float = 0.0
    paper_table: dict = field(default_factory=lambda: {
        -0.86: {"lambda": 0.0000, "cvar": -0.8971, "E_WT": 1.0247, "avg_phi": 0.3104},
        -0.94: {"lambda": 0.06001, "cvar": -0.9413, "E_WT": 1.0192, "avg_phi": 0.2415,
                "eta": -0.947499},
    })


@dataclass
class Exp4Config:
    """Square-root price impact experiment (exp_4)."""
    name: str = "exp4_sqrt_impact"
    model: ModelParams = field(default_factory=ModelParams)
    c_values: Tuple[float, ...] = (-0.86, -0.94)
    grid: GridParams = field(default_factory=lambda: GridParams(
        n_time=20,
        w_min=0.05, w_max=3.5, n_w=50,
        phi_state_min=-0.3, phi_state_max=1.5, n_phi_state=30,
        u_min=-8.0, u_max=15.0, n_u=50,
        n_quad=5,
    ))
    search: DualSearchParams = field(default_factory=lambda: DualSearchParams(
        eta_low=-2.0, eta_high=0.0,
        n_golden=10, lambda_max=2.0, n_bisect=15, bisect_tol=1e-4
    ))
    mc: MCParams = field(default_factory=lambda: MCParams(
        n_mc_residual=5_000, n_mc_eval=30_000, seed=42
    ))
    # Square-root impact coefficient Lambda_{3/2}
    Lambda_32: float = 0.0045
    # Initial risky exposure
    phi0: float = 0.0
    paper_table: dict = field(default_factory=lambda: {
        -0.86: {"lambda": 0.0000, "cvar": -0.8839, "E_WT": 1.0260, "avg_phi": 0.3500},
        -0.94: {"lambda": 0.07818, "cvar": -0.9409, "E_WT": 1.0180, "avg_phi": 0.2422},
    })
