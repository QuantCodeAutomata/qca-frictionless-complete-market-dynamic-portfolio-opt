"""
Experiment 3: Dynamic Portfolio Optimization with Terminal CVaR Constraint
and Quadratic Trading-Rate Regularization (Lambda = 0.01, phi0 = 0).

Two-dimensional state: (W_t, phi_t). Control: u_t = dphi_t/dt.

Reproduces the paper's Table 4 results:
  c = -0.86 (nonbinding): lambda ≈ 0.0000,  CVaR ≈ -0.8971, E[W_T] ≈ 1.0247, avg_phi ≈ 0.3104
  c = -0.94 (binding):    lambda ≈ 0.06001, CVaR ≈ -0.9413, E[W_T] ≈ 1.0192, avg_phi ≈ 0.2415
  Binding-case pair: (lambda, eta) ≈ (0.06001, -0.947499)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.config import Exp3Config
from src.solver import solve_2d
from src.metrics import metrics_to_dataframe
from src.plotting import plot_all_for_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MERTON_PHI = 0.08 / (5.0 * 0.20 ** 2)   # = 0.40


def run_exp3() -> list:
    """Run experiment 3 for both CVaR limits."""
    config = Exp3Config()
    all_metrics = []

    for c in config.c_values:
        log.info("\n>>> Running Exp3 quadratic regularization  |  c = %.2f  |  Lambda = %.4f",
                 c, config.Lambda)
        m = solve_2d(config, c)
        all_metrics.append(m)

        plot_all_for_metric(
            m, alpha=config.model.alpha,
            results_dir=RESULTS_DIR / "exp3",
            merton_phi=MERTON_PHI,
            is_2d=True,
        )

    return all_metrics


if __name__ == "__main__":
    metrics_list = run_exp3()

    df = metrics_to_dataframe(metrics_list)
    print("\n===  Experiment 3 Summary  ===")
    print(df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp3_summary.csv", index=False)
    log.info("Results saved to %s", RESULTS_DIR / "exp3_summary.csv")
