"""
Experiment 4: Dynamic Portfolio Optimization with Terminal CVaR Constraint
and Square-Root Price Impact (Lambda_32 = 0.0045, phi0 = 0).

Two-dimensional state: (W_t, phi_t). Control: u_t = dphi_t/dt.
Impact cost enters both wealth dynamics and running objective.

Reproduces the paper's Table 5 results:
  c = -0.86 (nonbinding): lambda ≈ 0.00000, CVaR ≈ -0.8839, E[W_T] ≈ 1.0260, avg_phi ≈ 0.3500
  c = -0.94 (binding):    lambda ≈ 0.07818, CVaR ≈ -0.9409, E[W_T] ≈ 1.0180, avg_phi ≈ 0.2422
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.config import Exp4Config
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


def run_exp4() -> list:
    """Run experiment 4 for both CVaR limits."""
    config = Exp4Config()
    all_metrics = []

    for c in config.c_values:
        log.info("\n>>> Running Exp4 sqrt impact  |  c = %.2f  |  Lambda_32 = %.6f",
                 c, config.Lambda_32)
        m = solve_2d(config, c)
        all_metrics.append(m)

        plot_all_for_metric(
            m, alpha=config.model.alpha,
            results_dir=RESULTS_DIR / "exp4",
            merton_phi=MERTON_PHI,
            is_2d=True,
        )

    return all_metrics


if __name__ == "__main__":
    metrics_list = run_exp4()

    df = metrics_to_dataframe(metrics_list)
    print("\n===  Experiment 4 Summary  ===")
    print(df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp4_summary.csv", index=False)
    log.info("Results saved to %s", RESULTS_DIR / "exp4_summary.csv")
