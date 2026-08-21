"""
Experiment 1: Frictionless Complete-Market Dynamic Portfolio Optimization
with Terminal CVaR Constraint.

Reproduces the paper's Table 2 results:
  c = -0.86 (nonbinding): lambda ≈ 0,    CVaR ≈ -0.8671, E[W_T] ≈ 1.0322, avg_phi ≈ 0.40
  c = -0.94 (binding):    lambda ≈ 0.072, CVaR ≈ -0.9406, E[W_T] ≈ 1.0242, avg_phi ≈ 0.2981

Model parameters: T=1, r=0, mu=0.08, sigma=0.20, gamma=5, w0=1, alpha=0.95, beta_perp=0
Merton benchmark: phi* = mu / (gamma * sigma^2) = 0.08 / (5 * 0.04) = 0.40
"""
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.config import Exp1Config
from src.solver import solve_1d
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


def run_exp1() -> list:
    """
    Run experiment 1 for both CVaR limits and return list of ExperimentMetrics.
    """
    config = Exp1Config()
    all_metrics = []

    for c in config.c_values:
        log.info("\n>>> Running Exp1 complete market  |  c = %.2f", c)
        m = solve_1d(config, c)
        all_metrics.append(m)

        # Save figures
        plot_all_for_metric(
            m, alpha=config.model.alpha,
            results_dir=RESULTS_DIR / "exp1",
            merton_phi=MERTON_PHI,
            is_2d=False,
        )

    return all_metrics


if __name__ == "__main__":
    metrics_list = run_exp1()

    # Summary table
    df = metrics_to_dataframe(metrics_list)
    print("\n===  Experiment 1 Summary  ===")
    print(df.to_string(index=False))

    # Save table
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp1_summary.csv", index=False)
    log.info("Results saved to %s", RESULTS_DIR / "exp1_summary.csv")
