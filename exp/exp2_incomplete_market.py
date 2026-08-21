"""
Experiment 2: Frictionless Incomplete-Market Dynamic Portfolio Optimization
with Terminal CVaR Constraint and Nontraded Endowment Risk (beta_perp = 0.02).

Reproduces the paper's Table 3 results:
  c = -0.86 (nonbinding): lambda ≈ 0,     CVaR ≈ -0.8620, E[W_T] ≈ 1.0322, avg_phi ≈ 0.40
  c = -0.94 (binding):    lambda ≈ 0.1400, CVaR ≈ -0.9405, E[W_T] ≈ 1.0206, avg_phi ≈ 0.2549

Same architecture as exp1 but with additional nontraded Brownian component
that adds variance to wealth dynamics without providing a hedge channel.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.config import Exp2Config
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


def run_exp2() -> list:
    """Run experiment 2 for both CVaR limits."""
    config = Exp2Config()
    all_metrics = []

    for c in config.c_values:
        log.info("\n>>> Running Exp2 incomplete market  |  c = %.2f  |  beta_perp = %.4f",
                 c, config.beta_perp)
        m = solve_1d(config, c)
        all_metrics.append(m)

        plot_all_for_metric(
            m, alpha=config.model.alpha,
            results_dir=RESULTS_DIR / "exp2",
            merton_phi=MERTON_PHI,
            is_2d=False,
        )

    return all_metrics


if __name__ == "__main__":
    metrics_list = run_exp2()

    df = metrics_to_dataframe(metrics_list)
    print("\n===  Experiment 2 Summary  ===")
    print(df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp2_summary.csv", index=False)
    log.info("Results saved to %s", RESULTS_DIR / "exp2_summary.csv")
