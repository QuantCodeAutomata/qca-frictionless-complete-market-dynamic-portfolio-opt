"""
Master script: run all four CVaR-constrained portfolio optimization experiments.

Results (metrics and figures) are saved to results/.
A consolidated RESULTS.md is written with all table values.

Usage
-----
    python exp/run_all.py
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.exp1_complete_market import run_exp1
from exp.exp2_incomplete_market import run_exp2
from exp.exp3_quadratic_reg import run_exp3
from exp.exp4_sqrt_impact import run_exp4
from src.metrics import metrics_to_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def write_results_md(all_exp_dfs: dict, elapsed: dict) -> None:
    """Write consolidated RESULTS.md with all summary tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / "RESULTS.md"

    paper_tables = {
        "exp1": {
            "caption": "Table 2 — Complete market",
            "rows": [
                ("−0.86", "nonbinding", "0.0000", "−0.8671", "1.0322", "0.4000"),
                ("−0.94", "binding",    "0.0720", "−0.9406", "1.0242", "0.2981"),
            ]
        },
        "exp2": {
            "caption": "Table 3 — Incomplete market (β⊥ = 0.02)",
            "rows": [
                ("−0.86", "nonbinding", "0.0000", "−0.8620", "1.0322", "0.4000"),
                ("−0.94", "binding",    "0.1400", "−0.9405", "1.0206", "0.2549"),
            ]
        },
        "exp3": {
            "caption": "Table 4 — Quadratic regularisation (Λ = 0.01)",
            "rows": [
                ("−0.86", "nonbinding", "0.0000",  "−0.8971", "1.0247", "0.3104"),
                ("−0.94", "binding",    "0.06001", "−0.9413", "1.0192", "0.2415"),
            ]
        },
        "exp4": {
            "caption": "Table 5 — Square-root impact (Λ₃⁄₂ = 0.0045)",
            "rows": [
                ("−0.86", "nonbinding", "0.0000",  "−0.8839", "1.0260", "0.3500"),
                ("−0.94", "binding",    "0.07818", "−0.9409", "1.0180", "0.2422"),
            ]
        },
    }

    with open(md_path, "w") as f:
        f.write("# Experiment Results\n\n")
        f.write(
            "Numerical reproduction of Tables 2–5 from the paper:\n"
            "*Dynamic Portfolio Optimization with Terminal CVaR Constraint*.\n\n"
            "Model parameters: T=1, r=0.00, μ=0.08, σ=0.20, γ=5, w₀=1, α=0.95.\n\n"
        )

        header = "| c | Binding | λ̂ | CVaR_α(−W_T) | E[W_T] | avg φ |\n"
        sep    = "|---|---------|---|------------|--------|-------|\n"

        for key, df in all_exp_dfs.items():
            exp_num = key.replace("exp", "")
            pt = paper_tables[key]
            f.write(f"## Experiment {exp_num}: {pt['caption']}\n\n")

            # Paper reference
            f.write("**Paper reference:**\n\n")
            f.write(header + sep)
            for row in pt["rows"]:
                f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |\n")
            f.write("\n")

            # Numerical results
            f.write("**Numerical results (this implementation):**\n\n")
            f.write(header + sep)
            for _, row in df.iterrows():
                f.write(
                    f"| {row['c']:.2f} | {'yes' if row['Binding'] else 'no'} "
                    f"| {row['lambda_hat']:.5f} | {row['CVaR_terminal']:.4f} "
                    f"| {row['E[W_T]']:.4f} | {row['avg_phi']:.4f} |\n"
                )
            f.write(f"\n*Elapsed: {elapsed.get(key, '?'):.1f}s*\n\n")

        f.write("## Figures\n\n")
        f.write(
            "All figures (exposure paths, trading-rate paths, terminal wealth distributions,\n"
            "interim CVaR diagnostics, and sample paths) are saved under `results/exp{1-4}/`.\n\n"
        )
        f.write("## Numerical Setup\n\n")
        f.write(
            "- **1D DP** (exp1, exp2): Gauss–Hermite quadrature, N_quad=7, N_W=80, N_φ=60,\n"
            "  N_time=40. Outer bisection 20 iterations, inner golden-section 12 iterations.\n"
            "- **2D DP** (exp3, exp4): Vectorised Bellman with RegularGridInterpolator.\n"
            "  N_W=25, N_φ=25, N_u=30, N_time=15. Same outer/inner iteration counts.\n"
            "- **Monte Carlo**: 2 000 paths during bisection, 20 000 paths for final evaluation.\n"
            "- **Seed**: fixed at 42 for reproducibility.\n"
        )

    log.info("RESULTS.md written to %s", md_path)


def main() -> None:
    all_metrics = {}
    all_dfs = {}
    elapsed = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for exp_name, runner in [
        ("exp1", run_exp1),
        ("exp2", run_exp2),
        ("exp3", run_exp3),
        ("exp4", run_exp4),
    ]:
        log.info("\n" + "=" * 70)
        log.info("Starting %s", exp_name.upper())
        log.info("=" * 70)
        t0 = time.time()
        metrics_list = runner()
        elapsed[exp_name] = time.time() - t0
        all_metrics[exp_name] = metrics_list
        df = metrics_to_dataframe(metrics_list)
        all_dfs[exp_name] = df
        df.to_csv(RESULTS_DIR / f"{exp_name}_summary.csv", index=False)
        log.info("%s completed in %.1fs", exp_name.upper(), elapsed[exp_name])

    write_results_md(all_dfs, elapsed)

    log.info("\nAll experiments completed.")
    for k, t in elapsed.items():
        log.info("  %s: %.1f s", k, t)


if __name__ == "__main__":
    main()
