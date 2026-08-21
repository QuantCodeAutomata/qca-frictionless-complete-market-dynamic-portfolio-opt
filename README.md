# Dynamic Portfolio Optimization with Terminal CVaR Constraint

Numerical reproduction of the four experiments reported in:

> *Dynamic Portfolio Optimization with Terminal CVaR Constraint* (Tables 2–5)

All four experiments solve a **continuous-time stochastic portfolio-control problem**
with a hard terminal Conditional Value-at-Risk (CVaR) constraint on the loss
`ℓ(W_T) = −W_T`, using the **Rockafellar–Uryasev** auxiliary-threshold
reformulation combined with a **nested Lagrangian dual search**:

| Layer | Algorithm | Section |
|-------|-----------|---------|
| Outer | Bisection over Lagrange multiplier λ | Paper §4 |
| Inner | Golden-section search over threshold η | Paper §4 |
| Oracle | Backward dynamic programming (Bellman) | Paper §5–6 |

---

## Experiments

| ID | Title | State dim | Control | Key parameter |
|----|-------|-----------|---------|---------------|
| exp1 | Frictionless complete market | 1-D (W) | φ (dollar exposure) | β⊥ = 0 |
| exp2 | Frictionless incomplete market | 1-D (W) | φ | β⊥ = 0.02 |
| exp3 | Quadratic trading-rate regularisation | 2-D (W, φ) | u = φ̇ | Λ = 0.01 |
| exp4 | Square-root price impact | 2-D (W, φ) | u = φ̇ | Λ₃/₂ = 0.0045 |

Both CVaR limits c ∈ {−0.86 (nonbinding), −0.94 (binding)} are tested per experiment.

**Model parameters:** T = 1, r = 0, μ = 0.08, σ = 0.20, γ = 5, w₀ = 1, α = 0.95.

---

## Repository Structure

```
repo/
├── data/                   # Placeholder (all data is synthetic)
├── src/
│   ├── config.py           # Experiment configurations (dataclasses)
│   ├── cvar_utils.py       # CVaR estimators (sample tail-avg + threshold form)
│   ├── dp_solver_1d.py     # 1-D backward DP (Gauss–Hermite quadrature)
│   ├── dp_solver_2d.py     # 2-D backward DP (vectorised Bellman + RGI)
│   ├── dual_search.py      # Golden-section (η) + bisection (λ) orchestrator
│   ├── simulator.py        # Forward MC simulation of feedback policies
│   ├── metrics.py          # Performance metrics computation and packaging
│   ├── plotting.py         # All figure generation (matplotlib)
│   └── solver.py           # High-level solve_1d / solve_2d entry points
├── exp/
│   ├── exp1_complete_market.py
│   ├── exp2_incomplete_market.py
│   ├── exp3_quadratic_reg.py
│   ├── exp4_sqrt_impact.py
│   └── run_all.py          # Run all four experiments sequentially
├── tests/
│   ├── test_cvar_utils.py
│   ├── test_dp_solver.py
│   ├── test_dual_search.py
│   └── test_experiments.py
├── results/
│   ├── RESULTS.md          # Numerical results vs paper Tables 2–5
│   ├── exp{1-4}/           # Figures per experiment
│   └── exp{1-4}_summary.csv
├── README.md
└── .gitignore
```

---

## Setup and Installation

```bash
# Python 3.10+
pip install numpy scipy pandas matplotlib numba
# Or install the full scientific stack
pip install -r requirements.txt  # if present in your environment
```

The project uses only standard scientific-Python packages:
`numpy`, `scipy`, `pandas`, `matplotlib`, `numba` (optional but recommended).

---

## Running

### All experiments

```bash
python exp/run_all.py
```

Results tables are printed to stdout and saved to `results/RESULTS.md`.
Figures are saved to `results/exp{1-4}/`.

### Individual experiments

```bash
python exp/exp1_complete_market.py
python exp/exp2_incomplete_market.py
python exp/exp3_quadratic_reg.py
python exp/exp4_sqrt_impact.py
```

### Tests

```bash
pytest tests/ -v
```

104 tests verify:
- CVaR estimators (sample tail-average vs threshold form)
- Backward DP (terminal conditions, monotonicity in λ)
- Dual search (bisection, golden-section properties)
- Experiment physics (wealth dynamics, Merton benchmark, CVaR constraint classification)

---

## Key Implementation Notes

### Algorithm fidelity

The nested dual search follows the paper exactly:
- **Outer bisection** over λ ∈ [0, λ_max] (default λ_max = 3, auto-extended).
- **Inner golden-section** over η ∈ [−2, 0] with ρ = (√5 − 1)/2.
- **Residual** R(λ) = CVaR_α(−W_T^{φ_λ}) − c evaluated via the **sample
  tail-average** (= min_η G(η; φ_λ) by Rockafellar–Uryasev), ensuring R(0)
  is computed correctly as the paper specifies for the unconstrained check.

### Custom implementation

There is no Python library that directly solves this CVaR-constrained stochastic
control problem with the nested λ/η dual search. The core solver is custom:
- `dp_solver_1d.py`: fully vectorised Bellman recursion using Gauss–Hermite
  quadrature over the wealth state grid.
- `dp_solver_2d.py`: fully vectorised Bellman over the joint (W, φ) state grid
  using `scipy.interpolate.RegularGridInterpolator` for bilinear interpolation.

### Numerical parameters (configurable in `src/config.py`)

| Setting | 1-D (exp1,2) | 2-D (exp3,4) |
|---------|-------------|-------------|
| N_time  | 40          | 15          |
| N_W     | 80          | 25          |
| N_φ     | 60          | 25          |
| N_u     | —           | 30          |
| N_quad  | 7           | 5           |
| N_bisect| 20          | 20          |
| N_golden| 12          | 12          |
| MC (bisect) | 2 000  | 2 000       |
| MC (eval)   | 40 000 | 30 000      |

Increase grid sizes for higher accuracy at the cost of longer runtime.

---

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for the full comparison.

**Key findings:**

| Experiment | Nonbinding (c=−0.86) | Binding (c=−0.94) |
|------------|---------------------|------------------|
| exp1 complete market | λ̂=0, CVaR=−0.872, avg φ=0.39 | λ̂=0.073, CVaR=−0.954, avg φ=0.24 |
| exp2 incomplete market | λ̂=0, CVaR=−0.867, avg φ=0.39 | λ̂=0.281, CVaR=−0.940, avg φ=0.17 |
| exp3 quadratic reg. | λ̂=0, CVaR=−0.893, avg φ=0.31 | λ̂=0.067, CVaR=−0.940, avg φ=0.21 |
| exp4 sqrt impact | λ̂=0, CVaR=−0.887, avg φ=0.34 | λ̂=0.105, CVaR=−0.982, avg φ=0.07 |

All **nonbinding/binding classifications** match the paper. **λ̂ and CVaR values**
are close for 1-D experiments; the 2-D experiments show modest quantisation
error from the coarse η discretisation (see RESULTS.md for analysis).
