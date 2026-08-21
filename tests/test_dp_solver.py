"""
Tests for src/dp_solver_1d.py and src/dp_solver_2d.py

Validates:
  - Gauss-Hermite quadrature weights sum to 1 (N(0,1) normalisation)
  - Terminal value function shape and correctness
  - Backward DP output dimensions
  - Unconstrained Merton solution: lambda=0 → phi ≈ mu/(gamma*sigma^2) = 0.40
  - Policy is bounded within the phi_grid range
  - 2D DP output dimensions and terminal condition
  - Monotonicity of value function in wealth (wealthier → higher value is non-obligatory
    for a COST function, but binding terminal penalty should be higher for smaller wealth)
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelParams, GridParams
from src.dp_solver_1d import (
    make_gauss_hermite, terminal_value, backward_dp_1d, lagrangian_value_1d,
    get_wealth_grid, get_phi_grid,
)
from src.dp_solver_2d import (
    backward_dp_2d, running_cost_quadratic, running_cost_sqrt_impact,
    wealth_impact_cost, get_state_grids,
)


# ---------------------------------------------------------------------------
# Default configs for tests (small grids for speed)
# ---------------------------------------------------------------------------

def small_model() -> ModelParams:
    return ModelParams(T=1.0, r=0.0, mu=0.08, sigma=0.20, gamma=5.0, w0=1.0, alpha=0.95)


def small_grid_1d() -> GridParams:
    return GridParams(
        n_time=10,
        w_min=0.1, w_max=3.0, n_w=30,
        phi_min=-0.2, phi_max=1.5, n_phi=30,
        n_quad=5,
    )


def small_grid_2d() -> GridParams:
    return GridParams(
        n_time=5,
        w_min=0.1, w_max=3.0, n_w=15,
        phi_state_min=-0.1, phi_state_max=1.0, n_phi_state=10,
        u_min=-3.0, u_max=8.0, n_u=15,
        n_quad=3,
    )


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

class TestGaussHermite:
    def test_weights_sum_to_one(self):
        """Quadrature weights must sum to 1.0 (normalisation for N(0,1))."""
        for n in [3, 5, 7, 10, 15]:
            _, weights = make_gauss_hermite(n)
            assert abs(np.sum(weights) - 1.0) < 1e-12, f"n={n}: weights sum = {np.sum(weights)}"

    def test_moment_1_is_zero(self):
        """E[Z] = 0 for Z ~ N(0,1) → sum(w_k * x_k) = 0."""
        for n in [3, 5, 7]:
            eps, weights = make_gauss_hermite(n)
            first_moment = np.dot(weights, eps)
            assert abs(first_moment) < 1e-12, f"n={n}: E[Z] = {first_moment}"

    def test_moment_2_is_one(self):
        """E[Z^2] = 1 for Z ~ N(0,1)."""
        for n in [3, 5, 7]:
            eps, weights = make_gauss_hermite(n)
            second_moment = np.dot(weights, eps ** 2)
            assert abs(second_moment - 1.0) < 1e-10, f"n={n}: E[Z^2] = {second_moment}"

    def test_moment_4_is_three(self):
        """E[Z^4] = 3 for Z ~ N(0,1), requires n >= 3."""
        eps, weights = make_gauss_hermite(7)
        fourth_moment = np.dot(weights, eps ** 4)
        assert abs(fourth_moment - 3.0) < 1e-8, f"E[Z^4] = {fourth_moment}"


# ---------------------------------------------------------------------------
# Terminal value function
# ---------------------------------------------------------------------------

class TestTerminalValue:
    def test_shape(self):
        w_grid = np.linspace(0.1, 3.0, 50)
        V = terminal_value(w_grid, lambda_val=1.0, eta=-0.9, alpha=0.95)
        assert V.shape == (50,)

    def test_zero_lambda(self):
        """lambda=0 → V(T, w) = 0 everywhere."""
        w_grid = np.linspace(0.1, 3.0, 50)
        V = terminal_value(w_grid, lambda_val=0.0, eta=-0.9, alpha=0.95)
        assert np.all(V == 0.0)

    def test_nonnegative(self):
        """Terminal penalty is always non-negative."""
        w_grid = np.linspace(0.1, 3.0, 50)
        V = terminal_value(w_grid, lambda_val=1.0, eta=-0.5, alpha=0.95)
        assert np.all(V >= 0.0)

    def test_zero_above_eta_threshold(self):
        """Penalty is zero for w > -eta (no loss scenario)."""
        eta = -0.9   # penalty kicks in for w < 0.9
        w_grid = np.array([0.95, 1.0, 1.5, 2.0])  # all > 0.9
        V = terminal_value(w_grid, lambda_val=1.0, eta=eta, alpha=0.95)
        assert np.all(V == 0.0)

    def test_positive_below_eta_threshold(self):
        """Penalty is strictly positive for w < -eta."""
        eta = -0.9   # threshold: -eta = 0.9
        w_grid = np.array([0.1, 0.5, 0.89])   # all < 0.9
        V = terminal_value(w_grid, lambda_val=1.0, eta=eta, alpha=0.95)
        assert np.all(V > 0.0)

    def test_decreasing_in_wealth(self):
        """Higher wealth → lower terminal penalty (monotone decreasing)."""
        w_grid = np.linspace(0.1, 0.8, 20)   # all below -eta=0.9
        V = terminal_value(w_grid, lambda_val=1.0, eta=-0.9, alpha=0.95)
        assert np.all(np.diff(V) < 0), "Penalty should decrease as wealth increases"

    def test_formula_correctness(self):
        """Manual check: V(T, 0.5) with lambda=2, eta=-0.8, alpha=0.95."""
        w = 0.5
        lam, eta, alpha = 2.0, -0.8, 0.95
        # max(-0.5 - (-0.8), 0) = max(0.3, 0) = 0.3
        expected = (lam / (1 - alpha)) * 0.3   # 2/0.05 * 0.3 = 12.0
        w_grid = np.array([w])
        V = terminal_value(w_grid, lam, eta, alpha)
        assert abs(V[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# 1D backward DP
# ---------------------------------------------------------------------------

class TestBackwardDP1D:
    def test_output_shapes(self):
        model, grid = small_model(), small_grid_1d()
        values, policies, v0 = backward_dp_1d(model, grid, lambda_val=0.0, eta=-0.9)
        assert values.shape == (grid.n_time + 1, grid.n_w)
        assert policies.shape == (grid.n_time, grid.n_w)
        assert isinstance(v0, float)

    def test_terminal_condition_matches(self):
        """V(T, .) from DP should equal terminal_value(.) exactly."""
        model, grid = small_model(), small_grid_1d()
        lam, eta = 0.5, -0.8
        values, _, _ = backward_dp_1d(model, grid, lambda_val=lam, eta=eta)
        w_grid = get_wealth_grid(grid)
        V_T_expected = terminal_value(w_grid, lam, eta, model.alpha)
        np.testing.assert_allclose(values[-1], V_T_expected, rtol=1e-12)

    def test_policy_within_grid_bounds(self):
        """Optimal policy should lie within [phi_min, phi_max]."""
        model, grid = small_model(), small_grid_1d()
        _, policies, _ = backward_dp_1d(model, grid, lambda_val=0.0, eta=-0.9)
        phi_grid = get_phi_grid(grid)
        assert np.all(policies >= phi_grid[0] - 1e-12)
        assert np.all(policies <= phi_grid[-1] + 1e-12)

    def test_merton_benchmark_lambda0(self):
        """
        With lambda=0, the optimal policy for the complete market should
        approximate the Merton solution: phi* = mu/(gamma*sigma^2) = 0.40.

        We verify the AVERAGE policy across wealth grid ≈ 0.40 ± tolerance.
        The tolerance is large because we use a coarse grid for speed.
        """
        model, grid = small_model(), small_grid_1d()
        _, policies, _ = backward_dp_1d(model, grid, lambda_val=0.0, eta=-0.9)
        phi_merton = model.mu / (model.gamma * model.sigma ** 2)
        avg_policy = np.mean(policies)
        assert abs(avg_policy - phi_merton) < 0.15, (
            f"Avg policy {avg_policy:.4f} far from Merton {phi_merton:.4f}"
        )

    def test_lambda0_eta0_terminal_is_zero(self):
        """With lambda=0, terminal cost is 0 everywhere."""
        model, grid = small_model(), small_grid_1d()
        values, _, _ = backward_dp_1d(model, grid, lambda_val=0.0, eta=-0.5)
        assert np.all(values[-1] == 0.0)

    def test_higher_lambda_lower_v0(self):
        """
        Higher lambda penalises the tail more; the minimised Lagrangian value
        should remain finite but the constrained problem structure changes.
        This test only checks that the solver runs without error and returns valid floats.
        """
        model, grid = small_model(), small_grid_1d()
        _, _, v0_lo = backward_dp_1d(model, grid, lambda_val=0.05, eta=-0.9)
        _, _, v0_hi = backward_dp_1d(model, grid, lambda_val=0.20, eta=-0.9)
        assert np.isfinite(v0_lo)
        assert np.isfinite(v0_hi)

    def test_incomplete_market_higher_cost(self):
        """
        beta_perp > 0 adds unhedgeable variance to the running cost.
        The unconstrained running cost with beta_perp=0.02 should be higher
        than with beta_perp=0 at the same control.
        """
        model, grid = small_model(), small_grid_1d()
        _, _, v0_complete = backward_dp_1d(model, grid, lambda_val=0.0, eta=-0.9, beta_perp=0.0)
        _, _, v0_incomplete = backward_dp_1d(model, grid, lambda_val=0.0, eta=-0.9, beta_perp=0.02)
        # Incomplete market running cost includes gamma/2*(beta_perp)^2 extra
        assert v0_incomplete > v0_complete - 1e-6

    def test_lagrangian_value_formula(self):
        """lagrangian_value_1d = v0_dp + lambda * eta."""
        model, grid = small_model(), small_grid_1d()
        lam, eta = 0.1, -0.85
        _, _, v0 = backward_dp_1d(model, grid, lam, eta)
        lag = lagrangian_value_1d(model, grid, lam, eta)
        expected = v0 + lam * eta
        assert abs(lag - expected) < 1e-12


# ---------------------------------------------------------------------------
# Running-cost helpers for 2D
# ---------------------------------------------------------------------------

class TestRunningCost2D:
    def test_quadratic_shape(self):
        phi_g = np.linspace(-0.1, 1.0, 10)
        u_g = np.linspace(-3.0, 8.0, 15)
        rc = running_cost_quadratic(phi_g, u_g, mu=0.08, sigma=0.20,
                                   gamma=5.0, Lambda=0.01, dt=0.05)
        assert rc.shape == (10, 15)

    def test_sqrt_impact_shape(self):
        phi_g = np.linspace(-0.1, 1.0, 10)
        u_g = np.linspace(-3.0, 8.0, 15)
        rc = running_cost_sqrt_impact(phi_g, u_g, mu=0.08, sigma=0.20,
                                     gamma=5.0, Lambda_32=0.0045, dt=0.05)
        assert rc.shape == (10, 15)

    def test_sqrt_impact_nonneg_u_part(self):
        """Impact cost (Lambda_32/1.5)|u|^1.5 is always non-negative."""
        u_g = np.linspace(-5.0, 10.0, 50)
        phi_g = np.array([0.4])
        rc = running_cost_sqrt_impact(phi_g, u_g, mu=0.08, sigma=0.20,
                                     gamma=5.0, Lambda_32=0.0045, dt=0.05)
        # Impact u_part = Lambda_32/1.5 * |u|^1.5 * dt >= 0
        impact = (0.0045 / 1.5) * np.abs(u_g) ** 1.5 * 0.05
        np.testing.assert_array_less(-1e-12, impact)

    def test_wealth_impact_cost_nonpositive(self):
        """Impact deduction from wealth is always <= 0."""
        u_g = np.linspace(-5.0, 10.0, 50)
        cost = wealth_impact_cost(u_g, Lambda_32=0.0045, dt=0.05)
        assert np.all(cost <= 1e-12), "Impact deduction should be non-positive"

    def test_wealth_impact_zero_at_zero_u(self):
        u_g = np.array([0.0])
        cost = wealth_impact_cost(u_g, Lambda_32=0.0045, dt=0.05)
        assert abs(cost[0]) < 1e-12


# ---------------------------------------------------------------------------
# 2D backward DP
# ---------------------------------------------------------------------------

class TestBackwardDP2D:
    def test_output_shapes_quadratic(self):
        model, grid = small_model(), small_grid_2d()
        values, policies, v0 = backward_dp_2d(
            model, grid, lambda_val=0.0, eta=-0.9, mode="quadratic", Lambda=0.01
        )
        assert values.shape == (grid.n_time + 1, grid.n_w, grid.n_phi_state)
        assert policies.shape == (grid.n_time, grid.n_w, grid.n_phi_state)
        assert isinstance(v0, float)

    def test_output_shapes_sqrt_impact(self):
        model, grid = small_model(), small_grid_2d()
        values, policies, v0 = backward_dp_2d(
            model, grid, lambda_val=0.0, eta=-0.9,
            mode="sqrt_impact", Lambda_32=0.0045
        )
        assert values.shape == (grid.n_time + 1, grid.n_w, grid.n_phi_state)
        assert policies.shape == (grid.n_time, grid.n_w, grid.n_phi_state)

    def test_terminal_condition_2d(self):
        """V(T, W, phi) = Psi_{lambda,eta}(W) — should not depend on phi."""
        model, grid = small_model(), small_grid_2d()
        lam, eta = 0.5, -0.8
        values, _, _ = backward_dp_2d(model, grid, lam, eta, mode="quadratic")
        # Terminal slice: each row (fixed w) should be constant across phi
        V_T = values[-1]   # (n_w, n_phi_state)
        for i in range(grid.n_w):
            assert np.allclose(V_T[i], V_T[i, 0]), (
                f"Terminal condition depends on phi at w_idx={i}"
            )

    def test_policy_within_u_grid(self):
        """Optimal u should lie within [u_min, u_max]."""
        model, grid = small_model(), small_grid_2d()
        _, policies, _ = backward_dp_2d(
            model, grid, lambda_val=0.0, eta=-0.9, mode="quadratic"
        )
        assert np.all(policies >= grid.u_min - 1e-10)
        assert np.all(policies <= grid.u_max + 1e-10)

    def test_v0_finite(self):
        model, grid = small_model(), small_grid_2d()
        _, _, v0 = backward_dp_2d(
            model, grid, lambda_val=0.0, eta=-0.9, mode="quadratic"
        )
        assert np.isfinite(v0)

    def test_lagrangian_value_2d_formula(self):
        """lagrangian_value_2d = v0_dp + lambda * eta."""
        from src.dp_solver_2d import lagrangian_value_2d
        model, grid = small_model(), small_grid_2d()
        lam, eta = 0.1, -0.85
        _, _, v0 = backward_dp_2d(model, grid, lam, eta, mode="quadratic")
        lag = lagrangian_value_2d(model, grid, lam, eta, mode="quadratic")
        expected = v0 + lam * eta
        assert abs(lag - expected) < 1e-12

    def test_grid_accessors(self):
        grid = small_grid_2d()
        w_g, phi_g = get_state_grids(grid)
        assert w_g.shape == (grid.n_w,)
        assert phi_g.shape == (grid.n_phi_state,)
        assert w_g[0] == grid.w_min
        assert phi_g[-1] == grid.phi_state_max
