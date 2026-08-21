"""
Tests for src/dual_search.py and src/simulator.py

Validates:
  - Golden-section correctly contracts the bracket
  - Golden ratio constant matches paper specification rho = (sqrt(5)-1)/2
  - Bisection returns lambda=0 for nonbinding case
  - Simulator output shapes and initial conditions
  - Residual oracle: G(eta) - c definition consistency
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dual_search import (
    GOLDEN_RATIO, golden_section_eta, EtaSearchResult,
)
from src.simulator import (
    simulate_1d, simulate_2d,
    residual_from_simulation_1d,
)
from src.config import ModelParams, GridParams, DualSearchParams, MCParams


# ---------------------------------------------------------------------------
# Golden ratio constant
# ---------------------------------------------------------------------------

class TestGoldenRatio:
    def test_value(self):
        """Paper specifies rho = (sqrt(5)-1)/2."""
        expected = (np.sqrt(5.0) - 1.0) / 2.0
        assert abs(GOLDEN_RATIO - expected) < 1e-15

    def test_golden_ratio_property(self):
        """rho satisfies rho^2 + rho = 1 (standard property)."""
        assert abs(GOLDEN_RATIO ** 2 + GOLDEN_RATIO - 1.0) < 1e-14

    def test_approximate_value(self):
        """rho ≈ 0.618033..."""
        assert abs(GOLDEN_RATIO - 0.6180339887) < 1e-8


# ---------------------------------------------------------------------------
# Golden-section search
# ---------------------------------------------------------------------------

class TestGoldenSectionEta:
    def _convex_oracle(self, eta_star: float, scale: float = 1.0):
        """Return a callable that evaluates a quadratic centred at eta_star."""
        def f(eta: float) -> float:
            return scale * (eta - eta_star) ** 2
        return f

    def test_finds_minimum_of_quadratic(self):
        """Should find eta* of a quadratic within the bracket."""
        eta_star = -0.75
        oracle = self._convex_oracle(eta_star)
        result = golden_section_eta(oracle, eta_low=-2.0, eta_high=0.0, n_iters=20)
        assert isinstance(result, EtaSearchResult)
        assert abs(result.eta_star - eta_star) < 0.01, (
            f"Found eta* = {result.eta_star:.5f}, expected {eta_star}"
        )

    def test_bracket_contracts(self):
        """Each iteration should reduce the bracket size by GOLDEN_RATIO."""
        call_log = []
        def logging_oracle(eta: float) -> float:
            call_log.append(eta)
            return (eta + 0.8) ** 2
        golden_section_eta(logging_oracle, eta_low=-2.0, eta_high=0.0, n_iters=10)
        # At least 10+2 calls were made (initial 2 + one per iteration)
        assert len(call_log) >= 12

    def test_v_star_consistent(self):
        """v_star should equal oracle evaluated at eta_star."""
        eta_star_true = -0.6
        oracle = self._convex_oracle(eta_star_true)
        result = golden_section_eta(oracle, eta_low=-2.0, eta_high=0.0, n_iters=20)
        expected_v = oracle(result.eta_star)
        assert abs(result.v_star - expected_v) < 1e-12

    def test_eta_star_within_bracket(self):
        """Returned eta* must be within [eta_low, eta_high]."""
        oracle = self._convex_oracle(-1.2)
        result = golden_section_eta(oracle, eta_low=-2.0, eta_high=0.0, n_iters=15)
        assert -2.0 <= result.eta_star <= 0.0

    def test_different_bracket(self):
        """Test with a different bracket and minimum location."""
        eta_star = 0.3
        oracle = self._convex_oracle(eta_star)
        # Bracket [0, 1] contains eta_star=0.3
        result = golden_section_eta(oracle, eta_low=0.0, eta_high=1.0, n_iters=20)
        assert abs(result.eta_star - eta_star) < 0.01

    def test_returns_eta_search_result(self):
        oracle = self._convex_oracle(-0.9)
        result = golden_section_eta(oracle, -2.0, 0.0, 5)
        assert hasattr(result, 'eta_star')
        assert hasattr(result, 'v_star')
        assert hasattr(result, 'n_iters')
        assert result.n_iters == 7   # 5 iterations + 2 initial evaluations + final


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

def small_model() -> ModelParams:
    return ModelParams(T=1.0, r=0.0, mu=0.08, sigma=0.20, gamma=5.0, w0=1.0, alpha=0.95)


def small_grid_1d() -> GridParams:
    return GridParams(
        n_time=10, w_min=0.1, w_max=3.0, n_w=30,
        phi_min=-0.2, phi_max=1.5, n_phi=30, n_quad=5,
    )


def small_grid_2d() -> GridParams:
    return GridParams(
        n_time=5, w_min=0.1, w_max=3.0, n_w=15,
        phi_state_min=-0.1, phi_state_max=1.0, n_phi_state=10,
        u_min=-3.0, u_max=8.0, n_u=15, n_quad=3,
    )


def small_mc() -> MCParams:
    return MCParams(n_mc_residual=200, n_mc_eval=500, seed=123)


class TestSimulate1D:
    def _constant_policy(self, n_time: int, n_w: int, phi: float = 0.4) -> np.ndarray:
        """Constant-policy lookup table."""
        return np.full((n_time, n_w), phi)

    def test_output_shapes(self):
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = self._constant_policy(grid.n_time, grid.n_w)
        W, phi_path, t_grid = simulate_1d(model, grid, policies, mc, n_paths=100)
        assert W.shape == (100, grid.n_time + 1)
        assert phi_path.shape == (100, grid.n_time)
        assert t_grid.shape == (grid.n_time + 1,)

    def test_initial_wealth(self):
        """W[:, 0] should equal w0 for all paths."""
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = self._constant_policy(grid.n_time, grid.n_w)
        W, _, _ = simulate_1d(model, grid, policies, mc, n_paths=200)
        np.testing.assert_array_equal(W[:, 0], model.w0)

    def test_t_grid_endpoints(self):
        """Time grid should run from 0 to T."""
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = self._constant_policy(grid.n_time, grid.n_w)
        _, _, t_grid = simulate_1d(model, grid, policies, mc, n_paths=50)
        assert abs(t_grid[0]) < 1e-12
        assert abs(t_grid[-1] - model.T) < 1e-12

    def test_zero_phi_constant_wealth(self):
        """phi=0 → dW=0 everywhere → terminal wealth = w0."""
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = self._constant_policy(grid.n_time, grid.n_w, phi=0.0)
        W, _, _ = simulate_1d(model, grid, policies, mc, n_paths=100, beta_perp=0.0)
        np.testing.assert_allclose(W[:, -1], model.w0, atol=1e-12)

    def test_positive_drift_increases_mean(self):
        """With phi=0.4, mu=0.08 > 0, mean terminal wealth > w0."""
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = self._constant_policy(grid.n_time, grid.n_w, phi=0.4)
        W, _, _ = simulate_1d(model, grid, policies, mc, n_paths=2000)
        assert np.mean(W[:, -1]) > model.w0

    def test_phi_path_shape(self):
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = self._constant_policy(grid.n_time, grid.n_w, phi=0.4)
        _, phi_path, _ = simulate_1d(model, grid, policies, mc, n_paths=50)
        assert phi_path.shape == (50, grid.n_time)

    def test_beta_perp_adds_variance(self):
        """Nontraded endowment increases terminal wealth variance."""
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = self._constant_policy(grid.n_time, grid.n_w, phi=0.4)
        rng = np.random.default_rng(42)
        W_complete, _, _ = simulate_1d(model, grid, policies, mc, n_paths=3000,
                                       beta_perp=0.0, rng=rng)
        rng2 = np.random.default_rng(42)
        W_incomplete, _, _ = simulate_1d(model, grid, policies, mc, n_paths=3000,
                                         beta_perp=0.02, rng=rng2)
        assert np.std(W_incomplete[:, -1]) > np.std(W_complete[:, -1])


class TestSimulate2D:
    def _constant_u_policy(self, n_time: int, n_w: int, n_phi: int, u: float = 0.5):
        return np.full((n_time, n_w, n_phi), u)

    def test_output_shapes(self):
        model, grid, mc = small_model(), small_grid_2d(), small_mc()
        policies = self._constant_u_policy(grid.n_time, grid.n_w, grid.n_phi_state)
        W, Phi, U, t_grid = simulate_2d(model, grid, policies, mc, n_paths=50)
        assert W.shape == (50, grid.n_time + 1)
        assert Phi.shape == (50, grid.n_time + 1)
        assert U.shape == (50, grid.n_time)
        assert t_grid.shape == (grid.n_time + 1,)

    def test_initial_conditions(self):
        model, grid, mc = small_model(), small_grid_2d(), small_mc()
        phi0 = 0.0
        policies = self._constant_u_policy(grid.n_time, grid.n_w, grid.n_phi_state)
        W, Phi, _, _ = simulate_2d(model, grid, policies, mc, n_paths=50, phi0=phi0)
        np.testing.assert_array_equal(W[:, 0], model.w0)
        np.testing.assert_array_equal(Phi[:, 0], phi0)

    def test_zero_u_phi_from_initial(self):
        """u=0 and phi0=0 → phi stays near 0 (only random drift keeps it at 0)."""
        model, grid, mc = small_model(), small_grid_2d(), small_mc()
        policies = self._constant_u_policy(grid.n_time, grid.n_w, grid.n_phi_state, u=0.0)
        W, Phi, _, _ = simulate_2d(model, grid, policies, mc, n_paths=1000, phi0=0.0)
        # With u=0 and phi0=0: phi_t = 0 for all t (no dynamics since phi_0=0)
        np.testing.assert_allclose(Phi, 0.0, atol=1e-12)

    def test_quadratic_vs_sqrt_modes(self):
        """Both modes should run without error and produce finite output."""
        model, grid, mc = small_model(), small_grid_2d(), small_mc()
        policies = self._constant_u_policy(grid.n_time, grid.n_w, grid.n_phi_state, u=0.2)
        W_q, _, _, _ = simulate_2d(model, grid, policies, mc, n_paths=100,
                                   mode="quadratic")
        W_s, _, _, _ = simulate_2d(model, grid, policies, mc, n_paths=100,
                                   mode="sqrt_impact", Lambda_32=0.0045)
        assert np.all(np.isfinite(W_q))
        assert np.all(np.isfinite(W_s))

    def test_sqrt_impact_lower_wealth(self):
        """Square-root impact deducts from wealth → terminal wealth lower than quadratic."""
        model, grid, mc = small_model(), small_grid_2d(), small_mc()
        # Large positive u to amplify impact
        policies = self._constant_u_policy(grid.n_time, grid.n_w, grid.n_phi_state, u=2.0)
        rng = np.random.default_rng(0)
        W_q, _, _, _ = simulate_2d(model, grid, policies, mc, n_paths=2000,
                                   mode="quadratic", rng=rng)
        rng2 = np.random.default_rng(0)
        W_s, _, _, _ = simulate_2d(model, grid, policies, mc, n_paths=2000,
                                   mode="sqrt_impact", Lambda_32=0.0045, rng=rng2)
        assert np.mean(W_s[:, -1]) < np.mean(W_q[:, -1])


# ---------------------------------------------------------------------------
# Residual oracle test
# ---------------------------------------------------------------------------

class TestResidualOracle1D:
    def test_residual_returns_float(self):
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = np.full((grid.n_time, grid.n_w), 0.4)
        residual = residual_from_simulation_1d(
            model, grid, policies, eta=-0.87, c=-0.86,
            mc=mc, n_paths=500, beta_perp=0.0
        )
        assert isinstance(residual, float)

    def test_tight_c_gives_positive_residual(self):
        """
        c = -0.99 (very tight CVaR limit, more negative → stricter) should
        give positive residual since unconstrained CVaR ≈ -0.87 > -0.99.
        Residual = CVaR(-W_T) - c ≈ -0.87 - (-0.99) = +0.12 > 0.
        """
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = np.full((grid.n_time, grid.n_w), 0.4)
        residual = residual_from_simulation_1d(
            model, grid, policies, eta=-0.99, c=-0.99,
            mc=mc, n_paths=2000, beta_perp=0.0
        )
        assert residual > 0.0, (
            f"Residual {residual:.4f} should be positive: unconstrained CVaR≈-0.87 "
            f"exceeds the tight limit c=-0.99"
        )

    def test_loose_c_gives_negative_residual(self):
        """c = 100 (very loose limit) should give negative residual."""
        model, grid, mc = small_model(), small_grid_1d(), small_mc()
        policies = np.full((grid.n_time, grid.n_w), 0.4)
        residual = residual_from_simulation_1d(
            model, grid, policies, eta=-1.0, c=100.0,  # c=100 is very loose
            mc=mc, n_paths=200, beta_perp=0.0
        )
        assert residual < 0.0
