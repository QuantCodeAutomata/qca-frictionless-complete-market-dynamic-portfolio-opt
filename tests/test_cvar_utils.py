"""
Tests for src/cvar_utils.py

Validates:
  - CVaR sample estimator mathematical properties
  - Rockafellar-Uryasev threshold form: G(eta) >= CVaR_alpha
  - Threshold consistency with sample estimator for known distributions
  - Edge cases (small samples, deterministic loss, extreme alpha)
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvar_utils import (
    cvar_sample, cvar_threshold_form, cvar_optimal_eta,
    var_sample, cvar_constraint_value,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_normal_losses(n: int, mean: float = 0.0, std: float = 1.0,
                       seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


# ---------------------------------------------------------------------------
# cvar_sample tests
# ---------------------------------------------------------------------------

class TestCVaRSample:
    def test_returns_float(self):
        losses = make_normal_losses(1000)
        result = cvar_sample(losses, 0.95)
        assert isinstance(result, float)

    def test_cvar_ge_var(self):
        """CVaR >= VaR always."""
        losses = make_normal_losses(5000, seed=1)
        alpha = 0.95
        assert cvar_sample(losses, alpha) >= var_sample(losses, alpha) - 1e-8

    def test_cvar_ge_mean(self):
        """CVaR >= mean(losses) for any alpha > 0."""
        losses = make_normal_losses(2000, seed=2)
        assert cvar_sample(losses, 0.8) >= np.mean(losses)

    def test_higher_alpha_higher_cvar(self):
        """CVaR is monotone in alpha."""
        losses = make_normal_losses(5000, seed=3)
        cvar80 = cvar_sample(losses, 0.80)
        cvar95 = cvar_sample(losses, 0.95)
        cvar99 = cvar_sample(losses, 0.99)
        assert cvar80 <= cvar95 + 1e-8
        assert cvar95 <= cvar99 + 1e-8

    def test_constant_losses(self):
        """All losses equal → CVaR equals that constant."""
        c = 3.14
        losses = np.full(1000, c)
        result = cvar_sample(losses, 0.95)
        assert abs(result - c) < 1e-10

    def test_single_element(self):
        """Single loss sample → CVaR equals that loss."""
        losses = np.array([2.5])
        assert abs(cvar_sample(losses, 0.95) - 2.5) < 1e-10

    def test_alpha_validation(self):
        """alpha outside (0,1) should raise AssertionError."""
        losses = make_normal_losses(100)
        with pytest.raises(AssertionError):
            cvar_sample(losses, 0.0)
        with pytest.raises(AssertionError):
            cvar_sample(losses, 1.0)

    def test_negative_losses_wealth_context(self):
        """In our context losses = -W_T can be negative (wealth > 1)."""
        rng = np.random.default_rng(42)
        W_T = 1.0 + rng.normal(0.04, 0.1, 10000)   # centered near 1.04
        losses = -W_T
        cvar = cvar_sample(losses, 0.95)
        # CVaR of loss ≈ -P5 of W_T: for normal losses centered at ~-1.04, CVaR < -1.0
        assert cvar < 0.0, "CVaR of negative losses should be negative for positive wealth"

    def test_regression_standard_normal(self):
        """For large N(0,1) sample, CVaR_0.95 ≈ 2.063."""
        rng = np.random.default_rng(0)
        losses = rng.standard_normal(200_000)
        cvar = cvar_sample(losses, 0.95)
        # Analytical: phi(qnorm(0.95)) / 0.05 = 2.063
        assert abs(cvar - 2.063) < 0.05, f"Expected ≈2.063, got {cvar:.4f}"


# ---------------------------------------------------------------------------
# cvar_threshold_form tests
# ---------------------------------------------------------------------------

class TestCVaRThresholdForm:
    def test_threshold_form_ge_sample_cvar(self):
        """G(eta) >= CVaR_alpha for any eta."""
        losses = make_normal_losses(5000, seed=10)
        alpha = 0.95
        cvar = cvar_sample(losses, alpha)
        for eta in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            g = cvar_threshold_form(losses, alpha, eta)
            assert g >= cvar - 1e-8, f"G({eta}) = {g:.4f} < CVaR = {cvar:.4f}"

    def test_optimal_threshold_matches_sample_cvar(self):
        """Minimising G(eta) over eta should recover CVaR_alpha(L)."""
        rng = np.random.default_rng(5)
        losses = rng.exponential(1.0, 10000)
        alpha = 0.90
        cvar_direct = cvar_sample(losses, alpha)
        cvar_thresh, eta_star = cvar_optimal_eta(losses, alpha, n_grid=1000)
        # Should be close (within Monte Carlo + grid error)
        assert abs(cvar_thresh - cvar_direct) < 0.02, (
            f"Threshold CVaR {cvar_thresh:.4f} vs sample CVaR {cvar_direct:.4f}"
        )

    def test_threshold_at_var_is_tight(self):
        """G evaluated at VaR_alpha should be close to CVaR_alpha."""
        rng = np.random.default_rng(7)
        losses = rng.standard_normal(50000)
        alpha = 0.95
        eta_star = var_sample(losses, alpha)
        g_at_var = cvar_threshold_form(losses, alpha, eta_star)
        cvar = cvar_sample(losses, alpha)
        assert abs(g_at_var - cvar) < 0.05


# ---------------------------------------------------------------------------
# cvar_constraint_value tests
# ---------------------------------------------------------------------------

class TestCVaRConstraintValue:
    def test_positive_when_cvar_exceeds_limit(self):
        """If CVaR > c, residual > 0 (constraint violated)."""
        losses = make_normal_losses(5000, seed=20)
        alpha = 0.95
        c = -5.0   # very tight limit (most losses positive, CVaR > c)
        eta = var_sample(losses, alpha)
        residual = cvar_constraint_value(losses, alpha, eta, c)
        assert residual > 0.0

    def test_negative_when_cvar_below_limit(self):
        """If CVaR < c, residual < 0 (constraint not violated)."""
        losses = np.full(1000, -0.9)   # all wealth well above zero
        alpha = 0.95
        c = -0.5   # very loose limit
        eta = -0.9
        residual = cvar_constraint_value(losses, alpha, eta, c)
        # G(-0.9) = -0.9 + 0/(0.05) = -0.9; residual = -0.9 - (-0.5) = -0.4 < 0
        assert residual < 0.0

    def test_zero_residual_at_exact_constraint(self):
        """When CVaR exactly equals c, residual ≈ 0."""
        losses = np.full(1000, 0.5)   # constant loss 0.5
        alpha = 0.95
        c = 0.5    # CVaR = 0.5 (constant losses)
        eta = 0.5  # G(0.5) = 0.5 + 0 = 0.5
        residual = cvar_constraint_value(losses, alpha, eta, c)
        assert abs(residual) < 1e-10


# ---------------------------------------------------------------------------
# var_sample tests
# ---------------------------------------------------------------------------

class TestVaRSample:
    def test_var_in_range(self):
        losses = make_normal_losses(5000)
        var = var_sample(losses, 0.95)
        # For N(0,1), VaR_0.95 ≈ 1.645
        assert 1.4 < var < 1.9

    def test_var_less_than_cvar(self):
        losses = make_normal_losses(5000, seed=99)
        alpha = 0.95
        assert var_sample(losses, alpha) <= cvar_sample(losses, alpha) + 1e-8
