"""Tests for utils.evaluator — error metrics."""

import numpy as np
import pytest

from core.models import ChannelTruth, RangeEstimate
from utils.evaluator import empty_error_store, range_error_m, summarize_errors


class TestRangeError:
    def test_perfect_estimate_zero_error(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0]),
            tau_paths_s=np.array([33e-9]),
            true_range_m=9.893151114,  # 33e-9 * LIGHT_SPEED_MPS
        )
        est = RangeEstimate(
            algorithm="test",
            protocol="uwb",
            estimated_tof_s=33e-9,
            estimated_range_m=10.0,
        )
        err = range_error_m(est, truth)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_late_estimate_positive_error(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0]),
            tau_paths_s=np.array([33e-9]),  # ~10m
            true_range_m=10.0,
        )
        est = RangeEstimate(
            algorithm="test",
            protocol="uwb",
            estimated_tof_s=36.7e-9,  # ~11m
            estimated_range_m=11.0,
        )
        err = range_error_m(est, truth)
        assert err > 0.5  # ~1m error

    def test_early_estimate_negative_error(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0]),
            tau_paths_s=np.array([50e-9]),
            true_range_m=15.0,  # > estimated_range_m to ensure negative error
        )
        est = RangeEstimate(
            algorithm="test",
            protocol="uwb",
            estimated_tof_s=40e-9,
            estimated_range_m=12.0,
        )
        err = range_error_m(est, truth)
        assert err < 0.0


class TestSummarizeErrors:
    def test_single_error(self):
        errors = {"uwb": [0.5, 0.3, -0.2, 0.1, -0.4]}
        summary = summarize_errors(errors)
        assert "uwb" in summary
        s = summary["uwb"]
        assert "bias_m" in s
        assert "std_m" in s
        assert "rmse_m" in s
        assert "p90_abs_m" in s

    def test_rmse_ge_abs_bias(self):
        """RMSE should be >= |bias| (since RMSE² = bias² + σ²)."""
        errors = {"uwb": [1.0, 1.0, 1.0, 1.0, 1.0]}
        summary = summarize_errors(errors)
        assert summary["uwb"]["rmse_m"] >= abs(summary["uwb"]["bias_m"])

    def test_zero_std_for_constant_errors(self):
        errors = {"uwb": [3.0, 3.0, 3.0]}
        summary = summarize_errors(errors)
        assert summary["uwb"]["std_m"] == pytest.approx(0.0, abs=1e-10)

    def test_p90_bounds(self):
        """P90 of absolute errors should be between median and max."""
        errors = {"wifi": [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0]}
        summary = summarize_errors(errors)
        p90 = summary["wifi"]["p90_abs_m"]
        arr = np.abs(np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0]))
        assert np.median(arr) <= p90 <= np.max(arr)

    def test_multiple_protocols(self):
        errors = {"uwb": [0.1, 0.2], "wifi": [0.5, 0.6, 0.7]}
        summary = summarize_errors(errors)
        assert len(summary) == 2


class TestEmptyErrorStore:
    def test_returns_defaultdict(self):
        store = empty_error_store()
        store["uwb"].append(0.1)
        assert len(store["uwb"]) == 1
        # Accessing missing key creates empty list
        assert store["wifi"] == []
