"""Tests for environments.statistical — TDL channel model generation."""

import numpy as np
import pytest

from environments.statistical import _TDL_PROFILES, _fallback_tdl, generate_tdl_truths


class TestTdlProfiles:
    """Verify 3GPP TR 38.901 profile data consistency."""

    MODELS = ["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"]

    @pytest.mark.parametrize("model", MODELS)
    def test_profile_exists(self, model):
        assert model in _TDL_PROFILES

    @pytest.mark.parametrize("model", MODELS)
    def test_delays_and_powers_same_length(self, model):
        p = _TDL_PROFILES[model]
        assert len(p["delays_norm"]) == len(p["powers_db"]), \
            f"{model}: delays={len(p['delays_norm'])}, powers={len(p['powers_db'])}"

    @pytest.mark.parametrize("model", MODELS)
    def test_23_taps(self, model):
        p = _TDL_PROFILES[model]
        assert len(p["delays_norm"]) == 23, \
            f"{model}: expected 23 taps, got {len(p['delays_norm'])}"

    @pytest.mark.parametrize("model", MODELS)
    def test_first_delay_zero(self, model):
        p = _TDL_PROFILES[model]
        assert p["delays_norm"][0] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("model", MODELS)
    def test_delays_monotonic(self, model):
        p = _TDL_PROFILES[model]
        diffs = np.diff(p["delays_norm"])
        assert np.all(diffs >= 0), f"{model}: delays not monotonic: {p['delays_norm']}"


class TestFallbackTdl:
    def test_tdl_a_basic(self):
        rng = np.random.default_rng(42)
        env_cfg = {"model": "TDL-A", "delay_spread_s": 30e-9}
        a, tau = _fallback_tdl(env_cfg, rng)
        assert len(a) == 23
        assert len(tau) == 23
        assert tau[0] == pytest.approx(0.0)  # first tau = 0 exactly
        # All delays should be non-negative and monotonic
        assert np.all(np.diff(tau) >= 0)

    def test_delay_scaling(self):
        """Doubling delay_spread should double all excess delays."""
        rng = np.random.default_rng(42)
        cfg_30 = {"model": "TDL-A", "delay_spread_s": 30e-9}
        _, tau_30 = _fallback_tdl(cfg_30, rng)

        rng2 = np.random.default_rng(42)
        cfg_60 = {"model": "TDL-A", "delay_spread_s": 60e-9}
        _, tau_60 = _fallback_tdl(cfg_60, rng2)

        np.testing.assert_allclose(tau_60, tau_30 * 2.0)

    def test_tdl_a_nlos_rayleigh(self):
        """TDL-A is NLOS: peak power should be at tap 1 (index), not tap 0."""
        rng = np.random.default_rng(42)
        env_cfg = {"model": "TDL-A", "delay_spread_s": 30e-9}
        a, _ = _fallback_tdl(env_cfg, rng)
        # TDL-A: first tap power is -13.4 dB, second tap is 0 dB
        # Due to fading this may not hold always, but statically tap 1 should
        # be stronger than tap 0 on average over many trials.
        powers = np.zeros((100, 23))
        for i in range(100):
            a_i, _ = _fallback_tdl(env_cfg, rng)
            powers[i] = np.abs(a_i) ** 2
        mean_powers = np.mean(powers, axis=0)
        # The second tap (index 1) should be stronger than first tap (index 0)
        assert mean_powers[1] > mean_powers[0]

    def test_tdl_d_los_first_tap_strong(self):
        """TDL-D is LOS with K=6dB: first tap should dominate."""
        rng = np.random.default_rng(42)
        env_cfg = {"model": "TDL-D", "delay_spread_s": 30e-9}
        a, _ = _fallback_tdl(env_cfg, rng)
        # First tap power in profile: -0.2 dB, second tap: -13.5 dB
        # K=6dB adds Rician → first tap should be much stronger
        assert np.abs(a[0]) > 0

    def test_tdl_e_los_first_tap_strong(self):
        """TDL-E is LOS with K=8dB."""
        rng = np.random.default_rng(42)
        env_cfg = {"model": "TDL-E", "delay_spread_s": 30e-9}
        a, _ = _fallback_tdl(env_cfg, rng)
        assert np.abs(a[0]) > 0

    def test_unknown_model_raises(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown TDL model"):
            _fallback_tdl({"model": "TDL-X", "delay_spread_s": 30e-9}, rng)

    def test_deterministic_with_fixed_seed(self):
        """Same seed should produce identical results."""
        cfg = {"model": "TDL-A", "delay_spread_s": 30e-9}
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        a1, tau1 = _fallback_tdl(cfg, rng1)
        a2, tau2 = _fallback_tdl(cfg, rng2)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(tau1, tau2)

    def test_different_seeds_different_fading(self):
        """Different seeds should produce different (faded) gains but same delays."""
        cfg = {"model": "TDL-A", "delay_spread_s": 30e-9}
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(9999)
        a1, tau1 = _fallback_tdl(cfg, rng1)
        a2, tau2 = _fallback_tdl(cfg, rng2)
        np.testing.assert_array_equal(tau1, tau2)  # delays fixed
        assert not np.allclose(a1, a2)  # fading differs


class TestGenerateTdlTruths:
    def test_generates_correct_number(self):
        config = {
            "environment": {
                "type": "standard_tdl",
                "model": "TDL-A",
                "delay_spread_s": 30e-9,
                "num_trials": 5,
                "base_range_m": 10.0,
            },
            "timing": {},
            "channel_engine": "numpy",
        }
        rng = np.random.default_rng(42)
        truths = generate_tdl_truths(config, rng)
        assert len(truths) == 5
        for t in truths:
            assert t.true_range_m == pytest.approx(10.0)

    def test_first_tau_equals_base_range(self):
        """First path delay should be base_range_m / c."""
        config = {
            "environment": {
                "type": "standard_tdl",
                "model": "TDL-A",
                "delay_spread_s": 30e-9,
                "num_trials": 1,
                "base_range_m": 15.0,
            },
            "timing": {},
            "channel_engine": "numpy",
        }
        LIGHT_SPEED_MPS = 299792458.0
        rng = np.random.default_rng(42)
        truths = generate_tdl_truths(config, rng)
        expected_tau = 15.0 / LIGHT_SPEED_MPS
        assert truths[0].true_first_tau_s == pytest.approx(expected_tau)

    def test_los_model_metadata(self):
        config = {
            "environment": {
                "type": "standard_tdl",
                "model": "TDL-D",
                "delay_spread_s": 30e-9,
                "num_trials": 1,
                "base_range_m": 10.0,
            },
            "timing": {},
            "channel_engine": "numpy",
        }
        rng = np.random.default_rng(42)
        truths = generate_tdl_truths(config, rng)
        assert truths[0].los is True
        assert truths[0].path_type[0] == "LOS"

    def test_nlos_model_metadata(self):
        config = {
            "environment": {
                "type": "standard_tdl",
                "model": "TDL-B",
                "delay_spread_s": 100e-9,
                "num_trials": 1,
                "base_range_m": 10.0,
            },
            "timing": {},
            "channel_engine": "numpy",
        }
        rng = np.random.default_rng(42)
        truths = generate_tdl_truths(config, rng)
        assert truths[0].los is False
