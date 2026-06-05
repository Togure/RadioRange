"""Tests for utils.runner — simulation pipeline and helpers."""

import numpy as np
import pytest

from core.models import ChannelTruth, LIGHT_SPEED_MPS
from utils.runner import _rng, _guess_bandwidth, run_single_trial, run_monte_carlo


class TestRng:
    """Tests for _rng() — deterministic seeded RNG."""

    def test_same_seed_same_result(self):
        rng1 = _rng(42)
        rng2 = _rng(42)
        a = rng1.random(10)
        b = rng2.random(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_result(self):
        rng1 = _rng(42)
        rng2 = _rng(99)
        a = rng1.random(10)
        b = rng2.random(10)
        assert not np.allclose(a, b)

    def test_suffix_changes_output(self):
        rng1 = _rng(42, 0)
        rng2 = _rng(42, 1)
        a = rng1.random(10)
        b = rng2.random(10)
        assert not np.allclose(a, b)

    def test_negative_seed_handled(self):
        """Negative seeds should be abs'd and wrapped."""
        rng = _rng(-42)
        assert isinstance(rng, np.random.Generator)
        # Should not raise


class TestGuessBandwidth:
    """Tests for _guess_bandwidth() — extract bandwidth from radio config."""

    def test_explicit_bandwidth(self):
        class FakeRadio:
            protocol = "uwb"
            config = {"radios": {"uwb": {"bandwidth_hz": 500e6}}}

        assert _guess_bandwidth(FakeRadio()) == 500e6

    def test_inferred_from_ofdm_params(self):
        class FakeRadio:
            protocol = "wifi"
            config = {"radios": {"wifi": {"subcarrier_spacing_hz": 312_500, "fft_size": 512}}}

        bw = _guess_bandwidth(FakeRadio())
        assert bw == 312_500 * 512  # 160 MHz

    def test_default_ofdm_params(self):
        class FakeRadio:
            protocol = "wifi"
            config = {"radios": {"wifi": {}}}

        bw = _guess_bandwidth(FakeRadio())
        assert bw == 312_500 * 512  # defaults


class TestRunSingleTrial:
    """Tests for run_single_trial() — the core simulation pipeline.

    These are integration-light tests using minimal mocks/real objects.
    """

    def _make_truth(self) -> ChannelTruth:
        """Minimal LOS truth — one direct path with delay 10 ns."""
        delay_s = 10e-9  # ~3 m
        return ChannelTruth(
            a_paths=np.array([0.8 + 0j]),
            tau_paths_s=np.array([delay_s]),
            path_type=np.array(["LOS"], dtype=object),
            path_order=np.array([1]),
            true_range_m=delay_s * LIGHT_SPEED_MPS,
            carrier_frequency_hz=8e9,
            los=True,
        )

    def test_uwb_los_no_impairments(self):
        """UWB + MaxPeak on a clean LOS path — should be near-zero error."""
        from utils.radio_factory import create_radio
        from algorithms import MaxPeakLde

        truth = self._make_truth()
        radio = create_radio("uwb")
        algo = MaxPeakLde()
        cfg = {"impairments": {}, "timing": {}}

        proto, err = run_single_trial(truth, radio, algo, cfg, seed=42)
        assert proto == "uwb"
        # With clean LOS and no impairments, error should be tiny
        assert abs(err) < 0.5, f"Expected near-zero error, got {err:.3f} m"

    def test_returns_protocol_and_float_error(self):
        from utils.radio_factory import create_radio
        from algorithms import ThresholdLde

        truth = self._make_truth()
        radio = create_radio("uwb")
        algo = ThresholdLde(peak_ratio=0.18)
        cfg = {"impairments": {}, "timing": {}}

        proto, err = run_single_trial(truth, radio, algo, cfg, seed=42)
        assert isinstance(proto, str)
        assert isinstance(err, float)

    def test_reproducibility_same_seed(self):
        from utils.radio_factory import create_radio
        from algorithms import LeadingEdgeLde

        truth = self._make_truth()
        radio = create_radio("uwb")
        algo = LeadingEdgeLde(n_sigma=4.0)
        cfg = {"impairments": {}, "timing": {}}

        _, err1 = run_single_trial(truth, radio, algo, cfg, seed=42)
        _, err2 = run_single_trial(truth, radio, algo, cfg, seed=42)
        assert err1 == pytest.approx(err2)

    def test_different_seed_different_error(self):
        """With impairments enabled, different seeds give different errors."""
        from utils.radio_factory import create_radio
        from algorithms import MaxPeakLde

        truth = self._make_truth()
        radio = create_radio("uwb")
        algo = MaxPeakLde()
        cfg = {"impairments": {"enable_sfo": True, "sfo_ppm": 20.0}, "timing": {}}

        _, err1 = run_single_trial(truth, radio, algo, cfg, seed=42)
        _, err2 = run_single_trial(truth, radio, algo, cfg, seed=99)
        # With SFO enabled, different seeds should produce different errors
        assert err1 != err2

    def test_wifi_protocol_works(self):
        from utils.radio_factory import create_radio
        from algorithms import MaxPeakLde

        truth = self._make_truth()
        radio = create_radio("wifi")
        algo = MaxPeakLde()
        cfg = {"impairments": {}, "timing": {}}

        proto, _ = run_single_trial(truth, radio, algo, cfg, seed=42)
        assert proto == "wifi"

    def test_all_five_algorithms(self):
        """Smoke test — all 5 LDE algorithms work on a clean LOS path."""
        from utils.radio_factory import create_radio
        from algorithms import (
            MaxPeakLde, ThresholdLde, LeadingEdgeLde, SearchBackLde, ChipLeadingEdgeLde,
        )

        truth = self._make_truth()
        radio = create_radio("uwb")
        cfg = {"impairments": {}, "timing": {}}

        for algo_cls in [MaxPeakLde, ThresholdLde, LeadingEdgeLde, SearchBackLde, ChipLeadingEdgeLde]:
            algo = algo_cls()
            proto, err = run_single_trial(truth, radio, algo, cfg, seed=42)
            assert proto == "uwb"
            assert isinstance(err, float)


class TestRunMonteCarlo:
    """Tests for run_monte_carlo() — multi-trial orchestration."""

    def _make_truths(self, n=10):
        """Generate n identical LOS truths for UWB."""
        delay_s = 10e-9
        truth = ChannelTruth(
            a_paths=np.array([0.8 + 0j]),
            tau_paths_s=np.array([delay_s]),
            path_type=np.array(["LOS"], dtype=object),
            path_order=np.array([1]),
            true_range_m=delay_s * LIGHT_SPEED_MPS,
            carrier_frequency_hz=8e9,
            los=True,
        )
        return {"uwb": [truth] * n}

    def test_returns_expected_structure(self):
        from utils.radio_factory import create_radio
        from algorithms import MaxPeakLde

        truths = self._make_truths(5)
        radios = {"uwb": create_radio("uwb")}
        algo = MaxPeakLde()
        cfg = {"impairments": {}, "timing": {}}

        errors = run_monte_carlo(truths, radios, algo, cfg, seed=42)
        assert "uwb" in errors
        assert len(errors["uwb"]) == 5
        assert all(isinstance(e, float) for e in errors["uwb"])

    def test_num_trials_limits_count(self):
        from utils.radio_factory import create_radio
        from algorithms import MaxPeakLde

        truths = self._make_truths(10)
        radios = {"uwb": create_radio("uwb")}
        algo = MaxPeakLde()
        cfg = {"impairments": {}, "timing": {}}

        errors = run_monte_carlo(truths, radios, algo, cfg, seed=42, num_trials=3)
        assert len(errors["uwb"]) == 3
