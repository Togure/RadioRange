"""Tests for core.models — data structure correctness."""

import numpy as np
import pytest

from core.models import ChannelTruth, LIGHT_SPEED_MPS, RadioObservation, RangeEstimate


class TestChannelTruth:
    def test_true_first_tau_s_basic(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0 + 0j, 0.5 + 0j, 0.2 + 0j]),
            tau_paths_s=np.array([10e-9, 20e-9, 30e-9]),
            true_range_m=2.99792458,  # 10e-9 * LIGHT_SPEED_MPS
        )
        assert truth.true_first_tau_s == pytest.approx(10e-9)

    def test_true_first_tau_s_zero_gain_path_ignored(self):
        truth = ChannelTruth(
            a_paths=np.array([0.0 + 0j, 0.5 + 0j, 0.2 + 0j]),
            tau_paths_s=np.array([5e-9, 20e-9, 30e-9]),
            true_range_m=5.99584916,  # 20e-9 * LIGHT_SPEED_MPS
        )
        assert truth.true_first_tau_s == pytest.approx(20e-9)

    def test_true_first_tau_s_all_zero_returns_nan(self):
        truth = ChannelTruth(
            a_paths=np.array([0.0 + 0j, 0.0 + 0j]),
            tau_paths_s=np.array([10e-9, 20e-9]),
        )
        assert np.isnan(truth.true_first_tau_s)

    def test_metadata_default(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0]),
            tau_paths_s=np.array([10e-9]),
        )
        assert truth.metadata == {}

    def test_metadata_custom(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0]),
            tau_paths_s=np.array([10e-9]),
            metadata={"source": "test"},
        )
        assert truth.metadata["source"] == "test"

    def test_angle_fields_none_by_default(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0]),
            tau_paths_s=np.array([10e-9]),
        )
        assert truth.aoa_azimuth_deg is None
        assert truth.aoa_elevation_deg is None
        assert truth.aod_azimuth_deg is None
        assert truth.aod_elevation_deg is None

    def test_frozen_dataclass(self):
        truth = ChannelTruth(
            a_paths=np.array([1.0]),
            tau_paths_s=np.array([10e-9]),
        )
        with pytest.raises(Exception):
            truth.true_range_m = 999.0  # type: ignore[misc]


class TestRadioObservation:
    @pytest.fixture
    def obs(self):
        n = 64
        return RadioObservation(
            protocol="uwb",
            t_discrete_s=np.arange(n, dtype=float) / 500e6,
            t_cont_s=np.arange(n * 4, dtype=float) / (500e6 * 4),
            frequency_hz=np.fft.fftfreq(n, d=1 / 500e6),
            h_clean=np.ones(n, dtype=np.complex128),
            h_observed=np.ones(n, dtype=np.complex128),
            cir_clean_discrete=np.zeros(n),
            cir_observed_discrete=np.zeros(n),
            cir_clean_cont=np.zeros(n * 4),
            cir_observed_cont=np.zeros(n * 4),
        )

    def test_backward_compat_h_frequency(self, obs):
        assert np.array_equal(obs.h_frequency, obs.h_observed)

    def test_backward_compat_cir_discrete(self, obs):
        assert np.array_equal(obs.cir_discrete, obs.cir_observed_discrete)

    def test_backward_compat_cir_cont(self, obs):
        assert np.array_equal(obs.cir_cont, obs.cir_observed_cont)


class TestRangeEstimate:
    def test_basic(self):
        est = RangeEstimate(
            algorithm="threshold",
            protocol="uwb",
            estimated_tof_s=33e-9,
            estimated_range_m=10.0,
        )
        assert est.algorithm == "threshold"
        assert est.protocol == "uwb"
        assert est.estimated_tof_s == pytest.approx(33e-9)
        assert est.estimated_range_m == pytest.approx(10.0)

    def test_tof_to_range_conversion(self):
        tof = 10.0 / LIGHT_SPEED_MPS
        est = RangeEstimate(
            algorithm="max_peak",
            protocol="wifi",
            estimated_tof_s=tof,
            estimated_range_m=tof * LIGHT_SPEED_MPS,
        )
        assert est.estimated_range_m == pytest.approx(10.0)
