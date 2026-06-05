"""Tests for core.transceivers.common — signal processing primitives."""

import numpy as np
import pytest

from core.transceivers.common import continuous_ofdm, continuous_uwb, path_response


class TestPathResponse:
    def test_single_path_unity_gain(self):
        freq = np.fft.fftfreq(64, d=1 / 20e6)
        a = np.array([1.0 + 0j])
        tau = np.array([0.0])
        h = path_response(freq, a, tau)
        np.testing.assert_allclose(h, 1.0 + 0j)

    def test_single_path_with_delay(self):
        """Path with delay d should produce H(f) = exp(-j*2*pi*f*d)."""
        freq = np.array([1e6, 2e6, 5e6])
        a = np.array([1.0 + 0j])
        tau = np.array([100e-9])
        h = path_response(freq, a, tau)
        expected = np.exp(-1j * 2.0 * np.pi * freq * 100e-9)
        np.testing.assert_allclose(h, expected)

    def test_two_paths_sum(self):
        freq = np.array([0.0, 1e6, 2e6])
        a = np.array([0.8 + 0j, 0.4 + 0j])
        tau = np.array([0.0, 50e-9])
        h = path_response(freq, a, tau)
        expected = 0.8 + 0.4 * np.exp(-1j * 2.0 * np.pi * freq * 50e-9)
        np.testing.assert_allclose(h, expected)

    def test_complex_gain(self):
        freq = np.array([1e6])
        a = np.array([0.6 + 0.8j])  # magnitude 1.0
        tau = np.array([0.0])
        h = path_response(freq, a, tau)
        assert abs(h[0]) == pytest.approx(1.0)

    def test_output_dtype(self):
        freq = np.fft.fftfreq(64, d=1 / 20e6)
        a = np.array([1.0 + 0j])
        tau = np.array([10e-9])
        h = path_response(freq, a, tau)
        assert h.dtype == np.complex128

    def test_empty_paths_returns_zeros(self):
        freq = np.array([1e6, 2e6])
        a = np.array([], dtype=np.complex128)
        tau = np.array([], dtype=float)
        h = path_response(freq, a, tau)
        np.testing.assert_allclose(h, 0.0)


class TestContinuousOfdm:
    def test_zero_padding_preserves_peak_position(self):
        """Zero-padded IFFT should interpolate without shifting the peak."""
        n = 64
        factor = 4
        # Create a simple H(f) that gives a clean peak in CIR
        h = np.ones(n, dtype=np.complex128)
        cir_discrete = np.abs(np.fft.ifft(h))
        cir_cont = np.abs(continuous_ofdm(h, factor))

        # Peak should be at index 0 for both
        assert np.argmax(cir_discrete) == 0
        assert np.argmax(cir_cont) == 0

    def test_output_length(self):
        n = 64
        factor = 5
        h = np.ones(n, dtype=np.complex128)
        cir = continuous_ofdm(h, factor)
        assert len(cir) == n * factor

    def test_factor_one_noop(self):
        n = 16
        h = np.random.normal(size=n) + 1j * np.random.normal(size=n)
        cir = continuous_ofdm(h, 1)
        cir_expected = np.fft.ifft(h)
        np.testing.assert_allclose(cir, cir_expected, atol=1e-14)


class TestContinuousUwb:
    def test_same_as_ofdm(self):
        """UWB and OFDM continuous CIR use the same zero-padded IFFT."""
        n = 32
        factor = 4
        h = np.random.normal(size=n) + 1j * np.random.normal(size=n)
        np.testing.assert_allclose(
            continuous_ofdm(h, factor), continuous_uwb(h, factor)
        )

    def test_output_length(self):
        n = 32
        factor = 3
        h = np.ones(n, dtype=np.complex128)
        cir = continuous_uwb(h, factor)
        assert len(cir) == n * factor
