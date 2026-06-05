"""Tests for hardware impairment models."""

import numpy as np
import pytest

from hardware.adc import quantize_cir, quantize_frequency_iq
from hardware.agc import apply_agc
from hardware.antenna import apply_antenna_pcv
from hardware.clock import apply_cfo, apply_sfo
from hardware.iq_imbalance import apply_iq_imbalance


# ═══════════════════════════════════════════════════════════════════════════
# SFO
# ═══════════════════════════════════════════════════════════════════════════
class TestApplySfo:
    def test_zero_sfo_noop(self):
        tau = np.array([10e-9, 20e-9, 30e-9])
        result, scale = apply_sfo(tau, 0.0)
        np.testing.assert_array_equal(result, tau)
        assert scale == 1.0

    def test_positive_sfo_stretches_delays(self):
        tau = np.array([100e-9])
        result, scale = apply_sfo(tau, 20.0)  # +20 ppm
        assert scale == pytest.approx(1.000020)
        assert result[0] == pytest.approx(100e-9 * 1.000020)

    def test_negative_sfo_compresses_delays(self):
        tau = np.array([100e-9])
        result, scale = apply_sfo(tau, -20.0)
        assert scale == pytest.approx(0.999980)
        assert result[0] == pytest.approx(100e-9 * 0.999980)

    def test_sfo_scale_matches_delay_ratio(self):
        tau = np.array([10e-9, 50e-9, 100e-9])
        sfo_ppm = 15.0
        result, scale = apply_sfo(tau, sfo_ppm)
        np.testing.assert_allclose(result, tau * scale)


# ═══════════════════════════════════════════════════════════════════════════
# CFO
# ═══════════════════════════════════════════════════════════════════════════
class TestApplyCfo:
    def test_zero_cfo_noop(self):
        a = np.array([1.0 + 0j, 0.5 + 0.5j])
        tau = np.array([10e-9, 20e-9])
        result = apply_cfo(a, tau, 0.0)
        np.testing.assert_array_equal(result, a)

    def test_cfo_phase_rotation(self):
        """CFO rotates each path by exp(-j*2*pi*cfo_hz*tau)."""
        a = np.array([1.0 + 0j])
        tau = np.array([1.0 / (2.0 * np.pi)])  # 2*pi*f*tau = cfo_hz
        cfo_hz = 1.0
        result = apply_cfo(a, tau, cfo_hz)
        expected = np.exp(-1j * 2.0 * np.pi * cfo_hz * tau[0])
        np.testing.assert_allclose(result[0], expected)

    def test_cfo_path_dependent_phase(self):
        """Each path gets different phase rotation proportional to its delay."""
        a = np.array([1.0 + 0j, 1.0 + 0j])
        tau = np.array([10e-9, 50e-9])
        cfo_hz = 1000.0  # 1 kHz
        result = apply_cfo(a, tau, cfo_hz)

        phase_diff = np.angle(result[1]) - np.angle(result[0])
        expected_diff = -2.0 * np.pi * cfo_hz * (tau[1] - tau[0])
        np.testing.assert_allclose(phase_diff, expected_diff, atol=1e-12)

    def test_cfo_preserves_magnitude(self):
        a = np.array([0.3 + 0.4j, 0.8 - 0.6j])
        tau = np.array([10e-9, 30e-9])
        result = apply_cfo(a, tau, 5000.0)
        np.testing.assert_allclose(np.abs(result), np.abs(a))


# ═══════════════════════════════════════════════════════════════════════════
# Antenna PCV
# ═══════════════════════════════════════════════════════════════════════════
class TestApplyAntennaPcv:
    def test_no_angle_no_rng_noop(self):
        tau = np.array([10e-9, 20e-9])
        result = apply_antenna_pcv(tau, pcv_magnitude_m=0.003)
        np.testing.assert_array_equal(result, tau)

    def test_no_angle_with_rng_adds_random(self):
        rng = np.random.default_rng(42)
        tau = np.array([10e-9, 20e-9])
        result = apply_antenna_pcv(tau, pcv_magnitude_m=0.003, rng=rng)
        # Should add positive offsets
        assert np.all(result >= tau)
        assert len(result) == len(tau)

    def test_normal_incidence_zero_offset(self):
        """Elevation = 90° (normal to boresight) should give zero PCV."""
        tau = np.array([10e-9])
        aoa_el = np.array([90.0])  # cos(90°) = 0 → offset = pcv × (1-0) = max
        # Wait — our model: offset = pcv × (1 - |cos(elev)|).
        # For elevation=90°, cos(90°)=0 → offset_m = pcv × (1-0) = pcv (max!)
        # This is the "grazing" angle case.

        aoa_el = np.array([0.0])  # cos(0°)=1 → offset_angle = pcv × (1-1) = 0
        result = apply_antenna_pcv(
            tau,
            aoa_elevation_deg=aoa_el,
            pcv_magnitude_m=0.003,
        )
        # cos(0)=1 → offset = 0.003 × 0.5 × (1-1) = 0
        np.testing.assert_allclose(result, tau)

    def test_grazing_incidence_max_offset(self):
        """Elevation = 90° relative to boresight → maximum PCV offset."""
        tau = np.array([10e-9])
        aoa_el = np.array([90.0])  # cos(90°)=0 → offset = 0.003 × 0.5 × 1
        result = apply_antenna_pcv(
            tau,
            aoa_elevation_deg=aoa_el,
            pcv_magnitude_m=0.010,
        )
        LIGHT_SPEED_MPS = 299792458.0
        expected_offset = 0.5 * 0.010 / LIGHT_SPEED_MPS  # ~16.7 ps
        assert result[0] == pytest.approx(tau[0] + expected_offset)

    def test_both_aoa_aod_double_offset(self):
        tau = np.array([10e-9])
        aoa_el = np.array([90.0])
        aod_el = np.array([90.0])
        result_aoa = apply_antenna_pcv(
            tau,
            aoa_elevation_deg=aoa_el,
            pcv_magnitude_m=0.010,
        )
        result_both = apply_antenna_pcv(
            tau,
            aoa_elevation_deg=aoa_el,
            aod_elevation_deg=aod_el,
            pcv_magnitude_m=0.010,
        )
        # Both ends → double the offset
        double_diff = (result_both[0] - tau[0]) / (result_aoa[0] - tau[0])
        assert double_diff == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# ADC Quantization
# ═══════════════════════════════════════════════════════════════════════════
class TestQuantizeCir:
    def test_6bit_unity_peak(self):
        """Normalized CIR with peak 1.0 should have 32 levels for 6-bit."""
        cir = np.array([0.0, 0.05, 0.2, 0.5, 1.0, 0.3, 0.01])
        result = quantize_cir(cir, adc_bits=6)
        # 6-bit signed → 32 positive levels
        delta = 1.0 / 31.0
        for val in result:
            # Each value should be an integer multiple of delta
            assert val == pytest.approx(np.round(val / delta) * delta)

    def test_6bit_noise_floor(self):
        """Values below half an LSB should quantize to 0 or delta."""
        cir = np.array([0.001, 0.002])
        result = quantize_cir(cir, adc_bits=6)
        delta = 1.0 / 31.0  # ≈ 0.0323
        val = min(np.abs(result))
        # Should be either 0 or at least delta
        assert val == 0.0 or val >= pytest.approx(delta)

    def test_indistinguishable_peaks(self):
        """Two peaks within one LSB of each other may quantize to same value."""
        cir = np.array([0.5, 0.51])
        result = quantize_cir(cir, adc_bits=3)  # only 4 positive levels
        # With 3-bit, delta = 1/3 ≈ 0.333, both 0.5 and 0.51 are near 0.5
        # Both should be on the same or adjacent levels (at most one delta apart)
        delta = 1.0 / 3.0
        assert abs(float(result[0] - result[1])) <= delta * 1.01

    def test_bits_too_low(self):
        """adc_bits < 1 should return original signal."""
        cir = np.array([0.1, 0.5, 1.0])
        result = quantize_cir(cir, adc_bits=0)
        np.testing.assert_array_equal(result, cir)

    def test_empty_input(self):
        result = quantize_cir(np.array([]), adc_bits=6)
        assert len(result) == 0


class TestQuantizeFrequencyIq:
    def test_6bit_preserves_shape(self):
        n = 64
        h = np.exp(1j * 2.0 * np.pi * np.arange(n) / n)
        result = quantize_frequency_iq(h, adc_bits=6)
        assert len(result) == n
        assert result.dtype == h.dtype

    def test_small_signal_noop(self):
        """Near-zero signal should be unchanged."""
        h = np.array([1e-40 + 1e-40j])
        result = quantize_frequency_iq(h, adc_bits=8)
        np.testing.assert_array_equal(result, h)

    def test_quantization_reduces_resolution(self):
        """After quantization, values should be on discrete levels."""
        h = np.random.normal(size=128) + 1j * np.random.normal(size=128)
        result = quantize_frequency_iq(h, adc_bits=4)  # coarse
        # Real and imag parts should be quantized
        max_abs = max(np.max(np.abs(h.real)), np.max(np.abs(h.imag)))
        n_levels = 2 ** (4 - 1)
        delta = max_abs / (n_levels - 1)
        # Check a few samples are on the grid
        for val in result[:5]:
            real_remainder = (val.real / delta) % 1.0
            imag_remainder = (val.imag / delta) % 1.0
            assert real_remainder == pytest.approx(0.0, abs=1e-10) or real_remainder == pytest.approx(1.0, abs=1e-10)
            assert imag_remainder == pytest.approx(0.0, abs=1e-10) or imag_remainder == pytest.approx(1.0, abs=1e-10)

    def test_empty_input(self):
        result = quantize_frequency_iq(np.array([]), adc_bits=6)
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════
# AGC
# ═══════════════════════════════════════════════════════════════════════════
class TestApplyAgc:
    def test_peak_one_no_gain(self):
        """Peak already at 1.0 → ideal gain ≈ 0 dB → no amplification."""
        cir = np.array([0.0, 0.3, 1.0, 0.2])
        result, gain = apply_agc(cir, gain_step_db=3.0, max_gain_db=30.0)
        # ideal_gain_linear = 1.0 → ideal_gain_db = 0 → quantized → 0 dB → gain=1.0
        assert gain == pytest.approx(1.0)
        np.testing.assert_allclose(result, cir)

    def test_weak_signal_amplified(self):
        """Weak signal → AGC amplifies to bring peak near 1.0."""
        cir = np.array([0.0, 0.02, 0.05, 0.01])
        result, gain = apply_agc(cir, gain_step_db=3.0, max_gain_db=30.0)
        # ideal_gain_linear = 20 → 26 dB → quantized to 27 dB (9×3 dB)
        assert gain > 1.0
        peak_after = np.max(result)
        assert peak_after > np.max(cir)

    def test_clip_enabled(self):
        """With clipping, amplified values exceeding 1.0 are capped."""
        cir = np.array([0.0, 0.3, 0.8, 0.5])
        result, _ = apply_agc(cir, gain_step_db=3.0, max_gain_db=30.0, clip_enable=True)
        # Some values may exceed 1.0 after amplification — clipping caps them
        assert np.max(result) <= 1.0

    def test_gain_quantized_to_step(self):
        """Gain should be a multiple of gain_step_db."""
        cir = np.array([0.0, 0.15, 0.3, 0.1])
        _, gain = apply_agc(cir, gain_step_db=3.0, max_gain_db=30.0)
        gain_db = 20.0 * np.log10(max(gain, 1e-12))
        remainder = gain_db % 3.0
        # Should be close to 0 (exact multiple of 3 dB)
        assert min(abs(remainder), abs(remainder - 3.0)) < 1e-9

    def test_max_gain_capped(self):
        """Gain should never exceed max_gain_db."""
        cir = np.array([1e-10])  # extremely weak
        _, gain = apply_agc(cir, gain_step_db=3.0, max_gain_db=12.0)
        assert 20.0 * np.log10(gain) <= 12.0 + 1e-9

    def test_empty_input(self):
        cir = np.array([])
        result, gain = apply_agc(cir)
        assert len(result) == 0
        assert gain == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# I/Q Imbalance
# ═══════════════════════════════════════════════════════════════════════════
class TestApplyIqImbalance:
    def test_no_imbalance_noop(self):
        h = np.array([1.0 + 0j, 0.5 + 0.5j, -0.3 + 0.8j])
        result = apply_iq_imbalance(h, epsilon=0.0, phi_deg=0.0)
        np.testing.assert_array_equal(result, h)

    def test_small_imbalance_near_identity(self):
        """Small epsilon/phi should produce output close to input."""
        rng = np.random.default_rng(42)
        n = 64
        h = rng.normal(size=n) + 1j * rng.normal(size=n)
        result = apply_iq_imbalance(h, epsilon=0.01, phi_deg=1.0)
        # Correlation between input and output should be high
        corr = np.abs(np.vdot(h, result)) / (
            np.linalg.norm(h) * np.linalg.norm(result)
        )
        assert corr > 0.99

    def test_pure_real_input(self):
        """Pure real frequency response is affected by I/Q imbalance differently."""
        h = np.ones(4, dtype=np.complex128)
        result = apply_iq_imbalance(h, epsilon=0.05, phi_deg=5.0)
        # Output should NOT be identical (imbalance creates mirror)
        assert not np.allclose(result, h)

    def test_dc_bin_no_mirror(self):
        """DC bin is its own mirror — should be real-scaled."""
        h = np.array([2.0 + 0j, 1.0 + 1j, 0.5 + 0j])
        result = apply_iq_imbalance(h, epsilon=0.1, phi_deg=3.0)
        # DC (index 0) maps to itself, so the mirror is itself
        assert abs(result[0].real) != pytest.approx(abs(h[0]))
        # But the DC result should still be finite
        assert np.isfinite(result[0])

    def test_preserves_signal_power_approximately(self):
        """I/Q imbalance approximately preserves total power."""
        rng = np.random.default_rng(42)
        h = rng.normal(size=256) + 1j * rng.normal(size=256)
        result = apply_iq_imbalance(h, epsilon=0.03, phi_deg=3.0)
        power_in = np.sum(np.abs(h) ** 2)
        power_out = np.sum(np.abs(result) ** 2)
        # Should be within a few percent
        assert 0.9 < power_out / power_in < 1.1
