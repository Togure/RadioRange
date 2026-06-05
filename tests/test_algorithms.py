"""Tests for algorithms.base_lde — refinement and noise statistics."""

import numpy as np
import pytest

from algorithms.base_lde import BaseLde
from core.models import RadioObservation


def _make_observation(
    t_discrete_s: np.ndarray,
    cir_discrete: np.ndarray,
    cir_cont: np.ndarray | None = None,
    factor: int = 4,
) -> RadioObservation:
    n = len(cir_discrete)
    if n > 1:
        dt = t_discrete_s[1] - t_discrete_s[0]
        t_cont_s = np.arange(n * factor, dtype=float) * (dt / factor)
    else:
        t_cont_s = np.array([t_discrete_s[0]])
    return RadioObservation(
        protocol="test",
        t_discrete_s=t_discrete_s,
        t_cont_s=t_cont_s,
        frequency_hz=np.fft.fftfreq(n, d=(t_discrete_s[1] - t_discrete_s[0])) if n > 1 else np.array([0.0]),
        h_clean=np.zeros(n, dtype=np.complex128),
        h_observed=np.zeros(n, dtype=np.complex128),
        cir_clean_discrete=np.zeros(n),
        cir_observed_discrete=cir_discrete,
        cir_clean_cont=np.zeros(n * factor),
        cir_observed_cont=cir_cont if cir_cont is not None else np.zeros(n * factor),
    )


class TestRefineTof:
    def test_factor_one_returns_discrete(self):
        """When no interpolation is available (factor=1), return discrete ToF."""
        n = 16
        dt = 2e-9
        obs = _make_observation(
            t_discrete_s=np.arange(n, dtype=float) * dt,
            cir_discrete=np.random.random(n),
            cir_cont=np.random.random(n),  # factor 1: same length
            factor=1,
        )
        # Override factor — the function derives factor from lengths
        # len(t_cont_s) // len(t_discrete_s). If both are n, factor=1.
        tof = BaseLde._refine_tof(obs, 5)
        assert tof == pytest.approx(5 * dt)

    def test_sub_bin_refinement_to_peak(self):
        """Refinement should find the local peak in the interpolated CIR."""
        n = 16
        factor = 8
        dt = 2e-9

        # Build an interpolated CIR with peak at cont index 45
        cir_cont = np.zeros(n * factor)
        cir_cont[45] = 1.0  # peak at cont index 45 → discrete index ~5
        # Also put a smaller peak one bin earlier at cont index 38
        cir_cont[38] = 0.5

        # Discrete: put peak at index 5 (same neighborhood as interpolated peak)
        coarse_idx = 5
        cir_discrete = np.zeros(n)
        cir_discrete[coarse_idx] = 1.0

        t_discrete_s = np.arange(n, dtype=float) * dt
        t_cont_s = np.arange(n * factor, dtype=float) * (dt / factor)

        obs = _make_observation(
            t_discrete_s=t_discrete_s,
            cir_discrete=cir_discrete,
            cir_cont=cir_cont,
            factor=factor,
        )

        tof = BaseLde._refine_tof(obs, coarse_idx)
        # Should find cont index 45
        assert tof == pytest.approx(t_cont_s[45])

    def test_refinement_at_start(self):
        n = 16
        factor = 4
        dt = 2e-9

        cir_cont = np.zeros(n * factor)
        cir_cont[1] = 1.0  # peak at cont index 1
        cir_cont[6] = 0.5

        cir_discrete = np.zeros(n)
        cir_discrete[0] = 1.0

        t_discrete_s = np.arange(n, dtype=float) * dt
        t_cont_s = np.arange(n * factor, dtype=float) * (dt / factor)

        obs = _make_observation(
            t_discrete_s=t_discrete_s,
            cir_discrete=cir_discrete,
            cir_cont=cir_cont,
            factor=factor,
        )
        tof = BaseLde._refine_tof(obs, 0)
        assert tof == pytest.approx(t_cont_s[1])


class TestNoiseStats:
    def test_pure_noise(self):
        rng = np.random.default_rng(42)
        envelope = np.abs(rng.normal(0, 1, size=1000))
        mean, std = BaseLde._noise_stats(envelope, tail_frac=0.3)
        # For half-normal of N(0,1): mean ≈ 0.798, std ≈ 0.603
        assert 0.6 < mean < 1.0
        assert 0.4 < std < 0.8

    def test_signal_tail_noise_separation(self):
        """When signal is in first half and noise in second half,
        noise stats should be estimated from the tail only."""
        rng = np.random.default_rng(42)
        n = 200
        envelope = np.zeros(n)
        envelope[:50] = 5.0  # strong signal in front
        envelope[50:] = np.abs(rng.normal(0, 0.5, size=150))  # noise in tail

        mean, std = BaseLde._noise_stats(envelope, tail_frac=0.3)
        # Should reflect the noise, not the signal
        assert mean < 2.0
        assert std < 2.0

    def test_short_signal_fallback(self):
        """Very short signal should use last half even if tail_frac gives < 4 samples."""
        envelope = np.array([1.0, 2.0, 3.0, 4.0, 0.1, 0.1])
        mean, std = BaseLde._noise_stats(envelope, tail_frac=0.25)
        assert np.isfinite(mean)
        assert np.isfinite(std)
