from __future__ import annotations

import numpy as np


def path_response(frequency_hz: np.ndarray, a_paths: np.ndarray, tau_paths_s: np.ndarray) -> np.ndarray:
    h_frequency = np.zeros_like(frequency_hz, dtype=np.complex128)
    for gain, delay_s in zip(a_paths, tau_paths_s):
        h_frequency += gain * np.exp(-1j * 2.0 * np.pi * frequency_hz * delay_s)
    return h_frequency


def continuous_ofdm(h_frequency: np.ndarray, factor: int) -> np.ndarray:
    """Upsample OFDM CIR via zero-padded IFFT (bandlimited sinc interpolation)."""
    n = len(h_frequency)
    h_pos = h_frequency[: n // 2]
    h_neg = h_frequency[n // 2 :]
    zeros = np.zeros((factor - 1) * n, dtype=h_frequency.dtype)
    return np.fft.ifft(np.concatenate([h_pos, zeros, h_neg])) * factor


def continuous_uwb(h_frequency: np.ndarray, factor: int) -> np.ndarray:
    """Upsample UWB CIR via zero-padded IFFT.

    UWB is impulse radio (not OFDM), but the mathematical operation —
    zero-padded IFFT for bandlimited interpolation — is identical.
    The function is kept separate so each protocol can evolve independently
    (e.g. UWB may later add window-specific shaping, asymmetric padding,
    or different normalisation).
    """
    n = len(h_frequency)
    h_pos = h_frequency[: n // 2]
    h_neg = h_frequency[n // 2 :]
    zeros = np.zeros((factor - 1) * n, dtype=h_frequency.dtype)
    return np.fft.ifft(np.concatenate([h_pos, zeros, h_neg])) * factor
