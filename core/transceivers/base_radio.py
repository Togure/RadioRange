from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.models import ChannelTruth, RadioObservation


class BaseRadio(ABC):
    protocol: str

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def observe(self, truth: ChannelTruth, rng: np.random.Generator) -> RadioObservation:
        raise NotImplementedError


def normalize_observation(cir: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(cir))
    if peak <= 0:
        return cir
    return cir / peak


def normalize_with_reference(cir: np.ndarray, reference: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(reference))
    if peak <= 0:
        return cir
    return cir / peak


def apply_snr_noise(
    h_frequency: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if snr_db >= 100.0:
        return h_frequency

    n_bins = len(h_frequency)
    cir_clean = np.fft.ifft(h_frequency)
    p_peak = float(np.max(np.abs(cir_clean) ** 2))

    sigma2_td = p_peak / (10.0 ** (snr_db / 10.0))
    sigma2_fd = sigma2_td * n_bins
    sigma_fd = np.sqrt(sigma2_fd / 2.0)

    noise = rng.normal(size=h_frequency.shape) + 1j * rng.normal(size=h_frequency.shape)
    return h_frequency + (sigma_fd * noise).astype(h_frequency.dtype)


def apply_cir_impairments(
    cir_discrete: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
    radio_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply CIR-domain AGC (analog gain control before ADC).

    AGC is the only impairment applied in the CIR domain.  ADC quantization
    is handled in the frequency domain by ``observe_frequency_response``
    (see ``hardware.adc.quantize_frequency_iq``), which correctly preserves
    the IFFT processing gain.

    Called from each radio's ``observe()`` after the raw CIR is computed.

    AGC parameters (gain_step_db, max_gain_db) are read from ``radio_cfg``
    when provided; otherwise they fall back to the global impairments section.
    The enable toggle and clip_enable always come from impairments.
    """
    impair_cfg = config.get("impairments", {})
    radio_cfg = radio_cfg or {}
    metadata: dict[str, Any] = {}

    if bool(impair_cfg.get("enable_agc", False)):
        from hardware.agc import apply_agc

        agc_gain_step_db = float(
            radio_cfg.get("agc_gain_step_db")
            or impair_cfg.get("agc_gain_step_db", 3.0)
        )
        agc_max_gain_db = float(
            radio_cfg.get("agc_max_gain_db")
            or impair_cfg.get("agc_max_gain_db", 30.0)
        )
        agc_settling_us = float(impair_cfg.get("agc_settling_us", 2.0))
        agc_clip_enable = bool(impair_cfg.get("agc_clip_enable", False))

        cir_discrete, agc_gain_linear = apply_agc(
            cir_discrete,
            gain_step_db=agc_gain_step_db,
            max_gain_db=agc_max_gain_db,
            settling_us=agc_settling_us,
            clip_enable=agc_clip_enable,
            rng=rng,
        )
        metadata["agc_gain_step_db"] = agc_gain_step_db
        metadata["agc_max_gain_db"] = agc_max_gain_db
        metadata["agc_clip_enable"] = agc_clip_enable
        metadata["agc_gain_linear"] = float(agc_gain_linear)
    else:
        metadata["agc_gain_linear"] = 1.0

    return cir_discrete, metadata


def _dmrs_2d_observation(
    h_clean: np.ndarray,
    frequency_hz: np.ndarray,
    active_mask: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """2D DM-RS time-frequency channel estimation (5G NR style).

    Creates one noisy CSI copy per OFDM symbol, samples at sparse 2D DM-RS
    positions, interpolates across both time and frequency via :func:`griddata`,
    and averages the estimated DM-RS symbols to produce a single 1D CSI vector.

    This replaces the 1D pilot-interpolation path when
    ``channel_estimation_mode`` is set to ``"2d"``.
    """
    n_subcarriers = len(h_clean)
    num_symbols = int(cfg.get("num_ofdm_symbols", 14))
    dmrs_symbols = list(cfg.get("dmrs_symbol_indices", [2, 6, 10]))
    dmrs_freq_spacing = int(cfg.get("dmrs_freq_spacing", 6))

    # ── per-symbol noise magnitudes ──────────────────────────────────────
    amp_std = float(cfg.get("csi_amplitude_std", 0.0))
    phase_std = float(cfg.get("csi_phase_std_rad", 0.0))
    csi_snr_db = cfg.get("csi_noise_snr_db")

    # ── per-symbol common phase error (inter-symbol phase variation) ────
    # These model effects that make different OFDM symbols see different
    # common phase rotations, degrading multi-symbol DM-RS averaging.
    dmrs_cpe_std_rad = float(cfg.get("dmrs_cpe_std_rad", 0.0))
    dmrs_residual_cfo_hz = float(cfg.get("dmrs_residual_cfo_hz", 0.0))
    if dmrs_residual_cfo_hz != 0.0:
        delta_f = float(frequency_hz[1] - frequency_hz[0])
        sym_duration_s = float(cfg.get("dmrs_symbol_duration_s", 1.0 / delta_f))

    # ── generate noisy CSI per OFDM symbol ───────────────────────────────
    h_2d = np.tile(h_clean, (num_symbols, 1)).astype(np.complex128)  # [sym, sc]

    for sym in range(num_symbols):
        row = h_2d[sym].copy()

        # Per-symbol common phase (phase noise + residual CFO drift)
        if dmrs_cpe_std_rad > 0.0:
            row *= np.exp(1j * dmrs_cpe_std_rad * rng.normal())
        if dmrs_residual_cfo_hz != 0.0:
            row *= np.exp(1j * 2.0 * np.pi * dmrs_residual_cfo_hz * sym * sym_duration_s)

        if amp_std > 0.0:
            row *= np.maximum(0.0, 1.0 + amp_std * rng.normal(size=row.shape))
        if phase_std > 0.0:
            row *= np.exp(1j * phase_std * rng.normal(size=row.shape))
        if csi_snr_db is not None:
            row = apply_snr_noise(row, float(csi_snr_db), rng)

        h_2d[sym] = row

    # ── 2D DM-RS mask ────────────────────────────────────────────────────
    active_indices = np.flatnonzero(active_mask)
    pilot_sc = active_indices[::dmrs_freq_spacing]
    # Ensure last active subcarrier is always a pilot → avoids edge gaps
    if pilot_sc[-1] != active_indices[-1]:
        pilot_sc = np.append(pilot_sc, active_indices[-1])

    dmrs_symbols_valid = [s for s in dmrs_symbols if 0 <= s < num_symbols]
    if not dmrs_symbols_valid:
        # Degenerate: fall back to 1D on first symbol
        h_1d = h_2d[0]
        return np.where(active_mask, h_1d, 0.0), {
            "channel_estimation": "2d_fallback_1d",
            "dmrs_total_pilots": 0,
        }

    known_pts: list[tuple[int, int]] = []
    known_vals: list[complex] = []
    for sym in dmrs_symbols_valid:
        for sc in pilot_sc:
            known_pts.append((sc, sym))
            known_vals.append(h_2d[sym, sc])

    if len(known_pts) < 4:
        h_1d = np.mean(h_2d[dmrs_symbols_valid], axis=0)
        return np.where(active_mask, h_1d, 0.0), {
            "channel_estimation": "2d_fallback_mean",
            "dmrs_total_pilots": len(known_pts),
        }

    # ── 2D linear interpolation (scipy gridata) ────────────────────────
    from scipy.interpolate import griddata

    pts = np.array(known_pts, dtype=float)           # [N, 2] — (sc, sym)
    vals = np.array(known_vals, dtype=np.complex128)  # [N]

    # Query all (subcarrier, symbol) positions
    query_pts = np.array(
        [(sc, sym) for sym in range(num_symbols) for sc in range(n_subcarriers)],
        dtype=float,
    )

    interp_real = griddata(pts, vals.real, query_pts, method="linear",
                           fill_value=0.0)
    interp_imag = griddata(pts, vals.imag, query_pts, method="linear",
                           fill_value=0.0)
    h_estimated_2d = (interp_real + 1j * interp_imag).reshape(
        num_symbols, n_subcarriers
    )

    # ── Average across DM-RS symbols → final 1D Ĥ ──────────────────────
    h_1d = np.mean(h_estimated_2d[dmrs_symbols_valid], axis=0)
    h_1d = np.where(active_mask, h_1d, 0.0)

    return h_1d, {
        "channel_estimation": "2d_dmrs",
        "num_ofdm_symbols": num_symbols,
        "dmrs_symbols": dmrs_symbols_valid,
        "dmrs_freq_spacing": dmrs_freq_spacing,
        "dmrs_total_pilots": len(known_pts),
        "dmrs_cpe_std_rad": dmrs_cpe_std_rad,
        "dmrs_residual_cfo_hz": dmrs_residual_cfo_hz,
    }


def observe_frequency_response(
    h_clean: np.ndarray,
    frequency_hz: np.ndarray,
    radio_cfg: dict[str, Any],
    rng: np.random.Generator,
    enable_mask_and_pilots: bool,
    impair_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build observed CSI/CIR from clean CSI.

    ``observation_model='snr'`` keeps the original compact model.  The
    ``explicit`` model applies independent device-observation effects so they
    can be swept one by one.
    """

    impair_cfg = impair_cfg or {}
    model = str(radio_cfg.get("observation_model", "snr")).lower()
    if model == "snr":
        snr_db = float(radio_cfg.get("snr_db", 60.0))
        h_obs = apply_snr_noise(h_clean, snr_db, rng)
        metadata = {
            "observation_model": model,
            "snr_db": snr_db,
        }
    elif model == "explicit":
        h_obs = np.array(h_clean, dtype=np.complex128, copy=True)
        metadata = {"observation_model": model}
    else:
        raise ValueError(f"Unknown observation_model: {model}")

    cfg = dict(radio_cfg.get("explicit_impairments", {})) if model == "explicit" else {}

    if model != "explicit":
        # SNR mode is intentionally compact: it skips the detailed CSI
        # observation terms but can still use hardware-level ADC quantization
        # below when enable_adc_quantization is set.
        active_mask = np.ones(len(h_obs), dtype=bool)
    else:
        if enable_mask_and_pilots:
            active_mask = _build_subcarrier_mask(len(h_obs), cfg)
            h_obs = np.where(active_mask, h_obs, 0.0)
            metadata["active_subcarriers"] = int(np.count_nonzero(active_mask))
        else:
            active_mask = np.ones(len(h_obs), dtype=bool)

        timing_offset_s = float(cfg.get("sampling_phase_offset_s", 0.0))
        timing_offset_s += float(cfg.get("random_sampling_phase_std_s", 0.0)) * rng.normal()
        if timing_offset_s != 0.0:
            h_obs *= np.exp(-1j * 2.0 * np.pi * frequency_hz * timing_offset_s)
        metadata["sampling_phase_offset_s"] = timing_offset_s

        sfo_residual_ppm = float(cfg.get("sfo_residual_ppm", cfg.get("sfo_ppm", 0.0)))
        if sfo_residual_ppm != 0.0:
            # First-order residual SFO after receiver digital compensation:
            # a small linear phase slope across frequency.
            sfo_delay_s = sfo_residual_ppm * 1e-6 * float(cfg.get("sfo_reference_delay_s", 100e-9))
            h_obs *= np.exp(-1j * 2.0 * np.pi * frequency_hz * sfo_delay_s)
            metadata["sfo_equiv_delay_s"] = sfo_delay_s
        metadata["sfo_residual_ppm"] = sfo_residual_ppm

        common_phase_rad = float(cfg.get("common_phase_offset_rad", 0.0))
        common_phase_rad += float(cfg.get("common_phase_std_rad", 0.0)) * rng.normal()
        if common_phase_rad != 0.0:
            h_obs *= np.exp(1j * common_phase_rad)
        metadata["common_phase_rad"] = common_phase_rad

        # ── Channel estimation ──────────────────────────────────────────
        # "ltf"     — WiFi 802.11 LTF: every active subcarrier estimated
        #              independently.  No pilots, no interpolation.
        # "2d"      — 5G NR DM-RS: sparse 2D time-frequency pilots +
        #              griddata interpolation + symbol averaging.
        # "1d"      — Legacy 1D pilot interpolation (backward compat).
        est_mode = str(cfg.get("channel_estimation_mode", "1d")).lower()

        if est_mode == "ltf" and enable_mask_and_pilots:
            amp_std = float(cfg.get("csi_amplitude_std", 0.0))
            if amp_std > 0.0:
                h_obs *= np.maximum(0.0, 1.0 + amp_std * rng.normal(size=h_obs.shape))
            metadata["csi_amplitude_std"] = amp_std

            phase_std = float(cfg.get("csi_phase_std_rad", 0.0))
            if phase_std > 0.0:
                h_obs *= np.exp(1j * phase_std * rng.normal(size=h_obs.shape))
            metadata["csi_phase_std_rad"] = phase_std

            csi_snr_db = cfg.get("csi_noise_snr_db")
            if csi_snr_db is not None:
                h_obs = apply_snr_noise(h_obs, float(csi_snr_db), rng)
                metadata["csi_noise_snr_db"] = float(csi_snr_db)

            metadata["channel_estimation"] = "ltf_full_coverage"

        elif est_mode == "2d" and enable_mask_and_pilots:
            # 2D DM-RS time-frequency interpolation — per-symbol noise,
            # sparse 2D sampling, griddata interpolation, symbol averaging.
            h_obs, est_meta = _dmrs_2d_observation(
                h_obs, frequency_hz, active_mask, cfg, rng,
            )
            metadata.update(est_meta)
        else:
            # 1D path: uniform per-subcarrier noise + optional pilot interp.
            amp_std = float(cfg.get("csi_amplitude_std", 0.0))
            if amp_std > 0.0:
                h_obs *= np.maximum(0.0, 1.0 + amp_std * rng.normal(size=h_obs.shape))
            metadata["csi_amplitude_std"] = amp_std

            phase_std = float(cfg.get("csi_phase_std_rad", 0.0))
            if phase_std > 0.0:
                h_obs *= np.exp(1j * phase_std * rng.normal(size=h_obs.shape))
            metadata["csi_phase_std_rad"] = phase_std

            csi_snr_db = cfg.get("csi_noise_snr_db")
            if csi_snr_db is not None:
                h_obs = apply_snr_noise(h_obs, float(csi_snr_db), rng)
                metadata["csi_noise_snr_db"] = float(csi_snr_db)

            if enable_mask_and_pilots:
                pilot_spacing = int(cfg.get("pilot_spacing_subcarriers", 1))
                if pilot_spacing > 1:
                    h_obs = _sparse_pilot_interpolate(
                        h_obs, frequency_hz, active_mask, pilot_spacing,
                    )
                    metadata["pilot_spacing_subcarriers"] = pilot_spacing
                    metadata["pilot_interpolation"] = "linear_real_imag"
                    metadata["channel_estimation"] = "1d_pilot_interp"

    # ADC amplitude quantization is a hardware effect, so it is only applied
    # when the global or per-radio switch is enabled.  adc_bits itself is a
    # device-profile parameter and is read from radio_cfg.
    adc_enabled = bool(
        impair_cfg.get("enable_adc_quantization", False)
        or radio_cfg.get("enable_adc_quantization", False)
        or cfg.get("enable_adc_quantization", False)
    )
    adc_bits = radio_cfg.get("adc_bits", cfg.get("adc_bits"))
    if adc_enabled and adc_bits is not None:
        adc_bits = int(adc_bits)
        if adc_bits > 0:
            from hardware.adc import quantize_frequency_iq

            h_obs = quantize_frequency_iq(h_obs, adc_bits=adc_bits)
            metadata["adc_bits"] = adc_bits
            metadata["enable_adc_quantization"] = True

    return h_obs.astype(h_clean.dtype), metadata


def _build_subcarrier_mask(n: int, cfg: dict[str, Any]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)

    guard_bins = int(cfg.get("guard_bins_each_side", 0))
    if guard_bins > 0:
        mask[:guard_bins] = False
        mask[n - guard_bins :] = False

    if bool(cfg.get("dc_null", False)):
        mask[0] = False

    active_fraction = cfg.get("active_subcarrier_fraction")
    if active_fraction is not None:
        active = max(1, min(n, int(round(float(active_fraction) * n))))
        frac_mask = np.zeros(n, dtype=bool)
        half = active // 2
        frac_mask[:half] = True
        frac_mask[-(active - half) :] = True
        mask &= frac_mask

    return mask


def _sparse_pilot_interpolate(
    h_obs: np.ndarray,
    frequency_hz: np.ndarray,
    active_mask: np.ndarray,
    pilot_spacing: int,
) -> np.ndarray:
    active_idx = np.flatnonzero(active_mask)
    if len(active_idx) < 2:
        return h_obs

    pilot_idx = active_idx[::pilot_spacing]
    if pilot_idx[-1] != active_idx[-1]:
        pilot_idx = np.append(pilot_idx, active_idx[-1])
    if len(pilot_idx) < 2:
        return h_obs

    order = np.argsort(frequency_hz[active_idx])
    active_sorted = active_idx[order]
    pilot_order = np.argsort(frequency_hz[pilot_idx])
    pilot_sorted = pilot_idx[pilot_order]

    interp = np.zeros_like(h_obs)
    interp_active_real = np.interp(
        frequency_hz[active_sorted],
        frequency_hz[pilot_sorted],
        h_obs[pilot_sorted].real,
    )
    interp_active_imag = np.interp(
        frequency_hz[active_sorted],
        frequency_hz[pilot_sorted],
        h_obs[pilot_sorted].imag,
    )
    interp[active_sorted] = interp_active_real + 1j * interp_active_imag
    return interp
