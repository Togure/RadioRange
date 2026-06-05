from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth
from hardware.antenna import apply_antenna_pcv
from hardware.clock import apply_cfo, apply_sfo


def apply_timing_impairments(
    truth: ChannelTruth,
    config: dict[str, Any],
    rng: np.random.Generator,
    radio_cfg: dict[str, Any] | None = None,
) -> ChannelTruth:
    """Apply all ChannelTruth-level impairments at once.

    Order of operations
    -------------------
    1. Antenna PCV  — positioning-level per-path delay offset from
       phase-centre variation.
    2. SFO          — scale tau_paths_s (sampling clock mismatch).
    3. CFO          — first-order per-path phase rotation approximation.
    4. Absolute biases — sync_bias_s + clock_bias_s + ADC phase offset.

    Configuration
    -------------
    Toggles are read from ``config["impairments"]``.  Device-specific
    magnitudes (sfo_ppm, cfo_hz, antenna_pcv_magnitude_m) are read from
    ``radio_cfg`` when provided; otherwise they fall back to the global
    impairments section.

    Antenna PCV (device-level):
      ``enable_antenna_offset`` (bool) — toggle in impairments.
      ``antenna_pcv_magnitude_m`` (float) — from radio_cfg or impairments.

    SFO (chip-level):
      ``enable_sfo`` (bool) — toggle in impairments.
      ``sfo_ppm`` (float) — from radio_cfg or impairments.

    CFO (chip-level):
      ``enable_cfo`` (bool) — toggle in impairments.
      ``cfo_hz`` (float) — from radio_cfg or impairments.

    ADC phase offset (positioning-level timing approximation):
      ``enable_adc_phase_offset`` (bool)
      ``adc_phase_offset_s`` (float): deterministic offset [s].
      If enable=True and value=0, a random Uniform(-0.5ns, +0.5ns) is drawn.

    Timing biases (system-level, from ``config["timing"]``):
      ``sync_bias_s`` (float): transmitter–receiver sync offset [s].
      ``clock_bias_s`` (float): absolute clock offset [s].
    """

    impair_cfg = config.get("impairments", {})
    timing_cfg = config.get("timing", {})
    radio_cfg = radio_cfg or {}

    a_paths = truth.a_paths
    tau_paths_s = truth.tau_paths_s

    # ── Antenna PCV ───────────────────────────────────────────────────
    pcv_magnitude_m = 0.0
    if bool(impair_cfg.get("enable_antenna_offset", False)):
        pcv_magnitude_m = float(
            radio_cfg.get("antenna_pcv_magnitude_m")
            or impair_cfg.get("antenna_pcv_magnitude_m", 0.003)
        )
        tau_paths_s = apply_antenna_pcv(
            tau_paths_s,
            aoa_azimuth_deg=truth.aoa_azimuth_deg,
            aoa_elevation_deg=truth.aoa_elevation_deg,
            aod_azimuth_deg=truth.aod_azimuth_deg,
            aod_elevation_deg=truth.aod_elevation_deg,
            pcv_magnitude_m=pcv_magnitude_m,
            rng=rng,
        )

    # ── SFO ────────────────────────────────────────────────────────────
    sfo_ppm = 0.0
    sfo_scale = 1.0
    if bool(impair_cfg.get("enable_sfo", False)):
        sfo_ppm = float(
            radio_cfg.get("sfo_ppm")
            or impair_cfg.get("sfo_ppm", 0.0)
        )
    tau_paths_s, sfo_scale = apply_sfo(tau_paths_s, sfo_ppm)

    # ── CFO ────────────────────────────────────────────────────────────
    # Positioning-layer approximation: we do NOT simulate OFDM ICI or a
    # full LO tracking loop.  Instead, the uncompensated CFO is modelled as
    # a per-path phase rotation of the complex gains.  This is the raw
    # physical error *before* receiver compensation.  Residuals after
    # compensation (CPE, phase noise) are layered independently in the
    # observation model — the two stages are serial: raw CFO → receiver
    # compensation → residual error.
    cfo_hz = 0.0
    if bool(impair_cfg.get("enable_cfo", False)):
        cfo_hz = float(
            radio_cfg.get("cfo_hz")
            or impair_cfg.get("cfo_hz", 0.0)
        )
    a_paths = apply_cfo(a_paths, tau_paths_s, cfo_hz)

    # ── Absolute timing biases ─────────────────────────────────────────
    adc_offset_s = 0.0
    if bool(impair_cfg.get("enable_adc_phase_offset", False)):
        adc_offset_s = float(impair_cfg.get("adc_phase_offset_s", 0.0))
        if adc_offset_s == 0.0:
            adc_offset_s = rng.uniform(-0.5e-9, 0.5e-9)

    total_bias_s = (
        float(timing_cfg.get("sync_bias_s", truth.sync_bias_s))
        + float(timing_cfg.get("clock_bias_s", truth.clock_bias_s))
        + adc_offset_s
    )

    tau_paths_s = tau_paths_s + total_bias_s

    if total_bias_s == 0.0 and sfo_ppm == 0.0 and cfo_hz == 0.0 and pcv_magnitude_m == 0.0:
        if bool(timing_cfg.get("rtt_mode", truth.rtt_mode)) == truth.rtt_mode:
            return truth

    return ChannelTruth(
        a_paths=a_paths,
        tau_paths_s=tau_paths_s,
        path_type=truth.path_type,
        path_order=truth.path_order,
        polarization=truth.polarization,
        aoa_azimuth_deg=truth.aoa_azimuth_deg,
        aoa_elevation_deg=truth.aoa_elevation_deg,
        aod_azimuth_deg=truth.aod_azimuth_deg,
        aod_elevation_deg=truth.aod_elevation_deg,
        carrier_frequency_hz=truth.carrier_frequency_hz,
        true_range_m=truth.true_range_m,
        los=truth.los,
        sync_bias_s=float(timing_cfg.get("sync_bias_s", truth.sync_bias_s)),
        clock_bias_s=float(timing_cfg.get("clock_bias_s", truth.clock_bias_s)),
        rtt_mode=bool(timing_cfg.get("rtt_mode", truth.rtt_mode)),
        metadata={
            **truth.metadata,
            "timing_bias_s": total_bias_s,
            "sfo_ppm": sfo_ppm,
            "sfo_scale": sfo_scale,
            "cfo_hz": cfo_hz,
            "antenna_pcv_magnitude_m": pcv_magnitude_m,
        },
    )
