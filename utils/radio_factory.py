"""Unified radio creation — single entry point for all scripts.

Every script duplicates radio instantiation.  This module provides the
canonical defaults and factory so all scripts share one source of truth.
"""

from __future__ import annotations

from typing import Any

from core.transceivers import FiveGNrRadio, UwbImpulseRadio, WifiOfdmRadio

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol defaults — canonical, matching paper §II Table I
# ═══════════════════════════════════════════════════════════════════════════════

RADIO_DEFAULTS: dict[str, dict[str, Any]] = {
    "uwb": {
        "bandwidth_hz": 499_200_000.0,
        "cir_bins": 1024,
        "interpolation_factor": 10,
        "window": "hamming",
        "accumulation_count": 128,
        "adc_bits": 6,
        "sfo_ppm": 20.0,
        "cfo_hz": 500.0,
        "snr_db": 30,
        "observation_model": "snr",
    },
    "wifi": {
        "fft_size": 512,
        "subcarrier_spacing_hz": 312_500.0,
        "interpolation_factor": 10,
        "adc_bits": 10,
        "sfo_ppm": 10.0,
        "cfo_hz": 300.0,
        "snr_db": 30,
        "observation_model": "snr",
    },
    "fiveg": {
        "fft_size": 4096,
        "subcarrier_spacing_hz": 30_000.0,
        "interpolation_factor": 10,
        "adc_bits": 12,
        "sfo_ppm": 5.0,
        "cfo_hz": 200.0,
        "snr_db": 25,
        "observation_model": "snr",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Carrier frequencies — canonical, matching paper §II Table I
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CARRIER_FREQ_HZ: dict[str, float] = {
    "uwb": 7_987_200_000.0,
    "wifi": 5_180_000_000.0,
    "fiveg": 4_800_000_000.0,
}

RADIO_CLASSES: dict[str, type] = {
    "uwb": UwbImpulseRadio,
    "wifi": WifiOfdmRadio,
    "fiveg": FiveGNrRadio,
}


def create_radio(
    protocol: str,
    carrier_frequency_hz: float | None = None,
    overrides: dict[str, Any] | None = None,
    impairments: dict[str, Any] | None = None,
) -> UwbImpulseRadio | WifiOfdmRadio | FiveGNrRadio:
    """Create and return a single radio for *protocol*.

    Parameters
    ----------
    protocol : {"uwb", "wifi", "fiveg"}
    carrier_frequency_hz : float or None
        Override the default carrier frequency.  None → protocol default.
    overrides : dict or None
        Merged on top of the radio section within the config dict.
    impairments : dict or None
        Impairment toggle dict (e.g. ``{"enable_agc": True, ...}``).
        Stored in ``config["impairments"]`` so CIR-domain impairments
        (AGC) can read their enable toggles at observe time.

    Returns
    -------
    Radio instance ready for ``.observe()``.
    """
    if protocol not in RADIO_CLASSES:
        raise ValueError(f"Unknown protocol: {protocol}.  Choose: uwb, wifi, fiveg")

    freq = carrier_frequency_hz or DEFAULT_CARRIER_FREQ_HZ[protocol]
    proto_defaults = dict(RADIO_DEFAULTS.get(protocol, {}))
    if overrides:
        # Legacy key normalization — old scripts use deprecated key names.
        # Map them so RADIO_DEFAULTS canonical keys are always overwritten.
        if "num_bins" in overrides and "cir_bins" not in overrides:
            overrides["cir_bins"] = overrides["num_bins"]
        proto_defaults.update(overrides)

    config: dict[str, Any] = {
        "radios": {
            protocol: {
                "enabled": True,
                "carrier_frequency_hz": freq,
                **proto_defaults,
            }
        },
        "impairments": dict(impairments) if impairments else {},
        "timing": {},
    }
    return RADIO_CLASSES[protocol](config)


def create_all_radios(
    protocols: list[str] | None = None,
    carrier_freqs: dict[str, float] | None = None,
    overrides: dict[str, dict] | None = None,
    impairments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create radios for multiple protocols.  Returns {protocol: radio}.

    Parameters
    ----------
    protocols : list of str
        Default: ["uwb", "wifi", "fiveg"]
    carrier_freqs : dict or None
        {protocol: carrier_frequency_hz} overrides.
    overrides : dict or None
        {protocol: {key: val}} per-protocol overrides.
    impairments : dict or None
        Impairment toggle dict passed to every radio.
    """
    if protocols is None:
        protocols = ["uwb", "wifi", "fiveg"]
    carrier_freqs = carrier_freqs or {}
    overrides = overrides or {}

    radios: dict[str, Any] = {}
    for proto in protocols:
        radios[proto] = create_radio(
            proto,
            carrier_frequency_hz=carrier_freqs.get(proto),
            overrides=overrides.get(proto),
            impairments=impairments,
        )
    return radios


def build_radios_from_config(
    config: dict,
    *,
    protocols: list[str] | None = None,
    observation_model: str | None = None,
) -> list:
    """Build a list of radio instances from a config dict.

    Replaces the duplicated ``for proto in ["uwb", "wifi", "fiveg"]``
    loop that appeared in 4 locations across the codebase.

    Parameters
    ----------
    config : dict
        Full config with ``"radios"`` and optional ``"impairments"`` keys.
    protocols : list of str or None
        Which protocols to consider.  None → ["uwb", "wifi", "fiveg"].
    observation_model : str or None
        If set, force every radio to this observation model (e.g. ``"snr"``
        for CIR comparison plots).  None → use whatever the config says.

    Returns
    -------
    list of radio instances (only enabled protocols).
    """
    if protocols is None:
        protocols = ["uwb", "wifi", "fiveg"]
    radios = []
    radio_cfg = config.get("radios", {})
    impairments = config.get("impairments", {})
    for proto in protocols:
        proto_cfg = radio_cfg.get(proto, {})
        if not proto_cfg.get("enabled", False):
            continue
        overrides = {k: v for k, v in proto_cfg.items()
                     if k not in ("enabled", "carrier_frequency_hz")}
        if observation_model:
            overrides["observation_model"] = observation_model
            # Carry over explicit-impairment SNR for SNR-model plots
            explicit = proto_cfg.get("explicit_impairments", {})
            if "csi_noise_snr_db" in explicit:
                overrides["snr_db"] = explicit["csi_noise_snr_db"]
        cfreq = proto_cfg.get("carrier_frequency_hz", None)
        radios.append(create_radio(proto, carrier_frequency_hz=cfreq,
                                   overrides=overrides, impairments=impairments))
    return radios
