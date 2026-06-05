from __future__ import annotations

import os
from typing import Any

import numpy as np

from core.models import ChannelTruth, LIGHT_SPEED_MPS

# ── 3GPP TR 38.901 Table 7.7.2 TDL power-delay profiles ──────────────────
# Normalized delays (unit: 1 ns RMS delay spread) and powers (dB).
# To get physical delays: tau = norm_delay × delay_spread_s.
# To get linear power:     p   = 10^(power_dB / 10).

_TDL_PROFILES: dict[str, dict[str, np.ndarray]] = {
    "TDL-A": {
        "delays_norm": np.array(
            [0.0000, 0.3819, 0.4025, 0.5868, 0.6390, 0.8291, 1.0779,
             1.1811, 1.5446, 1.8373, 2.2442, 2.7351, 3.3275, 4.0431,
             4.9066, 5.9464, 7.2056, 8.7314, 10.5901, 12.8603, 15.6448,
             19.0748, 22.9590],
            dtype=float,
        ),
        "powers_db": np.array(
            [-13.4, 0.0, -2.2, -4.0, -6.0, -8.1, -10.0, -11.9, -13.9,
             -15.4, -17.8, -20.3, -22.8, -22.8, -22.8, -22.8, -22.8,
             -22.8, -22.8, -22.8, -22.8, -22.8, -22.8],
            dtype=float,
        ),
    },
    "TDL-B": {
        "delays_norm": np.array(
            [0.0000, 0.1073, 0.2092, 0.3140, 0.4441, 0.6057, 0.8104,
             1.0697, 1.4009, 1.8258, 2.3713, 3.0719, 3.9715, 5.1260,
             6.6067, 8.5073, 10.9534, 14.1117, 18.1977, 23.4891, 30.3726,
             39.3558, 50.0000],
            dtype=float,
        ),
        "powers_db": np.array(
            [0.0, -0.7, -1.5, -2.2, -3.0, -3.9, -4.8, -5.8, -6.9,
             -8.1, -9.4, -10.8, -12.3, -13.9, -15.6, -17.4, -19.3,
             -21.3, -23.4, -25.6, -27.9, -30.3, -30.3],
            dtype=float,
        ),
    },
    "TDL-C": {
        "delays_norm": np.array(
            [0.0000, 0.5408, 1.0622, 1.5743, 2.0905, 2.6296, 3.2111,
             3.8608, 4.6127, 5.5121, 6.6214, 8.0898, 10.0000, 12.4527,
             15.6088, 19.5775, 24.3218, 30.0000, 36.8926, 45.1841,
             55.0534, 66.7527, 80.0000],
            dtype=float,
        ),
        "powers_db": np.array(
            [-4.8, -3.9, -3.2, -2.6, -2.0, -1.6, -1.2, -0.9, -0.7,
             -0.4, -0.3, -0.1, 0.0, -0.1, -0.3, -0.4, -0.7, -1.0,
             -1.4, -2.0, -2.7, -3.8, -3.8],
            dtype=float,
        ),
    },
    "TDL-D": {
        "delays_norm": np.array(
            [0.0000, 0.0349, 0.0810, 0.1037, 0.1468, 0.1982, 0.2587,
             0.3310, 0.4192, 0.5234, 0.6533, 0.8055, 0.9838, 1.1947,
             1.4410, 1.7293, 2.0660, 2.4590, 2.9170, 3.4489, 4.0659,
             4.7780, 5.5975],
            dtype=float,
        ),
        "powers_db": np.array(
            [-0.2, -13.5, -18.8, -21.0, -22.8, -24.2, -25.3, -26.4,
             -27.6, -28.8, -30.1, -31.5, -33.0, -34.5, -36.1, -37.8,
             -39.5, -41.3, -43.2, -45.1, -47.1, -49.2, -50.6],
            dtype=float,
        ),
    },
    "TDL-E": {
        "delays_norm": np.array(
            [0.0000, 0.0357, 0.0837, 0.1072, 0.1518, 0.2050, 0.2676,
             0.3424, 0.4337, 0.5415, 0.6759, 0.8333, 1.0178, 1.2360,
             1.4909, 1.7894, 2.1379, 2.5445, 3.0183, 3.5685, 4.2068,
             4.9436, 5.7907],
            dtype=float,
        ),
        "powers_db": np.array(
            [-0.03, -22.03, -30.71, -34.22, -37.25, -39.01, -41.12,
             -43.61, -45.85, -48.89, -51.01, -53.04, -54.80, -56.88,
             -58.69, -60.63, -62.31, -64.00, -65.43, -66.42, -67.87,
             -69.27, -70.62],
            dtype=float,
        ),
    },
}


def generate_tdl_truths(
    config: dict[str, Any],
    rng: np.random.Generator,
    carrier_frequency_hz: float | None = None,
) -> list[ChannelTruth]:
    env_cfg = config["environment"]
    timing_cfg = config.get("timing", {})

    # ── engine selection ────────────────────────────────────────────
    # "auto"  → try Sionna, fall back to numpy (default)
    # "sionna" → Sionna only; raise if unavailable
    # "numpy" → skip Sionna entirely
    engine = str(config.get("channel_engine", env_cfg.get("engine", "auto"))).lower()
    if engine == "numpy":
        sionna_tdl = None
    elif engine == "sionna":
        sionna_tdl = _try_build_sionna_tdl(env_cfg, carrier_frequency_hz)
        if sionna_tdl is None:
            raise RuntimeError(
                "channel_engine='sionna' but Sionna TDL is not available. "
                "Install sionna or set channel_engine='auto' / 'numpy'."
            )
    else:
        sionna_tdl = _try_build_sionna_tdl(env_cfg, carrier_frequency_hz)

    truths: list[ChannelTruth] = []
    num_trials = int(env_cfg.get("num_trials", 1))
    base_range_m = float(env_cfg.get("base_range_m", 10.0))
    base_tau_s = base_range_m / LIGHT_SPEED_MPS

    channel_source = "fallback_tdl" if sionna_tdl is None else "sionna_tdl"
    model = str(env_cfg.get("model", "TDL-A"))
    is_los_model = model in ("TDL-D", "TDL-E")

    for _ in range(num_trials):
        if sionna_tdl is None:
            a_paths, excess_tau_s = _fallback_tdl(env_cfg, rng)
        else:
            a_paths, excess_tau_s = _sample_sionna_tdl(sionna_tdl)

        tau_paths_s = base_tau_s + np.asarray(excess_tau_s, dtype=float)
        a_paths_arr = np.asarray(a_paths, dtype=np.complex128)

        n_paths = len(a_paths_arr)
        path_type_arr = np.full(n_paths, "statistical_multipath", dtype=object)
        if is_los_model:
            path_type_arr[0] = "LOS"

        truths.append(
            ChannelTruth(
                a_paths=a_paths_arr,
                tau_paths_s=tau_paths_s,
                path_type=path_type_arr,
                path_order=None,
                polarization=None,
                carrier_frequency_hz=carrier_frequency_hz,
                true_range_m=base_range_m,
                los=is_los_model,
                sync_bias_s=float(timing_cfg.get("sync_bias_s", 0.0)),
                clock_bias_s=float(timing_cfg.get("clock_bias_s", 0.0)),
                rtt_mode=bool(timing_cfg.get("rtt_mode", False)),
                metadata={
                    "environment": model,
                    "channel_source": channel_source,
                },
            )
        )

    return truths


def _try_build_sionna_tdl(
    env_cfg: dict[str, Any],
    carrier_frequency_hz: float | None = None,
):
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from sionna.phy.channel.tr38901 import TDL

        freq = carrier_frequency_hz or float(env_cfg.get("carrier_frequency_hz", 5e9))
        return TDL(
            model=str(env_cfg.get("model", "TDL-A")).replace("TDL-", ""),
            delay_spread=float(env_cfg.get("delay_spread_s", 30e-9)),
            carrier_frequency=freq,
            min_speed=0.0,
            max_speed=0.0,
            num_rx_ant=1,
            num_tx_ant=1,
        )
    except Exception:
        return None


def _sample_sionna_tdl(tdl):
    a, tau = tdl(batch_size=1, num_time_steps=1, sampling_frequency=160e6)
    a_np = _to_numpy(a)[0, 0, 0, 0, 0, :, 0]
    tau_np = _to_numpy(tau)[0, 0, 0, :]
    return a_np, tau_np


def _fallback_tdl(env_cfg: dict[str, Any], rng: np.random.Generator):
    """Generate TDL paths using 3GPP TR 38.901 standard power-delay profiles.

    Uses the fixed normalized delays and per-tap powers from the 3GPP spec
    (Table 7.7.2), scaled by the configured delay spread.  Rayleigh fading
    is applied independently to each tap.  For LOS models (TDL-D, TDL-E),
    the first tap represents the LOS component and is boosted by the
    appropriate K-factor.
    """
    model = str(env_cfg.get("model", "TDL-A"))
    delay_spread_s = float(env_cfg.get("delay_spread_s", 30e-9))

    profile = _TDL_PROFILES.get(model)
    if profile is None:
        raise ValueError(
            f"Unknown TDL model: {model}. "
            f"Supported: {', '.join(sorted(_TDL_PROFILES))}"
        )

    # Scale normalized delays by the desired RMS delay spread
    excess_tau_s = profile["delays_norm"] * delay_spread_s

    # Linear power from dB
    powers_linear = 10.0 ** (profile["powers_db"] / 10.0)

    # 3GPP-style K-factor for LOS models (TR 38.901 Table 7.7.2-4/5)
    _K_DB = {"TDL-D": 6.0, "TDL-E": 8.0}

    n_taps = len(excess_tau_s)
    if model in _K_DB:
        # LOS: first tap is the LOS component, remaining taps are NLOS.
        # The per-tap powers already embed the Rician K-factor.
        # TDL-D: first tap power -0.2 dB ≈ 0.955 × total power
        # TDL-E: first tap power -0.03 dB ≈ 0.993 × total power
        # The K-factor is already baked into the standardized power profile,
        # so we generate Rayleigh fading for all taps (including the first)
        # and the relative power distribution already captures the LOS/NLOS mix.
        # However, to be consistent with Sionna's output, we also add a
        # deterministic LOS component scaled by √(K/(K+1)) to the first tap.
        k_linear = 10.0 ** (_K_DB[model] / 10.0)
        los_power_frac = k_linear / (k_linear + 1.0)  # K/(K+1)
        diffuse_power_frac = 1.0 / (k_linear + 1.0)   # 1/(K+1)

        # Diffuse part: Rayleigh fading on all taps
        fading = (rng.normal(size=n_taps) + 1j * rng.normal(size=n_taps)) / np.sqrt(2.0)

        a_paths = np.sqrt(powers_linear) * fading
        # Add deterministic LOS component to first tap with random phase
        los_phase = rng.uniform(0.0, 2.0 * np.pi)
        a_paths[0] = (
            np.sqrt(los_power_frac * powers_linear[0]) * np.exp(1j * los_phase)
            + np.sqrt(diffuse_power_frac * powers_linear[0]) * fading[0]
        )
    else:
        # NLOS (TDL-A, TDL-B, TDL-C): pure Rayleigh fading
        fading = (rng.normal(size=n_taps) + 1j * rng.normal(size=n_taps)) / np.sqrt(2.0)
        a_paths = np.sqrt(powers_linear) * fading

    return a_paths, excess_tau_s


def _to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
