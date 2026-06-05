from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth, LIGHT_SPEED_MPS


def generate_manual_path_truths(
    config: dict[str, Any],
    rng: np.random.Generator,
    carrier_frequency_hz: float | None = None,
) -> list[ChannelTruth]:
    env_cfg = config["environment"]
    timing_cfg = config.get("timing", {})
    paths_cfg = env_cfg.get("paths", [])
    if not paths_cfg:
        raise ValueError("manual_paths environment requires environment.paths.")

    tau_paths_s = np.asarray([float(path["tau_s"]) for path in paths_cfg], dtype=float)
    amplitudes = np.asarray([float(path.get("amplitude", 1.0)) for path in paths_cfg], dtype=float)
    phases_rad = np.asarray(
        [
            float(path.get("phase_rad", rng.uniform(0.0, 2.0 * np.pi)))
            for path in paths_cfg
        ],
        dtype=float,
    )
    a_paths = amplitudes * np.exp(1j * phases_rad)
    order = np.argsort(tau_paths_s)

    true_range_m = float(env_cfg.get("true_range_m", np.min(tau_paths_s) * LIGHT_SPEED_MPS))
    metadata_paths = []
    for idx in order:
        path = dict(paths_cfg[int(idx)])
        path.setdefault("azimuth_deg", None)
        path.setdefault("elevation_deg", None)
        metadata_paths.append(path)

    aoa_az_raw = [path.get("azimuth_deg") for path in metadata_paths]
    aoa_el_raw = [path.get("elevation_deg") for path in metadata_paths]
    if all(v is None for v in aoa_az_raw) and all(v is None for v in aoa_el_raw):
        aoa_az = aoa_el = None
    else:
        aoa_az = np.asarray([v if v is not None else 0.0 for v in aoa_az_raw], dtype=float)
        aoa_el = np.asarray([v if v is not None else 0.0 for v in aoa_el_raw], dtype=float)

    num_trials = int(env_cfg.get("num_trials", 1))

    # Per-path metadata
    path_types = np.array(
        [str(path.get("type", "manual")) for path in paths_cfg], dtype=object
    )[order]
    path_orders = np.array(
        [path.get("order", None) for path in paths_cfg], dtype=object
    )
    path_orders = path_orders[order] if any(o is not None for o in path_orders) else None
    pol_values = np.array(
        [path.get("polarization", None) for path in paths_cfg], dtype=object
    )
    pol_values = pol_values[order] if any(p is not None for p in pol_values) else None

    truth = ChannelTruth(
        a_paths=a_paths[order].astype(np.complex128),
        tau_paths_s=tau_paths_s[order],
        path_type=path_types,
        path_order=path_orders,
        polarization=pol_values,
        aoa_azimuth_deg=aoa_az,
        aoa_elevation_deg=aoa_el,
        carrier_frequency_hz=carrier_frequency_hz,
        true_range_m=true_range_m,
        los=bool(env_cfg.get("los", True)),
        sync_bias_s=float(timing_cfg.get("sync_bias_s", 0.0)),
        clock_bias_s=float(timing_cfg.get("clock_bias_s", 0.0)),
        rtt_mode=bool(timing_cfg.get("rtt_mode", False)),
        metadata={
            "environment": "manual_paths",
            "channel_source": "manual_paths",
            "paths": metadata_paths,
        },
    )
    return [truth for _ in range(num_trials)]
