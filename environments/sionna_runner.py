"""Shared Sionna RT runner — antenna setup, PathSolver, and path extraction.

Every Sionna-based environment (simple_room, floorplan, custom_scene,
sionna_builtin) shares the same Tx/Rx placement and ray-tracing pipeline.
This module provides the common ``run_sionna_rt_from_scene()`` entry point
that each generator calls after creating its Mitsuba scene.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth, LIGHT_SPEED_MPS


def run_sionna_rt_from_scene(
    scene,
    env_cfg: dict[str, Any],
    carrier_frequency_hz: float | None = None,
) -> list[dict[str, Any]]:
    """Set up antennas, Tx/Rx, PathSolver, and extract path records.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Already-loaded Sionna scene (from ``load_scene`` or
        ``load_scene_from_string``).
    env_cfg : dict
        The ``config["environment"]`` sub-dict.  Must contain
        ``tx_position_m``, ``rx_position_m``, and optionally
        ``max_reflections``, ``los``, ``specular_reflection``,
        ``diffuse_reflection``, ``refraction``, ``diffraction``.
    carrier_frequency_hz : float | None
        Scene centre frequency.  If non-None, ``scene.frequency`` is set.

    Returns
    -------
    list[dict]
        Path records, each a dict with keys:
        name, order, distance_m, gain, path_type, aoa_azimuth_deg,
        aoa_elevation_deg, aod_azimuth_deg, aod_elevation_deg.
    """
    from sionna.rt import (
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
    )

    tx = np.asarray(env_cfg["tx_position_m"], dtype=float)
    rx = np.asarray(env_cfg["rx_position_m"], dtype=float)
    max_reflections = int(env_cfg.get("max_reflections", 1))

    if carrier_frequency_hz is not None:
        scene.frequency = carrier_frequency_hz

    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.add(Transmitter(name="tx", position=tx.tolist()))
    scene.add(Receiver(name="rx", position=rx.tolist()))

    use_diffuse = bool(env_cfg.get("diffuse_reflection", False))
    use_refraction = bool(env_cfg.get("refraction", False))
    scattering = float(env_cfg.get("scattering_coefficient", 0.3))
    num_samples = int(env_cfg.get("num_samples", 100000))

    if use_diffuse or use_refraction:
        for mat in scene.radio_materials.values():
            if use_diffuse:
                mat.scattering_coefficient = scattering
            if use_refraction:
                itu_type = str(getattr(mat, "itu_type", ""))
                if "glass" in itu_type.lower():
                    mat.thickness = 0.02
                elif "wood" in itu_type.lower() or "chipboard" in itu_type.lower():
                    mat.thickness = 0.05
                else:
                    mat.thickness = 0.15

    paths = PathSolver()(
        scene,
        max_depth=max_reflections,
        los=bool(env_cfg.get("los", True)),
        specular_reflection=bool(env_cfg.get("specular_reflection", True)),
        diffuse_reflection=use_diffuse,
        refraction=use_refraction,
        diffraction=bool(env_cfg.get("diffraction", False)),
        samples_per_src=num_samples,
    )
    a, tau = paths.cir(out_type="numpy", normalize_delays=False)
    a_paths = np.asarray(a[0, 0, 0, 0, :, 0], dtype=np.complex128)
    tau_paths_s = np.asarray(tau[0, 0, :], dtype=float)

    theta_r = np.asarray(paths.theta_r.numpy()[0, 0, :], dtype=float)
    phi_r = np.asarray(paths.phi_r.numpy()[0, 0, :], dtype=float)
    theta_t = np.asarray(paths.theta_t.numpy()[0, 0, :], dtype=float)
    phi_t = np.asarray(paths.phi_t.numpy()[0, 0, :], dtype=float)
    interactions = _paths_to_numpy(paths.interactions)

    # Extract interaction vertices (3D coordinates of each bounce/refraction point)
    # shape: (max_depth, num_tx, num_rx, max_num_paths, 3)
    raw_vertices = _paths_to_numpy(paths.vertices)
    # Squeeze tx/rx dims: result shape (max_depth, num_paths, 3)
    if raw_vertices.ndim >= 5:
        vertices_all = raw_vertices[:, 0, 0, :, :]
    elif raw_vertices.ndim == 4:
        vertices_all = raw_vertices[:, 0, :, :]
    else:
        vertices_all = raw_vertices

    # Baseline noise floor: discard only numerically negligible paths.
    noise_floor_db = float(env_cfg.get("gain_threshold_db", -150.0))
    abs_gains = np.abs(a_paths)
    max_gain = float(np.max(abs_gains)) if abs_gains.size > 0 else 1.0
    gain_floor = max_gain * (10.0 ** (noise_floor_db / 20.0)) if max_gain > 0 else 0.0

    active = (
        (abs_gains >= gain_floor)
        & np.isfinite(tau_paths_s)
        & (tau_paths_s > 0.0)
    )
    active_indices = np.flatnonzero(active)

    # Classify every active path by type and sort by gain within each type.
    # Keep at most ``max_paths_per_type`` strongest paths per type.
    max_per_type = int(env_cfg.get("max_paths_per_type", 10))

    type_buckets: dict[str, list[int]] = {}
    for path_idx in active_indices:
        order = _interaction_order(interactions, int(path_idx))
        if order == 0:
            ptype = "LOS"
        else:
            types_in_path = _interaction_type_set(interactions, int(path_idx))
            if len(types_in_path) == 1:
                ptype = types_in_path.pop()
            else:
                ptype = "fix"
        type_buckets.setdefault(ptype, []).append(int(path_idx))

    kept_indices: list[int] = []
    for ptype, pindices in type_buckets.items():
        pindices_sorted = sorted(pindices, key=lambda i: abs_gains[i], reverse=True)
        kept_indices.extend(pindices_sorted[:max_per_type])

    # Build records for kept paths, sorted by delay
    records: list[dict[str, Any]] = []
    for path_idx in sorted(kept_indices, key=lambda i: tau_paths_s[i]):
        distance_m = float(tau_paths_s[path_idx] * LIGHT_SPEED_MPS)
        order = _interaction_order(interactions, path_idx)
        aoa_el = float(90.0 - np.degrees(theta_r[path_idx]))
        aoa_az = float(np.degrees(phi_r[path_idx]))
        aod_el = float(90.0 - np.degrees(theta_t[path_idx]))
        aod_az = float(np.degrees(phi_t[path_idx]))

        if order == 0:
            ptype = "LOS"
        else:
            types_in_path = _interaction_type_set(interactions, path_idx)
            if len(types_in_path) == 1:
                ptype = types_in_path.pop()
            else:
                ptype = "fix"

        path_vertices: list[list[float]] = []
        path_interaction_types: list[int] = []
        if vertices_all.size > 0 and order > 0:
            for vi in range(min(order, vertices_all.shape[0])):
                coord = vertices_all[vi, path_idx, :]
                if np.any(np.isfinite(coord)):
                    path_vertices.append([float(coord[0]), float(coord[1]), float(coord[2])])
                    itype = _interaction_code_at(interactions, path_idx, vi)
                    if itype is not None:
                        path_interaction_types.append(itype)

        records.append(
            {
                "name": "los" if order == 0 else f"sionna_path_{path_idx}",
                "order": order,
                "distance_m": distance_m,
                "gain": complex(a_paths[path_idx]),
                "path_type": ptype,
                "aoa_azimuth_deg": aoa_az,
                "aoa_elevation_deg": aoa_el,
                "aod_azimuth_deg": aod_az,
                "aod_elevation_deg": aod_el,
                "vertices": path_vertices,
                "interaction_types": path_interaction_types,
            }
        )
    if not records:
        raise RuntimeError(
            "Sionna RT returned no active paths for the configured Tx/Rx."
        )
    return records


def _paths_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _interaction_order(interactions: np.ndarray, path_idx: int) -> int:
    if interactions.size == 0:
        return 0
    values = np.asarray(interactions)
    if values.ndim >= 4:
        vals = values[:, 0, 0, path_idx]
    else:
        vals = values.reshape(-1, values.shape[-1])[:, path_idx]
    return int(np.count_nonzero(vals))


def _interaction_code_at(
    interactions: np.ndarray, path_idx: int, depth: int,
) -> int | None:
    """Return the interaction type code at a given depth for a path, or None."""
    vals = _paths_to_numpy(interactions)
    if vals.size == 0:
        return None
    if vals.ndim >= 4:
        col = vals[:, 0, 0, path_idx]
    else:
        col = vals[:, path_idx]
    if depth >= len(col):
        return None
    code = int(col[depth])
    return code if code != 0 else None


_INTERACTION_CODE_TO_NAME: dict[int, str] = {
    1: "specular", 2: "diffuse", 4: "refraction", 8: "diffraction",
}


def _interaction_type_set(
    interactions: np.ndarray, path_idx: int,
) -> set[str]:
    """Return the set of interaction type names present in a single path."""
    vals = _paths_to_numpy(interactions)
    if vals.size == 0:
        return set()
    if vals.dtype.kind in ("i", "u"):
        col = vals[:, path_idx] if vals.ndim <= 2 else vals[:, 0, 0, path_idx]
    else:
        col = vals[:, path_idx] if vals.ndim <= 2 else vals[:, 0, 0, path_idx]
    types: set[str] = set()
    for v in np.asarray(col).flat:
        if v == 0:
            continue
        name = _INTERACTION_CODE_TO_NAME.get(int(v))
        if name:
            types.add(name)
    return types


def _has_interaction_type(
    interactions: np.ndarray,
    path_idx: int,
    interaction_type: str,
) -> bool:
    """Check whether any interaction along a path matches the given type.

    Sionna InteractionType enum values (bitmask):
      s — specular reflection  (1)
      diffuse — diffuse scattering (2)
      r / t — refraction  (4)
      d — diffraction  (8)
    """
    vals = _paths_to_numpy(interactions)
    if vals.size == 0:
        return False
    if vals.dtype.kind in ("i", "u"):
        type_codes = {"s": 1, "diffuse": 2, "r": 4, "t": 4, "d": 8}
        code = type_codes.get(interaction_type, -1)
        if code < 0:
            return False
        col = vals[:, path_idx] if vals.ndim <= 2 else vals[:, 0, 0, path_idx]
        return bool(np.any(col == code))
    col = vals[:, path_idx] if vals.ndim <= 2 else vals[:, 0, 0, path_idx]
    return bool(np.any(np.asarray(col).astype(str) == interaction_type))


def paths_to_channel_truth(
    path_records: list[dict[str, Any]],
    rx: np.ndarray,
    tx: np.ndarray,
    env_cfg: dict[str, Any],
    timing_cfg: dict[str, Any],
    carrier_frequency_hz: float | None,
    *,
    environment: str,
    channel_source: str,
    engine: str = "sionna_rt",
    extra_metadata: dict[str, Any] | None = None,
) -> list[ChannelTruth]:
    """Build ``ChannelTruth`` objects from path records.

    Shared by all environment modules (simple_room, floorplan, custom_scene,
    sionna_builtin) so that AoA/AoD extraction, path-type arrays, and
    metadata assembly live in one place.
    """
    tau_paths_s = np.asarray(
        [p["distance_m"] / LIGHT_SPEED_MPS for p in path_records], dtype=float,
    )
    a_paths = np.asarray([p["gain"] for p in path_records], dtype=np.complex128)
    true_range_m = float(np.linalg.norm(rx - tx))

    path_type_arr = np.array(
        [
            "LOS" if (p.get("order") is not None and int(p["order"]) == 0)
            else str(p.get("path_type", "reflection"))
            for p in path_records
        ],
        dtype=object,
    )
    path_order_arr = np.array(
        [int(p.get("order") or 0) for p in path_records], dtype=int,
    )
    pol_arr = (
        np.array(
            [str(p.get("polarization", "unknown")) for p in path_records],
            dtype=object,
        )
        if any(p.get("polarization") is not None for p in path_records)
        else None
    )

    num_trials = int(env_cfg.get("num_trials", 1))

    def _first(*values: Any) -> Any:
        for v in values:
            if v is not None:
                return v
        return None

    aoa_az_vals = [
        _first(p.get("aoa_azimuth_deg"), p.get("azimuth_deg"), 0.0)
        for p in path_records
    ]
    aoa_el_raw = [
        _first(p.get("aoa_elevation_deg"), p.get("elevation_deg"))
        for p in path_records
    ]
    aoa_el = (
        None
        if all(v is None for v in aoa_el_raw)
        else np.asarray([v if v is not None else 0.0 for v in aoa_el_raw], dtype=float)
    )
    aoa_az = np.asarray(aoa_az_vals, dtype=float)

    aod_az_raw = [p.get("aod_azimuth_deg") for p in path_records]
    aod_az = (
        np.asarray([float(v) for v in aod_az_raw if v is not None], dtype=float)
        if any(v is not None for v in aod_az_raw)
        else None
    )
    aod_el_raw = [p.get("aod_elevation_deg") for p in path_records]
    aod_el = (
        np.asarray([float(v) for v in aod_el_raw if v is not None], dtype=float)
        if any(v is not None for v in aod_el_raw)
        else None
    )

    metadata: dict[str, Any] = {
        "environment": environment,
        "channel_source": channel_source,
        "engine": engine,
        "paths": path_records,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    truth = ChannelTruth(
        a_paths=a_paths,
        tau_paths_s=tau_paths_s,
        path_type=path_type_arr,
        path_order=path_order_arr,
        polarization=pol_arr,
        aoa_azimuth_deg=aoa_az,
        aoa_elevation_deg=aoa_el,
        aod_azimuth_deg=aod_az,
        aod_elevation_deg=aod_el,
        carrier_frequency_hz=carrier_frequency_hz,
        true_range_m=true_range_m,
        los=True,
        sync_bias_s=float(timing_cfg.get("sync_bias_s", 0.0)),
        clock_bias_s=float(timing_cfg.get("clock_bias_s", 0.0)),
        rtt_mode=bool(timing_cfg.get("rtt_mode", False)),
        metadata=metadata,
    )
    return [truth for _ in range(num_trials)]
