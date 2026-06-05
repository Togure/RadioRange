from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from core.models import ChannelTruth
from environments.materials import (
    _DEFAULT_MATERIAL,
    _fresnel_reflection_power,
    _resolve_wall_materials,
    _wall_dir_to_name,
)
from environments.sionna_runner import (
    paths_to_channel_truth,
    run_sionna_rt_from_scene,
)


def generate_simple_room_truths(
    config: dict[str, Any],
    rng: np.random.Generator,
    carrier_frequency_hz: float | None = None,
) -> list[ChannelTruth]:
    env_cfg = config["environment"]
    timing_cfg = config.get("timing", {})

    global_engine = str(config.get("channel_engine", "auto")).lower()
    if global_engine == "numpy":
        engine = "image_method"
    elif global_engine == "sionna":
        engine = "sionna_rect_room"
    else:
        engine = str(env_cfg.get("engine", "image_method"))
    if engine == "sionna_box":
        engine = "sionna_rect_room"

    dimensions_m = np.asarray(env_cfg["dimensions_m"], dtype=float)
    tx = np.asarray(env_cfg["tx_position_m"], dtype=float)
    rx = np.asarray(env_cfg["rx_position_m"], dtype=float)
    if dimensions_m.shape != tx.shape or tx.shape != rx.shape:
        raise ValueError(
            "simple_room dimensions_m, tx_position_m, and rx_position_m must match."
        )
    if dimensions_m.size not in (2, 3):
        raise ValueError("simple_room supports 2D or 3D rectangular rooms.")

    wall_materials = _resolve_wall_materials(env_cfg)

    if engine == "sionna_rect_room":
        if dimensions_m.size != 3:
            raise ValueError("Sionna RT simple_room engine requires a 3D room.")
        path_records = _sionna_rect_room_paths(env_cfg, wall_materials, carrier_frequency_hz)
        channel_source = "sionna_rt_rect_room"
    elif engine == "image_method":
        max_reflections = int(env_cfg.get("max_reflections", 1))
        freq = carrier_frequency_hz or float(
            env_cfg.get("carrier_frequency_hz", 5e9)
        )
        raw_loss = env_cfg.get("reflection_loss")
        if raw_loss is not None:
            # Backward compat: single reflection loss for all walls.
            wall_reflection_coeffs = {
                w: float(raw_loss) for w in wall_materials
            }
        else:
            wall_reflection_coeffs = {
                w: _fresnel_reflection_power(freq, wall_materials[w])
                for w in wall_materials
            }
        path_records = _image_method_paths(
            dimensions_m=dimensions_m,
            tx=tx,
            rx=rx,
            max_reflections=max_reflections,
            wall_reflection_coeffs=wall_reflection_coeffs,
        )
        channel_source = "image_method_room"
    else:
        raise ValueError(f"Unknown simple_room engine: {engine}")

    return paths_to_channel_truth(
        path_records=path_records,
        rx=rx,
        tx=tx,
        env_cfg=env_cfg,
        timing_cfg=timing_cfg,
        carrier_frequency_hz=carrier_frequency_hz,
        environment="simple_room",
        channel_source=channel_source,
        engine=engine,
        extra_metadata={
            "room_dimension": f"{dimensions_m.size}D",
            "wall_materials": dict(wall_materials),
        },
    )


def _sionna_rect_room_paths(
    env_cfg: dict[str, Any],
    wall_materials: dict[str, str],
    carrier_frequency_hz: float | None = None,
) -> list[dict[str, Any]]:
    from sionna.rt import load_scene_from_string

    dimensions_m = np.asarray(env_cfg["dimensions_m"], dtype=float)

    scene = load_scene_from_string(
        _rect_room_xml(
            dimensions_m=dimensions_m,
            wall_materials=wall_materials,
            wall_thickness_m=float(env_cfg.get("wall_thickness_m", 0.1)),
        ),
        merge_shapes=False,
    )
    return run_sionna_rt_from_scene(scene, env_cfg, carrier_frequency_hz)


def _rect_room_xml(
    dimensions_m: np.ndarray,
    wall_materials: dict[str, str],
    wall_thickness_m: float,
) -> str:
    """Generate a Sionna-compatible Mitsuba 3 scene with per-wall radio materials.

    Uses ``xml_builder`` for all XML generation; only the room-specific
    wall positions and half-extents are computed here.
    """
    from environments.xml_builder import (
        build_bsdf_definitions,
        build_cube_shape,
        build_scene_xml,
    )

    dx, dy, dz = [float(v) for v in dimensions_m]
    t = max(float(wall_thickness_m), 0.01)
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    ht = t / 2.0

    walls = [
        ("floor",        (hx, hy, -ht),      (hx, hy, ht)),
        ("ceiling",      (hx, hy, dz + ht),   (hx, hy, ht)),
        ("wall_x_min",   (-ht, hy, hz),       (ht, hy, hz)),
        ("wall_x_max",   (dx + ht, hy, hz),   (ht, hy, hz)),
        ("wall_y_min",   (hx, -ht, hz),       (hx, ht, hz)),
        ("wall_y_max",   (hx, dy + ht, hz),   (hx, ht, hz)),
    ]

    materials = set(wall_materials.values())

    shape_parts: list[str] = []
    for name, center, half in walls:
        mat = wall_materials.get(name, _DEFAULT_MATERIAL)
        shape_parts.append(build_cube_shape(name, center, half, mat))

    return build_scene_xml(
        build_bsdf_definitions(materials),
        shape_parts,
    )


def _image_method_paths(
    dimensions_m: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
    max_reflections: int,
    wall_reflection_coeffs: dict[str, float],
) -> list[dict[str, Any]]:
    """Image-method path enumeration with per-wall reflection coefficients.

    Each axis mirrors the transmitter across a pair of parallel walls.
    *wall_reflection_coeffs* maps wall names ("wall_x_max", "floor", ...) to
    Fresnel power reflection coefficients Γ ∈ (0, 1).

    For a path with reflections on walls w₁, w₂, ... the total reflected power
    is ∏ Γ_wᵢ.  The free-space path loss factor is 1/distance_m.

    When a wall name is missing from the coeff dict the gain for that bounce
    is silently skipped (treated as Γ = 1).
    """
    ndim = dimensions_m.size
    records: list[dict[str, Any]] = []
    axis_range = list(range(-max_reflections, max_reflections + 1))

    for reflection_order in range(max_reflections + 1):
        for axes in product(axis_range, repeat=ndim):
            if sum(abs(axis) for axis in axes) != reflection_order:
                continue

            # Image of transmitter → arrival direction at receiver (AoA)
            image_tx = tx.copy()
            wall_labels: list[str] = []
            bounce_wall_names: list[str] = []
            for dim, axis in enumerate(axes):
                if axis == 0:
                    continue
                img = float(tx[dim])
                start_max = axis > 0
                for i in range(abs(axis)):
                    if (start_max and i % 2 == 0) or (not start_max and i % 2 == 1):
                        img = 2.0 * dimensions_m[dim] - img  # max wall
                        bounce_wall_names.append(_wall_dir_to_name(dim, +1))
                    else:
                        img = -img  # min wall
                        bounce_wall_names.append(_wall_dir_to_name(dim, -1))
                image_tx[dim] = img
                # Label: "x×2", "z×1", etc.
                axis_name = {0: "x", 1: "y", 2: "z"}[dim]
                wall_labels.append(f"{axis_name}×{abs(axis)}")

            aoa_vector = rx - image_tx
            distance_m = float(np.linalg.norm(aoa_vector))
            if distance_m <= 0:
                continue

            aoa_az, aoa_el = _angles_deg(aoa_vector)

            # Image of receiver → departure direction at transmitter (AoD)
            image_rx = rx.copy()
            for dim, axis in enumerate(axes):
                if axis == -1:
                    image_rx[dim] = -rx[dim]
                elif axis == 1:
                    image_rx[dim] = 2.0 * dimensions_m[dim] - rx[dim]

            aod_vector = image_rx - tx
            aod_az, aod_el = _angles_deg(aod_vector)

            # Gain = ∏Γ_wall / distance  (per-wall Fresnel × free-space path loss)
            reflection_gain = 1.0
            for wall_name in bounce_wall_names:
                reflection_gain *= wall_reflection_coeffs.get(wall_name, 1.0)
            gain_mag = reflection_gain / distance_m
            records.append(
                {
                    "name": "los" if reflection_order == 0 else "+".join(wall_labels),
                    "order": reflection_order,
                    "distance_m": distance_m,
                    "gain": complex(gain_mag, 0.0),
                    "aoa_azimuth_deg": aoa_az,
                    "aoa_elevation_deg": aoa_el,
                    "aod_azimuth_deg": aod_az,
                    "aod_elevation_deg": aod_el,
                }
            )

    records.sort(key=lambda item: item["distance_m"])
    return records


def _angles_deg(vector: np.ndarray) -> tuple[float, float | None]:
    azimuth = float(np.degrees(np.arctan2(vector[1], vector[0])))
    if vector.size == 2:
        return azimuth, None
    horizontal = float(np.linalg.norm(vector[:2]))
    elevation = float(np.degrees(np.arctan2(vector[2], horizontal)))
    return azimuth, elevation
