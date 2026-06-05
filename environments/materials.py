"""Indoor material electromagnetic properties and wall-material resolution.

Provides the ITU-R P.2040 material database, Fresnel reflection coefficient
computation, and per-wall material assignment logic used by all environment
modules (simple_room, floorplan, custom_scene, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ── Indoor material EM properties at radio frequencies ──────────────────
# ε' = real part of relative permittivity
# σ  = conductivity [S/m]
# ior = refractive index n = √ε' (used by Sionna dielectric BSDF / Fresnel)
# type = "dielectric" for non-conductive materials, "conductor" for metals
#
# Sources: ITU-R P.2040, various IEEE measurement papers at ~2–6 GHz.
#
# Complex permittivity model (Sionna radio-material + image method Fresnel):
#   ε_c(f) = ε' - j·σ / (2π·f·ε₀)
#   Γ = |√ε_c − 1|² / |√ε_c + 1|²  (normal-incidence power reflection)

_MATERIAL_DB: dict[str, dict[str, Any]] = {
    "itu_concrete":     {"epsilon": 5.31, "sigma": 0.033,   "ior": 2.304, "type": "dielectric"},
    "itu_brick":        {"epsilon": 3.75, "sigma": 0.038,   "ior": 1.936, "type": "dielectric"},
    "itu_glass":        {"epsilon": 6.27, "sigma": 0.004,   "ior": 2.504, "type": "dielectric"},
    "itu_wood":         {"epsilon": 1.99, "sigma": 0.005,   "ior": 1.411, "type": "dielectric"},
    "itu_plasterboard": {"epsilon": 2.94, "sigma": 0.010,   "ior": 1.715, "type": "dielectric"},
    "itu_metal":        {"epsilon": 1.00, "sigma": 1e7,     "ior": None,  "type": "conductor"},
    "itu_ceiling_board":{"epsilon": 1.50, "sigma": 0.001,   "ior": 1.225, "type": "dielectric"},
    "itu_floorboard":   {"epsilon": 3.66, "sigma": 0.024,   "ior": 1.913, "type": "dielectric"},
    "itu_very_dry_ground":  {"epsilon": 3.0,  "sigma": 0.00015, "ior": 1.732, "type": "dielectric"},
    "itu_medium_dry_ground":{"epsilon": 15.0, "sigma": 0.035,   "ior": 3.873, "type": "dielectric"},
    "itu_wet_ground":   {"epsilon": 30.0, "sigma": 0.15,    "ior": 5.477, "type": "dielectric"},
    "itu_tile":         {"epsilon": 7.50, "sigma": 0.015,   "ior": 2.739, "type": "dielectric"},
}

_DEFAULT_MATERIAL = "itu_concrete"

# ── Wall naming conventions ─────────────────────────────────────────────
# 3D room: 6 walls    2D room: 4 walls (no floor/ceiling)
_WALL_NAMES_3D = ["floor", "ceiling", "wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max"]
_WALL_NAMES_2D = ["wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max"]


def get_material_props(name: str) -> dict[str, Any]:
    """Return material properties dict, falling back to *itu_concrete* on unknown name."""
    return _MATERIAL_DB.get(name, _MATERIAL_DB[_DEFAULT_MATERIAL])


def _wall_dir_to_name(dim: int, direction: int) -> str:
    """Map (dimension, direction) to wall name for image-method bounce tracking.

    dim: 0=x, 1=y, 2=z
    direction: -1 → min wall, +1 → max wall
    """
    if dim == 0:
        return "wall_x_max" if direction > 0 else "wall_x_min"
    if dim == 1:
        return "wall_y_max" if direction > 0 else "wall_y_min"
    if dim == 2:
        return "ceiling" if direction > 0 else "floor"
    raise ValueError(f"Invalid dimension: {dim}")


def _resolve_wall_materials(env_cfg: dict[str, Any]) -> dict[str, str]:
    """Resolve per-wall material names from environment config.

    Precedence:
      1. ``materials`` dict with per-wall keys (floor, ceiling, wall_x_min, ...)
      2. ``material`` single string → all walls use the same material
      3. Falls back to ``itu_concrete`` for any missing wall.
    """
    dimensions_m = np.asarray(env_cfg["dimensions_m"], dtype=float)
    wall_names = _WALL_NAMES_2D if dimensions_m.size == 2 else _WALL_NAMES_3D

    materials_cfg = env_cfg.get("materials")
    if isinstance(materials_cfg, dict):
        return {w: str(materials_cfg.get(w, _DEFAULT_MATERIAL)) for w in wall_names}

    single = str(env_cfg.get("material", _DEFAULT_MATERIAL))
    return {w: single for w in wall_names}


def _fresnel_reflection_power(frequency_hz: float, material: str) -> float:
    """Normal-incidence Fresnel power reflection coefficient Γ.

    Uses the complex permittivity of *material* at the given *frequency_hz*.
    Falls back to concrete if the material is unknown.
    """
    eps0 = 8.854187817e-12
    omega = 2.0 * np.pi * max(frequency_hz, 1e6)

    props = _MATERIAL_DB.get(material)
    if props is None:
        props = _MATERIAL_DB[_DEFAULT_MATERIAL]

    eps_real = float(props["epsilon"])
    sigma = float(props["sigma"])

    eps_imag = sigma / (omega * eps0)
    eps_c = complex(eps_real, -eps_imag)

    sqrt_eps = np.sqrt(eps_c)
    gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
    return float(np.abs(gamma) ** 2)
