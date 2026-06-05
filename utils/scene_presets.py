"""Scene presets — canonical scene names shared by main.py and all scripts.

Extracted from main.py so scripts can reference the same scene definitions
without importing the CLI entry point.
"""

from __future__ import annotations

from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Scene presets — same as main.py:_SCENE_PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

SCENE_PRESETS: dict[str, dict[str, Any]] = {
    # ═════════════════════════════════════════════════════════════════════
    # Category 1 — Statistical models (3GPP TR 38.901, no materials)
    # ═════════════════════════════════════════════════════════════════════
    "tdl_a": {
        "category": "statistical",
        "description": "TDL-A  (NLOS, 23 taps)",
        "environment": {"type": "standard_tdl", "model": "TDL-A"},
    },
    "tdl_b": {
        "category": "statistical",
        "description": "TDL-B  (NLOS, 23 taps)",
        "environment": {"type": "standard_tdl", "model": "TDL-B"},
    },
    "tdl_c": {
        "category": "statistical",
        "description": "TDL-C  (NLOS, 24 taps)",
        "environment": {"type": "standard_tdl", "model": "TDL-C"},
    },
    "tdl_d": {
        "category": "statistical",
        "description": "TDL-D  (LOS, 13 taps + K-factor)",
        "environment": {"type": "standard_tdl", "model": "TDL-D"},
    },
    "tdl_e": {
        "category": "statistical",
        "description": "TDL-E  (LOS, 14 taps + K-factor)",
        "environment": {"type": "standard_tdl", "model": "TDL-E"},
    },

    # ═════════════════════════════════════════════════════════════════════
    # Category 2 — Deterministic (no materials)
    # ═════════════════════════════════════════════════════════════════════
    "two_path": {
        "category": "deterministic",
        "description": "LOS + ground reflection — two-path model",
        "environment": {"type": "manual_paths"},
    },

    # ═════════════════════════════════════════════════════════════════════
    # Category 3 — Sionna built-in scenes (materials baked into the scene)
    # ═════════════════════════════════════════════════════════════════════
    "box": {
        "category": "sionna_builtin",
        "description": "Simple enclosed box (canonical)",
        "environment": {
            "type": "sionna_builtin_scene",
            "scene_name": "box",
            "tx_position_m": [-1.5, 0.0, 1.0],
            "rx_position_m": [1.5, 0.0, 1.0],
            "max_reflections": 3,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "scattering_coefficient": 0.3,
            "num_samples": 10000,
            "max_paths_per_type": 10,
        },
    },
    "box_knife": {
        "category": "sionna_builtin",
        "description": "Box with knife-edge obstacle",
        "environment": {
            "type": "sionna_builtin_scene",
            "scene_name": "box_knife",
            "tx_position_m": [-2.0, -1.0, 0.2],
            "rx_position_m": [2.0, 1.0, 0.2],
            "max_reflections": 3,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "scattering_coefficient": 0.3,
            "num_samples": 10000,
            "max_paths_per_type": 10,
        },
    },
    "box_knife_concrete": {
        "category": "sionna_builtin",
        "description": "Box + knife-edge, all-concrete walls",
        "environment": {
            "type": "sionna_builtin_scene",
            "scene_name": "box_knife_concrete",
            "tx_position_m": [-2.0, -1.0, 0.2],
            "rx_position_m": [2.0, 1.0, 0.2],
            "max_reflections": 3,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "scattering_coefficient": 0.3,
            "num_samples": 10000,
            "max_paths_per_type": 10,
        },
    },
    "munich": {
        "category": "sionna_builtin",
        "description": "Munich city centre",
        "environment": {
            "type": "sionna_builtin_scene",
            "scene_name": "munich",
            "tx_position_m": [-100.0, 50.0, 1.5],
            "rx_position_m": [-40.0, 50.0, 1.5],
            "max_reflections": 3,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "scattering_coefficient": 0.3,
            "num_samples": 200000,
            "max_paths_per_type": 10,
        },
    },
    "etoile": {
        "category": "sionna_builtin",
        "description": "Etoile urban scene",
        "environment": {
            "type": "sionna_builtin_scene",
            "scene_name": "etoile",
            "tx_position_m": [-20.0, 30.0, 1.5],
            "rx_position_m": [40.0, -20.0, 1.5],
            "max_reflections": 3,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "scattering_coefficient": 0.3,
            "num_samples": 10000,
            "max_paths_per_type": 10,
        },
    },
    "florence": {
        "category": "sionna_builtin",
        "description": "Florence city scene",
        "environment": {
            "type": "sionna_builtin_scene",
            "scene_name": "florence",
            "tx_position_m": [300.0, 300.0, 2.0],
            "rx_position_m": [350.0, 350.0, 2.0],
            "max_reflections": 3,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "scattering_coefficient": 0.3,
            "num_samples": 10000,
            "max_paths_per_type": 10,
        },
    },

    # ═════════════════════════════════════════════════════════════════════
    # Category 4 — Procedural rooms (materials defined in preset)
    # ═════════════════════════════════════════════════════════════════════
    "simple_room": {
        "category": "procedural",
        "description": "4×4×3m rectangular room (Sionna RT)",
        "environment": {
            "type": "simple_room",
            "engine": "sionna_rect_room",
            "dimensions_m": [4.0, 4.0, 3.0],
            "tx_position_m": [2.0, 1.0, 1.5],
            "rx_position_m": [3.0, 3.0, 1.5],
            "max_reflections": 1,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "diffraction": False,
            "materials": {
                "floor": "itu_concrete",
                "ceiling": "itu_ceiling_board",
                "wall_x": "itu_brick",
                "wall_y": "itu_plasterboard",
            },
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Impairment presets — same as main.py:_IMPAIRMENT_PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

IMPAIRMENT_PRESETS: dict[str, dict] = {
    "none": {
        "enable_antenna_offset": False,
        "enable_sfo": False,
        "enable_cfo": False,
        "enable_adc_phase_offset": False,
        "enable_agc": False,
        "enable_iq_imbalance": False,
    },
    "full": {
        "enable_antenna_offset": True,
        "enable_sfo": True,
        "enable_cfo": True,
        "enable_adc_phase_offset": True,
        "enable_agc": True,
        "enable_iq_imbalance": True,
    },
}


def get_scene(name: str) -> dict[str, Any]:
    """Return a deep copy of the scene preset, or raises KeyError."""
    import copy
    if name not in SCENE_PRESETS:
        raise KeyError(f"Unknown scene: '{name}'. Available: {list(SCENE_PRESETS)}")
    return copy.deepcopy(SCENE_PRESETS[name])


def get_impairments(preset: str) -> dict:
    """Return a deep copy of the impairment preset."""
    import copy
    if preset not in IMPAIRMENT_PRESETS:
        raise KeyError(f"Unknown impairments preset: '{preset}'. Choose: none, full")
    return copy.deepcopy(IMPAIRMENT_PRESETS[preset])
