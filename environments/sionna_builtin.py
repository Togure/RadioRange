"""Sionna built-in scene loader — munich, etoile, florence, etc.

Each scene ships as a Mitsuba 3 XML file inside a namespace package under
``sionna.rt.scenes``.  This module resolves the scene name to its XML path,
loads it via ``load_scene``, and feeds it through the shared RT pipeline.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from core.models import ChannelTruth


def generate_sionna_builtin_truths(
    config: dict[str, Any],
    rng: np.random.Generator,
    carrier_frequency_hz: float | None = None,
) -> list[ChannelTruth]:
    env_cfg = config["environment"]
    timing_cfg = config.get("timing", {})
    scene_name = str(env_cfg["scene_name"]).lower()

    tx = np.asarray(env_cfg["tx_position_m"], dtype=float)
    rx = np.asarray(env_cfg["rx_position_m"], dtype=float)

    xml_path = _resolve_scene_path(scene_name)

    from sionna.rt import load_scene

    from environments.sionna_runner import (
        paths_to_channel_truth,
        run_sionna_rt_from_scene,
    )

    scene = load_scene(xml_path, merge_shapes=False)
    path_records = run_sionna_rt_from_scene(scene, env_cfg, carrier_frequency_hz)

    return paths_to_channel_truth(
        path_records=path_records,
        rx=rx,
        tx=tx,
        env_cfg=env_cfg,
        timing_cfg=timing_cfg,
        carrier_frequency_hz=carrier_frequency_hz,
        environment="sionna_builtin_scene",
        channel_source=f"sionna_rt_{scene_name}",
        extra_metadata={"scene_name": scene_name},
    )


def _resolve_scene_path(scene_name: str) -> str:
    """Look up a Sionna built-in scene and return the absolute XML path."""
    import importlib

    try:
        mod = importlib.import_module(f"sionna.rt.scenes.{scene_name}")
    except ImportError:
        available = list_builtin_scenes()
        raise ValueError(
            f"Unknown Sionna built-in scene: '{scene_name}'. "
            f"Available: {available}"
        ) from None

    # Each scene package contains one .xml file
    for p in mod.__path__:
        for fn in os.listdir(p):
            if fn.endswith(".xml"):
                return os.path.join(p, fn)

    raise FileNotFoundError(
        f"No .xml file found in sionna.rt.scenes.{scene_name}"
    )


def list_builtin_scenes() -> list[str]:
    """Return the names of all available Sionna built-in scenes."""
    import sionna.rt.scenes as _scenes

    names: list[str] = []
    for p in _scenes.__path__:
        if not os.path.isdir(p):
            continue
        for entry in sorted(os.listdir(p)):
            full = os.path.join(p, entry)
            if os.path.isdir(full) and not entry.startswith("_"):
                # Check it contains at least one .xml file
                if any(
                    f.endswith(".xml")
                    for f in os.listdir(full)
                ):
                    names.append(entry)
    return names
