"""Custom 3D scene import — OBJ/PLY meshes and pre-built Mitsuba XML.

Supports two modes:

  **Mode A — mesh file**
  Reference an external ``.obj`` or ``.ply`` mesh with a uniform radio material.
  The module wraps it in a Mitsuba 3 scene via ``xml_builder``.

  **Mode B — pre-built XML**
  The user provides a complete Mitsuba 3 XML scene file (with BSDFs, shapes,
  and transforms already defined).  The file is loaded directly.

Both modes feed into the shared ``run_sionna_rt_from_scene`` pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth


def generate_custom_scene_truths(
    config: dict[str, Any],
    rng: np.random.Generator,
    carrier_frequency_hz: float | None = None,
) -> list[ChannelTruth]:
    env_cfg = config["environment"]
    timing_cfg = config.get("timing", {})
    scene_cfg = env_cfg["custom_scene"]

    tx = np.asarray(env_cfg["tx_position_m"], dtype=float)
    rx = np.asarray(env_cfg["rx_position_m"], dtype=float)

    mesh_file = scene_cfg.get("mesh_file")
    scene_xml_file = scene_cfg.get("scene_xml_file")

    if mesh_file is not None:
        scene_xml = _mesh_to_xml(scene_cfg)
        channel_source = "sionna_rt_mesh"
    elif scene_xml_file is not None:
        scene_xml = _read_scene_xml(scene_cfg)
        channel_source = "sionna_rt_custom_xml"
    else:
        raise ValueError(
            "custom_scene requires either 'mesh_file' or 'scene_xml_file'."
        )

    from sionna.rt import load_scene_from_string

    from environments.sionna_runner import (
        paths_to_channel_truth,
        run_sionna_rt_from_scene,
    )

    scene = load_scene_from_string(scene_xml, merge_shapes=False)
    path_records = run_sionna_rt_from_scene(scene, env_cfg, carrier_frequency_hz)

    return paths_to_channel_truth(
        path_records=path_records,
        rx=rx,
        tx=tx,
        env_cfg=env_cfg,
        timing_cfg=timing_cfg,
        carrier_frequency_hz=carrier_frequency_hz,
        environment="custom_scene",
        channel_source=channel_source,
        extra_metadata={
            "mesh_file": mesh_file,
            "scene_xml_file": scene_xml_file,
            "material": scene_cfg.get("material"),
        },
    )


def _mesh_to_xml(scene_cfg: dict[str, Any]) -> str:
    """Build Mitsuba XML referencing an external OBJ or PLY mesh."""
    from pathlib import Path

    from environments.xml_builder import (
        build_bsdf_definitions,
        build_obj_shape,
        build_ply_shape,
        build_scene_xml,
    )

    filename = str(scene_cfg["mesh_file"])
    material = str(scene_cfg.get("material", "itu_concrete"))
    translate = tuple(
        float(v) for v in scene_cfg.get("translate", [0.0, 0.0, 0.0])
    )
    scale = tuple(
        float(v) for v in scene_cfg.get("scale", [1.0, 1.0, 1.0])
    )

    suffix = Path(filename).suffix.lower()
    if suffix == ".obj":
        shape = build_obj_shape(
            "custom_mesh", filename, material, translate=translate, scale=scale,
        )
    elif suffix == ".ply":
        shape = build_ply_shape(
            "custom_mesh", filename, material, translate=translate, scale=scale,
        )
    else:
        raise ValueError(
            f"Unsupported mesh format: '{suffix}'.  Use .obj or .ply."
        )

    return build_scene_xml(build_bsdf_definitions({material}), [shape])


def _read_scene_xml(scene_cfg: dict[str, Any]) -> str:
    """Read a pre-built Mitsuba 3 XML scene from disk."""
    path = str(scene_cfg["scene_xml_file"])
    with open(path, "r") as fh:
        return fh.read()
