"""Floorplan image → 3D scene → Sionna RT channel truths.

Users draw a top-down floorplan PNG where each wall material is a distinct
colour.  The module classifies pixels, extracts axis-aligned wall rectangles
via connected-components labelling, emits Mitsuba 3 XML, and runs Sionna RT.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth
from environments.sionna_runner import paths_to_channel_truth, run_sionna_rt_from_scene


def generate_floorplan_truths(
    config: dict[str, Any],
    rng: np.random.Generator,
    carrier_frequency_hz: float | None = None,
) -> list[ChannelTruth]:
    env_cfg = config["environment"]
    timing_cfg = config.get("timing", {})
    floorplan_cfg = env_cfg["floorplan"]

    tx = np.asarray(env_cfg["tx_position_m"], dtype=float)
    rx = np.asarray(env_cfg["rx_position_m"], dtype=float)

    labels, material_by_label = _parse_floorplan_image(floorplan_cfg)

    walls = _extract_walls(labels, material_by_label, floorplan_cfg)

    # Serialize wall geometry for 3D visualization (lightweight, scene-level)
    _wall_geo: list[dict[str, Any]] = []
    for w in walls:
        cx, cy, cz = w["center"]
        hx, hy, hz = w["half_extents"]
        _wall_geo.append({
            "center": [float(cx), float(cy), float(cz)],
            "half_extents": [float(hx), float(hy), float(hz)],
            "material": str(w["material"]),
        })

    from sionna.rt import load_scene_from_string

    scene_xml = _floorplan_to_xml(walls, labels, floorplan_cfg, material_by_label)
    scene = load_scene_from_string(scene_xml, merge_shapes=False)
    path_records = run_sionna_rt_from_scene(scene, env_cfg, carrier_frequency_hz)

    return paths_to_channel_truth(
        path_records=path_records,
        rx=rx,
        tx=tx,
        env_cfg=env_cfg,
        timing_cfg=timing_cfg,
        carrier_frequency_hz=carrier_frequency_hz,
        environment="floorplan",
        channel_source="sionna_rt_floorplan",
        extra_metadata={
            "wall_materials": sorted(set(material_by_label.values())),
            "wall_geometry": _wall_geo,
            "room_size_m": [float(labels.shape[1] / floorplan_cfg["pixels_per_meter"]),
                            float(labels.shape[0] / floorplan_cfg["pixels_per_meter"])],
            "scene_xml": scene_xml,
        },
    )


# ── Image parsing ──────────────────────────────────────────────────────────

def _parse_floorplan_image(
    floorplan_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[int, str]]:
    """Classify every pixel by L2 distance to configured colours.

    Returns
    -------
    labels : np.ndarray  (H, W) int32
        -1 for background, 0..M-1 for each material class.
    material_by_label : dict[int, str]
        Maps label index to material name.
    """
    from PIL import Image

    image_path = str(floorplan_cfg["image_path"])
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img, dtype=np.float64)  # (H, W, 3)

    color_mapping = floorplan_cfg["color_mapping"]
    background_color = np.array(
        floorplan_cfg.get("background_color", [255, 255, 255]), dtype=np.float64,
    )
    default_tolerance = float(floorplan_cfg.get("default_tolerance", 30))

    H, W = pixels.shape[:2]
    labels = np.full((H, W), -1, dtype=np.int32)
    material_by_label: dict[int, str] = {}

    # Assign background first (so explicit wall colours take precedence)
    bg_dist = np.sqrt(np.sum((pixels - background_color) ** 2, axis=2))
    labels[bg_dist <= default_tolerance] = -1

    for idx, entry in enumerate(color_mapping):
        color = np.array(entry["color"], dtype=np.float64)
        tol = float(entry.get("tolerance", default_tolerance))
        dist = np.sqrt(np.sum((pixels - color) ** 2, axis=2))
        mask = dist <= tol
        labels[mask] = idx
        material_by_label[idx] = str(entry["material"])

    return labels, material_by_label


# ── Wall extraction ────────────────────────────────────────────────────────

def _extract_walls(
    labels: np.ndarray,
    material_by_label: dict[int, str],
    floorplan_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Scan-based axis-aligned wall extraction.

    Horizontal runs (per row) and vertical runs (per column) are grouped
    independently, then overlapping wall rectangles are deduplicated so
    each pixel belongs to at most one wall.
    """
    pixels_per_meter = float(floorplan_cfg["pixels_per_meter"])
    wall_height_m = float(floorplan_cfg.get("wall_height_m", 3.0))
    min_length_px = int(floorplan_cfg.get("min_wall_length_px", 3))

    all_walls: list[dict[str, Any]] = []
    used = np.zeros(labels.shape, dtype=bool)

    for label, material in material_by_label.items():
        binary = (labels == label)
        if not np.any(binary):
            continue

        h_mask, v_mask = _split_by_orientation(binary)

        h_walls = _extract_horizontal_walls(h_mask, min_length_px)
        v_walls = _extract_vertical_walls(v_mask, min_length_px)

        for w in h_walls:
            ym, yM, xm, xM = w["y_min"], w["y_max"], w["x_min"], w["x_max"]
            used[ym:yM + 1, xm:xM + 1] = True
            all_walls.append(_wall_rect(w, material, pixels_per_meter, wall_height_m, "h"))

        for w in v_walls:
            ym, yM, xm, xM = w["y_min"], w["y_max"], w["x_min"], w["x_max"]
            if not np.any(binary[ym:yM + 1, xm:xM + 1] & ~used[ym:yM + 1, xm:xM + 1]):
                continue
            all_walls.append(_wall_rect(w, material, pixels_per_meter, wall_height_m, "v"))

    return all_walls


def _split_by_orientation(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classify each wall pixel as horizontal or vertical by local run lengths.

    A pixel at (y, x) is horizontal if its contiguous run in the x-direction
    is at least as long as its run in the y-direction; vertical otherwise.
    This prevents vertical walls (tall, thin columns) from creating short
    horizontal runs that bridge unrelated horizontal wall segments.
    """
    H, W = binary.shape

    h_runs = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        run_start = 0
        in_run = False
        for x in range(W):
            if binary[y, x]:
                if not in_run:
                    run_start = x
                    in_run = True
            else:
                if in_run:
                    h_runs[y, run_start:x] = x - run_start
                    in_run = False
        if in_run:
            h_runs[y, run_start:W] = W - run_start

    v_runs = np.zeros((H, W), dtype=np.int32)
    for x in range(W):
        run_start = 0
        in_run = False
        for y in range(H):
            if binary[y, x]:
                if not in_run:
                    run_start = y
                    in_run = True
            else:
                if in_run:
                    v_runs[run_start:y, x] = y - run_start
                    in_run = False
        if in_run:
            v_runs[run_start:H, x] = H - run_start

    h_mask = (h_runs >= v_runs) & binary
    v_mask = (v_runs > h_runs) & binary
    return h_mask, v_mask


def _wall_rect(
    w: dict[str, int],
    material: str,
    ppm: float,
    wall_height_m: float,
    orient: str,
) -> dict[str, Any]:
    """Convert pixel bounding-box to world-space cube dict."""
    y_min, y_max = w["y_min"], w["y_max"]
    x_min, x_max = w["x_min"], w["x_max"]
    center_x = (x_min + x_max) / 2.0 / ppm
    center_y = (y_min + y_max) / 2.0 / ppm
    half_x = (x_max - x_min + 1) / 2.0 / ppm
    half_y = (y_max - y_min + 1) / 2.0 / ppm
    return {
        "name": f"wall_{orient}_{y_min}_{x_min}",
        "center": (center_x, center_y, wall_height_m / 2.0),
        "half_extents": (half_x, half_y, wall_height_m / 2.0),
        "material": material,
    }


def _extract_horizontal_walls(
    binary: np.ndarray, min_length_px: int,
) -> list[dict[str, int]]:
    """Row-scan: group overlapping runs in adjacent rows → horizontal wall rects."""
    H, W = binary.shape
    runs: list[tuple[int, int, int]] = []  # (row, x_start, x_end)

    for y in range(H):
        row = binary[y]
        in_run = False
        start = 0
        for x in range(W):
            if row[x] and not in_run:
                start = x
                in_run = True
            elif not row[x] and in_run:
                if x - start >= min_length_px:
                    runs.append((y, start, x - 1))
                in_run = False
        if in_run and W - start >= min_length_px:
            runs.append((y, start, W - 1))

    return _merge_runs(runs, axis="y")


def _extract_vertical_walls(
    binary: np.ndarray, min_length_px: int,
) -> list[dict[str, int]]:
    """Column-scan: group overlapping runs in adjacent columns → vertical wall rects."""
    H, W = binary.shape
    runs: list[tuple[int, int, int]] = []  # (col, y_start, y_end)

    for x in range(W):
        col = binary[:, x]
        in_run = False
        start = 0
        for y in range(H):
            if col[y] and not in_run:
                start = y
                in_run = True
            elif not col[y] and in_run:
                if y - start >= min_length_px:
                    runs.append((x, start, y - 1))
                in_run = False
        if in_run and H - start >= min_length_px:
            runs.append((x, start, H - 1))

    rects = _merge_runs(runs, axis="x")
    # Convert column-based rects back to (x_min, x_max, y_min, y_max)
    result: list[dict[str, int]] = []
    for r in rects:
        result.append({
            "x_min": r["x_min"], "x_max": r["x_max"],
            "y_min": r["y_min"], "y_max": r["y_max"],
        })
    return result


def _merge_runs(
    runs: list[tuple[int, int, int]], axis: str,
) -> list[dict[str, int]]:
    """Union-find merge of runs that are adjacent along *axis* and overlap."""
    n = len(runs)
    if n == 0:
        return []
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    # Index runs by their primary axis position
    by_pos: dict[int, list[int]] = {}
    for i, (pos, s, e) in enumerate(runs):
        by_pos.setdefault(pos, []).append(i)

    positions = sorted(by_pos.keys())
    for idx, pos in enumerate(positions):
        next_pos = pos + 1
        if next_pos not in by_pos:
            # Allow small gaps (door openings)
            for gap in range(2, 6):
                if next_pos + gap - 1 in by_pos:
                    next_pos = next_pos + gap - 1
                    break
            else:
                continue
        for i in by_pos[pos]:
            _, s_i, e_i = runs[i]
            for j in by_pos.get(next_pos, []):
                _, s_j, e_j = runs[j]
                if s_i <= e_j and s_j <= e_i:
                    _union(i, j)

    # Collect groups
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    rects: list[dict[str, int]] = []
    for indices in groups.values():
        if axis == "y":
            positions_list = [runs[i][0] for i in indices]  # rows
            starts = [runs[i][1] for i in indices]
            ends = [runs[i][2] for i in indices]
            rects.append({
                "y_min": min(positions_list), "y_max": max(positions_list),
                "x_min": min(starts), "x_max": max(ends),
            })
        else:
            positions_list = [runs[i][0] for i in indices]  # cols
            starts = [runs[i][1] for i in indices]
            ends = [runs[i][2] for i in indices]
            rects.append({
                "x_min": min(positions_list), "x_max": max(positions_list),
                "y_min": min(starts), "y_max": max(ends),
            })

    return rects


# ── Mitsuba XML generation ─────────────────────────────────────────────────

def _floorplan_to_xml(
    walls: list[dict[str, Any]],
    labels: np.ndarray,
    floorplan_cfg: dict[str, Any],
    material_by_label: dict[int, str],
) -> str:
    """Build Mitsuba 3 XML from extracted walls + optional floor/ceiling."""
    from environments.xml_builder import (
        build_bsdf_definitions,
        build_cube_shape,
        build_scene_xml,
    )

    pixels_per_meter = float(floorplan_cfg["pixels_per_meter"])
    wall_height_m = float(floorplan_cfg.get("wall_height_m", 3.0))
    H, W = labels.shape

    materials: set[str] = set()
    shape_parts: list[str] = []

    for wall in walls:
        materials.add(wall["material"])
        shape_parts.append(build_cube_shape(
            wall["name"], wall["center"], wall["half_extents"], wall["material"],
        ))

    if floorplan_cfg.get("generate_floor_ceiling", True):
        floor_mat = str(floorplan_cfg.get("floor_material", "itu_concrete"))
        ceiling_mat = str(floorplan_cfg.get("ceiling_material", "itu_ceiling_board"))
        materials.update([floor_mat, ceiling_mat])

        room_w = W / pixels_per_meter
        room_h = H / pixels_per_meter
        hw, hh = room_w / 2.0, room_h / 2.0

        shape_parts.append(build_cube_shape(
            "floor",
            (hw, hh, -0.01),
            (hw, hh, 0.01),
            floor_mat,
        ))
        shape_parts.append(build_cube_shape(
            "ceiling",
            (hw, hh, wall_height_m + 0.01),
            (hw, hh, 0.01),
            ceiling_mat,
        ))

    return build_scene_xml(build_bsdf_definitions(materials), shape_parts)
