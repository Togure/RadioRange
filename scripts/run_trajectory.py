#!/usr/bin/env python3
"""
Trajectory Ranging Simulation — User Walking Through a Floorplan Environment
============================================================================

A user (RX device) walks along a smooth trajectory through a floorplan while
one or more fixed anchors (TX) provide ranging signals.  At each waypoint the
full RF pipeline runs: Sionna RT → impairments → observation → ranging →
evaluation.  Supports UWB, WiFi, and 5G simultaneously.

This module is a **shared simulation engine** — it exports ``build_config()``,
``run_trajectory_simulation()``, ``plot_results()``, ``save_results_csv()``,
and ``save_summary()``, which are imported by ``scripts/run_measure.py``.

**The public CLI entry point is** ``radiorange --mode measure`` (via ``main.py``).
See ``MODES.md#measure`` for usage.

The default scene is a 25m × 3m straight corridor (right wall: first 1/3 glass,
rest concrete).

Outputs (via run_measure.py → outputs/measure/<scene>_<timestamp>/):
  - trajectory_results.csv          per-waypoint ranging errors
  - error_map.png                   top-down floorplan + error heatmap
  - error_vs_distance.png           ranging error vs distance along path
  - error_cdf.png                   CDF comparison across protocols
  - rmse_summary.txt                summary statistics
  - rt_cache/                       per-waypoint RT caches (reusable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── project root ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.models import ChannelTruth, LIGHT_SPEED_MPS
from environments.registry import generate_truths
from hardware.impairments import apply_timing_impairments
from algorithms import (
    ChipLeadingEdgeLde,
    LeadingEdgeLde,
    MaxPeakLde,
    SearchBackLde,
    ThresholdLde,
)
from algorithms.multipath import CFARDetector, CLEANDetector, PeakFinder
from utils.scene_presets import IMPAIRMENT_PRESETS
from utils.radio_factory import RADIO_DEFAULTS, DEFAULT_CARRIER_FREQ_HZ, create_radio
from utils.runner import _rng, _PROTO_SEEDS

# ═══════════════════════════════════════════════════════════════════════════
# corridor floorplan generation (25m × 3m, glass + concrete)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_corridor_floorplan(output_path: Path) -> None:
    """Generate straight enclosed corridor floorplan PNG.

    Layout (top-down view, x=length, y=width):
      - Corridor: 25m (x) × 3m (y), fully enclosed
      - Top wall (y≈0): all concrete
      - Bottom wall (y≈3m): first 1/3 = glass, rest = concrete
      - End walls at x=0 and x=25: concrete
      - Wall height: 3m, device height: 1.5m
    """
    from PIL import Image

    ppm = 20.0
    wt = 3  # wall thickness in pixels
    width_px = int(25.0 * ppm) + 2 * wt   # 506
    height_px = int(3.0 * ppm) + 2 * wt    # 66

    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    COLOR_CONCRETE = np.array([0, 0, 0], dtype=np.uint8)
    COLOR_GLASS = np.array([0, 0, 255], dtype=np.uint8)

    glass_end_px = int(25.0 / 3 * ppm) + wt  # ~169

    # Top wall: all concrete
    img[0:wt, :] = COLOR_CONCRETE
    # Bottom wall: glass first 1/3, then concrete
    img[height_px - wt:height_px, 0:glass_end_px] = COLOR_GLASS
    img[height_px - wt:height_px, glass_end_px:width_px] = COLOR_CONCRETE
    # End walls
    img[:, 0:wt] = COLOR_CONCRETE
    img[:, width_px - wt:width_px] = COLOR_CONCRETE

    Image.fromarray(img).save(str(output_path))
    room_w, room_h = width_px / ppm, height_px / ppm
    print(f"  Generated floorplan: {output_path}  ({room_w:.1f}×{room_h:.1f}m)")


def _generate_t_junction_floorplan(output_path: Path) -> None:
    """Generate T-junction corridor floorplan PNG.

    Layout (top-down, x=right, y=down):
      - Y corridor: 2.5m × 7.15m (vertical), x=[12,14.5], y=[0,7.15]
      - X corridor: ~14.5m × 2.5m (horizontal), x=[0,14.5], y=[0,2.5]
      - Junction: x=[12,14.5], y=[0,2.5] — X corridor right end meets Y corridor
      - Metal panel: Y left wall (x=12), y=[3.5,6]
      - All other walls: tile
      - Wall height: 3m, device height: 1.5m
    """
    from PIL import Image

    ppm = 20.0
    wt = 3  # wall thickness in pixels

    # Corridor dimensions (world metres)
    Y_L, Y_W = 7.15, 2.5      # Y corridor: length × width
    Y_X0, Y_X1 = 12.0, 14.5   # Y corridor x-range (shifted right)
    X_MAX = Y_X1               # X corridor right end = junction
    X_MIN = 0.0
    X_L = X_MAX - X_MIN        # ~14.5m

    width_px = int(X_MAX * ppm) + 2 * wt    # 14.5*20+6 = 296
    height_px = int(Y_L * ppm) + 2 * wt      # 7.15*20+6 = 149

    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    COLOR_TILE = np.array([200, 130, 60], dtype=np.uint8)
    COLOR_METAL = np.array([192, 192, 192], dtype=np.uint8)

    def _x(x_m: float) -> int:
        return int(x_m * ppm) + wt

    def _y(y_m: float) -> int:
        return int(y_m * ppm) + wt

    x0 = _x(0)
    x12 = _x(Y_X0)
    x145 = _x(Y_X1)
    y35, y6 = _y(3.5), _y(6.0)
    y715 = _y(Y_L)

    # ── Bottom wall: y=[0, 0.15], x=[0, X_MAX] ──
    img[_y(0):_y(0.15), _x(0):_x(X_MAX)] = COLOR_TILE

    # ── X corridor top wall (left of Y corridor only; right of Y = junction) ──
    img[_y(2.5):_y(2.65), _x(0):_x(Y_X0)] = COLOR_TILE

    # ── X corridor left end wall (x=0) ──
    img[_y(0.15):_y(2.65), _x(0):_x(0.15)] = COLOR_TILE

    # ── Y corridor left wall (x=12, y>2.5): tile → metal → tile ──
    img[_y(2.5):y35, x12:_x(12.15)] = COLOR_TILE
    img[y35:y6, x12:_x(12.15)] = COLOR_METAL
    img[y6:y715, x12:_x(12.15)] = COLOR_TILE

    # ── Y corridor right wall (x=14.5, y>2.5) ──
    img[_y(2.5):y715, _x(14.35):x145] = COLOR_TILE

    # ── Y corridor top end wall (y=7.15) ──
    img[y715:_y(7.30), x12:x145] = COLOR_TILE

    Image.fromarray(img).save(str(output_path))
    total_w = width_px / ppm
    total_h = height_px / ppm
    print(f"  Generated T-junction floorplan: {output_path}  ({total_w:.1f}×{total_h:.1f}m)")
    print(f"    X corridor: {X_L:.1f}×2.5m  |  Y corridor: 2.5×{Y_L:.2f}m")
    print(f"    Metal panel: Y left wall (x=12), y=3.5→6.0m  |  TX: (13.25, 7.0)")


# ═══════════════════════════════════════════════════════════════════════════
# scene parameters (default: straight corridor)
# ═══════════════════════════════════════════════════════════════════════════

CORRIDOR_L_M = 25.0
CORRIDOR_W_M = 3.0
WALL_HEIGHT_M = 3.0
DEVICE_HEIGHT_M = 1.5
PPM = 20.0

# Anchor (TX) — fixed position near the starting end
ANCHOR_POS = (2.0, CORRIDOR_W_M / 2, DEVICE_HEIGHT_M)

# Trajectory: user walks along corridor center line
TRAJECTORY_START_X = 4.0
TRAJECTORY_END_X = 23.0
TRAJECTORY_Y = CORRIDOR_W_M / 2
TRAJECTORY_Z = DEVICE_HEIGHT_M

DEFAULT_CRUISE_SPEED_MPS = 0.5
DEFAULT_SAMPLING_PERIOD_S = 0.2   # 5 Hz position sampling
DEFAULT_NUM_WAYPOINTS = 40

RT_NUM_SAMPLES = 10_000
RT_MAX_REFLECTIONS = 2

ALGORITHMS = {
    "max_peak": MaxPeakLde,
    "threshold": lambda: ThresholdLde(peak_ratio=0.18),
    "leading_edge": lambda: LeadingEdgeLde(n_sigma=4.0),
    "search_back": lambda: SearchBackLde(peak_ratio=0.18),
    "chip_lde": lambda: ChipLeadingEdgeLde(),
}

MULTIPATH_ALGORITHMS = {
    "peak_finder": lambda: PeakFinder(threshold_db=20.0),
    "cfar": lambda: CFARDetector(guard_cells=3, reference_cells=10, pf=0.01),
    "clean": lambda: CLEANDetector(max_iterations=20, residual_threshold_db=15.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    t_s: float
    x: float; y: float; z: float


@dataclass
class TrajectoryResult:
    waypoints: list[Waypoint]
    anchor_pos: tuple[float, float, float]
    # per_protocol[protocol] = dict(algorithm_name → np.array of (N,) errors in meters)
    errors: dict[str, dict[str, np.ndarray]]
    true_ranges_m: np.ndarray  # (N,) ground-truth 3D distance per waypoint
    metadata: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════
# trajectory generation — naturalistic walking model
# ═══════════════════════════════════════════════════════════════════════════

def generate_trajectory(
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    cruise_speed_mps: float = 0.5,
    sampling_period_s: float = 0.2,
    num_waypoints: int | None = None,
    lateral_sway_m: float = 0.18,
    corridor_width_m: float = 3.0,
    seed: int = 42,
) -> list[Waypoint]:
    """Generate naturalistic human walking trajectory.

    Four-layer model for realistic motion:
      1. Path geometry — straight main axis + lateral sway oscillation
      2. Speed profile — smoothstep acceleration / deceleration (C² continuous)
      3. Gait variation — ±8% speed oscillation at ~1.6 Hz (step frequency)
      4. Body sway — multi-frequency lateral oscillation:
           - ~0.25 Hz, ±18cm  (slow drift / balance correction)
           - ~0.7 Hz,  ±8cm   (body sway during gait cycle)
           - ~1.6 Hz,  ±3cm   (step-to-step lateral shift)

    The resulting waypoints have natural-looking position scatter rather than
    a perfectly straight line, mimicking real human walking with a handheld device.
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    direction = end - start
    total_length = float(np.linalg.norm(direction))
    unit_fwd = direction / total_length

    # Perpendicular unit vector in horizontal plane (for lateral sway)
    unit_perp = np.array([-unit_fwd[1], unit_fwd[0], 0.0], dtype=float)
    # Clamp lateral sway to stay within corridor
    center_y = start[1]
    half_width = (corridor_width_m / 2) - 0.3  # 30cm margin from walls
    max_sway = min(lateral_sway_m, half_width)

    # ── Speed profile: smoothstep accel (first 10%) + cruise + decel (last 10%) ──
    accel_dist = min(1.0, total_length * 0.12)
    decel_dist = min(1.0, total_length * 0.12)

    def _speed_at_arc(s: float) -> float:
        if s <= 0:
            return 0.0
        if s < accel_dist:
            t = s / accel_dist
            return cruise_speed_mps * t * t * (3.0 - 2.0 * t)
        if s > total_length - decel_dist:
            t_remaining = max(0.0, min(1.0, (total_length - s) / decel_dist))
            return cruise_speed_mps * t_remaining * t_remaining * (3.0 - 2.0 * t_remaining)
        return cruise_speed_mps

    # ── Lateral sway: multi-frequency body oscillation ──
    # Independent random phases for each component
    phase_slow = rng.uniform(0, 2 * np.pi)
    phase_body = rng.uniform(0, 2 * np.pi)
    phase_step = rng.uniform(0, 2 * np.pi)
    phase_gait = rng.uniform(0, 2 * np.pi)

    def _lateral_offset(t_s: float) -> float:
        """Compound lateral displacement at time t (meters)."""
        sway = (
            max_sway * 0.60 * np.sin(2 * np.pi * 0.25 * t_s + phase_slow)   # slow drift
            + max_sway * 0.30 * np.sin(2 * np.pi * 0.70 * t_s + phase_body)  # body sway
            + max_sway * 0.10 * np.sin(2 * np.pi * 1.60 * t_s + phase_step)  # step lateral
        )
        return float(sway)

    # ── Generate waypoints ──
    # ── Single straight segment ──
    key_pts: list[np.ndarray] = [start, end]

    all_waypoints: list[Waypoint] = []
    t_cum = 0.0

    for seg_idx in range(len(key_pts) - 1):
        seg_start = key_pts[seg_idx]
        seg_end = key_pts[seg_idx + 1]
        seg_dir = seg_end - seg_start
        seg_len = float(np.linalg.norm(seg_dir))
        if seg_len < 1e-6:
            continue
        seg_fwd = seg_dir / seg_len
        seg_perp = np.array([-seg_fwd[1], seg_fwd[0], 0.0], dtype=float)

        # Number of waypoints for this segment (proportional to length)
        seg_n = max(3, int(num_waypoints * seg_len / total_length)) if num_waypoints else None

        # Accel/decel per segment
        seg_accel = min(0.8, seg_len * 0.12)
        seg_decel = min(0.8, seg_len * 0.12)

        def _seg_speed(arc: float) -> float:
            if arc <= 0:
                return 0.0
            if seg_idx == 0 and arc < seg_accel:
                t_frac = arc / seg_accel
                return cruise_speed_mps * t_frac * t_frac * (3.0 - 2.0 * t_frac)
            if seg_idx == len(key_pts) - 2 and arc > seg_len - seg_decel:
                t_frac = max(0.0, min(1.0, (seg_len - arc) / seg_decel))
                return cruise_speed_mps * t_frac * t_frac * (3.0 - 2.0 * t_frac)
            return cruise_speed_mps

        if seg_n is not None:
            base_dt = seg_len / (cruise_speed_mps * seg_n)
            for i in range(seg_n):
                frac = (i + 0.5) / seg_n
                ss = frac * seg_len
                gait_dt = base_dt * (1.0 + 0.08 * np.sin(2 * np.pi * 1.6 * t_cum + phase_gait))
                pos = seg_start + seg_fwd * ss
                lat = _lateral_offset(t_cum)
                pos = pos + seg_perp * lat
                all_waypoints.append(Waypoint(t_s=t_cum, x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
                t_cum += gait_dt
        else:
            s_seg = 0.0
            while s_seg < seg_len:
                v_gait = 0.08 * np.sin(2 * np.pi * 1.6 * t_cum + phase_gait)
                v = _seg_speed(s_seg) * (1.0 + v_gait)
                v = max(0.05, v)
                pos = seg_start + seg_fwd * s_seg
                lat = _lateral_offset(t_cum)
                pos = pos + seg_perp * lat
                all_waypoints.append(Waypoint(t_s=t_cum, x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
                ds = v * sampling_period_s
                s_seg += ds
                t_cum += sampling_period_s

    # Ensure final point
    final_pt = key_pts[-1]
    all_waypoints.append(Waypoint(t_s=t_cum, x=float(final_pt[0]), y=float(final_pt[1]), z=float(final_pt[2])))

    return all_waypoints


def generate_trajectory_straight(
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    cruise_speed_mps: float = 0.5,
    sampling_period_s: float = 0.2,
    num_waypoints: int | None = None,
    lateral_sway_m: float = 0.18,
    corridor_width_m: float = 3.0,
    seed: int = 42,
) -> list[Waypoint]:
    """Convenience wrapper: single straight segment via generate_trajectory."""
    return generate_trajectory(
        start_xyz, end_xyz,
        cruise_speed_mps=cruise_speed_mps,
        sampling_period_s=sampling_period_s,
        num_waypoints=num_waypoints,
        lateral_sway_m=lateral_sway_m,
        corridor_width_m=corridor_width_m,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# config builder
# ═══════════════════════════════════════════════════════════════════════════




def build_config(
    floorplan_path: Path,
    anchor_pos: tuple[float, float, float],
    rx_pos: tuple[float, float, float],
    protocols: list[str],
    impairments: dict,
    color_mapping: list[dict] | None = None,
    ppm: float | None = None,
) -> dict:
    if ppm is None:
        ppm = PPM
    if color_mapping is None:
        color_mapping = [
            {"color": [0, 0, 0], "material": "itu_concrete"},
            {"color": [0, 0, 255], "material": "itu_glass"},
        ]
    return {
        "seed": 42,
        "timing": {},
        "impairments": impairments,
        "environment": {
            "type": "floorplan",
            "tx_position_m": list(anchor_pos),
            "rx_position_m": list(rx_pos),
            "max_reflections": RT_MAX_REFLECTIONS,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": False,
            "scattering_coefficient": 0.3,
            "num_samples": RT_NUM_SAMPLES,
            "max_paths_per_type": 10,
            "num_trials": 1,
            "floorplan": {
                "image_path": str(floorplan_path.resolve()),
                "pixels_per_meter": ppm,
                "wall_height_m": WALL_HEIGHT_M,
                "default_tolerance": 40,
                "color_mapping": color_mapping,
                "background_color": [255, 255, 255],
                "floor_material": "itu_concrete",
                "ceiling_material": "itu_ceiling_board",
                "generate_floor_ceiling": True,
            },
        },
        "radios": {p: {"enabled": True, "carrier_frequency_hz": DEFAULT_CARRIER_FREQ_HZ[p], **RADIO_DEFAULTS[p]} for p in protocols},
    }


# ═══════════════════════════════════════════════════════════════════════════
# simulation runner
# ═══════════════════════════════════════════════════════════════════════════

def run_trajectory_simulation(
    waypoints: list[Waypoint],
    anchor_pos: tuple[float, float, float],
    floorplan_path: Path,
    protocols: list[str],
    algo_names: list[str],
    impairments: dict,
    rt_cache_dir: Path,
    skip_rt: bool = False,
    seed: int = 42,
    color_mapping: list[dict] | None = None,
    ppm: float | None = None,
) -> TrajectoryResult:
    """Run full RF pipeline at each waypoint along the trajectory."""

    # Build algorithm instances
    algos: dict[str, Any] = {}
    for name in algo_names:
        if name in ALGORITHMS:
            algos[name] = ALGORITHMS[name]()
        else:
            print(f"  Warning: unknown algorithm '{name}', skipping")

    if not algos:
        raise ValueError("No valid algorithms specified")

    # Build radios (via shared radio_factory)
    radios: dict[str, Any] = {}
    for proto in protocols:
        radios[proto] = create_radio(proto, impairments=impairments)

    # Initialize error storage
    errors: dict[str, dict[str, list[float]]] = {
        proto: {name: [] for name in algo_names}
        for proto in protocols
    }
    true_ranges: list[float] = []
    n_total = len(waypoints)
    scene_meta: dict[str, Any] = {}  # populated from first RT truth

    for wi, wp in enumerate(waypoints):
        rx_pos = (wp.x, wp.y, wp.z)
        true_range_m = float(np.linalg.norm(
            np.array(rx_pos) - np.array(anchor_pos)
        ))
        true_ranges.append(true_range_m)

        t_start = time.time()

        for proto in protocols:
            rt_subdir = rt_cache_dir / proto / f"waypoint_{wi:04d}"

            # ── Phase 1: RT (or load from cache) ──
            if skip_rt and rt_subdir.exists():
                from environments.persistence import load_truths
                truths = load_truths(rt_subdir)
                truth = truths[0]
            else:
                cfg = build_config(floorplan_path, anchor_pos, rx_pos, [proto], impairments, color_mapping=color_mapping, ppm=ppm)
                freq = cfg["radios"][proto]["carrier_frequency_hz"]
                rng_rt = _rng(seed, wi * 100 + _PROTO_SEEDS[proto])
                truths = generate_truths(cfg, rng_rt, carrier_frequency_hz=freq)
                truth = truths[0]

                # Save to cache
                from environments.persistence import save_truths
                rt_subdir.mkdir(parents=True, exist_ok=True)
                save_truths([truth], rt_subdir)
                rt_subdir.joinpath("config.json").write_text(
                    json.dumps(cfg, indent=2, default=str), encoding="utf-8",
                )

            # Capture scene metadata from first waypoint
            if wi == 0 and proto == protocols[0]:
                scene_meta = {
                    "room_size_m": truth.metadata.get("room_size_m"),
                    "wall_geometry": truth.metadata.get("wall_geometry"),
                    "floorplan": str(floorplan_path.stem),
                    "cruise_speed_mps": DEFAULT_CRUISE_SPEED_MPS,
                }
                # Detect glass boundary from wall materials
                wall_geo = truth.metadata.get("wall_geometry")
                if wall_geo:
                    for w in wall_geo:
                        mat = w.get("material", "")
                        if "glass" in mat:
                            cx = w["center"][0]
                            hx = w["half_extents"][0]
                            scene_meta["glass_boundary_x_m"] = float(cx + hx)
                            break

            # ── Phase 2: impairments ──
            impair_rng = _rng(seed, wi * 1000 + _PROTO_SEEDS[proto] + 1)
            radio_cfg = {"sfo_ppm": 20.0, "cfo_hz": 500.0}
            impaired = apply_timing_impairments(truth, {"impairments": impairments}, impair_rng, radio_cfg=radio_cfg)

            # ── Phase 3: observation ──
            obs_rng = _rng(seed, wi * 1000 + _PROTO_SEEDS[proto] + 2)
            observation = radios[proto].observe(impaired, obs_rng)

            # ── Phase 4: first-path estimation (same as rt_cache_interactive) ──
            for algo_name, algo in algos.items():
                estimate = algo.estimate(observation)
                error_m = estimate.estimated_range_m - impaired.true_range_m
                if not np.isfinite(error_m):
                    error_m = float("nan")
                errors[proto][algo_name].append(float(error_m))

        elapsed = time.time() - t_start
        print(f"  [{wi+1:4d}/{n_total}] x={wp.x:5.1f}m  true_range={true_range_m:6.2f}m  "
              f"({elapsed:.1f}s)" + (" (cached)" if skip_rt and rt_subdir.exists() else ""))

    # Convert lists to arrays
    errors_arr: dict[str, dict[str, np.ndarray]] = {}
    for proto in protocols:
        errors_arr[proto] = {}
        for name in algo_names:
            arr = np.array(errors[proto][name], dtype=float)
            errors_arr[proto][name] = arr

    return TrajectoryResult(
        waypoints=waypoints,
        anchor_pos=anchor_pos,
        errors=errors_arr,
        true_ranges_m=np.array(true_ranges, dtype=float),
        metadata={
            "protocols": protocols,
            "algorithms": algo_names,
            "impairments": impairments,
            "room_size_m": scene_meta.get("room_size_m", [CORRIDOR_L_M, CORRIDOR_W_M]),
            "wall_geometry": scene_meta.get("wall_geometry"),
            "glass_boundary_x_m": scene_meta.get("glass_boundary_x_m"),
            "floorplan": scene_meta.get("floorplan", "unknown"),
            "cruise_speed_mps": scene_meta.get("cruise_speed_mps", 0.5),
            "wall_height_m": WALL_HEIGHT_M,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(result: TrajectoryResult, output_dir: Path) -> None:
    """Generate all trajectory visualization plots with consistent color scales."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    protocols = result.metadata["protocols"]
    algo_names = result.metadata["algorithms"]

    PROTO_COLORS = {"uwb": "#FF7A00", "wifi": "#2563EB", "fiveg": "#059669"}
    ALGO_STYLES = {
        "max_peak":     ("dotted",  "o"),
        "threshold":    ("dashed",  "s"),
        "leading_edge": ("dashdot", "D"),
        "search_back":  ((0, (3, 1, 1, 1)), "^"),   # dense dotted
        "chip_lde":     ("solid",   "v"),
    }

    xs = np.array([wp.x for wp in result.waypoints])
    ys = np.array([wp.y for wp in result.waypoints])
    ds = result.true_ranges_m

    # ── Global error scale: P10–P90, clipped (consistent across ALL plots) ──
    all_abs_errors = []
    for proto in protocols:
        for algo_name in algo_names:
            ae = np.abs(result.errors[proto][algo_name])
            all_abs_errors.append(ae[np.isfinite(ae)])
    all_err = np.concatenate(all_abs_errors) if all_abs_errors else np.array([0.0, 1.0])
    global_vmin = float(np.percentile(all_err, 10))
    global_vmax = float(np.percentile(all_err, 90))
    global_vmax = max(global_vmax, global_vmin + 0.1)  # ensure non-zero range
    global_vmin = max(global_vmin, 0.0)
    print(f"  Global error color scale: {global_vmin:.2f} → {global_vmax:.2f} m (P10–P90, outliers clipped)")

    def _clip_err(abs_err: np.ndarray) -> np.ndarray:
        """Clip errors to [vmin, vmax] so outliers don't wash out the color scale."""
        return np.clip(abs_err, global_vmin, global_vmax)

    # Scene geometry
    room_l = result.metadata.get("room_size_m", [CORRIDOR_L_M, CORRIDOR_W_M])
    room_x, room_y = room_l[0], room_l[1] if len(room_l) >= 2 else CORRIDOR_W_M
    wall_geo = result.metadata.get("wall_geometry")

    # ── Material style definitions ──
    MATERIAL_STYLES = {
        "itu_concrete":    {"fc": "#8B5E3C", "edge": "#5C3A1E", "hatch": None,
                            "alpha": 0.80, "label": "Concrete"},
        "itu_glass":       {"fc": "#BAE6FD", "edge": "#7DD3FC", "hatch": None,
                            "alpha": 0.60, "label": "Glass"},
        "itu_brick":       {"fc": "#A0522D", "edge": "#5C3A1E", "hatch": None,
                            "alpha": 0.80, "label": "Brick"},
        "itu_wood":        {"fc": "#8B6914", "edge": "#5C3A1E", "hatch": None,
                            "alpha": 0.80, "label": "Wood"},
        "itu_metal":       {"fc": "#6B5B4F", "edge": "#5C3A1E", "hatch": None,
                            "alpha": 0.85, "label": "Metal"},
        "itu_plasterboard":{"fc": "#C4A882", "edge": "#8B6B4A", "hatch": None,
                            "alpha": 0.70, "label": "Plaster"},
        "itu_ceiling_board":{"fc": "#E8D5C0", "edge": "#C4A882", "hatch": None,
                            "alpha": 0.55, "label": "Ceiling"},
    }
    _DEFAULT_MAT = {"fc": "#8B5E3C", "edge": "#5C3A1E", "hatch": None,
                    "alpha": 0.80, "label": "Wall"}

    def _draw_floorplan_background(ax):
        """Draw floorplan with material-differentiated walls, no legend."""
        if wall_geo:
            for w in wall_geo:
                cx, cy = w["center"][0], w["center"][1]
                hx, hy = w["half_extents"][0], w["half_extents"][1]
                mat = w.get("material", "")
                style = MATERIAL_STYLES.get(mat, _DEFAULT_MAT)
                ax.add_patch(patches.Rectangle(
                    (cx - hx, cy - hy), 2 * hx, 2 * hy,
                    facecolor=style["fc"], alpha=style["alpha"],
                    edgecolor=style["edge"], linewidth=1.2,
                ))
        else:
            ax.add_patch(plt.Rectangle((0, 0), room_x, room_y,
                         fill=False, edgecolor="gray", linewidth=1.5,
                         linestyle="--"))

        # TX anchor — small square
        ax.plot(result.anchor_pos[0], result.anchor_pos[1],
                "s", markersize=10, markerfacecolor="#DC2626",
                markeredgecolor="#7F1D1D", markeredgewidth=1.0,
                zorder=6)

        margin = 1.0
        ax.set_xlim(-margin, room_x + margin)
        ax.set_ylim(-margin, room_y + margin)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.18, color="#B0A090", linewidth=0.4)
        ax.set_facecolor("#FEFDFB")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1: ERROR MAP — one subplot per protocol, ALL algorithms share
    #            the same color scale per protocol row.
    # ══════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(1, len(protocols), figsize=(5.5 * len(protocols), 5))
    if len(protocols) == 1:
        axes1 = [axes1]

    for pi, proto in enumerate(protocols):
        ax = axes1[pi]
        _draw_floorplan_background(ax)

        # Show error for the *first* algorithm as colored scatter
        primary_algo = algo_names[0]
        abs_err = np.abs(result.errors[proto][primary_algo])
        scatter = ax.scatter(xs, ys, c=_clip_err(abs_err), cmap="RdYlGn_r", s=50,
                             edgecolors="#1F2937", linewidth=0.5,
                             vmin=global_vmin, vmax=global_vmax,
                             zorder=4)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label(f"|Error| (m)  [{primary_algo}]", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.set_title(f"{proto.upper()} — Error Map ({primary_algo})", fontsize=11,
                     fontweight="bold")

    fig1.suptitle(f"Ranging Error Map — {result.metadata.get('floorplan', '')}  "
                  f"(color {global_vmin:.2f}–{global_vmax:.2f}m P10–P90)",
                  fontsize=12, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(output_dir / "error_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: error_map.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 2: ERROR MAP GRID — protocol (row) × algorithm (col)
    #            Single shared colorbar, all cells use the SAME scale.
    # ══════════════════════════════════════════════════════════════════════
    n_proto = len(protocols)
    n_algo = len(algo_names)
    fig2, axes2 = plt.subplots(n_proto, n_algo,
                                figsize=(3.2 * n_algo, 3.0 * n_proto),
                                sharex=True, sharey=True)
    if n_proto == 1 and n_algo == 1:
        axes2 = np.array([[axes2]])
    elif n_proto == 1:
        axes2 = axes2[np.newaxis, :]
    elif n_algo == 1:
        axes2 = axes2[:, np.newaxis]

    for pi, proto in enumerate(protocols):
        for ai, algo_name in enumerate(algo_names):
            ax = axes2[pi, ai]
            _draw_floorplan_background(ax)

            abs_err = np.abs(result.errors[proto][algo_name])
            scatter = ax.scatter(xs, ys, c=_clip_err(abs_err), cmap="RdYlGn_r", s=22,
                                 edgecolors="#1F2937", linewidth=0.3,
                                 vmin=global_vmin, vmax=global_vmax,
                                 zorder=4)

            if pi == 0:
                ax.set_title(algo_name, fontsize=10, fontweight="bold")
            if ai == 0:
                ax.set_ylabel(f"{proto.upper()}\nY (m)", fontsize=9)
            if pi == n_proto - 1:
                ax.set_xlabel("X (m)", fontsize=9)

    # Single shared colorbar
    fig2.subplots_adjust(right=0.92, wspace=0.08, hspace=0.08)
    cbar_ax = fig2.add_axes([0.94, 0.12, 0.015, 0.74])
    norm = matplotlib.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn_r")
    fig2.colorbar(sm, cax=cbar_ax, label="|Error| (m)")
    fig2.suptitle(f"Error Map Grid — All Protocols × All Algorithms  "
                  f"(uniform scale {global_vmin:.2f}–{global_vmax:.2f}m P10–P90)",
                  fontsize=13, fontweight="bold")
    fig2.savefig(output_dir / "error_map_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: error_map_grid.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 3: Error vs distance along path
    # ══════════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(len(protocols), 1, figsize=(11, 3.0 * len(protocols)),
                                sharex=True)
    if len(protocols) == 1:
        axes3 = [axes3]

    for pi, proto in enumerate(protocols):
        ax = axes3[pi]
        for algo_name in algo_names:
            linestyle, marker = ALGO_STYLES.get(algo_name, ("solid", "o"))
            err = np.array(result.errors[proto][algo_name])
            ax.plot(ds, err, linestyle=linestyle, color=PROTO_COLORS[proto],
                    marker=marker, markersize=4, markevery=max(1, len(ds) // 15),
                    label=algo_name, alpha=0.75)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{proto.upper()}\nRange Error (m)")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        glass_end = result.metadata.get("glass_boundary_x_m")
        if glass_end is not None:
            ax.axvline(x=glass_end, color="blue", linewidth=0.8, linestyle=":", alpha=0.4)

    axes3[-1].set_xlabel("True Distance from Anchor (m)")
    fig3.suptitle("Ranging Error vs Distance Along Path", fontsize=13, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(output_dir / "error_vs_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: error_vs_distance.png")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 4: Error CDF
    # ══════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(9, 5.5))
    for proto in protocols:
        for algo_name in algo_names:
            linestyle, _ = ALGO_STYLES.get(algo_name, ("solid", "o"))
            abs_err = np.abs(result.errors[proto][algo_name])
            abs_err = abs_err[np.isfinite(abs_err)]
            if len(abs_err) == 0:
                continue
            sorted_err = np.sort(abs_err)
            cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
            ax4.plot(sorted_err, cdf, linestyle=linestyle, color=PROTO_COLORS[proto],
                     label=f"{proto.upper()} {algo_name}", linewidth=1.3)

    ax4.set_xlabel("Absolute Range Error (m)")
    ax4.set_ylabel("CDF")
    ax4.set_title("Trajectory Ranging Error CDF", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower right", fontsize=6.5, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(0, 1.05)
    fig4.tight_layout()
    fig4.savefig(output_dir / "error_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: error_cdf.png")


def save_results_csv(result: TrajectoryResult, output_dir: Path) -> None:
    """Save trajectory results as CSV."""
    rows = []
    header = ["waypoint", "t_s", "x_m", "y_m", "z_m", "true_range_m"]
    for proto in result.metadata["protocols"]:
        for algo in result.metadata["algorithms"]:
            header.append(f"{proto}_{algo}_error_m")

    for wi, wp in enumerate(result.waypoints):
        row = [str(wi), f"{wp.t_s:.3f}", f"{wp.x:.4f}", f"{wp.y:.4f}", f"{wp.z:.4f}",
               f"{result.true_ranges_m[wi]:.6f}"]
        for proto in result.metadata["protocols"]:
            for algo in result.metadata["algorithms"]:
                err = result.errors[proto][algo][wi]
                row.append(f"{err:.6f}" if np.isfinite(err) else "nan")
        rows.append(",".join(row))

    csv_path = output_dir / "trajectory_results.csv"
    csv_path.write_text("\n".join([",".join(header)] + rows), encoding="utf-8")
    print(f"  Saved: trajectory_results.csv")


def save_summary(result: TrajectoryResult, output_dir: Path) -> None:
    """Compute and save RMSE / P90 summary."""
    lines = []
    lines.append("=" * 75)
    lines.append("TRAJECTORY RANGING ERROR SUMMARY")
    lines.append("=" * 75)
    lines.append(f"{'Protocol':<8} {'Algorithm':<16} {'RMSE(m)':>10} {'P90(m)':>10} "
                 f"{'MeanErr(m)':>12} {'Std(m)':>10}")
    lines.append("-" * 75)

    for proto in result.metadata["protocols"]:
        for algo in result.metadata["algorithms"]:
            err = result.errors[proto][algo]
            valid = err[np.isfinite(err)]
            if len(valid) == 0:
                lines.append(f"{proto:<8} {algo:<16} {'N/A':>10} {'N/A':>10}")
                continue
            rmse = np.sqrt(np.mean(valid ** 2))
            p90 = np.percentile(np.abs(valid), 90)
            mean_err = np.mean(valid)
            std_err = np.std(valid)
            lines.append(f"{proto:<8} {algo:<16} {rmse:10.4f} {p90:10.4f} "
                         f"{mean_err:12.4f} {std_err:10.4f}")

    lines.append("-" * 75)
    lines.append(f"Total waypoints: {len(result.waypoints)}")
    lines.append(f"Floorplan: {result.metadata.get('floorplan', 'unknown')}")
    lines.append(f"Anchor: {result.anchor_pos}")
    lines.append(f"Speed: {result.metadata.get('cruise_speed_mps', '?')} m/s | "
                 f"Duration: {result.waypoints[-1].t_s:.1f}s")

    summary_path = output_dir / "rmse_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: rmse_summary.txt")
    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Trajectory ranging simulation — user walking through a floorplan",
    )
    parser.add_argument("--floorplan", default=None,
                        help="Path to floorplan PNG (default: auto-generate 25m×3m corridor)")
    parser.add_argument("--scene", default="corridor", choices=["corridor", "t_junction"],
                        help="Built-in scene preset (default: corridor)")
    parser.add_argument("--waypoints", type=int, default=DEFAULT_NUM_WAYPOINTS,
                        help=f"Number of waypoints (default: {DEFAULT_NUM_WAYPOINTS})")
    parser.add_argument("--speed", type=float, default=DEFAULT_CRUISE_SPEED_MPS,
                        help=f"Cruise walking speed in m/s (default: {DEFAULT_CRUISE_SPEED_MPS})")
    parser.add_argument("--impairments", default="none", choices=["none", "full"],
                        help="Impairment preset (default: none)")
    parser.add_argument("--algorithms", default="max_peak,threshold,leading_edge,search_back,chip_lde",
                        help="Comma-separated first-path algorithm names")
    parser.add_argument("--protocols", default="uwb,wifi,fiveg",
                        help="Comma-separated protocols")
    parser.add_argument("--no-rt", action="store_true",
                        help="Skip RT, use cached truths only")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: auto-generated under outputs/trajectory/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    protocols = [s.strip() for s in args.protocols.split(",")]
    algo_names = [s.strip() for s in args.algorithms.split(",")]
    impairments = IMPAIRMENT_PRESETS[args.impairments]

    # ── Floorplan & scene setup ──
    if args.floorplan:
        floorplan_path = Path(args.floorplan)
        if not floorplan_path.exists():
            print(f"Floorplan not found: {floorplan_path}")
            sys.exit(1)
        scene_name = floorplan_path.stem

    elif args.scene == "t_junction":
        floorplan_path = _PROJECT_ROOT / "floorplans" / "t_junction.png"
        if not floorplan_path.exists():
            _generate_t_junction_floorplan(floorplan_path)
        scene_name = "t_junction"

    else:  # corridor
        floorplan_path = _PROJECT_ROOT / "floorplans" / "corridor_straight_25m.png"
        if not floorplan_path.exists():
            _generate_corridor_floorplan(floorplan_path)
        scene_name = "corridor_straight"

    # ── Output directory ──
    if args.output:
        output_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "outputs" / "trajectory" / f"{scene_name}_{ts}"
    rt_cache_dir = output_dir / "rt_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Scene parameters ──
    TILE_COLORS = [
        {"color": [200, 130, 60], "material": "itu_tile"},
        {"color": [192, 192, 192], "material": "itu_metal"},
        {"color": [0, 0, 0], "material": "itu_concrete"},
        {"color": [0, 0, 255], "material": "itu_glass"},
    ]
    CORRIDOR_COLORS = [
        {"color": [0, 0, 0], "material": "itu_concrete"},
        {"color": [0, 0, 255], "material": "itu_glass"},
    ]

    if scene_name == "t_junction":
        anchor_pos = (13.25, 7.0, DEVICE_HEIGHT_M)
        color_mapping = TILE_COLORS
        scene_desc = "T-junction 14.5m+7.15m — tile walls + metal panel"
        # Two-segment trajectory: Y corridor → turn → X corridor (leftward)
        seg1_len = 6.0 - 1.25   # 4.75m down Y corridor
        seg2_len = 13.25 - 1.0  # 12.25m left along X corridor
        total_len = seg1_len + seg2_len
        n1 = max(5, int(args.waypoints * seg1_len / total_len))
        n2 = args.waypoints - n1
        seg1 = generate_trajectory(
            start_xyz=(13.25, 6.0, DEVICE_HEIGHT_M),
            end_xyz=(13.25, 1.25, DEVICE_HEIGHT_M),
            cruise_speed_mps=args.speed,
            num_waypoints=n1,
            corridor_width_m=CORRIDOR_W_M,
            seed=args.seed,
        )
        t_offset = seg1[-1].t_s + 0.2
        seg2 = generate_trajectory(
            start_xyz=(13.25, 1.25, DEVICE_HEIGHT_M),
            end_xyz=(1.0, 1.25, DEVICE_HEIGHT_M),
            cruise_speed_mps=args.speed,
            num_waypoints=n2,
            corridor_width_m=CORRIDOR_W_M,
            seed=args.seed + 1000,
        )
        # Offset second segment times and combine
        for wp in seg2:
            wp.t_s += t_offset
        waypoints = seg1 + seg2
    else:
        anchor_pos = ANCHOR_POS
        color_mapping = CORRIDOR_COLORS
        scene_desc = f"{CORRIDOR_L_M}m × {CORRIDOR_W_M}m corridor"
        waypoints = generate_trajectory(
            start_xyz=(TRAJECTORY_START_X, TRAJECTORY_Y, TRAJECTORY_Z),
            end_xyz=(TRAJECTORY_END_X, CORRIDOR_W_M / 2, TRAJECTORY_Z),
            cruise_speed_mps=args.speed,
            num_waypoints=args.waypoints,
            corridor_width_m=CORRIDOR_W_M,
            seed=args.seed,
        )

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"Trajectory Simulation — {scene_desc}")
    print(f"{'='*60}")
    print(f"  Anchor (TX): {anchor_pos}")
    print(f"  Waypoints: {len(waypoints)}  |  Duration: {waypoints[-1].t_s:.1f}s")
    print(f"  Speed: {args.speed} m/s  |  Protocols: {protocols}")
    print(f"  Algorithms: {algo_names}  |  Impairments: {args.impairments}")
    print(f"  Output: {output_dir}")
    print(f"\n  Generated {len(waypoints)} waypoints")
    print(f"  Duration: {waypoints[-1].t_s:.1f}s  "
          f"  Mean step: {(waypoints[-1].x - waypoints[0].x) / len(waypoints):.2f}m")

    # ── Run simulation ──
    print(f"\n{'─'*60}")
    print(f"Running simulation...")
    if args.no_rt:
        print(f"  (RT skipped — loading from cache)")
    print(f"{'─'*60}")

    t0 = time.time()
    result = run_trajectory_simulation(
        waypoints=waypoints,
        anchor_pos=anchor_pos,
        floorplan_path=floorplan_path,
        protocols=protocols,
        algo_names=algo_names,
        impairments=impairments,
        rt_cache_dir=rt_cache_dir,
        skip_rt=args.no_rt,
        seed=args.seed,
        color_mapping=color_mapping,
    )
    total_elapsed = time.time() - t0
    print(f"\n  Simulation complete in {total_elapsed:.1f}s "
          f"({total_elapsed / len(waypoints):.1f}s per waypoint)")

    # ── Save results ──
    print(f"\n{'─'*60}")
    print(f"Saving results...")
    print(f"{'─'*60}")
    save_results_csv(result, output_dir)
    save_summary(result, output_dir)
    plot_results(result, output_dir)

    # ── Save metadata ──
    meta = {
        "corridor_l_m": CORRIDOR_L_M, "corridor_w_m": CORRIDOR_W_M,
        "wall_height_m": WALL_HEIGHT_M,
        "anchor_pos": list(ANCHOR_POS),
        "trajectory_start_x": TRAJECTORY_START_X,
        "trajectory_end_x": TRAJECTORY_END_X,
        "num_waypoints": len(waypoints),
        "cruise_speed_mps": args.speed,
        "duration_s": waypoints[-1].t_s,
        "protocols": protocols,
        "algorithms": algo_names,
        "impairments": args.impairments,
        "total_runtime_s": total_elapsed,
    }
    output_dir.joinpath("metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
