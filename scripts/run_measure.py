#!/usr/bin/env python3
"""
Measure Ranging Simulation — Built-in Scenes or User-Provided Floorplan + Trajectory
=====================================================================================

Generates simulated ranging measurements along a trajectory.  Two usage modes:

1. Built-in demo (no files needed):
   python3 scripts/run_measure.py --scene corridor --radios uwb

2. Custom floorplan + waypoints:
   python3 scripts/run_measure.py \
       --floorplan my_office.png --waypoints-file path.csv \
       --floorplan-width-m 15.0 --tx 2 1.5 1.5 --radios uwb

Outputs (saved to outputs/measure/<name>_<timestamp>/):
  - trajectory_results.csv          per-waypoint ranging errors
  - error_map.png / error_map_grid.png
  - error_vs_distance.png / error_cdf.png
  - rmse_summary.txt / metadata.json
  - rt_cache/                       per-waypoint RT caches (reusable)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── project root ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_trajectory import (
    Waypoint,
    TrajectoryResult,
    run_trajectory_simulation,
    plot_results,
    save_results_csv,
    save_summary,
    generate_trajectory,
    _generate_corridor_floorplan,
    _generate_t_junction_floorplan,
    DEFAULT_CRUISE_SPEED_MPS,
    CORRIDOR_L_M,
    CORRIDOR_W_M,
    WALL_HEIGHT_M,
    DEVICE_HEIGHT_M,
    ALGORITHMS,
)
from utils.scene_presets import IMPAIRMENT_PRESETS


# ═══════════════════════════════════════════════════════════════════════════
# CSV loader
# ═══════════════════════════════════════════════════════════════════════════

def _load_waypoints_csv(csv_path: Path, cruise_speed_mps: float = 0.5) -> list[Waypoint]:
    """Load waypoints from CSV (x,y,z), synthesizing timestamps from cruise speed."""
    lines = [l.strip() for l in csv_path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise ValueError(f"Waypoints file is empty: {csv_path}")

    first = lines[0].lower().replace(" ", "")
    start = 1 if first.startswith("x,") or first.startswith("t,") or first in ("x,y,z", "x;y;z") else 0

    pts: list[list[float]] = []
    for line in lines[start:]:
        sep = ";" if ";" in line else ","
        parts = [x.strip() for x in line.split(sep)]
        if len(parts) < 2:
            continue
        vals = [float(x) for x in parts[:3]]
        while len(vals) < 3:
            vals.append(DEVICE_HEIGHT_M)
        pts.append(vals)

    if not pts:
        raise ValueError(f"No valid waypoints found in {csv_path}")

    waypoints: list[Waypoint] = []
    t_cum = 0.0
    prev = np.array(pts[0][:3])
    for i, pt in enumerate(pts):
        pos = np.array(pt[:3])
        if i > 0:
            t_cum += float(np.linalg.norm(pos - prev)) / cruise_speed_mps
        waypoints.append(Waypoint(t_s=t_cum, x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
        prev = pos
    return waypoints


# ═══════════════════════════════════════════════════════════════════════════
# scene / trajectory presets
# ═══════════════════════════════════════════════════════════════════════════

def _setup_corridor(args) -> tuple[Path, list[Waypoint], tuple[float, float, float], float, list[dict]]:
    """Generate corridor floorplan + straight trajectory. Returns (png_path, waypoints, anchor, ppm, color_mapping)."""
    ppm = 20.0
    anchor = (2.0, CORRIDOR_W_M / 2, DEVICE_HEIGHT_M)
    color_mapping = [
        {"color": [0, 0, 0], "material": "itu_concrete"},
        {"color": [0, 0, 255], "material": "itu_glass"},
    ]

    png_path = _PROJECT_ROOT / "floorplans" / "corridor_straight_25m.png"
    if not png_path.exists():
        _generate_corridor_floorplan(png_path)

    waypoints = generate_trajectory(
        start_xyz=(4.0, CORRIDOR_W_M / 2, DEVICE_HEIGHT_M),
        end_xyz=(23.0, CORRIDOR_W_M / 2, DEVICE_HEIGHT_M),
        cruise_speed_mps=args.speed,
        num_waypoints=args.waypoints,
        corridor_width_m=CORRIDOR_W_M,
        seed=args.seed,
    )
    return png_path, waypoints, anchor, ppm, color_mapping


def _setup_t_junction(args) -> tuple[Path, list[Waypoint], tuple[float, float, float], float, list[dict]]:
    """Generate T-junction floorplan + two-segment trajectory."""
    ppm = 20.0
    anchor = (13.25, 7.0, DEVICE_HEIGHT_M)
    color_mapping = [
        {"color": [200, 130, 60], "material": "itu_tile"},
        {"color": [192, 192, 192], "material": "itu_metal"},
        {"color": [0, 0, 0], "material": "itu_concrete"},
        {"color": [0, 0, 255], "material": "itu_glass"},
    ]

    png_path = _PROJECT_ROOT / "floorplans" / "t_junction.png"
    if not png_path.exists():
        _generate_t_junction_floorplan(png_path)

    seg1_len = 4.75   # down Y corridor
    seg2_len = 12.25  # left along X corridor
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
    for wp in seg2:
        wp.t_s += t_offset
    return png_path, seg1 + seg2, anchor, ppm, color_mapping


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Measure — ranging simulation along a trajectory",
    )
    # ── input source (mutually exclusive) ──
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", choices=["corridor", "t_junction"],
                     help="Built-in scene preset (auto-generates floorplan + trajectory)")
    src.add_argument("--floorplan", metavar="PNG",
                     help="Path to custom floorplan PNG (requires --waypoints-file, --floorplan-width-m, --tx)")

    # ── custom-floorplan options ──
    parser.add_argument("--waypoints-file", metavar="CSV",
                         help="Path to waypoints CSV (N×3: x,y,z) — required with --floorplan")
    parser.add_argument("--floorplan-width-m", type=float,
                         help="Physical width of floorplan in meters — required with --floorplan")
    parser.add_argument("--tx", type=float, nargs=3, metavar=("X", "Y", "Z"),
                         help="Fixed anchor (TX) position in meters — required with --floorplan")

    # ── simulation ──
    parser.add_argument("--speed", type=float, default=DEFAULT_CRUISE_SPEED_MPS,
                        help=f"Walking speed in m/s (default: {DEFAULT_CRUISE_SPEED_MPS})")
    parser.add_argument("--waypoints", type=int, default=40,
                        help="Number of waypoints for built-in scenes (default: 40)")
    parser.add_argument("--impairments", default="none", choices=["none", "full"])
    parser.add_argument("--algorithms", default="max_peak,threshold,leading_edge,search_back,chip_lde",
                        help="Comma-separated LDE algorithm names")
    parser.add_argument("--radios", default="uwb,wifi,fiveg",
                        help="Comma-separated protocols")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of Monte Carlo trials per waypoint (default: 1)")
    parser.add_argument("--no-rt", action="store_true",
                        help="Skip RT, use cached truths only")
    parser.add_argument("--output", help="Output directory (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── Validate custom-floorplan args ──
    if args.floorplan:
        missing = []
        if not args.waypoints_file:
            missing.append("--waypoints-file")
        if not args.floorplan_width_m:
            missing.append("--floorplan-width-m")
        if not args.tx:
            missing.append("--tx")
        if missing:
            parser.error(f"--floorplan requires: {', '.join(missing)}")

    # ── Protocols & algorithms ──
    protocols = [s.strip() for s in args.radios.split(",")]
    algo_names = [s.strip() for s in args.algorithms.split(",")]
    impairments = IMPAIRMENT_PRESETS[args.impairments]

    # ── Setup scene + trajectory ──
    if args.scene:
        if args.scene == "corridor":
            floorplan_path, waypoints, anchor, ppm, color_mapping = _setup_corridor(args)
            scene_desc = f"{CORRIDOR_L_M}m × {CORRIDOR_W_M}m corridor — concrete + glass"
        else:  # t_junction
            floorplan_path, waypoints, anchor, ppm, color_mapping = _setup_t_junction(args)
            scene_desc = "T-junction 14.5m+7.15m — tile walls + metal panel"
        scene_name = args.scene
    else:
        floorplan_path = Path(args.floorplan)
        if not floorplan_path.exists():
            print(f"Floorplan not found: {floorplan_path}")
            sys.exit(1)
        waypoints_path = Path(args.waypoints_file)
        if not waypoints_path.exists():
            print(f"Waypoints file not found: {waypoints_path}")
            sys.exit(1)

        from PIL import Image
        img = Image.open(floorplan_path)
        img_w_px, img_h_px = img.size
        ppm = img_w_px / args.floorplan_width_m
        room_h_m = img_h_px / ppm
        anchor = tuple(args.tx)

        print(f"Floorplan: {floorplan_path}  ({img_w_px}×{img_h_px} px)")
        print(f"  Physical size: {args.floorplan_width_m:.1f} × {room_h_m:.1f} m")
        print(f"  PPM: {ppm:.2f}")

        waypoints = _load_waypoints_csv(waypoints_path, cruise_speed_mps=args.speed)
        scene_name = floorplan_path.stem
        scene_desc = f"custom floorplan ({args.floorplan_width_m:.1f}×{room_h_m:.1f}m)"
        color_mapping = [
            {"color": [0, 0, 0], "material": "itu_concrete"},
            {"color": [0, 0, 255], "material": "itu_glass"},
            {"color": [200, 130, 60], "material": "itu_tile"},
            {"color": [192, 192, 192], "material": "itu_metal"},
        ]

    # ── Output directory ──
    if args.output:
        output_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "outputs" / "measure" / f"{scene_name}_{ts}"
    rt_cache_dir = output_dir / "rt_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"Measure Simulation — {scene_desc}")
    print(f"{'='*60}")
    print(f"  Anchor (TX):  {anchor}")
    print(f"  Waypoints:    {len(waypoints)}  |  Duration: {waypoints[-1].t_s:.1f}s")
    print(f"  Speed:        {args.speed} m/s  |  PPM: {ppm:.2f}")
    print(f"  Protocols:    {protocols}")
    print(f"  Algorithms:   {algo_names}  |  Impairments: {args.impairments}")
    print(f"  Output:       {output_dir}")

    # ── Run ──
    print(f"\n{'─'*60}")
    print("Running simulation...")
    if args.no_rt:
        print("  (RT skipped — loading from cache)")
    print(f"{'─'*60}")

    t0 = time.time()
    result = run_trajectory_simulation(
        waypoints=waypoints,
        anchor_pos=anchor,
        floorplan_path=floorplan_path,
        protocols=protocols,
        algo_names=algo_names,
        impairments=impairments,
        rt_cache_dir=rt_cache_dir,
        skip_rt=args.no_rt,
        seed=args.seed,
        color_mapping=color_mapping,
        ppm=ppm,
    )
    elapsed = time.time() - t0
    print(f"\n  Simulation complete in {elapsed:.1f}s "
          f"({elapsed / len(waypoints):.1f}s per waypoint)")

    # ── Save ──
    print(f"\n{'─'*60}")
    print("Saving results...")
    print(f"{'─'*60}")
    save_results_csv(result, output_dir)
    save_summary(result, output_dir)
    plot_results(result, output_dir)

    output_dir.joinpath("metadata.json").write_text(json.dumps({
        "floorplan": str(floorplan_path),
        "anchor_pos": list(anchor),
        "num_waypoints": len(waypoints),
        "speed_mps": args.speed,
        "duration_s": waypoints[-1].t_s,
        "ppm": ppm,
        "protocols": protocols,
        "algorithms": algo_names,
        "impairments": args.impairments,
        "total_runtime_s": elapsed,
    }, indent=2), encoding="utf-8")

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
