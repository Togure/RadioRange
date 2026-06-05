#!/usr/bin/env python3
"""
Fingerprint Radio Map Simulation — WiFi RSSI + Ranging on a Floorplan Grid
===========================================================================

Given a floorplan PNG and multiple WiFi AP positions, generates a fingerprint
radio map: for each grid point × AP pair, computes RSSI (from Sionna RT path
gains) and range estimate (via LDE algorithm).

Usage:
  python3 scripts/run_fingerprint.py \\
      --floorplan floorplans/complex_building.png \\
      --floorplan-width-m 42 \\
      --aps ap_positions.csv \\
      --grid-spacing 2.0 \\
      --algo leading_edge --tx-power 20

Outputs (saved to outputs/fingerprint/<name>_<timestamp>/):
  - fingerprint_db.csv               per grid-point × AP results
  - rssi_heatmap_<ap_id>.png          RSSI heatmap per AP
  - range_error_heatmap_<ap_id>.png   ranging error heatmap per AP
  - metadata.json
  - rt_cache/                         per-measurement RT caches (reusable)
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

from core.models import ChannelTruth
from environments.registry import generate_truths
from hardware.impairments import apply_timing_impairments
from algorithms import (
    ChipLeadingEdgeLde,
    LeadingEdgeLde,
    MaxPeakLde,
    SearchBackLde,
    ThresholdLde,
)
from utils.radio_factory import DEFAULT_CARRIER_FREQ_HZ, RADIO_DEFAULTS, create_radio
from utils.runner import _rng, _PROTO_SEEDS
from utils.fingerprint import generate_grid_points, compute_rssi, build_radio_map
from utils.scene_presets import IMPAIRMENT_PRESETS

# ═══════════════════════════════════════════════════════════════════════════
# algorithm factory
# ═══════════════════════════════════════════════════════════════════════════

ALGORITHMS: dict[str, Any] = {
    "max_peak": lambda: MaxPeakLde(),
    "threshold": lambda: ThresholdLde(peak_ratio=0.18),
    "leading_edge": lambda: LeadingEdgeLde(n_sigma=4.0),
    "search_back": lambda: SearchBackLde(peak_ratio=0.18),
    "chip_lde": lambda: ChipLeadingEdgeLde(),
}


# ═══════════════════════════════════════════════════════════════════════════
# config builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_config(
    floorplan_path: Path,
    tx_pos: tuple[float, float, float],
    rx_pos: tuple[float, float, float],
    ppm: float,
    impairments: dict,
    color_mapping: list[dict] | None = None,
) -> dict:
    if color_mapping is None:
        color_mapping = [
            {"color": [0, 0, 0],       "material": "itu_concrete"},
            {"color": [128, 64, 0],    "material": "itu_brick"},
            {"color": [0, 0, 255],     "material": "itu_glass"},
            {"color": [139, 90, 43],   "material": "itu_wood"},
            {"color": [192, 192, 192], "material": "itu_metal"},
            {"color": [255, 200, 100], "material": "itu_plasterboard"},
        ]
    return {
        "seed": 42,
        "timing": {},
        "impairments": impairments,
        "environment": {
            "type": "floorplan",
            "tx_position_m": list(tx_pos),
            "rx_position_m": list(rx_pos),
            "max_reflections": 4,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "scattering_coefficient": 0.3,
            "num_samples": 100_000,
            "max_paths_per_type": 10,
            "num_trials": 1,
            "floorplan": {
                "image_path": str(floorplan_path.resolve()),
                "pixels_per_meter": ppm,
                "wall_height_m": 3.0,
                "default_tolerance": 40,
                "color_mapping": color_mapping,
                "background_color": [255, 255, 255],
                "floor_material": "itu_concrete",
                "ceiling_material": "itu_ceiling_board",
                "generate_floor_ceiling": True,
            },
        },
        "radios": {
            "wifi": {
                "enabled": True,
                "carrier_frequency_hz": DEFAULT_CARRIER_FREQ_HZ["wifi"],
                **RADIO_DEFAULTS["wifi"],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# single measurement runner
# ═══════════════════════════════════════════════════════════════════════════

def _make_measure_runner(
    floorplan_path: Path,
    ppm: float,
    impairments: dict,
    algo_name: str,
    tx_power_dbm: float,
    seed: int,
    rt_cache_dir: Path | None,
    skip_rt: bool,
    color_mapping: list[dict],
) -> Any:
    """Return a callable ``run(ap_pos, rx_pos) → dict``."""
    algo = ALGORITHMS[algo_name]()
    radio = create_radio("wifi", impairments=impairments)
    proto = "wifi"
    call_idx = [0]  # mutable counter

    def _run_one(ap_pos: tuple[float, float, float], rx_pos: tuple[float, float, float]) -> dict[str, Any]:
        idx = call_idx[0]
        call_idx[0] += 1

        true_range_m = float(np.linalg.norm(np.array(rx_pos) - np.array(ap_pos)))

        # ── RT ──
        paths = []
        try:
            rt_subdir = rt_cache_dir / f"meas_{idx:06d}" if rt_cache_dir else None
            if skip_rt and rt_subdir and rt_subdir.exists():
                from environments.persistence import load_truths
                truths = load_truths(rt_subdir)
                truth = truths[0]
            else:
                cfg = _build_config(floorplan_path, ap_pos, rx_pos, ppm, impairments, color_mapping)
                freq = DEFAULT_CARRIER_FREQ_HZ[proto]
                rng_rt = _rng(seed, idx * 100 + _PROTO_SEEDS[proto])
                truths = generate_truths(cfg, rng_rt, carrier_frequency_hz=freq)
                truth = truths[0]
                if rt_subdir:
                    from environments.persistence import save_truths
                    rt_subdir.mkdir(parents=True, exist_ok=True)
                    save_truths([truth], rt_subdir)
            paths_raw = truth.paths if hasattr(truth, "paths") else []
            path_gains = list(truth.a_paths) if hasattr(truth, "a_paths") and len(truth.a_paths) > 0 else []
            n_paths = len(path_gains)
        except Exception:
            return {
                "rssi_dbm": float("nan"),
                "range_m": float("nan"),
                "true_range_m": true_range_m,
                "range_error_m": float("nan"),
                "n_paths": 0,
            }

        # ── RSSI from RT paths ──
        rssi_dbm = compute_rssi(path_gains if path_gains else list(paths_raw) if paths_raw else [], tx_power_dbm=tx_power_dbm)

        # ── Impairments + observation + LDE ──
        impair_rng = _rng(seed, idx * 1000 + _PROTO_SEEDS[proto] + 1)
        radio_cfg = {"sfo_ppm": 10.0, "cfo_hz": 300.0}
        impaired = apply_timing_impairments(truth, {"impairments": impairments}, impair_rng, radio_cfg=radio_cfg)

        obs_rng = _rng(seed, idx * 1000 + _PROTO_SEEDS[proto] + 2)
        observation = radio.observe(impaired, obs_rng)

        estimate = algo.estimate(observation)
        range_m = estimate.estimated_range_m
        range_error_m = range_m - true_range_m
        if not np.isfinite(range_error_m):
            range_error_m = float("nan")

        return {
            "rssi_dbm": rssi_dbm,
            "range_m": float(range_m),
            "true_range_m": true_range_m,
            "range_error_m": float(range_error_m),
            "n_paths": n_paths,
        }

    return _run_one


# ═══════════════════════════════════════════════════════════════════════════
# visualization
# ═══════════════════════════════════════════════════════════════════════════

def _plot_radio_maps(
    records: list[dict],
    ap_ids: list[str],
    ap_positions: list[tuple[str, float, float, float]],
    grid_points: list[tuple[float, float, float]],
    floorplan_path: Path,
    ppm: float,
    output_dir: Path,
    grid_spacing_m: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # Build AP lookup
    ap_lookup = {ap_id: (x, y, z) for ap_id, x, y, z in ap_positions}

    # Load floorplan image for background
    img = Image.open(floorplan_path).convert("RGB")
    img_w_px, img_h_px = img.size
    room_w_m = img_w_px / ppm
    room_h_m = img_h_px / ppm
    bg = np.asarray(img, dtype=np.uint8)

    xs = np.array([p[0] for p in grid_points])
    ys = np.array([p[1] for p in grid_points])

    for ap_id in ap_ids:
        ap_x, ap_y, _ = ap_lookup[ap_id]
        ap_records = [r for r in records if r["ap_id"] == ap_id]
        n = len(ap_records)
        rssi = np.array([r["rssi_dbm"] for r in ap_records])
        range_m = np.array([r["range_m"] for r in ap_records])
        range_err = np.abs(np.array([r["range_error_m"] for r in ap_records]))

        valid = np.isfinite(rssi) & np.isfinite(range_m) & np.isfinite(range_err)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))

        for ax, data, title, cmap, label in [
            (ax1, rssi, f"RSSI — {ap_id}", "RdYlBu_r", "RSSI (dBm)"),
            (ax2, range_m, f"Estimated Range — {ap_id}", "plasma", "Range (m)"),
            (ax3, range_err, f"Range Error — {ap_id}", "RdYlGn_r", "|Error| (m)"),
        ]:
            ax.imshow(bg, extent=[0, room_w_m, room_h_m, 0], aspect="equal", alpha=0.25)
            sc = ax.scatter(xs[valid], ys[valid], c=data[valid], cmap=cmap,
                           s=55, edgecolors="none", zorder=4)
            cb = plt.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
            cb.set_label(label, fontsize=7)
            cb.ax.tick_params(labelsize=6)
            # Mark AP position
            ax.scatter(ap_x, ap_y, marker="*", s=250, c="#DC2626", edgecolors="#7F1D1D",
                      linewidth=1.0, zorder=6)
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlim(0, room_w_m)
            ax.set_ylim(room_h_m, 0)
            ax.grid(True, alpha=0.10)

        fig.suptitle(f"Fingerprint Radio Map — {ap_id}  "
                     f"(grid {grid_spacing_m:.1f}m, {n} points, ★ = AP)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"radiomap_{ap_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: radiomap_{ap_id}.png")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fingerprint — WiFi RSSI + ranging radio map on a floorplan grid",
    )
    parser.add_argument("--floorplan", required=True, help="Path to floorplan PNG")
    parser.add_argument("--floorplan-width-m", type=float, required=True,
                        help="Physical width of the floorplan in meters")
    parser.add_argument("--aps", default=None, help="Path to AP positions CSV (default: auto-generate 2 APs)")
    parser.add_argument("--grid-spacing", type=float, default=2.0,
                        help="Grid spacing in meters (default: 2.0)")
    parser.add_argument("--algo", default="leading_edge",
                        choices=["max_peak", "threshold", "leading_edge", "search_back", "chip_lde"],
                        help="LDE algorithm (default: leading_edge)")
    parser.add_argument("--tx-power", type=float, default=20.0,
                        help="WiFi transmit power in dBm (default: 20)")
    parser.add_argument("--impairments", default="none", choices=["none", "full"])
    parser.add_argument("--trials", type=int, default=1,
                        help="Monte Carlo trials per grid point (default: 1)")
    parser.add_argument("--no-rt", action="store_true", help="Skip RT, use cache")
    parser.add_argument("--output", help="Output directory (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── Resolve paths ──
    floorplan_path = Path(args.floorplan)
    if not floorplan_path.exists():
        print(f"Floorplan not found: {floorplan_path}")
        sys.exit(1)
    # ── Compute PPM ──
    from PIL import Image
    img = Image.open(floorplan_path)
    ppm = img.size[0] / args.floorplan_width_m
    room_h_m = img.size[1] / ppm
    print(f"Floorplan: {floorplan_path}  ({img.size[0]}×{img.size[1]} px)")
    print(f"  Physical size: {args.floorplan_width_m:.1f} × {room_h_m:.1f} m  |  PPM: {ppm:.2f}")

    # ── Load or auto-generate APs ──
    ap_positions: list[tuple[str, float, float, float]] = []
    if args.aps:
        aps_path = Path(args.aps)
        if not aps_path.exists():
            print(f"AP file not found: {aps_path}")
            sys.exit(1)
        ap_lines = [l.strip() for l in aps_path.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.strip().startswith("#")]
        start = 1 if ap_lines[0].lower().startswith("ap_id") else 0
        for line in ap_lines[start:]:
            sep = ";" if ";" in line else ","
            parts = [x.strip() for x in line.split(sep)]
            if len(parts) >= 4:
                ap_positions.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    else:
        # Auto-generate 2 APs at 1/3 and 2/3 of room dimensions
        ap_h = 2.5
        ap_positions = [
            ("ap1", args.floorplan_width_m / 3, room_h_m / 2, ap_h),
            ("ap2", args.floorplan_width_m * 2 / 3, room_h_m / 2, ap_h),
        ]
    print(f"APs: {len(ap_positions)}")
    for ap in ap_positions:
        print(f"  {ap[0]}: ({ap[1]:.1f}, {ap[2]:.1f}, {ap[3]:.1f})")

    # ── Generate grid ──
    grid_points, _, _ = generate_grid_points(
        floorplan_path, ppm, grid_spacing_m=args.grid_spacing,
        device_height_m=1.5, margin_m=0.5,
    )
    print(f"Grid: {len(grid_points)} points at {args.grid_spacing}m spacing")

    # ── Output dir ──
    if args.output:
        output_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "outputs" / "fingerprint" / f"{floorplan_path.stem}_{ts}"
    rt_cache_dir = output_dir / "rt_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    impairments = IMPAIRMENT_PRESETS[args.impairments]

    print(f"\n{'='*60}")
    print(f"Fingerprint Radio Map")
    print(f"{'='*60}")
    print(f"  Floorplan:     {floorplan_path}")
    print(f"  APs:           {len(ap_positions)}")
    print(f"  Grid points:   {len(grid_points)} × {len(ap_positions)} APs"
          f" = {len(grid_points) * len(ap_positions)} measurements")
    print(f"  Algorithm:     {args.algo}  |  TX power: {args.tx_power} dBm")
    print(f"  Output:        {output_dir}")

    # ── Build radio map ──
    color_mapping = None  # use built-in default (same as main.py)

    runner = _make_measure_runner(
        floorplan_path, ppm, impairments, args.algo, args.tx_power,
        args.seed, rt_cache_dir, args.no_rt, color_mapping,
    )

    print(f"\n{'─'*60}")
    print("Building radio map...")
    print(f"{'─'*60}")

    t0 = time.time()
    radio_map = build_radio_map(ap_positions, grid_points, runner)
    elapsed = time.time() - t0
    print(f"\n  Complete in {elapsed:.1f}s ({elapsed / len(radio_map['records']):.2f}s per measurement)")

    # ── Save CSV ──
    records = radio_map["records"]
    csv_path = output_dir / "fingerprint_db.csv"
    cols = ["ap_id", "ap_x", "ap_y", "ap_z", "rx_x", "rx_y", "rx_z",
            "rssi_dbm", "range_m", "true_range_m", "range_error_m", "n_paths"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"  Saved: fingerprint_db.csv")

    # ── Visualize ──
    print(f"\n{'─'*60}")
    print("Generating heatmaps...")
    print(f"{'─'*60}")
    _plot_radio_maps(records, radio_map["ap_ids"], ap_positions, grid_points,
                     floorplan_path, ppm, output_dir, args.grid_spacing)

    # ── Metadata ──
    output_dir.joinpath("metadata.json").write_text(json.dumps({
        "floorplan": str(floorplan_path),
        "floorplan_width_m": args.floorplan_width_m,
        "ppm": ppm,
        "aps": [{"id": a[0], "x": a[1], "y": a[2], "z": a[3]} for a in ap_positions],
        "n_aps": len(ap_positions),
        "n_grid_points": len(grid_points),
        "grid_spacing_m": args.grid_spacing,
        "algo": args.algo,
        "tx_power_dbm": args.tx_power,
        "impairments": args.impairments,
        "total_runtime_s": elapsed,
    }, indent=2), encoding="utf-8")

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
