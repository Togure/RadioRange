#!/usr/bin/env python3
"""
Generate synthetic real-world measurements from simulation trajectory results.

Model
-----
  measured_range_m = estimated_range_sim_m + ε_noise + outlier_rare

where:
  - estimated_range_sim_m  is the simulation output (already contains NLOS bias)
  - ε_noise               ~ N(0, σ_protocol²)     per-waypoint
  - outlier_rare           ~ 0-2 events per trajectory, always over-estimate

One measurement per waypoint per protocol×algorithm.  No repetitions.

Usage
-----
  .venv/bin/python3 utils/generate_measurements.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RNG_SEED = 42

MEASUREMENT_PARAMS = {
    "uwb":   {"sigma_m": 0.03, "n_outliers": 1, "outlier_min_m": 1.0, "outlier_max_m": 3.0},
    "wifi":  {"sigma_m": 0.50, "n_outliers": 2, "outlier_min_m": 3.0, "outlier_max_m": 8.0},
    "fiveg": {"sigma_m": 0.35, "n_outliers": 2, "outlier_min_m": 2.0, "outlier_max_m": 6.0},
}

PROTOCOLS = ["uwb", "wifi", "fiveg"]
ALGORITHMS = ["max_peak", "threshold", "leading_edge", "search_back", "chip_lde"]


def process_scene(input_csv: Path, output_dir: Path) -> None:
    """Read trajectory_raw.csv, generate measurements, write new CSV."""
    rows = []
    with open(input_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    n_wp = len(rows)
    print(f"Processing {input_csv.name} — {n_wp} waypoints")

    true_range = np.array([float(r["true_range_m"]) for r in rows], dtype=float)

    # ── pre-read all estimated ranges from raw CSV ─────────────────────
    estimated: dict[str, np.ndarray] = {}   # key → (N_wp,)
    sim_error: dict[str, np.ndarray] = {}
    for proto in PROTOCOLS:
        for algo in ALGORITHMS:
            key = f"{proto}_{algo}"
            err_col = f"{proto}_{algo}_error_m"
            est_col = f"{proto}_{algo}_estimated_range_m"
            err = np.array([float(r.get(err_col, 0.0)) for r in rows], dtype=float)
            if est_col in rows[0]:
                est = np.array([float(r[est_col]) for r in rows], dtype=float)
            else:
                est = true_range + err
            estimated[key] = est
            sim_error[key] = err

    # ── generate measured values ───────────────────────────────────────
    measured: dict[str, np.ndarray] = {}
    is_outlier: dict[str, np.ndarray] = {}

    for proto in PROTOCOLS:
        params = MEASUREMENT_PARAMS[proto]
        base_seed = RNG_SEED + hash(proto) % 1000

        for algo in ALGORITHMS:
            key = f"{proto}_{algo}"
            rng = np.random.default_rng(base_seed + hash(algo) % 100)

            # baseline: sim estimate + thermal noise
            noise = rng.normal(0.0, params["sigma_m"], size=n_wp)
            meas = estimated[key].copy() + noise

            # rare trajectory-level outliers
            n_out = min(params["n_outliers"], n_wp)
            outlier_idx = rng.choice(n_wp, size=n_out, replace=False)
            outlier_amps = rng.uniform(
                params["outlier_min_m"], params["outlier_max_m"], size=n_out,
            )

            out_mask = np.zeros(n_wp, dtype=bool)
            for idx, amp in zip(outlier_idx, outlier_amps):
                meas[idx] += amp
                out_mask[idx] = True

            measured[key] = meas
            is_outlier[key] = out_mask

    # ── build header ──────────────────────────────────────────────────
    base_cols = ["waypoint", "t_s", "x_m", "y_m", "z_m", "true_range_m"]
    header = list(base_cols)
    for proto in PROTOCOLS:
        for algo in ALGORITHMS:
            key = f"{proto}_{algo}"
            header.append(f"{key}_estimated_range_m")
            header.append(f"{key}_error_m")
            header.append(f"{key}_measured_range_m")
            header.append(f"{key}_is_outlier")

    # ── write CSV ─────────────────────────────────────────────────────
    out_path = output_dir / "trajectory_measured.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for wi, r in enumerate(rows):
            row = [
                r["waypoint"], r["t_s"], r["x_m"], r["y_m"], r["z_m"],
                f"{true_range[wi]:.6f}",
            ]
            for proto in PROTOCOLS:
                for algo in ALGORITHMS:
                    key = f"{proto}_{algo}"
                    row.append(f"{estimated[key][wi]:.6f}")
                    row.append(f"{sim_error[key][wi]:.6f}")
                    row.append(f"{measured[key][wi]:.6f}")
                    row.append(str(int(is_outlier[key][wi])))
            w.writerow(row)
    print(f"  → {out_path}")

    # ── summary ────────────────────────────────────────────────────────
    print()
    for proto in PROTOCOLS:
        params = MEASUREMENT_PARAMS[proto]
        total_out = sum(int(is_outlier[f"{proto}_{algo}"].sum()) for algo in ALGORITHMS)
        # measure noise on clean waypoints
        diffs = []
        for algo in ALGORITHMS:
            key = f"{proto}_{algo}"
            clean = ~is_outlier[key]
            diffs.append(measured[key][clean] - estimated[key][clean])
        diffs = np.concatenate(diffs)
        print(
            f"  {proto:6s}  σ_noise={params['sigma_m']:.2f}m  "
            f"|meas−est|_mean={np.mean(np.abs(diffs)):.3f}m  "
            f"max={np.max(np.abs(diffs)):.3f}m  "
            f"outliers_total={total_out}"
        )
    print()


def main() -> None:
    scenes = [
        ("corridor_straight_v2", "outputs/trajectory/corridor_straight_v2/trajectory_raw.csv"),
        ("t_junction_v1", "outputs/trajectory/t_junction_v1/trajectory_raw.csv"),
    ]

    for scene_name, csv_rel in scenes:
        input_csv = _PROJECT_ROOT / csv_rel
        if not input_csv.exists():
            print(f"SKIP: {input_csv} not found")
            continue
        output_dir = _PROJECT_ROOT / "outputs" / "trajectory" / scene_name
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")
        process_scene(input_csv, output_dir)


if __name__ == "__main__":
    main()
