"""Interactive 3D ray-tracing visualization + CIR ranging simulation.

============================================================================
  Usage (cache path is the ONLY required input)
============================================================================

  Step 1 — Generate RT cache (run once per scene/geometry):
    python3 main.py --scene box_knife_concrete --radios uwb \\
      --tx -2.0 0.0 0.2 --rx 2.0 0.0 0.2 --trials 300 \\
      --dump-truths cache/rt/my_run

  Step 2 — Visualize:
    python3 scripts/rt_cache_interactive.py cache/rt/my_run

  Step 2b — With CLI overrides (all optional):
    python3 scripts/rt_cache_interactive.py cache/rt/my_run \\
      --trials 300 --impairments full

  SNR is per-radio default: UWB=30dB, WiFi=30dB, 5G=25dB

============================================================================
  Where to change simulation parameters
============================================================================

  Scene geometry / RT settings → main.py  _SCENE_PRESETS["box_knife_concrete"]
    Key parameters:
      tx_position_m / rx_position_m   — TX/RX 3D coordinates
      max_reflections                  — max bounce depth (default 3)
      scattering_coefficient           — 0=pure specular, 1=pure diffuse (default 0.3)
      num_samples                      — rays shot from TX (default 10000)
      max_paths_per_type              — top-N strongest per path type (default 10)

  Radio / algorithm settings → this script's CLI (--trials, --impairments)
  SNR uses each sensor's own default — no global override needed

============================================================================
  Output
============================================================================

  outputs/interactive/<scene>_chip_sim.html  — interactive 3D + CIR + CDF
  outputs/interactive/<scene>_results.csv     — per-trial error data

.venv/bin/python3 scripts/rt_cache_interactive.py cache/rt/box_knife_concrete
.venv/bin/python3 scripts/rt_cache_interactive.py cache/rt/simple_room

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.runner import _rng

_OUTPUT_DIR = Path("outputs/interactive")
_CACHE_ROOT = Path("cache/rt")
_LIGHT_SPEED = 299_792_458.0


# ═══════════════════════════════════════════════════════════════════════════════
# data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_cache_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    alt = _CACHE_ROOT / raw
    if alt.exists():
        return alt
    print(f"Cache not found: {raw}  (also tried {alt})")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# algorithm factories
# ═══════════════════════════════════════════════════════════════════════════════

def _build_algorithms() -> dict[str, Any]:
    from algorithms import (
        ChipLeadingEdgeLde,
        LeadingEdgeLde,
        MaxPeakLde,
        SearchBackLde,
        ThresholdLde,
    )
    return {
        "MaxPeak": MaxPeakLde(),
        "Threshold(0.18)": ThresholdLde(peak_ratio=0.18),
        "LeadingEdge(4σ)": LeadingEdgeLde(n_sigma=4.0, tail_frac=0.25, min_run=3),
        "SearchBack(0.18)": SearchBackLde(peak_ratio=0.18),
        "ChipLDE(10dB)": ChipLeadingEdgeLde(threshold_db=10.0, tail_frac=0.25, min_run=3),
    }


def _build_multipath_algorithms() -> dict[str, Any]:
    from algorithms.multipath import CFARDetector, CLEANDetector, PeakFinder
    return {
        "PeakFinder(-20dB)": PeakFinder(threshold_db=20.0, min_peak_spacing_ns=2.0),
        "CFAR(1e-2)": CFARDetector(guard_cells=3, reference_cells=10, pf=0.01, min_peak_spacing_ns=2.0),
        "CLEAN(15dB)": CLEANDetector(max_iterations=20, residual_threshold_db=15.0, min_peak_spacing_ns=2.0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# matching helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_evaluable_path(path_type: str, path_order: int) -> bool:
    if path_order == 0:
        return True
    if path_order == 1:
        p = str(path_type).lower()
        return p in ("specular", "reflection")
    return False


def _get_evaluable_mask(gt_types, gt_orders, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        t = str(gt_types[i]) if gt_types is not None and i < len(gt_types) else "reflection"
        o = int(gt_orders[i]) if gt_orders is not None and i < len(gt_orders) else 0
        if _is_evaluable_path(t, o):
            mask[i] = True
    return mask


def _match_detected_to_gt(
    detected: list, gt_tau_s: np.ndarray, gt_gains: np.ndarray, bandwidth_hz: float,
    gt_types=None, gt_orders=None,
) -> dict:
    resolution_s = 1.0 / bandwidth_hz if bandwidth_hz > 0 else 2e-9
    active = np.abs(gt_gains) > 1e-12
    gt_indices_all = np.flatnonzero(active)
    gt_tau_active = gt_tau_s[active]
    gt_gains_active = gt_gains[active]

    n_active = len(gt_tau_active)
    if gt_types is not None and gt_orders is not None:
        if hasattr(gt_types, "tolist"):
            gt_types = gt_types.tolist()
        if hasattr(gt_orders, "tolist"):
            gt_orders = gt_orders.tolist()
        gt_types_active = []
        gt_orders_active = []
        for idx in gt_indices_all:
            gt_types_active.append(str(gt_types[idx]) if idx < len(gt_types) else "reflection")
            gt_orders_active.append(int(gt_orders[idx]) if idx < len(gt_orders) else 0)
        evaluable = _get_evaluable_mask(gt_types_active, gt_orders_active, n_active)
    else:
        evaluable = np.ones(n_active, dtype=bool)

    evaluable_active_indices = gt_indices_all[np.flatnonzero(evaluable)]

    detected_tau = np.array([d.estimated_tof_s for d in detected])

    gt_matched = [False] * n_active
    det_matched = [False] * len(detected_tau)
    hits = []

    gt_order = np.argsort(np.abs(gt_gains_active))[::-1]
    for gi in gt_order:
        if not evaluable[gi]:
            continue
        gt_t = gt_tau_active[gi]
        best_j, best_err = -1, float("inf")
        for j, dt in enumerate(detected_tau):
            if det_matched[j]:
                continue
            err = abs(dt - gt_t)
            if err < resolution_s and err < best_err:
                best_err = err
                best_j = j
        if best_j >= 0:
            hits.append((int(gt_indices_all[gi]), best_j, float(best_err)))
            gt_matched[gi] = True
            det_matched[best_j] = True

    misses = [int(gt_indices_all[gi]) for gi, m in enumerate(gt_matched) if not m and evaluable[gi]]
    false_alarms = [j for j, m in enumerate(det_matched) if not m]

    hit_dist_errors_m = [float(err_s) * _LIGHT_SPEED for _, _, err_s in hits]

    fa_nearest_dist_m: list[float] = []
    if false_alarms and len(gt_tau_active) > 0:
        for j in false_alarms:
            dtau = abs(detected_tau[j] - gt_tau_active)
            fa_nearest_dist_m.append(float(np.min(dtau)) * _LIGHT_SPEED)

    hit_rmse_m = float(np.sqrt(np.mean(np.array(hit_dist_errors_m) ** 2))) if hit_dist_errors_m else 0.0

    return {
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "n_gt": int(np.sum(evaluable)), "n_detected": len(detected),
        "n_evaluable": int(np.sum(evaluable)),
        "resolution_s": resolution_s,
        "hit_dist_errors_m": hit_dist_errors_m,
        "hit_rmse_m": hit_rmse_m,
        "fa_nearest_dist_m": fa_nearest_dist_m,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# single-trial runner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_chip_simulation(
    truth, radio, algo, impairments_cfg: dict, rng: np.random.Generator,
) -> float:
    from hardware.impairments import apply_timing_impairments
    observed = apply_timing_impairments(truth, impairments_cfg, rng)
    obs_rng = np.random.default_rng(rng.integers(2 ** 31))
    observation = radio.observe(observed, obs_rng)
    estimate = algo.estimate(observation)
    return (estimate.estimated_tof_s - observed.true_first_tau_s) * _LIGHT_SPEED


# ═══════════════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════════════

def _export_csv(cache_path: Path, errors: dict[str, dict[str, list[float]]],
                config_info: dict) -> Path:
    protocols = sorted(errors.keys())
    algos = sorted(errors[protocols[0]].keys()) if protocols else []
    out_path = _OUTPUT_DIR / f"{cache_path.name}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene", "protocol", "algorithm", "trial", "error_m"])
        for proto in protocols:
            for algo in algos:
                for ti, err in enumerate(errors[proto].get(algo, [])):
                    writer.writerow([
                        config_info.get("scene", ""), proto, algo, ti, f"{err:.6f}",
                    ])
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RT Cache → Chip Simulation + 3D Scene Visualization",
    )
    parser.add_argument("cache", help="Path to cache directory or name under cache/rt/")
    parser.add_argument("--trials", type=int, default=None, metavar="N",
                        help="Number of trials to run (default: all available)")
    parser.add_argument("--impairments", default="none", choices=["none", "full"])
    args = parser.parse_args()

    cache_path = _resolve_cache_path(args.cache)
    print(f"Loading cached truths from {cache_path}")

    # ── run_info ───────────────────────────────────────────────────────────
    run_info_path = cache_path / "run_info.json"
    config_info = json.loads(run_info_path.read_text(encoding="utf-8")) if run_info_path.exists() else {}
    tx_pos = np.asarray(config_info.get("tx_position_m", [0, 0, 0]), dtype=float)
    rx_pos = np.asarray(config_info.get("rx_position_m", [0, 0, 0]), dtype=float)
    scene_name = config_info.get("scene_name", "") or config_info.get("scene", "")

    # ── load truths ────────────────────────────────────────────────────────
    from environments.persistence import load_truths

    all_truths: dict[str, list] = {}
    for subdir in sorted(cache_path.iterdir()):
        if subdir.is_dir() and (subdir / "truths.npz").exists():
            all_truths[subdir.name] = load_truths(subdir)
            print(f"  {subdir.name}: {len(all_truths[subdir.name])} trials")
    if not all_truths:
        print("No cached truths found. Run 'main.py --dump-truths' first.")
        sys.exit(1)

    protocols = sorted(all_truths.keys())
    max_trials = min(len(v) for v in all_truths.values())
    num_trials = min(args.trials or max_trials, max_trials)
    print(f"Running {num_trials} trials per protocol")

    # ── carrier freqs ──────────────────────────────────────────────────────
    carrier_freq_hz: dict[str, float] = {}
    for proto in protocols:
        t0 = all_truths[proto][0]
        carrier_freq_hz[proto] = t0.carrier_frequency_hz or t0.metadata.get("carrier_frequency_hz") or 5e9

    # ── impairments ────────────────────────────────────────────────────────
    if args.impairments == "full":
        impairments_cfg = {
            "impairments": {
                "enable_antenna_offset": True, "enable_sfo": True,
                "enable_cfo": True, "enable_adc_phase_offset": True,
                "enable_agc": True, "enable_iq_imbalance": True,
            },
            "timing": {},
        }
    else:
        impairments_cfg = {"impairments": {}, "timing": {}}

    # ── radios (using shared factory) ──────────────────────────────────────
    from utils.radio_factory import RADIO_DEFAULTS, create_radio

    radios: dict[str, Any] = {}
    for proto in protocols:
        radios[proto] = create_radio(
            proto,
            carrier_frequency_hz=carrier_freq_hz[proto],
            impairments=impairments_cfg.get("impairments", {}),
        )

    # ── chip simulation ────────────────────────────────────────────────────
    from hardware.impairments import apply_timing_impairments

    algos = _build_algorithms()
    mp_algos = _build_multipath_algorithms()
    seed = config_info.get("seed", 42)
    errors: dict[str, dict[str, list[float]]] = {p: {a: [] for a in algos} for p in protocols}
    mp_errors: dict[str, dict[str, list[float]]] = {p: {a: [] for a in mp_algos} for p in protocols}

    for trial_idx in range(num_trials):
        for proto in protocols:
            truth = all_truths[proto][trial_idx]
            for algo_name, algo in algos.items():
                rng = _rng(seed, trial_idx * 1000 + hash(algo_name))
                errors[proto][algo_name].append(
                    _run_chip_simulation(truth, radios[proto], algo, impairments_cfg, rng)
                )
            for mp_name, mp_algo in mp_algos.items():
                rng = _rng(seed, trial_idx * 1000 + hash(mp_name))
                mp_truth = apply_timing_impairments(truth, impairments_cfg, rng)
                mp_obs_rng = np.random.default_rng(rng.integers(2 ** 31))
                mp_obs = radios[proto].observe(mp_truth, mp_obs_rng)
                mp_result = mp_algo.detect(mp_obs)
                first_path_range_err = (
                    mp_result.paths[0].estimated_range_m - truth.true_range_m
                ) if mp_result.paths else float("nan")
                mp_errors[proto][mp_name].append(first_path_range_err)

    # ── CIR data + first-trial multipath results for visualization ─────────
    cir_data: dict[str, dict] = {}
    multipath_viz: dict[str, dict] = {}
    for proto in protocols:
        truth = all_truths[proto][0]
        rng = _rng(seed, 9999)
        observed = apply_timing_impairments(truth, impairments_cfg, rng)
        obs_rng = _rng(seed, 9999 + hash(proto))
        obs = radios[proto].observe(observed, obs_rng)
        cir_data[proto] = {
            "t_discrete_s": obs.t_discrete_s, "t_cont_s": obs.t_cont_s,
            "cir_observed_discrete": obs.cir_observed_discrete,
            "cir_observed_cont": obs.cir_observed_cont,
            "cir_clean_discrete": obs.cir_clean_discrete,
            "cir_clean_cont": obs.cir_clean_cont,
            "true_first_tau_s": truth.true_first_tau_s,
        }

        proto_bw_hz = float(RADIO_DEFAULTS.get(proto, {}).get("bandwidth_hz",
                           RADIO_DEFAULTS.get(proto, {}).get("subcarrier_spacing_hz", 312_500.0)
                           * RADIO_DEFAULTS.get(proto, {}).get("fft_size", 512)))

        multipath_viz[proto] = {}
        for mp_name, mp_algo in mp_algos.items():
            rng_viz = _rng(seed, hash(mp_name))
            mp_result = mp_algo.detect(obs)
            match = _match_detected_to_gt(
                mp_result.paths, truth.tau_paths_s, np.abs(truth.a_paths), proto_bw_hz,
                gt_types=truth.path_type, gt_orders=truth.path_order,
            )
            gt_types_list: list = []
            if truth.path_type is not None:
                gt_types_list = truth.path_type.tolist() if hasattr(truth.path_type, "tolist") else list(truth.path_type)
            gt_orders_list: list = []
            if truth.path_order is not None:
                gt_orders_list = truth.path_order.tolist() if hasattr(truth.path_order, "tolist") else list(truth.path_order)

            multipath_viz[proto][mp_name] = {
                "result": mp_result,
                "match": match,
                "detected_paths": [
                    {"estimated_tof_s": p.estimated_tof_s, "amplitude": p.amplitude,
                     "confidence": p.confidence, "is_first_path": p.is_first_path}
                    for p in mp_result.paths
                ],
                "gt_paths": {
                    "tau_s": truth.tau_paths_s.tolist() if hasattr(truth.tau_paths_s, "tolist") else list(truth.tau_paths_s),
                    "gains": np.abs(truth.a_paths).tolist() if hasattr(np.abs(truth.a_paths), "tolist") else list(np.abs(truth.a_paths)),
                    "types": gt_types_list,
                    "orders": gt_orders_list,
                },
            }

    # ── first-path detection on first trial for visualization markers ───────
    first_path_viz: dict[str, dict[str, float]] = {}
    for proto in protocols:
        first_path_viz[proto] = {}
        obs = radios[proto].observe(
            apply_timing_impairments(
                all_truths[proto][0], impairments_cfg,
                _rng(seed, 9999),
            ),
            _rng(seed, 9999 + hash(proto)),
        )
        for algo_name, algo in algos.items():
            estimate = algo.estimate(obs)
            first_path_viz[proto][algo_name] = float(estimate.estimated_tof_s)

    # ── output — delegate to html_report module ────────────────────────────
    from utils.html_report import build_html_report

    html_path = build_html_report(
        cache_path, cir_data, errors, config_info,
        scene_name, tx_pos, rx_pos, all_truths[protocols[0]][0],
        multipath_results=multipath_viz,
        multipath_errors=mp_errors,
        first_path_viz=first_path_viz,
    )
    print(f"HTML: {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")
    csv_path = _export_csv(cache_path, errors, config_info)
    print(f"CSV:  {csv_path}")

    print("\nProtocol     Algorithm          RMSE(m)   P90|err|(m)")
    print("-" * 55)
    for proto in protocols:
        for algo in sorted(algos.keys()):
            errs = errors[proto].get(algo, [])
            if not errs:
                continue
            arr = np.array(errs)
            print(f"{proto:<12} {algo:<20} {np.sqrt(np.mean(arr**2)):8.4f}  "
                  f"{np.percentile(np.abs(arr), 90):10.4f}")
        for mp_name in sorted(mp_algos.keys()):
            errs = mp_errors[proto].get(mp_name, [])
            if not errs:
                continue
            arr = np.array([e for e in errs if not np.isnan(e)])
            if len(arr) == 0:
                continue
            print(f"{proto:<12} {mp_name:<20} {np.sqrt(np.mean(arr**2)):8.4f}  "
                  f"{np.percentile(np.abs(arr), 90):10.4f}")


if __name__ == "__main__":
    main()
