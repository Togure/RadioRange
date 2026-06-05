from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from algorithms import ChipLeadingEdgeLde, LeadingEdgeLde, MaxPeakLde, SearchBackLde, ThresholdLde
from environments import generate_truths
from environments.persistence import load_truths, save_truths
from hardware.impairments import apply_timing_impairments
from utils.config import load_config
from utils.evaluator import empty_error_store, summarize_errors
from utils.radio_factory import DEFAULT_CARRIER_FREQ_HZ, RADIO_DEFAULTS, build_radios_from_config, create_radio
from utils.runner import _PROTO_SEEDS, _rng, run_single_trial
from utils.scene_presets import IMPAIRMENT_PRESETS, SCENE_PRESETS
from utils.visualizer import plot_cir_comparison, plot_error_comparison

# ═══════════════════════════════════════════════════════════════════════════════
# builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_radios(config: dict) -> list:
    """Build radio instances from config via the shared factory."""
    return build_radios_from_config(config)


def build_algorithm(config: dict):
    algo_cfg = config.get("algorithms", {})
    primary = algo_cfg.get("primary", "threshold")
    if primary == "max_peak":
        return MaxPeakLde()
    if primary == "threshold":
        return ThresholdLde(peak_ratio=float(algo_cfg.get("threshold_peak_ratio", 0.18)))
    if primary == "leading_edge":
        return LeadingEdgeLde(
            n_sigma=float(algo_cfg.get("leading_edge_n_sigma", 4.0)),
            tail_frac=float(algo_cfg.get("leading_edge_tail_frac", 0.25)),
            min_run=int(algo_cfg.get("leading_edge_min_run", 3)),
        )
    if primary == "search_back":
        return SearchBackLde(
            peak_ratio=float(algo_cfg.get("search_back_peak_ratio", 0.18)),
        )
    if primary == "chip_lde":
        return ChipLeadingEdgeLde(
            threshold_db=float(algo_cfg.get("chip_lde_threshold_db", 10.0)),
            tail_frac=float(algo_cfg.get("chip_lde_tail_frac", 0.25)),
            min_run=int(algo_cfg.get("chip_lde_min_run", 3)),
        )
    raise ValueError(f"Unknown LDE algorithm: {primary}")


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_sionna_rt(config: dict) -> bool:
    env_type = str(config.get("environment", {}).get("type", ""))
    if env_type == "simple_room":
        engine = str(config.get("environment", {}).get("engine", ""))
        return engine in {"sionna_box", "sionna_rect_room"}
    return env_type in {"floorplan", "sionna_builtin_scene", "custom_scene"}


# ═══════════════════════════════════════════════════════════════════════════════
# multi-algorithm builder (for --mode compare-algos / compare-materials)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_all_algorithms() -> dict[str, Any]:
    """Return all 5 LDE algorithms for comparison modes."""
    return {
        "MaxPeak": MaxPeakLde(),
        "Threshold(0.18)": ThresholdLde(peak_ratio=0.18),
        "LeadingEdge(4σ)": LeadingEdgeLde(n_sigma=4.0, tail_frac=0.25, min_run=3),
        "SearchBack(0.18)": SearchBackLde(peak_ratio=0.18),
        "ChipLDE(10dB)": ChipLeadingEdgeLde(threshold_db=10.0, tail_frac=0.25, min_run=3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# mode: compare-algos — run all 5 algorithms on one scene
# ═══════════════════════════════════════════════════════════════════════════════

def _run_compare_algos_mode(
    config: dict, radios: list, all_truths: dict,
    seed: int, num_trials: int, output_dir: Path,
) -> None:
    """Run all 5 algorithms and print/plot comparison."""
    algos = _build_all_algorithms()
    errors: dict[str, dict] = {an: empty_error_store() for an in algos}

    for trial_idx in range(num_trials):
        for radio in radios:
            truth = all_truths[radio.protocol][trial_idx]
            radio_cfg = config.get("radios", {}).get(radio.protocol, {})
            for algo_name, algo in algos.items():
                proto, err = run_single_trial(
                    truth, radio, algo, config, seed, trial_idx,
                    radio_cfg=radio_cfg,
                )
                errors[algo_name][proto].append(err)

    _print_compare_algos_table(errors, radios)

    # Per-protocol error comparison plot
    plot_error_comparison(
        {p: errors["Threshold(0.18)"][p] for p in [r.protocol for r in radios]},
        output_dir / "ranging_error_comparison.png",
    )

    # Best-algorithm per protocol
    print("\nBest algorithm per protocol (by RMSE):")
    for proto in [r.protocol for r in radios]:
        best = min(algos.keys(), key=lambda an: np.sqrt(np.mean(
            np.array(errors[an][proto]) ** 2
        )))
        rmse = np.sqrt(np.mean(np.array(errors[best][proto]) ** 2))
        print(f"  {proto}: {best} (RMSE={rmse:.3f}m)")


def _print_compare_algos_table(errors: dict, radios: list) -> None:
    """Print multi-algorithm comparison table."""
    algos = list(errors.keys())
    protocols = [r.protocol for r in radios]

    sep = "─" * 110
    print(f"\n{sep}")
    print("  Multi-Algorithm Comparison")
    print(sep)

    header = f"{'Algorithm':<20}"
    for p in protocols:
        header += f"  {p.upper()} RMSE   {p.upper()} P90  "
    header += f"{'Winner':>8}"
    print(header)
    print(sep)

    for algo_name in algos:
        s = summarize_errors(errors[algo_name])
        line = f"{algo_name:<20}"

        rmse_list = []
        for p in protocols:
            stats = s.get(p, {})
            line += f"  {stats.get('rmse_m', 0):8.3f}  {stats.get('p90_abs_m', 0):8.3f}"
            rmse_list.append((p.upper(), stats.get('rmse_m', 999)))

        winner = min(rmse_list, key=lambda x: x[1])[0]
        line += f"  {winner:>6}"
        print(line)

    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# mode: compare-materials — sweep wall materials (image method, fixed room)
# ═══════════════════════════════════════════════════════════════════════════════

_MATERIALS_COMPARE: dict[str, dict] = {
    "itu_concrete": {"epsilon": 5.31, "sigma": 0.033, "desc": "Concrete (ITU)"},
    "itu_brick":    {"epsilon": 3.75, "sigma": 0.038, "desc": "Brick (ITU)"},
    "itu_metal":    {"epsilon": 1.0,  "sigma": 1e7,   "desc": "Metal (conductor)"},
}

_MATERIALS_ROOM_CFG: dict[str, Any] = {
    "type": "simple_room",
    "dimensions_m": [12.0, 10.0, 4.0],
    "tx_position_m": [5.0, 5.0, 1.5],
    "rx_position_m": [9.0, 8.0, 1.2],
    "max_reflections": 2,
    "engine": "image_method",
}


def _run_compare_materials_mode(
    config: dict, radios: list, seed: int,
    num_trials: int, output_dir: Path,
) -> None:
    """Sweep 3 wall materials and compare ranging accuracy."""
    from copy import deepcopy

    algos = _build_all_algorithms()

    # Pre-generate truths per material (image method → shared truth across radios)
    print(f"\nGenerating truths for {num_trials} trials per material...")
    truths_by_material: dict[str, list] = {}
    for mat_name in _MATERIALS_COMPARE:
        mat_cfg = deepcopy(config)
        mat_cfg.setdefault("environment", {}).update(_MATERIALS_ROOM_CFG)
        mat_cfg["environment"]["material"] = mat_name
        mat_cfg["environment"]["num_trials"] = num_trials

        rng = _rng(seed, 0)
        truths = generate_truths(mat_cfg, rng)
        while len(truths) < num_trials:
            rng = _rng(seed, len(truths) * 1000)
            truths.extend(generate_truths(deepcopy(mat_cfg), rng))
        truths_by_material[mat_name] = truths[:num_trials]

        t0 = truths[0]
        n_paths = len(t0.a_paths)
        print(f"  {mat_name:<20}: {n_paths} paths, LOS={t0.true_range_m:.3f}m")

    # Run all algorithms × materials
    print(f"\nRunning {num_trials} trials per material...")
    errors: dict[str, dict[str, dict[str, list[float]]]] = {
        mat: {an: empty_error_store() for an in algos}
        for mat in _MATERIALS_COMPARE
    }

    _material_errors_seen: set[tuple] = set()

    for mat_name in _MATERIALS_COMPARE:
        truths = truths_by_material[mat_name]
        for trial_idx in range(num_trials):
            truth = truths[trial_idx]
            for radio in radios:
                radio_cfg = config.get("radios", {}).get(radio.protocol, {})
                for algo_name, algo in algos.items():
                    try:
                        proto, err = run_single_trial(
                            truth, radio, algo, config, seed, trial_idx,
                            radio_cfg=radio_cfg,
                        )
                        errors[mat_name][algo_name][proto].append(err)
                    except Exception as exc:
                        # Log once per material+algo+proto, don't spam
                        _err_key = (mat_name, algo_name, proto)
                        if _err_key not in _material_errors_seen:
                            _material_errors_seen.add(_err_key)
                            print(f"  ⚠ {mat_name}/{algo_name}/{proto}: {exc}", file=sys.stderr)

    _print_compare_materials_table(errors, algos, radios, num_trials)


def _print_compare_materials_table(
    errors: dict, algos: dict, radios: list, num_trials: int,
) -> None:
    """Print material × algorithm comparison table."""
    protocols = [r.protocol for r in radios]
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  Wall Material Comparison — Ranging Accuracy (Image Method)")
    print(f"  Trials: {num_trials} | Room: 12×10×4 m | LOS ≈ 5 m")
    print(sep)

    header = f"{'Material':<20} {'Algo':<18}"
    for p in protocols:
        header += f"  {p.upper()} RMSE   {p.upper()} P90  "
    print(header)
    print("-" * 120)

    for mat_name in _MATERIALS_COMPARE:
        for algo_name in algos:
            e = errors[mat_name][algo_name]
            line = f"{_MATERIALS_COMPARE[mat_name]['desc']:<20} {algo_name:<18}"
            for p in protocols:
                arr = np.array([abs(x) for x in e.get(p, [])])
                if len(arr) == 0:
                    line += f"  {'N/A':>8}  {'N/A':>8}"
                else:
                    rmse = float(np.sqrt(np.mean(arr ** 2)))
                    p90 = float(np.percentile(np.abs(arr), 90))
                    line += f"  {rmse:8.3f}  {p90:8.3f}"
            print(line)
        print("-" * 120)

    print(f"\nPlots saved to {Path('outputs') / 'material_comparison'}/")


# ═══════════════════════════════════════════════════════════════════════════════
# mode: rt-viz — RT cache → chip simulation → interactive HTML report
# ═══════════════════════════════════════════════════════════════════════════════

def _run_rt_viz_mode(
    config: dict, cache_path: Path, seed: int,
    num_trials: int | None, output_dir: Path,
) -> None:
    """Load RT cache, run chip + multipath simulation, generate HTML."""
    from environments.persistence import load_truths
    from utils.html_report import build_html_report
    from scripts.rt_cache_interactive import (
        _build_algorithms, _build_multipath_algorithms,
        _run_chip_simulation, _match_detected_to_gt, _export_csv,
        _LIGHT_SPEED,
    )

    print(f"Loading cached truths from {cache_path}")

    # Load all protocol subdirs
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
    n_trials = min(num_trials or max_trials, max_trials)
    print(f"Running {n_trials} trials per protocol")

    # Radios
    radios: dict[str, Any] = {}
    for proto in protocols:
        radios[proto] = _make_single_radio(config, proto)

    # Algorithms
    algos = _build_algorithms()
    mp_algos = _build_multipath_algorithms()

    # Run chip simulation
    impairments_cfg = {"impairments": config.get("impairments", {}), "timing": {}}
    errors: dict[str, dict[str, list[float]]] = {
        p: {a: [] for a in algos} for p in protocols
    }

    for trial_idx in range(n_trials):
        for proto in protocols:
            truth = all_truths[proto][trial_idx]
            for algo_name, algo in algos.items():
                rng = _rng(seed, trial_idx * 1000 + hash(algo_name))
                errors[proto][algo_name].append(
                    _run_chip_simulation(truth, radios[proto], algo, impairments_cfg, rng)
                )

    # CIR data for first trial
    cir_data: dict[str, dict] = {}
    for proto in protocols:
        truth = all_truths[proto][0]
        rng = _rng(seed, 9999)
        observed = apply_timing_impairments(truth, impairments_cfg, rng)
        obs_rng = _rng(seed, 9999 + hash(proto))
        obs = radios[proto].observe(observed, obs_rng)
        cir_data[proto] = {
            "t_discrete_s": obs.t_discrete_s,
            "t_cont_s": obs.t_cont_s,
            "cir_observed_discrete": obs.cir_observed_discrete,
            "cir_observed_cont": obs.cir_observed_cont,
            "cir_clean_discrete": obs.cir_clean_discrete,
            "cir_clean_cont": obs.cir_clean_cont,
            "true_first_tau_s": truth.true_first_tau_s,
        }

    # Multipath detection
    multipath_viz: dict[str, dict] = {}
    for proto in protocols:
        truth = all_truths[proto][0]
        proto_bw_hz = float(RADIO_DEFAULTS.get(proto, {}).get(
            "bandwidth_hz",
            RADIO_DEFAULTS.get(proto, {}).get("subcarrier_spacing_hz", 312_500.0)
            * RADIO_DEFAULTS.get(proto, {}).get("fft_size", 512),
        ))
        rng = _rng(seed, 9999)
        impaired = apply_timing_impairments(truth, impairments_cfg, rng)
        obs_rng = _rng(seed, 9999 + hash(proto))
        obs = radios[proto].observe(impaired, obs_rng)

        multipath_viz[proto] = {}
        for mp_name, mp_algo in mp_algos.items():
            mp_result = mp_algo.detect(obs)
            match = _match_detected_to_gt(
                mp_result.paths, truth.tau_paths_s, np.abs(truth.a_paths),
                proto_bw_hz,
                gt_types=truth.path_type, gt_orders=truth.path_order,
            )
            multipath_viz[proto][mp_name] = {
                "result": mp_result, "match": match,
                "detected_paths": [
                    {"estimated_tof_s": p.estimated_tof_s, "amplitude": p.amplitude,
                     "confidence": p.confidence, "is_first_path": p.is_first_path}
                    for p in mp_result.paths
                ],
                "gt_paths": {
                    "tau_s": truth.tau_paths_s.tolist() if hasattr(truth.tau_paths_s, "tolist") else list(truth.tau_paths_s),
                    "gains": np.abs(truth.a_paths).tolist() if hasattr(np.abs(truth.a_paths), "tolist") else list(np.abs(truth.a_paths)),
                    "types": list(truth.path_type) if truth.path_type is not None else [],
                    "orders": list(truth.path_order) if truth.path_order is not None else [],
                },
            }

    # Build HTML report
    run_info_path = cache_path / "run_info.json"
    config_info = json.loads(run_info_path.read_text(encoding="utf-8")) if run_info_path.exists() else {}
    tx_pos = np.asarray(config_info.get("tx_position_m", [0, 0, 0]), dtype=float)
    rx_pos = np.asarray(config_info.get("rx_position_m", [0, 0, 0]), dtype=float)
    scene_name = config_info.get("scene_name", "") or config_info.get("scene", "")

    first_path_viz: dict[str, dict[str, float]] = {}
    for proto in protocols:
        first_path_viz[proto] = {}
        impaired = apply_timing_impairments(
            all_truths[proto][0], impairments_cfg, _rng(seed, 9999),
        )
        obs = radios[proto].observe(impaired, _rng(seed, 9999 + hash(proto)))
        for algo_name, algo in algos.items():
            estimate = algo.estimate(obs)
            first_path_viz[proto][algo_name] = float(estimate.estimated_tof_s)

    html_path = build_html_report(
        cache_path, cir_data, errors, config_info,
        scene_name, tx_pos, rx_pos, all_truths[protocols[0]][0],
        multipath_results=multipath_viz,
        multipath_errors={p: {a: [] for a in mp_algos} for p in protocols},
        first_path_viz=first_path_viz,
    )
    print(f"HTML: {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")

    csv_path = _export_csv(cache_path, errors, config_info)
    print(f"CSV:  {csv_path}")

    # Summary table
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


def _make_single_radio(config: dict, proto: str):
    """Create a single radio for *proto* from config."""
    radio_cfg = config.get("radios", {}).get(proto, {})
    overrides = {k: v for k, v in radio_cfg.items()
                 if k not in ("enabled", "carrier_frequency_hz")}
    cfreq = radio_cfg.get("carrier_frequency_hz", None)
    return create_radio(proto, carrier_frequency_hz=cfreq,
                        overrides=overrides,
                        impairments=config.get("impairments", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# mode: measure — ranging simulation along a trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def _run_measure_mode(args: argparse.Namespace) -> None:
    """Run measure simulation — built-in scene or custom floorplan + CSV (delegates to script)."""
    import sys as _sys

    _argv = ["run_measure"]
    if args.floorplan_image:
        _argv.extend(["--floorplan", str(args.floorplan_image)])
    if args.waypoints_file:
        _argv.extend(["--waypoints-file", str(args.waypoints_file)])
    if args.floorplan_width_m:
        _argv.extend(["--floorplan-width-m", str(args.floorplan_width_m)])
    if args.tx:
        _argv.extend(["--tx", str(args.tx[0]), str(args.tx[1]), str(args.tx[2])])
    # Built-in scene
    if not args.floorplan_image:
        _argv.extend(["--scene", args.trajectory_scene])
    _argv.extend(["--waypoints", str(args.waypoints)])
    _argv.extend(["--speed", str(args.speed)])
    _argv.extend(["--impairments", args.impairments or "none"])
    if args.radios:
        _argv.extend(["--radios", args.radios])
    if args.algo:
        _argv.extend(["--algorithms", args.algo])
    _argv.extend(["--trials", str(args.trials or 1)])
    _argv.extend(["--seed", str(args.seed or 42)])

    _old = _sys.argv
    _sys.argv = _argv
    try:
        from scripts.run_measure import main as _measure_main
        _measure_main()
    finally:
        _sys.argv = _old


# ═══════════════════════════════════════════════════════════════════════════════
# mode: fingerprint — WiFi RSSI + ranging radio map on floorplan grid
# ═══════════════════════════════════════════════════════════════════════════════

def _run_fingerprint_mode(args: argparse.Namespace) -> None:
    """Run fingerprint radio map simulation (delegates to script)."""
    import sys as _sys

    if not args.floorplan_image:
        print("Error: --floorplan-image is required for --mode fingerprint")
        _sys.exit(1)
    if not args.floorplan_width_m:
        print("Error: --floorplan-width-m is required for --mode fingerprint")
        _sys.exit(1)
    _argv = ["run_fingerprint",
             "--floorplan", str(args.floorplan_image),
             "--floorplan-width-m", str(args.floorplan_width_m),
             "--grid-spacing", str(args.grid_spacing)]
    if args.aps:
        _argv.extend(["--aps", str(args.aps)])
    _argv.extend(["--algo", args.algo or "leading_edge"])
    _argv.extend(["--tx-power", str(args.tx_power)])
    _argv.extend(["--impairments", args.impairments or "none"])
    _argv.extend(["--trials", str(args.trials or 1)])
    _argv.extend(["--seed", str(args.seed or 42)])

    _old = _sys.argv
    _sys.argv = _argv
    try:
        from scripts.run_fingerprint import main as _fp_main
        _fp_main()
    finally:
        _sys.argv = _old


# ═══════════════════════════════════════════════════════════════════════════════
# mode: ablation — impairment ablation + degradation sweeps
# ═══════════════════════════════════════════════════════════════════════════════

def _run_ablation_mode(args: argparse.Namespace) -> None:
    """Run impairment ablation / sweeps (delegates to script)."""
    import sys as _sys

    _argv = ["run_ablation", "--mode", args.ablation_mode]
    n_trials = str(args.trials or 100)
    _argv.extend(["--ablation-trials", n_trials])
    _argv.extend(["--sweep-trials", n_trials])
    _argv.extend(["--seed", str(args.seed or 42)])

    _old = _sys.argv
    _sys.argv = _argv
    try:
        from scripts.run_impairment_ablation_final import main as _abl_main
        _abl_main()
    finally:
        _sys.argv = _old


# ═══════════════════════════════════════════════════════════════════════════════
# mode: interactive — 3D HTML visualization for canonical scenes
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_interactive_cache(
    scene_def: dict, trials: int, seed: int, cache_path: Path,
) -> None:
    """Generate RT cache for a single interactive scene.

    Uses the same code paths as ``--dump-truths``: assembles a config from
    a synthetic ``argparse.Namespace``, builds radios, generates truths, and
    saves them alongside ``run_info.json``.
    """
    from environments.persistence import save_truths

    ns_dict: dict[str, Any] = {
        "config": None, "experiment": None,
        "scene": scene_def.get("scene"),
        "radios": "all",
        "impairments": "none",
        "algo": None,
        "trials": trials,
        "seed": seed,
        "output": None,
        "tx": scene_def.get("tx"),
        "rx": scene_def.get("rx"),
        "floorplan_image": scene_def.get("floorplan"),
        "floorplan_ppm": 20.0,
        "floorplan_wall_height": 3.0,
        "floorplan_tolerance": 40.0,
        "floorplan_bg": "255,255,255",
        "num_samples": scene_def.get("num_samples", 100000),
        "max_reflections": scene_def.get("max_reflections", 2),
        "dump_truths": None,
        "from_cache": None,
    }
    ns = argparse.Namespace(**ns_dict)

    config, _source = _assemble_config(ns)
    config["seed"] = seed

    radios = build_radios(config)

    all_truths = _generate_truths_for_radios(config, radios, seed)

    # Persist truths
    cache_path.mkdir(parents=True, exist_ok=True)
    for proto, truths in all_truths.items():
        save_truths(truths, cache_path / proto)

    # run_info.json
    first_proto = list(all_truths.keys())[0]
    first_truth = all_truths[first_proto][0]
    env_cfg = config.get("environment", {})

    run_info: dict[str, Any] = {
        "source": f"interactive:{scene_def.get('cache_name', 'unknown')}",
        "scene": str(env_cfg.get("type", "")),
        "scene_name": str(env_cfg.get("scene_name", scene_def.get("cache_name", ""))),
        "tx_position_m": env_cfg.get("tx_position_m", [0, 0, 0]),
        "rx_position_m": env_cfg.get("rx_position_m", [0, 0, 0]),
        "seed": seed,
        "num_trials": trials,
        "radios": [r.protocol for r in radios],
    }

    wall_geo = first_truth.metadata.get("wall_geometry")
    if wall_geo is not None:
        run_info["wall_geometry"] = wall_geo

    scene_xml = first_truth.metadata.get("scene_xml")
    if scene_xml:
        (cache_path / "scene.xml").write_text(str(scene_xml), encoding="utf-8")
        run_info["has_scene_xml"] = True

    (cache_path / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  Cache saved → {cache_path}")


def _run_interactive_mode(args: argparse.Namespace, seed: int, output_dir: Path) -> None:
    """Interactive 3D HTML reports for 3 canonical scenes.

    For each scene the function:
    1. ensures an RT cache exists (generating one if needed), then
    2. delegates to ``scripts/rt_cache_interactive.py`` to build an
       interactive 3D HTML report with CIR + CDF + ray-path visualization.
    """
    import sys as _sys

    trials = args.trials if args.trials is not None else 100

    INTERACTIVE_SCENES: list[dict[str, Any]] = [
        {
            "cache_name": "box_knife",
            "label": "Box + Knife-Edge (NLOS)",
            "scene": "box_knife",
        },
        {
            "cache_name": "box",
            "label": "Simple Box (LOS)",
            "scene": "box",
        },
        {
            "cache_name": "etoile",
            "label": "Etoile / Arc de Triomphe (Urban LOS)",
            "scene": "etoile",
        },
        {
            "cache_name": "complex_building",
            "label": "Complex Building (Custom Floorplan)",
            "scene": None,
            "floorplan": "floorplans/complex_building.png",
            "tx": [5.0, 10.0, 1.5],
            "rx": [32.0, 20.0, 1.5],
            "max_reflections": 4,
            "num_samples": 200000,
        },
    ]

    print("=" * 60)
    print("  RadioRange-Sim — Interactive Visualization")
    print("=" * 60)
    print()
    print(f"{len(INTERACTIVE_SCENES)} interactive 3D HTML reports will be generated:")
    for d in INTERACTIVE_SCENES:
        print(f"  • {d['label']}")
    print()

    html_paths: list[tuple[str, Path]] = []

    for scene_def in INTERACTIVE_SCENES:
        cache_path = Path(f"cache/rt/{scene_def['cache_name']}")
        label = scene_def["label"]

        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")

        # ── Step 1: ensure cache ─────────────────────────────────────────
        if not (cache_path / "run_info.json").exists():
            print("  Generating RT cache (this may take a few minutes)...")
            try:
                _generate_interactive_cache(scene_def, trials, seed, cache_path)
            except Exception as exc:
                print(f"  ✗ Failed: {exc}", file=_sys.stderr)
                continue
        else:
            print(f"  Using existing cache → {cache_path}")

        # ── Step 2: run rt_cache_interactive ─────────────────────────────
        print("  Building interactive HTML...")
        _argv = [
            "rt_cache_interactive", str(cache_path),
            "--trials", str(trials),
        ]
        _old = _sys.argv
        _sys.argv = _argv
        try:
            from scripts.rt_cache_interactive import main as _rt_main
            _rt_main()
        except Exception as exc:
            print(f"  ✗ Failed: {exc}", file=_sys.stderr)
            import traceback
            traceback.print_exc()
            continue
        finally:
            _sys.argv = _old

        html_path = output_dir / "interactive" / f"{scene_def['cache_name']}_chip_sim.html"
        if html_path.exists():
            html_paths.append((label, html_path))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Visualization complete! Open in browser:")
    print(f"{'=' * 60}")
    for label, path in html_paths:
        size_kb = path.stat().st_size / 1024
        print(f"\n  {label}:")
        print(f"    file://{path.resolve()}")
        print(f"    ({size_kb:.0f} KB)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# trial execution
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_truths_for_radios(
    config: dict, radios: list, seed: int,
) -> dict[str, list]:
    if not _is_sionna_rt(config):
        rng = _rng(seed, 0)
        shared = generate_truths(config, rng)
        return {radio.protocol: shared for radio in radios}

    truths: dict[str, list] = {}
    for radio in radios:
        proto_cfg = config.get("radios", {}).get(radio.protocol, {})
        freq = float(proto_cfg.get(
            "carrier_frequency_hz",
            config.get("environment", {}).get(
                "carrier_frequency_hz",
                DEFAULT_CARRIER_FREQ_HZ.get(radio.protocol, 5e9),
            ),
        ))
        rng = _rng(seed, _PROTO_SEEDS[radio.protocol])
        truths[radio.protocol] = generate_truths(config, rng, carrier_frequency_hz=freq)
    return truths


def _run_trial(
    config: dict, radio, truth, trial_idx: int, seed: int, algorithm,
) -> tuple[str, float]:
    """Delegate to the canonical single-trial pipeline in utils/runner."""
    radio_cfg = config.get("radios", {}).get(radio.protocol, {})
    return run_single_trial(truth, radio, algorithm, config, seed, trial_idx,
                            radio_cfg=radio_cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_snr_radios(config: dict) -> list:
    """Build radios forced to SNR observation model (for CIR comparison plots)."""
    return build_radios_from_config(config, observation_model="snr")


def _generate_plot_truths(
    config: dict, radios: list, seed: int,
) -> dict[str, object]:
    plot_rng = _rng(seed, 9999)
    truths: dict[str, object] = {}
    for radio in radios:
        radio_cfg = config.get("radios", {}).get(radio.protocol, {})
        if _is_sionna_rt(config):
            freq = float(radio_cfg.get(
                "carrier_frequency_hz",
                config.get("environment", {}).get(
                    "carrier_frequency_hz",
                    DEFAULT_CARRIER_FREQ_HZ.get(radio.protocol, 5e9),
                ),
            ))
            truth_rng = _rng(seed, 9999 + _PROTO_SEEDS[radio.protocol])
            raw = generate_truths(config, truth_rng, carrier_frequency_hz=freq)[0]
        else:
            raw = generate_truths(config, plot_rng)[0]
        truths[radio.protocol] = apply_timing_impairments(
            raw, config, plot_rng, radio_cfg=radio_cfg,
        )
    return truths


# ═══════════════════════════════════════════════════════════════════════════════
# config assembly — merges CLI overrides onto a (possibly empty) config base
# ═══════════════════════════════════════════════════════════════════════════════

def _default_radio_config(protocols: list[str]) -> dict:
    """Return a minimal radio section enabling the given protocols with
    sensible defaults (matching config/radio_profiles/default_radios.yaml)."""
    radios: dict[str, dict] = {}
    for p in ["uwb", "wifi", "fiveg"]:
        enabled = p in protocols
        defaults = dict(RADIO_DEFAULTS.get(p, {}))
        radios[p] = {
            "enabled": enabled,
            "carrier_frequency_hz": DEFAULT_CARRIER_FREQ_HZ[p],
            **defaults,
        }
    return {"radios": radios}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* recursively.  Lists are replaced, not merged."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _assemble_config(args: argparse.Namespace) -> tuple[dict, str]:
    """Build the final config dict from experiment file + CLI overrides.

    Returns (config, source_label) where *source_label* is a human-readable
    description of where the config came from (for logging).
    """
    # 1. base — either from experiment file or empty
    config_path = args.experiment or args.config
    if config_path:
        config = load_config(config_path)
        source = config_path
    else:
        config = {}
        source = "cli"

    # 2. scene preset
    if args.scene:
        preset = SCENE_PRESETS.get(args.scene)
        if preset is None:
            print(f"Unknown scene: '{args.scene}'. Use --list-scenes to see available scenes.")
            sys.exit(1)
        config.setdefault("environment", {})
        _deep_merge(config["environment"], preset.get("environment", {}))
        # If user didn't specify radios, default to all three
        if not args.radios:
            config.setdefault("radios", {})
            for p in ["uwb", "wifi", "fiveg"]:
                config["radios"].setdefault(p, {})["enabled"] = True
        source = f"{source}+scene={args.scene}"

    # 3. radios
    if args.radios:
        wanted = _parse_radio_list(args.radios)
        radio_defaults = _default_radio_config(wanted)
        config.setdefault("radios", {})
        _deep_merge(config["radios"], radio_defaults["radios"])

    # If no radios are enabled yet, enable all three with sensible defaults
    if not config.get("radios") or not any(
        r.get("enabled") for r in config.get("radios", {}).values()
    ):
        config.setdefault("radios", {})
        radio_defaults = _default_radio_config(["uwb", "wifi", "fiveg"])
        _deep_merge(config["radios"], radio_defaults["radios"])

    # 4. impairments
    if args.impairments:
        preset = IMPAIRMENT_PRESETS.get(args.impairments)
        if preset is None:
            print(f"Unknown impairments preset: '{args.impairments}'. Choose: none, full")
            sys.exit(1)
        config.setdefault("impairments", {})
        _deep_merge(config["impairments"], preset)

    # 5. algorithm
    if args.algo:
        config.setdefault("algorithms", {})
        config["algorithms"]["primary"] = args.algo

    # 6. trials
    if args.trials is not None:
        config.setdefault("environment", {})
        config["environment"]["num_trials"] = args.trials

    # 7. output dir
    if args.output:
        config["output_dir"] = args.output
    elif "output_dir" not in config:
        config["output_dir"] = "outputs"

    # 8. tx / rx position overrides
    if args.tx is not None:
        config.setdefault("environment", {})
        config["environment"]["tx_position_m"] = args.tx
    if args.rx is not None:
        config.setdefault("environment", {})
        config["environment"]["rx_position_m"] = args.rx

    # 8b. floorplan image (enables type=floorplan)
    if args.floorplan_image is not None:
        bg_color = [int(x) for x in args.floorplan_bg.split(",")]
        config.setdefault("environment", {})
        config["environment"]["type"] = "floorplan"
        config["environment"].setdefault("los", True)
        config["environment"].setdefault("specular_reflection", True)
        config["environment"].setdefault("diffuse_reflection", True)
        config["environment"].setdefault("refraction", True)
        config["environment"].setdefault("diffraction", True)
        config["environment"].setdefault("scattering_coefficient", 0.3)
        config["environment"]["max_reflections"] = args.max_reflections or 2
        if args.num_samples is not None:
            config["environment"]["num_samples"] = args.num_samples
        fp = config["environment"].setdefault("floorplan", {})
        fp["image_path"] = args.floorplan_image
        fp["pixels_per_meter"] = args.floorplan_ppm
        fp["wall_height_m"] = args.floorplan_wall_height
        fp["default_tolerance"] = args.floorplan_tolerance
        fp["background_color"] = bg_color
        fp.setdefault("color_mapping", [
            {"color": [0, 0, 0],       "material": "itu_concrete"},
            {"color": [128, 64, 0],    "material": "itu_brick"},
            {"color": [0, 0, 255],     "material": "itu_glass"},
            {"color": [139, 90, 43],   "material": "itu_wood"},
            {"color": [192, 192, 192], "material": "itu_metal"},
            {"color": [255, 200, 100], "material": "itu_plasterboard"},
        ])
        fp.setdefault("generate_floor_ceiling", True)
        fp.setdefault("floor_material", "itu_concrete")
        fp.setdefault("ceiling_material", "itu_ceiling_board")

    # 9. seed
    if args.seed is not None:
        config["seed"] = args.seed

    # 10. ensure bare-minimum sections exist
    config.setdefault("seed", 42)
    config.setdefault("timing", {})
    config.setdefault("plots", {"time_window_ns": 120.0})
    config.setdefault("algorithms", {}).setdefault("primary", "threshold")
    config.setdefault("environment", {}).setdefault("num_trials", 1)
    config.setdefault("impairments", {})

    return config, source


def _parse_radio_list(raw: str) -> list[str]:
    """Parse comma-separated radio names.  'all' expands to all three."""
    if raw.strip().lower() == "all":
        return ["uwb", "wifi", "fiveg"]
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def _cache_dir_name(config: dict) -> str:
    """Derive a human-readable cache directory name from the config."""
    env = config.get("environment", {})
    scene = env.get("type", "unknown")
    if scene == "sionna_builtin_scene":
        scene = env.get("scene_name", "sionna")
    if scene == "standard_tdl":
        scene = env.get("model", "tdl")
    if scene == "simple_room":
        engine = env.get("engine", "sionna")
        scene = f"room_{engine.split('_')[1] if '_' in engine else engine}"

    radios_enabled = [
        p for p in ["uwb", "wifi", "fiveg"]
        if config.get("radios", {}).get(p, {}).get("enabled", False)
    ]
    proto = "+".join(radios_enabled) if radios_enabled else "all"

    freq = 5e9
    for p in radios_enabled:
        f = config.get("radios", {}).get(p, {}).get("carrier_frequency_hz")
        if f:
            freq = f
            break
    freq_mhz = int(freq / 1e6)

    max_refl = int(env.get("max_reflections", 1))

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{scene}_{proto}_{freq_mhz}MHz_r{max_refl}_{date_str}"


# ═══════════════════════════════════════════════════════════════════════════════
# debug output
# ═══════════════════════════════════════════════════════════════════════════════

def _print_cir_samples(observations: list, num_samples: int = 12) -> None:
    print("\nClean(real) CIR vs observed CIR samples")
    for obs in observations:
        n = min(num_samples, len(obs.t_discrete_s))
        clean = np.asarray(obs.cir_clean_discrete[:n], dtype=float)
        observed = np.asarray(obs.cir_observed_discrete[:n], dtype=float)
        delta = observed - clean
        print(f"[{obs.protocol}] first {n} discrete samples")
        print(f"  observation model: {obs.metadata.get('observation_model', 'unknown')}")
        if "snr_db" in obs.metadata:
            print(f"  snr_db: {obs.metadata['snr_db']}")
        if "csi_noise_snr_db" in obs.metadata:
            print(f"  csi_noise_snr_db: {obs.metadata['csi_noise_snr_db']}")
        if "active_subcarriers" in obs.metadata:
            print(f"  active_subcarriers: {obs.metadata['active_subcarriers']}")
        if "pilot_spacing_subcarriers" in obs.metadata:
            print(f"  pilot_spacing_subcarriers: {obs.metadata['pilot_spacing_subcarriers']}")
        print(
            "  phase/timing: "
            f"common_phase={obs.metadata.get('common_phase_rad', 0.0):.4g} rad, "
            f"sampling_offset={obs.metadata.get('sampling_phase_offset_s', 0.0) * 1e9:.4g} ns, "
            f"sfo={obs.metadata.get('sfo_residual_ppm', obs.metadata.get('sfo_ppm', 0.0)):.4g} ppm"
        )
        print("  t_ns     :", np.array2string(obs.t_discrete_s[:n] * 1e9, precision=3, separator=", "))
        print("  clean    :", np.array2string(clean, precision=4, separator=", "))
        print("  observed :", np.array2string(observed, precision=4, separator=", "))
        print("  obs-clean:", np.array2string(delta, precision=4, separator=", "))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RadioRange-Sim — RF ranging error simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_cli_epilog(),
    )

    # config file (optional — can run entirely from CLI)
    p.add_argument("--config", default=None, help="Path to a JSON/YAML config file.")
    p.add_argument("--experiment", default=None, help="Alias for --config (run presets).")

    # scene
    p.add_argument("--scene", default=None, metavar="NAME",
                   help="Scene preset: box, box_knife, box_knife_concrete, etoile, munich, florence, simple_room, tdl_a..tdl_e, two_path")
    p.add_argument("--list-scenes", action="store_true", help="List available scene presets and exit.")

    # radios
    p.add_argument("--radios", default=None, metavar="LIST",
                   help="Comma-separated radios: uwb,wifi,fiveg  or 'all'")

    # impairments
    p.add_argument("--impairments", default=None, metavar="PRESET",
                   help="Impairment preset: none, full")

    # algorithm
    p.add_argument("--algo", default=None, metavar="NAME",
                   choices=["max_peak", "threshold", "leading_edge", "search_back", "chip_lde"],
                   help="LDE algorithm: max_peak, threshold, leading_edge, search_back, chip_lde")

    # experiment mode
    p.add_argument("--mode", default="single", metavar="MODE",
                   choices=["single", "compare-algos", "compare-materials",
                            "rt-viz", "measure", "ablation", "interactive",
                            "fingerprint"],
                   help="Experiment mode: single (default), compare-algos, "
                        "compare-materials, rt-viz, measure, ablation, interactive, fingerprint")

    # measure options
    p.add_argument("--waypoints", type=int, default=40, metavar="N",
                   help="Number of waypoints for measure mode (default: 40).")
    p.add_argument("--speed", type=float, default=0.5, metavar="MPS",
                   help="Cruise speed in m/s for measure mode (default: 0.5).")
    p.add_argument("--trajectory-scene", default="corridor", metavar="SCENE",
                   choices=["corridor", "t_junction"],
                   help="Built-in scene for measure mode: corridor or t_junction (default: corridor).")
    p.add_argument("--waypoints-file", default=None, metavar="PATH",
                   help="Path to waypoints CSV (N×3) for measure mode (custom floorplan).")
    p.add_argument("--floorplan-width-m", type=float, default=None, metavar="M",
                   help="Physical width of floorplan in meters (for custom floorplan in measure mode).")

    # fingerprint options
    p.add_argument("--aps", default=None, metavar="PATH",
                   help="Path to AP positions CSV (ap_id,x,y,z) for fingerprint mode.")
    p.add_argument("--grid-spacing", type=float, default=2.0, metavar="M",
                   help="Grid spacing in meters for fingerprint mode (default: 2.0).")
    p.add_argument("--tx-power", type=float, default=20.0, metavar="DBM",
                   help="WiFi TX power in dBm for fingerprint mode (default: 20).")

    # ablation options
    p.add_argument("--ablation-mode", default="ablation", metavar="SUB",
                   choices=["ablation", "sweeps", "all"],
                   help="Ablation sub-mode: ablation, sweeps, or all (default: ablation).")

    # other overrides
    p.add_argument("--trials", type=int, default=None, metavar="N", help="Number of Monte Carlo trials.")
    p.add_argument("--seed", type=int, default=None, metavar="N", help="Random seed.")
    p.add_argument("--output", default=None, metavar="DIR", help="Output directory.")
    p.add_argument("--tx", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="TX position override (3 floats).")
    p.add_argument("--rx", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="RX position override (3 floats).")

    # floorplan
    p.add_argument("--floorplan-image", default=None, metavar="PATH",
                   help="Path to floorplan PNG (enables type=floorplan).")
    p.add_argument("--floorplan-ppm", type=float, default=20.0, metavar="N",
                   help="Pixels per meter for floorplan (default: 20).")
    p.add_argument("--floorplan-wall-height", type=float, default=3.0, metavar="M",
                   help="Wall height in meters (default: 3.0).")
    p.add_argument("--floorplan-tolerance", type=float, default=40.0, metavar="N",
                   help="Color matching tolerance (default: 40).")
    p.add_argument("--floorplan-bg", default="255,255,255", metavar="R,G,B",
                   help="Background color (default: 255,255,255).")
    p.add_argument("--num-samples", type=int, default=None, metavar="N",
                   help="Sionna RT ray count (default: 100000, increase for NLOS).")
    p.add_argument("--max-reflections", type=int, default=None, metavar="N",
                   help="Max ray bounces (default: 2, increase for corridors).")

    # cache
    p.add_argument("--dump-truths", default=None, metavar="DIR",
                   help="Run RT only, save ChannelTruth to cache/DIR, then exit.")
    p.add_argument("--from-cache", default=None, metavar="PATH",
                   help="Skip RT, load ChannelTruth from a cache directory.")

    return p.parse_args()


def _cli_epilog() -> str:
    lines = ["examples:",
             "  # Quick start: interactive 3D visualization (4 canonical scenes)",
             "  %(prog)s --mode interactive",
             "",
             "  # Single scene: UWB + threshold LDE on Munich",
             "  %(prog)s --scene munich --radios uwb --algo threshold",
             "",
             "  # Compare all radios on etoile with full impairments",
             "  %(prog)s --scene etoile --radios all --impairments full --algo leading_edge",
             "",
             "  # Override TX/RX positions",
             "  %(prog)s --scene munich --tx 200 200 1.5 --rx 220 210 1.5",
             "",
             "  # Use experiment file as base, override algorithm",
             "  %(prog)s --experiment config/runs/tdl_a.yaml --algo max_peak",
             "",
             "  # Monte Carlo with custom trials on simple_room",
             "  %(prog)s --scene simple_room --radios uwb --algo threshold --trials 500",
             "",
             "  # Cache RT results for later reuse",
             "  %(prog)s --scene munich --radios uwb --dump-truths cache/rt/my_run",
             "",
             "  # Replay chip-level sim from cached RT",
             "  %(prog)s --from-cache cache/rt/my_run --algo leading_edge --impairments full --trials 2000",
             "",
             "  # Floorplan: PNG image → 3D scene → full pipeline",
             "  %(prog)s --floorplan-image floorplans/my_office.png \\",
             "           --tx 2 1 1.5 --rx 17 13 1.5 --radios uwb \\",
             "           --dump-truths cache/rt/my_floorplan",
    ]
    return "\n".join(lines)


def _print_scene_list() -> None:
    from utils.scene_presets import SCENE_PRESETS

    CATEGORY_LABELS = {
        "statistical":    "Statistical (3GPP TR 38.901, no materials)",
        "deterministic":  "Deterministic (no materials)",
        "sionna_builtin": "Sionna built-in scenes (materials baked in)",
        "procedural":     "Procedural rooms (materials in preset)",
    }

    # Collect and group by category
    by_category: dict[str, list[str]] = {}
    for name, preset in SCENE_PRESETS.items():
        cat = preset.get("category", "other")
        by_category.setdefault(cat, []).append(name)

    print("Available scene presets (--scene NAME):\n")

    for cat_key, cat_label in CATEGORY_LABELS.items():
        names = by_category.get(cat_key, [])
        if not names:
            continue
        print(f"  {cat_label}:")
        for name in sorted(names):
            desc = SCENE_PRESETS[name].get("description", name)
            print(f"    {name:<20} {desc}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    if args.list_scenes:
        _print_scene_list()
        return

    # Interactive mode — self-contained, bypasses normal config assembly
    if args.mode == "interactive":
        seed = args.seed if args.seed is not None else 42
        output_dir = Path(args.output) if args.output else Path("outputs")
        _run_interactive_mode(args, seed, output_dir)
        return

    # Measure mode — built-in scene or custom floorplan + waypoints CSV
    if args.mode == "measure":
        _run_measure_mode(args)
        return

    # Fingerprint mode — WiFi RSSI + ranging radio map
    if args.mode == "fingerprint":
        _run_fingerprint_mode(args)
        return

    config, source = _assemble_config(args)
    seed = int(config.get("seed", 42))
    output_dir = Path(config.get("output_dir", "outputs"))

    radios = build_radios(config)
    if not radios:
        print("No radios enabled. Use --radios uwb,wifi,fiveg or check your config.")
        sys.exit(1)

    algorithm = build_algorithm(config)
    env_cfg = config["environment"]
    num_trials = int(env_cfg.get("num_trials", 1))

    # ── cache: dump mode ──────────────────────────────────────────────────
    if args.dump_truths:
        if args.dump_truths:
            cache_dir = Path(args.dump_truths)
        else:
            cache_dir = Path("cache/rt") / _cache_dir_name(config)
        print(f"RT cache mode — saving truths to {cache_dir}")
        all_truths = _generate_truths_for_radios(config, radios, seed)
        # Flatten per-radio dict into labelled list for storage
        for proto, truths in all_truths.items():
            proto_dir = cache_dir / proto
            save_truths(truths, proto_dir)
        # Write a top-level run_info.json
        first_proto = list(all_truths.keys())[0]
        first_truth = all_truths[first_proto][0]
        wall_geo = first_truth.metadata.get("wall_geometry")
        room_size = first_truth.metadata.get("room_size_m")
        run_info = {
            "source": source,
            "scene": str(env_cfg.get("type", "")),
            "scene_name": str(env_cfg.get("scene_name", "")),
            "tx_position_m": env_cfg.get("tx_position_m", [0, 0, 0]),
            "rx_position_m": env_cfg.get("rx_position_m", [0, 0, 0]),
            "seed": seed,
            "num_trials": num_trials,
            "radios": [r.protocol for r in radios],
        }
        if wall_geo is not None:
            run_info["wall_geometry"] = wall_geo
        if room_size is not None:
            run_info["room_size_m"] = room_size
        scene_xml = first_truth.metadata.get("scene_xml")
        if scene_xml:
            (cache_dir / "scene.xml").write_text(str(scene_xml), encoding="utf-8")
            run_info["has_scene_xml"] = True
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Saved: {cache_dir}")
        for proto in all_truths:
            print(f"  {proto}/  truths.npz + config.json")
        return

    # ── cache: load mode ──────────────────────────────────────────────────
    if args.from_cache:
        cache_path = Path(args.from_cache)
        if not cache_path.exists():
            print(f"Cache directory not found: {cache_path}")
            sys.exit(1)
        print(f"Loading cached truths from {cache_path}")
        all_truths: dict[str, list] = {}
        for radio in radios:
            proto_dir = cache_path / radio.protocol
            if not proto_dir.exists():
                print(f"  No cache for {radio.protocol} in {cache_path}, generating...")
                rng = _rng(seed, _PROTO_SEEDS[radio.protocol])
                proto_cfg = config.get("radios", {}).get(radio.protocol, {})
                freq = float(proto_cfg.get(
                    "carrier_frequency_hz",
                    config.get("environment", {}).get(
                        "carrier_frequency_hz",
                        DEFAULT_CARRIER_FREQ_HZ.get(radio.protocol, 5e9),
                    ),
                ))
                all_truths[radio.protocol] = generate_truths(config, rng, carrier_frequency_hz=freq)
            else:
                all_truths[radio.protocol] = load_truths(proto_dir)
                print(f"  {radio.protocol}: {len(all_truths[radio.protocol])} trials loaded")
        num_trials = min(len(v) for v in all_truths.values())
        print(f"  Using {num_trials} trials (min across protocols)")
    else:
        # ── pre-generate all truths ───────────────────────────────────────
        all_truths = _generate_truths_for_radios(config, radios, seed)

    # ── mode dispatch ────────────────────────────────────────────────────
    if args.mode == "compare-algos":
        _run_compare_algos_mode(config, radios, all_truths, seed, num_trials, output_dir)
        return

    if args.mode == "compare-materials":
        _run_compare_materials_mode(config, radios, seed, num_trials, output_dir)
        return

    if args.mode == "rt-viz":
        if not args.from_cache:
            print("--mode rt-viz requires --from-cache PATH")
            sys.exit(1)
        cache_path = Path(args.from_cache)
        if not cache_path.exists():
            print(f"Cache directory not found: {cache_path}")
            sys.exit(1)
        _run_rt_viz_mode(config, cache_path, seed, args.trials, output_dir)
        return

    if args.mode == "ablation":
        _run_ablation_mode(args)
        return

    # ── mode: single (default) — run every trial ────────────────────────
    errors_by_protocol = empty_error_store()
    for trial_idx in range(num_trials):
        for radio in radios:
            truth = all_truths[radio.protocol][trial_idx]
            protocol, error_m = _run_trial(config, radio, truth, trial_idx, seed, algorithm)
            errors_by_protocol[protocol].append(error_m)

    # ── generate plot observations ──────────────────────────────────────
    plot_truths = _generate_plot_truths(config, radios, seed)
    first_truth = plot_truths[radios[0].protocol]

    # Detailed-error observations
    plot_rng = _rng(seed, 9999)
    detailed_obs = [
        radio.observe(plot_truths[radio.protocol], plot_rng) for radio in radios
    ]

    # SNR-based observations (for comparison plots)
    snr_radios = _build_snr_radios(config)
    snr_rng = _rng(seed, 9999)
    snr_truths = _generate_plot_truths(config, snr_radios, seed)
    snr_obs = [
        radio.observe(snr_truths[radio.protocol], snr_rng) for radio in snr_radios
    ]

    # ── plots ───────────────────────────────────────────────────────────
    time_window_ns = float(config.get("plots", {}).get("time_window_ns", 120.0))

    plot_cir_comparison(
        snr_obs, first_truth,
        output_dir / "snr_based_cir_comparison.png",
        time_window_ns=time_window_ns, observed=True,
        title_prefix="SNR-Based Observed",
    )
    plot_cir_comparison(
        detailed_obs, first_truth,
        output_dir / "detailed_error_cir_comparison.png",
        time_window_ns=time_window_ns, observed=True,
        title_prefix="Detailed Error Observed",
    )
    plot_cir_comparison(
        detailed_obs, first_truth,
        output_dir / "clean_cir_comparison.png",
        time_window_ns=time_window_ns, observed=False,
    )
    plot_error_comparison(errors_by_protocol, output_dir / "ranging_error_comparison.png")

    # ── report ──────────────────────────────────────────────────────────
    summary = summarize_errors(errors_by_protocol)
    sample_proto = radios[0].protocol

    print("RadioRange-Sim run complete")
    print(f"Source: {source}")
    print(f"Environment: {env_cfg.get('type', 'standard_tdl')}")
    print(f"Channel source: {first_truth.metadata.get('channel_source', 'unknown')}")
    print(f"Algorithm: {algorithm.name}")
    print(f"Trials per radio: {len(errors_by_protocol.get(sample_proto, []))}")
    for radio in radios:
        proto_cfg = config.get("radios", {}).get(radio.protocol, {})
        freq = float(proto_cfg.get(
            "carrier_frequency_hz",
            config.get("environment", {}).get(
                "carrier_frequency_hz",
                DEFAULT_CARRIER_FREQ_HZ.get(radio.protocol, 5e9),
            ),
        ))
        print(f"  {radio.protocol:>5} carrier: {freq * 1e-9:.3f} GHz")
    print("Protocol         Bias(m)     Std(m)    RMSE(m)  P90|err|(m)")
    for protocol, stats in summary.items():
        print(
            f"{protocol:<10}"
            f"{stats['bias_m']:>12.3f}"
            f"{stats['std_m']:>11.3f}"
            f"{stats['rmse_m']:>11.3f}"
            f"{stats['p90_abs_m']:>13.3f}"
        )
    for name in [
        "clean_cir_comparison.png",
        "snr_based_cir_comparison.png",
        "detailed_error_cir_comparison.png",
        "ranging_error_comparison.png",
    ]:
        print(f"Saved: {output_dir / name}")

    print("\nSNR-based CIR samples")
    _print_cir_samples(snr_obs)
    print("\nDetailed-error CIR samples")
    _print_cir_samples(detailed_obs)


if __name__ == "__main__":
    main()
