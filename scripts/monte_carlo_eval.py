from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np

from algorithms import (
    ChipLeadingEdgeLde,
    LeadingEdgeLde,
    MaxPeakLde,
    SearchBackLde,
    ThresholdLde,
)
from environments import generate_truths
from hardware.impairments import apply_timing_impairments
from utils.config import load_config
from utils.evaluator import empty_error_store, range_error_m, summarize_errors
from utils.radio_factory import build_radios_from_config
from utils.runner import _PROTO_SEEDS, _rng

# ── environments (config_path, env_type, label) ──────────────────────────
_ENVIRONMENTS = [
    # TDL statistical models (fast)
    ("TDL-A", "config/runs/quick_start.yaml", "standard_tdl"),
    ("TDL-B", "config/runs/quick_start.yaml", "standard_tdl"),
    ("TDL-C", "config/runs/quick_start.yaml", "standard_tdl"),
    ("TDL-D", "config/runs/quick_start.yaml", "standard_tdl"),
    ("TDL-E", "config/runs/quick_start.yaml", "standard_tdl"),
    # Deterministic two-path
    ("two_path", "config/runs/two_path.yaml", "manual_paths"),
    # Ray-tracing rooms (may need Sionna)
    ("simple_room_2d", "config/runs/simple_room_2d.yaml", "simple_room"),
    ("simple_room_3d", "config/runs/simple_room_3d.yaml", "simple_room"),
]

# ── algorithms ──────────────────────────────────────────────────────────
_ALGORITHMS = {
    "Threshold(0.30)": ThresholdLde(peak_ratio=0.30),
    "Chip(10dB)": ChipLeadingEdgeLde(threshold_db=10.0, tail_frac=0.25, min_run=3),
    "LeadingEdge(4σ)": LeadingEdgeLde(n_sigma=4.0, tail_frac=0.25, min_run=3),
    "MaxPeak": MaxPeakLde(),
    "SearchBack(0.30)": SearchBackLde(peak_ratio=0.30),
}



def run_environment(
    env_label: str,
    config_path: str,
    env_type: str,
    num_trials: int,
) -> dict:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))

    # Override for TDL model labels
    if env_type == "standard_tdl":
        config = copy.deepcopy(config)
        config["environment"]["model"] = env_label
        config["environment"]["num_trials"] = num_trials
    else:
        config = copy.deepcopy(config)
        config["environment"]["num_trials"] = num_trials

    # Each radio gets its own truth when sionna is involved;
    # for TDL we can share truth across radios.
    is_sionna = str(config.get("environment", {}).get("engine", "")) in {"sionna_box", "sionna_rect_room"}
    radios = build_radios_from_config(config)

    # Pre-generate truths
    if is_sionna:
        radio_truths: dict[str, list] = {}
        for radio in radios:
            rng = _rng(seed, _PROTO_SEEDS[radio.protocol])
            radio_truths[radio.protocol] = generate_truths(config, rng)
    else:
        base_rng = _rng(seed)
        shared_truths = generate_truths(config, base_rng)

    results: dict[str, dict] = {}

    for algo_name, algo in _ALGORITHMS.items():
        errors_by_protocol = empty_error_store()

        for trial_idx in range(num_trials):
            if is_sionna:
                for radio in radios:
                    truth = radio_truths[radio.protocol][trial_idx]
                    radio_cfg = config.get("radios", {}).get(radio.protocol, {})
                    impair_rng = _rng(seed, trial_idx * 1000 + 1)
                    observed = apply_timing_impairments(truth, config, impair_rng, radio_cfg=radio_cfg)
                    obs_rng = _rng(seed, trial_idx * 1000 + _PROTO_SEEDS[radio.protocol])
                    observation = radio.observe(observed, obs_rng)
                    estimate = algo.estimate(observation)
                    errors_by_protocol[observation.protocol].append(
                        range_error_m(estimate, observed)
                    )
            else:
                truth = shared_truths[trial_idx]
                for radio in radios:
                    radio_cfg = config.get("radios", {}).get(radio.protocol, {})
                    impair_rng = _rng(seed, trial_idx * 1000 + 1)
                    observed = apply_timing_impairments(truth, config, impair_rng, radio_cfg=radio_cfg)
                    obs_rng = _rng(seed, trial_idx * 1000 + _PROTO_SEEDS[radio.protocol])
                    observation = radio.observe(observed, obs_rng)
                    estimate = algo.estimate(observation)
                    errors_by_protocol[observation.protocol].append(
                        range_error_m(estimate, observed)
                    )

        summary = summarize_errors(errors_by_protocol)
        results[algo_name] = {proto: summary[proto] for proto in errors_by_protocol}

    return results


def print_results(env_label: str, results: dict):
    print(f"\n{'─' * 110}")
    print(f"  {env_label}")
    print(f"{'─' * 110}")
    header = (
        f"{'Algorithm':<20}"
        f"{'UWB RMSE':>10} {'UWB P90':>10}"
        f"{'WiFi RMSE':>10} {'WiFi P90':>10}"
        f"{'5G RMSE':>10} {'5G P90':>10}"
        f"{'Winner':>8}"
    )
    print(header)
    print(f"{'─' * 110}")

    for algo_name in _ALGORITHMS:
        s = results[algo_name]
        u = s.get("uwb", {})
        w = s.get("wifi", {})
        f = s.get("fiveg", {})

        rmse_list = [
            ("UWB", u.get("rmse_m", 999)),
            ("WiFi", w.get("rmse_m", 999)),
            ("5G", f.get("rmse_m", 999)),
        ]
        winner = min(rmse_list, key=lambda x: x[1])[0]

        print(
            f"{algo_name:<20}"
            f"{u.get('rmse_m', 0):10.3f} {u.get('p90_abs_m', 0):10.3f}"
            f"{w.get('rmse_m', 0):10.3f} {w.get('p90_abs_m', 0):10.3f}"
            f"{f.get('rmse_m', 0):10.3f} {f.get('p90_abs_m', 0):10.3f}"
            f"{winner:>8}"
        )

    print(f"{'─' * 110}")


def main():
    np.set_printoptions(precision=3, suppress=True)
    num_trials = 200

    print("=" * 110)
    print(f"  Monte Carlo Evaluation — {num_trials} trials per environment")
    print(f"  Algorithms: {', '.join(_ALGORITHMS)}")
    print(f"  Radios: UWB, WiFi, 5G NR")
    print(f"  Environments: {', '.join(e[0] for e in _ENVIRONMENTS)}")
    print("=" * 110)

    t_start = time.perf_counter()

    for env_label, config_path, env_type in _ENVIRONMENTS:
        t_env = time.perf_counter()
        try:
            results = run_environment(env_label, config_path, env_type, num_trials)
            print_results(env_label, results)
            elapsed = time.perf_counter() - t_env
            print(f"  ⏱  {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - t_env
            print(f"\n  ✗ {env_label} FAILED after {elapsed:.1f}s: {exc}")

    total = time.perf_counter() - t_start
    print(f"\n{'=' * 110}")
    print(f"  Total time: {total:.1f}s ({total / 60:.1f} min)")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
