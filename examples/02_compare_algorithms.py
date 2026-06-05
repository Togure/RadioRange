"""Example 2 — Compare all 5 LDE algorithms on one scene.

Demonstrates the --mode compare-algos workflow from Python.

Run:
    python3 examples/02_compare_algorithms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.radio_factory import create_radio
from utils.runner import run_single_trial
from utils.evaluator import empty_error_store, summarize_errors
from environments import generate_truths
from algorithms import (
    ChipLeadingEdgeLde, LeadingEdgeLde, MaxPeakLde,
    SearchBackLde, ThresholdLde,
)

# 1. All 5 algorithms
algos: dict = {
    "MaxPeak": MaxPeakLde(),
    "Threshold(0.18)": ThresholdLde(peak_ratio=0.18),
    "LeadingEdge(4σ)": LeadingEdgeLde(n_sigma=4.0, tail_frac=0.25, min_run=3),
    "SearchBack(0.18)": SearchBackLde(peak_ratio=0.18),
    "ChipLDE(10dB)": ChipLeadingEdgeLde(threshold_db=10.0, tail_frac=0.25, min_run=3),
}

# 2. Radios: UWB + WiFi + 5G
radios = {
    "uwb": create_radio("uwb"),
    "wifi": create_radio("wifi"),
    "fiveg": create_radio("fiveg"),
}

# 3. TDL-A channel
config: dict = {
    "seed": 42,
    "impairments": {},
    "timing": {},
    "environment": {"type": "standard_tdl", "model": "TDL-A", "num_trials": 100},
}
rng = np.random.default_rng(42)
truths = generate_truths(config, rng)
num_trials = min(100, len(truths))

# 4. Run all algorithms
errors: dict[str, dict] = {an: empty_error_store() for an in algos}
for trial_idx in range(num_trials):
    for proto, radio in radios.items():
        truth = truths[trial_idx]
        for algo_name, algo in algos.items():
            _, err_m = run_single_trial(truth, radio, algo, config, 42, trial_idx)
            errors[algo_name][proto].append(err_m)

# 5. Print comparison table
print(f"{'Algorithm':<20} {'UWB RMSE':>10} {'WiFi RMSE':>10} {'5G RMSE':>10}")
print("-" * 55)
for algo_name in algos:
    s = summarize_errors(errors[algo_name])
    print(f"{algo_name:<20} "
          f"{s.get('uwb', {}).get('rmse_m', 0):10.3f} "
          f"{s.get('wifi', {}).get('rmse_m', 0):10.3f} "
          f"{s.get('fiveg', {}).get('rmse_m', 0):10.3f}")

# Best per protocol
for p in ["uwb", "wifi", "fiveg"]:
    best = min(algos, key=lambda a: summarize_errors(errors[a]).get(p, {}).get("rmse_m", 999))
    rmse = summarize_errors(errors[best]).get(p, {}).get("rmse_m", 0)
    print(f"  Best {p}: {best} ({rmse:.3f}m)")
