"""Example 1 — Basic UWB ranging on a statistical TDL channel.

This is the simplest possible run: one radio, one scene, one algorithm.
No Sionna required. Runs in seconds.

Run:
    python3 examples/01_basic_uwb_tdl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the examples/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.radio_factory import create_radio
from utils.runner import run_single_trial
from utils.evaluator import summarize_errors
from environments import generate_truths
from algorithms import ThresholdLde

# 1. Build a UWB radio with default parameters
radio = create_radio("uwb")

# 2. Choose a scene: TDL-A (NLOS, 23 taps)
config: dict = {
    "seed": 42,
    "impairments": {},       # no impairments
    "timing": {},
    "environment": {
        "type": "standard_tdl",
        "model": "TDL-A",
        "num_trials": 200,
    },
}

# 3. Generate channel truths
import numpy as np

rng = np.random.default_rng(42)
truths = generate_truths(config, rng)

# 4. Choose an algorithm
algo = ThresholdLde(peak_ratio=0.18)

# 5. Run Monte Carlo
errors: list[float] = []
num_trials = min(200, len(truths))
for trial_idx in range(num_trials):
    _, err_m = run_single_trial(truths[trial_idx], radio, algo, config, 42, trial_idx)
    errors.append(err_m)

# 6. Report
arr = np.array(errors)
rmse = float(np.sqrt(np.mean(arr**2)))
p90 = float(np.percentile(np.abs(arr), 90))
print(f"UWB / TDL-A / ThresholdLDE: {num_trials} trials")
print(f"  RMSE = {rmse:.4f} m")
print(f"  P90  = {p90:.4f} m")
