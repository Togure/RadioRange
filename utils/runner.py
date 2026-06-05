"""Standard simulation pipeline — shared by all scripts.

Every script reinvents the same loop: apply impairments → observe →
estimate → record error.  This module provides the canonical
implementation so scripts only define *what* to run, not *how*.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.models import ChannelTruth, LIGHT_SPEED_MPS

# Protocol-specific seeds for per-frequency ray-tracing.
_PROTO_SEEDS = {"uwb": 101, "wifi": 202, "fiveg": 303}


def run_single_trial(
    truth: ChannelTruth,
    radio,
    algorithm,
    impairments_cfg: dict,
    seed: int,
    trial_idx: int = 0,
    *,
    radio_cfg: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """Run one ranging trial and return (protocol, error_m).

    This is the canonical pipeline shared by every script:

        ChannelTruth → apply impairments → radio.observe() → algo.estimate()
        → range_error_m(estimate, observed)

    Parameters
    ----------
    truth : ChannelTruth
    radio : BaseRadio instance
    algorithm : BaseLde (or any object with .estimate(observation))
    impairments_cfg : dict
        Full config dict or one containing ``"impairments"`` and ``"timing"`` keys.
    seed : int
        Base seed for reproducibility.
    trial_idx : int
        Trial index, mixed into seed for per-trial independence.
    radio_cfg : dict or None
        Per-protocol radio config for device-specific impairment magnitudes
        (sfo_ppm, cfo_hz, antenna_pcv_magnitude_m).  None → read from
        impairments_cfg globals.

    Returns
    -------
    (protocol, error_m) where error_m = estimated_range - true_range.
    """
    from hardware.impairments import apply_timing_impairments
    from utils.evaluator import range_error_m

    impair_rng = _rng(seed, trial_idx * 1000 + 1)
    observed = apply_timing_impairments(truth, impairments_cfg, impair_rng,
                                        radio_cfg=radio_cfg)

    obs_rng = _rng(seed, trial_idx * 1000 + _PROTO_SEEDS.get(radio.protocol, 0))
    observation = radio.observe(observed, obs_rng)

    estimate = algorithm.estimate(observation)
    error_m = range_error_m(estimate, observed)
    return observation.protocol, error_m


def run_monte_carlo(
    truths: dict[str, list[ChannelTruth]],
    radios: dict[str, Any],
    algorithm,
    impairments_cfg: dict,
    seed: int,
    num_trials: int | None = None,
) -> dict[str, list[float]]:
    """Run Monte Carlo ranging simulation across protocols.

    Parameters
    ----------
    truths : {protocol: [ChannelTruth, ...]}
    radios : {protocol: radio_instance}
    algorithm : LDE estimator
    impairments_cfg : dict with "impairments" key
    seed : int
    num_trials : int or None
        Number of trials.  None → min trials across all protocols.

    Returns
    -------
    {protocol: [error_m per trial]}
    """
    from utils.evaluator import empty_error_store

    protocols = sorted(truths.keys())
    max_avail = min(len(truths[p]) for p in protocols)
    n = num_trials if num_trials is not None else max_avail
    n = min(n, max_avail)

    errors = empty_error_store()
    for trial_idx in range(n):
        for proto in protocols:
            truth = truths[proto][trial_idx]
            _, err_m = run_single_trial(
                truth, radios[proto], algorithm, impairments_cfg, seed, trial_idx,
            )
            errors[proto].append(err_m)
    return errors


def run_multipath_detection(
    truth: ChannelTruth,
    radio,
    mp_algorithm,
    impairments_cfg: dict,
    seed: int,
) -> tuple[Any, dict]:
    """Run multipath detection on a single truth.

    Returns (MultipathResult, match_dict).
    """
    from hardware.impairments import apply_timing_impairments
    from algorithms.multipath import _match_detected_to_gt  # noqa — re-exported below

    rng = _rng(seed, 0)
    mp_truth = apply_timing_impairments(truth, impairments_cfg, rng)
    mp_obs_rng = _rng(seed, 1)
    mp_obs = radio.observe(mp_truth, mp_obs_rng)
    mp_result = mp_algorithm.detect(mp_obs)

    # Match against GT — use the same matching logic as rt_cache_interactive
    from core.models import LIGHT_SPEED_MPS

    proto_bw = _guess_bandwidth(radio)
    match = _match_detected_to_gt(
        mp_result.paths, truth.tau_paths_s, np.abs(truth.a_paths),
        proto_bw,
        gt_types=truth.path_type, gt_orders=truth.path_order,
    )
    return mp_result, match


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _rng(seed: int, suffix: int = 0) -> np.random.Generator:
    return np.random.default_rng(abs(seed + suffix) % (2 ** 31))


def _guess_bandwidth(radio) -> float:
    """Guess the effective bandwidth from a radio instance's config."""
    cfg = getattr(radio, "config", {})
    radio_cfg = cfg.get("radios", {}).get(radio.protocol, {})
    bw = radio_cfg.get("bandwidth_hz")
    if bw:
        return float(bw)
    scs = radio_cfg.get("subcarrier_spacing_hz", 312_500.0)
    fft = radio_cfg.get("fft_size", 512)
    return float(scs) * float(fft)


