from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from core.models import ChannelTruth, LIGHT_SPEED_MPS, RangeEstimate


def range_error_m(estimate: RangeEstimate, truth: ChannelTruth) -> float:
    return (estimate.estimated_tof_s - truth.true_first_tau_s) * LIGHT_SPEED_MPS


def summarize_errors(errors_by_protocol: dict[str, Iterable[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for protocol, errors in errors_by_protocol.items():
        arr = np.asarray(list(errors), dtype=float)
        if arr.size == 0:
            continue
        summary[protocol] = {
            "bias_m": float(np.mean(arr)),
            "std_m": float(np.std(arr)),
            "rmse_m": float(np.sqrt(np.mean(arr**2))),
            "p90_abs_m": float(np.percentile(np.abs(arr), 90)),
        }
    return summary


def empty_error_store() -> defaultdict[str, list[float]]:
    return defaultdict(list)
