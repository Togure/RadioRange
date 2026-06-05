from __future__ import annotations

import numpy as np

from algorithms.base_lde import BaseLde
from core.models import LIGHT_SPEED_MPS, RadioObservation, RangeEstimate


class ChipLeadingEdgeLde(BaseLde):
    """Realistic UWB chip leading-edge detector.

    Mimics the behaviour of commercial UWB transceivers (e.g. Qorvo DW1000):
    1. Estimate noise floor from the CIR tail.
    2. Set detection threshold at `threshold_db` dB above the noise floor.
    3. Scan left-to-right for the first sample exceeding the threshold.
    4. Require a minimum run of consecutive above-threshold samples to
       reject isolated noise spikes.

    With typical UWB SNR (~20 dB) and default 10 dB threshold, the
    effective threshold lands at roughly 0.25–0.35 of the normalised
    CIR peak — consistent with the "≈ 0.3" rule-of-thumb reported by
    users of commercial UWB modules.
    """

    name = "chip_leading_edge"

    def __init__(
        self,
        threshold_db: float = 10.0,
        tail_frac: float = 0.25,
        min_run: int = 3,
    ):
        self.threshold_db = threshold_db
        self.tail_frac = tail_frac
        self.min_run = min_run

    def estimate(self, observation: RadioObservation) -> RangeEstimate:
        envelope = np.abs(observation.cir_discrete)
        noise_mean, noise_std = self._noise_stats(envelope, self.tail_frac)

        # Noise power: mean of squared envelope in the tail gives σ²
        # Threshold = noise_floor * 10^(dB/10)  →  linear multiplier
        noise_floor = noise_mean
        threshold_linear = noise_floor * (10.0 ** (self.threshold_db / 10.0))

        # Also enforce at least N-sigma above noise to handle very low noise floors
        sigma_threshold = noise_mean + 4.0 * noise_std
        threshold = max(threshold_linear, sigma_threshold)

        # Find first sustained run above threshold
        run_count = 0
        first_idx = 0
        found = False
        for i, val in enumerate(envelope):
            if val >= threshold:
                run_count += 1
                if run_count >= self.min_run:
                    first_idx = i - self.min_run + 1
                    found = True
                    break
            else:
                run_count = 0

        if not found:
            first_idx = int(np.argmax(envelope))

        estimated_tof_s = self._refine_tof(observation, first_idx)
        return RangeEstimate(
            algorithm=self.name,
            protocol=observation.protocol,
            estimated_tof_s=estimated_tof_s,
            estimated_range_m=estimated_tof_s * LIGHT_SPEED_MPS,
            metadata={
                "sample_index": first_idx,
                "threshold": float(threshold),
                "threshold_db": self.threshold_db,
                "noise_floor": float(noise_floor),
                "noise_std": float(noise_std),
            },
        )
