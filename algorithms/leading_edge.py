from __future__ import annotations

import numpy as np

from algorithms.base_lde import BaseLde
from core.models import LIGHT_SPEED_MPS, RadioObservation, RangeEstimate


class LeadingEdgeLde(BaseLde):
    """First sustained run above a noise-floor adaptive threshold.

    Operates on the discrete CIR at Nyquist-rate samples for robust
    detection, then refines to sub-sample resolution via the interpolated
    continuous CIR around the detected coarse sample.
    """

    name = "leading_edge"

    def __init__(self, n_sigma: float = 4.0, tail_frac: float = 0.25, min_run: int = 3):
        self.n_sigma = n_sigma
        self.tail_frac = tail_frac
        self.min_run = min_run

    def estimate(self, observation: RadioObservation) -> RangeEstimate:
        envelope = np.abs(observation.cir_discrete)
        noise_mean, noise_std = self._noise_stats(envelope, self.tail_frac)
        threshold = noise_mean + self.n_sigma * noise_std

        # Find first sustained run above threshold
        run_count = 0
        coarse_idx = 0
        found = False
        for i, val in enumerate(envelope):
            if val >= threshold:
                run_count += 1
                if run_count >= self.min_run:
                    coarse_idx = i - self.min_run + 1
                    found = True
                    break
            else:
                run_count = 0

        if not found:
            coarse_idx = int(np.argmax(envelope))

        # Refine to sub-sample using interpolated CIR
        estimated_tof_s = self._refine_tof(observation, coarse_idx)
        estimated_range_m = estimated_tof_s * LIGHT_SPEED_MPS

        return RangeEstimate(
            algorithm=self.name,
            protocol=observation.protocol,
            estimated_tof_s=estimated_tof_s,
            estimated_range_m=estimated_range_m,
            metadata={
                "sample_index": coarse_idx,
                "threshold": float(threshold),
                "noise_mean": float(noise_mean),
                "noise_std": float(noise_std),
            },
        )
