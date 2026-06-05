from __future__ import annotations

import numpy as np

from algorithms.base_lde import BaseLde
from core.models import LIGHT_SPEED_MPS, RadioObservation, RangeEstimate


class SearchBackLde(BaseLde):
    """Finds the max peak, then walks backward to locate the leading edge.

    Uses a threshold relative to the global peak (like ThresholdLde) so the
    detector adapts to varying SNR levels.  Starting from the peak and
    walking backward avoids false-triggers on isolated pre-signal noise
    spikes that a left-to-right scan would hit.
    """

    name = "search_back"

    def __init__(self, peak_ratio: float = 0.18):
        self.peak_ratio = peak_ratio

    def estimate(self, observation: RadioObservation) -> RangeEstimate:
        envelope = np.abs(observation.cir_discrete)
        peak_idx = int(np.argmax(envelope))
        threshold = self.peak_ratio * float(envelope[peak_idx])

        if threshold <= 0:
            coarse_idx = peak_idx
        else:
            # Walk left from peak; stop when envelope drops below threshold.
            coarse_idx = 0
            for i in range(peak_idx, -1, -1):
                if envelope[i] < threshold:
                    coarse_idx = i + 1
                    break

        estimated_tof_s = self._refine_tof(observation, coarse_idx)
        return RangeEstimate(
            algorithm=self.name,
            protocol=observation.protocol,
            estimated_tof_s=estimated_tof_s,
            estimated_range_m=estimated_tof_s * LIGHT_SPEED_MPS,
            metadata={
                "sample_index": coarse_idx,
                "peak_index": peak_idx,
                "threshold": float(threshold),
            },
        )
