from __future__ import annotations

import numpy as np

from algorithms.base_lde import BaseLde
from core.models import LIGHT_SPEED_MPS, RadioObservation, RangeEstimate


class ThresholdLde(BaseLde):
    name = "threshold"

    def __init__(self, peak_ratio: float = 0.18):
        self.peak_ratio = peak_ratio

    def estimate(self, observation: RadioObservation) -> RangeEstimate:
        envelope = np.abs(observation.cir_discrete)
        threshold = self.peak_ratio * float(np.max(envelope))
        candidates = np.flatnonzero(envelope >= threshold)
        idx = int(candidates[0]) if len(candidates) else int(np.argmax(envelope))
        estimated_tof_s = self._refine_tof(observation, idx)
        return RangeEstimate(
            algorithm=self.name,
            protocol=observation.protocol,
            estimated_tof_s=estimated_tof_s,
            estimated_range_m=estimated_tof_s * LIGHT_SPEED_MPS,
            metadata={"sample_index": idx, "threshold": threshold},
        )
