from __future__ import annotations

import numpy as np

from algorithms.base_lde import BaseLde
from core.models import LIGHT_SPEED_MPS, RadioObservation, RangeEstimate


class MaxPeakLde(BaseLde):
    name = "max_peak"

    def estimate(self, observation: RadioObservation) -> RangeEstimate:
        idx = int(np.argmax(np.abs(observation.cir_discrete)))
        estimated_tof_s = self._refine_tof(observation, idx)
        return RangeEstimate(
            algorithm=self.name,
            protocol=observation.protocol,
            estimated_tof_s=estimated_tof_s,
            estimated_range_m=estimated_tof_s * LIGHT_SPEED_MPS,
            metadata={"sample_index": idx},
        )
