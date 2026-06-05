from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


LIGHT_SPEED_MPS = 299_792_458.0


@dataclass(frozen=True)
class ChannelTruth:
    """Physical path truth before protocol-specific observation.

    Every array field with one entry per multipath component shares the
    same ordering (typically sorted by increasing delay).
    """

    a_paths: np.ndarray            # complex path gains (unitless)
    tau_paths_s: np.ndarray        # absolute one-way TOF per path [s]

    # Per-path classification.  None = unavailable.
    path_type: np.ndarray | None = None     # "LOS" / "reflection" / "diffraction" / "scattering"
    path_order: np.ndarray | None = None    # reflection order (0 = LOS, 1 = first bounce, …)
    polarization: np.ndarray | None = None  # "V" / "H" / "VH" / "HV" per path

    # Angles-of-arrival at the receiver (degrees).  None = unavailable.
    aoa_azimuth_deg: np.ndarray | None = None
    aoa_elevation_deg: np.ndarray | None = None

    # Angles-of-departure from the transmitter (degrees).  None = unavailable.
    aod_azimuth_deg: np.ndarray | None = None
    aod_elevation_deg: np.ndarray | None = None

    # Carrier frequency the channel was generated for [Hz].
    # None means the channel is treated as frequency-independent.
    carrier_frequency_hz: float | None = None

    true_range_m: float = 0.0
    los: bool = True
    sync_bias_s: float = 0.0     # transmitter–receiver sync offset [s]
    clock_bias_s: float = 0.0    # absolute clock offset at the receiver [s]
    rtt_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def true_first_tau_s(self) -> float:
        """Geometric LOS delay — always true_range_m / c, independent of RT paths."""
        if self.true_range_m <= 0:
            return float("nan")
        return self.true_range_m / LIGHT_SPEED_MPS


@dataclass(frozen=True)
class RadioObservation:
    """What a radio front-end exposes to first-path detection algorithms."""

    protocol: str
    t_discrete_s: np.ndarray
    t_cont_s: np.ndarray
    frequency_hz: np.ndarray
    h_clean: np.ndarray
    h_observed: np.ndarray
    cir_clean_discrete: np.ndarray
    cir_observed_discrete: np.ndarray
    cir_clean_cont: np.ndarray
    cir_observed_cont: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def h_frequency(self) -> np.ndarray:
        """Backward-compatible alias for device-observed CSI."""
        return self.h_observed

    @property
    def cir_discrete(self) -> np.ndarray:
        """Backward-compatible alias for device-observed discrete CIR."""
        return self.cir_observed_discrete

    @property
    def cir_cont(self) -> np.ndarray:
        """Backward-compatible alias for device-observed interpolated CIR."""
        return self.cir_observed_cont


@dataclass(frozen=True)
class RangeEstimate:
    algorithm: str
    protocol: str
    estimated_tof_s: float
    estimated_range_m: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathComponent:
    """Single detected multipath component."""
    estimated_tof_s: float
    estimated_range_m: float
    amplitude: float
    confidence: float = 1.0
    is_first_path: bool = False


@dataclass(frozen=True)
class MultipathResult:
    """Output of a multipath identification algorithm."""
    algorithm: str
    protocol: str
    paths: list[PathComponent]        # sorted by TOF ascending
    noise_floor_db: float = -100.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_range_estimate(self) -> RangeEstimate:
        """Degrade to first-path-only RangeEstimate using paths[0]."""
        if not self.paths:
            return RangeEstimate(
                algorithm=self.algorithm,
                protocol=self.protocol,
                estimated_tof_s=float("nan"),
                estimated_range_m=float("nan"),
                metadata={"error": "no_paths_detected"},
            )
        p0 = self.paths[0]
        return RangeEstimate(
            algorithm=self.algorithm,
            protocol=self.protocol,
            estimated_tof_s=p0.estimated_tof_s,
            estimated_range_m=p0.estimated_range_m,
            metadata=self.metadata,
        )
