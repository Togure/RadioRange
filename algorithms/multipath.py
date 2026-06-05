"""Multipath component identification algorithms.

Each algorithm takes a RadioObservation and returns a MultipathResult
containing all detected propagation path components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.models import LIGHT_SPEED_MPS, MultipathResult, PathComponent, RadioObservation


class BaseMultipathDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, observation: RadioObservation) -> MultipathResult:
        raise NotImplementedError

    @staticmethod
    def _refine_tof(observation: RadioObservation, coarse_idx: int) -> float:
        """Sub-sample ToF refinement via continuous (interpolated) CIR."""
        factor = len(observation.t_cont_s) // len(observation.t_discrete_s)
        if factor < 2:
            return float(observation.t_discrete_s[coarse_idx])
        cont_start = coarse_idx * factor
        half_window = max(factor, 3)
        lo = max(0, cont_start - half_window)
        hi = min(len(observation.t_cont_s), cont_start + half_window + 1)
        if lo >= hi:
            return float(observation.t_discrete_s[coarse_idx])
        window = np.abs(observation.cir_observed_cont[lo:hi])
        peak_offset = int(np.argmax(window))

        # Quadratic interpolation around the peak for sub-grid resolution
        if 1 <= peak_offset < len(window) - 1:
            y_left = window[peak_offset - 1]
            y_peak = window[peak_offset]
            y_right = window[peak_offset + 1]
            denom = y_left - 2.0 * y_peak + y_right
            if abs(denom) > 1e-20:
                delta = 0.5 * (y_left - y_right) / denom
                delta = max(-0.5, min(0.5, delta))
                t_step = observation.t_cont_s[1] - observation.t_cont_s[0]
                return float(observation.t_cont_s[lo + peak_offset] + delta * t_step)

        return float(observation.t_cont_s[lo + peak_offset])

    @staticmethod
    def _noise_floor(envelope: np.ndarray, tail_frac: float = 0.3) -> float:
        n = len(envelope)
        tail_start = int(n * (1.0 - tail_frac))
        if tail_start >= n:
            tail_start = n // 2
        tail = envelope[tail_start:]
        if len(tail) < 4:
            tail = envelope[n // 2:]
        return float(np.mean(tail) + 3.0 * np.std(tail))


class PeakFinder(BaseMultipathDetector):
    """Detect multipath components via local peak finding with thresholding.

    Parameters
    ----------
    threshold_db : float
        Peaks below (max_peak - threshold_db) are discarded.  Default 20 dB.
    min_peak_spacing_ns : float
        Minimum time separation between detected peaks [ns].  Closer peaks are
        merged, keeping the stronger one.  Default 2.0 ns.
    """

    name = "PeakFinder"

    def __init__(self, threshold_db: float = 20.0, min_peak_spacing_ns: float = 2.0):
        self.threshold_db = threshold_db
        self.min_peak_spacing_ns = min_peak_spacing_ns

    def detect(self, observation: RadioObservation) -> MultipathResult:
        envelope = np.abs(observation.cir_observed_discrete)
        t_discrete_s = observation.t_discrete_s
        n = len(envelope)

        if n == 0:
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=-100.0,
            )

        max_val = float(np.max(envelope))
        if max_val <= 1e-20:
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=-100.0,
            )

        noise_floor = self._noise_floor(envelope)
        noise_floor_db = float(20.0 * np.log10(max(noise_floor, 1e-20)))
        abs_threshold = max_val * (10.0 ** (-self.threshold_db / 20.0))
        abs_threshold = max(abs_threshold, noise_floor)

        candidate_indices = self._find_peaks(envelope, abs_threshold)
        if not candidate_indices:
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=noise_floor_db,
            )

        merged = self._merge_close_peaks(candidate_indices, envelope, t_discrete_s)
        paths = self._build_paths(merged, observation, max_val)

        return MultipathResult(
            algorithm=self.name, protocol=observation.protocol,
            paths=paths, noise_floor_db=noise_floor_db,
            metadata={
                "threshold_db": self.threshold_db,
                "min_peak_spacing_ns": self.min_peak_spacing_ns,
                "abs_threshold": float(abs_threshold),
                "max_amplitude": float(max_val),
            },
        )

    @staticmethod
    def _find_peaks(envelope: np.ndarray, threshold: float) -> list[int]:
        n = len(envelope)
        candidates = []
        for i in range(1, n - 1):
            if envelope[i] <= threshold:
                continue
            if envelope[i] > envelope[i - 1] and envelope[i] >= envelope[i + 1]:
                candidates.append(i)
        return candidates

    def _merge_close_peaks(self, indices: list[int], envelope: np.ndarray,
                           t_discrete_s: np.ndarray) -> list[tuple]:
        min_spacing_s = self.min_peak_spacing_ns * 1e-9
        merged = []
        for idx in indices:
            t = t_discrete_s[idx]
            amp = envelope[idx]
            keep = True
            for j, (mj, mt, ma) in enumerate(merged):
                if abs(t - mt) < min_spacing_s:
                    keep = False
                    if amp > ma:
                        merged[j] = (idx, t, amp)
                    break
            if keep:
                merged.append((idx, t, amp))
        merged.sort(key=lambda x: x[1])
        return merged

    def _build_paths(self, merged: list[tuple], observation: RadioObservation,
                     max_val: float) -> list[PathComponent]:
        paths = []
        for idx, t_coarse, amp in merged:
            tof_s = self._refine_tof(observation, idx)
            range_m = tof_s * LIGHT_SPEED_MPS
            confidence = min(1.0, float(amp / max_val))
            paths.append(PathComponent(
                estimated_tof_s=float(tof_s),
                estimated_range_m=float(range_m),
                amplitude=float(amp),
                confidence=float(confidence),
                is_first_path=(len(paths) == 0),
            ))
        return paths


class CFARDetector(BaseMultipathDetector):
    """CA-CFAR (Cell-Averaging Constant False Alarm Rate) multipath detector.

    For each sample, estimates the local noise floor from neighbouring
    reference cells (excluding guard cells), then sets an adaptive threshold
    derived from the desired false-alarm probability.

    Parameters
    ----------
    guard_cells : int
        Number of guard cells on each side of the cell under test.  Default 3.
    reference_cells : int
        Number of reference cells on each side for noise estimation.  Default 10.
    pf : float
        Desired false-alarm probability.  Default 0.01 (1 %).
    min_peak_spacing_ns : float
        Minimum time separation between detected peaks [ns].  Default 2.0 ns.
    """

    name = "CFAR"

    def __init__(self, guard_cells: int = 3, reference_cells: int = 10,
                 pf: float = 0.01, min_peak_spacing_ns: float = 2.0):
        self.guard_cells = guard_cells
        self.reference_cells = reference_cells
        self.pf = pf
        self.min_peak_spacing_ns = min_peak_spacing_ns

    def detect(self, observation: RadioObservation) -> MultipathResult:
        envelope = np.abs(observation.cir_observed_discrete)
        t_discrete_s = observation.t_discrete_s
        n = len(envelope)

        if n == 0:
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=-100.0,
            )

        max_val = float(np.max(envelope))
        if max_val <= 1e-20:
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=-100.0,
            )

        N_total = 2 * self.reference_cells
        if N_total == 0:
            alpha = 1.0
        else:
            alpha = N_total * (self.pf ** (-1.0 / N_total) - 1.0)

        threshold = self._compute_cfar_threshold(envelope, alpha)

        candidate_indices = self._find_peaks_cfar(envelope, threshold)
        if not candidate_indices:
            noise_floor = self._noise_floor(envelope)
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=float(20.0 * np.log10(max(noise_floor, 1e-20))),
            )

        merged = self._merge_cfar_peaks(candidate_indices, envelope, t_discrete_s)
        paths = self._build_cfar_paths(merged, observation, max_val)

        noise_floor = self._noise_floor(envelope)
        return MultipathResult(
            algorithm=self.name, protocol=observation.protocol,
            paths=paths,
            noise_floor_db=float(20.0 * np.log10(max(noise_floor, 1e-20))),
            metadata={
                "guard_cells": self.guard_cells,
                "reference_cells": self.reference_cells,
                "pf": self.pf, "alpha": float(alpha),
                "max_amplitude": float(max_val),
            },
        )

    def _compute_cfar_threshold(self, envelope: np.ndarray, alpha: float) -> np.ndarray:
        n = len(envelope)
        G = self.guard_cells
        R = self.reference_cells
        threshold = np.zeros(n)

        for i in range(n):
            ref_vals = []
            # Lagging window
            lag_start = max(0, i - G - R)
            lag_end = max(0, i - G)
            if lag_start < lag_end:
                ref_vals.extend(envelope[lag_start:lag_end])
            # Leading window
            lead_start = min(n, i + G + 1)
            lead_end = min(n, i + G + 1 + R)
            if lead_start < lead_end:
                ref_vals.extend(envelope[lead_start:lead_end])

            if len(ref_vals) > 0:
                threshold[i] = float(np.mean(ref_vals)) * alpha
            else:
                threshold[i] = float(np.max(envelope))  # no detection at edges

        noise_floor = self._noise_floor(envelope)
        threshold = np.maximum(threshold, noise_floor)
        return threshold

    @staticmethod
    def _find_peaks_cfar(envelope: np.ndarray, threshold: np.ndarray) -> list[int]:
        n = len(envelope)
        candidates = []
        for i in range(1, n - 1):
            if envelope[i] <= threshold[i]:
                continue
            if envelope[i] > envelope[i - 1] and envelope[i] >= envelope[i + 1]:
                candidates.append(i)
        return candidates

    def _merge_cfar_peaks(self, indices: list[int], envelope: np.ndarray,
                          t_discrete_s: np.ndarray) -> list[tuple]:
        min_spacing_s = self.min_peak_spacing_ns * 1e-9
        merged = []
        for idx in indices:
            t = t_discrete_s[idx]
            amp = envelope[idx]
            keep = True
            for j, (mj, mt, ma) in enumerate(merged):
                if abs(t - mt) < min_spacing_s:
                    keep = False
                    if amp > ma:
                        merged[j] = (idx, t, amp)
                    break
            if keep:
                merged.append((idx, t, amp))
        merged.sort(key=lambda x: x[1])
        return merged

    def _build_cfar_paths(self, merged: list[tuple], observation: RadioObservation,
                          max_val: float) -> list[PathComponent]:
        paths = []
        for idx, t_coarse, amp in merged:
            tof_s = self._refine_tof(observation, idx)
            range_m = tof_s * LIGHT_SPEED_MPS
            confidence = min(1.0, float(amp / max_val))
            paths.append(PathComponent(
                estimated_tof_s=float(tof_s),
                estimated_range_m=float(range_m),
                amplitude=float(amp),
                confidence=float(confidence),
                is_first_path=(len(paths) == 0),
            ))
        return paths


class CLEANDetector(BaseMultipathDetector):
    """CLEAN iterative deconvolution multipath detector.

    Iteratively finds the strongest peak in the residual CIR, records it,
    then subtracts a scaled and shifted sinc-template before searching for
    the next peak.  Continues until the residual peak drops below a
    configurable threshold or the maximum iteration count is reached.

    Parameters
    ----------
    max_iterations : int
        Maximum number of paths to extract.  Default 20.
    residual_threshold_db : float
        Stop when residual peak falls below (max_initial_peak - threshold_db).
        Default 15 dB.
    min_peak_spacing_ns : float
        Minimum time separation between detected peaks [ns].  Default 2.0 ns.
    """

    name = "CLEAN"

    def __init__(self, max_iterations: int = 20, residual_threshold_db: float = 15.0,
                 min_peak_spacing_ns: float = 2.0):
        self.max_iterations = max_iterations
        self.residual_threshold_db = residual_threshold_db
        self.min_peak_spacing_ns = min_peak_spacing_ns

    def detect(self, observation: RadioObservation) -> MultipathResult:
        t_discrete_s = observation.t_discrete_s
        cir_complex = observation.cir_observed_discrete.astype(np.complex128, copy=True)
        envelope = np.abs(cir_complex)
        n = len(envelope)

        if n == 0:
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=-100.0,
            )

        initial_max = float(np.max(envelope))
        if initial_max <= 1e-20:
            return MultipathResult(
                algorithm=self.name, protocol=observation.protocol,
                paths=[], noise_floor_db=-100.0,
            )

        # Build sinc template from estimated bandwidth
        template = self._build_sinc_template(t_discrete_s, observation)
        half_tpl = len(template) // 2

        abs_threshold = initial_max * (10.0 ** (-self.residual_threshold_db / 20.0))
        noise_floor_val = self._noise_floor(envelope)
        abs_threshold = max(abs_threshold, noise_floor_val)

        detected_peaks: list[tuple[int, float, float]] = []  # (idx, tof_s, amp)

        for iteration in range(self.max_iterations):
            env = np.abs(cir_complex)
            peak_idx = int(np.argmax(env))
            peak_amp = float(env[peak_idx])

            if peak_amp < abs_threshold:
                break

            peak_val = complex(cir_complex[peak_idx])
            tof_s = self._refine_tof(observation, peak_idx)
            detected_peaks.append((peak_idx, tof_s, peak_amp))

            # Subtract scaled + shifted template
            shift = peak_idx - half_tpl
            lo_res = max(0, shift)
            hi_res = min(n, shift + len(template))
            lo_tpl = max(0, -shift)
            hi_tpl = len(template) - max(0, shift + len(template) - n)
            seg_len = hi_res - lo_res
            if seg_len > 0:
                cir_complex[lo_res:hi_res] -= peak_val * template[lo_tpl:hi_tpl]

        # Sort by TOF ascending, then merge close peaks
        detected_peaks.sort(key=lambda x: x[1])
        merged = self._merge_clean_peaks(detected_peaks)

        # Build PathComponents
        paths = []
        for idx, tof_s, amp in merged:
            range_m = tof_s * LIGHT_SPEED_MPS
            confidence = min(1.0, float(amp / initial_max))
            paths.append(PathComponent(
                estimated_tof_s=float(tof_s),
                estimated_range_m=float(range_m),
                amplitude=float(amp),
                confidence=float(confidence),
                is_first_path=(len(paths) == 0),
            ))

        noise_floor_db = float(20.0 * np.log10(max(noise_floor_val, 1e-20)))
        return MultipathResult(
            algorithm=self.name, protocol=observation.protocol,
            paths=paths, noise_floor_db=noise_floor_db,
            metadata={
                "max_iterations": self.max_iterations,
                "residual_threshold_db": self.residual_threshold_db,
                "iterations_used": len(detected_peaks),
                "abs_threshold": float(abs_threshold),
                "max_amplitude": float(initial_max),
            },
        )

    def _build_sinc_template(self, t_discrete_s: np.ndarray,
                             observation: RadioObservation) -> np.ndarray:
        """Build a sinc-pulse template from protocol bandwidth."""
        bw_hz = self._estimate_bandwidth(observation)
        dt = float(t_discrete_s[1] - t_discrete_s[0])
        # Main lobe + 3 side lobes each side ≈ 8/BW of support
        n_template = max(21, int(8.0 / (bw_hz * dt)))
        if n_template % 2 == 0:
            n_template += 1
        t_tpl = (np.arange(n_template) - n_template // 2) * dt
        template = np.sinc(2.0 * bw_hz * t_tpl)
        window = np.hamming(n_template)
        template = template * window
        template = template.astype(np.complex128)
        template /= np.max(np.abs(template))
        return template

    @staticmethod
    def _estimate_bandwidth(observation: RadioObservation) -> float:
        if observation.frequency_hz is not None and len(observation.frequency_hz) > 1:
            return float(observation.frequency_hz[-1] - observation.frequency_hz[0])
        return 500e6  # UWB default

    def _merge_clean_peaks(self, peaks: list[tuple[int, float, float]]
                           ) -> list[tuple[int, float, float]]:
        min_spacing_s = self.min_peak_spacing_ns * 1e-9
        merged = []
        for idx, tof_s, amp in peaks:
            keep = True
            for j, (mj, mt, ma) in enumerate(merged):
                if abs(tof_s - mt) < min_spacing_s:
                    keep = False
                    if amp > ma:
                        merged[j] = (idx, tof_s, amp)
                    break
            if keep:
                merged.append((idx, tof_s, amp))
        return merged
