from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from core.models import ChannelTruth, LIGHT_SPEED_MPS, RadioObservation, RangeEstimate


def plot_cir_comparison(
    observations: list[RadioObservation],
    truth: ChannelTruth,
    output_path: str | Path,
    time_window_ns: float = 120.0,
    observed: bool = True,
    title_prefix: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for obs in observations:
        if observed:
            cir_discrete = obs.cir_observed_discrete
            cir_cont = obs.cir_observed_cont
        else:
            cir_discrete = obs.cir_clean_discrete
            cir_cont = obs.cir_clean_cont

        axes[0].plot(obs.t_discrete_s * 1e9, np.abs(cir_discrete), ".-", label=obs.protocol)
        axes[1].plot(obs.t_cont_s * 1e9, np.abs(cir_cont), "-", label=obs.protocol)

    for i, tau_s in enumerate(truth.tau_paths_s):
        if np.abs(truth.a_paths[i]) > 1e-12:
            style = dict(color="grey", linestyle="--", alpha=0.45, linewidth=0.7)
            axes[0].axvline(tau_s * 1e9, **style)
            axes[1].axvline(tau_s * 1e9, **style)

    for ax in axes:
        ax.axvline(truth.true_first_tau_s * 1e9, color="k", linestyle=":", linewidth=1.5, label="true first path")
        ax.set_xlim(0.0, time_window_ns)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Normalized |CIR|")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    prefix = title_prefix or ("Observed" if observed else "Clean (Real)")
    axes[0].set_title(f"{prefix} Discrete Radio Samples")
    axes[1].set_title(f"{prefix} Interpolated Analog View")
    fig.suptitle(f"{prefix} CIR Comparison: UWB vs WiFi vs 5G")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(
    errors_by_protocol: dict[str, Iterable[float]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for protocol, errors in errors_by_protocol.items():
        arr = np.sort(np.abs(np.asarray(list(errors), dtype=float)))
        if arr.size == 0:
            continue
        cdf = np.arange(1, arr.size + 1) / arr.size
        axes[0].plot(arr, cdf, label=protocol)
        axes[1].hist(arr, bins=24, alpha=0.45, label=protocol)

    axes[0].set_title("Ranging Error CDF")
    axes[0].set_xlabel("Error (m)")
    axes[0].set_ylabel("CDF")
    axes[1].set_title("Ranging Error Histogram")
    axes[1].set_xlabel("Error (m)")
    axes[1].set_ylabel("Count")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Ranging Result Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cir_with_paths(
    observation: RadioObservation,
    truth: ChannelTruth,
    estimate: RangeEstimate | None = None,
    output_path: str | Path | None = None,
    time_window_ns: float = 120.0,
    title: str = "",
) -> None:
    """CIR waveform overlaid with true multipath markers and detection result.

    Shows the discrete CIR samples as stems, the interpolated continuous CIR
    as a line, grey dashed lines at each true path delay, a black dotted line
    at the true first path, and a red dashed line at the algorithm's estimate
    (when provided).

    Useful for debugging: you can see at a glance whether the detector locked
    onto the correct path or a later arrival.
    """
    fig, (ax_disc, ax_cont) = plt.subplots(1, 2, figsize=(15, 5))

    for ax, t, cir, label, marker in [
        (ax_disc, observation.t_discrete_s * 1e9,
         np.abs(observation.cir_observed_discrete), "observed discrete", "."),
        (ax_cont, observation.t_cont_s * 1e9,
         np.abs(observation.cir_observed_cont), "observed continuous", "-"),
    ]:
        if "discrete" in label:
            ax.stem(t, cir, linefmt="C0-", markerfmt="C0.", basefmt="C0-", label=label)
            ax.plot(t, cir, "C0-", alpha=0.3, linewidth=0.5)
        else:
            ax.plot(t, cir, "C0-", alpha=0.8, linewidth=0.8, label=label)

        # True multipath markers — color-coded by reflection order
        for i in range(len(truth.tau_paths_s)):
            if np.abs(truth.a_paths[i]) <= 1e-12:
                continue
            tau_ns = truth.tau_paths_s[i] * 1e9
            order = int(truth.path_order[i]) if i < len(truth.path_order) else 1
            if not (0 <= tau_ns <= time_window_ns):
                continue
            order_colors = {0: "#2e8b57", 1: "#1a5276", 2: "#7f8c8d"}
            color = order_colors.get(order, "#bdc3c7")
            order_alphas = {0: 0.9, 1: 0.75, 2: 0.55}
            alpha = order_alphas.get(order, 0.4)
            order_lw = {0: 2.0, 1: 1.4, 2: 1.0}
            lw = order_lw.get(order, 0.6)
            ax.axvline(tau_ns, color=color, linestyle="--", alpha=alpha, linewidth=lw)

        # True first path
        first_tau_ns = truth.true_first_tau_s * 1e9
        ax.axvline(first_tau_ns, color="black", linestyle=":", linewidth=1.8,
                   label=f"true first path ({first_tau_ns:.2f} ns)")

        # Algorithm estimate
        if estimate is not None:
            est_tau_ns = estimate.estimated_tof_s * 1e9
            err_m = (estimate.estimated_tof_s - truth.true_first_tau_s) * LIGHT_SPEED_MPS
            ax.axvline(est_tau_ns, color="red", linestyle="--", linewidth=1.5,
                       label=f"estimate ({est_tau_ns:.2f} ns, err={err_m:.3f} m)")

        ax.set_xlim(0, time_window_ns)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Normalized |CIR|")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

    ax_disc.set_title("Discrete Sampling Grid")
    ax_cont.set_title("Interpolated (Continuous)")
    fig.suptitle(title or f"CIR Decomposition — {observation.protocol} | {truth.metadata.get('environment', '')}")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_frequency_response(
    observation: RadioObservation,
    output_path: str | Path | None = None,
    title: str = "",
) -> None:
    """Frequency-domain view: clean vs observed H(f), magnitude and phase.

    Shows how hardware impairments and observation noise distort the channel
    frequency response.  The top row displays |H(f)| (dB scale) and the
    bottom row shows phase (radians wrapped to [-π, π]).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    freq_mhz = observation.frequency_hz * 1e-6

    # Magnitude
    axes[0, 0].plot(freq_mhz, 20 * np.log10(np.abs(observation.h_clean) + 1e-15),
                    "C0-", alpha=0.8, linewidth=0.6, label="clean")
    axes[0, 0].set_title("|H(f)| — Clean")
    axes[0, 0].set_xlabel("Frequency (MHz)")
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(freq_mhz, 20 * np.log10(np.abs(observation.h_observed) + 1e-15),
                    "C1-", alpha=0.8, linewidth=0.6, label="observed")
    axes[0, 1].set_title("|H(f)| — Observed")
    axes[0, 1].set_xlabel("Frequency (MHz)")
    axes[0, 1].set_ylabel("Magnitude (dB)")
    axes[0, 1].grid(True, alpha=0.25)

    # Phase
    axes[1, 0].plot(freq_mhz, np.angle(observation.h_clean),
                    "C0-", alpha=0.8, linewidth=0.4, label="clean")
    axes[1, 0].set_title("Phase — Clean")
    axes[1, 0].set_xlabel("Frequency (MHz)")
    axes[1, 0].set_ylabel("Phase (rad)")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(freq_mhz, np.angle(observation.h_observed),
                    "C1-", alpha=0.8, linewidth=0.4, label="observed")
    axes[1, 1].set_title("Phase — Observed")
    axes[1, 1].set_xlabel("Frequency (MHz)")
    axes[1, 1].set_ylabel("Phase (rad)")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(title or f"Frequency Response — {observation.protocol}")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_power_delay_profile(
    a_paths: np.ndarray,
    tau_paths_s: np.ndarray,
    output_path: str | Path | None = None,
    title: str = "",
    label: str = "",
) -> None:
    """Power-delay profile: per-path power vs delay with stem plot.

    Shows the magnitude of each multipath component as a stem, ordered by
    delay.  Useful for inspecting the channel realization before it enters
    the radio observation pipeline.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    power_linear = np.abs(a_paths) ** 2
    power_db = 10 * np.log10(power_linear + 1e-20)
    tau_ns = tau_paths_s * 1e9

    ax.stem(tau_ns, power_db, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or f"Power-Delay Profile{f' — {label}' if label else ''}")

    # Mark LOS if first tap is much stronger
    if len(power_db) >= 2:
        peak_db = np.max(power_db)
        first_db = power_db[0]
        if first_db >= peak_db - 1.0:
            ax.annotate("LOS", (tau_ns[0], first_db),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=9, color="green", ha="center")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
