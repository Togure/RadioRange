"""Compare ranging accuracy across different wall materials.

Materials compared: itu_concrete (baseline), itu_brick, itu_metal
Fixed geometry: 12×10×4 m room, LOS ≈ 5 m, max_reflections=2
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np

from algorithms import LeadingEdgeLde, MaxPeakLde, SearchBackLde, ThresholdLde
from core.models import LIGHT_SPEED_MPS
from environments import generate_truths
from hardware.impairments import apply_timing_impairments
from utils.config import load_config
from utils.radio_factory import build_radios_from_config
from utils.runner import _PROTO_SEEDS, _rng
from utils.visualizer import plot_cir_with_paths

# ── Shared config ──────────────────────────────────────────────────────────
_BASE_CONFIG = {
    "seed": 42,
    "channel_engine": "auto",
    "timing": {"sync_bias_s": 0.0, "clock_bias_s": 0.0, "rtt_mode": False},
    "environment": {
        "type": "simple_room",
        "dimensions_m": [12.0, 10.0, 4.0],
        "tx_position_m": [5.0, 5.0, 1.5],
        "rx_position_m": [9.0, 8.0, 1.2],
        "max_reflections": 2,
        "wall_thickness_m": 0.1,
        "num_trials": 1,
    },
    "radios": {
        "uwb":   {"enabled": True, "carrier_frequency_hz": 7.9872e9, "bandwidth_hz": 499.2e6,  "num_bins": 512,  "interpolation_factor": 10, "window": "hamming", "adc_bits": 8, "observation_model": "snr", "snr_db": 35.0},
        "wifi":  {"enabled": True, "carrier_frequency_hz": 5.18e9,   "fft_size": 512,  "subcarrier_spacing_hz": 312500.0, "interpolation_factor": 10, "adc_bits": 10, "observation_model": "snr", "snr_db": 30.0},
        "fiveg": {"enabled": True, "carrier_frequency_hz": 4.8e9,    "fft_size": 4096, "subcarrier_spacing_hz": 30000.0,  "interpolation_factor": 10, "adc_bits": 12, "observation_model": "snr", "snr_db": 25.0},
    },
}

_MATERIALS = {
    "itu_concrete": {"epsilon": 5.31, "sigma": 0.033, "desc": "Concrete (ITU)"},
    "itu_brick":    {"epsilon": 3.75, "sigma": 0.038, "desc": "Brick (ITU)"},
    "itu_metal":    {"epsilon": 1.0,  "sigma": 1e7,   "desc": "Metal (conductor)"},
}

_ALGOS = {
    "Threshold(0.18)": ThresholdLde(peak_ratio=0.18),
    "LeadingEdge(4σ)": LeadingEdgeLde(n_sigma=4.0, tail_frac=0.25, min_run=3),
    "SearchBack(0.18)": SearchBackLde(peak_ratio=0.18),
    "MaxPeak": MaxPeakLde(),
}

NUM_TRIALS = 50


def _build_radios(config: dict) -> list:
    return build_radios_from_config(config)


def _make_config(material: str, engine: str) -> dict:
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg["environment"]["material"] = material
    cfg["environment"]["engine"] = engine
    return cfg


def _pre_generate(config: dict, num_trials: int, seed: int) -> list:
    """Image method only — shared truth across radios."""
    rng = _rng(seed)
    truths = generate_truths(copy.deepcopy(config), rng)
    while len(truths) < num_trials:
        rng = _rng(seed, len(truths) * 1000)
        truths.extend(generate_truths(copy.deepcopy(config), rng))
    return truths[:num_trials]


def _run_one_trial(config: dict, radios: list, truth, trial_idx: int,
                   seed: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for radio in radios:
        impair_rng = _rng(seed, trial_idx * 1000 + 1)
        observed = apply_timing_impairments(truth, config, impair_rng,
                                            radio_cfg=config["radios"][radio.protocol])
        obs_rng = _rng(seed, trial_idx * 1000 + _PROTO_SEEDS[radio.protocol])
        observation = radio.observe(observed, obs_rng)
        algo_errors: dict[str, float] = {}
        for algo_name, algo in _ALGOS.items():
            estimate = algo.estimate(observation)
            error_m = (estimate.estimated_tof_s - observed.true_first_tau_s) * LIGHT_SPEED_MPS
            algo_errors[algo_name] = error_m
        result[observation.protocol] = algo_errors
    return result


def _compute_stats(errors_m: list[float]) -> dict[str, float]:
    if not errors_m:
        return {"rmse": 999.0, "p90": 999.0}
    arr = np.array(errors_m, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "p90": float(np.percentile(np.abs(arr), 90)),
    }


# ── Fresnel reflection coefficient ─────────────────────────────────────────
def fresnel_gamma(epsilon: float, sigma: float, freq_hz: float) -> float:
    """Power reflection coefficient at normal incidence."""
    eps0 = 8.854187817e-12
    omega = 2.0 * np.pi * freq_hz
    eps_complex = epsilon - 1j * sigma / (omega * eps0)
    sqrt_eps = np.sqrt(eps_complex)
    r = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
    return float(np.abs(r) ** 2)


def _plot_cir_comparison(truths_by_material: dict, radios: list, config: dict,
                         output_dir: Path):
    """One figure per radio: overlay CIR from all materials."""
    import matplotlib.pyplot as plt

    for radio in radios:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        seed = config["seed"]

        for mat_name, mat_info in _MATERIALS.items():
            truth = truths_by_material[mat_name][0]
            impair_rng = _rng(seed, 1)
            observed = apply_timing_impairments(truth, config, impair_rng,
                                                radio_cfg=config["radios"][radio.protocol])
            obs_rng = _rng(seed, _PROTO_SEEDS[radio.protocol])
            obs = radio.observe(observed, obs_rng)

            label = mat_info["desc"]
            axes[0].plot(obs.t_discrete_s * 1e9, np.abs(obs.cir_observed_discrete),
                         ".-", alpha=0.7, markersize=3, label=label)
            axes[1].plot(obs.t_cont_s * 1e9, np.abs(obs.cir_observed_cont),
                         "-", alpha=0.7, label=label)

            # Mark true first path
            first_tau_ns = truth.true_first_tau_s * 1e9
            for ax in axes:
                ax.axvline(first_tau_ns, color="grey", linestyle=":", alpha=0.5, linewidth=0.8)

        for ax in axes:
            ax.set_xlim(0, 120)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Normalized |CIR|")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[0].set_title("Discrete Samples")
        axes[1].set_title("Interpolated Continuous")
        fig.suptitle(f"Material Comparison — {radio.protocol.upper()}")
        fig.tight_layout()
        fig.savefig(output_dir / f"cir_material_comparison_{radio.protocol}.png",
                    dpi=160, bbox_inches="tight")
        plt.close(fig)


def main():
    np.set_printoptions(precision=3, suppress=True)
    output_dir = PROJECT / "outputs" / "material_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = _BASE_CONFIG["seed"]

    print("=" * 80)
    print("  Wall Material Comparison — Ranging Accuracy")
    print(f"  Monte Carlo trials: {NUM_TRIALS}")
    print(f"  Room: 12×10×4 m, LOS ≈ 5 m")
    mat_list = ", ".join(f"{k} ({v['desc']})" for k, v in _MATERIALS.items())
    print(f"  Materials: {mat_list}")
    print("=" * 80)

    # ── Compute Fresnel Γ for each material ─────────────────────────────────
    print("\nFresnel power reflection coefficients |Γ|² at normal incidence:")
    print(f"{'Material':<20} {'ε':>6} {'σ':>10} {'@8GHz':>8} {'@5.2GHz':>8} {'@4.8GHz':>8}")
    print("-" * 60)
    for mat_name, mat_info in _MATERIALS.items():
        g8 = fresnel_gamma(mat_info["epsilon"], mat_info["sigma"], 8e9)
        g5 = fresnel_gamma(mat_info["epsilon"], mat_info["sigma"], 5.2e9)
        g4 = fresnel_gamma(mat_info["epsilon"], mat_info["sigma"], 4.8e9)
        print(f"{mat_name:<20} {mat_info['epsilon']:>6.2f} {mat_info['sigma']:>10.2g} {g8:>8.4f} {g5:>8.4f} {g4:>8.4f}")

    # ── Pre-generate truths (image method) ──────────────────────────────────
    print(f"\nGenerating truths for {NUM_TRIALS} trials per material...")
    truths_by_material: dict[str, list] = {}
    for mat_name in _MATERIALS:
        cfg = _make_config(mat_name, "image_method")
        truths_by_material[mat_name] = _pre_generate(cfg, NUM_TRIALS, seed)
        t0 = truths_by_material[mat_name][0]
        n = len(t0.a_paths)
        print(f"  {mat_name:<20}: {n} paths, LOS={t0.true_range_m:.3f}m")

    # ── Debug CIR overlay plots ─────────────────────────────────────────────
    print("\nGenerating CIR overlay plots...")
    cfg_img = _make_config("itu_concrete", "image_method")
    radios = _build_radios(cfg_img)
    _plot_cir_comparison(truths_by_material, radios, cfg_img, output_dir)
    print("  CIR overlay plots saved.")

    # ── Monte Carlo ─────────────────────────────────────────────────────────
    print(f"\nRunning {NUM_TRIALS} Monte Carlo trials per material...")
    errors: dict[str, dict[str, dict[str, list[float]]]] = {
        mat: {an: {"uwb": [], "wifi": [], "fiveg": []} for an in _ALGOS}
        for mat in _MATERIALS
    }

    for mat_name in _MATERIALS:
        cfg = _make_config(mat_name, "image_method")
        mat_radios = _build_radios(cfg)
        truths = truths_by_material[mat_name]

        for trial_idx in range(NUM_TRIALS):
            try:
                trial_result = _run_one_trial(cfg, mat_radios, truths[trial_idx],
                                              trial_idx, seed)
                for proto, algo_errors in trial_result.items():
                    for algo_name, err in algo_errors.items():
                        errors[mat_name][algo_name][proto].append(abs(err))
            except Exception:
                pass

    # ── Results table ───────────────────────────────────────────────────────
    lines: list[str] = []
    sep = "=" * 110
    lines.append(sep)
    lines.append("  Wall Material Comparison — Ranging Accuracy (Image Method)")
    lines.append(f"  Trials: {NUM_TRIALS} | Room: 12×10×4 m | LOS ≈ 5 m")
    lines.append(sep)
    header = (
        f"{'Material':<20} {'Algo':<18}"
        f"{'UWB RMSE':>10} {'UWB P90':>10}"
        f"{'WiFi RMSE':>10} {'WiFi P90':>10}"
        f"{'5G RMSE':>10} {'5G P90':>10}"
    )
    lines.append(header)
    lines.append("-" * 110)

    for mat_name in _MATERIALS:
        for algo_name in _ALGOS:
            e = errors[mat_name][algo_name]
            u = _compute_stats(e["uwb"])
            w = _compute_stats(e["wifi"])
            f = _compute_stats(e["fiveg"])
            lines.append(
                f"{_MATERIALS[mat_name]['desc']:<20} {algo_name:<18}"
                f"{u['rmse']:10.3f} {u['p90']:10.3f}"
                f"{w['rmse']:10.3f} {w['p90']:10.3f}"
                f"{f['rmse']:10.3f} {f['p90']:10.3f}"
            )
        lines.append("-" * 110)

    lines.append(f"\nPlots saved to {output_dir}/")

    # Print + save
    for line in lines:
        print(line)

    txt_path = output_dir / "accuracy_table.txt"
    txt_path.write_text("\n".join(lines))
    print(f"\nAccuracy table saved to {txt_path}")


if __name__ == "__main__":
    main()
