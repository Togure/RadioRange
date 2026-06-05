#!/usr/bin/env python3
"""
Comprehensive Impairment Ablation + Degradation Sweeps
======================================================

Per `docs/requirements_impairment_ablation.md`:

  Part A — Ablation Table
    - 4 Sionna scenes (box_los, box_screen_nlos, etoile_los, munich_nlos)
    - 13 impairment configs (12 individual + "all")
    - Fixed algorithms: UWB → ChipLDE, WiFi → Threshold, 5G → Threshold
    - Δ RMSE (%) relative to BSL (multipath-only baseline)
    - Output: ablation_table.csv, ablation_pivot.csv

  Part B — Degradation Curves
    - box_los scene, 100 trials per data point
    - 3 sweep parameters:
      (a) I/Q gain ε: 1–30%  (8 steps)
      (b) SNR:         5–40 dB (10 steps)
      (c) ADC bits:    2–16    (9 steps)
    - 3 curves per panel (UWB / WiFi / 5G) with ±1σ error bands
    - BSL reference (horizontal dashed per protocol)
    - Output: CSV + combined 3-panel PNG figure

Key optimization: RT paths depend only on (scene, position, carrier freq) —
NOT on impairment parameters.  Cache once per (scene, trial, protocol), then
apply all parameter values to the same cached paths.

Usage:
  .venv/bin/python3 scripts/run_impairment_ablation_final.py --mode ablation
  .venv/bin/python3 scripts/run_impairment_ablation_final.py --mode sweeps
  .venv/bin/python3 scripts/run_impairment_ablation_final.py --mode all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.models import ChannelTruth
from environments.registry import generate_truths
from hardware.impairments import apply_timing_impairments
from algorithms import ChipLeadingEdgeLde, ThresholdLde
from utils.radio_factory import RADIO_DEFAULTS, DEFAULT_CARRIER_FREQ_HZ, create_radio

# ═══════════════════════════════════════════════════════════════════════════
# protocol → algorithm (FIXED per user specification)
# ═══════════════════════════════════════════════════════════════════════════

PROTOCOL_ALGO = {
    "uwb":   ("chip_lde",    lambda: ChipLeadingEdgeLde()),
    "wifi":  ("threshold",   lambda: ThresholdLde(peak_ratio=0.18)),
    "fiveg": ("threshold",   lambda: ThresholdLde(peak_ratio=0.18)),
}
PROTOCOLS = ["uwb", "wifi", "fiveg"]

# ═══════════════════════════════════════════════════════════════════════════
# base radio configs — derived from shared RADIO_DEFAULTS + experiment-specific
# explicit-impairment overrides (observation_model="explicit" for BSL baseline)
# ═══════════════════════════════════════════════════════════════════════════

def _make_base_radio_dict(proto: str) -> dict:
    """Build baseline radio config from RADIO_DEFAULTS + ablation overrides."""
    base = dict(RADIO_DEFAULTS[proto])
    base["enabled"] = True
    base["carrier_frequency_hz"] = DEFAULT_CARRIER_FREQ_HZ[proto]
    base["observation_model"] = "explicit"

    _EXTRA: dict[str, dict] = {
        "uwb": {
            "antenna_pcv_magnitude_m": 0.005,
            "iq_gain_imbalance": 0.03,
            "iq_phase_imbalance_deg": 3.0,
            "explicit_impairments": {
                "csi_noise_snr_db": 60, "csi_amplitude_std": 0.0,
                "csi_phase_std_rad": 0.0, "common_phase_std_rad": 0.0,
                "random_sampling_phase_std_s": 0.0, "sfo_residual_ppm": 0.0,
                "sfo_reference_delay_s": 100e-9,
            },
        },
        "wifi": {
            "antenna_pcv_magnitude_m": 0.003,
            "iq_gain_imbalance": 0.015,
            "iq_phase_imbalance_deg": 1.5,
            "explicit_impairments": {
                "csi_noise_snr_db": 60, "csi_amplitude_std": 0.0,
                "csi_phase_std_rad": 0.0, "common_phase_std_rad": 0.0,
                "random_sampling_phase_std_s": 0.0, "sfo_residual_ppm": 0.0,
                "sfo_reference_delay_s": 100e-9,
                "dc_null": False, "active_subcarrier_fraction": 1.0,
                "channel_estimation_mode": "1d", "pilot_spacing_subcarriers": 1,
            },
        },
        "fiveg": {
            "antenna_pcv_magnitude_m": 0.003,
            "iq_gain_imbalance": 0.01,
            "iq_phase_imbalance_deg": 1.0,
            "explicit_impairments": {
                "csi_noise_snr_db": 60, "csi_amplitude_std": 0.0,
                "csi_phase_std_rad": 0.0, "common_phase_std_rad": 0.0,
                "random_sampling_phase_std_s": 0.0, "sfo_residual_ppm": 0.0,
                "sfo_reference_delay_s": 100e-9,
                "dc_null": False, "active_subcarrier_fraction": 1.0,
                "channel_estimation_mode": "1d", "pilot_spacing_subcarriers": 1,
                "num_ofdm_symbols": 14, "dmrs_symbol_indices": [2, 6, 10],
                "dmrs_freq_spacing": 6, "dmrs_residual_cfo_hz": 0.0,
                "dmrs_cpe_std_rad": 0.0,
            },
        },
    }
    extra = _EXTRA[proto]
    base["antenna_pcv_magnitude_m"] = extra["antenna_pcv_magnitude_m"]
    base["iq_gain_imbalance"] = extra["iq_gain_imbalance"]
    base["iq_phase_imbalance_deg"] = extra["iq_phase_imbalance_deg"]
    base["explicit_impairments"] = dict(extra["explicit_impairments"])
    return base


_BASE_RADIO: dict[str, dict] = {p: _make_base_radio_dict(p) for p in PROTOCOLS}

# ═══════════════════════════════════════════════════════════════════════════
# scene definitions (4 scenes from requirements)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SceneConfig:
    name: str
    env_type: str
    scene_name: str
    category: str
    tx_fixed: tuple[float, float, float]
    rx_region: dict
    room_dims: tuple[float, float, float] | None
    engine: str
    max_reflections: int
    description: str


SCENE_CONFIGS: list[SceneConfig] = [
    SceneConfig(
        name="box_los", env_type="sionna_builtin_scene", scene_name="box",
        category="simple_los", tx_fixed=(-2.0, 0.0, 1.5),
        rx_region={"x": (0.0, 4.0), "y": (-2.0, 2.0), "z": 1.5},
        room_dims=None, engine="sionna_rt", max_reflections=2,
        description="Empty 10m box, all positions LOS",
    ),
    SceneConfig(
        name="box_screen_nlos", env_type="sionna_builtin_scene",
        scene_name="box_one_screen", category="simple_nlos",
        tx_fixed=(-3.5, 0.0, 1.5),
        rx_region={"x": (1.0, 4.0), "y": (-2.5, 2.5), "z": 1.5},
        room_dims=None, engine="sionna_rt", max_reflections=3,
        description="Box with metal screen — TX/RX opposite sides, all NLOS",
    ),
    SceneConfig(
        name="etoile_los", env_type="sionna_builtin_scene", scene_name="etoile",
        category="complex_los", tx_fixed=(0.0, 0.0, 1.5),
        rx_region={"x": (-20.0, 20.0), "y": (-20.0, 20.0), "z": 1.5},
        room_dims=None, engine="sionna_rt", max_reflections=3,
        description="Outdoor plaza with buildings — mostly LOS, rich multipath",
    ),
    SceneConfig(
        name="munich_nlos", env_type="sionna_builtin_scene", scene_name="munich",
        category="complex_nlos", tx_fixed=(-30.0, 40.0, 1.5),
        rx_region={"x": (-50.0, -10.0), "y": (10.0, 50.0), "z": 1.5},
        room_dims=None, engine="sionna_rt", max_reflections=3,
        description="Urban canyon — significant NLOS, complex multipath",
    ),
]

# ═══════════════════════════════════════════════════════════════════════════
# impairment configs (13 — same as run_impairment_ablation_multi.py)
# ═══════════════════════════════════════════════════════════════════════════

_IMPAIRMENT_LABELS = [
    "none", "antenna_pcv", "sfo", "cfo", "adc_timing", "adc_quant",
    "agc", "iq_imbalance", "csi_noise", "common_phase", "channel_est",
    "dmrs_2d", "all",
]


def _base_imp() -> dict[str, bool]:
    return {k: False for k in [
        "enable_antenna_offset", "enable_sfo", "enable_cfo",
        "enable_adc_phase_offset", "enable_adc_quantization",
        "enable_agc", "agc_clip_enable", "enable_iq_imbalance",
    ]}


def _empty_ro() -> dict[str, dict]:
    return {p: {} for p in PROTOCOLS}


def _build_ablation_configs() -> list[tuple[str, dict, dict]]:
    cfgs: list[tuple[str, dict, dict]] = []

    # 0. none
    cfgs.append(("none", _base_imp(), _empty_ro()))

    # 1. antenna_pcv
    imp = _base_imp(); imp["enable_antenna_offset"] = True
    cfgs.append(("antenna_pcv", imp, _empty_ro()))

    # 2. sfo
    imp = _base_imp(); imp["enable_sfo"] = True
    cfgs.append(("sfo", imp, _empty_ro()))

    # 3. cfo
    imp = _base_imp(); imp["enable_cfo"] = True
    cfgs.append(("cfo", imp, _empty_ro()))

    # 4. adc_timing
    imp = _base_imp(); imp["enable_adc_phase_offset"] = True
    cfgs.append(("adc_timing", imp, _empty_ro()))

    # 5. adc_quant
    imp = _base_imp(); imp["enable_adc_quantization"] = True
    cfgs.append(("adc_quant", imp, _empty_ro()))

    # 6. agc
    imp = _base_imp(); imp["enable_agc"] = True; imp["agc_clip_enable"] = True
    cfgs.append(("agc", imp, _empty_ro()))

    # 7. iq_imbalance
    imp = _base_imp(); imp["enable_iq_imbalance"] = True
    cfgs.append(("iq_imbalance", imp, _empty_ro()))

    # 8. csi_noise
    imp = _base_imp()
    ro = {
        p: {"explicit_impairments": {
            "csi_noise_snr_db": {"uwb": 30, "wifi": 30, "fiveg": 25}[p],
            "csi_amplitude_std": {"uwb": 0.04, "wifi": 0.025, "fiveg": 0.03}[p],
            "csi_phase_std_rad": {"uwb": 0.08, "wifi": 0.04, "fiveg": 0.05}[p],
        }} for p in PROTOCOLS
    }
    cfgs.append(("csi_noise", imp, ro))

    # 9. common_phase
    imp = _base_imp()
    ro = {
        p: {"explicit_impairments": {
            "common_phase_std_rad": {"uwb": 0.04, "wifi": 0.025, "fiveg": 0.03}[p],
            "sfo_residual_ppm": {"uwb": 2, "wifi": 1, "fiveg": 1}[p],
            "random_sampling_phase_std_s": {"uwb": 0.35e-9, "wifi": 0.12e-9, "fiveg": 0.18e-9}[p],
        }} for p in PROTOCOLS
    }
    cfgs.append(("common_phase", imp, ro))

    # 10. channel_est
    imp = _base_imp()
    ro = {
        "uwb": {"explicit_impairments": {}},
        "wifi": {"explicit_impairments": {
            "channel_estimation_mode": "1d", "pilot_spacing_subcarriers": 4,
            "dc_null": True, "active_subcarrier_fraction": 0.82,
        }},
        "fiveg": {"explicit_impairments": {
            "channel_estimation_mode": "1d", "pilot_spacing_subcarriers": 6,
            "dc_null": True, "active_subcarrier_fraction": 0.78,
        }},
    }
    cfgs.append(("channel_est", imp, ro))

    # 11. dmrs_2d
    imp = _base_imp()
    ro = {
        "uwb": {"explicit_impairments": {}},
        "wifi": {"explicit_impairments": {}},
        "fiveg": {"explicit_impairments": {
            "channel_estimation_mode": "2d", "dc_null": True,
            "active_subcarrier_fraction": 0.78,
            "num_ofdm_symbols": 14, "dmrs_symbol_indices": [2, 6, 10],
            "dmrs_freq_spacing": 6, "dmrs_residual_cfo_hz": 50,
            "dmrs_cpe_std_rad": 0.02,
        }},
    }
    cfgs.append(("dmrs_2d", imp, ro))

    # 12. all
    imp = {k: True for k in [
        "enable_antenna_offset", "enable_sfo", "enable_cfo",
        "enable_adc_phase_offset", "enable_adc_quantization",
        "enable_agc", "enable_iq_imbalance",
    ]}
    imp["agc_clip_enable"] = True
    ro = {
        "uwb": {"explicit_impairments": {
            "csi_noise_snr_db": 30, "csi_amplitude_std": 0.04,
            "csi_phase_std_rad": 0.08, "common_phase_std_rad": 0.04,
            "sfo_residual_ppm": 2, "random_sampling_phase_std_s": 0.35e-9,
        }},
        "wifi": {"explicit_impairments": {
            "csi_noise_snr_db": 30, "csi_amplitude_std": 0.025,
            "csi_phase_std_rad": 0.04, "common_phase_std_rad": 0.025,
            "sfo_residual_ppm": 1, "random_sampling_phase_std_s": 0.12e-9,
            "channel_estimation_mode": "1d", "pilot_spacing_subcarriers": 4,
            "dc_null": True, "active_subcarrier_fraction": 0.82,
        }},
        "fiveg": {"explicit_impairments": {
            "csi_noise_snr_db": 25, "csi_amplitude_std": 0.03,
            "csi_phase_std_rad": 0.05, "common_phase_std_rad": 0.03,
            "sfo_residual_ppm": 1, "random_sampling_phase_std_s": 0.18e-9,
            "channel_estimation_mode": "2d", "pilot_spacing_subcarriers": 6,
            "dc_null": True, "active_subcarrier_fraction": 0.78,
            "num_ofdm_symbols": 14, "dmrs_symbol_indices": [2, 6, 10],
            "dmrs_freq_spacing": 6, "dmrs_residual_cfo_hz": 50,
            "dmrs_cpe_std_rad": 0.02,
        }},
    }
    cfgs.append(("all", imp, ro))

    return cfgs


# ═══════════════════════════════════════════════════════════════════════════
# config builder helpers
# ═══════════════════════════════════════════════════════════════════════════

def _deep_radio(proto: str, overrides: dict | None = None) -> dict:
    import copy
    r = copy.deepcopy(_BASE_RADIO[proto])
    if overrides:
        if "explicit_impairments" in overrides:
            r["explicit_impairments"].update(overrides.pop("explicit_impairments"))
        r.update(overrides)
    return r


def _make_radio(proto: str, cfg: dict):
    """Create a radio via the shared factory, using cfg's radio section as overrides."""
    radio_section = cfg.get("radios", {}).get(proto, {})
    overrides = {k: v for k, v in radio_section.items()
                 if k not in ("enabled", "carrier_frequency_hz")}
    cfreq = radio_section.get("carrier_frequency_hz", None)
    return create_radio(proto, carrier_frequency_hz=cfreq,
                        overrides=overrides,
                        impairments=cfg.get("impairments", {}))


def build_config(
    scene: SceneConfig,
    tx_pos: tuple,
    rx_pos: tuple,
    protocol: str,
    impairments: dict,
    radio_overrides: dict | None = None,
) -> dict:
    env_cfg = {
        "type": scene.env_type,
        "tx_position_m": list(tx_pos),
        "rx_position_m": list(rx_pos),
        "max_reflections": scene.max_reflections,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": False,
        "num_trials": 1,
    }
    if scene.env_type == "sionna_builtin_scene":
        env_cfg["scene_name"] = scene.scene_name
    else:
        env_cfg["dimensions_m"] = list(scene.room_dims)
        env_cfg["engine"] = scene.engine or "sionna_rect_room"
        env_cfg["materials"] = {
            "floor": "itu_concrete", "ceiling": "itu_ceiling_board",
            "wall_x_min": "itu_concrete", "wall_x_max": "itu_concrete",
            "wall_y_min": "itu_concrete", "wall_y_max": "itu_concrete",
        }
        env_cfg["wall_thickness_m"] = 0.15

    radio_cfg = _deep_radio(protocol, radio_overrides)
    return {
        "seed": 42, "timing": {},
        "impairments": impairments,
        "environment": env_cfg,
        "radios": {protocol: radio_cfg},
    }


# ═══════════════════════════════════════════════════════════════════════════
# position sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_positions(
    scene: SceneConfig, n_trials: int, seed: int = 42,
) -> list[tuple[tuple, tuple]]:
    rng = np.random.default_rng(seed + hash(scene.name) % 10000)
    region = scene.rx_region
    pairs = []
    for _ in range(n_trials):
        rx_x = float(rng.uniform(*region["x"]))
        rx_y = float(rng.uniform(*region["y"]))
        rx_z = region.get("z", 1.5)
        pairs.append((scene.tx_fixed, (rx_x, rx_y, rx_z)))
    return pairs


from utils.runner import _rng

# ═══════════════════════════════════════════════════════════════════════════
# radio class lookup
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# data record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrialRecord:
    scene: str
    category: str
    trial: int
    true_range_m: float
    protocol: str
    algorithm: str
    impairment: str
    sweep_param: str          # "" for ablation, "iq_gain"/"snr"/"adc_bits" for sweeps
    sweep_value: float        # 0 for ablation, parameter value for sweeps
    estimated_range_m: float
    error_m: float


# ═══════════════════════════════════════════════════════════════════════════
# PART A — Ablation Study
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation(
    scenes: list[SceneConfig],
    n_trials: int,
    seed: int,
    out_dir: Path,
) -> list[TrialRecord]:
    """Run 13-config ablation across all scenes."""
    impairment_configs = _build_ablation_configs()
    n_imp = len(impairment_configs)
    all_records: list[TrialRecord] = []
    t_total_start = time.time()

    for scene_idx, scene in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"ABLATION Scene {scene_idx+1}/{len(scenes)}: {scene.name} "
              f"({scene.category})")
        print(f"  {scene.description}")
        print(f"  TX: {scene.tx_fixed}  |  RX region: {scene.rx_region}")
        print(f"{'='*60}")

        positions = sample_positions(scene, n_trials, seed=seed)
        dists = [np.linalg.norm(np.array(rx) - np.array(tx))
                 for tx, rx in positions]
        print(f"  {len(positions)} positions, dist range: "
              f"[{np.min(dists):.1f}, {np.max(dists):.1f}] m")

        t0 = time.time()
        scene_records: list[TrialRecord] = []

        for trial_idx, (tx, rx) in enumerate(positions):
            true_range_m = float(np.linalg.norm(np.array(rx) - np.array(tx)))
            trial_seed = seed + scene_idx * 10000 + trial_idx
            algo_instances = {
                p: PROTOCOL_ALGO[p][1]() for p in PROTOCOLS
            }

            for proto in PROTOCOLS:
                algo_name, _ = PROTOCOL_ALGO[proto]

                # RT once per protocol per trial
                cfg_rt = build_config(scene, tx, rx, proto, {}, {})
                freq = cfg_rt["radios"][proto]["carrier_frequency_hz"]
                rt_seed = trial_seed * 100 + hash(proto) % 100

                try:
                    truths = generate_truths(
                        cfg_rt, _rng(rt_seed), carrier_frequency_hz=freq,
                    )
                    truth = truths[0]
                except Exception:
                    continue

                for imp_label, imp_dict, radio_ov in impairment_configs:
                    cfg = build_config(
                        scene, tx, rx, proto, imp_dict,
                        radio_overrides=radio_ov.get(proto, {}),
                    )
                    radio = _make_radio(proto, cfg)

                    impair_rng = _rng(trial_seed * 10 + hash(proto) % 100 + 1)
                    radio_p = {
                        "sfo_ppm": _BASE_RADIO[proto].get("sfo_ppm", 20.0),
                        "cfo_hz": _BASE_RADIO[proto].get("cfo_hz", 500.0),
                    }
                    impaired = apply_timing_impairments(
                        truth, cfg, impair_rng, radio_cfg=radio_p,
                    )
                    obs_rng = _rng(trial_seed * 10 + hash(proto) % 100 + 2)
                    observation = radio.observe(impaired, obs_rng)

                    algo = algo_instances[proto]
                    estimate = algo.estimate(observation)
                    error_m = float(
                        estimate.estimated_range_m - impaired.true_range_m
                    )
                    if not np.isfinite(error_m):
                        error_m = float("nan")

                    scene_records.append(TrialRecord(
                        scene=scene.name, category=scene.category,
                        trial=trial_idx, true_range_m=true_range_m,
                        protocol=proto, algorithm=algo_name,
                        impairment=imp_label,
                        sweep_param="", sweep_value=0.0,
                        estimated_range_m=float(estimate.estimated_range_m),
                        error_m=error_m,
                    ))

            if (trial_idx + 1) % max(1, n_trials // 4) == 0 or trial_idx == 0:
                n_done = trial_idx + 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_trials - n_done) / rate if rate > 0 else 0
                print(f"    [{n_done:3d}/{n_trials}] "
                      f"recs={len(scene_records)}  "
                      f"rate={rate:.2f}/s  ETA={eta/60:.0f}min", flush=True)

        elapsed_scene = time.time() - t0
        print(f"  → {len(scene_records)} records in {elapsed_scene:.1f}s "
              f"({elapsed_scene/n_trials:.1f}s/trial)")

        # Save per-scene CSV
        scene_dir = out_dir / "ablation" / "per_scene" / scene.name
        scene_dir.mkdir(parents=True, exist_ok=True)
        _save_records_csv(scene_records, scene_dir / "per_trial_results.csv")
        _save_scene_summary(scene_records, scene_dir / "summary.csv")
        all_records.extend(scene_records)

    t_total = time.time() - t_total_start
    print(f"\nABLATION total: {t_total/60:.1f} min, {len(all_records)} records")

    # Build & save master tables
    _build_ablation_master_table(all_records, out_dir / "ablation")
    return all_records


def _build_ablation_master_table(
    records: list[TrialRecord], out_dir: Path,
) -> None:
    """Build comprehensive master table and pivot table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by (scene, protocol, impairment)
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in records:
        if np.isfinite(r.error_m):
            groups[(r.scene, r.protocol, r.impairment)].append(r.error_m)

    # Compute RMSE per group
    rmse_map: dict[tuple[str, str, str], float] = {}
    for (scene, proto, imp), errs in groups.items():
        arr = np.array(errs, dtype=float)
        rmse_map[(scene, proto, imp)] = float(np.sqrt(np.mean(arr ** 2)))

    # BSL lookup
    bsl_rmse: dict[tuple[str, str], float] = {}
    for (scene, proto, imp), rmse in rmse_map.items():
        if imp == "none":
            bsl_rmse[(scene, proto)] = rmse

    # Flat master table
    master_path = out_dir / "ablation_table.csv"
    with open(master_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scene", "category", "protocol", "algorithm", "impairment",
            "bsl_rmse_m", "rmse_m", "delta_m", "delta_pct", "n_trials",
        ])
        for r in records:
            if not np.isfinite(r.error_m):
                continue
            key = (r.scene, r.protocol, r.impairment)
            rmse = rmse_map.get(key)
            if rmse is None:
                continue
            bsl = bsl_rmse.get((r.scene, r.protocol), 0.0)
            delta = rmse - bsl
            pct = (delta / bsl * 100.0) if bsl > 0.001 else 0.0
            w.writerow([
                r.scene, r.category, r.protocol, r.algorithm, r.impairment,
                f"{bsl:.4f}", f"{rmse:.4f}", f"{delta:.4f}", f"{pct:.1f}",
                len(groups.get(key, [])),
            ])
    print(f"  Saved: {master_path.name}")

    # Pivot table: scene × protocol rows, impairment columns (Δ%)
    pivot_path = out_dir / "ablation_pivot.csv"
    with open(pivot_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Scene", "Category", "Protocol", "Algorithm",
                  "BSL_RMSE_m"] + _IMPAIRMENT_LABELS
        w.writerow(header)
        for scene_cfg in SCENE_CONFIGS:
            scene_name = scene_cfg.name
            for proto in PROTOCOLS:
                algo_name, _ = PROTOCOL_ALGO[proto]
                bsl = bsl_rmse.get((scene_name, proto), 0.0)
                row = [scene_name, scene_cfg.category, proto, algo_name,
                       f"{bsl:.4f}"]
                for imp in _IMPAIRMENT_LABELS:
                    rmse = rmse_map.get((scene_name, proto, imp))
                    if rmse is not None and bsl > 0.001:
                        pct = (rmse - bsl) / bsl * 100.0
                        row.append(f"{pct:+.1f}%")
                    elif rmse is not None:
                        row.append(f"{rmse:.4f}")
                    else:
                        row.append("N/A")
                w.writerow(row)
    print(f"  Saved: {pivot_path.name}")

    # Print console summary
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY — Δ RMSE (%) per Scene × Protocol")
    print("=" * 100)
    header_printed = False
    for scene_cfg in SCENE_CONFIGS:
        scene_name = scene_cfg.name
        if not header_printed:
            print(f"{'Scene':<22} {'Proto':<8}", end="")
            for imp in _IMPAIRMENT_LABELS:
                print(f"{imp:>11}", end="")
            print("\n" + "-" * 160)
            header_printed = True

        for proto in PROTOCOLS:
            algo_name, _ = PROTOCOL_ALGO[proto]
            bsl = bsl_rmse.get((scene_name, proto), 0.0)
            print(f"{scene_name:<22} {proto:<8}", end="")
            for imp in _IMPAIRMENT_LABELS:
                rmse = rmse_map.get((scene_name, proto, imp))
                if rmse is not None and bsl > 0.001:
                    pct = (rmse - bsl) / bsl * 100.0
                    mark = " *" if abs(pct) > 5 else ""
                    print(f"{pct:+10.1f}%{mark}", end="")
                else:
                    print(f" {'N/A':>10} ", end="")
            print(f"  (BSL={bsl:.3f}m, algo={algo_name})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# PART B — Degradation Sweeps
# ═══════════════════════════════════════════════════════════════════════════

# Sweep definitions per requirements §3.2
SWEEP_DEFS = {
    "iq_gain": {
        "label": "I/Q Gain Error ε",
        "param": "iq_gain_error",
        "values": [0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30],
        "xlabel": "I/Q Gain Error ε",
        "xunit": "",
        "xscale": "linear",
        "panel_label": "(a) I/Q Gain Error ε",
        "n_values": 8,
    },
    "snr": {
        "label": "SNR",
        "param": "snr_db",
        "values": [5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
        "xlabel": "SNR (dB)",
        "xunit": "dB",
        "xscale": "linear",
        "panel_label": "(b) SNR",
        "n_values": 10,
    },
    "adc_bits": {
        "label": "ADC Bits",
        "param": "adc_bits",
        "values": [2, 3, 4, 5, 6, 8, 10, 12, 16],
        "xlabel": "ADC Bits",
        "xunit": "bits",
        "xscale": "linear",
        "panel_label": "(c) ADC Bits",
        "n_values": 9,
    },
}

# Reference SNR values for noise scaling (per requirements §5.2)
_REF_SNR_DB = {"uwb": 30, "wifi": 30, "fiveg": 25}
_REF_AMPLITUDE_STD = {"uwb": 0.04, "wifi": 0.025, "fiveg": 0.03}
_REF_PHASE_STD_RAD = {"uwb": 0.08, "wifi": 0.04, "fiveg": 0.05}


def _build_sweep_config(
    proto: str,
    sweep_key: str,
    value: float,
) -> tuple[dict, dict]:
    """Build impairment dict + radio overrides for a sweep parameter value.

    Only the swept impairment is enabled; all others are off (isolated).
    """
    imp = _base_imp()  # all off
    radio_ov: dict = {}

    if sweep_key == "iq_gain":
        # Sweep I/Q gain imbalance ε, fix φ at protocol default
        imp["enable_iq_imbalance"] = True
        default_phi = _BASE_RADIO[proto]["iq_phase_imbalance_deg"]
        radio_ov = {
            "iq_gain_imbalance": value,
            "iq_phase_imbalance_deg": default_phi,
        }

    elif sweep_key == "snr":
        # Sweep SNR via explicit impairment model.
        # Scale amplitude/phase std as 1/√SNR  (requirements §5.2)
        snr_linear_ref = 10 ** (_REF_SNR_DB[proto] / 10.0)
        snr_linear_current = 10 ** (value / 10.0)
        scale = np.sqrt(snr_linear_ref / snr_linear_current)

        radio_ov = {
            "explicit_impairments": {
                "csi_noise_snr_db": value,
                "csi_amplitude_std": _REF_AMPLITUDE_STD[proto] * scale,
                "csi_phase_std_rad": _REF_PHASE_STD_RAD[proto] * scale,
            },
        }

    elif sweep_key == "adc_bits":
        # Sweep ADC quantization bits
        imp["enable_adc_quantization"] = True
        radio_ov = {"adc_bits": int(value)}

    return imp, radio_ov


def run_sweeps(
    scene: SceneConfig,
    n_trials: int,
    seed: int,
    out_dir: Path,
) -> dict[str, list[TrialRecord]]:
    """Run all 3 degradation sweeps on a single scene."""
    all_sweep_records: dict[str, list[TrialRecord]] = {}
    sweep_dir = out_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    for sweep_key, sweep_def in SWEEP_DEFS.items():
        values = list(sweep_def["values"])
        print(f"\n{'='*60}")
        print(f"SWEEP: {sweep_def['label']}")
        print(f"  Values: {values}")
        print(f"  Scene: {scene.name}  |  Trials per point: {n_trials}")
        print(f"{'='*60}")

        positions = sample_positions(scene, n_trials, seed=seed)
        dists = [np.linalg.norm(np.array(rx) - np.array(tx))
                 for tx, rx in positions]
        print(f"  {len(positions)} positions, dist range: "
              f"[{np.min(dists):.1f}, {np.max(dists):.1f}] m")

        sweep_records: list[TrialRecord] = []
        t0 = time.time()
        n_steps = len(values)

        for trial_idx, (tx, rx) in enumerate(positions):
            true_range_m = float(np.linalg.norm(np.array(rx) - np.array(tx)))
            trial_seed = seed + trial_idx
            algo_instances = {
                p: PROTOCOL_ALGO[p][1]() for p in PROTOCOLS
            }

            for proto in PROTOCOLS:
                algo_name, _ = PROTOCOL_ALGO[proto]

                # RT once per protocol per trial (independent of parameter value)
                cfg_rt = build_config(scene, tx, rx, proto, {}, {})
                freq = cfg_rt["radios"][proto]["carrier_frequency_hz"]
                rt_seed = trial_seed * 100 + hash(proto) % 100

                try:
                    truths = generate_truths(
                        cfg_rt, _rng(rt_seed), carrier_frequency_hz=freq,
                    )
                    truth = truths[0]
                except Exception:
                    continue

                # Also compute BSL (no impairments) once
                cfg_bsl = build_config(scene, tx, rx, proto, _base_imp(), {})
                radio_bsl = _make_radio(proto, cfg_bsl)
                impair_rng_bsl = _rng(trial_seed * 10 + hash(proto) % 100 + 1)
                radio_p = {
                    "sfo_ppm": _BASE_RADIO[proto].get("sfo_ppm", 20.0),
                    "cfo_hz": _BASE_RADIO[proto].get("cfo_hz", 500.0),
                }
                impaired_bsl = apply_timing_impairments(
                    truth, cfg_bsl, impair_rng_bsl, radio_cfg=radio_p,
                )
                obs_rng_bsl = _rng(trial_seed * 10 + hash(proto) % 100 + 2)
                obs_bsl = radio_bsl.observe(impaired_bsl, obs_rng_bsl)
                est_bsl = algo_instances[proto].estimate(obs_bsl)
                bsl_error = float(est_bsl.estimated_range_m - impaired_bsl.true_range_m)
                if not np.isfinite(bsl_error):
                    bsl_error = float("nan")

                sweep_records.append(TrialRecord(
                    scene=scene.name, category=scene.category,
                    trial=trial_idx, true_range_m=true_range_m,
                    protocol=proto, algorithm=algo_name,
                    impairment="none",
                    sweep_param=sweep_key, sweep_value=0.0,
                    estimated_range_m=float(est_bsl.estimated_range_m),
                    error_m=bsl_error,
                ))

                # Loop over sweep values
                for val in values:
                    imp_dict, radio_ov = _build_sweep_config(
                        proto, sweep_key, val,
                    )
                    cfg = build_config(
                        scene, tx, rx, proto, imp_dict,
                        radio_overrides=radio_ov,
                    )
                    radio = _make_radio(proto, cfg)

                    impair_rng = _rng(trial_seed * 10 + hash(proto) % 100 + 1)
                    impaired = apply_timing_impairments(
                        truth, cfg, impair_rng, radio_cfg=radio_p,
                    )
                    obs_rng = _rng(trial_seed * 10 + hash(proto) % 100 + 2)
                    observation = radio.observe(impaired, obs_rng)

                    algo = algo_instances[proto]
                    estimate = algo.estimate(observation)
                    error_m = float(
                        estimate.estimated_range_m - impaired.true_range_m
                    )
                    if not np.isfinite(error_m):
                        error_m = float("nan")

                    sweep_records.append(TrialRecord(
                        scene=scene.name, category=scene.category,
                        trial=trial_idx, true_range_m=true_range_m,
                        protocol=proto, algorithm=algo_name,
                        impairment=sweep_key,
                        sweep_param=sweep_key, sweep_value=val,
                        estimated_range_m=float(estimate.estimated_range_m),
                        error_m=error_m,
                    ))

            if (trial_idx + 1) % max(1, n_trials // 5) == 0 or trial_idx == 0:
                n_done = trial_idx + 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_trials - n_done) / rate if rate > 0 else 0
                print(f"    [{n_done:3d}/{n_trials}] "
                      f"recs={len(sweep_records)}  "
                      f"rate={rate:.2f}/s  ETA={eta/60:.0f}min", flush=True)

        elapsed_sweep = time.time() - t0
        print(f"  → {len(sweep_records)} records in {elapsed_sweep:.1f}s "
              f"({elapsed_sweep/n_trials:.1f}s/trial, "
              f"{len(sweep_records)/n_trials:.0f} recs/trial)")

        all_sweep_records[sweep_key] = sweep_records

        # Save per-sweep CSV
        _save_records_csv(sweep_records,
                          sweep_dir / f"{sweep_key}_per_trial.csv")
        _save_sweep_summary(sweep_records,
                            sweep_dir / f"{sweep_key}_summary.csv")

    return all_sweep_records


# ═══════════════════════════════════════════════════════════════════════════
# CSV output helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_records_csv(records: list[TrialRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scene", "category", "trial", "true_range_m",
            "protocol", "algorithm", "impairment",
            "sweep_param", "sweep_value",
            "estimated_range_m", "error_m",
        ])
        for r in records:
            w.writerow([
                r.scene, r.category, r.trial, f"{r.true_range_m:.6f}",
                r.protocol, r.algorithm, r.impairment,
                r.sweep_param, f"{r.sweep_value}" if r.sweep_value else "0",
                f"{r.estimated_range_m:.6f}" if np.isfinite(r.estimated_range_m) else "nan",
                f"{r.error_m:.6f}" if np.isfinite(r.error_m) else "nan",
            ])


def _save_scene_summary(records: list[TrialRecord], path: Path) -> None:
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        if np.isfinite(r.error_m):
            groups[(r.protocol, r.algorithm, r.impairment)].append(r.error_m)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "algorithm", "impairment",
                     "n_valid", "mean_err_m", "median_err_m",
                     "rmse_m", "p90_m", "std_m"])
        for (proto, algo, imp), errs in sorted(groups.items()):
            arr = np.array(errs, dtype=float)
            w.writerow([
                proto, algo, imp, len(arr),
                f"{np.mean(arr):.6f}", f"{np.median(arr):.6f}",
                f"{np.sqrt(np.mean(arr**2)):.6f}",
                f"{np.percentile(np.abs(arr), 90):.6f}",
                f"{np.std(arr):.6f}",
            ])


def _save_sweep_summary(records: list[TrialRecord], path: Path) -> None:
    """Aggregate sweep results: per (protocol, sweep_value) → RMSE, mean, std."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        if np.isfinite(r.error_m):
            groups[(r.protocol, r.sweep_value, r.impairment)].append(r.error_m)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "sweep_value", "impairment",
                     "n_valid", "mean_err_m", "median_err_m",
                     "rmse_m", "p90_m", "std_m"])
        for (proto, val, imp), errs in sorted(groups.items()):
            arr = np.array(errs, dtype=float)
            w.writerow([
                proto, f"{val}", imp, len(arr),
                f"{np.mean(arr):.6f}", f"{np.median(arr):.6f}",
                f"{np.sqrt(np.mean(arr**2)):.6f}",
                f"{np.percentile(np.abs(arr), 90):.6f}",
                f"{np.std(arr):.6f}",
            ])


# ═══════════════════════════════════════════════════════════════════════════
# PART C — Plotting
# ═══════════════════════════════════════════════════════════════════════════

# Protocol display config
_PROTO_STYLE = {
    "uwb":   {"color": "#2196F3", "marker": "o", "label": "UWB", "ls": "-"},
    "wifi":  {"color": "#FF9800", "marker": "s", "label": "WiFi", "ls": "-"},
    "fiveg": {"color": "#4CAF50", "marker": "^", "label": "5G", "ls": "-"},
}

_SWEEP_ORDER = ["iq_gain", "snr", "adc_bits"]


def _compute_sweep_curves(
    records: list[TrialRecord], sweep_key: str,
) -> dict[str, dict]:
    """Compute per-protocol RMSE, std across trials for each sweep value."""
    sweep_values = sorted(set(
        r.sweep_value for r in records
        if r.sweep_param == sweep_key and r.sweep_value > 0
    ))
    bsl_values = [r for r in records
                  if r.sweep_param == sweep_key and r.sweep_value == 0.0]

    # BSL RMSE per protocol
    bsl_rmse: dict[str, float] = {}
    for proto in PROTOCOLS:
        errs = [r.error_m for r in bsl_values
                if r.protocol == proto and np.isfinite(r.error_m)]
        if errs:
            bsl_rmse[proto] = float(np.sqrt(np.mean(np.array(errs) ** 2)))

    curves: dict[str, dict] = {}
    for proto in PROTOCOLS:
        x, y, y_std = [], [], []
        for val in sweep_values:
            errs = [r.error_m for r in records
                    if r.sweep_param == sweep_key
                    and r.protocol == proto
                    and r.sweep_value == val
                    and np.isfinite(r.error_m)]
            if len(errs) < 2:
                continue
            arr = np.array(errs, dtype=float)
            x.append(val)
            y.append(float(np.sqrt(np.mean(arr ** 2))))
            y_std.append(float(np.std(arr)))
        curves[proto] = {
            "x": np.array(x), "y": np.array(y), "y_std": np.array(y_std),
        }

    return curves, bsl_rmse


def plot_degradation_curves(
    all_sweep_records: dict[str, list[TrialRecord]],
    out_dir: Path,
) -> None:
    """Generate 3-panel degradation figure + individual panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Use a clean publication style
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "lines.linewidth": 1.3,
        "lines.markersize": 4.5, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.32, left=0.06, right=0.98,
                        top=0.90, bottom=0.13)

    for idx, sweep_key in enumerate(_SWEEP_ORDER):
        if sweep_key not in all_sweep_records:
            continue
        ax = axes[idx]
        sweep_def = SWEEP_DEFS[sweep_key]
        records = all_sweep_records[sweep_key]
        curves, bsl_rmse = _compute_sweep_curves(records, sweep_key)

        for proto in PROTOCOLS:
            if proto not in curves:
                continue
            style = _PROTO_STYLE[proto]
            c = curves[proto]

            # Shaded ±1σ band
            ax.fill_between(
                c["x"], c["y"] - c["y_std"], c["y"] + c["y_std"],
                color=style["color"], alpha=0.12,
            )
            # Main curve
            ax.plot(
                c["x"], c["y"],
                marker=style["marker"], color=style["color"],
                linestyle=style["ls"], label=style["label"],
                markersize=4.5, linewidth=1.3,
            )
            # BSL reference (dashed)
            if proto in bsl_rmse:
                ax.axhline(
                    y=bsl_rmse[proto],
                    color=style["color"], linestyle="dashed",
                    linewidth=0.7, alpha=0.6,
                )

        ax.set_title(sweep_def["panel_label"], fontweight="bold", pad=6)
        ax.set_xlabel(sweep_def["xlabel"])
        ax.set_ylabel("RMSE (m)" if idx == 0 else "")
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Formatting per panel
        if sweep_key == "iq_gain":
            ax.set_xlim(0, 0.32)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
            )
        elif sweep_key == "snr":
            ax.set_xlim(3, 42)
            # Annotate UWB +21 dB accumulation gain
            ax.annotate(
                "UWB +21 dB\ncoherent accum.",
                xy=(15, ax.get_ylim()[1] * 0.75 if ax.get_ylim()[1] > 0 else 2),
                xytext=(8, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 3),
                fontsize=6.5, color="#2196F3",
                arrowprops=dict(arrowstyle="->", color="#2196F3", lw=0.8),
            )
            ax.invert_xaxis()  # High SNR (good) on left
        elif sweep_key == "adc_bits":
            ax.set_xlim(1.5, 17)
            # Mark UWB default 6-bit
            ax.axvline(x=6, color="#2196F3", linestyle="dotted",
                       linewidth=0.7, alpha=0.5)
            ax.annotate("UWB\ndefault", xy=(6, 0), xytext=(7.5, 0.3),
                        fontsize=6, color="#2196F3", alpha=0.7)

        if idx == 1:  # SNR panel
            ax.legend(loc="lower left", framealpha=0.85, edgecolor="gray",
                      fontsize=7)
        elif idx == 0:
            ax.legend(loc="upper left", framealpha=0.85, edgecolor="gray",
                      fontsize=7)

    fig.suptitle("Fig 5. Ranging RMSE Degradation Under Swept Impairment Parameters",
                 fontweight="bold", fontsize=11, y=0.97)
    fig.text(0.5, 0.01,
             "Scene: box_los (simple LOS)  |  UWB→ChipLDE, WiFi→Threshold, "
             "5G→Threshold  |  Shaded band: ±1σ  |  Dashed: BSL reference",
             ha="center", fontsize=7, style="italic", color="gray")

    # Save
    combined_path = fig_dir / "degradation_combined.png"
    fig.savefig(combined_path)
    print(f"  Saved: {combined_path}")

    # Individual panels
    for idx, sweep_key in enumerate(_SWEEP_ORDER):
        fig_i, ax_i = plt.subplots(figsize=(5, 3.5))
        sweep_def = SWEEP_DEFS[sweep_key]
        records = all_sweep_records.get(sweep_key, [])
        if not records:
            plt.close(fig_i)
            continue
        curves, bsl_rmse = _compute_sweep_curves(records, sweep_key)

        for proto in PROTOCOLS:
            if proto not in curves:
                continue
            style = _PROTO_STYLE[proto]
            c = curves[proto]
            ax_i.fill_between(
                c["x"], c["y"] - c["y_std"], c["y"] + c["y_std"],
                color=style["color"], alpha=0.12,
            )
            ax_i.plot(
                c["x"], c["y"],
                marker=style["marker"], color=style["color"],
                linestyle=style["ls"], label=style["label"],
                markersize=5, linewidth=1.5,
            )
            if proto in bsl_rmse:
                ax_i.axhline(
                    y=bsl_rmse[proto],
                    color=style["color"], linestyle="dashed",
                    linewidth=0.7, alpha=0.6,
                )

        ax_i.set_title(sweep_def["panel_label"], fontweight="bold")
        ax_i.set_xlabel(sweep_def["xlabel"])
        ax_i.set_ylabel("RMSE (m)")
        if sweep_key == "iq_gain":
            ax_i.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
            )
        elif sweep_key == "snr":
            ax_i.invert_xaxis()
        ax_i.legend(loc="best", framealpha=0.85)
        ax_i.grid(True, alpha=0.3)

        indiv_path = fig_dir / f"degradation_{sweep_key}.png"
        fig_i.savefig(indiv_path)
        plt.close(fig_i)
        print(f"  Saved: {indiv_path}")

    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive Impairment Ablation + Degradation Sweeps",
    )
    parser.add_argument(
        "--mode", choices=["ablation", "sweeps", "all"], default="all",
        help="Which study to run",
    )
    parser.add_argument(
        "--ablation-trials", type=int, default=100,
        help="Trials per scene for ablation (default: 100)",
    )
    parser.add_argument(
        "--sweep-trials", type=int, default=100,
        help="Trials per data point for sweeps (default: 100)",
    )
    parser.add_argument(
        "--munich-trials", type=int, default=50,
        help="Trials for munich scene (default: 50, slower RT)",
    )
    parser.add_argument(
        "--skip-munich", action="store_true",
        help="Skip munich scene (slow RT)",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip PNG figure generation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _PROJECT_ROOT / "outputs" / "impairment_ablation_final" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Algorithms: UWB→chip_lde, WiFi→threshold, 5G→threshold")
    print()

    all_sweep_records: dict[str, list[TrialRecord]] = {}

    # ── Part A: Ablation ────────────────────────────────────────────────
    if args.mode in ("ablation", "all"):
        print("=" * 70)
        print("PART A — IMPAIRMENT ABLATION STUDY")
        print("=" * 70)
        print(f"Trials per scene: {args.ablation_trials} "
              f"(munich: {args.munich_trials})")
        print(f"Scenes: {[s.name for s in SCENE_CONFIGS]}")
        print(f"Total configs: {len(_IMPAIRMENT_LABELS)} "
              f"(× {len(PROTOCOLS)} protocols "
              f"× {len(SCENE_CONFIGS)} scenes)")
        print()

        scenes_to_run = list(SCENE_CONFIGS)
        if args.skip_munich:
            scenes_to_run = [s for s in scenes_to_run
                             if s.name != "munich_nlos"]
            print("  (munich skipped via --skip-munich)")

        # Adjust trial count for munich
        n_trials_per_scene = {}
        for s in scenes_to_run:
            n_trials_per_scene[s.name] = (
                args.munich_trials if s.name == "munich_nlos"
                else args.ablation_trials
            )

        # Run each scene with its trial count
        # (we modify run_ablation to support per-scene trial counts)
        ablation_records = _run_ablation_per_scene(
            scenes_to_run, n_trials_per_scene, args.seed, out_dir,
        )

    # ── Part B: Degradation Sweeps ──────────────────────────────────────
    if args.mode in ("sweeps", "all"):
        print("\n" + "=" * 70)
        print("PART B — DEGRADATION SWEEPS")
        print("=" * 70)
        print(f"Scene: box_los  |  Trials per point: {args.sweep_trials}")
        sweep_keys = list(SWEEP_DEFS.keys())
        print(f"Sweeps: {sweep_keys}")
        print(f"Total data points: "
              f"{sum(SWEEP_DEFS[k]['n_values'] for k in sweep_keys)} "
              f"parameter values "
              f"× {args.sweep_trials} trials "
              f"× {len(PROTOCOLS)} protocols")
        print()

        sweep_scene = next(s for s in SCENE_CONFIGS if s.name == "box_los")
        all_sweep_records = run_sweeps(
            sweep_scene, args.sweep_trials, args.seed, out_dir,
        )

        # Generate sweep summary CSV
        _save_sweep_combined_summary(all_sweep_records, out_dir / "sweeps")

        # Plots
        if not args.skip_plots:
            print("\nGenerating degradation plots ...")
            plot_degradation_curves(all_sweep_records, out_dir)

    # ── Metadata ────────────────────────────────────────────────────────
    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "ablation_trials": args.ablation_trials,
        "sweep_trials": args.sweep_trials,
        "munich_trials": args.munich_trials,
        "skip_munich": args.skip_munich,
        "algorithms": {p: PROTOCOL_ALGO[p][0] for p in PROTOCOLS},
        "protocols": list(PROTOCOLS),
        "impairment_configs": _IMPAIRMENT_LABELS,
        "sweep_definitions": {
            k: {"label": v["label"], "values": list(v["values"])}
            for k, v in SWEEP_DEFS.items()
        },
        "scenes": [
            {"name": s.name, "category": s.category,
             "description": s.description}
            for s in SCENE_CONFIGS
        ],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    print(f"\n{'='*70}")
    print(f"Done. Results: {out_dir}")
    if args.mode in ("ablation", "all"):
        print(f"  ablation/ablation_table.csv")
        print(f"  ablation/ablation_pivot.csv")
        print(f"  ablation/per_scene/")
    if args.mode in ("sweeps", "all"):
        print(f"  sweeps/iq_gain_per_trial.csv")
        print(f"  sweeps/snr_per_trial.csv")
        print(f"  sweeps/adc_bits_per_trial.csv")
        print(f"  sweeps/sweep_summary.csv")
        if not args.skip_plots:
            print(f"  figures/degradation_combined.png")
            print(f"  figures/degradation_iq_gain.png")
            print(f"  figures/degradation_snr.png")
            print(f"  figures/degradation_adc_bits.png")
    print(f"  metadata.json")


def _run_ablation_per_scene(
    scenes: list[SceneConfig],
    n_trials_map: dict[str, int],
    seed: int,
    out_dir: Path,
) -> list[TrialRecord]:
    """Run ablation with per-scene trial counts."""
    impairment_configs = _build_ablation_configs()
    all_records: list[TrialRecord] = []

    for scene_idx, scene in enumerate(scenes):
        n_trials = n_trials_map[scene.name]
        print(f"\n{'='*60}")
        print(f"ABLATION Scene {scene_idx+1}/{len(scenes)}: {scene.name} "
              f"({scene.category}) — {n_trials} trials")
        print(f"  {scene.description}")
        print(f"{'='*60}")

        positions = sample_positions(scene, n_trials, seed=seed)
        dists = [np.linalg.norm(np.array(rx) - np.array(tx))
                 for tx, rx in positions]
        print(f"  Dist range: [{np.min(dists):.1f}, {np.max(dists):.1f}] m")

        t0 = time.time()
        scene_records: list[TrialRecord] = []

        for trial_idx, (tx, rx) in enumerate(positions):
            true_range_m = float(np.linalg.norm(np.array(rx) - np.array(tx)))
            trial_seed = seed + scene_idx * 10000 + trial_idx
            algo_instances = {
                p: PROTOCOL_ALGO[p][1]() for p in PROTOCOLS
            }

            for proto in PROTOCOLS:
                algo_name, _ = PROTOCOL_ALGO[proto]
                cfg_rt = build_config(scene, tx, rx, proto, {}, {})
                freq = cfg_rt["radios"][proto]["carrier_frequency_hz"]
                rt_seed = trial_seed * 100 + hash(proto) % 100

                try:
                    truths = generate_truths(
                        cfg_rt, _rng(rt_seed), carrier_frequency_hz=freq,
                    )
                    truth = truths[0]
                except Exception:
                    continue

                for imp_label, imp_dict, radio_ov in impairment_configs:
                    cfg = build_config(
                        scene, tx, rx, proto, imp_dict,
                        radio_overrides=radio_ov.get(proto, {}),
                    )
                    radio = _make_radio(proto, cfg)
                    impair_rng = _rng(trial_seed * 10 + hash(proto) % 100 + 1)
                    radio_p = {
                        "sfo_ppm": _BASE_RADIO[proto].get("sfo_ppm", 20.0),
                        "cfo_hz": _BASE_RADIO[proto].get("cfo_hz", 500.0),
                    }
                    impaired = apply_timing_impairments(
                        truth, cfg, impair_rng, radio_cfg=radio_p,
                    )
                    obs_rng = _rng(trial_seed * 10 + hash(proto) % 100 + 2)
                    observation = radio.observe(impaired, obs_rng)
                    algo = algo_instances[proto]
                    estimate = algo.estimate(observation)
                    error_m = float(
                        estimate.estimated_range_m - impaired.true_range_m
                    )
                    if not np.isfinite(error_m):
                        error_m = float("nan")
                    scene_records.append(TrialRecord(
                        scene=scene.name, category=scene.category,
                        trial=trial_idx, true_range_m=true_range_m,
                        protocol=proto, algorithm=algo_name,
                        impairment=imp_label,
                        sweep_param="", sweep_value=0.0,
                        estimated_range_m=float(estimate.estimated_range_m),
                        error_m=error_m,
                    ))

            if (trial_idx + 1) % max(1, n_trials // 4) == 0 or trial_idx == 0:
                n_done = trial_idx + 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_trials - n_done) / rate if rate > 0 else 0
                print(f"    [{n_done:3d}/{n_trials}] "
                      f"recs={len(scene_records)}  "
                      f"rate={rate:.2f}/s  ETA={eta/60:.0f}min", flush=True)

        elapsed_scene = time.time() - t0
        print(f"  → {len(scene_records)} records in {elapsed_scene:.1f}s "
              f"({elapsed_scene/n_trials:.1f}s/trial)")

        scene_dir = out_dir / "ablation" / "per_scene" / scene.name
        scene_dir.mkdir(parents=True, exist_ok=True)
        _save_records_csv(scene_records, scene_dir / "per_trial_results.csv")
        _save_scene_summary(scene_records, scene_dir / "summary.csv")
        all_records.extend(scene_records)

    _build_ablation_master_table(all_records, out_dir / "ablation")
    return all_records


def _save_sweep_combined_summary(
    all_sweep_records: dict[str, list[TrialRecord]],
    out_dir: Path,
) -> None:
    """Create a combined sweep summary CSV for easy plotting."""
    path = out_dir / "sweep_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep_param", "protocol", "sweep_value",
                     "n_valid", "mean_err_m", "median_err_m",
                     "rmse_m", "std_m", "p90_m"])
        for sweep_key in _SWEEP_ORDER:
            if sweep_key not in all_sweep_records:
                continue
            records = all_sweep_records[sweep_key]

            # BSL first (sweep_value = 0)
            for proto in PROTOCOLS:
                errs = [r.error_m for r in records
                        if r.protocol == proto
                        and r.sweep_value == 0.0
                        and np.isfinite(r.error_m)]
                if errs:
                    arr = np.array(errs, dtype=float)
                    w.writerow([
                        sweep_key, proto, "BSL", len(arr),
                        f"{np.mean(arr):.6f}", f"{np.median(arr):.6f}",
                        f"{np.sqrt(np.mean(arr**2)):.6f}",
                        f"{np.std(arr):.6f}",
                        f"{np.percentile(np.abs(arr), 90):.6f}",
                    ])

            # Sweep values
            sweep_values = sorted(set(
                r.sweep_value for r in records
                if r.sweep_param == sweep_key and r.sweep_value > 0
            ))
            for val in sweep_values:
                for proto in PROTOCOLS:
                    errs = [r.error_m for r in records
                            if r.protocol == proto
                            and r.sweep_param == sweep_key
                            and r.sweep_value == val
                            and np.isfinite(r.error_m)]
                    if errs:
                        arr = np.array(errs, dtype=float)
                        w.writerow([
                            sweep_key, proto, f"{val}",
                            len(arr),
                            f"{np.mean(arr):.6f}", f"{np.median(arr):.6f}",
                            f"{np.sqrt(np.mean(arr**2)):.6f}",
                            f"{np.std(arr):.6f}",
                            f"{np.percentile(np.abs(arr), 90):.6f}",
                        ])
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    main()
