"""Save and load ChannelTruth lists to/from disk.

Format: one .npz file for numpy arrays + one .json sidecar for metadata.
The two files together fully reconstruct a list[ChannelTruth].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.models import ChannelTruth


def _truth_to_dict(truth: ChannelTruth) -> dict[str, Any]:
    """Extract serialisable fields from a single ChannelTruth."""
    meta_serialisable = {}
    for k, v in truth.metadata.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            meta_serialisable[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                meta_serialisable[k] = v
            except (TypeError, ValueError):
                meta_serialisable[k] = str(v)
        else:
            meta_serialisable[k] = str(v)

    return {
        "true_range_m": truth.true_range_m,
        "los": truth.los,
        "sync_bias_s": truth.sync_bias_s,
        "clock_bias_s": truth.clock_bias_s,
        "rtt_mode": truth.rtt_mode,
        "carrier_frequency_hz": truth.carrier_frequency_hz,
        "num_paths": int(len(truth.a_paths)),
        "metadata": meta_serialisable,
    }


def save_truths(truths: list[ChannelTruth], directory: str | Path) -> Path:
    """Save a list of ChannelTruth objects to *directory*.

    Creates ``truths.npz`` (numpy arrays) and ``config.json`` (scalar metadata).
    Returns the output directory path.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    num_trials = len(truths)

    # Stack arrays with trial index as first dimension
    a_paths_stack = np.stack([t.a_paths for t in truths])
    tau_paths_stack = np.stack([t.tau_paths_s for t in truths])

    arrays: dict[str, np.ndarray] = {
        "a_paths": a_paths_stack,
        "tau_paths_s": tau_paths_stack,
        "num_trials": np.array(num_trials),
    }

    # Optional per-path arrays — only saved when ALL truths have them
    for field in [
        "path_type", "path_order", "polarization",
        "aoa_azimuth_deg", "aoa_elevation_deg",
        "aod_azimuth_deg", "aod_elevation_deg",
    ]:
        values = [getattr(t, field) for t in truths]
        if all(v is not None for v in values):
            try:
                arrays[field] = np.stack(values)
            except (ValueError, TypeError):
                arrays[field] = np.array(values, dtype=object)

    np.savez(directory / "truths.npz", **arrays)

    # Sidecar JSON
    sidecar = {
        "num_trials": num_trials,
        "trials": [_truth_to_dict(t) for t in truths],
    }
    (directory / "config.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return directory


def load_truths(directory: str | Path) -> list[ChannelTruth]:
    """Load a list of ChannelTruth objects from a directory created by :func:`save_truths`."""
    directory = Path(directory)
    npz_path = directory / "truths.npz"
    json_path = directory / "config.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"truths.npz not found in {directory}")
    if not json_path.exists():
        raise FileNotFoundError(f"config.json not found in {directory}")

    data = np.load(npz_path, allow_pickle=True)
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))

    num_trials = int(data["num_trials"])
    a_paths_all = data["a_paths"]
    tau_paths_all = data["tau_paths_s"]

    optional_fields = [
        "path_type", "path_order", "polarization",
        "aoa_azimuth_deg", "aoa_elevation_deg",
        "aod_azimuth_deg", "aod_elevation_deg",
    ]

    truths: list[ChannelTruth] = []
    for i in range(num_trials):
        trial_meta = sidecar["trials"][i]
        meta = trial_meta.get("metadata", {})

        kwargs: dict[str, Any] = {
            "a_paths": a_paths_all[i],
            "tau_paths_s": tau_paths_all[i],
            "true_range_m": float(trial_meta.get("true_range_m", 0.0)),
            "los": bool(trial_meta.get("los", True)),
            "sync_bias_s": float(trial_meta.get("sync_bias_s", 0.0)),
            "clock_bias_s": float(trial_meta.get("clock_bias_s", 0.0)),
            "rtt_mode": bool(trial_meta.get("rtt_mode", False)),
            "carrier_frequency_hz": trial_meta.get("carrier_frequency_hz"),
            "metadata": meta,
        }

        for field in optional_fields:
            if field in data:
                kwargs[field] = data[field][i]

        truths.append(ChannelTruth(**kwargs))

    return truths
