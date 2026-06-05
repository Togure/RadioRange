"""Fingerprint radio map utilities — grid sampling, RSSI, and radio map construction."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any
from PIL import Image

LIGHT_SPEED_MPS = 299_792_458.0


def _load_floorplan_mask(
    image_path: Path,
    ppm: float,
    background_color: tuple[int, int, int] = (255, 255, 255),
    tolerance: int = 40,
) -> np.ndarray:
    """Load floorplan PNG and return a boolean mask (True = free space, False = wall).

    Returns (mask, img_w_px, img_h_px).
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    bg = np.array(background_color, dtype=np.uint8)
    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    return diff <= tolerance, arr.shape[1], arr.shape[0]


def generate_grid_points(
    image_path: Path,
    ppm: float,
    grid_spacing_m: float = 2.0,
    device_height_m: float = 1.5,
    margin_m: float = 0.5,
    background_color: tuple[int, int, int] = (255, 255, 255),
    tolerance: int = 40,
) -> tuple[list[tuple[float, float, float]], float, float]:
    """Generate a regular grid of free-space sample points within a floorplan.

    Args:
        image_path: Path to floorplan PNG.
        ppm: Pixels per meter.
        grid_spacing_m: Distance between adjacent grid points in meters.
        device_height_m: Z coordinate for all grid points.
        margin_m: Minimum distance from walls (meters).
        background_color: RGB tuple for the background (free space).
        tolerance: Color matching tolerance.

    Returns:
        (grid_points, room_width_m, room_height_m) where grid_points is a list
        of (x, y, z) tuples in meters.
    """
    mask, img_w, img_h = _load_floorplan_mask(image_path, ppm, background_color, tolerance)
    room_w_m = img_w / ppm
    room_h_m = img_h / ppm

    # Convert margin to pixels
    margin_px = int(margin_m * ppm)

    # Erode mask by margin to keep points away from walls
    if margin_px > 0:
        from scipy.ndimage import binary_erosion
        struct = np.ones((2 * margin_px + 1, 2 * margin_px + 1), dtype=bool)
        mask = binary_erosion(mask, structure=struct)

    points: list[tuple[float, float, float]] = []
    px_spacing = grid_spacing_m * ppm

    for row_px in np.arange(margin_px, img_h - margin_px, px_spacing):
        y = row_px / ppm
        for col_px in np.arange(margin_px, img_w - margin_px, px_spacing):
            x = col_px / ppm
            r, c = int(row_px), int(col_px)
            if 0 <= r < img_h and 0 <= c < img_w and mask[r, c]:
                points.append((float(x), float(y), float(device_height_m)))

    return points, room_w_m, room_h_m


def compute_rssi(
    paths: list[dict[str, Any]] | list[Any] | np.ndarray,
    tx_power_dbm: float = 20.0,
    rx_antenna_gain_dbi: float = 0.0,
) -> float:
    """Compute RSSI (dBm) from Sionna RT path gains.

    Each path contributes |gain|² to the total received power.
    RSSI = 10·log₁₀(Σ|a_i|²) + P_tx_dBm + G_rx_dBi

    Args:
        paths: List of complex gains (numpy or Python), or list of objects/dicts
               with a ``gain`` attribute.
        tx_power_dbm: Transmit power in dBm (default 20 dBm, typical WiFi).
        rx_antenna_gain_dbi: Receive antenna gain in dBi.

    Returns:
        RSSI in dBm.
    """
    total_power_linear = 0.0
    for g in paths:
        if isinstance(g, (complex, np.complex64, np.complex128)):
            total_power_linear += float(np.abs(g) ** 2)
        elif hasattr(g, "gain"):
            total_power_linear += float(np.abs(g.gain) ** 2)
        elif isinstance(g, dict):
            total_power_linear += float(np.abs(g["gain"]) ** 2)

    if total_power_linear <= 0:
        return -120.0  # noise floor

    rssi = 10.0 * np.log10(total_power_linear) + tx_power_dbm + rx_antenna_gain_dbi
    return float(rssi)


def build_radio_map(
    ap_positions: list[tuple[str, float, float, float]],
    grid_points: list[tuple[float, float, float]],
    run_single_measurement,
) -> dict[str, Any]:
    """Build a fingerprint radio map by running RT at every (AP, grid point) pair.

    Args:
        ap_positions: List of (ap_id, x, y, z) tuples.
        grid_points: List of (x, y, z) tuples from ``generate_grid_points``.
        run_single_measurement: Callable(ap_pos, rx_pos) → dict with keys
            'rssi_dbm', 'range_m', 'true_range_m', 'paths'.

    Returns:
        Dict with keys: 'records' (list of per-measurement dicts),
        'ap_ids', 'n_aps', 'n_points', 'grid_spacing_m'.
    """
    records: list[dict[str, Any]] = []
    n_aps = len(ap_positions)
    n_points = len(grid_points)
    n_total = n_aps * n_points

    idx = 0
    for ap_id, ap_x, ap_y, ap_z in ap_positions:
        ap_pos = (ap_x, ap_y, ap_z)
        for gx, gy, gz in grid_points:
            result = run_single_measurement(ap_pos, (gx, gy, gz))
            result["ap_id"] = ap_id
            result["ap_x"] = ap_x
            result["ap_y"] = ap_y
            result["ap_z"] = ap_z
            result["rx_x"] = gx
            result["rx_y"] = gy
            result["rx_z"] = gz
            records.append(result)
            idx += 1
            if idx % 100 == 0:
                print(f"  [{idx}/{n_total}] AP={ap_id}  rx=({gx:.1f}, {gy:.1f})  "
                      f"RSSI={result.get('rssi_dbm', -999):.0f} dBm  "
                      f"range_err={result.get('range_error_m', 0):.2f}m")

    return {
        "records": records,
        "ap_ids": [ap[0] for ap in ap_positions],
        "n_aps": n_aps,
        "n_points": n_points,
    }
