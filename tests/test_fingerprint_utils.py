"""Tests for utils.fingerprint — RSSI computation, grid generation, radio map."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.fingerprint import compute_rssi, generate_grid_points, build_radio_map


class TestComputeRssi:
    """Tests for compute_rssi() — RSSI from path gains."""

    def test_single_path_numpy_complex(self):
        """One strong LOS path should give expected RSSI."""
        paths = np.array([0.8 + 0j])
        rssi = compute_rssi(paths, tx_power_dbm=20.0, rx_antenna_gain_dbi=0.0)
        # P_rx = 10*log10(|0.8|²) + 20 = 10*log10(0.64) + 20 ≈ -1.94 + 20 ≈ 18.06
        expected = 10.0 * np.log10(0.64) + 20.0
        assert rssi == pytest.approx(expected, rel=0.01)

    def test_single_path_python_complex(self):
        paths = [0.5 + 0.5j]  # |gain|² = 0.5
        rssi = compute_rssi(paths, tx_power_dbm=20.0, rx_antenna_gain_dbi=0.0)
        expected = 10.0 * np.log10(0.5) + 20.0  # ≈ -3.01 + 20 ≈ 16.99
        assert rssi == pytest.approx(expected, rel=0.01)

    def test_multiple_paths_sum(self):
        """Two equal paths → power doubles → +3 dB."""
        paths = np.array([0.5 + 0j, 0.5 + 0j])
        rssi = compute_rssi(paths, tx_power_dbm=20.0, rx_antenna_gain_dbi=0.0)
        # |0.5|² + |0.5|² = 0.25 + 0.25 = 0.5
        expected = 10.0 * np.log10(0.5) + 20.0
        assert rssi == pytest.approx(expected, rel=0.01)

    def test_dict_with_gain_key(self):
        paths = [{"gain": 0.8 + 0j}]
        rssi = compute_rssi(paths, tx_power_dbm=20.0)
        expected = 10.0 * np.log10(0.64) + 20.0
        assert rssi == pytest.approx(expected, rel=0.01)

    def test_object_with_gain_attr(self):
        class FakePath:
            def __init__(self, gain):
                self.gain = gain
        paths = [FakePath(0.3 + 0j)]
        rssi = compute_rssi(paths, tx_power_dbm=0.0)
        expected = 10.0 * np.log10(0.09)
        assert rssi == pytest.approx(expected, rel=0.01)

    def test_no_power_returns_noise_floor(self):
        """Zero paths → noise floor."""
        rssi = compute_rssi([], tx_power_dbm=20.0)
        assert rssi == -120.0

    def test_all_zero_gains_returns_noise_floor(self):
        """All-zero complex gains → noise floor."""
        paths = [0.0 + 0j, 0.0 + 0j]
        rssi = compute_rssi(paths)
        assert rssi == -120.0

    def test_tx_power_adds_directly(self):
        """+10 dBm TX power → +10 dB RSSI."""
        paths = np.array([1.0 + 0j])
        rssi_20 = compute_rssi(paths, tx_power_dbm=20.0, rx_antenna_gain_dbi=0.0)
        rssi_30 = compute_rssi(paths, tx_power_dbm=30.0, rx_antenna_gain_dbi=0.0)
        assert rssi_30 == pytest.approx(rssi_20 + 10.0, rel=0.001)

    def test_rx_gain_adds_directly(self):
        """+3 dBi RX antenna gain → +3 dB RSSI."""
        paths = np.array([1.0 + 0j])
        rssi_0 = compute_rssi(paths, tx_power_dbm=20.0, rx_antenna_gain_dbi=0.0)
        rssi_3 = compute_rssi(paths, tx_power_dbm=20.0, rx_antenna_gain_dbi=3.0)
        assert rssi_3 == pytest.approx(rssi_0 + 3.0, rel=0.001)

    def test_weak_path_below_noise_floor(self):
        """Very weak path still returns a real RSSI value."""
        paths = [1e-6 + 0j]  # |g|² = 1e-12 → -120 dB
        rssi = compute_rssi(paths, tx_power_dbm=0.0, rx_antenna_gain_dbi=0.0)
        # 10*log10(1e-12) = -120, + 0 = -120, which is exactly noise floor
        assert rssi <= -119.0  # allow float margin

    def test_numpy_complex128(self):
        paths = np.array([0.7 + 0j], dtype=np.complex128)
        rssi = compute_rssi(paths, tx_power_dbm=0.0, rx_antenna_gain_dbi=0.0)
        expected = 10.0 * np.log10(0.49)
        assert rssi == pytest.approx(expected, rel=0.01)

    def test_mixed_types(self):
        """Mix of numpy complex, Python complex, and dict paths."""
        paths = [
            np.complex64(0.5 + 0j),
            complex(0.3, 0.0),
            {"gain": 0.2 + 0j},
        ]
        rssi = compute_rssi(paths, tx_power_dbm=0.0, rx_antenna_gain_dbi=0.0)
        # total_power = 0.25 + 0.09 + 0.04 = 0.38
        expected = 10.0 * np.log10(0.38)
        assert rssi == pytest.approx(expected, rel=0.01)


class TestBuildRadioMap:
    """Tests for build_radio_map() — the orchestration loop."""

    def test_single_ap_single_point(self):
        """One AP × one grid point → one record."""
        def fake_runner(ap_pos, rx_pos):
            true_range = float(np.linalg.norm(np.array(rx_pos) - np.array(ap_pos)))
            return {
                "rssi_dbm": -45.0,
                "range_m": true_range + 0.1,
                "true_range_m": true_range,
                "range_error_m": 0.1,
                "n_paths": 3,
            }

        result = build_radio_map(
            ap_positions=[("ap1", 10.0, 5.0, 2.5)],
            grid_points=[(5.0, 5.0, 1.5)],
            run_single_measurement=fake_runner,
        )

        assert result["n_aps"] == 1
        assert result["n_points"] == 1
        assert result["ap_ids"] == ["ap1"]
        assert len(result["records"]) == 1
        r = result["records"][0]
        assert r["ap_id"] == "ap1"
        assert r["rssi_dbm"] == -45.0
        assert r["range_error_m"] == 0.1

    def test_two_aps_two_points(self):
        """2 APs × 2 grid points → 4 records."""
        def fake_runner(ap_pos, rx_pos):
            true_range = float(np.linalg.norm(np.array(rx_pos) - np.array(ap_pos)))
            return {
                "rssi_dbm": -50.0,
                "range_m": true_range,
                "true_range_m": true_range,
                "range_error_m": 0.0,
                "n_paths": 1,
            }

        result = build_radio_map(
            ap_positions=[
                ("ap1", 10.0, 5.0, 2.5),
                ("ap2", 30.0, 5.0, 2.5),
            ],
            grid_points=[
                (5.0, 5.0, 1.5),
                (15.0, 5.0, 1.5),
            ],
            run_single_measurement=fake_runner,
        )

        assert len(result["records"]) == 4
        ap_ids_in_records = {r["ap_id"] for r in result["records"]}
        assert ap_ids_in_records == {"ap1", "ap2"}

    def test_record_has_all_required_keys(self):
        """Each record must include AP position, RX position, and measurement keys."""
        def fake_runner(ap_pos, rx_pos):
            true_range = float(np.linalg.norm(np.array(rx_pos) - np.array(ap_pos)))
            return {
                "rssi_dbm": -55.0,
                "range_m": true_range,
                "true_range_m": true_range,
                "range_error_m": 0.05,
                "n_paths": 2,
            }

        result = build_radio_map(
            ap_positions=[("ap1", 5.0, 5.0, 2.5)],
            grid_points=[(10.0, 10.0, 1.5)],
            run_single_measurement=fake_runner,
        )

        r = result["records"][0]
        required = {"ap_id", "ap_x", "ap_y", "ap_z", "rx_x", "rx_y", "rx_z",
                    "rssi_dbm", "range_m", "true_range_m", "range_error_m"}
        assert required.issubset(set(r.keys()))
