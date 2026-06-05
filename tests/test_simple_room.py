"""Tests for environments.simple_room — room geometry and material helpers."""

import numpy as np
import pytest

from environments.materials import (
    _fresnel_reflection_power,
    _resolve_wall_materials,
    _wall_dir_to_name,
)
from environments.simple_room import _image_method_paths


class TestWallDirToName:
    def test_x_axis(self):
        assert _wall_dir_to_name(0, -1) == "wall_x_min"
        assert _wall_dir_to_name(0, +1) == "wall_x_max"

    def test_y_axis(self):
        assert _wall_dir_to_name(1, -1) == "wall_y_min"
        assert _wall_dir_to_name(1, +1) == "wall_y_max"

    def test_z_axis(self):
        assert _wall_dir_to_name(2, -1) == "floor"
        assert _wall_dir_to_name(2, +1) == "ceiling"

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="Invalid dimension"):
            _wall_dir_to_name(3, 1)


class TestResolveWallMaterials:
    def test_single_material_uniform(self):
        """Single material string → all walls get same material."""
        cfg = {"dimensions_m": [8.0, 6.0], "material": "itu_brick"}
        result = _resolve_wall_materials(cfg)
        assert len(result) == 4  # 2D room
        for name in ["wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max"]:
            assert result[name] == "itu_brick"

    def test_per_wall_dict(self):
        """Materials dict → per-wall assignment."""
        cfg = {
            "dimensions_m": [4.0, 4.0, 3.0],
            "materials": {
                "floor": "itu_metal",
                "ceiling": "itu_ceiling_board",
                "wall_x_min": "itu_brick",
                "wall_x_max": "itu_brick",
                "wall_y_min": "itu_plasterboard",
                "wall_y_max": "itu_plasterboard",
            },
        }
        result = _resolve_wall_materials(cfg)
        assert result["floor"] == "itu_metal"
        assert result["ceiling"] == "itu_ceiling_board"
        assert result["wall_x_min"] == "itu_brick"
        assert result["wall_y_max"] == "itu_plasterboard"

    def test_partial_dict_fills_default(self):
        """Missing walls in materials dict fall back to itu_concrete."""
        cfg = {
            "dimensions_m": [8.0, 6.0],
            "materials": {"wall_x_min": "itu_glass"},
        }
        result = _resolve_wall_materials(cfg)
        assert result["wall_x_min"] == "itu_glass"
        assert result["wall_x_max"] == "itu_concrete"
        assert result["wall_y_min"] == "itu_concrete"

    def test_no_material_key_defaults(self):
        """No material key at all → all walls itu_concrete."""
        cfg = {"dimensions_m": [4.0, 4.0, 3.0]}
        result = _resolve_wall_materials(cfg)
        assert len(result) == 6
        for v in result.values():
            assert v == "itu_concrete"

    def test_2d_vs_3d_wall_count(self):
        """2D room returns 4 walls, 3D room returns 6."""
        cfg_2d = {"dimensions_m": [8.0, 6.0]}
        cfg_3d = {"dimensions_m": [4.0, 4.0, 3.0]}
        assert len(_resolve_wall_materials(cfg_2d)) == 4
        assert len(_resolve_wall_materials(cfg_3d)) == 6


class TestFresnelReflectionPower:
    def test_dielectric_between_zero_and_one(self):
        """Dielectric Γ should be between 0 and 1."""
        gamma = _fresnel_reflection_power(5e9, "itu_concrete")
        assert 0.0 < gamma < 1.0

    def test_metal_near_one(self):
        """Metal (high conductivity) → Γ ≈ 1."""
        gamma = _fresnel_reflection_power(5e9, "itu_metal")
        assert gamma > 0.99

    def test_frequency_dependence(self):
        """Higher frequency → lower Γ for dielectrics (skin effect)."""
        g_low = _fresnel_reflection_power(1e9, "itu_concrete")
        g_high = _fresnel_reflection_power(10e9, "itu_concrete")
        # Γ should differ measurably with frequency
        assert abs(g_low - g_high) > 1e-6

    def test_different_materials_give_different_gamma(self):
        """Concrete and brick should have measurably different Γ at same freq."""
        g_concrete = _fresnel_reflection_power(5e9, "itu_concrete")
        g_brick = _fresnel_reflection_power(5e9, "itu_brick")
        assert abs(g_concrete - g_brick) > 1e-4

    def test_unknown_material_falls_back_to_concrete(self):
        """Unknown material → same Γ as concrete."""
        g_unknown = _fresnel_reflection_power(5e9, "unicorn_dust")
        g_concrete = _fresnel_reflection_power(5e9, "itu_concrete")
        assert g_unknown == g_concrete


class TestImageMethodPaths:
    DIMS = np.array([8.0, 6.0])
    TX = np.array([1.0, 1.0])
    RX = np.array([6.5, 4.5])
    WALL_COEFFS = {
        "wall_x_min": 0.15,
        "wall_x_max": 0.15,
        "wall_y_min": 0.25,
        "wall_y_max": 0.25,
    }

    def test_los_path_present_and_order_zero(self):
        records = _image_method_paths(
            self.DIMS, self.TX, self.RX, max_reflections=1, wall_reflection_coeffs=self.WALL_COEFFS,
        )
        los = [r for r in records if r["order"] == 0]
        assert len(los) == 1
        assert los[0]["name"] == "los"
        assert los[0]["gain"].real > 0

    def test_los_distance_correct(self):
        records = _image_method_paths(
            self.DIMS, self.TX, self.RX, max_reflections=0, wall_reflection_coeffs=self.WALL_COEFFS,
        )
        expected_distance = float(np.linalg.norm(self.RX - self.TX))
        np.testing.assert_allclose(records[0]["distance_m"], expected_distance, rtol=1e-10)

    def test_paths_sorted_by_distance(self):
        records = _image_method_paths(
            self.DIMS, self.TX, self.RX, max_reflections=2, wall_reflection_coeffs=self.WALL_COEFFS,
        )
        distances = [r["distance_m"] for r in records]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]

    def test_reflection_count_matches_order(self):
        records = _image_method_paths(
            self.DIMS, self.TX, self.RX, max_reflections=2, wall_reflection_coeffs=self.WALL_COEFFS,
        )
        for r in records:
            assert r["order"] <= 2

    def test_first_order_reflections_use_wall_gamma(self):
        """First-order paths should have gain = Γ_wall / distance."""
        records = _image_method_paths(
            self.DIMS, self.TX, self.RX, max_reflections=1, wall_reflection_coeffs=self.WALL_COEFFS,
        )
        first_order = [r for r in records if r["order"] == 1]
        assert len(first_order) > 0
        for r in first_order:
            expected_gain = min(self.WALL_COEFFS.values()) ** 1 / r["distance_m"]
            # Single bounce on one wall → gain ≈ Γ / d
            assert 0 < r["gain"].real <= 1.0 / r["distance_m"]

    def test_per_wall_coefficients_different_effect(self):
        """Asymmetric wall coefficients produce different gains for x vs y reflections."""
        asymmetric = {
            "wall_x_min": 0.1,
            "wall_x_max": 0.9,
            "wall_y_min": 0.1,
            "wall_y_max": 0.9,
        }
        records = _image_method_paths(
            self.DIMS, self.TX, self.RX, max_reflections=1, wall_reflection_coeffs=asymmetric,
        )
        first_order = [r for r in records if r["order"] == 1]
        gains = [r["gain"].real for r in first_order]
        # Gains should differ because walls have different Γ
        assert len(set(round(g, 10) for g in gains)) > 1

    def test_zero_reflections_max_reflections_zero(self):
        """max_reflections=0 → only LOS path."""
        records = _image_method_paths(
            self.DIMS, self.TX, self.RX, max_reflections=0, wall_reflection_coeffs=self.WALL_COEFFS,
        )
        assert len(records) == 1
        assert records[0]["order"] == 0

    def test_3d_room_wall_count(self):
        """3D room produces paths with floor/ceiling bounces."""
        dims_3d = np.array([4.0, 4.0, 3.0])
        tx_3d = np.array([1.0, 1.0, 1.5])
        rx_3d = np.array([2.0, 2.0, 1.5])
        coeffs_3d = {
            "floor": 0.2,
            "ceiling": 0.3,
            "wall_x_min": 0.15,
            "wall_x_max": 0.15,
            "wall_y_min": 0.25,
            "wall_y_max": 0.25,
        }
        records = _image_method_paths(
            dims_3d, tx_3d, rx_3d, max_reflections=1, wall_reflection_coeffs=coeffs_3d,
        )
        los = [r for r in records if r["order"] == 0]
        assert len(los) == 1
        # 3D with 6 walls → first order has 6 reflection paths
        first_order = [r for r in records if r["order"] == 1]
        assert len(first_order) == 6
