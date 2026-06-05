"""Tests for environments.floorplan — image parsing and wall extraction."""

import numpy as np
import pytest

from environments.floorplan import (
    _extract_walls,
    _parse_floorplan_image,
    _split_by_orientation,
)


def _make_test_image(colors_px):
    """Create an RGB image from a dict of {color_tuple: list_of_pixel_coords}."""
    import tempfile
    from PIL import Image

    all_pixels = []
    for coords in colors_px.values():
        all_pixels.extend(coords)
    if not all_pixels:
        H, W = 10, 10
    else:
        H = max(y for y, _ in all_pixels) + 1
        W = max(x for _, x in all_pixels) + 1

    img = np.full((H, W, 3), 255, dtype=np.uint8)
    for color, coords in colors_px.items():
        for y, x in coords:
            img[y, x] = color

    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(img).save(path.name)
    return path.name, H, W


class TestSplitByOrientation:
    def test_horizontal_line(self):
        """A single row of pixels should be entirely horizontal."""
        arr = np.zeros((10, 10), dtype=bool)
        arr[5, 2:8] = True
        h_mask, v_mask = _split_by_orientation(arr)
        assert np.all(h_mask[5, 2:8])
        assert not np.any(v_mask)

    def test_vertical_line(self):
        """A single column of pixels should be entirely vertical."""
        arr = np.zeros((10, 10), dtype=bool)
        arr[2:8, 5] = True
        h_mask, v_mask = _split_by_orientation(arr)
        assert np.all(v_mask[2:8, 5])
        assert not np.any(h_mask)

    def test_thick_horizontal_wall(self):
        """3 rows × 20 cols → all horizontal."""
        arr = np.zeros((10, 30), dtype=bool)
        arr[4:7, 5:25] = True
        h_mask, v_mask = _split_by_orientation(arr)
        # All pixels should be horizontal (width 20 > height 3)
        assert np.all(h_mask[4:7, 5:25])
        assert not np.any(v_mask)

    def test_thick_vertical_wall(self):
        """20 rows × 3 cols → all vertical."""
        arr = np.zeros((30, 10), dtype=bool)
        arr[5:25, 4:7] = True
        h_mask, v_mask = _split_by_orientation(arr)
        assert np.all(v_mask[5:25, 4:7])
        assert not np.any(h_mask)

    def test_cross_shape(self):
        """A cross where the center goes to the longer arm."""
        arr = np.zeros((20, 20), dtype=bool)
        arr[8:12, 4:16] = True  # horizontal bar (4 rows × 12 cols)
        arr[4:16, 8:12] = True  # vertical bar (12 rows × 4 cols)
        h_mask, v_mask = _split_by_orientation(arr)

        # Center pixel (10, 10): h_run=12, v_run=12 → equal → horizontal
        center_h = h_mask[10, 10]
        center_v = v_mask[10, 10]
        assert center_h and not center_v

    def test_square(self):
        """A square (equal dimensions) should be mostly horizontal."""
        arr = np.zeros((20, 20), dtype=bool)
        arr[5:15, 5:15] = True
        h_mask, v_mask = _split_by_orientation(arr)
        # Most should be horizontal (h_run >= v_run for all pixels)
        assert np.sum(h_mask) >= np.sum(v_mask)


class TestParseFloorplanImage:
    def test_single_color_wall(self):
        """One colour → one material label."""
        path, H, W = _make_test_image({
            (0, 0, 0): [(y, x) for y in range(5) for x in range(100)],
        })
        cfg = {
            "image_path": path,
            "color_mapping": [{"color": [0, 0, 0], "material": "itu_concrete"}],
            "background_color": [255, 255, 255],
            "default_tolerance": 40,
        }
        labels, mat_map = _parse_floorplan_image(cfg)
        assert len(mat_map) == 1
        assert mat_map[0] == "itu_concrete"
        assert np.all(labels[:5, :] == 0)
        assert np.all(labels[5:, :] == -1)

    def test_two_materials(self):
        """Two colours → two distinct labels."""
        path, H, W = _make_test_image({
            (0, 0, 0): [(y, 0) for y in range(10)],
            (128, 64, 0): [(y, 5) for y in range(10)],
        })
        cfg = {
            "image_path": path,
            "color_mapping": [
                {"color": [0, 0, 0], "material": "itu_concrete"},
                {"color": [128, 64, 0], "material": "itu_brick"},
            ],
            "background_color": [255, 255, 255],
            "default_tolerance": 30,
        }
        labels, mat_map = _parse_floorplan_image(cfg)
        assert len(mat_map) == 2
        assert mat_map[0] == "itu_concrete"
        assert mat_map[1] == "itu_brick"
        assert labels[0, 0] == 0
        assert labels[0, 5] == 1

    def test_background_ignored(self):
        """Pixels close to background_color → label -1."""
        path, H, W = _make_test_image({
            (0, 0, 0): [(2, 2)],
        })
        cfg = {
            "image_path": path,
            "color_mapping": [{"color": [0, 0, 0], "material": "itu_concrete"}],
            "background_color": [255, 255, 255],
            "default_tolerance": 40,
        }
        labels, _ = _parse_floorplan_image(cfg)
        # White pixels → background (-1); black pixel → itu_concrete
        assert labels[0, 0] == -1
        assert labels[2, 2] == 0


class TestExtractWalls:
    def test_simple_rectangular_room(self):
        """A simple unfilled rectangle border → 4 walls."""
        H, W = 200, 300
        arr = np.full((H, W), -1, dtype=np.int32)
        # Outer border, 6px thick
        arr[0:6, :] = 0
        arr[-6:, :] = 0
        arr[:, 0:6] = 0
        arr[:, -6:] = 0
        mat_map = {0: "itu_concrete"}
        cfg = {"pixels_per_meter": 20, "wall_height_m": 3.0}

        walls = _extract_walls(arr, mat_map, cfg)
        # Should get: top, bottom, left, right (4 walls)
        assert len(walls) >= 4

        materials = {w["material"] for w in walls}
        assert materials == {"itu_concrete"}

    def test_no_walls_for_background_only(self):
        """An all-background image produces no walls."""
        arr = np.full((50, 50), -1, dtype=np.int32)
        mat_map = {}
        cfg = {"pixels_per_meter": 10, "wall_height_m": 3.0}
        walls = _extract_walls(arr, mat_map, cfg)
        assert len(walls) == 0

    def test_single_vertical_wall(self):
        """A single vertical line → 1 wall."""
        H, W = 100, 50
        arr = np.full((H, W), -1, dtype=np.int32)
        arr[10:90, 24:28] = 0  # 4px wide vertical wall
        mat_map = {0: "itu_brick"}
        cfg = {"pixels_per_meter": 10, "wall_height_m": 3.0}

        walls = _extract_walls(arr, mat_map, cfg)
        assert len(walls) == 1
        w = walls[0]
        # Should be roughly centered
        cx, cy, cz = w["center"]
        assert 2.4 < cx < 2.7  # ~2.6m (col 26/10)
        assert 4.5 < cy < 5.5  # ~5.0m (row 50/10)
        assert cz == 1.5

    def test_single_horizontal_wall(self):
        """A single horizontal line → 1 wall."""
        H, W = 50, 100
        arr = np.full((H, W), -1, dtype=np.int32)
        arr[24:28, 10:90] = 0  # 4px thick horizontal wall
        mat_map = {0: "itu_brick"}
        cfg = {"pixels_per_meter": 10, "wall_height_m": 3.0}

        walls = _extract_walls(arr, mat_map, cfg)
        assert len(walls) == 1
        w = walls[0]
        hx, hy, hz = w["half_extents"]
        assert hx > hy  # wider than tall
        assert hz == 1.5

    def test_wall_with_door_gap(self):
        """A wall with a gap (door) → 2 wall segments."""
        H, W = 20, 200
        arr = np.full((H, W), -1, dtype=np.int32)
        arr[8:12, 5:80] = 0
        arr[8:12, 120:195] = 0  # door from x=80 to x=120
        mat_map = {0: "itu_concrete"}
        cfg = {"pixels_per_meter": 10, "wall_height_m": 3.0}

        walls = _extract_walls(arr, mat_map, cfg)
        assert len(walls) == 2
