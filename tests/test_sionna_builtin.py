"""Tests for environments.sionna_builtin — scene resolution and listing."""

import pytest

sionna = pytest.importorskip("sionna", reason="Sionna not installed")

from environments.sionna_builtin import _resolve_scene_path, list_builtin_scenes


class TestListBuiltinScenes:
    def test_returns_non_empty_list(self):
        scenes = list_builtin_scenes()
        assert len(scenes) > 5
        assert "etoile" in scenes
        assert "munich" in scenes
        assert "florence" in scenes

    def test_all_entries_are_strings(self):
        for name in list_builtin_scenes():
            assert isinstance(name, str)
            assert len(name) > 0


class TestResolveScenePath:
    def test_etoile_resolves_to_xml(self):
        path = _resolve_scene_path("etoile")
        assert path.endswith("etoile.xml")
        assert "etoile" in path

    def test_munich_resolves_to_xml(self):
        path = _resolve_scene_path("munich")
        assert path.endswith("munich.xml")
        assert "munich" in path

    def test_unknown_scene_raises(self):
        with pytest.raises(ValueError, match="Unknown Sionna built-in scene"):
            _resolve_scene_path("nonexistent_scene_xyz")
