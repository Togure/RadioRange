"""Tests for environments.custom_scene — mesh XML generation and XML loading."""

import os
import tempfile

from environments.custom_scene import _mesh_to_xml, _read_scene_xml


class TestMeshToXml:
    def test_obj_generates_valid_xml(self):
        """OBJ mesh → valid Mitsuba scene XML with correct structure."""
        cfg = {
            "mesh_file": "scenes/simple_box.obj",
            "material": "itu_brick",
            "translate": [1.0, 0.0, 2.0],
            "scale": [2.0, 2.0, 2.0],
        }
        xml = _mesh_to_xml(cfg)
        assert '<scene version="3.0.0">' in xml
        assert 'type="obj"' in xml
        assert 'itu_brick' in xml
        assert "scenes/simple_box.obj" in xml
        assert '<scale x="2.0" y="2.0" z="2.0"/>' in xml
        assert '<translate x="1.0" y="0.0" z="2.0"/>' in xml

    def test_obj_defaults(self):
        """Default translate/scale → identity transform."""
        cfg = {"mesh_file": "scenes/simple_box.obj", "material": "itu_concrete"}
        xml = _mesh_to_xml(cfg)
        assert '<translate x="0.0" y="0.0" z="0.0"/>' in xml
        assert '<scale x="1.0" y="1.0" z="1.0"/>' in xml

    def test_ply_generates_valid_xml(self):
        """PLY mesh → valid Mitsuba scene XML."""
        cfg = {
            "mesh_file": "scenes/test.ply",
            "material": "itu_glass",
        }
        xml = _mesh_to_xml(cfg)
        assert '<scene version="3.0.0">' in xml
        assert 'type="ply"' in xml
        assert 'itu_glass' in xml
        assert "scenes/test.ply" in xml

    def test_unsupported_format_raises(self):
        """Unknown file extension raises ValueError."""
        import pytest
        cfg = {"mesh_file": "scenes/bad.gltf", "material": "itu_concrete"}
        with pytest.raises(ValueError, match="Unsupported mesh format"):
            _mesh_to_xml(cfg)


class TestReadSceneXml:
    def test_reads_file_content(self):
        """Should read the XML file contents as a string."""
        content = '<scene version="3.0.0">\n  <!-- test -->\n</scene>\n'
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False,
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = _read_scene_xml({"scene_xml_file": path})
            assert result == content
        finally:
            os.unlink(path)
