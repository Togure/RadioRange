"""Mitsuba 3 XML scene builder for Sionna RT.

Produces Mitsuba 3 XML fragments (BSDF, shapes, scene wrapper) that are
consumed by Sionna's ``load_scene_from_string()``.  All shape builders
accept world-space centre / half-extent tuples and emit axis-aligned
geometry with a ``to_world`` transform.

The ``mat-custom-`` prefix on BSDF IDs tells Sionna to use the explicit
permittivity/conductivity values rather than overriding with built-in
ITU-R P.2040 defaults (see ``sionna.rt.scene_utils.process_xml()``).
"""

from __future__ import annotations

from typing import Sequence

from environments.materials import get_material_props


def build_radio_material_bsdf(material_name: str) -> str:
    """Single ``<bsdf type="radio-material">`` node with ε' and σ from the DB."""
    props = get_material_props(material_name)
    eps = float(props["epsilon"])
    sigma = float(props["sigma"])
    return (
        f'  <bsdf type="radio-material" id="mat-custom-{material_name}">\n'
        f'    <float name="relative_permittivity" value="{eps}"/>\n'
        f'    <float name="conductivity" value="{sigma}"/>\n'
        f"  </bsdf>"
    )


def build_bsdf_definitions(material_names: set[str]) -> list[str]:
    """BSDF definitions for every unique material in *material_names*."""
    return [build_radio_material_bsdf(name) for name in sorted(material_names)]


def build_transform(
    translate: tuple[float, float, float],
    scale: tuple[float, float, float] | None = None,
) -> str:
    """``to_world`` transform with optional scale followed by translate."""
    tx, ty, tz = translate
    lines = []
    if scale is not None:
        sx, sy, sz = scale
        lines.append(f'      <scale x="{sx}" y="{sy}" z="{sz}"/>')
    lines.append(f'      <translate x="{tx}" y="{ty}" z="{tz}"/>')
    return "\n".join(lines)


def _make_transform_block(inner: str) -> str:
    return f"    <transform name=\"to_world\">\n{inner}\n    </transform>"


def build_cube_shape(
    name: str,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    material: str,
) -> str:
    """Axis-aligned cuboid wall / obstacle."""
    return (
        f'  <shape type="cube" id="{name}">\n'
        + _make_transform_block(build_transform(center, half_extents))
        + "\n"
        f'    <ref id="mat-custom-{material}"/>\n'
        f"  </shape>"
    )


def build_rectangle_shape(
    name: str,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    material: str,
) -> str:
    """Axis-aligned rectangle (flat in XY by default, useful for floor/ceiling)."""
    return (
        f'  <shape type="rectangle" id="{name}">\n'
        + _make_transform_block(build_transform(center, half_extents))
        + "\n"
        f'    <ref id="mat-custom-{material}"/>\n'
        f"  </shape>"
    )


def build_obj_shape(
    name: str,
    filename: str,
    material: str,
    translate: tuple[float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> str:
    """Shape referencing an external ``.obj`` mesh file."""
    lines = [f'  <shape type="obj" id="{name}">']
    if translate is not None or scale is not None:
        lines.append(
            _make_transform_block(
                build_transform(
                    translate or (0.0, 0.0, 0.0),
                    scale,
                )
            )
        )
    lines.append(f'    <string name="filename" value="{filename}"/>')
    lines.append(f'    <ref id="mat-custom-{material}"/>')
    lines.append(f"  </shape>")
    return "\n".join(lines)


def build_ply_shape(
    name: str,
    filename: str,
    material: str,
    translate: tuple[float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> str:
    """Shape referencing an external ``.ply`` mesh file."""
    lines = [f'  <shape type="ply" id="{name}">']
    if translate is not None or scale is not None:
        lines.append(
            _make_transform_block(
                build_transform(
                    translate or (0.0, 0.0, 0.0),
                    scale,
                )
            )
        )
    lines.append(f'    <string name="filename" value="{filename}"/>')
    lines.append(f'    <ref id="mat-custom-{material}"/>')
    lines.append(f"  </shape>")
    return "\n".join(lines)


def build_scene_xml(
    bsdf_strings: Sequence[str],
    shape_strings: Sequence[str],
) -> str:
    """Wrap BSDF definitions and shape elements in a Mitsuba 3 scene document."""
    parts = list(bsdf_strings) + list(shape_strings)
    return '<scene version="3.0.0">\n' + "\n".join(parts) + "\n</scene>"
