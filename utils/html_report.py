"""HTML report builder — extracted from scripts/rt_cache_interactive.py.

ALL logic is preserved character-for-character from the original.
The public entry point is ``build_html_report()``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_LIGHT_SPEED = 299_792_458.0

# ═══════════════════════════════════════════════════════════════════════════════
# Visual constants
# ═══════════════════════════════════════════════════════════════════════════════

_TYPE_STYLES = {
    "los":         {"color": "#1E3A8A", "width": 4.5, "dash": "solid",  "label": "LOS"},
    "specular":    {"color": "#EF4444", "width": 2.5, "dash": "solid",  "label": "Specular"},
    "diffuse":     {"color": "#3B82F6", "width": 1.8, "dash": "dash",   "label": "Diffuse"},
    "refraction":  {"color": "#8B5CF6", "width": 2.0, "dash": "longdash", "label": "Refraction"},
    "diffraction": {"color": "#10B981", "width": 2.0, "dash": "dot",    "label": "Diffraction"},
    "fix":         {"color": "#6B7280", "width": 2.0, "dash": "solid",  "label": "Mixed"},
}

_WALL_MATERIAL_COLORS: dict[str, str] = {
    "itu_concrete":      "#C5BEB0",
    "itu_brick":         "#C48830",
    "itu_glass":         "#93C5FD",
    "itu_wood":          "#8B6914",
    "itu_metal":         "#9CA3AF",
    "itu_plasterboard":  "#E5E7EB",
    "itu_ceiling_board": "#F5F5F4",
    "itu_floorboard":    "#D6D3D1",
}

_COLORS_PROTO = {
    "uwb": "#FF7A00",
    "wifi": "#2563EB",
    "fiveg": "#059669",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3D mesh loading
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_scene_xml_path(scene_name: str) -> str | None:
    import importlib
    import os

    try:
        mod = importlib.import_module(f"sionna.rt.scenes.{scene_name}")
    except (ImportError, ModuleNotFoundError):
        return None
    for p in mod.__path__:
        for fn in os.listdir(p):
            if fn.endswith(".xml"):
                return os.path.join(p, fn)
    return None


def _load_meshes(scene_name: str, mid: np.ndarray, max_tri_total: int = 50000):
    """Load PLY meshes nearest-first, returning Plotly-ready data and search triangles."""
    import trimesh

    xml_path = _resolve_scene_xml_path(scene_name)
    if xml_path is None:
        return None, None, None, None, []
    meshes_dir = Path(xml_path).parent / "meshes"
    if not meshes_dir.is_dir():
        return None, None, None, None, []

    entries = sorted(meshes_dir.glob("*.ply"))
    if not entries:
        return None, None, None, None, []

    def _dist_key(fp):
        try:
            m = trimesh.load(str(fp), process=False, skip_materials=True)
            v = np.asarray(m.vertices, dtype=float)
            return float(np.linalg.norm(v.mean(axis=0)[:2] - mid[:2])) if v.size else float("inf")
        except Exception:
            return float("inf")

    entries.sort(key=_dist_key)

    all_verts, all_faces, all_triangles, vert_offset, total_tri = [], [], [], 0, 0
    for ply_file in entries:
        if total_tri >= max_tri_total:
            break
        try:
            mesh = trimesh.load(str(ply_file), process=False)
        except Exception:
            continue
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int) if mesh.faces is not None else None
        if verts.size == 0 or faces is None or faces.size == 0:
            continue
        if total_tri + faces.shape[0] > max_tri_total:
            continue
        for f in faces:
            all_triangles.append(verts[f])
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)
        total_tri += faces.shape[0]

    if not all_verts:
        return None, None, None, None, []

    mega_verts = np.concatenate(all_verts, axis=0)
    mega_faces = np.concatenate(all_faces, axis=0)
    return (
        mega_verts,
        mega_faces[:, 0].tolist(),
        mega_faces[:, 1].tolist(),
        mega_faces[:, 2].tolist(),
        all_triangles,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# reflection-point reconstruction
# ═══════════════════════════════════════════════════════════════════════════════


def _angles_to_unit(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


def _reconstruct_reflection_point(
    tx: np.ndarray, rx: np.ndarray,
    aod_az_deg: float, aod_el_deg: float,
    aoa_az_deg: float, aoa_el_deg: float,
    total_distance_m: float,
) -> np.ndarray | None:
    u_d = _angles_to_unit(aod_az_deg, aod_el_deg)
    v = tx - rx
    d_total = total_distance_m
    denom = 2.0 * (float(np.dot(v, u_d)) + d_total)
    if abs(denom) < 1e-12:
        return None
    d1 = (d_total ** 2 - float(np.dot(v, v))) / denom
    d2 = d_total - d1
    if d1 < 0.0 or d2 < 0.0:
        return None
    return tx + d1 * u_d


# ═══════════════════════════════════════════════════════════════════════════════
# RadioMap computation
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_radiomap(scene_name: str, tx_pos: np.ndarray, rx_pos: np.ndarray,
                      max_depth: int = 3, cell_size: float | None = None,
                      num_samples: int = 200000,
                      scattering: float = 0.3,
                      xml_path: str | None = None) -> dict | None:
    from sionna.rt import (
        load_scene, PlanarArray, RadioMapSolver,
        Receiver, Transmitter,
    )

    if xml_path is not None and Path(xml_path).exists():
        scene = load_scene(xml_path, merge_shapes=True)
    else:
        resolved = _resolve_scene_xml_path(scene_name)
        if resolved is None:
            return None
        scene = load_scene(resolved, merge_shapes=True)
    scene.frequency = 2.4e9
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.add(Transmitter(name="tx", position=tx_pos.tolist()))
    scene.add(Receiver(name="rx", position=rx_pos.tolist()))

    for mat in scene.radio_materials.values():
        mat.scattering_coefficient = scattering
        itu_type = str(getattr(mat, "itu_type", ""))
        if "glass" in itu_type.lower():
            mat.thickness = 0.02
        elif "wood" in itu_type.lower() or "chipboard" in itu_type.lower():
            mat.thickness = 0.05
        else:
            mat.thickness = 0.15

    solver = RadioMapSolver()
    bbox = scene.mi_scene.bbox()
    rm_center = [float(bbox.center().x), float(bbox.center().y), float(tx_pos[2])]
    rm_size = [float(bbox.max.x - bbox.min.x), float(bbox.max.y - bbox.min.y)]
    if cell_size is None:
        cell_size = max(rm_size) / 60.0
        cell_size = max(cell_size, 0.2)
    rm = solver(
        scene,
        center=rm_center,
        orientation=(0, 0, 0),
        size=rm_size,
        cell_size=(cell_size, cell_size),
        samples_per_tx=num_samples,
        max_depth=max_depth,
        los=True, specular_reflection=True,
        diffuse_reflection=True,
        refraction=True, diffraction=True,
    )
    pg = rm.path_gain.numpy()
    pg_db = np.where(pg > 0, 10.0 * np.log10(pg), np.nan)
    centers = rm.cell_centers.numpy()
    return {
        "path_gain_db": pg_db[0],
        "cell_centers": centers,
        "cells_per_dim": (int(rm.cells_per_dim.x[0]), int(rm.cells_per_dim.y[0])),
        "size": (float(rm.size.x[0]), float(rm.size.y[0])),
        "center": (float(rm.center.x[0]), float(rm.center.y[0]), float(rm.center.z[0])),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Wall meshes
# ═══════════════════════════════════════════════════════════════════════════════


def _build_wall_meshes(wall_geometry: list[dict]) -> list:
    """Build semi-transparent Mesh3d traces for floorplan walls."""
    import plotly.graph_objects as go

    traces: list = []
    for w in wall_geometry:
        cx, cy, cz = w["center"]
        hx, hy, hz = w["half_extents"]
        material = w.get("material", "itu_concrete")
        color = _WALL_MATERIAL_COLORS.get(material, "#C5BEB0")

        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        z0, z1 = cz - hz, cz + hz
        verts = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=float)

        i_idx = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
        j_idx = [1, 2, 5, 6, 3, 4, 2, 5, 3, 6, 7, 4]
        k_idx = [2, 3, 6, 7, 4, 7, 5, 6, 6, 7, 4, 5]

        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i_idx, j=j_idx, k=k_idx,
            color=color, opacity=0.20, flatshading=True,
            name="Wall", showlegend=False,
            hovertemplate="Wall<extra></extra>",
            lighting=dict(ambient=0.55, diffuse=0.65, specular=0.15,
                          roughness=0.85, facenormalsepsilon=1e-4),
            lightposition=dict(x=200, y=200, z=500),
        ))

    return traces


# ═══════════════════════════════════════════════════════════════════════════════
# Button / menu builders
# ═══════════════════════════════════════════════════════════════════════════════


def _make_zoom_button(label: str, half_span: float, mid: np.ndarray,
                      z_stretch: float = 3.0) -> dict:
    return dict(
        label=label, method="relayout",
        args=[{
            "scene.xaxis.range": [mid[0] - half_span, mid[0] + half_span],
            "scene.yaxis.range": [mid[1] - half_span, mid[1] + half_span],
            "scene.zaxis.range": [-half_span / z_stretch,
                                   max(10, mid[2] + half_span / z_stretch * 2)],
        }],
    )


def _make_cir_toggle_dropdown(
    disc_indices: list[int], cont_indices: list[int],
) -> dict:
    """Build a dropdown to toggle between Discrete and Continuous CIR.

    Uses ``method="restyle"`` with per-trace visibility arrays so the toggle
    does not interfere with the protocol / algorithm filter dropdowns (which
    use ``method="update"`` and replace the full visibility array).
    """
    all_cir = disc_indices + cont_indices
    n_disc = len(disc_indices)
    n_cont = len(cont_indices)

    disc_vals = [True] * n_disc + [False] * n_cont
    cont_vals = [False] * n_disc + [True] * n_cont

    return dict(
        type="dropdown",
        buttons=[
            dict(label="Discrete CIR", method="restyle",
                 args=[{"visible": disc_vals}, all_cir]),
            dict(label="Continuous CIR", method="restyle",
                 args=[{"visible": cont_vals}, all_cir]),
        ],
        direction="down", showactive=True,
        x=0.27, y=1.00, xanchor="left", yanchor="top",
        pad=dict(r=2, t=4),
    )


def _make_z_stretch_button(label: str, z_stretch: float) -> dict:
    return dict(
        label=label, method="relayout",
        args=[{"scene.aspectratio.x": 1, "scene.aspectratio.y": 1,
               "scene.aspectratio.z": 1.0 / z_stretch}],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Path type mapping
# ═══════════════════════════════════════════════════════════════════════════════


def _map_path_type(ptype_raw: str, order: int) -> str:
    """Map ChannelTruth path_type to toggle key."""
    if order == 0:
        return "los"
    p = str(ptype_raw).lower()
    if p == "fix":
        return "fix"
    if p in ("specular", "reflection"):
        return "specular"
    if p in ("diffuse", "scattering"):
        return "diffuse"
    if p == "refraction":
        return "refraction"
    if p == "diffraction":
        return "diffraction"
    return "specular"


# ═══════════════════════════════════════════════════════════════════════════════
# 3D trace builder
# ═══════════════════════════════════════════════════════════════════════════════


def _build_3d_traces(
    scene_name: str,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    truth_first,
    mid: np.ndarray,
    verts, i_f, j_f, k_f, z_stretch: float,
) -> tuple[list, dict[str, list[int]]]:
    """Build 3D traces from cached truth data.  Returns (traces, trace_idx_by_type)."""
    import plotly.graph_objects as go

    traces = []
    trace_idx_by_type: dict[str, list[int]] = {k: [] for k in _TYPE_STYLES}

    # ── building mesh ──────────────────────────────────────────────────────
    if verts is not None:
        mesh_opacity = 0.20
        edge_opacity = 0.70
        edge_width = 2
        mesh_color = "#C5BEB0"

        t = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i_f, j=j_f, k=k_f,
            color=mesh_color, opacity=mesh_opacity, flatshading=True,
            name="Buildings", showlegend=False,
            hovertemplate="Building<extra></extra>",
            lighting=dict(ambient=0.55, diffuse=0.65, specular=0.15,
                          roughness=0.85, facenormalsepsilon=1e-4),
            lightposition=dict(x=200, y=200, z=500),
        )
        traces.append(t)

        # wireframe
        edges: set[tuple[int, int]] = set()
        for a, b, c in zip(i_f, j_f, k_f):
            edges.add((min(a, b), max(a, b)))
            edges.add((min(b, c), max(b, c)))
            edges.add((min(c, a), max(c, a)))
        ex, ey, ez = [], [], []
        for a, b in edges:
            ex.extend([float(verts[a, 0]), float(verts[b, 0]), None])
            ey.extend([float(verts[a, 1]), float(verts[b, 1]), None])
            ez.extend([float(verts[a, 2]), float(verts[b, 2]), None])
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="#000000", width=edge_width),
            opacity=edge_opacity,
            name="Edges", showlegend=False, hoverinfo="skip",
        ))

    # ── extract path data ──────────────────────────────────────────────────
    a_paths = np.asarray(truth_first.a_paths)
    tau_paths_s = np.asarray(truth_first.tau_paths_s)
    active = np.abs(a_paths) > 1e-12
    sorted_idx = np.argsort(np.abs(a_paths[active]))[::-1]
    active_indices = np.flatnonzero(active)

    path_types = truth_first.path_type
    path_orders = truth_first.path_order
    aoa_az = truth_first.aoa_azimuth_deg
    aoa_el = truth_first.aoa_elevation_deg
    aod_az = truth_first.aod_azimuth_deg
    aod_el = truth_first.aod_elevation_deg

    # Build lookup: path_idx -> vertices from metadata
    _path_records_raw = truth_first.metadata.get("paths", [])
    if isinstance(_path_records_raw, str):
        import ast
        _path_records = ast.literal_eval(_path_records_raw)
    else:
        _path_records = _path_records_raw
    _vertices_lookup: dict[int, list[list[float]]] = {}
    _itypes_lookup: dict[int, list[int]] = {}
    _CODE_TO_STYLE_KEY: dict[int, str] = {1: "specular", 2: "diffuse", 4: "refraction", 8: "diffraction"}
    for _pos, _rec in enumerate(_path_records):
        _verts = _rec.get("vertices")
        if _verts and isinstance(_verts, list) and len(_verts) > 0:
            _vertices_lookup[_pos] = _verts
        _itypes = _rec.get("interaction_types")
        if _itypes and isinstance(_itypes, list) and len(_itypes) > 0:
            _itypes_lookup[_pos] = _itypes

    # Compute min order per type across ALL active paths (for solid/dash logic)
    _min_ord_3d: dict[str, int] = {}
    for _si in sorted_idx[:50]:
        _pi = int(active_indices[int(_si)])
        _ord = int(path_orders[_pi]) if path_orders is not None else 0
        _pt = str(path_types[_pi]) if path_types is not None else "reflection"
        _tk = _map_path_type(_pt, _ord)
        if _ord == 0:
            continue
        if _tk not in _min_ord_3d or _ord < _min_ord_3d[_tk]:
            _min_ord_3d[_tk] = _ord

    # Per-type counters — synced with table: top 10 per pure type, 1 LOS, 10 mix
    los_done = False
    max_per_type: dict[str, int] = {"specular": 10, "diffuse": 10, "refraction": 10, "diffraction": 10, "fix": 10}
    type_counts: dict[str, int] = defaultdict(int)

    for rank, sort_pos in enumerate(sorted_idx):
        path_idx = int(active_indices[int(sort_pos)])
        gain_abs = float(np.abs(a_paths[path_idx]))
        if gain_abs < 1e-20:
            continue
        distance_m = float(tau_paths_s[path_idx] * _LIGHT_SPEED)
        order = int(path_orders[path_idx]) if path_orders is not None else 0
        ptype_raw = str(path_types[path_idx]) if path_types is not None else "reflection"

        type_key = _map_path_type(ptype_raw, order)
        style = _TYPE_STYLES[type_key]

        # ── Fix (mixed) ────────────────────────────────────────────────────
        if type_key == "fix":
            _flimit = max_per_type.get("fix", 10)
            if type_counts["fix"] >= _flimit:
                continue
            type_counts["fix"] += 1

            pverts = _vertices_lookup.get(path_idx)
            itypes = _itypes_lookup.get(path_idx)
            if pverts and len(pverts) >= 1:
                px = [tx_pos[0]] + [v[0] for v in pverts] + [rx_pos[0]]
                py = [tx_pos[1]] + [v[1] for v in pverts] + [rx_pos[1]]
                pz = [tx_pos[2]] + [v[2] for v in pverts] + [rx_pos[2]]
                opacity = 0.65
            else:
                px = [tx_pos[0], rx_pos[0]]
                py = [tx_pos[1], rx_pos[1]]
                pz = [tx_pos[2], rx_pos[2]]
                opacity = 0.35

            t = go.Scatter3d(
                x=px, y=py, z=pz,
                mode="lines",
                line=dict(color=style["color"], width=1.6, dash=style["dash"]),
                opacity=opacity,
                showlegend=False, legendgroup="fix",
                hovertemplate=f"fix {order}b  {distance_m:.1f}m  "
                              f"{20 * np.log10(gain_abs + 1e-20):.0f}dB<extra></extra>",
            )
            traces.append(t)
            trace_idx_by_type["fix"].append(len(traces) - 1)

            if pverts and len(pverts) >= 1:
                for vi, v in enumerate(pverts):
                    v_type_code = itypes[vi] if itypes and vi < len(itypes) else 0
                    v_tk = _CODE_TO_STYLE_KEY.get(v_type_code, "fix")
                    v_style = _TYPE_STYLES.get(v_tk, _TYPE_STYLES["specular"])
                    t_marker = go.Scatter3d(
                        x=[v[0]], y=[v[1]], z=[v[2]],
                        mode="markers",
                        marker=dict(size=3.5, color=v_style["color"], symbol="circle",
                                   line=dict(color="#000000", width=0.5)),
                        opacity=0.80,
                        showlegend=False, legendgroup="fix",
                        hovertemplate=f"{v_style['label']} v{vi}  "
                                      f"[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]<extra></extra>",
                    )
                    traces.append(t_marker)
                    trace_idx_by_type["fix"].append(len(traces) - 1)
            continue

        # ── LOS ────────────────────────────────────────────────────────────
        if type_key == "los":
            if los_done:
                continue
            los_done = True
            t = go.Scatter3d(
                x=[tx_pos[0], rx_pos[0]], y=[tx_pos[1], rx_pos[1]],
                z=[tx_pos[2], rx_pos[2]],
                mode="lines",
                line=dict(color=style["color"], width=style["width"] + 1.0, dash=style["dash"]),
                name=style["label"], legendgroup="los", showlegend=False,
                hovertemplate=f"LOS  {distance_m:.1f}m<extra></extra>",
            )
            traces.append(t)
            trace_idx_by_type["los"].append(len(traces) - 1)
            continue

        # ── 1-bounce ───────────────────────────────────────────────────────
        if order == 1:
            limit = max_per_type.get(type_key, 10)
            if type_counts[type_key] >= limit:
                continue

            pverts = _vertices_lookup.get(path_idx)
            if pverts and len(pverts) >= 1:
                R = np.asarray(pverts[0], dtype=float)
            else:
                aod_a = float(aod_az[path_idx]) if aod_az is not None else 0.0
                aod_e = float(aod_el[path_idx]) if aod_el is not None else 0.0
                aoa_a = float(aoa_az[path_idx]) if aoa_az is not None else 0.0
                aoa_e = float(aoa_el[path_idx]) if aoa_el is not None else 0.0
                R = _reconstruct_reflection_point(
                    tx_pos, rx_pos, aod_a, aod_e, aoa_a, aoa_e, distance_m,
                )
                if R is None:
                    R = (tx_pos + rx_pos) / 2.0
            type_counts[type_key] += 1
            j = type_counts[type_key]
            _min_o = _min_ord_3d.get(type_key, 1)
            b1_dash = "solid" if order == _min_o else "dash"
            t = go.Scatter3d(
                x=[tx_pos[0], R[0], rx_pos[0]],
                y=[tx_pos[1], R[1], rx_pos[1]],
                z=[tx_pos[2], R[2], rx_pos[2]],
                mode="lines",
                line=dict(color=style["color"], width=style["width"] + 1.2, dash=b1_dash),
                opacity=0.85,
                name=style["label"] if j == 1 else None,
                legendgroup=type_key, showlegend=False,
                hovertemplate=f"{style['label']} 1b  {distance_m:.1f}m  "
                              f"R=[{R[0]:.1f},{R[1]:.1f},{R[2]:.1f}]<extra></extra>",
            )
            traces.append(t)
            trace_idx_by_type[type_key].append(len(traces) - 1)
            t2 = go.Scatter3d(
                x=[R[0]], y=[R[1]], z=[R[2]],
                mode="markers",
                marker=dict(size=3.5, color=style["color"], symbol="circle",
                           line=dict(color="#000000", width=0.5)),
                opacity=0.85,
                showlegend=False, legendgroup=type_key,
                hovertemplate=f"{style['label']} pt  "
                              f"R=[{R[0]:.1f},{R[1]:.1f},{R[2]:.1f}]<extra></extra>",
            )
            traces.append(t2)
            trace_idx_by_type[type_key].append(len(traces) - 1)
            continue

        # ── multi-bounce ───────────────────────────────────────────────────
        if order >= 2:
            limit = max_per_type.get(type_key, 10)
            if type_counts[type_key] >= limit:
                continue
            type_counts[type_key] += 1

            pverts = _vertices_lookup.get(path_idx)
            if pverts and len(pverts) >= 2:
                px = [tx_pos[0]] + [v[0] for v in pverts] + [rx_pos[0]]
                py = [tx_pos[1]] + [v[1] for v in pverts] + [rx_pos[1]]
                pz = [tx_pos[2]] + [v[2] for v in pverts] + [rx_pos[2]]
                opacity = 0.65
            else:
                px = [tx_pos[0], rx_pos[0]]
                py = [tx_pos[1], rx_pos[1]]
                pz = [tx_pos[2], rx_pos[2]]
                opacity = 0.35

            _min_o = _min_ord_3d.get(type_key, 2)
            mb_dash = "solid" if order == _min_o else "dash"
            t = go.Scatter3d(
                x=px, y=py, z=pz,
                mode="lines",
                line=dict(color=style["color"], width=1.8, dash=mb_dash),
                opacity=opacity,
                showlegend=False, legendgroup=type_key,
                hovertemplate=f"{style['label']} {order}b  {distance_m:.1f}m  "
                              f"{20 * np.log10(gain_abs + 1e-20):.0f}dB<extra></extra>",
            )
            traces.append(t)
            trace_idx_by_type[type_key].append(len(traces) - 1)

            if pverts and len(pverts) >= 1:
                v0 = pverts[0]
                t_marker = go.Scatter3d(
                    x=[v0[0]], y=[v0[1]], z=[v0[2]],
                    mode="markers",
                    marker=dict(size=3.5, color=style["color"], symbol="circle",
                               line=dict(color="#000000", width=0.5)),
                    opacity=0.80,
                    showlegend=False, legendgroup=type_key,
                    hovertemplate=f"{style['label']} v0  "
                                  f"[{v0[0]:.2f},{v0[1]:.2f},{v0[2]:.2f}]<extra></extra>",
                )
                traces.append(t_marker)
                trace_idx_by_type[type_key].append(len(traces) - 1)

    # ── TX marker ──────────────────────────────────────────────────────────
    traces.append(go.Scatter3d(
        x=[tx_pos[0]], y=[tx_pos[1]], z=[tx_pos[2]],
        mode="markers+text",
        marker=dict(size=6, color="#E60000", symbol="diamond",
                     line=dict(color="#800000", width=1)),
        text=["Tx"], textposition="top center",
        textfont=dict(size=10, color="#E60000", family="Arial Black"),
        name="Tx", showlegend=False,
        hovertemplate=f"Tx [{tx_pos[0]:.1f}, {tx_pos[1]:.1f}, {tx_pos[2]:.1f}]<extra></extra>",
    ))
    traces.append(go.Scatter3d(
        x=[rx_pos[0]], y=[rx_pos[1]], z=[rx_pos[2]],
        mode="markers+text",
        marker=dict(size=6, color="#0060FF", symbol="circle",
                     line=dict(color="#001860", width=1)),
        text=["Rx"], textposition="top center",
        textfont=dict(size=10, color="#0060FF", family="Arial Black"),
        name="Rx", showlegend=False,
        hovertemplate=f"Rx [{rx_pos[0]:.1f}, {rx_pos[1]:.1f}, {rx_pos[2]:.1f}]<extra></extra>",
    ))

    return traces, trace_idx_by_type


# ═══════════════════════════════════════════════════════════════════════════════
# RadioMap trace builder
# ═══════════════════════════════════════════════════════════════════════════════


def _build_radiomap_trace(
    scene_name: str, tx_pos: np.ndarray, rx_pos: np.ndarray, max_depth: int = 3,
    xml_path: str | None = None,
) -> tuple[Any, list[int]]:
    """Compute RadioMap and return a Plotly Surface trace."""
    import plotly.graph_objects as go

    try:
        rm_data = _compute_radiomap(scene_name, tx_pos, rx_pos,
                                     max_depth=max_depth, cell_size=None,
                                     num_samples=200000, xml_path=xml_path)
    except Exception:
        return None, []

    if rm_data is None:
        return None, []

    pg_db = rm_data["path_gain_db"]
    centers = rm_data["cell_centers"]
    x_1d = centers[0, :, 0]
    y_1d = centers[:, 0, 1]
    z_2d = np.full_like(pg_db, rm_data["center"][2])
    SENTINEL = -200.0
    pg_db = np.where(np.isfinite(pg_db), pg_db, SENTINEL)
    rm_min = float(np.nanpercentile(pg_db, 2))
    rm_max = float(np.nanpercentile(pg_db, 98))
    if rm_min < -130.0:
        rm_min = -130.0
    if rm_max <= rm_min:
        rm_max = rm_min + 60.0

    trace = go.Surface(
        x=x_1d, y=y_1d, z=z_2d,
        surfacecolor=pg_db,
        colorscale=[[0, 'black'], [0.001, 'darkblue'], [0.25, 'blue'],
                     [0.5, 'green'], [0.75, 'yellow'], [1, 'red']],
        cmin=rm_min, cmax=rm_max,
        colorbar=dict(
            title=dict(text="Path Gain [dB]", font=dict(size=10)),
            x=1.02, len=0.5, thickness=15,
        ),
        name="RadioMap",
        legendgroup="radiomap",
        showlegend=True,
        visible="legendonly",
        hoverinfo="skip",
        opacity=0.7,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
    )
    return trace, [0]


# ═══════════════════════════════════════════════════════════════════════════════
# Multipath identification row
# ═══════════════════════════════════════════════════════════════════════════════


def _build_multipath_row2(
    fig, cir_data: dict, multipath_results: dict[str, dict[str, dict]],
    colors_proto: dict, protocols: list[str], mp_row: int, global_max: float,
    first_path_viz: dict[str, dict[str, float]] | None = None,
    algos: list[str] | None = None,
) -> None:
    """Add multipath identification traces + first-path detection markers to row 2."""
    import plotly.graph_objects as go

    fp_algo_name = "Threshold(0.18)"
    fp_y_max = global_max * 1.02

    for proto in protocols:
        cd = cir_data[proto]
        color = colors_proto.get(proto, "#333333")
        t_disc = cd["t_discrete_s"]
        cir_disc = cd["cir_observed_discrete"]
        true_tau_ns = cd.get("true_first_tau_s", 0.0) * 1e9

        # Background: discrete CIR curve
        fig.add_trace(go.Scatter(
            x=t_disc * 1e9, y=np.abs(cir_disc),
            mode="lines", line=dict(color=color, width=1.2),
            name=f"{proto} CIR", legendgroup=f"mp_{proto}", showlegend=True,
            hovertemplate="%{x:.2f} ns  |CIR|=%{y:.4f}<extra></extra>",
        ), row=mp_row, col=2)

        # GT true-range dashed line (same reference as CIR row)
        if true_tau_ns > 0:
            fig.add_trace(go.Scatter(
                x=[true_tau_ns, true_tau_ns], y=[0, fp_y_max],
                mode="lines", line=dict(color="black", width=2.5, dash="dash"),
                name="GT (true range)", legendgroup="truth_mp",
                showlegend=(proto == protocols[0]),
                hovertemplate="True range = %{x:.2f} ns<extra></extra>",
            ), row=mp_row, col=2)

        proto_mp = multipath_results.get(proto, {})
        _reserved_keys = {"gt_paths", "match", "detected_paths", "result"}
        mp_algos_here = [k for k in proto_mp if k not in _reserved_keys]
        if not mp_algos_here:
            continue
        first_algo = mp_algos_here[0]
        mp_data = proto_mp.get(first_algo, {})

        # Ground truth paths
        gt_data = mp_data.get("gt_paths", {})
        gt_tau_ns = np.asarray(gt_data.get("tau_s", [])) * 1e9
        gt_gains = np.asarray(gt_data.get("gains", []))
        gt_types = gt_data.get("types", [])
        gt_orders = gt_data.get("orders", [])

        match_data = mp_data.get("match", {})
        fa_det_indices = set(match_data.get("false_alarms", []))

        # ── GT path stem markers — top 10 by gain, colored by TYPE ──────────
        active = np.abs(gt_gains) > 1e-12
        if np.any(active):
            active_indices = np.flatnonzero(active)
            _min_order: dict[str, int] = {}
            for i in active_indices:
                pt = str(gt_types[i]) if gt_types is not None and i < len(gt_types) else "reflection"
                od = int(gt_orders[i]) if gt_orders is not None and i < len(gt_orders) else 0
                tk = _map_path_type(pt, od)
                if tk not in _min_order or od < _min_order[tk]:
                    _min_order[tk] = od

            sorted_idx = np.argsort(np.abs(gt_gains[active_indices]))[::-1][:10]
            top_gt = active_indices[sorted_idx]
            max_gain = float(np.max(np.abs(gt_gains[active])))
            type_shown: dict[str, bool] = {}

            for gi in top_gt:
                idx = int(gi)
                tau_ns_val = float(gt_tau_ns[idx]) if gt_tau_ns.size > 0 else 0.0
                gain_val = float(np.abs(gt_gains[idx]))
                marker_h = (gain_val / max_gain) * global_max * 0.85 if max_gain > 0 else 0.0

                ptype_raw = str(gt_types[idx]) if gt_types is not None and idx < len(gt_types) else "reflection"
                order = int(gt_orders[idx]) if gt_orders is not None and idx < len(gt_orders) else 0
                tkey = _map_path_type(ptype_raw, order)
                style = _TYPE_STYLES[tkey]

                first_of_type = tkey not in type_shown
                if first_of_type:
                    type_shown[tkey] = True
                label = f"{style['label']}" if order == 0 else f"{style['label']} {order}b"

                min_ord = _min_order.get(tkey, 1)
                if tkey == "fix":
                    stem_dash = "solid"
                    stem_width = 2.2
                elif order == 0 or order == min_ord:
                    stem_dash = "solid"
                    stem_width = 2.5 if order == 0 else 2.2
                else:
                    stem_dash = "dash"
                    stem_width = 2.2

                fig.add_trace(go.Scatter(
                    x=[tau_ns_val, tau_ns_val], y=[0, marker_h],
                    mode="lines",
                    line=dict(color=style["color"], width=stem_width, dash=stem_dash),
                    name=f"GT {label}", legendgroup=f"mp_gt_{tkey}",
                    showlegend=first_of_type,
                    hovertemplate=(
                        f"GT {label}  τ=%{{x:.2f}}ns  "
                        f"|g|={gain_val:.4f}<extra></extra>"
                    ),
                ), row=mp_row, col=2)

        # ── Detected path markers (▼) — placed ON the CIR curve ─────────────
        det_paths = mp_data.get("detected_paths", [])
        det_shown = False
        fa_shown = False
        for j, dp in enumerate(det_paths):
            tau_ns_val = float(dp["estimated_tof_s"]) * 1e9
            t_arr_ns = np.asarray(t_disc) * 1e9
            bin_idx = np.argmin(np.abs(t_arr_ns - tau_ns_val))
            cir_val_at_tau = float(np.abs(cir_disc[bin_idx])) if bin_idx < len(cir_disc) else 0.0

            if j in fa_det_indices:
                marker_color = color
                marker_symbol = "triangle-down-open"
                fa_label = "False Alarm"
                show = not fa_shown
                fa_shown = True
            else:
                marker_color = color
                marker_symbol = "triangle-down"
                fa_label = "Detected"
                show = not det_shown
                det_shown = True

            fig.add_trace(go.Scatter(
                x=[tau_ns_val], y=[cir_val_at_tau],
                mode="markers",
                marker=dict(size=9, color=marker_color, symbol=marker_symbol,
                           line=dict(color="#000000", width=0.8)),
                name=fa_label, legendgroup="mp_det",
                showlegend=show,
                hovertemplate=(
                    f"Detected tau={tau_ns_val:.2f}ns  "
                    f"|CIR|={cir_val_at_tau:.4f}<extra></extra>"
                ),
            ), row=mp_row, col=2)

        # ── algorithm first-path estimate (dotted to distinguish from GT) ────
        if first_path_viz is not None:
            fp_proto = first_path_viz.get(proto, {})
            tau_s = fp_proto.get(fp_algo_name)
            if tau_s is not None and not np.isnan(tau_s):
                tau_ns_val = float(tau_s) * 1e9
                fig.add_trace(go.Scatter(
                    x=[tau_ns_val, tau_ns_val], y=[0, fp_y_max],
                    mode="lines",
                    line=dict(color=color, width=1.2, dash="dot"),
                    name=f"{fp_algo_name} est", legendgroup=f"fp_{proto}",
                    showlegend=True,
                    hovertemplate=(
                        f"{fp_algo_name} est. tau0={tau_ns_val:.2f}ns"
                        f"<extra></extra>"
                    ),
                ), row=mp_row, col=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Path data tables
# ═══════════════════════════════════════════════════════════════════════════════


def _build_path_tables(truth_first, protocols: list[str]) -> dict[str, list[str]]:
    """Build per-type path tables, each with top-10 by gain.  Returns {type_key: html_lines}."""
    a_paths = np.asarray(truth_first.a_paths)
    tau_paths_s = np.asarray(truth_first.tau_paths_s)
    active = np.abs(a_paths) > 1e-12
    active_indices = np.flatnonzero(active)

    type_buckets: dict[str, list[int]] = {}
    for idx in active_indices:
        raw = str(truth_first.path_type[idx]) if truth_first.path_type is not None else "reflection"
        ord_ = int(truth_first.path_order[idx]) if truth_first.path_order is not None else 0
        tk = _map_path_type(raw, ord_)
        type_buckets.setdefault(tk, []).append(idx)

    tables: dict[str, list[str]] = {}
    display_order = ["los", "specular", "diffuse", "refraction", "diffraction", "fix"]

    for tk in display_order:
        style = _TYPE_STYLES[tk]
        color_hex = style["color"]
        lines: list[str] = []

        if tk not in type_buckets or not type_buckets[tk]:
            lines.append(f"<b style='color:{color_hex}'>{style['label']}</b>  "
                         f"<span style='font-size:7px'>(0/0)</span>")
            lines.append("<span style='font-size:7px'>rank  dB      r(m)   ord</span>")
            lines.append("<span style='font-size:7px; color:#9CA3AF;'>— no paths —</span>")
            tables[tk] = lines
            continue

        bucket = type_buckets[tk]
        bucket_sorted = sorted(bucket, key=lambda i: abs(a_paths[i]), reverse=True)[:10]

        lines.append(f"<b style='color:{color_hex}'>{style['label']}</b>  "
                     f"<span style='font-size:7px'>({len(bucket_sorted)}/{len(bucket)})</span>")
        lines.append("<span style='font-size:7px'>rank  dB      r(m)   ord</span>")

        for rank, idx in enumerate(bucket_sorted):
            gain_db = 20 * np.log10(np.abs(a_paths[idx]) + 1e-20)
            dist_m = tau_paths_s[idx] * _LIGHT_SPEED
            order = int(truth_first.path_order[idx]) if truth_first.path_order is not None else 0
            lines.append(
                f"<span style='color:{color_hex}'>#{rank+1:2d}</span> "
                f"{gain_db:5.0f}dB {dist_m:6.1f}m  {order}b"
            )
        tables[tk] = lines

    return tables


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HTML builder — the public entry point
# ═══════════════════════════════════════════════════════════════════════════════


def build_html_report(
    cache_path: Path,
    cir_data: dict[str, dict],
    errors: dict[str, dict[str, list[float]]],
    config_info: dict,
    scene_name: str,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    truth_first,
    multipath_results: dict[str, dict[str, dict]] | None = None,
    multipath_errors: dict[str, dict[str, list[float]]] | None = None,
    first_path_viz: dict[str, dict[str, float]] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Build the interactive HTML dashboard and return the output path.

    ALL logic is preserved character-for-character from the original
    ``scripts/rt_cache_interactive.py:_build_html()``.

    Parameters
    ----------
    cache_path : Path
        Used for scene.xml lookup and output filename derivation.
    cir_data : {protocol: {t_discrete_s, cir_observed_discrete, ...}}
    errors : {protocol: {algo_name: [error_m per trial]}}
    config_info : dict with keys "scene_name", "wall_geometry", etc.
    scene_name : str
    tx_pos, rx_pos : np.ndarray
    truth_first : ChannelTruth for the first trial
    multipath_results : optional {protocol: {algo: {result, match, ...}}}
    multipath_errors : optional {protocol: {algo: [error_m]}}
    first_path_viz : optional {protocol: {algo: estimated_tof_s}}
    output_dir : Path or None — directory for output HTML (default: outputs/interactive)

    Returns
    -------
    Path to the written HTML file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if output_dir is None:
        output_dir = Path("outputs/interactive")

    protocols = list(cir_data.keys())
    algos = list(errors[protocols[0]].keys())
    n_protos = len(protocols)
    mid = (tx_pos + rx_pos) / 2.0
    z_stretch = 3.0

    # ── load meshes ────────────────────────────────────────────────────────
    verts, i_f, j_f, k_f, _triangles = _load_meshes(scene_name, mid, max_tri_total=50000)

    # ── build 3D traces ────────────────────────────────────────────────────
    traces_3d, trace_idx_by_type = _build_3d_traces(
        scene_name, tx_pos, rx_pos, truth_first, mid,
        verts, i_f, j_f, k_f, z_stretch,
    )

    # ── RadioMap ───────────────────────────────────────────────────────────
    scene_xml_path = cache_path / "scene.xml"
    rm_trace, _rm_toggle_idx = _build_radiomap_trace(
        scene_name, tx_pos, rx_pos, max_depth=5,
        xml_path=str(scene_xml_path) if scene_xml_path.exists() else None,
    )

    # ── floorplan wall meshes ──────────────────────────────────────────────
    wall_geo = config_info.get("wall_geometry")
    wall_mesh_traces: list = []
    if wall_geo:
        wall_mesh_traces = _build_wall_meshes(wall_geo)

    # ── path data for CIR markers (gain-thresholded, top 10) ──────────────
    a_paths = np.asarray(truth_first.a_paths)
    tau_paths_s = np.asarray(truth_first.tau_paths_s)
    path_types = truth_first.path_type
    path_orders = truth_first.path_order
    abs_gains = np.abs(a_paths)
    active_mask = abs_gains > 1e-12
    max_gain = float(np.max(abs_gains)) if abs_gains.size > 0 else 1.0
    cir_marker_threshold_db = float(config_info.get("gain_threshold_db", -150.0))
    gain_floor = max_gain * (10.0 ** (cir_marker_threshold_db / 20.0)) if max_gain > 0 else 0.0
    active_mask = active_mask & (abs_gains >= gain_floor)
    sorted_by_gain = np.argsort(abs_gains[active_mask])[::-1]
    active_indices = np.flatnonzero(active_mask)
    top_n = min(10, len(sorted_by_gain))
    top_indices = active_indices[sorted_by_gain[:top_n]]
    top_gains = abs_gains[top_indices]
    top_tau_ns = tau_paths_s[top_indices] * 1e9
    top_types = [
        _map_path_type(str(path_types[i]) if path_types is not None else "reflection",
                       int(path_orders[i]) if path_orders is not None else 0)
        for i in top_indices
    ]
    top_orders = [
        int(path_orders[i]) if path_orders is not None else 0
        for i in top_indices
    ]

    _min_order_by_type: dict[str, int] = {}
    for tkey, order in zip(top_types, top_orders):
        if order == 0:
            continue
        if tkey not in _min_order_by_type or order < _min_order_by_type[tkey]:
            _min_order_by_type[tkey] = order

    # ── subplot grid: 3 rows (CIR toggle, multipath ID, CDF) ──────────────
    total_rows = 3
    subplot_titles = [
        "",
        "<b>CIR</b> — Discrete / Continuous",
        "<b>Multipath Identification</b>",
        "<b>Sampling-Ranging Results</b>",
        "<b>Ranging Error CDF</b>",
    ]

    specs: list[list[dict]] = [
        [{"type": "scene", "rowspan": 3}, {"type": "xy", "colspan": 2}, None],
        [None, {"type": "xy", "colspan": 2}, None],
        [None, {"type": "xy"}, {"type": "xy"}],
    ]

    col_widths = [0.60, 0.20, 0.20]
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=subplot_titles,
        specs=specs,
        column_widths=col_widths,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    # ── add 3D traces ──────────────────────────────────────────────────────
    for t in wall_mesh_traces:
        fig.add_trace(t, row=1, col=1)
    for t in traces_3d:
        fig.add_trace(t, row=1, col=1)

    rm_trace_idx_in_fig: list[int] = []
    if rm_trace is not None:
        fig.add_trace(rm_trace, row=1, col=1)
        rm_trace_idx_in_fig.append(len(fig.data) - 1)

    trace_idx_by_type_in_fig: dict[str, list[int]] = {
        k: [] for k in _TYPE_STYLES
    }
    for i, t in enumerate(fig.data):
        lg = t.legendgroup or ""
        if lg in trace_idx_by_type_in_fig:
            trace_idx_by_type_in_fig[lg].append(i)

    # ── CIR rows ──────────────────────────────────────────────────────────
    colors_proto = _COLORS_PROTO

    cir_row = 1
    mp_row = 2
    disc_only_indices: list[int] = []
    cont_only_indices: list[int] = []

    global_disc_max = max(
        float(np.max(np.abs(cir_data[p]["cir_observed_discrete"])))
        if cir_data[p]["cir_observed_discrete"].size > 0 else 0.0
        for p in protocols
    )
    global_cont_max = max(
        float(np.max(np.abs(cir_data[p]["cir_observed_cont"])))
        if cir_data[p]["cir_observed_cont"].size > 0 else 0.0
        for p in protocols
    )

    for pi, proto in enumerate(protocols):
        cd = cir_data[proto]
        color = colors_proto.get(proto, "#333333")
        t_disc = cd["t_discrete_s"]
        cir_disc = cd["cir_observed_discrete"]
        t_cont = cd["t_cont_s"]
        cir_cont = cd["cir_observed_cont"]
        true_tau_ns = cd.get("true_first_tau_s", 0.0) * 1e9

        # ── Row 1: discrete CIR traces ──────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=t_disc * 1e9, y=np.abs(cir_disc),
            mode="lines+markers", marker=dict(size=3),
            line=dict(color=color, width=1.5),
            name=f"{proto} observed", legendgroup=proto, showlegend=True,
            hovertemplate="%{x:.2f} ns  |CIR|=%{y:.4f}<extra></extra>",
        ), row=cir_row, col=2)
        disc_only_indices.append(len(fig.data) - 1)

        if true_tau_ns > 0 and pi == 0:
            fig.add_trace(go.Scatter(
                x=[true_tau_ns, true_tau_ns], y=[0, global_disc_max * 1.05],
                mode="lines", line=dict(color="black", width=2.5, dash="dash"),
                name="GT (true range)", legendgroup="truth", showlegend=True,
                hovertemplate="True range = %{x:.2f} ns<extra></extra>",
            ), row=cir_row, col=2)

        # ── Row 1: continuous CIR traces (initially hidden) ─────────────────
        fig.add_trace(go.Scatter(
            x=t_cont * 1e9, y=np.abs(cir_cont),
            mode="lines", line=dict(color=color, width=1.5),
            fill='tozeroy',
            fillcolor=f"rgba({','.join([str(int(color[i:i+2], 16)) for i in (1, 3, 5)])}, 0.08)",
            name=f"{proto} cont", legendgroup=proto, showlegend=False,
            visible=False,
            hovertemplate="%{x:.2f} ns  |CIR|=%{y:.4f}<extra></extra>",
        ), row=cir_row, col=2)
        cont_only_indices.append(len(fig.data) - 1)

        if true_tau_ns > 0 and pi == 0:
            fig.add_trace(go.Scatter(
                x=[true_tau_ns, true_tau_ns], y=[0, global_cont_max * 1.05],
                mode="lines", line=dict(color="black", width=2.5, dash="dash"),
                name="GT (true range)", legendgroup="truth", showlegend=False,
                visible=False,
                hovertemplate="True range = %{x:.2f} ns<extra></extra>",
            ), row=cir_row, col=2)
            cont_only_indices.append(len(fig.data) - 1)

    # ── multipath markers on discrete CIR (top 10) ─────────────────────────
    type_shown: dict[str, bool] = {}
    cir_disc_first = cir_data[protocols[0]]["cir_observed_discrete"]
    cir_disc_max0 = float(np.max(np.abs(cir_disc_first))) if cir_disc_first.size > 0 else 1.0
    for rank, (path_idx, gain, tau_ns, tkey, order) in enumerate(zip(
        top_indices, top_gains, top_tau_ns, top_types, top_orders,
    )):
        style = _TYPE_STYLES[tkey]
        marker_h = (gain / top_gains[0]) * cir_disc_max0 * 0.85 if top_gains[0] > 0 else 0.0
        if marker_h <= 0:
            continue
        first_of_type = tkey not in type_shown
        if first_of_type:
            type_shown[tkey] = True
        label = f"{style['label']} #{rank + 1}" if order == 0 else f"{style['label']} {order}b #{rank + 1}"
        min_ord = _min_order_by_type.get(tkey, 1)
        if tkey == "fix":
            marker_dash = "solid"
            marker_width = 2.2
        elif order == 0 or order == min_ord:
            marker_dash = "solid"
            marker_width = 2.5 if order == 0 else 2.2
        else:
            marker_dash = "dash"
            marker_width = 2.2
        fig.add_trace(go.Scatter(
            x=[tau_ns, tau_ns], y=[0, marker_h],
            mode="lines",
            line=dict(color=style["color"], width=marker_width, dash=marker_dash),
            name=f"τ {label}", legendgroup=f"tau_{tkey}",
            showlegend=(first_of_type),
            hovertemplate=(
                f"{label}  τ=%{{x:.2f}}ns  "
                f"|g|=%{{customdata[0]:.4f}}  {20*np.log10(gain+1e-20):.0f}dB"
                f"<extra></extra>"
            ),
            customdata=[[float(gain)]],
        ), row=cir_row, col=2)
        disc_only_indices.append(len(fig.data) - 1)

    _all_active_tau = tau_paths_s[active_mask]
    if _all_active_tau.size > 0:
        _max_tau_ns = float(np.max(_all_active_tau)) * 1e9
        _cir_x_max = min(_max_tau_ns * 1.15 + 10.0, _max_tau_ns + 50.0)
    else:
        _cir_x_max = 100.0

    fig.update_xaxes(title_text="Time (ns)", row=cir_row, col=2, range=[0, _cir_x_max])
    fig.update_yaxes(title_text="|CIR|", row=cir_row, col=2)

    # ── Row 2: Multipath Identification ────────────────────────────────────
    if multipath_results is not None:
        _build_multipath_row2(fig, cir_data, multipath_results, colors_proto,
                              protocols, mp_row, global_disc_max,
                              first_path_viz=first_path_viz, algos=algos)

    fig.update_xaxes(title_text="Time (ns)", row=mp_row, col=2, range=[0, _cir_x_max])
    fig.update_yaxes(title_text="|CIR|", row=mp_row, col=2)

    # ── Row 3: Scatter + CDF ───────────────────────────────────────────────
    result_row = 3

    trace_owner = []
    for t in fig.data:
        lg = t.legendgroup or ""
        if lg in ["uwb", "wifi", "fiveg"]:
            trace_owner.append(lg)
        elif lg == "radiomap":
            trace_owner.append("radiomap")
        else:
            trace_owner.append("base")

    symbols_algo = {
        "MaxPeak": "circle",
        "Threshold(0.18)": "square",
        "LeadingEdge(4σ)": "diamond",
        "SearchBack(0.18)": "triangle-up",
        "ChipLDE(10dB)": "cross",
    }

    dash_algo = {
        "MaxPeak": "dot",
        "Threshold(0.18)": "dash",
        "LeadingEdge(4σ)": "solid",
        "SearchBack(0.18)": "dashdot",
        "ChipLDE(10dB)": "longdash",
    }

    # Scatter plot
    for algo in algos:
        for proto in protocols:
            errs = errors[proto].get(algo, [])
            if not errs:
                continue
            x_vals = list(range(1, len(errs) + 1))
            proto_color = colors_proto.get(proto, "#333333")

            fig.add_trace(go.Scatter(
                x=x_vals, y=errs, mode="markers",
                marker=dict(
                    color=proto_color,
                    size=5,
                    symbol=symbols_algo.get(algo, "circle")
                ),
                name=f"{proto.upper()} ({algo})",
                legendgroup=f"{proto}_{algo}",
                showlegend=True,
                hovertemplate="trial %{x}  err=%{y:.4f}m<extra></extra>",
            ), row=result_row, col=2)
            trace_owner.append(proto)

    fig.update_xaxes(title_text="Trial #", row=result_row, col=2)
    fig.update_yaxes(title_text="Ranging Error (m)", row=result_row, col=2)

    # CDF curves
    for algo in algos:
        for proto in protocols:
            errs = errors[proto].get(algo, [])
            if not errs:
                continue
            sorted_errs = np.sort(np.abs(errs))
            cdf_y = np.linspace(0, 1, len(sorted_errs))
            proto_color = colors_proto.get(proto, "#333333")

            fig.add_trace(go.Scatter(
                x=sorted_errs, y=cdf_y, mode="lines",
                line=dict(
                    color=proto_color,
                    width=2,
                    dash=dash_algo.get(algo, "solid")
                ),
                name=f"{proto.upper()} ({algo})",
                legendgroup=f"{proto}_{algo}",
                showlegend=False,
                hovertemplate="|err|=%{x:.3f}m  P=%{y:.2f}<extra></extra>",
            ), row=result_row, col=3)
            trace_owner.append(proto)

    fig.update_xaxes(title_text="|Error| (m)", row=result_row, col=3, autorange=True)
    fig.update_yaxes(title_text="CDF", row=result_row, col=3, range=[0, 1.05])

    # ── path tables ────────────────────────────────────────────────────────
    path_tables = _build_path_tables(truth_first, protocols)

    type_x_positions = {
        "los": 0.00, "specular": 0.09, "diffuse": 0.18,
        "refraction": 0.27, "diffraction": 0.36, "fix": 0.45,
    }

    for tk, lines in path_tables.items():
        xp = type_x_positions.get(tk, 0.00)
        fig.add_annotation(
            x=xp, y=-0.08, xref="paper", yref="paper",
            text="<br>".join(lines),
            showarrow=False,
            font=dict(size=10.5, family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace", color="#374151"),
            bgcolor="#F9FAFB",
            borderpad=6,
            bordercolor="#E5E7EB",
            borderwidth=1,
            align="left",
            xanchor="left", yanchor="top",
        )

    # ── RMSE/P90 stats table ───────────────────────────────────────────────
    stats_lines = ["<b>Ranging Performance: RMSE / P90 (m)</b>", ""]
    header = f"{'Algorithm':<18}" + "".join([f"{p.upper():<14}" for p in protocols])
    stats_lines.append(f"<b>{header.replace(' ', '&nbsp;')}</b>")

    for algo in algos:
        row_str = f"{algo[:16]:<18}"
        for proto in protocols:
            errs = errors[proto].get(algo, [])
            if not errs:
                row_str += f"{'—':<14}"
                continue
            arr = np.array(errs)
            rmse = float(np.sqrt(np.mean(arr ** 2)))
            p90 = float(np.percentile(np.abs(arr), 90))
            val_str = f"{rmse:.2f}/{p90:.2f}"
            row_str += f"{val_str:<14}"
        stats_lines.append(row_str.replace(" ", "&nbsp;"))

    # ── multipath detection stats table ────────────────────────────────────
    if multipath_results is not None:
        _reserved_keys = {"gt_paths", "match", "detected_paths", "result"}
        mp_algos = sorted(set(
            a for p in protocols
            for a in multipath_results.get(p, {})
            if a not in _reserved_keys
        ))
        mp_stats_lines = ["<b>Multipath Detection</b>  (direct + 1b specular)", ""]

        total_gt = 0
        for algo in mp_algos:
            for proto in protocols:
                mp_data = multipath_results.get(proto, {}).get(algo, {})
                match = mp_data.get("match", {})
                total_gt += match.get("n_gt", 0)

        if total_gt == 0:
            mp_stats_lines.append(
                "<span style='color:#9CA3AF;'>"
                "No evaluable paths — no LOS or 1-bounce specular in this scene"
                "</span>"
            )
        else:
            ALGO_W = 16
            COL_W = 15
            mp_header = f"{'Algorithm':<{ALGO_W}}" + "".join([f"{p.upper():<{COL_W}}" for p in protocols])
            mp_stats_lines.append(f"<b>{mp_header.replace(' ', '&nbsp;')}</b>")

            for algo in mp_algos:
                row_str = f"{algo[:14]:<{ALGO_W}}"
                for proto in protocols:
                    mp_data = multipath_results.get(proto, {}).get(algo, {})
                    match = mp_data.get("match", {})
                    n_gt = match.get("n_gt", 0)
                    n_hit = len(match.get("hits", []))
                    n_fa = len(match.get("false_alarms", []))
                    rmse = match.get("hit_rmse_m", 0.0)
                    if n_gt == 0:
                        val_str = "—"
                    elif n_hit > 0:
                        val_str = f"{n_hit}/{n_gt} ±{rmse:.2f} FA:{n_fa}"
                    else:
                        val_str = f"0/{n_gt} FA:{n_fa}"
                    row_str += f"{val_str:<{COL_W}}"
                mp_stats_lines.append(row_str.replace(" ", "&nbsp;"))

        fig.add_annotation(
            x=0.52, y=-0.08, xref="paper", yref="paper",
            text="<br>".join(mp_stats_lines),
            showarrow=False,
            font=dict(size=10.5, family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace", color="#1F2937"),
            bgcolor="#F9FAFB",
            borderpad=10,
            bordercolor="#D1D5DB",
            borderwidth=1,
            align="left",
            xanchor="left", yanchor="top",
        )

    # RMSE/P90 text box
    fig.add_annotation(
        x=1.0, y=-0.08, xref="paper", yref="paper",
        text="<br>".join(stats_lines),
        showarrow=False,
        font=dict(size=10.5, family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace", color="#1F2937"),
        bgcolor="#F9FAFB",
        borderpad=10,
        bordercolor="#D1D5DB",
        borderwidth=1,
        align="left",
        xanchor="right", yanchor="top",
    )

    # ── protocol + algorithm filter dropdowns ──────────────────────────────
    scene_label = config_info.get("scene_name", "") or config_info.get("scene", "unknown")

    # Build trace classification arrays for filtering
    trace_protos = []
    trace_algos = []

    for t in fig.data:
        lg = t.legendgroup or ""
        found_algo = None
        for a in algos:
            if lg.endswith(f"_{a}"):
                found_algo = a
                break
        if found_algo:
            trace_algos.append(found_algo)
            p = lg.split(f"_{found_algo}")[0]
            trace_protos.append(p)
        else:
            if lg == "radiomap":
                trace_algos.append("radiomap")
                trace_protos.append("radiomap")
            elif lg in ["uwb", "wifi", "fiveg"]:
                trace_algos.append("base")
                trace_protos.append(lg)
            else:
                trace_algos.append("base")
                trace_protos.append("base")

    # Protocol filter buttons
    protocol_buttons = [
        dict(
            label="All", method="update",
            args=[
                {"visible": [True if x != "radiomap" else "legendonly" for x in trace_protos]},
                {"title": dict(text=f"<b>{scene_label}</b> <span style='color:#6B7280; font-size:12px;'> — All Sensors Dashboard</span>")}
            ]
        ),
        dict(
            label="UWB", method="update",
            args=[
                {"visible": [True if x in ["base", "uwb"] else ("legendonly" if x == "radiomap" else False) for x in trace_protos]},
                {"title": dict(text=f"<b>{scene_label}</b> <span style='color:#6B7280; font-size:12px;'> — UWB Only</span>")}
            ]
        ),
        dict(
            label="WiFi", method="update",
            args=[
                {"visible": [True if x in ["base", "wifi"] else ("legendonly" if x == "radiomap" else False) for x in trace_protos]},
                {"title": dict(text=f"<b>{scene_label}</b> <span style='color:#6B7280; font-size:12px;'> — WiFi Only</span>")}
            ]
        ),
        dict(
            label="5G NR", method="update",
            args=[
                {"visible": [True if x in ["base", "fiveg"] else ("legendonly" if x == "radiomap" else False) for x in trace_protos]},
                {"title": dict(text=f"<b>{scene_label}</b> <span style='color:#6B7280; font-size:12px;'> — 5G NR Only</span>")}
            ]
        )
    ]

    # Algorithm filter buttons
    algo_buttons = [
        dict(
            label="All", method="update",
            args=[
                {"visible": [True if x != "radiomap" else "legendonly" for x in trace_algos]},
                {"title": dict(text=f"<b>{scene_label}</b> <span style='color:#6B7280; font-size:12px;'> — All Algorithms</span>")}
            ]
        )
    ]
    for algo in algos:
        algo_buttons.append(
            dict(
                label=algo, method="update",
                args=[
                    {"visible": [True if x in ["base", algo] else ("legendonly" if x == "radiomap" else False) for x in trace_algos]},
                    {"title": dict(text=f"<b>{scene_label}</b> <span style='color:#6B7280; font-size:12px;'> — [{algo}] only</span>")}
                ]
            )
        )

    # Zoom + Z buttons
    zoom_margin = max(3.0, float(np.linalg.norm(tx_pos - rx_pos)) * 2.5)
    zoom_buttons = [_make_zoom_button(f"±{z:.0f}m", z, mid, z_stretch) for z in [30, 100, 300]]
    z_buttons = [_make_z_stretch_button(f"Z ×{int(s)}", s) for s in [1, 3, 8]]

    # Path toggle buttons
    path_toggle_buttons = []
    for tk in ["los", "specular", "diffuse", "refraction", "diffraction", "fix"]:
        indices = trace_idx_by_type_in_fig.get(tk, [])
        if indices:
            path_toggle_buttons.append(
                dict(label=_TYPE_STYLES[tk]["label"], method="restyle",
                     args=[{"visible": "legendonly"}, indices], args2=[{"visible": True}, indices])
            )
    if rm_trace_idx_in_fig:
        path_toggle_buttons.append(
            dict(label="RadioMap", method="restyle", args=[{"visible": "legendonly"}, rm_trace_idx_in_fig], args2=[{"visible": True}, rm_trace_idx_in_fig])
        )

    updatemenus = [
        _make_cir_toggle_dropdown(disc_only_indices, cont_only_indices),
        dict(type="dropdown", buttons=protocol_buttons, direction="down",
             showactive=True, x=0.38, y=1.00, xanchor="left", yanchor="top"),
        dict(type="dropdown", buttons=algo_buttons, direction="down",
             showactive=True, x=0.44, y=1.00, xanchor="left", yanchor="top"),
        dict(type="buttons", buttons=zoom_buttons, direction="right", pad=dict(r=2, t=4), showactive=True, x=0.005, y=1.00, xanchor="left", yanchor="top"),
        dict(type="buttons", buttons=z_buttons, direction="right", pad=dict(r=2, t=4), showactive=True, x=0.005, y=0.94, xanchor="left", yanchor="top"),
    ]

    if path_toggle_buttons:
        split_idx = 3
        row1_buttons = path_toggle_buttons[:split_idx]
        row2_buttons = path_toggle_buttons[split_idx:]

        if row1_buttons:
            updatemenus.append(
                dict(type="buttons", buttons=row1_buttons, direction="right", pad=dict(r=2, t=4), x=0.14, y=1.00, xanchor="left", yanchor="top")
            )
        if row2_buttons:
            updatemenus.append(
                dict(type="buttons", buttons=row2_buttons, direction="right", pad=dict(r=2, t=4), x=0.14, y=0.94, xanchor="left", yanchor="top")
            )

    # Layout
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#1F2937"),
        title=dict(
            text=f"<b>{scene_label}</b> <span style='color:#6B7280; font-size:12px;'> — Ray-Tracing Simulation & Channel Estimation Dashboard</span>",
            font=dict(size=16, color="#111827"),
            x=0.02, xanchor="left", y=0.98, yanchor="top"
        ),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="#F9FAFB", gridcolor="#E5E7EB", showbackground=True),
            yaxis=dict(title="Y (m)", backgroundcolor="#F9FAFB", gridcolor="#E5E7EB", showbackground=True),
            zaxis=dict(title="Z (m)", backgroundcolor="#F1F5F9", gridcolor="#E5E7EB", showbackground=True),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.0 / z_stretch),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        updatemenus=updatemenus,
        showlegend=True,
        legend=dict(
            orientation="v", yanchor="top", y=0.95, xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.95)", bordercolor="#E5E7EB", borderwidth=1, font=dict(size=10)
        ),
        height=max(1100, 300 + 250 * n_protos),
        margin=dict(l=20, r=150, t=110, b=260),
    )

    for menu in updatemenus:
        menu.update(bgcolor="#FFFFFF", bordercolor="#D1D5DB", font=dict(size=10, color="#374151"))

    fig.add_annotation(x=0.001, y=1.03, xref="paper", yref="paper", text="<b>Zoom / Z:</b>", showarrow=False, font=dict(size=9, color="#555555"))
    if path_toggle_buttons:
        fig.add_annotation(x=0.135, y=1.03, xref="paper", yref="paper", text="<b>Paths:</b>", showarrow=False, font=dict(size=9, color="#555555"))
    fig.add_annotation(x=0.265, y=1.03, xref="paper", yref="paper", text="<b>CIR Mode:</b>", showarrow=False, font=dict(size=9, color="#555555"))
    fig.add_annotation(x=0.39, y=1.03, xref="paper", yref="paper", text="<b>Sensor:</b>", showarrow=False, font=dict(size=9, color="#555555"))
    fig.add_annotation(x=0.45, y=1.03, xref="paper", yref="paper", text="<b>Algorithm:</b>", showarrow=False, font=dict(size=9, color="#555555"))

    out_path = output_dir / f"{cache_path.name}_chip_sim.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), auto_play=False, include_plotlyjs="cdn")

    return out_path
