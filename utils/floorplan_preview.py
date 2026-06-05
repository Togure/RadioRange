"""Preview extracted walls from a floorplan PNG before running expensive RT.

Usage:
  .venv/bin/python3 utils/floorplan_preview.py floorplans/my_office.png
  .venv/bin/python3 utils/floorplan_preview.py floorplans/my_office.png --ppm 25 --wall-height 2.8
  .venv/bin/python3 utils/floorplan_preview.py floorplans/my_office.png --dump-config my_office.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Default color → material mapping (user can override with --color)
_DEFAULT_COLORS: list[dict] = [
    {"color": [0, 0, 0],       "material": "itu_concrete",      "label": "Concrete"},
    {"color": [128, 64, 0],    "material": "itu_brick",         "label": "Brick"},
    {"color": [0, 0, 255],     "material": "itu_glass",         "label": "Glass"},
    {"color": [139, 90, 43],   "material": "itu_wood",          "label": "Wood"},
    {"color": [192, 192, 192], "material": "itu_metal",         "label": "Metal"},
    {"color": [255, 200, 100], "material": "itu_plasterboard",  "label": "Plasterboard"},
]

# Display colors for the preview plot (distinct from material colors)
_PREVIEW_COLORS = [
    "#404040", "#C48830", "#2563EB", "#8B6914", "#9CA3AF", "#F59E0B",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview floorplan wall extraction")
    parser.add_argument("image", help="Path to floorplan PNG")
    parser.add_argument("--ppm", type=float, default=20.0,
                        help="Pixels per meter (default: 20)")
    parser.add_argument("--wall-height", type=float, default=3.0,
                        help="Wall height in meters (default: 3.0)")
    parser.add_argument("--tolerance", type=float, default=40.0,
                        help="Color matching tolerance (default: 40)")
    parser.add_argument("--bg-color", default="255,255,255",
                        help="Background color R,G,B (default: 255,255,255)")
    parser.add_argument("--tx", default=None,
                        help="TX position x,y,z in meters (e.g., 2,1,1.5)")
    parser.add_argument("--rx", default=None,
                        help="RX position x,y,z in meters")
    parser.add_argument("--dump-config", default=None, metavar="PATH",
                        help="Save a ready-to-use config YAML")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Show matplotlib preview (default: True)")
    parser.add_argument("--no-show", action="store_false", dest="show",
                        help="Skip preview, only print stats")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    bg_color = [int(x) for x in args.bg_color.split(",")]

    # Build floorplan config for the parser
    floorplan_cfg = {
        "image_path": str(image_path.resolve()),
        "pixels_per_meter": args.ppm,
        "wall_height_m": args.wall_height,
        "default_tolerance": args.tolerance,
        "color_mapping": _DEFAULT_COLORS,
        "background_color": bg_color,
        "generate_floor_ceiling": True,
        "floor_material": "itu_concrete",
        "ceiling_material": "itu_ceiling_board",
    }

    # ── Parse ────────────────────────────────────────────────────────────
    from environments.floorplan import _extract_walls, _parse_floorplan_image

    print(f"Loading: {image_path}")
    labels, material_by_label = _parse_floorplan_image(floorplan_cfg)
    walls = _extract_walls(labels, material_by_label, floorplan_cfg)

    # ── Stats ────────────────────────────────────────────────────────────
    H, W = labels.shape
    room_w = W / args.ppm
    room_h = H / args.ppm
    n_wall_pixels = int(np.sum(labels >= 0))
    n_bg_pixels = int(np.sum(labels < 0))

    print(f"\nImage: {W}×{H} px  →  {room_w:.1f}×{room_h:.1f} m  ({args.ppm} px/m)")
    print(f"Wall height: {args.wall_height} m")
    print(f"Wall pixels: {n_wall_pixels}  |  Background: {n_bg_pixels}")
    print(f"Extracted walls: {len(walls)}")

    material_counts: dict[str, int] = {}
    for w in walls:
        material_counts[w["material"]] = material_counts.get(w["material"], 0) + 1
    print(f"Materials: {dict(material_counts)}")

    total_wall_len_m = 0.0
    for w in walls:
        hx, hy, _ = w["half_extents"]
        total_wall_len_m += max(hx, hy) * 2  # longer dimension × 2
    print(f"Total wall length: {total_wall_len_m:.1f} m")

    # ── Preview plot ─────────────────────────────────────────────────────
    if args.show and len(walls) > 0:
        _draw_preview(labels, walls, material_by_label, floorplan_cfg, args)

    # ── Dump config ──────────────────────────────────────────────────────
    if args.dump_config:
        _dump_config(args, image_path, bg_color, room_w, room_h)


def _draw_preview(
    labels: np.ndarray,
    walls: list[dict],
    material_by_label: dict[int, str],
    floorplan_cfg: dict,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    ppm = floorplan_cfg["pixels_per_meter"]
    H, W = labels.shape

    fig, (ax_img, ax_walls) = plt.subplots(1, 2, figsize=(14, max(6, H / W * 7)))

    # Left: original image
    img = Image.open(args.image).convert("RGB")
    ax_img.imshow(img)
    ax_img.set_title("Original PNG", fontsize=11)
    ax_img.set_xlabel("px")
    ax_img.set_ylabel("px")

    # Right: extracted walls (world coordinates)
    ax_walls.set_xlim(0, W / ppm)
    ax_walls.set_ylim(H / ppm, 0)  # y inverted to match image convention
    ax_walls.set_aspect("equal")
    ax_walls.set_title(f"Extracted Walls ({len(walls)} segments)", fontsize=11)
    ax_walls.set_xlabel("X (m)")
    ax_walls.set_ylabel("Y (m)")
    ax_walls.grid(True, alpha=0.3)

    # Color mapping per material
    mat_labels = sorted(set(w["material"] for w in walls))
    mat_color = {}
    for i, m in enumerate(mat_labels):
        mat_color[m] = _PREVIEW_COLORS[i % len(_PREVIEW_COLORS)]

    for w in walls:
        cx, cy, _ = w["center"]
        hx, hy, _ = w["half_extents"]
        mat = w["material"]
        rect = patches.Rectangle(
            (cx - hx, cy - hy), 2 * hx, 2 * hy,
            linewidth=0.8, edgecolor="#333333",
            facecolor=mat_color.get(mat, "#AAAAAA"), alpha=0.7,
        )
        ax_walls.add_patch(rect)

    # TX / RX
    if args.tx:
        tx = [float(x) for x in args.tx.split(",")]
        ax_walls.plot(tx[0], tx[1], "r*", markersize=12, label="TX")
        ax_img.plot(tx[0] * ppm, tx[1] * ppm, "r*", markersize=12)
    if args.rx:
        rx = [float(x) for x in args.rx.split(",")]
        ax_walls.plot(rx[0], rx[1], "bo", markersize=10, label="RX")
        ax_img.plot(rx[0] * ppm, rx[1] * ppm, "bo", markersize=10)

    # Legend for materials
    from matplotlib.lines import Line2D
    legend_elements = []
    for m in mat_labels:
        name = m.replace("itu_", "").replace("_", " ").title()
        legend_elements.append(
            patches.Patch(facecolor=mat_color[m], alpha=0.7, label=name)
        )
    if args.tx:
        legend_elements.append(Line2D([0], [0], marker="*", color="w",
                                       markerfacecolor="r", markersize=10, label="TX"))
    if args.rx:
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor="b", markersize=8, label="RX"))
    ax_walls.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def _dump_config(
    args: argparse.Namespace, image_path: Path, bg_color: list, room_w: float, room_h: float,
) -> None:
    tx = [float(x) for x in args.tx.split(",")] if args.tx else [room_w / 4, room_h / 2, 1.5]
    rx = [float(x) for x in args.rx.split(",")] if args.rx else [room_w * 3 / 4, room_h / 2, 1.5]

    config = {
        "environment": {
            "type": "floorplan",
            "tx_position_m": tx,
            "rx_position_m": rx,
            "max_reflections": 2,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "diffraction": False,
            "floorplan": {
                "image_path": str(image_path.resolve()),
                "pixels_per_meter": args.ppm,
                "wall_height_m": args.wall_height,
                "default_tolerance": args.tolerance,
                "color_mapping": _DEFAULT_COLORS,
                "background_color": bg_color,
                "floor_material": "itu_concrete",
                "ceiling_material": "itu_ceiling_board",
                "generate_floor_ceiling": True,
            },
        }
    }

    out_path = Path(args.dump_config)
    out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nConfig saved to: {out_path}")
    print(f"  TX: {tx}  RX: {rx}")
    print(f"\nNext step:")
    print(f"  python3 main.py --config {out_path} --radios uwb --trials 300 --dump-truths cache/rt/my_floorplan")
    print(f"  python3 scripts/rt_cache_interactive.py cache/rt/my_floorplan")


if __name__ == "__main__":
    main()
