"""Visualize waypoints overlaid on a DDNet map.

Usage:
    python visualize_waypoints.py data/maps/Tutorial.map \
        --ghost ghosts/Tutorial.waypoints.json \
        --pathfind ghosts/Tutorial.pathfind.waypoints.json \
        --output waypoints_vis.png
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import twmap
except ImportError:
    print("Error: twmap package required. Install with: pip install twmap", file=sys.stderr)
    sys.exit(1)

# Tile colors (RGBA)
TILE_COLORS = {
    0: (0.15, 0.15, 0.2, 1.0),   # AIR - dark background
    1: (0.5, 0.5, 0.5, 1.0),     # SOLID
    2: (0.8, 0.1, 0.1, 1.0),     # DEATH
    3: (0.4, 0.4, 0.5, 1.0),     # NOHOOK
    9: (0.2, 0.4, 0.8, 1.0),     # FREEZE
    11: (0.2, 0.7, 0.3, 1.0),    # UNFREEZE
    12: (0.1, 0.2, 0.6, 1.0),    # DFREEZE
    33: (0.0, 1.0, 0.0, 1.0),    # START
    34: (1.0, 0.8, 0.0, 1.0),    # FINISH
}
# Time checkpoints
for i in range(35, 60):
    TILE_COLORS[i] = (0.9, 0.5, 0.1, 1.0)


def render_map(map_path):
    """Render map game layer as an RGBA image."""
    m = twmap.Map(map_path)
    width = m.game_layer().width()
    height = m.game_layer().height()

    game_tiles = None
    for g in m.groups:
        for l in g.layers:
            if hasattr(l, 'tiles') and l.tiles is not None:
                tiles = np.array(l.tiles)
                if tiles.shape[0] == height and tiles.shape[1] == width:
                    if l.kind() == 'Game':
                        game_tiles = tiles
                        break

    if game_tiles is None:
        print("Error: no game layer found", file=sys.stderr)
        sys.exit(1)

    tile_ids = game_tiles[:, :, 0]
    img = np.zeros((height, width, 4))
    default_color = TILE_COLORS[0]

    for y in range(height):
        for x in range(width):
            tid = int(tile_ids[y, x])
            img[y, x] = TILE_COLORS.get(tid, default_color)

    return img, width, height


def load_waypoints_json(path):
    """Load waypoints from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return np.array(data["points"]), data.get("source", "unknown"), data.get("num_points", len(data["points"]))


def main():
    parser = argparse.ArgumentParser(description="Visualize waypoints on DDNet map")
    parser.add_argument("map_file", help="Path to .map file")
    parser.add_argument("--ghost", help="Ghost waypoints JSON")
    parser.add_argument("--pathfind", help="Pathfinding waypoints JSON")
    parser.add_argument("--output", "-o", default="waypoints_vis.png", help="Output image (default: waypoints_vis.png)")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    args = parser.parse_args()

    if not args.ghost and not args.pathfind:
        print("Error: provide at least one of --ghost or --pathfind", file=sys.stderr)
        sys.exit(1)

    print("Rendering map...", file=sys.stderr)
    img, map_w, map_h = render_map(args.map_file)

    # Split into horizontal strips for readability on wide maps
    aspect = map_w / max(map_h, 1)
    if aspect > 8:
        n_strips = max(2, int(aspect / 8))
        strip_w = map_w * 32 / n_strips
        fig, axes = plt.subplots(n_strips, 1, figsize=(40, 6 * n_strips))
        if n_strips == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            x0 = i * strip_w
            x1 = (i + 1) * strip_w
            ax.imshow(img, extent=[0, map_w * 32, map_h * 32, 0], interpolation='nearest', aspect='equal')
            ax.set_xlim(x0, x1)
            ax.set_ylim(map_h * 32, 0)

            if args.ghost:
                pts, src, n = load_waypoints_json(args.ghost)
                ax.plot(pts[:, 0], pts[:, 1], '-', color='cyan', linewidth=2.5, alpha=0.85, label=f'Ghost ({n} pts)')
                ax.plot(pts[0, 0], pts[0, 1], 'o', color='lime', markersize=8, zorder=5)
                ax.plot(pts[-1, 0], pts[-1, 1], 's', color='red', markersize=8, zorder=5)

            if args.pathfind:
                pts, src, n = load_waypoints_json(args.pathfind)
                ax.plot(pts[:, 0], pts[:, 1], '-', color='magenta', linewidth=2.5, alpha=0.85, label=f'Pathfind ({n} pts)')
                ax.plot(pts[0, 0], pts[0, 1], 'o', color='lime', markersize=8, zorder=5)
                ax.plot(pts[-1, 0], pts[-1, 1], 's', color='red', markersize=8, zorder=5)

            ax.set_ylabel("Y")
            if i == 0:
                ax.set_title("Waypoints on Map")
                ax.legend(loc='upper right', fontsize=10)
            ax.set_xlabel(f"X ({int(x0)}-{int(x1)})")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(min(map_w * 0.15, 40), min(map_h * 0.15, 20)))
        ax.imshow(img, extent=[0, map_w * 32, map_h * 32, 0], interpolation='nearest', aspect='equal')

        if args.ghost:
            pts, src, n = load_waypoints_json(args.ghost)
            ax.plot(pts[:, 0], pts[:, 1], '-', color='cyan', linewidth=2.5, alpha=0.85, label=f'Ghost ({n} pts)')
            ax.plot(pts[0, 0], pts[0, 1], 'o', color='lime', markersize=8, zorder=5)
            ax.plot(pts[-1, 0], pts[-1, 1], 's', color='red', markersize=8, zorder=5)

        if args.pathfind:
            pts, src, n = load_waypoints_json(args.pathfind)
            ax.plot(pts[:, 0], pts[:, 1], '-', color='magenta', linewidth=2.5, alpha=0.85, label=f'Pathfind ({n} pts)')
            ax.plot(pts[0, 0], pts[0, 1], 'o', color='lime', markersize=8, zorder=5)
            ax.plot(pts[-1, 0], pts[-1, 1], 's', color='red', markersize=8, zorder=5)

        ax.set_xlabel("X (world units)")
        ax.set_ylabel("Y (world units)")
        ax.set_title("Waypoints on Map")
        ax.legend(loc='upper right')
        ax.set_xlim(0, map_w * 32)
        ax.set_ylim(map_h * 32, 0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
