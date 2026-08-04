"""Generate waypoints from a DDNet map using BFS pathfinding.

Builds a tile-level graph from the game/front/tele layers and finds
the shortest path from start to finish, respecting teleporters and
avoiding death tiles.

Usage:
    python map_pathfinder.py path/to/map.map [--output waypoints.json] [--spacing 64]

Output format matches ghost_parser.py for interchangeable use.
"""

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from typing import Optional

import numpy as np

try:
    import twmap
except ImportError:
    print("Error: twmap package required. Install with: pip install twmap", file=sys.stderr)
    sys.exit(1)


# DDNet tile constants (from mapitems.h)
TILE_AIR = 0
TILE_SOLID = 1
TILE_DEATH = 2
TILE_NOHOOK = 3
TILE_FREEZE = 9
TILE_UNFREEZE = 11
TILE_DFREEZE = 12
TILE_DUNFREEZE = 13
TILE_TELEIN = 26
TILE_TELEOUT = 27
TILE_START = 33
TILE_FINISH = 34
TILE_TIME_CP_FIRST = 35
TILE_TIME_CP_LAST = 59
TILE_STOP = 60
TILE_STOPS = 61
TILE_STOPA = 62
TILE_TELECHECK = 29
TILE_TELECHECKOUT = 30
TILE_TELECHECKIN = 31
TILE_THROUGH = 6


def load_map_layers(map_path: str):
    """Load game, front, and tele layers from a DDNet map."""
    m = twmap.Map(map_path)
    width = m.game_layer().width()
    height = m.game_layer().height()

    # Find layers by kind in the groups
    game_tiles = None
    front_tiles = None
    tele_tiles = None

    for g in m.groups:
        for l in g.layers:
            if hasattr(l, 'tiles') and l.tiles is not None:
                tiles = np.array(l.tiles)
                if tiles.shape[0] == height and tiles.shape[1] == width:
                    kind = l.kind()
                    if kind == 'Game':
                        game_tiles = tiles
                    elif kind == 'Front':
                        front_tiles = tiles
                    elif kind == 'Tele':
                        tele_tiles = tiles

    print(f"Map: {width}x{height}", file=sys.stderr)
    print(f"  Game layer: {'found' if game_tiles is not None else 'MISSING'}", file=sys.stderr)
    print(f"  Front layer: {'found' if front_tiles is not None else 'none'}", file=sys.stderr)
    print(f"  Tele layer: {'found' if tele_tiles is not None else 'none'}", file=sys.stderr)

    return width, height, game_tiles, front_tiles, tele_tiles


def build_pathfinding_graph(width, height, game_tiles, front_tiles, tele_tiles,
                            freeze_cost=5, tele_cost=1):
    """Build an adjacency graph from map tiles.

    Returns:
        graph: dict mapping (x, y) -> list of ((nx, ny), cost)
        start_pos: (x, y) of start tile
        finish_pos: (x, y) of finish tile
        checkpoints: dict mapping cp_index -> (x, y)
        tele_map: dict mapping tele_number -> {'in': [(x,y),...], 'out': [(x,y),...]}
    """
    # Extract tile index arrays (channel 0 = tile ID, channel 1 = flags)
    game_idx = game_tiles[:, :, 0]
    front_idx = front_tiles[:, :, 0] if front_tiles is not None else np.zeros_like(game_idx)

    start_pos = None
    finish_positions = []
    checkpoints = {}

    # Build teleporter mapping from tele layer
    tele_map = defaultdict(lambda: {'in': [], 'out': [], 'checkin': [], 'checkout': []})
    if tele_tiles is not None:
        tele_num = tele_tiles[:, :, 0]  # teleporter number
        tele_type = tele_tiles[:, :, 1]  # teleporter type
        for y in range(height):
            for x in range(width):
                num = int(tele_num[y, x])
                typ = int(tele_type[y, x])
                if num == 0:
                    continue
                if typ == TILE_TELEIN:
                    tele_map[num]['in'].append((x, y))
                elif typ == TILE_TELEOUT:
                    tele_map[num]['out'].append((x, y))
                elif typ == TILE_TELECHECKIN:
                    tele_map[num]['checkin'].append((x, y))
                elif typ == TILE_TELECHECKOUT:
                    tele_map[num]['checkout'].append((x, y))

    # Scan for special tiles
    for y in range(height):
        for x in range(width):
            tile = int(game_idx[y, x])
            ftile = int(front_idx[y, x])

            if tile == TILE_START:
                start_pos = (x, y)
            elif tile == TILE_FINISH:
                finish_positions.append((x, y))

            # Time checkpoints in game or front layer
            for t in [tile, ftile]:
                if TILE_TIME_CP_FIRST <= t <= TILE_TIME_CP_LAST:
                    cp_idx = t - TILE_TIME_CP_FIRST
                    if cp_idx not in checkpoints:
                        checkpoints[cp_idx] = (x, y)

    if start_pos is None:
        print("WARNING: No start tile found!", file=sys.stderr)
    if not finish_positions:
        print("WARNING: No finish tile found!", file=sys.stderr)

    finish_pos = finish_positions[0] if finish_positions else None

    print(f"  Start: {start_pos}, Finish: {finish_pos}", file=sys.stderr)
    print(f"  Checkpoints: {len(checkpoints)}", file=sys.stderr)
    print(f"  Teleporters: {len(tele_map)} groups", file=sys.stderr)

    # Build adjacency graph
    # Nodes are (x, y) tile positions
    # Edges connect walkable adjacent tiles with movement costs
    graph = defaultdict(list)

    def is_passable(x, y):
        """Check if a tile is passable (not solid, not death)."""
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        tile = int(game_idx[y, x])
        ftile = int(front_idx[y, x])
        # Death in either layer = impassable
        if tile == TILE_DEATH or ftile == TILE_DEATH:
            return False
        # Solid = impassable
        if tile == TILE_SOLID or tile == TILE_NOHOOK:
            return False
        return True

    def tile_cost(x, y):
        """Get movement cost for entering a tile."""
        tile = int(game_idx[y, x])
        ftile = int(front_idx[y, x])
        if tile == TILE_FREEZE or ftile == TILE_FREEZE:
            return freeze_cost
        if tile == TILE_DFREEZE or ftile == TILE_DFREEZE:
            return freeze_cost * 2
        return 1

    # 4-directional movement (8-dir would be more accurate but 4 is simpler)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # Add diagonals with sqrt(2) cost
    diag_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue

            # Cardinal directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    cost = tile_cost(nx, ny)
                    graph[(x, y)].append(((nx, ny), cost))

            # Diagonals (only if both cardinal neighbors are passable to avoid cutting corners)
            for dx, dy in diag_directions:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny) and is_passable(x + dx, y) and is_passable(x, y + dy):
                    cost = tile_cost(nx, ny) * 1.414
                    graph[(x, y)].append(((nx, ny), cost))

    # Add teleporter edges
    for num, tele_data in tele_map.items():
        # Regular teleporters: all TELEIN tiles connect to all TELEOUT tiles
        for in_pos in tele_data['in']:
            for out_pos in tele_data['out']:
                graph[in_pos].append((out_pos, tele_cost))

        # Checkpoint teleporters
        for in_pos in tele_data['checkin']:
            for out_pos in tele_data['checkout']:
                graph[in_pos].append((out_pos, tele_cost))

    return graph, start_pos, finish_pos, checkpoints, tele_map


def dijkstra(graph, start, end, width=0, height=0):
    """Run Dijkstra's shortest path algorithm.

    Returns list of (x, y) positions from start to end, or empty list if unreachable.
    """
    import heapq

    if start is None or end is None:
        return []

    dist = {start: 0}
    prev = {start: None}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        if u == end:
            break

        for v, w in graph.get(u, []):
            new_dist = d + w
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    if end not in prev:
        return []

    # Reconstruct path
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path


def tile_path_to_world(path: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """Convert tile coordinates to world coordinates (center of tile)."""
    return [(x * 32.0 + 16.0, y * 32.0 + 16.0) for x, y in path]


def resample_waypoints(positions: list[tuple[float, float]], spacing: float = 64.0) -> list[tuple[float, float]]:
    """Resample positions to equally-spaced waypoints."""
    if len(positions) < 2:
        return positions

    waypoints = [positions[0]]
    accumulated = 0.0

    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.001:
            continue

        accumulated += dist

        while accumulated >= spacing:
            overshoot = accumulated - spacing
            t = 1.0 - (overshoot / dist) if dist > 0 else 1.0
            t = max(0.0, min(1.0, t))
            wx = positions[i - 1][0] + t * dx
            wy = positions[i - 1][1] + t * dy
            waypoints.append((wx, wy))
            accumulated -= spacing

    last = positions[-1]
    if waypoints[-1] != last:
        waypoints.append(last)

    return waypoints


def save_waypoints(waypoints, output_path, source="pathfinding", map_name=""):
    """Save waypoints as JSON."""
    total_dist = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        total_dist += math.sqrt(dx * dx + dy * dy)

    data = {
        "source": source,
        "map": map_name,
        "num_points": len(waypoints),
        "total_distance": round(total_dist, 1),
        "spacing": round(total_dist / max(len(waypoints) - 1, 1), 1),
        "points": [[round(x, 1), round(y, 1)] for x, y in waypoints],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Saved {len(waypoints)} waypoints to {output_path} "
          f"(total distance: {total_dist:.0f} units)", file=sys.stderr)


def save_waypoints_binary(waypoints, output_path):
    """Save waypoints as binary for C++ server loading."""
    import struct
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(waypoints)))
        for x, y in waypoints:
            f.write(struct.pack('<ff', x, y))
    print(f"Saved {len(waypoints)} waypoints to {output_path} (binary)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate waypoints from DDNet map via pathfinding")
    parser.add_argument("map_file", help="Path to .map file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--binary", "-b", default=None, help="Output binary file for C++ server")
    parser.add_argument("--spacing", type=float, default=64.0,
                        help="Spacing between waypoints in world units (32=1 tile, default: 64)")
    parser.add_argument("--freeze-cost", type=int, default=5,
                        help="Cost multiplier for freeze tiles (default: 5)")
    args = parser.parse_args()

    # Load map
    width, height, game_tiles, front_tiles, tele_tiles = load_map_layers(args.map_file)

    if game_tiles is None:
        print("Error: could not find game layer!", file=sys.stderr)
        sys.exit(1)

    # Build graph
    print("Building pathfinding graph...", file=sys.stderr)
    graph, start_pos, finish_pos, checkpoints, tele_map = build_pathfinding_graph(
        width, height, game_tiles, front_tiles, tele_tiles,
        freeze_cost=args.freeze_cost,
    )
    print(f"Graph: {len(graph)} walkable tiles", file=sys.stderr)

    if start_pos is None or finish_pos is None:
        print("Error: need both start and finish tiles!", file=sys.stderr)
        sys.exit(1)

    # Find path from start to finish
    print("Running Dijkstra...", file=sys.stderr)
    tile_path = dijkstra(graph, start_pos, finish_pos, width, height)

    if not tile_path:
        print("Error: no path found from start to finish!", file=sys.stderr)
        sys.exit(1)

    print(f"Path found: {len(tile_path)} tiles", file=sys.stderr)

    # Convert to world coordinates and resample
    world_path = tile_path_to_world(tile_path)
    waypoints = resample_waypoints(world_path, spacing=args.spacing)

    # Extract map name from path
    import os
    map_name = os.path.splitext(os.path.basename(args.map_file))[0]

    # Save
    output_path = args.output or f"{map_name}.pathfind.waypoints.json"
    save_waypoints(waypoints, output_path, source="pathfinding", map_name=map_name)

    if args.binary:
        save_waypoints_binary(waypoints, args.binary)


if __name__ == "__main__":
    main()
