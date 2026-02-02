#!/usr/bin/env python3
"""
Observation Space Observer

Connects to a running DDNet-Server with RL bots and displays the observation
space for a specified bot index. Useful for debugging and understanding what
the agent sees.

Usage:
    python observe.py --bot-index 0
    python observe.py --bot-index 0 --server-index 1
    python observe.py --bot-index 0 --continuous  # Live updates
    python observe.py --bot-index 0 --use-embeddings  # Show raw tile IDs
"""

import argparse
import mmap
import struct
import time
import numpy as np
import os
import sys

# Constants from teeworlds_server_env.py
TILE_GRID_WIDTH = 43
TILE_GRID_HEIGHT = 31
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT
HEADER_SIZE = 32
ACTION_SIZE = 32
OBS_SIZE = 1400
BOT_DATA_SIZE = 1448
MAX_RL_BOTS = 128
SHM_PATH_BASE = "/dev/shm/ddnet_rl_server"

# Tile type names (for raw IDs - common DDNet tiles)
TILE_NAMES = {
    0: "AIR",
    1: "SOLID",
    2: "DEATH",
    3: "NOHOOK",
    9: "FREEZE",
    10: "UNFREEZE",
    16: "WALLJUMP",
    17: "EHOOK_START",
    18: "EHOOK_END",
    32: "REFILL_JUMPS",
    33: "START",
    34: "FINISH",
    35: "TIME_CP_1",  # Time checkpoints 35-59
    60: "STOP",
    61: "STOP_L",
    62: "STOP_R",
    63: "STOP_U",
    64: "STOP_D",
    # Add more as needed
}


def get_shm_path(server_index: int = 0) -> str:
    if server_index == 0:
        return SHM_PATH_BASE
    return f"{SHM_PATH_BASE}_{server_index}"


def get_tile_name(tile_id: int) -> str:
    """Get human-readable name for a tile ID."""
    if tile_id in TILE_NAMES:
        return TILE_NAMES[tile_id]
    if 35 <= tile_id <= 59:
        return f"CP_{tile_id - 34}"
    return f"T{tile_id}"


class ObservationReader:
    """Reads observations from shared memory without controlling the server."""

    def __init__(self, server_index: int = 0):
        self.server_index = server_index
        self.shm_path = get_shm_path(server_index)
        self.shm_file = None
        self.shm_mmap = None

    def connect(self) -> bool:
        """Connect to server shared memory (read-only)."""
        if not os.path.exists(self.shm_path):
            print(f"Shared memory not found: {self.shm_path}")
            print("Make sure the DDNet-Server is running with sv_rl_bots enabled.")
            return False

        try:
            self.shm_file = open(self.shm_path, 'rb')
            total_size = HEADER_SIZE + (BOT_DATA_SIZE * MAX_RL_BOTS)
            self.shm_mmap = mmap.mmap(self.shm_file.fileno(), total_size, access=mmap.ACCESS_READ)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def close(self):
        if self.shm_mmap:
            self.shm_mmap.close()
        if self.shm_file:
            self.shm_file.close()

    def read_header(self) -> dict:
        """Read server header."""
        self.shm_mmap.seek(0)
        data = self.shm_mmap.read(HEADER_SIZE)
        server_ready, python_ready, shutdown, _pad1, num_bots, server_tick, _pad2a, _pad2b = struct.unpack('<iiii iiii', data)
        return {
            'server_ready': server_ready,
            'python_ready': python_ready,
            'shutdown': shutdown,
            'num_bots': num_bots,
            'server_tick': server_tick,
        }

    def read_observation(self, bot_index: int, use_embeddings: bool = False) -> dict:
        """Read observation for a specific bot."""
        offset = HEADER_SIZE + (bot_index * BOT_DATA_SIZE)

        # Offsets within SBotData
        OBS_START = 32
        LOCAL_PLAYER = OBS_START + 8
        TILES = LOCAL_PLAYER + 36
        EXTRA_STATE = TILES + 1333
        CHECKPOINT = EXTRA_STATE + 4
        REWARD_OFF = OBS_START + 1400
        DONE_OFF = REWARD_OFF + 4

        # Read raw data
        self.shm_mmap.seek(offset)
        raw = self.shm_mmap.read(BOT_DATA_SIZE)

        # Parse action (what Python last sent)
        action = struct.unpack_from('<iiiiiiii', raw, 0)

        # Parse player floats
        pos_x, pos_y, vel_x, vel_y = struct.unpack_from('<ffff', raw, LOCAL_PLAYER)

        # Parse player bytes
        direction = struct.unpack_from('<b', raw, LOCAL_PLAYER + 16)[0]
        jumped = raw[LOCAL_PLAYER + 17]
        hook_state = raw[LOCAL_PLAYER + 18]
        is_frozen = raw[LOCAL_PLAYER + 30]
        health = raw[LOCAL_PLAYER + 32]
        armor = raw[LOCAL_PLAYER + 33]

        # Parse extra state
        is_grounded = raw[EXTRA_STATE]
        jumps_remaining = raw[EXTRA_STATE + 1]
        can_hook = raw[EXTRA_STATE + 2]

        # Parse checkpoint
        cp_x, cp_y = struct.unpack_from('<ff', raw, CHECKPOINT)
        cp_idx, total_cp = struct.unpack_from('<ii', raw, CHECKPOINT + 8)

        # Parse tiles
        tile_raw = raw[TILES:TILES + TILE_GRID_SIZE]
        if use_embeddings:
            tiles = np.frombuffer(tile_raw, dtype=np.uint8).reshape(TILE_GRID_HEIGHT, TILE_GRID_WIDTH).astype(np.int64)
        else:
            tiles = np.frombuffer(tile_raw, dtype=np.uint8).reshape(TILE_GRID_HEIGHT, TILE_GRID_WIDTH).astype(np.float32) / 15.0

        # Parse reward/done
        reward = struct.unpack_from('<f', raw, REWARD_OFF)[0]
        done = struct.unpack_from('<i', raw, DONE_OFF)[0]

        return {
            'action': {
                'direction': action[0],
                'jump': action[1],
                'hook': action[2],
                'fire': action[3],
                'aim_x': action[4],
                'aim_y': action[5],
                'weapon': action[6],
                'reset': action[7],
            },
            'position': (pos_x, pos_y),
            'velocity': (vel_x, vel_y),
            'direction': direction,
            'jumped': jumped,
            'hook_state': hook_state,
            'is_frozen': bool(is_frozen),
            'health': health,
            'armor': armor,
            'is_grounded': bool(is_grounded),
            'jumps_remaining': jumps_remaining,
            'can_hook': bool(can_hook),
            'checkpoint': {
                'x': cp_x,
                'y': cp_y,
                'index': cp_idx,
                'total': total_cp,
                'rel_x': (cp_x - pos_x) / 2000.0 if total_cp > 0 else 0,
                'rel_y': (cp_y - pos_y) / 2000.0 if total_cp > 0 else 0,
            },
            'tiles': tiles,
            'reward': reward,
            'done': bool(done),
            'use_embeddings': use_embeddings,
        }


def render_tiles_ascii(tiles: np.ndarray, use_embeddings: bool = False, center_crop: int = 0, show_player: bool = True):
    """Render tile grid as ASCII art."""
    h, w = tiles.shape
    orig_cy, orig_cx = h // 2, w // 2

    # Optionally crop to center
    if center_crop > 0:
        half = center_crop // 2
        tiles = tiles[orig_cy - half:orig_cy + half + 1, orig_cx - half:orig_cx + half + 1]
        h, w = tiles.shape

    # Player is at center of the (possibly cropped) grid
    player_y, player_x = h // 2, w // 2

    # Tile ID to character mapping (works for both raw IDs and old categories)
    char_map = {
        0: ' ',   # AIR
        1: '█',   # SOLID
        2: 'X',   # DEATH
        3: '░',   # NOHOOK
        9: '*',   # FREEZE
        10: '+',  # UNFREEZE
        16: 'W',  # WALLJUMP
        17: 'E',  # EHOOK_START
        18: 'e',  # EHOOK_END
        26: '⊙',  # TELEIN
        27: '⊚',  # TELEOUT
        29: '⊛',  # TELECHECK
        32: 'J',  # REFILL_JUMPS
        33: 'S',  # START
        34: 'F',  # FINISH
        60: '!',  # STOP
        61: '<',  # STOP_L
        62: '>',  # STOP_R
        63: '^',  # STOP_U
        64: 'v',  # STOP_D
        103: '⊕', # TELEINHOOK
        104: '⊗', # TELEINWEAPON
    }

    # Old 16-category ETileType mapping (for servers not yet rebuilt)
    old_category_map = {
        0: ' ',   # EMPTY
        1: '█',   # SOLID
        2: 'X',   # DEATH
        3: '░',   # NOHOOK
        4: '*',   # FREEZE
        5: '+',   # UNFREEZE
        6: 'S',   # SPAWN
        7: 'F',   # FINISH
        8: 'C',   # CHECKPOINT
        9: '?',   # OTHER
    }

    lines = []
    for y in range(h):
        row = ""
        for x in range(w):
            # Show player at center
            if show_player and y == player_y and x == player_x:
                row += 'O'
                continue

            if use_embeddings:
                tid = int(tiles[y, x])
            else:
                # Convert normalized float back to category
                tid = int(round(tiles[y, x] * 15))

            # Try raw tile map first, then old category map
            if tid in char_map:
                row += char_map[tid]
            elif tid in old_category_map:
                row += old_category_map[tid]
            elif 35 <= tid <= 59:
                row += str((tid - 34) % 10)  # Time checkpoint number
            elif tid == 0:
                row += ' '
            else:
                row += '?'
        lines.append(row)

    return lines


def print_observation(obs: dict, show_tiles: bool = True, tile_crop: int = 0):
    """Pretty print an observation."""
    print("\n" + "=" * 60)
    print("OBSERVATION")
    print("=" * 60)

    print(f"\n[Position]  x={obs['position'][0]:.1f}, y={obs['position'][1]:.1f}")
    print(f"[Velocity]  vx={obs['velocity'][0]:.2f}, vy={obs['velocity'][1]:.2f}")
    print(f"[State]     grounded={obs['is_grounded']}, jumps={obs['jumps_remaining']}, "
          f"can_hook={obs['can_hook']}, frozen={obs['is_frozen']}")
    print(f"[Movement]  direction={obs['direction']}, jumped={obs['jumped']}, hook={obs['hook_state']}")
    print(f"[Stats]     health={obs['health']}, armor={obs['armor']}")

    cp = obs['checkpoint']
    if cp['total'] > 0:
        print(f"\n[Checkpoint] {cp['index']}/{cp['total']} at ({cp['x']:.1f}, {cp['y']:.1f})")
        print(f"             relative: ({cp['rel_x']:.3f}, {cp['rel_y']:.3f})")
    else:
        print(f"\n[Checkpoint] None")

    print(f"\n[Reward] {obs['reward']:.4f}  [Done] {obs['done']}")

    act = obs['action']
    print(f"\n[Last Action] dir={act['direction']}, jump={act['jump']}, hook={act['hook']}, "
          f"aim=({act['aim_x']}, {act['aim_y']})")

    if show_tiles:
        tiles = obs['tiles']
        use_emb = obs['use_embeddings']

        print(f"\n[Tiles] shape={tiles.shape}, dtype={tiles.dtype}")
        if use_emb:
            unique = np.unique(tiles)
            print(f"        unique IDs: {len(unique)} - {sorted(unique)[:20]}{'...' if len(unique) > 20 else ''}")
            print("        Legend: .=air #=solid X=death ~=nohook *=freeze W=walljump J=refill S=start F=finish")

        lines = render_tiles_ascii(tiles, use_emb, tile_crop)

        print(f"\n        Legend: █=solid ░=nohook X=death *=freeze ⊙=tele O=player")
        print("        " + "-" * (len(lines[0]) + 2) if lines else "")
        for line in lines:
            print(f"        |{line}|")
        print("        " + "-" * (len(lines[0]) + 2) if lines else "")


def main():
    parser = argparse.ArgumentParser(description="Observe bot observation space")
    parser.add_argument("--bot-index", "-b", type=int, default=0, help="Bot index to observe")
    parser.add_argument("--server-index", "-s", type=int, default=0, help="Server index")
    parser.add_argument("--continuous", "-c", action="store_true", help="Continuous updates (refresh in place)")
    parser.add_argument("--interval", "-i", type=float, default=0.5, help="Update interval (seconds)")
    parser.add_argument("--use-embeddings", "-e", action="store_true", help="Show raw tile IDs")
    parser.add_argument("--no-tiles", action="store_true", help="Don't show tile grid")
    parser.add_argument("--tile-crop", type=int, default=0, help="Crop tiles to NxN around player (0=full)")
    args = parser.parse_args()

    reader = ObservationReader(args.server_index)
    if not reader.connect():
        sys.exit(1)

    try:
        header = reader.read_header()

        if args.bot_index >= header['num_bots']:
            print(f"Error: Bot index {args.bot_index} out of range (0-{header['num_bots']-1})")
            sys.exit(1)

        if args.continuous:
            # Hide cursor for cleaner display
            print("\033[?25l", end="")

        first_run = True
        while True:
            obs = reader.read_observation(args.bot_index, args.use_embeddings)
            header = reader.read_header()

            if args.continuous:
                # Move cursor to top-left and clear screen
                print("\033[H\033[J", end="")

            if first_run and not args.continuous:
                print(f"Connected to server {args.server_index}")
                print(f"  Bots: {header['num_bots']}, Tick: {header['server_tick']}")
                first_run = False
            elif args.continuous:
                print(f"Server {args.server_index} | Tick: {header['server_tick']} | Bot {args.bot_index} | Ctrl+C to exit")

            print_observation(obs, show_tiles=not args.no_tiles, tile_crop=args.tile_crop)

            if not args.continuous:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass
    finally:
        if args.continuous:
            # Show cursor again
            print("\033[?25h", end="")
            print("\nStopped.")
        reader.close()


if __name__ == "__main__":
    main()
