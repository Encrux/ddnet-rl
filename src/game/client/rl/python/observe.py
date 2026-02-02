#!/usr/bin/env python3
"""Visualize RL observation space in terminal.

Reads from shared memory and displays the tile grid as ASCII art,
along with player state information.

Usage:
    python observe.py --shm /ddnet_rl_training_0 --bot 0
"""

import argparse
import mmap
import os
import struct
import sys
import time
from typing import Optional

import numpy as np

# Constants matching C++ rl_protocol.h
TILE_GRID_WIDTH = 29
TILE_GRID_HEIGHT = 21
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT

# Observation layout
OBS_PLAYER = 16
OBS_TILES = TILE_GRID_SIZE
OBSERVATION_SIZE = OBS_PLAYER + OBS_TILES  # = 625

# Tile characters for ASCII rendering
# Based on DDNet mapitems.h tile indices
TILE_CHARS = {
    0: ' ',    # TILE_AIR
    1: '#',    # TILE_SOLID
    2: 'X',    # TILE_DEATH
    3: '%',    # TILE_NOHOOK
    4: '~',    # TILE_NOLASER
    5: '/',    # TILE_THROUGH_CUT
    6: '|',    # TILE_THROUGH
    7: '^',    # TILE_JUMP
    9: '*',    # TILE_FREEZE
    10: '!',   # TILE_TELEINEVIL
    11: 'o',   # TILE_UNFREEZE
    12: 'D',   # TILE_DFREEZE
    13: 'U',   # TILE_DUNFREEZE
    16: 'W',   # TILE_WALLJUMP
}

# Special entity tiles (index >= ENTITY_OFFSET = 191)
ENTITY_OFFSET = 191


def get_tile_char(tile_id: int) -> str:
    """Convert tile ID to ASCII character."""
    tile_id = int(tile_id)

    # Check if it's an entity tile
    if tile_id >= ENTITY_OFFSET:
        entity_id = tile_id - ENTITY_OFFSET
        if entity_id == 1:  # ENTITY_SPAWN
            return 'S'
        elif entity_id in (2, 3):  # ENTITY_SPAWN_RED/BLUE
            return 's'
        elif entity_id in (6, 7):  # ENTITY_ARMOR/HEALTH
            return '+'
        elif entity_id in (8, 9, 10, 11):  # Weapons
            return 'w'
        elif entity_id == 49:  # ENTITY_DOOR
            return 'D'
        else:
            return 'E'  # Generic entity

    # Known tile types
    if tile_id in TILE_CHARS:
        return TILE_CHARS[tile_id]

    # Unknown tiles
    if tile_id > 0:
        return '?'
    return ' '


def render_tile_grid(tile_ids: np.ndarray) -> str:
    """Render 29x21 tile grid as ASCII art.

    Player is always at center (14, 10).
    """
    lines = []

    # Top border
    lines.append('┌' + '─' * TILE_GRID_WIDTH + '┐')

    for y in range(TILE_GRID_HEIGHT):
        row = '│'
        for x in range(TILE_GRID_WIDTH):
            if x == 14 and y == 10:  # Center = player position
                row += 'P'
            else:
                tile_id = tile_ids[y * TILE_GRID_WIDTH + x]
                row += get_tile_char(tile_id)
        row += '│'
        lines.append(row)

    # Bottom border
    lines.append('└' + '─' * TILE_GRID_WIDTH + '┘')

    return '\n'.join(lines)


def format_player_state(player_obs: np.ndarray) -> str:
    """Format player observation as readable string."""
    # SPlayerObs layout (16 floats):
    # pos_x, pos_y, vel_x, vel_y, angle,
    # health, armor, weapon, jump_state,
    # hook_state, hook_pos_x, hook_pos_y,
    # alive, team, is_local, padding

    pos_x = player_obs[0]
    pos_y = player_obs[1]
    vel_x = player_obs[2]
    vel_y = player_obs[3]
    angle = player_obs[4]
    health = player_obs[5]
    armor = player_obs[6]
    weapon = player_obs[7]
    jump_state = player_obs[8]
    hook_state = player_obs[9]
    hook_pos_x = player_obs[10]
    hook_pos_y = player_obs[11]
    alive = player_obs[12]

    # Hook state names (from DDNet)
    hook_states = {
        0: 'idle',
        1: 'retract_start',
        2: 'retract',
        3: 'retract_end',
        4: 'flying',
        5: 'grabbed',
    }
    hook_state_name = hook_states.get(int(hook_state * 5), f'{hook_state:.2f}')

    lines = [
        f"Position: ({pos_x:+.3f}, {pos_y:+.3f})",
        f"Velocity: ({vel_x:+.3f}, {vel_y:+.3f})",
        f"Angle: {angle:+.3f}  Alive: {'Yes' if alive > 0.5 else 'No'}",
        f"Jump: {jump_state:.0f}  Hook: {hook_state_name}",
        f"Hook pos: ({hook_pos_x:+.3f}, {hook_pos_y:+.3f})",
    ]
    return '\n'.join(lines)


def format_game_state() -> str:
    """Format game state observation (removed - no longer in observation)."""
    return "(Game state removed from observation)"


class ObservationReader:
    """Reads observations from shared memory."""

    def __init__(self, shm_name: str):
        self.shm_name = shm_name
        self.mmap_obj: Optional[mmap.mmap] = None
        self.header_size = 768
        self.transition_size = 0
        self.capacity = 0

    def open(self) -> bool:
        """Open shared memory region."""
        try:
            # Try /dev/shm path
            shm_path = f"/dev/shm{self.shm_name}"
            if os.path.exists(shm_path):
                fd = os.open(shm_path, os.O_RDONLY)
                size = os.fstat(fd).st_size
                self.mmap_obj = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                os.close(fd)

                # Read capacity and transition size from header
                self.mmap_obj.seek(200)
                self.capacity = struct.unpack('I', self.mmap_obj.read(4))[0]
                self.transition_size = struct.unpack('I', self.mmap_obj.read(4))[0]

                return True
            return False
        except Exception as e:
            print(f"Failed to open shared memory: {e}")
            return False

    def close(self):
        """Close shared memory."""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None

    def get_latest_observation(self, bot_id: int = 0) -> Optional[np.ndarray]:
        """Get the latest observation for a specific bot.

        Searches backward from write index to find a transition from the given bot.
        """
        if not self.mmap_obj:
            return None

        # Read write/read indices
        self.mmap_obj.seek(0)
        write_idx = struct.unpack('Q', self.mmap_obj.read(8))[0]

        if write_idx == 0:
            return None

        # Search backward for a transition from this bot
        for i in range(min(100, write_idx)):
            idx = (write_idx - 1 - i) % self.capacity
            offset = self.header_size + idx * self.transition_size

            self.mmap_obj.seek(offset)
            data = self.mmap_obj.read(self.transition_size)

            # Read observation
            obs = np.frombuffer(data[:OBSERVATION_SIZE * 4], dtype=np.float32).copy()

            # Read bot_id (at end of transition after action, log_probs, value, reward, done/truncated)
            # Action: 5 ints + 2 floats = 28 bytes
            # Log probs: 6 floats = 24 bytes
            # Value, reward: 2 floats = 8 bytes
            # Done, truncated, padding: 4 bytes
            # Bot ID: 4 bytes
            bot_id_offset = OBSERVATION_SIZE * 4 + 28 + 24 + 8 + 4
            trans_bot_id = struct.unpack_from('i', data, bot_id_offset)[0]

            if trans_bot_id == bot_id:
                return obs

        # Fallback: return latest observation regardless of bot
        idx = (write_idx - 1) % self.capacity
        offset = self.header_size + idx * self.transition_size
        self.mmap_obj.seek(offset)
        data = self.mmap_obj.read(OBSERVATION_SIZE * 4)
        return np.frombuffer(data, dtype=np.float32).copy()


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RL observation space in terminal',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--shm', default='/ddnet_rl_training_0',
                        help='Shared memory name')
    parser.add_argument('--bot', type=int, default=0,
                        help='Bot ID to observe')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Update rate (frames per second)')
    parser.add_argument('--no-clear', action='store_true',
                        help='Do not clear screen between frames')
    args = parser.parse_args()

    # Open shared memory
    print(f"Opening shared memory: {args.shm}")
    reader = ObservationReader(args.shm)
    if not reader.open():
        print(f"Error: Could not open shared memory {args.shm}")
        print(f"Make sure the DDNet server is running with RL bots enabled.")
        return 1

    print(f"Connected! Capacity: {reader.capacity}, Transition size: {reader.transition_size}")
    print(f"Observing bot {args.bot} at {args.fps} FPS")
    print("Press Ctrl+C to stop.\n")
    time.sleep(1)

    # Legend
    legend = (
        "Legend: ' '=air  #=solid  X=death  %=nohook  *=freeze  "
        "o=unfreeze  W=walljump  P=player  S=spawn  E=entity"
    )

    try:
        frame = 0
        while True:
            obs = reader.get_latest_observation(args.bot)

            if obs is None:
                if not args.no_clear:
                    clear_screen()
                print("Waiting for observations...")
                time.sleep(0.5)
                continue

            # Parse observation components
            player_state = obs[:OBS_PLAYER]
            tile_ids = obs[OBS_PLAYER:]

            # Build output
            output_lines = []

            if not args.no_clear:
                clear_screen()

            output_lines.append(f"=== DDNet RL Observer - Bot {args.bot} - Frame {frame} ===")
            output_lines.append("")
            output_lines.append(format_game_state())
            output_lines.append("")
            output_lines.append(format_player_state(player_state))
            output_lines.append("")
            output_lines.append(render_tile_grid(tile_ids))
            output_lines.append("")
            output_lines.append(legend)

            print('\n'.join(output_lines))

            frame += 1
            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        reader.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
