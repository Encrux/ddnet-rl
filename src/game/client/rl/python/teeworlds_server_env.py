"""
Teeworlds Server-Side Bot Environment for RL Training

A Gymnasium-compatible vectorized environment that communicates directly with
server-side bots via a single shared memory region. No client processes needed.

This provides much better performance than the client-based approach:
- Single process (server only)
- No network overhead
- Direct shared memory communication

Usage:
    env = TeeWorldsServerEnv(num_bots=63)
    obs = env.reset()
    for _ in range(1000):
        actions = policy(obs)
        obs, rewards, dones, truncated, infos = env.step(actions)
    env.close()
"""

import mmap
import struct
import time
import numpy as np
import subprocess
import os
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from teeworlds_env import StructuredObservation

# Try to import numba for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: define a no-op jit decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Tile grid constants (match C++ rl_protocol.h)
TILE_GRID_WIDTH = 43   # 21 left + 1 center + 21 right
TILE_GRID_HEIGHT = 31  # 15 up + 1 center + 15 down
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT  # 1333


def _warmup_numba_jit():
    """Pre-compile Numba JIT functions to avoid compilation delay during training."""
    if not NUMBA_AVAILABLE:
        return

    # Create tiny dummy arrays to trigger JIT compilation
    n = 2

    # Legacy interleaved layout warmup
    raw = np.zeros(n * 1448, dtype=np.uint8)
    features = np.zeros((n, 68), dtype=np.float32)
    player_state = np.zeros((n, 15), dtype=np.float32)
    checkpoint = np.zeros((n, 4), dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)

    # Trigger compilation (first call compiles, subsequent calls are fast)
    _parse_flat_obs_numba(raw, n, features, rewards, dones)
    _parse_cnn_obs_numba(raw, n, player_state, checkpoint, rewards, dones)

    # Contiguous layout warmup
    obs_arr = np.zeros((n, 1400), dtype=np.uint8)
    _parse_flat_obs_contiguous_numba(obs_arr, n, features, rewards, dones)

# Constants matching C++ rl_protocol.h
MAX_RL_BOTS = 128
SHM_PATH_BASE = "/dev/shm/ddnet_rl_server"

def get_shm_path(server_index: int = 0) -> str:
    """Get shared memory path for a server index."""
    if server_index == 0:
        return SHM_PATH_BASE
    return f"{SHM_PATH_BASE}_{server_index}"

# Struct sizes (must match C++ rl_protocol.h)
HEADER_SIZE = 32        # SServerRLHeader
ACTION_SIZE = 32        # SAction
OBS_SIZE = 1400         # SStructuredObservation (CNN-optimized: tiles + player + checkpoint)
BOT_DATA_SIZE = 1448    # SBotData (legacy interleaved layout: 32 + 1400 + 16)

# Contiguous layout offsets (matches SContiguousServerRLShm in rl_protocol.h)
# Layout: [Header][Action0..N][Obs0..N][Reward0..N][Done0..N][Truncated0..N][BotClientId0..N]
CONTIGUOUS_ACTIONS_OFFSET = HEADER_SIZE  # 32
CONTIGUOUS_OBS_OFFSET = CONTIGUOUS_ACTIONS_OFFSET + MAX_RL_BOTS * ACTION_SIZE  # 32 + 128*32 = 4128
CONTIGUOUS_REWARDS_OFFSET = CONTIGUOUS_OBS_OFFSET + MAX_RL_BOTS * OBS_SIZE  # 4128 + 128*1400 = 183328
CONTIGUOUS_DONES_OFFSET = CONTIGUOUS_REWARDS_OFFSET + MAX_RL_BOTS * 4  # 183328 + 512 = 183840
CONTIGUOUS_TRUNCATED_OFFSET = CONTIGUOUS_DONES_OFFSET + MAX_RL_BOTS * 4  # 183840 + 512 = 184352
CONTIGUOUS_BOTIDS_OFFSET = CONTIGUOUS_TRUNCATED_OFFSET + MAX_RL_BOTS * 4  # 184352 + 512 = 184864
CONTIGUOUS_SHM_SIZE = CONTIGUOUS_BOTIDS_OFFSET + MAX_RL_BOTS * 4  # 184864 + 512 = 185376

# Global reference for signal handler cleanup
_active_server_env: Optional['TeeWorldsServerEnv'] = None


def _cleanup_on_exit():
    """Cleanup function called on exit."""
    global _active_server_env
    if _active_server_env is not None:
        try:
            _active_server_env.close()
        except:
            pass


def _signal_handler(signum, frame):
    """Handle Ctrl+C and other signals."""
    global _active_server_env
    print(f"\nReceived signal {signum}, cleaning up...")
    if _active_server_env is not None:
        try:
            _active_server_env.close()
        except:
            pass
    raise KeyboardInterrupt()


def _get_default_server_path() -> str:
    """Get the default server path relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, "../../../../../build/DDNet-Server"))


@dataclass
class ServerEnvConfig:
    """Configuration for the server-side bot environment."""
    server_path: str = None  # Will be resolved in __post_init__
    server_port: int = 8303
    sv_tickrate: int = 50
    frame_skip: int = 1      # 1 = no skip, 4 = act every 4th tick (decision rate = tickrate/frame_skip)
    max_episode_steps: int = 2000
    startup_wait: float = 3.0
    action_timeout: float = 0.5  # Timeout waiting for observations (increased for debugging)
    use_dict_obs: bool = False   # Return dict observations for CNN policy
    use_embeddings: bool = False # Return raw tile IDs for embedding lookup (requires use_dict_obs=True)
    map_name: str = "Tutorial"   # Map to load (use novice maps for training)
    sv_name: str = "RL Training" # Server name
    allow_spectators: bool = True  # Allow connecting to watch
    debug_timing: bool = False   # Print step timing debug info

    def __post_init__(self):
        if self.server_path is None:
            self.server_path = _get_default_server_path()
        # use_embeddings implies use_dict_obs
        if self.use_embeddings:
            self.use_dict_obs = True


# Numba-accelerated CNN observation parsing
@jit(nopython=True, cache=True, parallel=True)
def _parse_flat_obs_numba(raw_arr: np.ndarray, n: int,
                           features: np.ndarray,  # (n, 68) - player(15) + checkpoint(4) + tiles(49)
                           rewards: np.ndarray,
                           dones: np.ndarray) -> None:
    """Parse flat observations for MLP policy - optimized for speed.

    Only extracts what's needed: player state, checkpoint, and 7x7 center tile crop.
    Much faster than CNN parsing since we skip the full 43x31 tile grid.
    """
    # Offsets within SBotData (1448 bytes per bot)
    OBS_START = 32  # After SAction
    LOCAL_PLAYER = OBS_START + 8  # After 8-byte header
    TILES = LOCAL_PLAYER + 36     # After SPlayerObs
    EXTRA_STATE = TILES + 1333    # After tile grid
    CHECKPOINT = EXTRA_STATE + 4  # After extra state
    REWARD_OFF = OBS_START + 1400 # After observation
    DONE_OFF = REWARD_OFF + 4
    BOT_SIZE = 1448

    # Tile grid center offsets (player is at center of 43x31 grid)
    # Center is at (21, 15), so 7x7 crop is (18:25, 12:19)
    TILE_CENTER_X = 21
    TILE_CENTER_Y = 15

    for i in prange(n):
        base = i * BOT_SIZE

        # Reward and done
        rewards[i] = raw_arr[base + REWARD_OFF:base + REWARD_OFF + 4].view(np.float32)[0]
        dones[i] = raw_arr[base + DONE_OFF:base + DONE_OFF + 4].view(np.int32)[0] != 0

        # Local player floats (pos_x, pos_y, vel_x, vel_y)
        pf = raw_arr[base + LOCAL_PLAYER:base + LOCAL_PLAYER + 16].view(np.float32)
        pos_x, pos_y, vel_x, vel_y = pf[0], pf[1], pf[2], pf[3]

        # Single-byte fields from SPlayerObs
        direction = np.int8(raw_arr[base + LOCAL_PLAYER + 16])
        jumped = raw_arr[base + LOCAL_PLAYER + 17]
        hook_state = raw_arr[base + LOCAL_PLAYER + 18]
        is_frozen = raw_arr[base + LOCAL_PLAYER + 30]
        health = raw_arr[base + LOCAL_PLAYER + 32]
        armor = raw_arr[base + LOCAL_PLAYER + 33]

        # Extra state
        is_grounded = raw_arr[base + EXTRA_STATE]
        jumps_remaining = raw_arr[base + EXTRA_STATE + 1]
        can_hook = raw_arr[base + EXTRA_STATE + 2]

        # Player state (15 features) - indices 0-14
        features[i, 0] = vel_x / 50.0
        features[i, 1] = vel_y / 50.0
        features[i, 2] = float(is_grounded)
        features[i, 3] = float(jumps_remaining) / 2.0
        features[i, 4] = float(can_hook)
        features[i, 5] = float(direction)
        features[i, 6] = float(is_frozen)
        features[i, 7] = 1.0 if hook_state == 0 else 0.0
        features[i, 8] = 1.0 if hook_state == 1 else 0.0
        features[i, 9] = 1.0 if hook_state == 2 else 0.0
        features[i, 10] = 0.0  # aim placeholder
        features[i, 11] = 0.0  # aim placeholder
        features[i, 12] = float(health) / 10.0
        features[i, 13] = float(armor) / 10.0
        features[i, 14] = float(jumped) / 3.0

        # Checkpoint (4 features) - indices 15-18
        cp_floats = raw_arr[base + CHECKPOINT:base + CHECKPOINT + 8].view(np.float32)
        cp_ints = raw_arr[base + CHECKPOINT + 8:base + CHECKPOINT + 16].view(np.int32)
        cp_x, cp_y = cp_floats[0], cp_floats[1]
        cp_idx, total_cp = float(cp_ints[0]), float(cp_ints[1])

        features[i, 15] = max(-1.0, min(1.0, (cp_x - pos_x) / 2000.0))
        features[i, 16] = max(-1.0, min(1.0, (cp_y - pos_y) / 2000.0))
        features[i, 17] = cp_idx / max(1.0, total_cp)
        features[i, 18] = total_cp / 20.0

        # 7x7 center tile crop (49 features) - indices 19-67
        for ty in range(7):
            for tx in range(7):
                grid_y = TILE_CENTER_Y - 3 + ty  # 12 to 18
                grid_x = TILE_CENTER_X - 3 + tx  # 18 to 24
                tile_offset = base + TILES + grid_y * 43 + grid_x
                features[i, 19 + ty * 7 + tx] = float(raw_arr[tile_offset]) / 15.0


@jit(nopython=True, cache=True, parallel=True)
def _parse_cnn_obs_numba(raw_arr: np.ndarray, n: int,
                          player_state: np.ndarray, # (n, 15)
                          checkpoint: np.ndarray,   # (n, 4)
                          rewards: np.ndarray,
                          dones: np.ndarray) -> None:
    """Parse CNN observations (except tiles) from raw bytes using Numba JIT.

    Tiles are parsed separately using numpy vectorized ops for speed.
    """
    # Offsets within SBotData (1448 bytes per bot)
    OBS_START = 32  # After SAction
    LOCAL_PLAYER = OBS_START + 8  # After 8-byte header
    TILES = LOCAL_PLAYER + 36     # After SPlayerObs
    EXTRA_STATE = TILES + 1333    # After tile grid
    CHECKPOINT = EXTRA_STATE + 4  # After extra state
    REWARD_OFF = OBS_START + 1400 # After observation
    DONE_OFF = REWARD_OFF + 4
    BOT_SIZE = 1448

    for i in prange(n):
        base = i * BOT_SIZE

        # Reward and done
        rewards[i] = raw_arr[base + REWARD_OFF:base + REWARD_OFF + 4].view(np.float32)[0]
        dones[i] = raw_arr[base + DONE_OFF:base + DONE_OFF + 4].view(np.int32)[0] != 0

        # Local player floats (pos_x, pos_y, vel_x, vel_y)
        pf = raw_arr[base + LOCAL_PLAYER:base + LOCAL_PLAYER + 16].view(np.float32)
        pos_x, pos_y, vel_x, vel_y = pf[0], pf[1], pf[2], pf[3]

        # Single-byte fields from SPlayerObs
        direction = np.int8(raw_arr[base + LOCAL_PLAYER + 16])
        jumped = raw_arr[base + LOCAL_PLAYER + 17]
        hook_state = raw_arr[base + LOCAL_PLAYER + 18]
        is_frozen = raw_arr[base + LOCAL_PLAYER + 30]
        health = raw_arr[base + LOCAL_PLAYER + 32]
        armor = raw_arr[base + LOCAL_PLAYER + 33]

        # Extra state
        is_grounded = raw_arr[base + EXTRA_STATE]
        jumps_remaining = raw_arr[base + EXTRA_STATE + 1]
        can_hook = raw_arr[base + EXTRA_STATE + 2]

        # Player state (15 features)
        player_state[i, 0] = vel_x / 50.0
        player_state[i, 1] = vel_y / 50.0
        player_state[i, 2] = float(is_grounded)
        player_state[i, 3] = float(jumps_remaining) / 2.0
        player_state[i, 4] = float(can_hook)
        player_state[i, 5] = float(direction)
        player_state[i, 6] = float(is_frozen)
        player_state[i, 7] = 1.0 if hook_state == 0 else 0.0
        player_state[i, 8] = 1.0 if hook_state == 1 else 0.0
        player_state[i, 9] = 1.0 if hook_state == 2 else 0.0
        player_state[i, 10] = 0.0  # aim placeholder
        player_state[i, 11] = 0.0  # aim placeholder
        player_state[i, 12] = float(health) / 10.0
        player_state[i, 13] = float(armor) / 10.0
        player_state[i, 14] = float(jumped) / 3.0

        # Checkpoint (4 features)
        cp_floats = raw_arr[base + CHECKPOINT:base + CHECKPOINT + 8].view(np.float32)
        cp_ints = raw_arr[base + CHECKPOINT + 8:base + CHECKPOINT + 16].view(np.int32)
        cp_x, cp_y = cp_floats[0], cp_floats[1]
        cp_idx, total_cp = float(cp_ints[0]), float(cp_ints[1])

        checkpoint[i, 0] = max(-1.0, min(1.0, (cp_x - pos_x) / 2000.0))
        checkpoint[i, 1] = max(-1.0, min(1.0, (cp_y - pos_y) / 2000.0))
        checkpoint[i, 2] = cp_idx / max(1.0, total_cp)
        checkpoint[i, 3] = total_cp / 20.0


# ============================================================================
# Contiguous Layout Numba Functions (for optimized SHM layout)
# ============================================================================

@jit(nopython=True, cache=True, parallel=True)
def _parse_flat_obs_contiguous_numba(obs_arr: np.ndarray, n: int,
                                       features: np.ndarray,
                                       rewards: np.ndarray,
                                       dones: np.ndarray) -> None:
    """Parse flat observations from CONTIGUOUS layout for MLP policy.

    obs_arr: shape (n, 1400) - contiguous observations
    rewards/dones: already extracted from contiguous arrays
    """
    # Offsets within SStructuredObservation (1400 bytes)
    LOCAL_PLAYER = 8
    TILES = LOCAL_PLAYER + 36  # 44
    EXTRA_STATE = TILES + 1333  # 1377
    CHECKPOINT = EXTRA_STATE + 4  # 1381

    # Tile grid center (player at center of 43x31)
    TILE_CENTER_X = 21
    TILE_CENTER_Y = 15

    for i in prange(n):
        # Local player floats (pos_x, pos_y, vel_x, vel_y)
        pf = obs_arr[i, LOCAL_PLAYER:LOCAL_PLAYER + 16].view(np.float32)
        pos_x, pos_y, vel_x, vel_y = pf[0], pf[1], pf[2], pf[3]

        # Single-byte fields from SPlayerObs
        direction = np.int8(obs_arr[i, LOCAL_PLAYER + 16])
        jumped = obs_arr[i, LOCAL_PLAYER + 17]
        hook_state = obs_arr[i, LOCAL_PLAYER + 18]
        is_frozen = obs_arr[i, LOCAL_PLAYER + 30]
        health = obs_arr[i, LOCAL_PLAYER + 32]
        armor = obs_arr[i, LOCAL_PLAYER + 33]

        # Extra state
        is_grounded = obs_arr[i, EXTRA_STATE]
        jumps_remaining = obs_arr[i, EXTRA_STATE + 1]
        can_hook = obs_arr[i, EXTRA_STATE + 2]

        # Player state (15 features) - indices 0-14
        features[i, 0] = vel_x / 50.0
        features[i, 1] = vel_y / 50.0
        features[i, 2] = float(is_grounded)
        features[i, 3] = float(jumps_remaining) / 2.0
        features[i, 4] = float(can_hook)
        features[i, 5] = float(direction)
        features[i, 6] = float(is_frozen)
        features[i, 7] = 1.0 if hook_state == 0 else 0.0
        features[i, 8] = 1.0 if hook_state == 1 else 0.0
        features[i, 9] = 1.0 if hook_state == 2 else 0.0
        features[i, 10] = 0.0  # aim placeholder
        features[i, 11] = 0.0  # aim placeholder
        features[i, 12] = float(health) / 10.0
        features[i, 13] = float(armor) / 10.0
        features[i, 14] = float(jumped) / 3.0

        # Checkpoint (4 features) - indices 15-18
        cp_floats = obs_arr[i, CHECKPOINT:CHECKPOINT + 8].view(np.float32)
        cp_ints = obs_arr[i, CHECKPOINT + 8:CHECKPOINT + 16].view(np.int32)
        cp_x, cp_y = cp_floats[0], cp_floats[1]
        cp_idx, total_cp = float(cp_ints[0]), float(cp_ints[1])

        features[i, 15] = max(-1.0, min(1.0, (cp_x - pos_x) / 2000.0))
        features[i, 16] = max(-1.0, min(1.0, (cp_y - pos_y) / 2000.0))
        features[i, 17] = cp_idx / max(1.0, total_cp)
        features[i, 18] = total_cp / 20.0

        # 7x7 center tile crop (49 features) - indices 19-67
        for ty in range(7):
            for tx in range(7):
                grid_y = TILE_CENTER_Y - 3 + ty  # 12 to 18
                grid_x = TILE_CENTER_X - 3 + tx  # 18 to 24
                tile_offset = TILES + grid_y * 43 + grid_x
                features[i, 19 + ty * 7 + tx] = float(obs_arr[i, tile_offset]) / 15.0


def _parse_tiles_vectorized(raw_arr: np.ndarray, n: int, tiles: np.ndarray) -> None:
    """Parse tile grids using numpy strided view (no Python loops).

    Uses as_strided to view non-contiguous tile data as a single array,
    then processes all bots at once with vectorized operations.
    """
    from numpy.lib.stride_tricks import as_strided

    BOT_SIZE = 1448
    TILES_OFFSET = 32 + 8 + 36  # OBS_START + header + LocalPlayer = 76
    TILE_SIZE = 1333  # 31 * 43

    # Use strided view to access all tile data at once without copying
    # Each bot's tiles are at offset (i * BOT_SIZE + TILES_OFFSET)
    # Stride between bots is BOT_SIZE, stride within tiles is 1
    tile_view = as_strided(
        raw_arr[TILES_OFFSET:],  # Start at first bot's tiles
        shape=(n, TILE_SIZE),
        strides=(BOT_SIZE, 1),
        writeable=False
    )

    # Single vectorized operation for all bots: convert to float, normalize, reshape
    # This is ~100x faster than a Python loop
    tiles[:, 0, :, :] = tile_view.astype(np.float32).reshape(n, 31, 43) * (1.0 / 15.0)


# Pre-compile Numba functions at module load (avoids compilation delay in threads)
_warmup_numba_jit()


class ServerShmClient:
    """
    Shared memory client for server-side bot communication.

    Communicates with the DDNet server's bot manager via a single shared memory region.
    """

    def __init__(self, num_bots: int, server_index: int = 0):
        self.num_bots = num_bots
        self.server_index = server_index
        self.shm_path = get_shm_path(server_index)
        self.shm_file = None
        self.shm_mmap = None
        self._connected = False

        # Use contiguous layout size
        self.total_size = CONTIGUOUS_SHM_SIZE

        # Pre-allocate action buffer for single memcpy writes
        self._action_buffer = np.zeros((MAX_RL_BOTS, 8), dtype=np.int32)

    def connect(self, max_retries: int = 60, retry_delay: float = 0.5) -> bool:
        """Connect to the server's shared memory."""
        for attempt in range(max_retries):
            if os.path.exists(self.shm_path):
                try:
                    self.shm_file = open(self.shm_path, 'r+b')
                    self.shm_mmap = mmap.mmap(self.shm_file.fileno(), self.total_size)
                    self._connected = True
                    print(f"Connected to server shared memory at {self.shm_path}")
                    return True
                except Exception as e:
                    print(f"Connection attempt {attempt + 1} failed: {e}")
                    if self.shm_file:
                        self.shm_file.close()
                        self.shm_file = None

            if attempt % 10 == 9:
                print(f"Waiting for server to create shared memory... ({attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

        print(f"Failed to connect to {self.shm_path}")
        return False

    def disconnect(self):
        """Disconnect from shared memory."""
        if self.shm_mmap:
            # Signal shutdown
            self._write_header_field('shutdown', 1)
            self.shm_mmap.close()
            self.shm_mmap = None
        if self.shm_file:
            self.shm_file.close()
            self.shm_file = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _read_header(self) -> dict:
        """Read the SServerRLHeader."""
        self.shm_mmap.seek(0)
        data = self.shm_mmap.read(HEADER_SIZE)

        # Unpack header: server_ready, python_ready, shutdown, padding, num_bots, server_tick, padding
        (server_ready, python_ready, shutdown, _pad1,
         num_bots, server_tick, _pad2a, _pad2b) = struct.unpack('<iiii iiii', data)

        return {
            'server_ready': server_ready,
            'python_ready': python_ready,
            'shutdown': shutdown,
            'num_bots': num_bots,
            'server_tick': server_tick,
        }

    def _write_header_field(self, field: str, value: int):
        """Write a field to the header."""
        offsets = {
            'server_ready': 0,
            'python_ready': 4,
            'shutdown': 8,
            'paused': 12,
        }
        if field in offsets:
            self.shm_mmap.seek(offsets[field])
            self.shm_mmap.write(struct.pack('<i', value))

    def _get_bot_data_offset(self, bot_index: int) -> int:
        """Get the offset to a bot's data in shared memory."""
        return HEADER_SIZE + (bot_index * BOT_DATA_SIZE)

    def write_action(self, bot_index: int, direction: int = 0, jump: int = 0,
                     hook: int = 0, fire: int = 0, aim_x: int = 0, aim_y: int = 0,
                     weapon: int = 0, reset: int = 0):
        """Write an action for a specific bot."""
        if not self._connected or bot_index >= self.num_bots:
            return

        offset = self._get_bot_data_offset(bot_index)
        self.shm_mmap.seek(offset)

        # SAction struct: direction, jump, hook, fire, aim_x, aim_y, weapon, reset
        action_data = struct.pack('<iiii iiii',
            direction, jump, hook, fire, aim_x, aim_y, weapon, reset)
        self.shm_mmap.write(action_data)

    def write_actions_batch(self, actions: Dict[str, np.ndarray]):
        """Write actions for all bots at once (single memcpy with contiguous layout)."""
        discrete_actions = actions['discrete']  # (num_bots, 3)
        continuous_actions = actions['continuous']  # (num_bots, 2)
        resets = actions.get('reset', np.zeros(self.num_bots, dtype=np.int32))

        # Vectorized action computation
        n = self.num_bots
        direction = discrete_actions[:, 0].astype(np.int32) - 1  # 0,1,2 -> -1,0,1
        jump = discrete_actions[:, 1].astype(np.int32)
        hook = discrete_actions[:, 2].astype(np.int32)

        # Continuous aim (vectorized)
        aim_angle = continuous_actions[:, 0] * np.pi
        aim_magnitude = (continuous_actions[:, 1] + 1.0) / 2.0
        aim_x = (np.cos(aim_angle) * aim_magnitude * 200).astype(np.int32)
        aim_y = (np.sin(aim_angle) * aim_magnitude * 200).astype(np.int32)

        # Build action array using pre-allocated buffer
        # SAction layout: direction, jump, hook, fire, aim_x, aim_y, weapon, reset
        self._action_buffer[:n, 0] = direction
        self._action_buffer[:n, 1] = jump
        self._action_buffer[:n, 2] = hook
        self._action_buffer[:n, 3] = 0  # fire
        self._action_buffer[:n, 4] = aim_x
        self._action_buffer[:n, 5] = aim_y
        self._action_buffer[:n, 6] = 0  # weapon
        self._action_buffer[:n, 7] = resets.astype(np.int32)

        # CONTIGUOUS LAYOUT: Single memcpy for all actions!
        # Actions are stored contiguously at CONTIGUOUS_ACTIONS_OFFSET
        self.shm_mmap.seek(CONTIGUOUS_ACTIONS_OFFSET)
        self.shm_mmap.write(self._action_buffer[:n].tobytes())

    def read_observation(self, bot_index: int) -> Tuple[Optional[StructuredObservation], float, bool]:
        """Read observation, reward, and done flag for a specific bot."""
        if not self._connected or bot_index >= self.num_bots:
            return None, 0.0, True

        offset = self._get_bot_data_offset(bot_index)

        # Skip action (32 bytes) to get to observation
        obs_offset = offset + ACTION_SIZE
        self.shm_mmap.seek(obs_offset)
        obs_data = self.shm_mmap.read(OBS_SIZE)

        # Read reward, done, truncated after observation
        self.shm_mmap.seek(obs_offset + OBS_SIZE)
        reward, done, truncated, bot_client_id = struct.unpack('<fiii', self.shm_mmap.read(16))

        try:
            obs = StructuredObservation.from_bytes(obs_data)
            return obs, reward, bool(done)
        except Exception as e:
            print(f"Error parsing observation for bot {bot_index}: {e}")
            return None, 0.0, True

    def read_observations_batch(self) -> Tuple[List[Optional[StructuredObservation]], np.ndarray, np.ndarray]:
        """Read all observations, rewards, and dones at once."""
        observations = []
        rewards = np.zeros(self.num_bots, dtype=np.float32)
        dones = np.zeros(self.num_bots, dtype=bool)

        for i in range(self.num_bots):
            obs, reward, done = self.read_observation(i)
            observations.append(obs)
            rewards[i] = reward
            dones[i] = done

        return observations, rewards, dones

    def read_observations_batch_fast(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read all observations as flat arrays directly (for MLP policy).

        Uses CONTIGUOUS layout for efficient single-memcpy reads.

        Returns:
            observations: np.ndarray of shape (num_bots, 68) - 15 player + 4 checkpoint + 49 tiles
            rewards: np.ndarray of shape (num_bots,)
            dones: np.ndarray of shape (num_bots,)
        """
        n = self.num_bots

        # CONTIGUOUS LAYOUT: Read observations, rewards, dones in single memcpy ops
        self.shm_mmap.seek(CONTIGUOUS_OBS_OFFSET)
        obs_raw = self.shm_mmap.read(n * OBS_SIZE)
        obs_arr = np.frombuffer(obs_raw, dtype=np.uint8).reshape(n, OBS_SIZE).copy()

        # Read rewards (contiguous float32 array)
        self.shm_mmap.seek(CONTIGUOUS_REWARDS_OFFSET)
        rewards = np.frombuffer(self.shm_mmap.read(n * 4), dtype=np.float32).copy()

        # Read dones (contiguous int32 array)
        self.shm_mmap.seek(CONTIGUOUS_DONES_OFFSET)
        dones = np.frombuffer(self.shm_mmap.read(n * 4), dtype=np.int32).astype(np.bool_)

        # Pre-allocate outputs - 68 features (15 + 4 + 49)
        features = np.zeros((n, 68), dtype=np.float32)

        # Use Numba-accelerated parsing if available
        if NUMBA_AVAILABLE:
            _parse_flat_obs_contiguous_numba(obs_arr, n, features, rewards, dones)
        else:
            self._parse_observations_flat_contiguous_python(obs_arr, n, features)

        # Sanitize NaN/inf (fast in-place operation)
        np.nan_to_num(features, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        return features, rewards, dones

    def read_observations_batch_cnn(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Read all observations as dict for CNN policy.

        Uses CONTIGUOUS layout for efficient single-memcpy reads.
        """
        n = self.num_bots

        # CONTIGUOUS LAYOUT: Read observations, rewards, dones in single memcpy ops
        # Observations are stored contiguously at CONTIGUOUS_OBS_OFFSET
        self.shm_mmap.seek(CONTIGUOUS_OBS_OFFSET)
        obs_raw = self.shm_mmap.read(n * OBS_SIZE)
        obs_arr = np.frombuffer(obs_raw, dtype=np.uint8).reshape(n, OBS_SIZE)

        # Read rewards (contiguous float32 array)
        self.shm_mmap.seek(CONTIGUOUS_REWARDS_OFFSET)
        rewards = np.frombuffer(self.shm_mmap.read(n * 4), dtype=np.float32).copy()

        # Read dones (contiguous int32 array)
        self.shm_mmap.seek(CONTIGUOUS_DONES_OFFSET)
        dones = np.frombuffer(self.shm_mmap.read(n * 4), dtype=np.int32).astype(np.bool_)

        # Pre-allocate outputs
        tiles = np.empty((n, 1, 31, 43), dtype=np.float32)
        player_state = np.empty((n, 15), dtype=np.float32)
        checkpoint = np.empty((n, 4), dtype=np.float32)

        # Offsets within SStructuredObservation (1400 bytes)
        # Layout: header(8) + LocalPlayer(36) + TileGrid(1333) + ExtraState(4) + Checkpoint(16) + Reserved(3)
        LOCAL_PLAYER = 8
        TILES = LOCAL_PLAYER + 36  # 44
        EXTRA_STATE = TILES + 1333  # 1377
        CHECKPOINT = EXTRA_STATE + 4  # 1381

        # ===== TILES: direct contiguous access, no strided view needed! =====
        # Each observation's tiles are at bytes 44:1377 (1333 bytes = 31*43)
        tile_data = obs_arr[:, TILES:TILES+1333].reshape(n, 31, 43)
        np.multiply(tile_data, 1.0/15.0, out=tiles[:, 0, :, :], casting='unsafe')

        # ===== PLAYER STATE: extract from contiguous observations =====
        # Float fields: pos_x(4), pos_y(4), vel_x(4), vel_y(4) at LOCAL_PLAYER
        float_data = obs_arr[:, LOCAL_PLAYER:LOCAL_PLAYER+16].view(np.float32).reshape(n, 4)
        pos_x = float_data[:, 0]
        pos_y = float_data[:, 1]
        vel_x = float_data[:, 2]
        vel_y = float_data[:, 3]

        # Byte fields
        direction = obs_arr[:, LOCAL_PLAYER + 16].astype(np.int8).astype(np.float32)
        jumped = obs_arr[:, LOCAL_PLAYER + 17].astype(np.float32)
        hook_state = obs_arr[:, LOCAL_PLAYER + 18].astype(np.float32)
        is_frozen = obs_arr[:, LOCAL_PLAYER + 30].astype(np.float32)
        health = obs_arr[:, LOCAL_PLAYER + 32].astype(np.float32)
        armor = obs_arr[:, LOCAL_PLAYER + 33].astype(np.float32)

        # Extra state
        is_grounded = obs_arr[:, EXTRA_STATE].astype(np.float32)
        jumps_remaining = obs_arr[:, EXTRA_STATE + 1].astype(np.float32)
        can_hook = obs_arr[:, EXTRA_STATE + 2].astype(np.float32)

        player_state[:, 0] = vel_x / 50.0
        player_state[:, 1] = vel_y / 50.0
        player_state[:, 2] = is_grounded
        player_state[:, 3] = jumps_remaining / 2.0
        player_state[:, 4] = can_hook
        player_state[:, 5] = direction
        player_state[:, 6] = is_frozen
        player_state[:, 7] = (hook_state == 0).astype(np.float32)
        player_state[:, 8] = (hook_state == 1).astype(np.float32)
        player_state[:, 9] = (hook_state == 2).astype(np.float32)
        player_state[:, 10] = 0.0
        player_state[:, 11] = 0.0
        player_state[:, 12] = health / 10.0
        player_state[:, 13] = armor / 10.0
        player_state[:, 14] = jumped / 3.0

        # ===== CHECKPOINT =====
        cp_floats = obs_arr[:, CHECKPOINT:CHECKPOINT+8].view(np.float32).reshape(n, 2)
        cp_ints = obs_arr[:, CHECKPOINT+8:CHECKPOINT+16].view(np.int32).reshape(n, 2)
        cp_x = cp_floats[:, 0]
        cp_y = cp_floats[:, 1]
        cp_idx = cp_ints[:, 0].astype(np.float32)
        total_cp = cp_ints[:, 1].astype(np.float32)

        checkpoint[:, 0] = np.clip((cp_x - pos_x) / 2000.0, -1.0, 1.0)
        checkpoint[:, 1] = np.clip((cp_y - pos_y) / 2000.0, -1.0, 1.0)
        checkpoint[:, 2] = cp_idx / np.maximum(1.0, total_cp)
        checkpoint[:, 3] = total_cp / 20.0

        # Sanitize NaN/inf
        np.nan_to_num(player_state, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.nan_to_num(checkpoint, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        return {'tiles': tiles, 'player_state': player_state, 'checkpoint': checkpoint}, rewards, dones

    def read_observations_batch_embed(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Read all observations for embedding-based policy.

        Returns raw tile IDs (int64) for nn.Embedding lookup instead of normalized floats.
        Uses CONTIGUOUS layout for efficient single-memcpy reads.

        Returns:
            observations: Dict with:
                'tiles': np.ndarray(shape=(n, 31, 43), dtype=np.int64) - raw tile IDs 0-255
                'player_state': np.ndarray(shape=(n, 15), dtype=np.float32)
                'checkpoint': np.ndarray(shape=(n, 4), dtype=np.float32)
            rewards: np.ndarray of shape (n,)
            dones: np.ndarray of shape (n,)
        """
        n = self.num_bots

        # CONTIGUOUS LAYOUT: Read observations, rewards, dones in single memcpy ops
        self.shm_mmap.seek(CONTIGUOUS_OBS_OFFSET)
        obs_raw = self.shm_mmap.read(n * OBS_SIZE)
        obs_arr = np.frombuffer(obs_raw, dtype=np.uint8).reshape(n, OBS_SIZE)

        # Read rewards (contiguous float32 array)
        self.shm_mmap.seek(CONTIGUOUS_REWARDS_OFFSET)
        rewards = np.frombuffer(self.shm_mmap.read(n * 4), dtype=np.float32).copy()

        # Read dones (contiguous int32 array)
        self.shm_mmap.seek(CONTIGUOUS_DONES_OFFSET)
        dones = np.frombuffer(self.shm_mmap.read(n * 4), dtype=np.int32).astype(np.bool_)

        # Pre-allocate outputs
        tiles = np.empty((n, 31, 43), dtype=np.int64)
        player_state = np.empty((n, 15), dtype=np.float32)
        checkpoint = np.empty((n, 4), dtype=np.float32)

        # Offsets within SStructuredObservation (1400 bytes)
        LOCAL_PLAYER = 8
        TILES = LOCAL_PLAYER + 36  # 44
        EXTRA_STATE = TILES + 1333  # 1377
        CHECKPOINT = EXTRA_STATE + 4  # 1381

        # ===== TILES: cast to int64 for embedding =====
        tile_data = obs_arr[:, TILES:TILES+1333].reshape(n, 31, 43)
        tiles[:] = tile_data.astype(np.int64)

        # ===== PLAYER STATE =====
        float_data = obs_arr[:, LOCAL_PLAYER:LOCAL_PLAYER+16].view(np.float32).reshape(n, 4)
        pos_x = float_data[:, 0]
        pos_y = float_data[:, 1]
        vel_x = float_data[:, 2]
        vel_y = float_data[:, 3]

        direction = obs_arr[:, LOCAL_PLAYER + 16].astype(np.int8).astype(np.float32)
        jumped = obs_arr[:, LOCAL_PLAYER + 17].astype(np.float32)
        hook_state = obs_arr[:, LOCAL_PLAYER + 18].astype(np.float32)
        is_frozen = obs_arr[:, LOCAL_PLAYER + 30].astype(np.float32)
        health = obs_arr[:, LOCAL_PLAYER + 32].astype(np.float32)
        armor = obs_arr[:, LOCAL_PLAYER + 33].astype(np.float32)

        is_grounded = obs_arr[:, EXTRA_STATE].astype(np.float32)
        jumps_remaining = obs_arr[:, EXTRA_STATE + 1].astype(np.float32)
        can_hook = obs_arr[:, EXTRA_STATE + 2].astype(np.float32)

        player_state[:, 0] = vel_x / 50.0
        player_state[:, 1] = vel_y / 50.0
        player_state[:, 2] = is_grounded
        player_state[:, 3] = jumps_remaining / 2.0
        player_state[:, 4] = can_hook
        player_state[:, 5] = direction
        player_state[:, 6] = is_frozen
        player_state[:, 7] = (hook_state == 0).astype(np.float32)
        player_state[:, 8] = (hook_state == 1).astype(np.float32)
        player_state[:, 9] = (hook_state == 2).astype(np.float32)
        player_state[:, 10] = 0.0
        player_state[:, 11] = 0.0
        player_state[:, 12] = health / 10.0
        player_state[:, 13] = armor / 10.0
        player_state[:, 14] = jumped / 3.0

        # ===== CHECKPOINT =====
        cp_floats = obs_arr[:, CHECKPOINT:CHECKPOINT+8].view(np.float32).reshape(n, 2)
        cp_ints = obs_arr[:, CHECKPOINT+8:CHECKPOINT+16].view(np.int32).reshape(n, 2)
        cp_x = cp_floats[:, 0]
        cp_y = cp_floats[:, 1]
        cp_idx = cp_ints[:, 0].astype(np.float32)
        total_cp = cp_ints[:, 1].astype(np.float32)

        checkpoint[:, 0] = np.clip((cp_x - pos_x) / 2000.0, -1.0, 1.0)
        checkpoint[:, 1] = np.clip((cp_y - pos_y) / 2000.0, -1.0, 1.0)
        checkpoint[:, 2] = cp_idx / np.maximum(1.0, total_cp)
        checkpoint[:, 3] = total_cp / 20.0

        # Sanitize NaN/inf
        np.nan_to_num(player_state, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.nan_to_num(checkpoint, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        return {'tiles': tiles, 'player_state': player_state, 'checkpoint': checkpoint}, rewards, dones

    def _parse_observations_cnn_python(self, raw: bytes, raw_arr: np.ndarray, n: int,
                                         tiles: np.ndarray, player_state: np.ndarray,
                                         checkpoint: np.ndarray, rewards: np.ndarray,
                                         dones: np.ndarray) -> None:
        """Fallback Python parsing for CNN observations when Numba is not available.

        New struct layout (1448 bytes per bot):
        - SAction: 32 bytes
        - SStructuredObservation: 1400 bytes
          - Header: 8 bytes (tick, flags, state, reserved)
          - LocalPlayer: 36 bytes
          - TileGrid: 1333 bytes (43x31)
          - ExtraState: 4 bytes
          - Checkpoint: 16 bytes
          - Reserved: 3 bytes
        - Reward: 4 bytes
        - Done: 4 bytes
        - Truncated: 4 bytes
        - BotClientId: 4 bytes
        """
        # Offsets within SBotData (1448 bytes per bot)
        OBS_START = 32  # After SAction
        LOCAL_PLAYER = OBS_START + 8  # After 8-byte header
        TILES = LOCAL_PLAYER + 36     # After SPlayerObs
        EXTRA_STATE = TILES + 1333    # After tile grid
        CHECKPOINT = EXTRA_STATE + 4  # After extra state
        REWARD_OFF = OBS_START + 1400 # After observation
        DONE_OFF = REWARD_OFF + 4

        bot_offsets = np.arange(n) * BOT_DATA_SIZE

        # Extract rewards and dones
        for i, off in enumerate(bot_offsets):
            rewards[i] = struct.unpack_from('<f', raw, int(off) + REWARD_OFF)[0]
            dones[i] = struct.unpack_from('<i', raw, int(off) + DONE_OFF)[0] != 0

        # Extract player floats
        pos_x = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER)[0] for o in bot_offsets])
        pos_y = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER + 4)[0] for o in bot_offsets])
        vel_x = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER + 8)[0] for o in bot_offsets])
        vel_y = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER + 12)[0] for o in bot_offsets])

        # Single-byte fields
        direction = raw_arr[bot_offsets + LOCAL_PLAYER + 16].astype(np.int8).astype(np.float32)
        jumped = raw_arr[bot_offsets + LOCAL_PLAYER + 17].astype(np.float32)
        hook_state = raw_arr[bot_offsets + LOCAL_PLAYER + 18].astype(np.float32)
        is_frozen = raw_arr[bot_offsets + LOCAL_PLAYER + 30].astype(np.float32)
        health = raw_arr[bot_offsets + LOCAL_PLAYER + 32].astype(np.float32)
        armor = raw_arr[bot_offsets + LOCAL_PLAYER + 33].astype(np.float32)
        is_grounded = raw_arr[bot_offsets + EXTRA_STATE].astype(np.float32)
        jumps_remaining = raw_arr[bot_offsets + EXTRA_STATE + 1].astype(np.float32)
        can_hook = raw_arr[bot_offsets + EXTRA_STATE + 2].astype(np.float32)

        # Player state (15 features)
        player_state[:, 0] = vel_x / 50.0
        player_state[:, 1] = vel_y / 50.0
        player_state[:, 2] = is_grounded
        player_state[:, 3] = jumps_remaining / 2.0
        player_state[:, 4] = can_hook
        player_state[:, 5] = direction
        player_state[:, 6] = is_frozen
        player_state[:, 7] = (hook_state == 0).astype(np.float32)
        player_state[:, 8] = (hook_state == 1).astype(np.float32)
        player_state[:, 9] = (hook_state == 2).astype(np.float32)
        player_state[:, 10] = 0.0  # aim placeholder
        player_state[:, 11] = 0.0  # aim placeholder
        player_state[:, 12] = health / 10.0
        player_state[:, 13] = armor / 10.0
        player_state[:, 14] = jumped / 3.0

        # Checkpoint (4 features)
        cp_x = np.array([struct.unpack_from('<f', raw, int(o) + CHECKPOINT)[0] for o in bot_offsets])
        cp_y = np.array([struct.unpack_from('<f', raw, int(o) + CHECKPOINT + 4)[0] for o in bot_offsets])
        cp_idx = np.array([struct.unpack_from('<i', raw, int(o) + CHECKPOINT + 8)[0] for o in bot_offsets], dtype=np.float32)
        total_cp = np.array([struct.unpack_from('<i', raw, int(o) + CHECKPOINT + 12)[0] for o in bot_offsets], dtype=np.float32)

        checkpoint[:, 0] = np.clip((cp_x - pos_x) / 2000.0, -1.0, 1.0)
        checkpoint[:, 1] = np.clip((cp_y - pos_y) / 2000.0, -1.0, 1.0)
        checkpoint[:, 2] = cp_idx / np.maximum(1.0, total_cp)
        checkpoint[:, 3] = total_cp / 20.0

        # Full tile grid (31, 43) - normalized by 15
        for i in range(n):
            tile_base = int(bot_offsets[i]) + TILES
            for ty in range(31):
                for tx in range(43):
                    tiles[i, 0, ty, tx] = float(raw_arr[tile_base + ty * 43 + tx]) / 15.0

    def _parse_observations_flat_python(self, raw: bytes, raw_arr: np.ndarray, n: int,
                                         features: np.ndarray, rewards: np.ndarray,
                                         dones: np.ndarray) -> None:
        """Fallback Python parsing for flat MLP observations when Numba is not available."""
        OBS_START = 32
        LOCAL_PLAYER = OBS_START + 8
        TILES = LOCAL_PLAYER + 36
        EXTRA_STATE = TILES + 1333
        CHECKPOINT = EXTRA_STATE + 4
        REWARD_OFF = OBS_START + 1400
        DONE_OFF = REWARD_OFF + 4

        # Tile grid center
        TILE_CENTER_X = 21
        TILE_CENTER_Y = 15

        bot_offsets = np.arange(n) * BOT_DATA_SIZE

        # Extract rewards and dones
        for i, off in enumerate(bot_offsets):
            rewards[i] = struct.unpack_from('<f', raw, int(off) + REWARD_OFF)[0]
            dones[i] = struct.unpack_from('<i', raw, int(off) + DONE_OFF)[0] != 0

        # Extract player floats
        pos_x = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER)[0] for o in bot_offsets])
        pos_y = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER + 4)[0] for o in bot_offsets])
        vel_x = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER + 8)[0] for o in bot_offsets])
        vel_y = np.array([struct.unpack_from('<f', raw, int(o) + LOCAL_PLAYER + 12)[0] for o in bot_offsets])

        # Single-byte fields
        direction = raw_arr[bot_offsets + LOCAL_PLAYER + 16].astype(np.int8).astype(np.float32)
        jumped = raw_arr[bot_offsets + LOCAL_PLAYER + 17].astype(np.float32)
        hook_state = raw_arr[bot_offsets + LOCAL_PLAYER + 18].astype(np.float32)
        is_frozen = raw_arr[bot_offsets + LOCAL_PLAYER + 30].astype(np.float32)
        health = raw_arr[bot_offsets + LOCAL_PLAYER + 32].astype(np.float32)
        armor = raw_arr[bot_offsets + LOCAL_PLAYER + 33].astype(np.float32)
        is_grounded = raw_arr[bot_offsets + EXTRA_STATE].astype(np.float32)
        jumps_remaining = raw_arr[bot_offsets + EXTRA_STATE + 1].astype(np.float32)
        can_hook = raw_arr[bot_offsets + EXTRA_STATE + 2].astype(np.float32)

        # Player state (15 features) - indices 0-14
        features[:, 0] = vel_x / 50.0
        features[:, 1] = vel_y / 50.0
        features[:, 2] = is_grounded
        features[:, 3] = jumps_remaining / 2.0
        features[:, 4] = can_hook
        features[:, 5] = direction
        features[:, 6] = is_frozen
        features[:, 7] = (hook_state == 0).astype(np.float32)
        features[:, 8] = (hook_state == 1).astype(np.float32)
        features[:, 9] = (hook_state == 2).astype(np.float32)
        features[:, 10] = 0.0
        features[:, 11] = 0.0
        features[:, 12] = health / 10.0
        features[:, 13] = armor / 10.0
        features[:, 14] = jumped / 3.0

        # Checkpoint (4 features) - indices 15-18
        cp_x = np.array([struct.unpack_from('<f', raw, int(o) + CHECKPOINT)[0] for o in bot_offsets])
        cp_y = np.array([struct.unpack_from('<f', raw, int(o) + CHECKPOINT + 4)[0] for o in bot_offsets])
        cp_idx = np.array([struct.unpack_from('<i', raw, int(o) + CHECKPOINT + 8)[0] for o in bot_offsets], dtype=np.float32)
        total_cp = np.array([struct.unpack_from('<i', raw, int(o) + CHECKPOINT + 12)[0] for o in bot_offsets], dtype=np.float32)

        features[:, 15] = np.clip((cp_x - pos_x) / 2000.0, -1.0, 1.0)
        features[:, 16] = np.clip((cp_y - pos_y) / 2000.0, -1.0, 1.0)
        features[:, 17] = cp_idx / np.maximum(1.0, total_cp)
        features[:, 18] = total_cp / 20.0

        # 7x7 center tile crop (49 features) - indices 19-67
        for ty in range(7):
            for tx in range(7):
                grid_y = TILE_CENTER_Y - 3 + ty
                grid_x = TILE_CENTER_X - 3 + tx
                tile_idx = grid_y * 43 + grid_x
                features[:, 19 + ty * 7 + tx] = raw_arr[bot_offsets + TILES + tile_idx].astype(np.float32) / 15.0

    def _parse_observations_flat_contiguous_python(self, obs_arr: np.ndarray, n: int,
                                                     features: np.ndarray) -> None:
        """Fallback Python parsing for flat MLP observations with CONTIGUOUS layout."""
        # Offsets within SStructuredObservation (1400 bytes)
        LOCAL_PLAYER = 8
        TILES = LOCAL_PLAYER + 36  # 44
        EXTRA_STATE = TILES + 1333  # 1377
        CHECKPOINT = EXTRA_STATE + 4  # 1381

        # Tile grid center
        TILE_CENTER_X = 21
        TILE_CENTER_Y = 15

        # Extract player floats
        float_data = obs_arr[:, LOCAL_PLAYER:LOCAL_PLAYER+16].view(np.float32).reshape(n, 4)
        pos_x = float_data[:, 0]
        pos_y = float_data[:, 1]
        vel_x = float_data[:, 2]
        vel_y = float_data[:, 3]

        # Single-byte fields
        direction = obs_arr[:, LOCAL_PLAYER + 16].astype(np.int8).astype(np.float32)
        jumped = obs_arr[:, LOCAL_PLAYER + 17].astype(np.float32)
        hook_state = obs_arr[:, LOCAL_PLAYER + 18].astype(np.float32)
        is_frozen = obs_arr[:, LOCAL_PLAYER + 30].astype(np.float32)
        health = obs_arr[:, LOCAL_PLAYER + 32].astype(np.float32)
        armor = obs_arr[:, LOCAL_PLAYER + 33].astype(np.float32)
        is_grounded = obs_arr[:, EXTRA_STATE].astype(np.float32)
        jumps_remaining = obs_arr[:, EXTRA_STATE + 1].astype(np.float32)
        can_hook = obs_arr[:, EXTRA_STATE + 2].astype(np.float32)

        # Player state (15 features) - indices 0-14
        features[:, 0] = vel_x / 50.0
        features[:, 1] = vel_y / 50.0
        features[:, 2] = is_grounded
        features[:, 3] = jumps_remaining / 2.0
        features[:, 4] = can_hook
        features[:, 5] = direction
        features[:, 6] = is_frozen
        features[:, 7] = (hook_state == 0).astype(np.float32)
        features[:, 8] = (hook_state == 1).astype(np.float32)
        features[:, 9] = (hook_state == 2).astype(np.float32)
        features[:, 10] = 0.0
        features[:, 11] = 0.0
        features[:, 12] = health / 10.0
        features[:, 13] = armor / 10.0
        features[:, 14] = jumped / 3.0

        # Checkpoint (4 features) - indices 15-18
        cp_floats = obs_arr[:, CHECKPOINT:CHECKPOINT+8].view(np.float32).reshape(n, 2)
        cp_ints = obs_arr[:, CHECKPOINT+8:CHECKPOINT+16].view(np.int32).reshape(n, 2)
        cp_x = cp_floats[:, 0]
        cp_y = cp_floats[:, 1]
        cp_idx = cp_ints[:, 0].astype(np.float32)
        total_cp = cp_ints[:, 1].astype(np.float32)

        features[:, 15] = np.clip((cp_x - pos_x) / 2000.0, -1.0, 1.0)
        features[:, 16] = np.clip((cp_y - pos_y) / 2000.0, -1.0, 1.0)
        features[:, 17] = cp_idx / np.maximum(1.0, total_cp)
        features[:, 18] = total_cp / 20.0

        # 7x7 center tile crop (49 features) - indices 19-67
        for ty in range(7):
            for tx in range(7):
                grid_y = TILE_CENTER_Y - 3 + ty
                grid_x = TILE_CENTER_X - 3 + tx
                tile_idx = TILES + grid_y * 43 + grid_x
                features[:, 19 + ty * 7 + tx] = obs_arr[:, tile_idx].astype(np.float32) / 15.0

    def wait_for_observations(self, timeout: float = 1.0) -> bool:
        """Wait for server to signal observations are ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            header = self._read_header()
            if header['shutdown']:
                return False
            if header['server_ready']:
                return True
            time.sleep(0.0001)  # 0.1ms
        return False

    def signal_actions_ready(self):
        """Signal to server that actions are ready."""
        self._write_header_field('server_ready', 0)
        self._write_header_field('python_ready', 1)

    def clear_actions_ready(self):
        """Clear the python_ready flag."""
        self._write_header_field('python_ready', 0)

    def signal_pause(self):
        """Signal to server to pause simulation (e.g., during PPO update)."""
        self._write_header_field('paused', 1)
        self.shm_mmap.flush()  # Ensure write is visible to server

    def signal_unpause(self):
        """Signal to server to resume simulation."""
        self._write_header_field('paused', 0)
        self.shm_mmap.flush()  # Ensure write is visible to server


class TeeWorldsServerEnv:
    """
    Server-side bot environment for Teeworlds RL training.

    This environment manages server-side bots directly, without client processes.
    Communication happens via a single shared memory region.

    Action space (hybrid):
        - discrete: MultiDiscrete([3, 2, 2])  # direction(-1,0,1), jump(0,1), hook(0,1)
        - continuous: Box([-1, -1], [1, 1])   # aim_angle, aim_magnitude (normalized)

    Observation space:
        - Box of shape (obs_dim,) containing player state, raycasts, local tiles, other players
    """

    def __init__(
        self,
        num_bots: int = 63,
        num_servers: int = 1,
        config: Optional[ServerEnvConfig] = None,
        auto_start: bool = True,
    ):
        self.num_bots_per_server = num_bots
        self.num_servers = num_servers
        self.num_bots = num_bots * num_servers  # Total bots across all servers
        self.num_envs = self.num_bots  # Alias for compatibility
        self.config = config or ServerEnvConfig()
        self.use_dict_obs = self.config.use_dict_obs
        self.use_embeddings = self.config.use_embeddings

        # Process management (list for multiple servers)
        self.server_procs: List[subprocess.Popen] = []

        # Shared memory clients (list for multiple servers)
        self.shm_clients: List[ServerShmClient] = []

        # Environment state
        self.obs_dim: Optional[int] = None
        self.episode_steps = np.zeros(self.num_bots, dtype=np.int32)
        self.episode_rewards = np.zeros(self.num_bots, dtype=np.float32)

        # Track state
        self._running = False
        self._closed = False
        self._shutting_down = False  # Fast exit flag for threads
        self._server_log = None  # Log file for server 0

        # Thread pool for parallel shm I/O (GIL is released during mmap operations)
        self._executor = ThreadPoolExecutor(max_workers=min(32, self.num_servers))

        # Timing instrumentation
        self._step_count = 0
        self._timing_write = 0.0
        self._timing_wait = 0.0
        self._timing_read = 0.0
        self._timing_total = 0.0

        # Register cleanup handlers
        global _active_server_env
        _active_server_env = self
        atexit.register(_cleanup_on_exit)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        if auto_start:
            self.start()

    @property
    def single_observation_space_shape(self) -> Tuple[int, ...]:
        """Shape of observation for a single environment."""
        if self.obs_dim is None:
            raise RuntimeError("Environment not started. Call start() first.")
        return (self.obs_dim,)

    @property
    def single_action_space(self) -> Dict[str, Any]:
        """Action space specification for a single environment."""
        return {
            'discrete': {
                'type': 'multi_discrete',
                'nvec': [3, 2, 2],  # direction, jump, hook
            },
            'continuous': {
                'type': 'box',
                'low': -1.0,
                'high': 1.0,
                'shape': (2,),  # aim_angle, aim_magnitude
            }
        }

    def start(self):
        """Start the DDNet server(s) with RL bots."""
        if self._running:
            return

        print(f"Starting Teeworlds server-side environment with {self.num_bots} bots ({self.num_servers} server(s), {self.num_bots_per_server} bots each)...")

        # Clean up stale shared memory for all servers
        self._cleanup_shared_memory()

        # Start ALL servers in parallel (no waiting between them)
        print(f"Launching {self.num_servers} servers in parallel...")
        for server_idx in range(self.num_servers):
            self._start_server_noblock(server_idx)

        # Single wait for all servers to initialize
        print(f"Waiting {self.config.startup_wait}s for servers to initialize...")
        time.sleep(self.config.startup_wait)

        # Connect to shared memory for all servers in parallel
        print("Connecting to shared memory...")
        def connect_shm(server_idx):
            shm_client = ServerShmClient(self.num_bots_per_server, server_idx)
            if not shm_client.connect(max_retries=30, retry_delay=0.2):
                return None, server_idx
            return shm_client, server_idx

        with ThreadPoolExecutor(max_workers=min(32, self.num_servers)) as executor:
            futures = [executor.submit(connect_shm, idx) for idx in range(self.num_servers)]
            results = [f.result() for f in futures]

        # Sort by server index and check for failures
        results.sort(key=lambda x: x[1])
        for shm_client, server_idx in results:
            if shm_client is None:
                raise RuntimeError(f"Failed to connect to server {server_idx} shared memory")
            self.shm_clients.append(shm_client)

        # Wait for first observations from all servers in parallel
        print("Waiting for initial observations...")
        def wait_obs(args):
            shm_client, idx = args
            success = shm_client.wait_for_observations(timeout=10.0)
            return success, idx

        with ThreadPoolExecutor(max_workers=min(32, self.num_servers)) as executor:
            futures = [executor.submit(wait_obs, (c, i)) for i, c in enumerate(self.shm_clients)]
            for f in futures:
                success, idx = f.result()
                if not success:
                    raise RuntimeError(f"Timeout waiting for initial observations from server {idx}")

        # Set observation dimension
        # Flat MLP mode: 68 features (15 player + 4 checkpoint + 49 tiles)
        # CNN mode: dict with tiles(1,31,43), player_state(15), checkpoint(4)
        self.obs_dim = 68 if not self.use_dict_obs else None
        self._running = True
        numba_status = "enabled (JIT)" if NUMBA_AVAILABLE else "disabled (install numba for 5-10x faster parsing)"
        if self.use_dict_obs:
            print(f"Environment ready. CNN observation mode, Numba: {numba_status}")
        else:
            print(f"Environment ready. Observation dim: {self.obs_dim}, Numba: {numba_status}")

    def _cleanup_shared_memory(self):
        """Clean up stale shared memory files and config files for all servers."""
        for server_idx in range(self.num_servers):
            shm_path = get_shm_path(server_idx)
            if os.path.exists(shm_path):
                try:
                    os.remove(shm_path)
                    print(f"  Removed stale shared memory: {shm_path}")
                except:
                    pass

        # Clear myServerconfig.cfg from all possible locations
        # DDNet-Server loads this after command-line args, so it would override them
        config_locations = [
            os.path.expanduser("~/.teeworlds/myServerconfig.cfg"),
        ]

        # Also check build/data directory
        server_dir = os.path.dirname(os.path.abspath(self.config.server_path))
        data_dir = os.path.join(server_dir, "data")
        config_locations.append(os.path.join(data_dir, "myServerconfig.cfg"))
        # Also clear any indexed configs from previous multi-server runs
        for i in range(10):
            config_locations.append(os.path.join(data_dir, f"myServerconfig_{i}.cfg"))

        for config_path in config_locations:
            if os.path.exists(config_path):
                try:
                    os.remove(config_path)
                    print(f"  Removed config: {config_path}")
                except Exception as e:
                    print(f"  Warning: Could not remove {config_path}: {e}")

    def _start_server(self, server_idx: int = 0):
        """Start a DDNet server with RL bots enabled."""
        server_dir = os.path.dirname(os.path.abspath(self.config.server_path))

        # Calculate port for this server
        server_port = self.config.server_port + server_idx

        # Server name with index
        if self.num_servers > 1:
            sv_name = f"{self.config.sv_name} #{server_idx}"
        else:
            sv_name = self.config.sv_name

        # Build server command with all settings as command-line arguments
        # DDNet-Server accepts console commands as positional arguments
        server_cmd = [
            self.config.server_path,
            f"sv_name \"{sv_name}\"",
            f"sv_port {server_port}",
            f"sv_map \"{self.config.map_name}\"",
            f"sv_tickrate {self.config.sv_tickrate}",
            f"sv_max_clients 128",
            f"sv_register 0",
            f"sv_rcon_password rltraining",
        ]

        if self.config.allow_spectators:
            server_cmd.extend([
                f"sv_spectator_slots 1",
                f"sv_max_clients_per_ip 4",
            ])

        # RL-specific settings (these must come after basic server setup)
        server_cmd.extend([
            f"sv_rl_frame_skip {self.config.frame_skip}",
            f"sv_rl_bots {self.num_bots_per_server} {server_idx}",
        ])

        # Server 0 logs to /tmp/rl_server.log for debugging (tail -f /tmp/rl_server.log)
        if server_idx == 0:
            self._server_log = open("/tmp/rl_server.log", "w")
            server_proc = subprocess.Popen(
                server_cmd,
                preexec_fn=os.setsid,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                cwd=server_dir,
            )
        else:
            server_proc = subprocess.Popen(
                server_cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=server_dir,
            )
        self.server_procs.append(server_proc)

        effective_rate = self.config.sv_tickrate // self.config.frame_skip
        print(f"  Server {server_idx}: PID={server_proc.pid}, port={server_port}")
        time.sleep(self.config.startup_wait)

    def _start_server_noblock(self, server_idx: int = 0):
        """Start a DDNet server without blocking (for parallel startup)."""
        server_dir = os.path.dirname(os.path.abspath(self.config.server_path))
        server_port = self.config.server_port + server_idx

        if self.num_servers > 1:
            sv_name = f"{self.config.sv_name} #{server_idx}"
        else:
            sv_name = self.config.sv_name

        server_cmd = [
            self.config.server_path,
            f"sv_name \"{sv_name}\"",
            f"sv_port {server_port}",
            f"sv_map \"{self.config.map_name}\"",
            f"sv_tickrate {self.config.sv_tickrate}",
            f"sv_max_clients 128",
            f"sv_register 0",
            f"sv_rcon_password rltraining",
        ]

        if self.config.allow_spectators:
            server_cmd.extend([
                f"sv_spectator_slots 1",
                f"sv_max_clients_per_ip 4",
            ])

        server_cmd.extend([
            f"sv_rl_frame_skip {self.config.frame_skip}",
            f"sv_rl_bots {self.num_bots_per_server} {server_idx}",
        ])

        # Server 0 logs to /tmp/rl_server.log for debugging (tail -f /tmp/rl_server.log)
        if server_idx == 0:
            self._server_log = open("/tmp/rl_server.log", "w")
            server_proc = subprocess.Popen(
                server_cmd,
                preexec_fn=os.setsid,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                cwd=server_dir,
            )
        else:
            server_proc = subprocess.Popen(
                server_cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=server_dir,
            )
        self.server_procs.append(server_proc)
        print(f"  Server {server_idx}: PID={server_proc.pid}, port={server_port}")

    def reset(self, seed: Optional[int] = None):
        """Reset all environments and return initial observations."""
        if not self._running:
            self.start()

        # Send reset signal for all bots on all servers
        for server_idx, shm_client in enumerate(self.shm_clients):
            reset_actions = {
                'discrete': np.zeros((self.num_bots_per_server, 3), dtype=np.int32),
                'continuous': np.zeros((self.num_bots_per_server, 2), dtype=np.float32),
                'reset': np.ones(self.num_bots_per_server, dtype=np.int32),
            }
            shm_client.write_actions_batch(reset_actions)
            shm_client.signal_actions_ready()

        # Wait for observations from all servers
        for server_idx, shm_client in enumerate(self.shm_clients):
            if not shm_client.wait_for_observations(timeout=2.0):
                print(f"Warning: Timeout waiting for reset observations from server {server_idx}")

        self.episode_steps[:] = 0
        self.episode_rewards[:] = 0.0

        # Read observations from all servers
        if self.use_embeddings:
            # Embedding mode: return raw tile IDs (int64) for nn.Embedding
            all_tiles = []
            all_player_state = []
            all_checkpoint = []
            for shm_client in self.shm_clients:
                obs_dict, _, _ = shm_client.read_observations_batch_embed()
                all_tiles.append(obs_dict['tiles'])
                all_player_state.append(obs_dict['player_state'])
                all_checkpoint.append(obs_dict['checkpoint'])
            return {
                'tiles': np.vstack(all_tiles),
                'player_state': np.vstack(all_player_state),
                'checkpoint': np.vstack(all_checkpoint),
            }, {}
        elif self.use_dict_obs:
            # CNN mode: return dict observations (normalized tiles)
            all_tiles = []
            all_player_state = []
            all_checkpoint = []
            for shm_client in self.shm_clients:
                obs_dict, _, _ = shm_client.read_observations_batch_cnn()
                all_tiles.append(obs_dict['tiles'])
                all_player_state.append(obs_dict['player_state'])
                all_checkpoint.append(obs_dict['checkpoint'])
            return {
                'tiles': np.vstack(all_tiles),
                'player_state': np.vstack(all_player_state),
                'checkpoint': np.vstack(all_checkpoint),
            }, {}
        else:
            # MLP mode: return flat observations
            all_observations = []
            for shm_client in self.shm_clients:
                observations, _, _ = shm_client.read_observations_batch_fast()
                all_observations.append(observations)
            return np.vstack(all_observations), {}  # (total_bots, 68)

    def _observations_to_dict(self, observations: List[Optional[StructuredObservation]]) -> Dict[str, np.ndarray]:
        """Convert list of observations to batched dict format for CNN."""
        tiles_list = []
        player_state_list = []
        checkpoint_list = []

        for obs in observations:
            if obs is not None:
                d = obs.to_dict_observation()
                tiles_list.append(d['tiles'])
                player_state_list.append(d['player_state'])
                checkpoint_list.append(d['checkpoint'])
            else:
                # Zero observations for missing
                tiles_list.append(np.zeros((1, TILE_GRID_HEIGHT, TILE_GRID_WIDTH), dtype=np.float32))
                player_state_list.append(np.zeros(15, dtype=np.float32))
                checkpoint_list.append(np.zeros(4, dtype=np.float32))

        return {
            'tiles': np.array(tiles_list, dtype=np.float32),
            'player_state': np.array(player_state_list, dtype=np.float32),
            'checkpoint': np.array(checkpoint_list, dtype=np.float32),
        }

    def _write_server_actions(self, server_idx: int, actions: Dict[str, np.ndarray]):
        """Write actions to one server (runs in thread)."""
        shm_client = self.shm_clients[server_idx]
        start_idx = server_idx * self.num_bots_per_server
        end_idx = start_idx + self.num_bots_per_server
        server_actions = {
            'discrete': actions['discrete'][start_idx:end_idx],
            'continuous': actions['continuous'][start_idx:end_idx],
        }
        if 'reset' in actions:
            server_actions['reset'] = actions['reset'][start_idx:end_idx]
        shm_client.write_actions_batch(server_actions)
        shm_client.signal_actions_ready()

    def _read_server_observations(self, server_idx: int):
        """Wait and read from one server (runs in thread)."""
        shm_client = self.shm_clients[server_idx]
        t_start = time.perf_counter()
        if not shm_client.wait_for_observations(timeout=self.config.action_timeout):
            return server_idx, None, None, None, True, 0.0, 0.0  # timeout_occurred=True
        t_wait = time.perf_counter()
        # Use embedding, CNN, or fast path depending on config
        if self.use_embeddings:
            observations, rewards, dones = shm_client.read_observations_batch_embed()
        elif self.use_dict_obs:
            observations, rewards, dones = shm_client.read_observations_batch_cnn()
        else:
            observations, rewards, dones = shm_client.read_observations_batch_fast()
        t_read = time.perf_counter()
        wait_ms = (t_wait - t_start) * 1000
        read_ms = (t_read - t_wait) * 1000
        return server_idx, observations, rewards, dones, False, wait_ms, read_ms

    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Take a step in all environments.

        Args:
            actions: Dict with keys:
                - 'discrete': np.ndarray of shape (num_envs, 3) with dtype int
                - 'continuous': np.ndarray of shape (num_envs, 2) with dtype float32

        Returns:
            observations: np.ndarray of shape (num_envs, obs_dim)
            rewards: np.ndarray of shape (num_envs,)
            dones: np.ndarray of shape (num_envs,) - True if episode ended
            truncated: np.ndarray of shape (num_envs,) - True if max steps reached
            infos: List of dicts with additional info per environment
        """
        t0 = time.perf_counter()

        # Parallel writes: submit all write tasks to thread pool
        t1 = time.perf_counter()
        write_futures = [
            self._executor.submit(self._write_server_actions, idx, actions)
            for idx in range(self.num_servers)
        ]
        for f in write_futures:
            f.result()  # Wait for all writes to complete
        t2 = time.perf_counter()

        # Parallel wait+read: submit all read tasks to thread pool
        read_futures = [
            self._executor.submit(self._read_server_observations, idx)
            for idx in range(self.num_servers)
        ]
        t3 = time.perf_counter()

        # Collect results in order
        results = [None] * self.num_servers
        wait_times = [0.0] * self.num_servers
        read_times = [0.0] * self.num_servers
        timeout_occurred = False
        for f in read_futures:
            idx, obs, rew, done, timed_out, wait_ms, read_ms = f.result()
            if timed_out:
                timeout_occurred = True
                print(f"Warning: Timeout waiting for observations from server {idx}")
            results[idx] = (obs, rew, done)
            wait_times[idx] = wait_ms
            read_times[idx] = read_ms
        t4 = time.perf_counter()

        # Timing instrumentation (log every 100 steps)
        self._step_count += 1
        self._timing_write += (t2 - t1)
        self._timing_wait += (t3 - t2)
        self._timing_read += (t4 - t3)
        self._timing_total += (t4 - t0)

        # Track per-server wait and read times
        if not hasattr(self, '_server_wait_times'):
            self._server_wait_times = [0.0] * self.num_servers
            self._server_read_times = [0.0] * self.num_servers
        for i, (wt, rt) in enumerate(zip(wait_times, read_times)):
            self._server_wait_times[i] += wt
            self._server_read_times[i] += rt

        if self._step_count % 10 == 0:
            if self.config.debug_timing:
                avg_write = self._timing_write / 10 * 1000  # ms
                avg_read = self._timing_read / 10 * 1000    # ms
                avg_total = self._timing_total / 10 * 1000  # ms

                # Show per-server wait and read times
                wait_avgs = [wt / 10 for wt in self._server_wait_times]
                read_avgs = [rt / 10 for rt in self._server_read_times]
                max_wait = max(wait_avgs)
                max_read = max(read_avgs)
                sum_read = sum(read_avgs)

                print(f"Step (ms): write={avg_write:.1f}, wait+read={avg_read:.1f}, total={avg_total:.1f} | "
                      f"server: max_wait={max_wait:.1f}, max_read={max_read:.1f}, sum_read={sum_read:.0f}")

            self._timing_write = 0.0
            self._timing_wait = 0.0
            self._timing_read = 0.0
            self._timing_total = 0.0
            self._server_wait_times = [0.0] * self.num_servers
            self._server_read_times = [0.0] * self.num_servers

        # Aggregate results from all servers (handles partial timeouts gracefully)
        all_rewards = []
        all_dones = []

        if self.use_embeddings:
            # Embedding mode: aggregate dict observations with int64 tiles
            all_tiles = []
            all_player_state = []
            all_checkpoint = []
            for obs, rew, done in results:
                if obs is not None:
                    all_tiles.append(obs['tiles'])
                    all_player_state.append(obs['player_state'])
                    all_checkpoint.append(obs['checkpoint'])
                    all_rewards.append(rew)
                    all_dones.append(done)
                else:
                    # Handle timeout case - append zeros
                    all_tiles.append(np.zeros((self.num_bots_per_server, TILE_GRID_HEIGHT, TILE_GRID_WIDTH), dtype=np.int64))
                    all_player_state.append(np.zeros((self.num_bots_per_server, 15), dtype=np.float32))
                    all_checkpoint.append(np.zeros((self.num_bots_per_server, 4), dtype=np.float32))
                    all_rewards.append(np.zeros(self.num_bots_per_server, dtype=np.float32))
                    all_dones.append(np.zeros(self.num_bots_per_server, dtype=bool))
            obs_out = {
                'tiles': np.vstack(all_tiles),
                'player_state': np.vstack(all_player_state),
                'checkpoint': np.vstack(all_checkpoint),
            }
        elif self.use_dict_obs:
            # CNN mode: aggregate dict observations (normalized float32 tiles)
            all_tiles = []
            all_player_state = []
            all_checkpoint = []
            for obs, rew, done in results:
                if obs is not None:
                    all_tiles.append(obs['tiles'])
                    all_player_state.append(obs['player_state'])
                    all_checkpoint.append(obs['checkpoint'])
                    all_rewards.append(rew)
                    all_dones.append(done)
                else:
                    # Handle timeout case - append zeros
                    all_tiles.append(np.zeros((self.num_bots_per_server, 1, TILE_GRID_HEIGHT, TILE_GRID_WIDTH), dtype=np.float32))
                    all_player_state.append(np.zeros((self.num_bots_per_server, 15), dtype=np.float32))
                    all_checkpoint.append(np.zeros((self.num_bots_per_server, 4), dtype=np.float32))
                    all_rewards.append(np.zeros(self.num_bots_per_server, dtype=np.float32))
                    all_dones.append(np.zeros(self.num_bots_per_server, dtype=bool))
            obs_out = {
                'tiles': np.vstack(all_tiles),
                'player_state': np.vstack(all_player_state),
                'checkpoint': np.vstack(all_checkpoint),
            }
        else:
            # MLP mode: aggregate flat observations
            all_observations = []
            for obs, rew, done in results:
                if obs is not None:
                    all_observations.append(obs)  # numpy array (num_bots, 68)
                    all_rewards.append(rew)
                    all_dones.append(done)
                else:
                    # Handle timeout case - append zeros
                    all_observations.append(np.zeros((self.num_bots_per_server, 68), dtype=np.float32))
                    all_rewards.append(np.zeros(self.num_bots_per_server, dtype=np.float32))
                    all_dones.append(np.zeros(self.num_bots_per_server, dtype=bool))
            obs_out = np.vstack(all_observations)  # (total_bots, 68)

        rewards = np.concatenate(all_rewards)
        dones = np.concatenate(all_dones)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i in range(self.num_envs):
            self.episode_steps[i] += 1
            self.episode_rewards[i] += rewards[i]

            info = {}

            # Check for truncation
            if self.episode_steps[i] >= self.config.max_episode_steps:
                truncated[i] = True
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_steps[i],
                }

            # Record episode stats on done
            if dones[i]:
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_steps[i],
                }

            infos.append(info)

        # Auto-reset environments that are done or truncated
        reset_mask = dones | truncated
        if np.any(reset_mask):
            self._reset_envs_fast(reset_mask, obs_out)

        return (
            obs_out,
            rewards,
            dones,
            truncated,
            infos
        )

    def _reset_envs_fast(self, mask: np.ndarray, observations):
        """Reset specific environments and update observations in place.

        Args:
            mask: Boolean mask of environments to reset
            observations: Either numpy array (MLP mode) or dict of arrays (CNN mode)
        """
        # Split mask across servers and send reset for masked environments
        for server_idx, shm_client in enumerate(self.shm_clients):
            start_idx = server_idx * self.num_bots_per_server
            end_idx = start_idx + self.num_bots_per_server
            server_mask = mask[start_idx:end_idx]

            if not np.any(server_mask):
                continue

            reset_actions = {
                'discrete': np.zeros((self.num_bots_per_server, 3), dtype=np.int32),
                'continuous': np.zeros((self.num_bots_per_server, 2), dtype=np.float32),
                'reset': server_mask.astype(np.int32),
            }
            shm_client.write_actions_batch(reset_actions)
            shm_client.signal_actions_ready()

        # Wait for servers to process reset and return fresh observations
        for server_idx, shm_client in enumerate(self.shm_clients):
            start_idx = server_idx * self.num_bots_per_server
            end_idx = start_idx + self.num_bots_per_server
            server_mask = mask[start_idx:end_idx]

            if not np.any(server_mask):
                continue

            if shm_client.wait_for_observations(timeout=0.1):
                # Read fresh observations for reset environments
                if self.use_embeddings:
                    fresh_obs, _, _ = shm_client.read_observations_batch_embed()
                    # Update only masked environments
                    for local_i in np.where(server_mask)[0]:
                        global_i = start_idx + local_i
                        observations['tiles'][global_i] = fresh_obs['tiles'][local_i]
                        observations['player_state'][global_i] = fresh_obs['player_state'][local_i]
                        observations['checkpoint'][global_i] = fresh_obs['checkpoint'][local_i]
                elif self.use_dict_obs:
                    fresh_obs, _, _ = shm_client.read_observations_batch_cnn()
                    # Update only masked environments
                    for local_i in np.where(server_mask)[0]:
                        global_i = start_idx + local_i
                        observations['tiles'][global_i] = fresh_obs['tiles'][local_i]
                        observations['player_state'][global_i] = fresh_obs['player_state'][local_i]
                        observations['checkpoint'][global_i] = fresh_obs['checkpoint'][local_i]
                else:
                    fresh_obs, _, _ = shm_client.read_observations_batch_fast()
                    # Update only masked environments
                    for local_i in np.where(server_mask)[0]:
                        global_i = start_idx + local_i
                        observations[global_i] = fresh_obs[local_i]

        # Reset counters for masked environments
        for i in np.where(mask)[0]:
            self.episode_steps[i] = 0
            self.episode_rewards[i] = 0.0

    def _reset_envs_observations(self, mask: np.ndarray, observations: List[Optional[StructuredObservation]]):
        """Reset specific environments and update observations in place (legacy version)."""
        # Split mask across servers and send reset for masked environments
        for server_idx, shm_client in enumerate(self.shm_clients):
            start_idx = server_idx * self.num_bots_per_server
            end_idx = start_idx + self.num_bots_per_server
            server_mask = mask[start_idx:end_idx]

            if not np.any(server_mask):
                continue

            reset_actions = {
                'discrete': np.zeros((self.num_bots_per_server, 3), dtype=np.int32),
                'continuous': np.zeros((self.num_bots_per_server, 2), dtype=np.float32),
                'reset': server_mask.astype(np.int32),
            }
            shm_client.write_actions_batch(reset_actions)
            shm_client.signal_actions_ready()

        # Wait for servers to process reset and return fresh observations
        for server_idx, shm_client in enumerate(self.shm_clients):
            start_idx = server_idx * self.num_bots_per_server
            end_idx = start_idx + self.num_bots_per_server
            server_mask = mask[start_idx:end_idx]

            if not np.any(server_mask):
                continue

            if shm_client.wait_for_observations(timeout=0.1):
                # Update observations for reset environments with fresh post-reset state
                for local_i in np.where(server_mask)[0]:
                    global_i = start_idx + local_i
                    obs, reward, done = shm_client.read_observation(local_i)
                    if obs:
                        observations[global_i] = obs

        # Reset counters for masked environments
        for i in np.where(mask)[0]:
            self.episode_steps[i] = 0
            self.episode_rewards[i] = 0.0

    def pause(self):
        """
        Signal all servers to pause simulation.

        Call this before PPO updates to prevent bots from taking actions
        with stale policy during the update. The servers will freeze physics
        until unpause() is called.

        Note: Requires sv_rl_allow_pause 1 on the server (default).
        """
        for shm_client in self.shm_clients:
            shm_client.signal_pause()

    def unpause(self):
        """
        Signal all servers to resume simulation.

        Call this after PPO updates to resume normal operation.
        """
        for shm_client in self.shm_clients:
            shm_client.signal_unpause()

    def get_detailed_timings(self) -> dict:
        """
        Get detailed timing breakdown for profiling.

        Returns timing stats accumulated since last reset_timing_stats() call,
        or since environment creation.

        Returns:
            dict with timing breakdown in milliseconds:
                - write_actions_ms: Average time to write actions
                - wait_for_obs_ms: Average time waiting for server
                - read_obs_ms: Average time to read observations
                - total_step_ms: Average total step time
                - step_count: Number of steps since last reset
                - per_server_wait_ms: List of avg wait times per server
                - per_server_read_ms: List of avg read times per server
        """
        if not hasattr(self, '_step_count') or self._step_count == 0:
            return {
                "write_actions_ms": 0.0,
                "wait_for_obs_ms": 0.0,
                "read_obs_ms": 0.0,
                "total_step_ms": 0.0,
                "step_count": 0,
                "per_server_wait_ms": [],
                "per_server_read_ms": [],
            }

        n = self._step_count
        result = {
            "write_actions_ms": self._timing_write / n * 1000,
            "wait_for_obs_ms": self._timing_wait / n * 1000,
            "read_obs_ms": self._timing_read / n * 1000,
            "total_step_ms": self._timing_total / n * 1000,
            "step_count": n,
        }

        # Add per-server timings if available
        if hasattr(self, '_server_wait_times') and self._server_wait_times:
            result["per_server_wait_ms"] = [wt / n for wt in self._server_wait_times]
        else:
            result["per_server_wait_ms"] = []

        if hasattr(self, '_server_read_times') and self._server_read_times:
            result["per_server_read_ms"] = [rt / n for rt in self._server_read_times]
        else:
            result["per_server_read_ms"] = []

        return result

    def reset_timing_stats(self):
        """Reset timing statistics for fresh profiling."""
        self._step_count = 0
        self._timing_write = 0.0
        self._timing_wait = 0.0
        self._timing_read = 0.0
        self._timing_total = 0.0
        if hasattr(self, '_server_wait_times'):
            self._server_wait_times = [0.0] * len(self._server_wait_times)
        if hasattr(self, '_server_read_times'):
            self._server_read_times = [0.0] * len(self._server_read_times)

    def close(self):
        """Clean up all resources."""
        if self._closed:
            return
        self._closed = True
        self._shutting_down = True  # Signal threads to exit fast
        self._running = False

        print("Closing server environment...")

        # Kill all servers first (fast, parallel) - this makes shm operations fail quickly
        for server_proc in self.server_procs:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            except:
                pass
        self.server_procs = []

        # Close server log file if open
        if hasattr(self, '_server_log') and self._server_log:
            try:
                self._server_log.close()
            except:
                pass

        # Shutdown thread pool without waiting
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False, cancel_futures=True)

        # Close shared memory (don't bother with clean disconnect - servers are dead)
        for shm_client in self.shm_clients:
            try:
                if shm_client.shm_mmap:
                    shm_client.shm_mmap.close()
                if shm_client.shm_file:
                    shm_client.shm_file.close()
            except:
                pass
        self.shm_clients = []

        print("Server environment closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FrameStackWrapper:
    """
    Stack frames for temporal context with separate control for tiles and player state.

    This wrapper maintains deques of recent observations and stacks them
    so the agent can see temporal patterns (e.g., velocity trajectory
    for double-jump timing).

    With frame_skip=3 and n_stack=8: 8 frames × 3 ticks × 20ms = 480ms coverage.

    Tiles and player_state can be stacked independently:
    - Tile stacking is expensive (8x VRAM) but useful for tracking other players
    - Player state stacking is cheap and sufficient for timing/velocity patterns
    """

    def __init__(self, env: TeeWorldsServerEnv, n_tile_stack: int = 1, n_player_state_stack: int = 8):
        """
        Args:
            env: The underlying TeeWorldsServerEnv (must use dict observations)
            n_tile_stack: Number of tile frames to stack (default: 1, no stacking)
            n_player_state_stack: Number of player state frames to stack (default: 8)
        """
        from collections import deque

        self.env = env
        self.n_tile_stack = n_tile_stack
        self.n_player_state_stack = n_player_state_stack

        # Separate deques for tiles and player state (can have different sizes)
        self.tile_frames = deque(maxlen=n_tile_stack)
        self.player_frames = deque(maxlen=n_player_state_stack)

        # Forward attributes from wrapped env
        self.num_bots = env.num_bots
        self.num_envs = env.num_envs
        self.config = env.config

    def reset(self, seed=None):
        """Reset and initialize frame stacks with copies of first observation."""
        obs, info = self.env.reset(seed=seed)

        # Initialize tile stack
        self.tile_frames.clear()
        for _ in range(self.n_tile_stack):
            self.tile_frames.append(obs['tiles'].copy())

        # Initialize player state stack
        self.player_frames.clear()
        for _ in range(self.n_player_state_stack):
            self.player_frames.append(obs['player_state'].copy())

        # Keep latest checkpoint (not stacked)
        self._latest_checkpoint = obs['checkpoint'].copy()

        return self._get_stacked_obs(), info

    def step(self, action):
        """Step environment and update frame stacks."""
        obs, reward, done, truncated, info = self.env.step(action)

        # Add new frames to stacks
        self.tile_frames.append(obs['tiles'].copy())
        self.player_frames.append(obs['player_state'].copy())
        self._latest_checkpoint = obs['checkpoint'].copy()

        # Handle episode resets (per-env): reset frame history for done envs
        reset_mask = done | truncated
        if np.any(reset_mask):
            for i in np.where(reset_mask)[0]:
                # Reset this env's tile history
                for frame in self.tile_frames:
                    frame[i] = obs['tiles'][i]
                # Reset this env's player state history
                for frame in self.player_frames:
                    frame[i] = obs['player_state'][i]

        return self._get_stacked_obs(), reward, done, truncated, info

    def _get_stacked_obs(self):
        """Stack observations along channel/feature dimension."""
        # Stack tiles: (n_envs, 1, H, W) * n_tile_stack -> (n_envs, n_tile_stack, H, W)
        stacked_tiles = np.concatenate(
            [f for f in self.tile_frames], axis=1
        )

        # Stack player_state: (n_envs, 15) * n_player_stack -> (n_envs, 15*n_player_stack)
        stacked_player = np.concatenate(
            [f for f in self.player_frames], axis=1
        )

        # Checkpoint stays single (doesn't change rapidly)
        return {
            'tiles': stacked_tiles,
            'player_state': stacked_player,
            'checkpoint': self._latest_checkpoint,
        }

    def pause(self):
        """Forward pause to underlying env."""
        self.env.pause()

    def unpause(self):
        """Forward unpause to underlying env."""
        self.env.unpause()

    def close(self):
        """Forward close to underlying env."""
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def make_server_env(num_bots: int = 63, num_servers: int = 1, **kwargs) -> TeeWorldsServerEnv:
    """Create a server-side Teeworlds environment."""
    config = ServerEnvConfig(**kwargs)
    return TeeWorldsServerEnv(num_bots=num_bots, num_servers=num_servers, config=config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-bots", type=int, default=8, help="Bots per server")
    parser.add_argument("--num-servers", type=int, default=1, help="Number of servers")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    total_bots = args.num_bots * args.num_servers
    print(f"Testing TeeWorldsServerEnv with {total_bots} bots ({args.num_servers} server(s), {args.num_bots} bots each)...")

    with TeeWorldsServerEnv(num_bots=args.num_bots, num_servers=args.num_servers) as env:
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")

        total_rewards = np.zeros(total_bots)

        for step in range(args.steps):
            # Random actions
            actions = {
                'discrete': np.random.randint([0, 0, 0], [3, 2, 2], size=(total_bots, 3)),
                'continuous': np.random.uniform(-1, 1, size=(total_bots, 2)).astype(np.float32),
            }

            obs, rewards, dones, truncated, infos = env.step(actions)
            total_rewards += rewards

            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}: mean_reward={rewards.mean():.4f}, dones={dones.sum()}")

        print(f"\nTotal rewards: {total_rewards}")
        print(f"Mean total reward: {total_rewards.mean():.4f}")
