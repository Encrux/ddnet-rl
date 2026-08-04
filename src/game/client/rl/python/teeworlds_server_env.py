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
from path_progress import PathProgress

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
    raw = np.zeros(n * BOT_DATA_SIZE, dtype=np.uint8)
    features = np.zeros((n, FLAT_OBS_DIM), dtype=np.float32)
    player_state = np.zeros((n, 15), dtype=np.float32)
    checkpoint = np.zeros((n, NAV_DIM), dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)

    # Trigger compilation (first call compiles, subsequent calls are fast)
    _parse_flat_obs_numba(raw, n, features, rewards, dones)
    _parse_cnn_obs_numba(raw, n, player_state, checkpoint, rewards, dones)

    # Contiguous layout warmup
    obs_arr = np.zeros((n, OBS_SIZE), dtype=np.uint8)
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
ACTION_SIZE = 36        # SAction (9 x int32: direction, jump, hook, fire, aim_x, aim_y, weapon, reset, restore_slot)
OBS_SIZE = 1452         # SStructuredObservation (tiles + player + nav)
BOT_DATA_SIZE = 1504    # SBotData (legacy interleaved layout: 36 + 1452 + 16)

# Navigation block: normalised progress + geometric path lookahead.
# Replaces the old 4-feature checkpoint block, which encoded mapper-authored
# time checkpoints (see rl_protocol.h for why those are unusable across maps).
# Aim discretisation. Sector count trades multimodality against head width; the
# residual means precision does not depend on it.
AIM_SECTORS = 16
AIM_RADIUS = 200.0  # arbitrary; only the direction of (aim_x, aim_y) is used

NUM_LOOKAHEAD = 8
NAV_DIM = 1 + 2 * NUM_LOOKAHEAD  # progress + (x, y) per lookahead point
FLAT_OBS_DIM = 15 + NAV_DIM + 49  # MLP mode: player + nav + 7x7 tile crop
# Per-index normalisation: lookahead k sits ~64 * 2^k units ahead, so dividing by
# its own expected magnitude keeps every feature O(1) instead of spanning 62..8000.
LOOKAHEAD_SCALE = np.array([64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0],
                           dtype=np.float32)

# Contiguous layout offsets (matches SContiguousServerRLShm in rl_protocol.h)
# Layout: [Header][Action0..N][Obs0..N][Reward0..N][Done0..N][Truncated0..N][BotClientId0..N]
CONTIGUOUS_ACTIONS_OFFSET = HEADER_SIZE  # 32
CONTIGUOUS_OBS_OFFSET = CONTIGUOUS_ACTIONS_OFFSET + MAX_RL_BOTS * ACTION_SIZE  # 32 + 128*36 = 4640
CONTIGUOUS_REWARDS_OFFSET = CONTIGUOUS_OBS_OFFSET + MAX_RL_BOTS * OBS_SIZE  # 4640 + 128*1400 = 183840
CONTIGUOUS_DONES_OFFSET = CONTIGUOUS_REWARDS_OFFSET + MAX_RL_BOTS * 4  # 183840 + 512 = 184352
CONTIGUOUS_TRUNCATED_OFFSET = CONTIGUOUS_DONES_OFFSET + MAX_RL_BOTS * 4  # 184352 + 512 = 184864
CONTIGUOUS_BOTIDS_OFFSET = CONTIGUOUS_TRUNCATED_OFFSET + MAX_RL_BOTS * 4  # 184864 + 512 = 185376
CONTIGUOUS_SHM_SIZE = CONTIGUOUS_BOTIDS_OFFSET + MAX_RL_BOTS * 4  # 185376 + 512 = 185888

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


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))


def _get_default_server_path() -> str:
    """Get the default server path relative to this script."""
    return os.path.join(_REPO_ROOT, "build/DDNet-Server")


class CheckpointSpawnManager:
    """Tracks agent progress and samples spawn checkpoints for adaptive restart.

    On reset, agents spawn at checkpoints sampled from the distribution of
    best recent episodes, with a configurable bias toward the map start.
    """

    def __init__(self, begin_bias: float = 0.2, buffer_size: int = 1000, top_fraction: float = 0.3):
        self.begin_bias = begin_bias
        self.top_fraction = top_fraction
        self.buffer: list = []
        self.buffer_size = buffer_size
        self._max_from_start = -1  # Frontier: highest time CP index reached from start (-1 = none)

    def record_episode(self, highest_checkpoint: int, from_start: bool):
        """Record the highest checkpoint reached in a completed episode.

        Args:
            highest_checkpoint: Highest checkpoint index reached (earned, not spawn offset).
            from_start: True if the agent spawned at the map beginning.
        """
        if highest_checkpoint < 0:
            highest_checkpoint = 0
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(highest_checkpoint)
        if from_start:
            self._max_from_start = max(self._max_from_start, highest_checkpoint)

    def sample_spawn_checkpoints(self, n: int) -> np.ndarray:
        """Sample n spawn checkpoint indices. -1 = map start."""
        result = np.full(n, -1, dtype=np.int32)
        if len(self.buffer) < 50 or self._max_from_start < 1:
            return result  # Not enough data, always spawn at start

        sorted_cps = sorted(self.buffer, reverse=True)
        top_k = max(1, int(len(sorted_cps) * self.top_fraction))
        top_cps = np.array(sorted_cps[:top_k])
        mean_cp = float(np.mean(top_cps))
        std_cp = max(1.0, float(np.std(top_cps)))

        # Cap spawns to the frontier proven from start
        max_spawn = self._max_from_start - 1  # Always leave at least 1 CP to reach

        for i in range(n):
            if np.random.random() < self.begin_bias:
                result[i] = -1  # spawn at start
            else:
                sampled = int(np.random.normal(mean_cp, std_cp))
                result[i] = max(0, min(sampled, max_spawn))
        return result

    def get_stats(self) -> dict:
        """Return stats for logging."""
        if not self.buffer:
            return {'spawn_mean_cp': 0.0, 'spawn_max_cp': 0, 'spawn_frontier': 0, 'spawn_buffer_size': 0}
        sorted_cps = sorted(self.buffer, reverse=True)
        top_k = max(1, int(len(sorted_cps) * self.top_fraction))
        top_cps = sorted_cps[:top_k]
        return {
            'spawn_mean_cp': float(np.mean(top_cps)),  # Mean of top episodes (what we sample from)
            'spawn_max_cp': max(self.buffer),  # Highest CP any agent reached
            'spawn_frontier': self._max_from_start,  # Highest CP reached from start (caps spawns)
            'spawn_buffer_size': len(self.buffer),
        }


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
    checkpoint_respawn: bool = False  # Enable checkpoint respawning
    begin_bias: float = 0.2          # Probability of spawning at map start
    top_fraction: float = 0.3        # Fraction of best episodes for spawn sampling
    # Virtual finish line: end the episode once normalised path progress reaches
    # this. 1.0 = the real end of the map. Lower values slice the map short for
    # fast iteration without editing the map itself (PLAN.md Phase 0.1).
    finish_progress: float = 1.0
    waypoints_path: Optional[str] = None  # Override waypoint file autodiscovery

    def __post_init__(self):
        if self.server_path is None:
            self.server_path = _get_default_server_path()
        # use_embeddings implies use_dict_obs
        if self.use_embeddings:
            self.use_dict_obs = True


# Numba-accelerated CNN observation parsing
@jit(nopython=True, cache=True, parallel=True)
def _parse_flat_obs_numba(raw_arr: np.ndarray, n: int,
                           features: np.ndarray,  # (n, FLAT_OBS_DIM) - player(15) + nav(17) + tiles(49)
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
    NAV = EXTRA_STATE + 4  # After extra state
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
        nav_f = raw_arr[base + NAV:base + NAV + 4 + 8 * NUM_LOOKAHEAD].view(np.float32)

        features[i, 15] = nav_f[0]  # normalised path progress
        for k in range(NUM_LOOKAHEAD):
            features[i, 16 + 2 * k] = min(4.0, max(-4.0, nav_f[1 + k] / LOOKAHEAD_SCALE[k]))
            features[i, 17 + 2 * k] = min(4.0, max(-4.0, nav_f[1 + NUM_LOOKAHEAD + k] / LOOKAHEAD_SCALE[k]))

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
    NAV = EXTRA_STATE + 4  # After extra state
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

        # Navigation: progress + geometric path lookahead
        nav_f = raw_arr[base + NAV:base + NAV + 4 + 8 * NUM_LOOKAHEAD].view(np.float32)

        checkpoint[i, 0] = nav_f[0]  # normalised path progress
        for k in range(NUM_LOOKAHEAD):
            checkpoint[i, 1 + 2 * k] = min(4.0, max(-4.0, nav_f[1 + k] / LOOKAHEAD_SCALE[k]))
            checkpoint[i, 2 + 2 * k] = min(4.0, max(-4.0, nav_f[1 + NUM_LOOKAHEAD + k] / LOOKAHEAD_SCALE[k]))


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
    NAV = EXTRA_STATE + 4  # 1381

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
        nav_f = obs_arr[i, NAV:NAV + 4 + 8 * NUM_LOOKAHEAD].view(np.float32)

        features[i, 15] = nav_f[0]  # normalised path progress
        for k in range(NUM_LOOKAHEAD):
            features[i, 16 + 2 * k] = min(4.0, max(-4.0, nav_f[1 + k] / LOOKAHEAD_SCALE[k]))
            features[i, 17 + 2 * k] = min(4.0, max(-4.0, nav_f[1 + NUM_LOOKAHEAD + k] / LOOKAHEAD_SCALE[k]))

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
        # SAction: direction, jump, hook, fire, aim_x, aim_y, weapon, reset, spawn_checkpoint
        self._action_buffer = np.zeros((MAX_RL_BOTS, 9), dtype=np.int32)

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

        # SAction struct: direction, jump, hook, fire, aim_x, aim_y, weapon, reset, spawn_checkpoint
        action_data = struct.pack('<iiiiiiiii',
            direction, jump, hook, fire, aim_x, aim_y, weapon, reset, -1)
        self.shm_mmap.write(action_data)

    def write_actions_batch(self, actions: Dict[str, np.ndarray]):
        """Write actions for all bots at once (single memcpy with contiguous layout)."""
        discrete_actions = actions['discrete']  # (num_bots, 4) - dir, jump, hook, aim sector
        continuous_actions = actions['continuous']  # (num_bots, 1) - aim residual
        resets = actions.get('reset', np.zeros(self.num_bots, dtype=np.int32))

        # Vectorized action computation
        n = self.num_bots
        direction = discrete_actions[:, 0].astype(np.int32) - 1  # 0,1,2 -> -1,0,1
        jump = discrete_actions[:, 1].astype(np.int32)
        hook = discrete_actions[:, 2].astype(np.int32)

        # Factored aim: Categorical sector + continuous residual within it.
        # The sector gives multimodality, the residual gives precision.
        sector = discrete_actions[:, 3]
        sector_center = (sector.astype(np.float32) + 0.5) * (2.0 * np.pi / AIM_SECTORS) - np.pi
        residual = np.tanh(continuous_actions[:, 0]) * (np.pi / AIM_SECTORS)
        aim_angle = sector_center + residual
        # Magnitude is fixed: DDNet uses only the direction of m_TargetX/Y.
        aim_x = (np.cos(aim_angle) * AIM_RADIUS).astype(np.int32)
        aim_y = (np.sin(aim_angle) * AIM_RADIUS).astype(np.int32)

        spawn_checkpoints = actions.get('spawn_checkpoint', np.full(n, -1, dtype=np.int32))

        # Build action array using pre-allocated buffer
        # SAction layout: direction, jump, hook, fire, aim_x, aim_y, weapon, reset, spawn_checkpoint
        self._action_buffer[:n, 0] = direction
        self._action_buffer[:n, 1] = jump
        self._action_buffer[:n, 2] = hook
        self._action_buffer[:n, 3] = 0  # fire
        self._action_buffer[:n, 4] = aim_x
        self._action_buffer[:n, 5] = aim_y
        self._action_buffer[:n, 6] = 0  # weapon
        self._action_buffer[:n, 7] = resets.astype(np.int32)
        self._action_buffer[:n, 8] = spawn_checkpoints.astype(np.int32)

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
            observations: np.ndarray of shape (num_bots, FLAT_OBS_DIM) - 15 player + 17 nav + 49 tiles
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

        # Pre-allocate outputs - FLAT_OBS_DIM features (15 + 17 + 49)
        features = np.zeros((n, FLAT_OBS_DIM), dtype=np.float32)

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
        checkpoint = np.empty((n, NAV_DIM), dtype=np.float32)

        # Offsets within SStructuredObservation (1400 bytes)
        # Layout: header(8) + LocalPlayer(36) + TileGrid(1333) + ExtraState(4) + Checkpoint(16) + Reserved(3)
        LOCAL_PLAYER = 8
        TILES = LOCAL_PLAYER + 36  # 44
        EXTRA_STATE = TILES + 1333  # 1377
        NAV = EXTRA_STATE + 4  # 1381

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

        # ===== NAV (progress + path lookahead) =====
        nav_raw = obs_arr[:, NAV:NAV + 4 + 8 * NUM_LOOKAHEAD].copy().view(np.float32).reshape(n, NAV_DIM)

        checkpoint[:, 0] = nav_raw[:, 0]  # normalised path progress
        checkpoint[:, 1::2] = np.clip(nav_raw[:, 1:1 + NUM_LOOKAHEAD] / LOOKAHEAD_SCALE, -4.0, 4.0)
        checkpoint[:, 2::2] = np.clip(nav_raw[:, 1 + NUM_LOOKAHEAD:] / LOOKAHEAD_SCALE, -4.0, 4.0)

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
                'checkpoint': np.ndarray(shape=(n, NAV_DIM), dtype=np.float32)
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
        checkpoint = np.empty((n, NAV_DIM), dtype=np.float32)

        # Offsets within SStructuredObservation (1400 bytes)
        LOCAL_PLAYER = 8
        TILES = LOCAL_PLAYER + 36  # 44
        EXTRA_STATE = TILES + 1333  # 1377
        NAV = EXTRA_STATE + 4  # 1381

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

        # ===== NAV (progress + path lookahead) =====
        nav_raw = obs_arr[:, NAV:NAV + 4 + 8 * NUM_LOOKAHEAD].copy().view(np.float32).reshape(n, NAV_DIM)

        checkpoint[:, 0] = nav_raw[:, 0]  # normalised path progress
        checkpoint[:, 1::2] = np.clip(nav_raw[:, 1:1 + NUM_LOOKAHEAD] / LOOKAHEAD_SCALE, -4.0, 4.0)
        checkpoint[:, 2::2] = np.clip(nav_raw[:, 1 + NUM_LOOKAHEAD:] / LOOKAHEAD_SCALE, -4.0, 4.0)

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
        NAV = EXTRA_STATE + 4  # After extra state
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

        # Navigation: progress + geometric path lookahead
        nav_raw = np.array(
            [struct.unpack_from('<' + 'f' * NAV_DIM, raw, int(o) + NAV) for o in bot_offsets],
            dtype=np.float32).reshape(len(bot_offsets), NAV_DIM)

        checkpoint[:, 0] = nav_raw[:, 0]  # normalised path progress
        checkpoint[:, 1::2] = np.clip(nav_raw[:, 1:1 + NUM_LOOKAHEAD] / LOOKAHEAD_SCALE, -4.0, 4.0)
        checkpoint[:, 2::2] = np.clip(nav_raw[:, 1 + NUM_LOOKAHEAD:] / LOOKAHEAD_SCALE, -4.0, 4.0)

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
        NAV = EXTRA_STATE + 4
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
        nav_raw = np.array(
            [struct.unpack_from('<' + 'f' * NAV_DIM, raw, int(o) + NAV) for o in bot_offsets],
            dtype=np.float32).reshape(len(bot_offsets), NAV_DIM)

        features[:, 15] = nav_raw[:, 0]  # normalised path progress
        features[:, 16:15 + NAV_DIM:2] = np.clip(nav_raw[:, 1:1 + NUM_LOOKAHEAD] / LOOKAHEAD_SCALE, -4.0, 4.0)
        features[:, 17:15 + NAV_DIM:2] = np.clip(nav_raw[:, 1 + NUM_LOOKAHEAD:] / LOOKAHEAD_SCALE, -4.0, 4.0)

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
        NAV = EXTRA_STATE + 4  # 1381

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
        nav_raw = obs_arr[:, NAV:NAV + 4 + 8 * NUM_LOOKAHEAD].copy().view(np.float32).reshape(n, NAV_DIM)

        features[:, 15] = nav_raw[:, 0]  # normalised path progress
        features[:, 16:15 + NAV_DIM:2] = np.clip(nav_raw[:, 1:1 + NUM_LOOKAHEAD] / LOOKAHEAD_SCALE, -4.0, 4.0)
        features[:, 17:15 + NAV_DIM:2] = np.clip(nav_raw[:, 1 + NUM_LOOKAHEAD:] / LOOKAHEAD_SCALE, -4.0, 4.0)

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
        self._agent_spawn_cp = np.full(self.num_bots, -1, dtype=np.int32)  # Track spawn checkpoint per agent

        # Path progress: the one map-independent progress coordinate (roadmap.md §1.7).
        # Drives the virtual finish line and every episode metric.
        self.path_progress = None
        wp = PathProgress.resolve(self.config.map_name, _REPO_ROOT, self.config.waypoints_path)
        if wp is not None:
            self.path_progress = PathProgress(
                wp, num_envs=self.num_bots, frame_skip=self.config.frame_skip)
            print(f"Path progress: {self.path_progress}")
            amb = self.path_progress.self_crossing_report()
            if amb["num_ambiguous"]:
                print(f"  note: path self-approaches at {amb['num_ambiguous']} point(s), "
                      f"e.g. {amb['examples'][:2]} — temporal matching is load-bearing here")
            if self.config.finish_progress < 1.0:
                fx, fy = self.path_progress.progress_to_world(self.config.finish_progress)
                print(f"  virtual finish at progress {self.config.finish_progress:.3f} "
                      f"-> world ({fx:.0f}, {fy:.0f})")
        elif self.config.finish_progress < 1.0:
            raise RuntimeError(
                f"finish_progress={self.config.finish_progress} requires a waypoint file "
                f"for map '{self.config.map_name}' (looked in waypoints/ and ghosts/)")

        # Was this episode started at the map beginning? Only these count toward
        # the headline from-start finish rate.
        self._episode_from_start = np.ones(self.num_bots, dtype=bool)
        self._last_progress = np.zeros(self.num_bots, dtype=np.float32)

        # Checkpoint respawn manager
        self.spawn_manager = None
        if self.config.checkpoint_respawn:
            self.spawn_manager = CheckpointSpawnManager(
                begin_bias=self.config.begin_bias,
                top_fraction=self.config.top_fraction,
            )

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
                'nvec': [3, 2, 2, AIM_SECTORS],  # direction, jump, hook, aim sector
            },
            'continuous': {
                'type': 'box',
                'low': -1.0,
                'high': 1.0,
                'shape': (1,),  # aim residual within the chosen sector
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
        # Flat MLP mode: FLAT_OBS_DIM features (15 player + 17 nav + 49 tiles)
        # CNN mode: dict with tiles(1,31,43), player_state(15), checkpoint(4)
        self.obs_dim = FLAT_OBS_DIM if not self.use_dict_obs else None
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
            # ORDER MATTERS: sv_rl_frame_skip is a no-op unless the bot manager
            # already exists (ConRLFrameSkip bails with "RL bots not active"), so
            # sv_rl_bots MUST come first. Reversed, frame skip silently stays 1.
            f"sv_rl_bots {self.num_bots_per_server} {server_idx}",
            f"sv_rl_frame_skip {self.config.frame_skip}",
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
            # ORDER MATTERS: sv_rl_frame_skip is a no-op unless the bot manager
            # already exists (ConRLFrameSkip bails with "RL bots not active"), so
            # sv_rl_bots MUST come first. Reversed, frame skip silently stays 1.
            f"sv_rl_bots {self.num_bots_per_server} {server_idx}",
            f"sv_rl_frame_skip {self.config.frame_skip}",
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
                'discrete': np.zeros((self.num_bots_per_server, 4), dtype=np.int32),
                'continuous': np.zeros((self.num_bots_per_server, 1), dtype=np.float32),
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
        self._agent_spawn_cp[:] = -1  # All agents start at map beginning on full reset
        self._episode_from_start[:] = True
        self._last_progress[:] = 0.0
        if self.path_progress is not None:
            self.path_progress.reset()
            self._warn_spawn_path_gap()

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
            return np.vstack(all_observations), {}  # (total_bots, FLAT_OBS_DIM)

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
                checkpoint_list.append(np.zeros(NAV_DIM, dtype=np.float32))

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
                    all_checkpoint.append(np.zeros((self.num_bots_per_server, NAV_DIM), dtype=np.float32))
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
                    all_checkpoint.append(np.zeros((self.num_bots_per_server, NAV_DIM), dtype=np.float32))
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
                    all_observations.append(np.zeros((self.num_bots_per_server, FLAT_OBS_DIM), dtype=np.float32))
                    all_rewards.append(np.zeros(self.num_bots_per_server, dtype=np.float32))
                    all_dones.append(np.zeros(self.num_bots_per_server, dtype=bool))
            obs_out = np.vstack(all_observations)  # (total_bots, 68)

        rewards = np.concatenate(all_rewards)
        dones = np.concatenate(all_dones)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = []

        # Path progress drives both the virtual finish line and every metric.
        positions = None
        finished = np.zeros(self.num_envs, dtype=bool)
        if self.path_progress is not None:
            positions = self._read_positions()
            self._last_progress = self.path_progress.update(positions)
            finished = self._last_progress >= self.config.finish_progress
            # A virtual finish ends the episode just like a server-side done.
            dones = dones | finished

        self.episode_steps += 1
        self.episode_rewards += rewards
        truncated[:] = self.episode_steps >= self.config.max_episode_steps

        for i in range(self.num_envs):
            info = {}
            if dones[i] or truncated[i]:
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_steps[i],
                    'progress': float(self._last_progress[i]),
                    'finished': bool(finished[i]),
                    'from_start': bool(self._episode_from_start[i]),
                }
                if positions is not None:
                    info['episode']['pos'] = positions[i].copy()
            infos.append(info)

        # Auto-reset environments that are done or truncated
        reset_mask = dones | truncated
        if np.any(reset_mask):
            spawn_cps = None
            if self.spawn_manager is not None:
                # Extract checkpoint indices from raw shm before reset overwrites them
                # cp_indices = m_NumCheckpointsReached, which includes spawn offset
                cp_indices = self._read_checkpoint_indices()
                for i in np.where(reset_mask)[0]:
                    # cp_indices[i] = m_HighestTimeCp (0-based index, -1 if none).
                    # For spawn at CP N: m_HighestTimeCp starts at N.
                    spawn_cp = self._agent_spawn_cp[i]
                    highest_cp = int(cp_indices[i])
                    if spawn_cp >= 0:
                        earned = max(0, highest_cp - spawn_cp)
                    else:
                        # Spawned at start: highest_cp is 0-based index, +1 for count
                        earned = highest_cp + 1 if highest_cp >= 0 else 0
                    # Only record if agent actually progressed beyond spawn
                    if earned > 0:
                        from_start = spawn_cp < 0
                        self.spawn_manager.record_episode(highest_cp, from_start=from_start)
                # Sample spawn checkpoints for resetting agents
                n_reset = int(reset_mask.sum())
                sampled = self.spawn_manager.sample_spawn_checkpoints(n_reset)
                spawn_cps = np.full(self.num_envs, -1, dtype=np.int32)
                spawn_cps[reset_mask] = sampled
                # Diagnostic: log spawn distribution every 1000 steps
                if self._step_count % 1000 == 0:
                    nonstart = sampled[sampled >= 0]
                    stats = self.spawn_manager.get_stats()
                    print(f"  [SPAWN] frontier={stats['spawn_frontier']} max_cp={stats['spawn_max_cp']:.0f} "
                          f"n_reset={n_reset} at_start={int((sampled < 0).sum())} "
                          f"nonstart_range=[{nonstart.min() if len(nonstart) else -1}, {nonstart.max() if len(nonstart) else -1}]")
                # Track what each agent spawns at for next episode
                reset_indices = np.where(reset_mask)[0]
                for j, i in enumerate(reset_indices):
                    self._agent_spawn_cp[i] = sampled[j]

            # New episodes: forget the tracked waypoint index, and record whether
            # they start at the map beginning (the from-start finish rate depends
            # on this being right).
            if self.path_progress is not None:
                self.path_progress.reset(reset_mask)
                self._last_progress[reset_mask] = 0.0
            if spawn_cps is not None:
                self._episode_from_start[reset_mask] = spawn_cps[reset_mask] < 0
            else:
                self._episode_from_start[reset_mask] = True

            self._reset_envs_fast(reset_mask, obs_out, spawn_checkpoints=spawn_cps)

        return (
            obs_out,
            rewards,
            dones,
            truncated,
            infos
        )

    def _reset_envs_fast(self, mask: np.ndarray, observations, spawn_checkpoints=None):
        """Reset specific environments and update observations in place.

        Args:
            mask: Boolean mask of environments to reset
            observations: Either numpy array (MLP mode) or dict of arrays (CNN mode)
            spawn_checkpoints: Optional int32 array of checkpoint indices per env (-1 = start)
        """
        # Split mask across servers and send reset for masked environments
        for server_idx, shm_client in enumerate(self.shm_clients):
            start_idx = server_idx * self.num_bots_per_server
            end_idx = start_idx + self.num_bots_per_server
            server_mask = mask[start_idx:end_idx]

            if not np.any(server_mask):
                continue

            reset_actions = {
                'discrete': np.zeros((self.num_bots_per_server, 4), dtype=np.int32),
                'continuous': np.zeros((self.num_bots_per_server, 1), dtype=np.float32),
                'reset': server_mask.astype(np.int32),
            }
            if spawn_checkpoints is not None:
                reset_actions['spawn_checkpoint'] = spawn_checkpoints[start_idx:end_idx]
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
                'discrete': np.zeros((self.num_bots_per_server, 4), dtype=np.int32),
                'continuous': np.zeros((self.num_bots_per_server, 1), dtype=np.float32),
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

    def _warn_spawn_path_gap(self, threshold: float = 128.0) -> None:
        """Warn once if the tee spawns far from the start of the reference path.

        Ghost recording begins at the START tile, not at the spawn tile, so a
        ghost-derived path can begin some distance ahead of where episodes
        actually start. Progress is flat over that gap, which means the shaping
        term gives no gradient for the opening seconds of every episode.
        """
        if getattr(self, '_spawn_gap_checked', False):
            return
        self._spawn_gap_checked = True
        try:
            spawn = self._read_positions()[0]
        except Exception:
            return
        wp0 = self.path_progress.xy[0]
        gap = float(np.linalg.norm(spawn - wp0))
        if gap > threshold:
            print(f"  WARNING: spawn ({spawn[0]:.0f}, {spawn[1]:.0f}) is {gap:.0f} units from "
                  f"path start ({wp0[0]:.0f}, {wp0[1]:.0f}) — progress is flat over that gap, "
                  f"so shaping gives no gradient there. Consider prepending the spawn to the "
                  f"waypoint file.")

    def _read_positions(self) -> np.ndarray:
        """Read every bot's world position from raw shm. Returns (num_envs, 2) float32.

        Position is deliberately absent from the policy observation (it doesn't
        generalise), but it's exactly what the metrics need.
        """
        # m_PosX/m_PosY are the first two fields of m_LocalPlayer, which follows
        # the 8-byte game-state header in SStructuredObservation.
        POS_OFFSET = 8
        result = np.zeros((self.num_envs, 2), dtype=np.float32)
        for server_idx, shm_client in enumerate(self.shm_clients):
            start_idx = server_idx * self.num_bots_per_server
            n = self.num_bots_per_server
            shm_client.shm_mmap.seek(CONTIGUOUS_OBS_OFFSET)
            obs_raw = shm_client.shm_mmap.read(n * OBS_SIZE)
            obs_arr = np.frombuffer(obs_raw, dtype=np.uint8).reshape(n, OBS_SIZE)
            xy = obs_arr[:, POS_OFFSET:POS_OFFSET + 8].copy().view(np.float32).reshape(n, 2)
            result[start_idx:start_idx + n] = xy
        return result

    def _read_checkpoint_indices(self) -> np.ndarray:
        """Read checkpoint indices from raw shm observations for all bots."""
        # m_CheckpointIndex is at byte offset 1389 within SStructuredObservation
        # Layout: header(8) + player(36) + tiles(1333) + extra(4) + NextCpX(4) + NextCpY(4) = 1389
        CP_INDEX_OFFSET = 8 + 36 + 1333 + 4 + 4 + 4  # 1389
        result = np.zeros(self.num_envs, dtype=np.int32)
        for server_idx, shm_client in enumerate(self.shm_clients):
            start_idx = server_idx * self.num_bots_per_server
            n = self.num_bots_per_server
            shm_client.shm_mmap.seek(CONTIGUOUS_OBS_OFFSET)
            obs_raw = shm_client.shm_mmap.read(n * OBS_SIZE)
            obs_arr = np.frombuffer(obs_raw, dtype=np.uint8).reshape(n, OBS_SIZE)
            cp_vals = obs_arr[:, CP_INDEX_OFFSET:CP_INDEX_OFFSET + 4].copy().view(np.int32).flatten()
            result[start_idx:start_idx + n] = cp_vals
        return result

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

    Uses a pre-allocated ring buffer for player state stacking to avoid
    per-step allocations. Tiles are passed through by reference when not
    stacked (n_tile_stack=1), since the underlying env returns fresh arrays.

    With frame_skip=3 and n_stack=8: 8 frames × 3 ticks × 20ms = 480ms coverage.
    """

    def __init__(self, env: TeeWorldsServerEnv, n_tile_stack: int = 1, n_player_state_stack: int = 8):
        self.env = env
        self.n_tile_stack = n_tile_stack
        self.n_player_state_stack = n_player_state_stack
        self._stack_tiles = n_tile_stack > 1

        # Forward attributes from wrapped env
        self.num_bots = env.num_bots
        self.num_envs = env.num_envs
        self.config = env.config

        n_envs = env.num_envs
        player_feat = 15

        if self._stack_tiles:
            from collections import deque
            self.tile_frames = deque(maxlen=n_tile_stack)

        # Pre-allocated ring buffer for player state: (n_envs, feat * n_stack)
        # Layout: [oldest_frame_feats ... newest_frame_feats]
        self._player_ring = np.zeros((n_envs, player_feat * n_player_state_stack), dtype=np.float32)
        self._player_feat = player_feat
        # Write pointer: index of the slot to write next (0..n_player_state_stack-1)
        self._player_ptr = 0

    def reset(self, seed=None):
        """Reset and initialize frame stacks with copies of first observation."""
        obs, info = self.env.reset(seed=seed)
        n_ps = self.n_player_state_stack
        feat = self._player_feat

        # Initialize tiles
        if self._stack_tiles:
            self.tile_frames.clear()
            for _ in range(self.n_tile_stack):
                self.tile_frames.append(obs['tiles'].copy())
        else:
            self._latest_tiles = obs['tiles']

        # Initialize player state ring buffer: tile all slots with the same observation
        ps = obs['player_state']  # (n_envs, 15)
        for i in range(n_ps):
            self._player_ring[:, i * feat:(i + 1) * feat] = ps
        self._player_ptr = 0

        # Checkpoint: just keep a reference (underlying env returns fresh arrays)
        self._latest_checkpoint = obs['checkpoint']

        return self._get_stacked_obs(), info

    def step(self, action):
        """Step environment and update frame stacks."""
        obs, reward, done, truncated, info = self.env.step(action)
        n_ps = self.n_player_state_stack
        feat = self._player_feat

        # Update tiles
        if self._stack_tiles:
            self.tile_frames.append(obs['tiles'].copy())
        else:
            self._latest_tiles = obs['tiles']

        # Write new player state into the ring buffer at current pointer
        ptr = self._player_ptr
        np.copyto(self._player_ring[:, ptr * feat:(ptr + 1) * feat], obs['player_state'])
        self._player_ptr = (ptr + 1) % n_ps

        # Checkpoint: just keep a reference
        self._latest_checkpoint = obs['checkpoint']

        # Handle episode resets: fill all ring buffer slots for done envs
        reset_mask = done | truncated
        if np.any(reset_mask):
            if self._stack_tiles:
                for frame in self.tile_frames:
                    frame[reset_mask] = obs['tiles'][reset_mask]
            # Vectorized reset: tile the current player_state across all slots
            ps_reset = obs['player_state'][reset_mask]  # (n_reset, 15)
            self._player_ring[reset_mask] = np.tile(ps_reset, n_ps)

        return self._get_stacked_obs(), reward, done, truncated, info

    def _get_stacked_obs(self):
        """Build stacked observation dict."""
        # Tiles: either stacked from deque or pass-through reference
        if self._stack_tiles:
            stacked_tiles = np.concatenate(list(self.tile_frames), axis=1)
        else:
            stacked_tiles = self._latest_tiles

        # Player state: reorder ring buffer so oldest frame is first
        ptr = self._player_ptr
        feat = self._player_feat
        n_ps = self.n_player_state_stack
        if ptr == 0:
            # Already in order: slot 0 is oldest, slot n-1 is newest
            stacked_player = self._player_ring
        else:
            # ptr points to the oldest slot, so roll: [ptr..n-1, 0..ptr-1]
            stacked_player = np.concatenate([
                self._player_ring[:, ptr * feat:],
                self._player_ring[:, :ptr * feat],
            ], axis=1)

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
