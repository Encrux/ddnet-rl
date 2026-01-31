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

# Import observation parsing from existing client code
from teeworlds_env import (
    StructuredObservation, PlayerObs, ProjectileObs,
    STRUCTURED_OBS_SIZE, MAX_OTHER_PLAYERS, MAX_PROJECTILES, MAX_PICKUPS,
    TILE_GRID_SIZE, TILE_GRID_WIDTH, TILE_GRID_HEIGHT, NUM_RAYCASTS
)

# Re-define locally for vectorized parsing
_TILE_GRID_SIZE = TILE_GRID_SIZE  # 1333

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
OBS_SIZE = 3020         # SStructuredObservation (43x31 grid + checkpoint info)
BOT_DATA_SIZE = 3072    # SBotData (32 + 3020 + 16 + 4)

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
    map_name: str = "Tutorial"   # Map to load (use novice maps for training)
    sv_name: str = "RL Training" # Server name
    allow_spectators: bool = True  # Allow connecting to watch

    def __post_init__(self):
        if self.server_path is None:
            self.server_path = _get_default_server_path()


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

        # Calculate total shared memory size
        self.total_size = HEADER_SIZE + (BOT_DATA_SIZE * MAX_RL_BOTS)

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
        """Write actions for all bots at once (vectorized, single memcpy)."""
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

        # Build action array: SAction is 8 int32s = 32 bytes
        # Layout: direction, jump, hook, fire, aim_x, aim_y, weapon, reset
        action_array = np.zeros((n, 8), dtype=np.int32)
        action_array[:, 0] = direction
        action_array[:, 1] = jump
        action_array[:, 2] = hook
        action_array[:, 3] = 0  # fire
        action_array[:, 4] = aim_x
        action_array[:, 5] = aim_y
        action_array[:, 6] = 0  # weapon
        action_array[:, 7] = resets.astype(np.int32)

        # Write all actions with strided writes (actions are at start of each SBotData)
        # Each SBotData is 3072 bytes, action is first 32 bytes
        raw_bytes = action_array.tobytes()  # n * 32 bytes contiguous

        # Write each action to the correct offset (unfortunately can't do single write due to stride)
        # But we can minimize syscalls by writing all actions first, then seeking once
        for i in range(n):
            offset = HEADER_SIZE + (i * BOT_DATA_SIZE)
            self.shm_mmap.seek(offset)
            self.shm_mmap.write(raw_bytes[i*32:(i+1)*32])

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
        """Read all observations as flat arrays directly (optimized with views).

        Returns:
            observations: np.ndarray of shape (num_bots, 141)
            rewards: np.ndarray of shape (num_bots,)
            dones: np.ndarray of shape (num_bots,)
        """
        n = self.num_bots

        # Read ALL bot data at once (single memcpy) - keep as bytes object
        start_offset = HEADER_SIZE
        total_size = n * BOT_DATA_SIZE
        self.shm_mmap.seek(start_offset)
        raw = self.shm_mmap.read(total_size)

        # Create 1D numpy view of raw bytes (no copy)
        raw_arr = np.frombuffer(raw, dtype=np.uint8)

        # Pre-allocate output
        features = np.zeros((n, 141), dtype=np.float32)

        # Offsets within SBotData
        OBS_START = ACTION_SIZE  # 32
        LOCAL_PLAYER = OBS_START + 8
        OTHER_PLAYERS = LOCAL_PLAYER + 36
        TILES = OBS_START + 1452
        RAYCASTS = OBS_START + 2785
        EXTRA_STATE = OBS_START + 2977
        CHECKPOINT = OBS_START + 2981
        REWARD_OFF = OBS_START + OBS_SIZE  # 3052
        DONE_OFF = REWARD_OFF + 4

        # Build index arrays for strided access (computed once, reused)
        bot_offsets = np.arange(n) * BOT_DATA_SIZE

        # Extract rewards (4 bytes each, stride=BOT_DATA_SIZE)
        reward_indices = bot_offsets + REWARD_OFF
        rewards = np.array([struct.unpack_from('<f', raw, int(idx))[0] for idx in reward_indices], dtype=np.float32)

        # Extract dones
        done_indices = bot_offsets + DONE_OFF
        dones = np.array([struct.unpack_from('<i', raw, int(idx))[0] != 0 for idx in done_indices], dtype=bool)

        # === Vectorized float extraction using stride tricks ===
        # Local player floats: pos_x, pos_y, vel_x, vel_y at LOCAL_PLAYER
        player_float_offsets = bot_offsets + LOCAL_PLAYER
        pos_x = np.array([struct.unpack_from('<f', raw, int(o))[0] for o in player_float_offsets])
        pos_y = np.array([struct.unpack_from('<f', raw, int(o)+4)[0] for o in player_float_offsets])
        vel_x = np.array([struct.unpack_from('<f', raw, int(o)+8)[0] for o in player_float_offsets])
        vel_y = np.array([struct.unpack_from('<f', raw, int(o)+12)[0] for o in player_float_offsets])

        # Single-byte fields using raw array indexing (very fast)
        direction = raw_arr[bot_offsets + LOCAL_PLAYER + 16].astype(np.int8).astype(np.float32)
        jumped = raw_arr[bot_offsets + LOCAL_PLAYER + 17].astype(np.float32)
        hook_state = raw_arr[bot_offsets + LOCAL_PLAYER + 18].astype(np.float32)
        is_frozen = raw_arr[bot_offsets + LOCAL_PLAYER + 30].astype(np.float32)
        health = raw_arr[bot_offsets + LOCAL_PLAYER + 32].astype(np.float32)
        armor = raw_arr[bot_offsets + LOCAL_PLAYER + 33].astype(np.float32)
        is_grounded = raw_arr[bot_offsets + EXTRA_STATE].astype(np.float32)
        jumps_remaining = raw_arr[bot_offsets + EXTRA_STATE + 1].astype(np.float32)
        can_hook = raw_arr[bot_offsets + EXTRA_STATE + 2].astype(np.float32)

        # Build features array
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
        features[:, 12] = health / 10.0
        features[:, 13] = armor / 10.0
        features[:, 14] = jumped / 3.0

        # Checkpoint floats
        cp_offsets = bot_offsets + CHECKPOINT
        cp_x = np.array([struct.unpack_from('<f', raw, int(o))[0] for o in cp_offsets])
        cp_y = np.array([struct.unpack_from('<f', raw, int(o)+4)[0] for o in cp_offsets])
        cp_idx = np.array([struct.unpack_from('<i', raw, int(o)+8)[0] for o in cp_offsets], dtype=np.float32)
        total_cp = np.array([struct.unpack_from('<i', raw, int(o)+12)[0] for o in cp_offsets], dtype=np.float32)

        features[:, 15] = np.clip((cp_x - pos_x) / 2000.0, -1.0, 1.0)
        features[:, 16] = np.clip((cp_y - pos_y) / 2000.0, -1.0, 1.0)
        features[:, 17] = cp_idx / np.maximum(1.0, total_cp)
        features[:, 18] = total_cp / 20.0

        # Raycasts - 48 floats per bot
        for i in range(n):
            base = int(bot_offsets[i]) + RAYCASTS
            features[i, 19:67] = struct.unpack_from('<48f', raw, base)

        # Tiles 7x7 center - extract using byte indexing
        # Center of 43x31 grid is (21, 15), we want 7x7 around it: rows 12-18, cols 18-24
        for ty in range(7):
            for tx in range(7):
                tile_row = 12 + ty
                tile_col = 18 + tx
                tile_local_offset = tile_row * 43 + tile_col
                tile_indices = bot_offsets + TILES + tile_local_offset
                features[:, 67 + ty * 7 + tx] = raw_arr[tile_indices].astype(np.float32) / 15.0

        # Other players (5 players x 5 features) - simplified, just extract if active
        for j in range(5):
            op_local = OTHER_PLAYERS + j * 36
            op_active = raw_arr[bot_offsets + op_local + 29]
            active_mask = op_active > 0

            if np.any(active_mask):
                for i in np.where(active_mask)[0]:
                    base = int(bot_offsets[i]) + op_local
                    op_px, op_py, op_vx, op_vy = struct.unpack_from('<ffff', raw, base)
                    op_hook = raw[base + 18]
                    feat_base = 116 + j * 5
                    features[i, feat_base] = (op_px - pos_x[i]) / 1000.0
                    features[i, feat_base + 1] = (op_py - pos_y[i]) / 1000.0
                    features[i, feat_base + 2] = op_vx / 50.0
                    features[i, feat_base + 3] = op_vy / 50.0
                    features[i, feat_base + 4] = float(op_hook) / 2.0

        # Sanitize NaN/inf
        np.nan_to_num(features, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        return features, rewards, dones

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

        # Start servers with RL bots
        for server_idx in range(self.num_servers):
            self._start_server(server_idx)

        # Connect to shared memory for each server
        for server_idx in range(self.num_servers):
            shm_client = ServerShmClient(self.num_bots_per_server, server_idx)
            if not shm_client.connect():
                raise RuntimeError(f"Failed to connect to server {server_idx} shared memory")
            self.shm_clients.append(shm_client)

        # Wait for first observations from all servers
        print("Waiting for initial observations...")
        for server_idx, shm_client in enumerate(self.shm_clients):
            if not shm_client.wait_for_observations(timeout=10.0):
                raise RuntimeError(f"Timeout waiting for initial observations from server {server_idx}")

        # Set observation dimension (fast path always returns 141)
        self.obs_dim = 141
        self._running = True
        print(f"Environment ready. Observation dim: {self.obs_dim}")

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

        server_proc = subprocess.Popen(
            server_cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=server_dir,  # Run from server directory so it finds data/
        )
        self.server_procs.append(server_proc)

        effective_rate = self.config.sv_tickrate // self.config.frame_skip
        print(f"Server {server_idx} started (PID: {server_proc.pid}, port: {server_port}, map: {self.config.map_name}, tickrate: {self.config.sv_tickrate}, decision_rate: {effective_rate}Hz, bots: {self.num_bots_per_server})")
        if self.config.allow_spectators:
            print(f"  Connect with DDNet client to 127.0.0.1:{server_port} to watch")
        time.sleep(self.config.startup_wait)

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

        # Read observations from all servers using fast path
        all_observations = []
        for shm_client in self.shm_clients:
            observations, _, _ = shm_client.read_observations_batch_fast()
            all_observations.append(observations)

        self.episode_steps[:] = 0
        self.episode_rewards[:] = 0.0

        obs_out = np.vstack(all_observations)  # (total_bots, 141)
        return obs_out, {}

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
        # Use fast path - returns flat arrays directly
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
            avg_write = self._timing_write / 10 * 1000  # ms
            avg_read = self._timing_read / 10 * 1000    # ms
            avg_total = self._timing_total / 10 * 1000  # ms
            budget = 20.0  # 50Hz = 20ms per tick

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
        # Fast path returns numpy arrays directly: obs is (num_bots, 141), rew is (num_bots,), done is (num_bots,)
        all_observations = []
        all_rewards = []
        all_dones = []
        for obs, rew, done in results:
            if obs is not None:
                all_observations.append(obs)  # numpy array (num_bots, 141)
                all_rewards.append(rew)
                all_dones.append(done)
            else:
                # Handle timeout case - append zeros
                all_observations.append(np.zeros((self.num_bots_per_server, 141), dtype=np.float32))
                all_rewards.append(np.zeros(self.num_bots_per_server, dtype=np.float32))
                all_dones.append(np.zeros(self.num_bots_per_server, dtype=bool))

        # Stack all observations into single array
        obs_out = np.vstack(all_observations)  # (total_bots, 141)
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

    def _reset_envs_fast(self, mask: np.ndarray, observations: np.ndarray):
        """Reset specific environments and update observations in place (fast numpy version)."""
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

    def close(self):
        """Clean up all resources."""
        if self._closed:
            return
        self._closed = True

        print("Closing server environment...")

        # Shutdown thread pool
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

        # Disconnect from all shared memory clients
        for server_idx, shm_client in enumerate(self.shm_clients):
            try:
                shm_client.disconnect()
            except:
                pass
        self.shm_clients = []

        # Terminate all servers
        for server_idx, server_proc in enumerate(self.server_procs):
            try:
                print(f"  Terminating server {server_idx}...")
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            except:
                pass
        self.server_procs = []

        self._running = False
        print("Server environment closed")

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
