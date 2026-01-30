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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import observation parsing from existing client code
from teeworlds_env import (
    StructuredObservation, PlayerObs, ProjectileObs,
    STRUCTURED_OBS_SIZE, MAX_OTHER_PLAYERS, MAX_PROJECTILES, MAX_PICKUPS,
    TILE_GRID_SIZE, TILE_GRID_WIDTH, TILE_GRID_HEIGHT, NUM_RAYCASTS
)

# Constants matching C++ rl_protocol.h
MAX_RL_BOTS = 64
SHM_PATH = "/dev/shm/ddnet_rl_server"

# Struct sizes (must match C++)
HEADER_SIZE = 32        # SServerRLHeader
ACTION_SIZE = 32        # SAction
OBS_SIZE = 2280         # SStructuredObservation
BOT_DATA_SIZE = 2336    # SBotData (32 + 2280 + 16 + 8)

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
    max_episode_steps: int = 2000
    startup_wait: float = 3.0
    action_timeout: float = 0.1  # Timeout waiting for Python to provide actions

    def __post_init__(self):
        if self.server_path is None:
            self.server_path = _get_default_server_path()


class ServerShmClient:
    """
    Shared memory client for server-side bot communication.

    Communicates with the DDNet server's bot manager via a single shared memory region.
    """

    def __init__(self, num_bots: int):
        self.num_bots = num_bots
        self.shm_path = SHM_PATH
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
        """Write actions for all bots at once."""
        discrete_actions = actions['discrete']  # (num_bots, 3)
        continuous_actions = actions['continuous']  # (num_bots, 2)
        resets = actions.get('reset', np.zeros(self.num_bots, dtype=np.int32))

        for i in range(self.num_bots):
            direction = int(discrete_actions[i, 0]) - 1  # 0,1,2 -> -1,0,1
            jump = int(discrete_actions[i, 1])
            hook = int(discrete_actions[i, 2])

            # Continuous aim
            aim_angle = float(continuous_actions[i, 0]) * np.pi
            aim_magnitude = (float(continuous_actions[i, 1]) + 1.0) / 2.0
            aim_x = int(np.cos(aim_angle) * aim_magnitude * 200)
            aim_y = int(np.sin(aim_angle) * aim_magnitude * 200)

            self.write_action(i, direction, jump, hook, 0, aim_x, aim_y, 0, int(resets[i]))

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
        config: Optional[ServerEnvConfig] = None,
        auto_start: bool = True,
    ):
        self.num_envs = num_bots
        self.num_bots = num_bots
        self.config = config or ServerEnvConfig()

        # Process management
        self.server_proc: Optional[subprocess.Popen] = None

        # Shared memory client
        self.shm_client: Optional[ServerShmClient] = None

        # Environment state
        self.obs_dim: Optional[int] = None
        self.episode_steps = np.zeros(num_bots, dtype=np.int32)
        self.episode_rewards = np.zeros(num_bots, dtype=np.float32)

        # Track state
        self._running = False
        self._closed = False

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
        """Start the DDNet server with RL bots."""
        if self._running:
            return

        print(f"Starting Teeworlds server-side environment with {self.num_bots} bots...")

        # Clean up stale shared memory
        self._cleanup_shared_memory()

        # Start server with RL bots
        self._start_server()

        # Connect to shared memory
        self.shm_client = ServerShmClient(self.num_bots)
        if not self.shm_client.connect():
            raise RuntimeError("Failed to connect to server shared memory")

        # Wait for first observations
        print("Waiting for initial observations...")
        if not self.shm_client.wait_for_observations(timeout=10.0):
            raise RuntimeError("Timeout waiting for initial observations")

        # Get observation dimension
        observations, _, _ = self.shm_client.read_observations_batch()
        valid_obs = [o for o in observations if o is not None]
        if valid_obs:
            self.obs_dim = len(valid_obs[0].to_flat_array())
        else:
            raise RuntimeError("No valid observations received")

        self._running = True
        print(f"Environment ready. Observation dim: {self.obs_dim}")

    def _cleanup_shared_memory(self):
        """Clean up stale shared memory files."""
        if os.path.exists(SHM_PATH):
            try:
                os.remove(SHM_PATH)
                print(f"  Removed stale shared memory: {SHM_PATH}")
            except:
                pass

    def _start_server(self):
        """Start the DDNet server with RL bots enabled."""
        # Create a temporary config file for RL settings
        # DDNet server reads myServerconfig.cfg from the data directory
        server_dir = os.path.dirname(os.path.abspath(self.config.server_path))
        data_dir = os.path.join(server_dir, "data")
        config_path = os.path.join(data_dir, "myServerconfig.cfg")

        # Write RL configuration
        os.makedirs(data_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(f"# RL Training Config (auto-generated)\n")
            f.write(f"sv_port {self.config.server_port}\n")
            f.write(f"sv_tickrate {self.config.sv_tickrate}\n")
            f.write(f"sv_max_clients 64\n")
            f.write(f"sv_rl_bots {self.num_bots}\n")
            f.write(f"sv_register 0\n")  # Don't register with master
            f.write(f"sv_rcon_password p")
            
        server_cmd = [self.config.server_path]

        self.server_proc = subprocess.Popen(
            server_cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=server_dir,  # Run from server directory so it finds data/
        )
        print(f"Server started (PID: {self.server_proc.pid}, tickrate: {self.config.sv_tickrate}, bots: {self.num_bots})")
        time.sleep(self.config.startup_wait)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset all environments and return initial observations."""
        if not self._running:
            self.start()

        # Send reset signal for all bots
        reset_actions = {
            'discrete': np.zeros((self.num_bots, 3), dtype=np.int32),
            'continuous': np.zeros((self.num_bots, 2), dtype=np.float32),
            'reset': np.ones(self.num_bots, dtype=np.int32),
        }
        self.shm_client.write_actions_batch(reset_actions)
        self.shm_client.signal_actions_ready()

        # Wait for observations
        if not self.shm_client.wait_for_observations(timeout=2.0):
            print("Warning: Timeout waiting for reset observations")

        # Read observations
        observations, _, _ = self.shm_client.read_observations_batch()

        obs_array = []
        for i, obs in enumerate(observations):
            if obs is not None:
                obs_array.append(obs.to_flat_array())
            else:
                obs_array.append(np.zeros(self.obs_dim, dtype=np.float32))

        self.episode_steps[:] = 0
        self.episode_rewards[:] = 0.0

        return np.array(obs_array, dtype=np.float32), {}

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
        # Write actions
        self.shm_client.write_actions_batch(actions)
        self.shm_client.signal_actions_ready()

        # Wait for observations
        if not self.shm_client.wait_for_observations(timeout=self.config.action_timeout):
            # Timeout - return last state with zero reward
            print("Warning: Timeout waiting for observations")
            return (
                np.zeros((self.num_envs, self.obs_dim), dtype=np.float32),
                np.zeros(self.num_envs, dtype=np.float32),
                np.zeros(self.num_envs, dtype=bool),
                np.zeros(self.num_envs, dtype=bool),
                [{'error': True} for _ in range(self.num_envs)]
            )

        # Read observations
        observations, rewards, dones = self.shm_client.read_observations_batch()

        obs_array = []
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i, obs in enumerate(observations):
            if obs is not None:
                obs_array.append(obs.to_flat_array())
            else:
                obs_array.append(np.zeros(self.obs_dim, dtype=np.float32))

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
            self._reset_envs(reset_mask, obs_array)

        return (
            np.array(obs_array, dtype=np.float32),
            rewards,
            dones,
            truncated,
            infos
        )

    def _reset_envs(self, mask: np.ndarray, observations: List[np.ndarray]):
        """Reset specific environments and update observations in place."""
        # Send reset for masked environments
        reset_actions = {
            'discrete': np.zeros((self.num_bots, 3), dtype=np.int32),
            'continuous': np.zeros((self.num_bots, 2), dtype=np.float32),
            'reset': mask.astype(np.int32),
        }
        self.shm_client.write_actions_batch(reset_actions)
        self.shm_client.signal_actions_ready()

        # Wait for server to process reset and return fresh observations
        if self.shm_client.wait_for_observations(timeout=0.1):
            # Update observations for reset environments with fresh post-reset state
            for i in np.where(mask)[0]:
                obs, reward, done = self.shm_client.read_observation(i)
                if obs:
                    observations[i] = obs.to_flat_array()

        # Reset counters for masked environments
        for i in np.where(mask)[0]:
            self.episode_steps[i] = 0
            self.episode_rewards[i] = 0.0

    def close(self):
        """Clean up all resources."""
        if self._closed:
            return
        self._closed = True

        print("Closing server environment...")

        # Disconnect from shared memory
        if self.shm_client:
            self.shm_client.disconnect()
            self.shm_client = None

        # Terminate server
        if self.server_proc:
            try:
                print("  Terminating server...")
                os.killpg(os.getpgid(self.server_proc.pid), signal.SIGKILL)
            except:
                pass
            self.server_proc = None

        self._running = False
        print("Server environment closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def make_server_env(num_bots: int = 63, **kwargs) -> TeeWorldsServerEnv:
    """Create a server-side Teeworlds environment."""
    config = ServerEnvConfig(**kwargs)
    return TeeWorldsServerEnv(num_bots=num_bots, config=config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-bots", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    print(f"Testing TeeWorldsServerEnv with {args.num_bots} bots...")

    with TeeWorldsServerEnv(num_bots=args.num_bots) as env:
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")

        total_rewards = np.zeros(args.num_bots)

        for step in range(args.steps):
            # Random actions
            actions = {
                'discrete': np.random.randint([0, 0, 0], [3, 2, 2], size=(args.num_bots, 3)),
                'continuous': np.random.uniform(-1, 1, size=(args.num_bots, 2)).astype(np.float32),
            }

            obs, rewards, dones, truncated, infos = env.step(actions)
            total_rewards += rewards

            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}: mean_reward={rewards.mean():.4f}, dones={dones.sum()}")

        print(f"\nTotal rewards: {total_rewards}")
        print(f"Mean total reward: {total_rewards.mean():.4f}")
