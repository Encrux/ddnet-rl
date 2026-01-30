"""
Teeworlds Vectorized Environment for RL Training

A Gymnasium-compatible vectorized environment that manages multiple DDNet clients
communicating via shared memory. Supports hybrid action spaces (multi-discrete + continuous).

Usage:
    env = TeeWorldsVecEnv(num_envs=8)
    obs = env.reset()
    for _ in range(1000):
        actions = policy(obs)  # Dict with 'discrete' and 'continuous' keys
        obs, rewards, dones, truncated, infos = env.step(actions)
    env.close()
"""

import numpy as np
import subprocess
import os
import signal
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import multiprocessing as mp
from teeworlds_env import TeeWorldsShmClient, StructuredObservation


@dataclass
class EnvConfig:
    """Configuration for the Teeworlds environment."""
    ddnet_path: str = "../../../../../build/DDNet"
    server_path: str = "../../../../../build/DDNet-Server"
    server_port: int = 8303
    sv_tickrate: int = 50
    max_episode_steps: int = 2000
    startup_wait: float = 5.0
    observation_timeout: float = 2.0


class TeeWorldsVecEnv:
    """
    Vectorized Teeworlds environment compatible with standard RL libraries.

    Action space (hybrid):
        - discrete: MultiDiscrete([3, 2, 2])  # direction(-1,0,1), jump(0,1), hook(0,1)
        - continuous: Box([-1, -1], [1, 1])   # aim_angle, aim_magnitude (normalized)

    Observation space:
        - Box of shape (obs_dim,) containing player state, raycasts, local tiles, other players
    """

    def __init__(
        self,
        num_envs: int = 8,
        config: Optional[EnvConfig] = None,
        auto_start: bool = True,
    ):
        self.num_envs = num_envs
        self.config = config or EnvConfig()

        # Process management
        self.server_proc: Optional[subprocess.Popen] = None
        self.client_procs: List[subprocess.Popen] = []
        self.worker_procs: List[mp.Process] = []
        self.parent_conns: List[mp.connection.Connection] = []
        self.child_conns: List[mp.connection.Connection] = []

        # Environment state
        self.obs_dim: Optional[int] = None
        self.episode_steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)

        # Track if environment is running
        self._running = False

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
        """Start the DDNet server, clients, and worker processes."""
        if self._running:
            return

        print(f"Starting Teeworlds environment with {self.num_envs} clients...")

        # Start server
        self._start_server()

        # Start clients
        self._start_clients()

        # Wait for clients to connect
        time.sleep(self.config.startup_wait)

        # Start worker processes
        self._start_workers()

        # Get initial observations to determine obs_dim
        initial_obs = self._get_initial_observations()
        self.obs_dim = initial_obs.shape[1]

        self._running = True
        print(f"Environment ready. Observation dim: {self.obs_dim}")

    def _start_server(self):
        """Start the DDNet server."""
        server_cmd = [
            self.config.server_path,
            f"sv_port {self.config.server_port}; "
            f"sv_tickrate {self.config.sv_tickrate}; "
            f"sv_max_clients_per_ip 64; "
            f"sv_max_clients 64"
        ]

        self.server_proc = subprocess.Popen(
            server_cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Server started (PID: {self.server_proc.pid}, tickrate: {self.config.sv_tickrate})")
        time.sleep(2)  # Wait for server to initialize

    def _start_clients(self):
        """Start DDNet clients."""
        for i in range(self.num_envs):
            cmd = [
                self.config.ddnet_path,
                f"connect localhost:{self.config.server_port}; "
                f"rl_enable state {i}; "
                f"snd_volume 0; "
                f"gfx_screen_width 320; "
                f"gfx_screen_height 240; "
                f"bind k kill; "
                f"player_name RL_{i}"
            ]

            proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.client_procs.append(proc)

            # Wait for shared memory to be created
            shm_path = f"/dev/shm/teeworlds_rl_{i}"
            for _ in range(60):
                if os.path.exists(shm_path):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Timeout waiting for {shm_path}")

            if (i + 1) % 10 == 0:
                print(f"  Started {i + 1}/{self.num_envs} clients")

        print(f"All {self.num_envs} clients started")

    def _start_workers(self):
        """Start worker processes for each environment."""
        for i in range(self.num_envs):
            parent_conn, child_conn = mp.Pipe()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

            proc = mp.Process(target=_env_worker, args=(i, child_conn, self.config.observation_timeout))
            proc.start()
            self.worker_procs.append(proc)

    def _get_initial_observations(self) -> np.ndarray:
        """Get initial observations from all workers."""
        observations = []
        for i, conn in enumerate(self.parent_conns):
            try:
                obs = conn.recv()
                if obs is None:
                    raise RuntimeError(f"Worker {i} failed to get initial observation")
                observations.append(obs)
            except Exception as e:
                raise RuntimeError(f"Failed to get observation from worker {i}: {e}")

        return np.array(observations, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset all environments and return initial observations."""
        if not self._running:
            self.start()

        # Send reset command to all workers
        for conn in self.parent_conns:
            conn.send({'command': 'reset'})

        # Collect observations
        observations = []
        for conn in self.parent_conns:
            obs = conn.recv()
            if obs is None:
                obs = np.zeros(self.obs_dim, dtype=np.float32)
            observations.append(obs)

        self.episode_steps[:] = 0
        self.episode_rewards[:] = 0.0

        return np.array(observations, dtype=np.float32), {}

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
        discrete_actions = actions['discrete']  # (num_envs, 3)
        continuous_actions = actions['continuous']  # (num_envs, 2)

        # Send actions to all workers
        for i, conn in enumerate(self.parent_conns):
            action_dict = self._convert_action(
                discrete_actions[i],
                continuous_actions[i]
            )
            conn.send({'command': 'step', 'action': action_dict})

        # Collect results
        observations = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i, conn in enumerate(self.parent_conns):
            result = conn.recv()

            if result is None or result.get('obs') is None:
                # Worker failed, use zeros
                observations.append(np.zeros(self.obs_dim, dtype=np.float32))
                infos.append({'error': True})
                continue

            obs = result['obs']
            reward = result['reward']
            done = result['done']
            info = result.get('info', {})

            observations.append(obs)
            rewards[i] = reward
            dones[i] = done

            self.episode_steps[i] += 1
            self.episode_rewards[i] += reward

            # Check for truncation (max steps reached)
            if self.episode_steps[i] >= self.config.max_episode_steps:
                truncated[i] = True
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_steps[i],
                }

            # If done, record episode stats
            if done:
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_steps[i],
                }

            infos.append(info)

        # Auto-reset environments that are done or truncated
        reset_mask = dones | truncated
        if np.any(reset_mask):
            self._reset_envs(reset_mask, observations)

        return (
            np.array(observations, dtype=np.float32),
            rewards,
            dones,
            truncated,
            infos
        )

    def _convert_action(
        self,
        discrete: np.ndarray,
        continuous: np.ndarray
    ) -> Dict[str, int]:
        """Convert network output to game action format."""
        # Discrete actions
        direction = int(discrete[0]) - 1  # 0,1,2 -> -1,0,1
        jump = int(discrete[1])
        hook = int(discrete[2])

        # Continuous aim: angle and magnitude
        # continuous[0] is angle in [-1, 1] -> [-pi, pi]
        # continuous[1] is magnitude in [-1, 1] -> [0, 1]
        aim_angle = float(continuous[0]) * np.pi
        aim_magnitude = (float(continuous[1]) + 1.0) / 2.0  # Map to [0, 1]

        # Convert to game coordinates
        aim_x = int(np.cos(aim_angle) * aim_magnitude * 200)
        aim_y = int(np.sin(aim_angle) * aim_magnitude * 200)

        return {
            'direction': direction,
            'jump': jump,
            'hook': hook,
            'fire': 0,
            'aim_x': aim_x,
            'aim_y': aim_y,
            'weapon': 0,
        }

    def _reset_envs(self, mask: np.ndarray, observations: List[np.ndarray]):
        """Reset specific environments and update observations in place."""
        for i in np.where(mask)[0]:
            self.parent_conns[i].send({'command': 'reset'})

        for i in np.where(mask)[0]:
            obs = self.parent_conns[i].recv()
            if obs is not None:
                observations[i] = obs
            self.episode_steps[i] = 0
            self.episode_rewards[i] = 0.0

    def close(self):
        """Clean up all resources."""
        print("Closing environment...")

        # Tell workers to close
        for conn in self.parent_conns:
            try:
                conn.send({'command': 'close'})
            except:
                pass

        # Wait for workers to finish
        for proc in self.worker_procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

        # Terminate clients
        for proc in self.client_procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass

        # Terminate server
        if self.server_proc:
            try:
                os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
            except:
                pass

        self._running = False
        print("Environment closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _env_worker(env_id: int, conn: mp.connection.Connection, timeout: float):
    """
    Worker process for a single environment.

    Communicates with the main process via pipe and with DDNet via shared memory.
    """
    # Wait for shared memory
    shm_path = f"/dev/shm/teeworlds_rl_{env_id}"
    for _ in range(60):
        if os.path.exists(shm_path):
            break
        time.sleep(0.5)
    else:
        conn.send(None)
        return

    # Connect to shared memory
    client = TeeWorldsShmClient(env_id=env_id)
    if not client.connect():
        conn.send(None)
        return

    # Get initial observation
    obs, info = client.wait_for_observation(timeout=timeout)
    if obs is None:
        conn.send(None)
        return

    # Send initial observation
    state = obs.to_flat_array()
    conn.send(state)

    # Main loop
    while True:
        try:
            msg = conn.recv()
        except:
            break

        if msg is None or msg.get('command') == 'close':
            break

        command = msg.get('command')

        if command == 'reset':
            # Send reset signal and kill command
            client.send_action(reset=1)
            client.reset()  # Sends kill keypress

            # Wait for respawn
            time.sleep(0.1)

            # Get new observation
            obs, info = client.wait_for_observation(timeout=timeout)
            if obs is None:
                conn.send(None)
            else:
                conn.send(obs.to_flat_array())

        elif command == 'step':
            action = msg.get('action', {})
            client.send_action(**action)

            # Get observation
            obs, info = client.wait_for_observation(timeout=timeout)

            if obs is None:
                conn.send(None)
            else:
                conn.send({
                    'obs': obs.to_flat_array(),
                    'reward': info.get('reward', 0.0),
                    'done': info.get('done', False),
                    'info': info,
                })

    client.disconnect()


# Convenience function for creating environment
def make_env(num_envs: int = 8, **kwargs) -> TeeWorldsVecEnv:
    """Create a Teeworlds vectorized environment."""
    config = EnvConfig(**kwargs)
    return TeeWorldsVecEnv(num_envs=num_envs, config=config)


if __name__ == "__main__":
    # Test the environment
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    print(f"Testing TeeWorldsVecEnv with {args.num_envs} environments...")

    with TeeWorldsVecEnv(num_envs=args.num_envs) as env:
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")

        total_rewards = np.zeros(args.num_envs)

        for step in range(args.steps):
            # Random actions
            actions = {
                'discrete': np.random.randint([0, 0, 0], [3, 2, 2], size=(args.num_envs, 3)),
                'continuous': np.random.uniform(-1, 1, size=(args.num_envs, 2)).astype(np.float32),
            }

            obs, rewards, dones, truncated, infos = env.step(actions)
            total_rewards += rewards

            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}: mean_reward={rewards.mean():.4f}, dones={dones.sum()}")

        print(f"\nTotal rewards: {total_rewards}")
        print(f"Mean total reward: {total_rewards.mean():.4f}")
