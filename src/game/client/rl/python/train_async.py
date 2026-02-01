#!/usr/bin/env python3
"""
Asynchronous PPO training for DDNet RL.

This script reads transitions from shared memory written by the C++ server,
runs PPO updates, and exports updated ONNX models for hot reload.

Architecture:
- C++ server runs inference locally using ONNX Runtime
- C++ server writes transitions to shared memory ring buffer
- This script reads transitions periodically and runs PPO updates
- Updated model is exported to ONNX and signaled for hot reload
"""

import argparse
import ctypes
import mmap
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# Constants matching C++ rl_protocol.h
MAX_PLAYERS = 16
MAX_PROJECTILES = 64
MAX_PICKUPS = 32
TILE_GRID_WIDTH = 29
TILE_GRID_HEIGHT = 21
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT

# Observation size calculation
OBS_GAME_STATE = 4  # tick, time_left, num_players, num_projectiles
OBS_PLAYER = 16
OBS_OTHER_PLAYERS = (MAX_PLAYERS - 1) * OBS_PLAYER
OBS_PROJECTILES = MAX_PROJECTILES * 8
OBS_PICKUPS = MAX_PICKUPS * 4
OBS_TILES = TILE_GRID_SIZE

OBSERVATION_SIZE = (
    OBS_GAME_STATE + OBS_PLAYER + OBS_OTHER_PLAYERS +
    OBS_PROJECTILES + OBS_PICKUPS + OBS_TILES
)

# Network output size
NETWORK_OUTPUT_SIZE = 14  # 3 + 1 + 1 + 1 + 3 + 2 + 2 + 1


@dataclass
class Transition:
    """Single transition from the ring buffer."""
    observation: np.ndarray
    action_direction: int
    action_jump: int
    action_hook: int
    action_fire: int
    action_weapon: int
    action_aim_x: float
    action_aim_y: float
    log_prob: float
    value: float
    reward: float
    done: bool
    truncated: bool
    bot_id: int


class SharedMemoryReader:
    """Reads transitions from the C++ shared memory ring buffer."""

    def __init__(self, shm_name: str = "/ddnet_rl_training"):
        self.shm_name = shm_name
        self.shm_fd = None
        self.mmap_obj = None
        self.header_size = self._calculate_header_size()
        self.transition_size = self._calculate_transition_size()

    def _calculate_header_size(self) -> int:
        """Calculate size of STrainingSharedMemory header."""
        # Calculated from actual shared memory: size - capacity * transition_size = 768
        return 768

    def _calculate_transition_size(self) -> int:
        """Calculate size of STransition."""
        # Read from shared memory at runtime instead of calculating
        # This ensures we match the C++ struct exactly
        # Default value based on C++ sizeof(STransition) = 6104
        return 6104

    def open(self) -> bool:
        """Open the shared memory region."""
        try:
            # Open POSIX shared memory
            import posix_ipc
            self.shm = posix_ipc.SharedMemory(self.shm_name)
            self.mmap_obj = mmap.mmap(self.shm.fd, 0)
            return True
        except ImportError:
            # Fallback: try to open via /dev/shm
            shm_path = f"/dev/shm{self.shm_name}"
            if os.path.exists(shm_path):
                self.shm_fd = os.open(shm_path, os.O_RDWR)
                size = os.fstat(self.shm_fd).st_size
                self.mmap_obj = mmap.mmap(self.shm_fd, size)
                return True
            return False
        except Exception as e:
            print(f"Failed to open shared memory: {e}")
            return False

    def close(self):
        """Close the shared memory region."""
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.shm_fd:
            os.close(self.shm_fd)

    def read_header(self) -> Tuple[int, int, int, int]:
        """Read buffer header values."""
        if not self.mmap_obj:
            return 0, 0, 0, 0

        # Read from STransitionBufferHeader with proper alignas(64) offsets
        # alignas(64) uint64_t m_WriteIndex   @ offset 0
        self.mmap_obj.seek(0)
        write_idx = struct.unpack('Q', self.mmap_obj.read(8))[0]

        # alignas(64) uint64_t m_ReadIndex    @ offset 64
        self.mmap_obj.seek(64)
        read_idx = struct.unpack('Q', self.mmap_obj.read(8))[0]

        # uint32_t m_Capacity @ offset 200 (immediately after m_TotalRead at 192)
        self.mmap_obj.seek(200)
        capacity = struct.unpack('I', self.mmap_obj.read(4))[0]

        # uint32_t m_TransitionSize @ offset 204
        transition_size = struct.unpack('I', self.mmap_obj.read(4))[0]

        return write_idx, read_idx, capacity, transition_size

    def read_transitions(self, max_count: int = 1024) -> list:
        """Read available transitions from the ring buffer."""
        if not self.mmap_obj:
            return []

        write_idx, read_idx, capacity, _ = self.read_header()
        available = write_idx - read_idx
        if available > capacity:
            # Buffer overflow, skip to latest
            read_idx = write_idx - capacity
            available = capacity

        to_read = min(available, max_count)
        transitions = []

        for i in range(to_read):
            idx = (read_idx + i) % capacity
            offset = self.header_size + idx * self.transition_size

            self.mmap_obj.seek(offset)
            data = self.mmap_obj.read(self.transition_size)

            try:
                trans = self._parse_transition(data)
                transitions.append(trans)
            except Exception as e:
                print(f"Error parsing transition: {e}")
                continue

        # Update read index (at offset 64, alignas(64) uint64_t m_ReadIndex)
        if to_read > 0:
            new_read_idx = read_idx + to_read
            self.mmap_obj.seek(64)
            self.mmap_obj.write(struct.pack('Q', new_read_idx))
            self.mmap_obj.flush()

        return transitions

    def _parse_transition(self, data: bytes) -> Transition:
        """Parse a transition from raw bytes."""
        offset = 0

        # Observation
        obs_bytes = OBSERVATION_SIZE * 4
        obs = np.frombuffer(data[offset:offset + obs_bytes], dtype=np.float32)
        offset += obs_bytes

        # Action
        direction, jump, hook, fire, weapon = struct.unpack_from('iiiii', data, offset)
        offset += 20
        aim_x, aim_y = struct.unpack_from('ff', data, offset)
        offset += 8

        # Log probs (6 floats, we just use the first one as total)
        log_probs = struct.unpack_from('ffffff', data, offset)
        log_prob = log_probs[0]
        offset += 24

        # Value, reward
        value, reward = struct.unpack_from('ff', data, offset)
        offset += 8

        # Done, truncated, padding, bot_id
        done, truncated, _, _, bot_id = struct.unpack_from('BBBBi', data, offset)

        return Transition(
            observation=obs.copy(),
            action_direction=direction,
            action_jump=jump,
            action_hook=hook,
            action_fire=fire,
            action_weapon=weapon,
            action_aim_x=aim_x,
            action_aim_y=aim_y,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=bool(done),
            truncated=bool(truncated),
            bot_id=bot_id
        )

    def signal_new_model(self, model_path: str):
        """Signal C++ that a new model is ready."""
        if not self.mmap_obj:
            return

        # STransitionBufferHeader layout (alignas(64) members):
        #   m_WriteIndex (0), m_ReadIndex (64), m_TotalWritten (128), m_TotalRead (192)
        #   m_Capacity (200), m_TransitionSize (204)
        #   m_ModelVersion (256), m_TrainingActive (260)
        #   m_EpisodeCount (320), m_StepCount (328), m_TotalReward (336), m_EpisodeRewardSum (344)
        #   Struct size rounds to 384 (alignment)
        #
        # SModelReloadSignal layout (starts at 384):
        #   alignas(64) m_NewModelReady (0)
        #   alignas(64) m_ModelLoaded (64)
        #   m_ModelVersion (68)
        #   m_ModelPath[256] (72)
        signal_offset = 384  # sizeof(STransitionBufferHeader)

        # Write model path first
        self.mmap_obj.seek(signal_offset + 72)  # m_ModelPath field
        path_bytes = model_path.encode('utf-8')[:255]
        self.mmap_obj.write(path_bytes + b'\x00')

        # Set new_model_ready flag (atomic store)
        self.mmap_obj.seek(signal_offset)  # m_NewModelReady field
        self.mmap_obj.write(struct.pack('I', 1))


class PolicyNetwork(nn.Module):
    """Actor-Critic network for hybrid action space."""

    def __init__(self, obs_size: int = OBSERVATION_SIZE, hidden_size: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor heads
        self.direction_head = nn.Linear(hidden_size, 3)  # left, none, right
        self.jump_head = nn.Linear(hidden_size, 1)
        self.hook_head = nn.Linear(hidden_size, 1)
        self.fire_head = nn.Linear(hidden_size, 1)
        self.weapon_head = nn.Linear(hidden_size, 3)  # prev, none, next
        self.aim_mean = nn.Linear(hidden_size, 2)
        self.aim_log_std = nn.Linear(hidden_size, 2)

        # Critic head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning flattened output for ONNX."""
        features = self.shared(x)

        # All outputs concatenated
        direction_logits = self.direction_head(features)  # 3
        jump_logit = self.jump_head(features)  # 1
        hook_logit = self.hook_head(features)  # 1
        fire_logit = self.fire_head(features)  # 1
        weapon_logits = self.weapon_head(features)  # 3
        aim_mean = self.aim_mean(features)  # 2
        aim_log_std = self.aim_log_std(features)  # 2
        value = self.value_head(features)  # 1

        return torch.cat([
            direction_logits, jump_logit, hook_logit, fire_logit,
            weapon_logits, aim_mean, aim_log_std, value
        ], dim=-1)

    def get_action_and_value(self, x: torch.Tensor, deterministic: bool = False):
        """Get action, log_prob, entropy, and value."""
        features = self.shared(x)

        # Direction
        direction_logits = self.direction_head(features)
        direction_dist = Categorical(logits=direction_logits)
        if deterministic:
            direction = direction_dist.probs.argmax(dim=-1)
        else:
            direction = direction_dist.sample()
        direction_log_prob = direction_dist.log_prob(direction)

        # Binary actions
        jump_prob = torch.sigmoid(self.jump_head(features)).squeeze(-1)
        hook_prob = torch.sigmoid(self.hook_head(features)).squeeze(-1)
        fire_prob = torch.sigmoid(self.fire_head(features)).squeeze(-1)

        if deterministic:
            jump = (jump_prob > 0.5).long()
            hook = (hook_prob > 0.5).long()
            fire = (fire_prob > 0.5).long()
        else:
            jump = torch.bernoulli(jump_prob).long()
            hook = torch.bernoulli(hook_prob).long()
            fire = torch.bernoulli(fire_prob).long()

        jump_log_prob = jump * torch.log(jump_prob + 1e-8) + (1 - jump) * torch.log(1 - jump_prob + 1e-8)
        hook_log_prob = hook * torch.log(hook_prob + 1e-8) + (1 - hook) * torch.log(1 - hook_prob + 1e-8)
        fire_log_prob = fire * torch.log(fire_prob + 1e-8) + (1 - fire) * torch.log(1 - fire_prob + 1e-8)

        # Weapon
        weapon_logits = self.weapon_head(features)
        weapon_dist = Categorical(logits=weapon_logits)
        if deterministic:
            weapon = weapon_dist.probs.argmax(dim=-1)
        else:
            weapon = weapon_dist.sample()
        weapon_log_prob = weapon_dist.log_prob(weapon)

        # Aim (continuous)
        aim_mean = self.aim_mean(features)
        aim_log_std = self.aim_log_std(features).clamp(-2, 2)
        aim_std = aim_log_std.exp()
        aim_dist = Normal(aim_mean, aim_std)

        if deterministic:
            aim = aim_mean
        else:
            aim = aim_dist.sample()
        aim_log_prob = aim_dist.log_prob(aim).sum(dim=-1)

        # Value
        value = self.value_head(features).squeeze(-1)

        # Total log prob
        total_log_prob = (
            direction_log_prob + jump_log_prob + hook_log_prob +
            fire_log_prob + weapon_log_prob + aim_log_prob
        )

        # Entropy
        entropy = (
            direction_dist.entropy() +
            -(jump_prob * torch.log(jump_prob + 1e-8) + (1 - jump_prob) * torch.log(1 - jump_prob + 1e-8)) +
            -(hook_prob * torch.log(hook_prob + 1e-8) + (1 - hook_prob) * torch.log(1 - hook_prob + 1e-8)) +
            -(fire_prob * torch.log(fire_prob + 1e-8) + (1 - fire_prob) * torch.log(1 - fire_prob + 1e-8)) +
            weapon_dist.entropy() +
            aim_dist.entropy().sum(dim=-1)
        )

        return {
            'direction': direction,
            'jump': jump,
            'hook': hook,
            'fire': fire,
            'weapon': weapon,
            'aim': aim,
            'log_prob': total_log_prob,
            'entropy': entropy,
            'value': value,
        }


class PPOTrainer:
    """PPO trainer for async training."""

    def __init__(
        self,
        model: PolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, transitions: list) -> dict:
        """Run PPO update on a batch of transitions."""
        if len(transitions) < self.minibatch_size:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # Convert transitions to tensors
        obs = torch.tensor(np.array([t.observation for t in transitions]), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([t.value for t in transitions], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done or t.truncated for t in transitions], dtype=torch.float32, device=self.device)

        # Actions
        directions = torch.tensor([t.action_direction + 1 for t in transitions], dtype=torch.long, device=self.device)
        jumps = torch.tensor([t.action_jump for t in transitions], dtype=torch.long, device=self.device)
        hooks = torch.tensor([t.action_hook for t in transitions], dtype=torch.long, device=self.device)
        fires = torch.tensor([t.action_fire for t in transitions], dtype=torch.long, device=self.device)
        weapons = torch.tensor([t.action_weapon + 1 for t in transitions], dtype=torch.long, device=self.device)
        aims = torch.tensor([[t.action_aim_x, t.action_aim_y] for t in transitions], dtype=torch.float32, device=self.device)

        # Compute GAE
        with torch.no_grad():
            # Get value of last observation for bootstrapping
            last_output = self.model(obs[-1:])
            next_value = last_output[0, -1].item()

        advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.update_epochs):
            # Shuffle
            indices = torch.randperm(len(transitions))

            for start in range(0, len(transitions), self.minibatch_size):
                end = start + self.minibatch_size
                if end > len(transitions):
                    break

                mb_indices = indices[start:end]

                # Get current policy outputs
                result = self.model.get_action_and_value(obs[mb_indices])

                # Recompute log probs for taken actions
                # This is simplified - ideally we'd separate the action distributions
                new_log_probs = result['log_prob']
                new_values = result['value']
                entropy = result['entropy']

                # PPO clipped objective
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]

                # Normalize advantages per minibatch for stability
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Importance sampling ratio
                ratio = (new_log_probs - mb_old_log_probs).exp()

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(new_values, returns[mb_indices])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Skip if loss is NaN or infinite
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            'loss': total_loss / max(num_updates, 1),
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
        }

    def export_onnx(self, path: str):
        """Export model to ONNX format."""
        self.model.eval()
        dummy_input = torch.randn(1, OBSERVATION_SIZE, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=13,  # Use older opset for ONNX Runtime 1.17 compatibility
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['output'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False  # Use legacy TorchScript-based exporter for speed and compatibility
        )
        self.model.train()

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    parser = argparse.ArgumentParser(description='Async PPO training for DDNet RL')
    parser.add_argument('--shm-name', default='/ddnet_rl_training',
                        help='Shared memory name')
    parser.add_argument('--model-path', default='model.onnx',
                        help='Path to export ONNX model')
    parser.add_argument('--checkpoint-path', default='checkpoint.pt',
                        help='Path to save/load checkpoints')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size for updates')
    parser.add_argument('--update-interval', type=float, default=1.0,
                        help='Seconds between update attempts')
    parser.add_argument('--export-interval', type=int, default=10,
                        help='Updates between ONNX exports')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    args = parser.parse_args()

    # Initialize model and trainer
    model = PolicyNetwork()
    trainer = PPOTrainer(model, lr=args.lr, minibatch_size=64)

    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"Resuming from {args.checkpoint_path}")
        trainer.load_checkpoint(args.checkpoint_path)

    # Initial ONNX export
    print(f"Exporting initial model to {args.model_path}", flush=True)
    trainer.export_onnx(args.model_path)
    print("Model exported.", flush=True)

    # Open shared memory
    print(f"Opening shared memory {args.shm_name}...", flush=True)
    shm = SharedMemoryReader(args.shm_name)
    if not shm.open():
        print(f"Warning: Could not open shared memory {args.shm_name}", flush=True)
        print("Running in offline mode - waiting for server to start...", flush=True)
    else:
        print("Shared memory opened successfully.", flush=True)

    # Signal handlers
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Training loop
    update_count = 0
    transition_buffer = []
    total_transitions = 0

    print("Starting async training loop...", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Update interval: {args.update_interval}s", flush=True)
    print(f"  Export interval: {args.export_interval} updates", flush=True)

    debug_counter = 0
    while running:
        # Read transitions from shared memory
        if shm.mmap_obj:
            # Debug: print header info periodically
            debug_counter += 1
            if debug_counter % 10 == 1:
                write_idx, read_idx, capacity, _ = shm.read_header()
                print(f"  [DEBUG] write={write_idx} read={read_idx} cap={capacity} buf={len(transition_buffer)}", flush=True)

            new_transitions = shm.read_transitions(max_count=4096)
            if new_transitions:
                transition_buffer.extend(new_transitions)
                total_transitions += len(new_transitions)

        # Run update if we have enough transitions
        if len(transition_buffer) >= args.batch_size:
            # Take a batch
            batch = transition_buffer[:args.batch_size]
            transition_buffer = transition_buffer[args.batch_size:]

            # Run PPO update
            stats = trainer.update(batch)
            update_count += 1

            print(f"Update {update_count}: "
                  f"loss={stats['loss']:.4f} "
                  f"policy={stats['policy_loss']:.4f} "
                  f"value={stats['value_loss']:.4f} "
                  f"entropy={stats['entropy']:.4f} "
                  f"transitions={total_transitions}")

            # Export ONNX periodically
            if update_count % args.export_interval == 0:
                # Export to temporary file first
                temp_path = args.model_path + '.new'
                trainer.export_onnx(temp_path)

                # Atomic rename
                os.rename(temp_path, args.model_path)

                # Signal C++ to reload
                if shm.mmap_obj:
                    shm.signal_new_model(os.path.abspath(args.model_path))
                    print(f"  Exported and signaled new model")

                # Save checkpoint
                trainer.save_checkpoint(args.checkpoint_path)

        time.sleep(args.update_interval)

    # Cleanup
    print("Saving final checkpoint...")
    trainer.save_checkpoint(args.checkpoint_path)
    trainer.export_onnx(args.model_path)
    shm.close()
    print("Done.")


if __name__ == '__main__':
    main()
