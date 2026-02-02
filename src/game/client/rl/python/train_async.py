#!/usr/bin/env python3
"""
Asynchronous V-trace training for DDNet RL.

This script reads transitions from shared memory written by the C++ server,
runs V-trace updates with off-policy correction, and exports updated ONNX
models for hot reload.

Architecture:
- C++ server runs inference locally using ONNX Runtime
- C++ server writes transitions to shared memory ring buffer
- This script reads transitions periodically and runs V-trace updates
- V-trace handles policy lag by using importance sampling with truncation
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
TILE_GRID_WIDTH = 29
TILE_GRID_HEIGHT = 21
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT

# Observation size calculation (simplified)
OBS_PLAYER = 16      # local player state
OBS_TILES = TILE_GRID_SIZE  # 29x21 = 609 tile IDs

OBSERVATION_SIZE = OBS_PLAYER + OBS_TILES  # = 625

# Network output size
NETWORK_OUTPUT_SIZE = 14  # 3 + 1 + 1 + 1 + 3 + 2 + 2 + 1

# Tile embedding configuration
NUM_TILE_TYPES = 256  # Tile IDs are 0-255
TILE_EMBEDDING_DIM = 16


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
        self.header_size = 768  # STrainingSharedMemory header size
        self.transition_size = 0  # Will be read from header

    def open(self) -> bool:
        """Open the shared memory region."""
        try:
            # Open POSIX shared memory
            import posix_ipc
            self.shm = posix_ipc.SharedMemory(self.shm_name)
            self.mmap_obj = mmap.mmap(self.shm.fd, 0)
        except ImportError:
            # Fallback: try to open via /dev/shm
            shm_path = f"/dev/shm{self.shm_name}"
            if os.path.exists(shm_path):
                self.shm_fd = os.open(shm_path, os.O_RDWR)
                size = os.fstat(self.shm_fd).st_size
                self.mmap_obj = mmap.mmap(self.shm_fd, size)
            else:
                return False
        except Exception as e:
            print(f"Failed to open shared memory: {e}")
            return False

        # Read transition size from header (offset 204)
        if self.mmap_obj:
            # Read capacity and transition size
            self.mmap_obj.seek(200)
            capacity = struct.unpack('I', self.mmap_obj.read(4))[0]
            self.transition_size = struct.unpack('I', self.mmap_obj.read(4))[0]

            # Calculate expected transition size based on current OBSERVATION_SIZE
            # Observation + Action(5 ints + 2 floats) + LogProbs(6 floats) + Value + Reward + flags + BotId
            expected_obs_bytes = OBSERVATION_SIZE * 4
            expected_trans_size = expected_obs_bytes + 28 + 24 + 4 + 4 + 4 + 4  # = OBSERVATION_SIZE*4 + 68

            print(f"Shared memory opened:")
            print(f"  capacity={capacity}")
            print(f"  transition_size={self.transition_size} bytes")
            print(f"  expected ~{expected_trans_size} bytes (obs={OBSERVATION_SIZE} floats = {expected_obs_bytes} bytes)")

            if self.transition_size != expected_trans_size and abs(self.transition_size - expected_trans_size) > 16:
                print(f"  WARNING: Size mismatch! Server may be using different observation size.")
                print(f"  If server was just restarted, this may indicate a build mismatch.")

            # Flush stale transitions by setting read index = write index
            self.mmap_obj.seek(0)
            write_idx = struct.unpack('Q', self.mmap_obj.read(8))[0]
            self.mmap_obj.seek(64)
            self.mmap_obj.write(struct.pack('Q', write_idx))
            self.mmap_obj.flush()
            print(f"  Flushed buffer (set read_idx = write_idx = {write_idx})")

            return True
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

        # Sanity check aim values - clamp to reasonable range
        if not (-100.0 <= aim_x <= 100.0) or np.isnan(aim_x) or np.isinf(aim_x):
            aim_x = 0.0
        if not (-100.0 <= aim_y <= 100.0) or np.isnan(aim_y) or np.isinf(aim_y):
            aim_y = 0.0

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
    """Actor-Critic network with tile embeddings for hybrid action space."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()

        # Tile embedding layer: maps tile IDs (0-255) to learned embeddings
        self.tile_embedding = nn.Embedding(NUM_TILE_TYPES, TILE_EMBEDDING_DIM)

        # Input size after embedding tiles
        # Player: 16, Tiles: 609 * TILE_EMBEDDING_DIM
        tile_features_size = TILE_GRID_SIZE * TILE_EMBEDDING_DIM
        total_input_size = OBS_PLAYER + tile_features_size

        # Running observation normalization for non-tile features (player state only)
        self.register_buffer('obs_mean', torch.zeros(OBS_PLAYER))
        self.register_buffer('obs_var', torch.ones(OBS_PLAYER))
        self.register_buffer('obs_count', torch.tensor(1e-4))

        # Shared feature extractor with LayerNorm for stability
        self.shared = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Initialize weights with orthogonal initialization for stability
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

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

        # Initialize actor heads with small weights for stable initial policy
        for head in [self.direction_head, self.jump_head, self.hook_head,
                     self.fire_head, self.weapon_head, self.aim_mean, self.aim_log_std]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        # Initialize value head with standard gain
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _process_observation(self, x: torch.Tensor, update_stats: bool = False) -> torch.Tensor:
        """Process raw observation into features with tile embeddings.

        Input x has shape [batch, 625]:
          - x[:, :16] = player state (16 floats, pre-normalized by C++ to ~[-1,1])
          - x[:, 16:] = tile IDs (609 floats, to be cast to int for embedding)

        Note: We don't apply running normalization here because:
        1. C++ already pre-normalizes observations to roughly [-1, 1]
        2. The network has LayerNorm which handles varying input scales
        3. Running stats would cause mismatch between ONNX inference and Python training
        """
        # Handle NaN/inf in entire observation first
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Split observation into components
        player_state = x[:, :OBS_PLAYER]  # [batch, 16]
        tile_ids_float = x[:, OBS_PLAYER:]  # [batch, 609]

        # Just clamp to handle any outliers from C++ normalization
        non_tile_features = player_state.clamp(-10.0, 10.0)

        # Cast tile IDs to long for embedding lookup
        tile_ids = tile_ids_float.clamp(0, NUM_TILE_TYPES - 1).long()

        # Embed tiles: [batch, 609] -> [batch, 609, 16]
        tile_features = self.tile_embedding(tile_ids)

        # Flatten tile features: [batch, 609, 16] -> [batch, 609 * 16]
        tile_flat = tile_features.flatten(start_dim=1)

        # Combine all features
        combined = torch.cat([non_tile_features, tile_flat], dim=1)

        return combined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning flattened output for ONNX."""
        combined = self._process_observation(x)
        features = self.shared(combined)

        # All outputs concatenated
        direction_logits = self.direction_head(features)  # 3
        jump_logit = self.jump_head(features)  # 1
        hook_logit = self.hook_head(features)  # 1
        fire_logit = self.fire_head(features)  # 1
        weapon_logits = self.weapon_head(features)  # 3
        aim_mean = self.aim_mean(features)  # 2
        aim_log_std = self.aim_log_std(features).clamp(-2, 2)  # 2, clamped to match get_action_and_value
        value = self.value_head(features)  # 1

        return torch.cat([
            direction_logits, jump_logit, hook_logit, fire_logit,
            weapon_logits, aim_mean, aim_log_std, value
        ], dim=-1)

    def get_action_and_value(self, x: torch.Tensor, deterministic: bool = False):
        """Get action, log_prob, entropy, and value."""
        combined = self._process_observation(x)
        features = self.shared(combined)

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

    def evaluate_actions(self, x: torch.Tensor, directions: torch.Tensor,
                         jumps: torch.Tensor, hooks: torch.Tensor,
                         fires: torch.Tensor, weapons: torch.Tensor,
                         aims: torch.Tensor) -> dict:
        """Evaluate log probs and entropy for given actions (for PPO update)."""
        # Don't update stats here - must match the normalization used in ONNX during inference
        combined = self._process_observation(x, update_stats=False)
        features = self.shared(combined)

        # Check for NaN in features and skip if found
        if torch.isnan(features).any():
            raise ValueError("NaN detected in network features")

        # Direction
        direction_logits = self.direction_head(features)
        direction_dist = Categorical(logits=direction_logits)
        direction_log_prob = direction_dist.log_prob(directions)

        # Binary actions - compute log prob of given actions
        jump_prob = torch.sigmoid(self.jump_head(features)).squeeze(-1)
        hook_prob = torch.sigmoid(self.hook_head(features)).squeeze(-1)
        fire_prob = torch.sigmoid(self.fire_head(features)).squeeze(-1)

        jump_log_prob = jumps * torch.log(jump_prob + 1e-8) + (1 - jumps) * torch.log(1 - jump_prob + 1e-8)
        hook_log_prob = hooks * torch.log(hook_prob + 1e-8) + (1 - hooks) * torch.log(1 - hook_prob + 1e-8)
        fire_log_prob = fires * torch.log(fire_prob + 1e-8) + (1 - fires) * torch.log(1 - fire_prob + 1e-8)

        # Weapon
        weapon_logits = self.weapon_head(features)
        weapon_dist = Categorical(logits=weapon_logits)
        weapon_log_prob = weapon_dist.log_prob(weapons)

        # Aim (continuous)
        aim_mean = self.aim_mean(features)
        aim_log_std = self.aim_log_std(features).clamp(-2, 2)
        aim_std = aim_log_std.exp()
        aim_dist = Normal(aim_mean, aim_std)
        aim_log_prob = aim_dist.log_prob(aims).sum(dim=-1)

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
            'log_prob': total_log_prob,
            'entropy': entropy,
            'value': value,
        }


class VTraceTrainer:
    """V-trace trainer for async training with off-policy correction.

    Based on IMPALA: https://arxiv.org/abs/1802.01561

    V-trace handles the policy lag between C++ ONNX inference and Python training
    by using importance sampling with truncation to correct for off-policy data.
    """

    def __init__(
        self,
        model: PolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        rho_bar: float = 0.5,      # Truncation for importance weights (lower = more stable)
        c_bar: float = 0.5,        # Truncation for trace coefficients (lower = more stable)
        clip_epsilon: float = 0.1, # PPO-style clipping for trust region (tighter)
        target_kl: float = 0.015,  # KL divergence threshold for early stopping
        value_coef: float = 0.5,   # Standard value coefficient
        entropy_coef: float = 0.01,    # Coefficient for entropy targeting
        target_entropy: float = 5.0,   # Target entropy - pulls policy toward this value
        max_grad_norm: float = 0.5,
        update_epochs: int = 1,    # V-trace typically uses fewer epochs (often just 1)
        minibatch_size: int = 256, # Larger minibatches work better with V-trace
    ):
        self.model = model
        # Use same decay for both moments + weight decay to prevent sudden large updates
        # Research: https://openreview.net/forum?id=m9Jfdz4ymO
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.9),      # Same decay prevents policy collapse
            weight_decay=1e-4,     # L2 regularization for stability
        )
        self.gamma = gamma
        self.rho_bar = rho_bar      # ρ̄ - truncation for policy gradient
        self.c_bar = c_bar          # c̄ - truncation for trace coefficient
        self.clip_epsilon = clip_epsilon  # PPO-style clipping
        self.target_kl = target_kl  # KL threshold for early stopping
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.target_entropy = target_entropy
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_vtrace(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        log_rhos: torch.Tensor,  # log(π_new / π_old) = new_log_prob - old_log_prob
        bootstrap_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute V-trace targets and advantages.

        V-trace corrects for off-policy data by using truncated importance sampling.
        This allows learning from data collected by older policy versions.

        Args:
            rewards: Tensor of rewards [T]
            values: Value estimates V(x_t) from current policy [T]
            dones: Episode termination flags [T]
            log_rhos: Log importance ratios log(π/μ) [T]
            bootstrap_value: V(x_T) for bootstrapping

        Returns:
            vtrace_targets: V-trace value targets [T]
            advantages: Policy gradient advantages [T]
        """
        T = len(rewards)

        # Compute importance ratios and clip them
        rhos = torch.exp(log_rhos.clamp(-20.0, 20.0))  # π(a|s) / μ(a|s)
        clipped_rhos = torch.clamp(rhos, max=self.rho_bar)  # ρ̄_t = min(ρ̄, ρ_t)
        cs = torch.clamp(rhos, max=self.c_bar)  # c_t = min(c̄, ρ_t)

        # Compute TD errors: δ_t = ρ̄_t * (r_t + γ * V(x_{t+1}) - V(x_t))
        # Note: we use clipped_rhos for the TD error
        values_t_plus_1 = torch.cat([values[1:], torch.tensor([bootstrap_value], device=values.device)])
        not_done = 1.0 - dones
        deltas = clipped_rhos * (rewards + self.gamma * values_t_plus_1 * not_done - values)

        # Compute V-trace targets iterating backwards
        # v_s = V(x_s) + Σ_{t=s}^{T-1} γ^{t-s} (Π_{i=s}^{t-1} c_i) * δ_t
        vtrace_targets = torch.zeros_like(rewards)
        vtrace_minus_v = torch.zeros_like(rewards)  # v_s - V(x_s)

        last_vtrace_minus_v = 0.0
        for t in reversed(range(T)):
            # v_{t+1} - V(x_{t+1}) term, propagated with discount and trace coef
            vtrace_minus_v[t] = deltas[t] + self.gamma * not_done[t] * cs[t] * last_vtrace_minus_v
            last_vtrace_minus_v = vtrace_minus_v[t]

        vtrace_targets = vtrace_minus_v + values

        # Policy gradient advantages: ρ̄_t * (r_t + γ * v_{t+1} - V(x_t))
        # where v_{t+1} is the V-trace target (not the value estimate)
        vtrace_t_plus_1 = torch.cat([vtrace_targets[1:], torch.tensor([bootstrap_value], device=values.device)])
        advantages = clipped_rhos * (rewards + self.gamma * vtrace_t_plus_1 * not_done - values)

        return vtrace_targets, advantages

    def update(self, transitions: list) -> dict:
        """Run V-trace update on a batch of transitions.

        Unlike PPO, V-trace properly handles off-policy data by using importance
        sampling with truncation. This is essential for async training where
        the C++ inference uses an older policy version.
        """
        if len(transitions) < self.minibatch_size:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'avg_rho': 0, 'clip_frac': 0}

        # Convert transitions to tensors
        obs = torch.tensor(np.array([t.observation for t in transitions]), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done or t.truncated for t in transitions], dtype=torch.float32, device=self.device)

        # Validate inputs - replace NaN/inf with safe values
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        old_log_probs = torch.nan_to_num(old_log_probs, nan=-10.0, posinf=-1.0, neginf=-100.0)
        old_log_probs = old_log_probs.clamp(-100.0, 0.0)  # Log probs should be <= 0
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=100.0, neginf=-100.0)
        rewards = rewards.clamp(-100.0, 100.0)  # Clamp extreme rewards

        # Actions
        directions = torch.tensor([t.action_direction + 1 for t in transitions], dtype=torch.long, device=self.device)
        jumps = torch.tensor([t.action_jump for t in transitions], dtype=torch.long, device=self.device)
        hooks = torch.tensor([t.action_hook for t in transitions], dtype=torch.long, device=self.device)
        fires = torch.tensor([t.action_fire for t in transitions], dtype=torch.long, device=self.device)
        weapons = torch.tensor([t.action_weapon + 1 for t in transitions], dtype=torch.long, device=self.device)
        aims = torch.tensor([[t.action_aim_x, t.action_aim_y] for t in transitions], dtype=torch.float32, device=self.device)

        # First pass: compute new log probs and values for all transitions
        # This is needed to compute V-trace targets
        with torch.no_grad():
            all_result = self.model.evaluate_actions(
                obs, directions, jumps.float(), hooks.float(), fires.float(), weapons, aims
            )
            new_log_probs_all = all_result['log_prob']
            values_all = all_result['value']

            # Bootstrap value for last transition
            last_output = self.model(obs[-1:])
            bootstrap_value = last_output[0, -1].item()

            # Compute log importance ratios: log(π_new / π_old)
            log_rhos = new_log_probs_all - old_log_probs

            # Compute V-trace targets and advantages
            vtrace_targets, advantages = self.compute_vtrace(
                rewards, values_all, dones, log_rhos, bootstrap_value
            )

            # Clip V-trace targets to be within reasonable range of current values
            # This prevents the value function from chasing extreme targets
            vtrace_targets = torch.clamp(
                vtrace_targets,
                values_all - 10.0,  # Don't decrease by more than 10
                values_all + 10.0   # Don't increase by more than 10
            )

            # Clip advantages to prevent extreme policy updates (tighter than before)
            advantages = torch.clamp(advantages, -5.0, 5.0)

            # Track average importance ratio for monitoring
            avg_rho = torch.exp(log_rhos.clamp(-20, 20)).mean().item()

        # V-trace update with PPO clipping for extra stability
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_frac = 0
        total_approx_kl = 0
        num_updates = 0
        early_stop = False

        for epoch in range(self.update_epochs):
            if early_stop:
                break

            # Shuffle
            indices = torch.randperm(len(transitions))

            for start in range(0, len(transitions), self.minibatch_size):
                if early_stop:
                    break

                end = start + self.minibatch_size
                if end > len(transitions):
                    break

                mb_indices = indices[start:end]

                try:
                    # Evaluate log probs for the actions that were actually taken
                    result = self.model.evaluate_actions(
                        obs[mb_indices],
                        directions[mb_indices],
                        jumps[mb_indices].float(),
                        hooks[mb_indices].float(),
                        fires[mb_indices].float(),
                        weapons[mb_indices],
                        aims[mb_indices]
                    )
                except ValueError as e:
                    print(f"  Warning: {e}, skipping minibatch")
                    continue

                new_log_probs = result['log_prob']
                new_values = result['value']
                entropy = result['entropy']

                # Get old log probs for this minibatch (from the behavior policy)
                mb_old_log_probs = old_log_probs[mb_indices]

                # V-trace advantages (already importance-weighted via clipped_rhos)
                mb_advantages = advantages[mb_indices]

                # Normalize advantages within minibatch for stability
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Compute importance ratio for PPO clipping
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio.clamp(-20, 20))

                # PPO-style clipped objective on top of V-trace
                # V-trace corrects the training objective, PPO clipping guards against destructive updates
                # Research: Sample Factory found this combination works well
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Track metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    # Approximate KL divergence for early stopping
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    total_approx_kl += approx_kl

                # KL early stopping - prevent destructive updates
                if approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

                # Value loss: MSE to V-trace targets (also clipped for stability)
                mb_vtrace_targets = vtrace_targets[mb_indices]
                value_loss = 0.5 * ((new_values - mb_vtrace_targets.clamp(-100, 100)) ** 2).mean()

                # Entropy targeting - pull toward target, penalize deviation in either direction
                entropy_loss = (entropy.mean() - self.target_entropy) ** 2

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Skip if loss is NaN, infinite, or too extreme (prevents catastrophic updates)
                if torch.isnan(loss) or torch.isinf(loss) or abs(loss.item()) > 5.0:
                    if abs(loss.item()) > 5.0:
                        print(f"  Warning: Extreme loss {loss.item():.2f}, skipping minibatch")
                    else:
                        print(f"  Warning: NaN/inf loss detected, skipping minibatch")
                    continue

                # Update
                self.optimizer.zero_grad()
                loss.backward()

                # Check for NaN gradients before stepping
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"  Warning: NaN/inf gradient norm, skipping update")
                    continue

                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_frac += clip_frac
                num_updates += 1

        return {
            'loss': total_loss / max(num_updates, 1),
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'avg_rho': avg_rho,  # Monitor importance ratio - should be close to 1.0
            'clip_frac': total_clip_frac / max(num_updates, 1),  # Fraction of ratios clipped
            'approx_kl': total_approx_kl / max(num_updates, 1),  # KL divergence for monitoring
            'early_stop': early_stop,  # Whether training stopped early due to KL
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
    trainer = VTraceTrainer(model, lr=args.lr, minibatch_size=64)

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
                  f"rho={stats['avg_rho']:.3f} "
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
