"""
PPO with Hybrid Action Space (Multi-Discrete + Continuous)

Based on CleanRL's single-file implementation approach.
Supports combined discrete (direction, jump, hook) and continuous (aim) actions.

Reference: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    # Training
    learning_rate: float = 3e-4
    num_steps: int = 256          # Steps per rollout per env
    num_minibatches: int = 4
    update_epochs: int = 4

    # PPO specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # Early stopping based on KL

    # Network
    hidden_dim: int = 256
    num_layers: int = 2

    # Action space
    discrete_dims: Tuple[int, ...] = (3, 2, 2)  # direction, jump, hook
    continuous_dim: int = 2  # aim_angle, aim_magnitude


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class HybridActor(nn.Module):
    """
    Actor network for hybrid action space.

    Outputs:
        - Logits for each discrete action dimension
        - Mean for continuous actions
        - Log std for continuous actions (learned parameter)
    """

    def __init__(self, obs_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(config.num_layers):
            layers.append(layer_init(nn.Linear(in_dim, config.hidden_dim)))
            layers.append(nn.Tanh())
            in_dim = config.hidden_dim
        self.shared = nn.Sequential(*layers)

        # Discrete action heads
        self.discrete_heads = nn.ModuleList([
            layer_init(nn.Linear(config.hidden_dim, dim), std=0.01)
            for dim in config.discrete_dims
        ])

        # Continuous action head
        self.continuous_mean = layer_init(
            nn.Linear(config.hidden_dim, config.continuous_dim), std=0.01
        )
        self.continuous_log_std = nn.Parameter(
            torch.zeros(config.continuous_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            discrete_logits: List of logits for each discrete dimension
            continuous_mean: Mean of continuous action distribution
            continuous_std: Std of continuous action distribution
        """
        shared_out = self.shared(x)

        discrete_logits = [head(shared_out) for head in self.discrete_heads]
        continuous_mean = self.continuous_mean(shared_out)
        continuous_std = self.continuous_log_std.exp().expand_as(continuous_mean)

        return discrete_logits, continuous_mean, continuous_std

    def get_action(
        self,
        x: torch.Tensor,
        action: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Get action and compute log probability and entropy.

        Args:
            x: Observation tensor
            action: Optional action dict (for computing log prob of given action)
            deterministic: If True, return mode instead of sampling

        Returns:
            action: Dict with 'discrete' and 'continuous' tensors
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
        """
        discrete_logits, cont_mean, cont_std = self.forward(x)

        # Create distributions
        discrete_dists = [Categorical(logits=logits) for logits in discrete_logits]
        cont_dist = Normal(cont_mean, cont_std)

        if action is None:
            # Sample new action
            if deterministic:
                discrete_actions = [torch.argmax(logits, dim=-1) for logits in discrete_logits]
                continuous_action = cont_mean
            else:
                discrete_actions = [d.sample() for d in discrete_dists]
                continuous_action = cont_dist.sample()
        else:
            # Use provided action
            discrete_actions = [action['discrete'][:, i] for i in range(len(discrete_dists))]
            continuous_action = action['continuous']

        # Compute log probabilities
        discrete_log_probs = torch.stack([
            d.log_prob(a) for d, a in zip(discrete_dists, discrete_actions)
        ], dim=-1).sum(dim=-1)

        continuous_log_prob = cont_dist.log_prob(continuous_action).sum(dim=-1)

        total_log_prob = discrete_log_probs + continuous_log_prob

        # Compute entropy
        discrete_entropy = torch.stack([d.entropy() for d in discrete_dists], dim=-1).sum(dim=-1)
        continuous_entropy = cont_dist.entropy().sum(dim=-1)
        total_entropy = discrete_entropy + continuous_entropy

        # Stack discrete actions
        action_out = {
            'discrete': torch.stack(discrete_actions, dim=-1),
            'continuous': continuous_action,
        }

        return action_out, total_log_prob, total_entropy


class Critic(nn.Module):
    """Value function network."""

    def __init__(self, obs_dim: int, config: PPOConfig):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for _ in range(config.num_layers):
            layers.append(layer_init(nn.Linear(in_dim, config.hidden_dim)))
            layers.append(nn.Tanh())
            in_dim = config.hidden_dim
        layers.append(layer_init(nn.Linear(config.hidden_dim, 1), std=1.0))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PPOAgent:
    """
    PPO Agent with hybrid action space.

    Handles rollout storage, advantage computation, and policy updates.
    """

    def __init__(
        self,
        obs_dim: int,
        num_envs: int,
        config: Optional[PPOConfig] = None,
        device: str = "cpu"
    ):
        self.config = config or PPOConfig()
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = torch.device(device)

        # Create networks
        self.actor = HybridActor(obs_dim, self.config).to(self.device)
        self.critic = Critic(obs_dim, self.config).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
            eps=1e-5
        )

        # Rollout storage
        self.num_steps = self.config.num_steps
        self._init_storage()

        # Training stats
        self.global_step = 0

    def _init_storage(self):
        """Initialize rollout storage buffers."""
        ns, ne = self.num_steps, self.num_envs

        self.obs_buffer = torch.zeros((ns, ne, self.obs_dim), device=self.device)
        self.discrete_actions = torch.zeros(
            (ns, ne, len(self.config.discrete_dims)), dtype=torch.long, device=self.device
        )
        self.continuous_actions = torch.zeros(
            (ns, ne, self.config.continuous_dim), device=self.device
        )
        self.log_probs = torch.zeros((ns, ne), device=self.device)
        self.rewards = torch.zeros((ns, ne), device=self.device)
        self.dones = torch.zeros((ns, ne), device=self.device)
        self.values = torch.zeros((ns, ne), device=self.device)

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Get action for given observation.

        Returns:
            action: Dict with 'discrete' and 'continuous' numpy arrays
            log_prob: Log probability numpy array
            value: Value estimate numpy array
        """
        obs_t = torch.from_numpy(obs).float().to(self.device)

        action, log_prob, _ = self.actor.get_action(obs_t, deterministic=deterministic)
        value = self.critic(obs_t)

        return (
            {k: v.cpu().numpy() for k, v in action.items()},
            log_prob.cpu().numpy(),
            value.squeeze(-1).cpu().numpy()
        )

    def store_transition(
        self,
        step: int,
        obs: np.ndarray,
        action: Dict[str, np.ndarray],
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray
    ):
        """Store a transition in the rollout buffer."""
        self.obs_buffer[step] = torch.from_numpy(obs).to(self.device)
        self.discrete_actions[step] = torch.from_numpy(action['discrete']).to(self.device)
        self.continuous_actions[step] = torch.from_numpy(action['continuous']).to(self.device)
        self.log_probs[step] = torch.from_numpy(log_prob).to(self.device)
        self.rewards[step] = torch.from_numpy(reward).to(self.device)
        self.dones[step] = torch.from_numpy(done.astype(np.float32)).to(self.device)
        self.values[step] = torch.from_numpy(value).to(self.device)

    @torch.no_grad()
    def compute_advantages(self, next_obs: np.ndarray, next_done: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Returns:
            advantages: Tensor of shape (num_steps, num_envs)
            returns: Tensor of shape (num_steps, num_envs)
        """
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        next_done_t = torch.from_numpy(next_done.astype(np.float32)).to(self.device)

        next_value = self.critic(next_obs_t).squeeze(-1)

        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done_t
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]

            delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + self.values

        return advantages, returns

    def update(self, next_obs: np.ndarray, next_done: np.ndarray) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout.

        Returns:
            Dictionary of training statistics
        """
        # Compute advantages
        advantages, returns = self.compute_advantages(next_obs, next_done)

        # Flatten batch
        b_obs = self.obs_buffer.reshape(-1, self.obs_dim)
        b_discrete = self.discrete_actions.reshape(-1, len(self.config.discrete_dims))
        b_continuous = self.continuous_actions.reshape(-1, self.config.continuous_dim)
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // self.config.num_minibatches

        # Training stats
        clipfracs = []
        all_pg_losses = []
        all_v_losses = []
        all_entropy_losses = []
        all_approx_kl = []

        for epoch in range(self.config.update_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_obs = b_obs[mb_indices]
                mb_action = {
                    'discrete': b_discrete[mb_indices],
                    'continuous': b_continuous[mb_indices],
                }
                mb_log_probs = b_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_values = b_values[mb_indices]

                # Forward pass
                _, new_log_prob, entropy = self.actor.get_action(mb_obs, mb_action)
                new_value = self.critic(mb_obs).squeeze(-1)

                # Policy loss
                logratio = new_log_prob - mb_log_probs
                ratio = logratio.exp()

                # Approximate KL for early stopping
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item())
                    all_approx_kl.append(approx_kl.item())

                # Clipped objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if self.config.clip_vloss:
                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        new_value - mb_values,
                        -self.config.clip_coef,
                        self.config.clip_coef
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * v_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                all_pg_losses.append(pg_loss.item())
                all_v_losses.append(v_loss.item())
                all_entropy_losses.append(entropy_loss.item())

            # Early stopping based on KL
            if self.config.target_kl is not None:
                if approx_kl > self.config.target_kl:
                    break

        # Return stats
        return {
            'policy_loss': np.mean(all_pg_losses),
            'value_loss': np.mean(all_v_losses),
            'entropy': np.mean(all_entropy_losses),
            'approx_kl': np.mean(all_approx_kl),
            'clipfrac': np.mean(clipfracs),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)


if __name__ == "__main__":
    # Quick test
    obs_dim = 137
    num_envs = 4
    num_steps = 64

    config = PPOConfig(num_steps=num_steps)
    agent = PPOAgent(obs_dim, num_envs, config)

    # Simulate rollout
    obs = np.random.randn(num_envs, obs_dim).astype(np.float32)

    for step in range(num_steps):
        action, log_prob, value = agent.get_action(obs)

        # Fake environment step
        next_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        reward = np.random.randn(num_envs).astype(np.float32)
        done = np.zeros(num_envs, dtype=bool)

        agent.store_transition(step, obs, action, log_prob, reward, done, value)
        obs = next_obs

    # Update
    stats = agent.update(obs, np.zeros(num_envs, dtype=bool))
    print("Update stats:", stats)
    print("Test passed!")
