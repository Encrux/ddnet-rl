"""
PPO with Hybrid Action Space (Multi-Discrete + Continuous)

Based on CleanRL's single-file implementation approach.
Supports combined discrete (direction, jump, hook) and continuous (aim) actions.
Includes CNN feature extractor for processing tile grids.

Reference: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass, field


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

    # CNN Feature Extractor (for dict observations)
    use_cnn: bool = False
    tile_grid_height: int = 31   # 1.5x original (21 -> 31)
    tile_grid_width: int = 43    # 1.5x original (29 -> 43)
    player_state_dim: int = 15   # Player state features
    checkpoint_dim: int = 4      # Checkpoint navigation features
    cnn_channels: Tuple[int, ...] = (32, 64, 64)  # CNN channel progression

    # Embedding Feature Extractor (for raw tile IDs)
    use_embeddings: bool = False     # Use tile embeddings instead of normalized values
    tile_embed_dim: int = 8          # Embedding dimension per tile
    tile_encoder: str = "mlp"        # "mlp" or "cnn" for processing embedded tiles

    # Mixed precision training (2-3x speedup on modern GPUs)
    use_amp: bool = True

    # torch.compile for faster forward/backward passes (PyTorch 2.0+)
    use_compile: bool = True

    # Frame stacking for temporal context
    # With frame_skip=3: 8 frames × 3 ticks × 20ms = 480ms coverage
    n_frame_stack: int = 1  # Number of tile frames to stack (1 = no stacking, expensive)
    n_player_state_stack: int = 8  # Number of player state frames to stack (cheap, enables timing)


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for tile grid observations.

    Processes the 2D tile grid through convolutional layers, then combines
    with player state and checkpoint information.

    Input:
        tiles: (batch, 1, H, W) - Single channel tile grid
        player_state: (batch, player_state_dim) - Player features
        checkpoint: (batch, checkpoint_dim) - Checkpoint navigation

    Output:
        features: (batch, output_dim) - Combined feature vector
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config

        h, w = config.tile_grid_height, config.tile_grid_width
        channels = config.cnn_channels
        n_stack = config.n_frame_stack

        # CNN for tile grid
        # Input: (n_stack, 31, 43), output after convs: depends on architecture
        # When frame stacking is enabled, tiles have n_stack channels instead of 1
        self.tile_encoder = nn.Sequential(
            # Conv1: 31x43 -> 15x21 (stride 2 with padding)
            layer_init(nn.Conv2d(n_stack, channels[0], kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            # Conv2: 15x21 -> 7x10 (stride 2)
            layer_init(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            # Conv3: 7x10 -> 3x5 (stride 2)
            layer_init(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        # After 3 stride-2 convs: H/8 x W/8 approximately
        # 31/8 ~ 4, 43/8 ~ 6 -> actual is (4, 6) with padding adjustments
        # Let's compute exactly:
        def conv_out_size(size, kernel=3, stride=2, padding=1):
            return (size + 2 * padding - kernel) // stride + 1

        h1 = conv_out_size(h)  # 31 -> 15
        w1 = conv_out_size(w)  # 43 -> 21
        h2 = conv_out_size(h1) # 15 -> 8
        w2 = conv_out_size(w1) # 21 -> 11
        h3 = conv_out_size(h2) # 8 -> 4
        w3 = conv_out_size(w2) # 11 -> 6

        cnn_flat_size = channels[2] * h3 * w3

        # Combine CNN features with player state and checkpoint
        # Tiles use n_frame_stack, player_state uses n_player_state_stack (can differ)
        # Checkpoint stays single (4 features)
        player_state_size = config.player_state_dim * config.n_player_state_stack
        combined_input_size = cnn_flat_size + player_state_size + config.checkpoint_dim

        self.combine = nn.Sequential(
            layer_init(nn.Linear(combined_input_size, config.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(config.hidden_dim, config.hidden_dim)),
            nn.Tanh(),
        )

        self.output_dim = config.hidden_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through CNN and combination layers.

        Args:
            obs: Dict with keys 'tiles', 'player_state', 'checkpoint'

        Returns:
            Combined feature tensor of shape (batch, hidden_dim)
        """
        tiles = obs['tiles']  # (batch, 1, H, W)
        player_state = obs['player_state']  # (batch, player_dim)
        checkpoint = obs['checkpoint']  # (batch, cp_dim)

        # Process tiles through CNN
        tile_features = self.tile_encoder(tiles)

        # Concatenate all features
        combined = torch.cat([tile_features, player_state, checkpoint], dim=-1)

        # Process through combination layers
        return self.combine(combined)


class EmbeddingFeatureExtractor(nn.Module):
    """
    Feature extractor using learned tile embeddings.

    Uses nn.Embedding(256, embed_dim) to convert raw tile IDs to dense vectors,
    then processes with either MLP or CNN encoder.

    Input:
        tiles: (batch, H, W) - Raw tile IDs (int64, values 0-255)
        player_state: (batch, player_state_dim) - Player features
        checkpoint: (batch, checkpoint_dim) - Checkpoint navigation

    Output:
        features: (batch, output_dim) - Combined feature vector
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config

        h, w = config.tile_grid_height, config.tile_grid_width
        embed_dim = config.tile_embed_dim
        tile_encoder = config.tile_encoder

        # Learned tile embeddings: 256 possible tile IDs -> embed_dim
        self.tile_embedding = nn.Embedding(256, embed_dim)

        if tile_encoder == "mlp":
            # MLP: Flatten embedded tiles and project
            # (batch, H, W, embed_dim) -> (batch, H*W*embed_dim) -> (batch, hidden_dim)
            flat_size = h * w * embed_dim  # 31 * 43 * 8 = 10664
            self.tile_encoder = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(flat_size, config.hidden_dim)),
                nn.ReLU(),
            )
            tile_out_dim = config.hidden_dim
        else:  # cnn
            # CNN: Process embedded tiles as channels
            # (batch, H, W, embed_dim) -> permute -> (batch, embed_dim, H, W)
            # Then apply conv layers
            self.tile_encoder = nn.Sequential(
                # Conv1: (embed_dim, 31, 43) -> (16, 16, 22)
                layer_init(nn.Conv2d(embed_dim, 16, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                # Conv2: (16, 16, 22) -> (32, 8, 11)
                layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                # Conv3: (32, 8, 11) -> (64, 4, 6)
                layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Calculate CNN output size
            def conv_out_size(size, kernel=3, stride=2, padding=1):
                return (size + 2 * padding - kernel) // stride + 1
            h1 = conv_out_size(h)  # 31 -> 16
            w1 = conv_out_size(w)  # 43 -> 22
            h2 = conv_out_size(h1) # 16 -> 8
            w2 = conv_out_size(w1) # 22 -> 11
            h3 = conv_out_size(h2) # 8 -> 4
            w3 = conv_out_size(w2) # 11 -> 6
            tile_out_dim = 64 * h3 * w3  # 64 * 4 * 6 = 1536

        self.use_cnn_encoder = (tile_encoder == "cnn")

        # Combine tile features with player_state (15 * n_player_state_stack) + checkpoint(4)
        player_state_size = config.player_state_dim * config.n_player_state_stack
        combined_input_size = tile_out_dim + player_state_size + config.checkpoint_dim

        self.combine = nn.Sequential(
            layer_init(nn.Linear(combined_input_size, config.hidden_dim)),
            nn.Tanh(),
        )

        self.output_dim = config.hidden_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through embeddings and combination layers.

        Args:
            obs: Dict with keys 'tiles' (int64), 'player_state', 'checkpoint'

        Returns:
            Combined feature tensor of shape (batch, hidden_dim)
        """
        tiles = obs['tiles']  # (batch, H, W) int64
        player_state = obs['player_state']  # (batch, player_dim)
        checkpoint = obs['checkpoint']  # (batch, cp_dim)

        # Embed tiles: (batch, H, W) -> (batch, H, W, embed_dim)
        tile_embeds = self.tile_embedding(tiles)

        if self.use_cnn_encoder:
            # Permute for CNN: (batch, H, W, embed_dim) -> (batch, embed_dim, H, W)
            tile_embeds = tile_embeds.permute(0, 3, 1, 2)

        # Process tiles through encoder
        tile_features = self.tile_encoder(tile_embeds)

        # Concatenate all features
        combined = torch.cat([tile_features, player_state, checkpoint], dim=-1)

        # Process through combination layers
        return self.combine(combined)


class FlatFeatureExtractor(nn.Module):
    """
    Simple MLP feature extractor for flat observation vectors.

    Used when use_cnn=False in config.
    """

    def __init__(self, obs_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = obs_dim
        for _ in range(config.num_layers):
            layers.append(layer_init(nn.Linear(in_dim, config.hidden_dim)))
            layers.append(nn.Tanh())
            in_dim = config.hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = config.hidden_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class HybridActor(nn.Module):
    """
    Actor network for hybrid action space.

    Supports flat observations (MLP), dict observations (CNN), and embeddings.

    Outputs:
        - Logits for each discrete action dimension
        - Mean for continuous actions
        - Log std for continuous actions (learned parameter)
    """

    def __init__(self, obs_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config
        self.use_cnn = config.use_cnn
        self.use_embeddings = config.use_embeddings

        # Feature extractor (Embedding, CNN, or MLP)
        if config.use_embeddings:
            self.feature_extractor = EmbeddingFeatureExtractor(config)
            feature_dim = self.feature_extractor.output_dim
        elif config.use_cnn:
            self.feature_extractor = CNNFeatureExtractor(config)
            feature_dim = self.feature_extractor.output_dim
        else:
            self.feature_extractor = FlatFeatureExtractor(obs_dim, config)
            feature_dim = self.feature_extractor.output_dim

        # Discrete action heads
        self.discrete_heads = nn.ModuleList([
            layer_init(nn.Linear(feature_dim, dim), std=0.01)
            for dim in config.discrete_dims
        ])

        # Continuous action head
        self.continuous_mean = layer_init(
            nn.Linear(feature_dim, config.continuous_dim), std=0.01
        )
        self.continuous_log_std = nn.Parameter(
            torch.zeros(config.continuous_dim)
        )

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Either flat tensor (batch, obs_dim) or dict with 'tiles', 'player_state', 'checkpoint'

        Returns:
            discrete_logits: List of logits for each discrete dimension
            continuous_mean: Mean of continuous action distribution
            continuous_std: Std of continuous action distribution
        """
        features = self.feature_extractor(x)

        discrete_logits = [head(features) for head in self.discrete_heads]
        continuous_mean = self.continuous_mean(features)
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

        # Create distributions (cast to float32, disable validation for speed)
        # NaN can occur with AMP, so we clamp to prevent crashes
        discrete_dists = [
            Categorical(logits=torch.clamp(logits.float(), -50, 50), validate_args=False)
            for logits in discrete_logits
        ]
        cont_dist = Normal(cont_mean.float(), cont_std.float().clamp(min=1e-6), validate_args=False)

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
    """Value function network. Supports flat, CNN, and embedding observations."""

    def __init__(self, obs_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config
        self.use_cnn = config.use_cnn
        self.use_embeddings = config.use_embeddings

        # Feature extractor (Embedding, CNN, or MLP)
        if config.use_embeddings:
            self.feature_extractor = EmbeddingFeatureExtractor(config)
            feature_dim = self.feature_extractor.output_dim
        elif config.use_cnn:
            self.feature_extractor = CNNFeatureExtractor(config)
            feature_dim = self.feature_extractor.output_dim
        else:
            self.feature_extractor = FlatFeatureExtractor(obs_dim, config)
            feature_dim = self.feature_extractor.output_dim

        # Value head
        self.value_head = layer_init(nn.Linear(feature_dim, 1), std=1.0)

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.value_head(features)


class PPOAgent:
    """
    PPO Agent with hybrid action space.

    Handles rollout storage, advantage computation, and policy updates.
    Supports both flat observations and dict observations (for CNN).
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
        self.use_cnn = self.config.use_cnn
        self.use_embeddings = self.config.use_embeddings

        # Create networks
        self.actor = HybridActor(obs_dim, self.config).to(self.device)
        self.critic = Critic(obs_dim, self.config).to(self.device)

        # Compile models for faster execution (PyTorch 2.0+)
        # Use "default" mode - reduce-overhead (CUDA graphs) causes recompilation
        # when batch sizes change between inference and PPO update
        if self.config.use_compile and hasattr(torch, 'compile'):
            try:
                self.actor = torch.compile(self.actor, mode="default")
                self.critic = torch.compile(self.critic, mode="default")
                print("Models compiled with torch.compile (default mode)")
            except Exception as e:
                print(f"torch.compile failed, using eager mode: {e}")

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
            eps=1e-5
        )

        # Mixed precision training
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Rollout storage
        self.num_steps = self.config.num_steps
        self._use_async_transfers = False  # Will be set by _init_async_transfer_buffers
        self._init_storage()

        # Training stats
        self.global_step = 0

    def _init_storage(self):
        """Initialize rollout storage buffers."""
        ns, ne = self.num_steps, self.num_envs
        cfg = self.config
        n_tile_stack = cfg.n_frame_stack
        n_player_stack = cfg.n_player_state_stack

        if self.use_embeddings:
            # Embedding observation storage (tiles as int64 for nn.Embedding)
            self.obs_tiles = torch.zeros(
                (ns, ne, cfg.tile_grid_height, cfg.tile_grid_width), dtype=torch.long, device=self.device
            )
            self.obs_player_state = torch.zeros(
                (ns, ne, cfg.player_state_dim * n_player_stack), device=self.device
            )
            self.obs_checkpoint = torch.zeros(
                (ns, ne, cfg.checkpoint_dim), device=self.device
            )
        elif self.use_cnn:
            # Dict observation storage (tiles as float32)
            # Tiles use n_frame_stack channels: (ns, ne, n_tile_stack, H, W)
            # Player state uses n_player_state_stack: (ns, ne, player_state_dim * n_player_stack)
            # Checkpoint stays single: (ns, ne, checkpoint_dim)
            self.obs_tiles = torch.zeros(
                (ns, ne, n_tile_stack, cfg.tile_grid_height, cfg.tile_grid_width), device=self.device
            )
            self.obs_player_state = torch.zeros(
                (ns, ne, cfg.player_state_dim * n_player_stack), device=self.device
            )
            self.obs_checkpoint = torch.zeros(
                (ns, ne, cfg.checkpoint_dim), device=self.device
            )
        else:
            # Flat observation storage
            self.obs_buffer = torch.zeros((ns, ne, self.obs_dim), device=self.device)

        self.discrete_actions = torch.zeros(
            (ns, ne, len(cfg.discrete_dims)), dtype=torch.long, device=self.device
        )
        self.continuous_actions = torch.zeros(
            (ns, ne, cfg.continuous_dim), device=self.device
        )
        self.log_probs = torch.zeros((ns, ne), device=self.device)
        self.rewards = torch.zeros((ns, ne), device=self.device)
        self.dones = torch.zeros((ns, ne), device=self.device)
        self.values = torch.zeros((ns, ne), device=self.device)

        # Initialize async transfer buffers for CUDA
        self._init_async_transfer_buffers()

    def _init_async_transfer_buffers(self):
        """Initialize pinned memory buffers for async GPU transfers."""
        if self.device.type != "cuda":
            self._use_async_transfers = False
            return

        self._use_async_transfers = True
        ne = self.num_envs
        cfg = self.config
        n_tile_stack = cfg.n_frame_stack
        n_player_stack = cfg.n_player_state_stack

        # Create CUDA streams for async transfers
        self._transfer_stream = torch.cuda.Stream()

        if self.use_embeddings:
            # Embedding mode: tiles as long
            self._pinned_tiles = torch.empty(
                (ne, cfg.tile_grid_height, cfg.tile_grid_width),
                dtype=torch.long, pin_memory=True
            )
            self._gpu_tiles = torch.empty(
                (ne, cfg.tile_grid_height, cfg.tile_grid_width),
                dtype=torch.long, device=self.device
            )
        elif self.use_cnn:
            # CNN mode: tiles as float with frame stacking
            self._pinned_tiles = torch.empty(
                (ne, n_tile_stack, cfg.tile_grid_height, cfg.tile_grid_width),
                dtype=torch.float32, pin_memory=True
            )
            self._gpu_tiles = torch.empty(
                (ne, n_tile_stack, cfg.tile_grid_height, cfg.tile_grid_width),
                dtype=torch.float32, device=self.device
            )
        else:
            # Flat mode
            self._pinned_obs = torch.empty(
                (ne, self.obs_dim), dtype=torch.float32, pin_memory=True
            )
            self._gpu_obs = torch.empty(
                (ne, self.obs_dim), dtype=torch.float32, device=self.device
            )

        # Player state and checkpoint buffers (for CNN/embedding modes)
        # Note: These match the raw observation dimensions, NOT the stacked dimensions
        # Frame stacking is handled by FrameStackWrapper before calling get_action
        if self.use_cnn or self.use_embeddings:
            # For inference: use stacked dimensions (what get_action receives)
            self._pinned_player = torch.empty(
                (ne, cfg.player_state_dim * n_player_stack),
                dtype=torch.float32, pin_memory=True
            )
            self._pinned_checkpoint = torch.empty(
                (ne, cfg.checkpoint_dim), dtype=torch.float32, pin_memory=True
            )
            self._gpu_player = torch.empty(
                (ne, cfg.player_state_dim * n_player_stack),
                dtype=torch.float32, device=self.device
            )
            self._gpu_checkpoint = torch.empty(
                (ne, cfg.checkpoint_dim), dtype=torch.float32, device=self.device
            )

        # Action output buffer (for D2H transfer)
        total_discrete = len(cfg.discrete_dims)
        self._pinned_discrete = torch.empty(
            (ne, total_discrete), dtype=torch.long, pin_memory=True
        )
        self._pinned_continuous = torch.empty(
            (ne, cfg.continuous_dim), dtype=torch.float32, pin_memory=True
        )

    def _obs_to_tensor_async(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert numpy observation to tensor using async pinned memory transfers."""
        if not self._use_async_transfers:
            return self._obs_to_tensor(obs)

        if isinstance(obs, dict):
            # Copy to pinned memory (fast CPU memcpy)
            if self.use_embeddings:
                self._pinned_tiles.copy_(torch.from_numpy(obs['tiles']))
            else:
                self._pinned_tiles.copy_(torch.from_numpy(obs['tiles']))
            self._pinned_player.copy_(torch.from_numpy(obs['player_state']))
            self._pinned_checkpoint.copy_(torch.from_numpy(obs['checkpoint']))

            # Async H2D transfer on separate stream
            with torch.cuda.stream(self._transfer_stream):
                self._gpu_tiles.copy_(self._pinned_tiles, non_blocking=True)
                self._gpu_player.copy_(self._pinned_player, non_blocking=True)
                self._gpu_checkpoint.copy_(self._pinned_checkpoint, non_blocking=True)

            # Sync default stream with transfer stream
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

            return {
                'tiles': self._gpu_tiles,
                'player_state': self._gpu_player,
                'checkpoint': self._gpu_checkpoint,
            }
        else:
            # Flat observation
            self._pinned_obs.copy_(torch.from_numpy(obs))

            with torch.cuda.stream(self._transfer_stream):
                self._gpu_obs.copy_(self._pinned_obs, non_blocking=True)

            torch.cuda.current_stream().wait_stream(self._transfer_stream)

            return self._gpu_obs

    def _obs_to_tensor(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert numpy observation to tensor(s)."""
        # Auto-detect observation type (dict vs flat array)
        if isinstance(obs, dict):
            if self.use_embeddings:
                # Embedding mode: tiles stay as int64/long for nn.Embedding
                return {
                    'tiles': torch.from_numpy(obs['tiles']).long().to(self.device),
                    'player_state': torch.from_numpy(obs['player_state']).float().to(self.device),
                    'checkpoint': torch.from_numpy(obs['checkpoint']).float().to(self.device),
                }
            else:
                # CNN mode: tiles are float32
                return {
                    'tiles': torch.from_numpy(obs['tiles']).float().to(self.device),
                    'player_state': torch.from_numpy(obs['player_state']).float().to(self.device),
                    'checkpoint': torch.from_numpy(obs['checkpoint']).float().to(self.device),
                }
        else:
            return torch.from_numpy(obs).float().to(self.device)

    def warmup(self, obs: Union[np.ndarray, Dict[str, np.ndarray]], num_iters: int = 10):
        """
        Warmup torch.compile by running inference passes.
        This triggers CUDA graph capture for reduce-overhead mode.
        Call this before training with a representative observation batch.
        """
        print(f"Warming up torch.compile ({num_iters} iterations)...")
        import time
        start = time.time()
        for i in range(num_iters):
            _ = self.get_action(obs)
        elapsed = time.time() - start
        print(f"Warmup complete in {elapsed:.1f}s (avg {elapsed/num_iters*1000:.0f}ms/iter)")

    @torch.no_grad()
    def get_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Get action for given observation.

        Uses async pinned memory transfers on CUDA for reduced latency.

        Args:
            obs: Either flat numpy array or dict with 'tiles', 'player_state', 'checkpoint'
            deterministic: If True, use mode instead of sampling

        Returns:
            action: Dict with 'discrete' and 'continuous' numpy arrays
            log_prob: Log probability numpy array
            value: Value estimate numpy array
        """
        # Use async transfers on CUDA, sync transfers on CPU
        if self._use_async_transfers:
            obs_t = self._obs_to_tensor_async(obs)
        else:
            obs_t = self._obs_to_tensor(obs)

        action, log_prob, _ = self.actor.get_action(obs_t, deterministic=deterministic)
        value = self.critic(obs_t)

        # Use async D2H transfer for actions on CUDA
        if self._use_async_transfers:
            with torch.cuda.stream(self._transfer_stream):
                self._transfer_stream.wait_stream(torch.cuda.current_stream())
                self._pinned_discrete.copy_(action['discrete'], non_blocking=True)
                self._pinned_continuous.copy_(action['continuous'], non_blocking=True)

            # Sync to ensure action data is ready
            self._transfer_stream.synchronize()

            return (
                {
                    'discrete': self._pinned_discrete.numpy().copy(),
                    'continuous': self._pinned_continuous.numpy().copy()
                },
                log_prob.cpu().numpy(),
                value.squeeze(-1).cpu().numpy()
            )
        else:
            return (
                {k: v.cpu().numpy() for k, v in action.items()},
                log_prob.cpu().numpy(),
                value.squeeze(-1).cpu().numpy()
            )

    @torch.no_grad()
    def get_action_profiled(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
        cuda_sync: bool = True
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Get action with detailed timing breakdown.

        Args:
            obs: Either flat numpy array or dict with 'tiles', 'player_state', 'checkpoint'
            deterministic: If True, use mode instead of sampling
            cuda_sync: If True, synchronize CUDA for accurate timing

        Returns:
            action: Dict with 'discrete' and 'continuous' numpy arrays
            log_prob: Log probability numpy array
            value: Value estimate numpy array
            timings: Dict with timing breakdown in milliseconds
        """
        import time
        timings = {}

        # Numpy -> Tensor conversion
        t0 = time.perf_counter()
        obs_t = self._obs_to_tensor(obs)
        if cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        timings["obs_to_tensor_ms"] = (time.perf_counter() - t0) * 1000

        # Actor forward
        t0 = time.perf_counter()
        action, log_prob, _ = self.actor.get_action(obs_t, deterministic=deterministic)
        if cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        timings["actor_forward_ms"] = (time.perf_counter() - t0) * 1000

        # Critic forward
        t0 = time.perf_counter()
        value = self.critic(obs_t)
        if cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        timings["critic_forward_ms"] = (time.perf_counter() - t0) * 1000

        # Tensor -> Numpy conversion
        t0 = time.perf_counter()
        action_np = {k: v.cpu().numpy() for k, v in action.items()}
        log_prob_np = log_prob.cpu().numpy()
        value_np = value.squeeze(-1).cpu().numpy()
        timings["tensor_to_numpy_ms"] = (time.perf_counter() - t0) * 1000

        # Total
        timings["total_ms"] = sum(timings.values())

        return action_np, log_prob_np, value_np, timings

    def store_transition(
        self,
        step: int,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        action: Dict[str, np.ndarray],
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray
    ):
        """Store a transition in the rollout buffer."""
        if self.use_embeddings:
            # Embedding mode: tiles as long for nn.Embedding
            self.obs_tiles[step] = torch.from_numpy(obs['tiles']).long().to(self.device)
            self.obs_player_state[step] = torch.from_numpy(obs['player_state']).to(self.device)
            self.obs_checkpoint[step] = torch.from_numpy(obs['checkpoint']).to(self.device)
        elif self.use_cnn:
            self.obs_tiles[step] = torch.from_numpy(obs['tiles']).to(self.device)
            self.obs_player_state[step] = torch.from_numpy(obs['player_state']).to(self.device)
            self.obs_checkpoint[step] = torch.from_numpy(obs['checkpoint']).to(self.device)
        else:
            self.obs_buffer[step] = torch.from_numpy(obs).to(self.device)

        self.discrete_actions[step] = torch.from_numpy(action['discrete']).to(self.device)
        self.continuous_actions[step] = torch.from_numpy(action['continuous']).to(self.device)
        self.log_probs[step] = torch.from_numpy(log_prob).to(self.device)
        self.rewards[step] = torch.from_numpy(reward).to(self.device)
        self.dones[step] = torch.from_numpy(done.astype(np.float32)).to(self.device)
        self.values[step] = torch.from_numpy(value).to(self.device)

    @torch.no_grad()
    def compute_advantages(
        self,
        next_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        next_done: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Returns:
            advantages: Tensor of shape (num_steps, num_envs)
            returns: Tensor of shape (num_steps, num_envs)
        """
        next_obs_t = self._obs_to_tensor(next_obs)
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

    def update(
        self,
        next_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        next_done: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout.

        Returns:
            Dictionary of training statistics
        """
        # Compute advantages
        advantages, returns = self.compute_advantages(next_obs, next_done)

        # Flatten batch
        batch_size = self.num_steps * self.num_envs
        cfg = self.config

        n_tile_stack = cfg.n_frame_stack
        n_player_stack = cfg.n_player_state_stack

        if self.use_embeddings:
            # Embedding mode: tiles as (batch, H, W) long tensor
            b_tiles = self.obs_tiles.reshape(-1, cfg.tile_grid_height, cfg.tile_grid_width)
            b_player_state = self.obs_player_state.reshape(-1, cfg.player_state_dim * n_player_stack)
            b_checkpoint = self.obs_checkpoint.reshape(-1, cfg.checkpoint_dim)
        elif self.use_cnn:
            # CNN mode: tiles use n_frame_stack, player_state uses n_player_state_stack
            b_tiles = self.obs_tiles.reshape(-1, n_tile_stack, cfg.tile_grid_height, cfg.tile_grid_width)
            b_player_state = self.obs_player_state.reshape(-1, cfg.player_state_dim * n_player_stack)
            b_checkpoint = self.obs_checkpoint.reshape(-1, cfg.checkpoint_dim)
        else:
            b_obs = self.obs_buffer.reshape(-1, self.obs_dim)

        b_discrete = self.discrete_actions.reshape(-1, len(cfg.discrete_dims))
        b_continuous = self.continuous_actions.reshape(-1, cfg.continuous_dim)
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        minibatch_size = batch_size // cfg.num_minibatches

        # Training stats
        clipfracs = []
        all_pg_losses = []
        all_v_losses = []
        all_entropy_losses = []
        all_approx_kl = []

        for epoch in range(cfg.update_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch observation
                if self.use_embeddings or self.use_cnn:
                    mb_obs = {
                        'tiles': b_tiles[mb_indices],
                        'player_state': b_player_state[mb_indices],
                        'checkpoint': b_checkpoint[mb_indices],
                    }
                else:
                    mb_obs = b_obs[mb_indices]

                mb_action = {
                    'discrete': b_discrete[mb_indices],
                    'continuous': b_continuous[mb_indices],
                }
                mb_log_probs = b_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_values = b_values[mb_indices]

                # Forward pass (with optional mixed precision)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, new_log_prob, entropy = self.actor.get_action(mb_obs, mb_action)
                    new_value = self.critic(mb_obs).squeeze(-1)

                    # Policy loss
                    logratio = new_log_prob - mb_log_probs
                    ratio = logratio.exp()

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

                # Approximate KL for early stopping (outside autocast)
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item())
                    all_approx_kl.append(approx_kl.item())

                # Optimize (with optional mixed precision)
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
            if cfg.target_kl is not None:
                if approx_kl > cfg.target_kl:
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
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
        }
        if self.scaler is not None:
            save_dict['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle torch.compile prefix mismatch
        # Compiled models have "_orig_mod." prefix, uncompiled don't
        def fix_state_dict(state_dict, target_module):
            ckpt_has_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())
            model_has_prefix = any(k.startswith('_orig_mod.') for k in target_module.state_dict().keys())

            if ckpt_has_prefix and not model_has_prefix:
                # Strip prefix: loading compiled checkpoint into uncompiled model
                return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            elif not ckpt_has_prefix and model_has_prefix:
                # Add prefix: loading uncompiled checkpoint into compiled model
                return {'_orig_mod.' + k: v for k, v in state_dict.items()}
            return state_dict

        self.actor.load_state_dict(fix_state_dict(checkpoint['actor_state_dict'], self.actor))
        self.critic.load_state_dict(fix_state_dict(checkpoint['critic_state_dict'], self.critic))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


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
