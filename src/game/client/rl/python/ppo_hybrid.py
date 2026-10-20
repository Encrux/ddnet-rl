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

from observation import SEMANTIC_CHANNELS, semantic_lookup


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

    # Action space.
    #
    # Aim is FACTORED: a Categorical over angular sectors times a continuous
    # residual within the chosen sector. Neither half works alone.
    #
    # A pure Gaussian over the angle is unimodal, so at a fork where the correct
    # action is "hook hard left OR hard right, never straight" it collapses to
    # the mean - the one action that fails. Hook movement is full of these.
    #
    # Pure binning is too coarse: m_HookLength is 380, so one bin of a 32-way
    # split (11.25 deg) subtends ~75 world units (2.3 tiles) at full hook range,
    # which misses the block. The residual restores full precision.
    #
    # Aim MAGNITUDE is gone. DDNet only uses the direction of m_TargetX/Y, so it
    # was a free dimension the entropy bonus could inflate forever at zero cost -
    # 1.42 of the observed 5.33 nats, 27% of the entropy budget, spent on an
    # action the game ignores.
    aim_sectors: int = 16
    discrete_dims: Tuple[int, ...] = (3, 2, 2, 16)  # direction, jump, hook, aim sector
    continuous_dim: int = 1  # aim residual within the sector

    # Learned action hold (see action_hold.py). Empty = disabled. When set, a
    # final discrete head of size len(hold_durations) is appended, and the env
    # latches the sampled action for the chosen number of decisions.
    hold_durations: Tuple[int, ...] = ()

    # Extra entropy coefficient applied to the duration head alone.
    #
    # Measured: with a plain shared ent_coef the policy DRIVES ITS OWN HOLDS
    # DOWN, from the 3.27 init prior to 2.19. A long hold is locally costly
    # everywhere on the map -- the agent is blind for its duration -- while its
    # exploration payoff is diffuse, delayed, and only collected at one
    # obstacle. Gradient descent correctly kills it, and kills the mechanism
    # with it. Ratchet shaping recovered some of this (2.19 -> 2.64) by removing
    # the penalty on the retreat, but did not restore the prior.
    #
    # This protects P(long hold) from that pressure. It is the on-policy
    # equivalent of what makes injected ez-greedy holds work: they cannot be
    # trained away because no gradient reaches them.
    hold_ent_coef: float = 0.0

    # CNN Feature Extractor (for dict observations)
    use_cnn: bool = False
    tile_grid_height: int = 31   # 1.5x original (21 -> 31)
    tile_grid_width: int = 43    # 1.5x original (29 -> 43)
    player_state_dim: int = 15   # Player state features
    checkpoint_dim: int = 34     # progress(1) + 8 lookahead (x,y) + high-water(1) + 16 hook-reach
    cnn_channels: Tuple[int, ...] = (32, 64, 64)  # CNN channel progression

    # Embedding Feature Extractor (for raw tile IDs)
    use_embeddings: bool = False     # Use tile embeddings instead of normalized values
    tile_embed_dim: int = 8
    semantic_tiles: bool = False     # fixed semantic lookup instead of a learned embedding
    tile_encoder: str = "mlp"        # "mlp" or "cnn" for processing embedded tiles

    # Mixed precision training (2-3x speedup on modern GPUs)
    use_amp: bool = True

    # torch.compile for faster forward/backward passes (PyTorch 2.0+)
    use_compile: bool = True

    # Frame stacking for temporal context
    # With frame_skip=3: 8 frames × 3 ticks × 20ms = 480ms coverage
    n_frame_stack: int = 1  # Number of tile frames to stack (1 = no stacking, expensive)
    n_player_state_stack: int = 8  # Number of player state frames to stack (cheap, enables timing)

    def __post_init__(self):
        # The duration head is appended LAST so that every existing consumer of
        # discrete_dims (env action packing reads columns 0-3, checkpoints, the
        # action-usage stats) keeps working on unchanged indices.
        if self.hold_durations:
            self.hold_durations = tuple(self.hold_durations)
            self.discrete_dims = tuple(self.discrete_dims) + (len(self.hold_durations),)


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



# ---------------------------------------------------------------------------
# Fixed semantic tile encoding (roadmap.md §1.9)
#
# The float32 path feeds the CNN a single channel of raw_tile_index / 15, so the
# distinctions that matter most for movement are neighbouring scalars:
#
#     SOLID   = 1  -> 0.067   (blocking, hook ATTACHES)
#     NOHOOK  = 3  -> 0.200   (blocking, hook does NOT attach)
#     THROUGH = 6  -> 0.400   (hook PASSES THROUGH)
#
# SOLID/NOHOOK/THROUGH are semantic opposites sitting 0.13-0.33 apart on a
# continuous axis, with NOLASER and THROUGH_CUT as distractors in between. A
# hook-climb depends on exactly that discrimination, so the encoding makes the
# single most important question the hardest one to extract.
#
# This table makes them orthogonal channels. It is FIXED, not learned: no
# parameters, no gradient, no exploration cost - only a reindexing of
# information the observation already carries.
#
# The id -> class mapping itself lives in observation.py, next to the wire format
# that produces the raw ids, so the policy and the observation viewer cannot
# drift apart about what a tile means.


def build_semantic_table(device=None) -> torch.Tensor:
    """(256, SEMANTIC_CHANNELS) one-hot lookup. Unmapped ids fall into OTHER."""
    classes = torch.from_numpy(semantic_lookup().astype(np.int64))
    t = torch.zeros(256, SEMANTIC_CHANNELS, dtype=torch.float32)
    t[torch.arange(256), classes] = 1.0
    return t.to(device) if device is not None else t


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

        # Fixed semantic lookup, or a learned embedding.
        self.semantic_tiles = getattr(config, 'semantic_tiles', False)
        if self.semantic_tiles:
            embed_dim = SEMANTIC_CHANNELS
            self.tile_embedding = nn.Embedding(256, embed_dim)
            with torch.no_grad():
                self.tile_embedding.weight.copy_(build_semantic_table())
            self.tile_embedding.weight.requires_grad_(False)
        else:
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

        # Bias the duration head toward SHORT holds at init. A uniform prior over
        # {1,2,4,8,16} means an average hold of 6.2 decisions from step one, which
        # would degrade the fine control the agent already has on terrain it has
        # solved. This prior gives a mean of ~3.3 while still drawing the longest
        # hold 5.8% of the time — five orders of magnitude more often than the
        # 0.425^16 it replaces, which is the entire point.
        if config.hold_durations:
            n_dur = len(config.hold_durations)
            with torch.no_grad():
                self.discrete_heads[-1].bias.copy_(
                    torch.linspace(1.0, -1.0, n_dur)
                )

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
            entropy: Total entropy of the action distribution
            head_entropies: Per-discrete-head entropy, (batch, n_heads)
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

        # Compute entropy. Per-head values are returned alongside the sum so a
        # single head can be given its own coefficient (see hold_ent_coef)
        # without paying for another forward pass.
        head_entropies = torch.stack([d.entropy() for d in discrete_dists], dim=-1)
        discrete_entropy = head_entropies.sum(dim=-1)
        continuous_entropy = cont_dist.entropy().sum(dim=-1)
        total_entropy = discrete_entropy + continuous_entropy

        # Stack discrete actions
        action_out = {
            'discrete': torch.stack(discrete_actions, dim=-1),
            'continuous': continuous_action,
        }

        return action_out, total_log_prob, total_entropy, head_entropies


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

        # Cache parameter list for gradient clipping (avoid recreating 384x per update)
        self._all_params = list(self.actor.parameters()) + list(self.critic.parameters())

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

        # Which rows the policy actually chose an action on. Without a duration
        # head every row is fresh, so this stays all-ones and the masked means
        # below reduce exactly to the plain means.
        self.fresh_mask = torch.ones((ns, ne), device=self.device)

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

        action, log_prob, _, _ = self.actor.get_action(obs_t, deterministic=deterministic)
        value = self.critic(obs_t)

        # Use pinned memory for faster D2H transfer on CUDA
        if self._use_async_transfers:
            # Synchronous copy to pinned memory (copy_() implicitly syncs on the source tensor)
            # Faster than .cpu() because it avoids allocation and uses pre-pinned DMA buffers
            self._pinned_discrete.copy_(action['discrete'])
            self._pinned_continuous.copy_(action['continuous'])

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
    def value_of(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """Critic value for an arbitrary observation batch.

        Used for time-limit bootstrapping: a truncated episode's remaining
        return is gamma * V(final observation), which has to be folded into the
        reward because the env has already auto-reset by the time we see it.
        """
        return self.critic(self._obs_to_tensor(obs)).squeeze(-1).cpu().numpy()

    def store_transition(
        self,
        step: int,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        action: Dict[str, np.ndarray],
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        fresh: Optional[np.ndarray] = None
    ):
        """Store a transition in the rollout buffer.

        `fresh` marks rows where the policy chose the action. On held rows the
        stored action is the latch, while `log_prob` belongs to the action that
        was sampled and thrown away — a deliberate mismatch that never reaches
        the loss, because `update()` masks those rows out of the surrogate
        objective. They are kept in the buffer because their rewards and values
        are real and the critic must learn them.
        """
        if self.use_embeddings:
            # Embedding mode: tiles as long for nn.Embedding
            tiles = obs['tiles']
            # Validate tile values for embedding (must be 0-255 for nn.Embedding(256, ...))
            if tiles.min() < 0 or tiles.max() >= 256:
                raise ValueError(
                    f"[store_transition] Invalid tile values at step {step}: "
                    f"min={tiles.min()}, max={tiles.max()} (expected 0-255)"
                )
            self.obs_tiles[step] = torch.from_numpy(tiles).long().to(self.device)
            self.obs_player_state[step] = torch.from_numpy(obs['player_state']).to(self.device)
            self.obs_checkpoint[step] = torch.from_numpy(obs['checkpoint']).to(self.device)
        elif self.use_cnn:
            self.obs_tiles[step] = torch.from_numpy(obs['tiles']).to(self.device)
            self.obs_player_state[step] = torch.from_numpy(obs['player_state']).to(self.device)
            self.obs_checkpoint[step] = torch.from_numpy(obs['checkpoint']).to(self.device)
        else:
            self.obs_buffer[step] = torch.from_numpy(obs).to(self.device)

        # Validate discrete actions before storing
        discrete = action['discrete']
        for i, dim in enumerate(self.config.discrete_dims):
            if (discrete[:, i] < 0).any() or (discrete[:, i] >= dim).any():
                invalid_mask = (discrete[:, i] < 0) | (discrete[:, i] >= dim)
                invalid_vals = discrete[invalid_mask, i].tolist()
                raise ValueError(
                    f"[store_transition] Invalid discrete action at step {step}, dim {i} (max={dim-1}): "
                    f"found values {invalid_vals[:5]}{'...' if len(invalid_vals) > 5 else ''}"
                )

        self.discrete_actions[step] = torch.from_numpy(discrete).to(self.device)
        self.continuous_actions[step] = torch.from_numpy(action['continuous']).to(self.device)
        self.log_probs[step] = torch.from_numpy(log_prob).to(self.device)
        self.rewards[step] = torch.from_numpy(reward).to(self.device)
        self.dones[step] = torch.from_numpy(done.astype(np.float32)).to(self.device)
        self.values[step] = torch.from_numpy(value).to(self.device)
        if fresh is None:
            self.fresh_mask[step] = 1.0
        else:
            self.fresh_mask[step] = torch.from_numpy(fresh.astype(np.float32)).to(self.device)

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
        b_fresh = self.fresh_mask.reshape(-1)

        # Normalize advantages over the rows that actually enter the surrogate
        # objective. With a duration head the held rows sit mid-maneuver and have
        # a systematically different advantage distribution, so including them
        # would bias the shift and scale of the gradient inputs.
        fresh_adv = b_advantages[b_fresh > 0]
        if fresh_adv.numel() > 1:
            b_advantages = (b_advantages - fresh_adv.mean()) / (fresh_adv.std() + 1e-8)
        else:
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
                mb_fresh = b_fresh[mb_indices]
                # Denominator for every policy-side mean. Guarded because a
                # minibatch could in principle contain no fresh rows.
                mb_fresh_n = mb_fresh.sum().clamp(min=1.0)

                # Validate discrete actions are within bounds (debug CUDA scatter/gather errors)
                for i, dim in enumerate(cfg.discrete_dims):
                    action_col = mb_action['discrete'][:, i]
                    if (action_col < 0).any() or (action_col >= dim).any():
                        invalid_mask = (action_col < 0) | (action_col >= dim)
                        invalid_vals = action_col[invalid_mask].cpu().tolist()
                        raise ValueError(
                            f"Invalid discrete action for dimension {i} (max={dim-1}): "
                            f"found values {invalid_vals[:5]}{'...' if len(invalid_vals) > 5 else ''}"
                        )

                # Forward pass (with optional mixed precision)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, new_log_prob, entropy, head_ents = self.actor.get_action(mb_obs, mb_action)
                    new_value = self.critic(mb_obs).squeeze(-1)

                    # Policy loss
                    logratio = new_log_prob - mb_log_probs
                    ratio = logratio.exp()

                    # Clipped objective
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    # Masked so that only steps where the policy actually chose an
                    # action contribute. On a held step the stored action came
                    # from the latch, not from pi_old, so its importance ratio is
                    # meaningless and optimising it would train the policy to
                    # reproduce its own hold.
                    pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_fresh).sum() / mb_fresh_n

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

                    # Entropy loss, masked to match pg_loss. Entropy itself is a
                    # property of the state, so held rows carry a valid value —
                    # but averaging it over all rows while pg_loss averages over
                    # a quarter of them would silently multiply the effective
                    # ent_coef. Run B established that the entropy bonus is
                    # load-bearing for exploration here, so its scale must not
                    # drift as a side effect of adding the duration head.
                    entropy_loss = (entropy * mb_fresh).sum() / mb_fresh_n

                    # Extra entropy pressure on the duration head alone. Without
                    # it the policy trains its own long holds away (3.27 -> 2.19
                    # measured), which removes the exploration mechanism the head
                    # exists to provide.
                    if cfg.hold_durations and cfg.hold_ent_coef > 0.0:
                        hold_entropy = (head_ents[:, -1] * mb_fresh).sum() / mb_fresh_n
                        entropy_loss = entropy_loss + (cfg.hold_ent_coef / max(cfg.ent_coef, 1e-8)) * hold_entropy

                    # Total loss
                    loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * v_loss

                # Approximate KL for early stopping (outside autocast)
                with torch.no_grad():
                    # Masked too: on held rows the ratio compares the latched
                    # action against the log-prob of a different, discarded one,
                    # which would make both diagnostics read as huge divergence.
                    old_approx_kl = (-logratio * mb_fresh).sum() / mb_fresh_n
                    approx_kl = (((ratio - 1) - logratio) * mb_fresh).sum() / mb_fresh_n
                    clipped = ((ratio - 1.0).abs() > self.config.clip_coef).float()
                    clipfracs.append(((clipped * mb_fresh).sum() / mb_fresh_n).item())
                    all_approx_kl.append(approx_kl.item())

                # Optimize (outside autocast)
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self._all_params, self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._all_params, self.config.max_grad_norm)
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
