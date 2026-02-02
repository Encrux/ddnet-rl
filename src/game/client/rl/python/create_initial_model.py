#!/usr/bin/env python3
"""Create an initial ONNX model for DDNet RL training."""

import torch
import torch.nn as nn

# Constants matching C++ rl_protocol.h
TILE_GRID_WIDTH = 29
TILE_GRID_HEIGHT = 21
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT

# Observation size calculation (simplified)
OBS_PLAYER = 16      # local player state
OBS_TILES = TILE_GRID_SIZE  # 29x21 = 609 tile IDs

OBSERVATION_SIZE = OBS_PLAYER + OBS_TILES  # = 625

# Network output: 3 + 1 + 1 + 1 + 3 + 2 + 2 + 1 = 14
NETWORK_OUTPUT_SIZE = 14

# Tile embedding configuration
NUM_TILE_TYPES = 256  # Tile IDs are 0-255
TILE_EMBEDDING_DIM = 16


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

        # Shared feature extractor with LayerNorm for stability
        self.shared = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Actor heads
        self.direction_head = nn.Linear(hidden_size, 3)
        self.jump_head = nn.Linear(hidden_size, 1)
        self.hook_head = nn.Linear(hidden_size, 1)
        self.fire_head = nn.Linear(hidden_size, 1)
        self.weapon_head = nn.Linear(hidden_size, 3)
        self.aim_mean = nn.Linear(hidden_size, 2)
        self.aim_log_std = nn.Linear(hidden_size, 2)

        # Critic head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning flattened output for ONNX.

        Input x has shape [batch, 625]:
          - x[:, :16] = player state (16 floats)
          - x[:, 16:] = tile IDs (609 floats, to be cast to int for embedding)
        """
        # Handle NaN/inf in entire observation first
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Split observation into components
        player_state = x[:, :OBS_PLAYER]  # [batch, 16]
        tile_ids_float = x[:, OBS_PLAYER:]  # [batch, 609]

        # Clamp player state to handle outliers (C++ pre-normalizes to ~[-1,1])
        player_state = player_state.clamp(-10.0, 10.0)

        # Cast tile IDs to long for embedding lookup
        tile_ids = tile_ids_float.clamp(0, NUM_TILE_TYPES - 1).long()

        # Embed tiles: [batch, 609] -> [batch, 609, 16]
        tile_features = self.tile_embedding(tile_ids)

        # Flatten tile features: [batch, 609, 16] -> [batch, 609 * 16]
        tile_flat = tile_features.flatten(start_dim=1)

        # Combine all features
        combined = torch.cat([player_state, tile_flat], dim=1)

        # Shared feature extraction
        features = self.shared(combined)

        # Output heads
        direction_logits = self.direction_head(features)
        jump_logit = self.jump_head(features)
        hook_logit = self.hook_head(features)
        fire_logit = self.fire_head(features)
        weapon_logits = self.weapon_head(features)
        aim_mean = self.aim_mean(features)
        aim_log_std = self.aim_log_std(features).clamp(-2, 2)
        value = self.value_head(features)

        return torch.cat([
            direction_logits, jump_logit, hook_logit, fire_logit,
            weapon_logits, aim_mean, aim_log_std, value
        ], dim=-1)


def main():
    import sys

    output_path = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"

    print(f"Creating initial model with tile embeddings...")
    print(f"  Observation size: {OBSERVATION_SIZE}")
    print(f"    - Player: {OBS_PLAYER}")
    print(f"    - Tile grid: {OBS_TILES} ({TILE_GRID_WIDTH}x{TILE_GRID_HEIGHT})")
    print(f"  Tile embedding: {NUM_TILE_TYPES} types -> {TILE_EMBEDDING_DIM}-dim")
    print(f"  Output size: {NETWORK_OUTPUT_SIZE}")

    model = PolicyNetwork()
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, OBSERVATION_SIZE)

    # Test forward pass
    output = model(dummy_input)
    print(f"  Test output shape: {output.shape}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['output'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )

    print(f"Model saved to: {output_path}")

    # Verify
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model verification: OK")


if __name__ == '__main__':
    main()
