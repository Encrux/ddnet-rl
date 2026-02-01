#!/usr/bin/env python3
"""Create an initial ONNX model for DDNet RL training."""

import torch
import torch.nn as nn

# Constants matching C++ rl_protocol.h
MAX_PLAYERS = 16
MAX_PROJECTILES = 64
MAX_PICKUPS = 32
TILE_GRID_WIDTH = 29
TILE_GRID_HEIGHT = 21
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT

# Observation size calculation
OBS_GAME_STATE = 4
OBS_PLAYER = 16
OBS_OTHER_PLAYERS = (MAX_PLAYERS - 1) * 16
OBS_PROJECTILES = MAX_PROJECTILES * 8
OBS_PICKUPS = MAX_PICKUPS * 4
OBS_TILES = TILE_GRID_SIZE

OBSERVATION_SIZE = (
    OBS_GAME_STATE + OBS_PLAYER + OBS_OTHER_PLAYERS +
    OBS_PROJECTILES + OBS_PICKUPS + OBS_TILES
)

# Network output: 3 + 1 + 1 + 1 + 3 + 2 + 2 + 1 = 14
NETWORK_OUTPUT_SIZE = 14


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
        """Forward pass returning flattened output for ONNX."""
        features = self.shared(x)

        direction_logits = self.direction_head(features)
        jump_logit = self.jump_head(features)
        hook_logit = self.hook_head(features)
        fire_logit = self.fire_head(features)
        weapon_logits = self.weapon_head(features)
        aim_mean = self.aim_mean(features)
        aim_log_std = self.aim_log_std(features)
        value = self.value_head(features)

        return torch.cat([
            direction_logits, jump_logit, hook_logit, fire_logit,
            weapon_logits, aim_mean, aim_log_std, value
        ], dim=-1)


def main():
    import sys

    output_path = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"

    print(f"Creating initial model...")
    print(f"  Observation size: {OBSERVATION_SIZE}")
    print(f"  Output size: {NETWORK_OUTPUT_SIZE}")

    model = PolicyNetwork()
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, OBSERVATION_SIZE)

    # Test forward pass
    output = model(dummy_input)
    print(f"  Test output shape: {output.shape}")

    # Export to ONNX using legacy TorchScript exporter for compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # Use older opset for compatibility
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['output'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy TorchScript-based exporter
    )

    print(f"Model saved to: {output_path}")

    # Verify
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model verification: OK")


if __name__ == '__main__':
    main()
