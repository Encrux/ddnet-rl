#!/usr/bin/env python3
"""
Teeworlds RL Evaluation Script

Runs a trained agent for visualization/evaluation.
Connect to the server to watch the bots play.

Usage:
    python eval.py --checkpoint runs/your_run/checkpoint_latest.pt

    # Then connect with DDNet client to localhost:8303
"""

import argparse
import time
import numpy as np
import torch

from teeworlds_server_env import TeeWorldsServerEnv, ServerEnvConfig, FrameStackWrapper
from ppo_hybrid import PPOAgent, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--num-bots", type=int, default=8,
                        help="Number of bots to run")
    parser.add_argument("--map", type=str, default="Tutorial",
                        help="Map to play on")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (no sampling)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, auto)")
    parser.add_argument("--port", type=int, default=8303,
                        help="Server port")
    parser.add_argument("--frame-skip", type=int, default=None,
                        help="Frame skip (default: use value from checkpoint)")
    parser.add_argument("--max-episode-steps", type=int, default=100000,
                        help="Max steps per episode before reset (default: 100000 = ~33 min)")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint to get config
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_config = checkpoint.get('config', PPOConfig())
    use_cnn = saved_config.use_cnn

    # Get stacking config from saved model
    n_frame_stack = getattr(saved_config, 'n_frame_stack', 1)
    n_player_state_stack = getattr(saved_config, 'n_player_state_stack', 1)
    needs_stacking = n_frame_stack > 1 or n_player_state_stack > 1

    # Get frame_skip from checkpoint or command line
    frame_skip = args.frame_skip if args.frame_skip is not None else getattr(saved_config, 'frame_skip', 3)

    # Create environment
    print(f"\nStarting server with {args.num_bots} bots on map '{args.map}'...")
    print(f"Connect to localhost:{args.port} to watch!")
    print(f"Frame skip: {frame_skip} (decision rate: {50 // frame_skip} Hz)")
    if needs_stacking:
        print(f"Frame stacking: tiles={n_frame_stack}, player_state={n_player_state_stack}")

    config = ServerEnvConfig(
        server_port=args.port,
        map_name=args.map,
        sv_name="RL Eval",
        use_dict_obs=use_cnn,  # Match training config
        frame_skip=frame_skip,
        max_episode_steps=args.max_episode_steps,
    )
    env = TeeWorldsServerEnv(num_bots=args.num_bots, config=config)

    # Wrap with frame stacking if needed
    if needs_stacking:
        env = FrameStackWrapper(env, n_tile_stack=n_frame_stack, n_player_state_stack=n_player_state_stack)

    try:
        # Get observation to determine dimensions
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs_dim = sum(np.prod(v.shape[1:]) for v in obs.values())
        else:
            obs_dim = obs.shape[1]

        # Create agent with saved config
        saved_config.use_compile = False  # Don't compile for eval

        agent = PPOAgent(
            obs_dim=obs_dim,
            num_envs=args.num_bots,
            config=saved_config,
            device=str(device),
        )
        agent.load(args.checkpoint)
        print("Agent loaded successfully!")

        # Run evaluation loop
        print("\n" + "="*50)
        print("Running evaluation - press Ctrl+C to stop")
        print("="*50 + "\n")

        episode_rewards = np.zeros(args.num_bots)
        episode_count = 0

        while True:
            # Get action
            action, _, _ = agent.get_action(obs, deterministic=args.deterministic)

            # Step environment
            obs, reward, done, truncated, infos = env.step(action)

            episode_rewards += reward

            # Log completed episodes
            for i, info in enumerate(infos):
                if 'episode' in info:
                    episode_count += 1
                    print(f"Episode {episode_count}: "
                          f"reward={info['episode']['r']:.2f}, "
                          f"length={info['episode']['l']}")
                    episode_rewards[i] = 0

            # Small sleep to not spam CPU
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nEvaluation stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
