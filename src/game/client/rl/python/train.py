#!/usr/bin/env python3
"""
Teeworlds RL Training Script

Trains a PPO agent with hybrid action space on the Teeworlds environment.
Uses CleanRL-style single-file approach with TensorBoard logging.

Usage:
    python train.py --num-envs 8 --total-timesteps 1000000

    # Resume training
    python train.py --resume runs/teeworlds_ppo/checkpoint.pt

    # Custom hyperparameters
    python train.py --learning-rate 1e-4 --ent-coef 0.02
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from teeworlds_vec_env import TeeWorldsVecEnv, EnvConfig
from teeworlds_server_env import TeeWorldsServerEnv, ServerEnvConfig
from ppo_hybrid import PPOAgent, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Teeworlds")

    # Environment
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--max-episode-steps", type=int, default=2000,
                        help="Maximum steps per episode")
    parser.add_argument("--sv-tickrate", type=int, default=50,
                        help="Server tickrate")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Frame skip (1=no skip, 4=act every 4th tick)")
    parser.add_argument("--server-side", action="store_true",
                        help="Use server-side bots (much more efficient)")
    parser.add_argument("--num-servers", type=int, default=1,
                        help="Number of servers (each with --num-envs bots)")
    parser.add_argument("--sync-inference", action="store_true",
                        help="Pause server during inference (slower but no stutter)")
    parser.add_argument("--sync-update", action="store_true",
                        help="Pause server during PPO update (prevents stale policy actions)")
    parser.add_argument("--map", type=str, default="Tutorial",
                        help="Map to train on (default: Tutorial)")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="Total timesteps to train")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="Steps per rollout per environment")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Number of minibatches for PPO update")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="Number of epochs per PPO update")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="PPO clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="Target KL divergence for early stopping")

    # Network
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of hidden layers")

    # Logging
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logging")
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--save-freq", type=int, default=100,
                        help="Save checkpoint every N updates")
    parser.add_argument("--log-freq", type=int, default=10,
                        help="Log to TensorBoard every N updates")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, auto)")

    # Paths
    parser.add_argument("--ddnet-path", type=str,
                        default="../../../../../build/DDNet",
                        help="Path to DDNet executable")
    parser.add_argument("--server-path", type=str,
                        default="../../../../../build/DDNet-Server",
                        help="Path to DDNet-Server executable")

    # CNN Policy
    parser.add_argument("--use-cnn", action="store_true",
                        help="Use CNN feature extractor for tile observations")

    # Curriculum learning
    parser.add_argument("--curriculum", type=str, default="none",
                        choices=["none", "map", "segment", "both"],
                        help="Curriculum learning mode")
    parser.add_argument("--maps-dir", type=str, default=None,
                        help="Path to ddnet-maps repository for curriculum learning")
    parser.add_argument("--starting-category", type=str, default="novice",
                        help="Starting difficulty category for curriculum")
    parser.add_argument("--success-threshold", type=float, default=0.8,
                        help="Success rate threshold to advance curriculum")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    args = parse_args()

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create run name (timestamp first for lexicographic sorting)
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{timestamp}_ppo_{args.num_envs}env"

    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Logging to: {run_dir}")

    # Create TensorBoard writer
    writer = SummaryWriter(run_dir)

    # Log hyperparameters
    writer.add_text("hyperparameters", str(vars(args)))

    # Create environment
    if args.server_side:
        # Server-side bots (more efficient, single process per server)
        total_bots = args.num_envs * args.num_servers
        if args.num_servers > 1:
            print(f"\nCreating SERVER-SIDE environment with {total_bots} bots ({args.num_servers} servers, {args.num_envs} bots each) on map '{args.map}'...")
        else:
            print(f"\nCreating SERVER-SIDE environment with {args.num_envs} bots on map '{args.map}'...")
        server_config = ServerEnvConfig(
            sv_tickrate=args.sv_tickrate,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_episode_steps,
            use_dict_obs=args.use_cnn,  # Enable dict observations for CNN
            map_name=args.map,
        )
        # Use default server_path (resolved relative to teeworlds_server_env.py)
        env = TeeWorldsServerEnv(num_bots=args.num_envs, num_servers=args.num_servers, config=server_config)
        # Update num_envs to reflect total bots for training loop
        args.num_envs = total_bots
    else:
        # Client-side environments (original approach)
        print(f"\nCreating CLIENT-SIDE environment with {args.num_envs} parallel instances...")
        env_config = EnvConfig(
            ddnet_path=args.ddnet_path,
            server_path=args.server_path,
            sv_tickrate=args.sv_tickrate,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_episode_steps,
        )
        env = TeeWorldsVecEnv(num_envs=args.num_envs, config=env_config)

    try:
        # Get observation dimension
        obs, _ = env.reset()
        if isinstance(obs, dict):
            # Dict observation (CNN mode)
            obs_dim = sum(np.prod(v.shape[1:]) for v in obs.values())
            print(f"Dict observation - tiles: {obs['tiles'].shape}, player: {obs['player_state'].shape}, checkpoint: {obs['checkpoint'].shape}")
        else:
            # Flat observation (fast path)
            obs_dim = obs.shape[1]
        print(f"Observation dimension: {obs_dim}")

        # Force use_cnn=False for fast path (flat observations)
        # TODO: Re-enable CNN support with dict observations if needed
        if args.use_cnn:
            print("Warning: --use-cnn is not supported with fast observation path, using MLP")

        # Create PPO config
        ppo_config = PPOConfig(
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_cnn=False,  # Always use MLP with fast flat observations
        )

        # Create agent
        agent = PPOAgent(
            obs_dim=obs_dim,
            num_envs=args.num_envs,
            config=ppo_config,
            device=str(device),
        )

        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from: {args.resume}")
            agent.load(args.resume)

        # Training loop
        num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
        print(f"\nStarting training for {args.total_timesteps:,} timesteps ({num_updates} updates)")
        print(f"Batch size: {args.num_steps * args.num_envs}")
        print(f"Server tickrate: {args.sv_tickrate}, Frame skip: {args.frame_skip}")
        if args.frame_skip > 1:
            effective_rate = args.sv_tickrate / args.frame_skip
            print(f"Effective decision rate: {effective_rate:.0f} Hz")
        if args.server_side:
            inference_mode = "sync (pause)" if args.sync_inference else "async (no pause)"
            update_mode = "sync (pause)" if args.sync_update else "async (no pause)"
            print(f"Inference: {inference_mode}, PPO update: {update_mode}")
        print("-" * 60)

        start_time = time.time()
        global_step = agent.global_step
        starting_global_step = global_step  # Track starting point for FPS calculation

        # Episode tracking
        episode_rewards = []
        episode_lengths = []

        for update in range(1, num_updates + 1):
            update_start_time = time.time()

            # Anneal learning rate (linear decay)
            if args.total_timesteps > 0:
                frac = 1.0 - (update - 1) / num_updates
                lr = frac * args.learning_rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = lr

            # Collect rollout
            for step in range(args.num_steps):
                global_step += args.num_envs

                # Get action (optionally pause during inference to prevent stale actions)
                if args.server_side and args.sync_inference:
                    env.pause()
                action, log_prob, value = agent.get_action(obs)
                if args.server_side and args.sync_inference:
                    env.unpause()

                # Environment step
                next_obs, reward, done, truncated, infos = env.step(action)

                # Store transition
                agent.store_transition(step, obs, action, log_prob, reward, done, value)

                obs = next_obs

                # Track episodes
                for info in infos:
                    if 'episode' in info:
                        episode_rewards.append(info['episode']['r'])
                        episode_lengths.append(info['episode']['l'])

            # PPO update (optionally pause to prevent bots from acting with stale policy)
            if args.server_side and args.sync_update:
                env.pause()
            stats = agent.update(obs, done | truncated)
            if args.server_side and args.sync_update:
                env.unpause()
            agent.global_step = global_step

            # Logging
            if update % args.log_freq == 0:
                elapsed = time.time() - start_time
                fps = (global_step - starting_global_step) / elapsed

                # TensorBoard logging
                writer.add_scalar("charts/learning_rate", lr, global_step)
                writer.add_scalar("charts/fps", fps, global_step)
                writer.add_scalar("losses/policy_loss", stats['policy_loss'], global_step)
                writer.add_scalar("losses/value_loss", stats['value_loss'], global_step)
                writer.add_scalar("losses/entropy", stats['entropy'], global_step)
                writer.add_scalar("losses/approx_kl", stats['approx_kl'], global_step)
                writer.add_scalar("losses/clipfrac", stats['clipfrac'], global_step)

                if episode_rewards:
                    mean_reward = np.mean(episode_rewards[-100:])
                    mean_length = np.mean(episode_lengths[-100:])
                    writer.add_scalar("charts/episodic_return", mean_reward, global_step)
                    writer.add_scalar("charts/episodic_length", mean_length, global_step)

                    print(f"Update {update:5d} | Step {global_step:9,d} | "
                          f"FPS {fps:6.0f} | "
                          f"Reward {mean_reward:7.2f} | "
                          f"Length {mean_length:6.0f} | "
                          f"PG Loss {stats['policy_loss']:7.4f} | "
                          f"Entropy {stats['entropy']:.4f}")
                else:
                    print(f"Update {update:5d} | Step {global_step:9,d} | "
                          f"FPS {fps:6.0f} | "
                          f"PG Loss {stats['policy_loss']:7.4f} | "
                          f"Entropy {stats['entropy']:.4f}")

            # Save checkpoint
            if update % args.save_freq == 0:
                checkpoint_path = os.path.join(run_dir, f"checkpoint_{update}.pt")
                agent.save(checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")

                # Also save as latest
                latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
                agent.save(latest_path)

        # Final save
        final_path = os.path.join(run_dir, "checkpoint_final.pt")
        agent.save(final_path)
        print(f"\nTraining complete! Final checkpoint: {final_path}")

        total_time = time.time() - start_time
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Average FPS: {global_step / total_time:.0f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        # Save checkpoint on interrupt if agent exists
        if 'agent' in dir():
            interrupt_path = os.path.join(run_dir, "checkpoint_interrupted.pt")
            agent.save(interrupt_path)
            print(f"Saved checkpoint: {interrupt_path}")

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Try to save checkpoint on error if agent exists
        if 'agent' in dir():
            try:
                error_path = os.path.join(run_dir, "checkpoint_error.pt")
                agent.save(error_path)
                print(f"Saved checkpoint: {error_path}")
            except:
                pass

    finally:
        # Cleanup
        env.close()
        writer.close()


if __name__ == "__main__":
    main()
