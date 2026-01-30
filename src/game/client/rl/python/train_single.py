#!/usr/bin/env python3
"""
Single-Environment Teeworlds RL Training

Simple training script for debugging and testing with a single DDNet client.
Start the server and client manually, then run this script.

Usage:
    1. Start server:  ./DDNet-Server "sv_port 8303"
    2. Start client:  ./DDNet "connect localhost:8303; rl_enable state 0; bind k kill"
    3. Run training:  python train_single.py

    Or with auto-start:
    python train_single.py --auto-start
"""

import os
import sys
import time
import argparse
import subprocess
import signal
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from teeworlds_env import TeeWorldsShmClient
from ppo_hybrid import PPOAgent, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on single Teeworlds env")

    # Auto-start
    parser.add_argument("--auto-start", action="store_true",
                        help="Automatically start server and client")
    parser.add_argument("--ddnet-path", type=str, default="../../../../../build/DDNet")
    parser.add_argument("--server-path", type=str, default="../../../../../build/DDNet-Server")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-steps", type=int, default=256,
                        help="Steps per PPO update")
    parser.add_argument("--max-episode-steps", type=int, default=2000)

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)

    # Network
    parser.add_argument("--hidden-dim", type=int, default=256)

    # Logging
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=50)

    # Device
    parser.add_argument("--device", type=str, default="cpu")

    # Debug
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Teeworlds RL - Single Environment Training")
    print("=" * 60)

    # Process management
    server_proc = None
    client_proc = None

    try:
        # Auto-start server and client if requested
        if args.auto_start:
            print("\nStarting server...")
            server_proc = subprocess.Popen(
                [args.server_path, "sv_port 8303; sv_max_clients_per_ip 64"],
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)

            print("Starting client...")
            client_proc = subprocess.Popen(
                [args.ddnet_path, "connect localhost:8303; rl_enable state 0; snd_volume 0; bind k kill; player_name RL_Agent"],
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for shared memory
            shm_path = "/dev/shm/teeworlds_rl_0"
            print(f"Waiting for shared memory: {shm_path}")
            for i in range(30):
                if os.path.exists(shm_path):
                    print("  Found!")
                    break
                time.sleep(1)
                if i % 5 == 4:
                    print(f"  Still waiting... ({i+1}s)")
            else:
                raise RuntimeError("Timeout waiting for shared memory")

            time.sleep(2)  # Extra time for client to fully connect

        # Connect to shared memory
        print("\nConnecting to DDNet client...")
        client = TeeWorldsShmClient(env_id=0)
        if not client.connect():
            raise RuntimeError("Failed to connect to shared memory. Is DDNet running with rl_enable?")

        # Get initial observation
        print("Getting initial observation...")
        obs, info = client.wait_for_observation(timeout=10.0)
        if obs is None:
            raise RuntimeError("Failed to get initial observation")

        state = obs.to_flat_array()
        obs_dim = len(state)
        print(f"Observation dimension: {obs_dim}")

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"teeworlds_single_{timestamp}"
        run_dir = os.path.join(args.log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Logging to: {run_dir}")

        writer = SummaryWriter(run_dir)

        # Create PPO config and agent
        # For single env, we need to adjust minibatches
        effective_batch = args.num_steps
        num_minibatches = min(args.num_minibatches, effective_batch // 32)
        num_minibatches = max(1, num_minibatches)

        ppo_config = PPOConfig(
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            num_minibatches=num_minibatches,
            update_epochs=args.update_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            hidden_dim=args.hidden_dim,
        )

        device = torch.device(args.device)
        agent = PPOAgent(obs_dim=obs_dim, num_envs=1, config=ppo_config, device=str(device))

        print(f"\nPPO Config:")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Steps per update: {args.num_steps}")
        print(f"  Minibatches: {num_minibatches}")
        print(f"  Update epochs: {args.update_epochs}")
        print(f"  Entropy coef: {args.ent_coef}")

        # Training loop
        num_updates = args.total_timesteps // args.num_steps
        print(f"\nStarting training for {args.total_timesteps:,} timesteps ({num_updates} updates)")
        print("-" * 60)

        global_step = 0
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        best_reward = float('-inf')

        start_time = time.time()

        for update in range(1, num_updates + 1):
            # Collect rollout
            for step in range(args.num_steps):
                global_step += 1

                # Get action from policy
                obs_batch = state.reshape(1, -1)  # Add batch dimension
                action, log_prob, value = agent.get_action(obs_batch)

                # Convert action to game format
                game_action = convert_action(action)

                if args.verbose and global_step % 100 == 0:
                    print(f"  Step {global_step}: action={game_action}")

                # Send action and get next observation
                client.send_action(**game_action)
                next_obs, info = client.wait_for_observation(timeout=2.0)

                if next_obs is None:
                    print(f"Warning: No observation at step {global_step}")
                    # Use previous state
                    next_state = state
                    reward = 0.0
                    done = False
                else:
                    next_state = next_obs.to_flat_array()
                    reward = info.get('reward', 0.0)
                    done = info.get('done', False)

                # Track episode stats
                episode_reward += reward
                episode_length += 1

                # Store transition
                agent.store_transition(
                    step,
                    obs_batch,
                    action,
                    log_prob,
                    np.array([reward], dtype=np.float32),
                    np.array([done], dtype=bool),
                    value
                )

                # Handle episode end
                if done or episode_length >= args.max_episode_steps:
                    episode_count += 1

                    writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)

                    if episode_reward > best_reward:
                        best_reward = episode_reward

                    if args.verbose or episode_count % 5 == 0:
                        print(f"Episode {episode_count}: reward={episode_reward:.2f}, length={episode_length}, best={best_reward:.2f}")

                    # Reset
                    client.send_action(reset=1)
                    client.reset()
                    time.sleep(0.1)

                    next_obs, _ = client.wait_for_observation(timeout=5.0)
                    if next_obs is not None:
                        next_state = next_obs.to_flat_array()

                    episode_reward = 0.0
                    episode_length = 0

                state = next_state

            # PPO Update
            next_obs_batch = state.reshape(1, -1)
            stats = agent.update(next_obs_batch, np.array([False], dtype=bool))

            # Logging
            if update % args.log_freq == 0:
                elapsed = time.time() - start_time
                fps = global_step / elapsed

                writer.add_scalar("charts/fps", fps, global_step)
                writer.add_scalar("losses/policy_loss", stats['policy_loss'], global_step)
                writer.add_scalar("losses/value_loss", stats['value_loss'], global_step)
                writer.add_scalar("losses/entropy", stats['entropy'], global_step)
                writer.add_scalar("losses/approx_kl", stats['approx_kl'], global_step)

                print(f"Update {update:4d} | Step {global_step:7,d} | FPS {fps:5.0f} | "
                      f"PG Loss {stats['policy_loss']:7.4f} | Entropy {stats['entropy']:.4f} | "
                      f"Episodes {episode_count}")

            # Save checkpoint
            if update % args.save_freq == 0:
                checkpoint_path = os.path.join(run_dir, f"checkpoint_{update}.pt")
                agent.save(checkpoint_path)
                print(f"  Saved: {checkpoint_path}")

        # Final save
        final_path = os.path.join(run_dir, "checkpoint_final.pt")
        agent.save(final_path)
        print(f"\nTraining complete! Final checkpoint: {final_path}")
        print(f"Best episode reward: {best_reward:.2f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted")

    finally:
        # Cleanup
        if 'client' in dir() and client:
            client.disconnect()

        if client_proc:
            try:
                os.killpg(os.getpgid(client_proc.pid), signal.SIGTERM)
            except:
                pass

        if server_proc:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            except:
                pass

        if 'writer' in dir():
            writer.close()


def convert_action(action: dict) -> dict:
    """Convert PPO action output to game action format."""
    discrete = action['discrete'][0]  # Remove batch dim
    continuous = action['continuous'][0]

    direction = int(discrete[0]) - 1  # 0,1,2 -> -1,0,1
    jump = int(discrete[1])
    hook = int(discrete[2])

    # Continuous aim
    aim_angle = float(continuous[0]) * np.pi
    aim_magnitude = (float(continuous[1]) + 1.0) / 2.0

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


if __name__ == "__main__":
    main()
