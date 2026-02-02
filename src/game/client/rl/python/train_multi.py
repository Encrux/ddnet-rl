#!/usr/bin/env python3
"""
Multi-server PPO training for DDNet RL.

Spawns multiple DDNet servers, each with 127 bots, and trains from all of them
in parallel. Each server writes to its own shared memory region.

Usage:
    # Start fresh training with 4 servers (252 bots total)
    python train_multi.py --num-servers 4

    # Resume existing training
    python train_multi.py --num-servers 4 --resume

    # Connect to already-running servers
    python train_multi.py --num-servers 4 --no-spawn
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Import from existing train_async
from train_async import (
    SharedMemoryReader, PolicyNetwork, PPOTrainer,
    OBSERVATION_SIZE, Transition
)


class MultiServerManager:
    """Manages multiple DDNet server processes."""

    def __init__(
        self,
        num_servers: int,
        server_executable: str,
        base_port: int = 8303,
        map_name: str = "Tutorial",
        frame_skip: int = 4,
    ):
        self.num_servers = num_servers
        self.server_executable = server_executable
        self.base_port = base_port
        self.map_name = map_name
        self.frame_skip = frame_skip
        self.processes: List[subprocess.Popen] = []
        self.shm_names: List[str] = []

    def start(self) -> List[str]:
        """Start all servers and return their shm names."""
        print(f"Starting {self.num_servers} servers...")

        for i in range(self.num_servers):
            shm_name = f"/ddnet_rl_training_{i}"
            port = self.base_port + i

            # Set environment for this server
            env = os.environ.copy()
            env["DDNET_RL_SHM_NAME"] = shm_name
            env["DDNET_RL_FRAME_SKIP"] = str(self.frame_skip)

            # Add ONNX Runtime and CUDA libraries to LD_LIBRARY_PATH
            # Auto-detect paths based on current environment
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            nvidia_base = os.path.join(conda_prefix, "lib/python3.14/site-packages/nvidia") if conda_prefix else ""
            onnx_root = os.environ.get("ONNXRUNTIME_ROOT", "")
            extra_lib_paths = []
            if onnx_root:
                extra_lib_paths.append(os.path.join(onnx_root, "lib"))
            if nvidia_base and os.path.exists(nvidia_base):
                extra_lib_paths.extend([
                    f"{nvidia_base}/cuda_runtime/lib",
                    f"{nvidia_base}/cublas/lib",
                    f"{nvidia_base}/cudnn/lib",
                    f"{nvidia_base}/cufft/lib",
                    f"{nvidia_base}/curand/lib",
                    f"{nvidia_base}/cusparse/lib",
                    f"{nvidia_base}/nccl/lib",
                ])
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(extra_lib_paths) + (":" + existing_ld_path if existing_ld_path else "")

            # Start server
            cmd = [
                self.server_executable,
                f"sv_map {self.map_name}",
                f"sv_port {port}",
                "sv_max_clients 128",
                "sv_spectator_slots 1",
                "sv_max_afk_time 0",
                "sv_rcon_password rltraining",
            ]

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.processes.append(proc)
            self.shm_names.append(shm_name)
            print(f"  Server {i}: port={port} shm={shm_name} pid={proc.pid}")

            # Small delay between server starts to avoid port binding race conditions
            time.sleep(0.2)

        # Wait for servers to initialize
        print("Waiting for servers to initialize...")
        time.sleep(3)

        return self.shm_names

    def stop(self):
        """Stop all servers."""
        print("Stopping servers...")
        for i, proc in enumerate(self.processes):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.processes.clear()

    def check_health(self) -> int:
        """Check how many servers are still running."""
        return sum(1 for p in self.processes if p.poll() is None)


class MultiSharedMemoryReader:
    """Reads transitions from multiple shared memory regions."""

    def __init__(self, shm_names: List[str]):
        self.readers: List[SharedMemoryReader] = []
        for name in shm_names:
            reader = SharedMemoryReader(name)
            self.readers.append(reader)

    def open_all(self) -> int:
        """Open all shared memory regions. Returns count of successfully opened."""
        opened = 0
        for reader in self.readers:
            if reader.open():
                opened += 1
        return opened

    def close_all(self):
        """Close all shared memory regions."""
        for reader in self.readers:
            reader.close()

    def read_all_transitions(self, max_per_reader: int = 1024) -> List[Transition]:
        """Read transitions from all readers."""
        all_transitions = []
        for reader in self.readers:
            if reader.mmap_obj:
                transitions = reader.read_transitions(max_count=max_per_reader)
                all_transitions.extend(transitions)
        return all_transitions

    def signal_all_new_model(self, model_path: str):
        """Signal all servers that a new model is ready."""
        abs_path = os.path.abspath(model_path)
        for reader in self.readers:
            if reader.mmap_obj:
                reader.signal_new_model(abs_path)


def backup_existing_run(model_path: str, checkpoint_path: str, log_dir: str):
    """Backup existing model, checkpoint, and logs to timestamped directory."""
    # Check if any files exist
    files_exist = (
        os.path.exists(model_path) or
        os.path.exists(checkpoint_path) or
        os.path.exists(f"{model_path}.data")
    )

    if not files_exist:
        return None

    # Create backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/{timestamp}_run"
    os.makedirs(backup_dir, exist_ok=True)

    # Backup files
    backed_up = []
    for src in [model_path, f"{model_path}.data", checkpoint_path]:
        if os.path.exists(src):
            dst = os.path.join(backup_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            backed_up.append(os.path.basename(src))

    print(f"Backed up existing files to: {backup_dir}/")
    print(f"  Files: {', '.join(backed_up)}")

    return backup_dir


def create_fresh_model(model_path: str):
    """Create a fresh randomly initialized model."""
    model = PolicyNetwork()
    trainer = PPOTrainer(model, lr=1e-4)
    trainer.export_onnx(model_path)
    print(f"Created fresh model: {model_path}")


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description='Multi-server PPO training for DDNet RL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Server configuration
    parser.add_argument('--num-servers', type=int, default=4,
                        help='Number of parallel servers (each with 127 bots)')
    parser.add_argument('--server-executable', type=str, default='./DDNet-Server',
                        help='Path to DDNet-Server executable')
    parser.add_argument('--base-port', type=int, default=8303,
                        help='Base port for servers')
    parser.add_argument('--map', type=str, default='Tutorial',
                        help='Map to load')
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='Only act every N frames (reduces transition rate, decorrelates samples)')

    # Training configuration
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8192,
                        help='Batch size for updates')
    parser.add_argument('--update-interval', type=float, default=0.1,
                        help='Seconds between reading transitions')
    parser.add_argument('--export-interval', type=int, default=10,
                        help='Updates between ONNX exports')

    # Paths
    parser.add_argument('--model-path', type=str, default='model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint.pt',
                        help='Path to checkpoint')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='TensorBoard log directory')

    # Resume / spawn control
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint')
    parser.add_argument('--no-spawn', action='store_true',
                        help='Do not spawn servers (connect to existing)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not backup existing files when starting fresh')

    args = parser.parse_args()

    # Calculate total bots
    bots_per_server = 127
    total_bots = args.num_servers * bots_per_server

    print()
    print("=" * 70)
    print("  DDNet RL Training - Multi-Server Mode")
    print("=" * 70)
    print(f"  Servers:        {args.num_servers}")
    print(f"  Bots/server:    {bots_per_server}")
    print(f"  Total bots:     {total_bots}")
    print(f"  Frame skip:     {args.frame_skip}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Export every:   {args.export_interval} updates")
    print("=" * 70)
    print()

    # Handle model initialization
    if args.resume:
        if not os.path.exists(args.checkpoint_path):
            print(f"ERROR: Cannot resume - checkpoint not found: {args.checkpoint_path}")
            return 1
        print(f"Resuming from: {args.checkpoint_path}")
    else:
        # Starting fresh - backup existing files
        if not args.no_backup:
            backup_existing_run(args.model_path, args.checkpoint_path, args.log_dir)

        # Create fresh model
        create_fresh_model(args.model_path)

    # Initialize model and trainer
    model = PolicyNetwork()
    trainer = PPOTrainer(model, lr=args.lr, minibatch_size=256)

    if args.resume:
        trainer.load_checkpoint(args.checkpoint_path)
        print("Checkpoint loaded.")

    # TensorBoard
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.num_servers:02d}srv_{total_bots:04d}bots"
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    print(f"TensorBoard: {args.log_dir}/{run_name}")
    print()

    # Start servers
    server_manager = None
    if not args.no_spawn:
        server_manager = MultiServerManager(
            num_servers=args.num_servers,
            server_executable=args.server_executable,
            base_port=args.base_port,
            map_name=args.map,
            frame_skip=args.frame_skip,
        )
        shm_names = server_manager.start()
    else:
        shm_names = [f"/ddnet_rl_training_{i}" for i in range(args.num_servers)]
        print(f"Connecting to existing servers...")

    # Open shared memory readers
    print("Opening shared memory...")
    multi_reader = MultiSharedMemoryReader(shm_names)

    num_expected = len(shm_names)
    for attempt in range(60):  # Wait up to 60 seconds for all servers
        opened = multi_reader.open_all()
        print(f"  Connected to {opened}/{num_expected} servers...", end='\r')
        if opened >= num_expected:
            print(f"  Connected to {opened}/{num_expected} servers    ")
            break
        time.sleep(1)
    else:
        # Accept partial connection if we have at least half
        opened = multi_reader.open_all()
        if opened >= num_expected // 2:
            print(f"\nWarning: Only {opened}/{num_expected} servers connected, continuing...")
        else:
            print(f"\nERROR: Only {opened}/{num_expected} servers connected")
            if server_manager:
                server_manager.stop()
            return 1

    # Signal handlers
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\n\nInterrupted! Saving and cleaning up...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Training state
    update_count = 0
    global_step = 0
    transition_buffer = []
    start_time = time.time()
    last_print_time = start_time

    # Episode tracking
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = {}  # bot_id -> cumulative reward
    current_episode_lengths = {}  # bot_id -> length

    max_buffer_size = args.batch_size * 4

    print()
    print("Training started. Press Ctrl+C to stop.")
    print("-" * 70)

    try:
        while running:
            # Read transitions from all servers
            new_transitions = multi_reader.read_all_transitions(max_per_reader=8192)

            for trans in new_transitions:
                transition_buffer.append(trans)
                global_step += 1

                # Track episode stats
                bot_id = trans.bot_id
                if bot_id not in current_episode_rewards:
                    current_episode_rewards[bot_id] = 0.0
                    current_episode_lengths[bot_id] = 0

                current_episode_rewards[bot_id] += trans.reward
                current_episode_lengths[bot_id] += 1

                if trans.done or trans.truncated:
                    episode_rewards.append(current_episode_rewards[bot_id])
                    episode_lengths.append(current_episode_lengths[bot_id])
                    current_episode_rewards[bot_id] = 0.0
                    current_episode_lengths[bot_id] = 0

            # Limit buffer size
            if len(transition_buffer) > max_buffer_size:
                transition_buffer = transition_buffer[-max_buffer_size:]

            # Run update if we have enough transitions
            if len(transition_buffer) >= args.batch_size:
                batch = transition_buffer[:args.batch_size]
                transition_buffer = transition_buffer[args.batch_size:]

                stats = trainer.update(batch)
                update_count += 1

                # Calculate metrics
                elapsed = time.time() - start_time
                sps = global_step / elapsed if elapsed > 0 else 0

                # Episode stats (last 100 episodes)
                if episode_rewards:
                    mean_reward = np.mean(episode_rewards[-100:])
                    mean_length = np.mean(episode_lengths[-100:])
                else:
                    mean_reward = 0.0
                    mean_length = 0.0

                # Log to TensorBoard
                writer.add_scalar('charts/SPS', sps, global_step)
                writer.add_scalar('charts/learning_rate', args.lr, global_step)
                writer.add_scalar('losses/policy_loss', stats['policy_loss'], global_step)
                writer.add_scalar('losses/value_loss', stats['value_loss'], global_step)
                writer.add_scalar('losses/entropy', stats['entropy'], global_step)
                writer.add_scalar('losses/total_loss', stats['loss'], global_step)

                if episode_rewards:
                    writer.add_scalar('charts/episodic_return', mean_reward, global_step)
                    writer.add_scalar('charts/episodic_length', mean_length, global_step)
                    writer.add_scalar('charts/episodes', len(episode_rewards), global_step)

                # Print progress
                print(f"Update {update_count:5d} | "
                      f"Step {global_step:10,d} | "
                      f"SPS {sps:5.0f} | "
                      f"Reward {mean_reward:8.2f} | "
                      f"Len {mean_length:6.0f} | "
                      f"Loss {stats['loss']:8.4f} | "
                      f"Ent {stats['entropy']:.2f}")

                # Export model periodically
                if update_count % args.export_interval == 0:
                    temp_path = args.model_path + '.new'
                    trainer.export_onnx(temp_path)
                    os.rename(temp_path, args.model_path)
                    multi_reader.signal_all_new_model(args.model_path)

                    # IMPORTANT: Clear transition buffer after model export!
                    # With PPO importance sampling, stale transitions from the old policy
                    # cause bad ratio estimates and training instability.
                    old_buffer_size = len(transition_buffer)
                    transition_buffer = []

                    # Save checkpoint
                    trainer.save_checkpoint(args.checkpoint_path)

                    print(f"  -> Model exported, {args.num_servers} servers signaled, cleared {old_buffer_size} stale transitions")

                # Check server health periodically
                if server_manager and update_count % 50 == 0:
                    running_servers = server_manager.check_health()
                    if running_servers < args.num_servers:
                        print(f"  WARNING: Only {running_servers}/{args.num_servers} servers running!")

            time.sleep(args.update_interval)

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        print()
        print("-" * 70)
        print("Saving final checkpoint...")
        trainer.save_checkpoint(args.checkpoint_path)
        trainer.export_onnx(args.model_path)

        multi_reader.close_all()

        if server_manager:
            server_manager.stop()

        writer.close()

        # Summary
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("  Training Summary")
        print("=" * 70)
        print(f"  Duration:       {format_time(elapsed)}")
        print(f"  Updates:        {update_count:,}")
        print(f"  Steps:          {global_step:,}")
        print(f"  Episodes:       {len(episode_rewards):,}")
        print(f"  Avg SPS:        {global_step / elapsed:.0f}")
        if episode_rewards:
            print(f"  Final reward:   {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  Checkpoint:     {args.checkpoint_path}")
        print(f"  Model:          {args.model_path}")
        print("=" * 70)


if __name__ == '__main__':
    main()
