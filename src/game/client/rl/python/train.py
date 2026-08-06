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
import signal
import atexit
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# Global reference to environment for cleanup
_global_env = None
_global_writer = None
_cleanup_done = False


def cleanup_system_resources():
    """
    Force cleanup of DDNet servers and SHM files.
    This is the nuclear option - kills all DDNet-Server processes and removes SHM files.
    """
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    print("\n[Cleanup] Cleaning up system resources...")

    # Kill any DDNet-Server processes
    try:
        result = subprocess.run(
            ["pkill", "-9", "-f", "DDNet-Server"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("[Cleanup] Killed DDNet-Server processes")
    except Exception as e:
        print(f"[Cleanup] pkill failed: {e}")

    # Remove SHM files
    try:
        import glob
        shm_files = glob.glob("/dev/shm/ddnet_rl_server*")
        for f in shm_files:
            try:
                os.unlink(f)
            except:
                pass
        if shm_files:
            print(f"[Cleanup] Removed {len(shm_files)} SHM files")
    except Exception as e:
        print(f"[Cleanup] SHM cleanup failed: {e}")

    print("[Cleanup] Done")


def graceful_cleanup():
    """Try graceful cleanup first, then force cleanup."""
    global _global_env, _global_writer

    # Try graceful close first
    if _global_env is not None:
        try:
            _global_env.close()
        except:
            pass
        _global_env = None

    if _global_writer is not None:
        try:
            _global_writer.close()
        except:
            pass
        _global_writer = None

    # Always force cleanup to ensure nothing is left behind
    cleanup_system_resources()


def signal_handler(signum, frame):
    """Handle termination signals."""
    sig_name = signal.Signals(signum).name
    print(f"\n[Signal] Received {sig_name}, cleaning up...")
    graceful_cleanup()
    sys.exit(128 + signum)


# Register atexit handler (catches normal exits, exceptions, sys.exit)
atexit.register(graceful_cleanup)

# Register signal handlers
for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
    try:
        signal.signal(sig, signal_handler)
    except (OSError, ValueError):
        pass  # Some signals may not be available on all platforms

# Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx/40xx)
torch.set_float32_matmul_precision('high')

from teeworlds_vec_env import TeeWorldsVecEnv, EnvConfig
from teeworlds_server_env import TeeWorldsServerEnv, ServerEnvConfig, FrameStackWrapper
from ppo_hybrid import PPOAgent, PPOConfig
from episode_tracker import EpisodeTracker
from action_hold import ActionHold


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Teeworlds")

    # Config file (loaded first, CLI args override)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (CLI args override config values)")

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
    parser.add_argument("--debug-timing", action="store_true",
                        help="Print step timing debug info")

    # Embedding Policy (tile embeddings)
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use tile embeddings (raw tile IDs -> learned embeddings)")
    parser.add_argument("--tile-embed-dim", type=int, default=8,
                        help="Tile embedding dimension (default: 8)")
    parser.add_argument("--tile-encoder", type=str, default="mlp",
                        choices=["mlp", "cnn"],
                        help="Encoder for embedded tiles: mlp or cnn (default: mlp)")

    # Frame stacking for temporal context
    parser.add_argument("--n-frame-stack", type=int, default=1,
                        help="Number of tile frames to stack (default: 1, expensive - use for multi-agent)")
    parser.add_argument("--n-player-state-stack", type=int, default=8,
                        help="Number of player state frames to stack (default: 8, cheap - enables timing)")

    # Checkpoint respawning
    parser.add_argument("--checkpoint-respawn", action="store_true",
                        help="Enable adaptive checkpoint respawning (spawn at frontier)")
    parser.add_argument("--begin-bias", type=float, default=0.2,
                        help="Probability of spawning at map start (default: 0.2)")
    parser.add_argument("--top-fraction", type=float, default=0.3,
                        help="Fraction of best episodes for spawn sampling (default: 0.3)")
    parser.add_argument("--finish-progress", type=float, default=1.0,
                        help="Virtual finish line as normalised path progress. 1.0 = real end of "
                             "map; lower slices the map short for fast iteration (default: 1.0)")
    parser.add_argument("--semantic-tiles", action="store_true",
                        help="Fixed 12-channel semantic tile encoding instead of raw index/15. "
                             "Makes SOLID / NOHOOK / THROUGH orthogonal rather than neighbouring "
                             "scalars (roadmap.md 1.9). Implies --use-embeddings plumbing.")
    parser.add_argument("--frontier-curriculum", action="store_true",
                        help="Restore agents near the stall-derived frontier (Phase 2.3). "
                             "Distinct from --curriculum, which selects maps/segments.")
    parser.add_argument("--frontier-window", type=int, default=12,
                        help="Bins before the frontier to sample restore slots from")
    parser.add_argument("--waypoints-path", type=str, default=None,
                        help="Override waypoint file autodiscovery (waypoints/, then ghosts/)")
    parser.add_argument("--restore-beta", type=float, default=0.5,
                        help="Restore cell weight ∝ visits**beta. 1 = natural occupancy, "
                             "0 = uniform over cells, negative = prefer rare cells. "
                             "-0.5 put 82%% of restores into transient hooked states.")
    parser.add_argument("--max-variant-share", type=float, default=0.4,
                        help="Ceiling on the share of restores any one cell variant may absorb")
    parser.add_argument("--max-episode-steps-restored", type=int, default=0,
                        help="Separate episode cap for RESTORED episodes (0 = same as "
                             "--max-episode-steps). From-start episodes need length to reach "
                             "the frontier; restored ones start next to it and only delay the "
                             "next restore.")
    parser.add_argument("--stall-patience", type=int, default=0,
                        help="Truncate after N consecutive steps without beating the episode's "
                             "own high-water progress. 0 disables. Cannot cut a productive "
                             "attempt, so it is safe for slow obstacles unlike a short cap.")
    parser.add_argument("--frontier-share", type=float, default=0.3,
                        help="Share of archive restores reserved for bins BEYOND the bulk "
                             "exploration edge -- states reached only by clearing an obstacle "
                             "the agent usually cannot. 0 = old edge-capped behaviour.")
    parser.add_argument("--legacy-archive", action="store_true",
                        help="Ablation: key archive cells by progress bin alone (latest-wins), "
                             "reproducing the pre-variant behaviour.")
    parser.add_argument("--ratchet", action="store_true",
                        help="Ratchet shaping: the potential is the episode's high-water progress, "
                             "so backtracking costs only time. Removes the certain, immediate "
                             "penalty on the first step of any obstacle needing a run-up.")
    parser.add_argument("--action-hold", action="store_true",
                        help="Learned action hold: the policy picks how many decisions to hold "
                             "each action for. Makes multi-step commitments (hook swings) "
                             "discoverable — see action_hold.py.")
    parser.add_argument("--hold-ent-coef", type=float, default=0.0,
                        help="Extra entropy coefficient on the duration head alone. Stops the "
                             "policy training its own long holds away (measured 3.27 -> 2.19).")
    parser.add_argument("--hold-durations", type=str, default="1,2,4,8,16",
                        help="Comma-separated hold lengths in decisions (default: 1,2,4,8,16). "
                             "At frame_skip 3 that spans 0.06s to 0.96s.")

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

    # Two-pass parsing: first grab --config, then apply YAML as defaults
    args, remaining = parser.parse_known_args()
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
        # Convert YAML keys (underscores) to argparse format and set as defaults
        defaults = {}
        for key, value in config.items():
            argparse_key = key.replace("-", "_")
            defaults[argparse_key] = value
        parser.set_defaults(**defaults)
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    global _global_env, _global_writer

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
    _global_writer = writer

    # Log hyperparameters
    writer.add_text("hyperparameters", str(vars(args)))

    # Frame/player state stacking requires CNN mode (dict observations)
    needs_stacking = args.n_frame_stack > 1 or args.n_player_state_stack > 1
    if needs_stacking and not args.use_cnn:
        print(f"Note: Stacking (tiles={args.n_frame_stack}, player={args.n_player_state_stack}) requires CNN mode, enabling --use-cnn")
        args.use_cnn = True

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
            max_episode_steps_restored=args.max_episode_steps_restored,
            stall_patience=args.stall_patience,
            use_dict_obs=args.use_cnn,  # Enable dict observations for CNN
            use_embeddings=args.use_embeddings or args.semantic_tiles,  # int64 tile ids
            map_name=args.map,
            debug_timing=args.debug_timing,
            checkpoint_respawn=args.checkpoint_respawn,
            begin_bias=args.begin_bias,
            top_fraction=args.top_fraction,
            finish_progress=args.finish_progress,
            waypoints_path=args.waypoints_path,
            curriculum=args.frontier_curriculum,
            frontier_window=args.frontier_window,
            ratchet_shaping=args.ratchet,
            archive_variants=not args.legacy_archive,
            restore_beta=args.restore_beta,
            max_variant_share=args.max_variant_share,
            frontier_share=args.frontier_share,
        )
        # Use default server_path (resolved relative to teeworlds_server_env.py)
        env = TeeWorldsServerEnv(num_bots=args.num_envs, num_servers=args.num_servers, config=server_config)
        _global_env = env  # Register for global cleanup
        # Update num_envs to reflect total bots for training loop
        args.num_envs = total_bots

        # Wrap with frame stacking for temporal context
        if args.n_frame_stack > 1 or args.n_player_state_stack > 1:
            env = FrameStackWrapper(env, n_tile_stack=args.n_frame_stack, n_player_state_stack=args.n_player_state_stack)
            _global_env = env  # Update global reference to wrapper
            tile_coverage_ms = args.n_frame_stack * args.frame_skip * 20
            player_coverage_ms = args.n_player_state_stack * args.frame_skip * 20
            print(f"Frame stacking: tiles={args.n_frame_stack} (~{tile_coverage_ms}ms), player_state={args.n_player_state_stack} (~{player_coverage_ms}ms)")
    else:
        # Client-side environments (original approach)
        print(f"\nCreating CLIENT-SIDE environment with {args.num_envs} parallel instances...")

        # Stacking not supported for client-side, disable if requested
        if args.n_frame_stack > 1 or args.n_player_state_stack > 1:
            print("Warning: Frame stacking not supported for client-side. Use --server-side for stacking.")
            print("         Disabling stacking for this run.")
            args.n_frame_stack = 1
            args.n_player_state_stack = 1

        env_config = EnvConfig(
            ddnet_path=args.ddnet_path,
            server_path=args.server_path,
            sv_tickrate=args.sv_tickrate,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_episode_steps,
            # Per-type caps and stall patience are server-side-env only; the
            # legacy client-side EnvConfig has no such fields.
        )
        env = TeeWorldsVecEnv(num_envs=args.num_envs, config=env_config)
        _global_env = env  # Register for global cleanup

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

        hold_durations = ()
        if args.action_hold:
            hold_durations = tuple(int(x) for x in str(args.hold_durations).split(",") if x.strip())
            print(f"Action hold: ON, durations={hold_durations} decisions "
                  f"({[d * args.frame_skip for d in hold_durations]} ticks)")

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
            use_cnn=args.use_cnn,
            use_embeddings=args.use_embeddings or args.semantic_tiles,
            semantic_tiles=args.semantic_tiles,
            tile_embed_dim=args.tile_embed_dim,
            tile_encoder='cnn' if args.semantic_tiles else args.tile_encoder,
            n_frame_stack=args.n_frame_stack,
            n_player_state_stack=args.n_player_state_stack,
            hold_durations=hold_durations,
            hold_ent_coef=args.hold_ent_coef,
        )

        # Create agent
        agent = PPOAgent(
            obs_dim=obs_dim,
            num_envs=args.num_envs,
            config=ppo_config,
            device=str(device),
        )

        action_hold = None
        if hold_durations:
            action_hold = ActionHold(
                num_envs=args.num_envs,
                num_discrete=len(ppo_config.discrete_dims),
                continuous_dim=ppo_config.continuous_dim,
                durations=hold_durations,
            )

        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from: {args.resume}")
            agent.load(args.resume)

        # Warmup torch.compile with actual observation shapes
        # This triggers CUDA graph capture for reduce-overhead mode
        agent.warmup(obs, num_iters=10)

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
        if args.checkpoint_respawn:
            print(f"Checkpoint respawn: ON (begin_bias={args.begin_bias}, top_fraction={args.top_fraction})")
        print("-" * 60)

        start_time = time.time()
        global_step = agent.global_step
        starting_global_step = global_step  # Track starting point for FPS calculation

        # Calculate starting update number for LR annealing when resuming
        batch_size = args.num_steps * args.num_envs
        starting_update = global_step // batch_size

        # Episode tracking
        episode_rewards = []
        episode_lengths = []

        # Cross-run-comparable metrics. Reward curves stop being comparable the
        # moment the reward function changes; these don't (PLAN.md Phase 0.2/0.3).
        base_env = env.env if hasattr(env, 'env') else env
        tracker = None
        if getattr(base_env, 'path_progress', None) is not None:
            tracker = EpisodeTracker(base_env.path_progress.bounds)
            print(f"Metrics: {tracker}")

        for update in range(1, num_updates + 1):
            rollout_start = time.time()

            # Anneal learning rate (linear decay) - use global update count for resuming
            if args.total_timesteps > 0:
                effective_update = starting_update + update
                frac = 1.0 - (effective_update - 1) / num_updates
                frac = max(frac, 0.0)  # Clamp to 0 if past total_timesteps
                lr = frac * args.learning_rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = lr

            # Collect rollout
            n_bootstrapped = 0
            for step in range(args.num_steps):
                global_step += args.num_envs

                # Get action (optionally pause during inference to prevent stale actions)
                if args.server_side and args.sync_inference:
                    env.pause()
                t_inf_start = time.time()
                action, log_prob, value = agent.get_action(obs)
                t_inf = time.time() - t_inf_start
                if t_inf > 0.5:  # Log slow inference (>500ms)
                    print(f"  [SLOW] Inference took {t_inf*1000:.0f}ms at step {step}")
                if args.server_side and args.sync_inference:
                    env.unpause()

                # Latch the action for as long as the policy asked. `executed`
                # is what the game sees and what goes in the buffer; `fresh`
                # marks the rows the policy is allowed to be graded on.
                if action_hold is not None:
                    executed, fresh = action_hold.apply(action)
                else:
                    executed, fresh = action, None

                # Environment step
                next_obs, reward, done, truncated, infos = env.step(executed)

                # Time-limit bootstrapping. A truncated episode has not really
                # ended, so its last transition needs gamma * V(final obs) added
                # to the reward and must then be marked terminal. Without this,
                # GAE either propagated advantage into the NEXT episode's states
                # (mid-rollout, because the env auto-resets) or dropped the
                # continuation value entirely (at a rollout boundary, where
                # next_done already included truncation). Both were happening,
                # in opposite directions. It matters more the shorter episodes
                # get, which is the direction stall_patience pushes.
                boot = getattr(base_env, '_boot_mask', None)
                final_stacked = getattr(env, '_final_stacked', None)
                if boot is not None and final_stacked is not None and boot.any():
                    v_final = agent.value_of(final_stacked)
                    reward = reward.copy()
                    reward[boot] += args.gamma * v_final[boot]
                    n_bootstrapped += int(boot.sum())

                # Truncation counts as terminal HERE because its continuation
                # value is now carried by the reward above.
                agent.store_transition(step, obs, executed, log_prob, reward,
                                       done | truncated, value, fresh)

                # A reset invalidates the latch: the state the action was chosen
                # for is gone, and under the curriculum the agent may have been
                # restored somewhere else on the map entirely.
                if action_hold is not None:
                    action_hold.clear(done | truncated)

                obs = next_obs

                # Per-step stall accounting -> stall map -> frontier estimate.
                if tracker is not None and getattr(base_env, '_stalled', None) is not None:
                    tracker.record_step(base_env._step_positions,
                                        base_env._last_progress,
                                        base_env._stalled)
                    tracker.record_diagnostics(
                        base_env._step_positions, base_env._stuck, base_env._thrash,
                        base_env._frozen, base_env._heading_cos, base_env._heading_valid,
                        base_env._oscillated, base_env._dwell_done,
                        base_env._dwell_steps_done)

                # Track episodes
                ep_batch = []
                for info in infos:
                    if 'episode' in info:
                        ep = info['episode']
                        episode_rewards.append(ep['r'])
                        episode_lengths.append(ep['l'])
                        if tracker is not None and 'pos' in ep:
                            ep_batch.append(ep)
                if ep_batch:
                    tracker.record(
                        progress=np.array([e['progress'] for e in ep_batch], dtype=np.float32),
                        positions=np.stack([e['pos'] for e in ep_batch]),
                        from_start=np.array([e['from_start'] for e in ep_batch]),
                        finished=np.array([e['finished'] for e in ep_batch]),
                        lengths=np.array([e['l'] for e in ep_batch]),
                    )

            # Refresh the curriculum's frontier estimate from accumulated stall
            # statistics. Once per update is plenty — the frontier moves slowly,
            # and re-deriving it per step would just add noise.
            if tracker is not None and base_env.spawn_manager is not None and \
               hasattr(base_env.spawn_manager, 'update_frontier'):
                base_env.spawn_manager.update_frontier(tracker.frontier_bin(),
                                                       tracker.reached_bins(),
                                                       tracker.exploration_edge())
                # Archive occupancy is per-server and only the server knows it.
                # Without this the sampler would request cells that are empty on
                # some servers and get a silent fallback to the map spawn.
                if hasattr(base_env, 'refresh_snapshot_visits'):
                    base_env.refresh_snapshot_visits()

            # PPO update (optionally pause to prevent bots from acting with stale policy)
            rollout_time = time.time() - rollout_start
            update_start = time.time()
            if args.server_side and args.sync_update:
                env.pause()
            stats = agent.update(obs, done | truncated)
            if args.server_side and args.sync_update:
                env.unpause()
            update_time = time.time() - update_start
            agent.global_step = global_step

            # Console logging (every update)
            elapsed = time.time() - start_time
            fps = (global_step - starting_global_step) / elapsed

            if episode_rewards:
                mean_reward = np.mean(episode_rewards[-100:])
                mean_length = np.mean(episode_lengths[-100:])
                # Headline metrics first: finish rate and progress, not reward.
                head = ""
                if tracker is not None:
                    ts = tracker.stats()
                    if ts:
                        head = (f"Fin {ts.get('episodes/finish_rate_from_start', 0.0):5.1%} | "
                                f"Prog {ts['progress/p50']:.3f}/{ts['progress/max']:.3f} | "
                                f"Stk {ts.get('stall/stuck_fraction', 0.0):3.0%}/"
                                f"Thr {ts.get('stall/thrash_fraction', 0.0):3.0%} | "
                                f"Head {ts.get('stall/heading_cos', 0.0):+.2f} | "
                                f"Fr {ts.get('stall/frontier_progress', -1.0):.3f} | ")
                print(f"Update {update:5d} | Step {global_step:9,d} | "
                      f"FPS {fps:6.0f} | "
                      f"{head}"
                      f"Reward {mean_reward:7.2f} | "
                      f"Length {mean_length:6.0f} | "
                      f"PG {stats['policy_loss']:6.3f} | "
                      f"Ent {stats['entropy']:.3f} | "
                      f"R/U {rollout_time:.1f}/{update_time:.1f}s")
            else:
                print(f"Update {update:5d} | Step {global_step:9,d} | "
                      f"FPS {fps:6.0f} | "
                      f"PG {stats['policy_loss']:6.3f} | "
                      f"Ent {stats['entropy']:.3f} | "
                      f"R/U {rollout_time:.1f}/{update_time:.1f}s")

            # TensorBoard logging (every log_freq updates)
            if update % args.log_freq == 0:
                writer.add_scalar("charts/learning_rate", lr, global_step)
                writer.add_scalar("charts/fps", fps, global_step)
                writer.add_scalar("losses/policy_loss", stats['policy_loss'], global_step)
                writer.add_scalar("losses/value_loss", stats['value_loss'], global_step)
                writer.add_scalar("losses/entropy", stats['entropy'], global_step)
                writer.add_scalar("losses/approx_kl", stats['approx_kl'], global_step)
                writer.add_scalar("losses/clipfrac", stats['clipfrac'], global_step)

                if episode_rewards:
                    writer.add_scalar("charts/episodic_return", mean_reward, global_step)
                    writer.add_scalar("charts/episodic_length", mean_length, global_step)

                # Action usage + PER-HEAD entropy. The aggregate entropy number
                # cannot distinguish "policy is uniform everywhere" from "sharp on
                # movement, uniform on an aim head that currently does not matter".
                for nm, val in (('hook', getattr(base_env, '_act_hook', 0.0)),
                                ('jump', getattr(base_env, '_act_jump', 0.0)),
                                ('move', getattr(base_env, '_act_move', 0.0))):
                    writer.add_scalar(f"actions/{nm}_rate", val, global_step)
                try:
                    import torch as _t
                    from torch.distributions import Categorical as _C, Normal as _N
                    with _t.no_grad():
                        logits, mu, sd = agent.actor(obs)
                        names = ['direction', 'jump', 'hook', 'aim_sector', 'hold']
                        for nm, lg in zip(names, logits):
                            e = _C(logits=lg.float().clamp(-50, 50), validate_args=False).entropy().mean()
                            writer.add_scalar(f"entropy/{nm}", e.item(), global_step)
                        writer.add_scalar("entropy/aim_residual",
                                          _N(mu.float(), sd.float().clamp(min=1e-6),
                                             validate_args=False).entropy().sum(-1).mean().item(),
                                          global_step)
                except Exception as _e:
                    if update <= args.log_freq * 2:
                        print(f"  [warn] per-head entropy logging unavailable: {_e}")

                # Hold usage. If mean_duration collapses to 1 the policy is
                # declining the extra head and the experiment is a no-op — that
                # needs to be visible, not inferred from the edge curve.
                if action_hold is not None:
                    for k, v in action_hold.stats().items():
                        writer.add_scalar(k, v, global_step)
                writer.add_scalar("charts/bootstrapped_truncations",
                                  n_bootstrapped / max(1, args.num_steps * args.num_envs),
                                  global_step)

                # Headline metrics: finish rate, progress percentiles, death heatmap
                if tracker is not None:
                    for k, v in tracker.stats().items():
                        writer.add_scalar(k, v, global_step)
                    prog_vals = tracker.progress_values()
                    if len(prog_vals):
                        writer.add_histogram("progress/hist", prog_vals, global_step)
                    if update % (args.log_freq * 10) == 0:
                        for src, tag in (("deaths", "progress/death_heatmap"),
                                         ("stalls", "progress/stall_heatmap"),
                                         ("stuck", "progress/stuck_heatmap"),
                                         ("thrash", "progress/thrash_heatmap"),
                                         ("oscillation", "progress/oscillation_heatmap")):
                            img = tracker.heatmap_image(base_env.path_progress.xy, source=src)
                            if img is not None:
                                writer.add_image(tag, img, global_step, dataformats='HWC')

                # Log curriculum / respawn stats
                if hasattr(base_env, 'spawn_manager') and base_env.spawn_manager is not None:
                    # Both managers expose get_stats(); log whatever keys they give
                    # rather than assuming the checkpoint-era key names.
                    for k, v in base_env.spawn_manager.get_stats().items():
                        tag = k if '/' in k else f"charts/{k}"
                        writer.add_scalar(tag, v, global_step)

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
        # Cleanup - use graceful cleanup with forced resource cleanup
        _global_env = None  # Clear reference so atexit doesn't double-close
        _global_writer = None
        try:
            env.close()
        except:
            pass
        try:
            writer.close()
        except:
            pass
        # Force cleanup any remaining processes/SHM
        cleanup_system_resources()


if __name__ == "__main__":
    main()
