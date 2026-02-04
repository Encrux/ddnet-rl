#!/usr/bin/env python3
"""
Quick profiling script for DDNet-RL inference performance.

Profiles inference without requiring a running server, using dummy observations.
Useful for quickly measuring GPU performance and identifying bottlenecks.

Usage:
    # Profile inference with CNN policy
    python profile_training.py --num-bots 16 --steps 200 --use-cnn

    # Profile with embeddings
    python profile_training.py --num-bots 16 --steps 200 --use-embeddings

    # Profile with PyTorch profiler for detailed GPU analysis
    python profile_training.py --num-bots 16 --steps 50 --use-cnn --torch-profile

    # Full profile with all options
    python profile_training.py --num-bots 64 --steps 200 --use-cnn --torch-profile --output ./profiles
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ppo_hybrid import PPOConfig, PPOAgent


def create_dummy_observation(num_bots: int, use_cnn: bool, use_embeddings: bool,
                              n_frame_stack: int = 1, n_player_state_stack: int = 1) -> dict | np.ndarray:
    """Create a dummy observation for profiling.

    Args:
        num_bots: Batch size
        use_cnn: Whether CNN policy is used
        use_embeddings: Whether embedding policy is used
        n_frame_stack: Number of tile frames stacked (for CNN)
        n_player_state_stack: Number of player state frames stacked
    """
    player_state_dim = 15 * n_player_state_stack

    if use_embeddings:
        # Embedding mode: raw tile IDs (int64)
        return {
            "tiles": np.random.randint(0, 256, (num_bots, 31, 43), dtype=np.int64),
            "player_state": np.random.rand(num_bots, player_state_dim).astype(np.float32),
            "checkpoint": np.random.rand(num_bots, 4).astype(np.float32),
        }
    elif use_cnn:
        # CNN mode: normalized tiles (float32)
        return {
            "tiles": np.random.rand(num_bots, n_frame_stack, 31, 43).astype(np.float32),
            "player_state": np.random.rand(num_bots, player_state_dim).astype(np.float32),
            "checkpoint": np.random.rand(num_bots, 4).astype(np.float32),
        }
    else:
        # MLP mode: flat observation
        return np.random.rand(num_bots, 68).astype(np.float32)


def profile_inference_only(agent: PPOAgent, obs, num_steps: int = 100, warmup: int = 10) -> np.ndarray:
    """Profile just the inference path."""
    print(f"\n=== Inference Profiling ({num_steps} steps, {warmup} warmup) ===")

    # Warmup
    for _ in range(warmup):
        agent.get_action(obs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profile
    timings = []
    for _ in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        action, _, _ = agent.get_action(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    timings = np.array(timings)
    print(f"Inference: mean={timings.mean():.2f}ms, "
          f"std={timings.std():.2f}ms, "
          f"p50={np.percentile(timings, 50):.2f}ms, "
          f"p95={np.percentile(timings, 95):.2f}ms, "
          f"p99={np.percentile(timings, 99):.2f}ms")
    print(f"Theoretical max FPS: {1000/timings.mean():.0f}")
    return timings


def profile_inference_detailed(agent: PPOAgent, obs, num_steps: int = 50) -> dict:
    """Profile inference with detailed breakdown."""
    print(f"\n=== Detailed Inference Breakdown ({num_steps} steps) ===")

    timings = {
        "obs_to_tensor_ms": [],
        "actor_forward_ms": [],
        "critic_forward_ms": [],
        "tensor_to_numpy_ms": [],
        "total_ms": [],
    }

    device = agent.device
    use_cnn = agent.use_cnn
    use_embeddings = agent.use_embeddings

    for _ in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total = time.perf_counter()

        # Obs -> Tensor
        t0 = time.perf_counter()
        obs_t = agent._obs_to_tensor(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["obs_to_tensor_ms"].append((time.perf_counter() - t0) * 1000)

        # Actor forward
        t0 = time.perf_counter()
        with torch.no_grad():
            action, log_prob, _ = agent.actor.get_action(obs_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["actor_forward_ms"].append((time.perf_counter() - t0) * 1000)

        # Critic forward
        t0 = time.perf_counter()
        with torch.no_grad():
            value = agent.critic(obs_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["critic_forward_ms"].append((time.perf_counter() - t0) * 1000)

        # Tensor -> Numpy
        t0 = time.perf_counter()
        action_np = {k: v.cpu().numpy() for k, v in action.items()}
        log_prob_np = log_prob.cpu().numpy()
        value_np = value.squeeze(-1).cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["tensor_to_numpy_ms"].append((time.perf_counter() - t0) * 1000)

        timings["total_ms"].append((time.perf_counter() - t_total) * 1000)

    # Print summary
    print(f"\n{'Component':<20} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("-" * 55)
    for name, key in [("Obs->Tensor", "obs_to_tensor_ms"),
                      ("Actor Forward", "actor_forward_ms"),
                      ("Critic Forward", "critic_forward_ms"),
                      ("Tensor->Numpy", "tensor_to_numpy_ms"),
                      ("Total", "total_ms")]:
        arr = np.array(timings[key])
        print(f"{name:<20} {arr.mean():>7.2f} {np.percentile(arr, 50):>8.2f} "
              f"{np.percentile(arr, 95):>8.2f} {np.percentile(arr, 99):>8.2f}")

    return timings


def profile_with_torch_profiler(agent: PPOAgent, obs, num_steps: int = 50, output_dir: Path = None):
    """Deep profiling with PyTorch profiler."""
    print(f"\n=== PyTorch Profiler ({num_steps} steps) ===")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_steps):
            action, _, _ = agent.get_action(obs)

    print("\nTop 15 CUDA operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\nTop 15 CPU operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export Chrome trace
        trace_path = output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"\nChrome trace saved to: {trace_path}")
        print(f"View with: chrome://tracing/ -> Load {trace_path}")

        # Export summary tables
        cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        (output_dir / "cuda_summary.txt").write_text(cuda_table)

        cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=30)
        (output_dir / "cpu_summary.txt").write_text(cpu_table)

        mem_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30)
        (output_dir / "memory_summary.txt").write_text(mem_table)

    return prof


def main():
    parser = argparse.ArgumentParser(description="Profile DDNet-RL inference performance")
    parser.add_argument("--num-bots", type=int, default=16,
                        help="Number of bots (batch size)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of steps to profile")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup steps")
    parser.add_argument("--use-cnn", action="store_true",
                        help="Use CNN feature extractor")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use tile embeddings")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--torch-profile", action="store_true",
                        help="Run PyTorch profiler for detailed GPU analysis")
    parser.add_argument("--detailed", action="store_true",
                        help="Profile with detailed inference breakdown")
    parser.add_argument("--output", type=str, default="./profiles",
                        help="Output directory for profiling results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, auto)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--n-frame-stack", type=int, default=1,
                        help="Number of tile frames to stack (default: 1)")
    parser.add_argument("--n-player-state-stack", type=int, default=1,
                        help="Number of player state frames to stack (default: 1)")
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dummy observation
    obs = create_dummy_observation(
        args.num_bots,
        args.use_cnn or args.use_embeddings,
        args.use_embeddings,
        n_frame_stack=args.n_frame_stack,
        n_player_state_stack=args.n_player_state_stack,
    )

    if isinstance(obs, dict):
        print(f"Observation: dict with tiles={obs['tiles'].shape}, "
              f"player_state={obs['player_state'].shape}, checkpoint={obs['checkpoint'].shape}")
    else:
        print(f"Observation: flat array {obs.shape}")

    # Determine obs_dim
    if isinstance(obs, dict):
        obs_dim = sum(np.prod(v.shape[1:]) for v in obs.values())
    else:
        obs_dim = obs.shape[1]

    # Create agent
    config = PPOConfig(
        hidden_dim=args.hidden_dim,
        use_cnn=args.use_cnn or args.use_embeddings,
        use_embeddings=args.use_embeddings,
        use_compile=not args.no_compile,
        n_frame_stack=args.n_frame_stack,
        n_player_state_stack=args.n_player_state_stack,
    )
    agent = PPOAgent(obs_dim, args.num_bots, config, device=device)

    # Warmup torch.compile
    print("\nWarming up model...")
    agent.warmup(obs, num_iters=args.warmup)

    # Output directory
    output_dir = Path(args.output)

    # Run profiles
    inf_timings = profile_inference_only(agent, obs, args.steps, warmup=0)

    if args.detailed:
        detailed_timings = profile_inference_detailed(agent, obs, min(args.steps, 100))

    if args.torch_profile:
        prof = profile_with_torch_profiler(
            agent, obs,
            num_steps=min(50, args.steps),
            output_dir=output_dir if args.torch_profile else None
        )

    # Save inference timings
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "inference_timings.npy", inf_timings)

    # Generate summary report
    report_lines = [
        "=" * 60,
        "DDNet-RL Inference Profile",
        "=" * 60,
        f"",
        f"Configuration:",
        f"  Batch size (num_bots): {args.num_bots}",
        f"  Device: {device}",
        f"  Policy: {'CNN' if args.use_cnn else ('Embeddings' if args.use_embeddings else 'MLP')}",
        f"  Hidden dim: {args.hidden_dim}",
        f"  torch.compile: {'disabled' if args.no_compile else 'enabled'}",
        f"",
        f"Results ({args.steps} steps):",
        f"  Mean:  {inf_timings.mean():.2f} ms",
        f"  Std:   {inf_timings.std():.2f} ms",
        f"  P50:   {np.percentile(inf_timings, 50):.2f} ms",
        f"  P95:   {np.percentile(inf_timings, 95):.2f} ms",
        f"  P99:   {np.percentile(inf_timings, 99):.2f} ms",
        f"  Min:   {inf_timings.min():.2f} ms",
        f"  Max:   {inf_timings.max():.2f} ms",
        f"",
        f"Theoretical max FPS: {1000/inf_timings.mean():.0f}",
        f"  At P50: {1000/np.percentile(inf_timings, 50):.0f} FPS",
        f"  At P95: {1000/np.percentile(inf_timings, 95):.0f} FPS",
        f"  At P99: {1000/np.percentile(inf_timings, 99):.0f} FPS",
        "=" * 60,
    ]

    report = "\n".join(report_lines)
    print(f"\n{report}")

    (output_dir / "inference_report.txt").write_text(report)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
