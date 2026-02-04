"""
Profiling utilities for DDNet-RL training.

Provides detailed timing instrumentation at multiple granularities:
1. High-level breakdown: Rollout vs Update time
2. Per-step breakdown: Inference, env.step (write/wait/read), buffer storage
3. Deep GPU profiling: PyTorch profiler with CUDA kernels, memory, TensorBoard export
4. Statistical analysis: Percentiles, variance, outlier detection

Usage:
    python train.py --profile --profile-steps 100 --profile-output ./profiles/

    # Or standalone profiling:
    python profile_training.py --num-bots 16 --steps 200 --use-cnn
"""

import time
import json
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import torch


@dataclass
class StepTimings:
    """Timing data for a single training step."""
    inference_ms: float = 0.0
    env_step_ms: float = 0.0
    buffer_store_ms: float = 0.0
    total_ms: float = 0.0

    # Detailed inference breakdown (optional)
    obs_to_tensor_ms: float = 0.0
    actor_forward_ms: float = 0.0
    critic_forward_ms: float = 0.0
    tensor_to_numpy_ms: float = 0.0

    # Detailed env breakdown (optional)
    env_write_ms: float = 0.0
    env_wait_ms: float = 0.0
    env_read_ms: float = 0.0


@dataclass
class ProfileConfig:
    """Configuration for profiling run."""
    enabled: bool = False
    warmup_steps: int = 10       # Skip first N steps (JIT warmup)
    profile_steps: int = 100     # Number of steps to profile
    output_dir: str = "./profiles"
    tensorboard: bool = False    # Export to TensorBoard
    memory: bool = False         # Track CUDA memory allocation
    cuda_sync: bool = True       # Force CUDA sync for accurate timing


class TrainingProfiler:
    """
    Orchestrates profiling across training components.

    Collects timing data during training and generates reports with
    statistical summaries and bottleneck analysis.
    """

    def __init__(self, config: ProfileConfig):
        self.config = config
        self.step_timings: List[Dict[str, float]] = []
        self.update_timings: List[Dict[str, float]] = []
        self.torch_profiler = None
        self._step_count = 0
        self._profiling_active = False
        self._last_inference_ms: float = 0.0

    def should_profile_step(self) -> bool:
        """Check if current step should be profiled."""
        if not self.config.enabled:
            return False
        in_range = (self._step_count >= self.config.warmup_steps and
                    self._step_count < self.config.warmup_steps + self.config.profile_steps)
        return in_range

    def increment_step(self):
        """Increment step counter."""
        self._step_count += 1

    def record_step_timing(self, timing: StepTimings):
        """Record timing for a step."""
        if self.should_profile_step():
            self.step_timings.append(asdict(timing))

    def record_step_timing_dict(self, timing_dict: Dict[str, float]):
        """Record timing from a dictionary."""
        if self.should_profile_step():
            self.step_timings.append(timing_dict)

    def record_update_timing(self, rollout_ms: float, update_ms: float):
        """Record timing for a PPO update."""
        if self.config.enabled:
            self.update_timings.append({
                'rollout_ms': rollout_ms,
                'update_ms': update_ms,
                'total_ms': rollout_ms + update_ms,
            })

    @contextmanager
    def profile_inference(self):
        """Context manager for profiling inference with CUDA events."""
        if not self.config.enabled or not torch.cuda.is_available():
            yield
            return

        if self.config.cuda_sync:
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        yield

        end.record()
        if self.config.cuda_sync:
            torch.cuda.synchronize()
        self._last_inference_ms = start.elapsed_time(end)

    def get_last_inference_ms(self) -> float:
        """Get the last recorded inference time in ms."""
        return self._last_inference_ms

    def start_torch_profiler(self):
        """Start PyTorch profiler for detailed GPU analysis."""
        if not self.config.enabled or not self.config.tensorboard:
            return

        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.torch_profiler = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=self.config.memory,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(str(output_dir / "tensorboard")),
        )
        self.torch_profiler.__enter__()
        self._profiling_active = True
        print(f"PyTorch profiler started (output: {output_dir}/tensorboard)")

    def stop_torch_profiler(self):
        """Stop profiler and export results."""
        if self.torch_profiler is None or not self._profiling_active:
            return

        self.torch_profiler.__exit__(None, None, None)
        self._profiling_active = False

        output_dir = Path(self.config.output_dir)

        # Key averages table
        try:
            table = self.torch_profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=30
            )
            (output_dir / "cuda_summary.txt").write_text(table)

            # CPU time table
            cpu_table = self.torch_profiler.key_averages().table(
                sort_by="cpu_time_total", row_limit=30
            )
            (output_dir / "cpu_summary.txt").write_text(cpu_table)

            # Memory table
            if self.config.memory:
                mem_table = self.torch_profiler.key_averages().table(
                    sort_by="self_cuda_memory_usage", row_limit=30
                )
                (output_dir / "memory_summary.txt").write_text(mem_table)

            print(f"\nPyTorch profiler results saved to {output_dir}/")
            print(f"View with: tensorboard --logdir={output_dir}/tensorboard")
        except Exception as e:
            print(f"Warning: Could not export profiler tables: {e}")

    def export_timing_stats(self):
        """Export collected timing statistics."""
        if not self.step_timings:
            print("No timing data collected.")
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON export
        (output_dir / "step_timings.json").write_text(
            json.dumps(self.step_timings, indent=2)
        )

        if self.update_timings:
            (output_dir / "update_timings.json").write_text(
                json.dumps(self.update_timings, indent=2)
            )

        # Summary statistics
        summary = self._compute_summary()
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        # Human-readable report
        report = self._generate_report(summary)
        (output_dir / "report.txt").write_text(report)
        print(report)

    def _compute_summary(self) -> dict:
        """Compute summary statistics from collected timings."""
        def stats(values: List[float]) -> dict:
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }

        result = {
            "inference": stats([t.get("inference_ms", 0) for t in self.step_timings]),
            "env_step": stats([t.get("env_step_ms", 0) for t in self.step_timings]),
            "buffer_store": stats([t.get("buffer_store_ms", 0) for t in self.step_timings]),
            "total": stats([t.get("total_ms", 0) for t in self.step_timings]),
            "num_samples": len(self.step_timings),
        }

        # Add detailed inference breakdown if available
        if any(t.get("obs_to_tensor_ms", 0) > 0 for t in self.step_timings):
            result["obs_to_tensor"] = stats([t.get("obs_to_tensor_ms", 0) for t in self.step_timings])
            result["actor_forward"] = stats([t.get("actor_forward_ms", 0) for t in self.step_timings])
            result["critic_forward"] = stats([t.get("critic_forward_ms", 0) for t in self.step_timings])
            result["tensor_to_numpy"] = stats([t.get("tensor_to_numpy_ms", 0) for t in self.step_timings])

        # Add detailed env breakdown if available
        if any(t.get("env_write_ms", 0) > 0 for t in self.step_timings):
            result["env_write"] = stats([t.get("env_write_ms", 0) for t in self.step_timings])
            result["env_wait"] = stats([t.get("env_wait_ms", 0) for t in self.step_timings])
            result["env_read"] = stats([t.get("env_read_ms", 0) for t in self.step_timings])

        # Add update timings if available
        if self.update_timings:
            result["rollout"] = stats([t.get("rollout_ms", 0) for t in self.update_timings])
            result["ppo_update"] = stats([t.get("update_ms", 0) for t in self.update_timings])

        return result

    def _generate_report(self, summary: dict) -> str:
        """Generate human-readable profiling report."""
        lines = [
            "=" * 60,
            "DDNet-RL Training Profile Report",
            "=" * 60,
            "",
            f"Samples: {summary['num_samples']} steps",
            "",
            "Per-Step Breakdown (ms):",
            "-" * 55,
            f"{'Component':<18} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}",
            "-" * 55,
        ]

        total_mean = summary["total"]["mean"] if summary["total"]["mean"] > 0 else 1.0

        for name, key in [("Inference", "inference"), ("Env Step", "env_step"),
                          ("Buffer", "buffer_store"), ("Total", "total")]:
            s = summary[key]
            if key != "total":
                pct = f"({s['mean']/total_mean*100:.0f}%)"
            else:
                pct = ""
            lines.append(
                f"{name:<18} {s['mean']:>7.2f} {s['p50']:>8.2f} {s['p95']:>8.2f} {s['p99']:>8.2f}  {pct}"
            )

        # Detailed inference breakdown
        if "obs_to_tensor" in summary:
            lines.extend([
                "",
                "Inference Breakdown (ms):",
                "-" * 55,
            ])
            for name, key in [("Obs->Tensor", "obs_to_tensor"), ("Actor Forward", "actor_forward"),
                              ("Critic Forward", "critic_forward"), ("Tensor->Numpy", "tensor_to_numpy")]:
                s = summary[key]
                lines.append(f"{name:<18} {s['mean']:>7.2f} {s['p50']:>8.2f} {s['p95']:>8.2f} {s['p99']:>8.2f}")

        # Detailed env breakdown
        if "env_write" in summary:
            lines.extend([
                "",
                "Env Step Breakdown (ms):",
                "-" * 55,
            ])
            for name, key in [("Write Actions", "env_write"), ("Wait for Obs", "env_wait"),
                              ("Read Obs", "env_read")]:
                s = summary[key]
                lines.append(f"{name:<18} {s['mean']:>7.2f} {s['p50']:>8.2f} {s['p95']:>8.2f} {s['p99']:>8.2f}")

        # Update timing breakdown
        if "rollout" in summary:
            lines.extend([
                "",
                "Update Breakdown (ms):",
                "-" * 55,
            ])
            for name, key in [("Rollout", "rollout"), ("PPO Update", "ppo_update")]:
                s = summary[key]
                lines.append(f"{name:<18} {s['mean']:>7.2f} {s['p50']:>8.2f} {s['p95']:>8.2f} {s['p99']:>8.2f}")

        lines.extend([
            "",
            "Theoretical Max FPS:",
            "-" * 40,
        ])

        if summary["total"]["p50"] > 0:
            lines.append(f"  At P50 latency: {1000/summary['total']['p50']:.0f} FPS")
        if summary["total"]["p95"] > 0:
            lines.append(f"  At P95 latency: {1000/summary['total']['p95']:.0f} FPS")
        if summary["total"]["p99"] > 0:
            lines.append(f"  At P99 latency: {1000/summary['total']['p99']:.0f} FPS")

        lines.extend([
            "",
            "Bottleneck Analysis:",
            "-" * 40,
        ])

        # Identify bottleneck
        inf_pct = summary["inference"]["mean"] / total_mean * 100
        env_pct = summary["env_step"]["mean"] / total_mean * 100

        if inf_pct > 60:
            lines.append(f"  -> GPU-bound: Inference is {inf_pct:.0f}% of step time")
            lines.append("     Recommendations: TensorRT, smaller network, better GPU")
        elif env_pct > 60:
            lines.append(f"  -> Environment-bound: Env step is {env_pct:.0f}% of step time")
            lines.append("     Recommendations: Check server tick rate, reduce obs size")
        else:
            lines.append(f"  -> Balanced: Inference {inf_pct:.0f}%, Env {env_pct:.0f}%")
            lines.append("     Both components contribute significantly")

        # Check for high variance (outliers)
        inf_cv = summary["inference"]["std"] / max(0.001, summary["inference"]["mean"])
        env_cv = summary["env_step"]["std"] / max(0.001, summary["env_step"]["mean"])

        if inf_cv > 0.5:
            lines.append(f"  -> High inference variance (CV={inf_cv:.2f}): possible JIT or GC stalls")
        if env_cv > 0.5:
            lines.append(f"  -> High env variance (CV={env_cv:.2f}): possible I/O or scheduling issues")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def print_step_summary(self, step_timings: List[StepTimings]):
        """Print summary of step timings to console."""
        if not step_timings:
            return

        inf_times = [s.inference_ms for s in step_timings]
        env_times = [s.env_step_ms for s in step_timings]
        total_times = [s.total_ms for s in step_timings]

        print(f"\n=== Step Timing (n={len(step_timings)}) ===")
        print(f"Inference:  p50={np.percentile(inf_times, 50):.2f}ms  "
              f"p95={np.percentile(inf_times, 95):.2f}ms  "
              f"p99={np.percentile(inf_times, 99):.2f}ms")
        print(f"Env Step:   p50={np.percentile(env_times, 50):.2f}ms  "
              f"p95={np.percentile(env_times, 95):.2f}ms  "
              f"p99={np.percentile(env_times, 99):.2f}ms")

        mean_inf = np.mean(inf_times)
        mean_env = np.mean(env_times)
        mean_total = np.mean(total_times)

        if mean_total > 0:
            print(f"Breakdown:  inference={mean_inf:.1f}ms ({mean_inf/mean_total*100:.0f}%)  "
                  f"env={mean_env:.1f}ms ({mean_env/mean_total*100:.0f}%)")
