#!/usr/bin/env python3
"""
Adaptive overnight benchmark suite for DDNet RL training.
Runs experiments sequentially, analyzes results, and adapts subsequent experiments.
"""
import subprocess
import time
import os
import re
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

SCRIPT_DIR = Path(__file__).parent / "src/game/client/rl/python"
LOG_DIR = Path(__file__).parent / "benchmark_logs"
RESULTS_FILE = LOG_DIR / "summary.json"
REPORT_FILE = LOG_DIR / "report.txt"

RUN_MINUTES = 35
MAX_TOTAL_HOURS = 9.5

# Python from the ddnet-rl conda env
PYTHON = str(Path.home() / "miniconda3/envs/ddnet-rl/bin/python")

# Common flags for all experiments
BASE_FLAGS = [
    PYTHON, str(SCRIPT_DIR / "train.py"),
    "--server-side", "--num-servers", "12", "--num-envs", "127",
    "--sync-update", "--total-timesteps", "20000000000",
    "--frame-skip", "3", "--log-freq", "1",
]


@dataclass
class ExperimentResult:
    name: str
    description: str
    flags: List[str]
    fps_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    length_history: List[float] = field(default_factory=list)
    ru_history: List[tuple] = field(default_factory=list)
    total_steps: int = 0
    total_updates: int = 0
    peak_reward: float = float('-inf')
    avg_fps: float = 0.0
    avg_reward_last10: float = 0.0
    avg_r_time: float = 0.0
    avg_u_time: float = 0.0
    wall_time_seconds: float = 0.0
    error: Optional[str] = None


def cleanup():
    """Kill leftover servers and clean SHM."""
    subprocess.run(["pkill", "-9", "-f", "DDNet-Server"],
                   capture_output=True, timeout=10)
    import glob
    for f in glob.glob("/dev/shm/ddnet_rl_server*"):
        try:
            os.unlink(f)
        except OSError:
            pass
    time.sleep(5)


def parse_log(log_file: Path) -> ExperimentResult:
    """Parse training log to extract metrics."""
    result = ExperimentResult(name="", description="", flags=[])

    if not log_file.exists():
        result.error = "Log file not found"
        return result

    text = log_file.read_text()
    # Match lines like: Update    5 | Step   975,360 | FPS  17483 | Reward  -11.79 | Length    380 | PG -0.004 | Ent 5.308 | R/U 3.9/7.1s
    pattern = r'Update\s+(\d+)\s*\|\s*Step\s+([\d,]+)\s*\|\s*FPS\s+([\d.]+)\s*\|\s*Reward\s+([-\d.]+)\s*\|\s*Length\s+([\d.]+)\s*\|.*?R/U\s+([\d.]+)/([\d.]+)s'

    for m in re.finditer(pattern, text):
        update = int(m.group(1))
        step = int(m.group(2).replace(',', ''))
        fps = float(m.group(3))
        reward = float(m.group(4))
        length = float(m.group(5))
        r_time = float(m.group(6))
        u_time = float(m.group(7))

        result.fps_history.append(fps)
        result.reward_history.append(reward)
        result.length_history.append(length)
        result.ru_history.append((r_time, u_time))
        result.total_updates = update
        result.total_steps = step

    if result.fps_history:
        result.peak_reward = max(result.reward_history)
        last10 = result.reward_history[-10:]
        result.avg_reward_last10 = sum(last10) / len(last10)
        last10_fps = result.fps_history[-10:]
        result.avg_fps = sum(last10_fps) / len(last10_fps)
        last10_ru = result.ru_history[-10:]
        result.avg_r_time = sum(r for r, u in last10_ru) / len(last10_ru)
        result.avg_u_time = sum(u for r, u in last10_ru) / len(last10_ru)

    # Check for errors
    if "Error" in text or "Traceback" in text:
        # Get last few lines around error
        lines = text.strip().split('\n')
        error_lines = [l for l in lines if 'Error' in l or 'Traceback' in l]
        if error_lines:
            result.error = error_lines[-1][:200]

    return result


def run_experiment(name: str, description: str, extra_flags: List[str],
                   run_minutes: int = RUN_MINUTES) -> ExperimentResult:
    """Run a single experiment with timeout."""
    log_file = LOG_DIR / f"{name}.log"
    flags = BASE_FLAGS + extra_flags + [
        "--run-name", f"bench_{name}",
        "--log-dir", str(LOG_DIR / "runs"),
    ]

    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Starting: {name}")
    print(f"  {description}")
    print(f"  Extra flags: {' '.join(extra_flags)}")
    print(f"  Duration: {run_minutes} minutes")
    print(f"{'='*60}")

    cleanup()

    wall_start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(flags, stdout=f, stderr=subprocess.STDOUT, env=env)
            try:
                proc.wait(timeout=run_minutes * 60)
            except subprocess.TimeoutExpired:
                # Graceful shutdown
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
    except Exception as e:
        print(f"  ERROR: {e}")

    wall_time = time.time() - wall_start

    # Wait for cleanup
    time.sleep(10)
    cleanup()

    # Parse results
    result = parse_log(log_file)
    result.name = name
    result.description = description
    result.flags = extra_flags
    result.wall_time_seconds = wall_time

    # Print summary
    print(f"\n[{time.strftime('%H:%M:%S')}] Finished: {name}")
    print(f"  FPS: {result.avg_fps:.0f} | Peak Reward: {result.peak_reward:.1f} | "
          f"Avg Reward (last 10): {result.avg_reward_last10:.1f}")
    print(f"  R/U: {result.avg_r_time:.1f}/{result.avg_u_time:.1f}s | "
          f"Updates: {result.total_updates} | Steps: {result.total_steps:,}")
    if result.error:
        print(f"  WARNING: {result.error}")

    return result


def generate_report(results: List[ExperimentResult]):
    """Generate human-readable analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"OVERNIGHT BENCHMARK REPORT - {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Hardware: RTX 3080 10GB, Ryzen 7 5800X3D 8c/16t, 32GB RAM")
    lines.append("=" * 70)
    lines.append("")

    # Summary table
    lines.append("RESULTS SUMMARY")
    lines.append("-" * 70)
    lines.append(f"{'Name':<28} {'FPS':>6} {'Peak Rew':>9} {'Avg Rew':>8} {'R/U':>10} {'Steps':>12}")
    lines.append("-" * 70)

    for r in results:
        ru = f"{r.avg_r_time:.1f}/{r.avg_u_time:.1f}s"
        lines.append(
            f"{r.name:<28} {r.avg_fps:>6.0f} {r.peak_reward:>9.1f} "
            f"{r.avg_reward_last10:>8.1f} {ru:>10} {r.total_steps:>12,}"
        )

    lines.append("-" * 70)
    lines.append("")

    # Detailed analysis per experiment
    lines.append("DETAILED RESULTS")
    lines.append("=" * 70)
    for r in results:
        lines.append(f"\n--- {r.name} ---")
        lines.append(f"  Description: {r.description}")
        lines.append(f"  Flags: {' '.join(r.flags)}")
        lines.append(f"  Wall time: {r.wall_time_seconds/60:.1f} min")
        lines.append(f"  Avg FPS (last 10 updates): {r.avg_fps:.0f}")
        lines.append(f"  Peak reward: {r.peak_reward:.2f}")
        lines.append(f"  Avg reward (last 10 updates): {r.avg_reward_last10:.2f}")
        lines.append(f"  Avg R/U time: {r.avg_r_time:.1f}/{r.avg_u_time:.1f}s")
        lines.append(f"  Total updates: {r.total_updates}")
        lines.append(f"  Total steps: {r.total_steps:,}")
        if r.error:
            lines.append(f"  ERROR: {r.error}")

        # Reward trajectory (first, 25%, 50%, 75%, last)
        if len(r.reward_history) >= 5:
            n = len(r.reward_history)
            trajectory = [
                r.reward_history[0],
                r.reward_history[n//4],
                r.reward_history[n//2],
                r.reward_history[3*n//4],
                r.reward_history[-1],
            ]
            lines.append(f"  Reward trajectory: {' -> '.join(f'{v:.1f}' for v in trajectory)}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")

    report = "\n".join(lines)
    REPORT_FILE.write_text(report)
    print(f"\nReport written to: {REPORT_FILE}")
    return report


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "runs").mkdir(exist_ok=True)

    all_results: List[ExperimentResult] = []
    global_start = time.time()

    def time_remaining():
        elapsed_hours = (time.time() - global_start) / 3600
        return MAX_TOTAL_HOURS - elapsed_hours

    def run_and_record(name, description, flags, run_minutes=RUN_MINUTES):
        if time_remaining() < (run_minutes / 60 + 0.1):
            print(f"\nSkipping {name}: only {time_remaining():.1f}h remaining")
            return None
        result = run_experiment(name, description, flags, run_minutes)
        all_results.append(result)
        # Save incremental results
        save_results(all_results)
        generate_report(all_results)
        return result

    def save_results(results):
        data = []
        for r in results:
            d = {
                'name': r.name, 'description': r.description,
                'flags': r.flags, 'avg_fps': r.avg_fps,
                'peak_reward': r.peak_reward,
                'avg_reward_last10': r.avg_reward_last10,
                'avg_r_time': r.avg_r_time, 'avg_u_time': r.avg_u_time,
                'total_steps': r.total_steps, 'total_updates': r.total_updates,
                'wall_time_seconds': r.wall_time_seconds,
                'error': r.error,
                'reward_history': r.reward_history,
                'fps_history': r.fps_history,
            }
            data.append(d)
        RESULTS_FILE.write_text(json.dumps(data, indent=2))

    print(f"Starting overnight benchmarks at {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Max runtime: {MAX_TOTAL_HOURS} hours")
    print(f"Log directory: {LOG_DIR}")
    print()

    # ================================================================
    # PHASE 1: Core experiments (planned)
    # ================================================================

    # 1. Baseline - the user's current config
    r_baseline = run_and_record(
        "01_baseline_cnn_stack4",
        "Baseline: CNN tiles + player_state stack=4 (user's current config)",
        ["--use-cnn", "--n-player-state-stack", "4",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 2. No stacking (isolate stacking effect with same CNN architecture)
    r_nostack = run_and_record(
        "02_cnn_nostack",
        "CNN tiles, no player state stacking (isolate stacking overhead)",
        ["--use-cnn", "--n-player-state-stack", "1", "--n-frame-stack", "1",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 3. More temporal context
    r_stack8 = run_and_record(
        "03_cnn_stack8",
        "CNN tiles + player_state stack=8 (more temporal context)",
        ["--use-cnn", "--n-player-state-stack", "8",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 4. Embeddings (better tile representation, slower)
    r_embed = run_and_record(
        "04_embed_cnn_stack4",
        "Learned tile embeddings + CNN encoder + stack=4",
        ["--use-embeddings", "--tile-encoder", "cnn", "--n-player-state-stack", "4",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 5. Higher learning rate
    r_lr_high = run_and_record(
        "05_lr_1e3",
        "Higher learning rate (1e-3 vs default 3e-4)",
        ["--use-cnn", "--n-player-state-stack", "4", "--learning-rate", "1e-3",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 6. Lower learning rate
    r_lr_low = run_and_record(
        "06_lr_1e4",
        "Lower learning rate (1e-4 vs default 3e-4)",
        ["--use-cnn", "--n-player-state-stack", "4", "--learning-rate", "1e-4",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 7. High entropy
    r_ent_high = run_and_record(
        "07_ent_high",
        "Higher entropy coefficient (0.05 vs default 0.01)",
        ["--use-cnn", "--n-player-state-stack", "4", "--ent-coef", "0.05",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 8. Low entropy
    r_ent_low = run_and_record(
        "08_ent_low",
        "Lower entropy coefficient (0.001 vs default 0.01)",
        ["--use-cnn", "--n-player-state-stack", "4", "--ent-coef", "0.001",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 9. Fewer minibatches = larger minibatch size (6096 vs 3048)
    r_mb32 = run_and_record(
        "09_mb32",
        "Fewer minibatches (32 vs 64) = 2x larger minibatch, half the forward/backward passes",
        ["--use-cnn", "--n-player-state-stack", "4", "--num-minibatches", "32",
         "--num-steps", "128", "--update-epochs", "4"],
    )

    # 10. Longer rollouts
    r_steps256 = run_and_record(
        "10_steps256",
        "Longer rollouts (256 vs 128 steps) = better advantage estimation",
        ["--use-cnn", "--n-player-state-stack", "4", "--num-steps", "256",
         "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 11. Larger network
    r_hidden512 = run_and_record(
        "11_hidden512",
        "Larger network (hidden_dim=512 vs 256)",
        ["--use-cnn", "--n-player-state-stack", "4", "--hidden-dim", "512",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # 12. CPU scaling test - fewer servers
    r_8srv = run_and_record(
        "12_8servers",
        "CPU scaling: 8 servers (1016 bots) vs 12 (1524 bots)",
        ["--use-cnn", "--n-player-state-stack", "4",
         "--num-servers", "8", "--num-envs", "127",
         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
    )

    # ================================================================
    # PHASE 2: Adaptive experiments based on Phase 1 findings
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE - Analyzing results for Phase 2")
    print("=" * 60)

    # Analyze what worked and design follow-up experiments
    if all_results:
        # Find best learning config by reward
        valid = [r for r in all_results if r.reward_history and not r.error]
        if valid:
            best_reward = max(valid, key=lambda r: r.avg_reward_last10)
            best_fps = max(valid, key=lambda r: r.avg_fps)

            print(f"\nBest reward: {best_reward.name} (avg last 10: {best_reward.avg_reward_last10:.1f})")
            print(f"Best FPS: {best_fps.name} ({best_fps.avg_fps:.0f})")

            # Adaptive experiment: combine best learning config with throughput optimizations
            # If embeddings learned much better, try smaller embed_dim for speed
            if r_embed and r_baseline:
                if (r_embed.avg_reward_last10 > r_baseline.avg_reward_last10 * 1.2):
                    print("\nEmbeddings showed >20% reward improvement, testing smaller embed_dim...")
                    run_and_record(
                        "13_embed_dim4",
                        "Adaptive: embeddings with smaller dim (4 vs 8) for speed",
                        ["--use-embeddings", "--tile-encoder", "cnn",
                         "--tile-embed-dim", "4", "--n-player-state-stack", "4",
                         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
                    )

            # If high entropy helped, try intermediate value
            if r_ent_high and r_baseline and r_ent_low:
                ent_rewards = [
                    (0.001, r_ent_low.avg_reward_last10 if r_ent_low else float('-inf')),
                    (0.01, r_baseline.avg_reward_last10 if r_baseline else float('-inf')),
                    (0.05, r_ent_high.avg_reward_last10 if r_ent_high else float('-inf')),
                ]
                best_ent = max(ent_rewards, key=lambda x: x[1])
                if best_ent[0] == 0.05:
                    print("\nHigh entropy was best, testing even higher (0.1)...")
                    run_and_record(
                        "13_ent_01",
                        "Adaptive: very high entropy (0.1) since 0.05 was best",
                        ["--use-cnn", "--n-player-state-stack", "4", "--ent-coef", "0.1",
                         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
                    )
                elif best_ent[0] == 0.001:
                    print("\nLow entropy was best, testing even lower (0.0001)...")
                    run_and_record(
                        "13_ent_00001",
                        "Adaptive: very low entropy (0.0001) since 0.001 was best",
                        ["--use-cnn", "--n-player-state-stack", "4", "--ent-coef", "0.0001",
                         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
                    )

            # If 256 steps helped, try 512
            if r_steps256 and r_baseline:
                if r_steps256.avg_reward_last10 > r_baseline.avg_reward_last10 * 1.1:
                    print("\nLonger rollouts helped, trying 512 steps...")
                    run_and_record(
                        "14_steps512",
                        "Adaptive: even longer rollouts (512 steps)",
                        ["--use-cnn", "--n-player-state-stack", "4", "--num-steps", "512",
                         "--num-minibatches", "64", "--update-epochs", "4"],
                    )

            # If hidden512 helped, try 3 layers
            if r_hidden512 and r_baseline:
                if r_hidden512.avg_reward_last10 > r_baseline.avg_reward_last10 * 1.1:
                    print("\nLarger network helped, trying 3 layers...")
                    run_and_record(
                        "14_3layers",
                        "Adaptive: 3 hidden layers instead of 2",
                        ["--use-cnn", "--n-player-state-stack", "4",
                         "--hidden-dim", "512", "--num-layers", "3",
                         "--num-steps", "128", "--num-minibatches", "64", "--update-epochs", "4"],
                    )

            # If fewer minibatches was fine, try even fewer
            if r_mb32 and r_baseline:
                if (r_mb32.avg_fps > r_baseline.avg_fps * 1.1 and
                    r_mb32.avg_reward_last10 >= r_baseline.avg_reward_last10 * 0.9):
                    print("\nFewer minibatches boosted FPS without hurting reward, trying 16...")
                    run_and_record(
                        "14_mb16",
                        "Adaptive: even fewer minibatches (16) for throughput",
                        ["--use-cnn", "--n-player-state-stack", "4", "--num-minibatches", "16",
                         "--num-steps", "128", "--update-epochs", "4"],
                    )

            # Combination run: take the best hyperparams found so far
            if len(all_results) >= 12 and time_remaining() > 1.0:
                # Find best lr, ent, architecture settings
                best_config_flags = list(best_reward.flags)  # Start from best overall config
                print(f"\nRunning combination of best settings (based on {best_reward.name})...")
                # Run for longer to see if gains compound
                run_and_record(
                    "15_best_combo",
                    f"Adaptive: best config from Phase 1 (based on {best_reward.name}), 45 min run",
                    best_config_flags,
                    run_minutes=45,
                )

    # ================================================================
    # Final report
    # ================================================================
    report = generate_report(all_results)
    print("\n\n")
    print(report)

    total_hours = (time.time() - global_start) / 3600
    print(f"\nTotal benchmark time: {total_hours:.1f} hours")
    print(f"Experiments completed: {len(all_results)}")


if __name__ == "__main__":
    main()
