#!/bin/bash
# Overnight hyperparameter benchmark suite
# Each experiment runs for ~35 minutes, 12 experiments total (~7-8 hours)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/src/game/client/rl/python" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/benchmark_logs"
RESULTS_FILE="$LOG_DIR/summary.txt"
RUN_MINUTES=35

mkdir -p "$LOG_DIR"

# Common flags
BASE="python $SCRIPT_DIR/train.py --server-side --num-servers 12 --num-envs 127 --sync-update --total-timesteps 20000000000 --frame-skip 3 --log-freq 1"

# Cleanup between runs
cleanup() {
    pkill -9 -f DDNet-Server 2>/dev/null || true
    rm -f /dev/shm/ddnet_rl_server* 2>/dev/null || true
    sleep 5
}

# Run a single experiment
run_experiment() {
    local name="$1"
    shift
    local flags="$@"
    local log_file="$LOG_DIR/${name}.log"

    echo "============================================================" | tee -a "$RESULTS_FILE"
    echo "[$(date '+%H:%M:%S')] Starting: $name" | tee -a "$RESULTS_FILE"
    echo "  Flags: $flags" | tee -a "$RESULTS_FILE"
    echo "============================================================" | tee -a "$RESULTS_FILE"

    cleanup

    # Run with timeout, send SIGTERM for graceful cleanup
    timeout --signal=TERM --kill-after=60 "${RUN_MINUTES}m" \
        $BASE $flags --run-name "bench_${name}" --log-dir "$LOG_DIR/runs" \
        > "$log_file" 2>&1 || true

    # Wait for cleanup
    sleep 10
    cleanup

    # Extract last 5 update lines for summary
    echo "" >> "$RESULTS_FILE"
    echo "--- Last 5 updates ---" >> "$RESULTS_FILE"
    grep "^Update" "$log_file" | tail -5 >> "$RESULTS_FILE" 2>/dev/null || echo "  (no updates found)" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    # Extract peak reward and FPS
    local peak_reward=$(grep "^Update" "$log_file" | awk -F'Reward' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    local avg_fps=$(grep "^Update" "$log_file" | tail -10 | awk -F'FPS' '{print $2}' | awk '{print $1}' | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print "N/A"}')
    local total_steps=$(grep "^Update" "$log_file" | tail -1 | awk -F'Step' '{print $2}' | awk '{print $1}' | tr -d ',')
    local total_updates=$(grep "^Update" "$log_file" | tail -1 | awk '{print $2}')

    echo "  Peak Reward: ${peak_reward:-N/A}" >> "$RESULTS_FILE"
    echo "  Avg FPS (last 10): ${avg_fps:-N/A}" >> "$RESULTS_FILE"
    echo "  Total Steps: ${total_steps:-N/A}" >> "$RESULTS_FILE"
    echo "  Total Updates: ${total_updates:-N/A}" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    echo "[$(date '+%H:%M:%S')] Finished: $name (FPS: ${avg_fps:-N/A}, Peak Reward: ${peak_reward:-N/A})" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
}

# Header
echo "========================================" > "$RESULTS_FILE"
echo "OVERNIGHT BENCHMARK - $(date)" >> "$RESULTS_FILE"
echo "Hardware: RTX 3080 10GB, Ryzen 7 5800X3D, 32GB RAM" >> "$RESULTS_FILE"
echo "Base config: 12 servers x 127 envs = 1524 bots, frame_skip=3" >> "$RESULTS_FILE"
echo "Run duration: ${RUN_MINUTES} minutes each" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# ========================================
# EXPERIMENT 1: Baseline CNN + stack=4 (control)
# ========================================
run_experiment "01_baseline_cnn_stack4" \
    --use-cnn --n-player-state-stack 4 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 2: CNN, no player state stacking
# Tests: is temporal stacking worth the overhead?
# ========================================
run_experiment "02_cnn_nostack" \
    --use-cnn --n-player-state-stack 1 --n-frame-stack 1 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 3: CNN + stack=8 (more temporal context)
# Tests: does more history help?
# ========================================
run_experiment "03_cnn_stack8" \
    --use-cnn --n-player-state-stack 8 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 4: Embeddings + CNN encoder + stack=4
# Tests: learned tile representations vs normalized floats
# ========================================
run_experiment "04_embed_cnn_stack4" \
    --use-embeddings --tile-encoder cnn --n-player-state-stack 4 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 5: Higher learning rate (1e-3)
# Tests: faster convergence vs instability
# ========================================
run_experiment "05_lr_1e3" \
    --use-cnn --n-player-state-stack 4 --learning-rate 1e-3 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 6: Lower learning rate (1e-4)
# Tests: slower but more stable learning
# ========================================
run_experiment "06_lr_1e4" \
    --use-cnn --n-player-state-stack 4 --learning-rate 1e-4 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 7: High entropy (0.05)
# Tests: more exploration
# ========================================
run_experiment "07_ent_high" \
    --use-cnn --n-player-state-stack 4 --ent-coef 0.05 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 8: Low entropy (0.001)
# Tests: more exploitation
# ========================================
run_experiment "08_ent_low" \
    --use-cnn --n-player-state-stack 4 --ent-coef 0.001 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 9: Fewer minibatches (32) = larger minibatch size
# Tests: gradient quality vs update throughput
# ========================================
run_experiment "09_mb32" \
    --use-cnn --n-player-state-stack 4 --num-minibatches 32 \
    --num-steps 128 --update-epochs 4

# ========================================
# EXPERIMENT 10: Longer rollouts (256 steps)
# Tests: better advantage estimation vs memory/throughput
# ========================================
run_experiment "10_steps256" \
    --use-cnn --n-player-state-stack 4 --num-steps 256 \
    --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 11: Larger network (hidden_dim=512)
# Tests: more model capacity
# ========================================
run_experiment "11_hidden512" \
    --use-cnn --n-player-state-stack 4 --hidden-dim 512 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

# ========================================
# EXPERIMENT 12: Fewer servers (8) - CPU scaling test
# Tests: is CPU saturated? What's the FPS/server ratio?
# ========================================
run_experiment "12_8servers" \
    --use-cnn --n-player-state-stack 4 --num-servers 8 --num-envs 127 \
    --num-steps 128 --num-minibatches 64 --update-epochs 4

echo "========================================" >> "$RESULTS_FILE"
echo "ALL EXPERIMENTS COMPLETE - $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

echo ""
echo "All experiments complete! Results in: $RESULTS_FILE"
echo "Detailed logs in: $LOG_DIR/"
