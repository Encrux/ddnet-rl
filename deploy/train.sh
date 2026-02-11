#!/usr/bin/env bash
set -euo pipefail

# DDNet RL Training Launcher
# Auto-detects hardware and launches training with benchmark-tuned defaults.
# All flags are overridable via command-line arguments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_DIR/src/game/client/rl/python"

# --- Hardware detection ---

NCPU=$(nproc)
NUM_SERVERS=$(( NCPU - 2 ))
if [ "$NUM_SERVERS" -lt 1 ]; then
    NUM_SERVERS=1
fi
if [ "$NUM_SERVERS" -gt 24 ]; then
    NUM_SERVERS=24
fi

echo "=== DDNet RL Training ==="
echo "CPUs: $NCPU (using $NUM_SERVERS servers)"

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "WARNING: nvidia-smi not found. Training will be slow on CPU."
fi

# --- Cleanup leftover processes/shm ---

echo ""
echo "Cleaning up leftover processes..."
pkill -9 -f DDNet-Server 2>/dev/null || true
sleep 0.5
rm -f /dev/shm/ddnet_rl_server* 2>/dev/null || true

# --- Find Python ---

if [ -d "$PROJECT_DIR/.venv" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: No Python found. Run 'uv venv && uv sync' first."
    exit 1
fi

echo "Python: $PYTHON"
echo ""

# --- Launch training ---

exec "$PYTHON" "$PYTHON_DIR/train.py" \
    --server-side \
    --num-servers "$NUM_SERVERS" \
    --use-cnn \
    --n-player-state-stack 4 \
    --num-steps 128 \
    --num-minibatches 64 \
    --update-epochs 4 \
    --frame-skip 3 \
    --learning-rate 3e-4 \
    --ent-coef 0.01 \
    "$@"
