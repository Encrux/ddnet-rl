#!/usr/bin/env bash
set -euo pipefail

# DDNet RL Bare-Metal Deploy Script
# For Lambda Labs, SSH machines, or anywhere without Docker.
# Installs deps, builds DDNet-Server, sets up Python env.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== DDNet RL Bare-Metal Deploy ==="
echo "Project dir: $PROJECT_DIR"
echo ""

# --- Step 1: System dependencies (server-only build) ---

echo "[1/4] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    cargo \
    rustc \
    libcurl4-openssl-dev \
    libssl-dev \
    libsqlite3-dev \
    libpng-dev \
    zlib1g-dev \
    python3 \
    git

# --- Step 2: Build DDNet-Server ---

echo ""
echo "[2/4] Building DDNet-Server..."
mkdir -p "$PROJECT_DIR/build"
cd "$PROJECT_DIR/build"
cmake -GNinja \
    -DCLIENT=OFF \
    -DTOOLS=OFF \
    -DVULKAN=OFF \
    -DVIDEORECORDER=OFF \
    -DPREFER_BUNDLED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    ..
ninja DDNet-Server

echo "Built: $PROJECT_DIR/build/DDNet-Server"

# --- Step 3: Install uv and Python deps ---

echo ""
echo "[3/4] Setting up Python environment..."

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$PROJECT_DIR"
uv venv
uv sync

echo "Python env ready at $PROJECT_DIR/.venv"

# --- Step 4: Verify ---

echo ""
echo "[4/4] Verifying setup..."

# Check server binary
if [ -x "$PROJECT_DIR/build/DDNet-Server" ]; then
    echo "  DDNet-Server: OK"
else
    echo "  DDNet-Server: FAILED"
    exit 1
fi

# Check map
if [ -f "$PROJECT_DIR/build/data/maps/Tutorial.map" ]; then
    echo "  Tutorial map: OK"
else
    echo "  Tutorial map: MISSING (training may fail)"
fi

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME"
    "$PROJECT_DIR/.venv/bin/python" -c "import torch; print(f'  PyTorch CUDA: {torch.cuda.is_available()} (torch {torch.__version__})')"
else
    echo "  GPU: not detected (will train on CPU)"
fi

echo ""
echo "=== Deploy complete ==="
echo "Start training with:"
echo "  ./deploy/train.sh"
echo ""
echo "Or with custom args:"
echo "  ./deploy/train.sh --num-servers 24 --total-timesteps 50000000"
