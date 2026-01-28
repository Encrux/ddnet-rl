# Teeworlds Reinforcement Learning Bot

## Overview

This module implements a reinforcement learning interface for Teeworlds/DDNet, allowing AI agents to play the game by modifying only the client side.

**Two observation modes are supported:**
- **Structured** (default): Game state vectors - headless compatible, faster for training
- **Rendered**: RGB pixel frames - requires display or Xvfb

---

## Architecture

```
┌─────────────────┐     Shared Memory (POSIX)   ┌──────────────────┐
│   DDNet Client  │  ◄────────────────────────► │   Python Agent   │
│   (C++ Game)    │   obs, reward, done, action │  (PyTorch/JAX)   │
└─────────────────┘                             └──────────────────┘
```

---

## Quick Start

### Structured Mode (Headless - Recommended for Training)

```bash
# Terminal 1: Start server
./build/DDNet-Server

# Terminal 2: Start client (works headless with -DHEADLESS_CLIENT=ON)
./build/DDNet "connect localhost:8303; rl_enable"

# Terminal 3: Run Python client
cd src/game/client/rl/python
python teeworlds_env.py --print
```

### Rendered Mode (Requires Display)

```bash
# Terminal 1: Start server
./build/DDNet-Server

# Terminal 2: Start client with display
./build/DDNet "connect localhost:8303; cl_overlay_entities 100; cl_showquads 0; cl_showhud 0; rl_enable render"

# OR with Xvfb for headless pixel capture:
xvfb-run -a ./build/DDNet "connect localhost:8303; rl_enable render 0 640 360"

# Terminal 3: View frames
python teeworlds_env.py --display
```

---

## Console Commands

| Command | Description |
|---------|-------------|
| `rl_enable` | Enable RL mode with structured observations (headless compatible) |
| `rl_enable state [env_id]` | Same as above, explicit structured mode |
| `rl_enable render [env_id] [width] [height]` | Enable with pixel observations (default 640x360) |
| `rl_disable` | Disable RL mode |
| `rl_status` | Show RL status and current mode |

---

## Observation Modes

### Structured Mode (Default)

Extracts game state directly from `CGameClient::m_Snap`. Fast, compact, headless-compatible.

**Size:** 1496 bytes per observation

| Field | Type | Description |
|-------|------|-------------|
| `tick` | uint32 | Game tick |
| `game_flags` | uint8 | Team game, CTF, etc. |
| `local_player` | SPlayerObs (36B) | Local player state |
| `other_players[15]` | SPlayerObs[] | Other visible players |
| `projectiles[32]` | SProjectileObs[] | Active projectiles |
| `flag_pos[2]` | vec2[] | CTF flag positions |

**SPlayerObs (36 bytes):**
- Position (x, y), Velocity (vx, vy)
- Direction, Jumped, HookState, Weapon
- Hook position, Hooked player ID
- Active, IsFrozen, Health, Armor

### Rendered Mode

Captures RGB pixel frames from OpenGL. Requires display or Xvfb.

**Size:** width × height × 3 bytes (e.g., 640×360×3 = 691,200 bytes)

---

## Folder Structure

```
src/game/client/rl/
├── README.md              # This file
├── rlbot.h                # Main RL bot component header
├── rlbot.cpp              # Main RL bot component implementation
├── observation.h          # Structured observation definitions
├── communication/
│   ├── comm_interface.h   # Abstract communication interface
│   ├── shm_comm.h         # Shared memory implementation
│   └── shm_comm.cpp
└── python/
    ├── teeworlds_env.py   # Python client (supports both modes)
    ├── stream_viewer.py   # Remote frame viewer
    └── requirements.txt   # Python dependencies
```

---

## Reward Functions

### Generic (all modes)
- `+1.0` per kill
- `-1.0` per death
- `-0.01` per tick (encourage speed)

### DDRace
- `+10.0` per checkpoint
- `-5.0` per death
- `+100.0` on finish
- `-0.001` per tick

### CTF
- `+50.0` flag capture
- `+10.0` flag return
- `+1.0` per kill

---

## Implementation Plan

### Phase 1: Basic Integration
1. [ ] Create `CRLBot` component
2. [ ] Implement shared memory communication
3. [ ] Build Python Gym wrapper
4. [ ] Test with random actions

### Phase 2: Observation & Reward
1. [ ] Implement structured observation extraction
2. [ ] Add nearby tile extraction (collision map)
3. [ ] Implement reward functions for DM/DDRace/CTF

### Phase 3: Training Pipeline
1. [ ] Integrate with Stable-Baselines3
2. [ ] Parallelize with multiple headless clients
3. [ ] Implement curriculum learning

### Phase 4: Evaluation
1. [ ] Test on various game modes
2. [ ] Compare to human players
3. [ ] Fine-tune observation space

---

## Building

```bash
# Build client (regular)
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Build headless client for training (no graphics window)
mkdir build-rl && cd build-rl
cmake -DHEADLESS_CLIENT=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## Usage

### Quick Start

```bash
# Terminal 1: Start server
./build/DDNet-Server

# Terminal 2: Start client with RL-friendly settings
./build/DDNet "connect localhost:8303; cl_overlay_entities 100; cl_showquads 0; cl_showhud 0; cl_showchat 0"

# In DDNet console (F1), enable RL mode:
rl_enable

# Terminal 3: Run Python client
cd src/game/client/rl/python
pip install -r requirements.txt
python teeworlds_env.py --display
```

### Console Commands

| Command | Description |
|---------|-------------|
| `rl_enable [env_id] [width] [height]` | Enable RL mode (default: env_id=0, 640x360 frames) |
| `rl_disable` | Disable RL mode |
| `rl_status` | Show RL status and current action |

### Recommended Client Settings

For a clean observation without UI clutter:

```
cl_overlay_entities 100   # Show game tiles (freeze, teleport, etc.)
cl_showquads 0            # Hide decorative backgrounds
cl_showhud 0              # Hide HUD
cl_showchat 0             # Hide chat
cl_show_others 0          # Hide other players (optional)
```

### Python Display/Stream

```bash
# Display frames locally
python teeworlds_env.py --display

# Stream frames over network (port 5555)
python teeworlds_env.py --stream

# View stream from another machine
python stream_viewer.py --host <ip>
```

---

## Headless Training

The DDNet client requires a graphics context to render frames. For headless training (no physical display), use **Xvfb** (X Virtual Framebuffer) which creates a virtual display in memory.

### Option 1: Xvfb (Recommended)

Install Xvfb:
```bash
# Ubuntu/Debian
sudo apt install xvfb

# Arch Linux
sudo pacman -S xorg-server-xvfb
```

Run DDNet with virtual display:
```bash
# Single client
xvfb-run -a -s "-screen 0 800x600x24" ./build/DDNet \
    "connect localhost:8303; cl_overlay_entities 100; cl_showquads 0; cl_showhud 0; cl_showchat 0; rl_enable"

# Or start Xvfb separately for multiple clients
Xvfb :99 -screen 0 800x600x24 &
export DISPLAY=:99

# Now run multiple clients on the same virtual display
./build/DDNet "connect localhost:8303; rl_enable 0" &
./build/DDNet "connect localhost:8303; rl_enable 1" &
./build/DDNet "connect localhost:8303; rl_enable 2" &
```

### Option 2: VirtualGL + Headless (GPU accelerated)

For faster training with GPU acceleration:
```bash
# Install VirtualGL
sudo apt install virtualgl

# Run with GPU offscreen rendering
vglrun -d :0 ./build/DDNet "connect localhost:8303; rl_enable"
```

### Autoexec Configuration

Create an autoexec file for RL-optimized settings:
```bash
# Create autoexec file
cat > ~/.teeworlds/autoexec_rl.cfg << 'EOF'
cl_overlay_entities 100
cl_showquads 0
cl_showhud 0
cl_showchat 0
gfx_vsync 0
gfx_refresh_rate 0
EOF
```

Run with autoexec:
```bash
xvfb-run -a ./build/DDNet -f autoexec_rl.cfg "connect localhost:8303; rl_enable"
```

### Start training server
```bash
./DDNet-Server "sv_map ctf5; sv_max_clients 16"
```

### Run Python training
```bash
cd src/game/client/rl/python
pip install -r requirements.txt
python train.py --env teeworlds --algo ppo
```

### Parallel Training Script

```bash
#!/bin/bash
# parallel_train.sh - Run N parallel training environments
NUM_ENVS=${1:-4}
SERVER_PORT=8303

# Start server
./build/DDNet-Server "sv_port $SERVER_PORT" &
SERVER_PID=$!
sleep 2

# Start Xvfb
Xvfb :99 -screen 0 800x600x24 &
XVFB_PID=$!
export DISPLAY=:99
sleep 1

# Launch clients
for i in $(seq 0 $((NUM_ENVS-1))); do
    ./build/DDNet "connect localhost:$SERVER_PORT; rl_enable $i" &
done

# Wait for Ctrl+C
trap "kill $SERVER_PID $XVFB_PID; pkill -f DDNet" INT
wait
```

---

## Key Integration Points

| File | Modification |
|------|--------------|
| `src/game/client/gameclient.h` | Add `CRLBot` component |
| `src/game/client/gameclient.cpp` | Register component |
| `src/game/client/components/controls.cpp` | Hook for action injection |
| `CMakeLists.txt` | Add new source files |

---

## Communication Protocol

### Message format (binary)

**Observation (C++ → Python):**
```
[4 bytes: tick]
[4 bytes: observation_size]
[N bytes: observation_data]
[4 bytes: reward (float)]
[1 byte: done]
[1 byte: truncated]
```

**Action (Python → C++):**
```
[1 byte: direction + 1]  // 0=left, 1=none, 2=right
[1 byte: jump]
[1 byte: hook]
[1 byte: fire]
[2 bytes: aim_angle]     // 0-65535 mapped to 0-2π
[1 byte: weapon]
```

---

## Notes

- Server modification is minimal: only simulation speed adjustment for faster training
- Client handles all RL logic
- Use Xvfb for headless training with frame capture
- Multiple clients can connect to same server for parallel training
