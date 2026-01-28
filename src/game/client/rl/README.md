# Teeworlds Reinforcement Learning Bot

## Overview

This module implements a reinforcement learning interface for Teeworlds/DDNet, allowing AI agents to play the game by modifying only the client side.

---

## Architecture

```
┌─────────────────┐     Shared Memory / ZMQ     ┌──────────────────┐
│   DDNet Client  │  ◄────────────────────────► │   Python Agent   │
│   (C++ Game)    │   obs, reward, done, action │  (PyTorch/JAX)   │
└─────────────────┘                             └──────────────────┘
```

---

## Folder Structure

```
src/game/client/rl/
├── README.md              # This file
├── rlbot.h                # Main RL bot component header
├── rlbot.cpp              # Main RL bot component implementation
├── observation.h          # Observation space definition
├── observation.cpp        # Observation extraction from game state
├── action.h               # Action space definition  
├── action.cpp             # Action injection into controls
├── reward.h               # Reward function definitions
├── reward.cpp             # Reward calculation logic
├── communication/
│   ├── comm_interface.h   # Abstract communication interface
│   ├── shm_comm.h         # Shared memory implementation
│   ├── shm_comm.cpp
│   ├── zmq_comm.h         # ZeroMQ implementation (optional)
│   └── zmq_comm.cpp
└── python/
    ├── teeworlds_env.py   # Gym environment wrapper
    ├── observation.py     # Observation space helpers
    ├── train.py           # Training script example
    └── requirements.txt   # Python dependencies
```

---

## Observation Space

Extracted from `CGameClient::m_Snap`:

| Feature | Type | Description |
|---------|------|-------------|
| `local_pos` | vec2 | Player position |
| `local_vel` | vec2 | Player velocity |
| `local_hook_state` | int | Hook state (idle, flying, grabbed) |
| `local_hook_pos` | vec2 | Hook position |
| `local_jumped` | int | Jump flags |
| `local_weapon` | int | Current weapon |
| `other_players[N]` | array | Other players' positions, velocities |
| `nearby_tiles[M]` | array | Collision tiles around player |
| `projectiles[K]` | array | Active projectiles |

---

## Action Space

Based on `CNetObj_PlayerInput`:

| Action | Type | Values |
|--------|------|--------|
| `direction` | Discrete | -1 (left), 0 (none), 1 (right) |
| `jump` | Binary | 0, 1 |
| `hook` | Binary | 0, 1 |
| `fire` | Binary | 0, 1 |
| `aim_angle` | Discrete/Continuous | 0-15 (16 directions) or radians |
| `weapon` | Discrete | 0-5 |

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

For training, run `rl_enable` after connection is established. You can use an autoexec config:

```bash
# Create autoexec file
echo "connect localhost:8303" > ~/.teeworlds/autoexec_rl.cfg
echo "cl_overlay_entities 100" >> ~/.teeworlds/autoexec_rl.cfg
echo "cl_showquads 0" >> ~/.teeworlds/autoexec_rl.cfg
echo "cl_showhud 0" >> ~/.teeworlds/autoexec_rl.cfg

# Run with autoexec
./build/DDNet -f autoexec_rl.cfg
```

Then enable RL mode via the Python client or manually.

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
- Headless mode (`-DHEADLESS_CLIENT=ON`) removes graphics overhead
- Multiple clients can connect to same server for parallel training
