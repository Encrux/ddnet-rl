# Plan: Tickrate-Agnostic RL Training

## Goal
Train RL agents at higher tickrates (e.g., 200) for faster wall-clock training, then deploy at standard 50 tick with consistent behavior.

## Key Insight
**Velocity values are tickrate-independent** - `VelX=10` means 10 units per tick at any tickrate. The physics relationships (thresholds, trajectories) remain consistent. The differences are:
1. Per-tick rewards accumulate faster at higher tickrates
2. Agent makes decisions more frequently at higher tickrates

## Approach: Scale Time-Based Rewards

The simplest effective solution is to scale per-tick rewards by `(50 / actual_tickrate)` so the total reward per second remains constant regardless of tickrate.

---

## Implementation

### 1. Add Tickrate to Shared Memory Header

**File:** `src/game/client/rl/communication/comm_interface.h`

Add `m_ServerTickrate` using reserved bytes:
```cpp
// Replace m_Reserved[23] with:
uint16_t m_ServerTickrate;    // Server tickrate (50, 100, 200, etc.)
uint8_t m_Reserved[21];       // Remaining reserved bytes
```

### 2. Set Tickrate in C++

**File:** `src/game/client/rl/rlbot.cpp`

In `SendStructuredObservation()`, set the tickrate:
```cpp
rl::SSharedMemoryHeader *pHeader = m_pComm->GetHeader();
if(pHeader)
{
    pHeader->m_ServerTickrate = (uint16_t)Client()->GameTickSpeed();
    // ... existing code ...
}
```

### 3. Scale Time-Based Rewards

**File:** `src/game/client/rl/rlbot.cpp`

Modify `CalculateReward()`:
```cpp
float CRLBot::CalculateReward(const rl::SStructuredObservation *pObs)
{
    const float BASE_TICKRATE = 50.0f;
    float tickrate = (float)Client()->GameTickSpeed();
    float tickScale = BASE_TICKRATE / tickrate;  // 0.25 at 200 tick, 1.0 at 50 tick

    float reward = 0.0f;

    // Time penalty - SCALE by tickrate
    reward -= 0.0005f * tickScale;

    // Progress reward - naturally balances (more samples at higher tick)
    float progress = (local.m_PosX - m_LastPosX) / 32.0f;
    reward += progress * 0.01f;  // Keep as-is

    // Velocity bonus - keep as-is (per-tick is fine)
    if(local.m_VelX > 0)
        reward += (local.m_VelX / 100.0f) * 0.002f;

    // Stuck counter - SCALE threshold by tickrate
    int stuckThreshold = (int)(100.0f * (tickrate / BASE_TICKRATE));  // 400 at 200 tick
    if(m_StuckCounter > stuckThreshold)
        reward -= 0.01f * tickScale;

    // Sparse rewards (checkpoint, finish, death) - keep fixed
    // These are one-time events, not per-tick accumulation
}
```

### 4. Parse Tickrate in Python

**File:** `src/game/client/rl/python/teeworlds_env.py`

Update `SharedMemoryHeader`:
```python
@dataclass
class SharedMemoryHeader:
    # ... existing fields up to m_Padding3 ...
    server_tickrate: int      # uint16 - NEW
    _reserved: bytes          # uint8[21] - reduced from 23

    STRUCT_FORMAT = '<BBBBIIIIfBBB1sbBBBhhBB3BH21s'  # Added 'H' for uint16
```

**Note:** Velocity normalization (`/50.0`) stays unchanged - it's for NN input scaling, not tickrate compensation.

### 5. Pass Tickrate to Training Config

**File:** `src/game/client/rl/python/train_single.py` (and `train.py`)

Add logging of tickrate for verification:
```python
print(f"Server tickrate: {args.sv_tickrate}")
```

---

## Files to Modify

| File | Change |
|------|--------|
| `comm_interface.h` | Add `m_ServerTickrate` field |
| `rlbot.cpp` | Scale rewards, set tickrate in header |
| `teeworlds_env.py` | Parse `m_ServerTickrate` from header |

---

## Verification

1. **Unit test reward scaling:**
   ```bash
   # Start with 200 tickrate
   ./DDNet-Server "sv_port 8303; sv_tickrate 200"
   ./DDNet "connect localhost:8303; rl_enable state 0"
   python train_single.py --sv-tickrate 200 --verbose --total-timesteps 10000
   ```

2. **Compare episode rewards:** At 200 tick vs 50 tick, episode rewards over same wall-clock time should be similar (since time penalty is scaled).

3. **Deploy test:** Train at 200 tick, then test at 50 tick:
   - Agent behavior should be similar (same velocity thresholds)
   - Agent may react less frequently (1/4 as many decisions per second)

---

## Notes

- **Action frequency difference:** At 200 tick, agent decides 4x per second vs 50 tick. This means finer-grained control during training. At deployment, agent has "coarser" control but learned thresholds still apply.

- **No frame-skipping needed:** The agent learns tick-based behavior that transfers directly.

- **Future enhancement:** Could add action-repeat during deployment if reaction time differences matter.
