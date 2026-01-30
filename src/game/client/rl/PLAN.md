# Teeworlds RL Refactor Plan

## Overview

Refactor the RL training system to use:
- **Hybrid action space**: Multi-discrete (movement) + continuous (aim)
- **CleanRL-based PPO**: Single-file, easy to modify, proven implementations
- **Improved observations**: Raycasts, better normalization, grounded detection
- **Dense reward shaping**: Progress-based rewards, mechanic bonuses, proper penalties

## Phase 1: Environment Wrapper (Gymnasium-compatible)

### 1.1 Create `teeworlds_vec_env.py`

A proper Gymnasium-compatible vectorized environment that wraps the shared memory communication.

```python
# Action space (Hybrid)
action_space = spaces.Dict({
    'discrete': spaces.MultiDiscrete([3, 2, 2]),  # direction, jump, hook
    'continuous': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # aim_x, aim_y (normalized)
})

# Observation space
observation_space = spaces.Dict({
    'player': spaces.Box(...),      # Player state features
    'raycasts': spaces.Box(...),    # Distance to solids/death/hookable in N directions
    'tiles': spaces.Box(...),       # Small local grid or CNN embedding
})
# Flattened for MLP policy: ~150-200 features total
```

### 1.2 Observation Changes

**Remove:**
- Raw 29x21 tile grid as 609 continuous values (categorical treated as ordinal)
- Absolute positions normalized by arbitrary constants

**Add:**
- **Raycasts** (16 directions x 3 types = 48 features):
  - Distance to nearest SOLID (0-1 normalized by max ray length)
  - Distance to nearest DEATH
  - Distance to nearest HOOKABLE (SOLID or NOHOOK boundary)
- **Player state** (~20 features):
  - Velocity (normalized by max speed ~50 units)
  - Is grounded (binary)
  - Can jump (based on m_Jumped: 0=can double jump, 1=can single jump, 2+=no jumps)
  - Hook state (one-hot: idle, flying, attached)
  - Is frozen (binary)
  - Current aim angle (sin, cos for continuity)
- **Small local grid** (7x7 = 49 tiles, one-hot 4 key types = 196 features):
  - Or better: CNN encoder to 32-dim embedding

**Total: ~100-150 features** (down from ~700)

### 1.3 Action Processing

```python
def process_action(action):
    discrete = action['discrete']
    continuous = action['continuous']

    direction = discrete[0] - 1  # 0,1,2 -> -1,0,1
    jump = discrete[1]
    hook = discrete[2]

    # Continuous aim: normalize from [-1,1] to game coordinates
    # Convert to angle, then to aim vector
    aim_angle = continuous[0] * np.pi  # -pi to pi
    aim_magnitude = (continuous[1] + 1) / 2  # 0 to 1, controls aim distance

    aim_x = int(np.cos(aim_angle) * aim_magnitude * 200)
    aim_y = int(np.sin(aim_angle) * aim_magnitude * 200)

    return {
        'direction': direction,
        'jump': jump,
        'hook': hook,
        'aim_x': aim_x,
        'aim_y': aim_y,
        'fire': 0,  # Not used for DDNet movement
        'weapon': 0,
    }
```

## Phase 2: C++ Reward & Observation Improvements

### 2.1 Reward Shaping (rlbot.cpp)

```cpp
float CRLBot::CalculateReward(const rl::SStructuredObservation* pObs) {
    float reward = 0.0f;
    const rl::SPlayerObs& local = pObs->m_LocalPlayer;

    // ===== DENSE REWARDS (every step) =====

    // 1. Progress reward: movement toward finish (assume rightward)
    //    Use actual progress, not just "new max distance"
    float progress = (local.m_PosX - m_LastPosX) / 32.0f;  // tiles moved
    reward += progress * 0.01f;

    // 2. Time penalty: encourage efficiency
    reward -= 0.0005f;

    // 3. Velocity bonus: reward moving fast in progress direction
    if (local.m_VelX > 0) {
        reward += (local.m_VelX / 100.0f) * 0.002f;
    }

    // ===== MECHANIC REWARDS =====

    // 4. Hook success: reward attaching hook (once per hook)
    if (local.m_HookState == 2 && m_LastHookState != 2) {
        reward += 0.05f;  // Successful hook attach
    }

    // 5. Hook momentum: reward speed gained while hooked
    if (local.m_HookState == 2) {
        float speed = std::sqrt(local.m_VelX*local.m_VelX + local.m_VelY*local.m_VelY);
        float lastSpeed = std::sqrt(m_LastVelX*m_LastVelX + m_LastVelY*m_LastVelY);
        if (speed > lastSpeed) {
            reward += (speed - lastSpeed) * 0.005f;
        }
    }

    // ===== SPARSE MILESTONES =====

    // 6. Checkpoint (track actual checkpoint index, not tick)
    if (m_CurrentCheckpointIndex > m_LastCheckpointIndex) {
        reward += 0.5f;
        m_LastCheckpointIndex = m_CurrentCheckpointIndex;
    }

    // 7. Finish
    if (IsOnFinishTile()) {
        reward += 5.0f;
        m_Done = true;
    }

    // ===== PENALTIES =====

    // 8. Death
    if (PlayerDied()) {
        reward -= 1.0f;
        m_Done = true;
    }

    // 9. Freeze
    if (local.m_IsFrozen && !m_WasFrozen) {
        reward -= 0.3f;
    }

    // 10. Stuck penalty (no progress for N ticks)
    if (std::abs(local.m_PosX - m_LastPosX) < 0.1f &&
        std::abs(local.m_PosY - m_LastPosY) < 0.1f) {
        m_StuckCounter++;
        if (m_StuckCounter > 100) {  // ~2 seconds at 50 tick
            reward -= 0.01f;
        }
    } else {
        m_StuckCounter = 0;
    }

    // Update state for next frame
    m_LastPosX = local.m_PosX;
    m_LastPosY = local.m_PosY;
    m_LastVelX = local.m_VelX;
    m_LastVelY = local.m_VelY;
    m_LastHookState = local.m_HookState;
    m_WasFrozen = local.m_IsFrozen;

    return reward;
}
```

### 2.2 Add Raycast Observations (rlbot.cpp)

Add a new function to cast rays from player position:

```cpp
void CRLBot::ExtractRaycasts(rl::SStructuredObservation* pObs) {
    // Cast rays in 16 directions (every 22.5 degrees)
    constexpr int NUM_RAYS = 16;
    constexpr float MAX_DIST = 20.0f * 32.0f;  // 20 tiles

    float playerX = pObs->m_LocalPlayer.m_PosX;
    float playerY = pObs->m_LocalPlayer.m_PosY;

    for (int i = 0; i < NUM_RAYS; i++) {
        float angle = (float)i * (2.0f * M_PI / NUM_RAYS);
        float dx = std::cos(angle);
        float dy = std::sin(angle);

        // Ray march to find distances
        pObs->m_aRaycastSolid[i] = CastRayToType(playerX, playerY, dx, dy, MAX_DIST, TILE_SOLID);
        pObs->m_aRaycastDeath[i] = CastRayToType(playerX, playerY, dx, dy, MAX_DIST, TILE_DEATH);
        pObs->m_aRaycastHookable[i] = CastRayToHookable(playerX, playerY, dx, dy, MAX_DIST);
    }
}

float CRLBot::CastRayToType(float startX, float startY, float dx, float dy,
                            float maxDist, int tileType) {
    CCollision* pCollision = GameClient()->Collision();
    float step = 8.0f;  // Check every 8 units (quarter tile)

    for (float dist = step; dist < maxDist; dist += step) {
        float x = startX + dx * dist;
        float y = startY + dy * dist;
        int tileX = (int)(x / 32.0f);
        int tileY = (int)(y / 32.0f);

        if (pCollision->GetTileIndex(tileY * pCollision->GetWidth() + tileX) == tileType) {
            return dist / maxDist;  // Normalized 0-1
        }
    }
    return 1.0f;  // Max distance, nothing found
}
```

### 2.3 Update Observation Struct (observation.h)

```cpp
// Add to SStructuredObservation:
constexpr int NUM_RAYCASTS = 16;

struct SStructuredObservation {
    // ... existing fields ...

    // Raycast observations (16 directions x 3 types = 48 floats = 192 bytes)
    float m_aRaycastSolid[NUM_RAYCASTS];    // Distance to solid
    float m_aRaycastDeath[NUM_RAYCASTS];    // Distance to death
    float m_aRaycastHookable[NUM_RAYCASTS]; // Distance to hookable surface

    // Player state extras
    uint8_t m_IsGrounded;      // Standing on solid
    uint8_t m_JumpsRemaining;  // 0, 1, or 2
};
```

## Phase 3: PPO Implementation (CleanRL-based)

### 3.1 Create `ppo_hybrid.py`

Based on CleanRL's `ppo_continuous_action.py` but with hybrid action space support.

Key modifications:
- Actor outputs both discrete logits AND continuous mean/std
- Separate entropy calculation for discrete and continuous parts
- Combined log probability for policy gradient

```python
class HybridActor(nn.Module):
    def __init__(self, obs_dim, discrete_dims, continuous_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Discrete heads (one per discrete action dimension)
        self.discrete_heads = nn.ModuleList([
            nn.Linear(128, dim) for dim in discrete_dims
        ])

        # Continuous head (mean and log_std)
        self.continuous_mean = nn.Linear(128, continuous_dim)
        self.continuous_log_std = nn.Parameter(torch.zeros(continuous_dim))

    def forward(self, x):
        shared_out = self.shared(x)

        discrete_logits = [head(shared_out) for head in self.discrete_heads]
        continuous_mean = self.continuous_mean(shared_out)
        continuous_std = self.continuous_log_std.exp()

        return discrete_logits, continuous_mean, continuous_std

    def get_action_and_log_prob(self, x, action=None):
        discrete_logits, cont_mean, cont_std = self.forward(x)

        # Discrete distributions
        discrete_dists = [Categorical(logits=logits) for logits in discrete_logits]

        # Continuous distribution
        cont_dist = Normal(cont_mean, cont_std)

        if action is None:
            # Sample new action
            discrete_actions = [d.sample() for d in discrete_dists]
            continuous_action = cont_dist.sample()
        else:
            # Use provided action (for PPO update)
            discrete_actions = [action['discrete'][:, i] for i in range(len(discrete_dists))]
            continuous_action = action['continuous']

        # Log probabilities
        discrete_log_probs = sum(d.log_prob(a) for d, a in zip(discrete_dists, discrete_actions))
        continuous_log_prob = cont_dist.log_prob(continuous_action).sum(-1)
        total_log_prob = discrete_log_probs + continuous_log_prob

        # Entropy
        discrete_entropy = sum(d.entropy() for d in discrete_dists)
        continuous_entropy = cont_dist.entropy().sum(-1)
        total_entropy = discrete_entropy + continuous_entropy

        action_out = {
            'discrete': torch.stack(discrete_actions, dim=-1),
            'continuous': continuous_action,
        }

        return action_out, total_log_prob, total_entropy
```

### 3.2 Training Loop Structure

```python
def train():
    # Hyperparameters
    num_envs = 63
    num_steps = 256        # Steps per env before update
    num_minibatches = 4
    update_epochs = 4
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Create vectorized environment
    envs = TeeWorldsVecEnv(num_envs=num_envs)

    # Create networks
    obs_dim = envs.single_observation_space.shape[0]
    actor = HybridActor(obs_dim, discrete_dims=[3, 2, 2], continuous_dim=2)
    critic = Critic(obs_dim)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate)

    # Storage
    obs_buffer = torch.zeros((num_steps, num_envs, obs_dim))
    actions_discrete = torch.zeros((num_steps, num_envs, 3), dtype=torch.long)
    actions_continuous = torch.zeros((num_steps, num_envs, 2))
    log_probs = torch.zeros((num_steps, num_envs))
    rewards = torch.zeros((num_steps, num_envs))
    dones = torch.zeros((num_steps, num_envs))
    values = torch.zeros((num_steps, num_envs))

    obs = envs.reset()

    for iteration in range(num_iterations):
        # Rollout
        for step in range(num_steps):
            with torch.no_grad():
                action, log_prob, _ = actor.get_action_and_log_prob(obs)
                value = critic(obs)

            next_obs, reward, done, info = envs.step(action)

            obs_buffer[step] = obs
            actions_discrete[step] = action['discrete']
            actions_continuous[step] = action['continuous']
            log_probs[step] = log_prob
            rewards[step] = reward
            dones[step] = done
            values[step] = value.squeeze()

            obs = next_obs

        # Compute advantages (GAE)
        advantages = compute_gae(rewards, values, dones, gamma, gae_lambda)
        returns = advantages + values

        # Flatten batch
        b_obs = obs_buffer.reshape(-1, obs_dim)
        b_actions = {'discrete': actions_discrete.reshape(-1, 3),
                     'continuous': actions_continuous.reshape(-1, 2)}
        b_log_probs = log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # PPO update
        for epoch in range(update_epochs):
            indices = torch.randperm(num_steps * num_envs)
            for start in range(0, len(indices), batch_size):
                mb_indices = indices[start:start + batch_size]

                _, new_log_prob, entropy = actor.get_action_and_log_prob(
                    b_obs[mb_indices],
                    {k: v[mb_indices] for k, v in b_actions.items()}
                )
                new_value = critic(b_obs[mb_indices]).squeeze()

                # Policy loss
                ratio = (new_log_prob - b_log_probs[mb_indices]).exp()
                adv = b_advantages[mb_indices]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -ratio * adv
                pg_loss2 = -torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_value - b_returns[mb_indices]) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + vf_coef * v_loss + ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
                optimizer.step()
```

## Phase 4: Episode Reset Handling

### Current Approach (Kill Command)
- Works but requires respawn delay
- Player state persists across deaths (inventory, etc.)

### Recommended: Keep Kill Command
The kill command approach is actually fine for DDNet:
- Server handles respawn logic correctly
- No need for rcon access or server modifications
- Each client manages its own reset independently

**Improvements to current reset:**
1. Track when kill was sent, wait for respawn confirmation
2. Reset internal state (max distance, checkpoint tracking) on respawn detection
3. Add small delay after kill to ensure clean state

```cpp
void CRLBot::RequestReset() {
    // Send kill command
    GameClient()->SendKill();
    m_ResetRequested = true;
    m_ResetRequestTick = Client()->GameTick(g_Config.m_ClDummy);
}

void CRLBot::CheckResetComplete() {
    if (!m_ResetRequested) return;

    // Wait for respawn (player position changes to spawn)
    int currentTick = Client()->GameTick(g_Config.m_ClDummy);
    if (currentTick - m_ResetRequestTick > 50) {  // 1 second timeout
        // Respawn complete, reset tracking state
        m_LastPosX = m_SpawnPosX;
        m_LastPosY = m_SpawnPosY;
        m_MaxDistanceFromSpawn = 0.0f;
        m_LastCheckpointIndex = -1;
        m_StuckCounter = 0;
        m_AccumulatedReward = 0.0f;
        m_ResetRequested = false;
    }
}
```

## Phase 5: File Structure

```
src/game/client/rl/
├── python/
│   ├── teeworlds_vec_env.py    # Gymnasium-compatible vectorized env
│   ├── ppo_hybrid.py           # CleanRL-based PPO with hybrid actions
│   ├── train.py                # Main training script
│   ├── networks.py             # Actor/Critic definitions
│   └── utils.py                # GAE, normalization, etc.
├── rlbot.cpp                   # Updated with new rewards, raycasts
├── rlbot.h
├── observation.h               # Updated with raycast fields
└── communication/
    └── shm_comm.cpp/h          # Shared memory (unchanged)
```

## Implementation Order

### Step 1: C++ Changes (Rewards & Observations)
1. Add raycast extraction to rlbot.cpp
2. Update observation.h with new fields
3. Implement new reward function
4. Add grounded detection and jump state

### Step 2: Python Environment
1. Create teeworlds_vec_env.py with Gymnasium interface
2. Parse new observation format
3. Implement hybrid action space conversion
4. Add observation normalization

### Step 3: PPO Implementation
1. Create ppo_hybrid.py based on CleanRL
2. Implement HybridActor and Critic
3. Add training loop with GAE
4. TensorBoard logging

### Step 4: Integration & Testing
1. Test with single environment first
2. Scale to full 63 environments
3. Tune hyperparameters
4. Monitor reward curves

## Hyperparameter Starting Points

```python
# PPO
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.2
ent_coef = 0.01      # May need to increase for exploration
vf_coef = 0.5
max_grad_norm = 0.5

# Training
num_envs = 63
num_steps = 256       # Steps per rollout
num_minibatches = 4
update_epochs = 4
total_timesteps = 10_000_000

# Network
hidden_dim = 256
num_layers = 2
```

## Success Metrics

1. **Learning signal**: Loss decreasing, rewards increasing
2. **Exploration**: Entropy staying reasonable (not collapsing to 0)
3. **Mechanics**: Agent uses hook (track hook state transitions)
4. **Progress**: Average distance from spawn increasing
5. **Completion**: Eventually reaching checkpoints/finish

## Potential Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| Reward hacking (oscillating for speed) | Progress reward dominates, time penalty |
| Sparse rewards (never reaches checkpoint) | Dense progress reward, exploration bonus |
| Action correlation (always hook+jump) | Entropy bonus, factored action heads |
| Observation scale mismatch | Running normalization, careful scaling |
| Training instability | Gradient clipping, PPO clipping, smaller LR |
