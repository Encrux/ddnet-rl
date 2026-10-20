# DDNet RL — Holistic Review and Roadmap

*Written 2026-08-03, after reviewing the `checkpoint-respawn` branch.*

Goal: an autonomous agent that finishes a real DDNet race map, start to finish, unassisted —
impressive enough to show a DDNet audience and an ML audience.

---

## TL;DR

1. **The algorithm is not your bottleneck.** PPO is fine. The reward function is actively
   fighting you, and the reset distribution is the real cost centre. Those are ~10× effects;
   swapping PPO for something newer is a ~2× effect.
2. **The single highest-leverage change is already sitting in the codebase**: `CSaveTee`
   (`src/game/server/save.h:38`) serialises a *complete* tee — position, velocity, hook state,
   freeze timer, jumps, weapons, tele-checkpoint. Snapshot/restore turns "walk 40 seconds to the
   frontier" into "teleport to the frontier with the correct momentum". This is real Go-Explore,
   and it's maybe 200 lines of C++.
3. **You are not as data-poor as you think.** DDNet `.gho` ghosts contain per-tick
   `Direction`, `HookState`, `Angle`, and `AttackTick` — that is nearly the full action vector,
   plus exact position and velocity. Your own `ghost_parser.py` already decodes them. You don't
   need teehistorian for a backward curriculum; you need *restore points*, and ghosts hand you
   those directly.
4. **Yes, integrate a foundation model — in the outer loop, not the inner loop.** DDNet maps are
   literally ASCII-legible 2D tile grids. An LLM reading "agent dies at this 43×31 tile block 94%
   of the time" and rewriting the reward/curriculum is both feasible today and genuinely the
   "reasoning about a hard frontier task" you want. A VLA in the control loop is not feasible.
5. **Skip world models.** DreamerV3-class methods buy sample efficiency. You have a free,
   deterministic, 21k FPS simulator. You'd be paying GPU time to avoid using CPU time you have.

---

## Part 1 — What's actually wrong right now

### 1.1 The reward function punishes correct play (the big one)

From `CBotManager::CalculateReward` (`src/game/server/rl/botmanager.cpp:699`), evaluated **every
tick** and accumulated across the frame-skip:

| Term | Value |
|---|---|
| New path progress | `Δ / 100` per world unit, **high-water-mark only** |
| Freeze (per tick) | `−0.1` |
| Freeze (on entry) | `−0.3` |
| Time | `−0.0005` per tick |
| Off-path (>128u) | up to `−0.01` per tick |
| Stuck (>50 ticks) | ramps `−0.01 × (n−50)/50` per tick |

Now do the units. `m_Tuning.m_GroundControlSpeed = 10.0` and `m_Vel` is per-tick
(`gamecore.cpp:531` multiplies by 50 to get per-second), so a tee at full ground speed covers
**500 world units/second**. Progress therefore earns at most **+5.0 reward/second**.

Freeze costs **0.1 × 50 = −5.0 reward/second**.

> **One second frozen exactly cancels one second of world-record-pace running.**

In DDNet, freeze is not failure — it is routine. You pass through freeze constantly on real maps,
and on many obstacles freezing deliberately *is* the solution. This reward encodes the prior
"freeze is nearly as bad as death", which is simply false, and it is why (per your notes) the
freeze penalty dominates early training.

It gets worse. Because progress is high-water-mark, re-traversing ground you've already covered
earns **zero**. So after the agent's first failed attempt at the frontier, every remaining tick of
that episode has *strictly negative* expected return: time penalty, plus freeze risk, plus the
stuck ramp, with no upside available. With `max_episode_steps: 2000` at `frame_skip: 3`, that's up
to 120 seconds of episode where the optimal policy is "stand still somewhere safe" or "die
quickly". That is exactly the reward-hacking pressure you observed.

**Fixes, in order of importance:**
- Delete the per-tick freeze penalty. Keep at most a small entry penalty (`−0.05`), or nothing at
  all — the time penalty already prices freezing correctly, because being frozen *costs time*.
- Make the time penalty carry the load: `−0.01/tick` (`−0.5/s`) is a defensible race objective.
  Progress reward gives shape; time penalty gives urgency; nothing else is needed.
- Drop the stuck ramp entirely. It's a proxy for "you're wasting time", which the time penalty
  already says, and it double-counts.
- The off-path penalty is **minor** — see §1.7. Make it a config flag and move on.
- Sanity-check the result the same way every time: **a trajectory that freezes for 2s and then
  clears an obstacle must score higher than one that never attempts it.** Write that as a unit test
  against `CalculateReward`.

### 1.2 The reset distribution is the actual compute sink

This is the problem you named, and your instinct is right that it's the barrier.

`CheckpointSpawnManager` (`teeworlds_server_env.py:135`) is the current answer: sample a DDNet
*time checkpoint* to respawn at, biased toward the frontier. Two hard limits:

- **Granularity.** DDNet time checkpoints are sparse — a handful per map, sometimes zero. You
  cannot express "restart 3 seconds before the hard jump" if there's no checkpoint there.
- **State, not just position.** `ResetRewardState(i, SpawnCp)` puts the tee *at* the checkpoint
  position, standing still. Most non-trivial DDNet obstacles require arriving with specific
  momentum, a live hook, or a jump in reserve. So respawning at a checkpoint often poses a
  *different and sometimes unsolvable* problem versus the one a from-start rollout faces — and
  whatever the agent learns there may not transfer back.

**The fix already exists in the repo.** `CSaveTee::Save(CCharacter*)` / `Load(CCharacter*)`
(`src/game/server/save.h:43-44`) is DDNet's `/save` `/load` machinery, and it captures everything
that matters: `m_CorePos`, `m_Vel`, `m_HookPos/Dir/Tick/State/HookedPlayer`, `m_FreezeTime`,
`m_Jumped/m_JumpedTotal/m_Jumps`, weapons, ninja, `m_TeleCheckpoint`, `m_DDRaceState`, and the
per-checkpoint split times. `CSaveTeam` does the same for a whole team, which is your on-ramp to
team race later.

That gives you an arbitrary-granularity, physics-exact restore primitive for near-free.

### 1.3 The policy has no memory

`ppo_hybrid.py` is a pure feedforward CNN+MLP. `n_player_state_stack: 4` at `frame_skip: 3` gives
12 ticks ≈ **0.24 seconds** of history. A hook swing lasts 1–2 seconds. The agent is being asked to
execute multi-second ballistic manoeuvres from a near-instantaneous snapshot.

This is a genuine architecture gap, not a nice-to-have. Either add a GRU over the encoder output
(and switch PPO to truncated-BPTT rollouts), or — cheaper, and worth trying first — stack far more
player state (16–32 frames) and feed it as a 1D temporal conv rather than a flat concat. Your
benchmark note that "stack=8 hurts" is consistent with *flat concatenation* scaling badly, not with
history being useless.

### 1.4 The action space fights the task

Two issues:

- **Aim magnitude is dead weight** — already in `improvements.md`. DDNet only uses the direction of
  `m_TargetX/Y`. Drop it.
- **A Gaussian over aim angle is the wrong distribution.** It's unimodal. At a fork where the
  correct action is "hook hard left *or* hook hard right, but never straight ahead", a Gaussian
  literally cannot represent the policy — it collapses to the mean, which is the one action that
  fails. Hook-based movement is full of these.

**But plain binning is also wrong**, because aim needs real precision. `m_HookLength = 380`
(`tuning.h:20`), so at max hook range one bin of a 32-way split (11.25°) subtends
`380 × 0.196 ≈ 75` world units ≈ **2.3 tiles** of error. That misses the block.

**Recommended: a factored aim head — coarse Categorical × continuous residual.**

```
sector ~ Categorical(logits)          # 16–32 sectors: captures multimodality
residual ~ Normal(mu[sector], sigma)  # captures precision
angle = sector_center + tanh(residual) * (half_sector_width)
log_prob = log p(sector) + log p(residual)
```

Unlimited precision, genuinely multimodal, no observability change, and a small diff to
`ppo_hybrid.py`. This is the standard classification+regression head pattern.

**Why not delta aim** (relative crosshair movement, mimicking a mouse) — tempting, but it loses on
three counts here:

1. **`SPlayerObs` has no angle field** (`rl_protocol.h:59`). With absolute aim that's fine — the
   policy emits the aim, it needn't remember it. With delta aim, current aim becomes hidden state
   and you have a POMDP until you add it to the observation.
2. **Hook direction is latched at fire time.** `m_HookDir = TargetDirection` is set once when the
   hook enters `HOOK_FLYING` (`gamecore.cpp:266`) and never updated again (the only other
   assignment, line 394, is the hook-through-teleporter edge case). `m_HookFireSpeed = 80` and
   `m_HookLength = 380`, so the hook fully extends in ~4.75 ticks — **under two agent decisions at
   `frame_skip: 3`**. Aim precision therefore matters on *exactly one tick*. Delta aim converts a
   one-tick precision problem into an N-step pre-aiming problem, with credit spread across every
   delta that led there. Strictly harder.
3. **Snap latency.** At 16.7 decisions/s with a bounded delta of ~15°/step, a 180° flip costs ~12
   steps ≈ 0.7 s. Humans flick instantly; a bounded delta structurally cannot.

The usual argument *for* delta aim is temporal smoothness — but per (2), the hook doesn't care
about the aim trace, only the fire-tick aim, so there is little to smooth. If jitter does turn out
to hurt, add an action-smoothness auxiliary penalty on top of absolute aim rather than changing the
parameterisation. Absolute aim is also what ghosts record, so it stays compatible with any later
behaviour cloning.

**Worth an experiment: spatial-softmax aim.** Hooks attach to *tiles*, and your CNN already
computes a spatial feature map. A 1×1 conv head → softmax over the ~25×25 subgrid within hook range
gives "aim at that tile", which is natively multimodal at exactly the granularity the geometry has.
Pair with a small sub-tile residual for grazing hooks.

### 1.5 Observation gaps

- The agent gets **one** next checkpoint (`m_NextCheckpointX/Y`). Linesight — the strongest
  comparable result in this genre — feeds the *next N virtual checkpoints* in egocentric
  coordinates. Give the agent the next ~10 waypoints relative to itself. You already have the
  waypoint file loaded server-side; this is cheap and high-value.
- No global progress scalar. Add normalised path progress so the value function can distinguish
  "early on the map" from "near the finish".
- `m_Reserved3[3]` gives you 3 spare bytes; you'll need to bump `SStructuredObservation` past 1400
  for the waypoint block. Fine — it's a `static_assert`, not a wire format anyone else depends on.

### 1.6 The off-path penalty, precisely (and why it's a minor term)

`botmanager.cpp:730-743`, evaluated every tick when waypoints are loaded:

```cpp
int ClosestIdx = FindClosestWaypoint(Pos, State.m_WaypointIndex, 100); // ±100 indices
float DistFromPath = |Pos − waypoint[ClosestIdx]|;                     // point, not segment
if(DistFromPath > 128.0f)
    Reward -= min((DistFromPath − 128) / 384, 1) * 0.01f;
```

- Deadzone: **0 penalty within 128 units (4 tiles)** of the nearest waypoint.
- Linear ramp from 128 → 512 units, saturating at `−0.01/tick` = **−0.5 reward/second** once you're
  ≥16 tiles off the line.
- Waypoint spacing is 64 units (`ghost_parser.resample_waypoints` default), so the
  point-vs-segment approximation adds ≤32 units of error — mostly swallowed by the deadzone.

**I over-weighted this in the first draft.** At −0.5/s against +5.0/s progress and −5.0/s freeze,
it's an order of magnitude smaller than the freeze penalty. It's a flag-it-and-move-on item, not a
priority. Two places it does bite: (a) hook flight takes wide ballistic arcs that legitimately
leave the line by more than 4 tiles, so it mildly taxes the manoeuvre you most want to learn; and
(b) in the post-frontier-failure regime where no progress is available, it's one more negative term
on an already all-negative pile.

**The same code path has a more serious latent issue.** `FindClosestWaypoint` searches only
`SearchStart ± 100` indices and initialises `BestIdx = SearchStart`, so if the tee moves further
than ~6400 units of *path arc* in one tick, it silently returns a wrong "closest". `ClosestIdx`
also feeds `GetPathProgress`, which drives the **main progress reward** — and because that reward
is high-water-mark, a single spurious forward jump permanently inflates `m_BestPathProgress` and
zeroes out all real progress reward for that whole region.

I checked whether this actually fires on Tutorial: the raw ghost has **one** genuine
discontinuity, a 160-unit teleport at tick index 2953 (`(44063,1833) → (43920,1904)`). Everything
else >60 units/tick is just fast free-fall. 160 units is well inside the ±100-index window, so
Tutorial is fine today. But this will break on any map with long-range teleporters — worth a cheap
guard (fall back to a full scan when the windowed best distance exceeds, say, 512 units).

### 1.7 Checkpoints are the wrong abstraction for a cross-map reward

DDNet conflates three unrelated things under "checkpoint", and they have wildly different
reliability guarantees:

| Concept | Tiles | Enforced semantics? |
|---|---|---|
| **Time checkpoints** (splits) | `TILE_TIME_CHECKPOINT_FIRST..LAST` (35–59, max 25) | **None.** Cosmetic. A mapper may place 0 or 25, out of order, anywhere, or in a region you cross twice. |
| **Tele checkpoints** (respawn) | `TILE_TELECHECK` 29, `TELECHECKIN` 31, `TELECHECKOUT` 30, `TELECHECKINEVIL` 63 | **Yes.** This is DDNet's actual respawn mechanism (`CSaveTee::m_TeleCheckpoint`). Place them wrong and the map is broken, so mappers get them right — and on long maps they're placed densely, because dying back to the start would be unplayable. |
| **Start / Finish** | `TILE_START` 33, `TILE_FINISH` 34 | **Yes.** Universal on every race map. |

The current code keys everything on the **least reliable** of the three
(`botmanager.cpp:977`, `TileIdx - TILE_TIME_CHECKPOINT_FIRST`). That's why a single reward function
doesn't survive contact with a second map: `m_CheckpointIndex` / `m_TotalCheckpoints` /
`m_NextCheckpointX/Y` all measure *mapper style*, not task structure. Respawn granularity is
likewise whatever the mapper happened to place — which is the granularity problem from §1.2
restated.

**Two rules follow:**

1. **If you must use a map-provided checkpoint, use the tele checkpoint, not the time checkpoint.**
   It has enforced semantics; the other doesn't.
2. **Better: don't use either.** The one progress coordinate well-defined on every race map is
   **normalised path arc-length ∈ [0,1]** along a reference trajectory. You already compute it
   (`GetPathProgress`). Purge the rest:
   - `m_NextCheckpointX/Y`, `m_CheckpointIndex`, `m_TotalCheckpoints` → replace with next-N
     waypoints egocentric + a normalised progress scalar.
   - `CheckpointSpawnManager` → `CSaveTee` snapshots spaced by *arc-length or ghost time*, which
     you choose (e.g. every 2 s) and which is therefore **identical on every map**. This is the
     underrated payoff of §3.1: respawn granularity becomes yours, not the mapper's.
   - The fallback checkpoint reward branch (`botmanager.cpp:753-789`) → delete once waypoints are
     mandatory. Two reward code paths with different scales is a silent source of cross-map
     inconsistency.

**Teleports break the path abstraction — make the path a graph, not a polyline.**
`resample_waypoints` walks the ghost polyline at fixed 64-unit spacing, so it *interpolates across*
a teleport, laying phantom waypoints through solid geometry. On Tutorial the only real
discontinuity is 160 units (§1.6), so it's 2–3 phantom points and harmless — on a map with a
long-range tele it creates a phantom corridor, and worse, two physically distant regions get
*adjacent* waypoint indices, so a tee near the tele-out can match a waypoint near the tele-in and
corrupt progress.

Fix:
- Detect discontinuities **by collision, not by distance** — check whether the straight segment
  between consecutive ghost samples passes through solid tiles. A distance threshold is unreliable
  because free-fall legitimately hits 72 units/tick, and ghosts are 25 Hz (§3.2), so ~145 units
  between samples is normal.
- Split into segments at portals; progress = Σ completed segment lengths + within-segment arc, with
  the portal edge contributing **zero** length (no free reward for the teleport distance).
- Constrain `FindClosestWaypoint` to the current segment ± window. This also fixes the wrong-match
  bug in §1.6.
- Evil tele (`TILE_TELEINEVIL` 10, `TELECHECKINEVIL` 63) zeroes velocity; normal tele preserves it.
  Irrelevant for `CSaveTee` snapshots (you capture actual state), but it must be *visible* — see
  §1.8.

### 1.8 The observation is blind to most of the map (elevate this to P0)

This is the same disease as §1.7, one level worse: no reward function can fix blindness.

`ExtractTileGrid` (`botmanager.cpp:642-697`) reads the **game layer and the tele layer only**.
DDNet gameplay lives across five layers, all already exposed by `CCollision`:

| Layer | Accessor | Read by the observation? | What's lost |
|---|---|---|---|
| Game | `GetTileIndex` | ✅ | — |
| Tele | `TeleLayer()` (collision.h:116) | ⚠️ **overwrites** the game tile (`:680-683`) | "tele-in placed on freeze" loses the freeze |
| **Front** | `FrontLayer()` (collision.h:118) | ❌ | Freeze, deep-freeze, live-freeze, stoppers, one-way platforms |
| **Speedup** | `SpeedupLayer()` (collision.h:117) | ❌ | Direction + force + max-speed boosters |
| **Switch** | `SwitchLayer()` (collision.h:119) | ❌ | Doors and gates — unsolvable if invisible |
| **Tune** | `TuneLayer()` (collision.h:120) | ❌ | Per-zone physics changes (gravity, hook) |

**Correction on the front layer.** What is verified: `collision.cpp:845` reads the front layer for
`TILE_FREEZE..TILE_TELE_LASER_DISABLE` and `TILE_LFREEZE..TILE_LUNFREEZE`, `:877-885` for stoppers,
`:607-625` for one-way platforms — so front-layer tiles *are* gameplay-active — and
`ExtractTileGrid` does not read it. What is **not** verified, and what I asserted without measuring,
is how often real maps actually put hazards there. DDNet mapping guidelines are largely enforced
and violations tend to get fixed, so the practical incidence may be near zero outside old maps.
Treat this as *open, and cheap to settle*: scan a map corpus for front-layer tiles in the
gameplay-relevant ranges and decide from the count, rather than from either of our priors.

Worth noting the codebase already hedges both ways — the checkpoint scanner at
`botmanager.cpp:984-996` **does** read the front layer for time checkpoints, while
`ExtractTileGrid` doesn't read it at all. Whatever the answer, resolve the inconsistency in one
direction.

The **speedup** and **switch** layers are a separate matter and don't depend on that question:
boosters and switch-gated doors are standard, not legacy, and they're invisible today.

Also dropped: **tile flags**. `m_Flags` carries rotation, and `TILE_STOP` / `TILE_THROUGH_DIR` are
*directional* (`collision.cpp:609`, `:883-885`). Without flags, a one-way platform is ambiguous
even when it's visible.

**Fix: multi-plane observation.** Replace the single `uint8 m_aTileGrid[1333]` with C planes —
game, front, tele-type, speedup (force/angle), switch(number/state), flags. At 4–6 planes that's
5.3–8 KB per bot; at 127 bots ~1 MB per step against the ~178 KB you move today. Given the
contiguous shm layout (commit `9b1c612b8`) that's affordable, and it is the single change that
decides whether multi-map training is possible at all.

### 1.9 Raw tile IDs as a float32 channel is semantically meaningless

`botmanager.cpp:687`: `TileType = (TileIdx > 255) ? 255 : (uint8_t)TileIdx` — raw DDNet tile
indices. Under `--use-cnn` these are fed as a **float32 grid**, so the network sees
`TILE_LFREEZE = 144` as numerically 16× `TILE_FREEZE = 9`, and `TILE_TELEIN = 26` sitting on a
continuous axis between `TELEOUT = 27` and `TELECHECK = 29`. The ordering carries no meaning.

(Note that `ETileType` in `rl_protocol.h:34-52` — `EMPTY=0 … OTHER=15` — documents an encoding that
is **not actually used** anywhere. Dead code that actively misleads; delete or wire it up.)

**Fix: a fixed, non-learned semantic lookup.** Map 256 tile IDs → K≈12 semantic channels
(solid / death / freeze / deep-freeze / unfreeze / tele-normal / tele-evil / tele-check / speedup /
stopper / through / switch), and apply it on GPU as `F.embedding(ids, fixed_table)`. Semantics with
zero learned parameters and no exploration cost.

**Re-measure before dismissing this.** Your benchmark note says "embeddings not worth it — 2× slower
FPS", but that measurement predates the contiguous shm layout and async GPU transfer work (commits
`9b1c612b8`, `034ee5fb8`, `8b634a396`). 127 × 1333 = 169k gathers per step is nothing for a 3080 —
the slowdown was almost certainly the int64 conversion and transfer path, not the gather. And the
comparison was never apples-to-apples: it pitted "slow but semantically correct" against "fast but
semantically broken".

This matters most for exactly the thing you're worried about: a semantic encoding **transfers**
across maps; raw-ID-as-float memorises per-map quirks.

### 1.10 Horizon

`gamma: 0.99` at 50/3 ≈ 16.7 decisions/s gives an effective horizon of ~100 decisions ≈ **6
seconds**. With a dense progress reward that's survivable, but it's short for a map segment that
takes 20s. Once the reward is dense and sane, push to `0.995`–`0.999` and watch value loss.

---

## Part 1b — The reward function to actually build

The design constraint, stated directly: **the reward may depend only on (a) artefacts you generate
yourself, and (b) map markers with enforced semantics.** That admits exactly three inputs — the
reference path (yours), elapsed time (universal), and `TILE_FINISH` (enforced on every race map).
Everything else in DDNet is mapper convention and will differ on the next map.

```
R(s, s') = −c_time  +  k · (γ·Φ(s') − Φ(s))  +  R_finish · [reached finish]

Φ(s) = normalised arc-length along the reference path ∈ [0, 1]
```

Three terms. No freeze penalty, no stuck ramp, no checkpoint bonus, no velocity-toward-target term,
no off-path penalty, no `m_BestPathProgress` state.

**Why the middle term is specifically a potential difference and not the current ratchet.**
[Ng, Harada & Russell (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
proved that shaping of the form `F(s,s') = γΦ(s') − Φ(s)` leaves the optimal policy **unchanged for
any Φ whatsoever**. That is precisely the robustness property you're asking for: if the ghost line
is mediocre, if the pathfinder route is suboptimal, if a segment near a teleport is garbage — the
shaping still cannot teach the agent the wrong thing. It only changes how quickly the value
function learns. A bad reference path costs you speed, never correctness.

The current high-water-mark progress reward has no such guarantee. `max(0, progress − best)` is
non-Markovian (it depends on best-so-far), so it is *not* potential-based, it *does* move the
optimum, and it produces the pathology in §1.1: once the frontier attempt fails, no progress reward
is reachable for the rest of the episode and every remaining tick is strictly negative. Under
potential shaping, moving back toward the frontier re-earns the reward, so the dead zone disappears.

Two honest caveats:
- With `γ < 1` and constant Φ, `γΦ − Φ = (γ−1)Φ ≠ 0` — a small drift proportional to progress. Keep
  Φ normalised to [0,1] and `k` modest, or use `γ = 1` inside the shaping term only (very common,
  technically breaks strict invariance, harmless in practice).
- `k` trades learning speed against freedom to deviate: large `k` learns fast and hugs the
  reference line, small `k` explores more. Unlike the ratchet, **no value of `k` changes the
  optimum** — so this is a pure speed/exploration dial, not a correctness risk.

Resist adding terms back. Every one you add is (i) not potential-based, so it *can* move the
optimum, and (ii) carries a hand-tuned scale that has no reason to hold on the next map. If freeze
avoidance is genuinely optimal, the time penalty already expresses it — being frozen costs time.

**On checkpoints as reward markers — your Tutorial example.** A checkpoint placed off the ideal
path is not an edge case to patch, it's proof the marker doesn't measure what you need. Two
consequences beyond the reward: `m_NextCheckpointX/Y` aims the agent at a place it shouldn't go,
and `CheckpointSpawnManager` respawns it somewhere the ghost never visited, where
`GetPathProgress` is ill-defined. Both dissolve when progress comes from the path.

**If you keep checkpoints for any purpose, derive their order from the path, not from the tile
index.** The current pipeline sorts by mapper-authored number and then reassigns sequential indices
(`botmanager.cpp:1063-1070`), which does handle *gaps* — but the sort key is still authored, and it
has three further failure modes:
- `TILE_CP`/`TILE_CP_F` are assigned index `−1` (`:971-975`), so every unindexed checkpoint sorts
  ahead of every indexed one, in arbitrary BFS-discovery order. Maps that mix both get garbage
  ordering.
- `std::sort` is not stable, so duplicate numbers — which mappers do produce — order arbitrarily.
- Authored numbering need not be monotonic along the route, especially where the route doubles back.

Replacing the sort key with `GetPathProgress(cp_center)` makes gaps, ties, non-monotonic numbering
and the indexed/unindexed mix all stop mattering at once, and lets you *discard* any checkpoint
further than some threshold from the path — which is exactly the Tutorial case. But this is only
worth doing if checkpoints survive at all; §3.1's `CSaveTee` snapshots supersede their remaining
use as respawn anchors.

## Part 2 — Is there something better than PPO?

Short answer: PPO is a reasonable choice for your throughput regime, and it's not what's holding
you back. But there are two real wins available, and one strong existence proof pointing at a
specific alternative.

### The existence proof: Linesight

[Linesight](https://github.com/Linesight-RL/linesight) is the closest published analogue to what
you're building — single-agent, deterministic physics, race against the clock, no self-play. It
reached human level in 2023 and by May 2024 beat **10 of 12 official Trackmania campaign world
records**. It uses [**IQN**](https://arxiv.org/abs/1806.06923) (Implicit Quantile Networks —
distributional, off-policy, replay buffer) with virtual checkpoints ~10m apart.

Why off-policy is the right shape for *your* problem specifically: **your environment is
deterministic.** When the agent finally nails the hard jump once, that trajectory is a permanent,
exactly-reproducible asset. PPO discards it after 4 epochs. A replay buffer lets you re-learn from
it hundreds of times. For hard exploration, that is the entire difference between "learned the
trick" and "saw it once and forgot".

### The options, ranked for your situation

| Approach | Verdict |
|---|---|
| **PPO + self-imitation** ([SIL](https://arxiv.org/abs/1806.05635)) | **Do this first.** Keep a prioritised buffer of your best-return trajectories, add an off-policy imitation loss on them. Small diff to `ppo_hybrid.py`, directly attacks "rare success gets forgotten". |
| **IQN + prioritised replay** | **The ambitious target.** Matches the Linesight recipe. Bigger rewrite, best evidence. |
| [**PQN**](https://arxiv.org/abs/2407.04811) (Parallelised Q-Network) | Interesting middle ground: no replay buffer, no target network, LayerNorm + Q(λ). Purpose-built for vectorised envs like yours, reported to beat PPO on wall-clock. But *no replay* means it doesn't buy you the rare-success-retention property, which is the thing you actually need. |
| **FastTD3 / SimbaV2** | Strong 2025–26 results, but they're continuous-control methods. Your action space is mostly discrete. Poor fit. |
| **DreamerV3 / TD-MPC2** (world models) | **Skip.** These trade compute for sample efficiency. You have a free deterministic simulator at 21k FPS — ~1.8B frames/day. You'd be spending GPU to save samples you aren't short of. Revisit only if the *world model itself* becomes the deliverable. |

**Recommendation:** don't switch algorithms until Part 1 and Part 3 are done. Then add
self-imitation to PPO. Treat IQN as a v2 milestone, not a prerequisite.

---

## Part 3 — Fixing the frontier problem properly

This is the core of the roadmap. Three layers, each useful on its own.

### 3.1 State snapshot/restore over shared memory

Extend the protocol so Python can say "reset bot *i* to snapshot *k*":

```c
// rl_protocol.h
struct SAction {
    ...
    int32_t m_RestoreSlot;   // -1 = normal spawn, >=0 = restore CSaveTee from slot
};
```

Server-side, keep `std::vector<CSaveTee> m_vSnapshots` per map (shared across bots — the tee state
is position/physics, not identity) plus an API to capture the *current* state of bot *i* into a new
slot. `CSaveTee::Save()` / `Load()` do the work; you're writing plumbing, not physics.

Two gotchas to plan for:
- `CSaveTee::Save(pChr, AddPenalty)` — pass `AddPenalty = false`, you don't want the `/save` time
  penalty.
- `m_aGameUuid` and `MatchCharacter` are for cross-session team saves. For in-process snapshots you
  can bypass the string serialisation entirely and keep `CSaveTee` objects in memory.

### 3.2 Backward curriculum from a ghost (do this before Go-Explore)

This is the highest value-per-line change on the list, and it's the direct answer to
"we spend all our time walking to the frontier".

[Salimans & Chen (2018)](https://arxiv.org/abs/1812.03381) solved Montezuma's Revenge from *a
single demonstration* with an embarrassingly simple recipe: reset the agent near the **end** of the
demo; when it succeeds reliably, move the reset point backward; repeat until the reset point is the
map start. Every episode is then ~5–15 seconds of work at the actual frontier. No walking.

You have the demonstration: `ghosts/Tutorial_Infinity_187.120_....gho`, and you can record a ghost
for any map by playing it once.

**How to turn a ghost into restore points.** From `ghost_parser.py:351`, each
`GHOSTDATA_TYPE_CHARACTER` item carries `X, Y, VelX, VelY, Angle, Direction, Weapon, HookState,
HookX, HookY, AttackTick`. That's enough to *construct* a `CSaveTee` directly — no inverse dynamics
needed. Every ~25 ticks along the ghost, emit a snapshot. A few fields aren't in the ghost
(`m_FreezeTime`, `m_Jumps` remaining, tele checkpoint); default them and accept small inaccuracy —
for a reset ladder that's harmless.

If you later want *exact* restore points, note that the ghost gives you three of the four inputs
already (`Direction`, `HookState` → hook, `Angle` → aim; `Fire` from `AttackTick`). Only **jump**
is missing, and physics is deterministic, so you can recover it by replaying the trace and doing a
1-step search over `jump ∈ {0,1}` per tick, keeping whichever matches the recorded next position.
That reconstructs a fully faithful input trace — and, incidentally, gives you real
**behaviour-cloning data** after all. Worth doing once the cheap version is working.

**Ghosts are 25 Hz, not 50 Hz.** Verified on the Tutorial ghost: every item is
`GHOSTDATA_TYPE_CHARACTER` (type 2) carrying an explicit tick stamp, and the stamps step by **2**
(`4251634, 4251636, 4251638, …`) — 4678 items for a 187.12 s run. Client ghosts are recorded from
*snapshots*, which arrive at half the tick rate. Three consequences:

- The header's `num_ticks` is an **item count**, not a game-tick count. Don't treat it as duration.
- For the restore ladder this is irrelevant — you want a snapshot every ~25 ticks anyway.
- For the jump-recovery trick it means searching 2 ticks at a time (4 combinations) instead of 1.
  Still trivial, and the recorded `VelX/VelY` gives you an exact match target.
- If you ever do behaviour cloning, use `frame_skip: 2` so decisions align with ghost frames.
  `frame_skip: 3` does not divide the ghost cadence.

**The curriculum controller** (Python side, replaces `CheckpointSpawnManager`):
- Maintain a cursor `s` into the snapshot ladder, starting at the last snapshot before the finish.
- Sample most resets from `[s, s+w]`, plus a `begin_bias` fraction from the true start (keep this —
  it's your honest metric and it prevents the from-start policy rotting).
- When rolling success rate from `s` exceeds ~70%, decrement `s`.
- Log `s` as the headline training curve. It *is* the progress metric.

### 3.3 Go-Explore archive (the generalisation)

Once 3.2 works, generalise to maps with no ghost, and to routes better than the human's:

- **Cell key**: `(tile_x // 4, tile_y // 4, sign(vel_x), vel_y bucket, hooking?, jumps_left)`.
  Include velocity — this is exactly what checkpoint-respawn gets wrong.
- **Archive entry**: best `CSaveTee` reaching that cell + ticks-to-reach + visit count.
- **Reset sampling**: weight toward high path-progress and low visit count.
- **Update**: any rollout reaching a novel cell, or an old cell faster, updates the archive.
- **Robustification**: anneal the reset distribution back toward the start as the frontier
  saturates, otherwise you get a policy that can do every segment but can't chain them
  (classic Go-Explore phase-2 failure).

References worth reading before implementing:
[Go-Explore](https://arxiv.org/abs/1901.10995),
[SCOUT: per-context reset curricula](https://arxiv.org/html/2607.26417),
[Emergent Dexterity via Diverse Resets](https://arxiv.org/pdf/2603.15789) — the last one argues
diverse resets alone often beat an explicit curriculum, which is a cheap ablation for you to run.

---

## Part 4 — Foundation models: what's actually feasible

Your instinct was "this might be too data- or compute-constrained". That's correct for one of the
three tiers and wrong for the other two.

### Tier 1 — LLM in the outer loop as reward/curriculum engineer ✅ do this

[Eureka](https://arxiv.org/abs/2310.12931) has an LLM read the environment source and the task
description, write an executable reward function, look at the resulting training statistics, and
rewrite it — evolutionary search over reward code. It beat human-expert reward design on 83% of 29
tasks. [RDA (2026)](https://arxiv.org/pdf/2606.01672) extends this with *visual trajectory
analysis*: instead of reflecting only on scalar statistics, the model looks at rendered rollouts to
diagnose failure modes.

**DDNet is an unusually good fit for this**, better than the robotics domains these were built for:

- **Maps are ASCII.** A DDNet map is a 2D tile grid. Your 43×31 observation window renders to a
  readable ASCII block in ~1400 characters. A model can *read the level*.
- **Failures are legible and localised.** "94% of episodes end within 2 tiles of (312, 88). Here is
  that neighbourhood as ASCII. Here are 5 rendered trajectories overlaid." That's a well-posed
  question a frontier model can genuinely reason about.
- **The output is small.** A shaping term, a subgoal, a curriculum threshold, a tuning value.
- **The cost is negligible.** One LLM call per few thousand GPU-seconds of training.

This *is* the "reasons about a hard frontier task the way a human would" system you described, and
it needs **zero** game data. Build it as: rollout stats + death heatmap + ASCII map excerpt +
current reward code → model → patched reward/curriculum config → retrain → repeat.

### Tier 2 — LLM as skill/subgoal proposer ⚠️ feasible, do after Tier 1

DDNet has a real, small, named vocabulary of manoeuvres — hook-fly, edge-hook, double-jump-hook,
rocket jump, speeder, corner-jump, drag, hammerfly. At the frontier, hand the model the local ASCII
map plus tee state and ask which manoeuvre applies and where the intermediate target is. Use the
answer as (a) a subgoal position for a goal-conditioned low-level policy, or (b) a temporary
shaping term.

The engineering that makes this work isn't the LLM — it's having a goal-conditioned low-level
policy in the first place. Sequence it after Tier 1.

### Tier 3 — foundation policy in the control loop ❌ not now

16 Hz × 127 parallel agents through a large model on one RTX 3080 is not happening, and you have no
pretraining corpus to distil from. Correctly ruled out.

### On teehistorian

Worth one email. [Teehistorian](https://ddnet.org/docs/libtw2/teehistorian/) records *all server
input* faithfully — it is exactly the IL corpus you'd want, and DDNet's own tooling
([`ddnet-scripts/teehistorian/archive.sh`](https://github.com/ddnet/ddnet-scripts/blob/master/teehistorian/archive.sh))
shows they index and xz-archive it on their infrastructure. It isn't published, but it exists, and
researchers have been given access before (Zwelf's parser/replayer work). Cost: one message to the
DDNet team. Upside: a real behaviour-cloning corpus.

Meanwhile, note that Tiers 1 and 2 need none of it, and §3.2's jump-recovery trick gives you
demonstration data from ghosts you record yourself.

---

## Part 5 — Prioritised plan

### P0 — Make the agent able to see, and the objective map-independent

*Reordered: §1.8 blindness now leads. No reward function fixes an unobservable hazard, and nothing
below generalises across maps until the observation does.*

1. **Rewrite `CalculateReward` to the three-term form in Part 1b.** Potential-based shaping,
   time penalty, finish bonus. Delete the freeze penalty, stuck ramp, off-path penalty,
   velocity-toward-checkpoint term, the fallback checkpoint branch, and `m_BestPathProgress`.
   This is a large *deletion*, which is the point.
2. **Purge time checkpoints** (§1.7, Part 1b) from reward and observation. Replace
   `m_NextCheckpointX/Y` / `m_CheckpointIndex` / `m_TotalCheckpoints` with next-10 waypoints
   egocentric + normalised arc-length.
3. **Segment the waypoint path at teleports** (§1.7); constrain `FindClosestWaypoint` to the
   current segment. Fixes the wrong-match progress corruption in §1.6 too.
4. **Speedup and switch layers into the observation** (§1.8). Settle the front-layer question with
   a corpus scan first and add it only if the count justifies it.
5. **Fixed semantic tile encoding** (§1.9), 256 → ~12 channels via a non-learned GPU lookup.
   Re-benchmark against the current float32-raw-ID path on the contiguous shm layout.
6. Add a regression test: *freeze-then-clear-obstacle must outscore never-attempting*.
7. Action space (§1.4): drop aim magnitude; factored Categorical-sector × continuous-residual head.
8. **Instrumentation before anything else** — death-position heatmap, path-progress histogram,
   from-start finish rate. You cannot debug a frontier you can't see. `visualize_waypoints.py` is
   the seed for this.
9. Change the headline metric from mean episode reward to **from-start finish rate** and
   **finish time vs. ghost**.

A useful forcing function for 1–4: pick **two** maps with visibly different mapper style and
require the same config to train on both. Cross-map validity is a property you have to test for,
not one you get by intending it.

### P1 — Fix the reset distribution (1–2 weeks, the big one)

7. `CSaveTee` snapshot/restore across shared memory (`m_RestoreSlot` in `SAction`).
8. Ghost → snapshot ladder: construct `CSaveTee` from ghost fields every ~25 ticks.
9. Backward curriculum controller replacing `CheckpointSpawnManager`.
10. Go-Explore archive with velocity-aware cell keys; anneal resets back toward start.

### P2 — Learning improvements

11. Self-imitation / prioritised replay of best trajectories on top of PPO.
12. Recurrence (GRU + truncated BPTT), or deep temporal-conv player-state stacking as the cheap
    first test.
13. Raise `gamma` to 0.995–0.999 once rewards are dense and sane.
14. If P0–P2 land and you want the ceiling: port to IQN + PER, the Linesight recipe.

### P3 — Foundation model

15. Eureka-style outer loop: stats + ASCII map + death heatmap → reward/curriculum patch.
16. LLM subgoal proposer at the frontier, over a goal-conditioned low-level policy.

### Stretch — team race

`CSaveTeam` already exists, so the snapshot/restore machinery carries over unchanged. A 2-player
shared-parameter policy on a map *designed* for two is far more impressive to a DDNet audience than
another solo run, and less hopeless than it sounds — but only attempt it after a solo Novice map
finishes reliably from start.

---

## Part 6 — Things to decide

- **Target map: keep Tutorial, and keep run_guy_25 as the stretch.** Don't switch to a shorter map
  — that treats the symptom.

  The reason long maps were catastrophic is that the cost is *superlinear*. To train at frontier
  distance `d`, each episode spends ~`d/v` seconds walking and ~`T` seconds at the frontier, so the
  useful fraction is `T/(d/v + T)`, decaying as `1/d`. Integrated over map length that's **O(L²)**.
  With state restore (§3.1–3.2) every episode starts *at* the frontier, so it's **O(L)**. For a
  187-second map that factor is the whole ballgame — and it means map length stops being the thing
  you design around.

  What you do want is a **fast smoke-test map** for P0 iteration: a ~20-second slice of Tutorial
  (truncate the waypoint file and set the finish early) so reward/action-space changes give signal
  in 15 minutes instead of overnight. Target map and iteration map are different jobs.
- **Route pinning.** The off-path penalty (§1.6) forecloses route discovery. It's a small term, so
  this is a cheap flag either way — but decide deliberately rather than by default.
- **Server count.** Your own benchmarks say 8 servers cycles faster than 12 on a 16-thread box
  despite lower raw FPS. The `default.yaml` still says 12.
