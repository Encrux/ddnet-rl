# Plan of Action

Companion to `roadmap.md` (analysis). This is the execution order, with file targets and an
acceptance test for every step. Section references point back into `roadmap.md`.

**Critical path:** Phase 0 → Phase 1 → **gate** → Phase 2 → **demo** → Phases 3–5 in any order.

Phases 0 and 1 are prerequisites for everything: without instrumentation you can't tell whether a
change helped, and without a trustworthy progress signal the Phase 2 curriculum has nothing to
drive it.

**Two rules that save rework:**
- Batch all `rl_protocol.h` changes into **one** commit (steps 1.3 + 1.4). Every protocol change
  costs a rebuild, a Python parser update, and a `static_assert` fix.
- Once you change the reward, **episode-reward curves are incomparable across runs.** Land the
  metrics in Phase 0 first or you lose the ability to compare anything to anything.

---

## Phase 0 — Make iteration fast and observable ✅ DONE

*Pure Python, no protocol churn, no C++ rebuild.*

**Shipped:** `path_progress.py`, `episode_tracker.py`, `test_phase0.py`,
`configs/slice.yaml`, plus edits to `teeworlds_server_env.py` and `train.py`.
`python test_phase0.py` runs the acceptance suite without a server.

**Findings from building it:**

1. **Wrong-strand matching is real and worse than assumed.** Tutorial's reference
   path passes within ~100 units of a point ~1000 units later along itself, only
   **~17 waypoints apart** — far inside the ±100 window the C++
   `FindClosestWaypoint` uses. Measured against ground truth (waypoints are
   resampled from the ghost, so cumulative ghost length *is* the true arc
   position), a wide spatial window gives max error **0.018** of the map and
   jumps of 1052 units in a single frame; physically-sized temporal matching
   gives **0.0045**, which is the resampling floor. §1.6/§1.7 in `roadmap.md`
   understated this — it does not need a teleport to fire, just a doubling-back
   path, which is most DDNet maps.
2. The fallback scan must be **forward-only**. A backward global snap mid-episode
   is always wrong, because it lets the match jump onto an earlier strand.
3. **The ghost path does not start at spawn.** Tee spawns at (334, 2225); the
   path starts at (769, 2347), 452 units away, and its first four waypoints run
   *backwards* (769 → 609). Ghost recording begins at the START tile, not the
   spawn tile. Progress is flat over that gap, so shaping gives no gradient for
   the opening seconds of every episode. The env now warns; see Phase 1.2b.
4. Cost of the progress computation: **0.069 ms/update at 127 envs** — 0.25%
   overhead at the observed step rate.

**Env repair, incidentally:** the venv's base interpreter was gone (miniconda
removed) though `site-packages` was intact. Relinked to a uv-installed CPython
3.13 and added `pyyaml`, rather than re-downloading multi-GB torch.

### 0.1 Virtual finish line → arbitrary-length map slices

Don't edit maps. Add `finish_progress: float` to config: the episode terminates with the finish
bonus when normalised path progress ≥ threshold. Gives you a 15-second slice of any map for fast
iteration, and the full map by setting it to 1.0.

- **Files:** `configs/default.yaml`, `teeworlds_server_env.py` (episode termination),
  `botmanager.cpp` (finish check, or keep it Python-side initially)
- **Accept:** `finish_progress: 0.08` on Tutorial gives episodes that end in ~15 s of game time; a
  full training run produces a visible learning curve in under 20 minutes.

### 0.2 Instrumentation

Three things, all logged per update:

- **Death-position heatmap** — 2-D histogram over tile coords at episode end, dumped as an image to
  TensorBoard every N updates. This is the one that lets you *see* the frontier.
- **Progress histogram** — P10/P50/P90/max of normalised progress at episode end.
- **From-start finish rate** — fraction of episodes that started at the map start and finished.

- **Files:** `train.py` (logging), `teeworlds_server_env.py` (expose terminal position + spawn
  origin per episode), `visualize_waypoints.py` (reuse its rendering)
- **Accept:** you can open the heatmap and point at the specific wall the agent dies on.

### 0.3 Switch the headline metric

Promote **from-start finish rate** and **finish time vs. ghost**; demote mean episode reward to a
diagnostic. Reward is about to change and stops being cross-run comparable.

- **Accept:** the top-line chart you look at first is finish rate.

---

## Phase 1 — One clean, map-independent signal

*~3–5 days.*

### 1.1 Reward rewrite (mostly deletion) — `roadmap.md` Part 1b

**File:** `botmanager.cpp:699-838` (`CalculateReward`), `botmanager.h:100-123` (`SBotRewardState`)

Target:

```cpp
float Phi = GetNormalizedProgress(Pos, State);          // ∈ [0,1]
float Reward = -m_Cfg.m_TimePenalty;                     // per tick
if(DecisionTick) {                                       // see gotcha below
    Reward += m_Cfg.m_ShapingK * (m_Gamma * Phi - State.m_LastPhi);
    State.m_LastPhi = Phi;
}
if(ReachedFinish) Reward += m_Cfg.m_FinishBonus;
```

**Delete:** freeze penalty (both terms), stuck counter + ramp, off-path penalty,
velocity-toward-checkpoint, checkpoint bonus, the entire no-waypoint fallback branch
(`:753-789`), and from `SBotRewardState`: `m_BestPathProgress`, `m_BestDistToCheckpoint`,
`m_LastDistToCheckpoint`, `m_StuckCounter`, `m_LastCheckpointRewarded`, `m_NumCheckpointsReached`,
`m_HighestTimeCp`, `m_WasFrozen`.

**Keep** `m_VisitedRegions` — it's already the right shape for the Phase 2 archive cell key.

> **Gotcha — apply shaping at decision ticks, not every tick.** Reward is currently accumulated
> per tick across the frame skip (`:574`). Summing `(γΦ_{t+1} − Φ_t)` over 3 ticks does **not**
> equal `γ³Φ_{t+3} − Φ_t`, so the telescoping that makes potential shaping policy-invariant breaks.
> The discount in the shaping term must be the agent's γ at the agent's decision rate. Time penalty
> stays per-tick; shaping fires once per decision.

> **Gotcha — γ must match Python's.** Plumb it through config rather than hardcoding it in two
> places that will drift.

- **Accept:** a regression test over synthetic trajectories where *freeze-for-2s-then-clear-the-obstacle*
  scores strictly higher than *never attempt*. Add it as a C++ unit test.

### 1.2 Segment the waypoint path at teleports — `roadmap.md` §1.7

**File:** `botmanager.cpp` (waypoint load, `FindClosestWaypoint:1181`, `GetPathProgress:1207`)

At waypoint load, walk the list and mark index `i` as a portal break if the segment
`(wp[i], wp[i+1])` intersects solid tiles — use `m_pCollision`, **not** a distance threshold
(free-fall legitimately reaches 72 units/tick and ghosts are 25 Hz, so ~145 units between samples
is normal). Store a per-waypoint segment id and per-segment cumulative distance, portal edges
contributing **zero** length. Constrain `FindClosestWaypoint` to the current segment ± window, with
a full-scan fallback when the windowed best distance exceeds ~512 units.

**Port the temporal matching from `path_progress.py`** — it is already written and validated, and
Phase 0 finding (1) shows the current C++ window is wrong on Tutorial today, not just on some
hypothetical teleporter map. Specifically: size the forward window from `MAX_TEE_SPEED * frame_skip`
rather than a fixed ±100, cap backward movement, and make the fallback forward-only.

- **Accept:** replay the Tutorial ghost's own positions through `GetPathProgress` and assert Φ
  sweeps 0 → 1 with max error ≤ 0.006 against cumulative ghost length. `test_phase0.py` does exactly
  this for the Python implementation; mirror it in C++. This single test catches segmentation bugs,
  the wrong-match corruption from §1.6, and phantom-waypoint smearing at once.

### 1.2b Make the reference path start at spawn

Phase 0 finding (3): the ghost path begins 452 units past the spawn and its first four waypoints run
backwards. Prepend the spawn position to the waypoint file (or trim the leading detour) so shaping
has a gradient from tick 0. The env already warns when the gap exceeds 128 units.

### 1.3 + 1.4 Protocol change — do these together, one commit

**File:** `rl_protocol.h`, `botmanager.cpp:642` (`ExtractTileGrid`), `teeworlds_server_env.py`
(obs parsing)

- **Remove:** `m_NextCheckpointX/Y`, `m_CheckpointIndex`, `m_TotalCheckpoints` (16 bytes).
- **Add:** `float m_Progress` + `float m_aNextWaypoints[10][2]` egocentric (84 bytes).
- **Add:** speedup and switch planes to the tile grid; stop letting the tele layer overwrite the
  game tile (`:680-683` — keep both); carry tile flags (rotation matters for `TILE_STOP` and
  `TILE_THROUGH_DIR`).
- **Delete** the unused `ETileType` enum (`rl_protocol.h:34-52`) — it documents an encoding the
  code doesn't implement.
- **Add** `int32_t m_RestoreSlot` to `SAction` now, replacing `m_SpawnCheckpoint`, so Phase 2
  doesn't need a second protocol churn. Leave it inert until 2.1.

- **Accept:** `static_assert`s pass, Python obs shape matches, one full training update runs
  end-to-end without shape errors.

### 1.5 Front-layer corpus scan (settles the open question in §1.8)

Needs a map corpus (`--maps-dir` / ddnet-maps). Load each map and count front-layer tiles in the
gameplay-relevant ranges (`TILE_FREEZE..TILE_TELE_LASER_DISABLE`, `TILE_LFREEZE..TILE_LUNFREEZE`,
stoppers, `THROUGH*`).

- **Decision rule:** add the front plane only if a non-trivial fraction of maps use it. Otherwise
  drop it *and* remove the front-layer special case in the checkpoint scanner
  (`botmanager.cpp:984-996`) so the codebase stops hedging both ways.

### 1.6 Fixed semantic tile encoding — §1.9

Fixed, non-learned 256 → ~12 channel lookup applied on GPU as `F.embedding(ids, fixed_table)`.

- **File:** `ppo_hybrid.py`
- **Accept:** benchmark against the current float32-raw-ID path *on the contiguous shm layout*. The
  old "embeddings 2× slower" result predates commits `9b1c612b8` / `8b634a396` and compared
  "slow but correct" against "fast but semantically broken".

### 1.7 Action space — §1.4

Drop aim magnitude. Replace the Gaussian aim angle with a factored head: Categorical over 16–32
sectors × a continuous residual within the sector.

- **File:** `ppo_hybrid.py` (`get_action:370`, distribution construction `:394-397`)
- **Accept:** log-prob and entropy are correct — verify by sampling 10⁵ actions and checking the
  empirical angle distribution matches the analytic one; then confirm no regression on the slice map.

---

## 🚦 Gate — run a baseline before building Phase 2

With the clean reward and the 15-second slice, train from a normal start.

- **If the agent solves the slice:** proceed to Phase 2. The frontier problem is now the only thing
  between you and the full map.
- **If it doesn't:** stop. Something more basic is broken (action space, observation, credit
  assignment), and state restore will not save it. Debug with the death heatmap first.

---

## Phase 2 — Kill the frontier problem

*~1–2 weeks. The main event.*

### 2.1 `CSaveTee` snapshot/restore over shared memory — §3.1

**Files:** `botmanager.h/cpp` (add `std::vector<CSaveTee> m_vSnapshots`, `CaptureSnapshot(int) -> int`),
reset path at `botmanager.cpp:355-378`

Restore replaces the `ForceSpawn(SpawnPos)` + `ResetRewardState(i, SpawnCp)` path: `ForceSpawn`
first so a character exists, then `m_vSnapshots[slot].Load(pChr)`.

- **Gotcha:** call `Save(pChr, /*AddPenalty=*/false)` — you don't want the `/save` time penalty.
- **Gotcha:** pass `std::nullopt` for team on `Load()`. Skip the string serialisation entirely and
  keep `CSaveTee` objects in memory; `m_aGameUuid`/`MatchCharacter` are for cross-session team saves.

- **Accept (determinism test):** capture at tick T, restore, replay an identical input sequence,
  assert bit-identical positions for 100 ticks. If this fails, everything downstream is unsound.

### 2.2 Ghost → snapshot ladder — §3.2

Construct `CSaveTee` objects directly from ghost fields (`X, Y, VelX, VelY, Angle, Direction,
Weapon, HookState, HookX, HookY, AttackTick`) every ~12 ghost frames (≈24 ticks ≈ 0.5 s — ghosts
are 25 Hz, so this is *half* the game tick rate).

Not in the ghost: `m_FreezeTime`, `m_Jumps`, `m_TeleCheckpoint`, `m_DDRaceState`. Default them.

- **Gotcha:** a rung where the ghost was mid-freeze restores as *unfrozen*, which is an invalid
  state that may be trivially or impossibly solvable. Skip rungs whose position sits on a freeze
  tile.

- **Accept:** every rung restores without immediate death, and Φ at rung *i* is strictly increasing
  in *i*.

### 2.3 Backward curriculum controller — §3.2

Python, replaces `CheckpointSpawnManager` (`teeworlds_server_env.py:135-199`).

- Cursor `s` over rungs, starting at the last rung before the finish.
- Sample most resets from `[s, s+w]`; keep `begin_bias` from the true start — it's your honest
  metric and it stops the from-start policy rotting.
- Advance `s -= 1` when rolling success rate from `s` exceeds ~0.7 over ≥N episodes.
- **Success = reaching rung `s+1`'s progress**, not finishing the map.
- Log `s` as a headline curve — it *is* the progress metric.

- **Accept:** `s` decreases monotonically; from-start finish rate rises as `s` approaches 0.

### 2.4 Go-Explore archive — §3.3

Generalises 2.3 to maps with no ghost, and to routes better than the human's. Cell key
`(tile_x//4, tile_y//4, sign(vel_x), vel_y bucket, hooking?, jumps_left)` — velocity in the key is
exactly what checkpoint respawn got wrong. `m_VisitedRegions` is the seed.

Anneal the reset distribution back toward the start as the frontier saturates, or you get a policy
that can do every segment and chain none of them.

### 2.5 🎯 Demo milestone

Tutorial finished from start, unassisted, on video, side-by-side with the human ghost.

---

## Phase 3 — Learning improvements

*Only after the demo. Each is independently droppable.*

- **3.1 Self-imitation** — prioritised buffer of best-return trajectories + off-policy imitation
  loss on `ppo_hybrid.py`. Highest value/effort ratio here; directly fixes "rare success gets
  forgotten".
- **3.2 Recurrence** — GRU over the encoder + truncated BPTT. Cheap first test: deep temporal-conv
  player-state stacking (16–32 frames) instead of flat concat.
- **3.3 γ → 0.995–0.999** once rewards are dense and sane. Watch value loss.
- **3.4 IQN + PER** — the Linesight recipe. Big rewrite, best evidence, only if you want the ceiling.

## Phase 4 — Generalisation

- **4.1 Two-map validation.** Pick two maps with visibly different mapper style; require the *same
  config* to train both. Cross-map validity is tested for, not intended.
- **4.2 run_guy_25.** Long is fine now — §3.1–3.2 turn the cost from O(L²) into O(L).

## Phase 5 — Foundation model

- **5.1 Eureka-style outer loop** — rollout stats + death heatmap + ASCII map excerpt + current
  reward code → model → patched reward/curriculum → retrain. Note Phase 0.2 already produces every
  input this needs.
- **5.2 LLM subgoal proposer** at the frontier, over a goal-conditioned low-level policy. Requires
  that low-level policy first.

---

## First session

1. `finish_progress` config + virtual finish (0.1)
2. Death heatmap + progress histogram + finish rate (0.2)
3. Flip the headline metric (0.3)
4. Baseline run on the 15-second slice with the **current** reward — you want the "before" picture
   before deleting anything.
