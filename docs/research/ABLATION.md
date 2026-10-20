# Ablation: action hold, ratchet shaping, archive cells — 2026-08-05

Seven runs, 40M steps each, Tutorial slice at `finish_progress 0.50`. Every arm
used the **same binary and the same Python**; all three changes are behind
runtime flags, because comparing arms built from different binaries is how
`sv_rl_frame_skip` stayed a no-op for the project's entire history.

Analysis: `python analyze_ablation.py` and `plot_ablation.py`.

---

## 1. The wall

The obstacle at world (33,615, 1583) — a ~15-tile hook-climb needing a backward
run-up — sits at normalised progress **0.4557**. Four independent runs reproduce
that number to four decimal places.

`stall/exploration_edge` is **the wrong metric for it**. It requires ≥200 visits
in a bin, so it reports where the *bulk* of agents reach and cannot see a
breakthrough that starts as a handful of episodes. It reads exactly 0.4550
(= bin 91/200) in every arm regardless of what happened. Use the max of
`progress/hist` instead. This nearly caused the whole experiment to be reported
as a null result.

## 2. Results

| arm | config | peak | crossed | tail | hold dur | frozen | stuck |
|---|---|---|---|---|---|---|---|
| A | control | 0.4557 | 0/60 | 0/12 | — | 0.1382 | 0.0804 |
| B | +hold | 0.4557 | 0/60 | 0/12 | 2.21 | 0.1244 | 0.1078 |
| C | +ratchet | **0.4685** | 8/60 | 5/12 | 2.66 | 0.0294 | 0.0223 |
| D | +archive (beta −0.5) | 0.4557 | 0/60 | 0/12 | 2.51 | 0.0373 | 0.0361 |
| E | +archive (beta +0.5) | **0.4673** | 12/60 | 11/12 | 2.82 | 0.0328 | 0.0300 |
| F | replicate of C | 0.4557 | 0/60 | 0/12 | 2.83 | 0.0494 | 0.0369 |
| G | +hold entropy 0.03 | 0.3084 | 0/60 | 0/12 | 4.32 | 0.0000 | 0.0463 |

"crossed" = logged batches whose max progress exceeded 0.4557.

## 3. What replicates

**Ratchet shaping is the one keeper.** Across C, D, E and F it cuts freezing
from 0.138 to 0.029–0.049 (**2.8–4.7×**) and stuck-time from 0.080 to 0.022–0.037
(**2.2–3.6×**). C and F are the same config on different seeds and both show it,
so this is not a seed artifact.

Mechanism confirmed independently: removing the retreat penalty raises the hold
duration the policy chooses (2.21 hold-only → 2.66/2.83 with ratchet). The
penalty was what suppressed commitment.

**Forcing commitment backfires.** G raised mean hold duration to 4.32 as designed
and peak progress collapsed to **0.3084** — it never even reached the wall. The
blind-execution cost of a long hold is real and dominant. An entropy floor of
0.03 on the duration head is far too strong.

**Nothing moves the bulk policy.** Every arm's `exploration_edge` is 0.4550.

## 4. What does NOT replicate — claims retracted

- **"Ratchet enables crossing the wall."** 2 of 5 hold+ratchet runs crossed
  (C, E); D, F and G did not. F is a byte-identical replicate of C and scored
  zero. The crossing is a rare stochastic event, not a property of any config.
- **"The fixed archive improves on C."** E has the strongest crossing signal
  (11/12 tail) but cannot be credited when C's own replicate crossed zero times.
- **"The duration head costs ~16% FPS."** Based on B (8968) vs A (10715), but C
  and F — same config — differ by 9012 vs 10803, a 20% spread. FPS here is
  dominated by CPU contention between runs.

**Seed noise is wider than previously estimated.** The overnight A/C pair gave
±0.016 on `p50_from_start`; C vs F here differ by 0.0235. Use ~0.024. Several
deltas flagged "REAL" during the run fall back under that threshold.

## 5. The archive weighting bug

Cell weight shipped as `visits**-0.5`, a strong rarity preference. Rare cells are
overwhelmingly *hooked* cells, and hooked cells are transient, so **82% of
restores landed mid-hook** (31.6% hook+fall, 25.6% hook+level, 25.0% hook+rise
against 17.8% for all grounded classes combined). `p50_from_start` collapsed
0.363 → 0.205.

Rarity is a bad proxy for "worth practising" when most rare states are doomed.
Go-Explore cells are reachable *and resumable*; a rare cell here is often a tee
mid-fall into a freeze. Now `visits**+0.5`, measured live at 30.5% hooked, with
a water-filled per-variant cap as a backstop.

## 5b. The wall, actually broken (H1, H2)

The archive was already recording the rare episodes that clear the wall, and the
sampler was discarding them. `CaptureSnapshot` has no upper bound, so a crossing
leaves a snapshot at bin 92+; but the sampler used `hi = exploration_edge`, whose
floor no rarely-visited bin can clear. Never offered, so never accumulates
visits, so never becomes the edge, so never offered — self-limiting, the same
shape as the dwell-estimator loop fixed in `f0fc15377`.

Fixed by running the window to the furthest bin that HAS a snapshot and reserving
`frontier_share` (0.3) of restores for bins beyond the bulk edge. A fixed share,
not a weight: a bin-93 cell with 20 visits scores 4.5 at beta 0.5 against 224 for
a bin-85 cell with 50,000, so weighting alone gives them 0.00% of restores.

| | prior 7 runs | H1 | H2 |
|---|---|---|---|
| `stall/exploration_edge` | 0.4550 (all 7) | **0.4694** | **0.4650** |
| `episodes/finish_rate` | 0.00% (all 7) | **24.70%** | **11.00%** |
| peak progress | <= 0.4685 | 0.5048 | 0.5009 |

2/2 against 0/7, on a bulk-policy metric that no single episode can move. H1's
trace shows the mechanism directly: stalled at the wall from 22M like every other
run, one crossing at 32.5M, then edge 0.4550 -> 0.4700 and finish rate compounding
0.05% -> 1.2% -> 14% -> 24.7% within 3M steps.

This is the directed version of the rarity weighting that failed in §5. "Rare"
alone selects tees mid-fall into a freeze; "rare AND past the frontier" can only
select achievements.

**Retracted:** H1's `p50_from_start` of 0.212 was reported as the Go-Explore
chaining cost of this change. H2 came in at 0.3241, in line with the controls. It
was seed variance.

## 5c. The episode cap invalidated the from-start metrics

`--finish-progress 0.50 --max-episode-steps 1200` needs **1560 steps at ghost
pace**. So for every run above:

- `finish_rate_from_start` was pinned at 0.00% by arithmetic, not difficulty;
- `p50_from_start` saturated — ceiling 0.3847, control at 0.3605, i.e. **93.7%**
  of it, so every comparison drawn on it sat in the top ~7% of its range;
- from-start episodes were truncated at 0.3847, **before the obstacle at 0.4557**,
  so the from-start policy never once reached the thing it was failing at.

The sizing rule was already in `configs/slice.yaml`; it was not re-derived when
CLI overrides moved the slice from 0.05 to 0.50. It applies to overrides too.

## 5d. Correct sizing works, and exposes a second trade (I1, I2)

At `max_episode_steps 3000` the from-start policy finally reaches the obstacle:
median from-start episode ends at **0.4518 / 0.4568**, essentially exactly the
wall at 0.4557, against a hard truncation at 0.3847 under the old cap. The agent
is now failing AT the wall rather than failing to arrive.

But the finish rate goes to zero and peak progress drops to 0.4740 (both runs,
identically). Every curriculum restore happens at an episode boundary, so tripling
episode length at a fixed step budget gives ~2.5x fewer restores. The curriculum
is what drives the frontier forward, and lengthening episodes throttles it —
archive max bin 93.5/92.5 against 95.6 at the short cap.

| | 1200 steps (H1/H2) | 3000 steps (I1/I2) |
|---|---|---|
| p50 from start | 0.212 / 0.324 | **0.452 / 0.457** |
| finish rate | **24.7% / 11.0%** | 0.00% / 0.00% |
| peak progress | 0.505 / 0.501 | 0.474 / 0.474 |
| archive max bin | 95.6 / 93.1 | 93.5 / 92.5 |

Both settings replicate 2/2 in opposite directions, so this is a real trade, not
seed noise. The two episode types want different caps: a from-start episode needs
~3000 steps to reach the wall, a restored episode starts near the frontier and
does not. Splitting the cap by episode type decouples curriculum turnover from
from-start reach.

Across all four frontier-share runs the exploration edge moved (0.4694, 0.4650,
0.4694, 0.4700) against 0.4550 in all seven runs without it — 4/4 vs 0/7.

## 5e. Episode length is a two-sided trade (J, K)

Splitting the cap by episode type was meant to take H's finish rate and I's
from-start reach together. Two attempts:

| run | config | edge | p50_from_start | finish |
|---|---|---|---|---|
| J1 | split caps, patience on ALL episodes | 0.4600 | 0.1953 | 0.00% |
| K1 | split caps, patience on RESTORED only | 0.4550 | **0.4524** | 0.00% |

**J1 was a plumbing error, not a design failure.** Applying stall patience to
from-start episodes killed them at the first place they stalled -- somewhere
early in the map, not at the frontier -- so they never traversed anything.
Median from-start progress hit 0.0556 at 20M against 0.21-0.31 for every other
config, and the run ended worse than BOTH configs it was meant to combine.
Fixed by gating patience on `~_episode_from_start`; `test_episode_caps.py` now
asserts that idle from-start episodes ignore patience entirely.

**K1 held the from-start column and ran out of budget on the other.** Its
p50_from_start of 0.4524 sits alongside I1/I2 (0.4518/0.4568), so the split works.
But it reached the wall at 29.9M leaving only 10.1M steps there, and H1's
crossing came 11.7M steps AFTER arriving. It never got enough time at the wall.

The slowness is arithmetic. With begin_bias 0.2, one episode in five starts from
the map beginning and those are now 3-4x longer than restored ones, so from-start
practice consumes ~55% of the step budget:

    0.2*2000 / (0.2*2000 + 0.8*400) ~= 0.55

That is the direct price of the long from-start cap: the frontier gets about
half the compute it had under the uniform 1200 cap, and arrives ~1.4x slower at
every milestone (0.30 at 15.6M vs H1's 13.0M, wall at 29.9M vs 20.8M).

Two levers if 60M is not enough: lower begin_bias to ~0.1 (same episode
structure, more compute at the frontier), or accept that these two goals need
separate runs.

## 5f. The hook aim-line, and the first from-start finishes (P1)

`m_HookState` was one-hot encoded over {0, 1, 2}. DDNet's enum is sparse and has
no value 2: RETRACTED=-1 (uint8 255), IDLE=0, RETRACT_START=1, RETRACT_END=3,
FLYING=4, GRABBED=5. So FLYING, GRABBED and RETRACTED all collapsed to the
all-zeros vector -- measured in play, the agent could not distinguish its hook
state 60.0% of the time, including the 41.1% it spent GRABBED. Our bug, not
DDNet's. Fixing it alone (runs N1/N2) changed nothing measurable; the agent had
been inferring attachment from the velocity signature.

What did work was giving it the information DDNet shows humans: the red/green
aim line, as 16 per-sector hook reachability probes aligned with the aim head.
Ported from CPlayers::RenderHookCollLine, sharing IntersectLineTeleHook with the
real hook so it cannot disagree with the physics -- including the diagonal
block-gap case where a hook aimed "into" no-hook tiles still reaches a hookable.
An observation, not a penalty on aiming at TILE_NOHOOK: no reward-hacking
surface, and a tile-type heuristic would get the gap case wrong.

| run | obs | edge | p50_from_start | peak | finish_from_start |
|---|---|---|---|---|---|
| I1 / I2 | hook-blind | 0.4694 / 0.4700 | 0.4518 / 0.4568 | 0.4740 / 0.4740 | 0% / 0% |
| N1 / N2 | hook fixed | 0.4550 / 0.4550 | 0.4519 / 0.3818 | 0.4729 / 0.4557 | 0% / 0% |
| **P1** | **+ aim line** | **0.4763** | **0.4586** | **0.5050** | **21.27%** |

Four consecutive 3000-cap runs capped at 0.4729-0.4740, which looked like a
second wall. P1 broke it at 36.4M and the trace shows the transition directly:
0.4731 x4, then 0.5003, 0.5031 x4, 0.5050. Overall finish rate 89.95%.

**`episodes/finish_rate_from_start` had been 0.0000 in every run of the project.**
Confirmed in-game by the user: the agent visibly clears the obstacle now.

Cost: FPS 10.8k -> 7.9k for 16 marches per bot per decision.

## 6. Where this leaves the wall

Exploration pressure does not solve it. Crossings happen occasionally and never
consolidate into the bulk policy, and the one change that manufactured more
commitment (G) made everything worse because blind execution is expensive
everywhere else on the map.

Superseded by §5b: replaying the agent's OWN rare crossings works and needs no
demonstration. Ghost seeding (Salimans & Chen 2018) drops to a fallback for
obstacles the agent never clears even once — self-seeding has nothing to
bootstrap from there. The machinery is in place either way: the archive stores
`CSaveTee` snapshots, cells hold hooked states without being overwritten, and
`ghost_parser.py` already decodes position, velocity and hook state.

Next binding constraint is chaining: `finish_rate_from_start` is 0.00% in every
run, but §5c shows that was structurally unreachable, so it has never actually
been measured.

The honest caveat for whatever comes next: at n=1 per arm and 65 minutes per run,
this design can only detect large effects. The crossing rate needs several
replicates per config to be measurable at all.
