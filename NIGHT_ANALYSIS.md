# Overnight session analysis — 2026-08-04/05

Five one-hour runs (A–E) on the Tutorial slice, plus instrumentation work.
Commits `d7384f2b1` and `f0fc15377`. Raw notes in the session scratchpad.

---

## 1. Where the agent actually is

Best result: **`progress/p50_from_start` = 0.398, exploration edge = 0.455** of the
Tutorial path (≈50,900 of 111,928 world units), after 107M cumulative steps.

For context on how fast this moved: at the start of the session the agent could not
finish 5% of the map; by the end the median from-start episode covers 40%.

| run | config | steps | edge | p50_from_start | thrash |
|---|---|---|---|---|---|
| A | baseline, dwell frontier | 36M | 0.386 | 0.345 | 0.688 |
| B | `ent_coef` 0.01→0.003 | 36M | 0.305 | 0.289 | 0.770 |
| C | replicate of A (new seed) | 36M | 0.359 | 0.329 | 0.753 |
| D | resume A → 107M | 71M more | **0.455** | **0.398** | 0.630 |
| E | sampler fix, resume D | in progress | 0.455 | 0.398 | 0.496 |

`finish_rate_from_start` is 0 throughout at `finish_progress 0.50` — expected, since
0.50 needs ~2.5× the pace the agent has. It is not a useful signal at this slice length.

---

## 2. What worked

**The frontier curriculum, decisively.** Established before tonight and unchallenged
since: from-start finish rate 0% → 49.7% on the 0.20 slice, and 1.4–1.9× fewer steps to
any given progress level, with the speedup *widening* with distance (1.40× at 0.03,
1.93× at 0.103). That widening is the O(L²)→O(L) signature and is the strongest single
result of the whole project.

**Continued training keeps paying, up to a point.** Run D's resume took the edge from
0.386 → 0.455 and cut mean dwell from 24 to 14.8 steps per bin. The agent got materially
faster through terrain it already knew.

**Placement must target the exploration edge.** See §4 — the fix is committed and its
mechanism is confirmed working (mean restore bin 73 → 85 against an edge at 91), with
thrash dropping 0.63 → 0.50 immediately on resume.

---

## 3. What didn't

**Lowering `ent_coef` (Run B).** Entropy fell 6.23 → 3.72 as intended and the exploration
edge plateaued: slope +0.0003/Mstep versus +0.0087 for baseline. The entropy bonus was
buying exploration the curriculum depends on. My causal story — high entropy → noisy
actions → thrashing — was backwards.

**Chasing "thrash" as an independent defect.** The ghost control settled this:

| | ghost @4 ticks | agent (Run A) |
|---|---|---|
| thrash | 0.078 | 0.688 |
| heading while progressing | +0.746 | +0.696 |
| heading while thrashing | +0.178 | +0.199 |

The agent's *motion quality matches a human in both regimes*. It just makes 4–9× more
non-productive steps — and that ratio is arithmetic, not a separate failure: the agent
runs at ~31% of ghost pace, and 1 − 0.31 = 0.69 = the thrash fraction. **Thrash is a
restatement of "too slow", already encoded in the time penalty.** It is a useful
*diagnostic* but not a lever.

**Two hypotheses tested and rejected outright:**
- *Vertical field of view.* Path sections taller than the 31-tile view hold 10.3% of
  stall time against 13.2% of the path — under-represented. Correlation +0.09.
- *Aim head sitting unused, hiding a sharp policy.* Action usage says the opposite: hook
  pressed on 42.5% of steps vs the ghost's 27.7% hook-engaged, moving on 94.1% vs 80.7%.
  The agent does *more* of everything.

---

## 4. The plateau, and the bug behind it

Run D's edge froze at **0.4550 for 64M consecutive steps** — slope exactly +0.00000 —
while every other metric kept improving. That is not noise or saturation.

Measured at the plateau:

```
exploration edge      bin 91   (progress 0.455)
frontier estimate     bin 79
mean restore bin      bin 73
sampler window        12 bins
```

**The sampler could never place an agent within 18 bins of the real frontier.** And the
loop was self-reinforcing: `frontier_bin()` returns the bin absorbing the most dwell
time, but *the curriculum is what puts agents in that bin*, so their dwell kept the
estimate pinned to its own restore point. The curriculum locked onto itself.

Fixed in `f0fc15377`: placement now targets the exploration edge — the furthest bin
agents reach, which no part of the curriculum can move on its own. Dwell is kept as a
diagnostic only.

Mechanism confirmed on resume: mean restore bin 73 → **85.1** (edge 91), well inside the
12-bin window.

**But the fix did not break the plateau.** 26M steps later the edge is still exactly
0.4550, slope +0.00000. The early improvement (thrash 0.63 → 0.41, heading 0.10 → 0.50)
was transient and reverted to thrash 0.74 as the policy adapted to the new restore
distribution.

So the placement bug was real, and worth fixing, but it was **not the cause of the
plateau**. Agents are now dropped within a window of the frontier and still cannot get
past it. That makes 0.455 a genuine capability limit, not a curriculum artifact — which
is a more informative result than the fix working would have been.

**The wall at 0.455** is at world (33,615, 1583), tile (1050, 49). The path there dips
then climbs ~15 tiles at near-constant x (wp 815 y=1600 → wp 823 y=1124) — a vertical
shaft that requires a hook.

A hook-climb is a multi-step precise sequence (aim, hook, swing, release, re-hook). With
a near-uniform aim distribution and no temporal structure in the policy, the probability
of stumbling onto it by exploration is negligible — and no amount of restoring *just
before* it changes that.

**This exposes a structural limit of the archive.** It is seeded exclusively from agent
experience, so it can only ever contain states the agent already reached. It therefore
cannot help past any wall the agent cannot cross unaided. The curriculum accelerates
progress through terrain that is already solvable; it cannot manufacture a capability.

---

## 5. Methodology findings (these matter more than any single result)

**Seed variance is large enough to invalidate n=1 conclusions.** Runs A and C are
identical configs with different seeds:

| metric | A | C | spread |
|---|---|---|---|
| exploration edge | 0.3862 | 0.3588 | 0.027 |
| thrash | 0.6880 | 0.7534 | 0.065 |
| heading, thrashing | 0.1990 | 0.1355 | 0.064 |
| entropy | 6.231 | 5.508 | 0.72 |

The A→B "effect" I reported as clear (thrash +0.082, heading −0.055) is **seed-scale**.
Only the edge *slope* difference (plateau vs +0.0087/Mstep) survives. Any future
single-run comparison needs an effect larger than ~0.03 on edge or ~0.065 on thrash to
mean anything.

**Always control against the ghost.** One cheap computation reframed the entire
"thrash" investigation. It should have been the first thing measured, not the tenth.

**Metrics that a control loop also drives cannot diagnose that loop.** The dwell-based
frontier estimator was fine as an observation and wrong as a placement key. Worth
checking the same property anywhere else a metric feeds back into behaviour.

**Operational:** importing `train.py` runs an `atexit` handler that `pkill`s all
DDNet-Server processes — verifying an edit that way kills any live run (cost: ~25 min and
one dead run). Runs need `setsid` plus an explicit liveness check.

---

## 6. Where to focus next

Ordered by expected value.

1. **Seed the archive from the ghost (roadmap §3.2).** This is now the top item. The
   archive cannot contain states beyond the wall because no agent has ever been there —
   but the ghost has. Constructing restore points from ghost fields (pos, vel, hook
   state are all in the `.gho`; `ghost_parser.py` already decodes them) puts the agent
   *past* 0.455 so it can learn the terrain beyond, and gives the backward curriculum
   something to walk back from. It is the one mechanism that breaks the
   experience-bootstrapping deadlock, and the parser work is already done.

   The stronger version, if restore-point seeding is not enough: recover the ghost's
   *actions*. Direction, hook state and aim angle are recorded directly; only jump is
   missing and is recoverable by a 2-tick search against the recorded velocity. That
   turns the ghost into behaviour-cloning data for exactly the segments the agent
   cannot discover.

2. **Widen `frontier_window` and raise `begin_bias`.** The window is 12 bins = 6% of the
   path; with restores now at the edge, a wider band gives more varied approach practice.
   `begin_bias` 0.2 → 0.3 remains untested and is the guard against Go-Explore phase-2
   (from-start p50 oscillates ±0.059 vs a baseline ±0.005 — segments learned faster than
   they are chained).

3. **The 452-unit spawn-to-path gap** (Phase 1 item 1.2b, still open). Progress is pinned
   at 0 there, so shaping gives no gradient for the opening seconds of *every* from-start
   episode, and it pollutes 4.2% of the stall signal. Cheap to fix, and from-start
   episodes are exactly what the headline metric measures.

4. **Freeze is real and rising** — 6.2% (A) → 9.7% (D), concentrated near the frontier.
   Tutorial is not as failsafe as assumed. Worth a per-bin freeze breakdown before it
   becomes the next wall.

5. **Per-head entropy still doesn't log** under `torch.compile` (FakeTensor shape
   mismatch on a direct actor call). Aggregate entropy cannot distinguish "uniform
   everywhere" from "sharp on movement, uniform on aim". Small fix, real diagnostic value.

**Not recommended:** more hyperparameter sweeps at n=1 (see §5), and reward shaping for
specific obstacles — the whole Part 1b argument was that situation-specific reward terms
are what made the old reward unfixable across maps, and reward cannot teach an agent to
act on information it does not have.
