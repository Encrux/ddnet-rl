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

## 6. Where this leaves the wall

Exploration pressure does not solve it. Crossings happen occasionally and never
consolidate into the bulk policy, and the one change that manufactured more
commitment (G) made everything worse because blind execution is expensive
everywhere else on the map.

That argues for removing the need to *discover* the maneuver at all:
**seed the archive from ghost states past 0.4557** (Salimans & Chen 2018,
reset-from-demonstration). It assumes only that a human *was* there, not that
they were good, so it survives the objection that the ghost is not optimal. All
the machinery now exists — the archive stores `CSaveTee` snapshots, cells can
hold hooked states without them being overwritten, and `ghost_parser.py` already
decodes position, velocity and hook state.

The honest caveat for whatever comes next: at n=1 per arm and 65 minutes per run,
this design can only detect large effects. The crossing rate needs several
replicates per config to be measurable at all.
