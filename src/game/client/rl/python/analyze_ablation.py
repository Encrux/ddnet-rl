"""Compare ablation arms from their TensorBoard event files.

Usage:  python analyze_ablation.py runs/ablate_A_control runs/ablate_B_hold ...

Reports the headline metric (exploration edge), the wall at progress 0.455 that
the previous policy sat on for 64M consecutive steps, and steps-to-milestone so
a faster-but-equal-endpoint arm is still visible.

Seed variance from runs A and C of the overnight session bounds what counts as a
result: an edge difference below ~0.03 or a thrash difference below ~0.065 is
indistinguishable from a reseed at n=1. Differences under those thresholds are
printed but flagged.

NEVER import train.py from here -- its atexit handler pkills all game servers.
"""

import os
import sys

import numpy as np

# Established by re-running an identical config with a different seed
# (overnight runs A vs C). Anything smaller is not a result at n=1.
SEED_NOISE = {'stall/exploration_edge': 0.027, 'progress/p50_from_start': 0.016,
              'stall/thrash_fraction': 0.065}

WALL = 0.455  # where the previous policy froze for 64M steps
MILESTONES = (0.30, 0.35, 0.39, WALL, 0.50)

KEYS = [
    'stall/exploration_edge',
    'progress/p50_from_start',
    'progress/p50',
    'stall/thrash_fraction',
    'stall/frozen_fraction',
    'stall/stuck_fraction',
    'losses/entropy',
    'charts/fps',
    'hold/mean_duration',
    'hold/fresh_fraction',
    'archive/cells_per_bin',
    'archive/occupied_slots',
    'archive/hooked_cells',
    'curriculum/edge_bin',
    'curriculum/restore_bin_mean',
]


def load(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = set(ea.Tags()['scalars'])
    out = {}
    for k in KEYS:
        if k in tags:
            ev = ea.Scalars(k)
            out[k] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def tail_mean(series, frac=0.15):
    """Mean over the last `frac` of the run -- less seed-jittery than a last value."""
    if series is None or len(series[1]) == 0:
        return float('nan')
    v = series[1]
    n = max(1, int(len(v) * frac))
    return float(v[-n:].mean())


def steps_to(series, threshold):
    """First global step at which the metric reached `threshold`."""
    if series is None:
        return None
    steps, vals = series
    hit = np.nonzero(vals >= threshold)[0]
    return int(steps[hit[0]]) if len(hit) else None


def slope_per_mstep(series, frac=0.3):
    """Late-run slope. A flat edge with everything else improving is a plateau,
    which is what the 0.455 wall looked like -- exactly +0.00000/Mstep."""
    if series is None or len(series[1]) < 4:
        return float('nan')
    steps, vals = series
    n = max(2, int(len(vals) * frac))
    s, v = steps[-n:] / 1e6, vals[-n:]
    if s[-1] - s[0] < 1e-9:
        return float('nan')
    return float(np.polyfit(s, v, 1)[0])


def fmt(x, nd=4):
    if x is None:
        return "  --  "
    if isinstance(x, float) and np.isnan(x):
        return "  --  "
    return f"{x:.{nd}f}"


def main(dirs):
    runs = {}
    for d in dirs:
        name = os.path.basename(d.rstrip('/')).replace('ablate_', '')
        if not os.path.isdir(d):
            print(f"  [skip] {d} does not exist")
            continue
        runs[name] = load(d)
    if not runs:
        print("no runs found")
        return 1

    names = list(runs)
    w = max(14, max(len(n) for n in names) + 2)

    print("=" * (24 + w * len(names)))
    print("ABLATION SUMMARY  (tail mean over final 15% of each run)")
    print("=" * (24 + w * len(names)))
    print(f"{'metric':<24}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (24 + w * len(names)))

    for k in KEYS:
        if not any(k in r for r in runs.values()):
            continue
        row = f"{k:<24}"
        for n in names:
            row += f"{fmt(tail_mean(runs[n].get(k))):>{w}}"
        print(row)

    print("\n" + "-" * (24 + w * len(names)))
    print(f"{'edge slope /Mstep':<24}" +
          "".join(f"{fmt(slope_per_mstep(runs[n].get('stall/exploration_edge')), 5):>{w}}"
                  for n in names))
    print(f"{'final step':<24}" +
          "".join(f"{(str(runs[n]['stall/exploration_edge'][0][-1]) if 'stall/exploration_edge' in runs[n] else '--'):>{w}}"
                  for n in names))

    print("\n" + "=" * (24 + w * len(names)))
    print("STEPS TO EDGE MILESTONE  (millions; '--' = never reached)")
    print("=" * (24 + w * len(names)))
    print(f"{'milestone':<24}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (24 + w * len(names)))
    for m in MILESTONES:
        label = f"edge {m:.3f}" + ("  <- WALL" if m == WALL else "")
        row = f"{label:<24}"
        for n in names:
            s = steps_to(runs[n].get('stall/exploration_edge'), m)
            row += f"{(f'{s/1e6:.1f}M' if s else '--'):>{w}}"
        print(row)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    base = names[0]
    for k, noise in SEED_NOISE.items():
        if k not in runs[base]:
            continue
        b = tail_mean(runs[base][k])
        print(f"\n{k}  (seed noise +/- {noise})")
        print(f"    {base:<18} {fmt(b)}   (reference)")
        for n in names[1:]:
            if k not in runs[n]:
                continue
            v = tail_mean(runs[n][k])
            d = v - b
            verdict = "REAL" if abs(d) > noise else "within seed noise"
            print(f"    {n:<18} {fmt(v)}   delta {d:+.4f}   {verdict}")

    print(f"\nThe wall at {WALL}:")
    for n in names:
        s = runs[n].get('stall/exploration_edge')
        if s is None:
            continue
        peak = float(s[1].max())
        broke = peak > WALL + 0.005
        hit = steps_to(s, WALL)
        note = (f"BROKE IT (peak {peak:.4f})" if broke else
                f"reached but did not pass (peak {peak:.4f})" if hit else
                f"did not reach it (peak {peak:.4f})")
        print(f"    {n:<18} {note}")

    print("\nCaveat: arms are ADDITIVE, so each delta is the effect of that change")
    print("GIVEN the ones before it, not in isolation. n=1 per arm.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        import glob
        args = sorted(glob.glob("runs/ablate_*"))
    sys.exit(main(args))
