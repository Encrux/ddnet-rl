"""Plot exploration-edge trajectories for the ablation arms.

Usage:  python plot_ablation.py [out.png] [runs/ablate_* ...]

The edge curve is the metric that matters: the previous policy pinned at 0.4550
with slope exactly +0.00000 for 64M steps, so a flat tail at that line means the
arm hit the same structural wall and a rising tail means it did not.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WALL = 0.455


def series(run_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        return None
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]) / 1e6, np.array([e.value for e in ev])


def main(out, dirs):
    panels = [
        ('stall/exploration_edge', 'exploration edge', True),
        ('progress/p50_from_start', 'p50 progress from start', False),
        ('stall/thrash_fraction', 'thrash fraction', False),
        ('losses/entropy', 'policy entropy', False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(dirs)))

    for ax, (tag, title, mark_wall) in zip(axes.ravel(), panels):
        for d, c in zip(dirs, colors):
            s = series(d, tag)
            if s is None:
                continue
            name = os.path.basename(d.rstrip('/')).replace('ablate_', '')
            ax.plot(s[0], s[1], label=name, color=c, lw=1.6)
        if mark_wall:
            ax.axhline(WALL, ls='--', c='crimson', lw=1.2)
            ax.text(0.01, WALL, ' wall 0.455', color='crimson', va='bottom',
                    fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_title(title)
        ax.set_xlabel('Msteps')
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    fig.suptitle('Ablation: learned action hold, ratchet shaping, archive cells')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[0] if args and args[0].endswith('.png') else 'ablation.png'
    dirs = [a for a in args if not a.endswith('.png')] or sorted(glob.glob('runs/ablate_*'))
    main(out, dirs)
