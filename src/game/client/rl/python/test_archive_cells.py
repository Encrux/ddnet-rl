"""Acceptance test for the (bin x hook x velocity) snapshot archive.

The claim under test: a mid-maneuver state can no longer be overwritten by a bot
walking through the same progress bin on the ground, and the sampler only ever
asks for cells that actually exist on the server it is asking.

Run: python test_archive_cells.py

NEVER import train.py here -- its atexit handler pkills all game servers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_server_env import (NUM_SNAPSHOT_SLOTS, NUM_SNAPSHOT_VARIANTS,
                                  NUM_PROGRESS_BINS, ProgressSpawnManager,
                                  ServerEnvConfig, TeeWorldsServerEnv)

N_BOTS = 16
N_SERVERS = 2
STEPS = 400

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def test_sampler_offline():
    """Sampler behaviour, with synthetic occupancy so nothing depends on a map."""
    print("\n[1] Sampler (synthetic occupancy)")
    bots_per_server = 4
    mgr = ProgressSpawnManager(begin_bias=0.0, window=4, num_bins=NUM_PROGRESS_BINS,
                               num_servers=2, bots_per_server=bots_per_server,
                               restore_beta=-0.5, max_variant_share=1.0)

    visits = np.zeros((2, NUM_SNAPSHOT_SLOTS), dtype=np.int64)
    # Server 0: bin 50 has a common walking cell and one rare hooked cell.
    visits[0, 50 * NUM_SNAPSHOT_VARIANTS + 1] = 1000   # level, no hook
    visits[0, 50 * NUM_SNAPSHOT_VARIANTS + 3] = 1      # hooked + rising
    # Server 1: nothing anywhere in the window.
    mgr.update_visits(visits)
    mgr.update_frontier(frontier_bin=50, reached_bins=np.array([50]), edge=50)

    envs_s0 = np.arange(0, bots_per_server)
    got = mgr.sample(np.tile(envs_s0, 500))
    check("only ever samples occupied cells",
          set(np.unique(got)).issubset({50 * NUM_SNAPSHOT_VARIANTS + 1,
                                        50 * NUM_SNAPSHOT_VARIANTS + 3}),
          f"unique slots {np.unique(got)[:6]}")

    stats = mgr.get_stats()
    check("restore_bin_mean reports the BIN not the raw slot",
          abs(stats.get('curriculum/restore_bin_mean', -1) - 50.0) < 1e-6,
          f"{stats.get('curriculum/restore_bin_mean')}")
    check("archive stats expose cells-per-bin",
          abs(stats.get('archive/cells_per_bin', 0) - 2.0) < 1e-6,
          f"{stats.get('archive/cells_per_bin')}")

    rare = (got == 50 * NUM_SNAPSHOT_VARIANTS + 3).mean()
    # beta=-0.5 means w ∝ visits^-0.5, so 1 vs 1000 visits gives the rare cell
    # ~97% of the mass. This arm exists to document how extreme that is; the
    # shipped default is beta=+0.5 (see test_restore_weighting).
    check("beta=-0.5 strongly prefers the rare cell",
          rare > 0.9, f"rare cell chosen {rare:.1%} of the time")

    envs_s1 = np.arange(bots_per_server, 2 * bots_per_server)
    got1 = mgr.sample(np.tile(envs_s1, 50))
    check("server with an empty window spawns at start instead of a dead slot",
          (got1 == -1).all(), f"unique {np.unique(got1)}")

    mgr2 = ProgressSpawnManager(begin_bias=1.0, window=4, num_servers=2,
                                bots_per_server=bots_per_server)
    mgr2.update_visits(visits)
    mgr2.update_frontier(50, np.array([50]), 50)
    check("begin_bias 1.0 always spawns at the start",
          (mgr2.sample(np.tile(envs_s0, 50)) == -1).all())


def test_restore_weighting():
    """The measured failure: beta=-0.5 sent 82% of restores into hooked cells.

    Realistic occupancy -- grounded cells visited often, hooked cells rarely,
    because hooked states are transient.
    """
    print("\n[1b] Restore weighting across variants")
    rng = np.random.default_rng(3)
    visits = np.zeros((1, NUM_SNAPSHOT_SLOTS), dtype=np.int64)
    for b in range(40, 60):
        for v in range(NUM_SNAPSHOT_VARIANTS):
            base = rng.integers(200, 2000) if v < 3 else rng.integers(1, 20)
            visits[0, b * NUM_SNAPSHOT_VARIANTS + v] = base

    def hooked_share(beta, cap=1.0):
        mgr = ProgressSpawnManager(begin_bias=0.0, window=12, num_servers=1,
                                   bots_per_server=8, restore_beta=beta,
                                   max_variant_share=cap)
        mgr.update_visits(visits)
        mgr.update_frontier(59, np.arange(40, 60), 59)
        got = mgr.sample(np.tile(np.arange(8), 400))
        got = got[got >= 0]
        return (got % NUM_SNAPSHOT_VARIANTS >= 3).mean()

    broken = hooked_share(-0.5)
    fixed = hooked_share(0.5, cap=0.4)
    natural = hooked_share(1.0)
    print(f"      hooked-cell restore share:  beta=-0.5 {broken:.1%} (the bug)  "
          f"beta=0.5 {fixed:.1%}  beta=1.0 {natural:.1%}")
    check("beta=-0.5 reproduces the observed failure", broken > 0.7, f"{broken:.1%}")
    check("beta=0.5 keeps restores mostly on the ground", fixed < 0.45, f"{fixed:.1%}")
    check("beta=0.5 still visits hooked cells sometimes", fixed > 0.02, f"{fixed:.1%}")
    check("beta=1.0 tracks natural occupancy (hooked cells are rare)",
          natural < 0.10, f"{natural:.1%}")

    # The cap must bind regardless of beta. With 6 variants present a cap must
    # be >= 1/6 to be satisfiable at all; 0.25 is feasible and should pull each
    # hooked variant down from ~30% to 25%, i.e. ~75% total instead of 91.6%.
    capped = hooked_share(-0.5, cap=0.25)
    check("per-variant cap bounds even a bad beta", capped <= 0.75 + 0.02,
          f"{capped:.1%} with cap 0.25 over 3 hooked variants")
    check("cap actually binds (below the uncapped 91.6%)", capped < broken - 0.1,
          f"{capped:.1%} vs uncapped {broken:.1%}")

    # An unsatisfiable cap must degrade to uniform rather than be ignored.
    infeasible = hooked_share(-0.5, cap=0.10)
    check("unsatisfiable cap degrades to uniform, not to the raw weighting",
          abs(infeasible - 0.5) < 0.03, f"{infeasible:.1%}, uniform would be 50.0%")


def test_archive_live():
    """Occupancy from a real server: do the extra cells actually get filled?"""
    print("\n[2] Live archive occupancy")
    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          max_episode_steps=100000, finish_progress=1.0,
                          curriculum=True, begin_bias=0.2)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=N_SERVERS, config=cfg)
    try:
        env.reset()
        n = env.num_envs
        rng = np.random.default_rng(7)
        for t in range(STEPS):
            env.step({
                'discrete': np.stack([
                    rng.integers(0, 3, n),
                    (rng.random(n) < 0.2).astype(np.int32),
                    (rng.random(n) < 0.35).astype(np.int32),  # hook often
                    rng.integers(0, 16, n),
                ], 1).astype(np.int32),
                'continuous': rng.uniform(-1, 1, (n, 1)).astype(np.float32),
            })

        visits = env.refresh_snapshot_visits()
        occupied = visits > 0
        per_server = occupied.sum(axis=1)
        by_variant = occupied.reshape(N_SERVERS, NUM_PROGRESS_BINS,
                                      NUM_SNAPSHOT_VARIANTS).sum(axis=(0, 1))

        print(f"      occupied cells per server: {per_server.tolist()}")
        print(f"      by variant {['rise', 'level', 'fall', 'hk+rise', 'hk+lvl', 'hk+fall']}: "
              f"{by_variant.tolist()}")

        check("archive gets populated", per_server.min() > 0, f"{per_server.tolist()}")
        check("visit counts are published to shm", visits.max() > 1, f"max {visits.max()}")

        bins_occupied = occupied.reshape(N_SERVERS, NUM_PROGRESS_BINS,
                                         NUM_SNAPSHOT_VARIANTS).any(axis=2).sum(axis=1)
        cells_per_bin = per_server / np.maximum(bins_occupied, 1)
        check("more than one cell per bin (the variant axis is doing work)",
              cells_per_bin.mean() > 1.2, f"{cells_per_bin.mean():.2f} cells/bin")

        hooked_cells = by_variant[3:].sum()
        check("hooked states get their own cells", hooked_cells > 0,
              f"{hooked_cells} hooked cells")
        check("servers keep independent archives",
              not np.array_equal(visits[0], visits[1]))

        # The bug this exists to prevent: a walking state overwriting a swing
        # state in the same bin. If both exist as separate cells, it cannot.
        grid = occupied.reshape(N_SERVERS, NUM_PROGRESS_BINS, NUM_SNAPSHOT_VARIANTS)
        both = (grid[..., :3].any(axis=2) & grid[..., 3:].any(axis=2)).sum()
        check("bins hold hooked AND unhooked cells simultaneously", both > 0,
              f"{both} bins with both")
    finally:
        env.close()


def main():
    print("=" * 70)
    print("Archive cell acceptance test")
    print("=" * 70)
    test_sampler_offline()
    test_restore_weighting()
    test_archive_live()
    print("\n" + "=" * 70)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    else:
        print("All archive cell tests passed")
    print("=" * 70)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
