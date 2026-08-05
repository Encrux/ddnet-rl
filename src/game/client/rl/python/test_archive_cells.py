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
                               rarity_alpha=0.5)

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
    # Rarity weighting with alpha 0.5: w ∝ visits^-0.5, so 1 vs 1000 gives the
    # rare cell 1/(1+1/sqrt(1000)) ≈ 0.969 of the mass.
    check("rarity weighting strongly prefers the rare hooked cell",
          rare > 0.9, f"rare cell chosen {rare:.1%} of the time")
    check("progress-only sampling would have picked it ~50% at best", 0.9 < rare <= 1.0)

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
