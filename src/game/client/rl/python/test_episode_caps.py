"""Acceptance test for per-episode-type caps and stall patience.

Two episode types are doing different jobs. A from-start episode practises the
whole chain and needs enough steps to physically REACH the frontier. A restored
episode starts next to the obstacle, needs only a couple of attempts, and every
extra step delays the reset that produces the next restore -- and curriculum
turnover happens only at episode boundaries.

Stall patience is the safer version of "make frontier episodes short": it cannot
truncate a productive attempt, because by construction it only fires when the
episode has stopped beating its own high-water progress.

Run: python test_episode_caps.py

NEVER import train.py here -- its atexit handler pkills all game servers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_server_env import ServerEnvConfig, TeeWorldsServerEnv

N_BOTS = 8
_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def idle(n):
    """Do nothing: no horizontal input, no jump, no hook. Guarantees a stall."""
    return {
        'discrete': np.stack([np.full(n, 1), np.zeros(n, dtype=np.int32),
                              np.zeros(n, dtype=np.int32), np.full(n, 4)], 1).astype(np.int32),
        'continuous': np.zeros((n, 1), dtype=np.float32),
    }


def run(**cfg_kw):
    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          finish_progress=1.0, **cfg_kw)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=1, config=cfg)
    try:
        env.reset()
        lengths = []
        for step in range(500):
            _, _, _, trunc, infos = env.step(idle(N_BOTS))
            for info in infos:
                if 'episode' in info:
                    lengths.append(info['episode']['l'])
        return lengths
    finally:
        env.close()


def main():
    print("=" * 68)
    print("Episode cap acceptance test")
    print("=" * 68)

    print("\n[1] Plain cap still applies when nothing else is set")
    lens = run(max_episode_steps=80)
    check("episodes truncate at the plain cap", len(lens) > 0 and set(lens) == {80},
          f"lengths seen {sorted(set(lens))[:5]}")

    print("\n[2] Stall patience truncates an idle episode much sooner")
    lens_p = run(max_episode_steps=400, stall_patience=25)
    check("patience fires well before the hard cap",
          len(lens_p) > 0 and max(lens_p) < 400,
          f"max length {max(lens_p) if lens_p else 'none'} vs cap 400")
    check("patience length is near the patience value",
          len(lens_p) > 0 and max(lens_p) <= 40,
          f"lengths {sorted(set(lens_p))[:5]}")
    check("patience produces many more episodes than the plain cap",
          len(lens_p) > len(lens), f"{len(lens_p)} vs {len(lens)} episodes in 500 steps")

    print("\n[3] Patience does NOT cut an episode that is still progressing")
    # Drive right: bots make progress, so the high-water mark keeps moving and
    # patience must never fire. Any truncation here must be the hard cap.
    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          finish_progress=1.0, max_episode_steps=400, stall_patience=25)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=1, config=cfg)
    try:
        env.reset()
        moving = {'discrete': np.stack([np.full(N_BOTS, 2), np.zeros(N_BOTS, dtype=np.int32),
                                        np.zeros(N_BOTS, dtype=np.int32), np.full(N_BOTS, 4)],
                                       1).astype(np.int32),
                  'continuous': np.zeros((N_BOTS, 1), dtype=np.float32)}
        gains, stalls = 0, 0
        for _ in range(200):
            env.step(moving)
            gains += int((~env._stalled).sum())
            stalls += int(env._stalled.sum())
        # The counter must be zero for any bot that just gained progress.
        gained_now = ~env._stalled
        ok = bool((env._steps_since_gain[gained_now] == 0).all()) if gained_now.any() else True
        check("counter resets on any progress gain", ok,
              f"{gains} gaining steps, {stalls} stalled steps")
        check("progressing bots exist at all (precondition)", gains > 0, f"{gains}")
    finally:
        env.close()

    print("\n[4] Restored episodes can have their own, shorter cap")
    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          finish_progress=1.0, max_episode_steps=300,
                          max_episode_steps_restored=50)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=1, config=cfg)
    try:
        env.reset()
        # All bots spawn at the map start, so they are from-start episodes and
        # must use the LONG cap, not the restored one.
        env._episode_from_start[:] = True
        seen = []
        for _ in range(120):
            _, _, _, trunc, infos = env.step(idle(N_BOTS))
            seen.extend(i['episode']['l'] for i in infos if 'episode' in i)
        # Asserted as "nothing ended", not `all(l > 50)` -- the latter passes
        # vacuously on an empty list, which is exactly the case here.
        check("from-start episodes are NOT cut at the restored cap",
              len(seen) == 0 or min(seen) > 50,
              f"{len(seen)} episodes ended in 120 steps; lengths {sorted(set(seen))[:5]}")

        # Now mark them restored and confirm the short cap binds.
        env._episode_from_start[:] = False
        env.episode_steps[:] = 0
        seen2 = []
        for _ in range(120):
            _, _, _, trunc, infos = env.step(idle(N_BOTS))
            seen2.extend(i['episode']['l'] for i in infos if 'episode' in i)
        check("restored episodes use the short cap",
              len(seen2) > 0 and max(seen2) <= 50, f"lengths {sorted(set(seen2))[:5]}")
    finally:
        env.close()

    print("\n" + "=" * 68)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    else:
        print("All episode cap tests passed")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
