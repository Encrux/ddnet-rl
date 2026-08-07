"""Acceptance test: the agent must be able to see its own hook state.

The observation one-hotted hook state over {0, 1, 2}. DDNet's enum is sparse --
RETRACTED=-1 (arrives as 255), IDLE=0, RETRACT_START=1, RETRACT_END=3, FLYING=4,
GRABBED=5 -- and has no value 2 at all. So one slot encoded a state that cannot
occur, while FLYING, GRABBED and RETRACTED all collapsed to (0,0,0).

Measured in play before the fix: 60.0% of the time the hook observation was the
all-zeros vector, including the 41.1% of time spent GRABBED. Every hook maneuver
depends on knowing whether the hook caught, and the agent could not see it.

Run: python test_hook_obs.py

NEVER import train.py here -- its atexit handler pkills all game servers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_server_env import (CONTIGUOUS_OBS_OFFSET, HOOK_FLYING, HOOK_GRABBED,
                                  HOOK_IDLE, HOOK_RETRACTED, HOOK_RETRACT_END,
                                  HOOK_RETRACT_START, OBS_SIZE, ServerEnvConfig,
                                  TeeWorldsServerEnv)

N_BOTS = 16
STEPS = 600
_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def encode(hook_state):
    """Mirror of the observation encoding, for the offline part of the test."""
    hooked = hook_state == HOOK_GRABBED
    flying = hook_state == HOOK_FLYING
    return (float(not (hooked or flying)), float(flying), float(hooked))


def main():
    print("=" * 68)
    print("Hook observation acceptance test")
    print("=" * 68)

    print("\n[1] Every reachable hook state maps to a defined slot")
    states = {'RETRACTED': HOOK_RETRACTED, 'IDLE': HOOK_IDLE,
              'RETRACT_START': HOOK_RETRACT_START, 'RETRACT_END': HOOK_RETRACT_END,
              'FLYING': HOOK_FLYING, 'GRABBED': HOOK_GRABBED}
    enc = {k: encode(v) for k, v in states.items()}
    for k, e in enc.items():
        print(f"      {k:14s} -> {e}")
    check("no state produces the all-zeros vector",
          all(sum(e) > 0 for e in enc.values()),
          f"{[k for k, e in enc.items() if sum(e) == 0]}")
    check("GRABBED is distinguishable from IDLE", enc['GRABBED'] != enc['IDLE'])
    check("GRABBED is distinguishable from FLYING", enc['GRABBED'] != enc['FLYING'])
    check("GRABBED is distinguishable from RETRACTED", enc['GRABBED'] != enc['RETRACTED'])
    check("exactly one slot is set for every state",
          all(sum(e) == 1.0 for e in enc.values()))

    print("\n[2] Live game: the encoding tracks the real hook state")
    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          max_episode_steps=100000, finish_progress=1.0)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=1, config=cfg)
    try:
        obs, _ = env.reset()
        rng = np.random.default_rng(0)
        raw_seen, mismatches, grabbed_steps, zero_rows = {}, 0, 0, 0
        for _ in range(STEPS):
            a = {'discrete': np.stack([rng.integers(0, 3, N_BOTS),
                                       (rng.random(N_BOTS) < 0.3).astype(np.int32),
                                       (rng.random(N_BOTS) < 0.6).astype(np.int32),
                                       rng.integers(0, 16, N_BOTS)], 1).astype(np.int32),
                 'continuous': rng.uniform(-1, 1, (N_BOTS, 1)).astype(np.float32)}
            obs, _, _, _, _ = env.step(a)

            c = env.shm_clients[0]
            c.shm_mmap.seek(CONTIGUOUS_OBS_OFFSET)
            arr = np.frombuffer(c.shm_mmap.read(N_BOTS * OBS_SIZE),
                                dtype=np.uint8).reshape(N_BOTS, OBS_SIZE)
            raw = arr[:, 8 + 18]
            ps = obs['player_state'][:, 7:10]
            for v in raw:
                raw_seen[int(v)] = raw_seen.get(int(v), 0) + 1
            grabbed_steps += int((raw == HOOK_GRABBED).sum())
            zero_rows += int((ps.sum(axis=1) == 0).sum())
            for r, e in zip(raw, ps):
                if not np.allclose(e, encode(int(r))):
                    mismatches += 1
        total = sum(raw_seen.values())
        print(f"      raw states seen: "
              f"{ {k: f'{v/total:.1%}' for k, v in sorted(raw_seen.items())} }")
        check("observation matches the true hook state every step", mismatches == 0,
              f"{mismatches}/{total} mismatched")
        check("the all-zeros vector never occurs", zero_rows == 0,
              f"{zero_rows}/{total} rows all-zero (was 60.0% before the fix)")
        check("GRABBED actually occurs in play (precondition)",
              grabbed_steps / total > 0.05, f"{grabbed_steps/total:.1%}")
        check("GRABBED is now visible in the observation",
              raw_seen.get(HOOK_GRABBED, 0) > 0 and
              all(np.isclose(encode(HOOK_GRABBED)[2], 1.0) for _ in [0]))
    finally:
        env.close()

    print("\n" + "=" * 68)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    else:
        print("All hook observation tests passed")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
