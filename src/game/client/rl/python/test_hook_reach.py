"""Acceptance test for per-sector hook reachability.

This is the information DDNet shows human players as the red/green aim line. The
client computes it in CPlayers::RenderHookCollLine, which is rendering code and
unreachable from the server, so the forward march was ported into the bot
manager (FillHookReach). It shares IntersectLineTeleHook with the real hook.

The test that matters is BEHAVIOURAL, not arithmetic: aim where the observation
says a hook would grab, fire, and check that it actually grabs -- and likewise
that sectors reporting 0 do not. A port that merely looks right but disagrees
with the physics would pass a unit test and fail this one.

Run: python test_hook_reach.py

NEVER import train.py here -- its atexit handler pkills all game servers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_server_env import (AIM_SECTORS, NUM_AIM_SECTORS, NUM_LOOKAHEAD,
                                  ServerEnvConfig, TeeWorldsServerEnv)

N_BOTS = 32
HR0 = 2 + 2 * NUM_LOOKAHEAD          # first hook-reach index inside the nav block
GRABBED = 9                          # player_state slot for HOOK_GRABBED
_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def act(n, sectors, hook):
    return {
        'discrete': np.stack([np.full(n, 1), np.zeros(n, dtype=np.int32),
                              np.full(n, hook, dtype=np.int32),
                              sectors.astype(np.int32)], 1).astype(np.int32),
        'continuous': np.zeros((n, 1), dtype=np.float32),
    }


def main():
    print("=" * 70)
    print("Hook reachability acceptance test")
    print("=" * 70)

    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          max_episode_steps=100000, finish_progress=1.0)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=1, config=cfg)
    try:
        obs, _ = env.reset()
        rng = np.random.default_rng(0)

        print("\n[1] Sanity")
        check("aim sector count matches the action head",
              NUM_AIM_SECTORS == AIM_SECTORS, f"{NUM_AIM_SECTORS} vs {AIM_SECTORS}")

        # Let the bots scatter so they see varied geometry.
        for t in range(120):
            obs, _, _, _, _ = env.step(act(N_BOTS, rng.integers(0, 16, N_BOTS),
                                           int(rng.random() < 0.3)))

        hr = obs['checkpoint'][:, HR0:HR0 + NUM_AIM_SECTORS]
        check("values are in [0, 1]", bool((hr >= 0).all() and (hr <= 1).all()),
              f"range [{hr.min():.3f}, {hr.max():.3f}]")
        check("the feature is not dead (some sector reachable somewhere)",
              bool((hr > 0).any()), f"{(hr > 0).mean():.1%} of sectors reachable")
        check("not everything is reachable (it discriminates)",
              bool((hr == 0).any()), f"{(hr == 0).mean():.1%} of sectors unreachable")

        print(f"\n      mean reachable sectors per bot: {(hr > 0).sum(axis=1).mean():.1f} / 16")

        print("\n[2] Behavioural: does the prediction match the physics?")
        yes_grab = yes_tot = no_grab = no_tot = 0
        for trial in range(40):
            hr = obs['checkpoint'][:, HR0:HR0 + NUM_AIM_SECTORS]
            # Half the bots aim at their best sector, half at a zero sector.
            sectors = np.zeros(N_BOTS, dtype=np.int32)
            predict = np.zeros(N_BOTS, dtype=bool)
            for i in range(N_BOTS):
                nz = np.nonzero(hr[i] > 0)[0]
                z = np.nonzero(hr[i] == 0)[0]
                if i % 2 == 0 and len(nz):
                    sectors[i] = nz[np.argmax(hr[i][nz])]
                    predict[i] = True
                elif len(z):
                    sectors[i] = z[rng.integers(len(z))]
                    predict[i] = False
                else:
                    sectors[i] = rng.integers(16)
                    predict[i] = hr[i][sectors[i]] > 0

            # Hold the hook long enough for it to fly its full length
            # (HookLength/FireSpeed ~ 5 ticks; frame_skip 3 -> 2 decisions).
            for _ in range(2):
                obs, _, _, _, _ = env.step(act(N_BOTS, sectors, 1))
            grabbed = obs['player_state'][:, -15:][:, GRABBED] > 0.5

            yes_grab += int(grabbed[predict].sum()); yes_tot += int(predict.sum())
            no_grab += int(grabbed[~predict].sum()); no_tot += int((~predict).sum())

            # Release and drift so the next trial sees fresh geometry.
            for _ in range(3):
                obs, _, _, _, _ = env.step(act(N_BOTS, rng.integers(0, 16, N_BOTS), 0))

        yr = yes_grab / max(yes_tot, 1)
        nr = no_grab / max(no_tot, 1)
        print(f"      aimed where reach > 0 : grabbed {yr:6.1%}  ({yes_grab}/{yes_tot})")
        print(f"      aimed where reach = 0 : grabbed {nr:6.1%}  ({no_grab}/{no_tot})")
        check("predicted-reachable aims usually grab", yr > 0.6, f"{yr:.1%}")
        check("predicted-unreachable aims rarely grab", nr < 0.15, f"{nr:.1%}")
        check("the signal separates the two cases", yr - nr > 0.5, f"gap {yr-nr:.1%}")
    finally:
        env.close()

    print("\n" + "=" * 70)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    else:
        print("All hook reachability tests passed")
    print("=" * 70)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
