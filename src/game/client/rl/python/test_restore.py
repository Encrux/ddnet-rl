"""Phase 2.1 acceptance test: snapshot/restore must be bit-exact.

If restoring a CSaveTee does not reproduce physics exactly, every curriculum
built on top of it is training on states the agent can never actually reach,
and nothing downstream is trustworthy. Run: python test_restore.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_server_env import NAV_DIM, ServerEnvConfig, TeeWorldsServerEnv

N_BOTS = 8
REPLAY_STEPS = 100
_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def spread(step, n):
    """Forward-biased but per-bot DIFFERENT, so bots fan out across progress bins."""
    r = np.random.default_rng(1000 + step)
    return {
        'discrete': np.stack([
            np.full(n, 2),                                  # hold right
            (r.random(n) < 0.25).astype(np.int32),           # occasional jump
            (r.random(n) < 0.15).astype(np.int32),           # occasional hook
            r.integers(0, 16, n),                            # varied aim
        ], 1).astype(np.int32),
        'continuous': r.uniform(-1, 1, (n, 1)).astype(np.float32),
    }


def lockstep(step, n):
    """IDENTICAL for every bot. This is what makes the test meaningful: after
    restoring all bots to one slot, identical inputs on deterministic physics
    must keep them identical. Any field CSaveTee failed to restore (velocity,
    hook tick, freeze timer, jumps) shows up as divergence."""
    r = np.random.default_rng(9000 + step)
    one = np.array([r.integers(0, 3), r.integers(0, 2), r.integers(0, 2), r.integers(0, 16)])
    return {
        'discrete': np.tile(one, (n, 1)).astype(np.int32),
        'continuous': np.full((n, 1), r.uniform(-1, 1), dtype=np.float32),
    }


def main():
    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          max_episode_steps=100000, finish_progress=1.0)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=1, config=cfg)
    try:
        env.reset()

        # Drive with PER-BOT DIFFERENT actions so bots end up in genuinely
        # different states — different positions, velocities, hook states.
        for k in range(60):
            env.step(spread(k, N_BOTS))
        pre = env._read_positions().copy()
        pre_spread = np.abs(pre - pre[0]).max()
        # This precondition is what gives the test teeth. If bots were already
        # identical going in, "identical after restore" would prove nothing.
        check("bots are in DIFFERENT states before restore", pre_spread > 1.0,
              f"max separation {pre_spread:.1f} units")

        progress = env.path_progress.update(pre)
        slot = int(progress.max() * 200)

        # Restore every bot into the same archive slot. If CSaveTee::Load misses
        # any field - velocity, hook tick, freeze timer, jumps remaining - the
        # bots keep their differing prior values and diverge under identical
        # inputs. Position alone matching is not sufficient.
        obs_buf = {'tiles': np.zeros((N_BOTS, 1, 31, 43), dtype=np.float32),
                   'player_state': np.zeros((N_BOTS, 15), dtype=np.float32),
                   'checkpoint': np.zeros((N_BOTS, NAV_DIM), dtype=np.float32)}
        env._reset_envs_fast(np.ones(N_BOTS, dtype=bool), obs_buf,
                             spawn_checkpoints=np.full(N_BOTS, slot, dtype=np.int32))
        start = env._read_positions().copy()
        dev = np.abs(start - start[0]).max()
        check("all bots collapse to one state on restore", dev < 1e-3,
              f"max deviation {dev:.6f} units (slot {slot})")

        traj = []
        for k in range(REPLAY_STEPS):
            env.step(lockstep(k, N_BOTS))
            traj.append(env._read_positions().copy())
        traj = np.array(traj)  # (steps, bots, 2)

        drift = np.abs(traj - traj[:, :1, :]).max()
        check(f"identical inputs stay in lockstep for {REPLAY_STEPS} steps", drift < 1e-3,
              f"max divergence {drift:.6f} units")
        moved = np.linalg.norm(traj[-1, 0] - start[0])
        check("replay actually exercised physics", moved > 32.0, f"{moved:.0f} units travelled")
    finally:
        env.close()


if __name__ == "__main__":
    import contextlib, io
    buf = io.StringIO()
    print("snapshot restore determinism")
    with contextlib.redirect_stdout(buf):
        try:
            main()
        finally:
            out = buf.getvalue()
    print("\n".join(l for l in out.splitlines() if l.strip().startswith(("PASS", "FAIL", "("))))
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        sys.exit(1)
    print("restore determinism verified")
