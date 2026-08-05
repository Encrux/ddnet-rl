"""Acceptance test for ratchet shaping and the high-water observation field.

The claim under test: with the ratchet on, giving ground back costs only TIME,
whereas plain potential shaping charges SHAPING_K per unit of progress lost. That
penalty is what makes the first step of any obstacle needing a run-up immediately
and certainly negative.

The test deliberately asserts NOTHING about Tutorial's geometry. It drives both
arms with a byte-identical recorded action sequence, then does the analysis
per-step on whatever trajectory actually resulted, selecting the steps where
progress happened to decrease. Physics is deterministic, so the two arms visit
exactly the same states and every reward difference is attributable to shaping.

    plain:    r_t = -time + K * (phi_t - phi_{t-1})
    ratchet:  r_t = -time + K * max(0, phi_t - highwater_{t-1})

so on any step where phi decreased, plain is strictly below -time and ratchet is
exactly -time.

Run: python test_ratchet.py

NEVER import train.py here -- its atexit handler pkills all game servers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_server_env import NUM_LOOKAHEAD, ServerEnvConfig, TeeWorldsServerEnv

N_BOTS = 8
SHAPING_K = 100.0
TIME_PER_DECISION = 0.001 * 3  # TIME_PENALTY per tick x frame_skip
HW = 1 + 2 * NUM_LOOKAHEAD      # index of the high-water field in the nav block
STEPS = 700

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def make_schedule(steps, n):
    """A varied but FIXED action sequence, so both arms replay it exactly.

    Wandering left and right guarantees the trajectory contains both progress
    gains and progress losses, which is all the analysis needs.
    """
    rng = np.random.default_rng(4242)
    out = []
    direction = 2
    for t in range(steps):
        if t % 40 == 0:                      # reverse periodically
            direction = 0 if direction == 2 else 2
        out.append({
            'discrete': np.stack([
                np.full(n, direction),
                (rng.random(n) < 0.15).astype(np.int32),
                (rng.random(n) < 0.10).astype(np.int32),
                rng.integers(0, 16, n),
            ], 1).astype(np.int32),
            'continuous': rng.uniform(-1, 1, (n, 1)).astype(np.float32),
        })
    return out


def run_arm(ratchet, schedule):
    cfg = ServerEnvConfig(frame_skip=3, use_dict_obs=True, map_name='Tutorial',
                          max_episode_steps=100000, finish_progress=1.0,
                          ratchet_shaping=ratchet)
    env = TeeWorldsServerEnv(num_bots=N_BOTS, num_servers=1, config=cfg)
    try:
        env.reset()
        rewards, progress, high_water = [], [], []
        for action in schedule:
            obs, rew, _, _, _ = env.step(action)
            rewards.append(rew.copy())
            progress.append(obs['checkpoint'][:, 0].copy())
            high_water.append(obs['checkpoint'][:, HW].copy())
        return (np.array(rewards), np.array(progress), np.array(high_water))
    finally:
        env.close()


def main():
    print("=" * 70)
    print("Ratchet shaping acceptance test")
    print("=" * 70)

    schedule = make_schedule(STEPS, N_BOTS)
    r_off, p_off, h_off = run_arm(False, schedule)
    r_on, p_on, h_on = run_arm(True, schedule)

    print("\n[0] Both arms are the same experiment")
    check("identical trajectories (deterministic physics, identical actions)",
          np.allclose(p_off, p_on, atol=1e-6),
          f"max |diff| {np.abs(p_off - p_on).max():.2e}")
    check("identical high-water observations",
          np.allclose(h_off, h_on, atol=1e-6),
          f"max |diff| {np.abs(h_off - h_on).max():.2e}")

    # Per-step progress deltas. Step 0 has no predecessor inside the window.
    dphi = np.diff(p_off, axis=0)
    lost = dphi < -1e-7
    gained = dphi > 1e-7
    print(f"\n[1] Trajectory — {lost.sum()} losing steps, {gained.sum()} gaining steps, "
          f"progress range [{p_off.min():.4f}, {p_off.max():.4f}]")
    check("trajectory contains steps that give ground back (precondition)",
          lost.sum() > 50, f"{lost.sum()} losing steps")

    ro, rn = r_off[1:], r_on[1:]

    print("\n[2] Steps where progress DECREASED")
    plain_on_loss = ro[lost].mean()
    ratchet_on_loss = rn[lost].mean()
    print(f"      plain   {plain_on_loss:+.5f}   ratchet {ratchet_on_loss:+.5f}   "
          f"time-only {-TIME_PER_DECISION:+.5f}")
    check("plain shaping charges well beyond the time cost",
          plain_on_loss < -2 * TIME_PER_DECISION, f"{plain_on_loss:.5f}")
    check("ratchet charges exactly the time cost and nothing more",
          abs(ratchet_on_loss + TIME_PER_DECISION) < 1e-4,
          f"{ratchet_on_loss:.5f} vs {-TIME_PER_DECISION:.5f}")
    check("per-step plain reward matches -time + K*dphi",
          np.allclose(ro[lost], -TIME_PER_DECISION + SHAPING_K * dphi[lost], atol=2e-3),
          f"max err {np.abs(ro[lost] - (-TIME_PER_DECISION + SHAPING_K * dphi[lost])).max():.2e}")

    print("\n[3] Steps where progress INCREASED")
    check("ratchet still pays for genuinely new ground",
          rn[gained].mean() > 0.0, f"{rn[gained].mean():+.5f}")
    # Restricted to GAINING steps on purpose. Across losing steps the ratchet
    # pays strictly more than plain -- that withholding of penalties is the whole
    # mechanism, verified in [2]. What must hold here is that the ratchet never
    # hands out a *bonus* plain shaping would not: on a gaining step it pays
    # K*(phi - highwater), and highwater >= phi_prev, so it can only be smaller.
    check("on gaining steps ratchet never pays more than plain",
          (rn[gained] <= ro[gained] + 1e-4).all(),
          f"max excess {(rn[gained] - ro[gained]).max():.2e}")
    check("ratchet withholds re-earned ground (pays less overall on gains)",
          rn[gained].sum() < ro[gained].sum(),
          f"ratchet {rn[gained].sum():.2f} vs plain {ro[gained].sum():.2f}")

    print("\n[4] The whole episode")
    print(f"      total plain {r_off.sum(axis=0).mean():+.3f}   "
          f"total ratchet {r_on.sum(axis=0).mean():+.3f}")
    # Telescoping: total shaping is K*(final high-water - initial), so the ratchet
    # total depends only on the endpoint -- exactly the property that keeps it
    # potential-based and non-farmable.
    predicted = SHAPING_K * (h_off[-1] - p_off[0]) - STEPS * TIME_PER_DECISION
    check("ratchet total telescopes to K*(high-water gained) - time",
          np.allclose(r_on.sum(axis=0), predicted, atol=0.05),
          f"actual {r_on.sum(axis=0).mean():.3f} vs predicted {predicted.mean():.3f}")

    print("\n[5] High-water field")
    check("high-water never below current progress", (h_on >= p_on - 1e-6).all())
    check("high-water is non-decreasing within an episode",
          (np.diff(h_on, axis=0) >= -1e-6).all(),
          f"min step {np.diff(h_on, axis=0).min():.2e}")
    check("high-water equals the running max of progress",
          np.allclose(h_on, np.maximum.accumulate(p_on, axis=0), atol=1e-4),
          f"max |diff| {np.abs(h_on - np.maximum.accumulate(p_on, axis=0)).max():.2e}")

    print("\n" + "=" * 70)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    else:
        print("All ratchet tests passed")
    print("=" * 70)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
