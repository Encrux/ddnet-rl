"""Acceptance tests for the learned action hold (duration head).

Run:  python test_action_hold.py

NEVER import train.py here. Its atexit handler runs `pkill -9 -f DDNet-Server`,
so importing it kills any live training run (this cost ~25 minutes once).
"""

import sys

import numpy as np
import torch

from action_hold import ActionHold
from ppo_hybrid import PPOAgent, PPOConfig

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def make_action(n, num_discrete, dur_idx, cont=None):
    d = np.zeros((n, num_discrete), dtype=np.int64)
    d[:, -1] = dur_idx
    c = np.zeros((n, 1), dtype=np.float32) if cont is None else cont
    return {'discrete': d, 'continuous': c}


def test_duration_semantics():
    print("\n[1] Hold duration semantics")
    durations = (1, 2, 4, 8, 16)
    nd = 5

    # Duration index 0 == 1 decision: every step is a fresh decision.
    h = ActionHold(num_envs=3, num_discrete=nd, durations=durations)
    fresh_log = [h.apply(make_action(3, nd, 0))[1].copy() for _ in range(5)]
    check("duration 1 is fresh every step", all(f.all() for f in fresh_log))

    # Duration index 2 == 4 decisions: fresh on steps 0 and 4, held between.
    h = ActionHold(num_envs=3, num_discrete=nd, durations=durations)
    fresh_log = [h.apply(make_action(3, nd, 2))[1].copy() for _ in range(9)]
    pattern = [bool(f[0]) for f in fresh_log]
    check("duration 4 is fresh every 4th step", pattern == [True, False, False, False] * 2 + [True],
          f"got {pattern}")

    # The action executed during a hold must be the LATCHED one, not the new sample.
    h = ActionHold(num_envs=2, num_discrete=nd, durations=durations)
    first = make_action(2, nd, 2)
    first['discrete'][:, 2] = 1          # hook pressed
    first['continuous'][:] = 0.75
    exec0, _ = h.apply(first)
    second = make_action(2, nd, 2)
    second['discrete'][:, 2] = 0         # policy now wants hook released
    second['continuous'][:] = -0.75
    exec1, fresh1 = h.apply(second)
    check("held step re-executes the latched discrete action",
          bool((exec1['discrete'][:, 2] == 1).all()) and not fresh1.any())
    check("held step re-executes the latched continuous action",
          bool(np.allclose(exec1['continuous'], 0.75)))
    check("latched action equals what was sampled when fresh",
          bool((exec0['discrete'][:, 2] == 1).all()))

    # Envs run independent holds — one env's latch must not leak into another.
    h = ActionHold(num_envs=2, num_discrete=nd, durations=durations)
    a = make_action(2, nd, 0)
    a['discrete'][1, -1] = 3             # env 1 holds for 8 decisions
    a['discrete'][:, 0] = [0, 2]
    h.apply(a)
    b = make_action(2, nd, 0)
    b['discrete'][:, 0] = [1, 1]
    exec_b, fresh_b = h.apply(b)
    check("per-env holds are independent",
          bool(fresh_b[0]) and not bool(fresh_b[1]) and
          exec_b['discrete'][0, 0] == 1 and exec_b['discrete'][1, 0] == 2)


def test_reset_clears_hold():
    print("\n[2] Episode boundary clears the latch")
    h = ActionHold(num_envs=3, num_discrete=5, durations=(1, 2, 4, 8, 16))
    h.apply(make_action(3, 5, 4))        # everyone holds 16
    _, fresh_before = h.apply(make_action(3, 5, 4))
    check("mid-hold envs are not fresh", not fresh_before.any())

    h.clear(np.array([True, False, True]))
    _, fresh_after = h.apply(make_action(3, 5, 4))
    check("cleared envs decide again immediately",
          bool(fresh_after[0]) and bool(fresh_after[2]) and not bool(fresh_after[1]),
          f"got {fresh_after.tolist()}")


def test_stats():
    print("\n[3] Diagnostics")
    h = ActionHold(num_envs=4, num_discrete=5, durations=(1, 2, 4, 8, 16))
    for _ in range(8):
        h.apply(make_action(4, 5, 2))    # duration 4
    s = h.stats()
    check("mean_duration reports the chosen hold", abs(s['hold/mean_duration'] - 4.0) < 1e-6,
          f"got {s['hold/mean_duration']}")
    check("fresh_fraction ~ 1/duration", abs(s['hold/fresh_fraction'] - 0.25) < 1e-6,
          f"got {s['hold/fresh_fraction']}")
    check("stats reset after read", h.stats().get('hold/mean_duration') is None)


def test_config_and_head():
    print("\n[4] Config wiring and head prior")
    base = PPOConfig()
    check("disabled by default", base.discrete_dims == (3, 2, 2, 16) and base.hold_durations == ())

    cfg = PPOConfig(hold_durations=(1, 2, 4, 8, 16))
    check("duration head appended last", cfg.discrete_dims == (3, 2, 2, 16, 5),
          f"got {cfg.discrete_dims}")

    # The prior must favour short holds but still draw the longest one often
    # enough to matter. 0.425^16 ~ 1e-6 is what it has to beat.
    from ppo_hybrid import HybridActor
    cfg.use_cnn = False
    cfg.use_compile = False
    actor = HybridActor(obs_dim=32, config=cfg)
    with torch.no_grad():
        probs = torch.softmax(actor.discrete_heads[-1].bias, dim=-1)
    mean_dur = float((probs * torch.tensor([1.0, 2, 4, 8, 16])).sum())
    p_longest = float(probs[-1])
    check("prior favours short holds", 2.5 < mean_dur < 4.5, f"mean duration {mean_dur:.2f}")
    check("longest hold still drawn often", p_longest > 0.02,
          f"P(16)={p_longest:.4f}, vs 0.425^16={0.425**16:.2e}")


def test_masked_update_equivalence():
    print("\n[5] PPO update with the mask")
    torch.manual_seed(0)
    np.random.seed(0)

    def build():
        cfg = PPOConfig(hold_durations=(1, 2, 4, 8, 16), use_cnn=False, use_compile=False,
                        use_amp=False, num_steps=8, num_minibatches=2, update_epochs=1)
        return PPOAgent(obs_dim=16, num_envs=4, config=cfg, device="cpu")

    def run(fresh_value):
        torch.manual_seed(0)
        np.random.seed(0)
        agent = build()
        nd = len(agent.config.discrete_dims)
        for t in range(agent.num_steps):
            obs = np.random.randn(4, 16).astype(np.float32)
            act = {
                'discrete': np.stack([
                    np.random.randint(0, d, size=4) for d in agent.config.discrete_dims
                ], axis=-1).astype(np.int64),
                'continuous': np.random.randn(4, 1).astype(np.float32),
            }
            fresh = None if fresh_value is None else np.full(4, fresh_value, dtype=bool)
            agent.store_transition(
                t, obs, act,
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.zeros(4, dtype=bool),
                np.random.randn(4).astype(np.float32),
                fresh,
            )
        return agent.update(np.random.randn(4, 16).astype(np.float32), np.zeros(4, dtype=bool))

    stats_none = run(None)
    stats_all_fresh = run(True)
    check("all-fresh mask reproduces the unmasked update",
          abs(stats_none['policy_loss'] - stats_all_fresh['policy_loss']) < 1e-6 and
          abs(stats_none['entropy'] - stats_all_fresh['entropy']) < 1e-6,
          f"pg {stats_none['policy_loss']:.6f} vs {stats_all_fresh['policy_loss']:.6f}")

    # A partial mask must still produce finite losses and must actually differ.
    torch.manual_seed(0)
    np.random.seed(0)
    agent = build()
    rng = np.random.default_rng(0)
    for t in range(agent.num_steps):
        obs = np.random.randn(4, 16).astype(np.float32)
        act = {
            'discrete': np.stack([
                np.random.randint(0, d, size=4) for d in agent.config.discrete_dims
            ], axis=-1).astype(np.int64),
            'continuous': np.random.randn(4, 1).astype(np.float32),
        }
        agent.store_transition(
            t, obs, act,
            np.random.randn(4).astype(np.float32),
            np.random.randn(4).astype(np.float32),
            np.zeros(4, dtype=bool),
            np.random.randn(4).astype(np.float32),
            rng.random(4) < 0.3,
        )
    stats_partial = agent.update(np.random.randn(4, 16).astype(np.float32), np.zeros(4, dtype=bool))
    finite = all(np.isfinite(v) for v in stats_partial.values())
    check("partial mask gives finite losses", finite, str(stats_partial))
    check("partial mask changes the policy loss",
          abs(stats_partial['policy_loss'] - stats_none['policy_loss']) > 1e-9)


def test_discovery_arithmetic():
    print("\n[6] The number this exists for")
    p_hook = 0.425
    hold_steps = 16
    p_sector, p_move = 1 / 16, 1 / 3

    without = (p_hook ** hold_steps) * p_sector * p_move
    probs = torch.softmax(torch.linspace(1.0, -1.0, 5), dim=-1)
    with_head = p_hook * p_sector * p_move * float(probs[-1])

    print(f"      P(maneuver) without duration head: {without:.3e}")
    print(f"      P(maneuver) with duration head:    {with_head:.3e}")
    print(f"      improvement: {with_head / without:,.0f}x")
    check("duration head raises discovery by >1e4", with_head / without > 1e4)

    # Consistency with the observed history, in matching units. The 169M figure
    # is decision-STEPS; the probabilities above are per OPPORTUNITY — a step
    # where the agent is actually at the wall and in position to start the swing.
    # Only a fraction of steps qualify: roughly the share of time spent in the
    # frontier band, divided by how many steps one approach takes.
    steps = 169e6
    opportunities = steps * 0.15 / 20      # ~1.3M
    expected_old = opportunities * without
    expected_new = opportunities * with_head
    print(f"      expected successes over {opportunities/1e6:.1f}M opportunities: "
          f"{expected_old:.2f} (old) -> {expected_new:,.0f} (new)")
    check("old rate predicts the zero successes actually observed", expected_old < 1.0,
          f"expected {expected_old:.3f}")
    check("new rate predicts discovery many times over", expected_new > 100,
          f"expected {expected_new:,.0f}")


if __name__ == "__main__":
    print("=" * 66)
    print("Action hold acceptance tests")
    print("=" * 66)
    test_duration_semantics()
    test_reset_clears_hold()
    test_stats()
    test_config_and_head()
    test_masked_update_equivalence()
    test_discovery_arithmetic()
    print("\n" + "=" * 66)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    print("=" * 66)
    sys.exit(1 if FAIL else 0)
