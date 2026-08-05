"""Learned action hold — temporally-extended actions chosen by the policy.

The policy emits a hold DURATION alongside the action. The chosen action is
latched and re-sent for that many decisions. This is rung 3 of the temporal
abstraction ladder:

    frame skip (fixed, config)  ->  random repeat (ez-greedy)
      ->  LEARNED repeat (here)  ->  full action chunking

`frame_skip: 3` is rung 1 and has been in every recent run. This stops
hard-coding the number and lets the policy pick it per decision.

WHY. Independent per-step sampling cannot discover a maneuver that requires
holding an input. At the Tutorial wall (progress 0.455) the agent must hold hook
for ~16 consecutive decisions while swinging. At the measured hook rate of 0.425
that is 0.425^16 ~ 1e-6 for the hold alone; with the aim sector and movement it
came to ~2.5e-8 per opportunity. The observed zero successes in 169M steps bound
the true value below ~6e-9, which agrees. With a duration head the hold is ONE
draw, so the same maneuver is ~1.7e-3 — roughly 70,000x more likely.

Secondary benefits: the effective horizon shrinks (a 1200-step episode becomes
~300 decisions), so GAE has less distance to carry credit; and the reward
penalty for a tactical retreat ends up adjacent to its payoff instead of ~25
steps away from it.

KNOWN COST. During a hold the agent is blind — it cannot react to anything
unexpected. That reactivity/coherence tradeoff is inherent to every method on
this ladder. The mitigation is that duration 1 stays available, so the policy
can buy precision back wherever it matters.

CREDIT ASSIGNMENT. Only fresh-decision steps are policy-gradient targets. Held
steps still carry reward and train the critic — GAE naturally accumulates across
them — but their action was not sampled from the current policy, so including
them in the surrogate objective would train the policy to imitate its own latch.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

DEFAULT_DURATIONS: Tuple[int, ...] = (1, 2, 4, 8, 16)


class ActionHold:
    """Latches a sampled action for a policy-chosen number of decisions.

    The duration is read from the LAST discrete action dimension, which is where
    `PPOConfig.__post_init__` appends the duration head.
    """

    def __init__(self, num_envs: int, num_discrete: int, continuous_dim: int = 1,
                 durations: Sequence[int] = DEFAULT_DURATIONS):
        if not durations:
            raise ValueError("durations must be non-empty")
        if min(durations) < 1:
            raise ValueError(f"durations must all be >= 1, got {tuple(durations)}")

        self.durations = np.asarray(durations, dtype=np.int32)
        self.num_envs = num_envs
        self._remaining = np.zeros(num_envs, dtype=np.int32)
        self._discrete = np.zeros((num_envs, num_discrete), dtype=np.int64)
        self._continuous = np.zeros((num_envs, continuous_dim), dtype=np.float32)

        # Diagnostics. If the policy collapses to duration 1 the head is being
        # ignored, and that should be visible in the logs rather than inferred.
        self._dur_sum = 0
        self._dur_n = 0
        self._fresh_steps = 0
        self._total_steps = 0

    def apply(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Return (executed action, fresh-decision mask).

        Envs whose hold has expired latch the newly sampled action; the rest
        re-execute what they latched earlier and the sampled action is discarded.
        """
        fresh = self._remaining <= 0

        if fresh.any():
            self._discrete[fresh] = action['discrete'][fresh]
            self._continuous[fresh] = action['continuous'][fresh]
            # Duration head is the last discrete dimension.
            chosen = self.durations[action['discrete'][fresh, -1]]
            self._remaining[fresh] = chosen
            self._dur_sum += int(chosen.sum())
            self._dur_n += int(chosen.size)

        # Every env burns one decision of its hold, including the one just set:
        # a duration of 1 means "execute now, decide again next step".
        self._remaining -= 1

        self._fresh_steps += int(fresh.sum())
        self._total_steps += self.num_envs

        return (
            {'discrete': self._discrete.copy(), 'continuous': self._continuous.copy()},
            fresh,
        )

    def clear(self, mask: np.ndarray) -> None:
        """Drop the hold for envs that just reset.

        An episode boundary breaks the premise of the latch: the state the action
        was chosen for no longer exists, and on a curriculum restore the agent may
        be somewhere else on the map entirely.
        """
        self._remaining[mask] = 0

    def stats(self) -> Dict[str, float]:
        """Diagnostics since the last call. Resets the accumulators."""
        out = {}
        if self._dur_n:
            out['hold/mean_duration'] = self._dur_sum / self._dur_n
        if self._total_steps:
            out['hold/fresh_fraction'] = self._fresh_steps / self._total_steps
        self._dur_sum = 0
        self._dur_n = 0
        self._fresh_steps = 0
        self._total_steps = 0
        return out
