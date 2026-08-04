"""Cross-run-comparable episode metrics.

Mean episode reward stops being comparable across runs the moment the reward
function changes, which it is about to. These metrics don't: they measure what
the agent *did*, not what it scored.

  - **from-start finish rate** — the headline number
  - **progress percentiles** — how far along the reference path episodes get
  - **death heatmap** — a 2-D histogram of where episodes end, rendered for
    TensorBoard. This is the one that lets you see the frontier instead of
    inferring it.

See ``PLAN.md`` Phase 0.2.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np


# Black -> red -> orange -> yellow -> white. Perceptually crude but high contrast
# against the dim path overlay, which is what matters for spotting a hot cell.
_HEAT = np.array([
    (0, 0, 0), (60, 0, 40), (120, 0, 40), (180, 20, 20),
    (230, 70, 0), (255, 140, 0), (255, 200, 30), (255, 255, 160), (255, 255, 255),
], dtype=np.float32)


def _colorize(norm: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB uint8 via linear interpolation over _HEAT."""
    x = np.clip(norm, 0.0, 1.0) * (len(_HEAT) - 1)
    lo = np.floor(x).astype(np.int32)
    hi = np.minimum(lo + 1, len(_HEAT) - 1)
    t = (x - lo)[..., None]
    return (_HEAT[lo] * (1 - t) + _HEAT[hi] * t).astype(np.uint8)


class EpisodeTracker:
    """Rolling metrics over recently-finished episodes.

    Args:
        bounds: (min_x, min_y, max_x, max_y) of the reference path, world units.
        cell_size: heatmap cell size in world units (32 = one tile).
        margin: world units of padding added around the path bounding box.
        history: how many recent episodes the percentile/finish-rate window keeps.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        cell_size: float = 64.0,
        margin: float = 1024.0,
        history: int = 2000,
    ):
        min_x, min_y, max_x, max_y = bounds
        self.origin = np.array([min_x - margin, min_y - margin], dtype=np.float32)
        self.cell_size = float(cell_size)
        self.grid_w = max(1, int(np.ceil((max_x - min_x + 2 * margin) / cell_size)))
        self.grid_h = max(1, int(np.ceil((max_y - min_y + 2 * margin) / cell_size)))

        # Cumulative across the whole run — the frontier is a slow-moving thing.
        self.deaths = np.zeros((self.grid_h, self.grid_w), dtype=np.int64)

        self._progress = deque(maxlen=history)
        self._from_start = deque(maxlen=history)
        self._finished = deque(maxlen=history)
        self._lengths = deque(maxlen=history)
        self.total_episodes = 0
        self.total_finishes = 0
        self.best_progress = 0.0

    # ------------------------------------------------------------------ record

    def record(
        self,
        progress: np.ndarray,
        positions: np.ndarray,
        from_start: np.ndarray,
        finished: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        """Record a batch of just-ended episodes (all args same length)."""
        n = len(progress)
        if n == 0:
            return

        cells = ((positions - self.origin) / self.cell_size).astype(np.int32)
        np.clip(cells[:, 0], 0, self.grid_w - 1, out=cells[:, 0])
        np.clip(cells[:, 1], 0, self.grid_h - 1, out=cells[:, 1])
        # Only unfinished episodes are "deaths" — a finish is not a failure site.
        died = ~finished.astype(bool)
        if died.any():
            np.add.at(self.deaths, (cells[died, 1], cells[died, 0]), 1)

        self._progress.extend(progress.tolist())
        self._from_start.extend(np.asarray(from_start, dtype=bool).tolist())
        self._finished.extend(np.asarray(finished, dtype=bool).tolist())
        self._lengths.extend(np.asarray(lengths).tolist())

        self.total_episodes += n
        self.total_finishes += int(finished.sum())
        self.best_progress = max(self.best_progress, float(progress.max()))

    # ------------------------------------------------------------------- stats

    def stats(self) -> dict:
        """Rolling stats over the history window. Empty dict if nothing recorded."""
        if not self._progress:
            return {}

        prog = np.fromiter(self._progress, dtype=np.float32)
        from_start = np.fromiter(self._from_start, dtype=bool)
        finished = np.fromiter(self._finished, dtype=bool)
        lengths = np.fromiter(self._lengths, dtype=np.float32)

        out = {
            "progress/p10": float(np.percentile(prog, 10)),
            "progress/p50": float(np.percentile(prog, 50)),
            "progress/p90": float(np.percentile(prog, 90)),
            "progress/max": float(prog.max()),
            "progress/best_ever": self.best_progress,
            "episodes/finish_rate": float(finished.mean()),
            "episodes/length_mean": float(lengths.mean()),
            "episodes/n": int(len(prog)),
        }

        # The headline metric. Episodes that started mid-map say nothing about
        # whether the agent can actually run the map.
        if from_start.any():
            out["episodes/finish_rate_from_start"] = float(finished[from_start].mean())
            out["progress/p50_from_start"] = float(np.percentile(prog[from_start], 50))
            out["progress/max_from_start"] = float(prog[from_start].max())
            out["episodes/n_from_start"] = int(from_start.sum())
        return out

    def progress_values(self) -> np.ndarray:
        """Raw progress values, for ``writer.add_histogram``."""
        return np.fromiter(self._progress, dtype=np.float32)

    # ------------------------------------------------------------------- image

    def heatmap_image(self, path_xy: Optional[np.ndarray] = None,
                      upscale: int = 1) -> Optional[np.ndarray]:
        """Render the death heatmap as HWC uint8, path overlaid in dim blue.

        Returns None if no deaths have been recorded yet.
        """
        if self.deaths.sum() == 0:
            return None

        # Log scale: death counts are extremely peaked at the frontier, and a
        # linear scale shows one bright pixel and nothing else.
        norm = np.log1p(self.deaths.astype(np.float32))
        norm /= max(norm.max(), 1e-6)
        img = _colorize(norm)

        if path_xy is not None and len(path_xy):
            cells = ((path_xy - self.origin) / self.cell_size).astype(np.int32)
            ok = ((cells[:, 0] >= 0) & (cells[:, 0] < self.grid_w) &
                  (cells[:, 1] >= 0) & (cells[:, 1] < self.grid_h))
            cx, cy = cells[ok, 0], cells[ok, 1]
            # Only draw where nothing died, so the path never hides a hot cell.
            cold = self.deaths[cy, cx] == 0
            img[cy[cold], cx[cold]] = (40, 70, 120)

        if upscale > 1:
            img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
        return img

    def __repr__(self) -> str:
        return (f"EpisodeTracker(grid={self.grid_w}x{self.grid_h} @{self.cell_size:.0f}u, "
                f"episodes={self.total_episodes}, finishes={self.total_finishes})")
