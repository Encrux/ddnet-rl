"""Normalised path progress along a reference trajectory.

Loads a waypoint file (the binary format emitted by ``ghost_parser.py`` and
``map_pathfinder.py``) and computes arc-length progress in [0, 1] for a batch of
world positions.

This is the one progress coordinate that is well-defined on every race map — it
depends only on a reference trajectory we generate ourselves, never on
mapper-authored markers like time checkpoints. See ``roadmap.md`` §1.7.

Used by:
  - the virtual finish line, which slices any map to an arbitrary length
    without editing the map
  - progress histograms and death heatmaps

The nearest-waypoint search is **temporally coherent**: the matched index may only
advance by as much as the tee could physically have moved since the last update.
A merely spatial window is not enough. Real reference paths double back on
themselves at close range — on Tutorial the ghost passes within ~100 units of a
point ~1000 units later along its own path, only ~17 waypoints apart — so a wide
window happily matches the wrong strand and the progress signal oscillates by ~1%
of the map. Under a high-water-mark reward that corruption would be permanent.
See ``roadmap.md`` §1.6, which records how the C++ side bounds its own search.

A full-scan fallback handles the genuinely-lost cases (teleport, mid-map respawn)
where temporal coherence is broken by design.
"""

from __future__ import annotations

import os
import struct
from typing import Optional, Tuple

import numpy as np


class PathProgress:
    """Vectorised arc-length progress along a polyline reference path.

    Args:
        waypoints_path: path to a ``.waypoints.bin`` file.
        num_envs: number of parallel environments to track.
        frame_skip: ticks per call to :meth:`update`. Used to size the forward
            window from physics when ``max_advance`` is not given explicitly.
        max_advance: how many waypoint indices the match may move *forward* per
            update. Leave as None to derive it: a tee tops out around
            ``MAX_TEE_SPEED`` units/tick, so one update covers at most
            ``MAX_TEE_SPEED * frame_skip`` units of arc. Sizing this from physics
            rather than generously is what keeps the match off the wrong strand
            where the path doubles back.
        max_retreat: how many indices the match may move *backward* per update.
            Small on purpose — real backtracking (setting up a jump) is slow.
        fallback_dist: if the best distance inside the window exceeds this (world
            units), scan forward from the tracked index instead. Escape hatch for
            teleports.
    """

    #: Conservative upper bound on tee speed in world units per tick. Measured
    #: max on the Tutorial ghost is ~72 (free fall); 100 leaves margin.
    MAX_TEE_SPEED = 100.0

    def __init__(
        self,
        waypoints_path: str,
        num_envs: int,
        frame_skip: int = 1,
        max_advance: Optional[int] = None,
        max_retreat: int = 2,
        fallback_dist: float = 512.0,
    ):
        self.path = waypoints_path
        self.num_envs = num_envs
        self._fallback_d2 = float(fallback_dist) ** 2

        self.xy = self._load(waypoints_path)  # (W, 2) float32
        if len(self.xy) < 2:
            raise ValueError(f"{waypoints_path}: need at least 2 waypoints, got {len(self.xy)}")

        seg = self.xy[1:] - self.xy[:-1]                      # (W-1, 2)
        seg_len = np.linalg.norm(seg, axis=1)                 # (W-1,)
        self.seg = seg.astype(np.float32)
        self.seg_len = seg_len.astype(np.float32)
        self._inv_seg_len2 = (1.0 / np.maximum(seg_len ** 2, 1e-6)).astype(np.float32)

        cum = np.zeros(len(self.xy), dtype=np.float32)
        np.cumsum(seg_len, out=cum[1:])
        self.cum = cum
        self.total_distance = float(cum[-1])
        self.spacing = float(np.median(seg_len))

        # Size the forward window from physics unless told otherwise.
        if max_advance is None:
            reach = self.MAX_TEE_SPEED * max(1, int(frame_skip))
            max_advance = int(np.ceil(reach / max(self.spacing, 1.0))) + 1
        self.max_advance = int(max_advance)
        self.max_retreat = int(max_retreat)

        # Per-env tracked waypoint index.
        self._idx = np.zeros(num_envs, dtype=np.int64)
        self._offsets = np.arange(-self.max_retreat, self.max_advance + 1, dtype=np.int64)
        self._env_ar = np.arange(num_envs)

    # ------------------------------------------------------------------ load

    @staticmethod
    def _load(path: str) -> np.ndarray:
        """Read ``[uint32 num_points][float32 x, float32 y]...``."""
        with open(path, "rb") as f:
            raw = f.read()
        if len(raw) < 4:
            raise ValueError(f"{path}: truncated waypoint file")
        (n,) = struct.unpack("<I", raw[:4])
        expected = 4 + n * 8
        if len(raw) < expected:
            raise ValueError(f"{path}: expected {expected} bytes for {n} points, got {len(raw)}")
        return np.frombuffer(raw, dtype=np.float32, count=n * 2, offset=4).reshape(n, 2).copy()

    @staticmethod
    def resolve(map_name: str, repo_root: str, explicit: Optional[str] = None) -> Optional[str]:
        """Find a waypoint file for ``map_name``, mirroring the server's lookup order."""
        if explicit:
            return explicit if os.path.exists(explicit) else None
        for d in ("waypoints", "ghosts"):
            for suffix in (".waypoints.bin", ".pathfind.waypoints.bin"):
                p = os.path.join(repo_root, d, f"{map_name}{suffix}")
                if os.path.exists(p):
                    return p
        return None

    # ------------------------------------------------------------- diagnostic

    def self_crossing_report(self, min_index_gap: Optional[int] = None,
                             near_dist: float = 128.0) -> dict:
        """Detect places where the path passes close to a distant part of itself.

        Anywhere this fires, a purely *spatial* nearest-waypoint match is
        ambiguous and the temporal coherence of :meth:`update` is what keeps the
        progress signal correct. The default index gap is ``max_advance``, i.e.
        it reports exactly the ambiguities the search window cannot resolve on
        its own. Cheap enough to run once at load.
        """
        if min_index_gap is None:
            min_index_gap = self.max_advance
        n = len(self.xy)
        near2 = near_dist ** 2
        worst = 0
        pairs = []
        # Chunked to keep the pairwise matrix small for long paths.
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            block = self.xy[start:stop]
            d2 = ((block[:, None, :] - self.xy[None, :, :]) ** 2).sum(-1)  # (b, n)
            rows = np.arange(start, stop)[:, None]
            cols = np.arange(n)[None, :]
            far = np.abs(rows - cols) > min_index_gap
            hit = far & (d2 < near2)
            if hit.any():
                ri, ci = np.nonzero(hit)
                worst += len(ri)
                for a, b in zip(ri[:5] + start, ci[:5]):
                    pairs.append((int(a), int(b)))
        return {"num_ambiguous": int(worst), "examples": pairs[:5]}

    # ---------------------------------------------------------------- runtime

    def reset(self, mask: Optional[np.ndarray] = None) -> None:
        """Reset the tracked index for the masked envs (all envs if None).

        A wrong tracked index is self-correcting via the full-scan fallback, so
        this is a hint rather than a requirement.
        """
        if mask is None:
            self._idx[:] = 0
        else:
            self._idx[mask] = 0

    def update(self, positions: np.ndarray) -> np.ndarray:
        """Advance the tracked index and return normalised progress in [0, 1].

        Args:
            positions: (num_envs, 2) float32 world coordinates.
        Returns:
            (num_envs,) float32 in [0, 1].
        """
        pos = np.asarray(positions, dtype=np.float32)
        n_wp = len(self.xy)

        # Windowed nearest-waypoint search around each env's tracked index.
        cand = np.clip(self._idx[:, None] + self._offsets[None, :], 0, n_wp - 1)  # (N, 2w+1)
        d2 = ((pos[:, None, :] - self.xy[cand]) ** 2).sum(-1)                     # (N, 2w+1)
        best = d2.argmin(axis=1)
        idx = cand[self._env_ar, best]
        best_d2 = d2[self._env_ar, best]

        # Fallback for envs whose window missed — teleports, or an agent that
        # wandered far off the reference path. Scanning is **forward-only**: a
        # backward global snap mid-episode is always wrong, because it would let
        # the match jump onto an earlier strand of a path that doubles back.
        # Only reset() may move the index backward without limit.
        stray = best_d2 > self._fallback_d2
        if stray.any():
            for e in np.nonzero(stray)[0]:
                start = int(self._idx[e])
                d2_fwd = ((pos[e] - self.xy[start:]) ** 2).sum(-1)
                idx[e] = start + int(d2_fwd.argmin())

        self._idx = idx

        # Project onto the segment starting at idx for sub-waypoint precision.
        i = np.minimum(idx, n_wp - 2)
        to_pos = pos - self.xy[i]
        proj = (to_pos * self.seg[i]).sum(-1) * self._inv_seg_len2[i]
        np.clip(proj, 0.0, 1.0, out=proj)
        dist = self.cum[i] + proj * self.seg_len[i]

        return (dist / max(self.total_distance, 1e-6)).astype(np.float32)

    # ------------------------------------------------------------------ extras

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) of the reference path, in world units."""
        lo = self.xy.min(axis=0)
        hi = self.xy.max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def progress_to_world(self, progress: float) -> np.ndarray:
        """World position at a given normalised progress — useful for slice finishes."""
        target = np.clip(progress, 0.0, 1.0) * self.total_distance
        i = int(np.searchsorted(self.cum, target, side="right") - 1)
        i = int(np.clip(i, 0, len(self.xy) - 2))
        frac = (target - self.cum[i]) / max(float(self.seg_len[i]), 1e-6)
        return self.xy[i] + np.clip(frac, 0.0, 1.0) * self.seg[i]

    def __len__(self) -> int:
        return len(self.xy)

    def __repr__(self) -> str:
        return (f"PathProgress({os.path.basename(self.path)}, {len(self.xy)} waypoints, "
                f"{self.total_distance:.0f} units)")
