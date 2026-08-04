"""Phase 0 acceptance tests — no server required.

Run: python test_phase0.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_tracker import EpisodeTracker
from ghost_parser import extract_positions
from path_progress import PathProgress

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
WAYPOINTS = os.path.join(REPO, "ghosts/Tutorial.waypoints.bin")
GHOST = os.path.join(REPO, "ghosts/Tutorial_Infinity_187.120_2026-03-12_23-11-23_"
                            "796a3716fe64657bfb8bc6af5f9422b197278919a9d875e43b9bbbcb73262fc0.gho")

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def test_progress_tracks_ghost():
    """Replaying the ghost's own positions must sweep progress 0 -> 1.

    This is the single most valuable test in Phase 0: it catches wrong-strand
    matching, phantom waypoints across teleports, and cumulative-distance bugs
    all at once.
    """
    print("progress vs ghost ground truth")
    gp = np.array(extract_positions(GHOST), dtype=np.float32)
    # Waypoints were resampled from exactly this polyline, so cumulative ghost
    # length is the ground-truth arc position at every frame.
    step = np.linalg.norm(np.diff(gp, axis=0), axis=1)
    gt = np.concatenate([[0.0], np.cumsum(step)])
    gt /= gt[-1]

    pp = PathProgress(WAYPOINTS, num_envs=1, frame_skip=3)
    est = np.array([pp.update(gp[k:k + 1])[0] for k in range(len(gp))])
    err = np.abs(est - gt)

    check("starts at 0", est[0] < 1e-3, f"{est[0]:.5f}")
    check("ends at 1", est[-1] > 0.999, f"{est[-1]:.5f}")
    # Floor is the 64-unit resampling corner-cut: path is ~2.6% shorter than the
    # raw ghost polyline, so ~0.003 error is inherent, not a defect.
    check("max error <= 0.006", err.max() <= 0.006, f"max={err.max():.4f} p99={np.percentile(err, 99):.4f}")
    check("unbiased", abs((est - gt).mean()) < 1e-3, f"mean={(est - gt).mean():+.5f}")


def test_no_wrong_strand_matching():
    """A wide spatial window matches the wrong strand where the path doubles back.

    Tutorial's ghost passes within ~100 units of a point ~1000 units later along
    its own path, only ~17 waypoints apart. Temporal coherence is what prevents
    the progress signal from oscillating by ~1% of the map there.
    """
    print("wrong-strand matching")
    gp = np.array(extract_positions(GHOST), dtype=np.float32)

    tight = PathProgress(WAYPOINTS, num_envs=1, frame_skip=3)
    wide = PathProgress(WAYPOINTS, num_envs=1, max_advance=96, max_retreat=96)
    e_tight = np.array([tight.update(gp[k:k + 1])[0] for k in range(len(gp))])
    e_wide = np.array([wide.update(gp[k:k + 1])[0] for k in range(len(gp))])

    j_tight = np.diff(e_tight).max()
    j_wide = np.diff(e_wide).max()
    # A tee cannot advance more than MAX_TEE_SPEED * ticks of arc in one update.
    phys_limit = PathProgress.MAX_TEE_SPEED * 2 / tight.total_distance * 3  # 2 ticks/ghost frame, 3x margin
    check("no physically impossible jumps", j_tight < phys_limit,
          f"max jump {j_tight * tight.total_distance:.0f} units (limit {phys_limit * tight.total_distance:.0f})")
    check("beats a wide spatial window", j_tight < j_wide,
          f"tight={j_tight * tight.total_distance:.0f}u vs wide={j_wide * wide.total_distance:.0f}u")


def test_teleport_fallback_is_forward_only():
    """After a jump, the match may snap forward but never backward."""
    print("teleport fallback")
    pp = PathProgress(WAYPOINTS, num_envs=1, frame_skip=3)
    for target in (0.1, 0.3, 0.5):
        pp.update(pp.progress_to_world(target)[None].astype(np.float32))
    before = float(pp.update(pp.progress_to_world(0.5)[None].astype(np.float32))[0])
    # Teleport far backwards in space: the match must not follow.
    after = float(pp.update(pp.progress_to_world(0.05)[None].astype(np.float32))[0])
    check("does not snap backward", after >= before - 1e-3, f"{before:.3f} -> {after:.3f}")

    pp.reset()
    reset_prog = float(pp.update(pp.progress_to_world(0.05)[None].astype(np.float32))[0])
    check("reset() releases the index", abs(reset_prog - 0.05) < 0.01, f"{reset_prog:.3f}")


def test_virtual_finish():
    """finish_progress must map to a reachable world position and fire correctly."""
    print("virtual finish line")
    pp = PathProgress(WAYPOINTS, num_envs=4, frame_skip=3)
    for thresh in (0.05, 0.5, 1.0):
        world = pp.progress_to_world(thresh)
        pos = np.repeat(world[None].astype(np.float32), 4, axis=0)
        pp.reset()
        # Walk up to the target so the temporal window can track it.
        for f in np.linspace(0.0, thresh, 200):
            prog = pp.update(np.repeat(pp.progress_to_world(float(f))[None].astype(np.float32), 4, 0))
        check(f"reaches threshold {thresh}", bool((prog >= thresh - 0.01).all()),
              f"progress={prog[0]:.4f} at world ({world[0]:.0f}, {world[1]:.0f})")


def test_tracker():
    print("episode tracker")
    pp = PathProgress(WAYPOINTS, num_envs=8, frame_skip=3)
    tr = EpisodeTracker(pp.bounds)
    rng = np.random.default_rng(0)
    prog = np.array([0.1, 0.2, 1.0, 1.0, 0.5], dtype=np.float32)
    fin = np.array([False, False, True, True, False])
    from_start = np.array([True, True, True, False, True])
    pos = np.stack([pp.progress_to_world(float(p)) for p in prog])
    tr.record(prog, pos, from_start, fin, np.array([100, 200, 300, 400, 500]))

    s = tr.stats()
    check("finish rate", abs(s["episodes/finish_rate"] - 2 / 5) < 1e-6, f"{s['episodes/finish_rate']:.3f}")
    # from_start episodes are indices 0,1,2,4 -> one finish out of four
    check("from-start finish rate excludes mid-map spawns",
          abs(s["episodes/finish_rate_from_start"] - 1 / 4) < 1e-6,
          f"{s['episodes/finish_rate_from_start']:.3f}")
    check("deaths exclude finishes", tr.deaths.sum() == 3, f"{tr.deaths.sum()} death cells")

    img = tr.heatmap_image(pp.xy)
    check("heatmap renders", img is not None and img.dtype == np.uint8 and img.ndim == 3,
          f"{None if img is None else img.shape}")


if __name__ == "__main__":
    import contextlib, io
    for fn in (test_progress_tracks_ghost, test_no_wrong_strand_matching,
               test_teleport_fallback_is_forward_only, test_virtual_finish, test_tracker):
        with contextlib.redirect_stderr(io.StringIO()):  # ghost_parser is chatty
            fn()
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        sys.exit(1)
    print("all Phase 0 acceptance tests passed")
