#!/usr/bin/env python3
"""Render what an RL bot actually sees, from the live shared memory.

Launches a server with bots and draws one bot's observation: the tile grid
reindexed through the same semantic table the policy uses, the hook aim-line
probes as their grab points, and the path lookahead. The point is to check an
encoding against runtime values instead of trusting the struct layout -- an
encoding bug here does not crash, it just plateaus, and one cost this project
four runs.

Usage:
    python visualize_obs.py                      # terminal, live
    python visualize_obs.py --frames 1           # one snapshot, then exit
    python visualize_obs.py --window             # interactive OpenCV window
    python visualize_obs.py --num-bots 4 --bot-id 2 --map Tutorial

Window controls (--window):
    a/d: move left/right    space: jump    h: hook
    r: toggle hook probes   l: toggle lookahead    q: quit
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from observation import (
    HOOK_LENGTH, NUM_AIM_SECTORS, NUM_LOOKAHEAD,
    TILE_GRID_HEIGHT, TILE_GRID_WIDTH,
    StructuredObservation, TileClass, aim_sector_direction,
)

TILE_SIZE = 32.0  # world units per tile

# One glyph, one background colour and one RGB per semantic class, so the
# terminal and the window agree with each other and with the policy's channels.
CLASS_STYLE = {
    TileClass.AIR:      (' ', '\033[40m',  (28, 28, 36)),
    TileClass.SOLID:    ('#', '\033[47m',  (205, 205, 205)),
    TileClass.NOHOOK:   ('o', '\033[43m',  (190, 180, 90)),
    TileClass.THROUGH:  ('.', '\033[100m', (95, 95, 110)),
    TileClass.DEATH:    ('X', '\033[41m',  (200, 55, 55)),
    TileClass.FREEZE:   ('*', '\033[46m',  (95, 190, 240)),
    TileClass.UNFREEZE: ('+', '\033[42m',  (95, 225, 120)),
    TileClass.TELE:     ('@', '\033[45m',  (195, 100, 240)),
    TileClass.BOOST:    ('>', '\033[44m',  (90, 110, 235)),
    TileClass.JUMP:     ('^', '\033[102m', (130, 240, 170)),
    TileClass.STOPPER:  ('=', '\033[107m', (160, 160, 175)),
    TileClass.OTHER:    ('?', '\033[100m', (120, 120, 120)),
}

RESET = '\033[0m'
FG_DARK = '\033[30m'
FG_LIGHT = '\033[97m'


def hook_grab_points(obs: StructuredObservation):
    """(sector, world offset, reach) for every sector whose hook would grab."""
    out = []
    for k in range(NUM_AIM_SECTORS):
        reach = float(obs.hook_reach[k])
        if reach <= 0.0:
            continue
        dist = obs.hook_reach_distance(k)
        dx, dy = aim_sector_direction(k)
        out.append((k, (dx * dist, dy * dist), reach))
    return out


def status_lines(obs: StructuredObservation, reward: float, done: bool):
    p = obs.local_player
    grabs = hook_grab_points(obs)
    reachable = ''.join('#' if obs.hook_reach[k] > 0 else '.'
                        for k in range(NUM_AIM_SECTORS))
    return [
        f"tick {obs.tick}   pos ({p.pos_x:8.1f}, {p.pos_y:8.1f})   "
        f"vel ({p.vel_x:6.2f}, {p.vel_y:6.2f})  |v| {p.speed:5.2f}",
        f"hook {p.hook_name:<14} grounded {str(obs.is_grounded):<5} "
        f"jumps {obs.jumps_remaining}  can_hook {str(obs.can_hook):<5} "
        f"frozen {p.is_frozen}",
        f"progress {obs.progress:.4f}   high-water {obs.max_progress:.4f}   "
        f"reward {reward:+.4f}   done {done}",
        f"hook probes [{reachable}]  {len(grabs)}/{NUM_AIM_SECTORS} would grab"
        + (f"   nearest {min(obs.hook_reach_distance(k) for k, _, _ in grabs):.0f}u"
           if grabs else ""),
    ]


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def render_terminal(obs: StructuredObservation, reward: float, done: bool,
                    use_color: bool = True) -> str:
    """The grid as text, with hook grab points and path lookahead overlaid."""
    classes = obs.semantic_grid
    cx, cy = TILE_GRID_WIDTH // 2, TILE_GRID_HEIGHT // 2

    # Overlays, lowest priority first; the tee always wins.
    overlay = {}
    for _sector, (ox, oy), _reach in hook_grab_points(obs):
        gx, gy = cx + int(round(ox / TILE_SIZE)), cy + int(round(oy / TILE_SIZE))
        if 0 <= gx < TILE_GRID_WIDTH and 0 <= gy < TILE_GRID_HEIGHT:
            overlay[(gx, gy)] = ('H', '\033[101m')
    for i in range(NUM_LOOKAHEAD):
        ox, oy = obs.lookahead[i]
        gx, gy = cx + int(round(ox / TILE_SIZE)), cy + int(round(oy / TILE_SIZE))
        if 0 <= gx < TILE_GRID_WIDTH and 0 <= gy < TILE_GRID_HEIGHT:
            overlay[(gx, gy)] = (str(i + 1), '\033[103m')
    overlay[(cx, cy)] = ('T', '\033[105m')

    rows = []
    for y in range(TILE_GRID_HEIGHT):
        cells = []
        for x in range(TILE_GRID_WIDTH):
            if (x, y) in overlay:
                char, bg = overlay[(x, y)]
                fg = FG_DARK
            else:
                char, bg, _ = CLASS_STYLE[TileClass(classes[y, x])]
                fg = FG_DARK if bg == '\033[47m' else FG_LIGHT
            cells.append(f"{bg}{fg}{char}{RESET}" if use_color else char)
        rows.append(''.join(cells))

    legend = '  '.join(
        f"{CLASS_STYLE[c][0]} {c.name.lower()}"
        for c in (TileClass.SOLID, TileClass.NOHOOK, TileClass.THROUGH,
                  TileClass.DEATH, TileClass.FREEZE, TileClass.TELE))
    footer = ['', 'T tee   H hook grab point   1-8 path lookahead', legend]
    return '\n'.join(status_lines(obs, reward, done) + [''] + rows + footer)


# ---------------------------------------------------------------------------
# OpenCV window
# ---------------------------------------------------------------------------


def render_window(obs: StructuredObservation, reward: float, done: bool,
                  scale: int, show_probes: bool, show_lookahead: bool):
    import cv2

    classes = obs.semantic_grid
    rgb = np.zeros((TILE_GRID_HEIGHT, TILE_GRID_WIDTH, 3), dtype=np.uint8)
    for c in TileClass:
        r, g, b = CLASS_STYLE[c][2]
        rgb[classes == int(c)] = (b, g, r)  # OpenCV is BGR

    img = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    half = scale // 2
    centre = (TILE_GRID_WIDTH // 2 * scale + half, TILE_GRID_HEIGHT // 2 * scale + half)

    def to_px(ox, oy):
        return (int(centre[0] + ox / TILE_SIZE * scale),
                int(centre[1] + oy / TILE_SIZE * scale))

    if show_probes:
        for sector in range(NUM_AIM_SECTORS):
            dx, dy = aim_sector_direction(sector)
            reach = float(obs.hook_reach[sector])
            if reach > 0.0:
                end = to_px(dx * obs.hook_reach_distance(sector),
                            dy * obs.hook_reach_distance(sector))
                cv2.line(img, centre, end, (120, 255, 120), 1)
                cv2.circle(img, end, max(2, scale // 4), (60, 255, 60), -1)
            else:
                # Draw the miss faintly at full extension: knowing where the
                # hook would NOT grab is half the signal.
                cv2.line(img, centre, to_px(dx * HOOK_LENGTH, dy * HOOK_LENGTH),
                         (70, 70, 140), 1)

    if show_lookahead:
        prev = centre
        for i in range(NUM_LOOKAHEAD):
            px = to_px(float(obs.lookahead[i, 0]), float(obs.lookahead[i, 1]))
            cv2.line(img, prev, px, (60, 220, 255), 1)
            cv2.circle(img, px, max(2, scale // 4), (0, 200, 255), -1)
            prev = px

    cv2.circle(img, centre, max(3, scale // 2 - 1), (255, 120, 255), -1)
    cv2.circle(img, centre, max(3, scale // 2 - 1), (255, 255, 255), 1)

    panel = np.zeros((22 * len(status_lines(obs, reward, done)) + 10,
                      img.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(status_lines(obs, reward, done)):
        cv2.putText(panel, line, (6, 18 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([panel, img])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args) -> int:
    from teeworlds_server_env import ServerEnvConfig, TeeWorldsServerEnv

    cv2 = None
    if args.window:
        try:
            import cv2  # noqa: F811
        except ImportError:
            print("OpenCV not available; falling back to the terminal renderer.\n"
                  "Install it with: uv pip install opencv-python")
            args.window = False

    config = ServerEnvConfig(map_name=args.map, max_episode_steps=10 ** 9,
                             frame_skip=args.frame_skip)
    env = TeeWorldsServerEnv(num_bots=args.num_bots, num_servers=1, config=config)
    if not env.shm_clients:
        print("No server came up -- is build/DDNet-Server built?")
        return 1
    shm = env.shm_clients[0]
    if args.bot_id >= args.num_bots:
        print(f"--bot-id {args.bot_id} is out of range for --num-bots {args.num_bots}")
        env.close()
        return 2

    print(f"Server up with {args.num_bots} bot(s) on {args.map}; showing bot {args.bot_id}.")

    show_probes = True
    show_lookahead = True
    direction, jump, hook = 0, 0, 0
    frame_time = 1.0 / args.fps if args.fps > 0 else 0.0
    frames = 0

    try:
        while args.frames <= 0 or frames < args.frames:
            start = time.time()
            obs, reward, done = shm.read_observation(args.bot_id)

            if obs is None:
                time.sleep(0.02)
                continue

            if args.window:
                img = render_window(obs, reward, done, args.scale,
                                    show_probes, show_lookahead)
                cv2.imshow('DDNet RL observation', img)
            else:
                text = render_terminal(obs, reward, done, use_color=not args.no_color)
                if args.frames != 1:
                    print('\033[2J\033[H', end='')
                print(text, flush=True)

            # Keep the server stepping. Aim straight right unless driven.
            discrete = np.zeros((args.num_bots, 4), dtype=np.int32)
            discrete[:, 0] = 1  # neutral direction for every bot
            discrete[args.bot_id, 0] = direction + 1
            discrete[args.bot_id, 1] = jump
            discrete[args.bot_id, 2] = hook
            shm.write_actions_batch({
                'discrete': discrete,
                'continuous': np.zeros((args.num_bots, 1), dtype=np.float32),
            })
            shm.signal_actions_ready()
            shm.wait_for_observations(timeout=0.2)
            jump = 0
            frames += 1

            if args.window:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    direction = -1
                elif key == ord('d'):
                    direction = 1
                elif key == ord('s'):
                    direction = 0
                elif key == ord(' '):
                    jump = 1
                elif key == ord('h'):
                    hook = 1 - hook
                elif key == ord('r'):
                    show_probes = not show_probes
                elif key == ord('l'):
                    show_lookahead = not show_lookahead
            elif frame_time:
                remaining = frame_time - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.close()
        if args.window and cv2 is not None:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a bot's observation from live shared memory")
    parser.add_argument('--num-bots', type=int, default=1)
    parser.add_argument('--bot-id', type=int, default=0)
    parser.add_argument('--map', type=str, default='Tutorial')
    parser.add_argument('--frame-skip', type=int, default=1)
    parser.add_argument('--frames', type=int, default=0,
                        help='Stop after N frames; 0 runs until interrupted')
    parser.add_argument('--window', action='store_true',
                        help='Interactive OpenCV window instead of the terminal')
    parser.add_argument('--scale', type=int, default=16, help='Pixels per tile (--window)')
    parser.add_argument('--fps', type=float, default=10.0, help='Refresh rate')
    parser.add_argument('--no-color', action='store_true', help='Plain ASCII output')
    # Accepted so existing commands keep working; server-side is the only mode now.
    parser.add_argument('--server-side', action='store_true', help=argparse.SUPPRESS)
    return run(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
