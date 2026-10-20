"""Acceptance test: observation.py and rl_protocol.h describe the same bytes.

A layout mismatch between the two sides does not raise. The server writes 1520
bytes, Python reads them at whatever offsets it believes in, and training
continues on misparsed input -- which is how the hook-state bug survived four
runs. So the offsets are pinned here against the C++ header itself, not against
a copy of them.

Covers:
  - the struct size, read out of rl_protocol.h's static_assert
  - every field offset, against the ones the batched training readers use
  - a byte-exact round trip through StructuredObservation.from_bytes
  - the semantic tile table, including that ppo_hybrid builds its one-hot from
    the same mapping the observation viewer colours tiles with

Run: python test_obs_layout.py

NEVER import train.py here -- its atexit handler pkills all game servers.
"""

import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import observation as obsmod
from observation import (NUM_AIM_SECTORS, NUM_LOOKAHEAD, PlayerObs,
                         SEMANTIC_CHANNELS, StructuredObservation,
                         TILE_GRID_HEIGHT, TILE_GRID_SIZE, TILE_GRID_WIDTH,
                         TileClass, semantic_lookup)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
_PROTOCOL_H = os.path.join(_REPO_ROOT, 'src', 'game', 'rl', 'rl_protocol.h')

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _failures.append(name)


def cpp_static_assert_size(header_text, struct_name):
    m = re.search(r'static_assert\(sizeof\(%s\)\s*==\s*(\d+)' % struct_name, header_text)
    return int(m.group(1)) if m else None


def cpp_constant(header_text, name):
    """Value of a `constexpr int` in the header, evaluating a product of two."""
    m = re.search(r'constexpr int %s\s*=\s*([^;]+);' % name, header_text)
    if not m:
        return None
    expr = m.group(1).strip()
    if expr.isdigit():
        return int(expr)
    factors = [f.strip() for f in expr.split('*')]
    try:
        return int(np.prod([cpp_constant(header_text, f) for f in factors]))
    except TypeError:
        return None


def test_against_header():
    print("\n[1] Sizes and constants against rl_protocol.h")
    if not os.path.exists(_PROTOCOL_H):
        check("rl_protocol.h found", False, _PROTOCOL_H)
        return
    text = open(_PROTOCOL_H, encoding='utf-8').read()

    obs_size = cpp_static_assert_size(text, 'SStructuredObservation')
    player_size = cpp_static_assert_size(text, 'SPlayerObs')
    check("SStructuredObservation size matches the C++ static_assert",
          obs_size == StructuredObservation.STRUCT_SIZE,
          f"C++ {obs_size}, Python {StructuredObservation.STRUCT_SIZE}")
    check("SPlayerObs size matches the C++ static_assert",
          player_size == PlayerObs.STRUCT_SIZE,
          f"C++ {player_size}, Python {PlayerObs.STRUCT_SIZE}")
    check("struct format string agrees with the declared size",
          struct.calcsize(PlayerObs.STRUCT_FORMAT) == PlayerObs.STRUCT_SIZE)

    for name, py in (('NUM_LOOKAHEAD', NUM_LOOKAHEAD),
                     ('NUM_AIM_SECTORS', NUM_AIM_SECTORS),
                     ('TILE_GRID_WIDTH', TILE_GRID_WIDTH),
                     ('TILE_GRID_HEIGHT', TILE_GRID_HEIGHT),
                     ('TILE_GRID_SIZE', TILE_GRID_SIZE)):
        cpp = cpp_constant(text, name)
        check(f"{name} matches", cpp == py, f"C++ {cpp}, Python {py}")

    # The offsets are a sum of field sizes, so a size agreement is only half the
    # story: the fields have to be in the same order too.
    check("offsets reach exactly the end of the struct",
          obsmod.OFF_HOOK_REACH + 4 * NUM_AIM_SECTORS + 3 == obs_size,
          f"{obsmod.OFF_HOOK_REACH} + {4 * NUM_AIM_SECTORS} + 3 vs {obs_size}")


def test_against_training_reader():
    print("\n[2] Offsets against the batched training readers")
    # teeworlds_server_env's hot path hardcodes these, derived independently.
    LOCAL_PLAYER = 8
    TILES = LOCAL_PLAYER + 36
    EXTRA_STATE = TILES + 1333
    NAV = EXTRA_STATE + 4
    check("local player offset", obsmod.OFF_LOCAL_PLAYER == LOCAL_PLAYER)
    check("tile grid offset", obsmod.OFF_TILE_GRID == TILES)
    check("extra state offset", obsmod.OFF_IS_GROUNDED == EXTRA_STATE)
    check("navigation block offset", obsmod.OFF_PROGRESS == NAV)

    from teeworlds_server_env import NAV_DIM, OBS_SIZE
    check("OBS_SIZE matches", OBS_SIZE == StructuredObservation.STRUCT_SIZE,
          f"env {OBS_SIZE}, observation.py {StructuredObservation.STRUCT_SIZE}")
    check("NAV_DIM covers the whole navigation block",
          NAV_DIM == 1 + 2 * NUM_LOOKAHEAD + 1 + NUM_AIM_SECTORS,
          f"{NAV_DIM}")
    check("the navigation block ends where hook reach ends",
          NAV + 4 * NAV_DIM == obsmod.OFF_HOOK_REACH + 4 * NUM_AIM_SECTORS)


def build_blob(**over):
    """A 1520-byte observation with distinguishable values in every field."""
    tile_grid = (np.arange(TILE_GRID_SIZE, dtype=np.int32) % 256).astype(np.uint8)
    lookahead_x = np.arange(NUM_LOOKAHEAD, dtype=np.float32) * 10.0 + 1.0
    lookahead_y = np.arange(NUM_LOOKAHEAD, dtype=np.float32) * -10.0 - 1.0
    hook_reach = np.linspace(0.0, 1.0, NUM_AIM_SECTORS, dtype=np.float32)

    blob = bytearray(StructuredObservation.STRUCT_SIZE)
    struct.pack_into('<IBB', blob, 0, over.get('tick', 123456), 7, 3)
    struct.pack_into(PlayerObs.STRUCT_FORMAT, blob, obsmod.OFF_LOCAL_PLAYER,
                     100.5, -200.25, 3.5, -4.75,   # pos, vel
                     -1, 2, obsmod.HOOK_GRABBED, 5,  # direction, jumped, hook, weapon
                     111.0, 222.0,                 # hook pos
                     -1, 1, 1, 1, 10, 4)           # hooked player, flags, health, armor
    blob[obsmod.OFF_TILE_GRID:obsmod.OFF_TILE_GRID + TILE_GRID_SIZE] = tile_grid.tobytes()
    struct.pack_into('<3Bx', blob, obsmod.OFF_IS_GROUNDED, 1, 2, 1)
    struct.pack_into('<f', blob, obsmod.OFF_PROGRESS, 0.4763)
    blob[obsmod.OFF_LOOKAHEAD_X:obsmod.OFF_LOOKAHEAD_X + 4 * NUM_LOOKAHEAD] = lookahead_x.tobytes()
    blob[obsmod.OFF_LOOKAHEAD_Y:obsmod.OFF_LOOKAHEAD_Y + 4 * NUM_LOOKAHEAD] = lookahead_y.tobytes()
    struct.pack_into('<f', blob, obsmod.OFF_MAX_PROGRESS, 0.5050)
    blob[obsmod.OFF_HOOK_REACH:obsmod.OFF_HOOK_REACH + 4 * NUM_AIM_SECTORS] = hook_reach.tobytes()
    return bytes(blob), tile_grid, lookahead_x, lookahead_y, hook_reach


def test_round_trip():
    print("\n[3] Round trip through from_bytes")
    blob, tile_grid, lx, ly, hook_reach = build_blob()
    obs = StructuredObservation.from_bytes(blob)

    check("tick", obs.tick == 123456, str(obs.tick))
    check("game flags / state", (obs.game_flags, obs.game_state) == (7, 3))

    p = obs.local_player
    check("position", (p.pos_x, p.pos_y) == (100.5, -200.25))
    check("velocity", (p.vel_x, p.vel_y) == (3.5, -4.75))
    check("direction keeps its sign", p.direction == -1, str(p.direction))
    check("hook state is not remapped", p.hook_state == obsmod.HOOK_GRABBED)
    check("hook state decodes to a name", p.hook_name == 'grabbed', p.hook_name)
    check("retracted decodes from 255",
          PlayerObs.from_bytes(struct.pack(
              PlayerObs.STRUCT_FORMAT, 0, 0, 0, 0, 0, 0, obsmod.HOOK_RETRACTED,
              0, 0, 0, 0, 1, 0, 1, 0, 0)).hook_name == 'retracted')
    check("hook position", (p.hook_pos_x, p.hook_pos_y) == (111.0, 222.0))
    check("health / armor", (p.health, p.armor) == (10, 4))
    check("speed", abs(p.speed - float(np.hypot(3.5, -4.75))) < 1e-6)

    check("tile grid shape", obs.tile_grid.shape == (TILE_GRID_HEIGHT, TILE_GRID_WIDTH))
    check("tile grid is row-major and byte-exact",
          np.array_equal(obs.tile_grid.ravel(), tile_grid))
    check("tile_at(0, 0) is the centre tile",
          obs.tile_at(0, 0) == int(obs.tile_grid[TILE_GRID_HEIGHT // 2,
                                                 TILE_GRID_WIDTH // 2]))

    check("grounded / jumps / can_hook",
          (obs.is_grounded, obs.jumps_remaining, obs.can_hook) == (True, 2, True))
    check("progress", abs(obs.progress - 0.4763) < 1e-6, f"{obs.progress}")
    check("high-water progress", abs(obs.max_progress - 0.5050) < 1e-6,
          f"{obs.max_progress}")
    check("lookahead x column", np.allclose(obs.lookahead[:, 0], lx))
    check("lookahead y column", np.allclose(obs.lookahead[:, 1], ly))
    check("hook reach", np.allclose(obs.hook_reach, hook_reach))
    check("parsed arrays own their memory (the blob may be a live mmap)",
          obs.tile_grid.flags.owndata and obs.hook_reach.flags.owndata
          and obs.lookahead.flags.owndata)

    # A short read must fail loudly rather than silently misparse.
    truncated = False
    try:
        StructuredObservation.from_bytes(blob[:-1])
    except ValueError:
        truncated = True
    check("a short observation raises instead of misparsing", truncated)


def test_semantics():
    print("\n[4] Semantic tile table")
    table = semantic_lookup()
    check("table covers all 256 ids", table.shape == (256,), str(table.shape))
    check("every entry is a valid class",
          set(np.unique(table)).issubset({int(c) for c in TileClass}))
    check("SEMANTIC_CHANNELS matches the enum", SEMANTIC_CHANNELS == len(TileClass))
    check("air is air", table[0] == int(TileClass.AIR))
    # The three that motivate the whole table: neighbouring ids, opposite meanings.
    check("SOLID(1) / NOHOOK(3) / THROUGH(6) land in distinct channels",
          len({table[1], table[3], table[6]}) == 3,
          f"{table[1]}, {table[3]}, {table[6]}")
    check("all three freeze variants map to FREEZE",
          {table[9], table[12], table[144]} == {int(TileClass.FREEZE)})
    check("unmapped ids fall into OTHER", table[200] == int(TileClass.OTHER))

    obs = StructuredObservation.from_bytes(build_blob()[0])
    check("semantic_grid reindexes without changing shape",
          obs.semantic_grid.shape == obs.tile_grid.shape)
    check("semantic_grid agrees with the table elementwise",
          np.array_equal(obs.semantic_grid, table[obs.tile_grid]))

    try:
        from ppo_hybrid import SEMANTIC_CHANNELS as ppo_channels, build_semantic_table
    except ImportError as e:
        check("ppo_hybrid importable (needs torch)", False, str(e))
        return
    one_hot = build_semantic_table().numpy()
    check("ppo_hybrid uses the same channel count", ppo_channels == SEMANTIC_CHANNELS)
    check("the policy's one-hot is built from this table",
          np.array_equal(one_hot.argmax(axis=1).astype(np.uint8), table))
    check("the one-hot really is one-hot", np.array_equal(one_hot.sum(axis=1),
                                                          np.ones(256, dtype=np.float32)))


def main():
    print("=" * 68)
    print("Observation layout acceptance tests")
    print("=" * 68)
    test_against_header()
    test_against_training_reader()
    test_round_trip()
    test_semantics()

    print("\n" + "=" * 68)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    else:
        print("All observation layout tests passed")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
