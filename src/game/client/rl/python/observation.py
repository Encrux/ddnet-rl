#!/usr/bin/env python3
"""Wire format for the DDNet RL observation.

This mirrors ``SStructuredObservation`` in ``src/game/rl/rl_protocol.h`` and is
the only Python definition of that layout: offsets, field order, sizes, and the
meaning of the raw tile ids the server writes. No transport, no environment
logic, so both the training env and the offline tooling can depend on it.

A mismatch with the C++ side shows up as silently misparsed observations, never
as an error -- that is what ``STRUCT_SIZE`` and ``test_obs_layout.py`` are for.

The hot training path does NOT go through the dataclasses here; it parses the
shared-memory block straight into batched numpy arrays
(``teeworlds_server_env.read_observations_batch_cnn`` and friends) using the
same offsets. These classes are for tooling that wants one bot's observation as
an object -- ``visualize_obs.py`` above all, since checking an encoding against
live runtime values is the only thing that catches an encoding bug.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geometry (must match rl_protocol.h)
# ---------------------------------------------------------------------------

TILE_GRID_WIDTH = 43   # 21 left + centre + 21 right
TILE_GRID_HEIGHT = 31  # 15 up + centre + 15 down
TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT  # 1333

NUM_LOOKAHEAD = 8      # path points ahead, at LOOKAHEAD_OFFSETS waypoints out
NUM_AIM_SECTORS = 16   # hook reachability probes, aligned with the aim head

LOOKAHEAD_OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128)

# Default DDNet hook tuning. The server encodes reach as
# max(0.05, 1 - distance / HOOK_LENGTH), so a reach value converts back to a
# distance only under the tuning the server was running.
HOOK_LENGTH = 380.0

# DDNet hook states (game/gamecore.h). The enum is SPARSE -- there is no value 2
# -- and m_HookState travels as uint8, so RETRACTED (-1) arrives as 255. The
# observation used to one-hot over {0, 1, 2}, which collapsed FLYING, GRABBED and
# RETRACTED into the all-zeros vector: measured in play, the agent could not tell
# its own hook state apart 60% of the time. Decode against these names, never
# against a contiguous range.
HOOK_IDLE = 0
HOOK_RETRACT_START = 1
HOOK_RETRACT_END = 3
HOOK_FLYING = 4
HOOK_GRABBED = 5
HOOK_RETRACTED = 255


# ---------------------------------------------------------------------------
# Tile semantics
# ---------------------------------------------------------------------------
#
# The grid carries RAW DDNet tile indices, clamped to uint8 -- not the semantic
# classes below. That is deliberate: the raw id is the lossless thing to put on
# the wire, and both consumers reindex it themselves. But raw ids are a terrible
# input to a network, because the distinctions that decide a hook climb are
# neighbouring integers:
#
#     SOLID   = 1    blocking, hook ATTACHES
#     NOHOOK  = 3    blocking, hook does NOT attach
#     THROUGH = 6    hook PASSES THROUGH
#
# Three semantic opposites within five of each other, with NOLASER and
# THROUGH_CUT as distractors in between. Hence this table, which is FIXED rather
# than learned: no parameters, no gradient, no exploration cost, only a
# reindexing of information the observation already carries.
#
# ppo_hybrid.build_semantic_table() turns it into the (256, 12) one-hot the
# policy uses, and visualize_obs.py colours tiles by it. One table, so the
# renderer and the network cannot disagree about what a tile means.


class TileClass(IntEnum):
    """Semantic class of a raw DDNet tile id."""
    AIR = 0
    SOLID = 1
    NOHOOK = 2
    THROUGH = 3
    DEATH = 4
    FREEZE = 5
    UNFREEZE = 6
    TELE = 7
    BOOST = 8
    JUMP = 9
    STOPPER = 10
    OTHER = 11


SEMANTIC_CHANNELS = len(TileClass)

TILE_SEMANTICS = {
    0: TileClass.AIR,
    1: TileClass.SOLID,                     # hookable wall
    3: TileClass.NOHOOK,                    # wall the hook slides off
    5: TileClass.THROUGH, 6: TileClass.THROUGH,    # THROUGH_CUT, THROUGH
    66: TileClass.THROUGH, 67: TileClass.THROUGH,  # THROUGH_ALL, THROUGH_DIR
    2: TileClass.DEATH,
    9: TileClass.FREEZE, 12: TileClass.FREEZE, 144: TileClass.FREEZE,
    11: TileClass.UNFREEZE, 13: TileClass.UNFREEZE, 145: TileClass.UNFREEZE,
    10: TileClass.TELE, 26: TileClass.TELE, 27: TileClass.TELE,
    30: TileClass.TELE, 31: TileClass.TELE, 63: TileClass.TELE,
    28: TileClass.BOOST, 29: TileClass.BOOST,
    7: TileClass.JUMP, 16: TileClass.JUMP,  # JUMP, WALLJUMP
    60: TileClass.STOPPER, 61: TileClass.STOPPER, 62: TileClass.STOPPER,
}


def semantic_lookup() -> np.ndarray:
    """(256,) uint8 table mapping raw tile id -> TileClass. Unmapped -> OTHER."""
    table = np.full(256, int(TileClass.OTHER), dtype=np.uint8)
    for tile_id, cls in TILE_SEMANTICS.items():
        table[tile_id] = int(cls)
    return table


_SEMANTIC_LOOKUP = semantic_lookup()


def classify_tiles(tile_grid: np.ndarray) -> np.ndarray:
    """Reindex an array of raw tile ids into TileClass values, shape preserved."""
    return _SEMANTIC_LOOKUP[tile_grid]


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


@dataclass
class PlayerObs:
    """``SPlayerObs`` -- one tee's state."""

    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    direction: int
    jumped: int
    hook_state: int
    weapon: int
    hook_pos_x: float
    hook_pos_y: float
    hooked_player: int
    active: bool
    is_frozen: bool
    is_local: bool
    health: int
    armor: int

    STRUCT_FORMAT = '<4fb3B2fb5B2x'
    STRUCT_SIZE = 36

    @classmethod
    def from_bytes(cls, data: bytes) -> 'PlayerObs':
        (pos_x, pos_y, vel_x, vel_y,
         direction, jumped, hook_state, weapon,
         hook_pos_x, hook_pos_y,
         hooked_player, active, is_frozen, is_local,
         health, armor) = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(
            pos_x=pos_x, pos_y=pos_y, vel_x=vel_x, vel_y=vel_y,
            direction=direction, jumped=jumped, hook_state=hook_state,
            weapon=weapon, hook_pos_x=hook_pos_x, hook_pos_y=hook_pos_y,
            hooked_player=hooked_player, active=bool(active),
            is_frozen=bool(is_frozen), is_local=bool(is_local),
            health=health, armor=armor,
        )

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vel_x, self.vel_y))

    @property
    def hook_name(self) -> str:
        return {
            HOOK_IDLE: 'idle',
            HOOK_RETRACT_START: 'retract_start',
            HOOK_RETRACT_END: 'retract_end',
            HOOK_FLYING: 'flying',
            HOOK_GRABBED: 'grabbed',
            HOOK_RETRACTED: 'retracted',
        }.get(self.hook_state, f'?{self.hook_state}')


assert struct.calcsize(PlayerObs.STRUCT_FORMAT) == PlayerObs.STRUCT_SIZE


# Field offsets within SStructuredObservation. The struct is packed, so these are
# exact byte positions and the floats are NOT 4-byte aligned -- copy before
# viewing a slice as float32.
OFF_TICK = 0
OFF_GAME_FLAGS = 4
OFF_GAME_STATE = 5
OFF_LOCAL_PLAYER = 8
OFF_TILE_GRID = OFF_LOCAL_PLAYER + PlayerObs.STRUCT_SIZE       # 44
OFF_IS_GROUNDED = OFF_TILE_GRID + TILE_GRID_SIZE               # 1377
OFF_JUMPS_REMAINING = OFF_IS_GROUNDED + 1
OFF_CAN_HOOK = OFF_IS_GROUNDED + 2
OFF_PROGRESS = OFF_IS_GROUNDED + 4                             # 1381
OFF_LOOKAHEAD_X = OFF_PROGRESS + 4                             # 1385
OFF_LOOKAHEAD_Y = OFF_LOOKAHEAD_X + 4 * NUM_LOOKAHEAD          # 1417
OFF_MAX_PROGRESS = OFF_LOOKAHEAD_Y + 4 * NUM_LOOKAHEAD         # 1449
OFF_HOOK_REACH = OFF_MAX_PROGRESS + 4                          # 1453
STRUCTURED_OBS_SIZE = OFF_HOOK_REACH + 4 * NUM_AIM_SECTORS + 3  # 1520


@dataclass
class StructuredObservation:
    """``SStructuredObservation`` -- what one bot sees on one decision tick."""

    tick: int
    game_flags: int
    game_state: int
    local_player: PlayerObs

    # Raw DDNet tile ids, clamped to uint8, centred on the tee. Out of bounds
    # reads back as SOLID so the edge of the map is a wall, not air.
    tile_grid: np.ndarray  # (TILE_GRID_HEIGHT, TILE_GRID_WIDTH) uint8

    is_grounded: bool
    jumps_remaining: int
    can_hook: bool

    # Navigation. Replaces the old time-checkpoint block: checkpoint placement,
    # numbering and even existence are mapper convention, while arc-length along
    # a reference path we generate ourselves is defined on every race map.
    progress: float            # normalised arc-length in [0, 1]
    lookahead: np.ndarray      # (NUM_LOOKAHEAD, 2) egocentric world offsets
    max_progress: float        # episode high-water progress

    # Per-sector hook reachability -- the red/green aim line DDNet draws for
    # human players, marched server-side by the same IntersectLineTeleHook the
    # hook itself uses. 0 means a hook fired that way would not grab; otherwise
    # in (0, 1], larger for a closer grab point.
    hook_reach: np.ndarray     # (NUM_AIM_SECTORS,) float32

    STRUCT_SIZE = STRUCTURED_OBS_SIZE

    @classmethod
    def from_bytes(cls, data: bytes) -> 'StructuredObservation':
        if len(data) < cls.STRUCT_SIZE:
            raise ValueError(
                f'observation is {len(data)} bytes, expected {cls.STRUCT_SIZE} '
                '-- Python and rl_protocol.h disagree about the layout')

        tick, game_flags, game_state = struct.unpack_from('<IBB', data, OFF_TICK)
        local_player = PlayerObs.from_bytes(
            data[OFF_LOCAL_PLAYER:OFF_LOCAL_PLAYER + PlayerObs.STRUCT_SIZE])

        tile_grid = np.frombuffer(
            data, dtype=np.uint8, count=TILE_GRID_SIZE, offset=OFF_TILE_GRID
        ).reshape(TILE_GRID_HEIGHT, TILE_GRID_WIDTH).copy()

        is_grounded, jumps_remaining, can_hook = struct.unpack_from(
            '<3B', data, OFF_IS_GROUNDED)

        # The struct is packed, so none of these floats are 4-byte aligned.
        # Slicing first hands frombuffer a freshly allocated (aligned) buffer
        # rather than relying on unaligned reads being tolerated.
        progress, = struct.unpack_from('<f', data, OFF_PROGRESS)
        lookahead_x = np.frombuffer(
            data[OFF_LOOKAHEAD_X:OFF_LOOKAHEAD_X + 4 * NUM_LOOKAHEAD],
            dtype=np.float32)
        lookahead_y = np.frombuffer(
            data[OFF_LOOKAHEAD_Y:OFF_LOOKAHEAD_Y + 4 * NUM_LOOKAHEAD],
            dtype=np.float32)
        max_progress, = struct.unpack_from('<f', data, OFF_MAX_PROGRESS)
        hook_reach = np.frombuffer(
            data[OFF_HOOK_REACH:OFF_HOOK_REACH + 4 * NUM_AIM_SECTORS],
            dtype=np.float32)

        return cls(
            tick=tick,
            game_flags=game_flags,
            game_state=game_state,
            local_player=local_player,
            tile_grid=tile_grid,
            is_grounded=bool(is_grounded),
            jumps_remaining=jumps_remaining,
            can_hook=bool(can_hook),
            progress=progress,
            lookahead=np.stack([lookahead_x, lookahead_y], axis=1).copy(),
            max_progress=max_progress,
            hook_reach=hook_reach.copy(),
        )

    @property
    def semantic_grid(self) -> np.ndarray:
        """The tile grid reindexed into TileClass values."""
        return classify_tiles(self.tile_grid)

    def tile_at(self, dx: int, dy: int) -> int:
        """Raw tile id at an offset in tiles from the tee, (0, 0) being its own."""
        return int(self.tile_grid[TILE_GRID_HEIGHT // 2 + dy,
                                  TILE_GRID_WIDTH // 2 + dx])

    def hook_reach_distance(self, sector: int) -> float:
        """World distance to the grab point in a sector, or inf if it would miss.

        Inverts the server's encoding, max(0.05, 1 - d / HOOK_LENGTH), so it is
        only exact under default hook tuning.
        """
        reach = float(self.hook_reach[sector])
        if reach <= 0.0:
            return float('inf')
        return (1.0 - reach) * HOOK_LENGTH


def aim_sector_direction(sector: int) -> Tuple[float, float]:
    """Unit vector at the centre of an aim sector.

    Must match FillHookReach in botmanager.cpp and the sector decoding in
    write_actions_batch: centre = (k + 0.5) * 2pi/N - pi. Sector 0 therefore
    points up-left, not right.
    """
    angle = (sector + 0.5) * (2.0 * np.pi / NUM_AIM_SECTORS) - np.pi
    return float(np.cos(angle)), float(np.sin(angle))
