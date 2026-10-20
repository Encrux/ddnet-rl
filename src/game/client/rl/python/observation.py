#!/usr/bin/env python3
"""
Shared observation structs for the DDNet RL interface.

These mirror the C++ structs in src/game/rl/rl_protocol.h and parse the
observation bytes delivered over shared memory. They are wire-format
definitions only — no transport, no environment logic — so both the
server-side env and the offline tooling can depend on them.

Keep the constants and struct layouts in sync with the C++ side; a mismatch
shows up as silently misparsed observations rather than an error.
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np


# Constants (must match C++ header)
MAX_OTHER_PLAYERS = 15
MAX_PROJECTILES = 32
MAX_PICKUPS = 16
TILE_GRID_WIDTH = 43   # 21 left + 1 center + 21 right (1.5x original)
TILE_GRID_HEIGHT = 31  # 15 up + 1 center + 15 down (1.5x original)
TILE_GRID_SIZE = 1333  # 43 * 31
NUM_RAYCASTS = 16
STRUCTURED_OBS_SIZE = 3020  # Updated for larger tile grid + checkpoint info


class TileType(IntEnum):
    """Tile types matching the C++ ETileType enum."""
    EMPTY = 0
    SOLID = 1
    DEATH = 2
    NOHOOK = 3
    FREEZE = 4
    UNFREEZE = 5
    TELEPORT = 6
    BOOST = 7
    STOPPER = 8
    SWITCH = 9
    JUMP = 10
    THROUGH = 11
    SPAWN = 12
    FINISH = 13
    CHECKPOINT = 14
    OTHER = 15



@dataclass
class PlayerObs:
    """Structured observation for a player (36 bytes)"""
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
    
    STRUCT_FORMAT = '<ffffbBBBffbBBBBB2x'  # 36 bytes
    STRUCT_SIZE = 36
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'PlayerObs':
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data[:cls.STRUCT_SIZE])
        return cls(
            pos_x=unpacked[0], pos_y=unpacked[1], vel_x=unpacked[2], vel_y=unpacked[3],
            direction=unpacked[4], jumped=unpacked[5], hook_state=unpacked[6], weapon=unpacked[7],
            hook_pos_x=unpacked[8], hook_pos_y=unpacked[9], hooked_player=unpacked[10],
            active=bool(unpacked[11]), is_frozen=bool(unpacked[12]), is_local=bool(unpacked[13]),
            health=unpacked[14], armor=unpacked[15]
        )


@dataclass
class ProjectileObs:
    """Structured observation for a projectile (20 bytes)"""
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    type: int
    active: bool
    
    STRUCT_FORMAT = '<ffffBB2x'  # 20 bytes
    STRUCT_SIZE = 20
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ProjectileObs':
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data[:cls.STRUCT_SIZE])
        return cls(
            pos_x=unpacked[0], pos_y=unpacked[1], vel_x=unpacked[2], vel_y=unpacked[3],
            type=unpacked[4], active=bool(unpacked[5])
        )


@dataclass
class StructuredObservation:
    """Full structured observation (3020 bytes)"""
    tick: int
    game_flags: int
    game_state: int
    num_active_players: int
    num_projectiles: int
    local_player: PlayerObs
    other_players: List[PlayerObs]
    projectiles: List[ProjectileObs]
    flag_pos: np.ndarray  # [2, 2] - [team][x/y]
    flag_carrier: Tuple[int, int]
    flag_at_stand: Tuple[bool, bool]
    tile_grid: np.ndarray  # [31, 43] - local tile grid centered on player (1.5x original)
    # New raycast observations
    raycast_solid: np.ndarray    # [16] - distance to solid in 16 directions
    raycast_death: np.ndarray    # [16] - distance to death tiles
    raycast_hookable: np.ndarray # [16] - distance to hookable surfaces
    # Extra player state
    is_grounded: bool
    jumps_remaining: int
    can_hook: bool
    # Checkpoint navigation
    next_checkpoint_x: float = 0.0
    next_checkpoint_y: float = 0.0
    checkpoint_index: int = -1
    total_checkpoints: int = 0
    
    def to_flat_array(self) -> np.ndarray:
        """Convert to a flat numpy array for RL training.

        Layout:
        - Player state: 15 features
        - Checkpoint info: 4 features
        - Raycasts: 48 features (16 dirs x 3 types)
        - Local tiles (7x7): 49 features
        - Other players: 25 features (5 players x 5 features)
        Total: ~141 features
        """
        features = []

        # Local player (normalized) - 15 features
        p = self.local_player
        features.extend([
            p.vel_x / 50.0,                    # Normalized velocity
            p.vel_y / 50.0,
            float(self.is_grounded),          # Binary
            float(self.jumps_remaining) / 2.0, # 0, 0.5, or 1
            float(self.can_hook),             # Binary
            p.direction,                      # -1, 0, 1
            float(p.is_frozen),               # Binary
            # Hook state one-hot (idle, flying, attached)
            float(p.hook_state == 0),
            float(p.hook_state == 1),
            float(p.hook_state == 2),
            # Aim direction (sin/cos for continuity) - computed from hook pos if hooked
            0.0, 0.0,  # Placeholder - will be set by Python training code
            p.health / 10.0,
            p.armor / 10.0,
            float(p.jumped) / 3.0,            # Jump state normalized
        ])

        # Checkpoint navigation - 4 features (direction-agnostic, signed)
        # Direction to next checkpoint (normalized, can be any direction)
        cp_dx = self.next_checkpoint_x - p.pos_x
        cp_dy = self.next_checkpoint_y - p.pos_y
        max_dist = 2000.0  # Normalization factor
        features.extend([
            np.clip(cp_dx / max_dist, -1.0, 1.0),  # Signed direction X
            np.clip(cp_dy / max_dist, -1.0, 1.0),  # Signed direction Y
            float(self.checkpoint_index) / max(1, self.total_checkpoints),  # Progress
            float(self.total_checkpoints) / 20.0,  # Total checkpoints (normalized)
        ])

        # Raycast observations - 48 features (already normalized 0-1)
        features.extend(self.raycast_solid.tolist())
        features.extend(self.raycast_death.tolist())
        features.extend(self.raycast_hookable.tolist())

        # Local tiles (7x7 around player) - 49 features
        # One-hot encode into 4 key types: empty, solid, death, other
        # For now, just use normalized tile values for the small grid
        center_x = TILE_GRID_WIDTH // 2
        center_y = TILE_GRID_HEIGHT // 2
        small_grid = self.tile_grid[center_y-3:center_y+4, center_x-3:center_x+4]
        tile_flat = small_grid.flatten().astype(np.float32) / 15.0
        features.extend(tile_flat.tolist())

        # Other players (top 5 closest) - 25 features
        for i in range(min(5, len(self.other_players))):
            op = self.other_players[i]
            if op.active:
                features.extend([
                    (op.pos_x - p.pos_x) / 1000.0,  # Relative position
                    (op.pos_y - p.pos_y) / 1000.0,
                    op.vel_x / 50.0, op.vel_y / 50.0,
                    op.hook_state / 2.0
                ])
            else:
                features.extend([0, 0, 0, 0, 0])

        # Pad to 5 players if fewer
        for _ in range(5 - min(5, len(self.other_players))):
            features.extend([0, 0, 0, 0, 0])

        return np.array(features, dtype=np.float32)

    def to_dict_observation(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary observation for CNN-based policy.

        Returns:
            Dict with:
                'tiles': (1, 31, 43) - Full tile grid for CNN
                'player_state': (15,) - Player state vector
                'checkpoint': (4,) - Checkpoint navigation info
        """
        p = self.local_player

        # Player state - 15 features
        player_state = np.array([
            p.vel_x / 50.0,
            p.vel_y / 50.0,
            float(self.is_grounded),
            float(self.jumps_remaining) / 2.0,
            float(self.can_hook),
            p.direction,
            float(p.is_frozen),
            float(p.hook_state == 0),
            float(p.hook_state == 1),
            float(p.hook_state == 2),
            0.0, 0.0,  # Aim placeholder
            p.health / 10.0,
            p.armor / 10.0,
            float(p.jumped) / 3.0,
        ], dtype=np.float32)

        # Checkpoint info - 4 features
        cp_dx = self.next_checkpoint_x - p.pos_x
        cp_dy = self.next_checkpoint_y - p.pos_y
        max_dist = 2000.0
        checkpoint = np.array([
            np.clip(cp_dx / max_dist, -1.0, 1.0),
            np.clip(cp_dy / max_dist, -1.0, 1.0),
            float(self.checkpoint_index) / max(1, self.total_checkpoints),
            float(self.total_checkpoints) / 20.0,
        ], dtype=np.float32)

        # Full tile grid normalized (1, H, W) for CNN
        tiles = self.tile_grid.astype(np.float32) / 15.0
        tiles = tiles.reshape(1, TILE_GRID_HEIGHT, TILE_GRID_WIDTH)

        return {
            'tiles': tiles,
            'player_state': player_state,
            'checkpoint': checkpoint,
        }

    def to_flat_array_legacy(self) -> np.ndarray:
        """Legacy flat array for backwards compatibility (~700 features)."""
        features = []

        # Game state
        features.extend([self.tick / 1000.0, self.game_flags, self.game_state])

        # Local player (normalized)
        p = self.local_player
        features.extend([
            p.pos_x / 10000.0, p.pos_y / 10000.0,
            p.vel_x / 100.0, p.vel_y / 100.0,
            p.direction, p.jumped, p.hook_state, p.weapon,
            p.hook_pos_x / 10000.0, p.hook_pos_y / 10000.0,
            p.hooked_player, int(p.is_frozen),
            p.health / 10.0, p.armor / 10.0
        ])

        # Other players (top 5 closest)
        for i in range(min(5, len(self.other_players))):
            op = self.other_players[i]
            if op.active:
                features.extend([
                    (op.pos_x - p.pos_x) / 1000.0,
                    (op.pos_y - p.pos_y) / 1000.0,
                    op.vel_x / 100.0, op.vel_y / 100.0,
                    op.hook_state
                ])
            else:
                features.extend([0, 0, 0, 0, 0])

        # Full tile grid
        tile_flat = self.tile_grid.flatten().astype(np.float32) / 15.0

        return np.concatenate([np.array(features, dtype=np.float32), tile_flat])
    
    def get_tile_at_relative(self, dx: int, dy: int) -> int:
        """
        Get the tile type at a position relative to the player.
        
        Args:
            dx: Horizontal offset (-14 to +14)
            dy: Vertical offset (-10 to +10)
            
        Returns:
            Tile type (0-15) or -1 if out of bounds
        """
        grid_x = dx + 14
        grid_y = dy + 10
        if 0 <= grid_x < TILE_GRID_WIDTH and 0 <= grid_y < TILE_GRID_HEIGHT:
            return int(self.tile_grid[grid_y, grid_x])
        return -1
    
    def get_nearby_tiles(self, radius: int = 3) -> np.ndarray:
        """
        Get tiles in a smaller square around the player.
        
        Args:
            radius: Number of tiles in each direction (default 3 = 7x7 grid)
            
        Returns:
            (2*radius+1, 2*radius+1) numpy array of tile types
        """
        center_x = 14  # Player is at center
        center_y = 10
        start_x = center_x - radius
        end_x = center_x + radius + 1
        start_y = center_y - radius
        end_y = center_y + radius + 1
        return self.tile_grid[start_y:end_y, start_x:end_x].copy()
    
    @staticmethod
    def fast_to_flat_array(data: bytes) -> np.ndarray:
        """Parse bytes directly to flat array, skipping intermediate objects.

        This is ~10x faster than from_bytes() + to_flat_array() because it
        avoids creating PlayerObs, ProjectileObs, and StructuredObservation objects.
        """
        # Pre-allocate output array
        # Layout: 15 player + 4 checkpoint + 48 raycasts + 49 tiles + 25 other_players = 141
        features = np.zeros(141, dtype=np.float32)

        # Offsets matching C++ SStructuredObservation (rl_protocol.h)
        # SPlayerObs is 36 bytes (not 64!)
        HEADER_SIZE = 8
        PLAYER_SIZE = 36
        LOCAL_PLAYER_OFFSET = HEADER_SIZE  # 8
        OTHER_PLAYERS_OFFSET = LOCAL_PLAYER_OFFSET + PLAYER_SIZE  # 44
        PROJECTILES_OFFSET = OTHER_PLAYERS_OFFSET + 15 * PLAYER_SIZE  # 584
        PICKUPS_OFFSET = PROJECTILES_OFFSET + 32 * 20  # 1224
        FLAGS_OFFSET = PICKUPS_OFFSET + 16 * 12  # 1416
        TILES_OFFSET = FLAGS_OFFSET + 36  # 1452 (flags=24 + padding=12)
        RAYCASTS_OFFSET = TILES_OFFSET + TILE_GRID_SIZE  # 2785
        EXTRA_STATE_OFFSET = RAYCASTS_OFFSET + 192  # 2977
        CHECKPOINT_OFFSET = EXTRA_STATE_OFFSET + 4  # 2981

        # Local player (36 bytes): parse matching SPlayerObs struct
        # Format: pos_x(0), pos_y(1), vel_x(2), vel_y(3), direction(4), jumped(5), hook_state(6), weapon(7),
        #         hook_pos_x(8), hook_pos_y(9), hooked_player(10), active(11), is_frozen(12), is_local(13),
        #         health(14), armor(15), padding
        local = struct.unpack_from('<ffffbBBBffbBBBBBxx', data, LOCAL_PLAYER_OFFSET)
        pos_x, pos_y = local[0], local[1]
        vel_x, vel_y = local[2], local[3]
        direction = local[4]
        jumped = local[5]
        hook_state = local[6]
        is_frozen = local[12]
        health = local[14]
        armor = local[15]

        # Extra state (4 bytes)
        extra = struct.unpack_from('<BBBB', data, EXTRA_STATE_OFFSET)
        is_grounded, jumps_remaining, can_hook = extra[0], extra[1], extra[2]

        # Checkpoint info (16 bytes)
        cp_data = struct.unpack_from('<ffii', data, CHECKPOINT_OFFSET)
        next_cp_x, next_cp_y, cp_index, total_cps = cp_data

        # Fill features array
        idx = 0

        # Local player (15 features)
        features[idx:idx+15] = [
            vel_x / 50.0,
            vel_y / 50.0,
            float(is_grounded),
            float(jumps_remaining) / 2.0,
            float(can_hook),
            float(direction),
            float(is_frozen),
            float(hook_state == 0),
            float(hook_state == 1),
            float(hook_state == 2),
            0.0, 0.0,  # Aim placeholder
            health / 10.0,
            armor / 10.0,
            float(jumped) / 3.0,
        ]
        idx += 15

        # Checkpoint (4 features)
        cp_dx = next_cp_x - pos_x
        cp_dy = next_cp_y - pos_y
        max_dist = 2000.0
        features[idx:idx+4] = [
            np.clip(cp_dx / max_dist, -1.0, 1.0),
            np.clip(cp_dy / max_dist, -1.0, 1.0),
            float(cp_index) / max(1, total_cps),
            float(total_cps) / 20.0,
        ]
        idx += 4

        # Raycasts (48 features) - read directly from bytes
        raycasts = np.frombuffer(data[RAYCASTS_OFFSET:RAYCASTS_OFFSET + 192], dtype=np.float32)
        features[idx:idx+48] = raycasts
        idx += 48

        # Tiles (49 features) - 7x7 center grid
        tile_grid_flat = np.frombuffer(data[TILES_OFFSET:TILES_OFFSET + TILE_GRID_SIZE], dtype=np.uint8)
        tile_grid = tile_grid_flat.reshape((TILE_GRID_HEIGHT, TILE_GRID_WIDTH))
        center_x = TILE_GRID_WIDTH // 2
        center_y = TILE_GRID_HEIGHT // 2
        small_grid = tile_grid[center_y-3:center_y+4, center_x-3:center_x+4]
        features[idx:idx+49] = small_grid.flatten().astype(np.float32) / 15.0
        idx += 49

        # Other players (25 features) - top 5
        for i in range(5):
            other_offset = OTHER_PLAYERS_OFFSET + i * PLAYER_SIZE
            other = struct.unpack_from('<ffffbBBBffbBBBBBxx', data, other_offset)
            other_active = other[11]  # active is at index 11
            if other_active:
                features[idx:idx+5] = [
                    (other[0] - pos_x) / 1000.0,  # rel pos_x
                    (other[1] - pos_y) / 1000.0,  # rel pos_y
                    other[2] / 50.0,  # vel_x
                    other[3] / 50.0,  # vel_y
                    other[6] / 2.0,   # hook_state
                ]
            # else: stays zeros
            idx += 5

        # Validate - replace NaN/Inf with zeros
        if not np.all(np.isfinite(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    @classmethod
    def from_bytes(cls, data: bytes) -> 'StructuredObservation':
        """Parse structured observation from raw bytes."""
        offset = 0

        # Game state header (8 bytes)
        tick, game_flags, game_state, num_active, num_proj = struct.unpack_from('<IBBBB', data, offset)
        offset += 8

        # Local player (36 bytes)
        local_player = PlayerObs.from_bytes(data[offset:offset + PlayerObs.STRUCT_SIZE])
        offset += PlayerObs.STRUCT_SIZE

        # Other players (36 * 15 = 540 bytes)
        other_players = []
        for _ in range(MAX_OTHER_PLAYERS):
            other_players.append(PlayerObs.from_bytes(data[offset:offset + PlayerObs.STRUCT_SIZE]))
            offset += PlayerObs.STRUCT_SIZE

        # Projectiles (20 * 32 = 640 bytes)
        projectiles = []
        for _ in range(MAX_PROJECTILES):
            projectiles.append(ProjectileObs.from_bytes(data[offset:offset + ProjectileObs.STRUCT_SIZE]))
            offset += ProjectileObs.STRUCT_SIZE

        # Pickups (12 * 16 = 192 bytes) - skip for now
        offset += 12 * MAX_PICKUPS

        # Flags (24 bytes)
        flag_pos = np.zeros((2, 2), dtype=np.float32)
        flag_pos[0, 0], flag_pos[0, 1] = struct.unpack_from('<ff', data, offset)
        offset += 8
        flag_pos[1, 0], flag_pos[1, 1] = struct.unpack_from('<ff', data, offset)
        offset += 8
        flag_carrier = struct.unpack_from('<bb', data, offset)
        offset += 2
        flag_at_stand_raw = struct.unpack_from('<BB', data, offset)
        flag_at_stand = (bool(flag_at_stand_raw[0]), bool(flag_at_stand_raw[1]))
        offset += 2
        offset += 12  # Padding

        # Tile grid (609 bytes) - 29x21 grid centered on player, row-major
        tile_grid_flat = np.frombuffer(data[offset:offset + TILE_GRID_SIZE], dtype=np.uint8)
        tile_grid = tile_grid_flat.reshape((TILE_GRID_HEIGHT, TILE_GRID_WIDTH))
        offset += TILE_GRID_SIZE

        # Raycast observations (16 * 4 * 3 = 192 bytes)
        raycast_solid = np.frombuffer(data[offset:offset + NUM_RAYCASTS * 4], dtype=np.float32).copy()
        offset += NUM_RAYCASTS * 4
        raycast_death = np.frombuffer(data[offset:offset + NUM_RAYCASTS * 4], dtype=np.float32).copy()
        offset += NUM_RAYCASTS * 4
        raycast_hookable = np.frombuffer(data[offset:offset + NUM_RAYCASTS * 4], dtype=np.float32).copy()
        offset += NUM_RAYCASTS * 4

        # Extra player state (4 bytes)
        is_grounded, jumps_remaining, can_hook, _ = struct.unpack_from('<BBBB', data, offset)
        offset += 4

        # Checkpoint info (16 bytes)
        next_cp_x, next_cp_y, cp_index, total_cps = struct.unpack_from('<ffii', data, offset)
        offset += 16

        # Reserved (11 bytes) - skip
        # offset += 11

        return cls(
            tick=tick,
            game_flags=game_flags,
            game_state=game_state,
            num_active_players=num_active,
            num_projectiles=num_proj,
            local_player=local_player,
            other_players=other_players,
            projectiles=projectiles,
            flag_pos=flag_pos,
            flag_carrier=flag_carrier,
            flag_at_stand=flag_at_stand,
            tile_grid=tile_grid.copy(),
            raycast_solid=raycast_solid,
            raycast_death=raycast_death,
            raycast_hookable=raycast_hookable,
            is_grounded=bool(is_grounded),
            jumps_remaining=jumps_remaining,
            can_hook=bool(can_hook),
            next_checkpoint_x=next_cp_x,
            next_checkpoint_y=next_cp_y,
            checkpoint_index=cp_index,
            total_checkpoints=total_cps,
        )
