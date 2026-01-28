"""
Teeworlds RL Environment - Shared Memory Client

This module provides a Python interface to receive game state/frames
from the DDNet client via shared memory.

Supports two observation modes:
- STRUCTURED (default): Game state vectors (headless compatible)
- RENDERED: RGB pixel frames (requires display or Xvfb)

Usage:
    python teeworlds_env.py [--env-id 0] [--display] [--stream PORT]
"""

import mmap
import struct
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum
import argparse
import os


# Observation modes (must match C++ enum)
class ObservationMode:
    RENDERED = 0
    STRUCTURED = 1


# Constants (must match C++ header)
MAX_OTHER_PLAYERS = 15
MAX_PROJECTILES = 32
MAX_PICKUPS = 16
TILE_GRID_WIDTH = 29
TILE_GRID_HEIGHT = 21
TILE_GRID_SIZE = 609  # 29 * 21
STRUCTURED_OBS_SIZE = 2088


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
class SharedMemoryHeader:
    """Matches the C++ SSharedMemoryHeader struct (64 bytes)"""
    # Sync flags
    client_ready: int       # uint8
    python_ready: int       # uint8
    shutdown: int           # uint8
    observation_mode: int   # uint8 (0=rendered, 1=structured)
    
    # Frame info
    tick: int               # uint32
    frame_width: int        # uint32
    frame_height: int       # uint32
    frame_channels: int     # uint32
    
    # RL signals
    reward: float           # float32
    done: int               # uint8
    truncated: int          # uint8
    _padding2: bytes        # uint8[2]
    
    # Action
    action_direction: int   # int8
    action_jump: int        # uint8
    action_hook: int        # uint8
    action_fire: int        # uint8
    action_aim_x: int       # int16
    action_aim_y: int       # int16
    action_weapon: int      # uint8
    _padding3: bytes        # uint8[3]
    
    # Reserved
    _reserved: bytes        # uint8[24]
    
    STRUCT_FORMAT = '<BBBBIIIIfBB2sbBBBhhB3s24s'
    STRUCT_SIZE = 64
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SharedMemoryHeader':
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data[:cls.STRUCT_SIZE])
        return cls(*unpacked)
    
    def to_bytes(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            self.client_ready, self.python_ready, self.shutdown, self.observation_mode,
            self.tick, self.frame_width, self.frame_height, self.frame_channels,
            self.reward, self.done, self.truncated, self._padding2,
            self.action_direction, self.action_jump, self.action_hook, self.action_fire,
            self.action_aim_x, self.action_aim_y, self.action_weapon, self._padding3,
            self._reserved
        )


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
    """Full structured observation (2088 bytes)"""
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
    tile_grid: np.ndarray  # [21, 29] - local tile grid centered on player
    
    def to_flat_array(self) -> np.ndarray:
        """Convert to a flat numpy array for RL training."""
        features = []
        
        # Game state
        features.extend([self.tick / 1000.0, self.game_flags, self.game_state])
        
        # Local player (normalized)
        p = self.local_player
        features.extend([
            p.pos_x / 10000.0, p.pos_y / 10000.0,  # Normalize position
            p.vel_x / 100.0, p.vel_y / 100.0,      # Normalize velocity
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
                    (op.pos_x - p.pos_x) / 1000.0,  # Relative position
                    (op.pos_y - p.pos_y) / 1000.0,
                    op.vel_x / 100.0, op.vel_y / 100.0,
                    op.hook_state
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        # Tile grid - flatten and normalize to [0, 1] range
        # Tile types are 0-15, so divide by 15
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
            tile_grid=tile_grid.copy()  # Copy to avoid reference to mmap buffer
        )


class TeeWorldsShmClient:
    """
    Shared memory client to receive game state/frames from DDNet.
    
    Supports two observation modes:
    - STRUCTURED: Game state vectors (headless compatible, faster)
    - RENDERED: RGB pixel frames (requires display or Xvfb)
    
    Example (structured mode - default):
        client = TeeWorldsShmClient(env_id=0)
        while True:
            obs, info = client.wait_for_observation()
            if obs is None:
                break
            # obs is a StructuredObservation object
            features = obs.to_flat_array()  # Convert to numpy for RL
            client.send_action(direction=1, jump=0)
    
    Example (rendered mode):
        client = TeeWorldsShmClient(env_id=0)
        while True:
            frame, info = client.wait_for_frame()
            if frame is None:
                break
            # frame is a numpy array (H, W, C)
            client.send_action(direction=1, jump=0)
    """
    
    def __init__(self, env_id: int = 0, timeout: float = 5.0):
        """
        Initialize the shared memory client.
        
        Args:
            env_id: Environment ID (matches rl_enable command in game)
            timeout: Timeout in seconds for waiting for observations
        """
        self.env_id = env_id
        self.timeout = timeout
        self.shm_path = f"/dev/shm/teeworlds_rl_{env_id}"
        self.shm_file = None
        self.shm_mmap = None
        self.header_size = SharedMemoryHeader.STRUCT_SIZE
        
        self.observation_mode = ObservationMode.STRUCTURED
        self.frame_width = 0
        self.frame_height = 0
        self.frame_channels = 0
        self.data_size = 0
        
        self._connected = False
    
    def connect(self, max_retries: int = 10, retry_delay: float = 0.5) -> bool:
        """
        Connect to the shared memory created by DDNet.
        
        Args:
            max_retries: Maximum connection attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if connected successfully
        """
        for attempt in range(max_retries):
            if os.path.exists(self.shm_path):
                try:
                    self.shm_file = open(self.shm_path, 'r+b')
                    
                    # Read header to get observation mode and dimensions
                    header_data = self.shm_file.read(self.header_size)
                    header = SharedMemoryHeader.from_bytes(header_data)
                    
                    self.observation_mode = header.observation_mode
                    self.frame_width = header.frame_width
                    self.frame_height = header.frame_height
                    self.frame_channels = header.frame_channels
                    
                    # Calculate data buffer size based on mode
                    if self.observation_mode == ObservationMode.STRUCTURED:
                        self.data_size = STRUCTURED_OBS_SIZE
                    else:
                        self.data_size = self.frame_width * self.frame_height * self.frame_channels
                    
                    # Map the full shared memory
                    total_size = self.header_size + self.data_size
                    self.shm_file.seek(0)
                    self.shm_mmap = mmap.mmap(self.shm_file.fileno(), total_size)
                    
                    self._connected = True
                    mode_name = "structured" if self.observation_mode == ObservationMode.STRUCTURED else "rendered"
                    print(f"Connected to {self.shm_path}")
                    print(f"Observation mode: {mode_name}")
                    if self.observation_mode == ObservationMode.RENDERED:
                        print(f"Frame size: {self.frame_width}x{self.frame_height}x{self.frame_channels}")
                    return True
                    
                except Exception as e:
                    print(f"Connection attempt {attempt + 1} failed: {e}")
                    if self.shm_file:
                        self.shm_file.close()
                        self.shm_file = None
            
            print(f"Waiting for DDNet to create shared memory... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        
        print(f"Failed to connect to {self.shm_path}")
        return False
    
    def disconnect(self):
        """Disconnect from shared memory."""
        if self.shm_mmap:
            # Signal shutdown
            self._write_header_field(2, 1, 'B')  # shutdown = 1
            self.shm_mmap.close()
            self.shm_mmap = None
        if self.shm_file:
            self.shm_file.close()
            self.shm_file = None
        self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected
    
    def is_structured_mode(self) -> bool:
        return self.observation_mode == ObservationMode.STRUCTURED
    
    def _read_header(self) -> SharedMemoryHeader:
        """Read the header from shared memory."""
        self.shm_mmap.seek(0)
        header_data = self.shm_mmap.read(self.header_size)
        return SharedMemoryHeader.from_bytes(header_data)
    
    def _write_header_field(self, offset: int, value: int, fmt: str = 'B'):
        """Write a single field to the header."""
        self.shm_mmap.seek(offset)
        self.shm_mmap.write(struct.pack(fmt, value))
    
    def _read_frame(self) -> np.ndarray:
        """Read the frame data from shared memory (rendered mode only)."""
        self.shm_mmap.seek(self.header_size)
        frame_data = self.shm_mmap.read(self.data_size)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        return frame.reshape((self.frame_height, self.frame_width, self.frame_channels))
    
    def _read_structured_observation(self) -> StructuredObservation:
        """Read structured observation from shared memory."""
        self.shm_mmap.seek(self.header_size)
        obs_data = self.shm_mmap.read(STRUCTURED_OBS_SIZE)
        return StructuredObservation.from_bytes(obs_data)
    
    def wait_for_observation(self, timeout: Optional[float] = None) -> Tuple[Optional[Any], Dict]:
        """
        Wait for a new observation from DDNet (works for both modes).
        
        Args:
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            For structured mode: (StructuredObservation, info_dict) or (None, {})
            For rendered mode: (np.ndarray frame, info_dict) or (None, {})
        """
        if not self._connected:
            return None, {}
        
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while True:
            header = self._read_header()
            
            # Check for shutdown
            if header.shutdown:
                print("Shutdown signal received")
                return None, {}
            
            # Check if observation is ready
            if header.client_ready:
                if self.observation_mode == ObservationMode.STRUCTURED:
                    obs = self._read_structured_observation()
                else:
                    obs = self._read_frame()
                    
                info = {
                    'tick': header.tick,
                    'reward': header.reward,
                    'done': bool(header.done),
                    'truncated': bool(header.truncated),
                }
                return obs, info
            
            # Check timeout
            if time.time() - start_time > timeout:
                print("Timeout waiting for observation")
                return None, {}
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
    
    def wait_for_frame(self, timeout: Optional[float] = None) -> Tuple[Optional[np.ndarray], dict]:
        """
        Wait for a new frame from DDNet (rendered mode only).
        
        Args:
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            Tuple of (frame, info_dict) or (None, {}) if shutdown/timeout
        """
        if self.observation_mode == ObservationMode.STRUCTURED:
            print("Warning: wait_for_frame() called in structured mode. Use wait_for_observation() instead.")
            return None, {}
            
        if not self._connected:
            return None, {}
        
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while True:
            header = self._read_header()
            
            # Check for shutdown
            if header.shutdown:
                print("Shutdown signal received")
                return None, {}
            
            # Check if frame is ready
            if header.client_ready:
                frame = self._read_frame()
                info = {
                    'tick': header.tick,
                    'reward': header.reward,
                    'done': bool(header.done),
                    'truncated': bool(header.truncated),
                }
                return frame, info
            
            # Check timeout
            if time.time() - start_time > timeout:
                print("Timeout waiting for frame")
                return None, {}
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
    
    def send_action(self, direction: int = 0, jump: int = 0, hook: int = 0, 
                    fire: int = 0, aim_x: int = 0, aim_y: int = 0, weapon: int = 0):
        """
        Send an action to DDNet.
        
        Args:
            direction: -1 (left), 0 (none), 1 (right)
            jump: 0 or 1
            hook: 0 or 1
            fire: 0 or 1
            aim_x: Aim X coordinate (relative to player)
            aim_y: Aim Y coordinate (relative to player)
            weapon: Weapon to switch to (0-5)
        """
        if not self._connected:
            return
        
        # Write action fields (starting at offset 28 in the header)
        action_offset = 28
        self.shm_mmap.seek(action_offset)
        self.shm_mmap.write(struct.pack('<bBBBhhB',
            direction, jump, hook, fire, aim_x, aim_y, weapon))
        
        # Signal action is ready and clear client_ready
        self._write_header_field(0, 0, 'B')  # client_ready = 0
        self._write_header_field(1, 1, 'B')  # python_ready = 1
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def display_frames(client: TeeWorldsShmClient):
    """Display frames using OpenCV (rendered mode only)."""
    if client.is_structured_mode():
        print("Cannot display frames in structured mode. Use --print to see observations.")
        return
        
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return
    
    print("Displaying frames. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        frame, info = client.wait_for_frame()
        if frame is None:
            break
        
        # Debug: check frame content
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: shape={frame.shape}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize for display
        display_size = (frame.shape[1] * 4, frame.shape[0] * 4)
        frame_display = cv2.resize(frame_bgr, display_size, interpolation=cv2.INTER_NEAREST)
        
        # Add info overlay
        fps = frame_count / (time.time() - start_time + 0.001)
        cv2.putText(frame_display, f"Tick: {info.get('tick', 0)} FPS: {fps:.1f}", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Teeworlds RL', frame_display)
        
        # Send random action for testing
        import random
        client.send_action(
            direction=random.choice([-1, 0, 1]),
            jump=random.randint(0, 1),
            hook=0,
            fire=0,
            aim_x=random.randint(-100, 100),
            aim_y=random.randint(-100, 100)
        )
        
        frame_count += 1
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


def print_structured_observations(client: TeeWorldsShmClient):
    """Print structured observations to console."""
    if not client.is_structured_mode():
        print("Cannot print structured observations in rendered mode. Use --display for frames.")
        return
    
    print("Receiving structured observations. Press Ctrl+C to stop.")
    obs_count = 0
    start_time = time.time()
    
    try:
        while True:
            obs, info = client.wait_for_observation()
            if obs is None:
                break
            
            obs_count += 1
            if obs_count % 30 == 0:
                fps = obs_count / (time.time() - start_time)
                p = obs.local_player
                print(f"[{obs_count}] Tick:{obs.tick} FPS:{fps:.1f} | "
                      f"Pos:({p.pos_x:.0f},{p.pos_y:.0f}) Vel:({p.vel_x:.1f},{p.vel_y:.1f}) "
                      f"Dir:{p.direction} Hook:{p.hook_state} Frozen:{p.is_frozen} "
                      f"Players:{obs.num_active_players}")
            
            # Send random action
            import random
            client.send_action(
                direction=random.choice([-1, 0, 1]),
                jump=random.randint(0, 1)
            )
    except KeyboardInterrupt:
        print(f"\nStopped. Received {obs_count} observations.")


def stream_frames(client: TeeWorldsShmClient, port: int = 5555):
    """Stream frames over ZeroMQ for remote viewing."""
    try:
        import zmq
        import cv2
    except ImportError:
        print("ZeroMQ or OpenCV not installed. Install with: pip install pyzmq opencv-python")
        return
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    print(f"Streaming frames on port {port}. Use stream_viewer.py to view.")
    
    while True:
        frame, info = client.wait_for_frame()
        if frame is None:
            break
        
        # Encode frame as JPEG
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Send frame with metadata
        socket.send_multipart([
            b'frame',
            struct.pack('<I', info.get('tick', 0)),
            encoded.tobytes()
        ])
        
        # Send random action
        import random
        client.send_action(
            direction=random.choice([-1, 0, 1]),
            jump=random.randint(0, 1)
        )
    
    socket.close()
    context.term()


def main():
    parser = argparse.ArgumentParser(description='Teeworlds RL Observation Receiver')
    parser.add_argument('--env-id', type=int, default=0, help='Environment ID')
    parser.add_argument('--display', action='store_true', help='Display frames using OpenCV (rendered mode)')
    parser.add_argument('--print', dest='print_obs', action='store_true', help='Print structured observations')
    parser.add_argument('--stream', type=int, default=0, help='Stream frames on specified port (rendered mode)')
    parser.add_argument('--save', type=str, default='', help='Save frames to directory (rendered mode)')
    args = parser.parse_args()
    
    with TeeWorldsShmClient(env_id=args.env_id) as client:
        if not client.is_connected():
            print("Failed to connect. Make sure DDNet is running with 'rl_enable' command.")
            return
        
        if args.display:
            display_frames(client)
        elif args.print_obs:
            print_structured_observations(client)
        elif args.stream:
            stream_frames(client, args.stream)
        elif args.save:
            if client.is_structured_mode():
                print("Cannot save frames in structured mode.")
                return
            os.makedirs(args.save, exist_ok=True)
            frame_num = 0
            while True:
                frame, info = client.wait_for_frame()
                if frame is None:
                    break
                
                # Save frame
                try:
                    import cv2
                    path = os.path.join(args.save, f"frame_{frame_num:06d}.png")
                    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except ImportError:
                    from PIL import Image
                    path = os.path.join(args.save, f"frame_{frame_num:06d}.png")
                    Image.fromarray(frame).save(path)
                
                # Send random action
                import random
                client.send_action(direction=random.choice([-1, 0, 1]))
                
                frame_num += 1
                if frame_num % 100 == 0:
                    print(f"Saved {frame_num} frames")
        else:
            # Auto-detect mode and run appropriate receiver
            print("Receiving observations. Press Ctrl+C to stop.")
            obs_count = 0
            start_time = time.time()
            
            try:
                while True:
                    obs, info = client.wait_for_observation()
                    if obs is None:
                        break
                    
                    obs_count += 1
                    if obs_count % 50 == 0:
                        fps = obs_count / (time.time() - start_time)
                        if client.is_structured_mode():
                            p = obs.local_player
                            print(f"Obs: {obs_count}, FPS: {fps:.1f}, Tick: {info.get('tick', 0)}, "
                                  f"Pos: ({p.pos_x:.0f}, {p.pos_y:.0f})")
                        else:
                            print(f"Frames: {obs_count}, FPS: {fps:.1f}, Tick: {info.get('tick', 0)}")
                    
                    # Send no-op action
                    client.send_action()
            except KeyboardInterrupt:
                print("\nStopped by user")


if __name__ == '__main__':
    main()
