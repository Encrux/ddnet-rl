"""
Teeworlds RL Environment - Shared Memory Client

This module provides a Python interface to receive game frames
from the DDNet client via shared memory.

Usage:
    python teeworlds_env.py [--env-id 0] [--display] [--stream PORT]
"""

import mmap
import struct
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import os


@dataclass
class SharedMemoryHeader:
    """Matches the C++ SSharedMemoryHeader struct (64 bytes)"""
    # Sync flags
    client_ready: int       # uint8
    python_ready: int       # uint8
    shutdown: int           # uint8
    _padding1: int          # uint8
    
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
            self.client_ready, self.python_ready, self.shutdown, self._padding1,
            self.tick, self.frame_width, self.frame_height, self.frame_channels,
            self.reward, self.done, self.truncated, self._padding2,
            self.action_direction, self.action_jump, self.action_hook, self.action_fire,
            self.action_aim_x, self.action_aim_y, self.action_weapon, self._padding3,
            self._reserved
        )


class TeeWorldsShmClient:
    """
    Shared memory client to receive frames from DDNet.
    
    Example:
        client = TeeWorldsShmClient(env_id=0)
        while True:
            frame, info = client.wait_for_frame()
            if frame is None:
                break
            # Process frame...
            client.send_action(direction=1, jump=0, hook=0, fire=0, aim_x=100, aim_y=0)
    """
    
    def __init__(self, env_id: int = 0, timeout: float = 5.0):
        """
        Initialize the shared memory client.
        
        Args:
            env_id: Environment ID (matches rl_enable command in game)
            timeout: Timeout in seconds for waiting for frames
        """
        self.env_id = env_id
        self.timeout = timeout
        self.shm_path = f"/dev/shm/teeworlds_rl_{env_id}"
        self.shm_file = None
        self.shm_mmap = None
        self.header_size = SharedMemoryHeader.STRUCT_SIZE
        
        self.frame_width = 0
        self.frame_height = 0
        self.frame_channels = 0
        self.frame_size = 0
        
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
                    
                    # Read header to get frame dimensions
                    header_data = self.shm_file.read(self.header_size)
                    header = SharedMemoryHeader.from_bytes(header_data)
                    
                    self.frame_width = header.frame_width
                    self.frame_height = header.frame_height
                    self.frame_channels = header.frame_channels
                    self.frame_size = self.frame_width * self.frame_height * self.frame_channels
                    
                    # Map the full shared memory
                    total_size = self.header_size + self.frame_size
                    self.shm_file.seek(0)
                    self.shm_mmap = mmap.mmap(self.shm_file.fileno(), total_size)
                    
                    self._connected = True
                    print(f"Connected to {self.shm_path}")
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
        """Read the frame data from shared memory."""
        self.shm_mmap.seek(self.header_size)
        frame_data = self.shm_mmap.read(self.frame_size)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        return frame.reshape((self.frame_height, self.frame_width, self.frame_channels))
    
    def wait_for_frame(self, timeout: Optional[float] = None) -> Tuple[Optional[np.ndarray], dict]:
        """
        Wait for a new frame from DDNet.
        
        Args:
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            Tuple of (frame, info_dict) or (None, {}) if shutdown/timeout
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
    """Display frames using OpenCV."""
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
    parser = argparse.ArgumentParser(description='Teeworlds RL Frame Receiver')
    parser.add_argument('--env-id', type=int, default=0, help='Environment ID')
    parser.add_argument('--display', action='store_true', help='Display frames using OpenCV')
    parser.add_argument('--stream', type=int, default=0, help='Stream frames on specified port')
    parser.add_argument('--save', type=str, default='', help='Save frames to directory')
    args = parser.parse_args()
    
    with TeeWorldsShmClient(env_id=args.env_id) as client:
        if not client.is_connected():
            print("Failed to connect. Make sure DDNet is running with 'rl_enable' command.")
            return
        
        if args.display:
            display_frames(client)
        elif args.stream:
            stream_frames(client, args.stream)
        elif args.save:
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
            # Just print frame info
            print("Receiving frames. Press Ctrl+C to stop.")
            frame_count = 0
            start_time = time.time()
            
            try:
                while True:
                    frame, info = client.wait_for_frame()
                    if frame is None:
                        break
                    
                    frame_count += 1
                    if frame_count % 50 == 0:
                        fps = frame_count / (time.time() - start_time)
                        print(f"Frames: {frame_count}, FPS: {fps:.1f}, Tick: {info.get('tick', 0)}")
                    
                    # Send no-op action
                    client.send_action()
            except KeyboardInterrupt:
                print("\nStopped by user")


if __name__ == '__main__':
    main()
