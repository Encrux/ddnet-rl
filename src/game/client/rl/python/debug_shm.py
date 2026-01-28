#!/usr/bin/env python3
"""Debug script to inspect shared memory contents."""

import mmap
import struct
import time
import os
import numpy as np

SHM_PATH = "/dev/shm/teeworlds_rl_0"
HEADER_SIZE = 64
HEADER_FORMAT = '<BBBBIIIIfBB2sbBBBhhB3s24s'

def main():
    print(f"Looking for shared memory at: {SHM_PATH}")
    
    if not os.path.exists(SHM_PATH):
        print(f"ERROR: {SHM_PATH} does not exist!")
        print("Make sure DDNet is running with 'rl_enable' command")
        return
    
    file_size = os.path.getsize(SHM_PATH)
    print(f"Shared memory file size: {file_size} bytes")
    
    with open(SHM_PATH, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        
        print("\n=== Header Debug ===")
        mm.seek(0)
        header_raw = mm.read(HEADER_SIZE)
        print(f"Raw header bytes: {header_raw.hex()}")
        
        unpacked = struct.unpack(HEADER_FORMAT, header_raw)
        print(f"\nUnpacked header:")
        print(f"  client_ready: {unpacked[0]}")
        print(f"  python_ready: {unpacked[1]}")
        print(f"  shutdown: {unpacked[2]}")
        print(f"  _padding1: {unpacked[3]}")
        print(f"  tick: {unpacked[4]}")
        print(f"  frame_width: {unpacked[5]}")
        print(f"  frame_height: {unpacked[6]}")
        print(f"  frame_channels: {unpacked[7]}")
        print(f"  reward: {unpacked[8]}")
        print(f"  done: {unpacked[9]}")
        print(f"  truncated: {unpacked[10]}")
        
        frame_width = unpacked[5]
        frame_height = unpacked[6]
        frame_channels = unpacked[7]
        
        if frame_width > 0 and frame_height > 0 and frame_channels > 0:
            frame_size = frame_width * frame_height * frame_channels
            print(f"\n=== Frame Debug ===")
            print(f"Expected frame size: {frame_size} bytes ({frame_width}x{frame_height}x{frame_channels})")
            
            mm.seek(HEADER_SIZE)
            frame_data = mm.read(min(frame_size, 1000))  # Read first 1000 bytes
            
            # Check if all zeros
            all_zeros = all(b == 0 for b in frame_data)
            print(f"First 1000 bytes all zeros: {all_zeros}")
            
            if not all_zeros:
                print(f"First 50 bytes: {frame_data[:50].hex()}")
                frame_arr = np.frombuffer(frame_data, dtype=np.uint8)
                print(f"Min: {frame_arr.min()}, Max: {frame_arr.max()}, Mean: {frame_arr.mean():.2f}")
            else:
                print("Frame buffer is empty (all zeros)")
        else:
            print("\nFrame dimensions are 0 - no frame data written yet")
        
        # Monitor for changes
        print("\n=== Monitoring for changes (5 seconds) ===")
        last_tick = unpacked[4]
        last_client_ready = unpacked[0]
        
        for i in range(50):
            time.sleep(0.1)
            mm.seek(0)
            header_raw = mm.read(HEADER_SIZE)
            unpacked = struct.unpack(HEADER_FORMAT, header_raw)
            tick = unpacked[4]
            client_ready = unpacked[0]
            
            if tick != last_tick or client_ready != last_client_ready:
                print(f"  [+{i*100}ms] tick: {tick}, client_ready: {client_ready}")
                last_tick = tick
                last_client_ready = client_ready
        
        print("\nDone monitoring.")
        mm.close()

if __name__ == "__main__":
    main()
