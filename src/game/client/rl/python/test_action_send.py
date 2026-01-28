#!/usr/bin/env python3
"""
Test script to verify that sending actions to the DDNet client works.
Cycles through left, right, jump, and idle actions and prints the player position.
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_env import TeeWorldsShmClient

actions = [
    {'direction': -1, 'jump': 0, 'hook': 0, 'fire': 0},  # left
    {'direction': 1, 'jump': 0, 'hook': 0, 'fire': 0},   # right
    {'direction': 0, 'jump': 1, 'hook': 0, 'fire': 0},   # jump
    {'direction': 0, 'jump': 0, 'hook': 0, 'fire': 0},   # idle
]

def main():
    client = TeeWorldsShmClient(env_id=0, timeout=5.0)
    assert client.connect(), "Failed to connect to DDNet client."
    print("Connected. Cycling actions...")
    idx = 0
    for step in range(40):
        action = actions[idx % len(actions)]
        client.send_action(**action)
        obs, info = client.wait_for_observation(timeout=1.0)
        if obs is None:
            print("No observation received.")
            continue
        print(f"Step {step:2d} | Action: {action} | Pos: ({obs.local_player.pos_x:.1f}, {obs.local_player.pos_y:.1f})")
        idx += 1
        time.sleep(0.2)
    client.disconnect()
    print("Done.")

if __name__ == '__main__':
    main()
