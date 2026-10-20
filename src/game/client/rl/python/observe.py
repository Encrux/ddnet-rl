#!/usr/bin/env python3
"""Attach to a running RL server and print one bot's observation.

Read-only and passive: it writes no actions and takes part in no handshake, so
it is safe to point at a live training server. Use visualize_obs.py instead when
you want it to start its own server, or want the interactive window.

Both tools parse the shared memory through observation.py, which is the only
Python definition of the layout -- an observation viewer that decodes the bytes
its own way is worse than none, because it agrees with nothing.

Usage:
    python observe.py                        # one snapshot of bot 0, server 0
    python observe.py -b 3 -s 1              # bot 3 on the second server
    python observe.py --continuous           # refresh in place
"""

import argparse
import mmap
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from observation import StructuredObservation
from teeworlds_server_env import (CONTIGUOUS_DONES_OFFSET, CONTIGUOUS_OBS_OFFSET,
                                  CONTIGUOUS_REWARDS_OFFSET, CONTIGUOUS_SHM_SIZE,
                                  HEADER_SIZE, OBS_SIZE, get_shm_path)
from visualize_obs import render_terminal


class ObservationReader:
    """Reads observations out of a server's shared memory without driving it."""

    def __init__(self, server_index: int = 0):
        self.server_index = server_index
        self.shm_path = get_shm_path(server_index)
        self.shm_file = None
        self.shm_mmap = None

    def connect(self) -> bool:
        if not os.path.exists(self.shm_path):
            print(f"Shared memory not found: {self.shm_path}")
            print("Is DDNet-Server running with sv_rl_bots enabled?")
            return False
        self.shm_file = open(self.shm_path, 'rb')
        self.shm_mmap = mmap.mmap(self.shm_file.fileno(), CONTIGUOUS_SHM_SIZE,
                                  access=mmap.ACCESS_READ)
        return True

    def close(self):
        if self.shm_mmap:
            self.shm_mmap.close()
        if self.shm_file:
            self.shm_file.close()

    def read_header(self) -> dict:
        self.shm_mmap.seek(0)
        (server_ready, python_ready, shutdown, _paused,
         num_bots, server_tick, _p0, _p1) = struct.unpack(
            '<8i', self.shm_mmap.read(HEADER_SIZE))
        return {'server_ready': server_ready, 'python_ready': python_ready,
                'shutdown': shutdown, 'num_bots': num_bots,
                'server_tick': server_tick}

    def read_observation(self, bot_index: int):
        self.shm_mmap.seek(CONTIGUOUS_OBS_OFFSET + bot_index * OBS_SIZE)
        obs = StructuredObservation.from_bytes(self.shm_mmap.read(OBS_SIZE))
        self.shm_mmap.seek(CONTIGUOUS_REWARDS_OFFSET + bot_index * 4)
        reward, = struct.unpack('<f', self.shm_mmap.read(4))
        self.shm_mmap.seek(CONTIGUOUS_DONES_OFFSET + bot_index * 4)
        done, = struct.unpack('<i', self.shm_mmap.read(4))
        return obs, reward, bool(done)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a bot's observation from a running server")
    parser.add_argument('--bot-index', '-b', type=int, default=0)
    parser.add_argument('--server-index', '-s', type=int, default=0)
    parser.add_argument('--continuous', '-c', action='store_true',
                        help='Refresh in place until interrupted')
    parser.add_argument('--interval', '-i', type=float, default=0.5)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    reader = ObservationReader(args.server_index)
    if not reader.connect():
        return 1

    try:
        header = reader.read_header()
        if args.bot_index >= header['num_bots']:
            print(f"Bot index {args.bot_index} out of range "
                  f"(server reports {header['num_bots']} bots)")
            return 2

        while True:
            obs, reward, done = reader.read_observation(args.bot_index)
            header = reader.read_header()
            if args.continuous:
                print('\033[H\033[J', end='')
            print(f"server {args.server_index}  bot {args.bot_index}/"
                  f"{header['num_bots']}  server tick {header['server_tick']}")
            print(render_terminal(obs, reward, done, use_color=not args.no_color),
                  flush=True)
            if not args.continuous:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        reader.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
