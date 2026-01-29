"""
Test script for DDNet RL reset functionality.

This script starts a DDNet client with kill bind and tests sending
the reset (kill) keypress via shared memory every 2 seconds.
"""

import subprocess
import time
import os
import signal
from teeworlds_env import TeeWorldsShmClient

# Configuration
DDNET_PATH = os.environ.get('DDNET_PATH', '../../../../../build/DDNet')  # Adjust as needed
SERVER_PORT = 8303
ENV_ID = 0

def main():
    # Start DDNet client with kill bind
    print("Starting DDNet client with kill bind...")
    client_cmd = [
        DDNET_PATH,
        f"connect localhost:{SERVER_PORT}; rl_enable state {ENV_ID}; snd_volume 0; bind k kill"
    ]
    client_proc = subprocess.Popen(
        client_cmd,
        preexec_fn=os.setsid,
        stdout=None,  # Print stdout to console
        stderr=None   # Print stderr to console
    )
    time.sleep(3)  # Wait for client to connect

    # Connect Python client
    print("Connecting Python client...")
    client = TeeWorldsShmClient(env_id=ENV_ID)
    if not client.connect(max_retries=10):
        print("Failed to connect to DDNet client")
        return

    print("Connected! Testing reset functionality...")

    try:
        # Wait for initial observation
        obs, info = client.wait_for_observation(timeout=10.0)
        if obs is None:
            print("Failed to get initial observation")
            return

        print("Initial observation received. Starting reset test...")

        reset_count = 0
        while True:
            # Send reset (kill) keypress
            client.reset()
            reset_count += 1
            print(f"Sent reset #{reset_count} at {time.time()}")

            # Wait 2 seconds
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        cleanup(None, client_proc, client)
    except Exception as e:
        print(f"Test failed with error: {e}")
        cleanup(None, client_proc, client)


def cleanup(server_proc, client_proc, client):
    """Clean up processes."""
    try:
        if client:
            client.disconnect()
        if client_proc.poll() is None:
            os.killpg(os.getpgid(client_proc.pid), signal.SIGTERM)
        if server_proc.poll() is None:
            server_proc.terminate()
    except:
        pass
    print("Cleanup complete")

if __name__ == '__main__':
    main()