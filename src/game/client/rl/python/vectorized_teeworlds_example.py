"""
Example: Vectorized Teeworlds RL Environment using Python multiprocessing

This script demonstrates how to launch multiple DDNet clients and connect to them from a single Python process using multiprocessing, aggregating experience for a single shared policy (DQN shown, but can be adapted for PPO/A2C/etc).

Assumptions:
- Each DDNet client is started externally with a unique --env-id (or use a script to launch them)
- Each environment is accessible via shared memory segment teeworlds_rl_{env_id}
- The TeeWorldsShmClient supports env_id argument

This is a minimal, illustrative example. For production, use a proper Gym vectorized wrapper and robust process management.
"""

import multiprocessing as mp
import numpy as np
import time
import subprocess
import os
import signal
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from teeworlds_env import TeeWorldsShmClient

NUM_ENVS = 2
NUM_STEPS = 1000  # Steps per episode
DDNET_PATH = os.environ.get('DDNET_PATH', '../../../../../build/DDNet')  # Adjust as needed
SERVER_PORT = 8303

# --- DQN Policy (copied from train_basic_agent.py) ---
OBS_SIZE = 700  # Will be set dynamically
N_ACTIONS = 5  # Not used directly, see below
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE = 1000

class QNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]
    def __len__(self):
        return len(self.buffer)

# Worker function for each environment
def env_worker(env_id, conn):
    import time
    import os
    shm_path = f"/dev/shm/teeworlds_rl_{env_id}"
    # Wait for shared memory file to exist
    for _ in range(60):
        if os.path.exists(shm_path):
            break
        time.sleep(0.5)
    else:
        conn.send(None)
        return
    client = TeeWorldsShmClient(env_id=env_id)
    # Retry connect if needed
    for _ in range(10):
        if client.connect():
            break
        time.sleep(0.5)
    else:
        conn.send(None)
        return
    obs, _ = client.wait_for_observation(timeout=5.0)
    if obs is None:
        conn.send(None)
        return
    state = obs.to_flat_array().astype(np.float32)
    conn.send(state)  # Send initial state
    while True:
        action = conn.recv()  # Receive action from main process
        if action.get('close', False):
            break
        # If reset requested, send 'kill' keypress (bound to 'k')
        if action.get('reset', False):
            client.reset()  # send keypress
            obs, info = client.wait_for_observation(timeout=5.0)
            if obs is None:
                conn.send(None)
                continue
            state = obs.to_flat_array().astype(np.float32)
            conn.send(state)
            continue
        client.send_action(**action)
        next_obs, info = client.wait_for_observation(timeout=2.0)
        if next_obs is None:
            conn.send((None, info))
            continue
        next_state = next_obs.to_flat_array().astype(np.float32)
        reward = info.get('reward', 0.0)
        done = info.get('done', False)
        conn.send((next_state, reward, done, info))
    client.disconnect()

# Main process: manages policy and aggregates experience
def main():


    # Start DDNet clients as subprocesses
    ddnet_procs = []
    for i in range(NUM_ENVS):
        cmd = [DDNET_PATH, f"connect localhost:{SERVER_PORT}; rl_enable state {i}; snd_volume 0; bind k kill"]
        proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        ddnet_procs.append(proc)
        # Wait for shared memory file to exist before starting worker
        shm_path = f"/dev/shm/teeworlds_rl_{i}"
        for _ in range(60):
            if os.path.exists(shm_path):
                print(f"Shared memory for environment {i} started successfully")
                break
            time.sleep(0.5)
        else:
            print(f"Timeout waiting for {shm_path}")
            return
    
    time.sleep(5)
    print("DDNet clients startup complete")

    # Start environment workers
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(NUM_ENVS)])
    procs = [mp.Process(target=env_worker, args=(i, child_conns[i])) for i in range(NUM_ENVS)]
    for p in procs:
        p.start()

    # Get initial states
    states = [parent_conns[i].recv() for i in range(NUM_ENVS)]
    obs_size = len(states[0])
    global OBS_SIZE
    OBS_SIZE = obs_size

    policy_net = QNet(OBS_SIZE, NUM_ENVS)
    target_net = QNet(OBS_SIZE, NUM_ENVS)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    steps_done = 0
    episode_rewards = [0 for _ in range(NUM_ENVS)]
    dones = [False for _ in range(NUM_ENVS)]
    episode_steps = [0 for _ in range(NUM_ENVS)]

    episode_count = 0
    try:
        while True:
            actions = []
            for i in range(NUM_ENVS):
                # Epsilon-greedy
                eps = 0.05 + (1.0 - 0.05) * np.exp(-1. * steps_done / 10000)
                if dones[i]:
                    # Request reset by sending a reset action
                    parent_conns[i].send({'reset': True})
                    states[i] = parent_conns[i].recv()
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    dones[i] = False
                    actions.append({'reset': True})
                    continue
                if np.random.rand() < eps:
                    action = {
                        'direction': np.random.choice([-1, 0, 1]),
                        'jump': np.random.randint(0, 2),
                        'hook': np.random.randint(0, 2),
                        'fire': np.random.randint(0, 2),
                        'aim_x': np.random.randint(-180, 181),
                        'aim_y': np.random.randint(-180, 181),
                        'weapon': np.random.randint(0, 5),
                    }
                else:
                    with torch.no_grad():
                        state_t = torch.tensor(states[i]).unsqueeze(0).float()
                        qvals = policy_net(state_t)
                        action_idx = int(torch.argmax(qvals)) % 5
                        # Map action_idx to a fixed action (expand as needed)
                        action = [
                            {'direction': 0, 'jump': 0, 'hook': 0, 'fire': 0, 'aim_x': 0, 'aim_y': 0, 'weapon': 0},
                            {'direction': -1, 'jump': 0, 'hook': 0, 'fire': 0, 'aim_x': 0, 'aim_y': 0, 'weapon': 0},
                            {'direction': 1, 'jump': 0, 'hook': 0, 'fire': 0, 'aim_x': 0, 'aim_y': 0, 'weapon': 0},
                            {'direction': 0, 'jump': 1, 'hook': 0, 'fire': 0, 'aim_x': 0, 'aim_y': 0, 'weapon': 0},
                            {'direction': 0, 'jump': 0, 'hook': 1, 'fire': 0, 'aim_x': 0, 'aim_y': 0, 'weapon': 0},
                        ][action_idx]
                parent_conns[i].send(action)
                actions.append(action)
            # Collect results
            for i in range(NUM_ENVS):
                if actions[i].get('reset', False):
                    continue
                result = parent_conns[i].recv()
                if result[0] is None:
                    print(f"Env {i} returned None observation.")
                    continue
                next_state, reward, done, info = result
                memory.push(states[i], actions[i], reward, next_state, done)
                states[i] = next_state
                episode_rewards[i] += reward
                episode_steps[i] += 1
                # End episode if done or max steps reached
                if done or episode_steps[i] >= NUM_STEPS:
                    print(f"[Env {i}] Episode {episode_count} finished with reward {episode_rewards[i]} after {episode_steps[i]} steps")
                    dones[i] = True
                steps_done += 1
            # Learn
            if len(memory) > MIN_REPLAY_SIZE:
                batch = memory.sample(BATCH_SIZE)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)
                batch_states = torch.tensor(batch_states, dtype=torch.float32)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
                batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32)
                batch_dones = torch.tensor(batch_dones, dtype=torch.bool)
                # For demonstration, use only the first action index (expand for full parametric)
                batch_action_idxs = torch.zeros(BATCH_SIZE, dtype=torch.long)
                qvals = policy_net(batch_states).gather(1, batch_action_idxs.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_qvals = target_net(batch_next_states).max(1)[0]
                    target = batch_rewards + GAMMA * next_qvals * (~batch_dones)
                loss = nn.functional.mse_loss(qvals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update target
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            # Increment episode count if all envs done
            if all(dones):
                print(f"All environments finished episode {episode_count}. Rewards: {episode_rewards}")
                episode_count += 1
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Tell workers to close
        for i in range(NUM_ENVS):
            parent_conns[i].send({'close': True})
        print(f"Final episode rewards: {episode_rewards}")


    # Cleanup
    for p in procs:
        p.terminate()
        p.join()

    # Terminate DDNet clients
    for proc in ddnet_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass

if __name__ == "__main__":
    main()
