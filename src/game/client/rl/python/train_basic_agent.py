#!/usr/bin/env python3
"""
Minimal DQN RL agent for Teeworlds tutorial race track.
- Uses structured observation via shared memory
- Trains on discrete actions: left, right, jump, do nothing
- Saves episode rewards to CSV

Requirements: pip install torch numpy

Usage:
    python train_basic_agent.py
"""
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teeworlds_env import TeeWorldsShmClient, StructuredObservation

# --- Hyperparameters ---
OBS_SIZE = 700  # Adjust if you change to_flat_array()
N_ACTIONS = 5  # direction, jump, hook, fire, weapon (discrete fields)

# Reasonable aim range (game expects -256..256, but -180..180 is safe for most maps)
AIM_MIN = -180
AIM_MAX = 180
WEAPON_IDS = [0, 1, 2, 3, 4]  # hammer, pistol, shotgun, grenade, laser

def sample_random_action():
    """Sample a random valid action dict."""
    return {
        'direction': random.choice([-1, 0, 1]),
        'jump': random.randint(0, 1),
        'hook': random.randint(0, 1),
        'fire': random.randint(0, 1),
        'aim_x': random.randint(AIM_MIN, AIM_MAX),
        'aim_y': random.randint(AIM_MIN, AIM_MAX),
        'weapon': random.choice(WEAPON_IDS),
    }

def clamp_action(action):
    """Clamp aim_x and aim_y to safe range."""
    action['aim_x'] = max(AIM_MIN, min(AIM_MAX, action.get('aim_x', 0)))
    action['aim_y'] = max(AIM_MIN, min(AIM_MAX, action.get('aim_y', 0)))
    if action.get('weapon') not in WEAPON_IDS:
        action['weapon'] = 0
    return action
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MIN_REPLAY_SIZE = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 1000
MAX_EPISODES = 500
MAX_STEPS = 2000

# --- Q-network ---
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

# --- Replay Buffer ---
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

# --- Main Training Loop ---
def main():
    client = TeeWorldsShmClient(env_id=0, timeout=10.0)
    assert client.connect(), "Failed to connect to game client. Start DDNet with rl_enable!"
    print("Connected to Teeworlds RL client.")

    # Probe obs size
    obs, _ = client.wait_for_observation(timeout=5.0)
    obs_vec = obs.to_flat_array()
    obs_size = obs_vec.shape[0]
    print(f"Observation size: {obs_size}")

    # Adjust OBS_SIZE if needed
    global OBS_SIZE
    OBS_SIZE = obs_size

    policy_net = QNet(OBS_SIZE, N_ACTIONS)
    target_net = QNet(OBS_SIZE, N_ACTIONS)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_done = 0
    episode_rewards = []

    for episode in range(1, MAX_EPISODES+1):
        obs, _ = client.wait_for_observation(timeout=5.0)
        state = obs.to_flat_array().astype(np.float32)
        total_reward = 0
        done = False
        for t in range(MAX_STEPS):
            # Epsilon-greedy for demonstration: random actions
            action = sample_random_action()
            action = clamp_action(action)
            client.send_action(**action)
            next_obs, info = client.wait_for_observation(timeout=2.0)
            if next_obs is None:
                print("No observation received, ending episode early.")
                break
            next_state = next_obs.to_flat_array().astype(np.float32)
            reward = info.get('reward', 0.0)
            done = info.get('done', False)
            # For a real agent, store (state, action, reward, next_state, done) in replay buffer
            state = next_state
            total_reward += reward
            steps_done += 1
            # Learning code would go here
            if done:
                break
        episode_rewards.append(total_reward)
        print(f"Episode {episode:3d} | Reward: {total_reward:.2f} | Steps: {t+1}")
        # Optionally save model
        if episode % 50 == 0:
            np.savetxt("episode_rewards.csv", episode_rewards, delimiter=",")
    print("Training complete.")
    client.disconnect()

if __name__ == '__main__':
    main()
