"""Roll out a trained checkpoint from the map start, with a configurable cap.

Exists to separate a MEASUREMENT question from a TRAINING one. Runs trained
under a 1200-step cap show p50_from_start ~0.21-0.36, but the ghost-pace ceiling
at that cap is 0.3847 -- below the obstacle at 0.4557 -- so those episodes were
truncated before they could reach it. That number therefore says nothing about
whether the policy COULD get there given time.

This answers it directly: take the checkpoint, give it a long cap, spawn every
episode at the map start, and see how far it actually gets. If a 1200-cap policy
reaches ~0.45 here, then short-cap training never harmed from-start capability
and the episode-length machinery was solving an artifact.

Usage:
    python eval_from_start.py runs/fr_H1_frontier/checkpoint_final.pt \
        --max-episode-steps 3000 --episodes 200

NEVER import train.py here -- its atexit handler pkills all game servers.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_hold import ActionHold
from ppo_hybrid import PPOAgent, PPOConfig
from teeworlds_server_env import ServerEnvConfig, TeeWorldsServerEnv, FrameStackWrapper

PATH_LEN = 111928.0
GHOST = 598.0
WALL = 0.4557


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--max-episode-steps", type=int, default=3000)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=127)
    ap.add_argument("--num-servers", type=int, default=8)
    ap.add_argument("--map", type=str, default="Tutorial")
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--finish-progress", type=float, default=0.50)
    ap.add_argument("--deterministic", action="store_true",
                    help="Use the distribution mode instead of sampling")
    # Architecture must match the checkpoint or the state dict will not load.
    ap.add_argument("--semantic-tiles", action="store_true", default=True)
    ap.add_argument("--action-hold", action="store_true", default=True)
    ap.add_argument("--hold-durations", type=str, default="1,2,4,8,16")
    ap.add_argument("--n-player-state-stack", type=int, default=4)
    args = ap.parse_args()

    hold = tuple(int(x) for x in args.hold_durations.split(",")) if args.action_hold else ()

    # begin_bias 1.0 and curriculum off: EVERY episode starts at the map start,
    # which is the whole point -- restored episodes would tell us nothing about
    # whether the policy can run the map.
    cfg = ServerEnvConfig(
        frame_skip=args.frame_skip, use_dict_obs=True, use_embeddings=args.semantic_tiles,
        map_name=args.map, max_episode_steps=args.max_episode_steps,
        finish_progress=args.finish_progress, curriculum=False, begin_bias=1.0,
    )
    bots_per_server = args.num_envs // args.num_servers
    total = bots_per_server * args.num_servers
    env = TeeWorldsServerEnv(num_bots=bots_per_server, num_servers=args.num_servers, config=cfg)
    env = FrameStackWrapper(env, n_tile_stack=1, n_player_state_stack=args.n_player_state_stack)

    obs, _ = env.reset()
    ppo = PPOConfig(use_cnn=True, use_embeddings=args.semantic_tiles,
                    semantic_tiles=args.semantic_tiles, tile_encoder='cnn',
                    n_frame_stack=1, n_player_state_stack=args.n_player_state_stack,
                    hold_durations=hold, use_compile=False)
    agent = PPOAgent(obs_dim=0, num_envs=total, config=ppo,
                     device="cuda" if torch.cuda.is_available() else "cpu")
    agent.load(args.checkpoint)
    print(f"loaded {args.checkpoint}")

    holder = ActionHold(total, len(ppo.discrete_dims), ppo.continuous_dim, hold) if hold else None

    finals, finished, lengths = [], [], []
    steps = 0
    while len(finals) < args.episodes:
        action, _, _ = agent.get_action(obs, deterministic=args.deterministic)
        executed, _ = holder.apply(action) if holder else (action, None)
        obs, _, done, trunc, infos = env.step(executed)
        if holder:
            holder.clear(done | trunc)
        for info in infos:
            if 'episode' in info:
                finals.append(info['episode']['progress'])
                finished.append(bool(info['episode']['finished']))
                lengths.append(info['episode']['l'])
        steps += 1
        if steps > args.max_episode_steps * 3:
            break
    env.close()

    p = np.array(finals[:args.episodes])
    secs = args.max_episode_steps * args.frame_skip / 50.0
    ceil = GHOST * secs / PATH_LEN
    print("\n" + "=" * 62)
    print(f"FROM-START EVALUATION  ({len(p)} episodes, cap {args.max_episode_steps}"
          f" = {secs:.0f}s, ghost-pace ceiling {ceil:.4f})")
    print("=" * 62)
    for q in (10, 25, 50, 75, 90):
        print(f"  p{q:<3d} progress          {np.percentile(p, q):.4f}")
    print(f"  max                    {p.max():.4f}")
    print(f"  mean episode length    {np.mean(lengths[:len(p)]):.0f}")
    print(f"  reached the wall       {(p >= WALL - 0.005).mean():.1%}   (wall {WALL})")
    print(f"  crossed the wall       {(p > WALL + 1e-4).mean():.1%}")
    print(f"  finished {args.finish_progress:.2f}         {np.mean(finished[:len(p)]):.1%}")
    print("=" * 62)


if __name__ == "__main__":
    main()
