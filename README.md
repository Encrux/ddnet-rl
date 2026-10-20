# DDNet RL

Reinforcement learning agents that learn to race [DDNet](https://ddnet.org) maps — a 2D
platformer whose movement is built around a grappling hook, momentum and freeze tiles. The
reward is the obvious one: get to the finish, fast.

This is a fork of the DDNet game. The agents are not scripted bots: they run inside a modified
DDNet server, see a local view of the map, and are trained from scratch with PPO.

There is a write-up of the project here: **https://boesch.dev/posts/ddnet-rl/**

## Status

Honest version: it works on one map, and the headline number has an *n* of 1.

On `Tutorial`, giving the agent DDNet's own hook aim-line as 16 per-sector reachability probes
broke a path-progress ceiling of ~0.474 that had held across four consecutive runs, and produced
the first non-zero from-start finish rate in the project's history:

| run | observation | path progress (edge) | finish rate, from start |
| --- | --- | --- | --- |
| I1 / I2 | hook-blind | 0.4694 / 0.4700 | 0% / 0% |
| N1 / N2 | hook state fixed | 0.4550 / 0.4550 | 0% / 0% |
| **P1** | **+ hook aim-line** | **0.4763** | **21.27%** |

Overall finish rate including runs restored mid-map was 89.95%. Confirmed in-game, but it is a
single run and the replicate is still pending. Notably, fixing the hook-state encoding *alone*
changed nothing measurable — the agent had been inferring hook attachment from its velocity
signature. What it lacked was line-of-sight along a direction.

Generalising across maps is **not** solved. See [`docs/research/roadmap.md`](docs/research/roadmap.md)
for what is currently believed to be wrong, and
[`docs/research/ABLATION.md`](docs/research/ABLATION.md) for the measurements — including three
claims that were later retracted.

## How it works

The bots live in the server process. Python only does inference and learning.

```
┌──────────────────────────────┐    POSIX shared memory    ┌──────────────────────┐
│  DDNet-Server (modified C++) │ /dev/shm/ddnet_rl_server* │  Python (PyTorch)    │
│                              │ ◄───────────────────────► │                      │
│  up to 127 headless bots     │    obs, reward, done  →   │  PPO, TensorBoard    │
│  physics, reward, snapshots  │         ←  actions        │  curriculum, archive │
└──────────────────────────────┘                           └──────────────────────┘
          × 8-12 server processes in parallel  →  ~1000-1500 bots training at once
```

Why this shape:

- **Bots in the server, not clients.** An earlier design ran one DDNet *client* per bot over
  shared memory; it cost a process and a render loop per agent. Moving the bots into the server
  removed both. One client slot is deliberately left free (`sv_max_clients 128`) so you can join
  a training server and watch.
- **Shared memory, not sockets.** No serialisation, no syscall per step.
- **Inference in Python, not C++.** An earlier attempt ran ONNX Runtime inside the server. The
  policy collapsed to a constant action and debugging it in C++ was painful, so inference moved
  out. That branch is kept as history, not as a build target.
- **Frame skip.** The agent acts every 3rd tick at `sv_tickrate 50`, i.e. ~17 decisions/second,
  with reward accumulating in between.

### Observations

Per bot, laid out by `rl_protocol.h` and parsed by `observation.py`:

- a **31×43 tile grid** centred on the tee, carrying raw DDNet tile ids
- **player state** — position, velocity, hook state and phase, freeze, jumps, grounded — stacked
  over the last 4 decisions
- **navigation** — normalised progress along the waypoint path, that episode's high-water
  progress, and **8 lookahead points** on the path (1, 2, 4 … 128 waypoints ahead) as egocentric
  offsets
- **16 hook-reach probes**, one per aim sector: DDNet's own red/green aim line, marched
  server-side by the same routine the real hook uses, so the two cannot disagree

That is 1427 floats in the default configuration — 1333 tiles, 4 × 15 player state, 34 navigation.

Raw tile ids make a poor network input: `SOLID = 1`, `NOHOOK = 3` and `THROUGH = 6` are
neighbouring integers with opposite meanings, and a hook climb turns on exactly that distinction.
`--semantic-tiles` reindexes them through a fixed 12-channel lookup — no learned parameters, just
orthogonal channels — before the encoder. It is not the default, and it has not been ablated.

There is no checkpoint in the observation and there are no raycasts. Both existed and were
removed: DDNet time checkpoints are mapper convention, and the raycasts were subsumed by the tile
grid plus the hook probes.

### Actions

A hybrid space: multi-discrete for direction, jump and hook, plus a **factored continuous aim**
(aim is not bucketed — some maps need it precise). A learned **duration head** lets the policy
hold an action for several decisions, which is what makes long hook swings discoverable at all.

### Reward

Three terms, deliberately no more:

```
R = −0.001 per tick                 time cost — the actual race objective
  + 100.0 · (γ·Φ′ − Φ) per decision progress along the waypoint path
  + 100.0 on finish
```

Φ is normalised progress along an explicit **waypoint path**. The shaping is a *potential
difference*, which by [Ng, Harada & Russell (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
leaves the optimal policy unchanged for any Φ — so a mediocre ghost line or a bad pathfinder
segment can cost learning speed but can never teach the wrong behaviour. It is applied once per
*decision*, not per tick: reward accumulates across the frame skip, and a sum of per-tick
potential differences does not telescope.

Everything else was deliberately removed — freeze penalty, stuck ramp, off-path penalty,
checkpoint bonus, velocity-toward-checkpoint. Each was a hand-tuned scale with no reason to hold
on the next map. The freeze penalty was the worst: −5.0/s against a maximum progress rate of
+5.0/s, in a game where passing through freeze is routine and sometimes mandatory.

`sv_rl_ratchet` (off by default) makes Φ the episode's **high-water** progress, so giving ground
back costs only time. Plain shaping charges for the run-up that hard obstacles need: at the 0.455
wall the correct play starts by backing up ~3 tiles — an immediate −0.086, about 29 decisions of
time penalty, against a payoff that during exploration almost never arrives. A high-water
potential is monotone, so it still telescopes and still preserves the optimum.

An earlier checkpoint-based reward broke on real maps: it assumed checkpoints were sequential and
reachable, and agents sprinted at an unreachable teleporter instead. Waypoints now come from
either a human **ghost replay** (`ghost_parser.py` decodes `.gho`) or a tile-level **BFS
pathfinder** that respects teleporters and avoids death tiles (`map_pathfinder.py`).

### Resets

Starting every episode at the spawn wastes nearly all compute re-walking solved ground, so resets
are drawn from an archive, in the spirit of [Go-Explore](https://arxiv.org/abs/1901.10995):

- episodes are snapshotted with DDNet's own `CSaveTee`, so a restore is exact — position,
  velocity and hook included
- archive cells are keyed by `(progress, hook state, velocity)`, not progress alone, so the
  agent can be restored *mid-swing* and not just mid-corridor
- sampling is capped at a **frontier**: the furthest point proven reachable from the start, so
  the agent is never dropped past its own demonstrated ability
- 20% of resets still start from the beginning, to keep whole-map skill

## Quick start

Requires a CUDA GPU for anything but a smoke test. Reference hardware is an RTX 3080 with a
Ryzen 5800X3D.

```sh
git clone --recursive https://github.com/Encrux/ddnet-rl
cd ddnet-rl

# 1. Build the server. Training needs only DDNet-Server; the client is optional (to watch).
cmake -B build -GNinja
ninja -C build DDNet-Server

# 2. Python environment
uv sync

# 3. Train — this sizes the server count to your CPU and clears stale
#    processes and shared memory first
./deploy/train.sh
```

To drive `train.py` yourself, note that `configs/default.yaml` is sized for the reference rig
above — 12 servers × 127 bots — so scale it down unless you have the cores for it:

```sh
cd src/game/client/rl/python
uv run python train.py --config configs/default.yaml --num-servers 4 --num-envs 32
```

Other things you will want:

```sh
# Watch what the agent actually sees (tiles, hook probes, path lookahead).
# Add --window for an interactive OpenCV view you can drive with a/d/space.
uv run python visualize_obs.py --num-bots 1

# Evaluate a checkpoint from the map start, which is the number that matters.
# It builds the network with --semantic-tiles and --action-hold on and has no
# way to turn them off, so it only loads checkpoints trained with both.
uv run python eval_from_start.py runs/<run>/checkpoint_final.pt

# Generate waypoints for a new map, from a human replay or from the map itself
uv run python ghost_parser.py path/to/replay.gho --output ghosts/MyMap.waypoints.json
uv sync --extra maps   # map_pathfinder.py needs twmap to read .map files
uv run python map_pathfinder.py path/to/MyMap.map --output ghosts/MyMap.waypoints.json
```

Training writes TensorBoard logs and checkpoints to `runs/<name>/`, resolved against the
directory you launched from — so `./deploy/train.sh` at the repo root and `train.py` from the
Python directory write to two different `runs/`.

The RL-specific server settings are console variables, and **order matters** — `sv_rl_bots` must
be issued before the others, which silently no-op without an active bot manager:

| variable | meaning |
| --- | --- |
| `sv_rl_bots <n> <server_idx>` | spawn `n` headless bots; creates the bot manager |
| `sv_rl_frame_skip <n>` | act every `n`th tick |
| `sv_rl_ratchet <0\|1>` | high-water progress shaping |
| `sv_rl_archive_variants <0\|1>` | key archive cells by hook/velocity, not progress alone |

## Layout

```
src/game/server/rl/          bots, reward and snapshots inside the server (C++)
src/game/rl/rl_protocol.h    the shared-memory structs -- the contract between the two sides
src/game/client/rl/python/   training, environment and tooling (Python)
  train.py                   entry point
  ppo_hybrid.py              PPO, hybrid action space, duration head
  teeworlds_server_env.py    vectorised env over the shared-memory servers
  observation.py             observation structs; must match rl_protocol.h
  curriculum.py              frontier curriculum and archive sampling
  ghost_parser.py            .gho ghost replay decoder
  map_pathfinder.py          tile-level BFS pathfinder
  visualize_obs.py           render a bot's observation (terminal, or --window)
  observe.py                 the same, read-only, against an already-running server
deploy/train.sh              hardware-aware launcher
docs/research/               the research journal: roadmap, plan, ablations, run notes
```

`observation.py` and `rl_protocol.h` describe the same bytes. If they disagree, observations are
silently misparsed rather than rejected — an encoding bug here cost this project four runs — so
`test_obs_layout.py` pins the Python offsets against the C++ header itself rather than against a
copy of them.

## Lessons

- Most of the work is infrastructure, logging and debugging. The learning is the short part.
- Verify encodings against live runtime values. The agent was hook-blind for 60% of its state and
  nothing crashed; the bug showed up only as a progress plateau.
- Metrics can quietly answer a different question than the one you asked. Three results in
  `ABLATION.md` were retracted for exactly that reason, and a whole benchmark sweep was
  invalidated when `sv_rl_frame_skip` turned out to be a no-op.
- A reward function that assumes the map is well-behaved will find the map that isn't.

## Relationship to DDNet, and licence

This is a modified fork of [DDNet](https://github.com/ddnet/ddnet), which is itself based on
Teeworlds. All game code, assets and build tooling are theirs; the RL work is the addition.
Upstream build instructions, dependencies and platform notes live in the
[DDNet repository](https://github.com/ddnet/ddnet) and its
[wiki](https://wiki.ddnet.org/wiki/Development) — this fork does not restate them.

Licensed under the same terms as DDNet; see [`license.txt`](license.txt). Copyright for the
original game remains with Magnus Auvinen, Shereef Marzouk, Dennis Felsing and the DDNet
contributors. This is an unofficial fork and is not affiliated with or endorsed by the DDNet
project.
