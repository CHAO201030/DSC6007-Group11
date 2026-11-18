# Beyond Self-Play: A Controlled Study of Partner Generalisation in a Collaborative Kitchen Environment

English | [简体中文](README.zh-CN.md)

**DSC6007 Group 11**

This project vendors the runtime environment of
[HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai)
v2.0.0 as the `GameEnv/` package, so the Gymnasium-style Overcooked environment can be
used directly **without `pip install`**.

The goal is a controlled study of **partner generalisation** in the two-player
cooperative Overcooked kitchen task: when training exposes an agent to only a
limited set of partners (including self-play), can the resulting policy still
collaborate effectively with unseen partners at evaluation time?

## Research Goals

- Compare self-play with partner-diverse training under a fixed layout, horizon,
  training budget and evaluation seed.
- Evaluate collaboration with held-out partners, distinguishing "works with the
  training partners" from "generalises to new partners".
- Keep the environment, observation encodings and evaluation protocol fixed so
  that differences between algorithms come mainly from the partner policies and
  training regime.

> Status: environment setup stage. Skeleton directories `agents/`, `algorithms/`,
> `common/`, `config/` and `checkpoints/` exist; implementations await the team's
> division of work.

## Features

- **Batteries included**: `GameEnv/` ships the MDP, planners, visualization, Gym
  wrapper, and all layouts/graphics assets.
- **No upstream installation**: no `pip install overcooked_ai`, avoiding duplicate
  module trees and class-identity issues.
- **Gymnasium 1.x compatible**: `reset(seed=...)` returns `(obs, info)` and
  `step()` returns a 5-tuple.
- **Minimal conda environment**: `environment.yml` contains only the dependencies
  needed for running, training and plotting.
- **Smoke tests**: a single command validates the MDP, planners, rendering and Gym
  API.

## Quick Start

```bash
# 1. Create the environment (Linux / macOS / Windows)
conda env create -f environment.yml
conda activate gameenv

# 2. Verify the environment
python test_env.py
```

Expected output (the script prints its check names in Chinese):

```
GameEnv smoke test | layout='cramped_room' steps=20 seed=0

[PASS] 模块来源 — GameEnv 独立加载，overcooked_ai 未安装
[PASS] 布局加载 — grid=(5, 4)
[PASS] MDP transition — timestep 0 -> 1
[PASS] 观测编码 — featurized=(96,), lossless=(5, 4, 26)
[PASS] OvercookedEnv rollout — 20 步, total_reward=0.00
[PASS] Gymnasium API — reset -> (obs, info); step -> 5-tuple
[PASS] seed 可复现 — agent_idx=1, obs 一致
[PASS] 渲染 — frame=(928, 1056, 3), dtype=uint8

8/8 项检查通过
```

> No need to `pip install` this repository; run from the repository root
> (`GameEnv` is imported by directory).

## Directory Structure

```
DSC6007-Group11/
├── GameEnv/                 # vendored Overcooked-AI environment (read-only, see VENDORING.md)
│   ├── actions.py           # actions and directions
│   ├── overcooked_mdp.py    # MDP: states, transitions, observation encoding
│   ├── overcooked_env.py    # OvercookedEnv + Gymnasium wrapper
│   ├── planners.py          # motion / medium-level planners
│   ├── state_visualizer.py  # pygame rendering
│   ├── data/                # layouts, graphics, fonts, planner cache
│   └── VENDORING.md         # upstream source and modification log
├── agents/                  # decision and partner/opponent policies (to be implemented)
│   └── __init__.py
├── algorithms/              # learning algorithms such as PPO (to be implemented)
│   └── __init__.py
├── common/                  # shared interfaces and utilities
│   └── __init__.py
├── config/                  # one YAML config per experiment
│   └── .gitkeep
├── checkpoints/             # weights and run metadata (contents are git-ignored)
│   └── .gitkeep
├── environment.yml          # minimal conda environment
├── test_env.py              # environment smoke tests
├── LICENSE                  # MIT for this repository
├── GameEnv/LICENSE.overcooked_ai
├── README.md                # this file (English; shown on GitHub)
└── README.zh-CN.md          # Chinese README
```

> `envs/` (environment wrappers), `scripts/` (training/evaluation entry points)
> and `tests/` (contract tests) will be added when the training pipeline is
> implemented.

## Code Organization Conventions

| Directory | Contains | Does not contain |
|---|---|---|
| `agents/` | `act(obs) -> int`, rule-based policies, partner/opponent policies, wrappers around trained policies | losses, optimizers, GAE |
| `algorithms/` | networks, losses, optimizers, rollout buffers, GAE, training loops | environment details, evaluation scripts, partner-selection logic |
| `common/` | shared `Protocol`s, observation/action utilities, seeding helpers | concrete algorithm implementations |
| `config/` | one YAML per experiment (layout, hyperparameters, seed, number of steps) | code |
| `checkpoints/` | `best.pt`, `last.pt`, `run.json`, etc. | anything except `.gitkeep` should be committed |

Changes to `common/` interfaces require group review; `GameEnv/` is read-only;
the contents of `checkpoints/` never enter git.

## Usage Examples

### 1. Using the MDP directly (no Gym needed)

```python
from GameEnv.actions import Action
from GameEnv.overcooked_mdp import OvercookedGridworld

mdp = OvercookedGridworld.from_layout_name("cramped_room")
state = mdp.get_standard_start_state()

next_state, info = mdp.get_state_transition(
    state, (Action.STAY, Action.STAY)
)
print(next_state.timestep)  # 1
```

### 2. Using `OvercookedEnv` (joint actions; recommended for multi-agent)

```python
from GameEnv.actions import Action
from GameEnv.overcooked_mdp import OvercookedGridworld
from GameEnv.overcooked_env import OvercookedEnv

mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)

state, reward, done, info = env.step((Action.INTERACT, Action.STAY))
print(reward)                       # sparse reward (non-zero only on soup delivery)
print(info["sparse_r_by_agent"])    # per-agent sparse rewards
print(info["shaped_r_by_agent"])    # per-agent shaping rewards
```

- `OvercookedEnv.step` takes a **joint action** `(action_p0, action_p1)`.
- It returns `(next_state, reward, done, info)`.
- `done` is determined by `horizon` (Overcooked has no true terminal states).

### 3. Using the Gymnasium interface

```python
import gymnasium as gym
from GameEnv.overcooked_mdp import OvercookedGridworld
from GameEnv.overcooked_env import OvercookedEnv

mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)

env = gym.make(
    "Overcooked-v0",
    base_env=base_env,
    featurize_fn=base_env.featurize_state_mdp,
    # obs is a dict containing OvercookedState, which a Box space cannot describe
    disable_env_checker=True,
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step((4, 4))  # (STAY, STAY)
```

- Observations are a dict: `both_agent_obs`, `overcooked_state`,
  `other_agent_env_idx`.
- The action space is declared as `Discrete(6)`, but `step()` actually takes
  `(own_action, other_agent_action)`.
- Training PPO/SB3 requires a wrapper that extracts observations into arrays and
  maps actions to a standard `Discrete` space.

### Action Indices

| Index | Action | Description |
|---|---|---|
| 0 | `NORTH` | move up |
| 1 | `SOUTH` | move down |
| 2 | `EAST` | move right |
| 3 | `WEST` | move left |
| 4 | `STAY` | stay in place |
| 5 | `INTERACT` | interact (pick up, put in pot, serve, ...) |

### Observation Encodings

```python
# For MLPs: a 96-dimensional vector per agent
obs_p0, obs_p1 = base_env.featurize_state_mdp(base_env.state)

# For CNNs: an (H, W, 26) tensor
enc_p0, enc_p1 = base_env.lossless_state_encoding_mdp(base_env.state)
print(enc_p0.shape)  # e.g. (5, 4, 26)
```

## Testing

```bash
python test_env.py                                  # cramped_room, 20 steps
python test_env.py --layout forced_coordination     # another layout
python test_env.py --steps 50 --seed 7              # custom steps and seed
python test_env.py --all-layouts                    # check all 49 bundled layouts
python test_env.py --skip-render                    # skip rendering on headless machines
```

| Flag | Default | Description |
|---|---|---|
| `--layout` | `cramped_room` | layout name |
| `--steps` | `20` | random rollout length |
| `--seed` | `0` | random seed |
| `--all-layouts` | off | check every layout under `GameEnv/data/layouts` |
| `--skip-render` | off | skip the pygame rendering check |

Set `GAMEENV_TEST_TRACEBACK=1` to print full tracebacks.

## Environment

| Dependency | Version (solver result) | Purpose |
|---|---|---|
| Python | 3.12 | |
| NumPy | 2.5.4 | MDP and observations |
| SciPy | 1.18.1 | sparse graph search in planners |
| Gymnasium | 1.3.0 | `gym.Env` and `Overcooked-v0` registration |
| pygame | 2.6.1 | rendering and `StateVisualizer` |
| PyTorch | 2.14.1 (CPU) | training |
| tqdm | 4.70.1 | progress bars |
| matplotlib / pandas / seaborn | 3.x / 3.x / 0.13 | learning-curve plots |
| opencv-python-headless | >=4.8 | resizing rendered frames (pip) |

- The default PyTorch build is **CPU-only**. On a CUDA machine, remove `pytorch`
  from the YAML, create the environment, then run:
  `pip install torch --index-url https://download.pytorch.org/whl/cu126`
- Headless machines (WSL/servers) need a dummy video driver: `test_env.py` sets
  `SDL_VIDEODRIVER=dummy` automatically; other scripts must set it themselves
  before importing `GameEnv`. The Gym wrapper's `env.render()` returns an RGB
  ndarray; for a live pygame window use `OvercookedEnv.display_states(...)` or
  `StateVisualizer.display_rendered_state(...)` with a working WSLg/X11 display.
- `GameEnv/data/planners/*.pkl` are planner caches generated on first use and are
  ignored by `.gitignore`; delete them and they will be recomputed on demand.

## Known Caveats

1. **Do not `pip install overcooked_ai`.** This repository uses `GameEnv` as the
   single environment source; installing the upstream package side by side creates
   two module trees and breaks class identity and pickles.
2. **`GameEnv/` is vendored code** and should stay read-only. If a change is
   unavoidable, record it in `GameEnv/VENDORING.md` instead of patching it ad hoc.
3. **Rewards are sparse.** By default reward is given only on soup delivery; for
   training you may add `info["shaped_r_by_agent"]` to the reward or design your
   own reward shaping.
4. **Horizon is not termination.** `done=True` means the horizon was reached
   (truncation). Distinguish `terminated` from `truncated` in training to avoid
   incorrect bootstrapping at truncation.
5. **Weights do not go into git.** Everything under `checkpoints/` except
   `.gitkeep` is ignored. Share weights via cloud storage, W&B artifacts or Git
   LFS instead of committing large files.

## Upstream and License

- Upstream project: [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) v2.0.0
- Changes in `GameEnv/` are limited to import paths, Gymnasium 1.x compatibility
  and lazy imports of optional notebook dependencies; see
  [`GameEnv/VENDORING.md`](GameEnv/VENDORING.md).
- Upstream code is MIT-licensed; see
  [`GameEnv/LICENSE.overcooked_ai`](GameEnv/LICENSE.overcooked_ai).
- This repository is MIT-licensed; see [`LICENSE`](LICENSE).
