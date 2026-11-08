# GameEnv — vendored Overcooked-AI environment

This package is a vendored copy of the parts of
[HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai)
needed to run the standard Gymnasium environment without `pip install`.

- Upstream version: **2.0.0**
- Upstream path: `overcooked_ai/src/overcooked_ai_py/`
- License: MIT (see `LICENSE.overcooked_ai`)

## What was copied

The flat module layout replaces the upstream subpackages
(`mdp/`, `planning/`, `visualization/`):

| GameEnv module | Upstream file |
|---|---|
| `actions.py` | `mdp/actions.py` |
| `overcooked_mdp.py` | `mdp/overcooked_mdp.py` |
| `overcooked_env.py` | `mdp/overcooked_env.py` |
| `overcooked_trajectory.py` | `mdp/overcooked_trajectory.py` |
| `layout_generator.py` | `mdp/layout_generator.py` |
| `planners.py` | `planning/planners.py` |
| `search.py` | `planning/search.py` |
| `state_visualizer.py` | `visualization/state_visualizer.py` |
| `pygame_utils.py` | `visualization/pygame_utils.py` |
| `visualization_utils.py` | `visualization/visualization_utils.py` |
| `utils.py` / `static.py` | `utils.py` / `static.py` |
| `data/` | `data/{layouts,planners,graphics,fonts}` |

`data/planners/*.pkl` were **not** copied. Those pickles embed upstream
module paths (`overcooked_ai_py.*`) and would deserialize upstream classes,
creating a second module tree. `MotionPlanner.from_pickle_or_compute` and
`MediumLevelActionManager.from_pickle_or_compute` regenerate and cache them
inside `GameEnv/data/planners/` on first use instead.

## Changes made

### Import rewrites (no algorithmic changes)

- `overcooked_ai_py.mdp.*` → `GameEnv.*`
- `overcooked_ai_py.planning.*` → `GameEnv.*`
- `overcooked_ai_py.visualization.*` → `GameEnv.*`
- `overcooked_ai_py.utils` / `overcooked_ai_py.static` → `GameEnv.utils` / `GameEnv.static`
- `overcooked_ai_py.data.planners` → `GameEnv.data.planners`
- `overcooked_env.py`: the lazy `AgentEvaluator` import/check inside the
  trajectory helper was disabled because the `agents/` subpackage is not
  vendored. All other methods are unchanged.
- `visualization_utils.py`: `IPython` and `ipywidgets` are imported inside the
  notebook helper functions instead of at module import, so the Jupyter stack
  is optional.

`__init__.py` registers the Gymnasium id `Overcooked-v0`
(entry point `GameEnv.overcooked_env:Overcooked`), matching upstream.

### Gymnasium 1.x compatibility (`overcooked_env.py`, class `Overcooked`)

Gymnasium 1.4 passes `seed`/`options` to `reset()` and expects
`(obs, info)` from `reset` and a 5-tuple from `step`. Upstream 2.0.0 used the
older API, so the vendored class was adapted:

- `reset(self, *, seed=None, options=None)` calls `super().reset(seed=seed)`
  and picks `agent_idx` with `self.np_random` (so seeding is reproducible);
- `reset` returns `(obs, info)`;
- `step` returns `(obs, reward, terminated, truncated, info)`.

The observation is a dict that includes the `OvercookedState` object, which
the declared `Box` observation space cannot describe. Create the env with
`disable_env_checker=True`; `OvercookedEnv` itself (joint-action, multi-agent)
is unaffected.

## Usage

Run from the repository root (or put the repository root on `sys.path`):

```python
import gymnasium
from GameEnv.overcooked_mdp import OvercookedGridworld
from GameEnv.overcooked_env import OvercookedEnv

mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp, horizon=400)

env = gymnasium.make(
    "Overcooked-v0",
    base_env=base_env,
    featurize_fn=base_env.featurize_state_mdp,
)
obs, info = env.reset()
# obs["both_agent_obs"], obs["overcooked_state"], obs["other_agent_env_idx"]
```

For multi-agent training, wrap `OvercookedEnv` directly (joint actions,
player-centric observations) instead of using the flattened single-agent
`Overcooked` class.

## Upgrading upstream

Re-copy the files above and re-apply the import rewrite:

```bash
grep -rn "overcooked_ai_py\." GameEnv --include="*.py"
```

There should be no matches. `GameEnv/utils.py` (`read_layout_dict`) and
`GameEnv/static.py` locate assets relative to `GameEnv/data/`, so keep that
directory structure intact.
