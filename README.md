# Beyond Self-Play: A Controlled Study of Partner Generalisation in a Collaborative Kitchen Environment

**DSC6007 Group 11**

本项目以 [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai)
v2.0.0 为基础，将其运行环境 vendored 为 `GameEnv/` 包，**无需 `pip install` 即可直接使用**
Gymnasium 风格的 Overcooked 环境。

研究目标是在 Overcooked 双人合作厨房任务中，以受控实验的方式考察智能体的
**伙伴泛化能力（partner generalisation）**：在训练阶段只接触有限伙伴（包括 self-play）
的情况下，策略能否与评估阶段遇到的未见伙伴有效协作。

## 研究目标

- 在统一的布局、horizon、训练预算和评估 seed 下，对比 self-play 与伙伴多样性训练
- 使用 held-out partners 评估协作表现，区分「与训练伙伴配合」和「泛化到新伙伴」
- 保持环境、观测编码和评估协议固定，使不同算法的差异尽量来自伙伴策略与训练方式

> 当前处于环境搭建阶段；`agents/`、`algorithms/`、`common/`、`config/`、`checkpoints/`
> 已建立目录骨架，具体实现待组内分工确定。

## 特性

- **开箱即用的环境**：`GameEnv/` 包含 MDP、planner、可视化、Gym 包装和全部布局/贴图资源
- **不依赖上游安装**：环境中不需要 `pip install overcooked_ai`，避免双模块树歧义
- **Gymnasium 1.x 兼容**：`reset(seed=...)` 返回 `(obs, info)`，`step()` 返回 5 元组
- **最小 conda 环境**：`environment.yml` 只包含运行、训练和画图所需依赖
- **冒烟测试**：`test_env.py` 一条命令验证 MDP、planner、渲染和 Gym 接口

## 快速开始

```bash
# 1. 创建环境（Linux / macOS / Windows 均可）
conda env create -f environment.yml
conda activate gameenv

# 2. 验证环境
python test_env.py
```

预期输出：

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

> 不需要 `pip install` 本仓库；在仓库根目录运行即可（`GameEnv` 通过目录导入）。

## 目录结构

```
DSC6007-Group11/
├── GameEnv/                 # vendored Overcooked-AI 环境（只读，见 VENDORING.md）
│   ├── actions.py           # 动作与方向定义
│   ├── overcooked_mdp.py    # MDP：状态、转移、观测编码
│   ├── overcooked_env.py    # OvercookedEnv + Gymnasium 包装
│   ├── planners.py          # 运动/中层规划器
│   ├── state_visualizer.py  # pygame 渲染
│   ├── data/                # 布局、贴图、字体、planner 缓存
│   └── VENDORING.md         # 上游来源、改动清单
├── agents/                  # 决策策略与伙伴/对手策略（待分工实现）
│   └── __init__.py
├── algorithms/              # 训练算法，如 PPO（待分工实现）
│   └── __init__.py
├── common/                  # 共享接口与工具
│   └── __init__.py
├── config/                  # 每个实验一份 YAML 配置
│   └── .gitkeep
├── checkpoints/             # 权重与 run 元数据（内容被 git 忽略）
│   └── .gitkeep
├── environment.yml          # 最小 conda 环境
├── test_env.py              # 环境冒烟测试
├── LICENSE                  # 本仓库 MIT
├── GameEnv/LICENSE.overcooked_ai
└── README.md
```

> `envs/`（环境 wrapper）、`scripts/`（训练/评估入口）、`tests/`（契约测试）将在实现训练流程时加入。

## 代码组织约定

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| `agents/` | `act(obs) -> int`、规则策略、伙伴/对手策略、训练后策略的封装 | loss、optimizer、GAE |
| `algorithms/` | 网络、loss、优化器、rollout buffer、GAE、训练循环 | 环境细节、评估脚本、伙伴选择逻辑 |
| `common/` | 共享 `Protocol`、观测/动作工具、seed 工具 | 具体算法实现 |
| `config/` | 每个实验一份 YAML（布局、超参数、seed、训练步数） | 代码 |
| `checkpoints/` | `best.pt`、`last.pt`、`run.json` 等 | 除 `.gitkeep` 外都不应提交 |

修改 `common/` 的接口需要全组 review；`GameEnv/` 只读；`checkpoints/` 的内容不会进入 git。

## 使用示例

### 1. 直接使用 MDP（不需要 Gym）

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

### 2. 使用 `OvercookedEnv`（joint action，推荐多智能体）

```python
from GameEnv.actions import Action
from GameEnv.overcooked_mdp import OvercookedGridworld
from GameEnv.overcooked_env import OvercookedEnv

mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)

state, reward, done, info = env.step((Action.INTERACT, Action.STAY))
print(reward)                       # 稀疏奖励（送汤时非零）
print(info["sparse_r_by_agent"])    # 每个 agent 的稀疏奖励
print(info["shaped_r_by_agent"])    # 每个 agent 的 shaping 奖励
```

- `OvercookedEnv` 的 `step` 接收**联合动作** `(action_p0, action_p1)`；
- 返回 `(next_state, reward, done, info)`；
- `done` 由 `horizon` 决定（Overcooked 没有真正的终止状态）。

### 3. 使用 Gymnasium 接口

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
    # obs 是含 OvercookedState 的 dict，无法用 Box 空间描述
    disable_env_checker=True,
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step((4, 4))  # (STAY, STAY)
```

- 观测是 dict：`both_agent_obs`、`overcooked_state`、`other_agent_env_idx`；
- 动作空间声明为 `Discrete(6)`，但 `step()` 实际接收 `(自己的动作, 对手的动作)` 二元组；
- 训练 PPO/SB3 前需要一层 wrapper，把观测提取为数组、动作映射为标准 `Discrete`。

### 动作编号

| 编号 | 动作 | 说明 |
|---|---|---|
| 0 | `NORTH` | 上移 |
| 1 | `SOUTH` | 下移 |
| 2 | `EAST` | 右移 |
| 3 | `WEST` | 左移 |
| 4 | `STAY` | 原地不动 |
| 5 | `INTERACT` | 交互（拾取/放锅/送餐等） |

### 观测编码

```python
# 面向 MLP：每个 agent 96 维向量
obs_p0, obs_p1 = base_env.featurize_state_mdp(base_env.state)

# 面向 CNN：(H, W, 26) 张量
enc_p0, enc_p1 = base_env.lossless_state_encoding_mdp(base_env.state)
print(enc_p0.shape)  # 例如 (5, 4, 26)
```

## 测试

```bash
python test_env.py                                  # 默认 cramped_room，20 步
python test_env.py --layout forced_coordination     # 换布局
python test_env.py --steps 50 --seed 7              # 指定步数和种子
python test_env.py --all-layouts                    # 检查全部 49 个内置布局
python test_env.py --skip-render                    # 无显示器环境跳过渲染
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--layout` | `cramped_room` | 布局名 |
| `--steps` | `20` | 随机 rollout 步数 |
| `--seed` | `0` | 随机种子 |
| `--all-layouts` | 关闭 | 检查 `GameEnv/data/layouts` 下所有布局 |
| `--skip-render` | 关闭 | 跳过 pygame 渲染检查 |

设置 `GAMEENV_TEST_TRACEBACK=1` 可打印完整 traceback。

## 环境说明

| 依赖 | 版本（求解结果） | 用途 |
|---|---|---|
| Python | 3.12 | |
| NumPy | 2.5.4 | MDP、观测 |
| SciPy | 1.18.1 | planner 的稀疏图搜索 |
| Gymnasium | 1.3.0 | `gym.Env` 与 `Overcooked-v0` 注册 |
| pygame | 2.6.1 | 渲染与 `StateVisualizer` |
| PyTorch | 2.14.1 (CPU) | 训练 |
| tqdm | 4.70.1 | 进度条 |
| matplotlib / pandas / seaborn | 3.x / 3.x / 0.13 | 画学习曲线 |
| opencv-python-headless | >=4.8 | 渲染帧缩放（pip 安装） |

- 默认安装的是 **CPU 版 PyTorch**。有 CUDA 的机器可以删掉 yaml 里的 `pytorch`，
  创建环境后执行：
  `pip install torch --index-url https://download.pytorch.org/whl/cu126`
- 无显示器环境（WSL/服务器）需要 dummy 视频驱动：`test_env.py` 会自动设置
  `SDL_VIDEODRIVER=dummy`，其他脚本请在 `import GameEnv` 之前自行设置。Gym 包装的
  `env.render()` 返回 RGB ndarray；要在 pygame 窗口中实时显示，可使用
  `OvercookedEnv.display_states(...)` 或 `StateVisualizer.display_rendered_state(...)`，
  并确保有 WSLg/X11 等可用显示环境。
- `GameEnv/data/planners/*.pkl` 是首次运行时生成的 planner 缓存，已被 `.gitignore` 忽略；
  删除后会按需重算。

## 已知注意事项

1. **不要 `pip install overcooked_ai`**。本仓库使用 `GameEnv` 作为唯一环境来源；
   同时安装上游包会产生两套模块树，导致类身份和 pickle 不兼容。
2. **`GameEnv/` 是 vendored 代码**，原则上只读。确需修改时记录到
   `GameEnv/VENDORING.md`，避免每个人按自己的需要打补丁。
3. **奖励稀疏**。默认只有送汤时给奖励；训练时可考虑把
   `info["shaped_r_by_agent"]` 加入 reward，或自行设计 reward shaping。
4. **horizon 不是终止**。`done=True` 表示到达 `horizon`（截断），训练时应正确区分
   `terminated` 与 `truncated`，避免在截断时错误 bootstrap。
5. **权重不进入 git**。`checkpoints/` 下除 `.gitkeep` 外全部被忽略；组员之间共享权重
   请使用网盘、W&B artifact 或 Git LFS，不要直接提交大文件。

## 上游与许可

- 上游项目：[HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) v2.0.0
- `GameEnv/` 的改动仅限导入路径、Gymnasium 1.x 接口适配和可选的 notebook 依赖延迟导入，
  详见 [`GameEnv/VENDORING.md`](GameEnv/VENDORING.md)
- 上游代码使用 MIT 许可，见 [`GameEnv/LICENSE.overcooked_ai`](GameEnv/LICENSE.overcooked_ai)
- 本仓库代码使用 MIT 许可，见 [`LICENSE`](LICENSE)
