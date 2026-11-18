# work_c/ — 评估协议（工作 C）

本目录是**独立的评估模块**，不依赖 `algorithms/`、`envs/` 的具体实现，只依赖：

- `GameEnv/`（vendored 只读环境）
- `agents/`（伙伴策略：`SPPolicy` / `StayBot` / `RandomBot` / `FCPPool`）

目标是回答：**给定一个 checkpoint，它和不同伙伴配合得怎么样，能否复现。**

---

## 目录结构

```
work_c/
├── eval/
│   ├── gameenv_wrapper.py   构造评估环境（座位固定 + 随机初始状态）
│   ├── run_episode.py       跑一局，记录 soups / shaped / 轨迹指纹
│   ├── run_sweep.py         按伙伴分组跑，产出 median/IQR + 配对 gap
│   └── aggregate.py         把 results/raw/*.json 汇总成 CSV
├── scripts/
│   ├── evaluate.py          单个 checkpoint -> 一份标准 JSON（验收交付物）
│   ├── run_all_experiments.py  批量评估多个 checkpoint（学习曲线数据）
│   ├── plot_results.py      从 CSV 出图
│   └── reproduce.py         一键全流程 + manifest 哈希校验
└── results/
    ├── raw/*.json           每个 checkpoint 的完整评估结果
    ├── summary_long.csv     长表：一行 = 一个 checkpoint x 一个分组
    ├── summary_wide.csv     宽表：一行 = 一个 checkpoint
    ├── learning_curve.csv   画图用的精简表
    ├── figures/*.png        学习曲线 + gap 图
    └── manifest.json        所有输出文件的 sha256（用于跨机比对）
```

---

## 快速开始

```bash
# 1. 单个 checkpoint 评估
python -m work_c.scripts.evaluate \
    --ckpt work_c/partners/self_play_snapshots/sp_seed0_best.pt \
    --seen-dir work_c/partners/self_play_snapshots_old \
    --heldout-ckpt work_c/partners/self_play_snapshots/sp_seed0_last.pt \
    --n-episodes 5 --seeds 0,1,2,3,4 \
    --out work_c/results/raw/sp_seed0_best.json

# 2. 一键跑全部 + 出表 + 出图
python -m work_c.scripts.reproduce

# 3. 重跑一遍并和上一次逐文件比对哈希
python -m work_c.scripts.reproduce --check
```

**验收标准**：`--check` 输出 `[check] 通过：14 个输出文件全部一致`。

---

## 关键设计（这些是踩过坑之后定的）

### 1. 座位固定

`GameEnv` 的 Gym wrapper 在每次 `reset()` 时随机 `agent_idx`（0 或 1），
并把 `both_agent_obs` 和 joint action 都按「策略智能体」为中心重排。

如果不处理，ego 有一半概率被喂成玩家 1 的观测。

**处理**：`gameenv_wrapper.py` 在 `reset()` 后强制 `agent_idx = 0`，
并从 `base_env.state` 重新 featurize，保证 `obs[0]` 恒为 player0。

### 2. 初始状态随机化（最重要）

确定性策略 + 固定起始状态时，**同一个伙伴的每一局完全相同**——
只有座位在变。所以「跑 10 局」其实只有 1~2 个样本。

实测：固定起始状态下 6 局只有 **1 条不同轨迹**；随机初始状态下 6 局有 **6 条**。

**处理**：`--start-state random_obj`（默认）使用
`mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.5)`。

### 3. 每局重新播种

- `np.random.seed(ep_seed)`：`start_state_fn` 内部用的是全局 `np.random`
- `policy.reset(ep_seed)`：带随机性的策略（如 `RandomBot`）必须每局重播，
  否则 RNG 跨局累积，单独重跑某个 seed 数字就变了

### 4. 配对计算 gap

```
gap = median(soups | seen 伙伴) − median(soups | held-out 伙伴)
```

两臂用**同一批 episode seed**，所以是配对比较，方差比「两个总均值相减」小得多。

同时报告两个指标：

| 指标 | 用途 |
|---|---|
| `soups_delivered` | 主指标（端到端上菜数） |
| `shaped_total` | 过程指标（放洋葱/取盘/取汤的塑形奖励之和） |

主指标全 0 时，过程指标仍能显示配合质量的变化。

---

## 指标定义

| 字段 | 含义 | 注意 |
|---|---|---|
| `soups_delivered` | 本局上菜数（`sparse_r_by_agent` 之和 / 20） | |
| `episode_length` | 本局实际步数 | **恒等于 horizon**，因为 done 只由 horizon 触发，无信息量 |
| `steps_to_first_soup` | 首次上菜的步数 | 没上菜时填 `horizon`，存在截断偏差 |
| `got_soup` | 本局是否上过菜 | 配合 `steps_to_first_soup` 一起看 |
| `shaped_total` | 本局塑形奖励累计 | `PLACEMENT_IN_POT_REW=3`、`DISH_PICKUP_REWARD=3`、`SOUP_PICKUP_REWARD=5` |
| `traj` | 动作序列的 md5 前 12 位 | 用来验证「N 局真的是 N 条轨迹」 |

---

## 支持的 checkpoint 格式

`agents/checkpoint.py` 的 `SPPolicy` 同时支持两种：

**旧格式（本目录自带的 11 个 `.pt`）**

```python
{"state_dict": {...}, "obs_dim": 96, "hidden": 128, "global_step": 460800}
```

**B 的格式（`algorithms/algo_utils.py::save_weights` 产出）**

```python
{"format_version": 1, "step": 12345,
 "param": {"obs_dim": 96, "hidden_dim": 128, "action_dim": 6, ...},
 "actor_state": {...},      # 键名形如 network.0.weight
 "critic_state": {...}, "optimizer_state": {...}}
```

判定依据是字典里有没有 `actor_state` 键。
加载时缺任何 actor 权重会**直接抛异常**，不会静默退回随机权重。

---

## 已知限制

| # | 限制 | 说明 |
|---|---|---|
| 1 | `episode_length` 恒等于 horizon | 该环境的 done 只由步数上限触发，此项无信息量 |
| 2 | `steps_to_first_soup` 有截断偏差 | 没上菜时填 horizon，中位数会被拉高 |
| 3 | 只有 1 个训练 seed | 现有 checkpoint 均来自 seed 0，无法给出训练方差 |
| 4 | `seen` 组只有 ego 自己 | 未把历史 checkpoint 放进 seen 组，配对口径偏保守 |
| 5 | `heldout` 含同源 `last.pt` | 严格说它属于「见过的家族」，不是真正的未见伙伴 |
| 6 | 跨机一致性未实测 | `--check` 只在单机验证过 |

---

## 与其它层的边界

| 层 | 提供 | 本目录是否使用 |
|---|---|---|
| `envs/`（A） | `OvercookedSingleEnv` / `EnvConfig` / `joint_index` | ❌ 不用，自建 `gameenv_wrapper.py` |
| `algorithms/`（B） | `PPO` / `save_weights` / `load_weights` | ⚠️ 只读取它产出的 checkpoint 文件 |
| `agents/`（C） | `SPPolicy` / `FCPPool` / 脚本 bot | ✅ 直接使用 |

**为什么不复用 `envs/`**：A 的 `OvercookedSingleEnv` 没有传 `start_state_fn`，
起始状态固定，会把上面第 2 条的核心修复退回去。

如果 A 之后加上了随机初始状态，只需改 `eval/gameenv_wrapper.py` 一个文件即可切换。
