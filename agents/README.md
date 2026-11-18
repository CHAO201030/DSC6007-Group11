# agents/ — 伙伴策略

三种伙伴策略的实现，供**训练层**和**评估层**共同使用。

本目录自成一体，不依赖仓库里的其他 Python 包，可直接 `from agents import ...`。

---

## 一、接口约定

所有策略对象都实现同一套接口：

```python
policy.reset(seed)                    # 每局开始时调用一次，用于重新播种
policy.act(obs, agent_index) -> int   # 返回动作索引 0..5
```

| 参数 | 类型 | 含义 |
|---|---|---|
| `obs` | `np.ndarray (2, 96)` float32 | `obs[0]` 恒为 **player0** 的观测，`obs[1]` 为 player1 |
| `agent_index` | `int` | 我是几号玩家：`0` 或 `1` |
| 返回值 | `int` | `0`=NORTH `1`=SOUTH `2`=EAST `3`=WEST `4`=STAY `5`=INTERACT |

> ⚠️ `reset(seed)` **必须每局调用**。带随机性的策略（如 `RandomBot`）内部靠它重新播种；
> 不调用的话 RNG 会跨局累积，导致"同样的 seed 每次结果不同"。

---

## 二、三个模块

| 模块 | 导出的类 | 用途 | 对应"三种伙伴"中的 |
|---|---|---|---|
| `scripted.py` | `StayBot` / `RandomBot` | 固定规则 bot | 脚本 bot（固定规则，作下界） |
| `checkpoint.py` | `SPPolicy` | 把 checkpoint 包成策略 | self-play 快照（当前策略的副本） |
| `pools.py` | `FCPPool` | 历史 checkpoint 池，均匀采样 | 历史 checkpoint 池（FCP） |

### `scripted.py`

```python
from agents import StayBot, RandomBot

bot = StayBot()          # 永远 STAY
bot = RandomBot(seed=0)  # 每步随机
```

### `checkpoint.py` — `SPPolicy`

```python
from agents import SPPolicy

p = SPPolicy("checkpoints/step0450560.pt", deterministic=True)
p.selftest()             # 建议加载后调用一次
```

| 属性 | 说明 |
|---|---|
| `p.path` | checkpoint 路径 |
| `p.digest` | 文件的 sha256（用于结果溯源） |
| `p.global_step` | 从 checkpoint 元信息读取的训练步数，读不到为 `-1` |
| `p.obs_dim` | 自动从权重形状推断（96 或 192） |

**加载行为**：
- 支持 `{"state_dict": ..., "obs_dim": 96, ...}` 或裸 `state_dict`
- 自动剥掉 `actor.` / `policy.` 前缀，自动丢弃 `critic.` / `value.` 权重
- 网络结构默认 `obs_dim -> 128 -> 128 -> 6`（`Tanh` 激活），`hidden` 从权重推断
- **缺键直接抛 `RuntimeError`**，不会静默保留随机权重

### `pools.py` — `FCPPool`

```python
from agents import FCPPool

pool = FCPPool(["ckpt_1.pt", "ckpt_2.pt", ...])
len(pool)                        # 池大小
paths = pool.sample(n=1, seed=42)  # 返回**路径列表**（不是策略对象）
```

- 采样是无放回均匀采样，结果与 `seed` 绑定、可复现
- 池为空时构造直接抛 `ValueError`

---

## 三、训练时怎么用

### 最小示例

```python
import glob
from agents import FCPPool, SPPolicy

# 1) 建池（一次性，在训练开始前）
pool = FCPPool(sorted(glob.glob("checkpoints/*.pt")))

# 2) 需要换对手时
paths = pool.sample(n=1, seed=global_step)
partner = SPPolicy(paths[0], deterministic=False)   # ★ 训练必须 False

# 3) 每局开始
partner.reset(ep_seed)

# 4) 每步
partner_action = partner.act(obs, agent_index=1)
```

### 三个必须注意的点

1. **`deterministic=False`**
   `SPPolicy` 默认 `deterministic=True`（argmax 确定性动作），那是**评估**用的。
   训练时如果不传 `False`，对手永远做同一个动作，探索完全没有。

2. **`sample()` 返回路径，不是策略对象**
   需要自己包一层 `SPPolicy(...)`。如果你想避免重复加载权重，可以自己缓存：

   ```python
   cache = {}
   def get_partner(path, seed):
       if path not in cache:
           cache[path] = SPPolicy(path, deterministic=False)
       p = cache[path]
       p.reset(seed)
       return p
   ```

   单个 checkpoint 约 240 KB，加载开销很小，但训练循环里每步都加载会明显变慢。

3. **`reset(seed)` 每局调用一次**
   对 `SPPolicy` 是空操作（确定性策略无内部随机状态），但 `RandomBot` 必须调用。
   统一调用不会出错，养成习惯。

### 用脚本 bot 当对手

```python
from agents import RandomBot, StayBot

partner = StayBot()      # 下界
partner = RandomBot()    # 下界
partner.reset(ep_seed)
```

---

## 四、评估时怎么用

```python
from agents import SPPolicy

ego = SPPolicy("checkpoints/best.pt", deterministic=True)
ego.selftest()

obs = np.zeros((2, 96), dtype=np.float32)
action = ego.act(obs, agent_index=0)
```

评估用的完整流水线在 `work_c/`：
- `python -m work_c.scripts.evaluate --ckpt <路径> --out results.json`
- `python -m work_c.scripts.reproduce`

---

## 五、已知限制

| # | 限制 | 说明 |
|---|---|---|
| 1 | `obs_dim == 192` 分支未验证 | 该分支对两个 agent 返回同一个拼接向量，若将来用 192 维模型会出问题 |
| 2 | 只支持 `96 -> 128 -> 128 -> 6` 的 MLP | 若是 CNN 或更深/更宽的网络，需要改 `PolicyNet` |
| 3 | 只支持离散动作空间（6 个） | 不支持连续动作 |
| 4 | checkpoint 必须是 PyTorch `.pt` | SB3 的 `.zip` 需要另写适配器 |
| 5 | `PolicyNet` 是临时实现 | 推理用的网络定义放在本目录，是为了能直接加载 checkpoint。正式的**网络结构归属 `algorithms/`（B）**；B 提供 `load_policy` 后应替换为调用它 |

---

## 六、修改本目录的约定

- 这里的接口被训练层和评估层同时使用
- 新增伙伴类型时，实现 `reset(seed)` + `act(obs, agent_index)` 两个方法即可