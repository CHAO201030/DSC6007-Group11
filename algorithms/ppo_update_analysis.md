# PPO::update 逻辑分析

> 分析对象：`algorithms/ppo.py` 中的 `PPO.update`（master 分支）。
> 重点：第 167 行开始的两层 `for` 循环，尤其是外层 `for _ in range(self.param.epochs)` 的作用。

## 1. 方法输入

```python
def update(self, obs, actions, old_log_probs, advantages, returns):
```

| 输入 | 含义 | 形状 |
|---|---|---|
| `obs` | rollout 时保存的观测 | `(T, N, obs_dim)` 或 `(T*N, obs_dim)` |
| `actions` | rollout 时实际执行的动作 | `(T, N)` |
| `old_log_probs` | 采样时策略的 log 概率 | `(T, N)` |
| `advantages` | GAE 计算出的优势 | `(T, N)` |
| `returns` | `advantages + values`，critic 的回归目标 | `(T, N)` |

`T` 是 rollout 长度，`N` 是 agent 数量（当前为 2）。

## 2. 进入循环前的预处理

```python
obs = obs.reshape(-1, obs.shape[-1])       # (T*N, obs_dim)
actions = actions.reshape(-1)              # (T*N,)
old_log_probs = old_log_probs.reshape(-1).detach()
advantages = advantages.reshape(-1)
returns = returns.reshape(-1)

n = obs.shape[0]                           # T*N

advantages = (advantages - advantages.mean()) / (
    advantages.std(unbiased=False) + 1e-8
)
advantages = advantages.detach()
returns = returns.detach()
```

关键点：

- 两个 agent 的样本被 flatten 到同一个 batch；
- `old_log_probs`、`advantages`、`returns` 都是**固定常量**，在整个 update 中不改变；
- `advantages` 做一次全局归一化；
- `returns` 只作为 critic 的监督目标，不参与归一化。

## 3. 外层 `for`：为什么需要 epochs

```python
for _ in range(int(self.param.epochs)):
    indices = torch.randperm(n, device=device)
    for start in range(0, n, batch_size):
        ...
```

### 3.1 一行回答

**外层 `for` 的作用是：让同一批 rollout 数据被 PPO 重复利用多个 epoch，而不是只做一次梯度更新。**

PPO 是 on-policy 算法，但它的设计允许在**同一批旧数据**上做有限次优化。这正是 PPO 相比 vanilla policy gradient 样本效率更高的原因之一。

### 3.2 为什么不能只更新一次

如果只做一次：

```text
采样一批 T 步数据 → 前向/反向一次 → 丢弃
```

问题：

- 每个样本只贡献一次梯度，样本利用率低；
- 环境采样通常比网络计算贵得多；
- critic 也需要多次迭代才能拟合好 returns；
- 策略更新一次通常不够。

所以 PPO 在一次 rollout 后做：

```text
epochs × minibatches 次梯度更新
```

例如：

```text
n = 800, batch_size = 64, epochs = 4
每个 epoch: 13 个 minibatch（最后一批 32 个样本）
总共: 4 × 13 = 52 次 optimizer.step()
```

## 4. 为什么可以重复使用旧数据

PPO 的 clipped surrogate 限制了新旧策略的差距：

```text
ratio = exp(new_log_prob - old_log_prob)

L = min(
    ratio * advantage,
    clip(ratio, 1 - eps, 1 + eps) * advantage
)
```

- `old_log_probs` 固定；
- 每次 minibatch 更新后，策略参数变了，`new_log_prob` 会变；
- `clip` 让 ratio 超出 `[1-eps, 1+eps]` 后不再提供额外收益；
- 因此策略不会在旧数据上走得太远。

这就是“可以多 epoch”的理论基础；clip 是对“数据逐渐 off-policy”的一种控制。

## 5. 内层 `for`：minibatch 更新

```python
for start in range(0, n, batch_size):
    idx = indices[start : start + batch_size]
```

- 每个 epoch 重新 `randperm(n)`；
- 打乱后按 `batch_size` 切分；
- 最后一批可能小于 `batch_size`；
- 每个 minibatch 独立做一次前向、反向和 optimizer step。

### 单个 minibatch 的计算

```python
logits = self.actor(batch_obs)
dist = Categorical(logits=logits)
new_log_probs = dist.log_prob(batch_actions)

log_ratio = new_log_probs - batch_old_log_probs
ratio = torch.exp(log_ratio)

policy_loss = -torch.min(
    ratio * batch_advantages,
    torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * batch_advantages,
).mean()

values = self.critic(batch_obs).squeeze(-1)
value_loss = F.mse_loss(values, batch_returns)

loss = policy_loss + self.param.value_coef * value_loss

self.optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(params, self.param.max_grad_norm)
self.optimizer.step()
```

注意：

- `old_log_probs` 来自 rollout，必须 detach；
- `ratio` 会在每个 minibatch 中用**当前参数**重新计算；
- 第一次更新时 ratio 应接近 1；
- `policy_loss` 是 `-L_clip`，因为 PyTorch 最小化；
- 没有 KL、没有 entropy，这是当前实现的明确设计；
- `value_loss` 是未裁剪的 MSE。

## 6. 两层 `for` 的分工

| 循环 | 作用 |
|---|---|
| 外层 `for _ in range(epochs)` | 对同一批数据重复优化；每个 epoch 重新打乱 |
| 内层 `for start in range(0, n, batch_size)` | 把数据切 minibatch，逐批梯度更新 |

如果只有内层循环，只相当于一个 epoch。

如果只有外层循环，没有 minibatch 切分，就是全批量更新，计算量大且泛化较差。

## 7. 为什么不用一层 `for` 遍历所有 minibatch

理论上可以写成：

```python
for _ in range(epochs * num_batches):
    ...
```

但两层循环更清晰：

- 外层表示“数据重复使用次数”；
- 内层表示“一次 pass 内的 minibatch”；
- 每个 epoch 需要重新 `randperm`，两层结构自然表达；
- `epochs` 和 `minibatch_size` 是两个独立超参数，分别控制“复用次数”和“每次更新批量”。

## 8. 指标累计

```python
weight = idx.numel()
policy_loss_sum += policy_loss.item() * weight
value_loss_sum += value_loss.item() * weight
clip_fraction_sum += clip_fraction * weight
seen += weight
```

- 每个 minibatch 的指标按样本数加权；
- `seen` 累计所有 epoch、所有 minibatch 的样本数；
- 正常情况下：

```text
seen = epochs × n
```

- 最后返回：

```python
denom = max(seen, 1)
return (
    policy_loss_sum / denom,
    value_loss_sum / denom,
    clip_fraction_sum / denom,
)
```

所以返回的是**所有更新步骤的按样本加权平均**，不是某一批的值。

## 9. 关键设计点

### 9.1 `old_log_probs` 固定

这是 PPO 能算 ratio 的前提。不能在 update 中重新采样或重新计算 old log prob。

### 9.2 advantages 固定

GAE 的优势在 update 前已经算好，update 只重复使用，不重新计算。

### 9.3 epochs 不能无限大

虽然 clip 限制了更新幅度，但：

- 数据会逐渐变成“更 off-policy”；
- critic 目标 returns 也是旧的；
- 太多次更新可能导致策略过拟合当前 rollout；
- clip fraction 持续很高是信号。

常见范围：`3 ~ 10`；当前配置是 `4`。

### 9.4 `clip_fraction` 的计算时机

```python
with torch.no_grad():
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
```

它是在 `optimizer.step()` 之后统计的，但使用的是本 minibatch 更新前的 `ratio`，表示“这一批里有多少样本被裁剪”。它不参与 loss，只是诊断指标。

### 9.5 没有 KL / entropy

当前实现：

- 没有 KL penalty；
- 没有 KL early stop；
- 没有 entropy bonus。

所以 update 的 loss 只有：

```text
policy_loss + value_coef * value_loss
```

epochs 的控制完全依赖 clip，而不是 target KL。

## 10. 潜在问题 / 改进点

1. **advantage 是全局归一化**  
   当前在所有 epoch 之前做一次归一化，之后固定。另一种做法是每个 minibatch 内部再归一化；两者都常见，但要保持一致。

2. **epochs = 0 时不更新**  
   `seen = 0`，返回 0 指标，参数不变。可以加校验。

3. **n = 0 时**  
   `randperm(0)` 不会进入内层循环，最后返回 0 指标；通常不会发生，但可以显式检查。

4. **没有 target KL early stop**  
   当前设计没有 KL；如果想更稳定，可以在每个 epoch 后计算 approx KL 并提前停止，但这与“无 KL”的设计目标冲突，需要团队决定。

5. **value loss 未裁剪**  
   标准 PPO 常直接使用 MSE；value clipping 是可选项，不是必需。

## 11. 一句话总结

```text
外层 for (epochs):
  让同一批 rollout 被重复使用多次，提高样本效率；
  每次重新打乱，避免固定顺序带来的偏差。

内层 for (minibatches):
  在每个 epoch 内切小批量，逐批计算 clipped surrogate
  和 value loss，做反向传播和 optimizer.step。
```

**第 167 行的 `for` 不是多余循环，而是 PPO 的核心：一次采样、多次优化。** 没有它，PPO 就退化成“每批数据只更新一次”的 on-policy 策略梯度，样本效率会明显下降；但 epochs 也不能太大，否则旧数据会过度 off-policy，clip fraction 会升高，训练变得不稳定。
