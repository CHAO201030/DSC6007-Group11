# 评估协议（work C）

> 本文件是工作 C 的评估说明。代码在 `scripts/`，数据在 `results/eval/`。

## 1. 目录

```
scripts/
├── evaluate.py            单点评估 -> 一份固定格式 JSON
├── run_all_experiments.py 批量评估（学习曲线 / 终点表）
├── transfer_matrix.py     ego × partner 交叉矩阵
├── plot_all.py            出图 F1–F6
├── check_repro.py         复现校验（两次运行逐字节比对）
├── fcp_curve_stats.py     训练曲线健康检查
├── eval/                  评估内核（环境包装、单局、分组扫描、汇总）
└── legacy/                已归档的旧脚本，不要使用

results/eval/
├── decay_s*.csv           兼容率衰减（每个训练 seed 一份）
├── transfer_stoch.csv     15 × 10 迁移矩阵
├── final_table.csv        四臂终点表
├── f42_cross.csv          FCP-scratch seed42 对异源 checkpoint 的验证
├── TABLES.md              报告用 T1/T2 表
├── REPRO_CHECK.md         复现校验记录
├── manifest.json          产物哈希清单
├── raw_final/             终点表 20 份原始 JSON
├── diag/                  定位用的诊断 JSON
├── curves/                各臂训练曲线 npz
└── figures/               F1–F6.png
```

## 2. 四条臂

| 目录 | 名字 | 初始化 | 伙伴 | 步数 |
|---|---|---|---|---|
| `checkpoints/ppo_single_env` | **SP** | 随机 | 自己是同一份权重（联合训练） | 8M |
| `checkpoints/fcp` | **FCP-scratch** | 随机 | 100% 池（5 seed × 16 checkpoint） | 8M |
| `checkpoints/fcp2` | **Arm P** | SP@8M | 50% 自身 8M + 50% 池 | 4M |
| `checkpoints/sp_ref` | **Arm S** | SP@8M | 100% 自身 8M（对照） | 4M |
| `checkpoints/ppo_vec_env` | **held-out** | — | 仅评估用，从未进入任何池 | — |

## 3. 评估协议

| 项 | 取值 |
|---|---|
| 起始状态 | `standard`（与训练一致）。`random_obj` 会让所有臂一起崩，仅作鲁棒性附录 |
| 动作 | 采样（`--stochastic`）；argmax 只用于快速冒烟 |
| 每格样本 | 3 个评估 seed × 5 局 = 15 局 |
| 配对 | 每局 `torch.manual_seed(ep_seed)`，两臂消耗同一串随机数 |
| 强 held-out | `ppo_vec_env` seed1–5 @8M |
| 弱 held-out | `StayBot` / `RandomBot` |

**跨臂可比性**：只有"两条臂都没见过"的集合能用于跨臂比较（vec + bot）。
`ppo_single_env` 的其它 seed 对 SP 臂未见、对 Arm P 已见，只能作为 **SP-only 诊断**。

## 4. checkpoint 自检规则

每个 seed 的最终 checkpoint 必须通过自对弈自检（standard 起点、采样动作、≥ 9 汤）：

- 未通过者回退到该 seed 最近的通过检查的 checkpoint，并在结果中注明；
- **seed526007**：8M 未通过（0 汤 / shaped 174，塌缩态），改用 7.5M（11 汤）；
- 规则对四条臂统一执行。

## 5. 常用命令

```bash
# 单点评估
python -m scripts.evaluate --ckpt <path> --heldout-ckpt <p1> <p2> \
    --n-episodes 5 --seeds 0,1,2 --start-state standard --stochastic --out out.json

# 批量
python -m scripts.run_all_experiments --ckpt-dir <dirs...> \
    --heldout-ckpt <paths...> --n-episodes 5 --seeds 0,1,2 \
    --start-state standard --stochastic --out-csv results/eval/c.csv --raw-dir results/eval/raw_x

# 交叉矩阵
python -m scripts.transfer_matrix --ego <...> --partner <...> \
    --episodes 5 --start-state standard --stochastic --out results/eval/m.csv

# 出图（F1–F6）
python -m scripts.plot_all

# 复现校验
python -m scripts.check_repro
```

## 6. 已知限制

| # | 限制 |
|---|---|
| 1 | 单学习器 + 冻结伙伴 ≠ 联合自对弈，Arm S 不是 SP 的严格对照 |
| 2 | held-out 与训练伙伴同属 PPO 家族，差异来自采样方式与训练随机性，不是策略类别 |
| 3 | 训练预算 8M 步、96 维 MLP、单机 CPU，远小于文献量级 |
| 4 | 只有 1/5 个 seed 的 FCP 学出有效策略，只能作 existence proof |
| 5 | 兼容率阈值取 ≥8 汤，属人为设定；≥6 / ≥10 的趋势方向相同 |
| 6 | 随机初始状态条件只做过一次诊断，未系统评估 |
| 7 | `scripts/legacy/` 下为过期脚本，不要使用 |

## 7. 与其它层的边界

| 层 | 提供 | 本层 |
|---|---|---|
| `envs/`（A） | `OvercookedSingleEnv` / `EnvConfig` / `joint_index` | 训练用；评估用 `scripts/eval/gameenv_wrapper.py`（需要随机初始状态与座位固定） |
| `algorithms/`（B） | `PPO` / `save_weights` / `load_policy` / `get_GAE` | 只读取 checkpoint，训练时复用 |
| `agents/`（C） | `SPPolicy` / `FCPPool` / 脚本 bot | 直接使用 |
